import os
import torch
import torch.nn.functional as F
import torch.nn as nn

def rating_to_weight(ratings):
    """Use ratings as mild confidence weights for positive-only training."""
    return 1.0 + 0.25 * (ratings.float() - 3.0).clamp(min=0.0)

def in_batch_softmax_loss(
    user_vecs,
    item_vecs,
    temperature=0.1,
    user_ids=None,
    target_items=None,
    sample_weights=None,
):
    logits = torch.matmul(user_vecs, item_vecs.T) / temperature

    if user_ids is not None or target_items is not None:
        batch_size = logits.size(0)
        off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=logits.device)
        false_negative_mask = torch.zeros_like(logits, dtype=torch.bool)

        if target_items is not None:
            same_target = target_items.view(-1, 1).eq(target_items.view(1, -1))
            false_negative_mask |= same_target

        if user_ids is not None:
            same_user = user_ids.view(-1, 1).eq(user_ids.view(1, -1))
            false_negative_mask |= same_user

        logits = logits.masked_fill(false_negative_mask & off_diagonal, -1e9)

    labels = torch.arange(user_vecs.size(0), device=user_vecs.device)
    losses = F.cross_entropy(logits, labels, reduction="none")

    if sample_weights is not None:
        weights = sample_weights.to(losses.device).float()
        return (losses * weights).sum() / weights.sum().clamp_min(1e-8)

    return losses.mean()

def save_checkpoint(path, model, optimizer, epoch, best_loss, preprocessor,
                    best_metric=None, metric_name=None):
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "epoch": epoch,
        "best_loss": best_loss,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "model_state": base_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "user_encoder": preprocessor.user_encoder,
        "movie_encoder": preprocessor.movie_encoder,
        "gender_encoder": preprocessor.gender_encoder,
        "age_encoder": preprocessor.age_encoder,
        "occ_encoder": preprocessor.occ_encoder,
        "num_genders": preprocessor.num_genders,
        "num_ages": preprocessor.num_ages,
        "num_occupations": preprocessor.num_occupations,
        "seq_len": preprocessor.seq_len,
        "genre2id": preprocessor.genre2id
    }, path)

def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        start_epoch = checkpoint["epoch"] + 1
        best_loss   = checkpoint["best_loss"]
        best_metric = checkpoint.get("best_metric")
    else:
        print("  [Warning] Checkpoint cũ, chỉ load được model weights.")
        state_dict  = checkpoint
        start_epoch = 0
        best_loss   = float("inf")
        best_metric = None

    is_parallel = isinstance(model, nn.DataParallel)
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

    if is_parallel and not has_module_prefix:
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    elif not is_parallel and has_module_prefix:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    metric_msg = f" | best metric: {best_metric:.4f}" if best_metric is not None else ""
    print(f"Resumed from epoch {start_epoch} | best loss: {best_loss:.4f}{metric_msg}")
    return start_epoch, best_loss, best_metric


def train(model, train_loader, optimizer, device,
          epochs, checkpoint_dir, preprocessor, temperature=0.05, resume=False,
          scheduler=None, val_loader=None, genre_matrix=None, val_user_seen_dict=None,
          eval_k=20, eval_every=1, metric_name=None, gradient_clip_norm=1.0,
          use_rating_weight=True, return_history=False):
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")

    start_epoch = 0
    best_loss   = float("inf")
    best_metric = None
    metric_name = metric_name or f"NDCG@{eval_k}"
    history = {
        "train_loss": [],
        "val_metrics": [],
    }

    if resume and os.path.exists(best_checkpoint_path):
        print("\nResuming training from checkpoint...")
        start_epoch, best_loss, best_metric = load_checkpoint(
            best_checkpoint_path, model, optimizer, device
        )

    print("\n--- START TRAINING ---")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            user_features = batch["user_inputs"]
            user_inputs = {
                "user_id":    user_features["user_id"].to(device),
                "gender":     user_features["gender"].to(device),
                "age":        user_features["age"].to(device),        
                "occupation": user_features["occupation"].to(device),
                "movie_hist": user_features["movie_hist"].to(device), 
            }
            item_features = batch["item_inputs"]
            item_inputs = {
                "target_movie":  item_features["target_movie"].to(device),
                "target_genres": item_features["target_genres"].to(device) 
            }
            ratings = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            user_vec, item_vec = model(user_inputs, item_inputs)
            
            sample_weights = rating_to_weight(ratings) if use_rating_weight else None
            loss = in_batch_softmax_loss(
                user_vec,
                item_vec,
                temperature=temperature,
                user_ids=user_inputs["user_id"],
                target_items=item_inputs["target_movie"],
                sample_weights=sample_weights,
            )
            
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                avg_batch_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_batch_loss:.4f}")

        avg_epoch_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_epoch_loss)
        print(f"Epoch {epoch+1} complete — Train Loss: {avg_epoch_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

        should_evaluate = (
            val_loader is not None
            and genre_matrix is not None
            and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1)
        )

        if should_evaluate:
            from test import evaluate

            metrics = evaluate(
                model=model,
                test_loader=val_loader,
                genre_matrix=genre_matrix,
                device=device,
                checkpoint_path=None,
                k=eval_k,
                user_seen_dict=val_user_seen_dict,
                verbose=False,
            )
            history["val_metrics"].append({"epoch": epoch + 1, **metrics})
            current_metric = metrics[metric_name]
            print(
                f"Validation — {metric_name}: {current_metric:.4f} | "
                f"Recall@{eval_k}: {metrics[f'Recall@{eval_k}']:.4f} | "
                f"MRR@{eval_k}: {metrics[f'MRR@{eval_k}']:.4f}"
            )

            if best_metric is None or current_metric > best_metric:
                best_metric = current_metric
                best_loss = avg_epoch_loss
                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    best_loss,
                    preprocessor,
                    best_metric=best_metric,
                    metric_name=metric_name,
                )
                print(f"New best checkpoint saved by {metric_name}: {best_metric:.4f}\n")
        elif avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(best_checkpoint_path, model, optimizer, epoch, best_loss, preprocessor)
            print(f"New best checkpoint saved by train loss: {best_loss:.4f}\n")

    if return_history:
        history["best_checkpoint_path"] = best_checkpoint_path
        history["best_metric"] = best_metric
        history["metric_name"] = metric_name
        return best_checkpoint_path, history

    return best_checkpoint_path

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def in_batch_softmax_loss(user_vecs, item_vecs, temperature=0.1):
    logits = torch.matmul(user_vecs, item_vecs.T) / temperature
    labels = torch.arange(user_vecs.size(0), device=user_vecs.device)
    return F.cross_entropy(logits, labels)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_loss,
    preprocessor,
    best_metric=None,
    metric_name=None,
):
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(
        {
            "epoch": epoch,
            "best_loss": best_loss,
            "best_metric": best_metric,
            "metric_name": metric_name,
            "model_state": base_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "user_encoder": preprocessor.user_encoder,
            "movie_encoder": preprocessor.movie_encoder,
            "genre2id": preprocessor.genre2id,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        best_metric = checkpoint.get("best_metric")
    else:
        print("  [Warning] Checkpoint cu, chi load duoc model weights.")
        state_dict = checkpoint
        start_epoch = 0
        best_loss = float("inf")
        best_metric = None

    is_parallel = isinstance(model, nn.DataParallel)
    has_module_prefix = any(key.startswith("module.") for key in state_dict)

    if is_parallel and not has_module_prefix:
        state_dict = {"module." + key: value for key, value in state_dict.items()}
    elif not is_parallel and has_module_prefix:
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    metric_text = (
        f" | best metric: {best_metric:.4f}"
        if best_metric is not None
        else ""
    )
    print(
        f"Resumed from epoch {start_epoch} | "
        f"best loss: {best_loss:.4f}{metric_text}"
    )
    return start_epoch, best_loss, best_metric


def train(
    model,
    train_loader,
    optimizer,
    device,
    epochs,
    checkpoint_dir,
    preprocessor,
    temperature=0.05,
    resume=False,
    val_loader=None,
    genre_matrix=None,
    val_user_seen_dict=None,
    eval_k=20,
    eval_every=1,
    return_history=False,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")

    start_epoch = 0
    best_loss = float("inf")
    best_metric = None
    best_epoch = None
    metric_name = f"NDCG@{eval_k}"
    history = {"train_loss": [], "val_metrics": []}

    if resume and os.path.exists(best_checkpoint_path):
        print("\nResuming training from checkpoint...")
        start_epoch, best_loss, best_metric = load_checkpoint(
            best_checkpoint_path,
            model,
            optimizer,
            device,
        )

    print("\n--- START TRAINING ---")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            user_features = batch["user_inputs"]
            user_inputs = {
                "user_id": user_features["user_id"].to(device),
                "gender": user_features["gender"].to(device),
                "age": user_features["age"].to(device),
                "occupation": user_features["occupation"].to(device),
                "movie_hist": user_features["movie_hist"].to(device),
            }
            item_features = batch["item_inputs"]
            item_inputs = {
                "target_movie": item_features["target_movie"].to(device),
                "target_genres": item_features["target_genres"].to(device),
            }

            optimizer.zero_grad()
            user_vec, item_vec = model(user_inputs, item_inputs)
            loss = in_batch_softmax_loss(
                user_vec,
                item_vec,
                temperature=temperature,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                avg_batch_loss = total_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {avg_batch_loss:.4f}"
                )

        avg_epoch_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_epoch_loss)
        print(
            f"Epoch {epoch + 1} complete | "
            f"Train Loss: {avg_epoch_loss:.4f}"
        )

        should_validate = (
            val_loader is not None
            and genre_matrix is not None
            and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1)
        )

        if should_validate:
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
            current_metric = metrics[metric_name]
            history["val_metrics"].append({"epoch": epoch + 1, **metrics})
            print(
                f"Validation | {metric_name}: {current_metric:.4f} | "
                f"Recall@{eval_k}: {metrics[f'Recall@{eval_k}']:.4f} | "
                f"MRR@{eval_k}: {metrics[f'MRR@{eval_k}']:.4f}"
            )

            if best_metric is None or current_metric > best_metric:
                best_metric = current_metric
                best_loss = avg_epoch_loss
                best_epoch = epoch + 1
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
                print(
                    f"New best checkpoint saved at epoch {best_epoch} "
                    f"by {metric_name}: {best_metric:.4f}\n"
                )
        elif avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch + 1
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch,
                best_loss,
                preprocessor,
            )
            print(
                f"New best checkpoint saved by train loss: "
                f"{best_loss:.4f}\n"
            )

    if return_history:
        history["best_metric"] = best_metric
        history["best_epoch"] = best_epoch
        history["metric_name"] = metric_name
        return best_checkpoint_path, history
    return best_checkpoint_path

import os
import torch
import torch.nn.functional as F
import torch.nn as nn


# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------

def in_batch_softmax_loss(user_vecs, item_vecs, temperature=0.05):
    """
    In-batch Softmax Loss.
    Temperature nhỏ hơn (0.05) → harder negatives → learning tốt hơn.
    """
    logits = torch.matmul(user_vecs, item_vecs.T) / temperature
    labels = torch.arange(user_vecs.size(0), device=user_vecs.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# CHECKPOINT UTILS
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, epoch, best_loss):
    """Lưu đầy đủ trạng thái: model, optimizer, epoch, best_loss."""
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    torch.save({
        "epoch":           epoch,
        "best_loss":       best_loss,
        "model_state":     base_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer, device):
    """Load checkpoint + xử lý DataParallel mismatch."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        start_epoch = checkpoint["epoch"] + 1
        best_loss   = checkpoint["best_loss"]
    else:
        print("  [Warning] Checkpoint cũ, chỉ load được model weights.")
        state_dict  = checkpoint
        start_epoch = 0
        best_loss   = float("inf")

    # Xử lý mismatch DataParallel / non-DataParallel
    is_parallel      = isinstance(model, nn.DataParallel)
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

    if is_parallel and not has_module_prefix:
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    elif not is_parallel and has_module_prefix:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(f"  ✓ Resumed from epoch {start_epoch} | best loss: {best_loss:.4f}")
    return start_epoch, best_loss


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

def train(model, train_loader, optimizer, device,
          epochs, checkpoint_dir, temperature=0.05, resume=False):
    """
    Huấn luyện model (chỉ track loss, không test metrics).
    Test metrics chỉ chạy 1 lần sau khi training xong.
    
    Args:
        resume: True → load best_checkpoint.pth và train tiếp
    
    Returns:
        str: đường dẫn best_checkpoint.pth
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")

    start_epoch = 0
    best_loss   = float("inf")

    # Resume nếu cần
    if resume:
        if os.path.exists(best_checkpoint_path):
            print("\nResuming training from checkpoint...")
            start_epoch, best_loss = load_checkpoint(
                best_checkpoint_path, model, optimizer, device
            )
        else:
            print("\n[Warning] Best checkpoint.pth not found. Train from scratch.")

    print("\n--- START TRAINING ---")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            user_inputs = {
                "user_id":    batch["user_id"].to(device),
                "gender":     batch["gender"].to(device),
                "occupation": batch["occupation"].to(device),
                "movie_hist": batch["user_hist"].to(device),
            }
            target_movies = batch["target_movie"].to(device)

            optimizer.zero_grad()
            user_vec, item_vec = model(user_inputs, target_movies)
            loss = in_batch_softmax_loss(user_vec, item_vec, temperature=temperature)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_batch_loss = total_loss / (batch_idx + 1)
            
            # In progress mỗi 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Batch Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_batch_loss:.4f}"
                )

        avg_epoch_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} complete — Train Loss: {avg_epoch_loss:.4f}")

        # ── Lưu best checkpoint theo training loss ───────────────────────────
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(best_checkpoint_path, model, optimizer, epoch, best_loss)
            print(f"  ✓ New best loss ({best_loss:.4f}) — saved checkpoint\n")
        else:
            print(f"  – No improvement (best loss: {best_loss:.4f})\n")

    print("--- TRAINING COMPLETE ---")
    return best_checkpoint_path
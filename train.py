import os
import torch
import torch.nn.functional as F
import torch.nn as nn


# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------

def in_batch_softmax_loss(user_vecs, item_vecs, item_freq=None, temperature=0.1):

    logits = torch.matmul(user_vecs, item_vecs.T) / temperature

    if item_freq is not None:
        log_freq = torch.log(item_freq.to(user_vecs.device) + 1e-8)
        logits = logits - log_freq.unsqueeze(0)

    labels = torch.arange(user_vecs.size(0)).to(user_vecs.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# CHECKPOINT UTILS
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, epoch, best_loss):
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    torch.save({
        "epoch":           epoch,
        "best_loss":       best_loss,
        "model_state":     base_model.state_dict(), # Luôn sạch sẽ
        "optimizer_state": optimizer.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer, device):
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

    # ── Xử lý mismatch DataParallel / non-DataParallel ───────────────────────
    is_parallel      = isinstance(model, torch.nn.DataParallel)
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

    if is_parallel and not has_module_prefix:
        # Checkpoint lưu không có DataParallel → thêm "module." vào key
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    elif not is_parallel and has_module_prefix:
        # Checkpoint lưu có DataParallel → bỏ "module." khỏi key
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(f"  ✓ Resumed from epoch {start_epoch} | best loss: {best_loss:.4f}")
    return start_epoch, best_loss


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

def train(model, train_loader, optimizer, device,
          epochs, checkpoint_dir, temperature=0.1, resume=False):
    """
    Huấn luyện model, hỗ trợ resume từ best_checkpoint.pth.

    Args:
        resume (bool): True → load best_checkpoint.pth và train tiếp.

    Returns:
        str: đường dẫn tới best_checkpoint.pth đã lưu.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")

    start_epoch = 0
    best_loss   = float("inf")

    # ── Resume ───────────────────────────────────────────────────────────────
    if resume:
        if os.path.exists(best_checkpoint_path):
            print("\nResuming training from checkpoint...")
            start_epoch, best_loss = load_checkpoint(
                best_checkpoint_path, model, optimizer, device
            )
        else:
            print("\n[Warning] Best checkpoint.pth not found. Train from sratch")

    # ── Training loop ─────────────────────────────────────────────────────────
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
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Batch {batch_idx+1}/{len(train_loader)} | "
                f"Batch Loss: {loss.item():.4f} | "
                f"Avg Loss: {avg_batch_loss:.4f}"
            )

        avg_epoch_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} complete — Avg Epoch Loss: {avg_epoch_loss:.4f}")

        # ── Lưu best checkpoint ───────────────────────────────────────────────
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(best_checkpoint_path, model, optimizer, epoch, best_loss)
            print(f"  ✓ New best loss ({best_loss:.4f}) — saved: {best_checkpoint_path}\n")
        else:
            print(f"  – No improvement (best: {best_loss:.4f})\n")

    print("--- TRAINING COMPLETE ---")
    return best_checkpoint_path
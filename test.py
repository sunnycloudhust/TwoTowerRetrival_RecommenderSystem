import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# METRICS COMPUTATION (OPTIMIZED)
# ---------------------------------------------------------------------------

def compute_all_metrics_optimized(targets, topk_indices, k=20):
    """Tính toán Recall, Precision, MRR, NDCG trong một vòng lặp."""
    recalls, precisions, mrrs, ndcgs = [], [], [], []
    top_k_matrix = topk_indices[:, :k] 
    
    for i in range(len(targets)):
        relevant = {targets[i]} if isinstance(targets[i], (int, np.integer)) else set(targets[i])
        if not relevant:
            recalls.append(0.0)
            precisions.append(0.0)
            mrrs.append(0.0)
            ndcgs.append(0.0)
            continue
            
        pred_row = top_k_matrix[i]
        hits = relevant & set(pred_row)
        
        # Recall & Precision
        recalls.append(len(hits) / len(relevant))
        precisions.append(len(hits) / k)
        
        # MRR & NDCG
        mrr, dcg = 0.0, 0.0
        for rank, item in enumerate(pred_row, 1):
            if item in relevant:
                if mrr == 0.0:
                    mrr = 1.0 / rank
                dcg += 1.0 / np.log2(rank + 1)
                
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
        mrrs.append(mrr)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        
    return {
        f"Recall@{k}": float(np.mean(recalls)),
        f"Precision@{k}": float(np.mean(precisions)),
        f"MRR@{k}": float(np.mean(mrrs)),
        f"NDCG@{k}": float(np.mean(ndcgs))
    }


# ---------------------------------------------------------------------------
# TOPK COMPUTATION (MEMORY TILING - OPTIMIZED)
# ---------------------------------------------------------------------------

def compute_topk_batched(user_vecs, item_vecs, k=20, batch_size=512):
    """Tính Top-K bằng memory tiling để tối ưu cache."""
    num_users = user_vecs.size(0)
    topk_indices_list = []
    item_vecs_t = item_vecs.T 

    with torch.no_grad():
        for start_idx in range(0, num_users, batch_size):
            end_idx = min(start_idx + batch_size, num_users)
            user_batch = user_vecs[start_idx:end_idx]
            batch_scores = torch.matmul(user_batch, item_vecs_t)
            _, batch_topk = torch.topk(batch_scores, k=k, dim=-1)
            topk_indices_list.append(batch_topk.cpu())

    return torch.cat(topk_indices_list, dim=0).numpy()


# ---------------------------------------------------------------------------
# EVALUATE FUNCTION
# ---------------------------------------------------------------------------

def evaluate(model, test_loader, device, checkpoint_path=None, k=20):
    """
    Evaluate model với checkpoint loading + tối ưu metrics.
    
    Args:
        checkpoint_path: Nếu có, load weights từ checkpoint trước evaluate
        k: Top-K để evaluate
    
    Returns:
        dict: Recall@k, Precision@k, MRR@k, NDCG@k
    """
    
    # ── Load checkpoint nếu có ──────────────────────────────────────────────
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract state_dict từ checkpoint
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
            epoch_info = checkpoint.get("epoch", -1)
            best_loss  = checkpoint.get("best_loss", float("inf"))
            print(f"  ✓ Loaded checkpoint from epoch {epoch_info+1} (best loss: {best_loss:.4f})")
        else:
            # Checkpoint cũ format
            state_dict = checkpoint
            print("  ✓ Loaded checkpoint (old format)")

        # Xử lý DataParallel mismatch
        is_parallel = isinstance(model, nn.DataParallel)
        has_module_prefix = any(key.startswith("module.") for key in state_dict.keys())

        if is_parallel and not has_module_prefix:
            state_dict = {"module." + key: value for key, value in state_dict.items()}
        elif not is_parallel and has_module_prefix:
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

        model.load_state_dict(state_dict)

    # ── Inference ───────────────────────────────────────────────────────────
    model.eval()
    all_user_vecs, all_item_vecs, all_targets = [], [], []

    print(f"\n[1/3] Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference", disable=False):
            user_inputs = {
                "user_id":    batch["user_id"].to(device),
                "gender":     batch["gender"].to(device),
                "occupation": batch["occupation"].to(device),
                "movie_hist": batch["user_hist"].to(device),
            }
            target_movies = batch["target_movie"].to(device)
            user_vec, item_vec = model(user_inputs, target_movies)
            
            all_user_vecs.append(user_vec.cpu())
            all_item_vecs.append(item_vec.cpu())
            all_targets.append(target_movies.cpu())

    user_vecs = torch.cat(all_user_vecs, dim=0)
    item_vecs = torch.cat(all_item_vecs, dim=0)
    targets   = torch.cat(all_targets,   dim=0).numpy()

    print(f"  User vectors shape: {user_vecs.shape}")
    print(f"  Item vectors shape: {item_vecs.shape}")

    # ── Tính Top-K ──────────────────────────────────────────────────────────
    print(f"\n[2/3] Computing Top-{k} rankings...")
    topk_indices = compute_topk_batched(user_vecs, item_vecs, k=k, batch_size=512)

    # ── Tính metrics ────────────────────────────────────────────────────────
    print(f"\n[3/3] Computing metrics...")
    metrics = compute_all_metrics_optimized(targets, topk_indices, k=k)

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS @{k}")
    print(f"{'='*50}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"{'='*50}\n")

    return metrics
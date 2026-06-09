import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm


def compute_all_metrics_optimized(targets, topk_indices, k=20):
    """Tính toán Recall, Precision, MRR, NDCG chuẩn cho Leave-One-Out."""
    recalls, precisions, mrrs, ndcgs = [], [], [], []
    top_k_matrix = topk_indices[:, :k]
    
    for i in range(len(targets)):
        target_item = targets[i]
        pred_row = top_k_matrix[i]
        
        hit = (target_item in pred_row)
        
        recalls.append(1.0 if hit else 0.0)
        precisions.append((1.0 / k) if hit else 0.0)
        
        mrr, ndcg = 0.0, 0.0
        if hit:
            rank = np.where(pred_row == target_item)[0][0] + 1
            mrr  = 1.0 / rank
            ndcg = 1.0 / np.log2(rank + 1)
            
        mrrs.append(mrr)
        ndcgs.append(ndcg)
        
    return {
        f"Recall@{k}":    float(np.mean(recalls)),
        f"Precision@{k}": float(np.mean(precisions)),
        f"MRR@{k}":       float(np.mean(mrrs)),
        f"NDCG@{k}":      float(np.mean(ndcgs))
    }

def compute_topk_batched(user_vecs, item_vecs, k=20, batch_size=512, seen_items=None):
    """
    Tính Top-K bằng Dot Product trên toàn bộ kho phim.
    
    Args:
        seen_items: list[set] — seen_items[i] là tập movie_id user i đã tương tác.
                    Nếu truyền vào, các phim này bị loại khỏi kết quả Top-K.
    """
    num_users  = user_vecs.size(0)
    item_vecs_t = item_vecs.T           # (64, num_movies) — transpose 1 lần dùng nhiều
    topk_list  = []

    with torch.no_grad():
        for start in range(0, num_users, batch_size):
            end         = min(start + batch_size, num_users)
            user_batch  = user_vecs[start:end]                          # (B, 64)
            batch_scores = torch.matmul(user_batch, item_vecs_t)        # (B, num_movies)

            # --- FIX 2: Mask các phim user đã xem ---
            # Gán điểm -inf để chúng không bao giờ lọt vào Top-K
            if seen_items is not None:
                for local_i, global_i in enumerate(range(start, end)):
                    for seen_id in seen_items[global_i]:
                        batch_scores[local_i, seen_id] = float("-inf")

            _, batch_topk = torch.topk(batch_scores, k=k, dim=-1)      # (B, k)
            topk_list.append(batch_topk.cpu())

    return torch.cat(topk_list, dim=0).numpy()                         # (num_users, k)



def evaluate(model, test_loader, genre_matrix, device, checkpoint_path=None, k=20, user_seen_dict=None):

    # 1. Load checkpoint 
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint

        is_parallel       = isinstance(model, nn.DataParallel)
        has_module_prefix = any(k_.startswith("module.") for k_ in state_dict.keys())
        if is_parallel and not has_module_prefix:
            state_dict = {"module." + k_: v for k_, v in state_dict.items()}
        elif not is_parallel and has_module_prefix:
            state_dict = {k_.replace("module.", "", 1): v for k_, v in state_dict.items()}

        model.load_state_dict(state_dict)
        print(f"  ✓ Model weights loaded from {checkpoint_path}")

    model.eval()
    
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    # -----------------------------------------------------------------------
    # 2. Xây dựng embedding cho TOÀN BỘ kho phim
    # -----------------------------------------------------------------------
    print(f"\n[1/3] Trích xuất embedding kho phim...")
    num_movies      = genre_matrix.shape[0]
    all_movie_ids   = torch.arange(num_movies, device=device)
    all_movie_genres = torch.tensor(genre_matrix, dtype=torch.long, device=device)

    global_item_vecs = []
    with torch.no_grad():
        for i in range(0, num_movies, 512):
            end_i    = min(i + 512, num_movies)
            # Dùng base_model.item_tower thay vì model.item_tower
            item_vec = base_model.item_tower(
                all_movie_ids[i:end_i],
                all_movie_genres[i:end_i]
            )
            global_item_vecs.append(item_vec.cpu())

    global_item_vecs = torch.cat(global_item_vecs, dim=0)   # (num_movies, 64)

    # -----------------------------------------------------------------------
    # 3. Trích xuất embedding user + thu thập seen_items
    # -----------------------------------------------------------------------
    print(f"[2/3] Trích xuất embedding người dùng tập Test...")
    all_user_vecs = []
    all_targets   = []
    all_seen_items = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            u_in = batch["user_inputs"]

            user_vec = base_model.user_tower(
                u_in["user_id"].to(device),
                u_in["gender"].to(device),
                u_in["age"].to(device),
                u_in["occupation"].to(device),
                u_in["movie_hist"].to(device)
            )
            all_user_vecs.append(user_vec.cpu())
            all_targets.append(batch["item_inputs"]["target_movie"])

            user_ids_np = u_in["user_id"].cpu().numpy()
            for uid in user_ids_np:
                seen = user_seen_dict.get(uid, set()) if user_seen_dict is not None else set()
                all_seen_items.append(seen)

    user_vecs = torch.cat(all_user_vecs, dim=0)   
    targets   = torch.cat(all_targets,   dim=0).numpy()

    # -----------------------------------------------------------------------
    # 4. Tính Top-K và Metrics
    # -----------------------------------------------------------------------
    print(f"[3/3] Đang tính toán Top-{k} và Metrics...")
    topk_indices = compute_topk_batched(
        user_vecs,
        global_item_vecs,
        k=k,
        seen_items=all_seen_items   # truyền seen_items vào để mask
    )
    metrics = compute_all_metrics_optimized(targets, topk_indices, k=k)

    print(f"\n{'='*40}")
    print(f"KẾT QUẢ ĐÁNH GIÁ (K={k})")
    print(f"{'='*40}")
    for name, val in metrics.items():
        print(f"  {name:15}: {val:.4f}")
    print(f"{'='*40}\n")

    return metrics
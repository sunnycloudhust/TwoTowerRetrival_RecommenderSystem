import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

# ---------------------------------------------------------------------------
# METRICS COMPUTATION
# ---------------------------------------------------------------------------

def compute_all_metrics_optimized(targets, topk_indices, k=20):
    """Tính toán Recall, Precision, MRR, NDCG chuẩn cho Leave-One-Out."""
    recalls, precisions, mrrs, ndcgs = [], [], [], []
    top_k_matrix = topk_indices[:, :k] 
    
    for i in range(len(targets)):
        target_item = targets[i]
        pred_row = top_k_matrix[i]
        
        # Kiểm tra hit
        hit = (target_item in pred_row)
        
        # Recall & Precision (Trong Leave-One-Out, len(relevant) luôn = 1)
        recalls.append(1.0 if hit else 0.0)
        precisions.append((1.0 / k) if hit else 0.0)
        
        # MRR & NDCG
        mrr, ndcg = 0.0, 0.0
        if hit:
            # Tìm vị trí của item đúng (rank bắt đầu từ 1)
            rank = np.where(pred_row == target_item)[0][0] + 1
            mrr = 1.0 / rank
            ndcg = 1.0 / np.log2(rank + 1)
            
        mrrs.append(mrr)
        ndcgs.append(ndcg) # IDCG luôn = 1 vì chỉ có 1 item đúng
        
    return {
        f"Recall@{k}": float(np.mean(recalls)),
        f"MRR@{k}": float(np.mean(mrrs)),
        f"NDCG@{k}": float(np.mean(ndcgs))
    }

# ---------------------------------------------------------------------------
# TOP-K COMPUTATION
# ---------------------------------------------------------------------------

def compute_topk_batched(user_vecs, item_vecs, k=20, batch_size=512):
    """Tính Top-K bằng Dot Product trên toàn bộ kho phim."""
    num_users = user_vecs.size(0)
    topk_indices_list = []
    item_vecs_t = item_vecs.T 

    with torch.no_grad():
        for start_idx in range(0, num_users, batch_size):
            end_idx = min(start_idx + batch_size, num_users)
            user_batch = user_vecs[start_idx:end_idx]
            
            # Tính scores: (Batch_size, 64) x (64, Num_Movies) -> (Batch_size, Num_Movies)
            batch_scores = torch.matmul(user_batch, item_vecs_t)
            _, batch_topk = torch.topk(batch_scores, k=k, dim=-1)
            topk_indices_list.append(batch_topk.cpu())

    return torch.cat(topk_indices_list, dim=0).numpy()

# ---------------------------------------------------------------------------
# EVALUATE FUNCTION
# ---------------------------------------------------------------------------

def evaluate(model, test_loader, genre_matrix, device, checkpoint_path=None, k=20):
    """
    Hàm đánh giá mô hình toàn diện.
    """
    # 1. Load weights nếu được yêu cầu
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        
        # Xử lý DataParallel mismatch
        is_parallel = isinstance(model, nn.DataParallel)
        has_module_prefix = any(key.startswith("module.") for key in state_dict.keys())
        if is_parallel and not has_module_prefix:
            state_dict = {"module." + key: value for key, value in state_dict.items()}
        elif not is_parallel and has_module_prefix:
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
            
        model.load_state_dict(state_dict)
        print(f"  ✓ Model weights loaded from {checkpoint_path}")

    model.eval()

    # 2. Xây dựng véc-tơ cho TOÀN BỘ KHO PHIM (Candidate Pool)
    print(f"\n[1/3] Trích xuất embedding kho phim...")
    num_movies = genre_matrix.shape[0]
    all_movie_ids = torch.arange(num_movies).to(device)
    all_movie_genres = torch.tensor(genre_matrix, dtype=torch.long).to(device)
    
    global_item_vecs = []
    with torch.no_grad():
        for i in range(0, num_movies, 512):
            end_i = min(i + 512, num_movies)
            # Chỉ sử dụng Tháp Item để lấy véc-tơ đại diện phim
            item_vec = model.item_tower(all_movie_ids[i:end_i], all_movie_genres[i:end_i])
            global_item_vecs.append(item_vec.cpu())
    
    global_item_vecs = torch.cat(global_item_vecs, dim=0) # (Total_Movies, 64)

    # 3. Trích xuất véc-tơ cho User trong tập Test
    print(f"[2/3] Trích xuất embedding người dùng tập Test...")
    all_user_vecs, all_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            u_in = batch["user_inputs"]
            # Chỉ sử dụng Tháp User
            user_vec = model.user_tower(
                u_in["user_id"].to(device), 
                u_in["gender"].to(device), 
                u_in["age"].to(device),
                u_in["occupation"].to(device), 
                u_in["movie_hist"].to(device)
            )
            all_user_vecs.append(user_vec.cpu())
            all_targets.append(batch["item_inputs"]["target_movie"])

    user_vecs = torch.cat(all_user_vecs, dim=0) # (Test_Users, 64)
    targets   = torch.cat(all_targets,   dim=0).numpy()

    # 4. Tính toán Rank và Metrics
    print(f"[3/3] Đang tính toán Top-{k} và Metrics...")
    topk_indices = compute_topk_batched(user_vecs, global_item_vecs, k=k)
    metrics = compute_all_metrics_optimized(targets, topk_indices, k=k)

    # In kết quả
    print(f"\n{'='*40}")
    print(f"KẾT QUẢ ĐÁNH GIÁ (K={k})")
    print(f"{'='*40}")
    for name, val in metrics.items():
        print(f"  {name:15}: {val:.4f}")
    print(f"{'='*40}\n")

    return metrics
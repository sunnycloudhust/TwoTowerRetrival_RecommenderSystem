import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. TỐI ƯU HÓA TÍNH TOÁN METRICS (Duyệt mảng 1 lần duy nhất)
# ---------------------------------------------------------------------------

def compute_all_metrics_optimized(targets, topk_indices, k=20):
    """
    Tính toán đồng thời Recall, Precision, MRR, NDCG từ kết quả Top-K.
    """
    recalls, precisions, mrrs, ndcgs = [], [], [], []
    top_k_matrix = topk_indices[:, :k] 
    
    for i in range(len(targets)):
        # Target có thể là một ID đơn lẻ hoặc một tập hợp
        relevant = {targets[i]} if isinstance(targets[i], (int, np.integer)) else set(targets[i])
        if not relevant:
            recalls.append(0.0); precisions.append(0.0); mrrs.append(0.0); ndcgs.append(0.0)
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
# 2. HÀM TÍNH TOP-K BẰNG KỸ THUẬT MEMORY TILING (Siêu tốc)
# ---------------------------------------------------------------------------

def compute_topk_batched(user_vecs, item_vecs, k=20, batch_size=512):
    """
    Nhân ma trận theo khối để tối ưu hóa CPU Cache và giảm thiểu độ trễ bộ nhớ.
    """
    num_users = user_vecs.size(0)
    topk_indices_list = []
    item_vecs_t = item_vecs.T 

    with torch.no_grad():
        # Duyệt qua từng khối User
        for start_idx in range(0, num_users, batch_size):
            end_idx = min(start_idx + batch_size, num_users)
            user_batch = user_vecs[start_idx:end_idx]
            
            # Nhân ma trận trên một khối nhỏ (Rất nhanh trên CPU)
            batch_scores = torch.matmul(user_batch, item_vecs_t)
            
            # Trích xuất Top-K ngay lập tức
            _, batch_topk = torch.topk(batch_scores, k=k, dim=-1)
            
            topk_indices_list.append(batch_topk.cpu())

    return torch.cat(topk_indices_list, dim=0).numpy()


# ---------------------------------------------------------------------------
# 3. HÀM EVALUATE CHÍNH
# ---------------------------------------------------------------------------

def evaluate(model, test_loader, device, checkpoint_path=None, k=20):
    """
    Hàm đánh giá mô hình Two-Tower tối ưu hóa cho cả GPU và CPU (MacBook).
    """
    # ── Nạp trọng số & Xử lý tương thích DataParallel ───────────────────────
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint

        is_parallel = isinstance(model, nn.DataParallel)
        has_module_prefix = any(key.startswith("module.") for key in state_dict.keys())

        if is_parallel and not has_module_prefix:
            state_dict = {"module." + key: value for key, value in state_dict.items()}
        elif not is_parallel and has_module_prefix:
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

        model.load_state_dict(state_dict)
        print(f"  ✓ Loaded best checkpoint: {checkpoint_path}")

    # ── Inference trích xuất Embedding ───────────────────────────────────────
    model.eval()
    all_user_vecs, all_item_vecs, all_targets = [], [], []

    print("\n[1/3] Đang trích xuất Embedding (Inference)...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
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

    # ── Tính toán Top-K (Sử dụng kỹ thuật Tiling tối ưu) ──────────────────────
    print(f"\n[2/3] Đang tính toán Top-{k} Rankings (Memory Tiling Mode)...")
    # Batch_size=512 là con số tối ưu để giữ dữ liệu trong CPU Cache
    topk_indices = compute_topk_batched(user_vecs, item_vecs, k=k, batch_size=512)

    # ── Tính toán Metrics ────────────────────────────────────────────────────
    print(f"\n[3/3] Đang tổng hợp Metrics...")
    metrics = compute_all_metrics_optimized(targets, topk_indices, k=k)

    print(f"\n--- KẾT QUẢ ĐÁNH GIÁ @{k} ---")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    return metrics
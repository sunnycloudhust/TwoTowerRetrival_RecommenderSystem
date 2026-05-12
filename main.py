import os
import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing.preprocessing import Preprocessor
from models.two_tower import TwoTowerModel
from dataset import get_dataloader
from train import train
from test import evaluate


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CONFIG = {
    "mode":         "test",        
    "epochs":       100,
    "batch_size":   1024,
    "lr":           0.01,
    "seq_len":      20,
    "temperature":  0.1,
    "eval_k":       20,
    "resume":       True,          
    "ratings_path": "dataset/ratings.dat",
    "users_path":   "dataset/users.dat",
    "checkpoint_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"),
    "final_model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "two_tower_model.pth"),
}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # 1. Device
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Device: {device} | GPUs: {num_gpus}")
    print(f"Running Mode: {CONFIG['mode'].upper()}") # <--- Thêm dòng in thông báo

    # 2. Preprocessing — chỉ encode, chưa build history
    print("\nPreprocessing data...")
    prep = Preprocessor(seq_len=CONFIG["seq_len"])
    ratings, users = prep.preprocess(CONFIG["ratings_path"], CONFIG["users_path"])

    num_users  = len(prep.user_encoder.classes_)
    num_movies = len(prep.movie_encoder.classes_)
    print(f"Users: {num_users} | Movies: {num_movies}")

    # 3. Split TRƯỚC khi build history (tránh data leak)
    print("\nSplitting data...")
    split_idx     = int(0.8 * len(ratings))
    train_ratings = ratings.iloc[:split_idx].reset_index(drop=True)
    test_ratings  = ratings.iloc[split_idx:].reset_index(drop=True)

    # 4. Build history CHỈ từ train_ratings
    print("Building user history from train set...")
    train_hist = prep.build_user_hist_array(train_ratings)

    # 5. DataLoaders
    print("Preparing DataLoaders...")
    # Nếu chỉ test, bạn vẫn cần train_loader nếu logic code của bạn yêu cầu, 
    # nhưng ta có thể tạo bình thường để luồng chuẩn bị dữ liệu không bị gãy.
    train_loader = get_dataloader(
        train_ratings, users, train_hist,
        batch_size=CONFIG["batch_size"], shuffle=True
    )
    test_loader = get_dataloader(
        test_ratings, users, train_hist,
        batch_size=CONFIG["batch_size"], shuffle=False
    )

    # 6. Model
    print("Initializing model...")
    model = TwoTowerModel(num_users=num_users, num_movies=num_movies)
    
    # --- LOGIC TỰ ĐỘNG THÍCH ỨNG SỐ LƯỢNG GPU ---
    if num_gpus > 1:
        print(f"  -> Phát hiện {num_gpus} GPUs. Tự động bật chế độ song song (DataParallel)...")
        model = nn.DataParallel(model)
    else:
        print(f"  -> Chạy trên chế độ Đơn thiết bị ({device}).")
        
    model.to(device)

    best_checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_checkpoint.pth")

    # -----------------------------------------------------------------------
    # PHÂN NHÁNH: TRAIN vs TEST
    # -----------------------------------------------------------------------
    if CONFIG["mode"] == "train":
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

        # 7. Train (resume nếu CONFIG["resume"] = True)
        best_checkpoint_path = train(
            model          = model,
            train_loader   = train_loader,
            optimizer      = optimizer,
            device         = device,
            epochs         = CONFIG["epochs"],
            checkpoint_dir = CONFIG["checkpoint_dir"],
            temperature    = CONFIG["temperature"],
            resume         = CONFIG["resume"],
        )

        # 8. Lưu final model
        torch.save(model.state_dict(), CONFIG["final_model_path"])
        print(f"\nFinal model saved: {CONFIG['final_model_path']}")

    # 9. Evaluate (Chạy khi mode là 'test' hoặc sau khi train xong)
    print("\n--- EVALUATING FROM BEST CHECKPOINT ---")
    if not os.path.exists(best_checkpoint_path):
        print(f"[ERROR] Không tìm thấy file checkpoint tại: {best_checkpoint_path}")
        return

    evaluate(
        model           = model,
        test_loader     = test_loader,
        device          = device,
        checkpoint_path = best_checkpoint_path,
        k               = CONFIG["eval_k"],
    )


if __name__ == "__main__":
    main()

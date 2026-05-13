import os
import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing.preprocessing import Preprocessor
from models.two_tower import TwoTowerModel
from dataset import get_dataloader
from train import train as train_model
from test import evaluate


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CONFIG = {
    "epochs":       100,
    "batch_size":   512,
    "lr":           0.001,
    "weight_decay": 1e-5,
    "seq_len":      20,
    "temperature":  0.05,
    "eval_k":       20,
    "resume":       False,
    "ratings_path": "dataset/ratings.dat",
    "users_path":   "dataset/users.dat",
    "checkpoint_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"),
    "final_model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "two_tower_model.pth"),
}


# ---------------------------------------------------------------------------
# HELPER: Load and prepare data once
# ---------------------------------------------------------------------------

class DataManager:

    def __init__(self):
        self.train_loader = None
        self.test_loader = None
        self.num_users = None
        self.num_movies = None
    
    def prepare(self):
        """Load data, preprocess, split, build history."""
        print("=" * 60)
        print("PREPARING DATA")
        print("=" * 60)
        
        # Preprocessing
        print("\nPreprocessing data...")
        prep = Preprocessor(seq_len=CONFIG["seq_len"])
        ratings, users = prep.preprocess(CONFIG["ratings_path"], CONFIG["users_path"])

        self.num_users  = int(len(prep.user_encoder.classes_))
        self.num_movies = int(len(prep.movie_encoder.classes_))
        print(f"Users: {self.num_users} | Movies: {self.num_movies}")

        # Split data
        print("\nSplitting data (80/20)...")
        split_idx     = int(0.8 * len(ratings))
        train_ratings = ratings.iloc[:split_idx].reset_index(drop=True)
        test_ratings  = ratings.iloc[split_idx:].reset_index(drop=True)
        print(f"Train: {len(train_ratings)} | Test: {len(test_ratings)}")

        # Build history từ train_ratings
        print("\nBuilding user history from train set...")
        train_hist = prep.build_user_hist_array(train_ratings)

        # DataLoaders
        print("\nPreparing DataLoaders...")
        self.train_loader = get_dataloader(
            train_ratings, users, train_hist,
            batch_size=CONFIG["batch_size"], shuffle=True
        )
        self.test_loader = get_dataloader(
            test_ratings, users, train_hist,
            batch_size=CONFIG["batch_size"], shuffle=False
        )
        print(f"Train batches: {len(self.train_loader)} | Test batches: {len(self.test_loader)}\n")


# ---------------------------------------------------------------------------
# TRAIN FUNCTION
# ---------------------------------------------------------------------------

def train():
    """Chạy training."""
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    # Setup
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Device: {device} | GPUs: {num_gpus}")
    print(f"Config: batch_size={CONFIG['batch_size']}, lr={CONFIG['lr']}, temperature={CONFIG['temperature']}\n")
    
    # Data
    data_manager = DataManager()
    data_manager.prepare()
    
    # Model
    print("Initializing model...")
    model = TwoTowerModel(num_users=data_manager.num_users, num_movies=data_manager.num_movies)
    if num_gpus > 1:
        print(f"Using DataParallel on {num_gpus} GPUs")
        model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Train
    best_checkpoint_path = train_model(
        model          = model,
        train_loader   = data_manager.train_loader,
        optimizer      = optimizer,
        device         = device,
        epochs         = CONFIG["epochs"],
        checkpoint_dir = CONFIG["checkpoint_dir"],
        temperature    = CONFIG["temperature"],
        resume         = CONFIG["resume"],
    )
    
    # Save final model
    torch.save(model.state_dict(), CONFIG["final_model_path"])
    print(f"\n✓ Final model saved: {CONFIG['final_model_path']}")
    
    return model, data_manager, best_checkpoint_path


# ---------------------------------------------------------------------------
# TEST FUNCTION
# ---------------------------------------------------------------------------

def test(model, data_manager, best_checkpoint_path):
    """Chạy evaluation."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluate(
        model           = model,
        test_loader     = data_manager.test_loader,
        device          = device,
        checkpoint_path = best_checkpoint_path,
        k               = CONFIG["eval_k"],
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # # ========== OPTION 1: Train + Test ==========
    model, data_manager, best_checkpoint_path = train()
    test(model, data_manager, best_checkpoint_path)
    
    # ========== OPTION 2: Chỉ Train ==========
    # model, data_manager, best_checkpoint_path = train()
    
    # ========== OPTION 3: Chỉ Test =========
    # device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_manager = DataManager()
    # data_manager.prepare()
    
    # print("Initializing model...")
    # model = TwoTowerModel(num_users=data_manager.num_users, num_movies=data_manager.num_movies)
    # num_gpus = torch.cuda.device_count()
    # if num_gpus > 1:
    #     model = nn.DataParallel(model)
    # model.to(device)
    
    # best_checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_checkpoint.pth")
    # test(model, data_manager, best_checkpoint_path)
    
    # print("\n" + "=" * 60)
    # print("✅ ALL DONE!")
    # print("=" * 60)

import torch
import torch.optim as optim
import torch.nn as nn
from preprocessing.preprocessing import Preprocessor
from dataset import get_dataloader
from models.two_tower import TwoTowerModel
from train import train
from test import evaluate

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DATA_PATHS = {
        "ratings": "dataset/ratings.dat",
        "users":   "dataset/users.dat",
        "movies":  "dataset/movies.dat"
    }

    HYPERPARAMS = {
        "seq_len":        10,
        "max_genres":     5,
        "batch_size":     500,
        "lr":             1e-3,
        "epochs":         200,
        "temperature":    0.07,
        "checkpoint_dir": "checkpoints"
    }

    # ---------------------------------------------------------
    # 2. Preprocessing
    # ---------------------------------------------------------
    print("\nPreprocessing data.")
    prep = Preprocessor(seq_len=HYPERPARAMS["seq_len"])
    train_df, test_df, genre_matrix = prep.preprocess(
        DATA_PATHS["ratings"],
        DATA_PATHS["users"],
        DATA_PATHS["movies"],
        max_genres=HYPERPARAMS["max_genres"]    # thêm dòng này
    )

    print(f"User count:  {prep.num_users}")
    print(f"Movie count: {prep.num_movies}")
    print(f"Train size:  {len(train_df)} | Test size: {len(test_df)}")

    # ---------------------------------------------------------
    # 3. DataLoaders
    # ---------------------------------------------------------
    print("\nDataloader initialization")
    train_loader = get_dataloader(
        train_df, genre_matrix,
        batch_size=HYPERPARAMS["batch_size"], shuffle=True
    )
    test_loader = get_dataloader(
        test_df, genre_matrix,
        batch_size=HYPERPARAMS["batch_size"], shuffle=False
    )

    # ---------------------------------------------------------
    # 4. Model & Optimizer
    # ---------------------------------------------------------
    print("\nModel initialization")
    model = TwoTowerModel(
        num_users=prep.num_users,
        num_movies=prep.num_movies,
        num_genres=prep.num_genres
    )
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs. DataParallel starts.")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=HYPERPARAMS["lr"], weight_decay=1e-5)

    # ---------------------------------------------------------
    # 5. Train
    # ---------------------------------------------------------
    best_model_path = train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=HYPERPARAMS["epochs"],
        checkpoint_dir=HYPERPARAMS["checkpoint_dir"],
        preprocessor=prep,
        temperature=HYPERPARAMS["temperature"],
        resume=True
    )
    # ---------------------------------------------------------
    # 6. Evaluate
    # ---------------------------------------------------------
    print("\nEvaluating on test set")
    metrics = evaluate(
        model=model,
        test_loader=test_loader,
        genre_matrix=genre_matrix,
        device=device,
        checkpoint_path=best_model_path,
        k=20
    )

    print("\n================================")

if __name__ == "__main__":
    main()
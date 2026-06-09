import os

import torch
import torch.optim as optim

from dataset import get_dataloader
from preprocessing.preprocessing import Preprocessor
from test import evaluate
from train import train
from utils import (
    build_model,
    build_user_seen_dict,
    plot_loss_histories,
    set_seed,
    split_train_validation,
    write_metrics_csv,
)


DATA_PATHS = {
    "ratings": "dataset/ratings.dat",
    "users": "dataset/users.dat",
    "movies": "dataset/movies.dat",
}

HYPERPARAMS = {
    "seq_len": 20,
    "max_genres": 5,
    "batch_size": 1024,
    "lr": 1e-2,
    "weight_decay": 1e-5,
    "epochs": 100,
    "temperature": 0.08,
    "eval_every": 1,
    "num_runs": 5,
    "seeds": [42, 2024, 3407, 12345, 98765],
}

EXPERIMENTS = [
    {
        "name": "temporal",
        "split": "temporal",
        "random_split": False,
        "k": 10,
    },
    {
        "name": "random",
        "split": "random",
        "random_split": True,
        "k": 20,
    },
]


def run_experiment(experiment, device):
    name = experiment["name"]
    split = experiment["split"]
    k = experiment["k"]

    print(f"\n{'#' * 72}")
    print(f"EXPERIMENT: {name} | split={split} | K={k}")
    print(f"{'#' * 72}")

    preprocessor = Preprocessor(seq_len=HYPERPARAMS["seq_len"])
    train_df, test_df, genre_matrix = preprocessor.preprocess(
        experiment["random_split"],
        DATA_PATHS["ratings"],
        DATA_PATHS["users"],
        DATA_PATHS["movies"],
        max_genres=HYPERPARAMS["max_genres"],
    )
    train_df, val_df = split_train_validation(train_df)
    print(f"User count:  {preprocessor.num_users}")
    print(f"Movie count: {preprocessor.num_movies}")
    print(
        f"Train size:  {len(train_df)} | "
        f"Validation size: {len(val_df)} | Test size: {len(test_df)}"
    )

    val_loader = get_dataloader(
        val_df,
        genre_matrix,
        batch_size=HYPERPARAMS["batch_size"],
        shuffle=False,
    )
    test_loader = get_dataloader(
        test_df,
        genre_matrix,
        batch_size=HYPERPARAMS["batch_size"],
        shuffle=False,
    )
    val_seen_dict = build_user_seen_dict(train_df)
    test_seen_dict = build_user_seen_dict(train_df, val_df)
    histories = []
    result_rows = []

    for run_id, seed in enumerate(HYPERPARAMS["seeds"], start=1):
        print(f"\n{'=' * 72}")
        print(f"{name} | run {run_id}/{HYPERPARAMS['num_runs']} | seed={seed}")
        print(f"{'=' * 72}")

        set_seed(seed)
        train_loader = get_dataloader(
            train_df,
            genre_matrix,
            batch_size=HYPERPARAMS["batch_size"],
            shuffle=True,
        )
        model = build_model(preprocessor, device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=HYPERPARAMS["lr"],
            weight_decay=HYPERPARAMS["weight_decay"],
        )

        checkpoint_dir = os.path.join(
            "checkpoints",
            name,
            f"run_{run_id}_seed_{seed}",
        )
        best_model_path, history = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epochs=HYPERPARAMS["epochs"],
            checkpoint_dir=checkpoint_dir,
            preprocessor=preprocessor,
            temperature=HYPERPARAMS["temperature"],
            resume=False,
            val_loader=val_loader,
            genre_matrix=genre_matrix,
            val_user_seen_dict=val_seen_dict,
            eval_k=k,
            eval_every=HYPERPARAMS["eval_every"],
            return_history=True,
        )
        history["run_id"] = run_id
        history["seed"] = seed
        histories.append(history)

        print(f"\nEvaluating {name} run {run_id}")
        metrics = evaluate(
            model=model,
            test_loader=test_loader,
            genre_matrix=genre_matrix,
            device=device,
            checkpoint_path=best_model_path,
            k=k,
            user_seen_dict=test_seen_dict,
        )
        result_rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "split": split,
                "k": k,
                "final_train_loss": history["train_loss"][-1],
                "best_val_ndcg": history["best_metric"],
                "best_epoch": history["best_epoch"],
                "best_checkpoint_path": best_model_path,
                **metrics,
            }
        )

        csv_path = os.path.join("outputs", name, "metrics_5_runs.csv")
        write_metrics_csv(result_rows, csv_path, k)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_path = os.path.join("outputs", name, "training_loss_5_runs.png")
    plot_loss_histories(
        histories,
        plot_path,
        title=f"Training Loss - {split.capitalize()} Split, K={k}",
    )
    print(f"\nSaved metrics: {csv_path}")
    print(f"Saved loss plot: {plot_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if len(HYPERPARAMS["seeds"]) != HYPERPARAMS["num_runs"]:
        raise ValueError("The number of seeds must equal num_runs.")

    for experiment in EXPERIMENTS:
        run_experiment(experiment, device)


if __name__ == "__main__":
    main()

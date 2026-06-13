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

BASE_HYPERPARAMS = {
    "seq_len": 20,
    "max_genres": 5,
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "temperature": 0.08,
    "eval_every": 1,
    "num_workers": 2,
    "dropout": 0.2,
    "grad_clip": 1.0,
    "num_runs": 5,
    "seeds": [42, 2024, 3407, 12345, 98765],
}

EXPERIMENTS = [
    {
        "k": 20,
        "random_split": True,
        "checkpoint_dir": "checkpoints/k20_5runs",
        "loss_plot_path": "outputs/k20/loss_curves_k20_5_runs.png",
        "final_results_csv_path": "outputs/k20/final_results_k20_5_runs.csv",
    },
    {
        "k": 10,
        "random_split": False,
        "checkpoint_dir": "checkpoints/k10_5runs",
        "loss_plot_path": "outputs/k10/loss_curves_k10_5_runs.png",
        "final_results_csv_path": "outputs/k10/final_results_k10_5_runs.csv",
    },
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seeds = BASE_HYPERPARAMS["seeds"][:BASE_HYPERPARAMS["num_runs"]]
    if len(seeds) != BASE_HYPERPARAMS["num_runs"]:
        raise ValueError("BASE_HYPERPARAMS['seeds'] must contain num_runs seeds.")

    for experiment in EXPERIMENTS:
        k = experiment["k"]
        split_name = "random" if experiment["random_split"] else "temporal"
        loss_histories = []
        final_rows = []

        print(f"\n{'#' * 72}")
        print(
            f"START EXPERIMENT | split={split_name} | K={k} | "
            f"runs={BASE_HYPERPARAMS['num_runs']}"
        )
        print(f"{'#' * 72}")

        print("\nPreprocessing data")
        prep = Preprocessor(seq_len=BASE_HYPERPARAMS["seq_len"])
        train_df, test_df, genre_matrix = prep.preprocess(
            experiment["random_split"],
            DATA_PATHS["ratings"],
            DATA_PATHS["users"],
            DATA_PATHS["movies"],
            max_genres=BASE_HYPERPARAMS["max_genres"],
        )
        train_df, val_df = split_train_validation(train_df)

        print(f"User count:  {prep.num_users}")
        print(f"Movie count: {prep.num_movies}")
        print(
            f"Train size: {len(train_df)} | Val size: {len(val_df)} | "
            f"Test size: {len(test_df)}"
        )

        val_loader = get_dataloader(
            val_df,
            genre_matrix,
            batch_size=BASE_HYPERPARAMS["batch_size"],
            shuffle=False,
            num_workers=BASE_HYPERPARAMS["num_workers"],
        )
        test_loader = get_dataloader(
            test_df,
            genre_matrix,
            batch_size=BASE_HYPERPARAMS["batch_size"],
            shuffle=False,
            num_workers=BASE_HYPERPARAMS["num_workers"],
        )
        val_seen_dict = build_user_seen_dict(train_df)
        test_seen_dict = build_user_seen_dict(train_df, val_df)

        for run_id, seed in enumerate(seeds, start=1):
            print(f"\n{'=' * 70}")
            print(f"K={k} | RUN {run_id}/{BASE_HYPERPARAMS['num_runs']} | seed={seed}")
            print(f"{'=' * 70}")

            set_seed(seed)
            train_loader = get_dataloader(
                train_df,
                genre_matrix,
                batch_size=BASE_HYPERPARAMS["batch_size"],
                shuffle=True,
                num_workers=BASE_HYPERPARAMS["num_workers"],
            )

            model = build_model(prep, device)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=BASE_HYPERPARAMS["lr"],
                weight_decay=BASE_HYPERPARAMS["weight_decay"],
            )

            run_checkpoint_dir = os.path.join(
                experiment["checkpoint_dir"],
                f"run_{run_id}_seed_{seed}",
            )
            best_model_path, history = train(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epochs=BASE_HYPERPARAMS["epochs"],
                checkpoint_dir=run_checkpoint_dir,
                preprocessor=prep,
                temperature=BASE_HYPERPARAMS["temperature"],
                resume=False,
                val_loader=val_loader,
                genre_matrix=genre_matrix,
                val_user_seen_dict=val_seen_dict,
                eval_k=k,
                eval_every=BASE_HYPERPARAMS["eval_every"],
                return_history=True,
            )

            history["run_id"] = run_id
            history["seed"] = seed
            loss_histories.append(history)

            print(f"\nEvaluating K={k} run {run_id} on test set")
            metrics = evaluate(
                model=model,
                test_loader=test_loader,
                genre_matrix=genre_matrix,
                device=device,
                checkpoint_path=best_model_path,
                k=k,
                user_seen_dict=test_seen_dict,
            )

            final_rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "split": split_name,
                    "k": k,
                    "final_train_loss": history["train_loss"][-1],
                    "best_val_ndcg": history["best_metric"],
                    "best_epoch": history["best_epoch"],
                    "best_checkpoint_path": best_model_path,
                    **metrics,
                }
            )
            write_metrics_csv(final_rows, experiment["final_results_csv_path"], k)

            if device.type == "cuda":
                torch.cuda.empty_cache()

        plot_loss_histories(
            loss_histories,
            experiment["loss_plot_path"],
            title=f"Training Loss by Epoch Across 5 Runs (K={k})",
        )

        print(f"\nK={k} summary:")
        recall_key = f"Recall@{k}"
        mrr_key = f"MRR@{k}"
        ndcg_key = f"NDCG@{k}"
        for row in final_rows:
            print(
                f"Run {row['run_id']} seed={row['seed']} | "
                f"final_loss={row['final_train_loss']:.4f} | "
                f"{recall_key}: {row[recall_key]:.4f} | "
                f"{mrr_key}: {row[mrr_key]:.4f} | "
                f"{ndcg_key}: {row[ndcg_key]:.4f}"
            )


if __name__ == "__main__":
    main()

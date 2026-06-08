import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn

from models.two_tower import TwoTowerModel


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def split_train_validation(train_df):
    user_counts = train_df.groupby("user_id").size()
    eligible_users = user_counts[user_counts >= 2].index
    val_idx = (
        train_df[train_df["user_id"].isin(eligible_users)]
        .groupby("user_id")
        .tail(1)
        .index
    )
    val_df = train_df.loc[val_idx].copy().reset_index(drop=True)
    train_df = train_df.drop(index=val_idx).reset_index(drop=True)
    return train_df, val_df


def build_user_seen_dict(*dfs):
    user_seen_dict = {}

    for df in dfs:
        if df is None or len(df) == 0:
            continue

        for user_id, group in df.groupby("user_id"):
            seen_movies = user_seen_dict.setdefault(user_id, set())
            seen_movies.update(group["target_movie"].astype(int).tolist())

            first_history = group.iloc[0]["history"]
            seen_movies.update(int(movie_id) for movie_id in first_history if movie_id != 0)

    return user_seen_dict


def build_model(prep, hyperparams, device):
    model = TwoTowerModel(
        num_users=prep.num_users,
        num_movies=prep.num_movies,
        num_genres=prep.num_genres,
        num_genders=prep.num_genders,
        num_ages=prep.num_ages,
        num_occupations=prep.num_occupations,
        max_seq_len=prep.seq_len,
        dropout=hyperparams["dropout"],
    )

    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs. DataParallel starts.")
        model = nn.DataParallel(model)

    return model.to(device)


def plot_loss_histories(histories, output_path, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib to plot loss curves: pip install matplotlib") from exc

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for history in histories:
        losses = history["train_loss"]
        epochs = range(1, len(losses) + 1)
        label = f"Run {history['run_id']} (seed={history['seed']})"
        plt.plot(epochs, losses, linewidth=1.8, label=label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Loss plot saved to: {output_path}")


def build_final_result_row(run_id, seed, history, metrics, k, best_model_path):
    return {
        "run_id": run_id,
        "seed": seed,
        "k": k,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "best_val_metric_name": history["metric_name"],
        "best_val_metric": history["best_metric"],
        f"Recall@{k}": metrics[f"Recall@{k}"],
        f"Precision@{k}": metrics[f"Precision@{k}"],
        f"MRR@{k}": metrics[f"MRR@{k}"],
        f"NDCG@{k}": metrics[f"NDCG@{k}"],
        "best_checkpoint_path": best_model_path,
    }


def write_final_results_csv(rows, output_path, k):
    if not rows:
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "run_id",
        "seed",
        "k",
        "final_train_loss",
        "best_val_metric_name",
        "best_val_metric",
        f"Recall@{k}",
        f"Precision@{k}",
        f"MRR@{k}",
        f"NDCG@{k}",
        "best_checkpoint_path",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Final results CSV saved to: {output_path}")

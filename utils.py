import csv
import os
import random

import matplotlib
import numpy as np
import torch
import torch.nn as nn

from models.two_tower import TwoTowerModel


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(preprocessor, device):
    model = TwoTowerModel(
        num_users=preprocessor.num_users,
        num_movies=preprocessor.num_movies,
        num_genres=preprocessor.num_genres,
    )
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs. DataParallel starts.")
        model = nn.DataParallel(model)
    return model.to(device)


def split_train_validation(train_df):
    group_sizes = train_df.groupby("user_id")["user_id"].transform("size")
    eligible_df = train_df[group_sizes >= 2]
    val_indices = eligible_df.groupby("user_id", sort=False).tail(1).index
    val_df = train_df.loc[val_indices].copy().reset_index(drop=True)
    train_df = train_df.drop(index=val_indices).reset_index(drop=True)
    return train_df, val_df


def build_user_seen_dict(*dataframes):
    user_seen_dict = {}
    for dataframe in dataframes:
        for user_id, group in dataframe.groupby("user_id"):
            seen_movies = user_seen_dict.setdefault(user_id, set())
            seen_movies.update(group["target_movie"].tolist())
            for history in group["history"]:
                seen_movies.update(movie_id for movie_id in history if movie_id != 0)
    return user_seen_dict


def plot_loss_histories(histories, output_path, title):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for history in histories:
        losses = history["train_loss"]
        epochs = range(1, len(losses) + 1)
        plt.plot(
            epochs,
            losses,
            label=f"Run {history['run_id']} (seed={history['seed']})",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_metrics_csv(rows, output_path, k):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metric_names = [
        f"Recall@{k}",
        f"Precision@{k}",
        f"MRR@{k}",
        f"NDCG@{k}",
    ]
    fieldnames = [
        "run_id",
        "seed",
        "split",
        "k",
        "final_train_loss",
        "best_val_ndcg",
        "best_epoch",
        "best_checkpoint_path",
        *metric_names,
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

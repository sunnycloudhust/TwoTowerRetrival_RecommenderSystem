import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def __init__(self, seq_len=20, num_neg=4):
        self.seq_len = seq_len
        self.num_neg = num_neg
    
        # encoders
        self.user_enc = LabelEncoder()
        self.movie_enc = LabelEncoder()
        self.gender_enc = LabelEncoder()
        self.age_enc = LabelEncoder()
        self.job_enc = LabelEncoder()

        self.num_users = None
        self.num_movies = None

    # ---------- Load data ----------
    def load(self, ratings_path, users_path):
        ratings = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )

        users = pd.read_csv(
            users_path,
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zip"]
        )

        return ratings, users

    # ---------- Merge & filter ----------
    def merge_and_filter(self, ratings, users):
        ratings = ratings[ratings["rating"] > 3]

        df = ratings.merge(users, on="user_id")
        df = df.sort_values(["user_id", "timestamp"])

        return df

    # ---------- Encode ----------
    def encode(self, df):
        df["user_id"] = self.user_enc.fit_transform(df["user_id"])
        df["movie_id"] = self.movie_enc.fit_transform(df["movie_id"])

        df["gender"] = self.gender_enc.fit_transform(df["gender"])
        df["age"] = self.age_enc.fit_transform(df["age"])
        df["occupation"] = self.job_enc.fit_transform(df["occupation"])

        self.num_users = df["user_id"].nunique()
        self.num_movies = df["movie_id"].nunique()

        return df

    # ---------- Build sequences ----------
    def build_sequences(self, df):
        user_sequences = {}
        user_features = {}

        for uid, group in df.groupby("user_id"):
            movies = group["movie_id"].tolist()

            user_sequences[uid] = movies

            # lấy feature của user (giống nhau cho mọi dòng)
            user_features[uid] = {
                "gender": group["gender"].iloc[0],
                "age": group["age"].iloc[0],
                "job": group["occupation"].iloc[0],
            }

        return user_sequences, user_features

    # ---------- Create samples ----------
    def create_samples(self, user_sequences, user_features):
        user_ids, seqs, targets = [], [], []
        genders, ages, jobs = [], [], []

        for uid, movies in user_sequences.items():
            for i in range(1, len(movies)):
                hist = movies[max(0, i - self.seq_len):i]
                target = movies[i]

                user_ids.append(uid)
                seqs.append(hist)
                targets.append(target)

                genders.append(user_features[uid]["gender"])
                ages.append(user_features[uid]["age"])
                jobs.append(user_features[uid]["job"])

        return user_ids, seqs, targets, genders, ages, jobs

    # ---------- Pre-padding ----------
    def pad_pre(self, sequences):
        padded = []

        for seq in sequences:
            if len(seq) > self.seq_len:
                seq = seq[-self.seq_len:]
            else:
                pad_len = self.seq_len - len(seq)
                seq = [0] * pad_len + seq

            padded.append(seq)

        return torch.tensor(padded, dtype=torch.long)

    # ---------- Negative sampling ----------
    def negative_sampling(self, user_ids, seqs, targets, genders, ages, jobs):
        new_u, new_s, new_t = [], [], []
        new_g, new_a, new_j = [], [], []
        labels = []

        for u, s, t, g, a, j in zip(user_ids, seqs, targets, genders, ages, jobs):
            # positive
            new_u.append(u)
            new_s.append(s)
            new_t.append(t)
            new_g.append(g)
            new_a.append(a)
            new_j.append(j)
            labels.append(1)

            # negatives
            for _ in range(self.num_neg):
                neg = np.random.randint(0, self.num_movies)
                while neg == t:
                    neg = np.random.randint(0, self.num_movies)

                new_u.append(u)
                new_s.append(s)
                new_t.append(neg)
                new_g.append(g)
                new_a.append(a)
                new_j.append(j)
                labels.append(0)

        return (
            torch.tensor(new_u),
            torch.stack(new_s),
            torch.tensor(new_t),
            torch.tensor(new_g),
            torch.tensor(new_a),
            torch.tensor(new_j),
            torch.tensor(labels, dtype=torch.float32),
        )

    # ---------- Full pipeline ----------
    def process(self, ratings_path, users_path):
        ratings, users = self.load(ratings_path, users_path)
        df = self.merge_and_filter(ratings, users)
        df = self.encode(df)

        user_sequences, user_features = self.build_sequences(df)

        user_ids, seqs, targets, genders, ages, jobs = self.create_samples(
            user_sequences, user_features
        )

        seqs = self.pad_pre(seqs)

        return self.negative_sampling(
            user_ids, seqs, targets, genders, ages, jobs
        )
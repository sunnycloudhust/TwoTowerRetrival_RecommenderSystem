# NOTE: preprocessing.py prepares data for training a recommendation model.
# The output of this module includes:
# 1. Encoded ratings data:
#    (user_id, movie_id, rating, timestamp) where user_id and movie_id are mapped to integer indices.
# 2. User feature table:
#    user_id (encoded), gender, age, occupation.
# 3. User history sequences:
#    fixed-length sequences of positively interacted movies (rating >= 3),
#    used as input for the User Tower in a Two-Tower recommendation model.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def __init__(self, seq_len=20):
        self.seq_len = seq_len
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()

    def load_data(self, ratings_path, users_path): #this function reads the data of 2 files: ratings and users
        ratings = pd.read_csv(ratings_path, sep="::", engine="python",
                              names=["user_id", "movie_id", "rating", "timestamp"])

        users = pd.read_csv(users_path, sep="::", engine="python",
                            names=["user_id", "gender", "age", "occupation", "zip"])
        return ratings, users

    def encode_ids(self, ratings, users): #this function encodes user_id and movie_id into continuous ranges
        # sort trước
        ratings = ratings.sort_values("timestamp").reset_index(drop=True)

        # fit trên ratings
        ratings["user_id"] = self.user_encoder.fit_transform(ratings["user_id"])
        ratings["movie_id"] = self.movie_encoder.fit_transform(ratings["movie_id"])

        # chỉ giữ user có trong ratings
        users = users[users["user_id"].isin(self.user_encoder.classes_)]

        # transform users
        users["user_id"] = self.user_encoder.transform(users["user_id"])

        return ratings, users

    def build_user_history(self, ratings): #this function only takes the rating >= 3 and output the history array of that user, for ex: user 0 → [1, 5, 9]
        # hyperparameter set
        ratings = ratings[ratings["rating"] >= 3]

        # sort by time
        ratings = ratings.sort_values("timestamp")
        user_hist = ratings.groupby("user_id")["movie_id"].apply(list)
        return user_hist

    def pad_sequence(self, seq): #this function normalizes the length of the user_history_sequence to len=20
        if len(seq) >= self.seq_len:
            return seq[-self.seq_len:]
        else:
            return [0] * (self.seq_len - len(seq)) + seq

    def preprocess(self, ratings_path, users_path): #this is the complete pipeline for the works above
        ratings, users = self.load_data(ratings_path, users_path)
        ratings, users = self.encode_ids(ratings, users)
        user_hist = self.build_user_history(ratings)
        # padding
        user_hist = user_hist.apply(self.pad_sequence)
        #convert to numpy
        user_hist_array = np.stack(user_hist.values)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def __init__(self, seq_len=20):
        self.seq_len = seq_len
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()

    def load_data(self, ratings_path, users_path):
        ratings = pd.read_csv(ratings_path, sep="::", engine="python",
                              names=["user_id", "movie_id", "rating", "timestamp"])
        users = pd.read_csv(users_path, sep="::", engine="python",
                            names=["user_id", "gender", "age", "occupation", "zip"])
        return ratings, users

    def encode_ids(self, ratings, users):
        ratings = ratings.sort_values("timestamp").reset_index(drop=True)

        # Encode user_id và movie_id (không +1 ở đây, để padding_idx=0 tự xử lý)
        ratings["user_id"] = self.user_encoder.fit_transform(ratings["user_id"])
        ratings["movie_id"] = self.movie_encoder.fit_transform(ratings["movie_id"]) + 1  # +1 để dành 0 cho padding

        users = users[users["user_id"].isin(self.user_encoder.classes_)]
        users["user_id"] = self.user_encoder.transform(users["user_id"])
        
        # Encode Gender "M"/"F" -> 0/1
        users["gender"] = self.gender_encoder.fit_transform(users["gender"])
        return ratings, users

    def pad_sequence(self, seq):
        """Pad sequence về độ dài seq_len, padding value = 0."""
        if len(seq) >= self.seq_len:
            return seq[-self.seq_len:]
        else:
            return [0] * (self.seq_len - len(seq)) + seq

    def build_user_hist_array(self, ratings_df):
        """Build user history array từ ratings_df được truyền vào."""
        num_users = len(self.user_encoder.classes_)
        
        # Lọc rating >= 3, sắp xếp theo time
        pos_ratings = ratings_df[ratings_df["rating"] >= 3].sort_values("timestamp")
        user_hist_dict = pos_ratings.groupby("user_id")["movie_id"].apply(list).to_dict()
        
        hist_list = []
        for i in range(num_users):
            seq = user_hist_dict.get(i, [])
            hist_list.append(self.pad_sequence(seq))
        return np.array(hist_list)

    def preprocess(self, ratings_path, users_path):
        """Load + encode IDs. Không build history ở đây để tránh leak."""
        ratings, users = self.load_data(ratings_path, users_path)
        ratings, users = self.encode_ids(ratings, users)
        return ratings, users
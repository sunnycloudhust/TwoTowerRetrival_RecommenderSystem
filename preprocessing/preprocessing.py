import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self, seq_len=20, max_genres=5):
        """
        Args:
            seq_len (int):    Độ dài tối đa chuỗi lịch sử xem phim.
            max_genres (int): Số lượng thể loại tối đa mỗi phim (default sẽ bị
                              override bởi tham số max_genres trong preprocess()).
        """
        self.seq_len        = seq_len
        self.max_genres     = max_genres
        self.user_encoder   = LabelEncoder()
        self.movie_encoder  = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.age_encoder    = LabelEncoder()
        self.occ_encoder    = LabelEncoder()
        self.num_users  = 0
        self.num_movies = 0
        self.num_genres = 0
        self.genre2id   = {}

    # -----------------------------------------------------------------------
    # Load raw data
    # -----------------------------------------------------------------------
    def load_users_data(self, users_path):
        return pd.read_csv(
            users_path, sep="::", engine="python",
            names=["user_id", "gender", "age", "occupation", "zip"]
        )

    def load_movies_data(self, movies_path):
        return pd.read_csv(
            movies_path, sep="::", engine="python",
            names=["movie_id", "title", "genres"], encoding="latin-1"
        )

    def load_ratings_data(self, ratings_path):
        return pd.read_csv(
            ratings_path, sep="::", engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )

    # -----------------------------------------------------------------------
    # Genre matrix
    # -----------------------------------------------------------------------
    def build_genre_matrix(self, movies_df, max_genres=5):
        """
        Xử lý đa thể loại và tạo ma trận tra cứu (num_movies, max_genres).
        Giá trị 0 là padding.
        """
        self.max_genres = max_genres

        # Xây dựng từ điển genre → id (bắt đầu từ 1, 0 dành cho padding)
        unique_genres = set()
        for g_str in movies_df["genres"]:
            unique_genres.update(g_str.split("|"))
        self.genre2id   = {g: i + 1 for i, g in enumerate(sorted(unique_genres))}
        self.num_genres = len(self.genre2id) + 1

        def encode_multi_genres(g_str):
            ids = [self.genre2id[g] for g in g_str.split("|")]
            if len(ids) >= self.max_genres:
                return ids[:self.max_genres]
            return ids + [0] * (self.max_genres - len(ids))

        movies = movies_df[movies_df["movie_id"].isin(self.movie_encoder.classes_)].copy()
        movies["movie_id_encoded"] = self.movie_encoder.transform(movies["movie_id"]) + 1

        genre_matrix = np.zeros((self.num_movies, self.max_genres), dtype=np.int64)
        for _, row in movies.iterrows():
            genre_matrix[row["movie_id_encoded"]] = encode_multi_genres(row["genres"])

        return genre_matrix

    # -----------------------------------------------------------------------
    # ID encoding
    # -----------------------------------------------------------------------
    def encode_ids(self, ratings_df):
        """Mã hóa User ID và Movie ID thành số nguyên liên tiếp."""
        ratings_df["user_id"]  = self.user_encoder.fit_transform(ratings_df["user_id"])
        self.num_users         = len(self.user_encoder.classes_)

        # Movie ID dịch +1: ID 0 dành cho padding trong sequence
        ratings_df["movie_id"] = self.movie_encoder.fit_transform(ratings_df["movie_id"]) + 1
        self.num_movies        = len(self.movie_encoder.classes_) + 1

        return ratings_df

    # -----------------------------------------------------------------------
    # User features
    # -----------------------------------------------------------------------
    def process_user_features(self, users_df):
        """Mã hóa các thuộc tính phụ của User."""
        users             = users_df[users_df["user_id"].isin(self.user_encoder.classes_)].copy()
        users["user_id"]  = self.user_encoder.transform(users["user_id"])
        users["gender"]   = self.gender_encoder.fit_transform(users["gender"])
        users["age"]      = self.age_encoder.fit_transform(users["age"])
        users["occupation"] = self.occ_encoder.fit_transform(users["occupation"])
        return users[["user_id", "gender", "age", "occupation"]]

    # -----------------------------------------------------------------------
    # Sequence padding
    # -----------------------------------------------------------------------
    def pad_sequence(self, seq):
        """Pre-padding về độ dài seq_len, padding value = 0."""
        if len(seq) >= self.seq_len:
            return seq[-self.seq_len:]
        return [0] * (self.seq_len - len(seq)) + seq

    # -----------------------------------------------------------------------
    # Sequential data generation  (Leave-One-Out)
    # -----------------------------------------------------------------------
    def generate_sequential_data(self, ratings_df):
        """
        Tạo train/test samples theo chuẩn Leave-One-Out:
        - Train: tất cả vị trí 1..n-2
        - Test:  vị trí cuối cùng (n-1)
        
        Chỉ giữ rating >= 3 làm tương tác tích cực.
        User có < 3 tương tác bị loại (không đủ để tạo ít nhất 1 train + 1 test sample).
        """
        pos_ratings = (
            ratings_df[ratings_df["rating"] >= 3]
            .sort_values(["user_id", "timestamp"])
        )

        train_samples, test_samples = [], []

        for user_id, group in pos_ratings.groupby("user_id"):
            movie_ids = group["movie_id"].tolist()
            ratings   = group["rating"].tolist()
            n         = len(movie_ids)

            if n < 3:
                continue

            for i in range(1, n):
                history_padded = self.pad_sequence(movie_ids[:i])
                sample = {
                    "user_id":      user_id,
                    "history":      history_padded,
                    "target_movie": movie_ids[i],
                    "rating":       ratings[i],
                }
                if i == n - 1:
                    test_samples.append(sample)
                else:
                    train_samples.append(sample)

        return pd.DataFrame(train_samples), pd.DataFrame(test_samples)

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------
    def preprocess(self, ratings_path, users_path, movies_path, max_genres=5):
        """
        Hàm thực thi chính.

        Args:
            max_genres (int): Truyền từ HYPERPARAMS để đồng bộ với toàn bộ pipeline.
                              --- FIX: Nhận tham số thay vì hardcode bên trong ---

        Returns:
            train_df, test_df, genre_matrix
        """
        # 1. Load
        ratings_raw = self.load_ratings_data(ratings_path)
        users_raw   = self.load_users_data(users_path)
        movies_raw  = self.load_movies_data(movies_path)

        # 2. Encode IDs
        ratings_encoded = self.encode_ids(ratings_raw)

        # 3. Genre matrix  — dùng max_genres từ tham số, không hardcode
        genre_matrix = self.build_genre_matrix(movies_raw, max_genres=max_genres)

        # 4. User features
        users_processed = self.process_user_features(users_raw)

        # 5. Train / Test split
        train_df, test_df = self.generate_sequential_data(ratings_encoded)

        # 6. Merge side info
        train_df = train_df.merge(users_processed, on="user_id", how="left")
        test_df  = test_df.merge(users_processed,  on="user_id", how="left")

        return train_df, test_df, genre_matrix
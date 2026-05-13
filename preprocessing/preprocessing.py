import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self, seq_len=20, max_genres=2):
        """
        Args:
            seq_len (int): Độ dài tối đa của chuỗi lịch sử xem phim (Mặc định: 20).
            max_genres (int): Số lượng thể loại cho mỗi phim (Mặc định: 2).
        """
        self.seq_len = seq_len
        self.max_genres = max_genres
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.age_encoder = LabelEncoder()
        self.occ_encoder = LabelEncoder()
        self.num_users = 0
        self.num_movies = 0
        self.num_genres = 0
        self.genre2id = {}
    ############################## Load data of the dataset #################################
    def load_users_data(self, users_path):
        users = pd.read_csv(users_path, sep="::", engine="python",
                            names=["user_id", "gender", "age", "occupation", "zip"])
        return users
    def load_movies_data(self, movies_path):
        return pd.read_csv(movies_path, sep="::", engine="python", 
                           names=["movie_id", "title", "genres"], encoding="latin-1")

    def load_ratings_data(self, ratings_path):
        return pd.read_csv(ratings_path, sep="::", engine="python",
                           names=["user_id", "movie_id", "rating", "timestamp"])
    ########################################################################################
    def build_genre_matrix(self, movies_df, max_genres=5):
        """
        Xử lý đa thể loại từ DataFrame movies gốc và tạo ma trận tra cứu đệm 0 an toàn.
        
        Args:
            movies_df (pd.DataFrame): Bảng phim gốc từ hàm load_movies_data.
            max_genres (int): Số lượng thể loại tối đa giữ lại cho mỗi phim.
        Returns:
            np.ndarray: Ma trận tra cứu ID thể loại, kích thước (num_movies, max_genres).
        """
        self.max_genres = max_genres
        
        # 1. Thu thập toàn bộ các thể loại đơn lẻ xuất hiện để xây dựng từ điển
        unique_genres = set()
        for g_str in movies_df["genres"]:
            unique_genres.update(g_str.split("|"))
        
        # Gán ID thể loại bắt đầu từ 1. ID 0 được giữ lại làm giá trị đệm (Padding)
        self.genre2id = {g: i + 1 for i, g in enumerate(sorted(unique_genres))}
        self.num_genres = len(self.genre2id) + 1 

        # Hàm trợ giúp nội bộ: Tách chuỗi và đệm 0 ở cuối (Post-padding)
        def encode_multi_genres(g_str):
            ids = [self.genre2id[g] for g in g_str.split("|")]
            if len(ids) >= self.max_genres:
                return ids[:self.max_genres]
            return ids + [0] * (self.max_genres - len(ids))
            
        # 2. Căn chỉnh Movie ID:
        # Lọc các phim hợp lệ xuất hiện trong tập ratings và map sang ID đã encode.
        # Bắt buộc +1 để khớp với logic dịch ID (tránh ghi đè vào hàng 0 của sequence padding).
        movies = movies_df[movies_df["movie_id"].isin(self.movie_encoder.classes_)].copy()
        movies["movie_id_encoded"] = self.movie_encoder.transform(movies["movie_id"]) + 1

        # 3. Khởi tạo ma trận rỗng toàn số 0 và đổ dữ liệu vào
        # self.num_movies đã được tính toán chính xác ở hàm encode_ids chạy trước đó
        genre_matrix = np.zeros((self.num_movies, self.max_genres), dtype=np.int64)
        for _, row in movies.iterrows():
            m_id = row["movie_id_encoded"]
            genre_matrix[m_id] = encode_multi_genres(row["genres"])
            
        return genre_matrix
    def encode_ids(self, ratings_df):
        """
        Mã hóa User ID và Movie ID thành số nguyên liên tiếp.
        
        Args:
            ratings_df (pd.DataFrame): Bảng ratings thô.
        Returns:
            pd.DataFrame: Bảng ratings đã được mã hóa ID.
        """
        # 1. Mã hóa User ID (0 -> num_users - 1)
        ratings_df["user_id"] = self.user_encoder.fit_transform(ratings_df["user_id"])
        self.num_users = len(self.user_encoder.classes_)
        
        # 2. Mã hóa Movie ID và DỊCH 1 ĐƠN VỊ (1 -> num_movies), ID 0 sẽ được dành riêng cho Padding trong chuỗi sequence
        ratings_df["movie_id"] = self.movie_encoder.fit_transform(ratings_df["movie_id"]) + 1
        self.num_movies = len(self.movie_encoder.classes_) + 1
        
        return ratings_df
    def process_user_features(self, users_df):
        """
        Mã hóa các thuộc tính phụ của User.
        Args:
            users_df (pd.DataFrame): Bảng user thô.
        Returns:
            pd.DataFrame: Bảng user đã mã hóa hoàn toàn.
        """
        users = users_df[users_df["user_id"].isin(self.user_encoder.classes_)].copy()
        users["user_id"] = self.user_encoder.transform(users["user_id"])
        users["gender"] = self.gender_encoder.fit_transform(users["gender"])
        users["age"] = self.age_encoder.fit_transform(users["age"])
        users["occupation"] = self.occ_encoder.fit_transform(users["occupation"])
        
        return users[["user_id", "gender", "age", "occupation"]]
    ### helper function ###
    def pad_sequence(self, seq):
        """
        Pad sequence về độ dài seq_len, padding value = 0 (Pre-padding).
        """
        if len(seq) >= self.seq_len:
            return seq[-self.seq_len:]
        else:
            return [0] * (self.seq_len - len(seq)) + seq
    
    ################## generate sequential data ###################
    def generate_sequential_data(self, ratings_df):
        pos_ratings = (
            ratings_df[ratings_df["rating"] >= 3]
            .sort_values(["user_id", "timestamp"])
        )
        train_samples = []
        test_samples = []
        for user_id, group in pos_ratings.groupby("user_id"):
            movie_ids = group["movie_id"].tolist()
            ratings = group["rating"].tolist()
            
            n = len(movie_ids)
            if n < 3:
                continue
                
            for i in range(1, n):
                history_padded = self.pad_sequence(movie_ids[:i])
                target_movie = movie_ids[i]
                target_rating = ratings[i]
                
                sample = {
                    "user_id": user_id,
                    "history": history_padded,
                    "target_movie": target_movie,
                    "rating": target_rating
                }
                if i == n - 1:
                    test_samples.append(sample)
                else:
                    train_samples.append(sample)
        return pd.DataFrame(train_samples), pd.DataFrame(test_samples)

    def preprocess(self, ratings_path, users_path, movies_path):
        """
        Hàm thực thi chính: Output trực tiếp ra Train DF, Test DF và Ma trận thể loại.
        """
        # 1. Load dữ liệu thô
        ratings_raw = self.load_ratings_data(ratings_path)
        users_raw = self.load_users_data(users_path)
        movies_raw = self.load_movies_data(movies_path)
        
        # 2. Mã hóa IDs chính (User, Movie)
        ratings_encoded = self.encode_ids(ratings_raw)
        
        # 3. Xây dựng ma trận thể loại phim
        genre_matrix = self.build_genre_matrix(movies_raw, max_genres=5)
        
        # 4. Mã hóa đặc trưng User
        users_processed = self.process_user_features(users_raw)
        
        # 5. Tạo và tách thẳng 2 tập Train / Test
        train_df, test_df = self.generate_sequential_data(ratings_encoded)
        
        # 6. Gộp thông tin User (Side Info) độc lập vào từng tập
        train_df = train_df.merge(users_processed, on="user_id", how="left")
        test_df = test_df.merge(users_processed, on="user_id", how="left")
        
        return train_df, test_df, genre_matrix
    
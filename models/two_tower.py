import torch
import torch.nn as nn
from .user_tower import UserTower
from .item_tower import ItemTower

class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        num_genres,
        num_genders=2,
        num_ages=7,
        num_occupations=21,
        max_seq_len=20,
        embedding_dim=64,
        genre_dim=16,
        dropout=0.2,
    ):
        """
        Args:
            num_users (int): Kích thước tập User (từ vocab).
            num_movies (int): Kích thước tập Movie (đã tính pad 0 từ preprocessing).
            num_genres (int): Kích thước tập Genres (đã tính pad 0 từ preprocessing).
        """
        super().__init__()

        # 1. Khởi tạo các lớp Embedding dùng chung (Shared Layers)
        # Bắt buộc padding_idx=0 để các véc-tơ đệm luôn là véc-tơ 0
        self.shared_movie_emb = nn.Embedding(num_movies, embedding_dim, padding_idx=0)
        self.shared_genre_emb = nn.Embedding(num_genres, genre_dim, padding_idx=0)
        
        # 2. Khởi tạo Hai Tháp (Bơm các shared layers vào)
        self.user_tower = UserTower(
            num_users=num_users,
            num_genders=num_genders,
            num_ages=num_ages,
            num_occupations=num_occupations,
            movie_emb_layer=self.shared_movie_emb,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.item_tower = ItemTower(
            self.shared_movie_emb,
            self.shared_genre_emb,
            embedding_dim=embedding_dim,
            genre_dim=genre_dim,
            dropout=dropout,
        )

    def forward(self, user_inputs, item_inputs):
        """
        Thực thi luồng đi qua 2 tháp.
        
        Args:
            user_inputs (dict): Chứa user_id, gender, age, occupation, user_hist.
            item_inputs (dict): Chứa target_movie (ID) và target_genres (Thuộc tính).
        Returns:
            user_vec, item_vec: Hai véc-tơ biểu diễn (đã L2 Normalized).
        """
        # Bơm đủ 5 features vào Tháp User
        user_vec = self.user_tower(
            user_inputs["user_id"], 
            user_inputs["gender"], 
            user_inputs["age"],          # Bổ sung Age
            user_inputs["occupation"], 
            user_inputs["movie_hist"]     # Khớp tên key từ Dataloader
        )
        
        # Bơm ID và Genres vào Tháp Item
        item_vec = self.item_tower(
            item_inputs["target_movie"], 
            item_inputs["target_genres"] # Bổ sung Genres
        )
        
        return user_vec, item_vec

    def compute_score(self, user_vec, item_vec):
        """
        Tính điểm tương đồng (Dot Product) giữa User và Item.
        Do 2 véc-tơ đã L2 Normalize, Dot Product chính là Cosine Similarity (giá trị -1 đến 1).
        """
        # (Batch_size, 64) * (Batch_size, 64) -> sum dim=1 -> (Batch_size,)
        scores = (user_vec * item_vec).sum(dim=1)
        return scores

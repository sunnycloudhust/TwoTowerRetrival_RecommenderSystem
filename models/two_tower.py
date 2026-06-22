import torch
import torch.nn as nn
import torch.nn.functional as F
from .user_tower import UserTower
from .item_tower import ItemTower

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies, num_genres):
        """
        Args:
            num_users (int): Number of encoded users.
            num_movies (int): Number of encoded movies, including padding id 0.
            num_genres (int): Number of encoded genres, including padding id 0.
        """
        super().__init__()

        self.shared_movie_emb = nn.Embedding(num_movies, 64, padding_idx=0)
        self.shared_genre_emb = nn.Embedding(num_genres, 16, padding_idx=0)
        
        self.user_tower = UserTower(num_users, self.shared_movie_emb)
        self.item_tower = ItemTower(self.shared_movie_emb, self.shared_genre_emb)

    def forward(self, user_inputs, item_inputs):
        """
        Args:
            user_inputs (dict): User ids, features, and history tensors.
            item_inputs (dict): Target movie ids and genre tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: L2-normalized user and item vectors.
        """
        user_vec = self.user_tower(
            user_inputs["user_id"], 
            user_inputs["gender"], 
            user_inputs["age"],
            user_inputs["occupation"], 
            user_inputs["movie_hist"]
        )
        
        item_vec = self.item_tower(
            item_inputs["target_movie"], 
            item_inputs["target_genres"]
        )
        
        return user_vec, item_vec

    def compute_score(self, user_vec, item_vec):
        """
        Return cosine similarity because both vectors are L2-normalized.
        """
        scores = (user_vec * item_vec).sum(dim=1)
        return scores

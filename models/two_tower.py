import torch
import torch.nn as nn
import torch.nn.functional as F
from .user_tower import UserTower
from .item_tower import ItemTower

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies):
        super().__init__()

        self.shared_movie_emb = nn.Embedding(num_movies + 1, 64, padding_idx=0)
        
        self.user_tower = UserTower(num_users, self.shared_movie_emb)
        self.item_tower = ItemTower(self.shared_movie_emb)

    def forward(self, user_inputs, target_movie_id):
        user_vec = self.user_tower(
            user_inputs["user_id"], 
            user_inputs["gender"], 
            user_inputs["occupation"], 
            user_inputs["movie_hist"]
        )
        item_vec = self.item_tower(target_movie_id)
        return user_vec, item_vec
    
    
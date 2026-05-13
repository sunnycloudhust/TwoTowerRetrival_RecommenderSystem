import torch
import torch.nn as nn
import torch.nn.functional as F

class ItemTower(nn.Module):
    def __init__(self, movie_emb_layer):
        super().__init__()
        self.movie_emb = movie_emb_layer
        
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, movie_id):
        # movie_id đã được cộng 1 ở preprocessing để tránh ID 0
        item_vector = self.movie_emb(movie_id) 
        item_vector = self.mlp(item_vector)
        
        # L2 Normalize (Phải giống hệt User Tower)
        return F.normalize(item_vector, p=2, dim=1)
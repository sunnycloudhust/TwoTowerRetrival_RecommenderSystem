import torch
import torch.nn as nn
import torch.nn.functional as F

class ItemTower(nn.Module):
    def __init__(self, movie_emb_layer):
        super().__init__()
        
        # Dùng chung layer embedding với User Tower
        self.movie_emb = movie_emb_layer
        
        # MLP để biến đổi movie embedding về không gian chung
        # Nếu movie_emb đã có size 64 và bạn muốn giữ nguyên, MLP có thể đơn giản hơn
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, movie_id):
        # movie_id: (batch_size)
        item_vector = self.movie_emb(movie_id) # (batch_size, 64)
        item_vector = self.mlp(item_vector)
        
        # L2 Normalization (quan trọng để đồng bộ với User Tower)
        return F.normalize(item_vector, p=2, dim=1)
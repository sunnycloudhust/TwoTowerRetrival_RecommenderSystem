import torch
import torch.nn as nn
import torch.nn.functional as F

class ItemTower(nn.Module):
    def __init__(self, movie_emb_layer, genre_emb_layer):
        super().__init__()
        self.movie_emb = movie_emb_layer
        self.genre_emb = genre_emb_layer 

        self.mlp = nn.Sequential(
            nn.Linear(64 + 16, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        )

    def forward(self, movie_id, movie_genres):
        """
        Args:
            movie_id: Tensor (B,)
            movie_genres: Tensor (B, max_genres) from genre_matrix.
        """
        id_vector = self.movie_emb(movie_id)
        
        g_emb = self.genre_emb(movie_genres)
        mask = (movie_genres != 0).float().unsqueeze(-1)
        genre_vector = (g_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        item_features = torch.cat([id_vector, genre_vector], dim=1)
        item_vector = self.mlp(item_features)
        
        return F.normalize(item_vector, p=2, dim=1)

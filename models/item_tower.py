import torch
import torch.nn as nn
import torch.nn.functional as F

class ItemTower(nn.Module):
    def __init__(self, movie_emb_layer, genre_emb_layer, embedding_dim=64, genre_dim=16, dropout=0.2):
        super().__init__()
        self.movie_emb = movie_emb_layer
        self.genre_emb = genre_emb_layer 

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + genre_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, movie_id, movie_genres):
        """
        Args:
            movie_id: Tensor (B,)
            movie_genres: Tensor (B, max_genres) truy xuất từ genre_matrix
        """
        id_vector = self.movie_emb(movie_id) # (B, 64)
        
        # Masked Pooling cho Genres (Gộp các véc-tơ thể loại của phim)
        g_emb = self.genre_emb(movie_genres) # (B, max_genres, 16)
        mask = (movie_genres != 0).float().unsqueeze(-1)
        genre_vector = (g_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1) # (B, 16)
        
        # Nối ID phim và Thuộc tính phim
        item_features = torch.cat([id_vector, genre_vector], dim=1) # (B, 80)
        item_vector = self.mlp(item_features)
        
        return F.normalize(item_vector, p=2, dim=1)

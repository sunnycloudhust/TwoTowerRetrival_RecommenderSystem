import torch
import torch.nn as nn

class UserTower(nn.Module):
    def __init__(self, num_users, num_movies, movie_emb_layer):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, 64)
        self.gender_emb = nn.Embedding(2, 8)
        self.occ_emb = nn.Embedding(21, 16)

        # Sử dụng chung layer embedding được truyền từ ngoài vào
        self.movie_emb = movie_emb_layer 

        self.mlp = nn.Sequential(
            nn.Linear(64 + 8 + 16 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, user_id, gender, occupation, movie_hist):
        u = self.user_emb(user_id)
        g = self.gender_emb(gender)
        o = self.occ_emb(occupation)

        # movie_hist: (batch, seq_len) -> (batch, seq_len, 64)
        m = self.movie_emb(movie_hist)
        
        # Pooling: batch seqlen 64 -> batch_size, 64
        hist = m.mean(dim=1)

        x = torch.cat([u, g, o, hist], dim=1)
        user_vector = self.mlp(x)
        

        # L2 Normalization để tính Cosine Similarity dễ dàng hơn
        return torch.nn.functional.normalize(user_vector, p=2, dim=1)
import torch
import torch.nn as nn

class UserTower(nn.Module):
    def __init__(self, num_users, num_movies):
        super().__init__()

        # embedding cho feature
        self.user_emb = nn.Embedding(num_users, 64)
        self.gender_emb = nn.Embedding(2, 8)
        self.occ_emb = nn.Embedding(21, 16)

        # shared với item tower
        self.movie_emb = nn.Embedding(num_movies, 64)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(64 + 8 + 16 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, user_id, gender, occupation, movie_hist):
        u = self.user_emb(user_id)
        g = self.gender_emb(gender)
        o = self.occ_emb(occupation)

        # movie_hist: (batch, seq_len)
        m = self.movie_emb(movie_hist)

        # pooling
        hist = m.mean(dim=1)

        x = torch.cat([u, g, o, hist], dim=1)

        return self.mlp(x)
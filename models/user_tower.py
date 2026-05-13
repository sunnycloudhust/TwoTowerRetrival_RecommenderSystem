import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, num_users, movie_emb_layer):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, 64)
        self.gender_emb = nn.Embedding(2, 8)
        self.occ_emb = nn.Embedding(21, 16)
        self.movie_emb = movie_emb_layer # Nhận layer có padding_idx=0

        self.mlp = nn.Sequential(
            nn.Linear(64 + 8 + 16 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, user_id, gender, occupation, movie_hist):
        u = self.user_emb(user_id)
        g = self.gender_emb(gender)
        o = self.occ_emb(occupation)

        # Masked Pooling: Bỏ qua padding (ID 0)
        m = self.movie_emb(movie_hist) # (B, seq_len, 64)
        mask = (movie_hist != 0).float().unsqueeze(-1) # (B, seq_len, 1)
        hist = (m * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        x = torch.cat([u, g, o, hist], dim=1)
        user_vector = self.mlp(x)
        return F.normalize(user_vector, p=2, dim=1)
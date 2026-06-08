import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(
        self,
        num_users,
        num_genders,
        num_ages,
        num_occupations,
        movie_emb_layer,
        max_seq_len,
        embedding_dim=64,
        dropout=0.2,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.gender_emb = nn.Embedding(num_genders, 8)
        self.age_emb = nn.Embedding(num_ages, 8)
        self.occ_emb = nn.Embedding(num_occupations, 16)
        self.movie_emb = movie_emb_layer

        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.attn_pool = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1)
        )

        mlp_in_dim = embedding_dim + 8 + 8 + 16 + embedding_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, user_id, gender, age, occupation, movie_hist):
        u = self.user_emb(user_id)
        g = self.gender_emb(gender)
        a = self.age_emb(age) 
        o = self.occ_emb(occupation)

        B, S = movie_hist.size()
        if S > self.max_seq_len:
            raise ValueError(f"movie_hist length {S} exceeds max_seq_len={self.max_seq_len}")

        positions = torch.arange(S, device=movie_hist.device).unsqueeze(0).expand(B, S)
        seq_emb = self.movie_emb(movie_hist) + self.pos_emb(positions)
        padding_mask = (movie_hist == 0)

        trans_out = self.transformer(seq_emb, src_key_padding_mask=padding_mask)

        mask = (movie_hist != 0).float().unsqueeze(-1)
        attn_scores = self.attn_pool(trans_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(padding_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        attn_hist = (trans_out * attn_weights * mask).sum(dim=1)

        # The history is pre-padded, so the last position is the most recent item.
        recent_hist = trans_out[:, -1, :]

        x = torch.cat([u, g, a, o, attn_hist, recent_hist], dim=1)
        user_vector = self.mlp(x)
        
        return F.normalize(user_vector, p=2, dim=1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, num_users, movie_emb_layer):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, 64)
        self.gender_emb = nn.Embedding(2, 8)
        self.age_emb = nn.Embedding(7, 8)       
        self.occ_emb = nn.Embedding(21, 16)
        self.movie_emb = movie_emb_layer        

        self.pos_emb = nn.Embedding(100, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.mlp = nn.Sequential(
            nn.Linear(64 + 8 + 8 + 16 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64) 
        )

    def forward(self, user_id, gender, age, occupation, movie_hist):
        u = self.user_emb(user_id)
        g = self.gender_emb(gender)
        a = self.age_emb(age) 
        o = self.occ_emb(occupation)

        B, S = movie_hist.size()
        positions = torch.arange(S, device=movie_hist.device).unsqueeze(0).expand(B, S)
        seq_emb = self.movie_emb(movie_hist) + self.pos_emb(positions)
        padding_mask = (movie_hist == 0)

        # 3. Đi qua Transformer
        trans_out = self.transformer(seq_emb, src_key_padding_mask=padding_mask)

        # 4. Masked Mean Pooling trên đầu ra của Transformer
        mask = (movie_hist != 0).float().unsqueeze(-1)
        hist = (trans_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # 5. Gộp toàn bộ véc-tơ đặc trưng
        x = torch.cat([u, g, a, o, hist], dim=1)
        user_vector = self.mlp(x)
        
        return F.normalize(user_vector, p=2, dim=1)
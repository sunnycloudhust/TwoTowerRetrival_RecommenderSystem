import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, num_users, movie_emb_layer):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, 64)
        self.gender_emb = nn.Embedding(2, 8)
        self.age_emb = nn.Embedding(7, 8)       # Thêm Embedding cho Age (7 nhóm tuổi)
        self.occ_emb = nn.Embedding(21, 16)
        self.movie_emb = movie_emb_layer        # Shared layer

        # Tổng input: 64 (User) + 8 (Gender) + 8 (Age) + 16 (Occ) + 64 (Hist) = 160
        self.mlp = nn.Sequential(
            nn.Linear(64 + 8 + 8 + 16 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64) # Output 64 khớp với Item Tower
        )

    def forward(self, user_id, gender, age, occupation, movie_hist):
        u = self.user_emb(user_id)
        g = self.gender_emb(gender)
        a = self.age_emb(age) # Trích xuất Age
        o = self.occ_emb(occupation)

        # Xử lý chuỗi lịch sử (Masked Pooling)
        m = self.movie_emb(movie_hist)
        mask = (movie_hist != 0).float().unsqueeze(-1)
        hist = (m * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Gộp toàn bộ véc-tơ ngữ cảnh của User
        x = torch.cat([u, g, a, o, hist], dim=1)
        user_vector = self.mlp(x)
        
        return F.normalize(user_vector, p=2, dim=1)
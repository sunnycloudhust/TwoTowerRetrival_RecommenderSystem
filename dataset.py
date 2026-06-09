import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MovieLensDataset(Dataset):
    def __init__(self, df, genre_matrix):
        """
        Lớp Dataset tối ưu hóa hiệu năng cao cho mô hình Two-Tower.
        
        Args:
            df (pd.DataFrame): Bảng dữ liệu đã qua tiền xử lý (train_df hoặc test_df)
                               chứa sẵn lịch sử trượt động, target, label và user features.
            genre_matrix (np.ndarray): Ma trận tra cứu nhanh ID thể loại phim.
        """
        # 1. Chuyển đổi toàn bộ dữ liệu Pandas sang NumPy Array trong __init__
        # Loại bỏ hoàn toàn nút thắt cổ chai (bottleneck) tra cứu của CPU.
        self.user_ids = df["user_id"].values
        self.genders = df["gender"].values
        self.ages = df["age"].values
        self.occupations = df["occupation"].values
        self.target_movies = df["target_movie"].values
        
        # Nhãn đánh giá: Ép kiểu float32 để tương thích trực tiếp với các hàm Loss PyTorch
        self.labels = df["rating"].values.astype(np.float32)
        
        # Cột 'history' chứa danh sách (list) các ID phim. Dùng np.vstack biến thành ma trận 2D tĩnh.
        self.histories = np.vstack(df["history"].values)
        
        # 2. Lưu trữ bảng tra cứu thể loại phim
        self.genre_matrix = genre_matrix

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        # Lấy ID phim mục tiêu để tra cứu thể loại cực nhanh (O(1))
        target_movie_id = self.target_movies[idx]
        target_genres = self.genre_matrix[target_movie_id]

        # Phân cụm Output dict gọn gàng, ánh xạ chuẩn xác vào hàm forward của TwoTowerModel
        return {
            "user_inputs": {
                "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
                "gender": torch.tensor(self.genders[idx], dtype=torch.long),
                "age": torch.tensor(self.ages[idx], dtype=torch.long),
                "occupation": torch.tensor(self.occupations[idx], dtype=torch.long),
                "movie_hist": torch.tensor(self.histories[idx], dtype=torch.long) # Khớp tên biến UserTower
            },
            "item_inputs": {
                "target_movie": torch.tensor(target_movie_id, dtype=torch.long),
                "target_genres": torch.tensor(target_genres, dtype=torch.long)
            },
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }

def get_dataloader(df, genre_matrix, batch_size=256, shuffle=True, num_workers=2):
    """
    Hàm khởi tạo DataLoader nạp dữ liệu song song.
    """
    dataset = MovieLensDataset(df, genre_matrix)
    # Khuyên dùng num_workers >= 2 để tận dụng đa luồng CPU chuẩn bị batch trước cho GPU
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
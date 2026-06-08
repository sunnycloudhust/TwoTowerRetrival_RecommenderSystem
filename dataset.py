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
        self.user_ids = torch.as_tensor(df["user_id"].to_numpy(), dtype=torch.long)
        self.genders = torch.as_tensor(df["gender"].to_numpy(), dtype=torch.long)
        self.ages = torch.as_tensor(df["age"].to_numpy(), dtype=torch.long)
        self.occupations = torch.as_tensor(df["occupation"].to_numpy(), dtype=torch.long)
        self.target_movies = torch.as_tensor(df["target_movie"].to_numpy(), dtype=torch.long)
        self.labels = torch.as_tensor(df["rating"].to_numpy(np.float32), dtype=torch.float32)
        self.histories = torch.as_tensor(np.vstack(df["history"].values), dtype=torch.long)
        self.genre_matrix = torch.as_tensor(genre_matrix, dtype=torch.long)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        # Lấy ID phim mục tiêu để tra cứu thể loại cực nhanh (O(1))
        target_movie_id = self.target_movies[idx]
        target_genres = self.genre_matrix[target_movie_id]

        # Phân cụm Output dict gọn gàng, ánh xạ chuẩn xác vào hàm forward của TwoTowerModel
        return {
            "user_inputs": {
                "user_id": self.user_ids[idx],
                "gender": self.genders[idx],
                "age": self.ages[idx],
                "occupation": self.occupations[idx],
                "movie_hist": self.histories[idx]
            },
            "item_inputs": {
                "target_movie": target_movie_id,
                "target_genres": target_genres
            },
            "label": self.labels[idx]
        }

def get_dataloader(
    df,
    genre_matrix,
    batch_size=256,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
    generator=None,
    worker_init_fn=None,
):
    """
    Hàm khởi tạo DataLoader nạp dữ liệu song song.
    """
    dataset = MovieLensDataset(df, genre_matrix)
    # Khuyên dùng num_workers >= 2 để tận dụng đa luồng CPU chuẩn bị batch trước cho GPU
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

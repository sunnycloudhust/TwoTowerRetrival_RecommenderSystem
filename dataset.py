import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MovieLensDataset(Dataset):
    def __init__(self, ratings, users_df, user_hist_array):
        """
        Args:
            ratings: DataFrame từ preprocessor (đã encode)
            users_df: DataFrame chứa gender, age, occupation (đã encode)
            user_hist_array: Numpy array chứa chuỗi lịch sử (đã padding)
        """
        self.ratings = ratings
        self.users_df = users_df.set_index("user_id")
        self.user_hist_array = user_hist_array

        # Mapping để lấy thông tin nhanh hơn
        self.user_ids = ratings["user_id"].values
        self.movie_ids = ratings["movie_id"].values
        
        # Chuẩn bị gender và occupation
        # Lưu ý: Gender cần chuyển sang số (M:0, F:1) nếu preprocessor chưa làm
        self.gender_map = {'M': 0, 'F': 1}

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        target_movie = self.movie_ids[idx]
        
        # Lấy metadata của user
        user_data = self.users_df.loc[uid]
        
        # Chuyển gender sang int nếu nó vẫn là 'M'/'F'
        gender = self.gender_map.get(user_data['gender'], user_data['gender'])
        occ = user_data['occupation']
        
        # Lấy history từ array đã prep (index của array tương ứng với encoded user_id)
        history = self.user_hist_array[uid]

        return {
            "user_id": torch.tensor(uid, dtype=torch.long),
            "gender": torch.tensor(gender, dtype=torch.long),
            "occupation": torch.tensor(occ, dtype=torch.long),
            "user_hist": torch.tensor(history, dtype=torch.long),
            "target_movie": torch.tensor(target_movie, dtype=torch.long)
        }

def get_dataloader(ratings, users_df, user_hist_array, batch_size=256, shuffle=True):
    dataset = MovieLensDataset(ratings, users_df, user_hist_array)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
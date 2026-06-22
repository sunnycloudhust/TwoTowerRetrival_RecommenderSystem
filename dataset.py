import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MovieLensDataset(Dataset):
    def __init__(self, df, genre_matrix):
        """
        Dataset for the two-tower retrieval model.
        
        Args:
            df (pd.DataFrame): Preprocessed train or test data.
            genre_matrix (np.ndarray): Movie-to-genre lookup matrix.
        """
        self.user_ids = df["user_id"].values
        self.genders = df["gender"].values
        self.ages = df["age"].values
        self.occupations = df["occupation"].values
        self.target_movies = df["target_movie"].values
        self.labels = df["rating"].values.astype(np.float32)
        self.histories = np.vstack(df["history"].values)
        self.genre_matrix = genre_matrix

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        target_movie_id = self.target_movies[idx]
        target_genres = self.genre_matrix[target_movie_id]

        return {
            "user_inputs": {
                "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
                "gender": torch.tensor(self.genders[idx], dtype=torch.long),
                "age": torch.tensor(self.ages[idx], dtype=torch.long),
                "occupation": torch.tensor(self.occupations[idx], dtype=torch.long),
                "movie_hist": torch.tensor(self.histories[idx], dtype=torch.long)
            },
            "item_inputs": {
                "target_movie": torch.tensor(target_movie_id, dtype=torch.long),
                "target_genres": torch.tensor(target_genres, dtype=torch.long)
            },
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }

def get_dataloader(df, genre_matrix, batch_size=256, shuffle=True, num_workers=2):
    dataset = MovieLensDataset(df, genre_matrix)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

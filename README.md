# TwoTowerRetrival_RecommenderSystem

A MovieLens-based two-tower recommendation system implemented in PyTorch. The project preprocesses MovieLens rating and user metadata, builds user history sequences, and trains a two-tower neural model for user-item matching.

## Features

- Data preprocessing for MovieLens `ratings.dat` and `users.dat`
- Label encoding of `user_id` and `movie_id`
- User history sequence padding for a fixed-length recommendation input
- Two-Tower architecture with shared movie embeddings
- In-batch softmax training loss for contrastive learning
- Model saving to `two_tower_model.pth`

## Repository Structure

- `train.py` - Main training script for the recommendation model.
- `dataset.py` - Dataset and DataLoader wrapper for training.
- `preprocessing/preprocessing.py` - Data loading, encoding, and user history preparation.
- `models/user_tower.py` - User tower network.
- `models/item_tower.py` - Item tower network.
- `models/two_tower.py` - Full two-tower model combining user and item towers.
- `dataset/README` - MovieLens dataset description.

## Dataset

This project is designed to use the MovieLens dataset files under `dataset/`:

- `dataset/ratings.dat`
- `dataset/users.dat`

The existing `dataset/README` indicates this is the MovieLens dataset format. The code expects MovieLens files with `::` separators.

## Requirements

- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn

Example installation:

```bash
pip install torch pandas numpy scikit-learn
```

## Usage

1. Place the MovieLens data files in the `dataset/` folder.
2. Run the training script:

```bash
python train.py
```

During training, the script will:

- preprocess ratings and user metadata
- encode users and movies
- build user history sequences
- create a training DataLoader
- train the two-tower model for 50 epochs
- save the model weights to `two_tower_model.pth`

## Training Details

- Sequence length for user history: `20`
- Batch size: `1024`
- Learning rate: `0.001`
- Number of epochs: `50`
- Loss: in-batch softmax cross-entropy over user/item vectors

## Notes

- `Preprocessor.preprocess()` encodes IDs and pads user histories.
- The user tower uses user embedding, gender embedding, occupation embedding, and pooled movie history.
- The item tower uses a shared movie embedding layer and an MLP.
- The model normalizes user and item vectors before computing similarity.

## Extending the Project

- Add evaluation logic to compute hit rate or recall.
- Include a movie metadata tower using `movies.dat`.
- Add hyperparameter configuration via command-line arguments.
- Support more dataset variants and negative sampling.


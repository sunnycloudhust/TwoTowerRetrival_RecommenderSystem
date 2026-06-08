# Two-Tower Retrieval Recommender System

A high-performance Two-Tower Neural Retrieval model implemented in PyTorch for session-based and collaborative filtering recommendation tasks. This project is built using the classic MovieLens 1M dataset format, optimizing user side-information (demographics and sequential interaction history) alongside item side-information (movie IDs and multi-genre profiles).

---

## ⚙️ Key Features
- **Two-Tower Architecture:** Implements standalone `UserTower` and `ItemTower` sub-networks mapping users and movies to a shared low-dimensional embedding space ($d=64$), with Transformer-based history encoding.
- **Optimized Performance:** Pre-computes and stacks DataFrames into static tensors during dataset initialization to reduce per-sample CPU overhead.
- **In-Batch Softmax Loss:** Leverages dynamic, highly efficient contrastive learning loss with a temperature scaling parameter to scale candidate retrieval.
- **Sequential Leave-One-Out Evaluation:** Built using industrial sequential prediction patterns. It drops items seen in the training history and performs dynamic `-inf` scoring masks on test lookups.
- **Distributed Training Ready:** Features automatic scaling across multiple GPUs using PyTorch's `nn.DataParallel`.
- **Comprehensive Evaluation:** Computes Top-$K$ retrieval performance via batched inner-product matrix multiplication, tracking **Recall@K**, **Precision@K**, **MRR@K**, and **NDCG@K**.

---

## 📁 Repository Structure
```text
├── dataset/                  # Contains raw data or descriptors (.dat files)
├── checkpoints/              # Model weights saved iteratively during training
├── models/
│   ├── __init__.py
│   ├── user_tower.py         # Sub-network processing User ID, Age, Gender, History, etc.
│   ├── item_tower.py         # Sub-network processing Target Movie & Multi-Genres
│   └── two_tower.py          # Wrapper routing shared layer embeddings & similarity steps
├── preprocessing/
│   ├── preprocessing.py      # Label encoding, history sliding window sequence building
│   └── run_preprocessing.ipynb
├── dataset.py                # Ultra-performant PyTorch Dataset mapping and Dataloaders
├── train.py                  # Training loops and checkpoint recovery mechanisms
├── test.py                   # Batched matrix lookups and standard LOO ranking metrics
├── utils.py                  # Experiment helpers: seeding, plotting, CSV summaries
└── main.py                   # Master orchestration script pipeline execution
```

---

## 🛠️ System Architecture

### 1. Data Pipeline & Data Preprocessing
The model expects interaction datasets tracking explicit ratings along chronological timelines. 
- Filters interaction matrices keeping only positive records ($\text{rating} \ge 3$).
- Uses **Leave-One-Out (LOO)** sequencing: the last recorded interaction of a user acts as the test query target, while sequences prior populate the training split.
- Automatically handles **Multi-Genre Fields** mapping variable-length categories (`Action|Sci-Fi|Thriller`) into rigid, padded fixed-width vectors.

### 2. Dual Network Pipeline
```
  [User Inputs] (ID, Gender, Age, Occupation)       [Item Inputs] (Movie ID)
                       │                                         │
        ┌──────────────┴──────────────┐           ┌──────────────┴──────────────┐
        ▼                             ▼           ▼                             ▼
 [Embeddings]                  [History Seq]   [Shared Movie Emb]        [Genres Vector]
        │                             │           (Padded Index 0)              │
        ▼                             ▼           │                             ▼
  [Concatenate]         [Transformer + Attention] │                     [Shared Genre Emb]
        │                             │           │                             │
        └──────────────┬──────────────┘           │                             ▼
                       ▼                          └──────────────┬──────────────┘
              [Fully Connected]                                  ▼
                       │                                [Fully Connected]
                       ▼                                         │
                  [User Tower]                                   ▼
                       │                                    [Item Tower]
                       ▼ (L2 Normalized Space)                   ▼
                User Vector ($u$)                         Item Vector ($v$)
                       │                                         │
                       └───────────────────┬─────────────────────┘
                                           ▼
                                 Cosine Similarity ($u \cdot v$)
```

---

## 🚀 Getting Started

### Prerequisites
Make sure your environment contains the necessary dependencies:
```bash
pip install torch numpy pandas scikit-learn tqdm matplotlib
```

### Dataset Placement
Download and extract the MovieLens format text tables (`ratings.dat`, `users.dat`, `movies.dat`) into the localized directory root:
```text
dataset/
  ├── ratings.dat
  ├── users.dat
  └── movies.dat
```

---

## 🏋️ Running the Pipeline

To execute data ingestion, dictionary compilation, training passes, and final out-of-sample prediction evaluations natively, run:
```bash
python main.py
```

### Pipeline Hyperparameters (Modifiable via `main.py`)
```python
BASE_HYPERPARAMS = {
    "seq_len": 20,
    "max_genres": 5,
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "temperature": 0.08,
    "random_split": False,
    "eval_every": 1,
    "num_workers": 2,
    "dropout": 0.2,
    "grad_clip": 1.0,
    "num_runs": 5,
    "seeds": [42, 2024, 3407, 12345, 98765],
}

EXPERIMENTS = [
    {
        "k": 20,
        "checkpoint_dir": "checkpoints/k20_5runs",
        "loss_plot_path": "outputs/k20/loss_curves_k20_5_runs.png",
        "final_results_csv_path": "outputs/k20/final_results_k20_5_runs.csv",
    },
    {
        "k": 10,
        "checkpoint_dir": "checkpoints/k10_5runs",
        "loss_plot_path": "outputs/k10/loss_curves_k10_5_runs.png",
        "final_results_csv_path": "outputs/k10/final_results_k10_5_runs.csv",
    },
]
```

---

## 📊 Evaluation Metrics

When training runs conclude, the system maps out embeddings for all candidate items across your catalog to evaluate validation and test batches. Checkpoints are selected by validation ranking quality. The default script runs 5 independent seeds for both $K=20$ and $K=10$.

Each experiment writes a final summary CSV with one row per completed run:
```text
outputs/k20/final_results_k20_5_runs.csv
outputs/k10/final_results_k10_5_runs.csv
```

Each CSV contains the final train loss from the last epoch, the best validation metric, final test metrics, seed, run id, and checkpoint path. Training-loss plots are saved separately:
```text
outputs/k20/loss_curves_k20_5_runs.png
outputs/k10/loss_curves_k10_5_runs.png
```

| Metric | Target Goal | Meaning |
| :--- | :--- | :--- |
| **Recall@K** | Higher is better | Measures if the exact skipped item fell anywhere inside your top recommendations. |
| **Precision@K** | Higher is better | Ratio of hit relevance dispersed inside the top recommendation set size. |
| **MRR@K** | Higher is better | Multiplicative inverse rank placement score tracking how close hits were to index 0. |
| **NDCG@K** | Higher is better | Graded relevance index penalized heavily based on sub-optimal position placement. |

---

## 💾 Checkpoints and Resuming Training
Weights and vocabulary states are managed inside `train.py`. The framework saves dictionary encoders (`user_encoder`, `movie_encoder`, user feature encoders, `genre2id`) and model weights together.

To resume an interrupted training session, toggle `resume=True` within the execution script's main sequence block in `main.py`:
```python
best_model_path = train(
    ...,
    resume=True
)
```

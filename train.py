import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn
from preprocessing.preprocessing import Preprocessor
from models.two_tower import TwoTowerModel
from dataset import get_dataloader

# --- LOSS FUNCTION ---
def in_batch_softmax_loss(user_vecs, item_vecs, temperature=0.1):
    """
    Compute In-batch Softmax Loss.
    Each user in the batch uses their corresponding item as Positive (diagonal),
    and items from other users as Negative.
    """
    logits = torch.matmul(user_vecs, item_vecs.T) / temperature
    labels = torch.arange(user_vecs.size(0)).to(user_vecs.device)
    return F.cross_entropy(logits, labels)

# --- EVALUATION METRICS ---
def recall_at_k(y_true, y_pred, k=20):
    """
    Calculate Recall@k
    y_true: list of relevant items (shape: (num_users, num_all_items) or (num_users,))
    y_pred: list of predicted ranks/scores (shape: (num_users, num_all_items))
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    
    # Get top-k items with highest scores
    top_k_indices = np.argsort(y_pred, axis=1)[:, -k:][:, ::-1]  # Sort descending
    
    recalls = []
    for i in range(len(y_true)):
        if isinstance(y_true[i], (int, np.integer)):
            # y_true[i] is an item id
            relevant = {y_true[i]}
        else:
            # y_true[i] is a list of relevant items
            relevant = set(y_true[i])
        
        top_k = set(top_k_indices[i])
        recall = len(relevant & top_k) / len(relevant) if len(relevant) > 0 else 0
        recalls.append(recall)
    
    return np.mean(recalls)

def precision_at_k(y_true, y_pred, k=20):
    """
    Calculate Precision@k
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    
    top_k_indices = np.argsort(y_pred, axis=1)[:, -k:][:, ::-1]
    
    precisions = []
    for i in range(len(y_true)):
        if isinstance(y_true[i], (int, np.integer)):
            relevant = {y_true[i]}
        else:
            relevant = set(y_true[i])
        
        top_k = set(top_k_indices[i])
        precision = len(relevant & top_k) / k if k > 0 else 0
        precisions.append(precision)
    
    return np.mean(precisions)

def mrr_at_k(y_true, y_pred, k=20):
    """
    Calculate Mean Reciprocal Rank@k
    MRR = 1/N * Σ(1/rank_of_first_relevant_item)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    
    top_k_indices = np.argsort(y_pred, axis=1)[:, -k:][:, ::-1]
    
    mrrs = []
    for i in range(len(y_true)):
        if isinstance(y_true[i], (int, np.integer)):
            relevant = {y_true[i]}
        else:
            relevant = set(y_true[i])
        
        top_k = top_k_indices[i]
        # Find position of first relevant item
        mrr = 0
        for rank, item in enumerate(top_k, 1):
            if item in relevant:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)
    
    return np.mean(mrrs)

def ndcg_at_k(y_true, y_pred, k=20):
    """
    Calculate Normalized Discounted Cumulative Gain@k
    DCG = Σ(rel_i / log2(i+1)) for i in 1..k
    NDCG = DCG / IDCG
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    
    top_k_indices = np.argsort(y_pred, axis=1)[:, -k:][:, ::-1]
    
    ndcgs = []
    for i in range(len(y_true)):
        if isinstance(y_true[i], (int, np.integer)):
            relevant = {y_true[i]}
        else:
            relevant = set(y_true[i])
        
        # Calculate DCG
        dcg = 0
        top_k = top_k_indices[i]
        for rank, item in enumerate(top_k, 1):
            if item in relevant:
                dcg += 1.0 / np.log2(rank + 1)
        
        # Calculate IDCG (ideal case: all relevant items at top)
        idcg = 0
        num_relevant = len(relevant)
        for rank in range(1, min(num_relevant, k) + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
    
    return np.mean(ndcgs)

def evaluate_metrics(model, test_loader, device, k=20):
    """
    Evaluate model on test set with metrics
    """
    model.eval()
    all_user_vecs = []
    all_item_vecs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            user_inputs = {
                'user_id': batch['user_id'].to(device),
                'gender': batch['gender'].to(device),
                'occupation': batch['occupation'].to(device),
                'movie_hist': batch['user_hist'].to(device)
            }
            target_movies = batch['target_movie'].to(device)
            
            user_vec, item_vec = model(user_inputs, target_movies)
            
            all_user_vecs.append(user_vec.cpu())
            all_item_vecs.append(item_vec.cpu())
            all_targets.append(target_movies.cpu())
    
    # Concatenate data
    user_vecs = torch.cat(all_user_vecs, dim=0)
    item_vecs = torch.cat(all_item_vecs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate scores: user vector * item vector
    scores = torch.matmul(user_vecs, item_vecs.T)  # Shape: (num_users, num_items)
    
    # Calculate metrics
    recall = recall_at_k(targets.numpy(), scores.numpy(), k=k)
    precision = precision_at_k(targets.numpy(), scores.numpy(), k=k)
    mrr = mrr_at_k(targets.numpy(), scores.numpy(), k=k)
    ndcg = ndcg_at_k(targets.numpy(), scores.numpy(), k=k)
    
    return {
        f'Recall@{k}': recall,
        f'Precision@{k}': precision,
        f'MRR@{k}': mrr,
        f'NDCG@{k}': ndcg
    }

########################## --- MAIN TRAINING LOOP ---##################################
def main():
    # 1. Configuration (Hyperparameters)
    EPOCHS = 100
    BATCH_SIZE = 1024
    LR = 0.01
    SEQ_LEN = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    # 2. Data preprocessing
    print("Preprocessing data...")
    prep = Preprocessor(seq_len=SEQ_LEN)
    
    # Get all 3 variables (ensure preprocessing.py has return statement)
    ratings, users, user_hist_array = prep.preprocess("dataset/ratings.dat", "dataset/users.dat")

    # 3. Get number of users and movies for embedding initialization
    num_users = len(prep.user_encoder.classes_)
    num_movies = len(prep.movie_encoder.classes_)
    print(f"Number of Users: {num_users}, Number of Movies: {num_movies}")

    # 4. Split data into train/test (80/20)
    print("Splitting data into train/test set...")
    split_idx = int(0.8 * len(ratings))
    train_ratings = ratings.iloc[:split_idx].reset_index(drop=True)
    test_ratings = ratings.iloc[split_idx:].reset_index(drop=True)
    
    # 5. Initialize DataLoader
    print("Preparing DataLoader...")
    train_loader = get_dataloader(train_ratings, users, user_hist_array, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = get_dataloader(test_ratings, users, user_hist_array, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Initialize model and optimizer
    print("Initializing model...")
    model = TwoTowerModel(num_users=num_users, num_movies=num_movies)
    
    # Apply DataParallel if multiple GPUs available
    if num_gpus > 1:
        print(f"Đang chạy song song trên {num_gpus} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Checkpoint directory for final model
    checkpoint_dir = 'checkpoints'
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 7. Training process
    print("\n--- START TRAINING ---")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to GPU/CPU
            user_inputs = {
                'user_id': batch['user_id'].to(DEVICE),
                'gender': batch['gender'].to(DEVICE),
                'occupation': batch['occupation'].to(DEVICE),
                'movie_hist': batch['user_hist'].to(DEVICE)
            }
            target_movies = batch['target_movie'].to(DEVICE)

            # Clear old gradients
            optimizer.zero_grad()

            # Forward pass: get vectors from 2 towers
            user_vec, item_vec = model(user_inputs, target_movies)

            # Calculate loss (Softmax Loss)
            loss = in_batch_softmax_loss(user_vec, item_vec, temperature=0.1)

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Print loss for each batch
            avg_batch_loss = total_loss / (batch_idx + 1)
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f} | Avg Loss: {avg_batch_loss:.4f}")

        # Save checkpoint for every epoch (no metric evaluation)
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print(f"Epoch checkpoint saved: {epoch_checkpoint_path}\n")

    # Save final model
    torch.save(model.state_dict(), "two_tower_model.pth")
    print("\nFinal model saved successfully at 'two_tower_model.pth'!")

    # Final evaluation (run once after training)
    print("\n--- FINAL EVALUATION @20 ---")
    final_metrics = evaluate_metrics(model, test_loader, DEVICE, k=20)
    for metric_name, metric_value in final_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

if __name__ == "__main__":
    main()


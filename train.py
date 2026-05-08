import torch
import torch.nn.functional as F
import torch.optim as optim

# --- IMPORT TỪ CÁC THƯ MỤC CỦA BẠN ---
from preprocessing.preprocessing import Preprocessor
from models.two_tower import TwoTowerModel
from dataset import get_dataloader

# --- HÀM LOSS ---
def in_batch_softmax_loss(user_vecs, item_vecs, temperature=0.1):
    """
    Hàm tính In-batch Softmax Loss.
    Mỗi user trong batch sẽ lấy item tương ứng làm Positive (đường chéo),
    và các item của người khác trong batch làm Negative.
    """
    logits = torch.matmul(user_vecs, item_vecs.T) / temperature
    labels = torch.arange(user_vecs.size(0)).to(user_vecs.device)
    return F.cross_entropy(logits, labels)

# --- VÒNG LẶP HUẤN LUYỆN CHÍNH ---
def main():
    # 1. Cấu hình (Hyperparameters)
    EPOCHS = 50
    BATCH_SIZE = 1024
    LR = 0.001
    SEQ_LEN = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {DEVICE}")

    # 2. Xử lý dữ liệu
    print("Đang tiền xử lý dữ liệu...")
    prep = Preprocessor(seq_len=SEQ_LEN)
    
    # Chỉ cần 1 dòng này là lấy được cả 3 biến (Đảm bảo preprocessing.py đã có dòng return)
    ratings, users, user_hist_array = prep.preprocess("dataset/ratings.dat", "dataset/users.dat")

    # 3. Lấy số lượng User và Movie để khởi tạo Embedding
    num_users = len(prep.user_encoder.classes_)
    num_movies = len(prep.movie_encoder.classes_)
    print(f"Số lượng Users: {num_users}, Số lượng Movies: {num_movies}")

    # 4. Khởi tạo DataLoader
    print("Đang chuẩn bị DataLoader...")
    train_loader = get_dataloader(ratings, users, user_hist_array, batch_size=BATCH_SIZE)

    # 5. Khởi tạo Model và Optimizer
    print("Đang khởi tạo mô hình...")
    model = TwoTowerModel(num_users=num_users, num_movies=num_movies).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 6. Quá trình Training
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Chuyển dữ liệu lên GPU/CPU
            user_inputs = {
                'user_id': batch['user_id'].to(DEVICE),
                'gender': batch['gender'].to(DEVICE),
                'occupation': batch['occupation'].to(DEVICE),
                'movie_hist': batch['user_hist'].to(DEVICE)
            }
            target_movies = batch['target_movie'].to(DEVICE)

            # Xóa gradient cũ
            optimizer.zero_grad()

            # Forward pass: Lấy vector từ 2 tháp
            user_vec, item_vec = model(user_inputs, target_movies)

            # Tính loss
            loss = in_batch_softmax_loss(user_vec, item_vec, temperature=0.1)

            # Backward pass và cập nhật trọng số
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # In log mỗi 100 batch
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # In Loss trung bình sau mỗi Epoch
        avg_loss = total_loss / len(train_loader)
        print(f"==> Hoàn thành Epoch {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.4f}\n")

    # 7. Lưu mô hình
    torch.save(model.state_dict(), "two_tower_model.pth")
    print("Đã lưu mô hình thành công tại 'two_tower_model.pth'!")

if __name__ == "__main__":
    main()
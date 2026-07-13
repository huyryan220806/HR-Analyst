"""
train.py
Module huấn luyện mô hình MLP cho bài toán dự đoán nghỉ việc nhân viên.
- Sử dụng BCEWithLogitsLoss với pos_weight để xử lý mất cân bằng lớp
- Huấn luyện trên tập Train, đánh giá trên tập Validation
- Lưu mô hình tốt nhất (best_model.pth)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import time
import copy


class HRDataset(Dataset):
    """PyTorch Dataset cho dữ liệu HR Attrition."""

    def __init__(self, X, y):
        """
        Args:
            X (np.ndarray): Đặc trưng, shape (n_samples, n_features).
            y (np.ndarray): Nhãn, shape (n_samples,).
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # shape (n, 1) cho BCEWithLogitsLoss

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def calculate_pos_weight(y_train):
    """
    Tính pos_weight cho BCEWithLogitsLoss để xử lý mất cân bằng lớp.
    pos_weight = số mẫu negative / số mẫu positive
    """
    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    pos_weight = n_negative / n_positive
    print(f"  Negative samples: {int(n_negative)}, Positive samples: {int(n_positive)}")
    print(f"  pos_weight = {pos_weight:.2f}")
    return torch.FloatTensor([pos_weight])


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Huấn luyện mô hình 1 epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        # Lưu predictions
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(y_batch.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return avg_loss, acc, f1


def validate(model, dataloader, criterion, device):
    """Đánh giá mô hình trên tập validation."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y_batch.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return avg_loss, acc, f1


def train_model(model, X_train, y_train, X_val, y_val,
                num_epochs=100, batch_size=32, learning_rate=0.001,
                patience=15, save_path='best_model.pth', device=None):
    """
    Huấn luyện mô hình với Early Stopping.

    Args:
        model: Mô hình PyTorch.
        X_train, y_train: Dữ liệu huấn luyện.
        X_val, y_val: Dữ liệu kiểm thử.
        num_epochs: Số epoch tối đa.
        batch_size: Kích thước batch.
        learning_rate: Tốc độ học.
        patience: Số epoch chờ trước khi early stopping.
        save_path: Đường dẫn lưu mô hình tốt nhất.
        device: Thiết bị tính toán (CPU/GPU).

    Returns:
        dict: Lịch sử huấn luyện (loss, accuracy, f1 cho train và val).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("5. HUẤN LUYỆN MÔ HÌNH")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping patience: {patience}")

    model = model.to(device)

    # Tạo DataLoader
    train_dataset = HRDataset(X_train, y_train)
    val_dataset = HRDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Tính pos_weight cho class imbalance
    print(f"\n  Xử lý mất cân bằng lớp (Class Imbalance):")
    pos_weight = calculate_pos_weight(y_train).to(device)

    # Định nghĩa Loss function và Optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                      patience=7)

    # Biến theo dõi
    best_val_f1 = 0.0
    best_model_state = None
    epochs_no_improve = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} {'Train Acc':>10} {'Train F1':>9} | "
          f"{'Val Loss':>9} {'Val Acc':>9} {'Val F1':>8} | {'Status'}")
    print("-" * 95)

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Training
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        # Validation
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device)

        # Lưu lịch sử
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # Cập nhật learning rate
        scheduler.step(val_f1)

        # Kiểm tra và lưu mô hình tốt nhất
        status = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            status = f"✓ Best (F1={val_f1:.4f})"

            # Lưu mô hình
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, save_path)
        else:
            epochs_no_improve += 1
            status = f"  (no improve: {epochs_no_improve}/{patience})"

        # In tiến trình mỗi 5 epoch hoặc khi có cải thiện
        if epoch % 5 == 0 or status.startswith("✓") or epoch == 1:
            print(f"{epoch:>6} | {train_loss:>10.4f} {train_acc:>10.4f} {train_f1:>9.4f} | "
                  f"{val_loss:>9.4f} {val_acc:>9.4f} {val_f1:>8.4f} | {status}")

        # Early Stopping
        if epochs_no_improve >= patience:
            print(f"\n  → Early Stopping tại epoch {epoch} (không cải thiện sau {patience} epochs)")
            break

    elapsed = time.time() - start_time
    print("-" * 95)
    print(f"\n  Huấn luyện hoàn tất trong {elapsed:.1f} giây")
    print(f"  Mô hình tốt nhất: Val F1-Score = {best_val_f1:.4f}")
    print(f"  Đã lưu mô hình tốt nhất vào: '{save_path}'")

    # Load lại mô hình tốt nhất
    model.load_state_dict(best_model_state)

    return history


def plot_training_history(history):
    """Vẽ biểu đồ lịch sử huấn luyện."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Lịch sử Huấn luyện Mô hình', fontsize=16, fontweight='bold')
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_title('Accuracy', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1-Score
    axes[2].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[2].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    axes[2].set_title('F1-Score', fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1-Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Đã lưu biểu đồ lịch sử huấn luyện vào 'training_history.png'")

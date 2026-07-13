"""
evaluate.py
Module đánh giá mô hình trên tập Test.
- Tải mô hình tốt nhất từ file .pth
- Dự báo trên tập Test
- Tính các chỉ số: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Vẽ Confusion Matrix và ROC Curve
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from train import HRDataset
from torch.utils.data import DataLoader


def load_best_model(model, save_path='best_model.pth', device=None):
    """
    Tải mô hình tốt nhất từ file checkpoint.

    Args:
        model: Mô hình PyTorch (cần khởi tạo trước với đúng kiến trúc).
        save_path: Đường dẫn file .pth.
        device: Thiết bị tính toán.

    Returns:
        model: Mô hình đã load trọng số.
        checkpoint: Thông tin checkpoint.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Đã tải mô hình từ '{save_path}'")
    print(f"  Epoch tốt nhất: {checkpoint['epoch']}")
    print(f"  Val F1-Score: {checkpoint['val_f1']:.4f}")
    print(f"  Val Accuracy: {checkpoint['val_acc']:.4f}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

    return model, checkpoint


def predict(model, X_test, device=None, batch_size=64):
    """
    Dự báo trên tập dữ liệu test.

    Args:
        model: Mô hình đã huấn luyện.
        X_test: Dữ liệu test (numpy array).
        device: Thiết bị tính toán.
        batch_size: Kích thước batch.

    Returns:
        y_pred: Nhãn dự đoán (0 hoặc 1).
        y_proba: Xác suất dự đoán (0.0 - 1.0).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    all_proba = []

    # Tạo dummy labels (không dùng khi predict)
    dummy_y = np.zeros(len(X_test), dtype=np.float32)
    test_dataset = HRDataset(X_test, dummy_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            proba = torch.sigmoid(outputs)
            all_proba.extend(proba.cpu().numpy().flatten())

    y_proba = np.array(all_proba)
    y_pred = (y_proba >= 0.5).astype(int)

    return y_pred, y_proba


def evaluate_model(y_test, y_pred, y_proba):
    """
    Đánh giá toàn diện kết quả dự báo trên tập test.

    Args:
        y_test: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        y_proba: Xác suất dự đoán.
    """
    print("\n" + "=" * 60)
    print("6. ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 60)

    # Các chỉ số đánh giá
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'Chỉ số':<25} {'Giá trị':>10}")
    print("-" * 37)
    print(f"{'Accuracy':<25} {acc:>10.4f}")
    print(f"{'Precision':<25} {prec:>10.4f}")
    print(f"{'Recall':<25} {rec:>10.4f}")
    print(f"{'F1-Score':<25} {f1:>10.4f}")
    print(f"{'ROC-AUC':<25} {roc_auc:>10.4f}")

    # Classification Report chi tiết
    print(f"\n{'Classification Report chi tiết':}")
    print("-" * 55)
    target_names = ['Không nghỉ (No)', 'Nghỉ việc (Yes)']
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc}


def plot_confusion_matrix(y_test, y_pred):
    """Vẽ Confusion Matrix."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Không nghỉ (No)', 'Nghỉ việc (Yes)'],
                yticklabels=['Không nghỉ (No)', 'Nghỉ việc (Yes)'],
                annot_kws={'size': 16}, ax=ax)
    ax.set_title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dự đoán (Predicted)', fontsize=12)
    ax.set_ylabel('Thực tế (Actual)', fontsize=12)

    # Thêm chú thích
    tn, fp, fn, tp = cm.ravel()
    info_text = f'TN={tn}  FP={fp}\nFN={fn}  TP={tp}'
    ax.text(2.3, 0.5, info_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Đã lưu Confusion Matrix vào 'confusion_matrix.png'")

    return cm


def plot_roc_curve(y_test, y_proba):
    """Vẽ đường cong ROC."""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#e74c3c', lw=2.5,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--',
            label='Random Classifier (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title('Đường cong ROC (Receiver Operating Characteristic)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Đã lưu đường cong ROC vào 'roc_curve.png'")

    return roc_auc

"""
model.py
Định nghĩa kiến trúc mạng nơ-ron MLP (Multi-Layer Perceptron) bằng PyTorch
cho bài toán dự đoán nghỉ việc của nhân viên (phân loại nhị phân).
"""

import torch
import torch.nn as nn


class HRAttritionMLP(nn.Module):
    """
    Mạng MLP cho bài toán dự đoán nghỉ việc.

    Kiến trúc:
        Input → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
              → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
              → Linear(64)  → BatchNorm → ReLU → Dropout(0.2)
              → Linear(32)  → BatchNorm → ReLU → Dropout(0.2)
              → Linear(1)   → Output (logit)

    Sử dụng BCEWithLogitsLoss nên đầu ra là logit (chưa qua Sigmoid).
    """

    def __init__(self, input_dim, dropout_rate=0.3):
        """
        Args:
            input_dim (int): Số lượng đặc trưng đầu vào.
            dropout_rate (float): Tỷ lệ dropout cho các tầng ẩn.
        """
        super(HRAttritionMLP, self).__init__()

        self.network = nn.Sequential(
            # Tầng ẩn 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Tầng ẩn 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Tầng ẩn 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.67),   # dropout nhẹ hơn

            # Tầng ẩn 4
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.67),

            # Tầng đầu ra - 1 nơ-ron (logit cho BCEWithLogitsLoss)
            nn.Linear(32, 1)
        )

        # Khởi tạo trọng số theo phương pháp Kaiming (He initialization)
        self._initialize_weights()

    def _initialize_weights(self):
        """Khởi tạo trọng số bằng phương pháp Kaiming He cho các tầng Linear."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Tensor đầu vào có shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Logit đầu ra có shape (batch_size, 1).
        """
        return self.network(x)

    def predict_proba(self, x):
        """
        Dự đoán xác suất nghỉ việc.

        Args:
            x (torch.Tensor): Tensor đầu vào.

        Returns:
            torch.Tensor: Xác suất nghỉ việc (đã qua Sigmoid), shape (batch_size, 1).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            proba = torch.sigmoid(logits)
        return proba


def get_model_summary(model, input_dim):
    """In tóm tắt kiến trúc mô hình."""
    print("=" * 60)
    print("KIẾN TRÚC MÔ HÌNH - HRAttritionMLP")
    print("=" * 60)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTổng số tham số: {total_params:,}")
    print(f"Số tham số huấn luyện được: {trainable_params:,}")
    print(f"Kích thước đầu vào: {input_dim}")
    return total_params


if __name__ == '__main__':
    # Chạy thử - giả sử có 44 features sau One-Hot Encoding
    input_dim = 44
    model = HRAttritionMLP(input_dim=input_dim)
    get_model_summary(model, input_dim)

    # Test forward pass
    dummy_input = torch.randn(8, input_dim)
    output = model(dummy_input)
    print(f"\nTest forward pass: input shape={dummy_input.shape} → output shape={output.shape}")

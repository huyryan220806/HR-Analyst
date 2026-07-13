# Kế hoạch Dự án: Dự đoán nguy cơ nghỉ việc của nhân viên (HR Analyst) bằng PyTorch

Dự án này nhằm xây dựng một mô hình học sâu sử dụng thư viện **PyTorch** để dự đoán khả năng nghỉ việc (`Attrition`) của nhân viên dựa trên tập dữ liệu nhân sự từ IBM (`WA_Fn-UseC_-HR-Employee-Attrition.csv`). Kế hoạch dưới đây chi tiết hóa các bước thực hiện từ phân tích dữ liệu, tiền xử lý, phân chia dữ liệu, xây dựng và huấn luyện mô hình cho đến bước đánh giá cuối cùng.

---

## User Review Required

> [!IMPORTANT]
> - **Môi trường chạy**: Bạn sẽ chạy code này trên Google Colab. Chúng tôi sẽ thiết kế mã nguồn dưới dạng các file script Python hoặc các cell code dễ dàng copy-paste vào Google Colab.
> - **Class Imbalance (Mất cân bằng lớp)**: Tỷ lệ nhân viên nghỉ việc (`Attrition = Yes`) thường thấp hơn nhiều so với số người ở lại (`Attrition = No`). Chúng ta cần áp dụng các kỹ thuật như chọn hàm Loss phù hợp (ví dụ: `BCEWithLogitsLoss` có cấu hình `pos_weight`) hoặc tính toán chỉ số F1-Score thay vì chỉ dùng Accuracy để đánh giá chính xác mô hình.
> - **Đặc trưng không cần thiết (Redundant Features)**: Một số cột như `EmployeeCount` (luôn là 1), `Over18` (luôn là 'Y'), `StandardHours` (luôn là 80) và `EmployeeNumber` (ID định danh) không mang lại giá trị dự báo và sẽ được loại bỏ trong quá trình tiền xử lý.

---

## Proposed Changes

Chúng ta sẽ tạo các file mã nguồn Python và cấu trúc dự án như sau tại thư mục `D:\VLU\253\PTDLHS\DoAnCK_HS`:
- `data_processing.py`: Phân tích dữ liệu sơ bộ (EDA), xử lý dữ liệu (chuyển đổi category sang numeric, chuẩn hóa dữ liệu số) và chia tập dữ liệu train/val/test phân tầng (Stratified Split).
- `model.py`: Định nghĩa kiến trúc mạng mạng nơ-ron đa tầng (Multi-Layer Perceptron - MLP) bằng PyTorch.
- `train.py`: Huấn luyện mô hình, kiểm thử trên tập validation, lưu lại mô hình tốt nhất (`best_model.pth`).
- `evaluate.py`: Đánh giá hiệu năng của mô hình tốt nhất trên tập test độc lập.
- `notebook_colab.ipynb` (hoặc hướng dẫn tương đương): File tổng hợp dạng Jupyter Notebook để bạn dễ dàng chạy trực tiếp trên Google Colab.

---

### 1. Phân tích & Tiền xử lý dữ liệu (Step 1 & Step 2)

#### [NEW] [data_processing.py](file:///D:/VLU/253/PTDLHS/DoAnCK_HS/data_processing.py)
File này chịu trách nhiệm:
- Đọc file CSV `WA_Fn-UseC_-HR-Employee-Attrition.csv`.
- Phân tích thống kê mô tả, kiểm tra giá trị thiếu (Missing values).
- Xử lý các biến phân loại (Categorical columns) bằng phương pháp Label Encoding hoặc One-Hot Encoding.
- Chuẩn hóa các biến số (Numerical columns) bằng `StandardScaler`.
- Chia dữ liệu thành 3 phần: **Train (70%)**, **Validation (15%)**, **Test (15%)** bằng phương pháp phân tầng (`StratifiedShuffleSplit` từ thư viện `scikit-learn`) dựa trên cột mục tiêu `Attrition` để đảm bảo tỷ lệ nghỉ việc đồng đều giữa các tập.
- Lưu trữ các tập dữ liệu đã xử lý để chuẩn bị cho quá trình huấn luyện.

---

### 2. Định nghĩa Mô hình (Step 3)

#### [NEW] [model.py](file:///D:/VLU/253/PTDLHS/DoAnCK_HS/model.py)
File này định nghĩa mô hình PyTorch:
- Xây dựng một mạng MLP (Multi-Layer Perceptron) kế thừa từ `torch.nn.Module`.
- Đầu vào: Số lượng đặc trưng (features) sau khi xử lý dữ liệu.
- Các tầng ẩn (Hidden layers): Sử dụng các tầng fully connected (`nn.Linear`), đi kèm với Batch Normalization (`nn.BatchNorm1d`), Dropout để chống quá khớp (overfitting) và hàm kích hoạt `nn.ReLU`.
- Đầu ra: 1 nơ-ron (sử dụng hàm kích hoạt Sigmoid cuối cùng hoặc kết hợp trực tiếp vào hàm Loss `BCEWithLogitsLoss` để tính xác suất nghỉ việc).

---

### 3. Huấn luyện Mô hình (Step 3)

#### [NEW] [train.py](file:///D:/VLU/253/PTDLHS/DoAnCK_HS/train.py)
File này thực hiện việc huấn luyện:
- Chuyển đổi dữ liệu từ file xử lý sang PyTorch `Dataset` và `DataLoader`.
- Định nghĩa hàm tối ưu (Optimizer) ví dụ: `Adam` hoặc `SGD` và hàm lỗi (Loss function): `BCEWithLogitsLoss` (phù hợp với bài toán phân loại nhị phân và dữ liệu mất cân bằng).
- Vòng lặp huấn luyện (Training loop):
  - Huấn luyện trên tập Train.
  - Đánh giá Loss và F1-Score trên tập Validation sau mỗi Epoch.
  - Theo dõi Metric tốt nhất trên tập Validation (ví dụ: F1-Score cao nhất hoặc Loss thấp nhất).
  - Lưu checkpoint mô hình tốt nhất thành file `best_model.pth` (hoặc `best_model.pt`).

---

### 4. Đánh giá kết quả (Step 4)

#### [NEW] [evaluate.py](file:///D:/VLU/253/PTDLHS/DoAnCK_HS/evaluate.py)
File này thực hiện đánh giá mô hình:
- Tải mô hình đã lưu `best_model.pth`.
- Chạy dự báo trên tập Test (dữ liệu hoàn toàn mới chưa được huấn luyện).
- Tính toán và in ra các chỉ số:
  - **Accuracy** (Độ chính xác tổng thể).
  - **Precision** (Độ chính xác của dự báo nghỉ việc).
  - **Recall** (Tỷ lệ phát hiện được nhân viên thực sự nghỉ việc).
  - **F1-Score** (Trung bình điều hòa giữa Precision và Recall).
  - **Confusion Matrix** (Ma trận nhầm lẫn để xem chi tiết dự báo đúng/sai ở từng lớp).
  - **ROC-AUC** (Đánh giá khả năng phân biệt lớp của mô hình).

---

## Verification Plan

### Automated Tests
- Chạy kiểm tra định dạng dữ liệu đầu vào và đầu ra của file tiền xử lý bằng cách in kích thước các tập dữ liệu `(X_train.shape, y_train.shape)`.
- Chạy huấn luyện thử nghiệm với số epoch nhỏ (ví dụ: 5 epoch) để đảm bảo không gặp lỗi cú pháp hay lỗi Tensor shape trong PyTorch.
- Kiểm tra tính hợp lệ của file `.pth` được lưu.

### Manual Verification
- Xác minh tỷ lệ phân bố lớp (`Attrition` Yes/No) trên cả 3 tập Train/Val/Test xem đã đúng tỷ lệ phân tầng chưa.
- Kiểm tra kết quả đánh giá trên tập Test để đảm bảo không bị rò rỉ dữ liệu (Data Leakage) từ tập huấn luyện.

# Dự Án Phân Tích Dữ Liệu Nhân Sự Và Dự Đoán Nguy Cơ Nghỉ Việc Của Nhân Viên

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

Dự án cuối kỳ môn Phân tích dữ liệu và Học sâu thực hiện bài toán phân tích dữ liệu nhân sự (HR Analytics) và xây dựng mô hình mạng nơ-ron nhân tạo MLP (Multi-Layer Perceptron) để dự đoán nguy cơ nghỉ việc chủ động (Attrition) của nhân viên dựa trên bộ dữ liệu IBM HR Analytics.

---

## Thông Tin Đồ Án

*   **Lớp học phần (LHP):** 253_71ITDS30203_0203
*   **Giảng viên hướng dẫn (GVHD):** ThS. Nguyễn Thị Mỹ Linh
*   **Sinh viên thực hiện 1:** Nguyễn Đình Huy (MSSV: 2474802010140)
*   **Sinh viên thực hiện 2:** Lê Quyết Tiến (MSSV: 2474802010386)

---

## Quy Trình Xử Lý Dữ Liệu

1.  **Tiền xử lý và Làm sạch:**
    *   Loại bỏ các trường dữ liệu không mang thông tin dự báo hoặc hằng số (EmployeeCount, StandardHours, EmployeeNumber, Over18).
    *   Loại bỏ các trường Rate có tương quan yếu (DailyRate, HourlyRate, MonthlyRate).
    *   Mã hóa One-Hot Encoding cho các biến danh mục.
2.  **Feature Engineering:**
    *   Xây dựng đặc trưng `CompanyLoyalty` (YearsAtCompany / (TotalWorkingYears + 1)).
    *   Xây dựng đặc trưng `PromotionStagnation` (YearsSinceLastPromotion / (YearsAtCompany + 1)).
    *   Xây dựng đặc trưng `IncomePerYear` (MonthlyIncome / (TotalWorkingYears + 1)).
3.  **Chuẩn hóa:**
    *   Áp dụng StandardScaler để đưa các biến số về dạng phân phối chuẩn có trung bình bằng 0 và phương sai bằng 1 (fit chỉ trên tập Train).
4.  **Cân bằng dữ liệu:**
    *   Sử dụng phương pháp SMOTE trên tập Train để giải quyết mất cân bằng lớp (Yes: 16.1%, No: 83.9%). Tập Validation và Test giữ nguyên phân phối thực.

---

## Kiến Trúc Mô Hình Đề Xuất

Mô hình được xây dựng dạng Feedforward Neural Network (MLP) sử dụng PyTorch:

*   **Input Layer:** 44 node đầu vào (sau khi thực hiện mã hóa và feature engineering).
*   **Hidden Layer 1:** 32 node, Batch Normalization, hàm kích hoạt ReLU, Dropout (0.3).
*   **Hidden Layer 2:** 16 node, Batch Normalization, hàm kích hoạt ReLU, Dropout (0.3).
*   **Output Layer:** 1 node, sigmoid để tính toán xác suất.
*   **Thuật toán tối ưu:** AdamW với Weight Decay (L2 Regularization) bằng 1.0e-2.
*   **Hàm mất mát (Loss):** BCEWithLogitsLoss tích hợp pos_weight để bù đắp sự mất cân bằng.
*   **Học tập thích ứng:** LR Scheduler MultiStepLR với milestones=[12, 24, 30] và hệ số giảm gamma=0.35.

---

## Kết Quả Đánh Giá Trên Tập Kiểm Thử (Test Set)

Sau khi quét tìm ngưỡng tối ưu F1-Score trên tập Validation, ngưỡng tối ưu xác định là 0.20. Kết quả đo được trên tập Test độc lập:

| Chỉ số (Metric) | Kết quả đạt được (Score) |
|---|---|
| Accuracy | 77.38% |
| Recall | 19.44% |
| Precision | 25.00% |
| F1-Score | 21.88% |
| ROC-AUC | 0.5482 |

---

## Cấu Trúc Thư Mục

*   `WA_Fn-UseC_-HR-Employee-Attrition.csv` - File dữ liệu đầu vào.
*   `DoAn_HR_Attrition.py` - Script chính thực hiện quy trình từ EDA, huấn luyện đến đánh giá.
*   `Output/Final/` - Thư mục lưu trữ kết quả đầu ra:
    *   `best_model.pth` - Trọng số mô hình tốt nhất lưu ở epoch 22.
    *   `eda_plots.png` - Biểu đồ phân tích khám phá dữ liệu.
    *   `correlation_matrix.png` - Heatmap ma trận tương quan.
    *   `correlation_with_attrition.png` - Tương quan các biến với Attrition.
    *   `training_history.png` - Đồ thị lịch sử huấn luyện.
    *   `test_evaluation.png` - Các đồ thị ma trận nhầm lẫn, ROC, PR và phân tích ngưỡng.
    *   `feature_importance.png` - Độ quan trọng đặc trưng bằng phương pháp Permutation Importance.
    *   `mlp_architecture.png` - Sơ đồ kiến trúc mạng MLP đầu đề xuất.
*   `pipeline_flowchart.png` - Sơ đồ quy trình hệ thống.
*   `BaoCao_HR_Attrition_HoanThien.docx` - Báo cáo đồ án hoàn chỉnh.

---

## Hướng Dẫn Chạy Chương Trình

1.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install torch pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
    ```
2.  Đặt file dữ liệu `WA_Fn-UseC_-HR-Employee-Attrition.csv` cùng thư mục với script chạy.
3.  Chạy script huấn luyện và đánh giá:
    ```bash
    python DoAn_HR_Attrition.py
    ```

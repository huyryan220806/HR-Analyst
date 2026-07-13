# -*- coding: utf-8 -*-
"""
=============================================================================
ĐỒ ÁN CUỐI KỲ: PHÂN TÍCH DỮ LIỆU NHÂN SỰ (HR ANALYST)
DỰ ĐOÁN NGUY CƠ NGHỈ VIỆC CỦA NHÂN VIÊN
=============================================================================
Môn: Phân tích dữ liệu và Học sâu
Chủ đề 23: Phân tích dữ liệu nhân sự (HR Analyst) dự đoán nghỉ việc
            của nhân viên. Học máy dự báo nguy cơ nghỉ việc,
            hỗ trợ chính sách giữ chân nhân tài.
Framework: PyTorch
Dataset: IBM HR Analytics Employee Attrition & Performance
=============================================================================

Phiên bản 5 - Tập trung: Đường Train & Val sát nhau
   Mô hình NHỎ: 32161 (phù hợp ~1000 mẫu train)
   Regularization MẠNH: Dropout 0.5, Weight Decay 5e-3
   Label Smoothing: giảm confidence  train loss không giảm quá sâu
   Mixup Data Augmentation: nội suy dữ liệu  tăng generalization
   Feature Engineering (3 biến), loại biến nhiễu
   Threshold tuning F1 trên Validation
   Biểu đồ: smoothing, best epoch, 2x2 eval, correlation bar chart

Hướng dẫn Google Colab:
1. Upload file 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
2. Mỗi phần "# %%" = 1 cell, chạy lần lượt
=============================================================================
"""

# %% [markdown]
# # ĐỒ ÁN: DỰ ĐOÁN NGUY CƠ NGHỈ VIỆC CỦA NHÂN VIÊN
# ## Phân tích dữ liệu nhân sự (HR Analyst) bằng PyTorch

# %% ============================
# CELL 1: IMPORT THƯ VIỆN
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import time
import copy

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')
print("\n Import thư viện thành công!")


# %% ============================
# CELL 2: ĐỌC VÀ TÌM HIỂU DỮ LIỆU
# ============================
# *** BƯỚC 1: Download dataset, tìm hiểu về dữ liệu ***

# --- Trên Google Colab: Upload file CSV ---
# from google.colab import files
# uploaded = files.upload()
# filepath = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'

# --- Trên máy local ---
filepath = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'

df = pd.read_csv(filepath)

print("=" * 65)
print("BƯỚC 1: TÌM HIỂU VỀ DỮ LIỆU")
print("=" * 65)
print(f"\nKích thước dữ liệu: {df.shape[0]} dòng × {df.shape[1]} cột")
print("\n--- 5 dòng đầu tiên ---")
df.head()


# %% ============================
# CELL 3: PHÂN TÍCH DỮ LIỆU SƠ BỘ (EDA)
# ============================
print("=" * 65)
print("PHÂN TÍCH DỮ LIỆU SƠ BỘ (EDA)")
print("=" * 65)

print("\n--- Thông tin kiểu dữ liệu ---")
print(df.dtypes)
print(f"\n--- Kiểm tra giá trị thiếu ---")
missing = df.isnull().sum()
if missing.sum() == 0:
    print(" Không có giá trị thiếu (Missing Values) trong dữ liệu!")
else:
    print(missing[missing > 0])

print(f"\n--- Thống kê mô tả (biến số) ---")
df.describe().T


# %% ============================
# CELL 4: PHÂN TÍCH BIẾN MỤC TIÊU
# ============================
print("=" * 65)
print("PHÂN BỐ BIẾN MỤC TIÊU - ATTRITION")
print("=" * 65)

attrition_counts = df['Attrition'].value_counts()
attrition_pct = df['Attrition'].value_counts(normalize=True) * 100
print(f"\n  No  (Không nghỉ): {attrition_counts['No']:>5} ({attrition_pct['No']:.1f}%)")
print(f"  Yes (Nghỉ việc) : {attrition_counts['Yes']:>5} ({attrition_pct['Yes']:.1f}%)")
print(f"\n  Dữ liệu mất cân bằng! Tỷ lệ nghỉ việc chỉ ~{attrition_pct['Yes']:.0f}%")

print(f"\n--- Các cột cần loại bỏ ---")
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
for col in constant_cols:
    print(f"   {col} = '{df[col].unique()[0]}' (giá trị hằng)")
print(f"   EmployeeNumber = ID nhân viên")
print(f"   DailyRate, HourlyRate, MonthlyRate = Nhiễu (|r| < 0.02)")

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"\nSố cột phân loại: {len(cat_cols)}")
print(f"Số cột số: {len(num_cols)}")
print(f"\nCác cột phân loại:")
for col in cat_cols:
    print(f"   {col}: {df[col].nunique()} giá trị - {list(df[col].unique())}")


# %% ============================
# CELL 5: TRỰC QUAN HÓA DỮ LIỆU
# ============================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('TRỰC QUAN HÓA DỮ LIỆU - HR Employee Attrition',
             fontsize=18, fontweight='bold', y=1.02)

colors = ['#27ae60', '#e74c3c']
colors_palette = {'No': '#27ae60', 'Yes': '#e74c3c'}

attrition_counts = df['Attrition'].value_counts()
axes[0, 0].pie(attrition_counts.values, labels=['Không nghỉ (No)', 'Nghỉ việc (Yes)'],
               autopct='%1.1f%%', colors=colors, startangle=90,
               explode=(0, 0.08), textprops={'fontsize': 12}, shadow=True)
axes[0, 0].set_title('Tỷ lệ Nghỉ việc (Attrition)', fontsize=13, fontweight='bold')

for label, color in zip(['No', 'Yes'], colors):
    subset = df[df['Attrition'] == label]['Age']
    axes[0, 1].hist(subset, bins=20, alpha=0.7, label=f'Attrition={label}',
                    color=color, edgecolor='white')
axes[0, 1].set_title('Phân bố Tuổi theo Attrition', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Tuổi')
axes[0, 1].set_ylabel('Số lượng')
axes[0, 1].legend()

sns.boxplot(data=df, x='Attrition', y='MonthlyIncome', palette=colors_palette, ax=axes[0, 2])
axes[0, 2].set_title('Thu nhập hàng tháng theo Attrition', fontsize=13, fontweight='bold')

overtime_attrition = pd.crosstab(df['OverTime'], df['Attrition'], normalize='index') * 100
overtime_attrition.plot(kind='bar', ax=axes[1, 0], color=colors, edgecolor='white', width=0.6)
axes[1, 0].set_title('Tỷ lệ Attrition theo OverTime', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Tăng ca (OverTime)')
axes[1, 0].set_ylabel('Tỷ lệ (%)')
axes[1, 0].legend(title='Attrition')
axes[1, 0].tick_params(axis='x', rotation=0)

sns.boxplot(data=df, x='Attrition', y='YearsAtCompany', palette=colors_palette, ax=axes[1, 1])
axes[1, 1].set_title('Số năm tại công ty theo Attrition', fontsize=13, fontweight='bold')

job_sat = pd.crosstab(df['JobSatisfaction'], df['Attrition'], normalize='columns') * 100
job_sat.plot(kind='bar', ax=axes[1, 2], color=colors, edgecolor='white', width=0.6)
axes[1, 2].set_title('Mức độ hài lòng công việc theo Attrition', fontsize=13, fontweight='bold')
axes[1, 2].set_xlabel('JobSatisfaction (1=Thấp  4=Cao)')
axes[1, 2].set_ylabel('Tỷ lệ (%)')
axes[1, 2].legend(title='Attrition')
axes[1, 2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Đã lưu 'eda_plots.png'")


# %% ============================
# CELL 6: PHÂN TÍCH TƯƠNG QUAN
# ============================
print("=" * 65)
print("TƯƠNG QUAN CỦA CÁC ĐẶC TRƯNG VỚI ATTRITION")
print("=" * 65)

df_corr = df.copy()
df_corr['Attrition'] = df_corr['Attrition'].map({'Yes': 1, 'No': 0})
df_corr['OverTime'] = df_corr['OverTime'].map({'Yes': 1, 'No': 0})
df_corr['Gender'] = df_corr['Gender'].map({'Male': 1, 'Female': 0})

cols_exclude = ['EmployeeCount', 'StandardHours', 'EmployeeNumber']
numeric_cols_corr = [c for c in df_corr.select_dtypes(include=[np.number]).columns
                     if c not in cols_exclude]
corr_matrix = df_corr[numeric_cols_corr].corr()
attrition_corr = corr_matrix['Attrition'].drop('Attrition').sort_values()

print("\nTop 10 tương quan dương (tăng nguy cơ nghỉ):")
for feat, val in attrition_corr.sort_values(ascending=False).head(10).items():
    print(f"  {feat:<28} {val:>+.4f} {'' * int(abs(val) * 50)}")
print("\nTop 10 tương quan âm (giảm nguy cơ nghỉ):")
for feat, val in attrition_corr.head(10).items():
    print(f"  {feat:<28} {val:>+.4f} {'' * int(abs(val) * 50)}")

# Biểu đồ thanh tương quan
fig, ax = plt.subplots(figsize=(10, 10))
bar_colors = ['#e74c3c' if v > 0 else '#3498db' for v in attrition_corr.values]
ax.barh(range(len(attrition_corr)), attrition_corr.values,
        color=bar_colors, edgecolor='white', height=0.7)
ax.set_yticks(range(len(attrition_corr)))
ax.set_yticklabels(attrition_corr.index, fontsize=10)
ax.set_xlabel('Hệ số tương quan Pearson', fontsize=12)
ax.set_title('Tương quan của từng Đặc trưng với Attrition\n(Đỏ = Tăng nguy cơ, Xanh = Giảm nguy cơ)',
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('correlation_with_attrition.png', dpi=150, bbox_inches='tight')
plt.show()

# Heatmap top 15
top_n_corr = 15
top_corr_features = attrition_corr.abs().sort_values(ascending=False).head(top_n_corr).index.tolist()
top_corr_features = ['Attrition'] + top_corr_features
corr_selected = df_corr[top_corr_features].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_selected, dtype=bool))
sns.heatmap(corr_selected, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, annot_kws={'size': 9},
            vmin=-1, vmax=1)
plt.title(f'Ma trận Tương quan - Top {top_n_corr} đặc trưng liên quan Attrition nhất',
          fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Đã lưu biểu đồ tương quan")


# %% ============================
# CELL 7: FEATURE ENGINEERING + TIỀN XỬ LÝ
# ============================
print("=" * 65)
print("BƯỚC 2 (phần 1): FEATURE ENGINEERING + TIỀN XỬ LÝ")
print("=" * 65)

df_processed = df.copy()

# 1. Loại bỏ cột không cần thiết + biến nhiễu
cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber',
                'DailyRate', 'HourlyRate', 'MonthlyRate']
df_processed.drop(columns=cols_to_drop, inplace=True)
print(f"\n Loại bỏ {len(cols_to_drop)} cột: {cols_to_drop}")

# 2. Mã hóa biến mục tiêu
df_processed['Attrition'] = df_processed['Attrition'].map({'Yes': 1, 'No': 0})

# 3. Feature Engineering - 3 biến mới
print(f"\n--- Feature Engineering ---")
df_processed['CompanyLoyalty'] = (
    df_processed['YearsAtCompany'] / (df_processed['TotalWorkingYears'] + 1)
)
print(f"   CompanyLoyalty = YearsAtCompany / (TotalWorkingYears + 1)")

df_processed['PromotionStagnation'] = (
    df_processed['YearsSinceLastPromotion'] / (df_processed['YearsAtCompany'] + 1)
)
print(f"   PromotionStagnation = YearsSinceLastPromotion / (YearsAtCompany + 1)")

df_processed['IncomePerYear'] = (
    df_processed['MonthlyIncome'] / (df_processed['TotalWorkingYears'] + 1)
)
print(f"   IncomePerYear = MonthlyIncome / (TotalWorkingYears + 1)")

new_features = ['CompanyLoyalty', 'PromotionStagnation', 'IncomePerYear']

# 4. Tách X, y
y = df_processed['Attrition'].values
X_df = df_processed.drop(columns=['Attrition'])

cat_columns = X_df.select_dtypes(include=['object']).columns.tolist()
num_columns = X_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 5. One-Hot Encoding
X_encoded = pd.get_dummies(X_df, columns=cat_columns, drop_first=True)
feature_names = X_encoded.columns.tolist()
X = X_encoded.values.astype(np.float32)
y = y.astype(np.float32)

print(f"\n Sau One-Hot Encoding: {X.shape[1]} đặc trưng")


# %% ============================
# CELL 8: CHIA DỮ LIỆU + CHUẨN HÓA
# ============================
# *** BƯỚC 2: Chia Train (70%), Val (15%), Test (15%) - Phân tầng ***

print("=" * 65)
print("BƯỚC 2 (phần 2): CHIA DỮ LIỆU + CHUẨN HÓA")
print("=" * 65)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

# Chuẩn hóa - fit trên Train ONLY
num_col_indices = [feature_names.index(c) for c in num_columns if c in feature_names]
scaler = StandardScaler()
X_train[:, num_col_indices] = scaler.fit_transform(X_train[:, num_col_indices])
X_val[:, num_col_indices] = scaler.transform(X_val[:, num_col_indices])
X_test[:, num_col_indices] = scaler.transform(X_test[:, num_col_indices])
# Lưu lại bản gốc (chưa SMOTE) để ĐO ĐIỂM (giúp biểu đồ khớp hoàn toàn với tập Val)
X_train_real = X_train.copy()
y_train_real = y_train.copy()

print(f"\n Đang áp dụng SMOTE để cân bằng tập Train...")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=RANDOM_STATE)
X_train, y_train = smote.fit_resample(X_train, y_train)
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
print("  Hoàn tất SMOTE (Tập Train 'Học' giờ là 50/50).")
print("  (Tập Train 'Đo điểm' vẫn giữ nguyên gốc 84/16).")

print(f"\n Chia phân tầng (stratified) & Sau SMOTE:")
print(f"  Train:      {len(y_train)} mẫu ({len(y_train)/len(y)*100:.0f}%)")
print(f"  Validation: {len(y_val)} mẫu ({len(y_val)/len(y)*100:.0f}%)")
print(f"  Test:       {len(y_test)} mẫu ({len(y_test)/len(y)*100:.0f}%)")
print(f"  StandardScaler fit trên Train only ")

print(f"\n{'Tập':<12} {'Tổng':>6} {'AT=0':>6} {'AT=1':>6} {'%AT=1':>8}")
print("-" * 42)
for name, y_set in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]:
    n = len(y_set)
    n_pos = int(y_set.sum())
    print(f"  {name:<10} {n:>6} {n - n_pos:>6} {n_pos:>6} {n_pos/n*100:>7.1f}%")


# %% ============================
# CELL 9: PYTORCH DATASET & DATALOADER
# ============================
print("=" * 65)
print("BƯỚC 3 (phần 1): CHUẨN BỊ DỮ LIỆU CHO PYTORCH")
print("=" * 65)


class HRDataset(Dataset):
    """PyTorch Dataset cho dữ liệu HR Attrition."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Tập dùng để HỌC (Đã SMOTE 50/50)
train_dataset = HRDataset(X_train, y_train)
# Tập dùng để ĐO ĐIỂM (Chưa SMOTE 84/16)
eval_train_dataset = HRDataset(X_train_real, y_train_real)

val_dataset = HRDataset(X_val, y_val)
test_dataset = HRDataset(X_test, y_test)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_train_loader = DataLoader(eval_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n  Train: {len(train_loader)} batches ({len(train_dataset)} mẫu)")
print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} mẫu)")
print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} mẫu)")
print(f" Chuẩn bị dữ liệu thành công!")


# Mo hinh (32-16-1) du suc hoc tot cho du lieu ~1000 mau
# GaussianNoise tích hợp trong model (tự tắt khi eval)
# Dropout 0.4 + Regularization

print("=" * 65)
print("BUOC 3 (phan 2): KIEN TRUC MO HINH MLP")
print("=" * 65)


class GaussianNoise(nn.Module):
    """Thêm nhiễu Gaussian vào input CHỈ trong lúc training.
    Khi model.eval(), noise tự động TẮT → Train và Val được đo công bằng."""
    def __init__(self, sigma=0.05):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma > 0:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


class HRAttritionMLP(nn.Module):
    """
    MLP cho du doan nghi viec.
    Kien truc: Input -> GaussianNoise -> 32 -> 16 -> 1
    GaussianNoise tích hợp trong model (tự tắt khi eval).
    Dropout 0.4, BatchNorm, Kaiming Init.
    """
    def __init__(self, input_dim, noise_sigma=0.08):
        super(HRAttritionMLP, self).__init__()

        self.network = nn.Sequential(
            GaussianNoise(noise_sigma),  # Noise tự tắt khi eval!

            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(16, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


INPUT_DIM = X_train.shape[1]
model = HRAttritionMLP(input_dim=INPUT_DIM, noise_sigma=0.08).to(device)

print(f"\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Input:          {INPUT_DIM} features")
print(f"  Kien truc:      {INPUT_DIM} -> GaussianNoise(0.08) -> 32 -> 16 -> 1")
print(f"  Tong tham so:   {total_params:,}")
print(f"  Dropout:        0.4")
print(f"  Regularization: Weight Decay 5e-3 (L2)")
print(f"  Augmentation:   GaussianNoise (tích hợp trong model) + Label Smoothing")


# %% ============================
# CELL 11: CẤU HÌNH HUẤN LUYỆN
# ============================
print("=" * 65)
print("BƯỚC 3 (phần 3): CẤU HÌNH HUẤN LUYỆN")
print("=" * 65)

NUM_EPOCHS = 100          # Cho chạy tối đa 100 epoch, Early Stopping sẽ tự dừng
LEARNING_RATE = 0.001     # LR chuẩn cho AdamW
WEIGHT_DECAY = 5e-3       # Giảm từ 2e-2 xuống 5e-3 (học từ Bank Churn)
LABEL_SMOOTH = 0.05       
MIXUP_ALPHA = 0.0         
NOISE_STD = 0.08          # Noise đã tích hợp trong model, biến này chỉ để hiển thị
PATIENCE = 10             # Early Stopping sớm hơn (học từ Bank Churn)
SAVE_PATH = 'best_model.pth'

# pos_weight cho du lieu mat can bang
n_pos_train = y_train.sum()
n_neg_train = len(y_train) - n_pos_train
pos_weight_value = n_neg_train / n_pos_train
pos_weight = torch.FloatTensor([pos_weight_value]).to(device)

# Loss function: BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Optimizer: AdamW
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Scheduler: ReduceLROnPlateau (học từ Bank Churn)
# Khi Val Loss chững lại 5 epoch → tự động giảm LR đi một nửa
# → Tạo hiệu ứng "đi ngang" TỰ NHIÊN thay vì milestones cứng
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print(f"\n  Cau hinh:")
print(f"    Epochs:          {NUM_EPOCHS}")
print(f"    Batch size:      {BATCH_SIZE}")
print(f"    Learning rate:   {LEARNING_RATE}")
print(f"    Weight decay:    {WEIGHT_DECAY} (L2)")
print(f"    Label smoothing: {LABEL_SMOOTH}")
print(f"    Noise std:       {NOISE_STD}")
print(f"    Optimizer:       AdamW")
print(f"    Loss:            BCEWithLogitsLoss (pos_weight={pos_weight_value:.2f})")
print(f"    Scheduler:       CosineAnnealingLR")
print(f"    Early stop:      patience={PATIENCE}")
print(f"\n  Dropout(0.4) + WD(2.0e-2) + SMOTE (Data Balanced)")


# %% ============================
# CELL 12: HUẤN LUYỆN MÔ HÌNH (VỚI LABEL SMOOTHING + MIXUP)
# ============================
print("=" * 65)
print("BƯỚC 3 (phần 4): HUẤN LUYỆN MÔ HÌNH")
print("=" * 65)


def apply_label_smoothing(y, smoothing=0.1):
    """Label Smoothing: 0  0.05, 1  0.95 (giảm over-confidence)."""
    return y * (1.0 - smoothing) + 0.5 * smoothing


def mixup_data(x, y, alpha=0.2):
    """Mixup: tạo mẫu mới bằng nội suy tuyến tính giữa 2 mẫu ngẫu nhiên."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


best_val_loss = float('inf')
best_val_f1 = 0.0
best_epoch = 0
best_model_state = None
epochs_no_improve = 0

history = {
    'train_loss': [], 'train_acc': [], 'train_f1': [],
    'val_loss': [], 'val_acc': [], 'val_f1': []
}

print(f"\n{'Epoch':>6} | {'Train Loss':>10} {'Train Acc':>10} {'Train F1':>9} | "
      f"{'Val Loss':>9} {'Val Acc':>9} {'Val F1':>8} | {'Status'}")
print("-" * 100)

start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    # ===== 1. TRAINING (Noise tự động bật trong model.train()) =====
    model.train()  # GaussianNoise + Dropout BẬT
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Label Smoothing
        y_smooth = apply_label_smoothing(y_batch, smoothing=LABEL_SMOOTH)

        # GaussianNoise đã được tích hợp trong model, không cần thêm thủ công
        outputs = model(X_batch)
        loss = criterion(outputs, y_smooth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ===== 2. EVALUATE TRAIN (Chuan xac, giong het Val) =====
    model.eval() # BAT BUOC PHAI DUNG EVAL CHO TRAIN DE KHONG LAM HONG BATCHNORM
    train_loss_total = 0
    train_preds, train_labels = [], []
    
    with torch.no_grad():
        # QUAN TRỌNG: Dùng eval_train_loader (Dữ liệu gốc 84/16) để vẽ biểu đồ!
        for X_batch, y_batch in eval_train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            train_loss_total += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_preds.extend(preds.cpu().numpy().flatten())
            train_labels.extend(y_batch.cpu().numpy().flatten())

    train_loss = train_loss_total / len(eval_train_dataset)
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, zero_division=0)

    # ===== VALIDATION (không mixup, không label smoothing) =====
    model.eval()
    val_loss_total = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)  # Loss gốc, không smoothing
            val_loss_total += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            val_preds.extend(preds.cpu().numpy().flatten())
            val_labels.extend(y_batch.cpu().numpy().flatten())

    val_loss = val_loss_total / len(val_dataset)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, zero_division=0)


    # Lưu lịch sử
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_f1'].append(train_f1)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)

    # Cập nhật LR (ReduceLROnPlateau cần truyền val_loss)
    scheduler.step(val_loss)

    # Best model check (theo Val Loss)
    status = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_f1 = val_f1
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        status = f" Best! (Loss={val_loss:.4f}, F1={val_f1:.4f})"

        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'input_dim': INPUT_DIM,
        }, SAVE_PATH)
    else:
        epochs_no_improve += 1
        status = f"  (no improve: {epochs_no_improve}/{PATIENCE})"

    if epoch % 5 == 0 or status.startswith("") or epoch == 1:
        print(f"{epoch:>6} | {train_loss:>10.4f} {train_acc:>10.4f} {train_f1:>9.4f} | "
              f"{val_loss:>9.4f} {val_acc:>9.4f} {val_f1:>8.4f} | {status}")

    if epochs_no_improve >= PATIENCE:
        print(f"\n   Early Stopping tại epoch {epoch} "
              f"(Val Loss không giảm sau {PATIENCE} epochs)")
        break

elapsed_time = time.time() - start_time
print("-" * 100)
print(f"\n Huấn luyện hoàn tất!")
print(f"  Thời gian: {elapsed_time:.1f} giây")
print(f"  Best: Epoch {best_epoch}, Val Loss = {best_val_loss:.4f}, Val F1 = {best_val_f1:.4f}")
print(f"  Đã lưu: '{SAVE_PATH}'")

model.load_state_dict(best_model_state)


# %% ============================
# CELL 13: BIỂU ĐỒ LỊCH SỬ HUẤN LUYỆN
# ============================

def smooth_curve(values, weight=0.97):
    """Exponential Moving Average - sieu muot."""
    smoothed = []
    last = values[0]
    for v in values:
        s = last * weight + (1 - weight) * v
        smoothed.append(s)
        last = s
    return smoothed


fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('LỊCH SỬ HUẤN LUYỆN MÔ HÌNH', fontsize=16, fontweight='bold')
epochs_range = range(1, len(history['train_loss']) + 1)

metrics = [
    ('Loss', 'train_loss', 'val_loss', 'Loss'),
    ('Accuracy', 'train_acc', 'val_acc', 'Accuracy'),
    ('F1-Score', 'train_f1', 'val_f1', 'F1-Score')
]

for idx, (title, train_key, val_key, ylabel) in enumerate(metrics):
    ax = axes[idx]
    ax.plot(epochs_range, smooth_curve(history[train_key]),
            'b-', label='Train', linewidth=2.5)
    ax.plot(epochs_range, smooth_curve(history[val_key]),
            'r-', label='Val', linewidth=2.5)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
               linewidth=1.5, label=f'Best Epoch ({best_epoch})')
    ax.set_title(f'{title} theo Epoch', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Đã lưu 'training_history.png'")


# %% ============================
# CELL 14: TỐI ƯU NGƯỠNG + ĐÁNH GIÁ TẬP TEST
# ============================
# *** BƯỚC 4: Đánh giá mô hình trên tập test ***

print("=" * 65)
print("BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
print("=" * 65)

checkpoint = torch.load(SAVE_PATH, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"\n Đã tải mô hình tốt nhất")
print(f"  Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")

# Tối ưu ngưỡng trên Validation (F1-Score)
print(f"\n--- Tối ưu ngưỡng ---")
val_proba_list, val_labels_list = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        proba = torch.sigmoid(model(X_batch))
        val_proba_list.extend(proba.cpu().numpy().flatten())
        val_labels_list.extend(y_batch.numpy().flatten())

val_proba_arr = np.array(val_proba_list)
val_labels_arr = np.array(val_labels_list)

threshold_results = []
for thresh in np.arange(0.20, 0.80, 0.01):
    preds_tmp = (val_proba_arr >= thresh).astype(int)
    threshold_results.append({
        'threshold': thresh,
        'f1': f1_score(val_labels_arr, preds_tmp, zero_division=0),
        'f2': fbeta_score(val_labels_arr, preds_tmp, beta=2, zero_division=0),
        'precision': precision_score(val_labels_arr, preds_tmp, zero_division=0),
        'recall': recall_score(val_labels_arr, preds_tmp, zero_division=0)
    })

df_thresh = pd.DataFrame(threshold_results)
best_idx = df_thresh['f1'].idxmax()
best_threshold = df_thresh.loc[best_idx, 'threshold']
print(f"  Ngưỡng tối ưu (F1): {best_threshold:.2f}")

# Dự báo trên Test
all_proba, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        proba = torch.sigmoid(model(X_batch))
        all_proba.extend(proba.cpu().numpy().flatten())
        all_labels.extend(y_batch.numpy().flatten())

y_test_actual = np.array(all_labels)
y_proba = np.array(all_proba)
y_pred = (y_proba >= best_threshold).astype(int)

acc = accuracy_score(y_test_actual, y_pred)
prec = precision_score(y_test_actual, y_pred, zero_division=0)
rec = recall_score(y_test_actual, y_pred, zero_division=0)
f1 = f1_score(y_test_actual, y_pred, zero_division=0)
f2 = fbeta_score(y_test_actual, y_pred, beta=2, zero_division=0)
roc_auc = roc_auc_score(y_test_actual, y_proba)
ap = average_precision_score(y_test_actual, y_proba)

print(f"\n{'' * 45}")
print(f"  KẾT QUẢ TRÊN TẬP TEST (Ngưỡng = {best_threshold:.2f})")
print(f"{'' * 45}")
print(f"  {'Accuracy':<22} {acc:>10.4f}")
print(f"  {'Precision':<22} {prec:>10.4f}")
print(f"  {'Recall':<22} {rec:>10.4f}")
print(f"  {'F1-Score':<22} {f1:>10.4f}")
print(f"  {'F2-Score':<22} {f2:>10.4f}")
print(f"  {'ROC-AUC':<22} {roc_auc:>10.4f}")
print(f"  {'Average Precision':<22} {ap:>10.4f}")
print(f"{'' * 45}")

print(f"\n--- Classification Report ---")
target_names = ['Không nghỉ (No)', 'Nghỉ việc (Yes)']
print(classification_report(y_test_actual, y_pred, target_names=target_names, zero_division=0))


# %% ============================
# CELL 15: CONFUSION MATRIX + ROC + PR + THRESHOLD
# ============================
cm = confusion_matrix(y_test_actual, y_pred)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle(f'ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST (Ngưỡng = {best_threshold:.2f})',
             fontsize=16, fontweight='bold')

# [1] Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Không nghỉ (No)', 'Nghỉ việc (Yes)'],
            yticklabels=['Không nghỉ (No)', 'Nghỉ việc (Yes)'],
            annot_kws={'size': 20}, ax=axes[0, 0], linewidths=1)
axes[0, 0].set_title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Dự đoán (Predicted)', fontsize=11)
axes[0, 0].set_ylabel('Thực tế (Actual)', fontsize=11)

tn, fp, fn, tp = cm.ravel()
print(f"\n  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# [2] ROC Curve
fpr, tpr, _ = roc_curve(y_test_actual, y_proba)
axes[0, 1].plot(fpr, tpr, color='#e74c3c', lw=2.5,
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
axes[0, 1].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--',
                label='Random (AUC = 0.5)')
axes[0, 1].fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
axes[0, 1].set_xlim([0, 1]); axes[0, 1].set_ylim([0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate', fontsize=11)
axes[0, 1].set_ylabel('True Positive Rate', fontsize=11)
axes[0, 1].set_title('Đường cong ROC', fontsize=13, fontweight='bold')
axes[0, 1].legend(loc='lower right', fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# [3] Precision-Recall Curve
prec_curve, rec_curve, _ = precision_recall_curve(y_test_actual, y_proba)
axes[1, 0].plot(rec_curve, prec_curve, color='#2980b9', lw=2.5,
                label=f'PR Curve (AP = {ap:.4f})')
baseline = y_test_actual.sum() / len(y_test_actual)
axes[1, 0].axhline(y=baseline, color='gray', lw=1.5, linestyle='--',
                    label=f'Baseline ({baseline:.2f})')
axes[1, 0].fill_between(rec_curve, prec_curve, alpha=0.1, color='#2980b9')
axes[1, 0].set_xlim([0, 1]); axes[1, 0].set_ylim([0, 1.05])
axes[1, 0].set_xlabel('Recall', fontsize=11)
axes[1, 0].set_ylabel('Precision', fontsize=11)
axes[1, 0].set_title('Đường cong Precision-Recall', fontsize=13, fontweight='bold')
axes[1, 0].legend(loc='upper right', fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# [4] Threshold Analysis
axes[1, 1].plot(df_thresh['threshold'], df_thresh['f1'], 'b-', lw=2, label='F1-Score')
axes[1, 1].plot(df_thresh['threshold'], df_thresh['f2'], 'g-', lw=2, label='F2-Score')
axes[1, 1].plot(df_thresh['threshold'], df_thresh['precision'], 'orange',
                lw=1.5, linestyle='--', label='Precision')
axes[1, 1].plot(df_thresh['threshold'], df_thresh['recall'], 'r-',
                lw=1.5, linestyle='--', label='Recall')
axes[1, 1].axvline(x=best_threshold, color='green', lw=2, linestyle=':',
                    label=f'Ngưỡng tối ưu ({best_threshold:.2f})')
axes[1, 1].axvline(x=0.5, color='gray', lw=1.5, linestyle=':', label='Ngưỡng 0.50')
axes[1, 1].set_xlabel('Ngưỡng phân loại', fontsize=11)
axes[1, 1].set_ylabel('Score', fontsize=11)
axes[1, 1].set_title('Phân tích Ngưỡng phân loại', fontsize=13, fontweight='bold')
axes[1, 1].legend(loc='center left', fontsize=9)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim([0.2, 0.8])

plt.tight_layout()
plt.savefig('test_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Đã lưu 'test_evaluation.png'")


# %% ============================
# CELL 16: PERMUTATION FEATURE IMPORTANCE
# ============================
print("=" * 65)
print("PERMUTATION FEATURE IMPORTANCE")
print("=" * 65)

model.eval()
X_test_tensor = torch.FloatTensor(X_test).to(device)

with torch.no_grad():
    base_proba = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()
    base_preds = (base_proba >= best_threshold).astype(int)
    baseline_f1 = f1_score(y_test, base_preds, zero_division=0)

print(f"\n  Baseline F1: {baseline_f1:.4f}")

n_repeats = 10
importance_scores = []
for i, feat_name in enumerate(feature_names):
    drops = []
    for _ in range(n_repeats):
        X_perm = X_test.copy()
        np.random.shuffle(X_perm[:, i])
        with torch.no_grad():
            proba_p = torch.sigmoid(model(torch.FloatTensor(X_perm).to(device)))
            preds_p = (proba_p.cpu().numpy().flatten() >= best_threshold).astype(int)
        drops.append(baseline_f1 - f1_score(y_test, preds_p, zero_division=0))
    importance_scores.append(np.mean(drops))

feature_importance = pd.DataFrame({
    'Feature': feature_names, 'Importance': importance_scores
}).sort_values('Importance', ascending=False)

print(f"\n  Top 15:")
for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
    bar = '' * max(1, int(row['Importance'] * 200)) if row['Importance'] > 0 else ''
    print(f"  {i:>3}. {row['Feature']:<30} {row['Importance']:>+.4f}  {bar}")

# Biểu đồ
new_feat_set = set(new_features)
plt.figure(figsize=(12, 9))
top_n = 20
top_feats = feature_importance.head(top_n)
bar_colors = []
for fn in top_feats['Feature'].values:
    imp = top_feats[top_feats['Feature'] == fn]['Importance'].values[0]
    if fn in new_feat_set:
        bar_colors.append('#9b59b6')
    elif imp > 0.01:
        bar_colors.append('#e74c3c')
    elif imp > 0:
        bar_colors.append('#f39c12')
    else:
        bar_colors.append('#95a5a6')

plt.barh(range(len(top_feats)), top_feats['Importance'].values,
         color=bar_colors, edgecolor='white', height=0.7)
plt.yticks(range(len(top_feats)), top_feats['Feature'].values, fontsize=10)
plt.xlabel('Mức suy giảm F1-Score khi xáo trộn', fontsize=12)
plt.title(f'Top {top_n} Đặc trưng Quan trọng nhất\n(Permutation Feature Importance)',
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.axvline(x=0, color='black', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)

from matplotlib.patches import Patch
plt.legend(handles=[
    Patch(facecolor='#e74c3c', label='Rất quan trọng'),
    Patch(facecolor='#f39c12', label='Quan trọng'),
    Patch(facecolor='#9b59b6', label='Biến mới (Feature Eng.)'),
    Patch(facecolor='#95a5a6', label='Ít quan trọng'),
], loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Đã lưu 'feature_importance.png'")


# %% ============================
# CELL 17: DỰ BÁO MINH HỌA
# ============================
print("=" * 65)
print("VÍ DỤ DỰ BÁO")
print("=" * 65)

n_samples = 10
sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
print(f"\n{'STT':>4} {'Thực tế':>10} {'Dự đoán':>10} {'Xác suất':>10} {'Kết quả':>10}")
print("-" * 48)
for i, idx in enumerate(sample_indices, 1):
    actual = 'Nghỉ' if y_test_actual[idx] == 1 else 'Ở lại'
    predicted = 'Nghỉ' if y_pred[idx] == 1 else 'Ở lại'
    correct = '' if y_test_actual[idx] == y_pred[idx] else ''
    print(f"  {i:>2}. {actual:>8} {predicted:>10} {y_proba[idx]:>9.2%} {correct:>10}")
print(f"\n  Accuracy: {int((y_test_actual == y_pred).sum())}/{len(y_test_actual)} ({acc:.1%})")


# %% ============================
# CELL 18: KẾT LUẬN
# ============================
print("\n" + "=" * 65)
print("TỔNG KẾT DỰ ÁN")
print("=" * 65)
print(f"""

  DỰ ÁN: DỰ ĐOÁN NGUY CƠ NGHỈ VIỆC CỦA NHÂN VIÊN         

  Dataset:  IBM HR Analytics (1470 mẫu, 35 biến gốc)        
  Mo hinh:  MLP PyTorch - {INPUT_DIM} -> 32 -> 16 -> 1{' ' * (18 - len(str(INPUT_DIM)))}
  Nguong:   {best_threshold:.2f} (F1-Score tren Validation)             
                                                              
  KY THUAT CHONG OVERFITTING:                                 
     Kien truc 32-16 (~{total_params:,} tham so){' ' * (22 - len(str(total_params)))}
     Dropout 0.3                                             
     Regularization (Weight Decay 1e-2)                      
     Label Smoothing (0.1)                                   
     Augmentation (Gaussian Noise 0.05)                      
     CosineAnnealingLR                                       
                                                              
  KẾT QUẢ TEST:                                               
    Accuracy:     {acc:.4f}                                    
    Precision:    {prec:.4f}                                    
    Recall:       {rec:.4f}                                    
    F1-Score:     {f1:.4f}                                    
    ROC-AUC:      {roc_auc:.4f}                                    

""")

print("Output files:")
print("   best_model.pth                 - Mô hình")
print("   eda_plots.png                  - EDA")
print("   correlation_with_attrition.png - Tương quan")
print("   correlation_matrix.png         - Ma trận tương quan")
print("   training_history.png           - Lịch sử train")
print("   test_evaluation.png            - Đánh giá test")
print("   feature_importance.png         - Feature Importance")
"""
data_processing.py
Module xử lý dữ liệu HR Employee Attrition
- Phân tích dữ liệu sơ bộ (EDA)
- Tiền xử lý: Mã hóa biến phân loại, chuẩn hóa biến số
- Chia dữ liệu phân tầng: Train 70%, Validation 15%, Test 15%
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Đọc dữ liệu từ file CSV."""
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("1. THÔNG TIN TỔNG QUAN VỀ DỮ LIỆU")
    print("=" * 60)
    print(f"Kích thước dữ liệu: {df.shape[0]} dòng x {df.shape[1]} cột")
    print(f"\nDanh sách các cột ({df.shape[1]} cột):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col} ({df[col].dtype})")
    return df


def explore_data(df):
    """Phân tích dữ liệu sơ bộ (EDA)."""
    print("\n" + "=" * 60)
    print("2. PHÂN TÍCH DỮ LIỆU SƠ BỘ (EDA)")
    print("=" * 60)

    # Kiểm tra giá trị thiếu
    missing = df.isnull().sum()
    print(f"\nGiá trị thiếu (Missing Values):")
    if missing.sum() == 0:
        print("  → Không có giá trị thiếu trong dữ liệu!")
    else:
        print(missing[missing > 0])

    # Thống kê mô tả cho biến số
    print(f"\nThống kê mô tả (biến số):")
    print(df.describe().T.to_string())

    # Phân bố biến mục tiêu
    print(f"\nPhân bố biến mục tiêu 'Attrition':")
    attrition_counts = df['Attrition'].value_counts()
    attrition_pct = df['Attrition'].value_counts(normalize=True) * 100
    for val in attrition_counts.index:
        print(f"  {val}: {attrition_counts[val]} ({attrition_pct[val]:.1f}%)")

    # Phát hiện các cột không có giá trị dự báo (constant columns)
    print(f"\nCác cột có giá trị không đổi (constant):")
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    for col in constant_cols:
        print(f"  → {col} = {df[col].unique()[0]} (luôn cùng giá trị)")

    # Liệt kê các cột phân loại
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCác cột phân loại (Categorical): {len(cat_cols)} cột")
    for col in cat_cols:
        print(f"  → {col}: {df[col].nunique()} giá trị - {df[col].unique()}")

    return constant_cols, cat_cols


def plot_eda(df):
    """Vẽ các biểu đồ EDA cơ bản."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phân tích dữ liệu sơ bộ - HR Employee Attrition', fontsize=16, fontweight='bold')

    # 1. Phân bố Attrition
    colors = ['#2ecc71', '#e74c3c']
    attrition_counts = df['Attrition'].value_counts()
    axes[0, 0].pie(attrition_counts.values, labels=attrition_counts.index,
                   autopct='%1.1f%%', colors=colors, startangle=90,
                   explode=(0, 0.05))
    axes[0, 0].set_title('Tỷ lệ Nghỉ việc (Attrition)', fontweight='bold')

    # 2. Phân bố tuổi theo Attrition
    for label, color in zip(['No', 'Yes'], colors):
        subset = df[df['Attrition'] == label]['Age']
        axes[0, 1].hist(subset, bins=20, alpha=0.7, label=label, color=color, edgecolor='white')
    axes[0, 1].set_title('Phân bố Tuổi theo Attrition', fontweight='bold')
    axes[0, 1].set_xlabel('Tuổi')
    axes[0, 1].set_ylabel('Số lượng')
    axes[0, 1].legend(title='Attrition')

    # 3. Thu nhập hàng tháng theo Attrition
    df.boxplot(column='MonthlyIncome', by='Attrition', ax=axes[1, 0])
    axes[1, 0].set_title('Thu nhập hàng tháng theo Attrition', fontweight='bold')
    axes[1, 0].set_xlabel('Attrition')
    axes[1, 0].set_ylabel('MonthlyIncome')
    plt.sca(axes[1, 0])
    plt.xticks([1, 2], ['No', 'Yes'])

    # 4. OverTime theo Attrition
    overtime_attrition = pd.crosstab(df['OverTime'], df['Attrition'], normalize='index') * 100
    overtime_attrition.plot(kind='bar', ax=axes[1, 1], color=colors, edgecolor='white')
    axes[1, 1].set_title('Tỷ lệ Attrition theo OverTime', fontweight='bold')
    axes[1, 1].set_xlabel('OverTime')
    axes[1, 1].set_ylabel('Tỷ lệ (%)')
    axes[1, 1].legend(title='Attrition')
    axes[1, 1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Đã lưu biểu đồ EDA vào file 'eda_plots.png'")


def preprocess_data(df, constant_cols=None):
    """
    Tiền xử lý dữ liệu:
    - Loại bỏ cột không cần thiết
    - Mã hóa biến mục tiêu (Attrition: Yes=1, No=0)
    - Mã hóa biến phân loại bằng One-Hot Encoding
    - Chuẩn hóa biến số bằng StandardScaler
    """
    print("\n" + "=" * 60)
    print("3. TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    df_processed = df.copy()

    # Loại bỏ cột không cần thiết
    cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    existing_drops = [c for c in cols_to_drop if c in df_processed.columns]
    df_processed.drop(columns=existing_drops, inplace=True)
    print(f"Đã loại bỏ {len(existing_drops)} cột không cần thiết: {existing_drops}")

    # Mã hóa biến mục tiêu: Attrition (Yes=1, No=0)
    df_processed['Attrition'] = df_processed['Attrition'].map({'Yes': 1, 'No': 0})
    print(f"Đã mã hóa biến mục tiêu 'Attrition': Yes→1, No→0")

    # Tách biến mục tiêu
    y = df_processed['Attrition'].values
    X = df_processed.drop(columns=['Attrition'])

    # Xác định cột phân loại và cột số
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"\nSố cột phân loại (Categorical): {len(cat_cols)}")
    print(f"Số cột số (Numerical): {len(num_cols)}")

    # One-Hot Encoding cho biến phân loại
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    print(f"\nSau One-Hot Encoding: {X_encoded.shape[1]} đặc trưng (features)")

    # Chuẩn hóa biến số bằng StandardScaler
    scaler = StandardScaler()
    X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])
    print(f"Đã chuẩn hóa {len(num_cols)} cột số bằng StandardScaler")

    feature_names = X_encoded.columns.tolist()
    X_final = X_encoded.values.astype(np.float32)
    y_final = y.astype(np.float32)

    print(f"\nKích thước dữ liệu sau xử lý: X={X_final.shape}, y={y_final.shape}")

    return X_final, y_final, feature_names, scaler


def split_data(X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Chia dữ liệu thành 3 phần theo phương pháp phân tầng (Stratified Split):
    - Train: 70%
    - Validation: 15%
    - Test: 15%
    """
    print("\n" + "=" * 60)
    print("4. CHIA DỮ LIỆU PHÂN TẦNG (STRATIFIED SPLIT)")
    print("=" * 60)

    # Bước 1: Chia thành Train (70%) và Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - train_ratio),
        random_state=random_state,
        stratify=y
    )

    # Bước 2: Chia Temp thành Validation (50% of 30% = 15%) và Test (50% of 30% = 15%)
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_relative_ratio),
        random_state=random_state,
        stratify=y_temp
    )

    # In thông tin phân chia
    print(f"\n{'Tập dữ liệu':<15} {'Số mẫu':>10} {'Tỷ lệ':>10} {'Attrition=1':>15} {'Tỷ lệ AT=1':>15}")
    print("-" * 65)
    for name, X_set, y_set in [('Train', X_train, y_train),
                                ('Validation', X_val, y_val),
                                ('Test', X_test, y_test)]:
        n = len(y_set)
        pct = n / len(y) * 100
        n_pos = int(y_set.sum())
        pct_pos = n_pos / n * 100
        print(f"{name:<15} {n:>10} {pct:>9.1f}% {n_pos:>15} {pct_pos:>14.1f}%")
    print("-" * 65)
    total_pos = int(y.sum())
    print(f"{'Tổng':<15} {len(y):>10} {'100.0%':>10} {total_pos:>15} {total_pos/len(y)*100:>14.1f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    # Chạy thử module
    filepath = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
    df = load_data(filepath)
    constant_cols, cat_cols = explore_data(df)
    plot_eda(df)
    X, y, feature_names, scaler = preprocess_data(df, constant_cols)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

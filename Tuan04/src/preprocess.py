from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CẤU HÌNH
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "train.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# ĐỌC DỮ LIỆU
# ============================================================

def load_data(data_path=DATA_PATH):
    """Đọc Ames Housing dataset."""
    return pd.read_csv(data_path)


# ============================================================
# XỬ LÝ OUTLIER
# ============================================================

def remove_outliers(df):
    """
    Loại các mẫu có GrLivArea > 4000
    và SalePrice < 300000.
    """
    df = df.copy()

    outlier_condition = (
        (df["GrLivArea"] > 4000)
        & (df["SalePrice"] < 300000)
    )

    df = df.loc[~outlier_condition].copy()
    df.reset_index(drop=True, inplace=True)

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):
    """Tạo các feature mới."""
    df = df.copy()

    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )

    df["HouseAge"] = (
        df["YrSold"] - df["YearBuilt"]
    )

    df["RemodAge"] = (
        df["YrSold"] - df["YearRemodAdd"]
    )

    df["TotalBath"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    return df


# ============================================================
# XỬ LÝ MISSING CÓ Ý NGHĨA "KHÔNG CÓ"
# ============================================================

def fill_none_values(df):
    """Thay NA bằng 'None' ở các biến mà NA có nghĩa là không có."""
    df = df.copy()

    none_cols = [
        "Alley",
        "MasVnrType",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature",
    ]

    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    return df


# ============================================================
# ORDINAL ENCODING
# ============================================================

def ordinal_encode(df):
    """Mã hóa các biến chất lượng theo thứ tự."""
    df = df.copy()

    quality_map = {
        "None": 0,
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5,
    }

    quality_cols = [
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
        "PoolQC",
    ]

    for col in quality_cols:
        if col in df.columns:
            df[col] = df[col].map(quality_map)

    return df


# ============================================================
# TẠO PREPROCESSOR
# ============================================================

def build_preprocessor(X_train):
    """Tạo pipeline xử lý numerical và categorical features."""

    numeric_features = (
        X_train
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    categorical_features = (
        X_train
        .select_dtypes(exclude=np.number)
        .columns
        .tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    try:
        onehot = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        onehot = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                onehot,
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def preprocess_data():
    """Thực hiện toàn bộ quy trình preprocessing."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Đọc dữ liệu
    df = load_data()

    print("Dataset ban đầu:", df.shape)

    # 2. Xóa duplicate nếu có
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    print("Duplicate:", duplicate_count)

    # 3. Tạo target log
    df["log_SalePrice"] = np.log1p(
        df["SalePrice"]
    )

    # 4. Xử lý outlier
    before_outlier = len(df)

    df = remove_outliers(df)

    print(
        "Outlier đã loại:",
        before_outlier - len(df)
    )

    # 5. Feature engineering
    df = create_features(df)

    # 6. Missing có nghĩa là không có
    df = fill_none_values(df)

    # 7. Ordinal encoding
    df = ordinal_encode(df)

    # 8. Tách X và y
    y = df["log_SalePrice"].copy()

    X = df.drop(
        columns=[
            "Id",
            "SalePrice",
            "log_SalePrice",
        ],
        errors="ignore",
    ).copy()

    # 9. Train/Test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # 10. Tạo preprocessor
    preprocessor = build_preprocessor(
        X_train
    )

    # Chỉ fit trên train
    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    # Test chỉ transform
    X_test_processed = (
        preprocessor.transform(X_test)
    )

    # 11. Lấy tên feature
    feature_names = (
        preprocessor.get_feature_names_out()
    )

    feature_names = [
        name
        .replace("num__", "")
        .replace("cat__", "")
        for name in feature_names
    ]

    # 12. Chuyển thành DataFrame
    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
    )

    # Reset index
    X_train_processed.reset_index(
        drop=True,
        inplace=True,
    )

    X_test_processed.reset_index(
        drop=True,
        inplace=True,
    )

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # 13. Kiểm tra missing
    if X_train_processed.isna().sum().sum() != 0:
        raise ValueError(
            "X_train vẫn còn missing values."
        )

    if X_test_processed.isna().sum().sum() != 0:
        raise ValueError(
            "X_test vẫn còn missing values."
        )

    # 14. Lưu CSV
    X_train_processed.to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False,
    )

    X_test_processed.to_csv(
        PROCESSED_DIR / "X_test.csv",
        index=False,
    )

    y_train.to_frame(
        name="log_SalePrice"
    ).to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False,
    )

    y_test.to_frame(
        name="log_SalePrice"
    ).to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False,
    )

    pd.DataFrame({
        "feature": feature_names
    }).to_csv(
        PROCESSED_DIR / "feature_names.csv",
        index=False,
    )

    # 15. Lưu preprocessor
    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor.pkl",
    )

    print("\nPreprocessing hoàn tất.")
    print(
        "X_train processed:",
        X_train_processed.shape
    )
    print(
        "X_test processed :",
        X_test_processed.shape
    )
    print(
        "Số feature sau preprocessing:",
        len(feature_names)
    )
    print(
        "Missing X_train:",
        X_train_processed.isna().sum().sum()
    )
    print(
        "Missing X_test:",
        X_test_processed.isna().sum().sum()
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    )


# ============================================================
# CHẠY FILE
# ============================================================

if __name__ == "__main__":
    preprocess_data()
"""Nạp dữ liệu và chia train/test dùng chung cho cả nhóm (cùng một phép chia -> so sánh công bằng).

Cách dùng:
    from src.data import load_data, get_train_test
    X_train, X_test, y_train, y_test = get_train_test(feature_set="C")

Tập test CHỈ được dùng một lần ở bước đánh giá cuối (nhóm trưởng chạy).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bỏ khoảng trắng/tab thừa, sửa lỗi chính tả tên cột."""
    df = df.copy()
    df.columns = [str(c).replace("\t", " ").strip() for c in df.columns]
    df = df.rename(columns=config.COLUMN_RENAMES)
    return df


def load_data(path=None) -> pd.DataFrame:
    """Đọc data.csv (dấu phân cách ';' hoặc ',', tự phát hiện) và chuẩn hóa tên cột."""
    path = path or config.DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy {path}. Chạy: python scripts/download_data.py"
        )
    # Không dùng sep=None: sniffer đoán sai vì tên cột chứa khoảng trắng, "/" và dấu nháy.
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline()
    sep = ";" if header.count(";") >= header.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    df = clean_columns(df)

    missing = [c for c in config.FEATURE_SETS["C"] + [config.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột sau khi chuẩn hóa tên: {missing}")
    df[config.TARGET] = df[config.TARGET].astype(str).str.strip()
    unknown = set(df[config.TARGET].unique()) - set(config.CLASSES)
    if unknown:
        raise ValueError(f"Nhãn lạ trong Target: {unknown}")
    return df


def get_train_test(feature_set: str = "C", df: pd.DataFrame = None):
    """Chia hold-out 80/20 phân tầng, random_state cố định.

    Trả về X_train, X_test, y_train, y_test (DataFrame/Series, giữ nguyên tên cột).
    feature_set: 'A' | 'B' | 'C' (xem config.FEATURE_SETS).
    """
    if feature_set not in config.FEATURE_SETS:
        raise KeyError(f"feature_set phải thuộc {list(config.FEATURE_SETS)}")
    df = load_data() if df is None else df
    X = df[config.FEATURE_SETS[feature_set]]
    y = df[config.TARGET]
    # Cùng random_state + stratify theo y => chỉ số train/test GIỐNG NHAU giữa A/B/C
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

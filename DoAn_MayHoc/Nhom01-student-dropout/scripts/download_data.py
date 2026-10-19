"""Tải dataset UCI ID 697 (Predict Students' Dropout and Academic Success) vào data/raw/data.csv.

Chạy:  python scripts/download_data.py
Cần:   pip install ucimlrepo
Nếu mạng bị chặn: tải thủ công từ
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
rồi đặt file CSV (dấu ';') vào data/raw/data.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402


def main():
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=697)
    X = ds.data.features
    y = ds.data.targets
    df = pd.concat([X, y], axis=1)
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA_FILE, sep=";", index=False)
    print(f"Đã lưu {config.DATA_FILE} - shape = {df.shape} (mong đợi: 4424 x 37 = 36 đặc trưng + Target)")


if __name__ == "__main__":
    main()

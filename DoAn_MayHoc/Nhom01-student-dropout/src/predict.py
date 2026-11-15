"""Hàm dự đoán dùng chung cho demo (SV3).

    from src.predict import predict_one
    out = predict_one(record, model="lr")          # record: dict {tên cột: giá trị} hoặc DataFrame 1 dòng
    out["label"], out["probabilities"], out["warnings"]

Hợp đồng đầu vào/đầu ra
- Đầu vào: đủ các cột của tập đặc trưng C (config.FEATURE_SETS["C"], 36 cột); thừa cột -> bỏ qua + cảnh báo.
- Lỗi (InputError): thiếu cột, giá trị không phải số / NaN, biến nhị phân khác 0/1, biến phân loại không phải số nguyên.
- Cảnh báo (không dừng): mã phân loại chưa từng thấy khi train (pipeline xử lý như nhóm hiếm/"Other"),
  giá trị số nằm ngoài khoảng min-max của tập train.
- Đầu ra: nhãn, xác suất đủ 3 lớp theo thứ tự pipeline.classes_ (= config.CLASSES), danh sách cảnh báo.
- Ngưỡng Dropout (tùy chọn): dropout_threshold=t thì nhãn = "Dropout" khi P(Dropout) >= t, ngược lại chọn lớp
  có xác suất lớn nhất trong hai lớp còn lại. Mặc định None = argmax. Ngưỡng phải chọn trên validation/CV (không dùng test).
Đây là demo minh họa, không phải công cụ ra quyết định thật.
"""
import json

import joblib
import numpy as np
import pandas as pd

from src import config

MODEL_FILES = {"lr": "lr_best.joblib", "dt": "dt_best.joblib"}
META_FILE = config.MODELS_DIR / "sv3_metadata.json"
_cache = {}


class InputError(ValueError):
    """Đầu vào không hợp lệ (thiếu cột, sai kiểu, sai miền)."""


def load_metadata() -> dict:
    if "meta" not in _cache:
        if not META_FILE.exists():
            raise FileNotFoundError(f"Không thấy {META_FILE}. Chạy notebooks/03_lr_dt.ipynb trước.")
        with open(META_FILE, encoding="utf-8") as f:
            _cache["meta"] = json.load(f)
    return _cache["meta"]


def load_model(name: str = "lr"):
    """Nạp pipeline đã fit; kiểm tra thứ tự lớp khớp config.CLASSES."""
    if name not in MODEL_FILES:
        raise KeyError(f"model phải thuộc {list(MODEL_FILES)}")
    if name not in _cache:
        pipe = joblib.load(config.MODELS_DIR / MODEL_FILES[name])
        classes = list(pipe.classes_)
        if sorted(classes) != sorted(config.CLASSES):
            raise ValueError(f"classes_ của mô hình ({classes}) không khớp config.CLASSES")
        _cache[name] = pipe
    return _cache[name]


def validate_input(record, feature_set: str = "C"):
    """Kiểm tra và chuẩn hóa một mẫu. Trả về (DataFrame 1 dòng đúng thứ tự cột, danh sách cảnh báo)."""
    meta = load_metadata()
    cols = config.FEATURE_SETS[feature_set]
    if isinstance(record, pd.DataFrame):
        if len(record) != 1:
            raise InputError(f"Cần đúng 1 dòng, nhận {len(record)}")
        record = record.iloc[0].to_dict()
    elif not isinstance(record, dict):
        raise InputError("record phải là dict hoặc DataFrame 1 dòng")

    warns = []
    missing = [c for c in cols if c not in record]
    if missing:
        raise InputError(f"Thiếu {len(missing)} cột: {missing}")
    extra = [c for c in record if c not in cols]
    if extra:
        warns.append(f"Bỏ qua cột thừa: {extra}")

    values = {}
    for c in cols:
        try:
            v = float(record[c])
        except (TypeError, ValueError):
            raise InputError(f"Cột '{c}' phải là số, nhận {record[c]!r}")
        if np.isnan(v) or np.isinf(v):
            raise InputError(f"Cột '{c}' bị thiếu/NaN/vô cực")
        if c in config.BINARY_COLS and v not in (0.0, 1.0):
            raise InputError(f"Cột nhị phân '{c}' phải là 0 hoặc 1, nhận {v}")
        if c in config.CATEGORICAL_COLS:
            if v != int(v):
                raise InputError(f"Cột phân loại '{c}' phải là mã số nguyên, nhận {v}")
            v = int(v)
            if v not in meta["known_categories"][c]:
                warns.append(f"'{c}' = {v} chưa từng thấy khi train, được xử lý như nhóm hiếm (Other)")
        elif c in meta["numeric_ranges"]:
            lo, hi = meta["numeric_ranges"][c]
            if not (lo <= v <= hi):
                warns.append(f"'{c}' = {v} nằm ngoài khoảng train [{lo:g}, {hi:g}]")
        values[c] = v
    return pd.DataFrame([values], columns=cols), warns


def predict_one(record, model: str = "lr", dropout_threshold: float = None, feature_set: str = "C") -> dict:
    """Dự đoán một mẫu. Trả về {'label', 'probabilities', 'model', 'threshold', 'warnings'}."""
    X, warns = validate_input(record, feature_set)
    pipe = load_model(model)
    classes = list(pipe.classes_)
    proba = pipe.predict_proba(X)[0]
    probs = {c: float(p) for c, p in zip(classes, proba)}
    if dropout_threshold is None:
        label = classes[int(np.argmax(proba))]
    elif probs[config.POSITIVE_CLASS] >= dropout_threshold:
        label = config.POSITIVE_CLASS
    else:
        rest = {c: p for c, p in probs.items() if c != config.POSITIVE_CLASS}
        label = max(rest, key=rest.get)
    return {"label": label, "probabilities": {c: probs[c] for c in config.CLASSES},
            "model": model, "threshold": dropout_threshold, "warnings": warns}


if __name__ == "__main__":
    from src.data import get_train_test

    X_train, _, y_train, _ = get_train_test("C")   # lấy mẫu minh họa từ TRAIN, không đụng tập test
    row = X_train.iloc[0].to_dict()
    for m in MODEL_FILES:
        print(m, "| nhãn thật:", y_train.iloc[0], "|", predict_one(row, model=m))

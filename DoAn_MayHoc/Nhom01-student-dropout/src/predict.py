"""Hàm dự đoán dùng chung cho demo.

    from src.predict import predict_one
    out = predict_one(record, model="gb", feature_set="C")
    out["label"], out["probabilities"], out["warnings"]

Hợp đồng đầu vào/đầu ra
- Đầu vào: đủ cột của tập A/B/C được chọn; thừa cột -> bỏ qua + cảnh báo.
- Lỗi (InputError): thiếu cột, giá trị không phải số / NaN, biến nhị phân khác 0/1, biến phân loại không phải số nguyên.
- Cảnh báo (không dừng): mã phân loại chưa từng thấy khi train (OneHotEncoder bỏ qua cột chưa biết),
  giá trị số nằm ngoài khoảng min-max của tập train.
- Đầu ra: nhãn, xác suất đủ 3 lớp theo thứ tự pipeline.classes_ (= config.CLASSES), danh sách cảnh báo.
- Mặc định dùng `pipeline.predict()` như quy tắc bàn giao `native_predict`.
- Ngưỡng Dropout (tùy chọn): dropout_threshold=t thì nhãn = "Dropout" khi P(Dropout) >= t, ngược lại chọn lớp
  có xác suất lớn nhất trong hai lớp còn lại. Ngưỡng phải chọn trên validation/CV (không dùng test).
Đây là demo minh họa, không phải công cụ ra quyết định thật.
"""
import json

import joblib
import numpy as np
import pandas as pd

from src import config

MODEL_FILES = {
    "lr": "lr_best.joblib",
    "dt": "dt_best.joblib",
    "rf": "rf_best.joblib",
    "gb": "gb_best.joblib",
    "svm": "svm_best_proba.joblib",
}
META_FILE = config.MODELS_DIR / "model_metadata.json"
_cache = {}


class InputError(ValueError):
    """Đầu vào không hợp lệ (thiếu cột, sai kiểu, sai miền)."""


def load_metadata() -> dict:
    if "meta" not in _cache:
        if not META_FILE.exists():
            raise FileNotFoundError(f"Không thấy metadata đầu vào: {META_FILE}")
        with open(META_FILE, encoding="utf-8") as f:
            _cache["meta"] = json.load(f)
    return _cache["meta"]


def model_path(name: str = "lr", feature_set: str = "C"):
    if name not in MODEL_FILES:
        raise KeyError(f"model phải thuộc {list(MODEL_FILES)}")
    if feature_set not in config.FEATURE_SETS:
        raise KeyError(f"feature_set phải thuộc {list(config.FEATURE_SETS)}")
    if feature_set == "C":
        return config.MODELS_DIR / MODEL_FILES[name]
    return config.MODELS_DIR / "demo" / feature_set / f"{name}_best.joblib"


def load_model(name: str = "lr", feature_set: str = "C"):
    """Nạp pipeline đã fit; kiểm tra thứ tự lớp khớp config.CLASSES."""
    key = (name, feature_set)
    if key not in _cache:
        path = model_path(name, feature_set)
        if not path.is_file():
            raise FileNotFoundError(f"Không thấy mô hình {name.upper()}-{feature_set}: {path}. Chạy scripts/build_demo_models.py.")
        pipe = joblib.load(path)
        classes = list(pipe.classes_)
        if sorted(classes) != sorted(config.CLASSES):
            raise ValueError(f"classes_ của mô hình ({classes}) không khớp config.CLASSES")
        _cache[key] = pipe
    return _cache[key]


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
                warns.append(f"'{c}' = {v} chưa từng thấy khi train; OneHotEncoder bỏ qua mã chưa biết")
        elif c in ("Admission grade", "Previous qualification (grade)") and not 0 <= v <= 200:
            raise InputError(f"Cột '{c}' phải nằm trong miền 0–200, nhận {v}")
        elif c in meta["numeric_ranges"]:
            lo, hi = meta["numeric_ranges"][c]
            if not (lo <= v <= hi):
                warns.append(f"'{c}' = {v} nằm ngoài khoảng train [{lo:g}, {hi:g}]")
        values[c] = v
    return pd.DataFrame([values], columns=cols), warns


def predict_one(record, model: str = "lr", dropout_threshold: float = None, feature_set: str = "C") -> dict:
    """Dự đoán một mẫu. Trả về {'label', 'probabilities', 'model', 'threshold', 'warnings'}."""
    X, warns = validate_input(record, feature_set)
    pipe = load_model(model, feature_set)
    classes = list(pipe.classes_)
    proba = pipe.predict_proba(X)[0]
    probs = {c: float(p) for c, p in zip(classes, proba)}
    if dropout_threshold is None:
        label = str(pipe.predict(X)[0])
    elif probs[config.POSITIVE_CLASS] >= dropout_threshold:
        label = config.POSITIVE_CLASS
    else:
        rest = {c: p for c, p in probs.items() if c != config.POSITIVE_CLASS}
        label = max(rest, key=rest.get)
    return {"label": str(label), "probabilities": {c: probs[c] for c in config.CLASSES},
            "model": model, "feature_set": feature_set,
            "decision_rule": "native_predict" if dropout_threshold is None else "dropout_threshold",
            "threshold": dropout_threshold, "warnings": warns}


if __name__ == "__main__":
    from src.data import get_train_test

    X_train, _, y_train, _ = get_train_test("C")   # lấy mẫu minh họa từ TRAIN, không đụng tập test
    row = X_train.iloc[0].to_dict()
    for m in MODEL_FILES:
        print(m, "| nhãn thật:", y_train.iloc[0], "|", predict_one(row, model=m))

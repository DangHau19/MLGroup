"""Quy tắc ngưỡng Dropout dùng chung (SV4-THR).

Quy tắc quyết định đa lớp (cố định, mô tả đầy đủ):
    1. p_D = P(Dropout). Nếu p_D >= t  ->  nhãn = "Dropout".
    2. Ngược lại -> nhãn = lớp có xác suất lớn nhất trong {Enrolled, Graduate}
       (KHÔNG xét Dropout; hòa thì lấy lớp đứng trước trong `classes`).
    Khi t <= 0 mọi mẫu là Dropout; khi t > 1 không mẫu nào là Dropout.
    Quy tắc gốc (argmax thuần) KHÔNG phải trường hợp đặc biệt của t, vì argmax có thể chọn
    Dropout khi p_D < 0,5; muốn so sánh dùng `predict_argmax`.

Mọi hàm chọn ngưỡng chỉ nhận dự đoán OOF/validation thuộc TRAIN. Không gọi evaluate_on_test.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import cross_val_predict

from src import config
from src.evaluate import get_cv

POSITIVE = config.POSITIVE_CLASS
GRID = np.round(np.arange(0.10, 0.801, 0.01), 2)


def oof_proba(pipeline, X, y):
    """Xác suất OOF trên train: mỗi dòng được dự đoán bởi mô hình không thấy dòng đó.
    Cột theo đúng thứ tự `config.CLASSES` (kiểm tra bằng classes_ của mô hình)."""
    proba = cross_val_predict(pipeline, X, y, cv=get_cv(), method="predict_proba", n_jobs=-1)
    classes = sorted(set(y))
    if list(classes) != list(config.CLASSES):
        raise ValueError(f"Thứ tự lớp {classes} khác config.CLASSES {config.CLASSES}")
    return pd.DataFrame(proba, index=X.index, columns=classes)


def predict_argmax(proba, classes=None):
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes if classes is not None else config.CLASSES)
    return classes[proba.argmax(axis=1)]


def apply_threshold(proba, classes=None, t=0.5, positive=POSITIVE):
    """Áp quy tắc ngưỡng cho ma trận xác suất (n, K) hoặc một vector (K,). `classes` phải là
    `pipeline.classes_` (hoặc `named_steps['clf'].classes_`), cùng thứ tự các cột."""
    proba = np.atleast_2d(np.asarray(proba, dtype=float))
    classes = np.asarray(classes if classes is not None else config.CLASSES)
    if proba.shape[1] != len(classes):
        raise ValueError(f"proba có {proba.shape[1]} cột, classes có {len(classes)}")
    if positive not in classes:
        raise ValueError(f"Không thấy lớp {positive!r} trong classes")
    k = int(np.where(classes == positive)[0][0])
    rest = proba.copy()
    rest[:, k] = -np.inf
    out = classes[rest.argmax(axis=1)].astype(object)
    out[proba[:, k] >= t] = positive
    return out


def _metrics(y_true, y_pred):
    labels = config.CLASSES
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    m = {"f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
         "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
         "accuracy": accuracy_score(y_true, y_pred)}
    for i, c in enumerate(labels):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"] = p[i], r[i], f[i]
    m["n_pred_dropout"] = int((np.asarray(y_pred) == POSITIVE).sum())
    return m


def threshold_table(y_true, proba, grid=GRID):
    """Bảng chỉ số theo từng ngưỡng; dòng đầu (t = 'argmax') là quy tắc gốc."""
    classes = list(proba.columns) if hasattr(proba, "columns") else list(config.CLASSES)
    P = proba.to_numpy() if hasattr(proba, "to_numpy") else np.asarray(proba)
    rows = [dict(threshold="argmax", **_metrics(y_true, predict_argmax(P, classes)))]
    for t in grid:
        rows.append(dict(threshold=float(t), **_metrics(y_true, apply_threshold(P, classes, t))))
    return pd.DataFrame(rows)


def choose_threshold(table):
    """TIÊU CHÍ (đặt trước khi nhìn test): trong các ngưỡng có Recall Dropout >= Recall Dropout của
    quy tắc argmax, chọn ngưỡng có Macro-F1 cao nhất; hòa thì chọn ngưỡng lớn hơn (ít báo động giả
    hơn). Nếu không ngưỡng nào đạt ràng buộc thì giữ argmax (trả về None)."""
    base = table[table.threshold == "argmax"].iloc[0]
    g = table[table.threshold != "argmax"].copy()
    g["threshold"] = g["threshold"].astype(float)
    ok = g[g.recall_Dropout >= base.recall_Dropout - 1e-12]
    if ok.empty:
        return None
    best = ok.f1_macro.max()
    return float(ok[np.isclose(ok.f1_macro, best)].threshold.max())


def nested_check(y_true, proba, folds):
    """Chọn ngưỡng trên OOF của 4 fold, chấm trên fold còn lại (ước lượng chọn ngưỡng có ăn may không).
    `folds`: danh sách mảng chỉ số vị trí của từng fold validation."""
    P = proba.to_numpy()
    classes = list(proba.columns)
    y = np.asarray(y_true)
    out = []
    for k, te in enumerate(folds):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != k])
        t = choose_threshold(threshold_table(y[tr], proba.iloc[tr]))
        pred_t = predict_argmax(P[te], classes) if t is None else apply_threshold(P[te], classes, t)
        pred_a = predict_argmax(P[te], classes)
        out.append(dict(fold=k, threshold=t,
                        f1_macro_thr=f1_score(y[te], pred_t, average="macro"),
                        f1_macro_argmax=f1_score(y[te], pred_a, average="macro"),
                        recall_dropout_thr=_metrics(y[te], pred_t)["recall_Dropout"],
                        recall_dropout_argmax=_metrics(y[te], pred_a)["recall_Dropout"]))
    return pd.DataFrame(out)

"""Độ đo và hàm đánh giá dùng chung. Mọi mô hình phải đánh giá qua file này.

- cross_validate_model: Stratified 5-fold CV trên tập TRAIN (dùng cho chọn mô hình/tuning).
- evaluate_on_test: đánh giá cuối trên tập TEST (chỉ nhóm trưởng chạy, một lần).
- log_result: ghi kết quả vào reports/cv_results.csv để gom bảng so sánh.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import config

# Scorer dùng cho CV / GridSearchCV (scoring chính của đồ án là f1_macro)
SCORING = {
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
    "recall_dropout": make_scorer(recall_score, labels=["Dropout"], average="macro", zero_division=0),
    "f1_enrolled": make_scorer(f1_score, labels=["Enrolled"], average="macro", zero_division=0),
}


def get_cv():
    """Stratified 5-fold, cố định random_state."""
    return StratifiedKFold(
        n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE
    )


def cross_validate_model(model, X_train, y_train, name=None, feature_set=None, log=True, n_jobs=-1):
    """CV 5-fold trên tập train. `model` nên là Pipeline (tiền xử lý nằm bên trong).

    Trả về dict: mean/std của từng độ đo (validation) và train_f1_macro để xem overfit.
    """
    res = cross_validate(
        model, X_train, y_train,
        cv=get_cv(), scoring=SCORING,
        return_train_score=True, n_jobs=n_jobs,
    )
    out = {"model": name or type(model).__name__, "feature_set": feature_set}
    for k in SCORING:
        out[f"{k}_mean"] = float(np.mean(res[f"test_{k}"]))
        out[f"{k}_std"] = float(np.std(res[f"test_{k}"]))
    out["train_f1_macro_mean"] = float(np.mean(res["train_f1_macro"]))
    out["fit_time_mean"] = float(np.mean(res["fit_time"]))
    if log:
        log_result(out, config.CV_RESULTS_FILE)
    return out


def compute_metrics(y_true, y_pred) -> dict:
    """Bộ độ đo trên một tập dự đoán (dùng cho test/validation)."""
    labels = config.CLASSES
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    m = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    for i, c in enumerate(labels):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"] = p[i], r[i], f[i]
    return m


def report(y_true, y_pred) -> str:
    """Classification report dạng text, thứ tự lớp cố định."""
    return classification_report(
        y_true, y_pred, labels=config.CLASSES, digits=3, zero_division=0
    )


def confusion(y_true, y_pred) -> pd.DataFrame:
    """Confusion matrix (hàng = thực tế, cột = dự đoán) dạng DataFrame."""
    cm = confusion_matrix(y_true, y_pred, labels=config.CLASSES)
    return pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in config.CLASSES],
        columns=[f"pred_{c}" for c in config.CLASSES],
    )


def plot_confusion(y_true, y_pred, title="Confusion Matrix", normalize=True, save_as=None):
    """Vẽ confusion matrix; save_as là tên file trong reports/figures/."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=config.CLASSES,
        normalize="true" if normalize else None,
        cmap="Blues", ax=ax, values_format=".2f" if normalize else "d",
    )
    ax.set_title(title)
    fig.tight_layout()
    if save_as:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / save_as, dpi=200)
    return fig


def log_result(row: dict, path) -> None:
    """Thêm một dòng kết quả vào file CSV (tạo mới nếu chưa có)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def evaluate_on_test(model, X_test, y_test, name, feature_set=None):
    """ĐÁNH GIÁ CUỐI trên tập test. Chỉ nhóm trưởng chạy, mỗi mô hình một lần.

    `model` phải đã fit trên toàn bộ tập train với siêu tham số tốt nhất.
    """
    warnings.warn(
        "evaluate_on_test: tập test chỉ được dùng MỘT LẦN ở bước cuối. "
        "Đừng dùng kết quả này để chỉnh lại mô hình.",
        stacklevel=2,
    )
    y_pred = model.predict(X_test)
    m = compute_metrics(y_test, y_pred)
    m.update({"model": name, "feature_set": feature_set})
    log_result(m, config.TEST_RESULTS_FILE)
    return m, y_pred

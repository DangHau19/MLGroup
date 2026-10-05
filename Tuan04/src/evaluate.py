import time
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

RESULT_COLUMNS = [
    "technique", "model", "n_features",
    "cv_log_rmse", "cv_log_r2",
    "test_log_rmse",
    "test_original_rmse", "test_original_mae", "test_original_r2",
    "cv_seconds", "train_seconds",
]

_MAX_PLAUSIBLE_LOG_PRICE = 25

def _take(data, index):
    return data.iloc[index] if hasattr(data, "iloc") else data[index]

def _as_vector(y):
    y = np.asarray(y, dtype=float).ravel()
    if np.isnan(y).any():
        raise ValueError("Target chứa NaN.")
    if y.max() > _MAX_PLAUSIBLE_LOG_PRICE:
        raise ValueError(
            "Target phải là log1p(SalePrice) (giá trị lớn nhất hiện là "
            f"{y.max():.1f}). Có vẻ bạn đang truyền giá gốc."
        )
    return y

def _to_price(log_values):
    return np.expm1(np.clip(log_values, 0, None))

def regression_scores(y_log, pred_log):
    y_price, pred_price = _to_price(y_log), _to_price(pred_log)
    return {
        "log_rmse": float(np.sqrt(mean_squared_error(y_log, pred_log))),
        "log_mae": float(mean_absolute_error(y_log, pred_log)),
        "log_r2": float(r2_score(y_log, pred_log)),
        "original_rmse": float(np.sqrt(mean_squared_error(y_price, pred_price))),
        "original_mae": float(mean_absolute_error(y_price, pred_price)),
        "original_r2": float(r2_score(y_price, pred_price)),
    }

def count_features(fitted_model, n_input):
    steps = getattr(fitted_model, "named_steps", {})
    for step in steps.values():
        if hasattr(step, "get_support"):
            return int(np.sum(step.get_support()))
    return int(n_input)

def evaluate_model(technique, name, model, X_train, y_train, X_test, y_test,
                   n_splits=5, seed=42, *, detailed=False, return_model=False):
   
    y_tr, y_te = _as_vector(y_train), _as_vector(y_test)
    if len(y_tr) != len(X_train) or len(y_te) != len(X_test):
        raise ValueError("Số dòng của X và y không khớp.")
    # --- Cross-validation trên tập train ---------------------------------
    folds = []
    started = time.perf_counter()
    for fit_idx, val_idx in KFold(n_splits, shuffle=True,
                                  random_state=seed).split(X_train):
        fold_model = clone(model).fit(_take(X_train, fit_idx), y_tr[fit_idx])
        folds.append(regression_scores(
            y_tr[val_idx], fold_model.predict(_take(X_train, val_idx))))
    cv_seconds = time.perf_counter() - started
    cv = {key: np.array([f[key] for f in folds]) for key in folds[0]}
    # --- Fit lần cuối trên toàn bộ train, báo cáo trên tập test ----------
    started = time.perf_counter()
    final_model = clone(model).fit(X_train, y_tr)
    train_seconds = time.perf_counter() - started
    test = regression_scores(y_te, final_model.predict(X_test))
    row = {
        "technique": technique,
        "model": name,
        "n_features": count_features(final_model, np.shape(X_train)[1]),
        "cv_log_rmse": cv["log_rmse"].mean(),
        "cv_log_r2": cv["log_r2"].mean(),
        "test_log_rmse": test["log_rmse"],
        "test_original_rmse": test["original_rmse"],
        "test_original_mae": test["original_mae"],
        "test_original_r2": test["original_r2"],
        "cv_seconds": cv_seconds,
        "train_seconds": train_seconds,
    }
    if detailed:
        row.update({
            "cv_log_rmse_std": cv["log_rmse"].std(ddof=1),
            "cv_log_mae": cv["log_mae"].mean(),
            "cv_original_rmse": cv["original_rmse"].mean(),
            "cv_original_mae": cv["original_mae"].mean(),
            "cv_original_r2": cv["original_r2"].mean(),
            "test_log_mae": test["log_mae"],
            "test_log_r2": test["log_r2"],
        })
    return (row, final_model) if return_model else row

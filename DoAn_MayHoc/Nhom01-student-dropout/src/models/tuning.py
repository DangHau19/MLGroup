#Tuning Gradient Boosting và SVM #

import sys
import time
import warnings

import joblib
import pandas as pd
from scipy.stats import loguniform
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

from src import config
from src.data import get_train_test
from src.evaluate import cross_validate_model, get_cv
from src.preprocessing import build_pipeline

warnings.filterwarnings("ignore")

GB_PARAM_DIST = {
    "clf__n_estimators": [100, 200, 300],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__max_depth": [2, 3, 4],
    "clf__subsample": [0.7, 0.85, 1.0],
}
GB_N_ITER = 12   # lưới thu hẹp (RandomizedSearchCV) để chạy không quá lâu

SVM_PARAM_GRID = [
    {"clf__kernel": ["linear"], "clf__C": [0.01, 0.1, 1],
     "clf__class_weight": [None, "balanced"]},
    {"clf__kernel": ["rbf"], "clf__C": [1, 3, 10, 30, 100],
     "clf__gamma": ["scale", 0.001, 0.003, 0.01],
     "clf__class_weight": [None, "balanced"]},
]


def _save(search, name, label, X_train, y_train, feature_set="C"):
    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cols = ["rank_test_score", "mean_test_score", "std_test_score", "mean_train_score",
            "mean_fit_time"] + [c for c in res.columns if c.startswith("param_")]
    res[cols].to_csv(config.RESULTS_DIR / f"{name}_search_results.csv", index=False)
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(search.best_estimator_, config.MODELS_DIR / f"{name}_best.joblib")
    # Đánh giá lại mô hình tốt nhất bằng cùng hàm chung (CV trên train) để vào bảng so sánh
    r = cross_validate_model(search.best_estimator_, X_train, y_train, name=label,
                             feature_set=feature_set, log=True)
    print(label, "best params:", search.best_params_)
    print({k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
    return r


def tune_gb(X_train, y_train, feature_set="C"):
    pipe = build_pipeline(GradientBoostingClassifier(random_state=config.RANDOM_STATE),
                          feature_set, scale=False)
    s = RandomizedSearchCV(pipe, GB_PARAM_DIST, n_iter=GB_N_ITER, scoring="f1_macro",
                           cv=get_cv(), random_state=config.RANDOM_STATE, n_jobs=-1,
                           return_train_score=True, refit=True)
    t = time.time()
    s.fit(X_train, y_train)
    print(f"GB search: {time.time() - t:.0f}s")
    return _save(s, "gb", "GradientBoosting(tuned)", X_train, y_train, feature_set)


def tune_svm(X_train, y_train, feature_set="C"):
    pipe = build_pipeline(SVC(random_state=config.RANDOM_STATE), feature_set, scale=True)  # SVM bắt buộc chuẩn hóa
    s = GridSearchCV(pipe, SVM_PARAM_GRID, scoring="f1_macro", cv=get_cv(), n_jobs=-1,
                     return_train_score=True, refit=True)
    t = time.time()
    s.fit(X_train, y_train)
    print(f"SVM search: {time.time() - t:.0f}s")
    return _save(s, "svm", "SVM(tuned)", X_train, y_train, feature_set)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    X_train, X_test, y_train, y_test = get_train_test("C")  # test KHÔNG dùng
    if which in ("svm", "all"):
        tune_svm(X_train, y_train)
    if which in ("gb", "all"):
        tune_gb(X_train, y_train)

"""Imbalance comparison: bảng đối chứng cân bằng lớp ĐẦY ĐỦ (mean ± std, CV 5 fold trên train) cho LR, RF, SVM, GB.

Chạy từ thư mục chứa README:
    .\\.venv\\Scripts\\python.exe -X utf8 scripts/compare_imbalance_strategies.py
Đầu ra: notebooks/Experiments/Logs/Experiment_05_Imbalance/imbalance_comparison_full.csv.
Chiến lược: none | class_weight (nếu mô hình hỗ trợ) | SMOTE (chỉ trên fold train, nằm TRONG pipeline).
GB không có tham số class_weight nên chỉ so none với SMOTE. Không dùng tập test.
Lưu ý SMOTE: nếu máy có imbalanced-learn thì dùng SMOTE của imblearn; nếu không dùng SimpleSMOTE
trong src/imbalance.py (cột `smote_impl` ghi rõ).
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC

from src import config
from src.data import get_train_test
from src.evaluate import get_cv
from src.imbalance import HAS_IMBLEARN, build_smote_pipeline
from src.models.proba import GB_BEST, SVM_BEST
from src.preprocessing import build_pipeline

warnings.filterwarnings("ignore")
SCORING = {"f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy"}
for cls in ["Dropout", "Enrolled"]:
    SCORING[f"precision_{cls}"] = make_scorer(precision_score, labels=[cls], average="macro", zero_division=0)
    SCORING[f"recall_{cls}"] = make_scorer(recall_score, labels=[cls], average="macro", zero_division=0)
    SCORING[f"f1_{cls}"] = make_scorer(f1_score, labels=[cls], average="macro", zero_division=0)

RS = config.RANDOM_STATE
MODELS = {   # tên -> (hàm tạo estimator theo class_weight, scale, hỗ trợ class_weight)
    "LogisticRegression": (lambda cw: LogisticRegression(max_iter=3000, class_weight=cw), True, True),
    "RandomForest": (lambda cw: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight=cw, random_state=RS, n_jobs=1), False, True),
    "SVM(RBF, best)": (lambda cw: SVC(random_state=RS, **{**SVM_BEST, "class_weight": cw}), True, True),
    "GradientBoosting(best)": (lambda cw: GradientBoostingClassifier(random_state=RS, **GB_BEST), False, False),
}


def main():
    X_train, _X_test, y_train, _y_test = get_train_test("C")
    del _X_test, _y_test
    rows = []
    for name, (mk, scale, has_cw) in MODELS.items():
        for strategy in ["none", "class_weight", "SMOTE"]:
            if strategy == "class_weight" and not has_cw:
                continue
            if strategy == "none":
                pipe = build_pipeline(mk(None), "C", scale)
            elif strategy == "class_weight":
                pipe = build_pipeline(mk("balanced"), "C", scale)
            else:
                pipe = build_smote_pipeline(mk(None), "C", scale)
            t = time.time()
            res = cross_validate(pipe, X_train, y_train, cv=get_cv(), scoring=SCORING,
                                 return_train_score=False, n_jobs=-1, error_score="raise")
            row = {"model": name, "strategy": strategy, "smote_impl": ("imblearn" if HAS_IMBLEARN else "SimpleSMOTE") if strategy == "SMOTE" else ""}
            for k in SCORING:
                row[f"{k}_mean"] = float(np.mean(res[f"test_{k}"]))
                row[f"{k}_std"] = float(np.std(res[f"test_{k}"]))
            rows.append(row)
            print(f"{name:24s} {strategy:12s} F1m={row['f1_macro_mean']:.4f}±{row['f1_macro_std']:.4f} "
                  f"R_Dr={row['recall_Dropout_mean']:.3f} P_Dr={row['precision_Dropout_mean']:.3f} F1_En={row['f1_Enrolled_mean']:.3f} ({time.time()-t:.0f}s)", flush=True)
    config.IMBALANCE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.IMBALANCE_LOG_DIR / "imbalance_comparison_full.csv", index=False)


if __name__ == "__main__":
    main()

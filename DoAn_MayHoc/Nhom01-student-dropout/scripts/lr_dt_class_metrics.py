"""Chỉ số theo lớp và confusion matrix (out-of-fold) cho LR/DT của . Chỉ dùng TRAIN, không dùng test.

    python scripts/lr_dt_class_metrics.py

Đọc siêu tham số từ models/model_metadata.json, dự đoán out-of-fold bằng Stratified 5-fold CV chung của nhóm,
ghi kết quả trong notebooks/Experiments/Logs/Experiment_03_LR_DT_Tuning/.
Lưu ý: chỉ số gộp trên toàn bộ dự đoán OOF nên Macro-F1 có thể lệch nhẹ so với trung bình 5 fold trong cv_summary.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data import get_train_test
from src.evaluate import confusion, get_cv
from src.preprocessing import build_pipeline


def make_pipeline(name, params):
    if name == "lr":
        est, scale = LogisticRegression(solver="saga", max_iter=2000, tol=1e-3, random_state=config.RANDOM_STATE), True
    else:
        est, scale = DecisionTreeClassifier(random_state=config.RANDOM_STATE), False
    return build_pipeline(est, "C", scale=scale).set_params(**params)


def main():
    meta = json.load(open(config.MODELS_DIR / "model_metadata.json", encoding="utf-8"))
    X, _, y, _ = get_train_test(meta["feature_set"])
    rows, cms = [], []
    for name, info in meta["models"].items():
        pred = cross_val_predict(make_pipeline(name, info["best_params"]), X, y, cv=get_cv(), n_jobs=1)
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=config.CLASSES, zero_division=0)
        for i, c in enumerate(config.CLASSES):
            rows.append({"model": name, "class": c, "precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])})
        rows.append({"model": name, "class": "macro_avg", "precision": p.mean(), "recall": r.mean(),
                     "f1": f1_score(y, pred, average="macro"), "support": int(s.sum())})
        rows.append({"model": name, "class": "accuracy", "precision": accuracy_score(y, pred), "recall": None, "f1": None, "support": int(s.sum())})
        rows.append({"model": name, "class": "balanced_accuracy", "precision": balanced_accuracy_score(y, pred), "recall": None, "f1": None, "support": int(s.sum())})
        cm = confusion(y, pred).reset_index().rename(columns={"index": "true"})
        cm.insert(0, "model", name)
        cms.append(cm)
    out = pd.DataFrame(rows).round(4)
    config.LR_DT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.LR_DT_LOG_DIR / "lr_dt_class_metrics_cv.csv", index=False)
    pd.concat(cms).to_csv(config.LR_DT_LOG_DIR / "lr_dt_confusion_cv.csv", index=False)
    print(out.to_string(index=False))
    print(pd.concat(cms).to_string(index=False))


if __name__ == "__main__":
    main()

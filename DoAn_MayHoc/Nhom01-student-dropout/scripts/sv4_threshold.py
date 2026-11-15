"""SV4-THR / SV4-PROBA: dự đoán OOF trên TRAIN, bảng ngưỡng Dropout, đồ thị, ngưỡng chốt.

Chạy từ thư mục chứa README:
    .\\.venv\\Scripts\\python.exe -X utf8 -m scripts.sv4_threshold
(hoặc: python -X utf8 scripts/sv4_threshold.py)
KHÔNG đọc/chấm tập test: X_test, y_test bị bỏ ngay sau khi chia. Không ghi đè mô hình .joblib.
Đầu ra: reports/threshold_table_{gb,svm}.csv, reports/oof_proba_{gb,svm}.csv,
        reports/threshold_nested_{gb,svm}.csv, reports/threshold_summary.json,
        reports/figures/sv4_threshold_{gb,svm}.png, models/threshold_config.json
"""
import json
import os
import platform
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import cross_val_predict

from src import config
from src.data import get_train_test
from src.evaluate import get_cv
from src.models.proba import build_gb, build_svm
from src.threshold import (GRID, apply_threshold, choose_threshold, nested_check, oof_proba,
                           predict_argmax, threshold_table)

N_JOBS = int(os.environ.get("N_JOBS", "-1"))


def ece(y_bin, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y_bin[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def plot(tab, name, chosen, path):
    g = tab[tab.threshold != "argmax"].copy(); g["threshold"] = g.threshold.astype(float)
    base = tab[tab.threshold == "argmax"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(g.threshold, g.precision_Dropout, label="Precision Dropout", color="#c0504d")
    ax.plot(g.threshold, g.recall_Dropout, label="Recall Dropout", color="#1f5fa8")
    ax.plot(g.threshold, g.f1_macro, label="Macro-F1 (3 lớp)", color="#2e8b57", lw=2.4)
    ax.axhline(base.recall_Dropout, color="#1f5fa8", ls=":", lw=1)
    ax.axhline(base.f1_macro, color="#2e8b57", ls=":", lw=1)
    if chosen is not None:
        ax.axvline(chosen, color="k", ls="--", lw=1); ax.text(chosen + 0.005, 0.12, f"t = {chosen:.2f}".replace(".", ","), fontsize=10)
    ax.set_xlabel("Ngưỡng t cho P(Dropout)"); ax.set_ylabel("Chỉ số (OOF, train)")
    ax.set_title(f"{name}: ngưỡng Dropout (đường chấm = quy tắc argmax)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left"); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    X_train, _X_test, y_train, _y_test = get_train_test("C")
    del _X_test, _y_test                                  # test KHÔNG được dùng ở đây
    folds = [va for _, va in get_cv().split(X_train, y_train)]
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary, cfg_models = {}, {}
    for key, label, pipe in [("gb", "Gradient Boosting", build_gb("C")),
                             ("svm", "SVM RBF (probability=True)", build_svm("C", probability=True))]:
        print(f"== {label}: OOF xác suất trên train ({len(X_train)} dòng)")
        P = oof_proba(pipe, X_train, y_train)
        P.to_csv(config.RESULTS_DIR / f"oof_proba_{key}.csv", index_label="row_index")
        tab = threshold_table(y_train, P)
        tab.to_csv(config.RESULTS_DIR / f"threshold_table_{key}.csv", index=False)
        t = choose_threshold(tab)
        nest = nested_check(y_train, P, folds)
        nest.to_csv(config.RESULTS_DIR / f"threshold_nested_{key}.csv", index=False)
        plot(tab, label, t, config.FIGURES_DIR / f"sv4_threshold_{key}.png")
        base = tab[tab.threshold == "argmax"].iloc[0]
        row = tab[tab.threshold == t].iloc[0] if t is not None else base
        y_d = (y_train.to_numpy() == config.POSITIVE_CLASS).astype(int)
        pd_ = P[config.POSITIVE_CLASS].to_numpy()
        info = dict(
            model=label, threshold=t,
            argmax={k: float(base[k]) for k in ["f1_macro", "recall_Dropout", "precision_Dropout", "f1_Dropout", "f1_Enrolled", "precision_Enrolled", "recall_Enrolled"]},
            chosen={k: float(row[k]) for k in ["f1_macro", "recall_Dropout", "precision_Dropout", "f1_Dropout", "f1_Enrolled", "precision_Enrolled", "recall_Enrolled"]},
            n_pred_dropout_argmax=int(base.n_pred_dropout), n_pred_dropout_chosen=int(row.n_pred_dropout),
            n_true_dropout=int(y_d.sum()),
            nested_f1_macro_thr_mean=float(nest.f1_macro_thr.mean()), nested_f1_macro_thr_std=float(nest.f1_macro_thr.std(ddof=0)),
            nested_f1_macro_argmax_mean=float(nest.f1_macro_argmax.mean()),
            nested_recall_dropout_thr_mean=float(nest.recall_dropout_thr.mean()),
            nested_recall_dropout_argmax_mean=float(nest.recall_dropout_argmax.mean()),
            nested_thresholds=[None if v is None or pd.isna(v) else float(v) for v in nest.threshold],
            nested_folds_thr_better=int((nest.f1_macro_thr > nest.f1_macro_argmax).sum()),
            log_loss=float(log_loss(y_train, P.to_numpy(), labels=list(P.columns))),
            brier_dropout=float(brier_score_loss(y_d, pd_)), ece_dropout=ece(y_d, pd_),
        )
        if key == "svm":     # predict() của SVC(probability=True) có thể khác argmax(predict_proba)
            pred = cross_val_predict(pipe, X_train, y_train, cv=get_cv(), n_jobs=N_JOBS)
            info["svm_predict_vs_argmax_proba_disagree"] = float((pred != predict_argmax(P.to_numpy(), list(P.columns))).mean())
        summary[key] = info
        cfg_models[key] = dict(threshold=t, positive_class=config.POSITIVE_CLASS,
                               classes_order=list(P.columns), feature_set="C",
                               expects="pipeline fit trên TOÀN train, có predict_proba")
        print(json.dumps({k: v for k, v in info.items() if k in ("threshold", "argmax", "chosen", "nested_f1_macro_thr_mean", "nested_f1_macro_argmax_mean")}, indent=1, ensure_ascii=False))
    (config.RESULTS_DIR / "threshold_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    config.MODELS_DIR.mkdir(exist_ok=True)
    cfg = dict(
        description="Cấu hình ngưỡng Dropout chọn trên dự đoán OOF của TRAIN (CV 5 fold, seed 42); không dùng test.",
        decision_rule="Nếu P(Dropout) >= threshold -> Dropout; ngược lại argmax trong {Enrolled, Graduate}. Xem src/threshold.py",
        selection_criterion="Trong các ngưỡng có Recall Dropout >= Recall Dropout của argmax, chọn Macro-F1 OOF cao nhất; hòa chọn ngưỡng lớn hơn.",
        grid=[float(GRID.min()), float(GRID.max()), 0.01], models=cfg_models, created=str(date.today()),
        versions=dict(python=platform.python_version(), scikit_learn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__),
        note="Ngưỡng gắn với mô hình tương ứng; nếu Quốc đổi mô hình/feature set phải chạy lại scripts/sv4_threshold.py.")
    (config.MODELS_DIR / "threshold_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

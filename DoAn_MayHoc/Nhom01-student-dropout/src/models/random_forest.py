"""Issue #10 (SV1): python -m src.models.random_forest --n-iter 20 --n-jobs 4.

Only training data is used. The held-out test remains reserved for issue #18.
"""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, learning_curve

from src import config
from src.data import get_train_test
from src.evaluate import cross_validate_model, get_cv, log_result
from src.preprocessing import build_pipeline


PARAM_DISTRIBUTIONS = {
    "clf__n_estimators": [100, 200, 400],
    "clf__max_depth": [None, 10, 20, 30],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", 0.5],
    "clf__class_weight": [None, "balanced", "balanced_subsample"],
}


def build_random_forest(feature_set="C"):
    """Keep feature engineering/encoding inside CV; trees do not need scaling."""
    return build_pipeline(
        RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=1),
        feature_set=feature_set, scale=False,
    )


def tune_random_forest(X_train, y_train, feature_set="C", n_iter=20, n_jobs=4):
    """Search train CV with a fixed seed, then refit the winner on all train rows."""
    search = RandomizedSearchCV(
        build_random_forest(feature_set), PARAM_DISTRIBUTIONS,
        n_iter=n_iter, scoring="f1_macro", cv=get_cv(),
        random_state=config.RANDOM_STATE, refit=True,
        return_train_score=True, error_score="raise",
        n_jobs=n_jobs, pre_dispatch="n_jobs", verbose=1,
    )
    return search.fit(X_train, y_train)


def compute_learning_curve(model, X_train, y_train, n_jobs=4):
    """Refit clones of the complete pipeline; keep validation folds untouched."""
    sizes, train, validation = learning_curve(
        model, X_train, y_train, train_sizes=np.linspace(0.2, 1.0, 5),
        cv=get_cv(), scoring="f1_macro", shuffle=True,
        random_state=config.RANDOM_STATE, error_score="raise",
        n_jobs=n_jobs, pre_dispatch="n_jobs",
    )
    table = pd.DataFrame({"train_size": sizes})
    for label, scores in [("train", train), ("validation", validation)]:
        table[f"{label}_f1_macro_mean"] = scores.mean(axis=1)
        table[f"{label}_f1_macro_std"] = scores.std(axis=1)
        for fold in range(scores.shape[1]):
            table[f"{label}_fold_{fold + 1}"] = scores[:, fold]
    return table


def plot_learning_curve(table, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = table["train_size"].to_numpy()
    for label, color in [("train", "tab:blue"), ("validation", "tab:orange")]:
        mean = table[f"{label}_f1_macro_mean"].to_numpy()
        std = table[f"{label}_f1_macro_std"].to_numpy()
        ax.plot(sizes, mean, "o-", label=label.capitalize(), color=color)
        ax.fill_between(sizes, mean - std, mean + std, alpha=0.15, color=color)
    ax.set(xlabel="Training samples per fold", ylabel="Macro-F1", ylim=(0, 1.02),
           title=f"Random Forest learning curve ({config.N_SPLITS}-fold stratified CV)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(summary, curve, path):
    metrics = summary["metrics"]
    lines = [
        "# Random Forest — Issue #10, SV1 Quốc", "",
        "## Phương pháp", "",
        f"Tập đặc trưng {summary['feature_set']}; {summary['train_rows']} mẫu train. "
        f"Chia train/test phân tầng theo cấu hình chung, seed {config.RANDOM_STATE}. "
        "Không đánh giá tập test; giữ cho issue #18.", "",
        f"RandomizedSearchCV {summary['n_iter']} cấu hình trong không gian 648 tổ hợp, "
        f"{config.N_SPLITS} fold, chọn theo Macro-F1. Tiền xử lý, One-Hot và các biến "
        "tạo thêm nằm trong Pipeline; không chuẩn hóa cho RF. Sau tuning, pipeline "
        "tốt nhất được fit lại trên toàn bộ train và lưu vào models/rf_best.joblib.", "",
        "## Siêu tham số được chọn", "", "```json",
        json.dumps(summary["best_params"], indent=2), "```", "",
        "## Các độ đo chung trên CV train", "",
        "| Độ đo | Trung bình | Độ lệch chuẩn |", "|---|---:|---:|",
    ]
    for metric in ["f1_macro", "balanced_accuracy", "accuracy", "recall_dropout", "f1_enrolled"]:
        lines.append(f"| {metric} | {metrics[metric + '_mean']:.4f} | {metrics[metric + '_std']:.4f} |")
    gap = float(curve.iloc[-1].train_f1_macro_mean - curve.iloc[-1].validation_f1_macro_mean)
    lines += ["", f"Macro-F1 train (CV): {metrics['train_f1_macro_mean']:.4f}.", "",
        "Đây là CV sau lựa chọn siêu tham số trên cùng train/fold, có thể lạc quan; "
        "không phải đánh giá độc lập hoặc kết quả test.", "",
        "## Learning curve", "", "![Learning curve](figures/rf_learning_curve.png)", "",
        "Đường bao là ±1 độ lệch chuẩn qua các fold; mỗi mức fit lại toàn bộ pipeline.", "",
        "| Số mẫu train/fold | Macro-F1 train | Macro-F1 validation |", "|---:|---:|---:|"]
    for row in curve.itertuples():
        lines.append(f"| {row.train_size} | {row.train_f1_macro_mean:.4f} | {row.validation_f1_macro_mean:.4f} |")
    lines += ["", f"Khoảng cách train–validation ở mức lớn nhất là {gap:.4f}. "
        + ("Khoảng cách còn lớn, cho thấy dấu hiệu overfit." if gap > 0.1
           else "Cần xem đồng thời độ lệch chuẩn và xu hướng khi tăng dữ liệu để đánh giá overfit."), "",
        "Learning curve dùng mẫu đã xáo trộn và cùng kích thước train tuyệt đối cho "
        "mọi fold; điểm cuối có thể khác nhẹ CV tuning dùng trọn phần train mỗi fold.", "",
        "## Tái lập", "", "```powershell",
        f".\\.venv\\Scripts\\python.exe -X utf8 -m src.models.random_forest --feature-set {summary['feature_set']} --n-iter {summary['n_iter']} --n-jobs 4",
        ".\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s tests -v", "```", "",
        "Phiên bản thư viện, SHA-256 dữ liệu, tham số và độ đo lưu trong rf_summary.json. "
        "requirements-rf.txt ghi chính xác các phiên bản dùng cho lượt chạy này. "
        "Chạy lệnh tại thư mục chứa README; cần dữ liệu gốc data/raw/data.csv.", "",
        "Sản phẩm: rf_search_results.csv, rf_learning_curve.csv, rf_summary.json, "
        "rf_cv_metrics.csv, figures/rf_learning_curve.png và ../models/rf_best.joblib. "
        "Kết quả cũng được thêm vào cv_results.csv bằng hàm log chung (file Git ignore).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(feature_set="C", n_iter=20, n_jobs=4):
    X_train, _, y_train, _ = get_train_test(feature_set)
    print(f"Training rows: {len(X_train)}; feature set: {feature_set}", flush=True)
    search = tune_random_forest(X_train, y_train, feature_set, n_iter, n_jobs)
    print(f"Best CV Macro-F1: {search.best_score_:.4f}; computing metrics and curve", flush=True)
    metrics = cross_validate_model(search.best_estimator_, X_train, y_train,
                                  name="RandomForest", feature_set=feature_set,
                                  log=False, n_jobs=n_jobs)
    curve = compute_learning_curve(search.best_estimator_, X_train, y_train, n_jobs)
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib"]
    summary = {
        "feature_set": feature_set, "train_rows": len(X_train),
        "random_state": config.RANDOM_STATE, "test_size": config.TEST_SIZE,
        "cv_folds": config.N_SPLITS, "n_iter": n_iter,
        "best_params": search.best_params_, "best_cv_f1_macro": float(search.best_score_),
        "param_distributions": PARAM_DISTRIBUTIONS, "metrics": metrics,
        "data_sha256": hashlib.sha256(config.DATA_FILE.read_bytes()).hexdigest(),
        "python": platform.python_version(), "versions": {p: version(p) for p in packages},
        "evaluation_note": "Post-selection CV on train; hold-out test was not evaluated.",
    }
    for directory in [config.MODELS_DIR, config.RESULTS_DIR, config.FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, config.MODELS_DIR / "rf_best.joblib")
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        config.RESULTS_DIR / "rf_search_results.csv", index=False)
    pd.DataFrame([metrics]).to_csv(config.RESULTS_DIR / "rf_cv_metrics.csv", index=False)
    curve.to_csv(config.RESULTS_DIR / "rf_learning_curve.csv", index=False)
    plot_learning_curve(curve, config.FIGURES_DIR / "rf_learning_curve.png")
    (config.RESULTS_DIR / "rf_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (config.ROOT / "requirements-rf.txt").write_text(
        f"# Issue #10 run environment: Python {summary['python']}\n"
        + "\n".join(f"{p}=={v}" for p, v in summary["versions"].items()) + "\n",
        encoding="utf-8")
    write_report(summary, curve, config.RESULTS_DIR / "03_random_forest.md")
    log_result(metrics, config.CV_RESULTS_FILE)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return search, curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", choices=list(config.FEATURE_SETS), default="C")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.n_iter <= 648 or args.n_jobs == 0:
        parser.error("--n-iter must be in 1..648; --n-jobs must be nonzero")
    run_experiment(args.feature_set, args.n_iter, args.n_jobs)


if __name__ == "__main__":
    main()

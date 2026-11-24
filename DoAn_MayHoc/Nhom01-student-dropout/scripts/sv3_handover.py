"""Gói bàn giao LR/DT cho Quốc (final_manifest) - chạy trong môi trường KHÓA của nhóm (requirements.txt).

    python scripts/sv3_handover.py --write

Việc làm (chỉ dùng TRAIN, không đụng tập test):
1. Đọc siêu tham số tốt nhất từ models/sv3_metadata.json.
2. Chạy CV 5-fold chung của nhóm, ghi bằng chứng CV cùng định dạng rf_cv_metrics.csv:
   reports/lr_cv_metrics.csv, reports/dt_cv_metrics.csv.
3. Với --write: fit lại trên toàn bộ train, ghi models/lr_best.joblib, models/dt_best.joblib, rồi thử nạp với
   warnings=error (cách Quốc kiểm tra) để chắc không lệch phiên bản.
4. Ghi models/sv3_handover.json: sha256 mô hình, số dòng train, sha256 dữ liệu, bằng chứng CV, phiên bản thư viện,
   quy tắc quyết định (native_predict). Quốc đối chiếu các trường này vào final_manifest.
"""
import argparse
import hashlib
import json
import platform
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data import get_train_test
from src.evaluate import cross_validate_model
from src.preprocessing import build_pipeline

NAMES = {"lr": "LogisticRegression", "dt": "DecisionTree"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_pipeline(name, params):
    if name == "lr":
        est, scale = LogisticRegression(solver="saga", max_iter=2000, tol=1e-3, random_state=config.RANDOM_STATE), True
    else:
        est, scale = DecisionTreeClassifier(random_state=config.RANDOM_STATE), False
    return build_pipeline(est, "C", scale=scale).set_params(**params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="fit lại trên toàn train và ghi models/*.joblib")
    args = ap.parse_args()
    meta = json.load(open(config.MODELS_DIR / "sv3_metadata.json", encoding="utf-8"))
    fs = meta["feature_set"]
    X_train, _, y_train, _ = get_train_test(fs)
    data_hash = sha(config.DATA_FILE)
    libs = {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]}
    print("Python", platform.python_version(), libs)

    entries = []
    for key, info in meta["models"].items():
        pipe = make_pipeline(key, info["best_params"])
        res = cross_validate_model(pipe, X_train, y_train, name=NAMES[key], feature_set=fs, log=False)
        evidence = config.RESULTS_DIR / f"{key}_cv_metrics.csv"
        pd.DataFrame([res]).to_csv(evidence, index=False)
        model_path = config.MODELS_DIR / f"{key}_best.joblib"
        if args.write:
            joblib.dump(pipe.fit(X_train, y_train), model_path)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                joblib.load(model_path)
            loads = "OK"
        except Exception as exc:  # noqa: BLE001
            loads = "TỪ CHỐI: " + type(exc).__name__
        print(f"{NAMES[key]}: CV Macro-F1 {res['f1_macro_mean']:.4f} ± {res['f1_macro_std']:.4f} "
              f"(metadata {info['cv_f1_macro']:.4f}); nạp với warnings=error: {loads}")
        entries.append({
            "name": key.upper(), "path": f"models/{key}_best.joblib", "feature_set": fs,
            "sha256": sha(model_path), "trained_rows": int(len(X_train)), "training_data_sha256": data_hash,
            "cv_evidence": f"reports/{key}_cv_metrics.csv", "cv_evidence_sha256": sha(evidence),
            "decision": {"kind": "native_predict"}, "loads_without_warning": loads == "OK",
        })
    out = {"python": platform.python_version(), "versions": libs, "data_sha256": data_hash,
           "models": entries, "note": "Chỉ dùng train/CV; không dùng test."}
    (config.MODELS_DIR / "sv3_handover.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã ghi models/sv3_handover.json")
    if not all(e["loads_without_warning"] for e in entries):
        print("CẢNH BÁO: chạy lại với --write trong môi trường khóa (Python 3.12.10, requirements.txt).")


if __name__ == "__main__":
    main()

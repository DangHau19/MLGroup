"""Tái lập LR/DT của  từ models/model_metadata.json (không tuning lại).

    python scripts/reproduce_lr_dt.py            # chỉ kiểm tra: CV Macro-F1 có khớp metadata không
    python scripts/reproduce_lr_dt.py --write    # kiểm tra, fit lại trên toàn train và ghi đè models/*.joblib

Chỉ dùng tập TRAIN (CV 5-fold chung của nhóm). Không đụng tập test.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data import get_train_test
from src.evaluate import cross_validate_model
from src.preprocessing import build_pipeline


def make_pipeline(name, params):
    if name == "lr":
        est, scale = LogisticRegression(solver="saga", max_iter=2000, tol=1e-3, random_state=config.RANDOM_STATE), True
    else:
        est, scale = DecisionTreeClassifier(random_state=config.RANDOM_STATE), False
    return build_pipeline(est, "C", scale=scale).set_params(**params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    meta = json.load(open(config.MODELS_DIR / "model_metadata.json", encoding="utf-8"))
    X_train, _, y_train, _ = get_train_test(meta["feature_set"])
    ok = True
    for name, info in meta["models"].items():
        pipe = make_pipeline(name, info["best_params"])
        r = cross_validate_model(pipe, X_train, y_train, name=name, feature_set=meta["feature_set"], log=False)
        diff = abs(r["f1_macro_mean"] - info["cv_f1_macro"])
        flag = "OK" if diff < 0.005 else "LỆCH"
        ok &= diff < 0.005
        print(f"{name}: CV Macro-F1 = {r['f1_macro_mean']:.4f} (metadata {info['cv_f1_macro']:.4f}) -> {flag}")
        if args.write:
            joblib.dump(pipe.fit(X_train, y_train), config.MODELS_DIR / Path(info["file"]).name)
            print("  đã ghi", info["file"])
    print("Tái lập thành công." if ok else "CẢNH BÁO: kết quả lệch metadata (kiểm tra phiên bản thư viện).")


if __name__ == "__main__":
    main()

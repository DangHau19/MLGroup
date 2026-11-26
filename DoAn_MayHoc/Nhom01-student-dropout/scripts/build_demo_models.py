r"""Fit A/B demo pipelines from the already selected C configurations.

Only the train split is used. The reserved hold-out is never loaded or scored.
Run from the project root with the locked .venv:
    .\.venv\Scripts\python.exe -X utf8 scripts\build_demo_models.py
"""
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data import get_train_test
from src.preprocessing import build_pipeline


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def params_for_models():
    metadata = json.loads((config.MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    rf = json.loads((config.RF_LOG_DIR / "rf_summary.json").read_text(encoding="utf-8"))
    gb = json.loads((config.MODELS_DIR / "gb_best.meta.json").read_text(encoding="utf-8"))
    svm = json.loads((config.MODELS_DIR / "svm_best_proba.meta.json").read_text(encoding="utf-8"))
    return {
        "lr": (LogisticRegression(solver="saga", max_iter=2000, tol=1e-3,
                                   random_state=config.RANDOM_STATE), True,
               metadata["models"]["lr"]["best_params"]),
        "dt": (DecisionTreeClassifier(random_state=config.RANDOM_STATE), False,
               metadata["models"]["dt"]["best_params"]),
        "rf": (RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=1), False,
               rf["best_params"]),
        "gb": (GradientBoostingClassifier(random_state=config.RANDOM_STATE), False,
               {f"clf__{k}": v for k, v in gb["params"].items()}),
        "svm": (SVC(random_state=config.RANDOM_STATE, probability=True), True,
                {f"clf__{k}": v for k, v in svm["params"].items()}),
    }


def main():
    data_hash = sha(config.DATA_FILE)
    cv = pd.read_csv(config.TIMEPOINT_LOG_DIR / "timepoint_cv.csv")
    expected = {"LR", "DT", "RF", "GB", "SVM"}
    if set(cv["model"].unique()) != expected:
        raise SystemExit("timepoint_cv.csv must contain the five locked models")

    specs = params_for_models()
    versions = {name: version(name) for name in
                ["numpy", "pandas", "scikit-learn", "joblib"]}
    registry = {"python": platform.python_version(), "versions": versions,
                "data_sha256": data_hash, "seed": config.RANDOM_STATE,
                "cv": "Stratified 5-fold, seed 42; configurations selected on C",
                "holdout_evaluated": False, "models": {}}

    for feature_set in ("A", "B"):
        X_train, _, y_train, _ = get_train_test(feature_set)
        dest = config.MODELS_DIR / "demo" / feature_set
        dest.mkdir(parents=True, exist_ok=True)
        for name, (estimator, scale, params) in specs.items():
            pipe = build_pipeline(estimator, feature_set=feature_set, scale=scale)
            pipe.set_params(**params)
            pipe.fit(X_train, y_train)
            if list(pipe.classes_) != config.CLASSES:
                raise ValueError(f"Unexpected classes in {name}-{feature_set}: {pipe.classes_}")
            model_path = dest / f"{name}_best.joblib"
            joblib.dump(pipe, model_path)
            p = pipe.predict_proba(X_train.iloc[:2])
            if p.shape != (2, len(config.CLASSES)) or not (abs(p.sum(axis=1) - 1) < 1e-8).all():
                raise ValueError(f"Invalid probability output from {name}-{feature_set}")
            group = cv[(cv.model.str.upper() == name.upper()) & (cv.feature_set == feature_set)]
            if len(group) != config.N_SPLITS:
                raise ValueError(f"Expected five CV folds for {name}-{feature_set}")
            registry["models"].setdefault(name, {})[feature_set] = {
                "path": model_path.relative_to(config.ROOT).as_posix(),
                "sha256": sha(model_path), "feature_set": feature_set,
                "trained_rows": int(len(X_train)), "training_data_sha256": data_hash,
                "selected_parameters_from_C": params,
                "cv_macro_f1_mean": float(group.f1_macro.mean()),
                "cv_macro_f1_std": float(group.f1_macro.std(ddof=0)),
                "decision": "native_predict",
            }
            print(f"Built {name.upper()}-{feature_set}: {model_path}")

    # C uses the canonical final artifacts consumed by final evaluation.
    for name in expected:
        model_name = name.lower()
        filename = "svm_best_proba.joblib" if model_name == "svm" else f"{model_name}_best.joblib"
        model_path = config.MODELS_DIR / filename
        meta_path = config.MODELS_DIR / f"{model_name}_best.meta.json"
        if model_name == "lr" or model_name == "dt":
            handover = json.loads((config.MODELS_DIR / "model_handover.json").read_text(encoding="utf-8"))
            entry = next(x for x in handover["models"] if x["name"].lower() == model_name)
            params = json.loads((config.MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))["models"][model_name]["best_params"]
        elif model_name == "rf":
            meta = json.loads((config.RF_LOG_DIR / "rf_summary.json").read_text(encoding="utf-8"))
            entry = {"trained_rows": meta["train_rows"], "training_data_sha256": meta["data_sha256"]}
            params = meta["best_params"]
        else:
            meta_filename = "svm_best_proba.meta.json" if model_name == "svm" else "gb_best.meta.json"
            meta_path = config.MODELS_DIR / meta_filename
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            entry = {"trained_rows": meta["train_rows"], "training_data_sha256": meta["data_file_sha256"]}
            params = meta["params"]
        group = cv[(cv.model.str.upper() == name.upper()) & (cv.feature_set == "C")]
        registry["models"].setdefault(model_name, {})["C"] = {
            "path": model_path.relative_to(config.ROOT).as_posix(),
            "sha256": sha(model_path), "feature_set": "C",
            "trained_rows": int(entry["trained_rows"]),
            "training_data_sha256": entry["training_data_sha256"],
            "selected_parameters_from_C": params,
            "cv_macro_f1_mean": float(group.f1_macro.mean()),
            "cv_macro_f1_std": float(group.f1_macro.std(ddof=0)),
            "decision": "native_predict",
        }
    registry_path = config.MODELS_DIR / "demo" / "model_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Registry written: {registry_path}; holdout_evaluated=false")


if __name__ == "__main__":
    main()

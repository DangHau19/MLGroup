"""Sequential D experiments: baselines, boosting, tuning, then v2."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json
import zipfile
import optuna
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import common
from models_D import FoldFeatureClassifier
from mlp_D import SmallMLPClassifier

OUT = common.ROOT / "evidence" / "models_D"


def read_cv_meta(name):
    relative = f"evidence/cv_logs/{name}_cv.json"
    path = common.ROOT / relative
    if path.exists():
        return json.loads(path.read_text())
    with zipfile.ZipFile(OUT / "intermediate_D.zip") as archive:
        return json.loads(archive.read(relative))


def data():
    train, test, _ = common.load_data()
    y, classes = common.encode_labels(train[common.LABEL_COL])
    columns = tuple(c for c in test.columns if c != common.ID_COL)
    raw = train.loc[:, columns]
    raw_test = test.loc[:, columns]
    reference = pd.concat([raw, raw_test], ignore_index=True)
    return raw, y, raw_test, reference, columns, classes, test[common.ID_COL].to_numpy()


def evaluate(estimator, family, version="v1", target_encoding=False, suffix="base", resume=True):
    X, y, Xt, ref, columns, classes, ids = data()
    name = f"{family}_{version}{'_te' if target_encoding else ''}_{suffix}_D"
    output = common.OOF_DIR / f"{name}_cv.json"
    expected = [output, common.OOF_DIR / f"{name}_oof.npy", common.OOF_DIR / f"{name}_test.npy"]
    if resume and all(path.exists() for path in expected):
        print(f"Already complete: {name}", flush=True)
        return json.loads(output.read_text())["cv_mean"]
    archive_path = OUT / "intermediate_D.zip"
    if resume and archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            members = set(archive.namelist())
            if all(path.relative_to(common.ROOT).as_posix() in members for path in expected):
                print(f"Archived result: {name}", flush=True)
                return read_cv_meta(name)["cv_mean"]
    model = FoldFeatureClassifier(estimator, columns, ref, version, target_encoding)
    with threadpool_limits(limits=4):
        result = common.cross_validate_model(model, X, y, Xt, name=name)
    assert result.oof_proba.shape == (len(y), len(classes))
    assert result.test_proba.shape == (len(ids), len(classes))
    for p in (result.oof_proba, result.test_proba):
        assert np.isfinite(p).all() and (p >= 0).all()
        assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
    common.save_oof(result)
    meta = json.loads(output.read_text())
    meta.update({"feature_version": version, "target_encoding": target_encoding,
                 "preprocessing": "fold-local median/scaler; global label-free frequencies",
                 "model_params": estimator.get_params(deep=True),
                 "classes": classes.tolist(), "oof_accuracy": common.score(y, result.oof_proba.argmax(1))["accuracy"]})
    output.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(result.summary(), flush=True)
    return result.cv_mean


def baselines(resume=True):
    models = {
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=4)),
        "svm": make_pipeline(StandardScaler(), SVC(C=5, gamma="scale", probability=True, random_state=42)),
        "extratrees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=4, random_state=42),
    }
    for family, estimator in models.items():
        evaluate(estimator, family, resume=resume)


def boosting_models():
    return {
        "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=31,
                                   min_child_samples=30, verbosity=-1, n_jobs=4, random_state=42),
        "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                 subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                                 n_jobs=4, random_state=42, eval_metric="mlogloss"),
        "catboost": CatBoostClassifier(iterations=500, depth=5, learning_rate=0.05,
                                       l2_leaf_reg=5, verbose=False, thread_count=4,
                                       random_seed=42, allow_writing_files=False),
        "mlp": make_pipeline(StandardScaler(), SmallMLPClassifier()),
    }


def boosting(resume=True):
    for family, estimator in boosting_models().items():
        evaluate(estimator, family, resume=resume)


def sampled_model(trial, family):
    choose = trial.suggest_categorical
    if family == "lightgbm":
        return LGBMClassifier(
            num_leaves=choose("num_leaves", [7, 15, 31, 63]),
            learning_rate=choose("learning_rate", [0.025, 0.05, 0.08]),
            n_estimators=choose("n_estimators", [200, 400, 650]),
            min_child_samples=choose("min_child_samples", [20, 50, 100]),
            subsample=choose("subsample", [0.8, 1.0]), subsample_freq=1,
            colsample_bytree=choose("colsample_bytree", [0.8, 1.0]),
            reg_lambda=choose("reg_lambda", [0.0, 2.0, 10.0]),
            verbosity=-1, n_jobs=4, random_state=42)
    if family == "xgboost":
        return XGBClassifier(
            max_depth=choose("max_depth", [3, 4, 5]),
            learning_rate=choose("learning_rate", [0.03, 0.06, 0.1]),
            n_estimators=choose("n_estimators", [200, 400, 650]),
            subsample=choose("subsample", [0.75, 0.9, 1.0]),
            colsample_bytree=choose("colsample_bytree", [0.75, 0.9, 1.0]),
            tree_method="hist", n_jobs=4, random_state=42, eval_metric="mlogloss")
    if family == "catboost":
        return CatBoostClassifier(
            depth=choose("depth", [4, 5, 6]),
            learning_rate=choose("learning_rate", [0.03, 0.06, 0.1]),
            l2_leaf_reg=choose("l2_leaf_reg", [3.0, 10.0, 20.0]),
            iterations=choose("iterations", [300, 500, 800]),
            verbose=False, thread_count=4, random_seed=42, allow_writing_files=False)
    if family == "svm":
        return make_pipeline(StandardScaler(), SVC(
            C=choose("C", [1.0, 5.0, 20.0]),
            gamma=choose("gamma", [0.003, 0.01, 0.03, "scale"]),
            probability=True, random_state=42))
    if family == "mlp":
        sizes = {"64": (64,), "64_32": (64, 32), "128_64": (128, 64)}
        return make_pipeline(StandardScaler(), SmallMLPClassifier(
            hidden_layers=sizes[choose("hidden_layers", list(sizes))],
            dropout=choose("dropout", [0.0, 0.15, 0.3]),
            weight_decay=choose("weight_decay", [0.0, 0.001, 0.01]),
            learning_rate=choose("learning_rate", [0.0005, 0.001, 0.002])))
    raise ValueError(family)


def tuning(resume=True, n_trials=3):
    """Optuna random search; all trials use exactly common.get_folds.

    Test predictions are exported, but never enter the objective. Selection
    uses CV accuracy only. The initial baseline also remains a candidate.
    """
    best = {}
    for family in ("lightgbm", "xgboost", "catboost", "svm", "mlp"):
        storage = f"sqlite:///{(OUT / 'tuning.sqlite3').as_posix()}"
        if not resume:
            try:
                optuna.delete_study(study_name=f"{family}_v1_D", storage=storage)
            except KeyError:
                pass
        study = optuna.create_study(
            study_name=f"{family}_v1_D", direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=42),
            storage=storage, load_if_exists=True)
        def objective(trial):
            estimator = sampled_model(trial, family)
            return evaluate(estimator, family, suffix=f"trial{trial.number:02d}", resume=resume)
        completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        if completed < n_trials:
            study.optimize(objective, n_trials=n_trials - completed)
        # Repair missing probability artifacts even when the SQLite trial is complete.
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                evaluate(sampled_model(optuna.trial.FixedTrial(trial.params), family), family,
                         suffix=f"trial{trial.number:02d}", resume=True)
        study.trials_dataframe().to_csv(OUT / f"{family}_trials.csv", index=False)
        baseline_name = f"{family}_v1_base_D"
        baseline = read_cv_meta(baseline_name)
        if study.best_value > baseline["cv_mean"]:
            trial = study.best_trial
            selected = f"{family}_v1_trial{trial.number:02d}_D"
            params = trial.params
        else:
            selected, params = baseline_name, None
        best[family] = {"selected_v1": selected, "search_params": params,
                        "cv_accuracy": read_cv_meta(selected)["cv_mean"],
                        "n_random_trials": n_trials}
        (OUT / "selected_v1.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(f"SELECTED {family}: {selected}, accuracy={best[family]['cv_accuracy']:.6f}", flush=True)


def selected_models():
    selected = json.loads((OUT / "selected_v1.json").read_text())
    models = boosting_models()
    models["svm"] = make_pipeline(StandardScaler(), SVC(C=5, gamma="scale", probability=True, random_state=42))
    for family, item in selected.items():
        if item["search_params"] is not None:
            models[family] = sampled_model(optuna.trial.FixedTrial(item["search_params"]), family)
    return models


def v2(resume=True):
    models = selected_models()
    for family in ("lightgbm", "xgboost", "catboost"):
        evaluate(models[family], family, version="v2", suffix="selected", resume=resume)
    models["extratrees"] = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=4, random_state=42)
    models["histgb"] = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=42)
    for family, estimator in models.items():
        evaluate(estimator, family, version="v2", target_encoding=True, suffix="selected", resume=resume)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["baselines", "boosting", "tuning", "v2"], default="baselines")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "tuning":
        tuning(resume=not args.force, n_trials=args.trials)
    else:
        {"baselines": baselines, "boosting": boosting, "v2": v2}[args.stage](resume=not args.force)


if __name__ == "__main__":
    main()

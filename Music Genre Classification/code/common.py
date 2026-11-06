"""
common.py - Code dung chung cho ca nhom (Challenge 2: Music Genre Classification).

QUY UOC BAT BUOC:
  1. SEED = 42, N_FOLDS = 5, StratifiedKFold(shuffle=True). Khong tu y doi.
  2. Nhan the loai duoc ma hoa thanh so 0..K-1 bang `encode_labels`.
  3. Moi mo hinh tra ve OOF XAC SUAT (n_mau x K) de E ensemble/stacking.
  4. Metric chinh: PRIMARY_METRIC = accuracy (= micro F1). DA XAC NHAN 06/11 bang bai nop thu (xem PHAN_CONG.md muc 1).

Dung:
    import sys; sys.path.append("../code")
    from common import *
    train, test, sub = load_data()
    ...
    res = cross_validate_model(model, X, y, X_test, name="lgbm")
    save_oof(res)
    make_submission(test_ids, res.test_proba, classes, "lgbm_v1.csv")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5
PRIMARY_METRIC = "accuracy"          # Kaggle cham F1 kieu micro = accuracy (xac nhan 06/11 bang bai nop thu)
#                                    Gia tri hop le: "macro_f1" | "weighted_f1" | "accuracy" (= micro F1)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUB_DIR = ROOT / "submissions"
OOF_DIR = ROOT / "evidence" / "cv_logs"

# Ten cot Id / nhan: tu dong doc tu sample_submission.csv (cot 1 = Id, cot 2 = nhan) neu co.
ID_COL = "Id"          # da xac nhan tu sample_submission.csv (Id, Class)
LABEL_COL = "Class"    # nhan 0..10 (11 the loai)


# --------------------------------------------------------------------------- #
# Du lieu
# --------------------------------------------------------------------------- #
def load_data(data_dir: Path | str = DATA_DIR):
    """Doc train.csv, test.csv (+ sample_submission.csv neu co). Tra ve (train, test, sample_sub|None).
    Neu du lieu la file audio, A se them ham doc rieng; ham nay van dung de doc file nhan/metadata."""
    global ID_COL, LABEL_COL
    d = Path(data_dir)
    tr, te, ss = d / "train.csv", d / "test.csv", d / "sample_submission.csv"
    if not tr.exists() or not te.exists():
        raise FileNotFoundError(
            f"Khong thay train.csv / test.csv trong {d}. Tai tu "
            "https://www.kaggle.com/competitions/shai-music-genre-classification/data va bo vao data/. "
            "Neu ten file khac, A cap nhat ham nay.")
    train, test = pd.read_csv(tr), pd.read_csv(te)
    sub = pd.read_csv(ss) if ss.exists() else None
    if sub is not None and sub.shape[1] >= 2:
        ID_COL, LABEL_COL = sub.columns[0], sub.columns[1]
    return train, test, sub


def encode_labels(labels) -> tuple[np.ndarray, np.ndarray]:
    """Chuoi the loai -> so nguyen 0..K-1. Tra ve (y, classes). classes[i] la ten the loai i."""
    classes, y = np.unique(np.asarray(labels), return_inverse=True)
    return y.astype(int), classes


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def score(y_true, y_pred_label) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred_label)),
            "macro_f1": float(f1_score(y_true, y_pred_label, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred_label, average="weighted"))}


def primary(y_true, y_pred_label) -> float:
    return score(y_true, y_pred_label)[PRIMARY_METRIC]


# --------------------------------------------------------------------------- #
# Cross-validation chung
# --------------------------------------------------------------------------- #
def get_folds(y, n_folds: int = N_FOLDS, seed: int = SEED):
    y = np.asarray(y)
    return list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def _proba(m, X, n_classes):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)
    p = np.zeros((len(X), n_classes))
    p[np.arange(len(X)), m.predict(X)] = 1.0
    return p


@dataclass
class CVResult:
    name: str
    oof_proba: np.ndarray                 # (n_train, K)
    test_proba: np.ndarray | None         # (n_test, K), trung binh cac fold
    fold_scores: list[float]
    fold_macro_f1: list[float]
    seconds: float = 0.0
    cv_mean: float = field(init=False)
    cv_std: float = field(init=False)

    def __post_init__(self):
        self.cv_mean = float(np.mean(self.fold_scores))
        self.cv_std = float(np.std(self.fold_scores))

    def summary(self) -> str:
        return (f"{self.name:<18} CV {PRIMARY_METRIC} = {self.cv_mean:.4f} (+/- {self.cv_std:.4f})  "
                f"macroF1 = {np.mean(self.fold_macro_f1):.4f}  [{self.seconds:.1f}s]")


def cross_validate_model(model, X, y, X_test=None, name: str = "model",
                         fit_params: dict | None = None, verbose: bool = True) -> CVResult:
    """Stratified K-fold chuan cua nhom. y la nhan da ma hoa 0..K-1 (xem encode_labels)."""
    Xa = X.values if hasattr(X, "values") else np.asarray(X)
    ya = np.asarray(y, dtype=int)
    Xt = None if X_test is None else (X_test.values if hasattr(X_test, "values") else np.asarray(X_test))
    K = int(ya.max()) + 1
    fit_params = fit_params or {}

    oof = np.zeros((len(ya), K))
    tp = None if Xt is None else np.zeros((len(Xt), K))
    acc, f1s = [], []
    t0 = time.time()
    for k, (tr, va) in enumerate(get_folds(ya)):
        m = clone(model)
        m.fit(Xa[tr], ya[tr], **fit_params)
        p = _proba(m, Xa[va], K)
        oof[va] = p
        s = score(ya[va], p.argmax(1))
        acc.append(s[PRIMARY_METRIC]); f1s.append(s["macro_f1"])
        if tp is not None:
            tp += _proba(m, Xt, K) / N_FOLDS
        if verbose:
            print(f"  [{name}] fold {k + 1}/{N_FOLDS}: {PRIMARY_METRIC}={acc[-1]:.4f}")
    res = CVResult(name, oof, tp, acc, f1s, seconds=time.time() - t0)
    if verbose:
        print(res.summary())
    return res


# --------------------------------------------------------------------------- #
# Luu / doc ket qua
# --------------------------------------------------------------------------- #
def save_oof(res: CVResult, tag: str | None = None) -> Path:
    tag = tag or res.name
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OOF_DIR / f"{tag}_oof.npy", res.oof_proba)
    if res.test_proba is not None:
        np.save(OOF_DIR / f"{tag}_test.npy", res.test_proba)
    meta = {"name": res.name, "metric": PRIMARY_METRIC, "cv_mean": res.cv_mean, "cv_std": res.cv_std,
            "macro_f1": float(np.mean(res.fold_macro_f1)), "fold_scores": res.fold_scores,
            "seconds": res.seconds, "seed": SEED, "n_folds": N_FOLDS}
    (OOF_DIR / f"{tag}_cv.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return OOF_DIR / f"{tag}_cv.json"


def load_oof(tag: str):
    oof = np.load(OOF_DIR / f"{tag}_oof.npy")
    tp = OOF_DIR / f"{tag}_test.npy"
    return oof, (np.load(tp) if tp.exists() else None)


def leaderboard_table() -> pd.DataFrame:
    """Bang so sanh tat ca mo hinh da luu (dung cho muc Results cua paper)."""
    rows = []
    for p in sorted(OOF_DIR.glob("*_cv.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"model": d["name"], f"cv_{d['metric']}": d["cv_mean"], "cv_std": d["cv_std"],
                     "macro_f1": d["macro_f1"], "seconds": d["seconds"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(df.columns[1], ascending=False).reset_index(drop=True)


def make_submission(test_ids, proba, classes, filename: str = "submission.csv") -> Path:
    """Ghi file nop Kaggle (ID_COL, LABEL_COL) tu ma tran xac suat (n_test, K)."""
    proba = np.asarray(proba, float)
    if proba.ndim != 2 or proba.shape[1] != len(classes):
        raise ValueError(f"proba phai co shape (n_test, {len(classes)}), dang la {proba.shape}")
    if not np.isfinite(proba).all():
        raise ValueError("proba chua NaN/inf. Kiem tra lai mo hinh.")
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({ID_COL: np.asarray(test_ids), LABEL_COL: np.asarray(classes)[proba.argmax(1)]})
    path = SUB_DIR / filename
    out.to_csv(path, index=False)
    print(f"Da ghi {path} ({len(out)} dong). Phan bo nhan du doan:\n{out[LABEL_COL].value_counts().head(10)}")
    return path


def log_submission(filename: str, kaggle_public_score: float, note: str = "", who: str = "") -> None:
    """Ghi nhat ky moi lan nop Kaggle vao evidence/submission_log.csv."""
    path = ROOT / "evidence" / "submission_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"time": pd.Timestamp.now().isoformat(timespec="seconds"), "who": who,
                         "file": filename, "public_score": kaggle_public_score, "note": note}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)

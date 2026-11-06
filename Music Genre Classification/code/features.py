"""
features.py - HOP DONG GIAO TIEP giua C (feature/preprocessing) va D, E (mo hinh).

C so huu file nay. GIU NGUYEN chu ky ham:

    X, y, X_test, test_ids, classes = build_features(train, test, version="v1")

  - X        : DataFrame so, KHONG NaN
  - y        : np.ndarray int 0..K-1 (nhan da ma hoa)
  - X_test   : DataFrame cung cot, cung thu tu cot voi X
  - test_ids : cot Id cua test
  - classes  : np.ndarray ten the loai (o day la 0..10), classes[i] <-> nhan i

Cac phien ban:
  v0 : chi cot so goc, dien median (baseline toi gian, KHONG xu ly duration lech don vi)
  v1 : v0 + sua don vi duration + co danh dau thieu + tan suat artist/track   <-- dung cai nay lam moc
  v2+: C phat trien tiep (xem PHAN_CONG.md muc C)

PHAT HIEN TU EDA (A, 06/11) - xem eda_notes trong PHAN_CONG.md:
  * Cot `duration_in min/ms` LAN DON VI: 14% dong < 1000 la don vi PHUT (~3.9), con lai la mili giay.
    Cac the loai 0, 3, 4, 7 ghi 100% bang phut -> co `duration_is_min` la dac trung rat manh.
  * Thieu du lieu CO QUY LUAT theo the loai (instrumentalness thieu 61% o lop 5, 0.2% o lop 7)
    -> luon giu co danh dau thieu (`*_missing`), dung chi dien roi thoi.
  * 63.8% artist trong test co xuat hien trong train.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
from common import encode_labels

ARTIST, TRACK, DURATION = "Artist Name", "Track Name", "duration_in min/ms"
MISSING_COLS = ["Popularity", "key", "instrumentalness"]


def build_features(train: pd.DataFrame, test: pd.DataFrame, version: str = "v1"):
    if version == "v0":
        return _v0_baseline(train, test)
    if version == "v1":
        return _v1(train, test)
    raise ValueError(f"Chua co phien ban '{version}'. C them vao day.")


def _prepare(train, test):
    id_col, label_col = common.ID_COL, common.LABEL_COL
    y, classes = encode_labels(train[label_col])
    test_ids = test[id_col].values if id_col in test.columns else np.arange(len(test))
    drop = [c for c in (id_col, label_col) if c in train.columns]
    return y, classes, test_ids, drop


def _v0_baseline(train: pd.DataFrame, test: pd.DataFrame):
    y, classes, test_ids, drop = _prepare(train, test)
    num_cols = [c for c in train.drop(columns=drop).select_dtypes(include=[np.number]).columns if c in test.columns]
    X, X_test = train[num_cols].copy(), test[num_cols].copy()
    med = X.median()                       # thong ke chi tinh tren train
    X, X_test = X.fillna(med), X_test.fillna(med)
    assert list(X.columns) == list(X_test.columns) and not X.isna().any().any()
    return X.reset_index(drop=True), y, X_test.reset_index(drop=True), test_ids, classes


def _v1(train: pd.DataFrame, test: pd.DataFrame):
    y, classes, test_ids, drop = _prepare(train, test)
    full = pd.concat([train.drop(columns=drop), test.drop(columns=[c for c in drop if c in test.columns])],
                     axis=0, ignore_index=True)
    n = len(train)

    # 1) duration: lan don vi phut/ms
    dur = full[DURATION].astype(float)
    is_min = (dur < 1000).astype(float)
    full["duration_is_min"] = is_min
    full["duration_ms"] = np.where(dur < 1000, dur * 60_000.0, dur)
    full = full.drop(columns=[DURATION])

    # 2) co danh dau thieu (truoc khi dien)
    for c in MISSING_COLS:
        full[f"{c}_missing"] = full[c].isna().astype(float)

    # 3) tan suat artist/track (khong dung nhan -> khong ro ri). Tinh tren train+test.
    full["artist_count"] = full.groupby(ARTIST)[ARTIST].transform("size").astype(float)
    full["track_count"] = full.groupby(TRACK)[TRACK].transform("size").astype(float)
    full["artist_in_train"] = full[ARTIST].isin(set(train[ARTIST])).astype(float)  # co artist trong train

    full = full.drop(columns=[ARTIST, TRACK])

    # 4) dien thieu bang median TINH TREN TRAIN
    num = full.select_dtypes(include=[np.number]).columns
    med = full.iloc[:n][num].median()
    full[num] = full[num].fillna(med)

    X = full.iloc[:n].reset_index(drop=True)
    X_test = full.iloc[n:].reset_index(drop=True)
    assert list(X.columns) == list(X_test.columns) and not X.isna().any().any() and not X_test.isna().any().any()
    return X, y, X_test, test_ids, classes

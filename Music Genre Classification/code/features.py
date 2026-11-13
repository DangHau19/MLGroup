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
    if version == "v2":
        return _v2(train, test)
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


def _v2(train: pd.DataFrame, test: pd.DataFrame):
    """
    Feature Engineering v2.
    Ke thua v1 va bo sung cac dac trung moi.
    """

    X, y, X_test, test_ids, classes = _v1(train, test)

    def add_features(df):
        df = df.copy()

        # 1. Log transformation
        for col in [
            "speechiness",
            "instrumentalness",
            "liveness",
            "acousticness"
        ]:
            if col in df.columns:
                df[col + "_log1p"] = np.log1p(
                    df[col].clip(lower=0)
                )

        # 2. Loudness clipping
        if "loudness" in df.columns:
            df["loudness_clipped"] = df["loudness"].clip(
                lower=-40,
                upper=5
            )

        # 3. Interaction features
        if {"energy", "acousticness"}.issubset(df.columns):
            df["energy_acousticness_ratio"] = (
                df["energy"] /
                (df["acousticness"].clip(lower=0) + 0.01)
            )

        if {"danceability", "valence"}.issubset(df.columns):
            df["danceability_x_valence"] = (
                df["danceability"] * df["valence"]
            )

        if {"loudness", "energy"}.issubset(df.columns):
            df["loudness_x_energy"] = (
                df["loudness"] * df["energy"]
            )

        # 4. Key + Mode
        if {"key", "mode"}.issubset(df.columns):
            key = df["key"].round().clip(0, 11).astype(int)
            mode = df["mode"].round().clip(0, 1).astype(int)

            df["key_mode_24"] = key + 12 * mode

            if "key_missing" in df.columns:
                df.loc[
                    df["key_missing"] == 1,
                    "key_mode_24"
                ] = -1

        # 5. Tempo features
        if "tempo" in df.columns:
            df["tempo_rounded_10"] = (
                np.round(df["tempo"] / 10) * 10
            )

            df["tempo_group_20"] = np.floor(
                df["tempo"].clip(lower=0) / 20
            )

        return df.replace([np.inf, -np.inf], np.nan)

    X = add_features(X)
    X_test = add_features(X_test)

    # Fill missing values
    medians = X.median().fillna(0)

    X = X.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)

    # Validate output
    assert list(X.columns) == list(X_test.columns)
    assert not X.isna().any().any()
    assert not X_test.isna().any().any()

    return X, y, X_test, test_ids, classes

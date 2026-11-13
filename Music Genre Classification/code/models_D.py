"""Fold-local adapter: keep C's features and A's CV API unchanged."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold

import common
from features import build_features, ARTIST, TRACK
from artist_target_encoder_C import ArtistTargetEncoder


class FoldFeatureClassifier(ClassifierMixin, BaseEstimator):
    """Accept raw unlabeled rows through common.cross_validate_model.

    Only label-free frequency counts use reference (train+test). Median,
    artist membership, scaling, model fitting and supervised encoding use
    the current training fold. Inner-OOF artist encoding is used for fit.
    """

    def __init__(self, estimator, columns, reference, version="v1", target_encoding=False,
                 smoothing=20.0):
        self.estimator = estimator
        self.columns = columns
        self.reference = reference
        self.version = version
        self.target_encoding = target_encoding
        self.smoothing = smoothing

    def _frame(self, X):
        frame = pd.DataFrame(X, columns=self.columns).copy()
        for col in frame.columns:
            if col not in (ARTIST, TRACK):
                frame[col] = pd.to_numeric(frame[col])
        return frame.reset_index(drop=True)

    def _counts(self, X, raw):
        X = X.copy()
        if self.version != "v0":
            for col, name in ((ARTIST, "artist_count"), (TRACK, "track_count")):
                counts = self.reference[col].value_counts(dropna=False)
                X[name] = raw[col].map(counts).fillna(0).to_numpy(dtype=float)
        return X

    def fit(self, X, y):
        self.train_ = self._frame(X)
        self.classes_ = np.unique(y)
        self.train_[common.LABEL_COL] = np.asarray(y, dtype=int)
        features, _, _, _, _ = build_features(self.train_, self.reference, self.version)
        features = self._counts(features, self.train_)
        if self.target_encoding:
            artists = self.train_[[ARTIST]]
            inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=common.SEED)
            te = np.empty((len(y), len(self.classes_)))
            for tr, va in inner.split(artists, y):
                enc = ArtistTargetEncoder(smoothing=self.smoothing).fit(artists.iloc[tr], np.asarray(y)[tr])
                te[va] = enc.transform(artists.iloc[va]).to_numpy()
            self.encoder_ = ArtistTargetEncoder(smoothing=self.smoothing).fit(artists, y)
            features = np.column_stack([features, te])
        self.model_ = clone(self.estimator).fit(np.asarray(features, dtype=float), y)
        return self

    def predict_proba(self, X):
        raw = self._frame(X)
        _, _, features, _, _ = build_features(self.train_, raw, self.version)
        features = self._counts(features, raw)
        if self.target_encoding:
            te = self.encoder_.transform(raw[[ARTIST]]).to_numpy()
            features = np.column_stack([features, te])
        return self.model_.predict_proba(np.asarray(features, dtype=float))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

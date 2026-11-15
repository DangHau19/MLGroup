
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold


class ArtistTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding theo Artist Name.

    fit_transform tao inner OOF cho du lieu train; transform dung mapping
    fit tren toan bo outer train. Nghe si chua biet nhan prior tu train.
    """

    def __init__(self, artist_col="Artist Name", smoothing=20.0,
                 n_splits=4, random_state=42):
        self.artist_col = artist_col
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = np.asarray(y, dtype=int)

        if self.artist_col not in X.columns:
            raise ValueError("Khong tim thay cot Artist Name")
        if len(X) != len(y):
            raise ValueError("X va y phai co cung so dong")

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        artists = X[self.artist_col].fillna("__MISSING__").astype(str)

        self.prior_ = np.array([
            np.mean(y == cls) for cls in self.classes_
        ])

        target = pd.DataFrame({
            "artist": artists.to_numpy(),
            "target": y
        })

        counts = target.groupby("artist").size()

        class_counts = pd.crosstab(
            target["artist"],
            target["target"]
        ).reindex(
            columns=self.classes_,
            fill_value=0
        )

        self.mapping_ = (
            class_counts.add(
                self.smoothing * self.prior_,
                axis=1
            ).div(
                counts + self.smoothing,
                axis=0
            )
        )

        return self

    def fit_transform(self, X, y=None):
        if y is None:
            raise ValueError("ArtistTargetEncoder can y de tao inner OOF")

        X = pd.DataFrame(X).copy().reset_index(drop=True)
        y = np.asarray(y, dtype=int)
        if len(X) != len(y):
            raise ValueError("X va y phai co cung so dong")

        self.fit(X, y)
        columns = [f"artist_te_{cls}" for cls in self.classes_]
        encoded = np.zeros((len(X), len(columns)), dtype=float)
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        for train_idx, valid_idx in cv.split(X, y):
            encoder = ArtistTargetEncoder(
                artist_col=self.artist_col,
                smoothing=self.smoothing,
                n_splits=self.n_splits,
                random_state=self.random_state
            )
            encoder.fit(X.iloc[train_idx], y[train_idx])
            fold_encoded = encoder.transform(X.iloc[valid_idx])
            fold_encoded = fold_encoded.reindex(columns=columns, fill_value=0.0)
            encoded[valid_idx] = fold_encoded.to_numpy(dtype=float)

        result = X.drop(columns=[self.artist_col]).copy()
        for i, column in enumerate(columns):
            result[column] = encoded[:, i]
        return result

    def transform(self, X):
        if not hasattr(self, "mapping_"):
            raise ValueError("ArtistTargetEncoder chua duoc fit")

        X = pd.DataFrame(X).copy()
        if self.artist_col not in X.columns:
            raise ValueError("Khong tim thay cot Artist Name")

        artists = X[self.artist_col].fillna("__MISSING__").astype(str)

        encoded = self.mapping_.reindex(
            artists.to_numpy()
       ).to_numpy(dtype=float, copy=True)

        missing = np.isnan(encoded).any(axis=1)
        encoded[missing] = self.prior_

        result = X.drop(columns=[self.artist_col]).reset_index(drop=True)

        for i, cls in enumerate(self.classes_):
            result[f"artist_te_{cls}"] = encoded[:, i]

        return result

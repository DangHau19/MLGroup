
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class ArtistTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding theo Artist Name.

    Chi hoc thong ke nhan tu du lieu trong fit().
    Nghe si chua biet se nhan prior cua cac lop.
    """

    def __init__(self, artist_col="Artist Name", smoothing=20.0):
        self.artist_col = artist_col
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = np.asarray(y, dtype=int)

        if self.artist_col not in X.columns:
            raise ValueError("Khong tim thay cot Artist Name")

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
    

    def transform(self, X):
        X = pd.DataFrame(X).copy()

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

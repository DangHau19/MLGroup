#Xử lý mất cân bằng lớp: class_weight vs SMOTE.#


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors

from src import config
from src.preprocessing import build_preprocessor

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:  # pragma: no cover
    HAS_IMBLEARN = False


class SimpleSMOTE:
    """SMOTE tối giản (nội suy giữa mẫu và láng giềng cùng lớp) để dự phòng khi thiếu imblearn."""

    def __init__(self, k_neighbors=5, random_state=config.RANDOM_STATE):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        n_max = counts.max()
        Xs, ys = [X], [y]
        for c, n in zip(classes, counts):
            need = n_max - n
            if need <= 0:
                continue
            Xc = X[y == c]
            k = min(self.k_neighbors, len(Xc) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(Xc)
            neigh = nn.kneighbors(Xc, return_distance=False)[:, 1:]
            base = rng.randint(0, len(Xc), need)
            nb = neigh[base, rng.randint(0, k, need)]
            gap = rng.rand(need, 1)
            Xs.append(Xc[base] + gap * (Xc[nb] - Xc[base]))
            ys.append(np.full(need, c))
        return np.vstack(Xs), np.concatenate(ys)


class _SmotePipeline(BaseEstimator, ClassifierMixin):
    """Bản dự phòng của imblearn Pipeline: tiền xử lý -> SMOTE (chỉ khi fit) -> mô hình."""

    def __init__(self, prep=None, clf=None):
        self.prep = prep
        self.clf = clf

    def fit(self, X, y):
        self.prep_ = clone(self.prep)
        Xt = self.prep_.fit_transform(X, y)
        Xr, yr = SimpleSMOTE().fit_resample(Xt, y)
        self.clf_ = clone(self.clf).fit(Xr, yr)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        return self.clf_.predict(self.prep_.transform(X))

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.prep_.transform(X))


def build_smote_pipeline(estimator, feature_set="C", scale=True):
    """Pipeline: tiền xử lý -> SMOTE (chỉ trên fold train) -> mô hình."""
    prep = build_preprocessor(feature_set, scale=scale)
    if HAS_IMBLEARN:
        return ImbPipeline([("prep", prep),
                            ("smote", SMOTE(random_state=config.RANDOM_STATE)),
                            ("clf", estimator)])
    return _SmotePipeline(prep=prep, clf=estimator)

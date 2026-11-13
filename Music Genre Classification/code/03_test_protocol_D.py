"""Meaningful integration checks for fold-local preprocessing and OOF alignment."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier

import common
from features import ARTIST, TRACK, build_features
from models_D import FoldFeatureClassifier


class CaptureClassifier(DummyClassifier):
    def fit(self, X, y, sample_weight=None):
        self.seen_ = np.asarray(X).copy()
        return super().fit(X, y, sample_weight=sample_weight)


def main():
    train, test, _ = common.load_data()
    columns = tuple(c for c in test if c != common.ID_COL)
    raw = train.loc[:, columns].iloc[:400].copy()
    y = train[common.LABEL_COL].iloc[:400].to_numpy()
    reference = pd.concat([train.loc[:, columns], test.loc[:, columns]], ignore_index=True)
    adapter = FoldFeatureClassifier(CaptureClassifier(), columns, reference, "v2", True)
    model = clone(adapter).fit(raw.to_numpy(), y)
    # A singleton artist's own label must not enter its inner-OOF encoding.
    counts = raw[ARTIST].value_counts()
    singleton = np.flatnonzero(raw[ARTIST].map(counts).to_numpy() == 1)[0]
    from sklearn.model_selection import StratifiedKFold
    for tr, va in StratifiedKFold(4, shuffle=True, random_state=42).split(raw, y):
        if singleton in va:
            prior = np.bincount(y[tr], minlength=11) / len(tr)
            np.testing.assert_allclose(model.model_.seen_[singleton, -11:], prior)
    # Non-label reference data cannot affect the fitted median.
    changed = reference.copy()
    changed["Popularity"] = 1e9
    plain = FoldFeatureClassifier(CaptureClassifier(), columns, reference, "v1").fit(raw, y)
    alternate = FoldFeatureClassifier(CaptureClassifier(), columns, changed, "v1").fit(raw, y)
    np.testing.assert_allclose(plain.model_.seen_, alternate.model_.seen_)
    expected, _, _, _, _ = build_features(train.iloc[:400], test.iloc[:20], "v1")
    pop_col = list(expected.columns).index("Popularity")
    np.testing.assert_allclose(plain.model_.seen_[:, pop_col], expected["Popularity"])
    # Prediction order and fitted data must not change when the request is reordered.
    before = model.model_.seen_.copy()
    p = model.predict_proba(test.loc[:19, columns])
    np.testing.assert_allclose(p[::-1], model.predict_proba(test.loc[:19, columns].iloc[::-1]))
    np.testing.assert_array_equal(before, model.model_.seen_)
    full_y, _ = common.encode_labels(train[common.LABEL_COL])
    coverage = np.zeros(len(full_y), dtype=int)
    for tr, va in common.get_folds(full_y):
        assert not np.intersect1d(tr, va).size
        coverage[va] += 1
    assert np.all(coverage == 1)
    print("PASS: fold-local median, singleton target leakage, cloning, row order, OOF coverage")


if __name__ == "__main__":
    main()

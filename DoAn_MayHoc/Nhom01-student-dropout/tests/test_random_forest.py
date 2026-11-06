"""Integration checks on training subsets; hold-out test is never scored."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src import config
from src.data import get_train_test
from src.models.random_forest import (
    build_random_forest, compute_learning_curve, tune_random_forest,
)


class RandomForestTests(unittest.TestCase):
    def test_feature_sets_and_unknown_category(self):
        for feature_set in config.FEATURE_SETS:
            with self.subTest(feature_set=feature_set):
                X, X_test, y, _ = get_train_test(feature_set)
                self.assertFalse(set(X.index) & set(X_test.index))
                X, _, y, _ = train_test_split(
                    X, y, train_size=180, stratify=y, random_state=42)
                pipe = build_random_forest(feature_set)
                pipe.set_params(clf__n_estimators=5).fit(X, y)
                unseen = X.iloc[:3].copy()
                unseen[config.CATEGORICAL_COLS[0]] = 99999
                probabilities = pipe.predict_proba(unseen)
                self.assertEqual(probabilities.shape, (3, 3))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                self.assertEqual(pipe.named_steps["clf"].n_jobs, 1)

    def test_search_curve_and_model_roundtrip(self):
        X, _, y, _ = get_train_test()
        X, _, y, _ = train_test_split(X, y, train_size=180, stratify=y, random_state=42)
        with patch("src.models.random_forest.PARAM_DISTRIBUTIONS", {
            "clf__n_estimators": [5], "clf__max_depth": [3],
        }):
            search = tune_random_forest(X, y, n_iter=1, n_jobs=1)
        self.assertEqual(search.n_splits_, config.N_SPLITS)
        self.assertEqual(search.scoring, "f1_macro")
        curve = compute_learning_curve(search.best_estimator_, X, y, n_jobs=1)
        self.assertEqual(len(curve), 5)
        self.assertTrue(np.isfinite(curve.to_numpy()).all())
        self.assertTrue(curve.train_size.is_monotonic_increasing)
        self.assertLess(curve.train_size.max(), len(X))
        self.assertTrue(curve.validation_f1_macro_mean.between(0, 1).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rf.joblib"
            joblib.dump(search.best_estimator_, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(loaded.predict(X), search.best_estimator_.predict(X))
            np.testing.assert_allclose(loaded.predict_proba(X), search.best_estimator_.predict_proba(X))


if __name__ == "__main__":
    unittest.main()

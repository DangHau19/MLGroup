"""Regression checks for data loading, fold-local MI and shared result logging."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import config
from src.data import clean_columns, get_train_test, load_data
from src.evaluate import log_result
from src.features import CurricularFeatures, engineered_columns, safe_ratio
from src.preprocessing import build_pipeline, get_feature_names, mutual_information_scores


class DataTests(unittest.TestCase):
    def test_source_aliases_and_string_paths(self):
        original = load_data()
        source = original.rename(columns={
            "Marital Status": " Marital\t status ",
            "Nationality": "Nacionality",
            "Application mode": "Application  mode",
        })
        with tempfile.TemporaryDirectory() as directory:
            for sep in (";", ","):
                path = Path(directory) / "data.csv"
                source.to_csv(path, sep=sep, index=False, encoding="utf-8-sig")
                pd.testing.assert_frame_equal(load_data(str(path)), original)

    def test_colliding_names_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "trùng"):
            clean_columns(pd.DataFrame(columns=["Nationality", "Nacionality"]))

    def test_shared_split_and_timepoints(self):
        reference = None
        for fs, size in (("A", 24), ("B", 30), ("C", 36)):
            train, test, y_train, y_test = get_train_test(fs)
            self.assertEqual(train.shape, (3539, size))
            self.assertEqual(len(test), 885)
            self.assertFalse(set(train.index) & set(test.index))
            indices = (train.index.tolist(), test.index.tolist())
            if reference is None:
                reference = indices
            self.assertEqual(indices, reference)
            transformed = CurricularFeatures(fs).fit_transform(train)
            added = set(transformed.columns) - set(train.columns)
            self.assertEqual(added, set(engineered_columns(fs)))


class PreprocessingTests(unittest.TestCase):
    def test_mi_uses_discrete_mask(self):
        X = pd.DataFrame({
            "Course_33": [0., 1., 0.], "Gender": [1., 0., 1.],
            "Admission grade": [100., 120., 130.],
            "Pass rate 1st sem": [0., .5, 1.],
        })
        with patch("src.preprocessing.mutual_info_classif", return_value=np.ones(4)) as score:
            mutual_information_scores(X, [0, 1, 0])
        np.testing.assert_array_equal(
            score.call_args.kwargs["discrete_features"], [True, True, False, False])

    def test_selection_fit_and_feature_names(self):
        X, _, y, _ = get_train_test("B")
        X = X.groupby(y, group_keys=False).head(40)
        y = y.loc[X.index]
        pipe = build_pipeline(LogisticRegression(max_iter=1000), "B", select_k=10)
        pipe.fit(X, y)
        self.assertEqual(len(get_feature_names(pipe)), 10)
        self.assertTrue(np.isfinite(pipe.predict_proba(X)).all())
        for k in (0, -1, True):
            with self.assertRaises(ValueError):
                build_pipeline(LogisticRegression(), select_k=k)

    def test_zero_denominator_and_invalid_timepoint(self):
        result = safe_ratio(pd.Series([2, 0], index=[7, 3]), pd.Series([4, 0], index=[7, 3]))
        pd.testing.assert_series_equal(result, pd.Series([.5, 0.], index=[7, 3]))
        with self.assertRaises(KeyError):
            engineered_columns("D")


class LoggingTests(unittest.TestCase):
    def test_reordered_and_extra_columns_and_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.touch()
            log_result({"model": "RF", "f1_macro": .7}, str(path))
            log_result({"f1_macro": .6, "model": "LR", "accuracy": .8}, path)
            result = pd.read_csv(path)
            self.assertEqual(result.model.tolist(), ["RF", "LR"])
            self.assertEqual(result.f1_macro.tolist(), [.7, .6])
            self.assertTrue(pd.isna(result.loc[0, "accuracy"]))
            self.assertEqual(result.loc[1, "accuracy"], .8)


if __name__ == "__main__":
    unittest.main()

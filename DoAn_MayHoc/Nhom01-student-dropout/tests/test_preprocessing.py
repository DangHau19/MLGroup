"""Kiểm thử tiền xử lý và đặc trưng (SV3). Chạy: python -X utf8 -m unittest discover -s tests -v"""
import unittest

import numpy as np
import pandas as pd

from src import config
from src.data import get_train_test
from src.features import CurricularFeatures, PASS_RATE_1, PASS_RATE_2, GRADE_CHANGE, safe_ratio
from src.preprocessing import build_preprocessor, get_column_groups


class TestFeatures(unittest.TestCase):
    def test_safe_ratio_zero_division(self):
        out = safe_ratio(pd.Series([3, 0, 5]), pd.Series([6, 0, 0]))
        self.assertEqual(out.tolist(), [0.5, 0.0, 0.0])

    def test_curricular_features_values(self):
        X, _, _, _ = get_train_test("C")
        row = X.iloc[[0]].copy()
        row["Curricular units 1st sem (enrolled)"] = 0          # ca biên: chia 0
        row["Curricular units 1st sem (approved)"] = 0
        row["Curricular units 2nd sem (enrolled)"] = 6
        row["Curricular units 2nd sem (approved)"] = 3
        row["Curricular units 1st sem (grade)"] = 10.0
        row["Curricular units 2nd sem (grade)"] = 13.0
        out = CurricularFeatures("C").fit_transform(row)
        self.assertEqual(out[PASS_RATE_1].iloc[0], 0.0)
        self.assertEqual(out[PASS_RATE_2].iloc[0], 0.5)
        self.assertEqual(out[GRADE_CHANGE].iloc[0], 3.0)

    def test_feature_set_A_adds_nothing(self):
        X, _, _, _ = get_train_test("A")
        out = CurricularFeatures("A").fit_transform(X)
        self.assertEqual(list(out.columns), list(X.columns))


class TestPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.Xte, cls.y, cls.yte = get_train_test("C")

    def test_all_feature_sets_no_nan(self):
        for fs in "ABC":
            X, _, y, _ = get_train_test(fs)
            out = build_preprocessor(fs, scale=True).fit_transform(X, y)
            self.assertEqual(len(out), len(X))
            self.assertFalse(out.isna().any().any(), fs)
            self.assertTrue(np.isfinite(out.to_numpy(dtype=float)).all(), fs)

    def test_unseen_category_does_not_fail(self):
        pre = build_preprocessor("C", scale=True).fit(self.X, self.y)
        row = self.X.iloc[[0]].copy()
        row["Course"] = 99999
        row["Nationality"] = 99999
        out = pre.transform(row)
        self.assertEqual(out.shape[0], 1)
        self.assertFalse(out.isna().any().any())

    def test_division_by_zero_row_is_finite(self):
        pre = build_preprocessor("C", scale=True).fit(self.X, self.y)
        row = self.X.iloc[[0]].copy()
        for c in ["Curricular units 1st sem (enrolled)", "Curricular units 2nd sem (enrolled)"]:
            row[c] = 0
        out = pre.transform(row)
        self.assertTrue(np.isfinite(out.to_numpy(dtype=float)).all())

    def test_scaler_fitted_on_train_only(self):
        pre = build_preprocessor("C", scale=True).fit(self.X, self.y)
        feat = pre.named_steps["features"].transform(self.X)
        _, _, num = get_column_groups("C")
        scaler = pre.named_steps["encode"].named_transformers_["num"]
        np.testing.assert_allclose(scaler.mean_, feat[num].astype(float).mean().to_numpy(), rtol=1e-6)

    def test_scale_false_keeps_numeric_raw(self):
        pre = build_preprocessor("C", scale=False).fit(self.X, self.y)
        out = pre.transform(self.X)
        np.testing.assert_allclose(out["Admission grade"].to_numpy(dtype=float),
                                   self.X["Admission grade"].to_numpy(dtype=float))

    def test_rare_codes_grouped(self):
        pre = build_preprocessor("C", scale=False).fit(self.X, self.y)
        names = pre.named_steps["encode"].get_feature_names_out()
        self.assertTrue(any("infrequent" in n for n in names))


if __name__ == "__main__":
    unittest.main()

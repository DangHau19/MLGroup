"""Kiểm thử hàm dự đoán dùng chung. Cần models/*.joblib và models/model_metadata.json."""
import unittest

import numpy as np
import pandas as pd

from src import config
from src.data import get_train_test
from src.predict import InputError, load_model, predict_one, validate_input


class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _, cls.y, _ = get_train_test("C")      # mẫu lấy từ TRAIN, không dùng test
        cls.row = X.iloc[5].to_dict()

    def test_valid_record_all_models(self):
        for m in ("lr", "dt"):
            out = predict_one(self.row, model=m)
            self.assertIn(out["label"], config.CLASSES)
            self.assertEqual(list(out["probabilities"]), config.CLASSES)
            self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0, places=6)
            self.assertEqual(out["warnings"], [])

    def test_classes_order_matches_config(self):
        for m in ("lr", "dt"):
            self.assertEqual(list(load_model(m).classes_), config.CLASSES)

    def test_label_is_argmax_by_default(self):
        out = predict_one(self.row)
        self.assertEqual(out["label"], max(out["probabilities"], key=out["probabilities"].get))

    def test_accepts_one_row_dataframe(self):
        out = predict_one(pd.DataFrame([self.row]))
        self.assertIn(out["label"], config.CLASSES)

    def test_two_row_dataframe_rejected(self):
        with self.assertRaises(InputError):
            predict_one(pd.DataFrame([self.row, self.row]))

    def test_missing_column(self):
        bad = dict(self.row); bad.pop("Debtor")
        with self.assertRaisesRegex(InputError, "Thiếu"):
            predict_one(bad)

    def test_wrong_type(self):
        bad = dict(self.row); bad["Admission grade"] = "abc"
        with self.assertRaises(InputError):
            predict_one(bad)

    def test_nan_rejected(self):
        bad = dict(self.row); bad["GDP"] = float("nan")
        with self.assertRaises(InputError):
            predict_one(bad)

    def test_binary_domain(self):
        bad = dict(self.row); bad["Gender"] = 2
        with self.assertRaises(InputError):
            predict_one(bad)

    def test_categorical_must_be_integer(self):
        bad = dict(self.row); bad["Course"] = 9500.5
        with self.assertRaises(InputError):
            predict_one(bad)

    def test_new_category_warns_but_predicts(self):
        odd = dict(self.row); odd["Course"] = 99999
        out = predict_one(odd)
        self.assertIn(out["label"], config.CLASSES)
        self.assertTrue(any("chưa từng thấy" in w for w in out["warnings"]))

    def test_out_of_range_numeric_warns(self):
        odd = dict(self.row); odd["Age at enrollment"] = 150
        out = predict_one(odd)
        self.assertTrue(any("ngoài khoảng" in w for w in out["warnings"]))

    def test_extra_column_ignored_with_warning(self):
        extra = dict(self.row); extra["foo"] = 1
        X, warns = validate_input(extra)
        self.assertNotIn("foo", X.columns)
        self.assertTrue(any("thừa" in w for w in warns))

    def test_dropout_threshold(self):
        self.assertEqual(predict_one(self.row, dropout_threshold=0.0)["label"], config.POSITIVE_CLASS)
        self.assertNotEqual(predict_one(self.row, dropout_threshold=1.01)["label"], config.POSITIVE_CLASS)

    def test_unknown_model_name(self):
        with self.assertRaises(KeyError):
            predict_one(self.row, model="unknown-model")


if __name__ == "__main__":
    unittest.main()

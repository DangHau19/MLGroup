import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from src import config
from src.data import get_train_test
from src.predict import InputError, MODEL_FILES, predict_one


class DemoModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads((config.MODELS_DIR / "demo" / "model_registry.json").read_text(encoding="utf-8"))

    def test_all_fifteen_model_feature_set_combinations_predict(self):
        for feature_set in ("A", "B", "C"):
            X_train, _, _, _ = get_train_test(feature_set)
            sample = X_train.iloc[0].to_dict()
            for model in MODEL_FILES:
                with self.subTest(feature_set=feature_set, model=model):
                    path = config.MODELS_DIR / "demo" / feature_set / f"{model}_best.joblib"
                    if feature_set == "C":
                        path = config.MODELS_DIR / MODEL_FILES[model]
                    self.assertTrue(path.is_file())
                    result = predict_one(sample, model=model, feature_set=feature_set)
                    self.assertIn(result["label"], config.CLASSES)
                    self.assertEqual(result["decision_rule"], "native_predict")
                    self.assertEqual(list(result["probabilities"]), config.CLASSES)
                    self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=7)

    def test_registry_covers_all_models_and_feature_sets(self):
        self.assertEqual(set(self.registry["models"]), set(MODEL_FILES))
        for model_key in MODEL_FILES:
            model_entries = self.registry["models"][model_key]
            self.assertEqual(set(model_entries), {"A", "B", "C"})
            for feature_set, entry in model_entries.items():
                path = config.ROOT / entry["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
                self.assertEqual(entry["feature_set"], feature_set)
                self.assertFalse(self.registry["holdout_evaluated"])

    def test_rejects_negative_admission_grade(self):
        X_train, _, _, _ = get_train_test("C")
        sample = X_train.iloc[0].to_dict()
        sample["Admission grade"] = -1
        with self.assertRaisesRegex(InputError, "Admission grade"):
            predict_one(sample, model="gb", feature_set="C")

    def test_rejects_missing_required_feature(self):
        X_train, _, _, _ = get_train_test("B")
        sample = X_train.iloc[0].to_dict()
        sample.pop("Curricular units 1st sem (approved)")
        with self.assertRaisesRegex(InputError, "Thiếu"):
            predict_one(sample, model="gb", feature_set="B")

    def test_threshold_is_optional_and_explicit(self):
        X_train, _, _, _ = get_train_test("C")
        sample = X_train.iloc[0].to_dict()
        result = predict_one(sample, model="gb", feature_set="C", dropout_threshold=0.45)
        self.assertEqual(result["decision_rule"], "dropout_threshold")
        self.assertEqual(result["threshold"], 0.45)


if __name__ == "__main__":
    unittest.main()

"""Decision-rule and holdout-gate tests use synthetic samples, never the real test set."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.sv1_final_evaluation import decide, evaluation_key, local_path, run


class FakeModel:
    classes_ = np.array(['Graduate', 'Dropout', 'Enrolled'])  # deliberately non-alphabetical
    def predict_proba(self, X):
        return np.array([[.5, .3, .2], [.2, .6, .2]])
    def predict(self, X):
        return np.array(['Graduate', 'Dropout'])


class FinalEvaluationTests(unittest.TestCase):
    def test_threshold_uses_labels_and_non_dropout_fallback(self):
        pred = decide(FakeModel(), [1, 2], {'kind': 'dropout_threshold', 'threshold': .7})
        np.testing.assert_array_equal(pred, ['Graduate', 'Graduate'])
        pred = decide(FakeModel(), [1, 2], {'kind': 'dropout_threshold', 'threshold': .3})
        np.testing.assert_array_equal(pred, ['Dropout', 'Dropout'])

    def test_native_and_invalid_rules(self):
        np.testing.assert_array_equal(decide(FakeModel(), [], {'kind': 'native_predict'}), ['Graduate', 'Dropout'])
        for rule in [{'kind': 'unknown'}, {'kind': 'dropout_threshold', 'threshold': 2}]:
            with self.assertRaises(ValueError):
                decide(FakeModel(), [], rule)

    def test_rename_cannot_bypass_identity_ledger(self):
        entry = {'name': 'RF', 'sha256': 'modelhash', 'feature_set': 'C', 'decision': {'kind': 'native_predict'}}
        self.assertEqual(evaluation_key(entry, 'data'), evaluation_key(dict(entry, name='RF_renamed'), 'data'))
        self.assertNotEqual(evaluation_key(entry, 'data'), evaluation_key(entry, 'other_data'))

    def test_paths_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with self.assertRaises(ValueError):
                local_path(root, '../outside')

    def test_blocked_preflight_never_scores_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / 'manifest.json'
            manifest.write_text('{}', encoding='utf-8')
            with patch('src.sv1_final_evaluation.preflight', return_value=(['Missing LR/DT'], [], 'data')):
                with patch('src.sv1_final_evaluation.evaluate_on_test') as evaluate:
                    self.assertEqual(run(manifest, execute=True), 2)
                    evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()

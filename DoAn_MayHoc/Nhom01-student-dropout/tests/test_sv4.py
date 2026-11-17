"""Test phần SV4: quy tắc ngưỡng, tham số GB/SVM khớp kết quả tuning, SMOTE chỉ áp trên fold train."""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import config
from src.models import proba as P
from src.threshold import apply_threshold, choose_threshold, predict_argmax, threshold_table

CLASSES = list(config.CLASSES)   # ['Dropout', 'Enrolled', 'Graduate']


class ThresholdRuleTests(unittest.TestCase):
    def test_classes_order_is_alphabetical(self):
        self.assertEqual(CLASSES, sorted(CLASSES))

    def test_dropout_if_prob_reaches_threshold(self):
        p = np.array([[0.40, 0.35, 0.25]])          # argmax = Dropout
        self.assertEqual(apply_threshold(p, CLASSES, 0.40)[0], "Dropout")   # bằng ngưỡng -> Dropout
        self.assertEqual(apply_threshold(p, CLASSES, 0.41)[0], "Enrolled")  # dưới ngưỡng -> argmax còn lại

    def test_below_threshold_ignores_dropout_even_if_argmax(self):
        p = np.array([[0.45, 0.10, 0.45]])
        self.assertEqual(apply_threshold(p, CLASSES, 0.60)[0], "Graduate")

    def test_extreme_thresholds(self):
        p = np.random.RandomState(0).dirichlet(np.ones(3), 50)
        self.assertTrue((apply_threshold(p, CLASSES, 0.0) == "Dropout").all())
        self.assertFalse((apply_threshold(p, CLASSES, 1.01) == "Dropout").any())

    def test_single_vector_and_shape_checks(self):
        self.assertEqual(apply_threshold([0.1, 0.2, 0.7], CLASSES, 0.5).shape, (1,))
        with self.assertRaises(ValueError):
            apply_threshold(np.ones((2, 2)), CLASSES, 0.5)

    def test_choose_threshold_keeps_recall_constraint(self):
        rng = np.random.RandomState(1)
        y = pd.Series(rng.choice(CLASSES, 400, p=[.3, .2, .5]))
        proba = pd.DataFrame(rng.dirichlet([2, 1, 2], 400), columns=CLASSES)
        tab = threshold_table(y, proba)
        t = choose_threshold(tab)
        base = tab[tab.threshold == "argmax"].iloc[0].recall_Dropout
        if t is not None:
            self.assertGreaterEqual(tab[tab.threshold == t].iloc[0].recall_Dropout, base - 1e-9)

    def test_argmax_matches_numpy(self):
        p = np.random.RandomState(2).dirichlet(np.ones(3), 20)
        self.assertEqual(list(predict_argmax(p, CLASSES)), [CLASSES[i] for i in p.argmax(1)])


class BestParamsMatchSearchTests(unittest.TestCase):
    """Hằng số trong src/models/proba.py phải trùng hạng 1 của tuning."""

    def _rank1(self, name):
        df = pd.read_csv(config.RESULTS_DIR / f"{name}_search_results.csv")
        return df.sort_values("rank_test_score").iloc[0]

    def test_gb(self):
        r = self._rank1("gb")
        self.assertEqual(int(r["param_clf__n_estimators"]), P.GB_BEST["n_estimators"])
        self.assertAlmostEqual(float(r["param_clf__learning_rate"]), P.GB_BEST["learning_rate"])
        self.assertEqual(int(r["param_clf__max_depth"]), P.GB_BEST["max_depth"])
        self.assertAlmostEqual(float(r["param_clf__subsample"]), P.GB_BEST["subsample"])

    def test_svm(self):
        r = self._rank1("svm")
        self.assertEqual(r["param_clf__kernel"], P.SVM_BEST["kernel"])
        self.assertAlmostEqual(float(r["param_clf__C"]), P.SVM_BEST["C"])
        self.assertAlmostEqual(float(r["param_clf__gamma"]), P.SVM_BEST["gamma"])
        self.assertEqual(r["param_clf__class_weight"], P.SVM_BEST["class_weight"])

    def test_svm_scaling_inside_pipeline_and_proba_flag(self):
        pipe = P.build_svm("C", probability=True)
        self.assertTrue(pipe.named_steps["clf"].probability)
        self.assertFalse(P.build_svm("C").named_steps["clf"].probability)
        enc = pipe.named_steps["prep"].named_steps["encode"]
        self.assertEqual(type(dict((n, t) for n, t, _ in enc.transformers)["num"]).__name__, "StandardScaler")


class SmoteOnlyOnTrainFoldTests(unittest.TestCase):
    """Chứng cứ chống leakage: bộ lấy mẫu chỉ thấy phần train của mỗi fold, không bao giờ thấy toàn bộ dữ liệu."""

    def test_sampler_sees_only_train_fold(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from src import imbalance
        from src.data import get_train_test
        from src.evaluate import get_cv

        X, _, y, _ = get_train_test("C")
        X, y = X.iloc[:600], y.iloc[:600]
        seen = []
        if imbalance.HAS_IMBLEARN:
            real = imbalance.SMOTE
            class Rec(real):
                def fit_resample(self, Xr, yr):
                    seen.append(len(Xr)); return super().fit_resample(Xr, yr)
            patcher = mock.patch.object(imbalance, "SMOTE", Rec)
        else:
            real = imbalance.SimpleSMOTE.fit_resample
            def rec(self, Xr, yr):
                seen.append(len(Xr)); return real(self, Xr, yr)
            patcher = mock.patch.object(imbalance.SimpleSMOTE, "fit_resample", rec)
        with patcher:
            pipe = imbalance.build_smote_pipeline(LogisticRegression(max_iter=2000), "C", True)
            pred = cross_val_predict(pipe, X, y, cv=get_cv())
        self.assertEqual(len(pred), len(X))                    # validation không bị nhân thêm mẫu
        self.assertEqual(len(seen), config.N_SPLITS)
        for n in seen:
            self.assertLess(n, len(X))                         # chỉ thấy phần train của fold
            self.assertAlmostEqual(n / len(X), 1 - 1 / config.N_SPLITS, delta=0.02)


if __name__ == "__main__":
    unittest.main()

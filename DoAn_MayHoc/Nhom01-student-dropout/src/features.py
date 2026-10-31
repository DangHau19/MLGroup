import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PASS_RATE_1 = "Pass rate 1st sem"
PASS_RATE_2 = "Pass rate 2nd sem"
GRADE_CHANGE = "Grade change 2nd vs 1st"


def engineered_columns(feature_set: str) -> list:
    """Tên các đặc trưng mới tương ứng với tập đặc trưng A/B/C."""
    if feature_set == "A":
        return []
    if feature_set == "B":
        return [PASS_RATE_1]
    return [PASS_RATE_1, PASS_RATE_2, GRADE_CHANGE]


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num/den, trả về 0 khi den = 0."""
    num = num.astype(float)
    den = den.astype(float)
    out = np.zeros(len(num), dtype=float)
    mask = den.to_numpy() != 0
    out[mask] = num.to_numpy()[mask] / den.to_numpy()[mask]
    return pd.Series(out, index=num.index)


class CurricularFeatures(BaseEstimator, TransformerMixin):
    """Thêm các đặc trưng tỷ lệ môn đạt và chênh lệch điểm TB giữa hai học kỳ."""

    def __init__(self, feature_set: str = "C"):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()
        fs = self.feature_set
        if fs in ("B", "C"):
            X[PASS_RATE_1] = safe_ratio(
                X["Curricular units 1st sem (approved)"],
                X["Curricular units 1st sem (enrolled)"],
            )
        if fs == "C":
            X[PASS_RATE_2] = safe_ratio(
                X["Curricular units 2nd sem (approved)"],
                X["Curricular units 2nd sem (enrolled)"],
            )
            X[GRADE_CHANGE] = (
                X["Curricular units 2nd sem (grade)"].astype(float)
                - X["Curricular units 1st sem (grade)"].astype(float)
            )
        return X

from functools import partial

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config
from src.features import CurricularFeatures, engineered_columns

MIN_FREQUENCY = 20 


def _discrete_mask(X):
    prefixes = tuple(f"{c}_" for c in config.CATEGORICAL_COLS)
    return np.array([
        c in config.BINARY_COLS or c.startswith(prefixes) for c in X.columns
    ])


def mutual_information_scores(X, y):
    """MI: One-Hot và nhị phân là rời rạc; các biến số còn lại là liên tục.

    X là DataFrame đầu ra của preprocessor. Tạo mask tại mỗi fold vì số cột
    One-Hot có thể thay đổi khi fit trên các tập train khác nhau.
    """
    return mutual_info_classif(
        X, y, discrete_features=_discrete_mask(X), random_state=config.RANDOM_STATE
    )


class MutualInformationSelector(SelectKBest):
    """Giữ mask theo tên trước khi SelectKBest chuyển DataFrame sang ndarray."""

    def __init__(self, k=10):
        super().__init__(score_func=mutual_information_scores, k=k)

    def fit(self, X, y):
        self.score_func = partial(
            mutual_info_classif, discrete_features=_discrete_mask(X),
            random_state=config.RANDOM_STATE,
        )
        return super().fit(X, y)


def get_column_groups(feature_set: str = "C", add_features: bool = True):
    """Trả về (cat_cols, bin_cols, num_cols) cho một tập đặc trưng."""
    cols = config.FEATURE_SETS[feature_set]
    cat = [c for c in config.CATEGORICAL_COLS if c in cols]
    binary = [c for c in config.BINARY_COLS if c in cols]
    num = [c for c in cols if c not in cat and c not in binary]
    if add_features:
        num = num + engineered_columns(feature_set)
    return cat, binary, num


def build_column_transformer(feature_set="C", scale=True, add_features=True,
                             min_frequency=MIN_FREQUENCY) -> ColumnTransformer:
    cat, binary, num = get_column_groups(feature_set, add_features)
    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency,
                        sparse_output=False)
    return ColumnTransformer(
        [
            ("cat", ohe, cat),
            ("num", StandardScaler() if scale else "passthrough", num),
            ("bin", "passthrough", binary),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_preprocessor(feature_set: str = "C", scale: bool = True,
                       add_features: bool = True, min_frequency: int = MIN_FREQUENCY):
    """Bộ tiền xử lý chưa fit, dùng làm bước đầu của Pipeline."""
    if feature_set not in config.FEATURE_SETS:
        raise KeyError(f"feature_set phải thuộc {list(config.FEATURE_SETS)}")
    ct = build_column_transformer(feature_set, scale, add_features, min_frequency)
    if not add_features:
        return ct
    return Pipeline([("features", CurricularFeatures(feature_set)), ("encode", ct)])


def build_pipeline(estimator, feature_set: str = "C", scale: bool = True,
                   add_features: bool = True, select_k: int = None,
                   min_frequency: int = MIN_FREQUENCY) -> Pipeline:
    """Pipeline = tiền xử lý [+ chọn đặc trưng MI] + mô hình."""
    steps = [("prep", build_preprocessor(feature_set, scale, add_features, min_frequency))]
    if select_k is not None:
        if isinstance(select_k, bool) or not isinstance(select_k, int) or select_k < 1:
            raise ValueError("select_k phải là số nguyên dương hoặc None")
        steps.append(("select", MutualInformationSelector(k=select_k)))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def get_feature_names(fitted_pipeline) -> list:
    """Tên đặc trưng sau tiền xử lý (và sau chọn đặc trưng nếu có) của pipeline đã fit."""
    prep = fitted_pipeline.named_steps["prep"]
    encoder = prep.named_steps["encode"] if isinstance(prep, Pipeline) else prep
    names = list(encoder.get_feature_names_out())
    if "select" in fitted_pipeline.named_steps:
        mask = fitted_pipeline.named_steps["select"].get_support()
        names = [n for n, m in zip(names, mask) if m]
    return names

"""Xây GB/SVM từ tham số tốt nhất đã tuning, có xác suất (predict_proba) cho bước chỉnh ngưỡng.

Vì sao có file này (Probability experiment, GB/SVM model build):
- `SVC` mặc định `probability=False` nên KHÔNG có predict_proba; muốn dùng ngưỡng phải bật
  `probability=True` (Platt scaling nội bộ, học bằng CV 5 fold trong lúc fit, chỉ dùng dữ liệu train).
- File .joblib phụ thuộc phiên bản scikit-learn đã lưu. `gb_best.joblib` lưu từ bản sklearn cũ
  không nạp được trên scikit-learn 1.9.1 (lỗi `No module named '_loss'`). Dựng lại pipeline từ
  tham số (file này) là cách tái lập chắc chắn; hằng số bên dưới được test đối chiếu với
  `notebooks/Experiments/Logs/Experiment_08_GB_SVM/*_search_results.csv` (xếp hạng 1).
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from src import config
from src.preprocessing import build_pipeline

# Hạng 1 của RandomizedSearchCV / GridSearchCV (scoring f1_macro, CV 5 fold trên train).
GB_BEST = dict(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.85)
SVM_BEST = dict(kernel="rbf", C=10, gamma=0.003, class_weight="balanced")


def build_gb(feature_set: str = "C"):
    """GB tốt nhất. GB hỗ trợ predict_proba sẵn; không cần chuẩn hóa."""
    return build_pipeline(
        GradientBoostingClassifier(random_state=config.RANDOM_STATE, **GB_BEST),
        feature_set, scale=False)


def build_svm(feature_set: str = "C", probability: bool = False):
    """SVM tốt nhất; chuẩn hóa nằm TRONG pipeline. probability=True để có predict_proba."""
    return build_pipeline(
        SVC(random_state=config.RANDOM_STATE, probability=probability, **SVM_BEST),
        feature_set, scale=True)

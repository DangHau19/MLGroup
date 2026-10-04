import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

def get_embedded_lasso_pipeline(model, random_state=42):
    """
    Kỹ thuật 1: Embedded Lasso
    Sử dụng LassoCV để tìm các hệ số, loại bỏ các feature có hệ số = 0.
    """
    # Khởi tạo LassoCV để tự động tìm alpha tốt nhất
    lasso_selector = SelectFromModel(LassoCV(cv=5, random_state=random_state, n_jobs=-1))
    
    # Tạo pipeline: Chọn feature -> Huấn luyện model
    pipeline = Pipeline([
        ('feature_selection', lasso_selector),
        ('model', model)
    ])
    return pipeline

def get_embedded_rf_pipeline(model, k_features=50, random_state=42):
    """
    Kỹ thuật 2: Embedded Random Forest
    Dựa vào feature_importances_ của RF để chọn top K feature.
    """
    # Khởi tạo RF để đánh giá tầm quan trọng
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    
    # Chọn top K feature quan trọng nhất (theo quy ước nhóm có thể quét k, mặc định k=50)
    rf_selector = SelectFromModel(rf, threshold=-np.inf, max_features=k_features)
    
    pipeline = Pipeline([
        ('feature_selection', rf_selector),
        ('model', model)
    ])
    return pipeline

# Code test thử ngay lập tức (không cần đợi TV2/TV3)
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.svm import SVR
    
    print("Đang tạo dữ liệu giả để test...")
    X, y = make_regression(n_samples=500, n_features=100, n_informative=20, random_state=42)
    
    print("Test Pipeline Embedded Lasso + SVR:")
    pipeline_lasso = get_embedded_lasso_pipeline(SVR())
    pipeline_lasso.fit(X, y)
    n_selected_lasso = pipeline_lasso.named_steps['feature_selection'].get_support().sum()
    print(f"-> Số feature Lasso giữ lại: {n_selected_lasso}")
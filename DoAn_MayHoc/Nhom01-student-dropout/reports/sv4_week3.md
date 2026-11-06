# SV4 - Kết quả tuần 3 (mất cân bằng, Gradient Boosting, SVM)

Tất cả số liệu là CV 5-fold trên tập train (feature set C, `random_state=42`). Tập test không được dùng.

## 1. Mất cân bằng lớp (issue #11, `notebooks/03_imbalance.ipynb`)
Xem `reports/imbalance_comparison.csv`. `class_weight="balanced"` cho Macro-F1 tốt nhất (LR 0,711; RF 0,725) và tăng F1 Enrolled mạnh nhất, nhưng Recall Dropout giảm nhẹ so với không xử lý (khoảng 0,775 xuống 0,74). Cần bước chỉnh ngưỡng (issue #14).

## 2. Gradient Boosting (issue #12, `src/models/tuning_sv4.py`)
RandomizedSearchCV 12 tổ hợp, scoring `f1_macro`. Tốt nhất: n_estimators=200, learning_rate=0,05, max_depth=4, subsample=0,85.
Macro-F1 val 0,719 ± 0,012, Recall Dropout 0,783, F1 Enrolled 0,493. Train Macro-F1 0,900, chênh lớn so với val nên mô hình overfit rõ. Lưu: `models/gb_best.joblib`.

## 3. SVM (issue #13)
GridSearchCV (linear: C, class_weight; RBF: C, gamma, class_weight), có StandardScaler. Tốt nhất: RBF, C=10, gamma=0,003, class_weight=balanced.
Macro-F1 val 0,721 ± 0,011, Recall Dropout 0,727, F1 Enrolled 0,540. Train 0,769, mức overfit nhẹ. Lưu: `models/svm_best.joblib`.

## 4. So sánh nhanh (val CV Macro-F1)
Baseline Dummy 0,222 / 0,338; LR 0,698 (không tuning); SVM 0,721; GB 0,719. Chênh lệch SVM và GB (0,002) nhỏ hơn độ lệch chuẩn nên chưa kết luận mô hình nào hơn.



# Bảng 3. So sánh CV accuracy các mô hình và ensemble — E

Accuracy là trung bình 5 fold (StratifiedKFold seed 42); std là độ lệch chuẩn giữa 5 fold. Với ensemble, điểm là **nested** (trọng số/meta-learner học trên 4 fold, chấm trên fold còn lại).

| Mô hình / chiến lược | CV accuracy | Std | Macro F1 | Log-loss |
|---|---:|---:|---:|---:|
| CatBoost (đơn) | 0.6548 | 0.0039 | 0.7007 | 0.8993 |
| LightGBM (đơn) | 0.6540 | 0.0072 | 0.7066 | 0.8994 |
| XGBoost (đơn) | 0.6511 | 0.0037 | 0.7028 | 0.9061 |
| HistGB (đơn) | 0.6415 | 0.0043 | 0.6925 | 0.9343 |
| MLP (đơn) | 0.6315 | 0.0058 | 0.6716 | 0.9472 |
| SVM (đơn) | 0.6291 | 0.0049 | 0.6660 | 0.9759 |
| ExtraTrees (đơn) | 0.6164 | 0.0050 | 0.6571 | 1.0439 |
| **Ensemble: Stacking: HistGradientBoosting nông** | **0.6626** | 0.0047 | 0.7100 | 0.8891 |
| Ensemble: Trọng số tối ưu accuracy (leo đồi) | 0.6581 | 0.0038 | 0.7070 | 0.8899 |
| Ensemble: Stacking: Logistic Regression | 0.6578 | 0.0012 | 0.7081 | 0.8882 |
| Ensemble: Trung bình đều 3 boosting (CatBoost, LightGBM, XGBoost) | 0.6575 | 0.0035 | 0.7083 | 0.8911 |
| Ensemble: Chọn tham lam có lặp (Caruana) | 0.6571 | 0.0038 | 0.7062 | 0.8903 |
| Ensemble: Trọng số tối ưu log-loss | 0.6568 | 0.0051 | 0.7065 | 0.8862 |
| Ensemble: Trung bình đều 7 mô hình | 0.6503 | 0.0026 | 0.6983 | 0.9044 |
| Ensemble: Trung bình hình học 7 mô hình | 0.6491 | 0.0013 | 0.6960 | 0.8926 |

Mốc baseline của A (HistGB v1): 0,5585. Oracle (ít nhất một trong 7 mô hình nền đúng): 0.7679.

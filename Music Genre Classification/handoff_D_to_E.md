# Bàn giao D → E

Trần Đình Bảo Quốc — 3124411252. Dữ liệu và seed theo quy ước nhóm. Đã chạy xong baseline, boosting, tuning và v2 theo đúng thứ tự phân công. Bàn giao **7 họ mô hình**, gồm LightGBM, XGBoost, CatBoost, HistGB, MLP, SVM, ExtraTrees.

| Tag dùng với common.load_oof | Accuracy | Std (5 folds) | Macro F1 |
| --- | --- | --- | --- |
| `catboost_v2_te_selected_D` | 0.654765 | 0.003912 | 0.700522 |
| `lightgbm_v2_te_selected_D` | 0.654001 | 0.007181 | 0.706566 |
| `xgboost_v2_te_selected_D` | 0.651084 | 0.003745 | 0.702729 |
| `histgb_v2_te_selected_D` | 0.641498 | 0.004274 | 0.692461 |
| `mlp_v2_te_selected_D` | 0.631495 | 0.005800 | 0.671531 |
| `svm_v2_te_selected_D` | 0.629063 | 0.004937 | 0.665659 |
| `extratrees_v2_te_selected_D` | 0.616351 | 0.005037 | 0.656789 |

Mỗi tag có đủ `evidence/cv_logs/<tag>_oof.npy`, `<tag>_test.npy`, `<tag>_cv.json`. OOF (14396,11) giữ nguyên thứ tự train; test (3600,11) giữ nguyên thứ tự test; cột xác suất tương ứng `classes=[0,1,...,10]`. Test là trung bình xác suất năm outer models. Tất cả tag trên dùng v2 + TE và cùng outer folds seed 42. Không trộn thứ tự dòng hoặc tự mã hóa lại nhãn.

```python
import sys
sys.path.insert(0, 'code')  # chạy tại thư mục dự án
from common import load_oof
oof, test_proba = load_oof('catboost_v2_te_selected_D')
```

`evidence/models_D/` có:

- `selected_models_D.csv`: bảy mô hình bàn giao; `all_results_D.csv`: toàn bộ 32 cấu hình (có cả baseline và trial, không coi các trial là kết quả đánh giá độc lập).
- `y_encoded_D.npy`, `classes_D.npy`, `train_ids_D.npy`, `test_ids_D.npy`, `fold_ids_D.npy` (fold 0–4), để kiểm tra alignment trước ensemble.
- `handoff_manifest_D.json`: SHA-256 dữ liệu, mã nguồn, ba file của mỗi mô hình; phiên bản thư viện và thứ tự lớp.
- `hyperparameters_D.csv`, các `*_trials.csv`, `selected_v1.json`, `tuning.sqlite3` cho toàn bộ quá trình chọn cấu hình.
- `intermediate_D.zip`: các bộ kết quả không được chọn và đầu ra baseline tái lập. Các tệp này được nén gọn; notebook/audit đọc trực tiếp, không cần giải nén. Chỉ 21 tệp kết quả của bảy mô hình bàn giao được giữ rời để `common.load_oof` dùng ngay.

Mô hình đơn tốt nhất: **catboost_v2_te_selected_D, accuracy 0.654765**. Đây là CV sau lựa chọn trên cùng folds, chưa phải đánh giá độc lập sau tuning; không khẳng định ensemble sẽ cải thiện. Không có điểm leaderboard mới cho mô hình D. Không dùng public score để chọn. Tần suất train+test không nhãn được dùng theo quy ước nhóm; TE luôn trong fold và nested OOF ở phần train. So với baseline A 0,5585, mô hình tốt nhất tăng 9.63 điểm phần trăm.

Chạy tái lập: xem phần D trong README và `code/03_models_tuning_D.ipynb`. Mục 5 của paper: `paper/05_models_experimental_setup_D.md`; Bảng 2: `paper/table_02_hyperparameters_D.md`.

E cần tự kiểm chứng lựa chọn/trọng số ensemble bằng phương pháp tách OOF theo phân công. Không xem CV được tối ưu trên chính OOF là điểm đánh giá độc lập. Phần này là bàn giao file trong workspace, chưa gửi tin nhắn nhóm hoặc nộp Kaggle.

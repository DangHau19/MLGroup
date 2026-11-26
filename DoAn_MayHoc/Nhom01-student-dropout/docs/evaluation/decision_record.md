# Biên bản khóa cấu hình đánh giá cuối

**Trạng thái:** cấu hình được khóa trước khi chấm hold-out. **Phạm vi:** so sánh năm pipeline trên tập đặc trưng C. Hold-out chỉ chấm một lượt theo manifest đã commit.

## Giao thức thống nhất

- Dữ liệu: `data/raw/data.csv`, SHA-256 `edc197d1381f25414a3757ab9bbae5f1f6d21aa6a964dbe98a148c15f3d63036`.
- Chia phân tầng một lần: 80% train (3.539 dòng), 20% hold-out (885 dòng), seed 42.
- Chọn tham số/đánh giá trước test: Stratified 5-fold CV trên train, shuffle, seed 42; độ đo chính Macro-F1. A/B/C dùng chung các chỉ số chia.
- Môi trường: Python 3.12.10, NumPy 2.5.4, pandas 3.1.1, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0.
- Phạm vi chốt: tập C (36 biến gốc; các đặc trưng dẫn xuất được tạo trong pipeline), năm mô hình LR, DT, RF, GB, SVM.
- Quy tắc cho cả năm mô hình: `pipeline.predict()` (`native_predict`). Không chọn quy tắc dựa trên kết quả hold-out.

## Cấu hình mô hình

| Mã | Artifact | Tham số đã chốt |
|---|---|---|
| LR | `models/lr_best.joblib` | `C=0.3`, `class_weight=balanced`, `l1_ratio=0.0` (L2) |
| DT | `models/dt_best.joblib` | entropy, `max_depth=4`, `min_samples_leaf=10`, `ccp_alpha=0` |
| RF | `models/rf_best.joblib` | 100 cây, `max_features=sqrt`, `min_samples_split=10`, `min_samples_leaf=2`, `class_weight=balanced` |
| GB | `models/gb_best.joblib` | 200 cây, `learning_rate=0.05`, `max_depth=4`, `subsample=0.85` |
| SVM | `models/svm_best.joblib` | RBF, `C=10`, `gamma=0.003`, `class_weight=balanced` |

## Căn cứ lựa chọn quy tắc

CV C tham chiếu trên train cho Macro-F1 trung bình: LR 0,7170 ± 0,0191; DT 0,6823 ± 0,0139; RF 0,7256 ± 0,0143; GB 0,7139 ± 0,0119; SVM 0,7213 ± 0,0113. RF là mức trung bình cao nhất nhưng chênh SVM chỉ 0,0043 và nhỏ hơn biến thiên giữa các fold; không diễn giải là RF vượt trội có ý nghĩa.

Thử nghiệm OOF phụ chọn GB `t=0,45` và SVM `t=0,44` theo tiêu chí đã đặt trước. Nested CV cho GB Macro-F1 0,7139 so với argmax 0,7139, ngưỡng tốt hơn ở 1/5 fold; SVM 0,7072 so với argmax 0,7093. Vì lợi ích không ổn định và quy tắc phải đồng nhất giữa năm mô hình, ngưỡng không được dùng trong bảng so sánh cuối. Ngưỡng vẫn được giữ trong `models/threshold_config.json` như một thử nghiệm đánh đổi Recall/Precision, không phải cấu hình triển khai.

## Giới hạn diễn giải

- CV diễn ra sau khi chọn siêu tham số trên cùng train nên có thể lạc quan; kết quả hold-out là đánh giá độc lập một lần nhưng cũng chỉ phản ánh mẫu dữ liệu này.
- Chênh lệch RF–CV còn lớn (Macro-F1 train khoảng 0,906 so với CV 0,726); cần ghi dấu hiệu overfit. GB cũng có khoảng cách train–CV lớn.
- F1 Enrolled thấp hơn hai lớp còn lại ở một số mô hình; báo cáo đủ Macro-F1, balanced accuracy, accuracy, recall Dropout, F1 từng lớp và confusion matrix.
- Hold-out không được dùng để chọn mô hình, tham số, tập biến hay ngưỡng. Sau khi có kết quả chỉ bổ sung báo cáo, không sửa cấu hình.

## Trạng thái xác nhận

Đây là biên bản khóa kỹ thuật để thực hiện yêu cầu hoàn thiện toàn bộ repo: năm artifact đã được dựng/kiểm tra trong cùng môi trường; chọn `native_predict` làm giao thức chung; phạm vi A/B/C đã được hoàn thiện cho phân tích và demo, còn so sánh cuối khóa trên C. Hash các nguồn, mô hình và bằng chứng CV được ghi trong `final_manifest.json`.

# Danh sách Issues (tạo trên GitHub, gắn vào Project board "Đồ án ML")

Cột board: Backlog → In progress → In review → Done. Nhãn gợi ý: `eda`, `preprocess`, `model`, `report`, `demo`, `week-N`.
Có thể tạo nhanh bằng `bash scripts/create_issues.sh` (cần `gh auth login` và đứng trong repo).

| # | Tiêu đề | Người nhận | Tuần | Nhãn |
|---|---|---|---|---|
| 1 | Tải dữ liệu, kiểm tra 4.424 dòng, mô tả nguồn trong README | SV2 Huy | 1 | eda, week-1 |
| 2 | EDA: phân bố lớp, countplot, boxplot, tương quan, trùng lặp, ngoại lệ | SV2 Huy | 2 | eda, week-2 |
| 3 | Bàn giao danh sách biến phân loại/số cho tiền xử lý | SV2 Huy | 2 | eda, week-2 |
| 4 | `preprocessing.py`: build_preprocessor(feature_set, scale) | SV3 Doanh | 2 | preprocess, week-2 |
| 5 | `features.py`: tỷ lệ môn đạt, chênh lệch điểm TB kỳ 2 so với kỳ 1 | SV3 Doanh | 3 | preprocess, week-3 |
| 6 | Lựa chọn đặc trưng (MI, L1, RF importance) | SV3 Doanh | 3 | preprocess, week-3 |
| 7 | Baseline DummyClassifier | SV3 Doanh | 3 | model, week-3 |
| 8 | Logistic Regression: tuning, hệ số theo lớp | SV3 Doanh | 3-4 | model, week-3 |
| 9 | Decision Tree: tuning, hình cây, overfit | SV3 Doanh | 3-4 | model, week-3 |
| 10 | Random Forest: tuning, learning curve | SV1 Quốc | 3-4 | model, week-3 |
| 11 | Xử lý mất cân bằng: class_weight vs SMOTE | SV4 Hậu | 3 | model, week-3 |
| 12 | Gradient Boosting: tuning | SV4 Hậu | 3-4 | model, week-3 |
| 13 | SVM: tuning (có chuẩn hóa) | SV4 Hậu | 3-4 | model, week-3 |
| 14 | Điều chỉnh ngưỡng lớp Dropout (trên CV/validation) | SV4 Hậu | 4-5 | model, week-4 |
| 15 | Thí nghiệm A/B/C cho các mô hình chính (RQ3) | SV5 Triết | 4-5 | model, week-4 |
| 16 | Phân tích lỗi, lớp Enrolled (RQ4) | SV5 Triết | 4-5 | model, week-4 |
| 17 | Diễn giải mô hình: feature/permutation importance (RQ2) | SV5 Triết | 4-5 | model, week-4 |
| 18 | Đánh giá cuối trên tập test, bảng tổng hợp RQ1 | SV1 Quốc | 5 | model, week-5 |
| 19 | Demo Streamlit | SV5 Triết, SV2 Huy | 5 | demo, week-5 |
| 20 | README hoàn chỉnh + ghim requirements + kiểm tra tái lập | SV5 Triết | 5 | demo, week-5 |
| 21 | Báo cáo: viết chương theo phân công, SV1 tổng hợp | Cả nhóm | 5-6 | report, week-5 |
| 22 | Poster đề tài | SV2 Huy | 5-6 | report, week-6 |
| 23 | Slide thuyết trình + phân người nói | SV5 Triết | 5-6 | report, week-6 |
| 24 | Video demo 2-4 phút | SV5 Triết | 5-6 | demo, week-6 |
| 25 | Đóng gói, bảng đóng góp %, nộp bài, in ấn | SV1 Quốc | 6 | report, week-6 |

#!/usr/bin/env bash
# Tạo labels và 25 Issues cho repo hiện tại bằng GitHub CLI.
# Yêu cầu: gh đã đăng nhập (gh auth login), đang đứng trong thư mục repo đã push lên GitHub.
# Gán người: thay các <github-username-...> bên dưới bằng username thật rồi bỏ comment dòng --assignee nếu muốn.
set -euo pipefail

for l in eda preprocess model report demo week-1 week-2 week-3 week-4 week-5 week-6; do
  gh label create "$l" --force >/dev/null 2>&1 || true
done

mk() { # title | labels | body
  gh issue create --title "$1" --label "$2" --body "$3"
}

mk "Tải dữ liệu, kiểm tra 4.424 dòng, mô tả nguồn trong README" "eda,week-1" "SV2 (Huy). Chạy scripts/download_data.py, xác minh shape, ghi nguồn/giấy phép vào README."
mk "EDA: phân bố lớp, countplot, boxplot, tương quan, trùng lặp, ngoại lệ" "eda,week-2" "SV2 (Huy). notebooks/01_eda.ipynb, 10-15 hình vào reports/figures/."
mk "Bàn giao danh sách biến phân loại/số cho tiền xử lý" "eda,week-2" "SV2 (Huy). Ghi chú biến mã hóa số, biến tương quan cao, mã hiếm cần gộp 'Other'."
mk "preprocessing.py: build_preprocessor(feature_set, scale)" "preprocess,week-2" "SV3 (Doanh). One-Hot handle_unknown=ignore; StandardScaler chỉ khi scale=True. Xong trước cuối Tuần 2."
mk "features.py: tỷ lệ môn đạt, chênh lệch điểm TB kỳ 2 so với kỳ 1" "preprocess,week-3" "SV3 (Doanh). Xử lý chia cho 0, đặt trong transformer."
mk "Lựa chọn đặc trưng (MI, L1, RF importance)" "preprocess,week-3" "SV3 (Doanh). Chỉ giữ nếu không giảm Macro-F1 validation; nằm trong Pipeline."
mk "Baseline DummyClassifier" "model,week-3" "SV3 (Doanh). most_frequent và stratified, Stratified 5-fold, ghi qua evaluate.cross_validate_model."
mk "Logistic Regression: tuning, hệ số theo lớp" "model,week-3" "SV3 (Doanh). C, penalty L1/L2 (saga), class_weight. Lưu models/lr_best.joblib."
mk "Decision Tree: tuning, hình cây, overfit" "model,week-3" "SV3 (Doanh). max_depth, min_samples_leaf, criterion, ccp_alpha. Lưu models/dt_best.joblib."
mk "Random Forest: tuning, learning curve" "model,week-3" "SV1 (Quốc). RandomizedSearchCV f1_macro. Lưu models/rf_best.joblib."
mk "Xử lý mất cân bằng: class_weight vs SMOTE" "model,week-3" "SV4 (Hậu). imblearn Pipeline, SMOTE chỉ trên fold train. Báo Recall Dropout."
mk "Gradient Boosting: tuning" "model,week-3" "SV4 (Hậu). n_estimators, learning_rate, max_depth, subsample. Thu hẹp lưới nếu chậm."
mk "SVM: tuning (có chuẩn hóa)" "model,week-3" "SV4 (Hậu). C, kernel, gamma, class_weight. Lưới nhỏ cho RBF."
mk "Điều chỉnh ngưỡng lớp Dropout (trên CV/validation)" "model,week-4" "SV4 (Hậu). Không dùng test. Đường cong P/R và bảng ngưỡng."
mk "Thí nghiệm A/B/C cho các mô hình chính (RQ3)" "model,week-4" "SV5 (Triết). notebooks/04_timepoint.ipynb. Dùng config.FEATURE_SETS."
mk "Phân tích lỗi, lớp Enrolled (RQ4)" "model,week-4" "SV5 (Triết). notebooks/05_error_analysis.ipynb."
mk "Diễn giải mô hình: feature/permutation importance (RQ2)" "model,week-4" "SV5 (Triết). Permutation importance tính trên validation."
mk "Đánh giá cuối trên tập test, bảng tổng hợp RQ1" "model,week-5" "SV1 (Quốc). evaluate_on_test, mỗi mô hình một lần."
mk "Demo Streamlit" "demo,week-5" "SV5 (Triết) + SV2 (Huy). app/streamlit_app.py, hiển thị xác suất 3 lớp và yếu tố chính."
mk "README hoàn chỉnh + ghim requirements + kiểm tra tái lập" "demo,week-5" "SV5 (Triết). Clone repo mới và chạy lại từ đầu."
mk "Báo cáo: viết chương theo phân công, SV1 tổng hợp" "report,week-5" "Cả nhóm. Xem bản đồ chương trong kế hoạch phân công."
mk "Poster đề tài" "report,week-6" "SV2 (Huy). docs/poster.pdf, khổ theo yêu cầu GV, có QR link GitHub."
mk "Slide thuyết trình + phân người nói" "report,week-6" "SV5 (Triết). docs/slide.pptx, 15-20 trang."
mk "Video demo 2-4 phút" "demo,week-6" "SV5 (Triết). Quay màn hình Streamlit."
mk "Đóng gói, bảng đóng góp %, nộp bài, in ấn" "report,week-6" "SV1 (Quốc). Zip: Nhom01_DuDoanNguyCoBoHoc_TranDinhBaoQuoc_3124411252.zip"
echo "Đã tạo xong issues."

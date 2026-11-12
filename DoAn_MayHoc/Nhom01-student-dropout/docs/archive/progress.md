# Rà soát tiến độ theo sản phẩm hiện có

> Tài liệu lưu trữ: ghi nhận tại thời điểm 08/11/2026, không thể hiện tiến độ hiện hành. Theo dõi phần bàn giao trên Sheets và issue #31–35, xem [nơi cộng tác](../README.md).

Ngày rà soát: 08/11/2026. Phạm vi: đồ án `Nhom01-student-dropout` trong checkout
hiện tại, dựa trên [bảng đầu việc ban đầu](issues.md). Chưa đối chiếu trạng thái
Issue/PR trên GitHub. Không suy ra tuần thực hiện từ ngày trên máy.

| Hạng mục | Bằng chứng trong repo | Trạng thái sản phẩm |
|---|---|---|
| #1 Tải và xác minh dữ liệu | `scripts/download_data.py`, `data/raw/data.csv`, README | Có dữ liệu 4.424 × 37 và mô tả nguồn |
| #2–3 EDA và bàn giao biến | `notebooks/01_EDA.ipynb`, `reports/01_eda.md`, 15 hình EDA | Đã có |
| #4 Tiền xử lý | `src/preprocessing.py`, notebook 02 | Đã có Pipeline, One-Hot, chuẩn hóa tùy mô hình |
| #5 Đặc trưng mới | `src/features.py`, `reports/features_ablation.csv` | Đã có tỷ lệ đạt, chênh lệch điểm và ablation trên train |
| #6 Chọn đặc trưng | notebook 02, `reports/feature_selection_*.csv` | Có xếp hạng MI/L1/RF và CV top-k MI; chưa có thí nghiệm chọn tập con bằng L1/RF nếu yêu cầu cả ba cách |
| #7 Baseline | notebook 02, `reports/baseline_table.csv` | Đã có Dummy most_frequent và stratified |
| #8–9 LR và Decision Tree | notebook 02 dùng LR mặc định và cây depth=5 | Có thí nghiệm hỗ trợ; thiếu tuning, hệ số theo lớp/hình cây và báo cáo riêng |
| #10 Random Forest | `src/models/random_forest.py`, mô hình, báo cáo, search/CV/learning curve và metadata | Đã có; CV sau tuning, chưa đánh giá test |
| #11–14 Cân bằng lớp, GB, SVM, ngưỡng Dropout | Chưa có module/notebook/báo cáo tương ứng | Chưa đủ sản phẩm; class_weight trong RF chưa thay cho so sánh class_weight vs SMOTE |
| #15–17 A/B/C, phân tích lỗi, diễn giải | Có API A/B/C và xếp hạng khám phá trong notebook 02 | Chưa có bộ thí nghiệm RQ3, phân tích lỗi và diễn giải mô hình được chọn |
| #18 Đánh giá cuối | Có hàm `evaluate_on_test`, chưa có bảng tổng hợp đã bàn giao | Chưa thực hiện đánh giá cuối |
| #19–20 Demo và tái lập toàn bộ | Có README, `requirements-rf.txt`; thư mục app chưa có ứng dụng | Chưa đủ; requirements chung vẫn là ràng buộc tối thiểu |
| #21–25 Báo cáo, poster, slide, video, đóng gói | Có báo cáo EDA/RF; chưa có đủ bộ sản phẩm cuối kỳ | Chưa đủ |

## Các lỗi đã sửa trong đợt rà soát

- Thống nhất tên cột `Marital status`/`Marital Status`, `Nacionality`/`Nationality`,
  gom khoảng trắng/tab thừa; báo lỗi rõ nếu chuẩn hóa tạo tên cột trùng.
- `load_data()` chấp nhận cả chuỗi đường dẫn và `Path`.
- MI dùng mask rời rạc cho One-Hot/nhị phân, cập nhật riêng trong từng fold
  trước khi SelectKBest chuyển dữ liệu thành mảng. Chạy lại notebook 02 để
  thay bảng/xếp hạng và hình MI cũ.
- Bỏ việc tắt toàn bộ cảnh báo trong notebook 02, sửa cách tìm thư mục gốc
  để chạy được từ thư mục repo, dự án hoặc notebooks.
- Sửa nhận xét lựa chọn đặc trưng vốn khẳng định top-k không cải thiện,
  trong khi bảng cũ có top-80 cao hơn toàn bộ một lượng nhỏ.
- Sửa đường dẫn báo cáo bàn giao EDA thành `reports/01_eda.md`.
- Ghi log theo tên cột, hỗ trợ thêm độ đo và file rỗng; CV báo lỗi trực tiếp
  nếu một fold thất bại thay vì âm thầm trả NaN.

## Công việc tiếp theo theo phụ thuộc

1. Hoàn thành tuning và bàn giao #8–9, #11–13 trên cùng train/fold/độ đo.
2. Chạy thí nghiệm thời điểm A/B/C, cân bằng lớp và ngưỡng trên train/CV.
3. Chốt mô hình/cấu hình rồi mới đánh giá hold-out ở #18; dùng kết quả để
   phân tích lỗi và diễn giải, không tiếp tục tuning theo điểm test.
4. Tích hợp demo, ghim môi trường chung đã kiểm tra và hoàn thiện sản phẩm cuối kỳ.

Đợt rà soát này kiểm tra các phần đã có; tập test chưa được dùng để chấm điểm.

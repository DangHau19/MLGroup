# Dự đoán tình trạng học tập của sinh viên

Đồ án phân loại sinh viên thành `Dropout`, `Enrolled` hoặc `Graduate` từ thông tin tuyển sinh, nhân khẩu học, kinh tế-xã hội và kết quả học tập. Dataset: **Predict Students' Dropout and Academic Success**, UCI Machine Learning Repository, mã 697.

- Nguồn dữ liệu: <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>
- Dữ liệu: `data/raw/data.csv` — 4.424 dòng, 36 biến đầu vào và `Target`.
- Notebook tổng quan: [`notebooks/00_Tong_quan_do_an.ipynb`](notebooks/00_Tong_quan_do_an.ipynb).

## Mục tiêu và phạm vi

Bài toán phân loại đa lớp. Ba tập đặc trưng mô tả thời điểm dự đoán: **A** (thông tin nhập học), **B** (A + kết quả học kỳ 1), **C** (B + kết quả học kỳ 2). So sánh cuối và artifact chính sử dụng tập C.

| Thuật toán | Artifact |
|---|---|
| Logistic Regression | `models/lr_best.joblib` |
| Decision Tree | `models/dt_best.joblib` |
| Random Forest | `models/rf_best.joblib` |
| SVM RBF | `models/svm_best.joblib` |
| Gradient Boosting | `models/gb_best.joblib` |

KNN có trong danh sách thuật toán gợi ý ban đầu nhưng chưa được triển khai; Gradient Boosting là mô hình bổ sung. Nếu rubric yêu cầu đủ các thuật toán gợi ý, cần bổ sung KNN và đánh giá trên train/CV, không dùng lại holdout đã khóa.

## Kết quả chính đã lưu

Cấu hình được chọn bằng Stratified 5-fold CV trên train; holdout gồm 885 dòng, split phân tầng 80/20, seed 42. Bảng này đọc từ kết quả đã lưu, không chấm lại holdout.

| Mô hình | CV Macro-F1 (mean ± SD) | Holdout Macro-F1 | Accuracy | Nhận xét |
|---|---:|---:|---:|---|
| Logistic Regression | 0,7170 ± 0,0191 | 0,6817 | 0,7220 | Baseline tuyến tính |
| Decision Tree | 0,6823 ± 0,0139 | 0,6410 | 0,7299 | Macro-F1 thấp nhất |
| Random Forest | 0,7256 ± 0,0143 | 0,7057 | 0,7469 | Macro-F1 holdout cao nhất theo trị số |
| SVM RBF | 0,7213 ± 0,0113 | 0,7056 | 0,7412 | F1 Enrolled cao nhất; gần ngang RF |
| Gradient Boosting | 0,7139 ± 0,0119 | 0,6925 | 0,7605 | Accuracy và Recall Dropout cao nhất |

RF và SVM chênh 0,00016 Macro-F1; không có cơ sở nói RF vượt trội rõ rệt. `Enrolled` là lớp khó nhất, nên cần báo cáo chỉ số theo lớp bên cạnh Accuracy. Chi tiết và confusion matrix: [`docs/evaluation/holdout_summary.md`](docs/evaluation/holdout_summary.md).

## Cấu trúc dự án

```text
data/raw/                 Dataset UCI
src/                      Nạp/chia dữ liệu, xử lý, feature engineering, predict/evaluate
src/models/               Pipeline, tuning và logic mô hình
models/                   Artifact, metadata và registry demo A/B/C
app/                      Demo Streamlit
notebooks/                Notebook tổng quan, EDA, baseline, thí nghiệm, từng mô hình
reports/                  CV evidence, kết quả holdout đã lưu và audit
scripts/                  Tải dữ liệu, dựng mô hình và kiểm tra
tests/                    Unit tests
docs/                     Schema, giao thức đánh giá, kết quả cuối
```

## Cài đặt

Dùng Python 3.12 và các phiên bản thư viện được khóa trong `requirements.txt`. Chạy trong PowerShell tại thư mục gốc repo:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Nếu chưa có `data/raw/data.csv`, tải từ UCI bằng script (cần mạng):

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/download_data.py
```

Hoặc tải CSV từ trang UCI ở trên và đặt vào `data/raw/data.csv`.

## Chạy demo, notebook và kiểm thử

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
.\.venv\Scripts\python.exe -m jupyter lab
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

Chọn kernel `.venv` để chạy notebook. Notebook tổng quan chỉ đọc dataset và kết quả đã lưu; notebook mô hình thực hiện train/CV.

## Quy trình đánh giá

- Chia phân tầng 80/20, seed 42; CV trên train dùng 5 folds, shuffle, seed 42.
- Tiền xử lý nằm trong pipeline để tránh leakage; cấu hình và A/B/C ở `src/config.py`.
- Độ đo chính: Macro-F1; kèm Accuracy, balanced accuracy, chỉ số từng lớp và confusion matrix.
- Holdout cuối đã được đánh giá một lần tại `reports/evaluation/holdout/`. Không chạy lại final evaluation với `--run`, và không dùng holdout để chỉnh tham số/ngưỡng.
- Kết quả mô tả dataset UCI này; chưa chứng minh khả năng khái quát sang trường hoặc khóa học khác.

## Bảng thông tin thành viên

| Họ và tên | Mã số sinh viên | Vai trò/nhiệm vụ | Phần công việc theo phân công | Tỷ lệ đóng góp (%) |
|---|---|---|---|---:|
| Trần Đình Bảo Quốc ([#31](https://github.com/DangHau19/MLGroup/issues/31)) | 3124411252 | Random Forest, tổng hợp kết quả và nghiệm thu | Random Forest; tổng hợp kết quả đánh giá; phối hợp hoàn thiện báo cáo cuối | 20% |
| Phan Lê Minh Huy ([#32](https://github.com/DangHau19/MLGroup/issues/32)) | Cần bổ sung | Dữ liệu, EDA, demo và poster | Tải/kiểm tra dữ liệu; EDA; giao diện demo; poster | 20% |
| Trần Văn Võ Doanh ([#33](https://github.com/DangHau19/MLGroup/issues/33)) | Cần bổ sung | Tiền xử lý, LR/DT và tích hợp dự đoán | Tiền xử lý; tạo đặc trưng; Logistic Regression; Decision Tree; tích hợp dự đoán | 20% |
| Hoàng Đăng Hậu ([#34](https://github.com/DangHau19/MLGroup/issues/34)) | Cần bổ sung | Mất cân bằng lớp, GB/SVM, ngưỡng và môi trường | Thí nghiệm mất cân bằng; Gradient Boosting; SVM; ngưỡng Dropout; kiểm tra môi trường | 20% |
| Nguyễn Minh Triết ([#35](https://github.com/DangHau19/MLGroup/issues/35)) | Cần bổ sung | A/B/C, phân tích lỗi, diễn giải và video demo | So sánh A/B/C; phân tích lỗi và importance; nội dung/video demo | 20% |

## Sản phẩm bàn giao

| Sản phẩm | Tình trạng trong repo |
|---|---|
| Mã nguồn, cấu hình, dataset, hướng dẫn chạy | Có |
| Notebook, artifact mô hình, log, biểu đồ, kết quả đánh giá | Có |
| Báo cáo hoàn chỉnh, slide, poster, video demo | Chưa có trong repo; bổ sung hoặc dẫn link Docs/Slides/Drive trước khi nộp |
| MSSV và tỷ lệ đóng góp thành viên | Chưa xác nhận; điền ở bảng trên |

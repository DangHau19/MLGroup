# Dự đoán tình trạng học tập của sinh viên

Mã nguồn phân loại sinh viên thành `Dropout`, `Enrolled` hoặc `Graduate` bằng năm pipeline: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting và SVM.

## Cấu trúc code

```text
data/raw/                       dữ liệu UCI gốc
src/                            mã nguồn dùng chung
src/models/                     logic huấn luyện và tuning
models/                         mô hình đã lưu và metadata
scripts/                        lệnh tải dữ liệu, tái lập và tạo notebook
app/                            demo Streamlit
tests/                          kiểm thử mã nguồn
notebooks/02_EDA/               khám phá dữ liệu và log EDA
notebooks/03_Baseline/          tiền xử lý, baseline và log
notebooks/Experiments/          notebook thí nghiệm và Logs/Experiment_* theo từng lượt
notebooks/models/<model>/       notebook riêng cho từng mô hình
reports/                        CV evidence đóng băng, hold-out và kiểm toán
docs/                           manifest đánh giá và tài liệu dự án
```

Không lưu các bản train/test đã biến đổi trùng lặp. Cách chia dữ liệu tập trung trong `src/data.py`; tiền xử lý nằm trong pipeline để tránh lệch split và data leakage.

## Thiết lập môi trường

Tại thư mục dự án trên Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -X utf8 scripts/download_data.py
```

Môi trường đã khóa theo Python 3.12.10. Dữ liệu được lưu tại `data/raw/data.csv`.

## Chạy code

Mở notebook bằng kernel `.venv`. Kết quả chạy được xếp theo notebook trong các thư mục `Logs/Experiment_*`; xem `notebooks/Experiments/Logs/README.md`. Hold-out đã khóa nằm trong `reports/evaluation/holdout/`; CV evidence được manifest tham chiếu vẫn ở gốc `reports/`.

Chạy demo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Chạy tuning Random Forest trên train/CV:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.models.random_forest --feature-set C --n-iter 20 --n-jobs 4
```

Kết quả hold-out cuối đã được ghi một lượt tại `reports/evaluation/holdout/`. Không chạy lại chế độ `--run` để chọn mô hình, ngưỡng hoặc quay demo.

## Quy tắc chung

- A/B/C, seed và cấu hình chia dữ liệu lấy từ `src/config.py` và `src/data.py`.
- Tất cả bước tiền xử lý nằm trong pipeline.
- Chọn cấu hình theo Macro-F1 trên Stratified 5-fold CV, seed 42.
- Bảng so sánh cuối dùng `pipeline.predict()` cho cả năm mô hình.
- Artifact và metadata/hash nằm trong `models/` và `docs/evaluation/`.

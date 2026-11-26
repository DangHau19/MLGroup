# Dự đoán nguy cơ bỏ học và kết quả học tập của sinh viên bằng các mô hình phân lớp

Đồ án cuối kỳ môn Máy học, Nhóm 1. Giảng viên: TS. Đỗ Như Tài.

Bài toán: phân lớp đa lớp `Target ∈ {Dropout, Enrolled, Graduate}` trên tập dữ liệu
[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
(UCI ID 697, CC BY 4.0, 4.424 mẫu, 36 đặc trưng + Target; proposal ghi 35, SV2 xác minh khi tải dữ liệu).

## Thành viên

| Họ tên | MSSV | Vai trò |
|---|---|---|
| Trần Đình Bảo Quốc | 3124411252 | Nhóm trưởng |
| Phan Lê Minh Huy | 3124411111 | Thành viên |
| Trần Văn Võ Doanh | 3124411052 | Thành viên |
| Hoàng Đăng Hậu | 3124411085 | Thành viên |
| Nguyễn Minh Triết | 3124411320 | Thành viên |

## Cài đặt

Môi trường chung của nhóm: **Python 3.12.10, Windows 64-bit**.
Các lệnh dưới đây dùng **PowerShell**, chạy tại thư mục chứa README này.
Mỗi máy tạo môi trường `.venv` riêng để tách thư viện của đồ án khỏi dự án khác.

### 1. Cài Python

Nếu máy chưa có Python 3.12.10:

```powershell
winget install --id Python.Python.3.12 --version 3.12.10 --exact --source winget
```

Mở lại terminal sau khi cài, rồi kiểm tra:

```powershell
py -3.12 --version
```

Kết quả cần là `Python 3.12.10`. Dùng `py -3.12` để chọn đúng phiên bản khi máy
có nhiều bản Python; lệnh `python -3.12` không phải cách chọn phiên bản trên Windows.

### 2. Lấy code

Nếu chưa có repo, mở terminal ở thư mục muốn lưu code:

```powershell
git clone https://github.com/DangHau19/MLGroup.git
cd MLGroup/DoAn_MayHoc/Nhom01-student-dropout
```

Nếu đã clone, mở terminal tại thư mục chứa README này và cập nhật:

```powershell
git pull --ff-only
```

Nếu Git báo có thay đổi cục bộ hoặc nhánh phân kỳ, xử lý thay đổi của mình và
trao đổi với nhóm trưởng trước khi pull tiếp.

### 3. Tạo môi trường và cài thư viện

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Python trong `.venv` cần là 3.12.10; `pip check` cần báo
`No broken requirements found.` Các lệnh dùng trực tiếp Python của `.venv`,
nên không cần kích hoạt môi trường hoặc thay đổi Execution Policy.

Sau mỗi lần pull có cập nhật `requirements.txt`, chạy lại lệnh cài thư viện
và `pip check`. Cài thư viện của đồ án bằng Python trong `.venv`.

**Môi trường thống nhất:** toàn nhóm chỉ cài từ [requirements.txt](requirements.txt),
đã ghim chính xác cả thư viện trực tiếp và phụ thuộc gián tiếp cho Python 3.12.10,
Windows 64-bit. Bộ chính: scikit-learn 1.9.1, pandas 3.1.1, NumPy 2.5.4,
SciPy 1.18.1, joblib 1.6.0. Dùng cùng môi trường này cho notebook, huấn luyện và demo.
Hai file requirements riêng RF/SV5 đã được hợp nhất; phiên bản của các lượt chạy cũ
vẫn được giữ trong metadata kết quả, không sửa thành phiên bản mới.

Nếu môi trường cũ có thư viện khác phiên bản, đổi tên để giữ lại rồi tạo `.venv` sạch
và cài theo các lệnh trên. Trong Jupyter/VS Code, chọn kernel của `.venv` này.
Không dùng `pip freeze` từ môi trường cá nhân để tự ghi đè file chung.
Khi cập nhật thư viện, cả nhóm cập nhật cùng file và kiểm tra lại trên môi trường sạch.

**Mô hình hiện tại:** LR/DT được fit lại trong `.venv` chung; RF đã kiểm chứng lại
trên train/CV; metadata GB/SVM ghi scikit-learn 1.9.1. Môi trường và SHA dữ liệu
của từng artifact nằm trong `models/sv3_handover.json`, `models/*.meta.json` và
`reports/evaluation/rf_audit/metadata.json`. Không sửa metadata cũ để giả định đã
chạy lại. Trước đánh giá cuối, phải dùng preflight của manifest để xác nhận mọi
hash, phiên bản và nguồn CV khớp.

Các pipeline demo A/B được dựng bằng cấu hình đã chọn trên C, không tuning riêng
trên hold-out. Chạy `scripts/build_demo_models.py` để tạo lại; registry và hash nằm
trong `models/demo/model_registry.json`.
Khi gặp lỗi khác máy, gửi kèm phiên bản Python, `python.exe -m pip freeze`
trong `.venv` và đầy đủ traceback.

### 4. Tải dữ liệu

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/download_data.py
```

Script lưu dữ liệu vào `data/raw/data.csv`, dùng dấu phân cách `;`.
Kích thước dự kiến: **4.424 dòng × 37 cột** (36 đặc trưng và `Target`).
Tham số `-X utf8` giúp in thông báo tiếng Việt trên terminal Windows.

Kiểm tra nhanh file vừa tải:

```powershell
.\.venv\Scripts\python.exe -X utf8 -c "import pandas as pd; df = pd.read_csv('data/raw/data.csv', sep=';'); print(df.shape); print(df['Target'].value_counts())"
```

Nếu không tải được tự động, tải thủ công từ link UCI ở trên và đặt vào `data/raw/data.csv`.
Giữ dữ liệu gốc trong `data/raw/`; các bản đã xử lý đặt trong `data/processed/`.

#### Kết quả xác minh dữ liệu - SV2 (Tuần 01)
**Người thực hiện:** Phan Lê Minh Huy (SV2)

**Thông tin bộ dữ liệu:**
- Tên: Predict Students' Dropout and Academic Success
- Nguồn: UCI Machine Learning Repository
- UCI Dataset ID: 697
- Giấy phép: CC BY 4.0
- Đường dẫn lưu: `data/raw/data.csv`
- Định dạng: CSV, phân cách bằng dấu `;`

**Kết quả kiểm tra:**

| Nội dung | Kết quả |
|---|---|
| Tổng số mẫu | 4.424 |
| Số đặc trưng | 36 |
| Số cột nhãn | 1 (Target) |
| Tổng số cột | 37 |
| Số lớp phân loại | 3 |

**Phân bố nhãn Target:**

| Nhãn | Số lượng | Tỷ lệ |
|---|---:|---:|
| Graduate | 2.209 | 49,93% |
| Dropout | 1.421 | 32,12% |
| Enrolled | 794 | 17,95% |
| Tổng | 4.424 | 100% |

**Kết luận xác minh:**

Bộ dữ liệu đã được tải thành công bằng
`scripts/download_data.py` và lưu đúng đường dẫn
`data/raw/data.csv`.

Kết quả kiểm tra bằng pandas xác nhận dữ liệu
có 4.424 mẫu, 36 đặc trưng đầu vào và một biến
mục tiêu Target gồm ba lớp.

Dữ liệu gốc được giữ nguyên để phục vụ các bước
EDA, tiền xử lý và xây dựng mô hình tiếp theo.

### 5. Mở notebook

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook
```

Nếu dùng VS Code, chọn interpreter/kernel là `.venv\Scripts\python.exe` của đồ án.
Các notebook sẽ được chạy theo thứ tự 01 → 02 → 03 → 04 → 05 khi người phụ trách
bàn giao. Hiện có `01_EDA.ipynb` và `02_preprocess.ipynb`; Random Forest chạy
bằng module theo hướng dẫn issue #10 bên dưới.

### Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Không tìm thấy `py` | Mở lại terminal sau khi cài Python; kiểm tra Python Launcher đã được cài. |
| Python trong `.venv` sai phiên bản | Đổi tên môi trường cũ để giữ lại rồi tạo `.venv` mới bằng `py -3.12 -m venv .venv`. |
| `ModuleNotFoundError` | Kiểm tra đang dùng Python/kernel của `.venv`, rồi cài lại từ `requirements.txt`. |
| `UnicodeEncodeError` khi tải dữ liệu | Chạy script với `-X utf8` như hướng dẫn ở trên. |
| Không tìm thấy `data.csv` | Chạy lệnh tải dữ liệu từ thư mục chứa README này. |
| Thiếu cột khi gọi `load_data()` | `load_data()` đã chuẩn hóa `Marital status` → `Marital Status`, `Nacionality` → `Nationality` và khoảng trắng/tab thừa. Nếu vẫn lỗi, kiểm tra đang dùng đúng dữ liệu UCI 697 và giữ nguyên thông báo tên cột bị thiếu để đối chiếu. |

## Cấu trúc repo

```text
data/           dữ liệu gốc raw/data.csv và thư mục processed/
notebooks/      EDA, tiền xử lý, phân tích và notebooks/models/ riêng từng mô hình
src/            data, feature engineering, preprocessing, prediction, evaluation
app/            Streamlit demo cho năm mô hình và tập A/B/C
models/         model C, artifact demo A/B và metadata/hash
reports/        nơi lưu bảng kết quả và figures/
docs/           hướng dẫn cộng tác, mẫu kiểm thử, checklist và tài liệu lịch sử
scripts/        huấn luyện, dựng demo, tái lập, đánh giá và kiểm tra
requirements.txt
README.md
```

Năm notebook riêng cho từng mô hình nằm trong `notebooks/models/`. Chạy demo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Demo nhận các tập A/B/C và năm mô hình; kết quả minh họa chỉ dùng dữ liệu train.

## Dùng chung trong code

Các hàm chia dữ liệu và đánh giá dùng chung cho mọi mô hình. Ví dụ dưới đây dùng
trong notebook đã chuẩn bị biến `pipeline` chứa tiền xử lý và mô hình.
Thư mục làm việc cần là thư mục chứa README để import được `src`.

```python
from src import config
from src.data import get_train_test
from src.evaluate import cross_validate_model

X_train, X_test, y_train, y_test = get_train_test(feature_set="C")  # A | B | C
res = cross_validate_model(pipeline, X_train, y_train, name="RandomForest", feature_set="C")
```

Quy tắc bắt buộc:

1. Cùng một phép chia (`get_train_test`) và cùng độ đo (`evaluate.py`) cho mọi mô hình.
2. Mọi bước tiền xử lý nằm trong `Pipeline`/`ColumnTransformer` (chống data leakage).
3. Tập test chỉ dùng một lần ở bước đánh giá cuối.
4. Độ đo chính: Macro-F1; tuning với `scoring="f1_macro"`.

Chia train/test phân tầng **80/20**, `random_state=42`; CV phân tầng **5-fold**
trên train theo `src/evaluate.py`.

| Tập đặc trưng | Nội dung theo `src/config.py` | Số biến |
|---|---|---:|
| A | Thông tin nền, được nhóm quy ước cho thời điểm nhập học | 24 |
| B | A + kết quả học kỳ 1 | 30 |
| C | B + kết quả học kỳ 2 | 36 |

Nhóm cần xác minh thời điểm ghi nhận các biến tài chính trong A trước khi
kết luận về khả năng cảnh báo ngay lúc nhập học.

## Random Forest — Issue #10 (SV1 Quốc)

Chạy từ thư mục chứa README sau khi cài thư viện và tải dữ liệu:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.models.random_forest --feature-set C --n-iter 20 --n-jobs 4
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

`RandomizedSearchCV` chọn cấu hình theo Macro-F1, Stratified 5-fold trên train,
seed 42. Toàn bộ tiền xử lý nằm trong Pipeline, `scale=False` cho RF. Pipeline
tốt nhất được fit lại trên toàn bộ train và lưu ở `models/rf_best.joblib`.
`--feature-set` nhận A/B/C; `--n-iter` chọn số cấu hình; `--n-jobs` chọn số tác vụ
CV song song. Mỗi lần chạy ghi đè bộ sản phẩm RF và thêm dòng vào log CV chung.

- [Báo cáo kết quả và learning curve](reports/03_random_forest.md).
- `reports/rf_search_results.csv`: tham số và điểm từng fold của các cấu hình.
- `reports/rf_cv_metrics.csv`: các độ đo chung của nhóm cho cấu hình được chọn.
- `reports/rf_learning_curve.csv`, `reports/figures/rf_learning_curve.png`: điểm
  train/validation theo số mẫu, trung bình và độ lệch chuẩn qua 5 fold.
- `reports/rf_summary.json`: tham số tốt nhất, metadata, SHA-256 dữ liệu, phiên bản.
- `requirements.txt`: môi trường chung để chạy RF và toàn bộ đồ án. Phiên bản
  thực tế của lượt chạy RF được ghi riêng trong `reports/rf_summary.json`.

Đọc mô hình từ thư mục chứa README bằng
`joblib.load("models/rf_best.joblib")`. Điểm CV/learning curve tính sau tuning
trên cùng train/fold có thể lạc quan; tập test được giữ cho issue #18.

## Quy trình Git

- Mỗi người một branch: `feature/<ten-viec>`; gộp vào `main` bằng Pull Request, nhóm trưởng review.
- Commit nhỏ, đều, nội dung rõ (ví dụ `eda: add class distribution plot`).
- Mỗi đầu việc có một Issue; nhận issue của mình và đóng bằng PR (`Closes #n`).

## Trạng thái

Phân công và bàn giao hiện hành xem [hướng dẫn cộng tác](docs/README.md).
[Bảng rà soát ngày 08/11/2026](docs/archive/progress.md) và danh sách dưới đây
là ghi nhận lịch sử, không phải trạng thái đóng/mở Issue hoặc tiến độ hiện tại.

- Đã có dữ liệu và EDA (#1–3), pipeline tiền xử lý (#4), đặc trưng mới (#5),
  xếp hạng MI/L1/RF và thử chọn top-k MI (#6), baseline Dummy (#7), Random Forest (#10).
- Notebook `02_preprocess.ipynb` có LR và cây độ sâu 5 để so sánh đặc trưng;
  chưa có tuning/bàn giao hoàn chỉnh cho Logistic Regression (#8) và Decision Tree (#9).
- Các hạng mục #11–25 chưa có đủ sản phẩm bàn giao trong checkout này;
  không xem các thư mục trống là phần đã hoàn thành.

MI xác định One-Hot và biến nhị phân là rời rạc; mask được tính riêng theo cột
sau tiền xử lý của từng fold. Các biến số còn lại dùng ước lượng MI liên tục.
Kết quả chọn đặc trưng là so sánh CV trên train; chênh lệch nhỏ cần được đọc
cùng độ lệch chuẩn, chưa đủ để khẳng định một lựa chọn tốt hơn rõ rệt.

Kiểm tra phần dùng chung và Random Forest bằng lệnh:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

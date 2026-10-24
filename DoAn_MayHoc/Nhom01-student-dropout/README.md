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

**Trạng thái thư viện:** [requirements.txt](requirements.txt) hiện dùng ràng buộc
phiên bản tối thiểu (`>=`), nên các máy cài ở thời điểm khác nhau vẫn có thể nhận
phiên bản khác nhau. SV5 sẽ ghim phiên bản chính xác sau khi nhóm kiểm tra bộ
thư viện thống nhất. Khi gặp lỗi khác máy, gửi kèm kết quả `--version`,
`python.exe -m pip freeze` trong `.venv` và thông báo lỗi cho nhóm.

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
bàn giao. Hiện thư mục `notebooks/` mới là khung, chưa có notebook để chạy.

### Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Không tìm thấy `py` | Mở lại terminal sau khi cài Python; kiểm tra Python Launcher đã được cài. |
| Python trong `.venv` sai phiên bản | Đổi tên môi trường cũ để giữ lại rồi tạo `.venv` mới bằng `py -3.12 -m venv .venv`. |
| `ModuleNotFoundError` | Kiểm tra đang dùng Python/kernel của `.venv`, rồi cài lại từ `requirements.txt`. |
| `UnicodeEncodeError` khi tải dữ liệu | Chạy script với `-X utf8` như hướng dẫn ở trên. |
| Không tìm thấy `data.csv` | Chạy lệnh tải dữ liệu từ thư mục chứa README này. |
| Thiếu cột `Marital status` khi gọi `load_data()` | Bản dữ liệu tải qua `ucimlrepo` có thể dùng tên `Marital Status`. Code hiện tại chưa chuẩn hóa biến thể này; nhóm cần thống nhất bổ sung ánh xạ tên cột trước khi chạy pipeline. |

## Cấu trúc repo

```text
data/           raw/ và processed/ (hiện chưa có dữ liệu)
notebooks/      nơi lưu notebook (hiện mới có khung thư mục)
src/            config.py, data.py, evaluate.py, models/
models/         nơi lưu mô hình huấn luyện (.joblib)
reports/        nơi lưu bảng kết quả và figures/
docs/           kickoff.md, issues.md; nơi lưu báo cáo và slide
scripts/        download_data.py, create_issues.sh
requirements.txt
README.md
```

Các phần dự kiến bổ sung: `src/preprocessing.py`, notebook `01_eda`,
`02_preprocess`, `03_models`, `04_timepoint`, `05_error_analysis` và
`app/streamlit_app.py`. Demo Streamlit sẽ có hướng dẫn chạy khi ứng dụng và
mô hình đã được bàn giao.

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

## Quy trình Git

- Mỗi người một branch: `feature/<ten-viec>`; gộp vào `main` bằng Pull Request, nhóm trưởng review.
- Commit nhỏ, đều, nội dung rõ (ví dụ `eda: add class distribution plot`).
- Mỗi đầu việc có một Issue; nhận issue của mình và đóng bằng PR (`Closes #n`).

## Trạng thái

Repo hiện có khung thư mục, script tải dữ liệu, cấu hình, hàm nạp/chia dữ liệu
và hàm đánh giá chung. Chưa có notebook, pipeline tiền xử lý, mô hình huấn luyện
hay demo Streamlit. README bản đầu do SV5 hoàn thiện; kết quả thí nghiệm và
hướng dẫn demo sẽ bổ sung khi có sản phẩm của nhóm.

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

```bash
git clone <URL repo chính>
cd Nhom01-student-dropout
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py  # tải dữ liệu vào data/raw/data.csv
```

Nếu không tải được tự động, tải thủ công từ link UCI ở trên và đặt vào `data/raw/data.csv`.

## Cấu trúc repo

```
data/            raw/ (dữ liệu gốc), processed/
notebooks/       01_eda, 02_preprocess, 03_models, 04_timepoint, 05_error_analysis
src/             config.py, data.py, preprocessing.py, evaluate.py, models/
models/          mô hình đã huấn luyện (.joblib)
reports/         figures/, cv_results.csv, test_results.csv
app/             streamlit_app.py (demo)
docs/            báo cáo, slide, poster, bảng đóng góp
scripts/         download_data.py
```

## Dùng chung trong code

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

## Quy trình Git

- Mỗi người một branch: `feature/<ten-viec>`; gộp vào `main` bằng Pull Request, nhóm trưởng review.
- Commit nhỏ, đều, nội dung rõ (ví dụ `eda: add class distribution plot`).
- Mỗi đầu việc có một Issue; nhận issue của mình và đóng bằng PR (`Closes #n`).

## Trạng thái

Đang thực hiện (Tuần 1: khung repo). README sẽ được SV5 hoàn thiện ở Tuần 5.

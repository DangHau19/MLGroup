# Challenge 2: Music Genre Classification (Nhóm)

Kaggle: https://www.kaggle.com/competitions/shai-music-genre-classification
Hạn nộp: **23:59 ngày 15/11** — nhóm trưởng nộp file zip.
Metric: **F1 micro = accuracy** (đã xác nhận bằng bài nộp thử 06/11). Luật: không có quy định cụ thể; nhóm **không dùng dữ liệu ngoài**.

> ⚠️ Đề bài ghi tên file zip là `challenge1_titanic_hoten_masv.zip` (nhiều khả năng giảng viên copy nhầm từ bài cũ).
> Tên hợp lý: `challenge2_musicgenre_hoten_masv.zip`. **A hỏi lại giảng viên** trước khi nộp.

## Thành viên

| | Họ tên | MSSV | Vai trò |
|---|---|---|---|
| A | (điền) | (điền) | Trưởng nhóm, tích hợp, đóng gói |
| B | (điền) | (điền) | EDA & hiểu dữ liệu, làm slide |
| C | (điền) | (điền) | Tiền xử lý & trích đặc trưng |
| D | (điền) | (điền) | Mô hình & tuning |
| E | (điền) | (điền) | Ensemble, đánh giá, poster |

## Cấu trúc

```
musicgenre_team/
├── data/        # dữ liệu Kaggle (KHÔNG commit)
├── code/        # common.py (A), features.py (C), 00_baseline_A.py, notebook 01..04 theo từng người
├── paper/  slide/  poster/  submissions/
└── evidence/    # leaderboard, cv_logs, submission_log.csv, ảnh làm việc nhóm
```

## Quy ước chung (BẮT BUỘC)

1. `SEED = 42`, `N_FOLDS = 5`, `StratifiedKFold(shuffle=True)`. Không tự đổi.
2. Nhãn mã hoá 0..K-1 bằng `common.encode_labels`. Metric của cuộc thi là **accuracy** (= F1 micro): chọn mô hình theo `common.PRIMARY_METRIC` (accuracy); macro F1 chỉ để tham khảo.
3. Mọi mô hình chạy qua `common.cross_validate_model` rồi `common.save_oof` (lưu **xác suất** OOF + test để E ensemble).
4. Dữ liệu đầu vào mô hình lấy qua `features.build_features`, không tự tiền xử lý riêng.
5. Mỗi lần nộp Kaggle ghi `common.log_submission`. Chọn mô hình theo **CV**, không theo điểm public.
6. Notebook phải chạy sạch từ trên xuống (Restart & Run All) trước khi báo xong.
7. Code tham khảo từ notebook Kaggle công khai phải ghi nguồn.

## Dữ liệu (đã xác nhận 06/11)

Tải từ trang Data của Kaggle (cần bấm Join) và **đổi tên** đúng như sau rồi bỏ vào `data/`: `train.csv` (14.396 dòng), `test.csv` (3.600 dòng), `sample_submission.csv` (cột `Id`, `Class`).
Dữ liệu là **bảng đặc trưng kiểu Spotify** (không phải file âm thanh): `Artist Name`, `Track Name`, `Popularity`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_in min/ms`, `time_signature`; nhãn `Class` có **11 thể loại (0–10)**, mất cân bằng (lớp 10 có 3.959 mẫu, lớp 4 chỉ 310).

Chạy baseline một lần để kiểm tra môi trường: `python code/00_baseline_A.py`.

## Dùng nhanh

```python
import sys; sys.path.append("../code")
from common import *
from features import build_features
train, test, sub = load_data()
X, y, X_test, test_ids, classes = build_features(train, test)
from sklearn.linear_model import LogisticRegression
res = cross_validate_model(LogisticRegression(max_iter=2000), X, y, X_test, name="logreg")
save_oof(res)
make_submission(test_ids, res.test_proba, classes, "logreg_v0.csv")
```

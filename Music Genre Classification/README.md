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
| B | Nguyễn Minh Triết | 3124411320 | EDA & hiểu dữ liệu, làm slide |
| C | Phan Lê Minh Huy | 3124411111 | Tiền xử lý & trích đặc trưng |
| D | Trần Đình Bảo Quốc | 3124411252 | Mô hình & tuning |
| E | Hoàng Đăng Hậu | 3124411085 | Ensemble, đánh giá, poster |

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

## Phần B — Nguyễn Minh Triết

- Notebook có đầu ra: [code/01_eda_B.ipynb](code/01_eda_B.ipynb).
- Bàn giao cho C: [eda_findings.md](eda_findings.md), đang chờ C xác nhận nhận bản bàn giao.
- Mục 3 của paper: [paper/03_dataset_eda.md](paper/03_dataset_eda.md).
- Hai hình dùng cho paper: `paper/figures/B_01_class_distribution.png` và
  `paper/figures/B_02_duration_and_features.png` (300 dpi).
- Hình bổ sung, số liệu kiểm chứng, phiên bản thư viện và SHA-256 dữ liệu nằm trong notebook.
- Slide **1 trang 16:9** đã cập nhật kết quả D/E: [slide/slide_B.pptx](slide/slide_B.pptx).
  CV stacking 0,6626 ± 0,0047 đối chiếu từ OOF/JSON; Kaggle public 0,68849,
  private 0,66388 theo ảnh bàn giao, trạng thái `Complete (after deadline)`.
  Nguồn và giới hạn đánh giá nằm trong speaker notes; B không huấn luyện phần D/E.
- [Rà soát chéo B → C ngày 15/11](review_notes.md): đầu ra/bảng cuối khớp, còn
  đường chạy notebook cũ và tích hợp target encoder cần C/A xử lý. Chưa xác nhận thay C.

Chạy lại EDA bằng PowerShell tại thư mục Challenge 2:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/requirements_B.txt
.\.venv\Scripts\python.exe -m notebook
```

Chọn kernel `.venv`, mở `code/01_eda_B.ipynb` rồi **Restart & Run All**.
Notebook cần ba CSV trong `data/`, giữ nguyên file gốc và ghi lại các đầu ra B cùng tên.
Phần B chỉ mô tả/kiểm chứng dữ liệu; không sửa pipeline của C hoặc chạy tuning của D.

## Phần D — Trần Đình Bảo Quốc (3124411252)

Đã hoàn thành tuần tự: cài môi trường và tái lập baseline A → kNN/SVM/ExtraTrees →
LightGBM/XGBoost/CatBoost/MLP → tuning Optuna → v2 và nested OOF target encoding →
notebook, mục 5 và bàn giao E. Tổng cộng **32 cấu hình mới / 160 outer-fold fits**, ngoài
lượt tái lập A. Tốt nhất **CatBoost v2 + TE: CV accuracy 0,654765 ± 0,003912,
macro F1 0,700522**. Đây là CV sau lựa chọn trên cùng folds, chưa có điểm Kaggle mới.

- Notebook có đầu ra: [code/03_models_tuning_D.ipynb](code/03_models_tuning_D.ipynb).
- Mục 5: [paper/05_models_experimental_setup_D.md](paper/05_models_experimental_setup_D.md).
- Bảng 2: [paper/table_02_hyperparameters_D.md](paper/table_02_hyperparameters_D.md).
- **Bàn giao E (7 họ mô hình):** [handoff_D_to_E.md](handoff_D_to_E.md), đủ `*_oof.npy`,
  `*_test.npy`, `*_cv.json` trong `evidence/cv_logs/`.
- Điểm mọi cấu hình, trial, siêu tham số, nhãn/ID/fold và manifest SHA-256:
  `evidence/models_D/`. Baseline A tái lập giữ riêng tại `evidence/baseline_reproduction_D/`.
- Tiến độ: [D_progress.md](D_progress.md); rà soát chéo: [review_notes.md](review_notes.md).

Tại thư mục dự án, dùng Python 3.12 và dữ liệu tên chuẩn `data/train.csv`, `test.csv`,
`sample_submission.csv`. Các tên chuẩn hiện là bản sao nguyên byte của CSV gốc, không sửa dữ liệu.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/requirements_D.txt
.\.venv\Scripts\python.exe code/03_reproduce_baseline_D.py
.\.venv\Scripts\python.exe code/03_models_tuning_D.py --stage baselines
.\.venv\Scripts\python.exe code/03_models_tuning_D.py --stage boosting
.\.venv\Scripts\python.exe code/03_models_tuning_D.py --stage tuning --trials 3
.\.venv\Scripts\python.exe code/03_models_tuning_D.py --stage v2
.\.venv\Scripts\python.exe code/03_audit_D.py
.\.venv\Scripts\python.exe code/03_report_D.py
```

Các lệnh mô hình mặc định dùng lại kết quả hoàn tất; thêm `--force` để chạy lại từng
stage (tuning sẽ tạo lại study của từng họ). Chạy lần lượt như trên. Để chạy notebook,
dùng `.\.venv\Scripts\python.exe -m notebook`, chọn kernel `.venv`, rồi Restart & Run All.
Notebook mặc định đọc các kết quả đã huấn luyện thực tế và audit lại; đổi
`RECOMPUTE=True` để huấn luyện lại toàn bộ từ đầu. Notebook kiểm SHA-256 mã nguồn/dữ liệu
và phiên bản thư viện trước khi dùng cache; nếu thay dữ liệu/mã nguồn, cần recompute.

Đã thu gọn **85 tệp trung gian thành một file** `evidence/models_D/intermediate_D.zip`.
Giữ rời đủ **21 tệp OOF/test/CV của bảy mô hình bàn giao**; bảng điểm, trial và mã tái lập
vẫn còn. Baseline tái lập giữ bảng so sánh, môi trường và log ở thư mục cũ, còn các
đầu ra dự đoán được đưa vào ZIP. Notebook và audit tự đọc kết quả trong ZIP, không cần
giải nén hoặc huấn luyện lại. Sau một lượt huấn luyện lại, có thể dọn bằng:
`.\.venv\Scripts\python.exe code/03_audit_D.py --compact`.

Đã chạy notebook từ kernel mới với 7/7 cell mã, không lỗi; kết quả ở
`evidence/models_D/notebook_execution_D.json`. Median/scaler fit trong outer train;
TE cho outer train dùng inner 4-fold OOF; chỉ tần suất **không nhãn** dùng train+test theo
quy tắc nhóm. Không sửa mã C hoặc số liệu A. Chưa gửi tin nhắn nhóm/nộp Kaggle.

### Ví dụ API chung

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

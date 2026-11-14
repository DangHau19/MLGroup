# SV5 Nguyễn Minh Triết — hướng dẫn chạy, kiểm thử chéo và bàn giao

Phạm vi theo [issue #35](https://github.com/DangHau19/MLGroup/issues/35), phần bổ sung
ngày 13/11/2026, `SV5_Triet_Huong_dan_cong_viec.docx` và `Sheets Máy Học.xlsx`
do nhóm cung cấp. Mã task dùng đúng hướng dẫn; không xem task đã hoàn tất khi chưa
có sản phẩm và người kiểm tra xác nhận.

## 1. Ai làm và kiểm tra phần nào

- Triết chạy và giải thích A/B/C, lỗi Enrolled, importance; ghi nhật ký, lượt chạy và link của mình.
- Hậu kiểm tra phần phân tích/video của Triết; Triết kiểm tra phần Quốc.
- Doanh bàn giao cấu hình LR/DT và hàm dự đoán. Quốc chốt RF/mô hình cuối và đánh giá test #18.
- Hậu bàn giao GB/SVM, ngưỡng và môi trường chung đã khóa. Huy/Doanh tích hợp demo.
- Không nhận việc của người khác là lượt tự chạy. Ghi rõ Codex hỗ trợ khi dùng AI thực thi.

## 2. Chuẩn bị môi trường riêng cho phân tích SV5

Mở PowerShell tại thư mục `DoAn_MayHoc/Nhom01-student-dropout` trong bản clone của mình.
Dùng Python **3.12.10** và kiểm tra thay đổi trước khi pull:

```powershell
git status --short --branch
git pull --ff-only
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-sv5.txt
.\.venv\Scripts\python.exe -m pip check
```

Nếu đã có `.venv`, kiểm tra phiên bản thay vì tạo đè môi trường đang dùng.
`requirements-sv5.txt` chỉ tái lập #15–17; đây **chưa phải** môi trường toàn nhóm
được Hậu nghiệm thu ở #20. Không thay `requirements.txt` bằng file SV5.
Dữ liệu gốc cần ở `data/raw/data.csv`, 4.424 × 37, không chỉnh sửa dữ liệu gốc.

Lỗi sau lần chuyển tài liệu vào `docs/archive/` đã được xử lý: notebook nhận diện
thư mục gốc bằng `src/config.py` và `src/data.py`, không còn phụ thuộc `docs/issues.md`.

## 3. Chạy thực tế và thu minh chứng

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_sv5_analysis.py
```

Script in đường dẫn thư mục kết quả. Mặc định là
`Documents/MLGroup_SV5/runs/<ngày>_SV5_ANALYSIS_<giờ>` trên máy người chạy.
Có thể chọn một thư mục mới, rỗng, nằm ngoài repo:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_sv5_analysis.py --output-dir C:\DuongDanCuaBan\SV5_RunMoi
```

Script thực hiện theo thứ tự:

1. Lưu bản trước của hai notebook và ba CSV vào `previous/` trong thư mục lượt chạy.
2. Chạy `pip check`, sau đó notebook 04 rồi 05 bằng Python/kernel của môi trường gọi script.
3. Kiểm tra 45 dòng CV, 3.539 dự đoán OOF, xác suất ba lớp, cùng fold và không lẫn hold-out.
4. Kiểm tra 375 dòng importance và chạy bộ unit test hiện có của đồ án.
5. Kiểm tra kỹ thuật mô hình RF của Quốc: nạp được, khớp tham số/hash và dự đoán/xác suất
   trên ba dòng **train**. Không chấm điểm test hoặc xác nhận thay toàn bộ issue #31.
6. Xuất bốn hình từ output notebook đã chạy, ba CSV, `run.log` và `metadata.json`.
   Metadata có thời gian, Python/thư viện, commit lúc bắt đầu, trạng thái Git, hash code/dữ liệu
   và mã lượt. Mô hình `.joblib` của nhóm được kiểm tra hash để bảo đảm không bị ghi đè.

Hai notebook và ba CSV `reports/sv5_*.csv` được cập nhật tại chỗ sau lượt chạy.
Xem `previous/` trước khi so sánh với bộ cũ. Nếu chạy lỗi, dừng bàn giao, đọc log và sửa;
không nhận các output cũ còn trong notebook là kết quả mới.

Không có lệnh `evaluate_on_test` trong script. Không chạy tuning của thành viên khác
hoặc đánh giá lại test để minh họa/quay video.

## 4. Đọc kết quả và giới hạn

| Task | Kết quả hiện có | Phần còn cần bổ sung |
|---|---|---|
| SV5-ABC | RF/GB/SVM × A/B/C, cùng 5 fold; mean ± SD và hình so sánh | Nhận cấu hình LR/DT từ Doanh để đủ 5 mô hình theo #35 |
| SV5-ERR | RF-C OOF, confusion matrix, độ đo từng lớp, nhóm Enrolled và ví dụ lỗi | Tách nhận xét test cuối sau khi Quốc bàn giao #18 |
| SV5-IMP | Impurity gộp One-Hot và permutation trên validation, độ biến thiên/tên biến | Cập nhật theo mô hình cuối nếu nhóm chọn khác RF-C |

- Tham số được chọn từ C và giữ cố định khi đổi A/B/C; chưa tuning riêng A/B.
  Đây là CV sau lựa chọn, có thể lạc quan, không phải nested CV hoặc test độc lập.
- Importance toàn cục không giải thích riêng cho một sinh viên. Biến tương quan có
  thể chia sẻ tín hiệu; thứ hạng không chứng minh quan hệ nhân quả.
- Hình `01_abc_cv.png`: RQ3. `02_rf_oof_confusion.png` và `03_enrolled_errors.png`: RQ4.
  `04_rf_validation_importance.png`: RQ2. Khi đưa vào báo cáo/UI/slide, chú thích mã lượt,
  mô hình, train/CV/OOF/validation và nguồn CSV tương ứng.

## 5. Hậu kiểm tra chéo phần Triết

Hậu dùng clone và môi trường riêng; đọc commit/PR bàn giao rồi chạy lại lệnh ở mục 3.
Ghi kết quả thực tế của mình, không sao chép log của Triết thành lượt tự chạy.

- [ ] Hai notebook chạy từ đầu, không còn `StopIteration`, không có output lỗi.
- [ ] A/B/C cùng train split/fold/seed; không dùng thông tin học kỳ 2 ở A/B.
- [ ] CSV có 45 dòng cho bộ 3 mô hình hiện có; xác nhận rõ còn thiếu LR/DT.
- [ ] OOF có 3.539 dòng duy nhất, không chứa 885 dòng hold-out, xác suất cộng gần 1.
- [ ] Confusion matrix và lỗi Enrolled khớp CSV; phân biệt OOF và kết quả test cuối.
- [ ] Permutation chỉ trên validation, 5 lần/biến/fold; tên biến và SD đúng nguồn.
- [ ] Hình, CSV và metadata khớp lượt chạy; mô hình dùng chung không bị ghi đè.
- [ ] Ghi lỗi, người sửa, commit sửa và kết quả chạy lại. Chỉ xác nhận phạm vi đã kiểm tra.

Nếu dùng đúng môi trường và dữ liệu, so sánh độ đo với sai số số học nhỏ (ví dụ 1e-6),
không so byte PNG, thời gian fit, execution timestamp hoặc JSON chứa đường dẫn máy.
Nếu khác phiên bản, xem metadata trước khi kết luận sai thuật toán.

## 6. Triết kiểm tra phần Quốc

Kiểm tra RF tự động trong script chỉ là bước kỹ thuật hỗ trợ. Để nghiệm thu phần Quốc,
đối chiếu thêm model/metadata, bảng tuning/CV/learning curve, giao thức đánh giá,
nguồn kết quả test đã lưu, báo cáo/slide và gói bàn giao theo #31.
Không chạy `evaluate_on_test` lần nữa. Ghi nhận thiếu/sai và gửi Quốc sửa; không tự
đánh dấu toàn bộ #31 đạt chỉ vì RF nạp và dự đoán được.

## 7. Ghi nhật ký, thí nghiệm và bàn giao trên Sheets chung

Dùng [bảng chung đang có trong issue #35](https://docs.google.com/spreadsheets/d/1JUVp3Pv8E0ntPUp_VHdhdkiRW-3sqHz93ovYEOcKFUE/edit),
không tạo thêm bảng hoặc nhật ký theo ngày trong repo.

| Tab | Cách nhập phần Triết |
|---|---|
| Cong_viec | Giữ SV5-MC/BC/SL/VD/VC; thêm ABC/ERR/IMP/REPRO/TALK/SHEETS nếu chưa có. Tìm theo mã + người, không phụ thuộc số dòng. |
| Nhat_ky | Mỗi buổi một dòng: ngày, giờ và số giờ thực tế, việc làm, kết quả, lỗi/cách xử lý, việc tiếp theo, commit/PR và minh chứng. |
| Thi_nghiem | Mỗi lượt mới một dòng, dùng đúng `run_id` từ metadata; ghi mô hình, dữ liệu/hash, CV/seed, tham số, môi trường, độ đo, giới hạn và link thư mục kết quả. |
| Ban_giao | Chỉ dòng Triết: link nhật ký, ảnh, phần báo cáo, slide 10–12, video cá nhân, kết quả, hướng dẫn chạy và issue #35. |

Upload hình, CSV/log/metadata của lượt chạy lên Drive theo người/mã lượt. Bộ có sẵn
trên GitHub dẫn đúng commit; không nhận kết quả cũ là vừa tự chạy. `previous/` là bản
đối chiếu, không upload như kết quả mới. Không push log/ảnh/video thô từ Drive vào repo.
Nếu chưa ghi nhận thời gian lao động của bản thân, để trống; thời gian máy chạy trong
metadata không thay cho giờ làm của thành viên.

Đặt **Chờ kiểm tra** khi đủ đầu ra trong phạm vi đã bàn giao; chỉ **Hoàn tất** sau xác nhận.
Không tự ghi Hậu đã kiểm tra hoặc xác nhận thay người khác.

## 8. Báo cáo, slide, video và các phần còn chờ

- **BC:** viết phần RQ2–RQ4 trong Docs chung, có tên Triết, dùng cùng bảng/hình nguồn.
- **SL:** làm slide 10–12 trong Slides chung; Huy rà hình thức, từng tác giả xác nhận phần mình.
- **TALK:** phối hợp phân người nói và thời lượng 15 slide, tổ chức tập; không ghi đã tập khi chưa diễn ra.
- **VD:** quay thực tế màn hình chạy phần phân tích và giải thích kết quả. Có thể dùng output đã lưu
  để trình bày, nhưng phải nói rõ nếu không phải lượt chạy mới. Không dựng ảnh thành video giả thao tác.
- **REPRO/VC:** chỉ kiểm thử clone sạch toàn demo và quay video chung sau khi app, hàm dự đoán,
  mô hình/ngưỡng và môi trường chung đã bàn giao. Ghi trong mẫu kiểm thử demo; không tự nhận đã đạt.

Giới thiệu video cá nhân: nguồn/commit → chạy `run_sv5_analysis.py` hoặc mở notebook đã chạy
→ A/B/C → lỗi Enrolled → importance → giới hạn/việc chờ. Video chung theo
`docs/Demo_va_Video/Kich_ban_video.md`, 2–4 phút, kiểm tra hình/tiếng/quyền mở trước bàn giao Quốc.

## 9. Đăng GitHub và đề nghị kiểm tra

Chỉ stage tệp của Triết đã sửa; không đưa thư mục log/backup/video vào commit.
Dùng nhánh `feature/triet-analysis-demo-validation`, PR về main với `Refs #35` và
#15–#17 liên quan. Nêu lệnh đã chạy, mã lượt, kết quả và các phần còn chờ; chưa dùng
`Closes #35` khi thiếu LR/DT, demo, video hoặc xác nhận chéo.

Trong bàn giao cho Hậu, dẫn PR/commit, file hướng dẫn này, notebook/CSV và thư mục
minh chứng. Hậu ghi phản hồi kiểm tra; Triết sửa và chạy lại phần bị ảnh hưởng.
Ảnh/log/video trên Drive và nội dung Docs/Slides/Sheets không cần commit vào GitHub.

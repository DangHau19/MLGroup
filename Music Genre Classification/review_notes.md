# Nhật ký rà soát chéo Challenge 2

## B (Nguyễn Minh Triết) rà phần C — 15/11/2026

Người rà: **Nguyễn Minh Triết, 3124411320**; Codex hỗ trợ đọc mã và chạy kiểm tra.
Nguồn được rà tại commit `c9ec07a827597045e33f431c6dfcc9b6f6ab4c74`.
Phạm vi theo `PHAN_CONG.md`: B đọc phần C, ghi lỗi để C xử lý; không sửa mã,
notebook, bảng kết quả hoặc hình của C. Không huấn luyện lại mô hình D/E và không nộp Kaggle.

**Kết luận:** đầu ra đặc trưng và số liệu ablation cuối khớp khi đối chiếu; còn
đường chạy notebook cũ và hạn chế tích hợp cần C/A xử lý trước khi xác nhận tái lập
toàn bộ phần C. Chưa có xác nhận C đã nhận bản bàn giao EDA của B.

### Các kiểm tra đã đạt

| Nội dung | Kết quả thực tế |
|---|---|
| `build_features` v0/v1/v2 | Train 14.396 dòng, test 3.600 dòng; lần lượt 14/21/32 đặc trưng. Cột train/test cùng thứ tự, giá trị hữu hạn, không NaN/inf. |
| Hợp đồng đầu ra | Trả đủ X, y, X_test, test_ids, classes; giữ thứ tự Id test, classes 0–10; không đưa Id/Class/chuỗi artist-track vào ma trận số. |
| Dữ liệu đầu vào | DataFrame train/test và hash các CSV, mã/kết quả C không đổi sau kiểm tra. |
| Duration và cờ thiếu | Các cờ khớp dữ liệu gốc. Ngưỡng <30 và <1000 hiện không phân loại khác dòng train nào; đây không phải bằng chứng xác nhận đơn vị từng dòng. |
| Xử lý theo fold của bản cuối | `prepare_fold_features` trong `02_ablation_foldsafe_C.py` giữ median train khi thay giá trị validation trong mẫu kiểm soát. |
| Target encoder | Nghệ sĩ chưa biết nhận prior; tổng xác suất bằng 1. Inner OOF 4 fold trên mẫu nghệ sĩ duy nhất khớp prior chỉ học từ inner train, không dùng nhãn của chính dòng được mã hóa. |
| Ablation cuối | Tính lại độ đo từ các OOF `histgb_foldsafe_*_C_oof.npy` và fold của `common.get_folds`; mean/std accuracy và mean macro-F1 khớp CSV/JSON trong sai số 1e-10. OOF/test đúng shape, hữu hạn, xác suất cộng bằng 1. |
| Paper và hình | Mục 4 khớp bảng ablation cuối; các nhãn số trên `paper/figures/ablation_C.png` cũng khớp. |

| Phiên bản | Accuracy ± SD, 5 fold | Macro-F1 trung bình fold |
|---|---:|---:|
| v0 | 0,538344 ± 0,005567 | 0,572662 |
| v1 | 0,559808 ± 0,001722 | 0,606438 |
| v2 | 0,558072 ± 0,003065 | 0,607497 |
| v2 + TE | 0,642053 ± 0,002693 | 0,693766 |

Rà soát ban đầu đối chiếu đầu ra đã lưu và kiểm tra chức năng trên dữ liệu/mẫu
kiểm soát. Ngày 15/11/2026, A đã chạy lại toàn bộ ablation fold-safe; kết quả mới
được ghi trong `paper/04_preprocessing_features_C.md` và các artifact C.

### Vấn đề cần chủ phần C/A xử lý

| Mã | Mức độ | Phát hiện và bằng chứng | Đề nghị xử lý | Trạng thái |
|---|---|---|---|---|
| B-C-01 | Quan trọng | Ô `CELL 6` gọi luồng legacy và ô kết quả đọc log không fold-safe; Run All không tái tạo bảng cuối. | A chuyển notebook sang `02_ablation_foldsafe_C.py`, đổi bảng sang `ablation_foldsafe_C.csv`, rồi chạy sạch. | Đã xử lý bởi A; chờ B kiểm lại. |
| B-C-02 | Quan trọng khi dùng luồng cũ | Tiền xử lý trước CV khiến validation ảnh hưởng median: mẫu kiểm soát đổi giá trị điền train **1 → 99**; wrapper fold-safe giữ **20**. | Không để lệnh cũ tiếp tục chạy luồng rò rỉ; dùng fold-safe runner cho mọi lần gọi. | Đã xử lý: `02_ablation_target_C.py` chuyển tiếp sang runner fold-safe; ablation được chạy lại. |
| B-C-03 | Tích hợp | `common.cross_validate_model` từng chuyển DataFrame thành `.values`, và encoder chưa có `fit_transform` inner OOF. | Giữ DataFrame qua CV; triển khai `fit_transform` OOF để dùng trong sklearn `Pipeline`; kiểm thử với CV chung. | Đã xử lý bởi A; kiểm thử tích hợp PASS; chờ B kiểm lại. |

Không đánh dấu toàn bộ C hoàn tất chỉ dựa vào shape/NaN hoặc notebook không có output lỗi.
Các giới hạn về cờ đơn vị nguồn, nghệ sĩ mới và thiếu cấu hình v1+TE đã được C nêu
trong paper; B không tự bổ sung thí nghiệm hay nhận công việc của C.

### Kiểm tra số liệu dùng cho slide B

- Bảng mô hình lấy từ `evidence/models_D/selected_models_D.csv`: CatBoost
  **0,654765 ± 0,003912**, LightGBM **0,654001 ± 0,007181**, XGBoost
  **0,651084 ± 0,003745**. Đây là kết quả D bàn giao.
- OOF `ensemble_stack_hgb_E_oof.npy`, nhãn/fold của D và CV JSON cho mean accuracy
  **0,6626142854**, SD **0,0047325463**. `summary_E.json`, `final_ensemble_E.json`
  và bảng 3 cùng nguồn này. Slide dùng **0,6626 ± 0,0047**.
- `evidence/leaderboard/kaggle_scores_E.csv` đã được A đồng bộ CV thành
  **0,6626142854** theo `summary_E.json`; public **0,68849** và private **0,66388**
  được đối chiếu với ảnh `20261115_my_submissions.jpg`.
- Public **0,68849**, private **0,66388** và trạng thái **Complete (after deadline)**
  của `ensemble_E.csv` khớp ảnh `20261115_my_submissions.jpg`. Đây là điểm từ ảnh
  bàn giao, không phải xác nhận nộp bài môn học hoặc thứ hạng hiện tại.
- Ảnh hiện có chưa hiển thị rõ tên tài khoản Kaggle/thời điểm tuyệt đối; `evidence/submission_log.csv`
  cũng chưa có. Không tự suy đoán hoặc tạo log hồi tố; A/E cần bổ sung minh chứng này
  trước khi đưa vào PDF nếu rubric yêu cầu.
- Điểm stacking là đánh giá meta-learner trên OOF nền đã có, không phải full nested
  CV tái huấn luyện mọi mô hình nền trong từng outer meta fold. Có thiên lệch sau
  lựa chọn cấu hình; không coi CV là ước lượng độc lập cuối. Chênh với CatBoost là
  khoảng **0,78 điểm phần trăm**.

### Bàn giao và việc tiếp theo

- Slide cuối của B: `slide/slide_B.pptx`, giữ một trang 16:9; nguồn chi tiết và giới
  hạn nằm trong speaker notes. B tổng hợp kết quả D/E, không nhận là lượt B tự huấn luyện.
- Bản nháp cũ được thay bằng bản này, không giữ thêm PPTX trùng trong repo.
- A đã xử lý B-C-01/B-C-03, chuyển tiếp lệnh legacy và chạy lại ablation ngày
  15/11/2026. B cần kiểm lại các thay đổi và ghi kết quả rà soát.
- A đã tiếp nhận EDA của B và chỉ đạo dùng làm căn cứ hoàn thiện C; chưa ghi nhận
  xác nhận cá nhân từ C, nhưng điều này không còn chặn việc hoàn thiện mã theo A.
- Chưa gửi tin nhắn nhóm hoặc sửa trạng thái thay C/D/E. Mã, số liệu và dữ liệu của
  các thành viên khác giữ nguyên trong lượt rà này.

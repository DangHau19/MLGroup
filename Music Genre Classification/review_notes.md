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

Đây là đối chiếu đầu ra đã lưu và kiểm tra chức năng trên dữ liệu/mẫu kiểm soát,
không phải một lượt huấn luyện lại toàn bộ ablation.

### Vấn đề cần chủ phần C/A xử lý

| Mã | Mức độ | Phát hiện và bằng chứng | Đề nghị xử lý | Trạng thái |
|---|---|---|---|---|
| B-C-01 | Quan trọng | Ô `CELL 6` của `code/02_features_C.ipynb` vẫn gọi `02_ablation_target_C.py`; ô kế tiếp đọc các log `histgb_v0_C`, `histgb_v1_C`, `histgb_v2_C`, `histgb_v2_artist_te_C`. Luồng này xây đặc trưng trên toàn train trước CV. Ô cuối chỉ đọc bảng fold-safe có sẵn, nên Run All chưa bảo đảm tái tạo bảng cuối. | C chuyển đường chạy và bảng được đọc sang `02_ablation_foldsafe_C.py`/`histgb_foldsafe_*`, rồi chạy sạch và xác nhận kết quả. | Chờ C sửa và B kiểm lại. |
| B-C-02 | Quan trọng khi dùng luồng cũ | `features.py` điền median trên tập được truyền vào. Nếu gọi một lần trên toàn train rồi chuyển cho CV, validation đã ảnh hưởng median: mẫu kiểm soát có 30 dòng train (Popularity=20, dòng đầu thiếu), 30 dòng validation đổi từ 1 sang 99 thì giá trị điền cho dòng train đầu đổi **1 → 99**. Với wrapper fold-safe, giá trị luôn **20**. | Giữ việc fit median trong từng outer train. Dùng wrapper fold-safe cho số liệu chính thức; không lấy đường chạy legacy làm bằng chứng chống leakage. | Bản fold-safe đạt kiểm tra; notebook/đường cũ vẫn cần đồng bộ. |
| B-C-03 | Tích hợp | Dùng trực tiếp `Pipeline(ArtistTargetEncoder, DummyClassifier)` với `common.cross_validate_model` trên mẫu kiểm soát báo `Khong tim thay cot Artist Name`: API chung chuyển DataFrame thành `.values` trước khi fit. Encoder độc lập hiện không có `fit_transform` tạo inner OOF; luồng cuối đang xử lý inner OOF thủ công và đã qua kiểm tra ở trên. | C phối hợp A thống nhất cách tích hợp giữ tên cột và tạo OOF cho phần train; hoặc ghi rõ dùng luồng thủ công fold-safe, không tuyên bố transformer hiện tại cắm trực tiếp vào API chung. | Chờ C/A xác nhận phương án. |

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
- `evidence/leaderboard/kaggle_scores_E.csv` ghi CV **0,6623**, chưa khớp các nguồn
  trên. Đề nghị E đồng bộ; B giữ nguyên tệp E và ghi nguồn trong speaker notes.
- Public **0,68849**, private **0,66388** và trạng thái **Complete (after deadline)**
  của `ensemble_E.csv` khớp ảnh `20261115_my_submissions.jpg`. Đây là điểm từ ảnh
  bàn giao, không phải xác nhận nộp bài môn học hoặc thứ hạng hiện tại.
- Điểm stacking là đánh giá meta-learner trên OOF nền đã có, không phải full nested
  CV tái huấn luyện mọi mô hình nền trong từng outer meta fold. Có thiên lệch sau
  lựa chọn cấu hình; không coi CV là ước lượng độc lập cuối. Chênh với CatBoost là
  khoảng **0,78 điểm phần trăm**.

### Bàn giao và việc tiếp theo

- Slide cuối của B: `slide/slide_B.pptx`, giữ một trang 16:9; nguồn chi tiết và giới
  hạn nằm trong speaker notes. B tổng hợp kết quả D/E, không nhận là lượt B tự huấn luyện.
- Bản nháp cũ được thay bằng bản này, không giữ thêm PPTX trùng trong repo.
- Chờ C xác nhận nhận EDA, sửa B-C-01/B-C-03 và chạy lại. Sau khi có commit sửa,
  B kiểm lại phần bị ảnh hưởng và ghi ngày/kết quả thực tế tại đây.
- Chưa gửi tin nhắn nhóm hoặc sửa trạng thái thay C/D/E. Mã, số liệu và dữ liệu của
  các thành viên khác giữ nguyên trong lượt rà này.

# Phân công chi tiết — Challenge 2: Music Genre Classification (5 thành viên)

Cập nhật **06/11** sau khi A đã xem dữ liệu thật. Hạn nộp **23:59 15/11**; mục tiêu nội bộ: A đóng gói và nộp trước **18:00 15/11**. Còn 9 ngày và cùng hạn với Challenge 1, nên mỗi người nhớ chia thời gian cho cả hai.

## 1. Bài toán và dữ liệu (đã xác nhận)

- Phân loại **11 thể loại nhạc** (nhãn 0–10) từ bảng đặc trưng kiểu Spotify. Train 14.396 dòng, test 3.600 dòng.
- Đầu vào: `Artist Name`, `Track Name` (chữ), `Popularity`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_in min/ms`, `time_signature`.
- File nộp: cột `Id`, `Class` (một nhãn 0–10 cho mỗi dòng test).
- **Metric (đã xác nhận 06/11): accuracy (= F1 micro).** Trang Evaluation chỉ ghi "F1" mà không nói kiểu trung bình, nên A nộp thử một file gán toàn bộ vào lớp 10: điểm public 0,266 và private 0,295 (≈ tỉ lệ lớp 10, 27,5%). Nếu là macro F1 thì chỉ ≈ 0,04, weighted ≈ 0,12. Vậy cả nhóm **chọn mô hình theo accuracy** (`PRIMARY_METRIC = "accuracy"` trong `common.py`); macro F1 chỉ để tham khảo/phân tích lỗi.
- **Luật cuộc thi:** tab Rules chỉ là trang mẫu ("Don't cheat!..."), không có quy định cụ thể. Để an toàn, **không dùng dữ liệu ngoài** (kể cả tra thể loại nghệ sĩ trên mạng); chỉ dùng `train.csv` và `test.csv`. Nếu muốn dùng dữ liệu ngoài, hỏi giảng viên trước.
- **Không có file âm thanh**, nên bỏ hẳn nhánh `librosa`/CNN. Mọi người làm theo hướng dữ liệu bảng.

## 2. Phát hiện ban đầu từ dữ liệu (A, 06/11) — B, C kiểm chứng và đào sâu

| # | Phát hiện | Hệ quả |
|---|---|---|
| 1 | **Mất cân bằng lớp:** lớp 10 có 3.959 mẫu, lớp 6 có 2.069, lớp 9 có 2.019; lớp 4 chỉ 310, lớp 3 chỉ 322 | Accuracy bị các lớp lớn (10, 6, 9) chi phối. Dùng StratifiedKFold (đã có). **Không** cân bằng lớp để "cải thiện": `class_weight="balanced"` làm giảm accuracy (0,5585 → 0,5272) |
| 2 | **`duration_in min/ms` lẫn đơn vị:** 2.075 dòng (14%) có giá trị < 30, là **phút** (~3,9); còn lại là mili giây. Các lớp **0, 3, 4, 7 ghi 100% bằng phút**; các lớp 1, 2, 5, 6, 8 hầu như toàn mili giây | Quy đổi về một đơn vị **và giữ cờ `duration_is_min`** (đây là đặc trưng rất mạnh, nhiều khả năng do nguồn dữ liệu). Nên ghi rõ trong paper là một hiện tượng của dữ liệu |
| 3 | **Thiếu dữ liệu có quy luật:** `instrumentalness` thiếu 24,6%, `key` 11,2%, `Popularity` 2,3%. Tỉ lệ thiếu `instrumentalness` khác nhau rất xa theo lớp (lớp 7: 0,2%; lớp 5: 61%; lớp 4: 54%) | Không chỉ điền median: luôn giữ cờ `*_missing`. Với mô hình cây, thử để NaN nguyên (HistGB/LightGBM/XGBoost/CatBoost xử lý được) |
| 4 | **Nghệ sĩ có giá trị lớn:** 63,8% nghệ sĩ trong test xuất hiện trong train. Nếu dự đoán theo thể loại phổ biến nhất của nghệ sĩ (chỉ trong train fold), đúng **62,8%** trên nhóm mẫu có nghệ sĩ đã biết (58,7% mẫu) | Target encoding theo nghệ sĩ có thể cải thiện lớn, **nhưng phải tính bên trong từng fold** để không rò rỉ nhãn |
| 5 | 23,9% tên bài trong test trùng tên bài trong train | Kiểm tra bài trùng (cover/bản khác) có cùng thể loại không |
| 6 | `key` không có giá trị 0 (chỉ 1–11); không có dòng trùng lặp hoàn toàn | B xem `key` thiếu có phải là 0 (nốt C) không |
| 7 | Không có cột `Id` nào trùng giữa train (1–14.396) và test (14.397–17.996) | Không dùng `Id` làm đặc trưng |

## 3. Kết quả tham chiếu (A đã chạy, 5-fold, seed 42) — D, E cần vượt qua

| Mô hình | accuracy v0 (số gốc) | **accuracy v1** (xem `features.py`) | macro F1 v1 |
|---|---|---|---|
| Logistic Regression | 0,4966 | 0,5256 | 0,5580 |
| Random Forest (300 cây) | 0,5123 | 0,5348 | 0,5888 |
| HistGradientBoosting | 0,5389 | **0,5585** | 0,6068 |

A chạy lại trên Windows ra số lệch nhẹ (HistGB v1: accuracy 0,5582, macro F1 0,6062), do khác phiên bản thư viện. Mốc để vượt: **accuracy ≈ 0,558 (CV)**.
- **Kiểm chứng trên Kaggle (A, 06/11):** `histgb_v1.csv` được **public 0,5746 · private 0,5509**, nhất quán với CV 0,558 (chênh trong khoảng nhiễu của ~1.800 mẫu mỗi phần). Điểm CV đáng tin, chọn mô hình theo CV.
- Riêng việc sửa đơn vị `duration`, thêm cờ thiếu và tần suất nghệ sĩ đã tăng khoảng 2–3 điểm accuracy.
- Trung bình xác suất đều RF + HistGB cho accuracy 0,543, **thấp hơn** HistGB đứng một mình (0,5585): E cần trọng số hoặc stacking, không chỉ trung bình đều.
- Ghi chú: `class_weight="balanced"` và chỉnh trọng số xác suất theo lớp chỉ tăng macro F1, **không giúp accuracy**, nên không dùng cho bài này.
Những con số này chỉ là mốc để so sánh. Các mô hình này chỉ dùng scikit-learn, chưa có LightGBM, XGBoost, CatBoost (môi trường chạy thử không cài được); mỗi người tự cài theo `requirements.txt`.

## 4. Việc ngay hôm nay (A, 06/11)

1. Đổi tên 3 file thành `train.csv`, `test.csv`, `sample_submission.csv` và bỏ vào `data/`.
2. Chạy `python code/00_baseline_A.py`, nộp `submissions/histgb_v1.csv` lên Kaggle, chụp leaderboard vào `evidence/leaderboard/`, ghi `log_submission`.
3. ~~Xác định kiểu F1~~ **Đã xong:** metric là accuracy (bài nộp thử 06/11). Ghi nhật ký bài nộp đầu bằng `log_submission('histgb_v1.csv', 0.57460, 'baseline v1; private=0.55092', 'A')`.
4. Hỏi giảng viên tên file zip (đề ghi `challenge1_titanic_...`, có vẻ nhầm).
5. Gửi `musicgenre_team.zip` và file này cho nhóm; chọn người nộp Kaggle; hẹn họp 20 phút (06 hoặc 07/11).

## Sơ đồ phụ thuộc

```
B (EDA) ──► C (features v2+) ──► D (models) ──► E (ensemble, poster)
   └─────────┴──────────────────┴────────────────┴──► A (paper, đóng gói)
```
`common.py` và `features.py` v0/v1 đã có sẵn, nên **D và E bắt đầu ngay hôm nay** trên v1, không chờ C. Khi C đẩy v2 lên, chỉ cần đổi tham số `version`.

---

## A — Trưởng nhóm, tích hợp

| Hạn | Việc | Sản phẩm |
|---|---|---|
| 06/11 | Mục 4 ở trên | README điền đủ, leaderboard đầu tiên |
| 06–11/11 | Theo dõi tiến độ mỗi 2 ngày, merge PR, giải quyết xung đột | — |
| 10/11 | Họp giữa kỳ 20 phút, kiểm tra ai đang trễ | Ảnh họp |
| 12/11 | Viết Abstract, Introduction, Conclusion với số liệu cuối | Mục 0, 1, 7 |
| 13/11 | Gộp paper, thống nhất văn phong, vẽ Hình 3 (sơ đồ pipeline), tài liệu tham khảo | `paper/` |
| 14/11 | Chạy lại toàn bộ code từ đầu, tổ chức rà soát chéo | Danh sách lỗi đã sửa |
| 15/11 (<18:00) | Đóng gói zip, kiểm tra mở được, nộp | File zip |

**Xong khi:** zip có `paper/ slide/ poster/ code/ evidence/ README.md`, mở ra chạy lại được.

---

## B — EDA · làm **slide**

**Notebook:** `code/01_eda_B.ipynb`. Bàn giao `eda_findings.md` cho C trước **08/11**.

| Hạn | Việc |
|---|---|
| 07/11 | Kiểm chứng 7 phát hiện ở mục 2 bằng biểu đồ và số liệu của chính B; báo A nếu thấy khác |
| 07/11 | Phân phối từng đặc trưng theo thể loại (violin/box plot), heatmap tương quan, cặp đặc trưng tương quan cao (`energy`–`loudness`–`acousticness`) |
| 08/11 | **Thiếu dữ liệu:** biểu đồ tỉ lệ thiếu theo thể loại; kiểm tra `key` thiếu có phải nốt C không; `Popularity` thiếu có đặc biệt không |
| 08/11 | **Nghệ sĩ/bài hát:** phân phối số bài mỗi nghệ sĩ, nghệ sĩ nào có nhiều thể loại, bài trùng tên giữa train và test có cùng thể loại không |
| 08/11 | Giảm chiều (PCA/UMAP) tô màu theo thể loại: cặp thể loại nào chồng lấn |
| 08/11 | **Bàn giao C:** `eda_findings.md` (1 trang): biến cần biến đổi, outlier (`loudness` thấp tới −40, `tempo`), cách xử lý thiếu và nghệ sĩ đề xuất |
| 09/11 | Viết mục 3 của paper + Hình 1 (phân bố lớp), Hình 2 (đặc trưng + hiện tượng `duration`); PNG 300 dpi vào `paper/figures/` |
| 12–13/11 | **Slide (1 trang 16:9):** bài toán + dữ liệu · pipeline · bảng CV các mô hình · kết quả leaderboard + kết luận. Làm khung trước, điền số sau 12/11 |

**Xong khi:** `eda_findings.md` được C xác nhận; notebook chạy sạch; hình có tiêu đề, nhãn trục.

---

## C — Tiền xử lý và đặc trưng

**File chính:** `code/features.py` (C sở hữu) + `code/02_features_C.ipynb`. **Giữ nguyên chữ ký** `build_features(train, test, version)` → `X, y, X_test, test_ids, classes`. v1 đã có, C bắt đầu từ đó.

| Hạn | Việc |
|---|---|
| 07/11 | Đọc `features.py` v1, chạy lại và hiểu từng bước; báo A nếu thấy lỗi |
| 09/11 | **v2 (đặc trưng):** biến đổi lệch (`log1p` cho `speechiness`, `instrumentalness`, `liveness`, `acousticness`), `loudness` cắt đuôi; đặc trưng tương tác/tỉ số (`energy/acousticness`, `danceability*valence`, `loudness*energy`); `key` và `mode` kết hợp thành 24 giọng; `tempo` làm tròn/nhóm; thử bỏ hoặc giữ NaN cho mô hình cây |
| 09/11 | **v2 (nghệ sĩ):** target encoding theo `Artist Name`: với mỗi fold, tính tần suất 11 lớp của nghệ sĩ **chỉ từ train fold** (có làm mịn, ví dụ theo m-estimate), và dùng nested OOF cho phần train để tránh rò rỉ. Nghệ sĩ lạ trong validation/test nhận giá trị prior của toàn bộ. Cách làm: đóng gói thành một `transformer` của sklearn (`fit`/`transform`) để đặt trong `Pipeline` của từng fold, vì `build_features` toàn cục không làm được việc này một cách an toàn |
| 10/11 | **Ablation:** cố định HistGB (hoặc LightGBM), chạy CV cho v0, v1, v2 và v2 + target encoding; ghi bảng cải thiện, đây là bảng quan trọng của paper |
| 10/11 | Viết mục 4 của paper + Bảng 1 (danh sách đặc trưng) |

**Quy tắc chống rò rỉ:** mọi thống kê dùng nhãn (target encoding) hoặc dùng để fit (scaler, median) chỉ được fit trên train fold. Chỉ thống kê **không dùng nhãn** (đếm tần suất nghệ sĩ trên train+test như v1) mới được tính toàn cục.

**Xong khi:** `build_features(..., version="v2")` trả đúng định dạng, không NaN; có transformer target encoding chạy được trong `cross_validate_model`; có bảng ablation.

---

## D — Mô hình và tuning

**Notebook:** `code/03_models_tuning_D.ipynb`. **Bắt đầu ngay** trên v1.

| Hạn | Việc |
|---|---|
| 07/11 | Cài môi trường (`requirements.txt`), chạy lại `00_baseline_A.py` để kiểm tra khớp số của A |
| 08/11 | Baseline đầy đủ: kNN, SVM (RBF), ExtraTrees, thêm vào các mô hình A đã chạy. Mô hình nhạy thang đo đặt trong `Pipeline` có scaler. Lưu bằng `save_oof` |
| 09/11 | Boosting: **LightGBM, XGBoost, CatBoost** (CatBoost thường mạnh với dữ liệu bảng có thiếu); thêm một MLP nhỏ. Không cần `class_weight` (metric là accuracy) |
| 09–11/11 | **Tuning** bằng Optuna hoặc RandomizedSearch với đúng 5-fold của `common`: LightGBM (`num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda`), XGBoost (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`), CatBoost (`depth`, `learning_rate`, `l2_leaf_reg`, `iterations`), SVM (`C`, `gamma`), MLP (số lớp, dropout, weight decay) |
| 10/11 | Chuyển sang v2 khi C đẩy lên; chạy lại các mô hình tốt nhất |
| 11/11 | Viết mục 5 + Bảng 2 (siêu tham số). **Bàn giao E:** `*_oof.npy`, `*_test.npy` của ≥ 6 mô hình tốt |

**Quy tắc:** không dùng test để tuning; không chọn mô hình theo điểm public một mình. Dùng early stopping bên trong fold (không dùng validation fold làm early stopping nếu đó cũng là tập đánh giá).

**Xong khi:** ≥ 6 mô hình (≥ 3 boosting, 1 SVM/MLP) có đủ `*_oof.npy`, `*_test.npy`, `*_cv.json`; bảng siêu tham số; notebook chạy lại được; mô hình tốt nhất vượt mốc **accuracy 0,5585** của HistGB v1.

---

## E — Ensemble, đánh giá · làm **poster**

**Notebook:** `code/04_ensemble_results_E.ipynb`. Bắt đầu sớm trên OOF của baseline (A đã lưu `*_v1`).

| Hạn | Việc |
|---|---|
| 09/11 | Viết nháp **Related Work** (phân loại thể loại nhạc, đặc trưng âm thanh kiểu Spotify, gradient boosting, ensemble/stacking), mỗi ý có trích dẫn |
| 09/11 | Khung ensemble trên OOF baseline: trung bình xác suất, trung bình có trọng số (tối ưu trọng số trên OOF), so với mô hình đơn tốt nhất (tối ưu theo accuracy). **Lưu ý:** soft voting đều RF + HistGB hiện thấp hơn HistGB riêng |
| 11/11 | **Stacking:** meta-learner (LogReg) học từ OOF xác suất của các mô hình cơ sở; chống rò rỉ bằng đúng OOF |
| 11/11 | **Tối ưu trọng số trộn mô hình theo accuracy:** tìm trọng số của từng mô hình bằng tối ưu trên OOF (ví dụ `scipy.optimize` hoặc Ridge không âm), kiểm chứng bằng cách chia đôi OOF để không tự lừa mình, rồi áp lên xác suất test |
| 11/11 | Chốt ensemble; `make_submission(...)` → gửi A nộp |
| 11/11 | **Phân tích:** bảng CV mọi mô hình (`leaderboard_table()`), **confusion matrix**, độ chính xác và F1 theo từng thể loại (lớp ít mẫu như 3, 4 có bị bỏ qua không), cặp thể loại hay nhầm và vì sao, tầm quan trọng đặc trưng (permutation importance) |
| 12/11 | Viết mục 6 + Bảng 3, Hình 4, 5 |
| 12–13/11 | **Poster** (A1/A0 dọc, 3 cột theo mẫu ICIT24: Intro/Data → Method → Results → Conclusion + QR GitHub; chữ ≥ 24 pt, tiêu đề ≥ 60 pt). Khung từ 11/11, điền số sau 12/11 |

**Xong khi:** ensemble tốt hơn mô hình đơn tốt nhất (nếu không, ghi rõ và giải thích), `submission.csv` đúng định dạng `Id,Class`, poster xuất PDF.

---

## Quy tắc chung

- **Họp:** 07/11 (khởi động, 20 phút), 10/11 (giữa kỳ), 12/11 (chốt kết quả), 14/11 (rà soát chéo).
- **Báo tiến độ:** mỗi người nhắn 1–2 dòng vào nhóm chat sau mỗi mốc; trễ > 1 ngày báo A ngay.
- **Nộp Kaggle:** chỉ người được A chỉ định nộp và ghi `log_submission`. Chọn bài cuối theo CV, không theo điểm public.
- **Rà soát chéo 14/11:** B đọc mục C; C đọc mục D; D đọc mục E; E đọc mục B; A đọc tất cả. Ghi lỗi vào file chung, không tự sửa mục người khác.
- **Hai challenge cùng hạn:** nếu nhóm muốn mọi người thử vai khác, có thể hoán đổi B/C/D/E so với Challenge 1 (A giữ nguyên).
- **Liêm chính:** code tham khảo từ Kaggle ghi nguồn; ghi trung thực ai làm gì trong README (lịch sử commit là bằng chứng).

# Schema đầu vào và tiền xử lý (SV3)

Nguồn: `src/config.py` (nhóm), `src/preprocessing.py`, `src/features.py`, `models/sv3_metadata.json`. Mô hình dự đoán dùng tập đặc trưng **C (36 cột)**. Miền giá trị lấy từ **tập train**.

## 1. Bảng cột đầu vào

| Cột | Nhóm | Miền giá trị | Xử lý trong Pipeline | Tập |
|---|---|---|---|---|
| Marital Status | Phân loại (mã số, không thứ tự) | 6 mã đã thấy; số nguyên | One-Hot (handle_unknown=ignore); mã < 20 mẫu gộp 'infrequent' (Other) | A,B,C |
| Application mode | Phân loại (mã số, không thứ tự) | 18 mã đã thấy; số nguyên | One-Hot (handle_unknown=ignore); mã < 20 mẫu gộp 'infrequent' (Other) | A,B,C |
| Course | Phân loại (mã số, không thứ tự) | 17 mã đã thấy; số nguyên | One-Hot (handle_unknown=ignore); mã < 20 mẫu gộp 'infrequent' (Other) | A,B,C |
| Daytime/evening attendance | Phân loại (mã số, không thứ tự) | 2 mã đã thấy; số nguyên | One-Hot (handle_unknown=ignore); mã < 20 mẫu gộp 'infrequent' (Other) | A,B,C |
| Previous qualification | Phân loại (mã số, không thứ tự) | 17 mã đã thấy; số nguyên | One-Hot (handle_unknown=ignore); mã < 20 mẫu gộp 'infrequent' (Other) | A,B,C |
| Nationality | Phân loại (mã số, không thứ tự) | 19 mã đã thấy; số nguyên | One-Hot (handle_unknown=ignore); mã < 20 mẫu gộp 'infrequent' (Other) | A,B,C |
| Mother's qualification | Phân loại (mã số, không thứ tự) | 28 mã đã thấy; số nguyên | One-Hot (handle_unknown=ignore); mã < 20 mẫu gộp 'infrequent' (Other) | A,B,C |
| Father's qualification | Phân loại (mã số, không thứ tự) | 32 mã đã thấy; số nguyên | One-Hot (handle_unknown=ignore); mã < 20 mẫu gộp 'infrequent' (Other) | A,B,C |
| Mother's occupation | Phân loại (mã số, không thứ tự) | 31 mã đã thấy; số nguyên | One-Hot (handle_unknown=ignore); mã < 20 mẫu gộp 'infrequent' (Other) | A,B,C |
| Father's occupation | Phân loại (mã số, không thứ tự) | 41 mã đã thấy; số nguyên | One-Hot (handle_unknown=ignore); mã < 20 mẫu gộp 'infrequent' (Other) | A,B,C |
| Displaced | Nhị phân | 0 hoặc 1 | Giữ nguyên | A,B,C |
| Educational special needs | Nhị phân | 0 hoặc 1 | Giữ nguyên | A,B,C |
| Debtor | Nhị phân | 0 hoặc 1 | Giữ nguyên | A,B,C |
| Tuition fees up to date | Nhị phân | 0 hoặc 1 | Giữ nguyên | A,B,C |
| Gender | Nhị phân | 0 hoặc 1 | Giữ nguyên | A,B,C |
| Scholarship holder | Nhị phân | 0 hoặc 1 | Giữ nguyên | A,B,C |
| International | Nhị phân | 0 hoặc 1 | Giữ nguyên | A,B,C |
| Application order | Số | [0, 9] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | A,B,C |
| Previous qualification (grade) | Số | [95, 190] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | A,B,C |
| Admission grade | Số | [95, 190] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | A,B,C |
| Age at enrollment | Số | [17, 70] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | A,B,C |
| Unemployment rate | Số | [7.6, 16.2] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | A,B,C |
| Inflation rate | Số | [-0.8, 3.7] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | A,B,C |
| GDP | Số | [-4.06, 3.51] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | A,B,C |
| Curricular units 1st sem (credited) | Số | [0, 20] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | B,C |
| Curricular units 1st sem (enrolled) | Số | [0, 26] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | B,C |
| Curricular units 1st sem (evaluations) | Số | [0, 45] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | B,C |
| Curricular units 1st sem (approved) | Số | [0, 26] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | B,C |
| Curricular units 1st sem (grade) | Số | [0, 18.875] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | B,C |
| Curricular units 1st sem (without evaluations) | Số | [0, 12] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | B,C |
| Curricular units 2nd sem (credited) | Số | [0, 19] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | C |
| Curricular units 2nd sem (enrolled) | Số | [0, 23] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | C |
| Curricular units 2nd sem (evaluations) | Số | [0, 33] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | C |
| Curricular units 2nd sem (approved) | Số | [0, 20] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | C |
| Curricular units 2nd sem (grade) | Số | [0, 18.5714] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | C |
| Curricular units 2nd sem (without evaluations) | Số | [0, 12] (min-max trên train) | StandardScaler (chỉ LR/SVM); cây giữ nguyên | C |
| Pass rate 1st sem | Đặc trưng mới (tự tính, không nhập) | approved/enrolled kỳ 1, enrolled=0 -> 0 | StandardScaler (chỉ LR/SVM) | B,C |
| Pass rate 2nd sem | Đặc trưng mới (tự tính, không nhập) | approved/enrolled kỳ 2, enrolled=0 -> 0 | StandardScaler (chỉ LR/SVM) | C |
| Grade change 2nd vs 1st | Đặc trưng mới (tự tính, không nhập) | điểm TB kỳ 2 - kỳ 1 | StandardScaler (chỉ LR/SVM) | C |

## 2. Hợp đồng hàm `predict_one`

- **Vào:** dict hoặc DataFrame 1 dòng có đủ 36 cột trên (đúng tên). Cột thừa bị bỏ qua + cảnh báo.
- **Lỗi (`InputError`):** thiếu cột; không phải số hoặc NaN; nhị phân khác 0/1; phân loại không phải số nguyên; DataFrame khác 1 dòng.
- **Cảnh báo (vẫn dự đoán):** mã phân loại chưa thấy khi train (xử lý như nhóm Other); giá trị số ngoài khoảng train.
- **Ra:** `label`, `probabilities` (đủ Dropout, Enrolled, Graduate theo `classes_`), `model`, `threshold`, `warnings`.
- **Ngưỡng Dropout:** `dropout_threshold=None` (argmax) hoặc `t`: nhãn Dropout nếu P(Dropout) >= t. Chưa chốt giá trị với Hậu.

## 3. Ví dụ trước/sau xử lý (một mẫu train, scale=True như LR)

**Trước (giá trị thô):**

| Cột | Giá trị |
|---|---|
| Course | 9238.0 |
| Admission grade | 116.9 |
| Curricular units 1st sem (enrolled) | 6.0 |
| Curricular units 1st sem (approved) | 6.0 |
| Curricular units 2nd sem (enrolled) | 6.0 |
| Curricular units 2nd sem (approved) | 3.0 |
| Curricular units 1st sem (grade) | 10.375 |
| Curricular units 2nd sem (grade) | 11.66666667 |
| Tuition fees up to date | 1.0 |

**Sau (một số cột đầu ra):**

| Cột | Giá trị |
|---|---|
| Admission grade | -0.6867 |
| Pass rate 1st sem | 0.8272 |
| Pass rate 2nd sem | -0.4156 |
| Grade change 2nd vs 1st | 0.5923 |
| Tuition fees up to date | 1.0000 |
| Cột One-Hot của Course bằng 1 | Course_9238 |
| Tổng số cột sau xử lý | 123 |

Kiểm tra tay: Pass rate 1st = 6.0/6.0 = 1.0000 (thô) rồi được chuẩn hóa bằng mean/std của train.

## 4. Các ca biên đã kiểm thử (`tests/`)

| Ca | Kết quả mong đợi | Test |
|---|---|---|
| enrolled = 0 (chia 0) | tỷ lệ = 0, không NaN/inf | `test_safe_ratio_zero_division`, `test_division_by_zero_row_is_finite` |
| Mã phân loại chưa thấy | không lỗi, vào nhóm Other; hàm dự đoán cảnh báo | `test_unseen_category_does_not_fail`, `test_new_category_warns_but_predicts` |
| Scaler chỉ fit trên train | mean của scaler = mean của train | `test_scaler_fitted_on_train_only` |
| Thiếu cột / sai kiểu / NaN / nhị phân sai / phân loại lẻ | `InputError` | các test `test_predict` |
| Giá trị số ngoài miền train | cảnh báo, vẫn dự đoán | `test_out_of_range_numeric_warns` |
| Ngưỡng Dropout 0 và 1,01 | luôn Dropout / không bao giờ Dropout | `test_dropout_threshold` |

Chạy: `.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v` (24 test).
Lưu ý: mô hình `.joblib` lưu bằng scikit-learn 1.8.0; nạp bằng bản khác sẽ có cảnh báo `InconsistentVersionWarning`. Sau khi Hậu khóa phiên bản, chạy lại `python scripts/reproduce_sv3.py --write`.
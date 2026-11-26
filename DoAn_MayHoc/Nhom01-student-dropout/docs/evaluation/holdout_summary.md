# Kết quả đánh giá cuối trên hold-out

Đánh giá hoàn tất ngày 26/11/2026 sau khi khóa cấu hình và vượt preflight. Runner chấm mỗi mô hình đúng một lần trên hold-out 885 dòng; metadata ghi commit `8d1f72175775945e65dc48a17b9e0d1d324fd2d5`, data SHA-256 `edc197d1381f25414a3757ab9bbae5f1f6d21aa6a964dbe98a148c15f3d63036`. Các file dự đoán, confusion matrix, classification report và ledger nằm trong `reports/evaluation/holdout/` và `reports/evaluation/holdout_ledger/`.

## So sánh tập C

| Mô hình | CV Macro-F1 | Hold-out Macro-F1 | Balanced accuracy | Accuracy | Recall Dropout | F1 Enrolled |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0,7170 ± 0,0191 | 0,6817 | 0,6922 | 0,7220 | 0,6831 | 0,4861 |
| Decision Tree | 0,6823 ± 0,0139 | 0,6410 | 0,6311 | 0,7299 | 0,6444 | 0,3610 |
| Random Forest | 0,7256 ± 0,0143 | **0,7057** | 0,7148 | 0,7469 | 0,7113 | 0,5144 |
| Gradient Boosting | 0,7139 ± 0,0119 | 0,6925 | 0,6868 | **0,7605** | **0,7500** | 0,4589 |
| SVM RBF | 0,7213 ± 0,0113 | 0,7056 | **0,7180** | 0,7412 | 0,6796 | **0,5287** |

RF và SVM chênh 0,00016 Macro-F1, vì vậy có thể xem là ngang nhau trên lượt này; không đủ căn cứ để nói một mô hình vượt trội có ý nghĩa. RF có Macro-F1 cao nhất theo trị số, còn SVM cân bằng accuracy giữa lớp tốt hơn và nhận diện Enrolled tốt nhất. GB có Recall Dropout và Accuracy cao nhất nhưng F1 Enrolled thấp hơn. Mục tiêu sử dụng quyết định cách chọn trong các ứng dụng sau; kết quả này không làm thay đổi cấu hình cuối đã khóa.

## F1 theo lớp

| Mô hình | F1 Dropout | F1 Enrolled | F1 Graduate |
|---|---:|---:|---:|
| LR | 0,7348 | 0,4861 | 0,8241 |
| DT | 0,7276 | 0,3610 | 0,8343 |
| RF | 0,7623 | 0,5144 | 0,8405 |
| GB | 0,7662 | 0,4589 | 0,8525 |
| SVM | 0,7569 | 0,5287 | 0,8312 |

Enrolled là lớp khó nhất ở cả năm mô hình. Điều này cần được nêu trong kết luận và giới hạn, dù độ đo tổng hợp của RF/SVM đạt khoảng 0,706.

## Confusion matrix của hai mô hình đứng đầu

Thứ tự hàng/cột: Dropout, Enrolled, Graduate; hàng là nhãn thật, cột là dự đoán.

**Random Forest**

| Nhãn thật / Dự đoán | Dropout | Enrolled | Graduate |
|---|---:|---:|---:|
| Dropout | 202 | 57 | 25 |
| Enrolled | 30 | 98 | 31 |
| Graduate | 14 | 67 | 361 |

**SVM RBF**

| Nhãn thật / Dự đoán | Dropout | Enrolled | Graduate |
|---|---:|---:|---:|
| Dropout | 193 | 64 | 27 |
| Enrolled | 20 | 106 | 33 |
| Graduate | 13 | 72 | 357 |

SVM nhận đúng 106/159 sinh viên Enrolled, RF nhận đúng 98/159. Cả hai còn nhầm nhiều Graduate thành Enrolled (67 RF, 72 SVM). GB tăng recall Dropout lên 0,750, nhưng F1 Enrolled còn 0,4589.

## Đối chiếu CV và hold-out

| Mô hình | Hold-out trừ CV Macro-F1 |
|---|---:|
| LR | −0,0354 |
| DT | −0,0413 |
| RF | −0,0199 |
| GB | −0,0213 |
| SVM | −0,0157 |

Điểm hold-out thấp hơn CV ở cả năm mô hình. Đây là đánh giá độc lập một lần trên một tập giữ riêng; không dùng các kết quả để điều chỉnh tham số, feature set, ngưỡng hay chọn lại mô hình.

## Phạm vi và tái lập

- Tập C, `native_predict` cho cả năm mô hình; seed 42, split phân tầng 80/20.
- Môi trường: Python 3.12.10, scikit-learn 1.9.1, pandas 3.1.1, NumPy 2.5.4, SciPy 1.18.1, joblib 1.6.0.
- Không chạy lại `--run`; dùng bảng và dự đoán đã lưu.
- Ngưỡng OOF GB=0,45 và SVM=0,44 là thử nghiệm phụ, không áp dụng vào bảng cuối vì nested CV không cho thấy cải thiện ổn định.
- Kết quả chỉ mô tả UCI 697; chưa chứng minh hiệu quả vận hành hoặc tính khái quát cho khóa học/cơ sở khác.

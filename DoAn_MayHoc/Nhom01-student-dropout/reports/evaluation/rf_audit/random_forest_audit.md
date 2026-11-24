# Giao thức đánh giá Random Forest và kết quả tổng hợp

Cập nhật 15/11/2026. Phạm vi hiện tại là train/CV; đánh giá test cuối chưa thực hiện vì chờ bàn giao cấu hình và mô hình đầy đủ.

## Giao thức đánh giá

Bài toán phân loại ba lớp Dropout, Enrolled và Graduate trên dữ liệu UCI 697 gồm 4.424 mẫu, 36 đặc trưng và Target. Dữ liệu gốc giữ nguyên; các tên cột được chuẩn hóa qua src/data.py. Mọi mô hình dùng cùng phép chia phân tầng 80/20 với seed 42: 3.539 mẫu train và 885 mẫu hold-out. A/B/C lần lượt gồm 24/30/36 biến theo src/config.py, dùng cùng chỉ số chia mẫu.

Lựa chọn cấu hình thực hiện trên train bằng Stratified 5-fold, shuffle=True, random_state=42. Độ đo tối ưu là Macro-F1; đồng thời theo dõi Balanced Accuracy, Accuracy, Recall Dropout và F1 Enrolled. Các bước tạo đặc trưng, One-Hot và chuẩn hóa nếu cần phải nằm trong Pipeline để chỉ học trên phần train của từng fold. RF không chuẩn hóa. Độ lệch chuẩn trong tài liệu này tính qua 5 fold với ddof=0.

Tập hold-out được giữ riêng cho #18. Cấu hình, tập biến, cân bằng lớp và quy tắc ngưỡng phải được chốt bằng train/CV và commit trước khi đánh giá. Mỗi mô hình/cấu hình chốt chỉ đánh giá hold-out một lần; lưu dự đoán, bảng kết quả và metadata để dùng lại trong báo cáo, hình và video. Không tuning hoặc thay ngưỡng theo kết quả test.

## Phương pháp và kiểm chứng Random Forest

RF dùng tập C và RandomizedSearchCV 20 cấu hình trong không gian 648 tổ hợp, tối ưu f1_macro. Bộ được chọn gồm 100 cây, max_depth=None, min_samples_split=10, min_samples_leaf=2, max_features=sqrt và class_weight=balanced. Pipeline được fit lại trên toàn train sau lựa chọn; tệp bàn giao là models/rf_best.joblib.

Kết quả RF-C CV: Macro-F1 0,7256 ± 0,0143; Balanced Accuracy 0,7311; Accuracy 0,7666; Recall Dropout 0,7308; F1 Enrolled 0,5439. Macro-F1 train trong CV đạt 0,9062. Đây là CV sau tuning trên cùng train/fold, có thể lạc quan, không phải nested CV hoặc điểm test độc lập.

Ở mức lớn nhất của learning curve, Macro-F1 train khoảng 0,9045 và validation 0,7216, khoảng cách khoảng 0,1829. Validation nhìn chung cải thiện khi tăng dữ liệu, nhưng khoảng cách train–validation còn lớn, cho thấy dấu hiệu overfit. Điểm cuối learning curve có thể khác nhẹ CV tuning vì kích thước mẫu train tuyệt đối được giữ bằng nhau qua các fold.

Lượt kiểm chứng 20261115_SV1_RF_AUDIT_151441 đã fit clone train, tính lại CV và learning curve. Kết quả khớp bản bàn giao trong sai số 1e-10; mô hình nạp và trả xác suất đúng ba lớp. Bộ kiểm tra đạt 14/14, gồm 9 kiểm thử có sẵn và 5 kiểm thử công cụ đánh giá cuối. Dữ liệu và mô hình gốc không đổi. Môi trường kiểm chứng dùng pandas 3.1.1, khác pandas 3.0.6 ghi trong lượt RF gốc; các độ đo vẫn khớp và khác biệt được giữ rõ trong metadata.

## So sánh các mô hình trong phạm vi hiện có

Bảng dưới lấy từ timepoint_cv.csv của phần Triết, tập C, cùng fold/seed và cấu hình chọn từ C. RF được Quốc kiểm chứng lại độc lập ở bước train/CV của gói này; GB/SVM trong bảng là kết quả tham chiếu đã lưu, không phải lượt Quốc vừa chạy. Không trộn bảng sv4_rebuild_cv.csv chạy ở sklearn 1.8.0 vào cùng bảng để kết luận mô hình cuối.

| Mô hình | Macro F1 CV trung bình ± SD | Balanced Accuracy | Recall Dropout | F1 Enrolled |
|---|---|---|---|---|
| RF | 0,7256 ± 0,0143 | 0,7311 | 0,7308 | 0,5439 |
| GB | 0,7139 ± 0,0119 | 0,7053 | 0,7827 | 0,4823 |
| SVM | 0,7213 ± 0,0113 | 0,7294 | 0,7273 | 0,5398 |


RF cao hơn SVM khoảng 0,0043 Macro-F1 trong bộ kết quả này, nhỏ hơn độ lệch chuẩn từng mô hình. Chưa có kiểm định hoặc đánh giá test để khẳng định RF vượt trội. GB có Recall Dropout cao hơn trong bộ tham chiếu nhưng F1 Enrolled thấp hơn; vì vậy cần cân nhắc đồng thời mục tiêu phát hiện Dropout và chất lượng đa lớp.

Baseline lưu trong baseline_table.csv có Dummy most_frequent Macro-F1 0,2220 ± 0,0002 và Dummy stratified 0,3383 ± 0,0092. Các giá trị này là kết quả đã bàn giao, làm mốc tham chiếu; chưa được chạy lại trong lượt kiểm chứng Quốc này. LR/DT chưa có tuning và pipeline hoàn chỉnh để đưa vào bảng so sánh cuối.

## Kết luận và phần cần hoàn thiện

Trong phạm vi train/CV hiện có, RF cho chất lượng đa lớp cao hơn baseline; kết quả Enrolled vẫn thấp hơn mức tổng thể và RF còn dấu hiệu overfit. Không suy rộng bảng này thành kết luận RQ1 cuối hoặc hiệu quả triển khai thực tế. Kết quả RQ2–RQ4 do Triết trình bày bổ sung góc nhìn thời điểm, lỗi và importance; importance toàn cục không phải giải thích từng sinh viên và không chứng minh nhân quả.

Trước #18 cần nhận LR/DT từ Doanh, GB/SVM và ngưỡng từ Hậu trong đúng môi trường chung; chốt phạm vi và cấu hình bằng CV. Demo cần Huy/Doanh tích hợp rồi Triết/Hậu kiểm tra clone sạch. Sau #18, bổ sung bảng test riêng, nhãn rõ CV/test, confusion matrix và kết luận cuối; giữ nguyên số liệu CV để truy vết.

## Nguồn số liệu

GitHub được rà tại 72a9545ff16bd5757e6254bef7286e299a93a410. RF: reports/rf_summary.json, rf_cv_metrics.csv, rf_learning_curve.csv. Tham chiếu đa mô hình: reports/timepoint_cv.csv, baseline_table.csv. Kiểm chứng mới: reports/evaluation/rf_audit/metadata.json, rf_cv_recheck.csv, rf_learning_curve_recheck.csv và tests.log. Không có điểm hold-out trong phần này.

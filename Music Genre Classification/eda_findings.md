# Bàn giao EDA từ B cho C
Nguyễn Minh Triết · 3124411320. **Trạng thái: A đã tiếp nhận để chỉ đạo hoàn thiện phần C; chưa ghi nhận xác nhận cá nhân từ C.**
Nguồn: ZIP Kaggle người dùng cung cấp; bằng chứng: `code/01_eda_B.ipynb` (bảng kiểm chứng và số liệu ở mục 8–9).

| Phát hiện đo trên dữ liệu | Đề xuất để C thử trong ablation |
|---|---|
| Train 14.396 × 18, test 3.600 × 17; 11 lớp. Class 10: 3.959, Class 4: 310; lệch 12.77 lần. | Giữ folds phân tầng seed 42. Tối ưu accuracy theo quy ước nhóm, báo thêm macro-F1; B chưa kiểm chứng thí nghiệm class_weight. |
| Duration <30: 2075 dòng (14.41%); giá trị nhỏ 0.502–29.886; nhóm lớn từ 55775. Ngưỡng <30 và <1000 khác ở 0 dòng train / 0 dòng test. | Giả thuyết nhóm nhỏ là phút, nhóm lớn ms: quy về giây/phút và giữ cờ nguồn. Hiện `features.py` dùng <1000, cần thống nhất ngưỡng theo kết quả này. Không coi cờ là đặc tính âm nhạc phổ quát. |
| Thiếu train: instrumentalness 24.60%, key 11.18%, Popularity 2.31%; thay đổi theo Class. Key không có 0 trong train/test. | Giữ missing flags; imputation/scaling chỉ fit train fold. Không tự thay mọi key NaN thành 0 khi chưa có bằng chứng từ nguồn. So NaN nguyên với imputer trên mô hình hỗ trợ. |
| Loudness -39.952 đến 1.342; tempo 30.557 đến 217.416. Bảng skew/IQR trong notebook. | Thử log1p cho biến lệch không âm và tương tác theo phân công. Cắt đuôi là lựa chọn phải ablation; không tự xóa cực trị nếu chưa xác minh lỗi. Key/mode/time_signature là mã rời rạc. |
| 2296/3.600 dòng test (63.78%) có artist đã thấy; 1375 nghệ sĩ train có nhiều lớp. Kiểm tra folds: coverage 58.68%, đúng 62.31% trên nhóm biết artist. | Artist hữu ích nhưng có đa nhãn. Encoding dùng nhãn cần fit trong từng outer fold và OOF bên trong train; nghệ sĩ lạ dùng prior. Kiểm tra grouped-by-artist bổ sung nếu mục tiêu là nghệ sĩ mới. |
| 860 dòng test (23.89%) trùng tên track; 607 dòng trùng cả artist/track; 1043 cặp artist/track train có nhiều nhãn. | Không gán nhãn theo tên trùng; test không có Class để xác minh. Tách exact match khỏi chuẩn hóa chuỗi và bản thu/cover. |

Spearman: energy–loudness 0.735, energy–acousticness -0.728. PCA 2D giữ 33.1% phương sai; các cặp có trộn láng giềng cao gồm 8–10, 6–10, 1–10, chỉ là dấu hiệu trên phép chiếu, chưa là lỗi mô hình.
Đối chiếu A: độ đúng nghệ sĩ đã biết 62,31% so với mốc 62,8%; cần kiểm tra quy tắc hòa/định nghĩa trước khi kết luận nguyên nhân khác biệt. Popularity thiếu ở 333 dòng (Class 0: 12,0%); nhóm thiếu có trung vị acousticness 0,144 so với 0,0801.
ID train/test không giao nhau; trùng dòng train bỏ Id: 0. Giữ dữ liệu gốc, không dùng Id làm feature. **C xác nhận nhận bàn giao: chưa có; A đã chỉ đạo dùng nội dung này làm căn cứ hoàn thiện C ngày 15/11/2026.**

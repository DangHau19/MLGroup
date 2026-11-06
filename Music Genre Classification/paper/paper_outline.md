# Dàn ý paper — Music Genre Classification

Độ dài gợi ý 6–8 trang, theo cấu trúc paper ICIT24 mà giảng viên đưa (A chỉnh lại khi có mẫu).

| # | Mục | Người viết | Hạn nháp |
|---|---|---|---|
| 0 | Title, Abstract | A | 12/11 |
| 1 | Introduction | A | 12/11 |
| 2 | Related Work (phân loại thể loại nhạc, đặc trưng âm thanh Spotify, gradient boosting, ensemble) | E | 08/11 |
| 3 | Dataset & EDA | B | 07/11 |
| 4 | Preprocessing & Feature Extraction | C | 09/11 |
| 5 | Models & Experimental Setup | D | 10/11 |
| 6 | Results & Discussion | E | 12/11 |
| 7 | Conclusion & Future Work | A | 12/11 |
| 8 | References | A | 13/11 |

## Hình/bảng dự kiến
- Hình 1: phân bố các thể loại (B) · Hình 2: đặc trưng theo thể loại + hiện tượng `duration` lẫn đơn vị phút/ms (B)
- Hình 3: sơ đồ pipeline (A) · Bảng 1: danh sách đặc trưng (C) · Bảng 2: siêu tham số (D)
- Bảng 3: so sánh CV mọi mô hình (E) · Hình 4: confusion matrix (E) · Hình 5: F1 theo từng thể loại (E)

## Khung Abstract (A)
1) Phân loại thể loại nhạc tự động: ứng dụng gợi ý/lập chỉ mục nhạc. 2) Thách thức: 11 thể loại mất cân bằng, dữ liệu thiếu có quy luật, đơn vị `duration` không thống nhất.
3) Phương pháp: <đặc trưng> + <mô hình> + <ensemble>. 4) Kết quả: CV **[..]**, public leaderboard **[..]**.
5) Kết luận: thành phần đóng góp nhiều nhất.

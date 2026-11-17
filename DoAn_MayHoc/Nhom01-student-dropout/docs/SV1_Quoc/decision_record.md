# Biên bản chốt quy tắc suy luận

Trạng thái: CHƯA CHỐT. File này là vị trí ghi nhận quyết định từ train/CV trước test, không phải xác nhận đã nhận đủ bàn giao.

| Mô hình | Tập biến | Cân bằng | Calibration | Quy tắc nhãn/ngưỡng | Bằng chứng CV | Người bàn giao | Quốc xác nhận |
|---|---|---|---|---|---|---|---|
| LR | Chờ Doanh | Chờ | Chờ | Chờ | Chờ | Doanh | Chưa |
| DT | Chờ Doanh | Chờ | Chờ | Chờ | Chờ | Doanh | Chưa |
| RF | C tham chiếu | balanced theo rf_summary | Không trong pipeline hiện có | Native predict đang là cấu hình tham chiếu, chưa chốt mô hình cuối | reports/rf_cv_metrics.csv | RF đã kiểm chứng | Chưa chốt cuối |
| GB | C tham chiếu | Chờ Hậu xác nhận | Chờ | Chờ | reports/sv4_rebuild_cv.csv cần tái lập môi trường mới | Hậu | Chưa |
| SVM | C tham chiếu | balanced | probability=True cần tái lập | Chờ ngưỡng/quyết định native | reports/sv4_rebuild_cv.csv cần tái lập | Hậu | Chưa |

Nếu chọn ngưỡng: ghi chính xác tiêu chí, ngưỡng, nguồn dự đoán validation/OOF, quy tắc nhiều lớp và cách phá hòa. Công cụ hiện hỗ trợ: p(Dropout)>=t thì Dropout, còn lại lấy xác suất lớn nhất trong Enrolled/Graduate. Chỉ dùng khi Hậu bàn giao đúng quy tắc này. Nếu quy tắc khác, phải sửa công cụ và kiểm thử trên dữ liệu giả/train trước commit; không tự đổi nghĩa ngưỡng.

Nếu giữ native_predict: ghi rõ lý do chọn từ CV và bằng chứng đối chứng ngưỡng; không bỏ trống #14 chỉ vì không dùng ngưỡng.

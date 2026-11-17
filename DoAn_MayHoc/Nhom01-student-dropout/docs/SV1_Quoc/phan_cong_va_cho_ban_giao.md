# Trạng thái phần Quốc


1. Doanh: chưa có models/lr_best.joblib, models/dt_best.joblib, tuning/CV/metadata hoàn chỉnh và hàm dự đoán/schema bàn giao. Quốc chưa thể chốt đủ 5 mô hình cho #18.
2. Hậu: đã có code rebuild và SVM probability=True, nhưng metadata bàn giao vẫn sklearn 1.8.0 (numpy 2.4.4, pandas 3.0.2), khác requirements chung sklearn 1.9.1. Kiểm tra thực tế: GB lỗi No module named '_loss'; SVM có cảnh báo lệch phiên bản. Cần Hậu xuất model/metadata bằng đúng .venv chung, bàn giao ngưỡng hoặc quyết định giữ native_predict có bằng chứng train/CV.
3. Huy và Doanh: app/streamlit_app.py chưa có trong snapshot GitHub này; chờ demo tích hợp và ca kiểm tra. Chưa quay video demo chung hoặc kiểm thử clone sạch cho sản phẩm chưa tồn tại.
4. Triết: phần RF/GB/SVM A/B/C, OOF và importance đã có; còn chờ LR/DT hoặc xác nhận chính thức phạm vi. Cần kiểm tra chéo đầu ra Quốc mới tạo, phân người nói và video chung sau khi demo đạt.
5. Cả nhóm: chưa đủ nội dung báo cáo 5 chương, 15 slide hoàn chỉnh, poster, video, nhật kí và link bàn giao của 5 người; cần xác nhận đóng góp phần trăm và kết quả kiểm tra chéo.
6. Quốc: cần quay video cá nhân thực tế theo kịch bản đã chuẩn bị; xác minh hạn/nơi nộp và yêu cầu in từ thông báo môn học. Không thể tạo bằng chứng nộp/in hoặc xác nhận của người khác thay thực tế.

Tỷ lệ đóng góp để trống đến khi 5 người xác nhận. Không đóng #31 hoặc #18 qua gói bàn giao này.

# Kiểm tra demo

| Thông tin | Giá trị |
|---|---|
| Ngày và người kiểm tra | 25/11/2026 — phần kiểm thử của Triết, Codex hỗ trợ thực thi |
| Bản kiểm thử | [PR #37](https://github.com/DangHau19/MLGroup/pull/37), commit `4a4e8f0948adb440d0251964e9f9b916c0354a3c`; main đối chiếu `58c7e2c` |
| Cách chạy | Giải nén snapshot từ `git archive` của commit PR, tạo venv mới, chạy AppTest và kiểm tra giao diện trình duyệt; chưa kiểm tra bản tích hợp sau merge |
| Môi trường | Windows, Python 3.12.10; Streamlit 1.67.0, scikit-learn 1.9.1; khớp 127 gói ghim, `pip check` đạt |
| Lệnh cài và khởi động | Trong thư mục dự án: `python -m pip install -r requirements.txt`, `python -m streamlit run app/streamlit_app.py` |
| Model ID, tập biến và ngưỡng | `models/gb_best.joblib`, tập C; dùng `predict` và `predict_proba` của pipeline, không áp dụng ngưỡng tùy chỉnh |
| Nguồn importance | App chưa hiển thị. `reports/importance.csv` của Triết là RF-C, không dùng để giải thích dự đoán GB |

[Minh chứng lượt kiểm thử](https://drive.google.com/drive/folders/1cZth3kcMm7d69k4gvKd6yPMjcqTvKRjG): metadata, log chạy, log thư viện, log unit test và 3 ảnh giao diện.

| Mã ca | Đầu vào và bước thử | Kết quả mong đợi | Kết quả thực tế | Đạt hoặc lỗi | Link log hoặc ảnh |
|---|---|---|---|---|---|
| TC01 | Mở ứng dụng | Form hiển thị, không lỗi khởi động | Có form và chọn tập biến, không có exception | Đạt | [Metadata][metadata] |
| TC02 | Chọn A, gửi dữ liệu hợp lệ | Có nhãn và xác suất 3 lớp | Chỉ báo hiện hỗ trợ C | Chưa đạt | [Ảnh A][anh-a], [metadata][metadata] |
| TC03 | Chọn B, gửi dữ liệu hợp lệ | Có nhãn và xác suất 3 lớp | Chỉ báo hiện hỗ trợ C | Chưa đạt | [Metadata][metadata] |
| TC04 | Chọn C, dùng hàng train ID 2283, đối chiếu pipeline | Nhãn, thứ tự lớp và xác suất khớp model | Graduate; Dropout/Enrolled/Graduate = 1.57%/2.49%/95.94%; tổng xác suất gốc = 1 | Đạt | [Metadata][metadata] |
| TC05 | Học kỳ 1: đăng ký 3, đạt 5 môn | Chặn đầu vào, không dự đoán | Hiển thị lỗi số môn đạt vượt số môn đăng ký | Đạt | [Metadata][metadata] |
| TC06 | C, nhập `Admission grade = -1` | Chặn điểm ngoài miền 0–200 | Vẫn nhận dữ liệu và dự đoán | Chưa đạt | [Ảnh điểm âm][anh-am], [metadata][metadata] |
| TC07 | Tạm bỏ model trong bản thử riêng, xóa cache rồi dự đoán C | Báo lỗi tải model, không văng ứng dụng | Có thông báo lỗi; model đã được khôi phục sau thử | Đạt | [Metadata][metadata] |
| TC08 | Xem kết quả C | Có yếu tố quan trọng, nguồn và giới hạn diễn giải | Chưa có phần importance | Chưa đạt | [Ảnh C][anh-c], [metadata][metadata] |

**Kết quả: 4/8 ca đạt, chưa đủ nghiệm thu demo.** Bộ unit test hiện có chạy bằng `python -m unittest discover -s tests -v`: **50/50 đạt** ([log][unit]).

Form tự tạo đủ cột, dùng ô số và danh mục cố định; chưa kiểm thử truyền record thiếu cột, sai kiểu hoặc mã mới qua API vì app không có đầu vào đó. Kiểm tra dự đoán dùng mẫu train, không chấm hold-out hay huấn luyện lại model. Ảnh trình duyệt dùng dữ liệu mặc định của form để minh họa giao diện, khác mẫu đối chiếu TC04.

## Phản hồi và sửa lỗi

| Mục cần xử lý | Bước tái hiện | Phần phụ trách | Bản sửa | Kết quả thử lại |
|---|---|---|---|---|
| Chưa dự đoán A/B | Chọn A hoặc B, gửi form | Huy — app | Chưa có | Chờ bản sửa |
| Điểm tuyển sinh âm vẫn được nhận | Chọn C, nhập điểm -1, gửi form | Huy — app | Chưa có | Chờ bản sửa |
| Thiếu yếu tố quan trọng | Dự đoán C, xem phần kết quả | Huy — app; phối hợp Triết về nguồn diễn giải | Chưa có | Chờ bản sửa |

Lượt này chỉ cập nhật biên bản của Triết; không sửa code Huy, không merge PR. Mã nguồn, model và dữ liệu gốc đã được đối chiếu không đổi sau kiểm thử.

[metadata]: https://drive.google.com/file/d/1zprYwG4CPva6K3IKnus5rMmH8P9FfGn5/view
[unit]: https://drive.google.com/file/d/16Oe-dRVzfMqcasGPo91rqiFyFgZ9GuC3/view
[anh-a]: https://drive.google.com/file/d/1zHnqFEpc91Zw-nPd1WC6JH2a-Y-S1lpD/view
[anh-c]: https://drive.google.com/file/d/19coEHgoIBSNAegXAvA75L5tRSf4_uxI3/view
[anh-am]: https://drive.google.com/file/d/1h1qH5qcw_HRZJkClJ9SOonctnnc7JakD/view


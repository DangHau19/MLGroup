# Bàn giao phần Quốc ngày 15 tháng 11 năm 2026

Nguồn GitHub đã rà: https://github.com/DangHau19/MLGroup/commit/72a9545ff16bd5757e6254bef7286e299a93a410. Issue #31 còn mở. Lượt kiểm chứng 20261115_SV1_RF_AUDIT_151441 dùng snapshot GitHub và code bổ sung trong gói; không chấm hold-out, không sửa data/model gốc. Metadata lưu cả hash code bổ sung chưa có trong commit nguồn; commit mới của Quốc sẽ là mốc bàn giao các file này.

RF được kiểm chứng CV/learning curve khớp bản bàn giao trong sai số 1e-10; 14/14 test đạt. 127 phiên bản khớp requirements. pandas 3.1.1 khác lượt RF cũ 3.0.6 và được ghi rõ, không sửa metadata lịch sử.

## Tệp chính
- reports/sv1/bao_cao_Quoc.md: phần báo cáo train/CV có số liệu và giới hạn.
- reports/sv1/metadata.json, tests.log: bằng chứng chạy thực tế; test log có dòng blocked từ kiểm thử dữ liệu giả, không phải đánh giá hold-out.
- scripts/sv1_audit.py: kiểm chứng train/CV, output bắt buộc là thư mục mới ngoài repo.
- src/sv1_final_evaluation.py: preflight và runner #18; scripts/sv1_refresh_manifest.py: chỉ cập nhật hash bản draft.
- final_manifest.json và decision_record.md: CHƯA CHỐT, còn phụ thuộc.
- nghiem_thu_issues.csv: rà 26 issue; dong_gop_cho_xac_nhan.csv: mẫu trống chờ xác nhận.

## Chuẩn bị đánh giá cuối sau khi đủ bàn giao
1. Nhận LR/DT, GB/SVM đúng môi trường, metadata train/CV và ngưỡng. Chốt quy tắc cụ thể trong decision_record.md.
2. Cập nhật đúng đường dẫn model/CV; trained_rows và training_data_sha256 chỉ điền theo metadata người bàn giao đã kiểm tra. Nếu CV dùng tên khác hai placeholder lr_cv_metrics/dt_cv_metrics, sửa manifest đúng tên thật.
3. Chạy python scripts/sv1_refresh_manifest.py khi ready_for_final_test=false. Script chỉ tính hash, không tạo bằng chứng huấn luyện hay xác nhận thay chủ mô hình.
4. Chỉ khi đủ bàn giao và xác nhận phạm vi, đổi confirmations thành true, ready_for_final_test=true rồi commit/push toàn bộ code, manifest, rule/evidence và model. Không đưa kết quả test vào bước lựa chọn này.
5. Chạy python -m src.sv1_final_evaluation --manifest docs/SV1_Quoc/final_manifest.json. Preflight kiểm tệp/model/hash, phiên bản, smoke trên TRAIN và dữ liệu byte-for-byte đã nằm trong HEAD. Phải đạt trước bước 6.
6. Chạy thêm --run đúng một lần để hoàn tất #18. Kết quả trong reports/sv1/final_test; ledger trong reports/sv1/test_ledger. Không xóa kết quả/ledger để chạy lại, kể cả lúc quay video. Nếu crash sau reserve, đọc file và log để xử lý, không tự đánh giá lại các model đã chạy.
7. Runner gọi evaluate_on_test với đúng predict/quy tắc đã chốt và lưu dự đoán, confusion CSV, classification report, metadata. Vẽ hình từ CSV dự đoán đã lưu, không gọi dự đoán test lại chỉ để đổi định dạng. Commit bảng final/ledger cần bàn giao và dẫn link cho nhóm.

Không có --run nào được thực hiện trong gói này. Runner đã kiểm thử các nhánh quyết định/chặn trên dữ liệu giả; chưa kiểm end-to-end trên hold-out thật vì chờ đầu vào. Đây là công cụ chuẩn bị, chưa thay nghiệm thu #18.

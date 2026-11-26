# Model notebooks

Mỗi notebook chạy riêng một model. Tất cả chỉ dùng tập train với Stratified 5-fold CV, seed 42; không chấm hold-out. Cấu hình được chọn trên C rồi giữ cố định khi so sánh A/B/C.

## Thứ tự đề xuất

1. `logistic_regression.ipynb`
2. `decision_tree.ipynb`
3. `random_forest.ipynb`
4. `gradient_boosting.ipynb`
5. `support_vector_machine.ipynb`

Mở kernel `.venv` của repo. Mỗi notebook in chỉ số CV, dự đoán một dòng train để kiểm tra giao diện và nhận xét được tính từ kết quả thực thi.

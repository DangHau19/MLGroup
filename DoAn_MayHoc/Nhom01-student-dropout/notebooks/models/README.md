# Notebook mô hình

Mỗi notebook tái lập một mô hình trên phần train bằng Stratified 5-fold CV, seed 42. Hold-out không được nạp vào các notebook này.

- `Logistic_Regression/logistic_regression.ipynb`
- `Decision_Tree/decision_tree.ipynb`
- `Random_Forest/random_forest.ipynb`
- `Gradient_Boosting/gradient_boosting.ipynb`
- `Support_Vector_Machine/support_vector_machine.ipynb`

Cấu hình được chọn trên C; kết quả so sánh A/B/C nằm trong `notebooks/Experiments/timepoint_abc.ipynb`. Chạy notebook từ repo hoặc từ thư mục notebook; phần khởi tạo tự tìm `src/config.py` để xác định thư mục gốc.

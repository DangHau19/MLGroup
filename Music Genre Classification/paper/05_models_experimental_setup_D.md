# 5. Models & Experimental Setup

**Thành viên D — Trần Đình Bảo Quốc, MSSV 3124411252.** Hoàn thành thực nghiệm trên dữ liệu của nhóm ngày 13/11/2026. Các số liệu trong mục này được sinh từ kết quả huấn luyện thực tế, không phải điểm leaderboard.

## 5.1. Mô hình và baseline tái lập

D chạy lại nguyên `00_baseline_A.py` vào thư mục riêng `evidence/baseline_reproduction_D/` để giữ bằng chứng cũ của A. HistGradientBoosting v1 đạt accuracy **0,558488**, Random Forest đạt **0,534801**, Logistic Regression đạt **0,525840**. HistGB lệch −0,000012 so với mốc làm tròn 0,5585 của A. Lượt tái lập này giữ nguyên cách tiền xử lý toàn cục của mã A; các thực nghiệm mới của D dùng adapter tiền xử lý theo fold như mục 5.2.

Trên v1, D bổ sung kNN (25 láng giềng, trọng số khoảng cách), SVM RBF, ExtraTrees (400 cây), LightGBM, XGBoost, CatBoost và MLP nhỏ. kNN, SVM và MLP đặt trong Pipeline với StandardScaler. Không dùng class weighting hay tái cân bằng lớp. Các mô hình cây nhận ma trận số do `features.build_features` của C tạo, không sử dụng Id, nhãn test, dữ liệu âm thanh hoặc dữ liệu ngoài. CatBoost trong thí nghiệm này dùng đặc trưng số và artist target encoding; không thử chế độ categorical/native text.

## 5.2. Cross-validation và chống rò rỉ

Mọi mô hình của D đi qua `common.cross_validate_model` rồi `common.save_oof`. Outer CV giữ nguyên **StratifiedKFold(n_splits=5, shuffle=True, random_state=42)** từ `common.get_folds`; không đổi thứ tự dòng hoặc mã lớp. Metric chọn mô hình là accuracy; macro F1 là thông tin bổ sung. Accuracy và macro F1 trong các bảng là trung bình của năm fold; std là độ lệch chuẩn quần thể của accuracy giữa các fold, không phải khoảng tin cậy.

`FoldFeatureClassifier` ở `code/models_D.py` nhận dữ liệu thô không có Id/Class, gọi lại hàm tạo đặc trưng của C bằng dữ liệu outer train tương ứng. Median và scaler chỉ fit trên outer train. Riêng tần suất nghệ sĩ/bài hát dùng bản tham chiếu không nhãn gồm train + test, theo quy tắc cho phép của nhóm. Các tần suất được cố định từ bản tham chiếu này để không phụ thuộc kích thước batch dự đoán; không đếm lặp outer train khi xây dựng dữ liệu phụ. Cờ nghệ sĩ đã biết được xác định từ outer train.

Khi thêm artist target encoding, adapter dùng `ArtistTargetEncoder(smoothing=20)` của C. Phần outer train được mã hóa bằng **Stratified 4-fold OOF nội bộ, shuffle=True, seed=42**. Outer validation và test chỉ được transform bằng encoder fit trên toàn bộ outer train. Nghệ sĩ chưa biết nhận prior của outer train; trong inner OOF, prior chỉ lấy từ inner train. Cách này ngăn mỗi dòng tự truyền nhãn vào đặc trưng của chính mình. D giữ nguyên file `features.py` và encoder của C.

Boosting dùng số vòng cố định của từng cấu hình, không early stop bằng outer validation. HistGB dùng early stopping nội bộ theo mặc định thư viện. MLP có tối đa 80 epoch, batch 256, early stopping theo cross-entropy trên 10% outer train được tách stratified với seed 42, patience 10; khôi phục trọng số epoch tốt nhất. MLP gồm các lớp ReLU, inverted dropout và softmax 11 lớp, tối ưu Adam với decoupled weight decay (AdamW), gradient clipping norm 5. MLP được triển khai bằng NumPy; finite-difference gradient check và bài thử học trên dữ liệu tổng hợp đã đạt.

## 5.3. Tuning và chuyển sang v2

D dùng **Optuna RandomSampler(seed=42)**: 3 cấu hình mới cho mỗi họ LightGBM, XGBoost, CatBoost, SVM và MLP, tổng cộng **15 cấu hình / 75 outer-fold fits**, ngoài baseline. Mỗi trial chạy đủ năm fold; objective chỉ là mean accuracy. Xác suất test được xuất để bàn giao, không dùng trong objective hoặc lựa chọn cấu hình. Giữ baseline nếu tuning không cải thiện; MLP thuộc trường hợp này. Study SQLite, toàn bộ trial CSV và cấu hình được chọn nằm trong `evidence/models_D/`.

Không gian tìm kiếm:

- LightGBM: num_leaves {7,15,31,63}; learning_rate {0.025,0.05,0.08}; n_estimators {200,400,650}; min_child_samples {20,50,100}; subsample {0.8,1} với subsample_freq=1; colsample_bytree {0.8,1}; reg_lambda {0,2,10}.
- XGBoost: max_depth {3,4,5}; learning_rate {0.03,0.06,0.1}; n_estimators {200,400,650}; subsample và colsample_bytree {0.75,0.9,1}; tree_method=hist.
- CatBoost: depth {4,5,6}; learning_rate {0.03,0.06,0.1}; l2_leaf_reg {3,10,20}; iterations {300,500,800}.
- SVM: C {1,5,20}; gamma {0.003,0.01,0.03,scale}; probability=True, random_state=42 cho xác suất phục vụ ensemble.
- MLP: lớp ẩn {(64,), (64,32), (128,64)}; dropout {0,0.15,0.3}; weight_decay {0,0.001,0.01}; learning_rate {0.0005,0.001,0.002}.

Đây là lấy mẫu ngẫu nhiên với ngân sách nhỏ, không duyệt hết tích Descartes và không chứng minh cấu hình tối ưu toàn cục. Sau tuning trên v1, D giữ siêu tham số đã chọn để thử ba boosting trên v2, rồi thử v2 + TE cho bảy họ mô hình. Không tuning tiếp trên test hoặc leaderboard.

## 5.4. Bảng 2 — Siêu tham số mô hình bàn giao

Tất cả bảy cấu hình dưới đây dùng **v2 + TE**, smoothing 20 và inner 4-fold như trên. Các tham số không liệt kê giữ mặc định theo phiên bản thư viện đã khóa. Mô hình CPU dùng tối đa 4 luồng; seed mô hình là 42. Bảng CSV đầy đủ và JSON chứa tham số estimator trước fit được lưu kèm kết quả; các thư viện boosting tự suy ra bài toán đa lớp từ 11 nhãn.

| Họ mô hình | Cấu hình bàn giao | Siêu tham số | CV accuracy |
| --- | --- | --- | --- |
| catboost | catboost_v2_te_selected_D | depth=4; learning_rate=0.1; l2_leaf_reg=20.0; iterations=800 | 0.654765 |
| lightgbm | lightgbm_v2_te_selected_D | num_leaves=15; learning_rate=0.025; n_estimators=200; min_child_samples=50; subsample=0.8; subsample_freq=1; colsample_bytree=1.0; reg_lambda=0.0 | 0.654001 |
| xgboost | xgboost_v2_te_selected_D | max_depth=4; learning_rate=0.03; n_estimators=400; subsample=1.0; colsample_bytree=0.75; tree_method=hist | 0.651084 |
| histgb | histgb_v2_te_selected_D | max_iter=150; learning_rate=0.08; max_leaf_nodes=31; early_stopping=auto | 0.641498 |
| mlp | mlp_v2_te_selected_D | hidden_layers=[64, 32]; dropout=0.15; weight_decay=0.001; learning_rate=0.001; max_epochs=80; patience=10 | 0.631495 |
| svm | svm_v2_te_selected_D | C=1.0; gamma=0.01; probability=True | 0.629063 |
| extratrees | extratrees_v2_te_selected_D | n_estimators=400; min_samples_leaf=2 | 0.616351 |

## 5.5. Kết quả kiểm chứng và bàn giao

D đã chạy **32 cấu hình**, tương ứng 160 outer-fold fits (chưa tính lượt tái lập A, inner target encoding và calibration của SVM). Mỗi cấu hình có OOF shape **(14396,11)**, test probability shape **(3600,11)** và JSON CV. Xác suất test là trung bình của năm outer models. Kiểm tra hữu hạn, miền [0,1], tổng hàng bằng 1, thứ tự lớp 0–10 và tính lại điểm từng fold từ OOF đều đạt cho toàn bộ 32 cấu hình.

| Tag dùng với common.load_oof | Accuracy | Std (5 folds) | Macro F1 |
| --- | --- | --- | --- |
| `catboost_v2_te_selected_D` | 0.654765 | 0.003912 | 0.700522 |
| `lightgbm_v2_te_selected_D` | 0.654001 | 0.007181 | 0.706566 |
| `xgboost_v2_te_selected_D` | 0.651084 | 0.003745 | 0.702729 |
| `histgb_v2_te_selected_D` | 0.641498 | 0.004274 | 0.692461 |
| `mlp_v2_te_selected_D` | 0.631495 | 0.005800 | 0.671531 |
| `svm_v2_te_selected_D` | 0.629063 | 0.004937 | 0.665659 |
| `extratrees_v2_te_selected_D` | 0.616351 | 0.005037 | 0.656789 |

Mô hình đơn dẫn đầu là **catboost**, mean CV accuracy **0.654765**, std **0.003912**, mean macro F1 **0.700522**. So với mốc 0,5585 của A, chênh lệch là **9.63 điểm phần trăm**. Phần tăng này kết hợp cả thay đổi mô hình và đặc trưng; không diễn giải toàn bộ là tác dụng của tuning. Chênh lệch rất nhỏ giữa CatBoost và LightGBM chưa chứng minh khác biệt có ý nghĩa thống kê.

Kết quả phục vụ lựa chọn cấu hình: tuning và báo cáo dùng chung outer folds, nên có thiên lệch lựa chọn; chưa phải ước lượng độc lập sau tuning hoặc nested hyperparameter CV. Stratified CV cho phép một nghệ sĩ xuất hiện ở cả train và validation; chưa kiểm tra tổng quát hóa tới nghệ sĩ mới bằng GroupKFold. Các tần suất không nhãn train+test là thiết lập transductive được nhóm cho phép, không phải quy trình hoàn toàn inductive. Không có điểm Kaggle mới cho các cấu hình D; không suy đoán public/private score.

E nhận bảy bộ xác suất tốt nhất theo từng họ, file nhãn/thứ tự dòng/fold và manifest SHA-256 tại `handoff_D_to_E.md`. Chỉ 21 tệp kết quả của bảy mô hình này được giữ rời; kết quả các cấu hình còn lại và đầu ra baseline tái lập được gộp tại `evidence/models_D/intermediate_D.zip`. Audit vẫn đọc và kiểm tra đủ 32 cấu hình trực tiếp từ ZIP. E tiếp tục ensemble và phân tích lỗi; D không chọn trọng số ensemble trong phần này.

## Tài liệu phương pháp để A tích hợp

- [LightGBM — Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
- [XGBoost — Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).
- [CatBoost — Common parameters](https://catboost.ai/docs/en/references/training-parameters/common).
- [scikit-learn — Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html).
- [Optuna — RandomSampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html).
- Kingma & Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), ICLR 2015.
- Loshchilov & Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), ICLR 2019.

Các tài liệu trên dùng để đối chiếu thuật toán/tham số, không bổ sung dữ liệu huấn luyện. Mã D không sao chép notebook Kaggle.

"""Build D's paper section, parameter table and E handoff from audited results."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

import common
from importlib import import_module

audit_all = import_module("03_audit_D").audit_all
OUT = common.ROOT / "evidence" / "models_D"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown_table(columns, rows):
    return "\n".join(["| " + " | ".join(columns) + " |",
                      "| " + " | ".join(["---"] * len(columns)) + " |"] +
                     ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def short_params(meta, family):
    keys = {
        "lightgbm": ["num_leaves", "learning_rate", "n_estimators", "min_child_samples", "subsample", "subsample_freq", "colsample_bytree", "reg_lambda"],
        "xgboost": ["max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree", "tree_method"],
        "catboost": ["depth", "learning_rate", "l2_leaf_reg", "iterations"],
        "svm": ["svc__C", "svc__gamma", "svc__probability"],
        "mlp": ["smallmlpclassifier__hidden_layers", "smallmlpclassifier__dropout", "smallmlpclassifier__weight_decay", "smallmlpclassifier__learning_rate", "smallmlpclassifier__max_epochs", "smallmlpclassifier__patience"],
        "extratrees": ["n_estimators", "min_samples_leaf"],
        "histgb": ["max_iter", "learning_rate", "max_leaf_nodes", "early_stopping"],
    }[family]
    return {key.split("__")[-1]: meta["model_params"][key] for key in keys}


def build_report():
    table = audit_all()
    train, test, sample = common.load_data()
    y, classes = common.encode_labels(train[common.LABEL_COL])
    assert test[common.ID_COL].is_unique and train[common.ID_COL].is_unique
    np.testing.assert_array_equal(test[common.ID_COL], sample[common.ID_COL])
    assert not set(train[common.ID_COL]).intersection(test[common.ID_COL])
    selected = table.groupby("family", sort=False).head(1).copy()
    # kNN is a diagnostic baseline; hand off the seven stronger model families.
    selected = selected[selected.family != "knn"].reset_index(drop=True)
    assert len(selected) >= 6
    assert {"lightgbm", "xgboost", "catboost"}.issubset(set(selected.family))
    assert {"svm", "mlp"}.intersection(set(selected.family))
    best = selected.iloc[0]
    assert best.cv_accuracy > 0.5585
    selected.to_csv(OUT / "selected_models_D.csv", index=False)
    np.save(OUT / "y_encoded_D.npy", y)
    np.save(OUT / "classes_D.npy", classes)
    np.save(OUT / "train_ids_D.npy", train[common.ID_COL].to_numpy())
    np.save(OUT / "test_ids_D.npy", test[common.ID_COL].to_numpy())
    fold_ids = np.empty(len(y), dtype=int)
    for k, (_, va) in enumerate(common.get_folds(y)):
        fold_ids[va] = k
    np.save(OUT / "fold_ids_D.npy", fold_ids)
    param_rows, handoff_rows = [], []
    manifest_models = []
    for row in selected.itertuples():
        meta_path = common.OOF_DIR / f"{row.model}_cv.json"
        meta = json.loads(meta_path.read_text())
        params = short_params(meta, row.family)
        text_params = "; ".join(f"{key}={value}" for key, value in params.items())
        param_rows.append([row.family, row.model, text_params, f"{row.cv_accuracy:.6f}"])
        handoff_rows.append([f"`{row.model}`", f"{row.cv_accuracy:.6f}", f"{row.cv_std:.6f}", f"{row.macro_f1:.6f}"])
        files = [common.OOF_DIR / f"{row.model}_{s}" for s in ("oof.npy", "test.npy", "cv.json")]
        manifest_models.append({"tag": row.model, "family": row.family,
                                "files": {str(p.relative_to(common.ROOT)): sha(p) for p in files}})
    pd.DataFrame(param_rows, columns=["family", "model", "parameters", "cv_accuracy"]).to_csv(OUT / "hyperparameters_D.csv", index=False)
    params_md = markdown_table(["Họ mô hình", "Cấu hình bàn giao", "Siêu tham số", "CV accuracy"], param_rows)
    (common.ROOT / "paper" / "table_02_hyperparameters_D.md").write_text("# Bảng 2. Siêu tham số mô hình — D\n\n" + params_md + "\n", encoding="utf-8")
    results_md = markdown_table(["Tag dùng với common.load_oof", "Accuracy", "Std (5 folds)", "Macro F1"], handoff_rows)
    paper = f"""# 5. Models & Experimental Setup

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

- LightGBM: num_leaves {{7,15,31,63}}; learning_rate {{0.025,0.05,0.08}}; n_estimators {{200,400,650}}; min_child_samples {{20,50,100}}; subsample {{0.8,1}} với subsample_freq=1; colsample_bytree {{0.8,1}}; reg_lambda {{0,2,10}}.
- XGBoost: max_depth {{3,4,5}}; learning_rate {{0.03,0.06,0.1}}; n_estimators {{200,400,650}}; subsample và colsample_bytree {{0.75,0.9,1}}; tree_method=hist.
- CatBoost: depth {{4,5,6}}; learning_rate {{0.03,0.06,0.1}}; l2_leaf_reg {{3,10,20}}; iterations {{300,500,800}}.
- SVM: C {{1,5,20}}; gamma {{0.003,0.01,0.03,scale}}; probability=True, random_state=42 cho xác suất phục vụ ensemble.
- MLP: lớp ẩn {{(64,), (64,32), (128,64)}}; dropout {{0,0.15,0.3}}; weight_decay {{0,0.001,0.01}}; learning_rate {{0.0005,0.001,0.002}}.

Đây là lấy mẫu ngẫu nhiên với ngân sách nhỏ, không duyệt hết tích Descartes và không chứng minh cấu hình tối ưu toàn cục. Sau tuning trên v1, D giữ siêu tham số đã chọn để thử ba boosting trên v2, rồi thử v2 + TE cho bảy họ mô hình. Không tuning tiếp trên test hoặc leaderboard.

## 5.4. Bảng 2 — Siêu tham số mô hình bàn giao

Tất cả bảy cấu hình dưới đây dùng **v2 + TE**, smoothing 20 và inner 4-fold như trên. Các tham số không liệt kê giữ mặc định theo phiên bản thư viện đã khóa. Mô hình CPU dùng tối đa 4 luồng; seed mô hình là 42. Bảng CSV đầy đủ và JSON chứa tham số estimator trước fit được lưu kèm kết quả; các thư viện boosting tự suy ra bài toán đa lớp từ 11 nhãn.

{params_md}

## 5.5. Kết quả kiểm chứng và bàn giao

D đã chạy **{len(table)} cấu hình**, tương ứng {len(table) * 5} outer-fold fits (chưa tính lượt tái lập A, inner target encoding và calibration của SVM). Mỗi cấu hình có OOF shape **(14396,11)**, test probability shape **(3600,11)** và JSON CV. Xác suất test là trung bình của năm outer models. Kiểm tra hữu hạn, miền [0,1], tổng hàng bằng 1, thứ tự lớp 0–10 và tính lại điểm từng fold từ OOF đều đạt cho toàn bộ {len(table)} cấu hình.

{results_md}

Mô hình đơn dẫn đầu là **{best.family}**, mean CV accuracy **{best.cv_accuracy:.6f}**, std **{best.cv_std:.6f}**, mean macro F1 **{best.macro_f1:.6f}**. So với mốc 0,5585 của A, chênh lệch là **{(best.cv_accuracy - 0.5585) * 100:.2f} điểm phần trăm**. Phần tăng này kết hợp cả thay đổi mô hình và đặc trưng; không diễn giải toàn bộ là tác dụng của tuning. Chênh lệch rất nhỏ giữa CatBoost và LightGBM chưa chứng minh khác biệt có ý nghĩa thống kê.

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
"""
    (common.ROOT / "paper" / "05_models_experimental_setup_D.md").write_text(paper, encoding="utf-8")
    dataset_hashes = {str(p.relative_to(common.ROOT)): sha(p) for p in common.DATA_DIR.glob("*.csv") if p.name in ("train.csv", "test.csv", "sample_submission.csv")}
    source_paths = [common.ROOT / "code" / n for n in ("common.py", "features.py", "artist_target_encoder_C.py", "models_D.py", "mlp_D.py", "03_models_tuning_D.py", "03_audit_D.py")]
    manifest = {"date": "2026-11-13", "seed": 42, "n_folds": 5, "inner_te_folds": 4,
                "metric": "accuracy", "classes": classes.tolist(), "n_train": len(train), "n_test": len(test),
                "python": platform.python_version(), "platform": platform.platform(),
                "dataset_sha256": dataset_hashes,
                "source_sha256": {str(p.relative_to(common.ROOT)): sha(p) for p in source_paths},
                "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost", "optuna", "nbclient")},
                "models": manifest_models}
    (OUT / "handoff_manifest_D.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    handoff = f"""# Bàn giao D → E

Trần Đình Bảo Quốc — 3124411252. Dữ liệu và seed theo quy ước nhóm. Đã chạy xong baseline, boosting, tuning và v2 theo đúng thứ tự phân công. Bàn giao **7 họ mô hình**, gồm LightGBM, XGBoost, CatBoost, HistGB, MLP, SVM, ExtraTrees.

{results_md}

Mỗi tag có đủ `evidence/cv_logs/<tag>_oof.npy`, `<tag>_test.npy`, `<tag>_cv.json`. OOF (14396,11) giữ nguyên thứ tự train; test (3600,11) giữ nguyên thứ tự test; cột xác suất tương ứng `classes=[0,1,...,10]`. Test là trung bình xác suất năm outer models. Tất cả tag trên dùng v2 + TE và cùng outer folds seed 42. Không trộn thứ tự dòng hoặc tự mã hóa lại nhãn.

```python
import sys
sys.path.insert(0, 'code')  # chạy tại thư mục dự án
from common import load_oof
oof, test_proba = load_oof('{best.model}')
```

`evidence/models_D/` có:

- `selected_models_D.csv`: bảy mô hình bàn giao; `all_results_D.csv`: toàn bộ {len(table)} cấu hình (có cả baseline và trial, không coi các trial là kết quả đánh giá độc lập).
- `y_encoded_D.npy`, `classes_D.npy`, `train_ids_D.npy`, `test_ids_D.npy`, `fold_ids_D.npy` (fold 0–4), để kiểm tra alignment trước ensemble.
- `handoff_manifest_D.json`: SHA-256 dữ liệu, mã nguồn, ba file của mỗi mô hình; phiên bản thư viện và thứ tự lớp.
- `hyperparameters_D.csv`, các `*_trials.csv`, `selected_v1.json`, `tuning.sqlite3` cho toàn bộ quá trình chọn cấu hình.
- `intermediate_D.zip`: các bộ kết quả không được chọn và đầu ra baseline tái lập. Các tệp này được nén gọn; notebook/audit đọc trực tiếp, không cần giải nén. Chỉ 21 tệp kết quả của bảy mô hình bàn giao được giữ rời để `common.load_oof` dùng ngay.

Mô hình đơn tốt nhất: **{best.model}, accuracy {best.cv_accuracy:.6f}**. Đây là CV sau lựa chọn trên cùng folds, chưa phải đánh giá độc lập sau tuning; không khẳng định ensemble sẽ cải thiện. Không có điểm leaderboard mới cho mô hình D. Không dùng public score để chọn. Tần suất train+test không nhãn được dùng theo quy ước nhóm; TE luôn trong fold và nested OOF ở phần train. So với baseline A 0,5585, mô hình tốt nhất tăng {(best.cv_accuracy - 0.5585) * 100:.2f} điểm phần trăm.

Chạy tái lập: xem phần D trong README và `code/03_models_tuning_D.ipynb`. Mục 5 của paper: `paper/05_models_experimental_setup_D.md`; Bảng 2: `paper/table_02_hyperparameters_D.md`.

E cần tự kiểm chứng lựa chọn/trọng số ensemble bằng phương pháp tách OOF theo phân công. Không xem CV được tối ưu trên chính OOF là điểm đánh giá độc lập. Phần này là bàn giao file trong workspace, chưa gửi tin nhắn nhóm hoặc nộp Kaggle.
"""
    (common.ROOT / "handoff_D_to_E.md").write_text(handoff, encoding="utf-8")
    print(f"Built paper section 5, table 2, handoff manifest for {len(selected)} families.")
    return selected


if __name__ == "__main__":
    print(build_report().to_string(index=False))

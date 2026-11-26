"""Create one reproducible train/CV notebook for each final model."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "models"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "logistic_regression": ("LR", "Logistic Regression", "lr_best.joblib"),
    "decision_tree": ("DT", "Decision Tree", "dt_best.joblib"),
    "random_forest": ("RF", "Random Forest", "rf_best.joblib"),
    "gradient_boosting": ("GB", "Gradient Boosting", "gb_best.joblib"),
    "support_vector_machine": ("SVM", "Support Vector Machine", "svm_best_proba.joblib"),
}


def cell(kind, source):
    return {"cell_type": kind, "metadata": {}, "source": source.splitlines(True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {})}


for slug, (key, label, artifact) in MODELS.items():
    cells = [
        cell("markdown", f"""# {label}

Notebook này tái lập kết quả **train/CV trên tập C** bằng pipeline đã khóa cho mô hình {label}. Siêu tham số được chọn bằng CV trên train; hold-out chưa được nạp hoặc chấm trong notebook này.

**Mục tiêu:** ghi lại quy trình, chỉ số và nhận xét cho riêng mô hình. CV này là đánh giá sau lựa chọn siêu tham số, không phải điểm test độc lập.
"""),
        cell("markdown", """## 1 Thiết lập

- Phân chia train/hold-out: phân tầng, 80/20, seed 42; chỉ lấy phần train.
- Đánh giá: Stratified 5-fold CV, shuffle=True, seed 42.
- Độ đo chính: Macro-F1; kèm balanced accuracy, accuracy, recall Dropout và F1 Enrolled.
- Quy tắc đầu ra: `native_predict`; không áp ngưỡng tùy chỉnh trong bảng so sánh chính.
"""),
        cell("code", f"""from pathlib import Path
import json
import warnings

import joblib
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().parent if Path.cwd().name == "models" else Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
import os, sys
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src import config
from src.data import get_train_test
from src.evaluate import cross_validate_model
from src.predict import predict_one

MODEL_KEY = "{key.lower()}"
MODEL_NAME = "{label}"
MODEL_PATH = config.MODELS_DIR / "{artifact}"
assert MODEL_PATH.is_file(), MODEL_PATH
X_train, _, y_train, _ = get_train_test("C")  # bỏ qua hold-out
model = joblib.load(MODEL_PATH)
assert list(model.classes_) == config.CLASSES
print(f"Model: {{MODEL_NAME}} | feature set C | train rows: {{len(X_train)}} | artifact: {{MODEL_PATH.name}}")
"""),
        cell("markdown", "## 2 Kết quả CV trên train\n\nÔ tiếp theo fit clone của pipeline trên từng fold. Tập hold-out không được sử dụng."),
        cell("code", f"""with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="The `probability` parameter")
    result = cross_validate_model(model, X_train, y_train, name=MODEL_NAME,
                                  feature_set="C", log=False, n_jobs=1)
metrics = pd.DataFrame([result])
display(metrics[["f1_macro_mean", "f1_macro_std", "balanced_accuracy_mean",
                 "accuracy_mean", "recall_dropout_mean", "f1_enrolled_mean",
                 "train_f1_macro_mean"]].round(4))
"""),
        cell("markdown", "## 3 Ví dụ dự đoán và nhận xét\n\nVí dụ lấy từ một dòng train để kiểm tra hợp đồng nhãn/xác suất. Đây không phải mẫu test."),
        cell("code", """example = X_train.iloc[0].to_dict()
prediction = predict_one(example, model=MODEL_KEY, feature_set="C")
display(prediction)

macro = float(metrics.loc[0, "f1_macro_mean"])
macro_sd = float(metrics.loc[0, "f1_macro_std"])
recall_dropout = float(metrics.loc[0, "recall_dropout_mean"])
f1_enrolled = float(metrics.loc[0, "f1_enrolled_mean"])
gap = float(metrics.loc[0, "train_f1_macro_mean"] - macro)
print("Nhận xét:")
print(f"- Macro-F1 CV = {macro:.4f} ± {macro_sd:.4f}; Recall Dropout = {recall_dropout:.4f}; F1 Enrolled = {f1_enrolled:.4f}.")
if gap > 0.10:
    print(f"- Chênh lệch train–CV = {gap:.4f}; có dấu hiệu overfit cần nêu khi so sánh.")
else:
    print(f"- Chênh lệch train–CV = {gap:.4f}; vẫn đối chiếu với biến động giữa các fold.")
print("- CV sau lựa chọn cấu hình trên train có thể lạc quan; không thay thế hold-out.")
"""),
        cell("markdown", """## 4 Nguồn và phạm vi

Giao thức và kết quả A/B/C nằm trong `reports/timepoint_cv.csv`. Kết quả train/CV cho cấu hình C được lưu ở bảng metrics riêng của từng model; GB/SVM có thêm `reports/sv4_rebuild_cv.csv`. Không điều chỉnh mô hình/ngưỡng theo hold-out. Kết quả cuối chỉ được thêm sau khi manifest được khóa và preflight đạt.
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3.12.10 (.venv)",
                                      "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12.10"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    for index, notebook_cell in enumerate(notebook["cells"]):
        notebook_cell["id"] = f"{slug[:4]}-{index:02d}"
    (OUT / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8")

readme = """# Model notebooks

Mỗi notebook chạy riêng một model. Tất cả chỉ dùng tập train với Stratified 5-fold CV, seed 42; không chấm hold-out. Cấu hình được chọn trên C rồi giữ cố định khi so sánh A/B/C.

## Thứ tự đề xuất

1. `logistic_regression.ipynb`
2. `decision_tree.ipynb`
3. `random_forest.ipynb`
4. `gradient_boosting.ipynb`
5. `support_vector_machine.ipynb`

Mở kernel `.venv` của repo. Mỗi notebook in chỉ số CV, dự đoán một dòng train để kiểm tra giao diện và nhận xét được tính từ kết quả thực thi.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
print(f"Wrote {len(MODELS)} model notebooks to {OUT}")

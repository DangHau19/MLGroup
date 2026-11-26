"""GB/SVM model build: dựng lại GB/SVM từ tham số tốt nhất (không tuning lại), CV lại, lưu mô hình + metadata.

Chạy từ thư mục chứa README:
    .\\.venv\\Scripts\\python.exe -X utf8 scripts/build_gb_svm_models.py
GHI ĐÈ models/gb_best.joblib, models/svm_best.joblib; GHI MỚI models/svm_best_proba.joblib,
models/*_best.meta.json, reports/sv4_rebuild_cv.csv. Giữ bản cũ trước khi chạy.
Không dùng tập test.
"""
import hashlib
import json
import platform
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from src import config
from src.data import get_train_test
from src.evaluate import cross_validate_model
from src.models.proba import GB_BEST, SVM_BEST, build_gb, build_svm

CMD = r".\.venv\Scripts\python.exe -X utf8 scripts/build_gb_svm_models.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    X_train, _X_test, y_train, _y_test = get_train_test("C")
    del _X_test, _y_test
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.GB_SVM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [("gb", "GradientBoosting(best)", build_gb("C"), GB_BEST, "notebooks/Experiments/Logs/Experiment_08_GB_SVM/gb_search_results.csv", "RandomizedSearchCV n_iter=12"),
            ("svm", "SVM(best)", build_svm("C", False), SVM_BEST, "notebooks/Experiments/Logs/Experiment_08_GB_SVM/svm_search_results.csv", "GridSearchCV"),
            ("svm_proba", "SVM(best, probability=True)", build_svm("C", True), SVM_BEST, "notebooks/Experiments/Logs/Experiment_08_GB_SVM/svm_search_results.csv", "GridSearchCV; probability=True thêm sau tuning")]
    rows = []
    for key, label, pipe, params, src, how in jobs:
        r = cross_validate_model(pipe, X_train, y_train, name=label, feature_set="C", log=False)
        rows.append(r)
        pipe.fit(X_train, y_train)
        out = config.MODELS_DIR / f"{key if key != 'svm_proba' else 'svm'}_best{'_proba' if key == 'svm_proba' else ''}.joblib"
        joblib.dump(pipe, out)
        clf = pipe.named_steps["clf"]
        meta = dict(
            name=label, file=out.name, feature_set="C", train_rows=int(len(X_train)),
            classes_=[str(c) for c in clf.classes_], has_predict_proba=bool(hasattr(clf, "predict_proba") and (key != "svm")),
            params=params, random_state=config.RANDOM_STATE, search=how, search_source=src,
            cv="StratifiedKFold(5, shuffle=True, seed=42) trên train (src.evaluate.get_cv)", scoring="f1_macro",
            cv_result={k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
            pipeline="prep(CurricularFeatures -> ColumnTransformer: OneHot+StandardScaler" + ("" if key != "gb" else " (scale=False)") + ") -> clf",
            scaling_inside_pipeline=key != "gb",
            data_file_sha256=sha(config.DATA_FILE),
            versions=dict(python=platform.python_version(), scikit_learn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, joblib=joblib.__version__),
            created=str(date.today()), command=CMD,
            load="joblib.load('models/" + out.name + "')  # cần cùng phiên bản scikit-learn; nếu lỗi, dựng lại bằng src.models.proba",
        )
        (config.MODELS_DIR / f"{out.stem}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(label, {k: round(v, 4) for k, v in r.items() if isinstance(v, float) and ("f1_macro" in k or "recall_dropout" in k or "f1_enrolled" in k)})
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "sv4_rebuild_cv.csv", index=False)


if __name__ == "__main__":
    main()

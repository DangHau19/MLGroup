"""Create and execute D's reproducible notebook in a fresh kernel."""
import json
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent


def main():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3.12 (project .venv)", "language": "python"}
    nb.cells = [
        md("""# 03 — Models & Tuning · thành viên D

**Trần Đình Bảo Quốc — 3124411252**. Thực hiện tuần tự theo `PHAN_CONG.md`: môi trường/baseline A → kNN/SVM/ExtraTrees → boosting/MLP → tuning → v2 → bàn giao E.

Tất cả 32 cấu hình đã được huấn luyện thực tế bằng CLI từng bước, với OOF/test/CV lưu trên đĩa. Notebook mặc định **đọc lại các kết quả này và kiểm tra lại**, không nhận là một lượt huấn luyện mới. Đổi `RECOMPUTE=True` để chạy lại toàn bộ baseline và 32 cấu hình từ đầu. Nếu thiếu artifact, từng bước sẽ tự huấn luyện phần thiếu. Không dùng dữ liệu ngoài hoặc leaderboard để tuning.

Chạy bằng `.venv` Python 3.12 sau khi cài `code/requirements_D.txt`. Dữ liệu tên chuẩn nằm trong `data/` (bản sao nguyên byte từ file gốc tên có `(1)`/`(2)`)."""),
        code("""from pathlib import Path
import os, sys, json, hashlib, importlib, importlib.metadata, subprocess
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
ROOT = Path.cwd().resolve()
if not (ROOT / 'PHAN_CONG.md').exists():
    ROOT = ROOT.parent
assert (ROOT / 'PHAN_CONG.md').exists()
sys.path.insert(0, str(ROOT / 'code'))
RECOMPUTE = False  # True = thực sự huấn luyện lại mọi cấu hình, cần thêm thời gian
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
import common
experiments = importlib.import_module('03_models_tuning_D')
audit_module = importlib.import_module('03_audit_D')
report = importlib.import_module('03_report_D')
assert common.SEED == 42 and common.N_FOLDS == 5 and common.PRIMARY_METRIC == 'accuracy'
manifest_path = ROOT / 'evidence/models_D/handoff_manifest_D.json'
if manifest_path.exists() and not RECOMPUTE:
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in {**manifest['dataset_sha256'], **manifest['source_sha256']}.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, f'{relative} changed; set RECOMPUTE=True'
    for name, expected in manifest['packages'].items():
        assert importlib.metadata.version(name) == expected, f'Install requirements_D.txt: {name}'
print('Python:', sys.executable)
print('Chế độ:', 'HUẤN LUYỆN LẠI' if RECOMPUTE else 'ĐỌC CACHE + AUDIT; huấn luyện nếu thiếu')
display(pd.read_csv(ROOT / 'data/train.csv').shape)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
"""),
        md("## 1. Tái lập baseline A\n\nGiữ nguyên mã A và cô lập đường dẫn đầu ra. HistGB v1 tái lập đạt 0,558488, khớp mốc làm tròn 0,5585. Baseline A là lượt kiểm chứng mã cũ; các mô hình D dùng tiền xử lý trong fold."),
        code("""baseline_csv = ROOT / 'evidence/baseline_reproduction_D/comparison.csv'
if RECOMPUTE or not baseline_csv.exists():
    subprocess.run([sys.executable, '-u', str(ROOT / 'code/03_reproduce_baseline_D.py')], cwd=ROOT, check=True)
display(pd.read_csv(baseline_csv))
display(pd.read_json(ROOT / 'evidence/baseline_reproduction_D/environment.json', typ='series'))
"""),
        md("## 2. Baseline bổ sung trên v1\n\nkNN và SVM dùng StandardScaler trong Pipeline. Median chỉ lấy từ outer train; tần suất không nhãn cố định trên train+test theo quy tắc nhóm. Giữ nguyên 5-fold, seed và thứ tự dòng."),
        code("""experiments.baselines(resume=not RECOMPUTE)
display(audit_module.audit_all().query("model.str.endswith('v1_base_D')", engine='python'))
"""),
        md("## 3. Ba boosting và MLP nhỏ trên v1\n\nKhông dùng class weights. Boosting dùng số vòng cố định. MLP NumPy có dropout, AdamW và inner holdout 10% để early stop; không sử dụng outer validation để chọn epoch."),
        code("""experiments.boosting(resume=not RECOMPUTE)
for script in ['03_test_protocol_D.py', '03_test_mlp_D.py']:
    subprocess.run([sys.executable, str(ROOT / 'code' / script)], cwd=ROOT, check=True)
"""),
        md("## 4. Tuning với Optuna\n\nRandomSampler seed 42, ba cấu hình mới/họ + baseline; mỗi trial chạy đủ năm fold qua common. Mục tiêu accuracy; không dùng test score. Tìm kiếm ngân sách nhỏ, không khẳng định tối ưu toàn cục. Giữ baseline khi trial không tốt hơn."),
        code("""experiments.tuning(resume=not RECOMPUTE, n_trials=3)
display(pd.read_json(ROOT / 'evidence/models_D/selected_v1.json').T)
for family in ['lightgbm', 'xgboost', 'catboost', 'svm', 'mlp']:
    print(family)
    display(pd.read_csv(ROOT / f'evidence/models_D/{family}_trials.csv'))
"""),
        md("## 5. Đặc trưng v2 và nested OOF target encoding\n\nGiữ siêu tham số chọn từ v1. Ba boosting được thử v2 thuần; bảy họ được thử v2 + TE. Encoder của C: smoothing 20, inner Stratified 4-fold seed 42 cho outer train; outer validation/test dùng mapping từ outer train. Nghệ sĩ lạ nhận prior. Không fit encoder trước khi chia outer CV."),
        code("""experiments.v2(resume=not RECOMPUTE)
all_results = audit_module.audit_all()
assert len(all_results) == 32
display(all_results)
"""),
        md("## 6. Mục 5, Bảng 2 và bàn giao E\n\nKiểm lại xác suất, điểm từng fold, lớp 0–10, ID/thứ tự dòng. Bàn giao bảy họ khác nhau cùng manifest SHA-256. CV sau tuning dùng lại các folds lựa chọn, chưa phải ước lượng độc lập sau tuning. Không có điểm Kaggle mới. Phần E chịu trách nhiệm ensemble/phân tích lỗi."),
        code("""selected = report.build_report()
display(selected)
display(pd.read_csv(ROOT / 'evidence/models_D/hyperparameters_D.csv'))
manifest = json.loads(manifest_path.read_text())
for item in manifest['models']:
    for relative, expected in item['files'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
assert len(selected) >= 6
assert {'lightgbm','xgboost','catboost'}.issubset(set(selected.family))
assert {'svm','mlp'}.intersection(set(selected.family))
assert selected.iloc[0].cv_accuracy > 0.5585
print('PASS: 32 cấu hình, 7 họ bàn giao; đủ OOF/test/CV; vượt baseline; manifest đúng.')
display(Markdown((ROOT / 'handoff_D_to_E.md').read_text(encoding='utf-8')))
"""),
    ]
    target = ROOT / "code/03_models_tuning_D.ipynb"
    nbf.write(nb, target)
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        nbf.write(nb, target)
    errors = [out for cell in nb.cells if cell.cell_type == "code" for out in cell.outputs if out.output_type == "error"]
    assert not errors
    status = {"notebook": str(target.relative_to(ROOT)), "fresh_kernel": True,
              "mode": "cache verification; all models previously trained sequentially by CLI",
              "code_cells": sum(c.cell_type == "code" for c in nb.cells),
              "executed_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in nb.cells),
              "errors": 0}
    (ROOT / "evidence/models_D/notebook_execution_D.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

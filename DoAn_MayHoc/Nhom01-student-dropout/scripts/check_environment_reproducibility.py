"""-ENV / -REPRO: kiểm tra môi trường từ bản clone sạch + venv mới, ghi log để dán vào Thi_nghiem/Nhat_ky.

Chạy từ thư mục chứa README, bằng Python TRONG .venv mới (đã cài requirements.txt):
    .\\.venv\\Scripts\\python.exe -X utf8 scripts/check_environment_reproducibility.py
    .\\.venv\\Scripts\\python.exe -X utf8 scripts/check_environment_reproducibility.py --write-pins   # ghim requirements.txt theo máy này
Việc script làm: ghi commit và phiên bản, `pip check`, kiểm tra dữ liệu (hình dạng + SHA-256),
nạp từng mô hình trong models/, chạy unittest, chạy thử ví dụ suy luận ngưỡng trên vài dòng TRAIN.
Không đụng tập test, không huấn luyện lại, không ghi đè mô hình. Log: notebooks/Experiments/Logs/Experiment_00_Environment/repro_log.txt
(log/ảnh lưu Drive, không push). Kết quả "ĐẠT" chỉ có nghĩa các bước trên chạy được trên máy đó.
"""
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LINES = []
EXPECTED_PY = "3.12.10"
# gói trực tiếp mà code/notebook đang dùng (import thật), kèm tên pip
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "seaborn",
            "imbalanced-learn", "ucimlrepo", "nbformat", "nbclient", "ipykernel", "jupyter"]


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout + p.stderr).strip()


def main():
    write_pins = "--write-pins" in sys.argv
    ok = {}
    log(f"# Kiểm tra tái lập - {datetime.now():%Y-%m-%d %H:%M}")
    code, out = run(["git", "rev-parse", "HEAD"])
    log(f"commit: {out if code == 0 else 'không đọc được (không phải clone git?)'}")
    code, out = run(["git", "status", "--short"])
    log(f"thay đổi cục bộ: {'không' if code == 0 and not out else out[:300]}")
    log(f"python: {platform.python_version()} ({sys.executable})  | mong đợi {EXPECTED_PY}")
    ok["python"] = platform.python_version() == EXPECTED_PY
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    log("phiên bản: " + json.dumps(versions, ensure_ascii=False))
    ok["packages_installed_core"] = all(versions[p] for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib"])

    code, out = run([sys.executable, "-m", "pip", "check"])
    log(f"pip check: {out}")
    ok["pip_check"] = code == 0

    try:
        from src import config
        from src.data import get_train_test, load_data
        df = load_data()
        sha = hashlib.sha256(config.DATA_FILE.read_bytes()).hexdigest()
        log(f"dữ liệu: shape={df.shape} (mong đợi (4424, 37)); sha256={sha}")
        log("nhãn: " + json.dumps(df[config.TARGET].value_counts().to_dict(), ensure_ascii=False))
        ok["data"] = df.shape == (4424, 37)
    except Exception as e:  # noqa: BLE001
        log(f"dữ liệu: LỖI {type(e).__name__}: {e}"); ok["data"] = False

    import joblib
    for name in ["gb_best", "svm_best", "svm_best_proba", "rf_best"]:
        path = ROOT / "models" / f"{name}.joblib"
        if not path.exists():
            log(f"mô hình {name}: không có file"); continue
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                m = joblib.load(path)
                msg = f"nạp được; classes_={list(m.classes_)}; predict_proba={'có' if hasattr(m, 'predict_proba') else 'KHÔNG'}"
                ok[f"model_{name}"] = True
            except Exception as e:  # noqa: BLE001
                msg = f"KHÔNG nạp được ({type(e).__name__}: {str(e)[:90]}). Dựng lại bằng scripts/build_gb_svm_models.py"
                ok[f"model_{name}"] = False
        extra = [str(x.message)[:110] for x in w if "unpickle" in str(x.message).lower()]
        log(f"mô hình {name}: {msg}" + (f" | CẢNH BÁO phiên bản: {extra[0]}" if extra else ""))

    code, out = run([sys.executable, "-W", "ignore", "-m", "unittest", "discover", "-s", "tests"])
    tail = out.splitlines()[-3:]
    log("unittest: " + " | ".join(tail))
    ok["unittest"] = code == 0

    try:
        from src.threshold import apply_threshold
        cfg = json.loads((ROOT / "models" / "threshold_config.json").read_text(encoding="utf-8"))
        m = joblib.load(ROOT / "models" / "gb_best.joblib")
        Xtr, _Xte, ytr, _yte = get_train_test("C"); del _Xte, _yte
        P = m.predict_proba(Xtr.iloc[:5])
        t = cfg["models"]["gb"]["threshold"]
        log(f"ví dụ suy luận (5 dòng TRAIN, t={t}): classes_={list(m.classes_)}; nhãn={list(apply_threshold(P, m.classes_, t))}")
        ok["inference_example"] = True
    except Exception as e:  # noqa: BLE001
        log(f"ví dụ suy luận: LỖI {type(e).__name__}: {str(e)[:120]}"); ok["inference_example"] = False

    log("tổng hợp: " + json.dumps(ok, ensure_ascii=False))
    log("KẾT LUẬN: " + ("ĐẠT các bước trên" if all(ok.values()) else "CÓ BƯỚC KHÔNG ĐẠT - xem dòng tương ứng"))
    (ROOT / "reports").mkdir(exist_ok=True)
    log_dir = ROOT / "notebooks" / "Experiments" / "Logs" / "Experiment_00_Environment"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "repro_log.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")

    if write_pins:
        if not all(ok.get(k) for k in ["pip_check", "packages_installed_core"]):
            print("Không ghim: pip check hoặc thư viện lõi chưa đạt."); return
        keep = [p for p in PACKAGES if versions[p] and p not in ("jupyter",)]
        text = (f"# Python {platform.python_version()}; ghim bằng scripts/check_environment_reproducibility.py --write-pins sau khi pip check đạt\n"
                + "\n".join(f"{p}=={versions[p]}" for p in keep) + "\n")
        (ROOT / "requirements.txt").write_text(text, encoding="utf-8")
        print("Đã ghi requirements.txt (ghim). Kiểm tra bằng git diff rồi mới commit.")


if __name__ == "__main__":
    main()

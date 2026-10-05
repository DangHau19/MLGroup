"""
merge_results.py (TV1) - Gộp và kiểm tra kết quả của TV3, TV4, TV5.

Chạy từ thư mục gốc dự án:
    python src/merge_results.py                  # yêu cầu đủ 18 dòng
    python src/merge_results.py --allow-partial  # gộp tạm khi chưa nhận đủ bài

Đầu ra: data/processed/feature_selection_results.csv  (+ in bảng xếp hạng)
Cũng dùng được như thư viện:  from src.merge_results import load, ranking
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "data/processed").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/processed"

try:  # ưu tiên cột chuẩn do TV3 định nghĩa
    from src.evaluate import RESULT_COLUMNS
except Exception:
    RESULT_COLUMNS = ["technique", "model", "n_features", "cv_log_rmse", "cv_log_r2",
                      "test_log_rmse", "test_original_rmse", "test_original_mae",
                      "test_original_r2", "cv_seconds", "train_seconds"]

K = 50
MODELS = ["SVR", "RandomForest", "XGBoost"]
TECHNIQUES = ["baseline", "filter_kbest", "filter_mi", "wrapper_rfe", "embedded_lasso", "embedded_rf"]
SOURCES = {"TV3 (baseline)": "data/processed/baseline_results.csv",
           "TV4 (filter+wrapper)": "data/processed/fs_results_filter_wrapper.csv",
           "TV5 (embedded)": "data/processed/fs_results_embedded.csv"}
EXTENDED = ["data/processed/baseline_results_extended.csv", "data/processed/fs_results_filter_wrapper_extended.csv",
            "data/processed/fs_results_embedded_extended.csv"]   # chứa thêm cv_log_rmse_std (nếu có)
OUT = DATA / "feature_selection_results.csv"


def load(allow_partial=False):
    """Trả về (DataFrame gộp, danh sách lỗi)."""
    errs, frames = [], []
    for owner, fname in SOURCES.items():
        p = DATA / fname
        if not p.exists():
            errs.append(f"[{owner}] chưa có {fname}")
            continue
        df = pd.read_csv(p)
        if list(df.columns) != RESULT_COLUMNS:
            errs.append(f"[{owner}] {fname}: cột/thứ tự cột không đúng RESULT_COLUMNS "
                        f"(thiếu {sorted(set(RESULT_COLUMNS)-set(df.columns))}, thừa {sorted(set(df.columns)-set(RESULT_COLUMNS))})")
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS), errs + ["Không có file hợp lệ nào."]
    m = pd.concat(frames, ignore_index=True)
    return m, errs + check(m, not allow_partial)


def check(df, complete):
    e = []
    if sorted(set(df.technique) - set(TECHNIQUES)):
        e.append(f"technique sai tên: {sorted(set(df.technique) - set(TECHNIQUES))}")
    if sorted(set(df.model) - set(MODELS)):
        e.append(f"model sai tên: {sorted(set(df.model) - set(MODELS))}")
    d = df[df.duplicated(["technique", "model"], keep=False)]
    if len(d):
        e.append("trùng (technique, model):\n" + d[["technique", "model"]].to_string(index=False))
    if df.isna().any().any():
        e.append(f"có NaN ở cột: {df.columns[df.isna().any()].tolist()}")
    fs = df[df.technique != "baseline"]
    bad = fs[fs.n_features != K]
    if len(bad):  # mọi kỹ thuật FS phải giữ đúng K feature để so sánh công bằng
        e.append(f"n_features khác K={K}:\n" + bad[["technique", "model", "n_features"]].to_string(index=False))
    if complete:
        miss = {(t, m) for t in TECHNIQUES for m in MODELS} - set(zip(df.technique, df.model))
        if miss:
            e.append(f"thiếu tổ hợp: {sorted(miss)}")
    return e


def load_std():
    """Gộp cv_log_rmse_std từ các file *_extended.csv hiện có (có thể thiếu một số kỹ thuật)."""
    fr = [pd.read_csv(DATA / f)[["technique", "model", "cv_log_rmse_std"]]
          for f in EXTENDED if (DATA / f).exists()]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(columns=["technique", "model", "cv_log_rmse_std"])


def ranking(df):
    out = df.copy()
    base = out[out.technique == "baseline"].set_index("model")["cv_log_rmse"]
    out["pct_vs_baseline"] = (out.cv_log_rmse / out.model.map(base) - 1) * 100  # âm = tốt hơn
    out["rank"] = out.groupby("model").cv_log_rmse.rank(method="min").astype(int)
    out = out.merge(load_std(), on=["technique", "model"], how="left")
    return out.sort_values(["model", "rank"]).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    merged, errs = load(a.allow_partial)
    for x in errs:
        print(" -", x)
    if merged.empty or (errs and not a.allow_partial):
        print("\nChưa ghi file gộp. Trả lại thành viên liên quan để sửa.")
        sys.exit(1)
    merged.to_csv(OUT, index=False)
    print(f"Đã ghi {len(merged)} dòng -> {OUT}\n")
    print(ranking(merged)[["model", "rank", "technique", "n_features", "cv_log_rmse", "cv_log_rmse_std",
                           "pct_vs_baseline", "test_log_rmse", "cv_seconds"]].round(4).to_string(index=False))

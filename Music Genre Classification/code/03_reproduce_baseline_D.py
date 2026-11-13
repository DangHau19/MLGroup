"""Run A's unchanged baseline with isolated output paths, preserving A's files."""
import contextlib
import json
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def main():
    output = common.ROOT / "evidence" / "baseline_reproduction_D"
    output.mkdir(parents=True, exist_ok=True)
    old_oof, old_sub = common.OOF_DIR, common.SUB_DIR
    try:
        common.OOF_DIR, common.SUB_DIR = output, output / "submissions"
        with (output / "run.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                runpy.run_path(str(Path(__file__).with_name("00_baseline_A.py")), run_name="__main__")
        result = common.leaderboard_table()
        result.to_csv(output / "comparison.csv", index=False)
        hist = json.loads((output / "histgb_v1_cv.json").read_text())
        print(result.to_string(index=False), flush=True)
        print(f"HistGB v1 difference from A's 0.5585: {hist['cv_mean'] - 0.5585:+.6f}", flush=True)
    finally:
        common.OOF_DIR, common.SUB_DIR = old_oof, old_sub


if __name__ == "__main__":
    main()

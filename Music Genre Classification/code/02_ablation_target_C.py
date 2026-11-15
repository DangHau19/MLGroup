"""Backward-compatible command that runs C's fold-safe ablation."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    safe_runner = Path(__file__).with_name("02_ablation_foldsafe_C.py")
    runpy.run_path(str(safe_runner), run_name="__main__")

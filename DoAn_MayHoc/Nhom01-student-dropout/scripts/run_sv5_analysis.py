"""Run SV5 notebooks, verify outputs and collect evidence outside the repository.

Usage: .venv/Scripts/python.exe -X utf8 scripts/run_sv5_analysis.py
No hold-out evaluation, shared model overwrite or Git operation is performed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="A new, empty evidence directory outside the repo")
    args = parser.parse_args()
    started = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    run_id = started.strftime("%Y%m%d_SV5_ANALYSIS_%H%M%S")
    output = (args.output_dir or Path.home() / "Documents" / "MLGroup_SV5" / "runs" / run_id).resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("Keep raw logs/evidence outside the repository; choose another --output-dir.")
    if output.exists() and any(output.iterdir()):
        parser.error("Evidence directory is not empty; use a new directory to preserve earlier runs.")
    output.mkdir(parents=True, exist_ok=True)

    import nbformat
    import numpy as np
    import pandas as pd
    from nbclient import NotebookClient
    from src import config
    from src.data import get_train_test

    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True, encoding="utf-8").strip()
    source_files = ["src/config.py", "src/data.py", "src/preprocessing.py", "src/features.py", "src/evaluate.py",
                    "reports/rf_summary.json", "reports/gb_search_results.csv", "reports/svm_search_results.csv",
                    "requirements-sv5.txt", "scripts/run_sv5_analysis.py"]
    names = ["04_timepoint.ipynb", "05_error_analysis.ipynb"]
    previous = output / "previous"
    previous.mkdir()
    for relative in [*["notebooks/" + n for n in names], "reports/sv5_timepoint_cv.csv",
                     "reports/sv5_rf_oof.csv", "reports/sv5_importance.csv"]:
        path = ROOT / relative
        if path.exists():
            shutil.copy2(path, previous / path.name)
    model_hashes = {p.name: sha(p) for p in (ROOT / "models").glob("*.joblib")}
    metadata = {
        "run_id": run_id, "started_at": started.isoformat(), "executor": "Codex hỗ trợ trên máy Triết",
        "git_commit_at_start": git("rev-parse", "HEAD"), "git_status_at_start": git("status", "--short"),
        "python": platform.python_version(), "python_executable": sys.executable,
        "versions": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "nbformat", "nbclient", "ipykernel"]},
        "data_sha256": sha(config.DATA_FILE), "source_sha256": {p: sha(ROOT / p) for p in source_files},
        "notebook_code_sha256": {}, "models": ["RF", "GB", "SVM"], "missing_model_handoffs": ["LR", "DT"],
        "scope": "Train-only 5-fold CV / OOF / validation; parameters selected on C, fixed across A/B/C.",
        "holdout_evaluated": False, "shared_model_hashes_before": model_hashes,
        "checks": {}, "figures": [], "status": "running",
    }
    t0 = time.perf_counter()
    with (output / "run.log").open("w", encoding="utf-8") as log:
        def emit(text: str) -> None:
            print(text, flush=True)
            log.write(text + "\n")
            log.flush()

        def command(label: str, command: list[str]) -> None:
            emit("COMMAND " + subprocess.list2cmdline(command))
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            emit(result.stdout + result.stderr)
            metadata["checks"][label] = {"exit_code": result.returncode}
            if result.returncode:
                raise RuntimeError(label + " failed; see run.log")

        try:
            emit(json.dumps({"run_id": run_id, "started_at": metadata["started_at"], "output": str(output)}, ensure_ascii=False))
            command("pip_check", [sys.executable, "-m", "pip", "check"])
            for name in names:
                path = ROOT / "notebooks" / name
                nb = nbformat.read(path, as_version=4)
                code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
                metadata["notebook_code_sha256"][name] = hashlib.sha256(code.encode()).hexdigest()
                for i, cell in enumerate(nb.cells):
                    if cell.cell_type == "code":
                        compile(cell.source, f"{name}:cell{i}", "exec")
                emit("START " + name)

                def progress(cell, cell_index, **kwargs):
                    if cell.cell_type == "code":
                        emit(f"{name}: cell {cell_index}")

                client = NotebookClient(nb, timeout=3600, resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress)
                try:
                    client.execute()
                finally:
                    nbformat.write(nb, path)
                assert not any(o.output_type == "error" for c in nb.cells if c.cell_type == "code" for o in c.outputs)
                figures = ["01_abc_cv.png"] if name.startswith("04") else ["02_rf_oof_confusion.png", "03_enrolled_errors.png", "04_rf_validation_importance.png"]
                image_index = 0
                for cell in nb.cells:
                    if cell.cell_type != "code":
                        continue
                    for item in cell.outputs:
                        if item.output_type == "stream":
                            log.write(item.text)
                        if "image/png" in item.get("data", {}):
                            target = output / figures[image_index]
                            target.write_bytes(base64.b64decode(item.data["image/png"]))
                            metadata["figures"].append({"file": target.name, "notebook": name, "cell_id": cell.id, "scope": metadata["scope"]})
                            image_index += 1
                assert image_index == len(figures), (name, image_index)
                emit("DONE " + name)

            cv = pd.read_csv(ROOT / "reports/sv5_timepoint_cv.csv")
            oof = pd.read_csv(ROOT / "reports/sv5_rf_oof.csv")
            importance = pd.read_csv(ROOT / "reports/sv5_importance.csv")
            X, holdout, y, _ = get_train_test("C")
            assert len(cv) == 45 and cv.groupby(["model", "feature_set"]).size().eq(5).all()
            assert cv.data_sha256.eq(metadata["data_sha256"]).all()
            assert len(oof) == len(X) and oof.row_id.is_unique and set(oof.row_id) == set(X.index)
            assert not set(oof.row_id) & set(holdout.index)
            assert (oof.actual.to_numpy() == y.loc[oof.row_id].to_numpy()).all()
            assert np.allclose(oof.filter(like="prob_").sum(axis=1), 1)
            assert len(importance) == 375 and importance.groupby(["method", "feature"]).size().eq(5).all()
            assert not any(df.isna().any().any() for df in [cv, oof, importance])
            metadata["checks"]["output_integrity"] = {"passed": True, "cv_rows": len(cv), "oof_rows": len(oof), "importance_rows": len(importance), "reserved_holdout_rows": len(holdout)}
            metadata["cv_f1_macro_mean"] = {f"{model}_{fs}": float(v) for (model, fs), v in cv.groupby(["model", "feature_set"]).f1_macro.mean().items()}
            command("existing_tests", [sys.executable, "-X", "utf8", "-B", "-m", "unittest", "discover", "-s", "tests", "-v"])

            # Technical cross-check of Quốc's existing RF artifact, not final issue acceptance.
            import joblib
            rf = joblib.load(ROOT / "models/rf_best.joblib")
            summary = json.loads((ROOT / "reports/rf_summary.json").read_text(encoding="utf-8"))
            assert summary["data_sha256"] == metadata["data_sha256"]
            assert all(rf.get_params()[key] == value for key, value in summary["best_params"].items())
            probabilities = rf.predict_proba(X.iloc[:3])
            assert set(rf.classes_) == set(config.CLASSES) and probabilities.shape == (3, 3)
            assert np.allclose(probabilities.sum(axis=1), 1)
            metadata["checks"]["quoc_rf_technical_smoke"] = {"passed": True, "rows_from_train": X.index[:3].tolist(), "classes": rf.classes_.tolist(), "predictions": rf.predict(X.iloc[:3]).tolist(), "note": "Only load/config/schema/probability checks; not acceptance of issue #31 or evaluation of test."}
            assert model_hashes == {p.name: sha(p) for p in (ROOT / "models").glob("*.joblib")}
            for name in ["sv5_timepoint_cv.csv", "sv5_rf_oof.csv", "sv5_importance.csv"]:
                shutil.copy2(ROOT / "reports" / name, output / name)
            metadata["status"] = "passed"
            emit("PASS: notebooks, output integrity, existing tests and RF technical smoke. No hold-out evaluation.")
        except Exception as exc:
            metadata["status"] = "failed"
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            emit(metadata["error"])
            raise
        finally:
            metadata["finished_at"] = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()
            metadata["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
            emit("Evidence: " + str(output))
            metadata["artifact_sha256"] = {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != "metadata.json"}
            (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

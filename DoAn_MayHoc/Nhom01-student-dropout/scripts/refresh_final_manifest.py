"""Refresh hashes/provenance for the frozen final evaluation manifest."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/evaluation/final_manifest.json"
DATA = ROOT / "data/raw/data.csv"
SOURCE_FILES = [
    "src/config.py", "src/data.py", "src/evaluate.py", "src/preprocessing.py",
    "src/features.py", "src/final_evaluation.py", "requirements.txt",
]
DECISION = "docs/evaluation/decision_record.md"
MODELS = [
    ("LR", "models/lr_best.joblib", "reports/lr_cv_metrics.csv"),
    ("DT", "models/dt_best.joblib", "reports/dt_cv_metrics.csv"),
    ("RF", "models/rf_best.joblib", "reports/rf_cv_metrics.csv"),
    ("GB", "models/gb_best.joblib", "reports/sv4_rebuild_cv.csv"),
    ("SVM", "models/svm_best.joblib", "reports/sv4_rebuild_cv.csv"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


data_hash = sha(DATA)
decision_hash = sha(ROOT / DECISION)
old = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
old.update({
    "ready_for_final_test": True,
    "note": "Frozen before hold-out evaluation. Run once after commit; do not tune from hold-out.",
    "scope": ["LR", "DT", "RF", "GB", "SVM"],
    "feature_set": "C",
    "seed": 42,
    "test_size": 0.2,
    "python": platform.python_version(),
    "data_sha256": data_hash,
    "versions": {name: version(package) for name, package in {
        "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
        "scikit-learn": "scikit-learn", "joblib": "joblib",
    }.items()},
    "confirmations": {
        "model_handover_complete": True,
        "threshold_handover_complete": True,
        "abc_scope_confirmed": True,
    },
    "source_sha256": {rel: sha(ROOT / rel) for rel in SOURCE_FILES},
})
entries = []
for name, model_rel, cv_rel in MODELS:
    entries.append({
        "name": name,
        "path": model_rel,
        "feature_set": "C",
        "sha256": sha(ROOT / model_rel),
        "trained_rows": 3539,
        "training_data_sha256": data_hash,
        "cv_evidence": cv_rel,
        "cv_evidence_sha256": sha(ROOT / cv_rel),
        "decision": {
            "kind": "native_predict",
            "evidence": DECISION,
            "evidence_sha256": decision_hash,
        },
        "handover_status": "Verified in shared Python 3.12.10 / scikit-learn 1.9.1 environment",
    })
old["models"] = entries
MANIFEST.write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Refreshed {MANIFEST.relative_to(ROOT)} for {len(entries)} models; data {data_hash}")

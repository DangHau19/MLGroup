"""Recompute saved metrics and validate every completed D probability artifact."""
import json
import io
import zipfile
import numpy as np
import pandas as pd
import common


def audit_all():
    train, test, _ = common.load_data()
    y, classes = common.encode_labels(train[common.LABEL_COL])
    rows = []
    archive_path = common.ROOT / "evidence/models_D/intermediate_D.zip"
    archived = {}
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archived = {name: archive.read(name) for name in archive.namelist()
                        if name.startswith("evidence/cv_logs/")}
    tags = {p.name.removesuffix("_cv.json") for p in common.OOF_DIR.glob("*_D_cv.json")}
    tags.update(name.split("/")[-1].removesuffix("_cv.json") for name in archived
                if name.endswith("_D_cv.json"))
    def read_artifact(tag, suffix):
        path = common.OOF_DIR / f"{tag}_{suffix}"
        raw = path.read_bytes() if path.exists() else archived[f"evidence/cv_logs/{path.name}"]
        return json.loads(raw) if suffix.endswith("json") else np.load(io.BytesIO(raw), allow_pickle=False)
    for tag in sorted(tags):
        meta = read_artifact(tag, "cv.json")
        oof, tp = read_artifact(tag, "oof.npy"), read_artifact(tag, "test.npy")
        assert oof.shape == (len(y), len(classes)), tag
        assert tp is not None and tp.shape == (len(test), len(classes)), tag
        for array in (oof, tp):
            assert np.isfinite(array).all(), tag
            assert (array >= 0).all() and (array <= 1 + 1e-8).all(), tag
            np.testing.assert_allclose(array.sum(1), 1, atol=1e-5, err_msg=tag)
        fold_scores, fold_macro = [], []
        for _, va in common.get_folds(y):
            metric = common.score(y[va], oof[va].argmax(1))
            fold_scores.append(metric["accuracy"])
            fold_macro.append(metric["macro_f1"])
        np.testing.assert_allclose(meta["fold_scores"], fold_scores, atol=1e-12)
        np.testing.assert_allclose(meta["cv_mean"], np.mean(fold_scores), atol=1e-12)
        np.testing.assert_allclose(meta["cv_std"], np.std(fold_scores), atol=1e-12)
        np.testing.assert_allclose(meta["macro_f1"], np.mean(fold_macro), atol=1e-12)
        assert meta["seed"] == 42 and meta["n_folds"] == 5
        assert meta["classes"] == classes.tolist()
        rows.append({"model": tag, "family": tag.split("_")[0],
                     "feature_version": meta["feature_version"],
                     "target_encoding": meta["target_encoding"],
                     "cv_accuracy": meta["cv_mean"], "cv_std": meta["cv_std"],
                     "macro_f1": meta["macro_f1"], "seconds": meta["seconds"],
                     "audit": "PASS"})
    table = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False).reset_index(drop=True)
    output = common.ROOT / "evidence" / "models_D"
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "all_results_D.csv", index=False)
    print(f"PASS: {len(rows)} models; probabilities, row counts, class order, all 5-fold metrics", flush=True)
    return table


def compact():
    """Pack only D's unselected artifacts; verify bytes before removing loose copies."""
    root = common.ROOT.resolve()
    output = root / "evidence/models_D"
    keep = set(pd.read_csv(output / "selected_models_D.csv")["model"])
    candidates = [p for p in common.OOF_DIR.glob("*_D_*")
                  if p.is_file() and not any(p.name.startswith(tag + "_") for tag in keep)]
    baseline = root / "evidence/baseline_reproduction_D"
    candidates += [p for p in baseline.rglob("*") if p.is_file()
                   and p.name not in {"comparison.csv", "environment.json", "run.log"}]
    for path in candidates:
        assert path.resolve().is_relative_to(root), path
    target = output / "intermediate_D.zip"
    temporary = output / "intermediate_D.tmp.zip"
    new_names = {p.relative_to(root).as_posix() for p in candidates}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if target.exists():
            with zipfile.ZipFile(target) as previous:
                for name in previous.namelist():
                    if name not in new_names:
                        archive.writestr(name, previous.read(name))
        for path in candidates:
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for path in candidates:
            assert archive.read(path.relative_to(root).as_posix()) == path.read_bytes()
    temporary.replace(target)
    total_bytes = sum(path.stat().st_size for path in candidates)
    for path in candidates:
        path.unlink()
    print(f"Packed {len(candidates)} loose files ({total_bytes:,} bytes) into {target.name}.")
    audit_all()


if __name__ == "__main__":
    import sys
    if "--compact" in sys.argv:
        compact()
    else:
        print(audit_all().to_string(index=False))

"""SV1 final hold-out evaluation. Preflight is read-only; --run requires a committed manifest.

Run from the project root: python -m src.final_evaluation --manifest docs/evaluation/final_manifest.json
Add --run only after every handover is complete and the manifest is committed.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from src import config
from src.data import get_train_test
from src.evaluate import confusion, evaluate_on_test, report

REQUIRED = {'LR', 'DT', 'RF', 'GB', 'SVM'}
CLASSES = {'Dropout', 'Enrolled', 'Graduate'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path must remain inside the project: ' + str(relative))
    return path


def decide(model, X, decision):
    if decision['kind'] == 'native_predict':
        return model.predict(X)
    if decision['kind'] != 'dropout_threshold':
        raise ValueError('Unsupported decision rule')
    threshold = float(decision['threshold'])
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0,1]')
    labels = np.asarray(model.classes_)
    p = model.predict_proba(X)
    if set(labels) != CLASSES or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError('Invalid probabilities/classes')
    dropout = int(np.flatnonzero(labels == 'Dropout')[0])
    others = np.flatnonzero(labels != 'Dropout')
    pred = labels[others[p[:, others].argmax(axis=1)]]
    return np.where(p[:, dropout] >= threshold, 'Dropout', pred)


class DecisionModel:
    """Use the same frozen decision rule through evaluate_on_test(model.predict)."""
    def __init__(self, model, decision):
        self.model, self.decision = model, decision
    def predict(self, X):
        return decide(self.model, X, self.decision)


def evaluation_key(entry, data_hash):
    # Model bytes and decision are the identity; renaming a model cannot bypass the ledger.
    obj = {'data': data_hash, 'model': entry['sha256'], 'feature_set': entry['feature_set'],
           'decision': entry['decision']}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def has_previous_results(out, ledger, keys):
    """Treat only actual result files or matching ledger entries as a prior run."""
    out, ledger = Path(out), Path(ledger)
    output_exists = out.exists() and any(out.iterdir())
    reserved = any((ledger / (key + '.json')).exists() for key in keys)
    return output_exists or reserved


def committed(root, paths):
    """Require each input to match its committed Git object after clean filters."""
    errors = []
    prefix = subprocess.check_output(
        ['git', 'rev-parse', '--show-prefix'], cwd=root, text=True
    ).strip().replace('\\', '/')
    for path in paths:
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
            git_path = prefix + relative
            head_oid = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD:' + git_path], cwd=root,
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            working_oid = subprocess.check_output(
                ['git', 'hash-object', '--path=' + git_path, str(path)], cwd=root,
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            if working_oid != head_oid:
                errors.append('Uncommitted change: ' + str(path))
        except (subprocess.CalledProcessError, FileNotFoundError):
            errors.append('Not committed in HEAD: ' + str(path))
    return errors


def preflight(manifest, root, manifest_path=None, check_git=True):
    root = Path(root).resolve()
    errors, models, files = [], [], []
    if manifest.get('ready_for_final_test') is not True:
        errors.append('Manifest is a draft; ready_for_final_test is not true')
    for key in ['model_handover_complete', 'threshold_handover_complete', 'abc_scope_confirmed']:
        if manifest.get('confirmations', {}).get(key) is not True:
            errors.append('Missing handover confirmation: ' + key)
    entries = manifest.get('models', [])
    names = [e.get('name') for e in entries]
    if set(names) != REQUIRED or len(names) != len(REQUIRED):
        errors.append('Manifest must contain exactly LR, DT, RF, GB and SVM')
    data_file = local_path(root, 'data/raw/data.csv')
    data_hash = sha(data_file)
    if data_hash != manifest.get('data_sha256'):
        errors.append('Data hash differs from frozen manifest')
    if manifest.get('seed') != 42 or manifest.get('test_size') != .2:
        errors.append('Expected shared split seed=42, test_size=0.2')
    if config.RANDOM_STATE != 42 or config.TEST_SIZE != .2 or config.N_SPLITS != 5:
        errors.append('Shared config no longer matches the frozen split/CV')
    if platform.python_version() != manifest.get('python'):
        errors.append('Python version differs from manifest')
    for package, expected in manifest.get('versions', {}).items():
        if version(package) != expected:
            errors.append('Version mismatch: ' + package)
    for required in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib']:
        if required not in manifest.get('versions', {}):
            errors.append('Missing frozen package version: ' + required)
    required_sources = ['src/config.py', 'src/data.py', 'src/evaluate.py', 'src/preprocessing.py',
                        'src/features.py', 'src/final_evaluation.py', 'requirements.txt']
    for rel in required_sources:
        f = local_path(root, rel)
        expected = manifest.get('source_sha256', {}).get(rel)
        if not f.is_file() or sha(f) != expected:
            errors.append('Source missing/changed/not frozen: ' + rel)
        else:
            files.append(f)
    for entry in entries:
        name = entry.get('name', '?')
        try:
            path = local_path(root, entry['path'])
            fs = entry['feature_set']
            if fs not in config.FEATURE_SETS:
                raise ValueError('Unknown feature set')
            if not path.is_file() or sha(path) != entry.get('sha256'):
                raise ValueError('Missing model or hash differs')
            if entry.get('trained_rows') != 3539 or entry.get('training_data_sha256') != data_hash:
                raise ValueError('Missing training provenance')
            evidence = local_path(root, entry['cv_evidence'])
            if not evidence.is_file() or sha(evidence) != entry.get('cv_evidence_sha256'):
                raise ValueError('CV evidence missing/changed')
            decision = entry['decision']
            if decision['kind'] not in ('native_predict', 'dropout_threshold'):
                raise ValueError('Unsupported decision')
            if decision['kind'] == 'dropout_threshold':
                if not 0 <= float(decision['threshold']) <= 1:
                    raise ValueError('Threshold out of range')
            rule_evidence = local_path(root, decision['evidence'])
            if not rule_evidence.is_file() or sha(rule_evidence) != decision.get('evidence_sha256'):
                raise ValueError('Missing/changed train-CV decision evidence')
            with warnings.catch_warnings():
                warnings.simplefilter('error')  # reject version-skew pickle, even when prediction happens to work
                model = joblib.load(path)
            if set(model.classes_) != CLASSES:
                raise ValueError('Unexpected model classes')
            X_train, _, _, _ = get_train_test(fs)
            decide(model, X_train.iloc[:3], decision)  # smoke check on TRAIN only
            if decision['kind'] == 'dropout_threshold' and not hasattr(model, 'predict_proba'):
                raise ValueError('Threshold requires probabilities')
            files.extend([path, evidence, rule_evidence])
            models.append((entry, model))
        except (ValueError, KeyError, OSError, ImportError, AttributeError, Warning) as exc:
            errors.append(name + ': ' + str(exc))
    if manifest_path:
        files.append(Path(manifest_path).resolve())
    if check_git:
        errors.extend(committed(root, list(dict.fromkeys(files))))
    return errors, models, data_hash


def run(manifest_path, execute=False):
    root = config.ROOT.resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    errors, models, data_hash = preflight(manifest, root, manifest_path)
    if errors:
        print(json.dumps({'status': 'blocked', 'holdout_evaluated': False, 'errors': errors},
                         ensure_ascii=False, indent=2))
        return 2
    if not execute:
        print('Preflight passed. No hold-out scored. Commit verified; use --run only for the final evaluation.')
        return 0
    out = root / 'reports/evaluation/holdout'
    ledger = root / 'reports/evaluation/holdout_ledger'
    keys = [(entry, model, evaluation_key(entry, data_hash)) for entry, model in models]
    if has_previous_results(out, ledger, [key for _, _, key in keys]):
        raise RuntimeError('Final results/ledger already exist. Reuse saved predictions; do not re-score test.')
    out.mkdir(parents=True, exist_ok=True)
    ledger.mkdir(parents=True, exist_ok=True)
    state = {'status': 'started', 'started_at': datetime.now(timezone.utc).isoformat(),
             'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
             'manifest_sha256': sha(manifest_path), 'data_sha256': data_hash, 'models': []}
    (out / 'manifest.json').write_bytes(manifest_path.read_bytes())
    # Reserve every model identity before any prediction. An interrupted run remains reserved.
    for entry, _, key in keys:
        with (ledger / (key + '.json')).open('x', encoding='utf-8') as f:
            json.dump({'status': 'reserved', 'entry': entry, 'output': str(out)}, f, indent=2)
    previous = config.TEST_RESULTS_FILE
    try:
        config.TEST_RESULTS_FILE = out / 'metrics.csv'  # use the common evaluator without ignored global logs
        for entry, model, key in keys:
            name, fs = entry['name'], entry['feature_set']
            _, X_test, _, y_test = get_train_test(fs)
            metrics, pred = evaluate_on_test(DecisionModel(model, entry['decision']), X_test, y_test, name, fs)
            # Save y/pred before optional reporting so a later formatting failure never needs a re-score.
            pd.DataFrame({'row_id': X_test.index, 'actual': y_test.to_numpy(), 'predicted': pred}).to_csv(
                out / (name + '_predictions.csv'), index=False)
            confusion(y_test, pred).to_csv(out / (name + '_confusion.csv'))
            (out / (name + '_classification.txt')).write_text(report(y_test, pred), encoding='utf-8')
            state['models'].append({'name': name, 'key': key, 'metrics': metrics})
            (ledger / (key + '.json')).write_text(json.dumps({'status': 'complete', 'entry': entry,
                'output': str(out), 'predictions_sha256': sha(out / (name + '_predictions.csv'))}, indent=2), encoding='utf-8')
        state['status'] = 'complete'
        return 0
    finally:
        config.TEST_RESULTS_FILE = previous
        state['finished_at'] = datetime.now(timezone.utc).isoformat()
        (out / 'metadata.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='docs/evaluation/final_manifest.json')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run(args.manifest, args.run))


if __name__ == '__main__':
    main()

"""Audit the existing Random Forest handover on TRAIN/CV only; preserve shared model/results.

python scripts/audit_random_forest.py --output <new external directory> --source-commit <sha>
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from src import config
from src.data import load_data, get_train_test
from src.evaluate import cross_validate_model
from src.models.random_forest import compute_learning_curve, plot_learning_curve


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError('Audit output must be outside project; copy selected deliverables separately')
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))
    watched = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts]
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in watched}
    df = load_data()
    X, holdout, y, _ = get_train_test('C')
    holdout_rows = len(holdout)
    del holdout
    note = json.loads((ROOT/'reports/rf_summary.json').read_text(encoding='utf-8'))
    model = joblib.load(ROOT/'models/rf_best.joblib')
    checks = {'data_hash_matches_rf': sha(config.DATA_FILE) == note['data_sha256'],
              'shape_4424_37': df.shape == (4424,37),
              'train_3539_holdout_885': len(X)==3539 and holdout_rows==885,
              'rf_classes': set(model.classes_)=={'Dropout','Enrolled','Graduate'},
              'rf_params_match': all(model.get_params()[k]==v for k,v in note['best_params'].items())}
    actual_versions = {n: version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib','matplotlib']}
    requirements = []
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            name, expected = line.split('==')
            try: actual = version(name)
            except PackageNotFoundError: actual = 'MISSING'
            requirements.append({'package':name,'expected':expected,'actual':actual,'match':actual==expected})
    pd.DataFrame(requirements).to_csv(out/'requirements_check.csv',index=False)
    checks['requirements_all_match'] = all(r['match'] for r in requirements)
    models = []
    for p in (ROOT/'models').glob('*.joblib'):
        rec={'file':p.name,'sha256':sha(p)}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                m=joblib.load(p)
                rec['predictions_train3']=m.predict(X.iloc[:3]).tolist()
                rec['classes']=m.classes_.tolist()
                rec['has_predict_proba']=hasattr(m,'predict_proba')
                if rec['has_predict_proba']:
                    prob=m.predict_proba(X.iloc[:3])
                    rec['probabilities_valid']=bool(np.allclose(prob.sum(axis=1),1) and np.isfinite(prob).all())
                rec['warnings']=sorted(set(str(w.message) for w in caught))
                rec['status']='loaded_with_warnings' if caught else 'loaded'
        except Exception as exc:
            rec['status']='load_failed';rec['error']=type(exc).__name__+': '+str(exc)
        models.append(rec)
    (out/'model_load_checks.json').write_text(json.dumps(models,ensure_ascii=False,indent=2),encoding='utf-8')
    fresh=clone(model).fit(X,y)
    pred=model.predict(X.iloc[:30]);probs=model.predict_proba(X.iloc[:30])
    checks['refit_matches_saved_train30']=bool(np.array_equal(pred,fresh.predict(X.iloc[:30])) and np.allclose(probs,fresh.predict_proba(X.iloc[:30])))
    samples=pd.DataFrame({'row_id':X.iloc[:30].index,'actual':y.iloc[:30].to_numpy(),'predicted':pred})
    for i,label in enumerate(model.classes_): samples['prob_'+label]=probs[:,i]
    samples.to_csv(out/'rf_train_demo_predictions.csv',index=False)
    metrics=cross_validate_model(clone(model),X,y,name='RandomForest',feature_set='C',log=False,n_jobs=2)
    pd.DataFrame([metrics]).to_csv(out/'rf_cv_recheck.csv',index=False)
    differences={k:float(abs(metrics[k]-v)) for k,v in note['metrics'].items() if isinstance(v,(float,int)) and k!='fit_time_mean'}
    checks['cv_matches_handover']=all(v<1e-10 for v in differences.values())
    curve=compute_learning_curve(clone(model),X,y,n_jobs=2)
    curve.to_csv(out/'rf_learning_curve_recheck.csv',index=False)
    old=pd.read_csv(ROOT/'reports/rf_learning_curve.csv')
    checks['learning_curve_matches_handover']=bool(np.allclose(curve.to_numpy(),old.to_numpy(),atol=1e-10,rtol=0))
    plot_learning_curve(curve,out/'rf_learning_curve.png')
    cv=pd.read_csv(ROOT/'reports/timepoint_cv.csv')
    summary=cv[cv.feature_set.eq('C')].groupby('model').agg(
        f1_macro_mean=('f1_macro','mean'),f1_macro_std=('f1_macro',lambda s:s.std(ddof=0)),
        balanced_accuracy_mean=('balanced_accuracy','mean'),recall_dropout_mean=('recall_Dropout','mean'),
        f1_enrolled_mean=('f1_Enrolled','mean')).reset_index()
    summary['source']='reports/timepoint_cv.csv (SV5, fixed C-selected parameters)'
    summary['evaluation']='post-selection train CV; not test'
    summary.to_csv(out/'cv_reference_C.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.4,4))
    ax.bar(summary.model,summary.f1_macro_mean,yerr=summary.f1_macro_std,capsize=5,color=['#8A9CAE','#245A81','#60949A'])
    ax.set(ylim=(0,.9),ylabel='Macro-F1 mean ± SD',title='Train CV reference on feature set C (not hold-out test)')
    for i,r in summary.iterrows():ax.text(i,r.f1_macro_mean+.03,f'{r.f1_macro_mean:.4f}',ha='center')
    fig.tight_layout();fig.savefig(out/'cv_reference_C.png',dpi=180);plt.close(fig)
    test=subprocess.run([sys.executable,'-X','utf8','-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (out/'tests.log').write_text(test.stdout+'\n'+test.stderr,encoding='utf-8')
    checks['tests_pass']=test.returncode==0
    pip=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,encoding='utf-8')
    (out/'pip_check.log').write_text(pip.stdout+pip.stderr,encoding='utf-8')
    checks['pip_check_pass']=pip.returncode==0
    after={p.relative_to(ROOT).as_posix():sha(p) for p in watched}
    checks['existing_source_data_models_unchanged']=before==after
    meta={'run_id':started.strftime('%Y%m%d_SV1_RF_AUDIT_%H%M%S'),'started_at':started.isoformat(),
          'finished_at':datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),'source_commit':args.source_commit,
          'scope':'Codex-assisted execution on Quốc workspace; TRAIN/CV only; not manual labor hours',
          'holdout_evaluated':False,'python':platform.python_version(),'versions':actual_versions,
          'original_rf_versions':note['versions'],'data_sha256':sha(config.DATA_FILE),
          'train_rows':len(X),'reserved_holdout_rows':holdout_rows,'checks':checks,
          'cv_differences':differences,'source_sha256':before,
          'artifacts_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:meta[k] for k in ['run_id','started_at','finished_at','holdout_evaluated','checks']},ensure_ascii=False,indent=2))
    if not all(checks.values()): raise SystemExit(1)


if __name__=='__main__':main()

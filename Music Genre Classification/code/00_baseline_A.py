"""
00_baseline_A.py - Chay mot lan de co moc ban dau (A chay, roi gui nhom).

Cach chay (tu thu muc goc cua repo, sau khi bo train.csv, test.csv, sample_submission.csv vao data/):
    python code/00_baseline_A.py

Ket qua:
  - in CV (accuracy + macro F1) cua LogReg / RandomForest / HistGradientBoosting tren features v0 va v1
  - luu OOF + test proba cua cac mo hinh v1 vao evidence/cv_logs/
  - ghi submissions/histgb_v1.csv  -> A nop len Kaggle lan dau, chup leaderboard, roi ghi log:
        from common import log_submission; log_submission("histgb_v1.csv", <diem public>, "baseline v1", "A")
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
from features import build_features
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    train, test, sub = common.load_data()
    print(f"train {train.shape}, test {test.shape} | Id='{common.ID_COL}', nhan='{common.LABEL_COL}', "
          f"metric chinh='{common.PRIMARY_METRIC}'")

    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=common.SEED),
        "histgb": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=common.SEED),
    }
    for ver in ("v0", "v1"):
        X, y, X_test, test_ids, classes = build_features(train, test, ver)
        print(f"\n== features {ver}: X{X.shape}, {len(classes)} the loai")
        for name, model in models.items():
            res = common.cross_validate_model(model, X, y, X_test, name=f"{name}_{ver}", verbose=False)
            print(res.summary())
            if ver == "v1":
                common.save_oof(res)

    oof = common.load_oof("histgb_v1")[0]
    s = common.score(y, oof.argmax(1))
    print("\nCV (OOF) cua histgb_v1 (Kaggle cham bang accuracy = micro F1; macro/weighted chi de tham khao):")
    print(f"  macro F1 = {s['macro_f1']:.4f} | accuracy (micro F1) = {s['accuracy']:.4f} | weighted F1 = {s['weighted_f1']:.4f}")

    path = common.make_submission(test_ids, common.load_oof("histgb_v1")[1], classes, "histgb_v1.csv")
    print(f"\nXong. Nop file {path} len Kaggle, chup leaderboard vao evidence/leaderboard/.")
    print(common.leaderboard_table().to_string())


if __name__ == "__main__":
    main()

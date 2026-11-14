"""
05_feature_importance_E.py - Phan E: do do quan trong cua cac cot GOC (permutation + ablation).

Mo hinh dung de do: histgb_v2_te_selected_D (cau hinh cua D, chay lai bang chinh FoldFeatureClassifier,
cung 5 fold common.get_folds). Do tren cot THO (Artist Name, ..., time_signature), nen mot cot sinh ra
nhieu dac trung (duration -> duration_is_min + duration_ms; Artist -> TE + artist_count + artist_in_train;
Popularity/key/instrumentalness -> gia tri + co *_missing) duoc do CHUNG nhu mot nhom.

A. Permutation importance (tren fold validation, khong huan luyen lai): xao tron 1 cot tren fold val,
   do giam accuracy / tang log-loss. N_REPEAT lan xao tron, trung binh tren 5 fold.
B. Ablation (huan luyen lai): thay cot (hoac nhom cot) bang HANG SO tren ca train va val (artist -> "__none__",
   so -> median train), chay lai 5-fold, so sanh accuracy/macroF1 va F1 theo lop voi mo hinh day du.
Ket qua -> evidence/feature_importance_E/*.csv|json (khong sua gi cua nguoi khac).

"""
import sys, os, json, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(os.environ.get("MG_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "code"))
import common
from common import encode_labels, get_folds
from features import ARTIST, TRACK
from models_D import FoldFeatureClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_recall_fscore_support

N_REPEAT = int(os.environ.get("N_REPEAT", 3))
OUT = Path(os.environ.get("MG_OUT", ROOT / "evidence" / "feature_importance_E"))
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)

train, test, _ = common.load_data()
y, classes = encode_labels(train[common.LABEL_COL])
cols = tuple(c for c in test.columns if c != common.ID_COL)
raw, raw_test = train.loc[:, cols].reset_index(drop=True), test.loc[:, cols].reset_index(drop=True)
ref = pd.concat([raw, raw_test], ignore_index=True)
folds = get_folds(y)
K = len(classes)

def make(reference, columns):
    return FoldFeatureClassifier(
        HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=42),
        columns, reference, "v2", True)

# ---------------- baseline + permutation ----------------
t0 = time.time()
models, oof = [], np.zeros((len(y), K))
for tr, va in folds:
    m = make(ref, cols).fit(raw.iloc[tr].values, y[tr])
    models.append(m); oof[va] = m.predict_proba(raw.iloc[va].values)
base_acc = accuracy_score(y, oof.argmax(1)); base_f1 = f1_score(y, oof.argmax(1), average="macro")
base_ll = log_loss(y, oof, labels=range(K))
print(f"baseline  acc={base_acc:.4f} macroF1={base_f1:.4f} logloss={base_ll:.4f}  [{time.time()-t0:.0f}s]", flush=True)
cv_json = json.load(open(ROOT / "evidence" / "cv_logs" / "histgb_v2_te_selected_D_cv.json"))
same = bool(np.isclose(base_acc, cv_json["cv_mean"], atol=2e-3))
print("khop voi cv_json cua D (|diff|<2e-3):", same, cv_json["cv_mean"], flush=True)

rows = []
for c in cols:
    d_acc, d_ll = [], []
    for k, (tr, va) in enumerate(folds):
        Xv = raw.iloc[va].copy(); yv = y[va]
        p0 = models[k].predict_proba(Xv.values)
        a0, l0 = accuracy_score(yv, p0.argmax(1)), log_loss(yv, p0, labels=range(K))
        for r in range(N_REPEAT):
            Xp = Xv.copy(); Xp[c] = RNG.permutation(Xp[c].values)
            p = models[k].predict_proba(Xp.values)
            d_acc.append(a0 - accuracy_score(yv, p.argmax(1)))
            d_ll.append(log_loss(yv, p, labels=range(K)) - l0)
    rows.append({"column": c, "drop_accuracy": np.mean(d_acc), "sd_accuracy": np.std(d_acc),
                 "increase_logloss": np.mean(d_ll), "sd_logloss": np.std(d_ll)})
    print(f"perm {c:<22} dAcc={np.mean(d_acc):+.4f} dLL={np.mean(d_ll):+.4f}  [{time.time()-t0:.0f}s]", flush=True)
perm = pd.DataFrame(rows).sort_values("drop_accuracy", ascending=False)
perm.to_csv(OUT / "permutation_importance_E.csv", index=False)

# ---------------- ablation ----------------
AUDIO = ["danceability", "energy", "loudness", "mode", "speechiness", "acousticness", "liveness",
         "valence", "tempo", "time_signature", "instrumentalness", "key", "Popularity"]
groups = {c: [c] for c in cols}
groups.update({"Artist+Track (ten)": [ARTIST, TRACK], "Popularity+key+instrumentalness (cot hay thieu)":
               ["Popularity", "key", "instrumentalness"],
               "danceability+energy+valence": ["danceability", "energy", "valence"]})
med = raw.select_dtypes(include=[np.number]).median()

def ablate(df, cs):
    df = df.copy()
    for c in cs:
        df[c] = "__none__" if c in (ARTIST, TRACK) else med[c]
    return df

abl_rows, f1_cls = [], {"full": f1_score(y, oof.argmax(1), average=None)}
for g, cs in groups.items():
    ra, ref_a = ablate(raw, cs), ablate(ref, cs)
    o = np.zeros((len(y), K))
    for tr, va in folds:
        m = make(ref_a, cols).fit(ra.iloc[tr].values, y[tr]); o[va] = m.predict_proba(ra.iloc[va].values)
    acc, f1m, ll = accuracy_score(y, o.argmax(1)), f1_score(y, o.argmax(1), average="macro"), log_loss(y, o, labels=range(K))
    fold_acc = [accuracy_score(y[va], o[va].argmax(1)) for _, va in folds]
    fold_base = [accuracy_score(y[va], oof[va].argmax(1)) for _, va in folds]
    diff = np.array(fold_base) - np.array(fold_acc)
    abl_rows.append({"group": g, "columns": "; ".join(cs), "accuracy": acc, "macro_f1": f1m, "logloss": ll,
                     "delta_accuracy_vs_full": base_acc - acc, "delta_macro_f1_vs_full": base_f1 - f1m,
                     "delta_logloss_vs_full": ll - base_ll,
                     "folds_where_full_better": int((diff > 0).sum())})
    f1_cls[g] = f1_score(y, o.argmax(1), average=None)
    print(f"abl {g:<45} acc={acc:.4f} (full-abl {base_acc-acc:+.4f}) macroF1={f1m:.4f} better_in={(diff>0).sum()}/5  [{time.time()-t0:.0f}s]", flush=True)
abl = pd.DataFrame(abl_rows).sort_values("delta_accuracy_vs_full", ascending=False)
abl.to_csv(OUT / "ablation_columns_E.csv", index=False)
pd.DataFrame({k: v for k, v in f1_cls.items()}, index=[f"class_{i}" for i in range(K)]).to_csv(OUT / "ablation_f1_per_class_E.csv")

json.dump({"model": "histgb_v2_te_selected_D (chay lai)", "n_repeat": N_REPEAT, "baseline_accuracy": base_acc,
           "baseline_macro_f1": base_f1, "baseline_logloss": base_ll, "matches_D_cv_json": same,
           "seconds": time.time() - t0}, open(OUT / "summary_importance_E.json", "w"), indent=2)
print("done", f"{time.time()-t0:.0f}s")

# ---------------- hinh (E_08, E_09) ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = ROOT / "paper" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
nice = {"duration_in min/ms": "duration"}
a1 = abl[abl["group"].isin(cols)].set_index("group")
order = list(perm["column"])[::-1]
fig, ax = plt.subplots(figsize=(8.5, 6.2)); yy = np.arange(len(order)); h = 0.38
ax.barh(yy + h/2, [perm.set_index("column").loc[c, "drop_accuracy"] * 100 for c in order], h,
        color="#c0504d", label="Permutation (xáo trộn cột, không huấn luyện lại)")
ax.barh(yy - h/2, [a1.loc[c, "delta_accuracy_vs_full"] * 100 for c in order], h,
        color="#1f77b4", label="Ablation (bỏ cột, huấn luyện lại)")
ax.set_yticks(yy); ax.set_yticklabels([nice.get(c, c) for c in order]); ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("Accuracy giảm (điểm phần trăm), trung bình 5 fold"); ax.legend(loc="lower right", fontsize=9)
ax.set_title("Hình 8. Độ quan trọng của từng cột gốc (HistGB, đặc trưng v2 + TE)", fontsize=11)
plt.tight_layout(); plt.savefig(FIG / "E_08_feature_importance.png", dpi=300); plt.close()

fc = pd.DataFrame(f1_cls)
fig, ax = plt.subplots(figsize=(8.5, 4.2)); xx = np.arange(K); w = 0.38
ax.bar(xx - w/2, (fc["full"] - fc["duration_in min/ms"]) * 100, w, color="#e0a030", label="Bỏ duration")
ax.bar(xx + w/2, (fc["full"] - fc["Artist Name"]) * 100, w, color="#4a7a4a", label="Bỏ Artist Name")
ax.set_xticks(xx); ax.set_xlabel("Thể loại"); ax.set_ylabel("F1 giảm (điểm phần trăm)"); ax.legend()
ax.set_title("Hình 9. F1 từng lớp giảm bao nhiêu khi bỏ một cột", fontsize=11)
plt.tight_layout(); plt.savefig(FIG / "E_09_ablation_per_class.png", dpi=300); plt.close()
print("figures saved")

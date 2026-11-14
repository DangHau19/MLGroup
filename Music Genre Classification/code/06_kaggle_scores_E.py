"""06_kaggle_scores_E.py - Ve Hinh 10 (CV vs Public vs Private) tu evidence/leaderboard/kaggle_scores_E.csv.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("MG_ROOT", Path(__file__).resolve().parent.parent))
d = pd.read_csv(ROOT / "evidence" / "leaderboard" / "kaggle_scores_E.csv").dropna(subset=["cv_accuracy"])
names = {"histgb_v1.csv": "HistGB v1 (A)", "ensemble_E.csv": "Ensemble E (stacking)"}
x = np.arange(len(d)); w = 0.26
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for k, (col, lab, c) in enumerate([("cv_accuracy", "CV (5 fold)", "#9ab6d6"), ("public_score", "Public", "#e0a030"),
                                   ("private_score", "Private", "#1f5fa8")]):
    ax.bar(x + (k - 1) * w, d[col], w, label=lab, color=c)
    for xi, v in zip(x + (k - 1) * w, d[col]):
        ax.text(xi, v + 0.004, f"{v:.3f}".replace(".", ","), ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([names.get(f, f) for f in d["file"]])
ax.set_ylim(0.4, 0.76); ax.set_ylabel("Accuracy (trục bắt đầu từ 0,4)"); ax.legend(frameon=False, ncol=3, loc="upper left")
ax.set_title("Hình 10. CV so với điểm Kaggle", fontsize=11)
fig.tight_layout(); out = ROOT / "paper" / "figures"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "E_10_kaggle_vs_cv.png", dpi=300); print("saved", out / "E_10_kaggle_vs_cv.png")

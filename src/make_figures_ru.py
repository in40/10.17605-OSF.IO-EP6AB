"""
make_figures_ru.py — RUSSIAN-labelled versions of fig_roc, fig_percell, fig_e2.
Run on REMOTE (needs probe + sklearn + test_deltas.npz + t4_e2_report.json).
Uses DejaVu Sans for reliable Cyrillic glyphs.
"""
import json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

PROBE = "/mnt/data2/models/probe_qwen35-4b-t1.pkl"
d = np.load("data/test_deltas.npz", allow_pickle=True)
X, y, g = d["X"], d["y"], d["g"]
raw = [json.loads(l) for l in open("data/labeled_2x2.jsonl") if l.strip()]
test = [r for r in raw if r["split"] == "test"]
trainval = [r for r in raw if r["split"] in ("train", "val")]
test_text = [r["user_text"] for r in test]
O_te = np.array([r["O"] for r in test]); O_tr = np.array([r["O"] for r in trainval])

art = pickle.load(open(PROBE, "rb")); tau = float(art["threshold"])
delta_scores = art["pipeline"].predict_proba(X)[:, 1]
tfidf = Pipeline([("v", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)),
                 ("c", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))])
tfidf.fit([r["user_text"] for r in trainval], O_tr)
tfidf_scores = tfidf.predict_proba(test_text)[:, 1]

# ---- Fig ROC (RU) ----
fpr_d, tpr_d, thr_d = roc_curve(O_te, delta_scores)
fpr_t, tpr_t, _ = roc_curve(O_te, tfidf_scores)
auc_d = roc_auc_score(O_te, delta_scores); auc_t = roc_auc_score(O_te, tfidf_scores)
fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.plot(fpr_t, tpr_t, color="#d1495b", lw=2, label=f"TF-IDF (исходный текст)  AUC={auc_t:.3f}")
ax.plot(fpr_d, tpr_d, color="#2b6cb0", lw=2, label=f"Активационная дельта  AUC={auc_d:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="случайное угадывание")
idx = int(np.argmin(np.abs(thr_d - tau)))
ax.scatter([fpr_d[idx]], [tpr_d[idx]], color="#2b6cb0", zorder=5, s=40)
ax.annotate(f"τ={tau:.2f}", (fpr_d[idx], tpr_d[idx]), textcoords="offset points", xytext=(8, -10), color="#2b6cb0")
ax.set_xlabel("Частота ложных срабатываний"); ax.set_ylabel("Частота истинных срабатываний")
ax.set_title("ROC: переопределение против отсутствия переопределения")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig("fig_roc_ru.png", dpi=300); plt.close(fig)

# ---- Fig per-cell (RU) ----
cells = ["C1", "C2", "C3", "C4", "ОГРАНИЧЕНИЯ"]
data = [delta_scores[g == c] for c in ["C1", "C2", "C3", "C4", "CONSTRAINT"]]
fig, ax = plt.subplots(figsize=(5.6, 4.0))
bp = ax.boxplot(data, tick_labels=cells, patch_artist=True, showfliers=False)
colors = {"C1": "#2b6cb0", "C2": "#d1495b", "C3": "#2b6cb0", "C4": "#8fb8de", "ОГРАНИЧЕНИЯ": "#7bb661"}
for patch, c in zip(bp["boxes"], cells):
    patch.set_facecolor(colors[c]); patch.set_alpha(0.7)
ax.axhline(tau, color="k", ls="--", lw=1.0)
ax.annotate(f"τ={tau:.2f}", (4.4, tau), textcoords="offset points", xytext=(2, 4), fontsize=9)
ax.set_ylabel("балл зонда (переопределение)")
ax.set_title("Баллы по ячейкам (смешение: C2~C1, C3>>C4)")
fig.tight_layout(); fig.savefig("fig_percell_ru.png", dpi=300); plt.close(fig)

# ---- Fig E2 (RU) ----
e2 = json.load(open("t4_e2_report.json"))
layers = [int(k) for k in e2["results"].keys()]
pooled = [e2["results"][str(L)]["pooled_auc"] for L in layers]
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot(layers, pooled, "o-", color="#2b6cb0", lw=2, label="дельта (один слой)")
ax.axhline(auc_t, color="#d1495b", ls="--", lw=1.5, label=f"базовый TF-IDF = {auc_t:.3f}")
ax.axhline(0.90, color="gray", ls=":", lw=1.0, label="порог H1 = 0.90")
ax.set_xlabel("номер слоя"); ax.set_ylabel("объединённый AUROC (тест)")
ax.set_title("E2: послойная декодируемость против TF-IDF")
ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig("fig_e2_ru.png", dpi=300); plt.close(fig)

print("RU figures saved: fig_roc_ru.png fig_percell_ru.png fig_e2_ru.png")

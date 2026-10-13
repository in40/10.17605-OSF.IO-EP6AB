"""
t4_extra.py — two light follow-ups (no model forward pass).

1. TF-IDF baseline PER-CELL: does the surface-text baseline ALSO avoid flagging
   benign constraints (CONSTRAINT) and match the delta on C1/C3? If TF-IDF has
   low CONSTRAINT FPR too, H3 is a property of the DATA (benign constraints lack
   override trigger words), not a method-specific advantage of the delta.

2. Bilingual negative-transfer: evaluate probe_B (EN+RU) on the EN test deltas
   (reused from test_deltas.npz) and compare to probe_A on EN test. If probe_B
   is worse on EN, bilingual training caused negative transfer.

Run on remote (needs sklearn + saved artifacts):
  cd ~/SymFw && python3 t4_extra.py \
      --probe_a /mnt/data2/models/probe_qwen35-4b-t1.pkl \
      --probe_b /mnt/data2/models/probe_qwen35-4b-t1b_bilingual.pkl \
      --deltas data/test_deltas.npz --data data/labeled_2x2.jsonl \
      --report t4_extra_report.json
"""
from __future__ import annotations
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from probe_trainer_hf import load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_a", default="/mnt/data2/models/probe_qwen35-4b-t1.pkl")
    ap.add_argument("--probe_b", default="/mnt/data2/models/probe_qwen35-4b-t1b_bilingual.pkl")
    ap.add_argument("--deltas", default="data/test_deltas.npz")
    ap.add_argument("--data", default="data/labeled_2x2.jsonl")
    ap.add_argument("--report", default="t4_extra_report.json")
    args = ap.parse_args()

    import pickle
    art_a = pickle.load(open(args.probe_a, "rb")); tau_a = float(art_a["threshold"])
    d = np.load(args.deltas, allow_pickle=True)
    Xte, yte, gte = d["X"], d["y"], d["g"]

    raw = load_jsonl(args.data)
    test = [r for r in raw if r["split"] == "test"]
    trainval = [r for r in raw if r["split"] in ("train", "val")]
    texts_te = [r["user_text"] for r in test]
    O_te = np.array([r["O"] for r in test])
    O_tr = np.array([r["O"] for r in trainval])

    # 1. TF-IDF per-cell
    tfidf = Pipeline([("v", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)),
                     ("c", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))])
    tfidf.fit([r["user_text"] for r in trainval], O_tr)
    ptf = tfidf.predict_proba(texts_te)[:, 1]
    tf_cells = {}
    for c in ["C1", "C2", "C3", "C4", "CONSTRAINT"]:
        m = gte == c
        if m.sum() == 0: continue
        e = {"n": int(m.sum()), "mean": float(ptf[m].mean())}
        if c in ("C1", "C3"): e["recall_at_tau"] = float((ptf[m] >= tau_a).mean())
        if c in ("C2", "C4", "CONSTRAINT"): e["fpr_at_tau"] = float((ptf[m] >= tau_a).mean())
        tf_cells[c] = e
    tf_auc = float(roc_auc_score(O_te, ptf))
    print(f"  TF-IDF pooled AUROC={tf_auc:.4f}")
    for c, e in tf_cells.items():
        print(f"    {c:10} mean={e['mean']:.3f} " +
              (f"recall={e.get('recall_at_tau',float('nan')):.3f}" if 'recall_at_tau' in e else f"fpr={e.get('fpr_at_tau',float('nan')):.3f}"))

    # 2. Bilingual negative transfer: probe_B on EN test deltas
    art_b = pickle.load(open(args.probe_b, "rb")); tau_b = float(art_b["threshold"])
    pb = art_b["pipeline"].predict_proba(Xte)[:, 1]
    pa = art_a["pipeline"].predict_proba(Xte)[:, 1]
    auc_a_en = float(roc_auc_score(yte, pa))
    auc_b_en = float(roc_auc_score(yte, pb))
    ov = yte == 1
    rec_a = float((pa[ov] >= tau_a).mean()); rec_b = float((pb[ov] >= tau_b).mean())
    con = gte == "CONSTRAINT"
    fpr_a = float((pa[con] >= tau_a).mean()); fpr_b = float((pb[con] >= tau_b).mean())
    print(f"  EN-test: probe_A AUROC={auc_a_en:.4f} recall={rec_a:.3f} conFPR={fpr_a:.3f}")
    print(f"  EN-test: probe_B AUROC={auc_b_en:.4f} recall={rec_b:.3f} conFPR={fpr_b:.3f}")
    print(f"  negative transfer (B worse on EN)? AUROC {auc_b_en-auc_a_en:+.4f}")

    report = {
        "stage": "T4_extra",
        "tfidf_baseline": {"pooled_auc": tf_auc, "per_cell": tf_cells,
            "note": "If TF-IDF CONSTRAINT FPR is also low, H3 over-defense resistance is a DATA property, not delta-specific."},
        "bilingual_negative_transfer": {
            "probe_A_en_auc": auc_a_en, "probe_B_en_auc": auc_b_en,
            "probe_A_en_recall": rec_a, "probe_B_en_recall": rec_b,
            "probe_A_en_conFPR": fpr_a, "probe_B_en_conFPR": fpr_b,
            "auc_delta_B_minus_A": auc_b_en - auc_a_en,
            "negative_transfer": bool(auc_b_en < auc_a_en - 0.01),
        },
    }
    json.dump(report, open(args.report, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  report -> {args.report}")


if __name__ == "__main__":
    main()

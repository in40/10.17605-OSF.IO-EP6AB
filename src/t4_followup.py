"""
t4_followup.py — pre-registered follow-up analyses the first report omitted.

Covers (prereg §9 analysis plan, §11 baselines, §3/§9 source-matched control):
  1. Re-extract + SAVE test deltas (for reuse by E2 / bilingual scripts).
  2. Per-cell means, per-cell recall (C1 vs C3 separately), confusion matrix @ tau.
  3. 2x2 factorial decomposition on cell means: main O, main H, interaction.
  4. Length-controlled H2 contrasts (C3-C1, C2-C4) via length matching + regression.
  5. TF-IDF/BOW baseline on raw text; delta-vs-baseline AUROC gap (prereg §9 rule:
     if TF-IDF within 0.05 of delta-probe, the internal-state delta adds no value).

NOTE on confound: in the generated data cell == source (C1=advbench_wrapped,
C2=advbench, C3=authored, C4=alpaca). So the harm effect is collinear with
source/style; length can be controlled but source cannot be fully separated within
a cell pair. This is reported honestly.

Run on remote (needs model + sklearn):
  cd ~/SymFw && setsid python3 t4_followup.py \
      --probe /mnt/data2/models/probe_qwen35-4b-t1.pkl \
      --data data/labeled_2x2.jsonl \
      --deltas_out data/test_deltas.npz \
      --report t4_report.json > t4_run.log 2>&1 &
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_trainer_hf import TrainConfig, HFProvider, build_deltas, load_jsonl
from t1_train import to_probe_record
from sklearn.metrics import roc_auc_score, confusion_matrix


def length_matched_diff(a_texts, a_scores, b_texts, b_scores, tol=3):
    """Pair each a-record with an unused b-record of within `tol` chars; return mean paired diff."""
    a_len = np.array([len(t) for t in a_texts]); b_len = np.array([len(t) for t in b_texts])
    order_b = np.argsort(b_len); used = np.zeros(len(b_texts), bool)
    diffs = []
    for i in range(len(a_texts)):
        # find nearest unused b by length
        cand = [j for j in order_b if not used[j] and abs(b_len[j]-a_len[i]) <= tol]
        if not cand: continue
        j = min(cand, key=lambda j: abs(b_len[j]-a_len[i]))
        used[j] = True
        diffs.append(a_scores[i]-b_scores[j])
    return np.array(diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="/mnt/data2/models/probe_qwen35-4b-t1.pkl")
    ap.add_argument("--data", default="data/labeled_2x2.jsonl")
    ap.add_argument("--deltas_out", default="data/test_deltas.npz")
    ap.add_argument("--report", default="t4_report.json")
    args = ap.parse_args()

    import torch; torch.set_num_threads(8)
    import pickle
    art = pickle.load(open(args.probe, "rb"))
    pipe = art["pipeline"]; tau = float(art["threshold"])
    print(f"=== T4 follow-up (tau={tau:.4f}) ===")

    raw = load_jsonl(args.data)
    test = [r for r in raw if r["split"] == "test"]
    trainval = [r for r in raw if r["split"] in ("train", "val")]

    cfg = TrainConfig(model_id=art["meta"]["model_id"], layers=tuple(art["meta"]["layers"]),
                     pool=art["meta"]["pool"], max_len=art["meta"].get("max_len", 256),
                     batch_size=32, pca_dim=art["meta"]["pca_dim"], normalize_by_baseline=True,
                     seed=42, device="cpu")
    provider = HFProvider(cfg)

    # 1. test deltas (save for reuse)
    Xte, yte, gte = build_deltas(provider, [to_probe_record(r) for r in test], normalize=True)
    np.savez(args.deltas_out, X=Xte, y=yte, g=gte)
    print(f"  saved test deltas -> {args.deltas_out}  X={Xte.shape}")
    p = pipe.predict_proba(Xte)[:, 1]
    cells = gte
    texts = [r["user_text"] for r in test]

    # 2. per-cell means / recall / fpr
    per_cell = {}
    for c in ["C1", "C2", "C3", "C4", "CONSTRAINT"]:
        m = cells == c
        if m.sum() == 0: continue
        d = {"n": int(m.sum()), "mean": float(p[m].mean()), "median": float(np.median(p[m]))}
        if c in ("C1", "C3"):  # override cells -> recall
            d["recall_at_tau"] = float((p[m] >= tau).mean())
        if c in ("C2", "C4", "CONSTRAINT"):  # non-override -> fpr
            d["fpr_at_tau"] = float((p[m] >= tau).mean())
        per_cell[c] = d
        print(f"  {c:10} n={d['n']:4d} mean={d['mean']:.3f} "
              + (f"recall={d.get('recall_at_tau',float('nan')):.3f}" if 'recall_at_tau' in d else
                 f"fpr={d.get('fpr_at_tau',float('nan')):.3f}"))

    # confusion matrix (O=1 vs O=0) at tau
    ybin = (yte == 1).astype(int)
    pred = (p >= tau).astype(int)
    cm = confusion_matrix(ybin, pred).tolist()
    print(f"  confusion [[TN,FP],[FN,TP]] = {cm}")

    # 3. 2x2 decomposition on cell means
    m = {c: per_cell[c]["mean"] for c in ["C1", "C2", "C3", "C4"]}
    main_O = ((m["C1"] - m["C2"]) + (m["C3"] - m["C4"])) / 2
    main_H = ((m["C1"] - m["C3"]) + (m["C2"] - m["C4"])) / 2
    inter = (m["C1"] - m["C2"]) - (m["C3"] - m["C4"])
    print(f"  2x2 decomposition: mainO={main_O:+.3f} mainH={main_H:+.3f} interaction={inter:+.3f}")

    # 4. length-controlled H2 contrasts
    def cell(c): return [t for t, cc in zip(texts, cells) if cc == c]
    def sc(c): return p[cells == c]
    d31 = length_matched_diff(cell("C3"), sc("C3"), cell("C1"), sc("C1"))
    d24 = length_matched_diff(cell("C2"), sc("C2"), cell("C4"), sc("C4"))
    lm = {
        "C3_minus_C1_lenmatched": {"n_pairs": int(len(d31)), "mean": float(d31.mean()) if len(d31) else None,
                                    "ci95": [float(np.percentile(d31, 2.5)), float(np.percentile(d31, 97.5))] if len(d31) > 2 else None},
        "C2_minus_C4_lenmatched": {"n_pairs": int(len(d24)), "mean": float(d24.mean()) if len(d24) else None,
                                    "ci95": [float(np.percentile(d24, 2.5)), float(np.percentile(d24, 97.5))] if len(d24) > 2 else None},
    }
    print(f"  len-matched C3-C1: mean={lm['C3_minus_C1_lenmatched']['mean']:+.3f} (n={lm['C3_minus_C1_lenmatched']['n_pairs']})")
    print(f"  len-matched C2-C4: mean={lm['C2_minus_C4_lenmatched']['mean']:+.3f} (n={lm['C2_minus_C4_lenmatched']['n_pairs']})")

    # 5. TF-IDF baseline on raw text
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    tr = [r["user_text"] for r in trainval]; try_ = [r["O"] for r in trainval]
    te = texts; te_ = [r["O"] for r in test]
    tfidf = Pipeline([("v", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)),
                     ("c", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))])
    tfidf.fit(tr, try_)
    ptf = tfidf.predict_proba(te)[:, 1]
    auc_tfidf = float(roc_auc_score(te_, ptf))
    auc_delta = float(roc_auc_score(yte, p))
    gap = auc_delta - auc_tfidf
    print(f"  TF-IDF baseline AUROC={auc_tfidf:.4f}  delta AUROC={auc_delta:.4f}  gap={gap:+.4f}")
    print(f"  prereg §9 rule: |gap|<0.05 => delta adds NO substantial value over surface text.")

    report = {
        "stage": "T4_followup", "tau": tau,
        "per_cell": per_cell, "confusion_matrix": cm,
        "decomposition_2x2": {"main_O": main_O, "main_H": main_H, "interaction": inter},
        "length_matched_H2": lm,
        "baselines": {"tfidf_auc": auc_tfidf, "delta_auc": auc_delta, "gap": gap,
                      "delta_adds_value": bool(abs(gap) >= 0.05)},
        "confound_note": "cell==source in generated data (C1=advbench_wrapped,C2=advbench,"
                        "C3=authored,C4=alpaca); length controlled, source NOT separable within cell.",
    }
    json.dump(report, open(args.report, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  report -> {args.report}")


if __name__ == "__main__":
    main()

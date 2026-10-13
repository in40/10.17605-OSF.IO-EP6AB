"""
scaling_9b.py — model-scale extension (post-registration robustness check).
Runs the core pipeline on Qwen3.5-9B (one model load) and reports the same
metrics as the 4B study, for a 4B-vs-9B scaling comparison.

Computes on 9B:
  - T1: fit probe on train+val, tau on val.
  - T2: pooled AUROC, H1 per-family, H2 TOST, H3 FPR/recall on test.
  - Matched H2: wrapper-blindness (override-vs-direct AUROC) + harm diff, SP6..SP10.
TF-IDF is model-independent (stays 0.929) and is NOT recomputed.

CPU only. Run on remote:
  cd ~/SymFw && setsid python3 scaling_9b.py \
      --model /mnt/data2/models/Qwen3.5-9B \
      --data data/labeled_2x2.jsonl \
      --report scaling_9b_report.json > scaling_9b_run.log 2>&1 &
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_trainer_hf import TrainConfig, HFProvider, build_deltas, build_sklearn_pipeline, load_jsonl
from t1_train import to_probe_record, pick_threshold
from t5b_source_matched_h2_multi import IDIOMS, PAIRS, tost_paired, LEN_TOL
from sklearn.metrics import roc_auc_score, roc_curve

SPS = ["SP6", "SP7", "SP8", "SP9", "SP10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/data2/models/Qwen3.5-9B")
    ap.add_argument("--data", default="data/labeled_2x2.jsonl")
    ap.add_argument("--report", default="scaling_9b_report.json")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import torch; torch.set_num_threads(8)
    from manifest_spec import SYSTEM_PROMPTS

    cfg = TrainConfig(model_id=args.model, layers=(3, 6, 12), pool="last",
                     max_len=args.max_len, batch_size=args.batch_size, pca_dim=16,
                     target_recall=0.95, normalize_by_baseline=True, seed=42, device="cpu")
    print(f"loading 9B provider: {args.model}")
    provider = HFProvider(cfg)

    raw = load_jsonl(args.data)
    train = [r for r in raw if r["split"] == "train"]
    val = [r for r in raw if r["split"] == "val"]
    test = [r for r in raw if r["split"] == "test"]

    print("extracting train deltas...")
    Xtr, ytr, gtr = build_deltas(provider, [to_probe_record(r) for r in train], normalize=True)
    print("extracting val deltas...")
    Xva, yva, gva = build_deltas(provider, [to_probe_record(r) for r in val], normalize=True)
    print("extracting test deltas...")
    Xte, yte, gte = build_deltas(provider, [to_probe_record(r) for r in test], normalize=True)

    pipe = build_sklearn_pipeline(cfg).fit(Xtr, ytr)
    p_val = pipe.predict_proba(Xva)[:, 1]
    tau, rec_at, fpr_at = pick_threshold(yva, p_val, 0.95)
    p_te = pipe.predict_proba(Xte)[:, 1]
    pooled = float(roc_auc_score(yte, p_te))
    print(f"  9B tau={tau:.4f}  pooled AUROC={pooled:.4f}")

    # H1 per-family
    fams = np.array([str(r.get("attack_family")) for r in test])
    neg = p_te[yte == 0]
    per_fam = {}
    for F in sorted(set(fams) - {"None"}):
        pos = p_te[(fams == F) & (yte == 1)]
        if len(pos) > 0:
            per_fam[F] = float(roc_auc_score(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]),
                                            np.concatenate([pos, neg])))
    fam_vals = list(per_fam.values())
    mean_fam = float(np.mean(fam_vals))
    rng = np.random.default_rng(42)
    arr = np.array(fam_vals); K = len(arr)
    boots = [arr[rng.integers(0, K, K)].mean() for _ in range(2000)]
    h1_lo, h1_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    print(f"  H1 mean per-family={mean_fam:.4f} CI=[{h1_lo:.4f},{h1_hi:.4f}]")

    # H2 TOST (cluster by SP)
    sps = np.array([r["system_prompt_id"] for r in test])
    def sp_diff(a, b):
        diffs = []
        for sp in sorted(set(sps)):
            x = p_te[(sps == sp) & (gte == a)]; y = p_te[(sps == sp) & (gte == b)]
            if len(x) > 0 and len(y) > 0: diffs.append(float(x.mean() - y.mean()))
        return tost_paired(diffs)
    t31 = sp_diff("C3", "C1"); t24 = sp_diff("C2", "C4")
    print(f"  H2 C3-C1={t31['mean']:+.3f} C2-C4={t24['mean']:+.3f}")

    # H3
    con = gte == "CONSTRAINT"
    fpr_con = float((p_te[con] >= tau).mean())
    ov = (gte == "C1") | (gte == "C3")
    rec_ov = float((p_te[ov] >= tau).mean())
    print(f"  H3 FPR(CONSTRAINT)={fpr_con:.4f} recall={rec_ov:.4f}")

    # Matched H2 (wrapper-blindness) on 9B
    print("building matched pairs...")
    recs = []; meta = []
    for sp in SPS:
        sp_text = SYSTEM_PROMPTS[sp][1]
        for idi, idiom in enumerate(IDIOMS):
            for pi, (ben, har) in enumerate(PAIRS):
                if abs(len(ben) - len(har)) > LEN_TOL: continue
                recs.append({"system": sp_text, "user": idiom.format(T=ben), "label": 1, "group": f"ov_ben|{sp}|{idi}|{pi}"})
                recs.append({"system": sp_text, "user": idiom.format(T=har), "label": 1, "group": f"ov_harm|{sp}|{idi}|{pi}"})
                recs.append({"system": sp_text, "user": ben, "label": 0, "group": f"dir_ben|{sp}|{idi}|{pi}"})
                recs.append({"system": sp_text, "user": har, "label": 0, "group": f"dir_harm|{sp}|{idi}|{pi}"})
                meta.append((sp, idi, pi))
    Xm, ym, gm = build_deltas(provider, recs, normalize=True)
    pm = pipe.predict_proba(Xm)[:, 1]
    def s(cond, sp, idi, pi):
        m = gm == f"{cond}|{sp}|{idi}|{pi}"
        return pm[m][0] if m.sum() else None
    ov_diffs, dir_diffs, ov_all, dir_all = [], [], [], []
    for (sp, idi, pi) in meta:
        ob, oh = s("ov_ben", sp, idi, pi), s("ov_harm", sp, idi, pi)
        db, dh = s("dir_ben", sp, idi, pi), s("dir_harm", sp, idi, pi)
        if None not in (ob, oh): ov_diffs.append(ob - oh); ov_all += [ob, oh]
        if None not in (db, dh): dir_diffs.append(db - dh); dir_all += [db, dh]
    yv = np.concatenate([np.ones(len(ov_all)), np.zeros(len(dir_all))])
    sv = np.array(ov_all + dir_all)
    matched_auc = float(roc_auc_score(yv, sv))
    print(f"  MATCHED override-vs-direct AUROC (9B) = {matched_auc:.4f}")

    report = {
        "stage": "scaling_9b", "model": args.model, "tau": tau,
        "pooled_auc": pooled,
        "H1": {"mean_per_family": mean_fam, "ci95": [h1_lo, h1_hi], "per_family": per_fam},
        "H2": {"C3_minus_C1": t31, "C2_minus_C4": t24},
        "H3": {"fpr_constraint": fpr_con, "override_recall": rec_ov},
        "matched_H2": {"override_vs_direct_auc": matched_auc,
                       "harm_override": tost_paired(ov_diffs), "harm_direct": tost_paired(dir_diffs)},
        "compare_4b": {"pooled_auc": 0.8947, "H1_mean": 0.8774, "matched_auc": 0.480,
                       "H3_fpr": 0.018, "tfidf_auc": 0.9288},
    }
    json.dump(report, open(args.report, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  report -> {args.report}")


if __name__ == "__main__":
    main()

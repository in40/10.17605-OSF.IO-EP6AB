"""
t3_bilingual_exploratory.py — EXPLORATORY (post-registration, T3): effect of
adding Russian training data on cross-lingual generalization.

This is NOT a confirmatory test and does NOT change the registered verdict.
It answers a secondary question: does training on EN+RU improve performance on
Russian, vs the EN-only confirmatory probe?

Pre-registration compliance (hard rules enforced here):
  1. The confirmatory EN test split is NEVER read by this script.
  2. Fitting uses ONLY train+val records: EN(train+val) + RU(train+val).
     A hard assert blocks any split=="test" record from entering the fit.
  3. Evaluation uses ONLY the Russian test split (a separate exploratory
     artifact = MT translation), never the EN confirmatory test.
  4. The report is explicitly labeled exploratory / post-registration.
  5. probe_A (EN-only) is the confirmatory probe from T1; it is reused here
     only to compare against probe_B on RU(test). Its confirmatory result
     (from T2) is unaffected.

Run AFTER T1 (needs probe_A) and AFTER T2 (so the single EN unblind is done):
  cd ~/SymFw && python t3_bilingual_exploratory.py \
      --probe_a /mnt/data2/models/probe_qwen35-4b-t1.pkl \
      --en data/labeled_2x2.jsonl \
      --ru data/labeled_2x2_ru.jsonl \
      --out_b /mnt/data2/models/probe_qwen35-4b-t1b_bilingual.pkl \
      --report t3_bilingual_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_trainer_hf import TrainConfig, HFProvider, build_deltas, build_sklearn_pipeline, load_jsonl
from t1_train import to_probe_record, pick_threshold
from sklearn.metrics import roc_auc_score


def load_split(path: str, splits: set) -> List[Dict]:
    out = []
    for r in load_jsonl(path):
        if r.get("split") in splits:
            out.append(to_probe_record(r))
    return out


def eval_on(probe, samples, provider, tau, label):
    X, y, g = build_deltas(provider, samples, normalize=True)
    p = probe.predict_proba(X)[:, 1]
    res = {"n": int(len(X)), "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None}
    # override recall (label==1) and benign-constraint FPR (group==CONSTRAINT)
    ov = y == 1
    if ov.sum():
        res["override_recall_at_tau"] = float((p[ov] >= tau).mean())
    con = g == "CONSTRAINT"
    if con.sum():
        res["constraint_fpr_at_tau"] = float((p[con] >= tau).mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_a", default="/mnt/data2/models/probe_qwen35-4b-t1.pkl")
    ap.add_argument("--en", default="data/labeled_2x2.jsonl")
    ap.add_argument("--ru", default="data/labeled_2x2_ru.jsonl")
    ap.add_argument("--out_b", default="/mnt/data2/models/probe_qwen35-4b-t1b_bilingual.pkl")
    ap.add_argument("--report", default="t3_bilingual_report.json")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(8)
    import pickle

    art_a = pickle.load(open(args.probe_a, "rb"))
    pipe_a = art_a["pipeline"]
    tau_a = float(art_a["threshold"])
    print("=== T3 EXPLORATORY: bilingual (EN+RU) training vs EN-only ===")
    print(f"  probe_A (EN-only) tau={tau_a:.4f}")

    # ---- Fit set: EN(train+val) + RU(train+val). NEVER test. ----
    en_fit = load_split(args.en, {"train", "val"})
    ru_fit = load_split(args.ru, {"train", "val"})
    fit = en_fit + ru_fit
    # HARD LEAK GUARD
    assert all(r["split"] != "test" for r in fit), "TEST RECORD LEAKED INTO BILINGUAL FIT"
    print(f"  fit: EN(train+val)={len(en_fit)}  RU(train+val)={len(ru_fit)}  total={len(fit)}")

    cfg = TrainConfig(
        model_id=art_a["meta"]["model_id"],
        layers=tuple(art_a["meta"]["layers"]),
        pool=art_a["meta"]["pool"],
        max_len=args.max_len,
        batch_size=args.batch_size,
        pca_dim=art_a["meta"]["pca_dim"],
        target_recall=args.target_recall,
        normalize_by_baseline=True,
        seed=args.seed,
        device="cpu",
    )
    provider = HFProvider(cfg)

    Xf, yf, gf = build_deltas(provider, fit, normalize=True)
    pipe_b = build_sklearn_pipeline(cfg).fit(Xf, yf)

    # tau for probe_B on the bilingual val portion
    val_mask = np.array([r["split"] == "val" for r in fit])
    tau_b, rec_b, fpr_b = pick_threshold(yf[val_mask], pipe_b.predict_proba(Xf[val_mask])[:, 1],
                                         args.target_recall)
    print(f"  probe_B (EN+RU) tau={tau_b:.4f} (val recall {rec_b:.3f})")

    # ---- Eval: RU(test) ONLY. Confirmatory EN test NOT read. ----
    ru_test = load_split(args.ru, {"test"})
    print(f"  eval on RU(test) only: n={len(ru_test)}  (EN confirmatory test untouched)")

    res_a = eval_on(pipe_a, ru_test, provider, tau_a, "probe_A_on_RUtest")
    res_b = eval_on(pipe_b, ru_test, provider, tau_b, "probe_B_on_RUtest")

    print("\n  === RU(test) comparison ===")
    print(f"  probe_A (EN-only):  AUROC={res_a['auc']:.4f}  "
          f"ov_recall={res_a.get('override_recall_at_tau', float('nan')):.3f}  "
          f"con_FPR={res_a.get('constraint_fpr_at_tau', float('nan')):.3f}")
    print(f"  probe_B (EN+RU):    AUROC={res_b['auc']:.4f}  "
          f"ov_recall={res_b.get('override_recall_at_tau', float('nan')):.3f}  "
          f"con_FPR={res_b.get('constraint_fpr_at_tau', float('nan')):.3f}")
    if res_a["auc"] is not None and res_b["auc"] is not None:
        print(f"  delta AUROC (B - A): {res_b['auc'] - res_a['auc']:+.4f}")

    # save probe_B
    with open(args.out_b, "wb") as f:
        pickle.dump({"pipeline": pipe_b, "threshold": tau_b,
                    "meta": {**art_a["meta"], "stage": "T3_exploratory_bilingual",
                            "n_fit": int(len(fit)), "n_en_fit": len(en_fit), "n_ru_fit": len(ru_fit)}}, f)
    print(f"\n  saved probe_B -> {args.out_b}")

    report = {
        "stage": "T3_bilingual_exploratory",
        "exploratory": True,
        "post_registration": True,
        "does_not_affect_confirmatory": True,
        "fit": {"en_trainval": len(en_fit), "ru_trainval": len(ru_fit), "total": len(fit)},
        "eval_on": "RU(test) only; EN confirmatory test NOT read",
        "tau_a": tau_a, "tau_b": tau_b,
        "probe_A_on_RUtest": res_a,
        "probe_B_on_RUtest": res_b,
        "delta_auc": (res_b["auc"] - res_a["auc"]) if (res_a["auc"] and res_b["auc"]) else None,
        "note": "Secondary analysis: effect of adding RU training data on cross-lingual generalization. "
                "RU is MT-translated (opus-mt-en-ru); translation quality is a limitation. "
                "Not a confirmatory hypothesis test.",
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  report -> {args.report}")
    print("  EXPLORATORY only. Confirmatory verdict remains from T2 (EN test).")


if __name__ == "__main__":
    main()

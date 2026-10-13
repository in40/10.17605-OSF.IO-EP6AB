"""
t1_train.py — T1: fit the override probe on the FROZEN train+val split.

Pre-registration rule (prereg §7 / starting-stopping rules):
  - T1 uses ONLY records with split in {train, val}.
  - The test split is NEVER read for fitting or threshold selection.
  - Threshold tau is chosen on VAL at target recall (0.95).
  - Test metrics are computed ONLY in T2 (single unblind), a separate script.

Label = O (override): C1,C3 -> 1 ; C2,C4,CONSTRAINT -> 0.
Delta = pool(system+user) - pool(system), normalized by ||pool(system)||.
Pipeline = StandardScaler -> PCA(16, whiten) -> LogisticRegression(C=1, balanced).

Run on the CPU box (Qwen3.5-4B, bf16, 8 threads):
  cd ~/SymFw && python t1_train.py \
      --data data/labeled_2x2.jsonl \
      --out /mnt/data2/models/probe_qwen35-4b-t1.pkl
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
from manifest_spec import SYSTEM_PROMPTS


def to_probe_record(r: Dict) -> Dict:
    """Map a labeled_2x2 record -> build_deltas() input schema."""
    sp_id = r["system_prompt_id"]
    if sp_id not in SYSTEM_PROMPTS:
        raise KeyError(f"unknown system_prompt_id: {sp_id}")
    return {
        "system": SYSTEM_PROMPTS[sp_id][1],   # full system prompt text
        "user": r["user_text"],
        "label": int(r["O"]),                # override = positive class
        "group": r["cell"],                  # C1/C2/C3/C4/CONSTRAINT for reporting
        "split": r["split"],
        "family": r.get("attack_family"),
        "ctype": r.get("constraint_type"),
    }


def pick_threshold(y_true, p, target_recall: float):
    """Smallest threshold achieving >= target_recall on the given set."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, p)
    idx = int(np.argmax(tpr >= target_recall))
    if tpr[idx] < target_recall:
        return float(thr[-1]), float(tpr[-1]), float(fpr[-1])
    return float(thr[idx]), float(tpr[idx]), float(fpr[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/labeled_2x2.jsonl")
    ap.add_argument("--model", default="/mnt/data2/models/Qwen3.5-4B")
    ap.add_argument("--out", default="/mnt/data2/models/probe_qwen35-4b-t1.pkl")
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 6, 12])
    ap.add_argument("--pca-dim", type=int, default=16)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(8)

    # ---- 1. Load + hard-guard the split ----
    raw = load_jsonl(args.data)
    recs = [to_probe_record(r) for r in raw]

    train_recs = [r for r in recs if r["split"] == "train"]
    val_recs = [r for r in recs if r["split"] == "val"]
    test_recs = [r for r in recs if r["split"] == "test"]

    # SAFETY: T1 must never touch test.
    assert len(train_recs) > 0 and len(val_recs) > 0, "missing train or val records"
    assert len(test_recs) > 0, "no test records found — wrong dataset?"
    # Confirm we are NOT including any test record in fitting:
    fitting = train_recs + val_recs
    assert all(r["split"] != "test" for r in fitting), "TEST RECORD LEAKED INTO T1"

    print("=== T1: fit override probe (train+val only) ===")
    print(f"  train={len(train_recs)}  val={len(val_recs)}  test={len(test_recs)} (NOT touched)")
    print(f"  model={args.model}  layers={args.layers}  pca={args.pca_dim}  "
          f"max_len={args.max_len}  batch={args.batch_size}  target_recall={args.target_recall}")

    # sanity: both classes present in train and val
    for name, rs in [("train", train_recs), ("val", val_recs)]:
        ys = [r["label"] for r in rs]
        assert 0 in ys and 1 in ys, f"{name} missing a class"
        print(f"  {name}: pos(override)={sum(ys)}  neg={len(ys)-sum(ys)}")

    # ---- 2. Extract deltas ----
    cfg = TrainConfig(
        model_id=args.model,
        layers=tuple(args.layers),
        pool="last",
        max_len=args.max_len,
        batch_size=args.batch_size,
        pca_dim=args.pca_dim,
        target_recall=args.target_recall,
        normalize_by_baseline=True,
        seed=args.seed,
        device="cpu",
    )
    provider = HFProvider(cfg)

    print("  extracting train deltas...")
    Xtr, ytr, gtr = build_deltas(provider, train_recs, normalize=True)
    print("  extracting val deltas...")
    Xva, yva, gva = build_deltas(provider, val_recs, normalize=True)
    print(f"  Xtr={Xtr.shape}  Xva={Xva.shape}")

    # ---- 3. Fit pipeline on TRAIN only ----
    pipe = build_sklearn_pipeline(cfg).fit(Xtr, ytr)

    # ---- 4. Pick tau on VAL at target recall ----
    p_val = pipe.predict_proba(Xva)[:, 1]
    tau, rec_at, fpr_at = pick_threshold(yva, p_val, args.target_recall)
    from sklearn.metrics import roc_auc_score
    auc_val = float(roc_auc_score(yva, p_val))

    print("=== VAL metrics (tau selection) ===")
    print(f"  val AUROC: {auc_val:.4f}")
    print(f"  tau: {tau:.4f}  (target recall {args.target_recall})")
    print(f"  recall@tau: {rec_at:.4f}")
    print(f"  fpr@tau:    {fpr_at:.4f}")
    print("  per-cell mean_p on val:")
    for cell in ["C1", "C2", "C3", "C4", "CONSTRAINT"]:
        m = gva == cell
        if m.sum() == 0:
            continue
        print(f"    {cell:10s} n={int(m.sum()):3d} mean_p={float(p_val[m].mean()):.3f} "
              f"flagged={int((p_val[m] >= tau).sum())}")

    # ---- 5. Save artifact (NO test metrics) ----
    artifact = {
        "pipeline": pipe,
        "threshold": tau,
        "meta": {
            "stage": "T1",
            "model_id": cfg.model_id,
            "layers": list(cfg.layers),
            "pool": cfg.pool,
            "pca_dim": cfg.pca_dim,
            "normalize_by_baseline": True,
            "max_len": cfg.max_len,
            "seed": cfg.seed,
            "target_recall": args.target_recall,
            "n_train": int(len(Xtr)),
            "n_val": int(len(Xva)),
            "n_test_withheld": int(len(test_recs)),
            "val_auc": auc_val,
            "val_recall_at_tau": rec_at,
            "val_fpr_at_tau": fpr_at,
            "label": "O (override=1)",
        },
    }
    with open(args.out, "wb") as f:
        import pickle
        pickle.dump(artifact, f)
    print(f"\nsaved T1 artifact -> {args.out}")
    print("T1 done. Test remains blinded. Run T2 (separate) only after OSF approval.")


if __name__ == "__main__":
    main()

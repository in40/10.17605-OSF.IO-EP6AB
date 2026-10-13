"""
t4_deberta.py — DeBERTa off-the-shelf injection-detector baseline (§11c).
Zero-shot: pretrained model (protectai/deberta-v3-base-prompt-injection-v2),
NOT trained on our split. Evaluated against our O (override) label on the test set.

CPU ONLY (torch.set_num_threads(8)); no GPU, no llama-server interaction.

Run on remote:
  cd ~/SymFw && setsid python3 t4_deberta.py \
      --data data/labeled_2x2.jsonl \
      --report t4_deberta_report.json > t4_deberta_run.log 2>&1 &
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import roc_auc_score
from probe_trainer_hf import load_jsonl

MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
BATCH = 16
MAX_LEN = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/labeled_2x2.jsonl")
    ap.add_argument("--report", default="t4_deberta_report.json")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(8)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"loading {MODEL} (CPU)...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to("cpu").eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print("labels:", id2label)
    inj_idx = [i for i, l in id2label.items() if "INJ" in l.upper()]
    inj_idx = inj_idx[0] if inj_idx else 1

    raw = load_jsonl(args.data)
    test = [r for r in raw if r["split"] == "test"]
    texts = [r["user_text"] for r in test]
    O = np.array([r["O"] for r in test])
    cells = np.array([r["cell"] for r in test])
    print(f"test n={len(test)}")

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to("cpu")
            out = model(**enc).logits
            p = torch.softmax(out, dim=1)[:, inj_idx].cpu().numpy()
            probs.extend(p)
            if (i // BATCH) % 10 == 0:
                sys.stdout.write(f"    {min(i+BATCH, len(texts))}/{len(texts)}\n"); sys.stdout.flush()
    probs = np.array(probs)

    auc = float(roc_auc_score(O, probs))
    print(f"\n  DeBERTa AUROC vs O (override): {auc:.4f}")
    print(f"  (compare: delta 0.895, TF-IDF 0.929)")

    per_cell = {}
    for c in ["C1", "C2", "C3", "C4", "CONSTRAINT"]:
        m = cells == c
        if m.sum() == 0: continue
        e = {"n": int(m.sum()), "mean_p_injection": float(probs[m].mean())}
        if c in ("C1", "C3"):
            e["recall_at_0.5"] = float((probs[m] >= 0.5).mean())
        if c in ("C2", "C4", "CONSTRAINT"):
            e["fpr_at_0.5"] = float((probs[m] >= 0.5).mean())
        per_cell[c] = e
        print(f"    {c:10} n={e['n']:4d} mean_p={e['mean_p_injection']:.3f} " +
              (f"recall@0.5={e.get('recall_at_0.5', float('nan')):.3f}" if 'recall_at_0.5' in e
               else f"fpr@0.5={e.get('fpr_at_0.5', float('nan')):.3f}"))

    report = {
        "stage": "T4_deberta_baseline",
        "model": MODEL,
        "zero_shot": True,
        "n_test": len(test),
        "auroc_vs_override": auc,
        "per_cell": per_cell,
        "comparison": {"delta_auc": 0.8947, "tfidf_auc": 0.9288, "deberta_auc": auc},
        "caveats": [
            "DeBERTa is a zero-shot pretrained detector, NOT trained on our split.",
            "Its 'injection' concept may differ from our 'override' construct (O).",
            "Threshold 0.5 is DeBERTa's native operating point, not our tau.",
        ],
    }
    json.dump(report, open(args.report, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  report -> {args.report}")


if __name__ == "__main__":
    main()

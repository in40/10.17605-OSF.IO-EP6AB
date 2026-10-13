"""
t4_e2_layers.py — E2 (prereg §15): layer-wise decodability of override.

One forward pass per item yields all hidden states; we pool the last token at each
of a ladder of layers and ask: at which depth is the override signal linearly
decodable, and does DEEPER help the families that failed at {3,6,12}
(F5 encoding 0.748, F8 indirect 0.803, F6 multilingual 0.857)?

For each layer L: delta_L = pool_L(sys+user) - pool_L(sys), normalized by
||pool_L(sys)||. Fit Scaler->PCA(16)->LogReg on train+val, AUROC on test
(pooled + per held-out family).

Run on remote:
  cd ~/SymFw && setsid python3 t4_e2_layers.py \
      --model /mnt/data2/models/Qwen3.5-4B \
      --data data/labeled_2x2.jsonl \
      --report t4_e2_report.json > t4_e2_run.log 2>&1 &
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from probe_trainer_hf import load_jsonl
from manifest_spec import SYSTEM_PROMPTS

LADDER = [1, 4, 8, 12, 16, 20, 24, 28, 31]
MAX_LEN = 256
BATCH = 32
PCA_DIM = 16
SEED = 42


def last_token_pool(hidden, attn):
    lengths = attn.sum(1) - 1
    idx = lengths.clamp(min=0)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]  # [b, d]


def extract_layer_vecs(model, tok, texts):
    """Return [n, len(LADDER), d] last-token vectors at LADDER layers."""
    import torch
    out_all = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                     max_length=MAX_LEN)
            o = model(**enc, output_hidden_states=True)
            vecs = [last_token_pool(o.hidden_states[L], enc["attention_mask"]).float().cpu().numpy()
                    for L in LADDER]
            out_all.append(np.stack(vecs, axis=1))  # [b, nL, d]
            if i % (BATCH*10) == 0:
                sys.stdout.write(f"    extracted {i+len(batch)}/{len(texts)}\n"); sys.stdout.flush()
    return np.concatenate(out_all, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/data2/models/Qwen3.5-4B")
    ap.add_argument("--data", default="data/labeled_2x2.jsonl")
    ap.add_argument("--report", default="t4_e2_report.json")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try:
        from transformers import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = model.to("cpu").eval()

    raw = load_jsonl(args.data)
    trainval = [r for r in raw if r["split"] in ("train", "val")]
    test = [r for r in raw if r["split"] == "test"]

    # baselines per unique system prompt (all layers)
    uniq_sp = sorted({r["system_prompt_id"] for r in raw})
    sp_texts = [SYSTEM_PROMPTS[s][1] for s in uniq_sp]
    print("extracting baselines (all layers)...")
    base = extract_layer_vecs(model, tok, sp_texts)  # [nSP, nL, d]
    base_map = {s: base[i] for i, s in enumerate(uniq_sp)}

    def full_vecs(recs):
        texts = [SYSTEM_PROMPTS[r["system_prompt_id"]][1] + "\n\n" + r["user_text"] for r in recs]
        return extract_layer_vecs(model, tok, texts)

    print("extracting train+val (all layers)...")
    Ftr = full_vecs(trainval)
    ytr = np.array([r["O"] for r in trainval])
    print("extracting test (all layers)...")
    Fte = full_vecs(test)
    yte = np.array([r["O"] for r in test])
    fam_te = np.array([str(r["attack_family"]) for r in test])

    results = {}
    for li, L in enumerate(LADDER):
        # deltas at layer L
        dtr = Ftr[:, li, :] - np.stack([base_map[r["system_prompt_id"]][li] for r in trainval])
        dte = Fte[:, li, :] - np.stack([base_map[r["system_prompt_id"]][li] for r in test])
        ntr = np.linalg.norm(np.stack([base_map[r["system_prompt_id"]][li] for r in trainval]), axis=1, keepdims=True) + 1e-8
        nte = np.linalg.norm(np.stack([base_map[r["system_prompt_id"]][li] for r in test]), axis=1, keepdims=True) + 1e-8
        dtr = dtr / ntr; dte = dte / nte
        probe = Pipeline_fit(dtr, ytr)
        pte = probe.predict_proba(dte)[:, 1]
        pooled = float(roc_auc_score(yte, pte))
        per_fam = {}
        for F in sorted(set(fam_te) - {"None"}):
            pos = pte[(fam_te == F) & (yte == 1)]
            neg = pte[yte == 0]
            if len(pos) > 0:
                per_fam[F] = float(roc_auc_score(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]),
                                                np.concatenate([pos, neg])))
        results[L] = {"pooled_auc": pooled, "per_family": per_fam}
        print(f"  layer {L:2d}: pooled AUROC={pooled:.4f}  F5={per_fam.get('F5',float('nan')):.3f} "
              f"F8={per_fam.get('F8',float('nan')):.3f} F6={per_fam.get('F6',float('nan')):.3f}")

    json.dump({"stage": "T4_E2_layerwise", "ladder": LADDER, "results": results},
              open(args.report, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  report -> {args.report}")


def Pipeline_fit(X, y):
    from sklearn.pipeline import Pipeline
    return Pipeline([("s", StandardScaler()),
                    ("p", PCA(n_components=PCA_DIM, whiten=True, random_state=SEED)),
                    ("c", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))]).fit(X, y)


if __name__ == "__main__":
    main()

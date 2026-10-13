"""
Probe Trainer (REAL) — scaffold для GPU-машины: обучение проба на реальных
hidden states через HuggingFace transformers + sklearn.

НЕ запускается в этой среде (нет torch/transformers/sklearn). Это заготовка
для переноса: mock-версия (probe_trainer.py) валидирует механику, эта —
боевой артефакт под реальную LLM.

Запуск (на GPU-боксе):
  python -m SymFw.probe_trainer_hf \
      --model Qwen/Qwen3-8B --layers 3 6 12 --pool last \
      --data data/labeled.jsonl --out probe_qwen3-8b.pkl \
      --pca-dim 128 --target-recall 0.95

Формат data/labeled.jsonl (по строке на пример):
  {"system": "...", "user": "...", "label": 0, "group": "clean"}
  {"system": "...", "user": "...", "label": 1, "group": "injection"}

vLLM caveat: vLLM не отдаёт per-layer hidden states из коробки для произвольных
промптов. Для проб-обучения используйте HF (полный контроль output_hidden_states).
Для продеплоя с vLLM нужен либо кастомный worker, либо отдельный hidden-state
экстрактор (HF) рядом с vLLM-сервером. VLLMProvider ниже — заглушка под это.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class TrainConfig:
    model_id: str
    layers: Tuple[int, ...] = (3, 6, 12)
    pool: str = "last"
    max_len: int = 1024
    batch_size: int = 8
    pca_dim: int = 128
    target_recall: float = 0.95
    l2: float = 1.0
    normalize_by_baseline: bool = True
    seed: int = 0
    device: str = "cuda"


class HFProvider:
    """Pooled hidden states из реальных слоёв HF-модели (замороженной)."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Для мультимодальных (Qwen3.5) берём языковую часть — она содержит все трансформер-слои
        try:
            from transformers import Qwen3_5ForCausalLM
            self.model = Qwen3_5ForCausalLM.from_pretrained(
                cfg.model_id, torch_dtype=torch.bfloat16)
        except ImportError:
            self.model = AutoModel.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16)
        self.model = self.model.to(cfg.device).eval()

    @torch.no_grad()
    def pool(self, texts: List[str]) -> np.ndarray:
        enc = self.tok(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.cfg.max_len,
        ).to(self.cfg.device)
        attn = enc["attention_mask"]
        out = self.model(**enc, output_hidden_states=True)
        vecs = [self._pool_masked(out.hidden_states[L], attn) for L in self.cfg.layers]
        return torch.cat(vecs, dim=1).float().cpu().numpy()  # [b, len(layers)*d]

    def _pool_masked(self, hs: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        if self.cfg.pool == "last":
            lengths = attn.sum(1) - 1
            idx = lengths.clamp(min=0)
            return hs[torch.arange(hs.size(0), device=hs.device), idx]
        if self.cfg.pool == "mean":
            m = attn.unsqueeze(-1).float()
            return (hs * m).sum(1) / m.sum(1).clamp(min=1)
        if self.cfg.pool == "weighted":
            seq = hs.shape[1]
            w = torch.linspace(0.5, 1.5, seq, device=hs.device).unsqueeze(0) * attn.float()
            w = w / w.sum(1, keepdim=True).clamp(min=1e-9)
            return (hs * w.unsqueeze(-1)).sum(1)
        raise ValueError(f"unknown pool: {self.cfg.pool}")


class VLLMProvider:
    """Заглушка: vLLM не отдаёт per-layer hidden states из коробки."""

    def __init__(self, *a, **k):
        raise NotImplementedError(
            "vLLM не отдаёт hidden states для произвольных промптов. "
            "Используй HFProvider для обучения; для продеплоя — кастомный "
            "worker или отдельный HF hidden-state экстрактор."
        )


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_deltas(provider, samples: List[Dict], normalize: bool = True):
    """Two-pass: DELTA = pool(system+user) - pool(system), baseline кэшируется."""
    import os

    # Group by system prompt to minimize baseline passes
    sys_to_samples: Dict[str, list] = {}
    for s in samples:
        sys_to_samples.setdefault(s["system"], []).append(s)

    # One forward pass per unique system prompt (baseline)
    baselines: Dict[str, np.ndarray] = {}
    for sp, sp_samples in sys_to_samples.items():
        baselines[sp] = provider.pool([sp])[0]

    # Batch all full text forward passes
    all_full_prompts = []
    sample_order = []
    for s in samples:
        all_full_prompts.append(f"{s['system']}\n\n{s['user']}")
        sample_order.append(s)

    # Process in batches
    batch_size = getattr(provider.cfg, "batch_size", 32)
    X, y, g = [], [], []
    total = len(all_full_prompts)

    for batch_start in range(0, total, batch_size):
        batch_prompts = all_full_prompts[batch_start:batch_start + batch_size]
        batch_vecs = provider.pool(batch_prompts)

        for local_idx, vec in enumerate(batch_vecs):
            idx = batch_start + local_idx
            s = sample_order[idx]
            sp = s["system"]
            d = vec - baselines[sp]
            if normalize:
                d = d / (np.linalg.norm(baselines[sp]) + 1e-8)
            X.append(d)
            y.append(int(s["label"]))
            g.append(s.get("group", "?"))
            if len(X) % 50 == 0:
                sys.stdout.write(f"    deltas {len(X)}/{total}\n")
                sys.stdout.flush()

        sys.stdout.write(f"    batch {batch_start//batch_size+1} done\n")
        sys.stdout.flush()

    return np.array(X, dtype=np.float32), np.array(y), np.array(g)


def build_sklearn_pipeline(cfg: TrainConfig) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=cfg.pca_dim, whiten=True, random_state=cfg.seed)),
        ("clf", LogisticRegression(max_iter=2000, C=cfg.l2, class_weight="balanced")),
    ])


def train(cfg: TrainConfig, X, y, groups):
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        X, y, groups, test_size=0.25, stratify=groups, random_state=cfg.seed
    )
    pipe = build_sklearn_pipeline(cfg).fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]

    fpr, tpr, thr = roc_curve(yte, p)
    idx = int(np.argmax(tpr >= cfg.target_recall))
    threshold = float(thr[idx]) if idx < len(thr) else 1.0
    fpr_at = float(fpr[idx])
    rec_at = float(tpr[idx])
    auc = float(roc_auc_score(yte, p))

    per_group = {}
    for grp in np.unique(gte):
        m = gte == grp
        per_group[str(grp)] = {
            "n": int(m.sum()),
            "mean_p": float(p[m].mean()),
            "flagged": int((p[m] >= threshold).sum()),
        }

    artifact = {
        "pipeline": pipe,
        "threshold": threshold,
        "meta": {
            "model_id": cfg.model_id, "layers": list(cfg.layers), "pool": cfg.pool,
            "pca_dim": cfg.pca_dim, "target_recall": cfg.target_recall,
            "normalize_by_baseline": cfg.normalize_by_baseline,
            "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        },
    }
    metrics = {
        "auc_test": auc, "threshold": threshold,
        "recall_at_thr": rec_at, "fpr_at_thr": fpr_at, "per_group": per_group,
    }
    return artifact, metrics


def cross_task_eval(cfg: TrainConfig, samples: List[Dict], provider):
    """Train на одних system_prompt'ах, test на других — мера переносимости."""
    train_sys = {s for s in samples if s.get("task", "A") == "A"}
    test_sys = {s for s in samples if s.get("task", "A") == "B"}
    Xtr, ytr, _ = build_deltas(provider, list(train_sys), cfg.normalize_by_baseline)
    Xte, yte, _ = build_deltas(provider, list(test_sys), cfg.normalize_by_baseline)
    pipe = build_sklearn_pipeline(cfg).fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 6, 12])
    ap.add_argument("--pool", default="last", choices=["last", "mean", "weighted"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="probe_real.pkl")
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = TrainConfig(
        model_id=args.model, layers=tuple(args.layers), pool=args.pool,
        pca_dim=args.pca_dim, target_recall=args.target_recall, l2=args.l2,
        batch_size=args.batch_size, device=args.device,
    )
    provider = HFProvider(cfg)
    samples = load_jsonl(args.data)
    X, y, groups = build_deltas(provider, samples, cfg.normalize_by_baseline)
    artifact, m = train(cfg, X, y, groups)

    print("=== Real probe training ===")
    print(f"model={cfg.model_id} layers={cfg.layers} pool={cfg.pool}")
    print(f"n={len(X)}  test AUROC={m['auc_test']:.4f}")
    print(f"threshold={m['threshold']:.3f}  recall@thr={m['recall_at_thr']:.3f}  fpr@thr={m['fpr_at_thr']:.3f}")
    for g, d in m["per_group"].items():
        print(f"  {g:10s} n={d['n']:3d} mean_p={d['mean_p']:.3f} flagged={d['flagged']}")

    with open(args.out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

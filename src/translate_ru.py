"""
translate_ru.py — EXPLORATORY: faithful EN->RU translation of the 2x2 dataset.

Uses Helsinki-NLP/opus-mt-en-ru (small Marian MT, CPU-fast, separate from the
probe model to avoid circularity). Output is a SEPARATE file; the frozen
labeled_2x2.jsonl is NOT modified.

Each RU record keeps identical metadata (cell/O/H/family/ctype/SP/split) and
carries `_en_original` for audit.

Run on remote (background, CPU):
  cd ~/SymFw && setsid python3 translate_ru.py > data/translate_ru.log 2>&1 &
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

SRC = Path("data/labeled_2x2.jsonl")
DST = Path("data/labeled_2x2_ru.jsonl")
BATCH = 32
MAX_LEN = 256

def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

def main():
    torch.set_num_threads(8)
    recs = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    distinct = sorted({r["user_text"] for r in recs})
    log(f"total={len(recs)}  distinct={len(distinct)}")

    log("loading opus-mt-en-ru...")
    tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    model.eval()

    cache = {}
    with torch.no_grad():
        for i in range(0, len(distinct), BATCH):
            batch = distinct[i:i + BATCH]
            enc = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=MAX_LEN)
            out = model.generate(**enc, max_new_tokens=MAX_LEN)
            trans = tok.batch_decode(out, skip_special_tokens=True)
            for s, t in zip(batch, trans):
                cache[s] = t.strip()
            if (i // BATCH) % 5 == 0:
                log(f"  translated {min(i+BATCH, len(distinct))}/{len(distinct)}")

    # compose output
    with open(DST, "w", encoding="utf-8") as f:
        for r in recs:
            ru = cache.get(r["user_text"], r["user_text"])
            out = dict(r)
            out["_en_original"] = r["user_text"]
            out["user_text"] = ru
            out["language"] = "ru"
            out["source"] = r.get("source", "") + "_ru_mt"
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    log(f"wrote {len(recs)} RU records -> {DST}")
    log("DONE")

if __name__ == "__main__":
    main()

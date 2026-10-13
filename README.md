# Activation-Delta Probes vs TF-IDF for Prompt-Injection Detection — Pre-Registered Study

**Pre-registration:** OSF `ep6ab` · DOI [10.17605/OSF.IO/EP6AB](https://doi.org/10.17605/OSF.IO/EP6AB) · CC0
**Headline (negative) result:** a plain TF-IDF baseline on raw text outperforms the activation-delta probe, and a source-matched test shows the probe is blind to the override wrapper (AUROC ≈ 0.48, at chance) — the apparent "override" signal is a content/source artifact, robust to model scale (4B→9B).

This repository contains the frozen data, the full analysis pipeline, figures, and the manuscript for the study.

---

## Repository structure

```
symfw-prereg-ep6ab/
├── README.md                  # this file
├── requirements.txt           # pinned dependencies
├── data/
│   ├── labeled_2x2.jsonl      # frozen 2×2 dataset (2850 records)
│   ├── labeled_2x2_ru.jsonl   # MT Russian translation (exploratory arm)
│   ├── pids_bench_v1.0.json   # PIDS-Bench (external validation; MIT/CC-BY/Apache)
│   ├── manifest_final.json    # T0 manifest: SHA-256 hashes + split + targets
│   └── test_deltas.npz        # cached test-set deltas (for reuse)
├── src/
│   ├── manifest_spec.py       # frozen taxonomies + split + T0 validator
│   ├── generate_2x2.py        # builds the 2×2 dataset
│   ├── t0_freeze.py           # T0: freeze manifest + hashes
│   ├── probe_trainer_hf.py    # HF provider, pooling, delta extraction, sklearn pipeline
│   ├── t1_train.py            # T1: fit probe on train+val, pick tau (test untouched)
│   ├── t2_unblind.py          # T2: single confirmatory unblind (H1/H2/H3)
│   ├── t3_pids_external.py    # T3: external H3 validation on PIDS-Bench
│   ├── t3_bilingual_exploratory.py  # T3: EN+RU vs EN on RU-test
│   ├── t4_followup.py         # T4: per-cell, 2×2 decomposition, TF-IDF baseline
│   ├── t4_e2_layers.py        # T4: E2 layer-wise decodability
│   ├── t4_extra.py            # T4: TF-IDF per-cell + bilingual negative transfer
│   ├── t4_deberta.py          # T4: DeBERTa zero-shot baseline
│   ├── t5_source_matched_h2.py       # T5: source-matched H2 (single SP)
│   ├── t5b_source_matched_h2_multi.py# T5: source-matched H2 (multi-SP)
│   ├── scaling_9b.py          # model-scale extension (4B -> 9B)
│   └── translate_ru.py        # MT EN->RU for the bilingual arm
├── figures/                   # generated figures (RU + EN)
│   ├── fig_method.png / fig_method_ru.png
│   ├── fig_percell.png / fig_percell_ru.png
│   ├── fig_roc.png / fig_roc_ru.png
│   └── fig_e2.png / fig_e2_ru.png
├── results/                   # JSON reports (t2/t3/t4/t5/scaling)
├── docs/
│   ├── PREREG_OSF_FULL.md     # full pre-registration text
│   ├── RESULTS.md             # results write-up
│   ├── PRIOR_WORK_POSITIONING.md
│   └── SESSION_SUMMARY.md
└── manuscript/
    └── manuscript_EP6AB.docx  # bilingual (RU+EN) journal manuscript
```

> Note: the working tree also contains early scaffolding and utility scripts (mock detectors, OSF helpers, dataset checks) that are **not** part of the reproduction path; the files under `src/` above are the canonical pipeline.

---

## Environment

- **Hardware:** CPU only (8 threads, AVX2/AVX512). No GPU required.
- **Models:** `Qwen/Qwen3.5-4B` (primary), `Qwen/Qwen3.5-9B` (scale check), `Helsinki-NLP/opus-mt-en-ru` (MT), `protectai/deberta-v3-base-prompt-injection-v2` (baseline). Downloaded from HuggingFace on first use.
- **Python:** 3.10+. Install deps:

```bash
pip install -r requirements.txt
```

---

## Reproduction order

```bash
# T0 — generate + freeze the 2×2 dataset and manifest (hashes)
python src/generate_2x2.py
python src/t0_freeze.py

# T1 — fit the probe on train+val only (test stays blinded)
python src/t1_train.py --data data/labeled_2x2.jsonl --out results/probe_qwen35-4b-t1.pkl

# T2 — single confirmatory unblind (H1/H2/H3)
python src/t2_unblind.py --probe results/probe_qwen35-4b-t1.pkl --data data/labeled_2x2.jsonl --report results/t2_report.json

# T3 — external + exploratory
python src/t3_pids_external.py --probe results/probe_qwen35-4b-t1.pkl --pids data/pids_bench_v1.0.json --report results/t3_pids_report.json --curated-h3-passed
python src/t3_bilingual_exploratory.py --probe_a results/probe_qwen35-4b-t1.pkl --ru data/labeled_2x2_ru.jsonl --report results/t3_bilingual_report.json

# T4 — baselines, decomposition, E2, per-cell, DeBERTa
python src/t4_followup.py --probe results/probe_qwen35-4b-t1.pkl --report results/t4_report.json
python src/t4_e2_layers.py --report results/t4_e2_report.json
python src/t4_extra.py --report results/t4_extra_report.json
python src/t4_deberta.py --report results/t4_deberta_report.json

# T5 — source-matched H2 (the decisive confound test)
python src/t5b_source_matched_h2_multi.py --probe results/probe_qwen35-4b-t1.pkl --report results/t5b_h2_multi_report.json

# Scale — 4B -> 9B
python src/scaling_9b.py --model /path/to/Qwen3.5-9B --report results/scaling_9b_report.json

# Figures + manuscript
python make_figures_ru.py && make_method_figure_ru.py
python build_manuscript.py
```

**Integrity guarantees enforced in code:**
- T1 asserts no `test` record enters fitting.
- T2 refuses to overwrite its report (single unblind) unless `--force`.
- T5 matched design holds idiom + system prompt + length constant, varying only harm.

---

## Data

- `labeled_2x2.jsonl` — 2850 records; fields: `record_id, system_prompt_id, user_text, cell, O, H, attack_family, constraint_type, source, language, template_id, split`.
- Split: train 861 / val 539 / test 1450. Cells: C1 700, C2 300, C3 550, C4 300, CONSTRAINT 1000.
- Frozen hashes (see `manifest_final.json`):
  - `dataset_sha256 = 1efee76b5c1f4d6d8d254cb33b856f6dae61ae46fa91efa64662bf763787ca05`
  - `taxonomy_sha256 = 97aa52f650302e9afe2910d141dd9f75d1fcf453b7a3c74f9db4a1c1fc41f849`

---

## License

- Study code and our data: **CC0 1.0** (matches the OSF registration).
- Third-party data retain their licenses: PIDS-Bench (MIT/CC-BY-4.0/Apache-2.0), AdvBench, Alpaca, DeBERTa — see `data/` and the References for attribution.

## Citation

```
Sorokin, A. (2026). Activation-Delta Probes Compared with TF-IDF for
Prompt-Injection Detection: A Pre-Registered Study. OSF Pre-registration
DOI 10.17605/OSF.IO/EP6AB.
```

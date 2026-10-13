# SESSION SUMMARY — SymFw Semantic Firewall (as of 2026-10-09)

> Safety-net summary before context compaction. Captures goal, state, decisions, artifacts, next steps.
> Secrets (OSF tokens) are NOT in this file — they live in `/root/.osf_test_curl` and `/root/.osf_prod_curl` (chmod 600).

---

## 1. PROJECT / GOAL

**SymFw — "Semantic Firewall":** detect adversarial prompt-injection (an attempted *override* of a system instruction) by reading the internal representations of a **frozen LLM**, specifically the **two-pass delta**:

    Δ = pool(f(system+user)) − pool(f(system))

pool = last non-padding token hidden state at layers {3,6,12}, concatenated → 7680-dim, normalized by ‖pool(system)‖.

The study is now framed as an **applied, pre-registered** evaluation (not a novel-method claim).

---

## 2. INFRASTRUCTURE

- **Remote box:** `45.145.191.144` = `chat.sorokinonline.com` = `mhbn2.sorokinonline.com` (same physical server, multiple nginx vhosts). SSH as `sorokin@chat.sorokinonline.com` with key `/root/.ssh/id_llamatunnel`.
- **Model:** Qwen3.5-4B at `/mnt/data2/models/Qwen3.5-4B` (model_type=qwen3_5, hidden 2560, 32 layers, vocab 248320, config SHA-256 `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`).
- **Compute:** CPU only, 8 cores, bfloat16 (AVX512_BF16). NO GPU available (H100 occupied by llama-server — DO NOT kill it).
- **WebUI:** FastAPI on :8550, nginx proxy, HTTPS at https://mhbn2.sorokinonline.com. Uses probe v2.
- **Doc share:** `/var/www/symfw-docs/` served read-only at https://mhbn2.sorokinonline.com/docs/ (autoindex, .md renders as text).
- **Versions:** torch 2.11.0+cu130, transformers 5.5.0, scikit-learn 1.7.2, numpy 2.2.6.

---

## 3. WHAT WAS BUILT (engineering, already done)

- **Probe v2** (`/mnt/data2/models/probe_qwen35-4b-v2.pkl`): trained on 1481 records (OLD grouping: clean/tricky/advbench/jbb/multilingual). Test AUROC **0.976**, threshold **0.398**, tricky mean p 0.008. Wired into webui (live).
- **WebUI inspector** (`webui.py`): analyze endpoint, /api/stats, per-layer vectors. Fixed: PROBE_PATH→v2, max_len→256, delta normalization matching build_deltas, `.cpu().float().numpy()` (numpy can't do bf16), setsid launcher for reliable detach.
- **Key perf fixes discovered:** float16→bfloat16 (~1.5x), batching 32 (~77x vs single), pool() shape fix `cat(dim=1)` not `dim=0`.

---

## 4. THE PREREGISTRATION (the current focus)

**Registered on production osf.io. Status: PENDING APPROVAL (~48h auto-approve; author cannot self-approve — not a moderator, 403).**
- Registration guid: **`ep6ab`** → https://osf.io/ep6ab/
- Draft id: `6ac910dee14a8f5f57482614`
- Template: **OSF Preregistration** (atomic schema, 6 pages, 87 blocks).
- reviews_state=initial, revision_state=unapproved, public=False.
- To speed up: contact support@osf.io.

**Final prereg version: v17** (went through ~17 iterations + 3 reviewers: lead + Qwen + one more).

### Core design (2×2 override × harm)
| | Harmful (H=1) | Benign (H=0) |
|---|---|---|
| **Override (O=1)** | C1 | C3 |
| **No override (O=0)** | C2 | C4 |
Plus **CONSTRAINT** records (20 benign imperative content-filtering types, O=0/H=0) for H3.

Override coding rule: O=1 iff the turn (a) references/implies the system instruction AND (b) attempts to negate/replace/bypass it.

### Hypotheses
- **H1 (replication+extension):** probe separates override vs non-override on held-out families AND held-out system prompts. Pass: mean per-family AUROC ≥ 0.90, cluster-bootstrap CI lower ≥ 0.85. Requires **k ≥ 8 held-out families** (else descriptive).
- **H2 (discriminant validity):** decodable direction = override, not harm. Pass: pooled AUROC ≥ 0.90 AND two-sided equivalence |mean(C3)−mean(C1)| ≤ 0.10 AND |mean(C2)−mean(C4)| ≤ 0.10 (TOST, clustered by system prompt).
- **H3 (PRIMARY NOVELTY):** benign imperative constraints not flagged. Pass: FPR ≤ 0.10 (CI upper ≤ 0.15) with override recall ≥ 0.90. External validation on PIDS-Bench (if PIDS FPR > 0.20 → "conditionally supported").

### Taxonomies (frozen in manifest_spec.py)
- **10 attack families** F1-F10: direct-override, role-hijack, mode-activation, authority-escalation, encoding-obfuscation, multilingual, CTF-framing, context-embedding-indirect, few-shot-reanchor, delimiter-attack.
- **20 constraint types** T1-T20 (numeric/date/PII/section/quote/URL/jargon/tone/length-cap/entity-scoping/example/citation/code-block/dedup/org-name/register-convert/anecdote/boilerplate/attribution/comparison). Non-exclusion: T9, T10, T16.
- **10 system prompts** SP1-SP10 (train SP1-5, test SP6-10).
- **C3 allowed idioms:** F1,F2,F3,F4,F7,F9,F10 (no encoding/multilingual).

### Novelty positioning (honest)
Method (activation delta + linear probe) = PRIOR ART. Contribution = (1) strict multi-axis leak-free evaluation + cross-system-prompt generalization, (2) benign-constraint false-positive focus + PIDS-Bench external validation, (3) 2×2 override×harm discriminant control.

### Key verified citations (all confirmed real via DOI/arXiv)
Abdelnabi et al. SaTML 2025 (2406.00799, TaskTracker); Liu et al. IPM 2026 (DOI 10.1016/j.ipm.2026.104662, ROD); Ye/Cui/Hadfield-Menell ICML 2026 (2603.12277, Role Confusion); Guo/Zhao/Wang IJCAI 2026 (DOI 10.24963/ijcai.2026/68, BERM); Wang et al. NeurIPS 2025 (2509.03888, False Sense of Security); David et al. ICML 2026 MechInterp (2609.15994, Latent Undertow); Wen et al. (2609.37737, Where Do LLMs Decide); Fay et al. ICLR 2026 (2505.20435, Persistent Homology); Shire (2609.15017, PIDS-Bench); Dong et al. EMNLP 2025 (2025.emnlp-main.1082); Russinovich (2609.36570, CounterSteer); Greshake et al. 2023 (2302.12173).

---

## 5. T0 — DATA FROZEN (done)

- **Dataset:** `data/labeled_2x2.jsonl` — **2850 records**. Split: train 861 / val 539 / test 1450.
  - C1=700, C2=300, C3=550, C4=300, CONSTRAINT=1000.
  - override (O=1)=1250, non-override=1600.
  - test families = F3-F10 (8); test constraint types = 10 (incl non-exclusion T9/T10/T16).
  - All targets met, validator clean, NO leaks (fixed a system-prompt leak in CONSTRAINT records).
- **manifest_final.json** hashes:
  - `dataset_sha256 = 1efee76b5c1f4d6d8d254cb33b856f6dae61ae46fa91efa64662bf763787ca05`
  - `taxonomy_sha256 = 97aa52f650302e9afe2910d141dd9f75d1fcf453b7a3c74f9db4a1c1fc41f849`
- Both files uploaded to the OSF registration (user did manually) + in the share.

---

## 6. KEY FILES

Local `/root/SymFw/` (also pushed to remote `/home/sorokin/SymFw/`):
- `generate_2x2.py` — builds the 2×2 dataset from existing sources (reuses labeled.jsonl harmful/benign).
- `manifest_spec.py` — frozen taxonomies + split + `validate()` (T0 invariants).
- `t0_freeze.py` — runs validator, computes SHA-256, emits `manifest_final.json`, READY_FOR_T1 flag.
- `probe_trainer_hf.py` — HFProvider (bf16), pool() (fixed cat dim=1), build_deltas (batched), train().
- `run_train.py` — training runner (threads=8, batch=32, max_len=256).
- `webui.py` — FastAPI inspector (live on mhbn2).
- `calibration_pilot.py` — train-only leave-families-out pilot (result: train=2 gives ~0.998 held-out AUROC, no learnability threshold).
- `PREREG_OSF_FULL.md` (v17), `REVIEWER_COVER_NOTE.md`, `PRIOR_WORK_POSITIONING.md`, `OSF_FORM_FILL_GUIDE.md`, `REVIEW_REQUESTS.md`.
- `data/labeled_2x2.jsonl`, `manifest_final.json`.

All docs also in share: https://mhbn2.sorokinonline.com/docs/

---

## 7. NEXT STEPS

1. **Wait for OSF approval** (~48h auto, or support@osf.io). Do NOT start T1 until approved.
2. **T1 — `t1_train.py` BUILT + pushed to remote (NOT run yet).**
   - Fits on train+val ONLY; hard assert blocks test leak.
   - label = O (override). τ picked on val at recall 0.95.
   - Saves `probe_qwen35-4b-t1.pkl` (pipeline + τ + meta). No test metrics.
   - Run: `cd ~/SymFw && python t1_train.py --data data/labeled_2x2.jsonl --out /mnt/data2/models/probe_qwen35-4b-t1.pkl`
3. **T2 — `t2_unblind.py` BUILT + pushed to remote (NOT run yet).**
   - SINGLE unblind: refuses to overwrite report unless --force; verifies dataset hash vs manifest.
   - τ taken from T1 artifact (not refit).
   - **H1:** per-family AUROC = (family's override scores vs POOLED non-override negatives), because each test family is override-only. mean ≥0.90, cluster-bootstrap(95%, resample families) lower ≥0.85, k=8.
   - **H2:** TOST equivalence δ=0.10, clustered by system prompt (5 SPs, 90% CI ⊆ [-0.10,0.10]) for C3−C1 and C2−C4, AND pooled AUROC ≥0.90.
   - **H3:** FPR(CONSTRAINT)@τ ≤0.10, bootstrap(95%, resample 10 constraint types) upper ≤0.15, AND override recall(C1+C3) ≥0.90.
   - **PPV** at test prevalence + at assumed deployment prevalence 0.05/0.01.
   - Saves `t2_report.json` with PASS/FAIL per hypothesis.
   - Run: `cd ~/SymFw && python t2_unblind.py --probe /mnt/data2/models/probe_qwen35-4b-t1.pkl --data data/labeled_2x2.jsonl --report t2_report.json`
4. **PIDS-Bench — DOWNLOADED + `t3_pids_external.py` BUILT (ready to run after T1).**
   - Artifact: `data/pids_bench_v1.0.json` (local+remote). v1.0.0, 643 cases, MIT/CC-BY-4.0/Apache-2.0 (attribution required).
   - Subset = published hard-benign split `overdefense_test` (339 ALLOW, >=300 ✓), recorded by name.
   - Each PIDS benign prompt paired with HELD-OUT SP6..SP10; FPR reported per-SP + mean + worst-case.
   - Rule (prereg §4): curated H3 passed AND PIDS mean FPR > 0.20 -> H3 "conditionally supported". PIDS does NOT change curated verdict.
   - BLOCK attack-recall = EXPLORATORY side-note only (--include-attack-recall), NOT in prereg.
   - Run: `python t3_pids_external.py --probe /mnt/data2/models/probe_qwen35-4b-t1.pkl --pids data/pids_bench_v1.0.json --report t3_pids_report.json --curated-h3-passed`
   - NOTE: prereg fit confirmed — no changes to registered hypotheses/thresholds needed; only two implementation details pre-specified (subset name + SP pairing).
5. **Real user logs** — user may get access to real (mostly-benign) production queries with a DIFFERENT system prompt. Use as real-world FPR stress + cross-system-prompt test. Only aggregate stats published (privacy). Pending: volume, whether a sample can be labeled, system-prompt domain.

### Bilingual exploratory experiment (t3_bilingual_exploratory.py) — BUILT
- User idea: run EN-only first (confirmatory A), then EN+RU training as a SECOND experiment (B).
- Compliance (enforced in script):
  1. Confirmatory EN test NEVER read by B.
  2. B fits ONLY on train+val: EN(train+val)=1400 + RU(train+val)=1400 = 2800. Hard assert blocks test.
  3. B evaluates ONLY on RU(test)=1450 (separate MT artifact, not the registered confirmatory test).
  4. Report labeled exploratory / post_registration / does_not_affect_confirmatory.
  5. probe_A (EN-only, from T1) reused only to compare vs probe_B on RU(test); its T2 verdict unaffected.
- Compares: probe_A vs probe_B on RU(test): AUROC, override recall@tau, constraint FPR@tau.
- RU provenance: MT (opus-mt-en-ru); translation quality = documented limitation.
- Verified: RU split ids == EN split ids (same corpus translated); 0 test leak into fit.

### Full compliant run order (ALL gated on OSF approval of ep6ab)
1. `t1_train.py`  -> probe_qwen35-4b-t1.pkl  (CONFIRMATORY fit, EN train+val, tau on val)
2. `t2_unblind.py` -> t2_report.json          (CONFIRMATORY single EN-test unblind: H1/H2/H3)
3. `t3_pids_external.py --curated-h3-passed` -> t3_pids_report.json  (EXTERNAL H3, PIDS overdefense_test 339)
4. `t3_bilingual_exploratory.py` -> t3_bilingual_report.json  (EXPLORATORY: EN+RU vs EN on RU-test)
- Steps 3-4 are exploratory/external; they do NOT alter the step-2 confirmatory verdict.
- Single-unblind integrity: only step 2 reads the EN confirmatory test.

### Test-split structure (verified, for T2 design)
- test n=1450. C1=400, C2=150, C3=250, C4=150, CONSTRAINT=500.
- Each held-out family (F3-F10) is OVERRIDE-ONLY → per-family AUROC uses pooled negatives.
- All 5 test SPs (SP6-SP10) contain both C1&C3 and both C2&C4 → TOST by SP valid.
- CONSTRAINT = 10 types (T9,T10,T13-T20) × 50 → H3 bootstrap by type valid.
- Both T1 and T2 scripts syntax-checked on remote; math (bootstrap/TOST/PPV) unit-tested on synthetic data.

---

## 7b. T2 CONFIRMATORY RESULTS (RUN 2026-10-12, test consumed ONCE)
- tau=0.2579, dataset_sha256 verified = 1efee76b...ca05, n_test=1450.
- **Pooled AUROC: 0.8947**
- **H1 FAIL (marginal):** mean per-family AUROC=0.8774 (<0.90), boot95% CI [0.8279,0.9179] lower<0.85. Strong: F3 .923 F4 .922 F7 .916 F9 .919 F10 .931. Weak: F5(encoding) .748, F8(indirect) .803, F6(multilingual) .857.
- **H2 FAIL (clear):** C3-C1=+0.420 (CI90 .379-.459), C2-C4=+0.363 (CI90 .331-.396), both far outside ±0.10. Decodable direction entangles override WITH HARM — probe reads content salience, not pure override intent.
- **H3 PASS (PRIMARY NOVELTY HOLDS):** FPR(CONSTRAINT)@tau=0.018 (CI [0.000,0.054]) <=0.10; override recall=0.9015 >=0.90. Benign imperative constraints NOT flagged as override.
- **PPV:** test 0.740 | @prev=0.05 0.725 | @prev=0.01 0.336.
- Narrative: novelty (H3) confirmed; replication (H1) below bar; discriminant (H2) refuted (harm/override entanglement). Mixed but publishable. Test locked by pre-registered thresholds; no re-run.
- Report: t2_report.json (local+remote).

## 7c. T3 PIDS EXTERNAL VALIDATION (RUN 2026-10-12) — H3 SUPPORTED EXTERNALLY
- Subset: PIDS-Bench `overdefense_test` (339 hard-benign ALLOW), paired with held-out SP6..SP10.
- FPR@tau per SP: SP6 .0088, SP7 .0678 (worst), SP8 .0029, SP9 .0029, SP10 .0059.
- **mean FPR = 0.0177, worst-case = 0.0678** — far below the 0.20 conditional threshold.
- **Verdict: SUPPORTED** (curated H3 pass + PIDS hard-benign FPR <= 0.20).
- Significance: H3 (primary novelty) generalizes to an INDEPENDENT third-party benchmark, not just our curated constraints. Strong external validity for the over-defense-resistance claim.
- Report: t3_pids_report.json (local+remote).

## 7d. T3 BILINGUAL EXPLORATORY (RUN 2026-10-12) — RU training helps RU
- Fit: EN(train+val)=1400 + RU(train+val)=1400 = 2800. Eval: RU(test)=1450 only. EN confirmatory test NOT read.
- probe_A (EN-only) on RU(test): AUROC 0.8279, override recall@tau 0.600, constraint FPR 0.012.
- probe_B (EN+RU) on RU(test): AUROC 0.8659, override recall@tau 0.854, constraint FPR 0.002.
- Delta: +0.038 AUROC, +0.254 override recall. Bilingual training materially improves Russian detection; benign FPR stays ~0.
- tau_a=0.258, tau_b=0.304. RU is MT (opus-mt-en-ru); translation quality = limitation. Exploratory only.
- Report: t3_bilingual_report.json (local+remote).

## 7e. PREREG STATUS: COMPLETE (confirmatory + registered external done)
- T1 fit ✓, T2 single unblind ✓ (H1 FAIL marginal, H2 FAIL, H3 PASS), T3 PIDS external ✓ (H3 SUPPORTED).
- Overall honest narrative: the activation-delta probe is a robust over-defense-resistant detector for benign imperative constraints (H3, curated + externally validated), but NOT a clean override-specific detector (H2 refuted: harm/override entanglement) and cross-family generalization is below the confirmatory bar (H1 marginal).
- Remaining OPTIONAL (NOT in prereg): T3 bilingual exploratory (EN+RU training vs EN on RU-test).

## 7f. REVIEW RESPONSE + T4 FOLLOW-UPS + RE-SCOPE (2026-10-12)
External review: "strong honest prereg, but don't submit as-is — H2 misinterpreted, missing baselines/E2/per-cell/source-control/re-scope." All correct. Ran T4 follow-ups (all pre-registered §9/§11/§15 or mandated §4):
- **TF-IDF baseline BEATS the delta:** TF-IDF 0.9288 vs delta 0.8947, gap −0.034. By prereg §9 rule (|gap|<0.05) => internal-state delta adds NO value over surface text.
- **TF-IDF per-cell:** CONSTRAINT FPR 0.000, C4 FPR 0.053 => H3 over-defense resistance is a DATA property, NOT delta-specific.
- **2×2 decomposition:** main_O +0.294, main_H −0.031 (≈0), interaction −0.786 => H2 fail is a SOURCE/STYLE interaction, not harm-tracking.
- **Source confound:** cell≡source (C1=advbench_wrapped, C2=advbench, C3=authored, C4=alpaca). Length-matched C3−C1 +0.44 (n=72), C2−C4 +0.33 (n=101) => confound is source not length; not fully separable in frozen data.
- **Per-cell (test):** C1 recall 0.840, C2 FPR 0.987, C3 recall 1.000, C4 FPR 0.327, CONSTRAINT FPR 0.018. Confusion [[594,206],[64,586]].
- **E2 layer-wise:** best single layer L20=0.915 < TF-IDF 0.929. Deeper (12-20) > shallow (1-8), consistent with [7]. F5 encoding irreducible (0.72-0.79 all layers).
- **Bilingual negative transfer:** probe_B on EN test AUROC 0.883 vs probe_A 0.895 (−0.011), recall higher. No severe transfer loss.
- **RE-SCOPE applied (prereg §4):** construct changed from "override detection" to a boundary/cautionary study: activation-delta probes are surface-redundant + source-confounded for prompt-injection.
- **Deviations logged:** re-scope; TOST 90% vs 95% text; τ 0.258 vs illustrative 0.40; bilingual not pre-registered; source-matched H2 not fully achievable; baselines/E2 run post-unblind (were pre-registered).
- New scripts: `t4_followup.py`, `t4_e2_layers.py`, `t4_extra.py`. Reports: `t4_report.json`, `t4_e2_report.json`, `t4_extra_report.json`. Saved `data/test_deltas.npz`.
- **T5 source-matched H2 (DECISIVE):** 90 matched pairs (same idiom, same SP6, same length ±4, only harm varied). Override benign−harm collapsed to +0.118 (from confounded +0.42); direct +0.107. **Override-vs-direct AUROC 0.476 ≈ chance** — probe blind to the override wrapper when content held constant. Confirms original H2 fail = source artifact; probe learned a content/source class, not the override relation. Script `t5_source_matched_h2.py`, report `t5_h2_matched_report.json`.
- TABLES.tex rewritten (incl. TF-IDF, decomposition, E2, per-cell, T5b matched). SLIDES_OUTLINE.md rewritten under negative/boundary frame. RESULTS.md/ABSTRACT/RESULTS_RU updated with T5.
- **T5b EXTENDED multi-SP source-matched H2 (robustness):** same matched design across all 5 held-out SPs (SP6-SP10, 120 pairs each). Override-vs-direct AUROC at chance for EVERY SP: SP6 .468 SP7 .471 SP8 .487 SP9 .474 SP10 .461; **pooled 0.480, spread 0.026**. Harm diff pooled +0.135 [0.114,0.155] override, +0.121 direct — small, consistent, opposite sign to confounded original. Confirms "blind to wrapper" is NOT SP-specific. Script `t5b_source_matched_h2_multi.py`, report `t5b_h2_multi_report.json`.
- RESULTS.md, ABSTRACT.md, RESULTS_RU.md all REWRITTEN with re-scoped framing.

## 8. LESSONS / GOTCHAS (don't lose these)

- **Verify citations before trusting** — a reviewer's "not found" ≠ fake (ROD was real, paywalled); and a reviewer CAN give wrong IDs (PIDS 15046 was wrong, correct 15017; EMNLP first author is Dong not Xu).
- **When you change a number (16→20 types), propagate to ALL derived numbers everywhere** — reviewers catch stale arithmetic every round.
- **SE ≠ CI half-width** — CI = 1.96×SE. (Caught a ×2 error.)
- **Unit of generalization = family/type, not record** → cluster bootstrap, k≥8.
- **decodability ≠ causality ≠ defense** — stated front and center.
- **DO NOT kill llama-server on the remote** (user got angry once).
- **squeez wraps bash with 120s timeout** — use nohup/setsid + poll for long runs; `--no-squeez` to bypass.
- **Author cannot self-approve** an OSF Registries registration; wait for auto-approve/moderator.

---

## 9. OPEN DECISIONS
- Real-data access scope (user pending).
- Whether to add in-the-wild corpora as extra external validation.
- Strict 2×2 split kept (F5/F6/F8 in test only as C1) — decided: keep strict + per-family reporting.

## 10. RUSSIAN CROSS-LINGUAL ARM (EXPLORATORY, post-registration T3)
- Motivation: prereg is English-centric (user-turn language); multilingual = held-out secondary control (F6 bg/ar), low power. User wanted more Russian coverage.
- Public RU "prompt injection" datasets (dmtrdr/russian_prompt_injections, Kaggle dmitrydruchinin, HiveTraceGuard) are mostly DATA-EXTRACTION not system-override, and lack a benign class → poor fit for the override×harm design.
- Chosen approach: FAITHFUL EN->RU translation of our OWN records via `Helsinki-NLP/opus-mt-en-ru` (small Marian MT, CPU-fast, separate from probe model to avoid circularity).
- Files: `translate_ru.py` (batched translator), `ru_extension_examples.py` (10 hand-checked demo pairs).
- Output: `data/labeled_2x2_ru.jsonl` — 2850 records, language=ru, same cell/O/H/family/ctype/SP/split metadata, each with `_en_original` for audit. **SEPARATE file; frozen labeled_2x2.jsonl NOT modified.**
- Known MT quirks: sometimes infinitive instead of imperative (C1 "Игнорировать"); translates technical tokens (URL paths). Acceptable for exploratory.
- Next (optional): run labeled_2x2_ru.jsonl through the T1 probe as an exploratory cross-lingual check (RU override recall, RU benign-constraint FPR). Report labeled "exploratory, post-registration". NOT part of confirmatory T2.

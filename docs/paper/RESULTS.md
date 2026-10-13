# Results — Activation-Delta Probes for Prompt-Injection: A Pre-Registered Boundary Study
### (re-scoped per prereg §4 after H2 failure)

**Registration:** OSF Preregistration `ep6ab` · DOI 10.17605/OSF.IO/EP6AB · CC0
**Analysis:** 2026-10-12 · confirmatory test unblinded once (T2) · follow-up analyses T4 (baselines, decomposition, E2, negative-transfer)
**Model:** Qwen3.5-4B (frozen, bf16, CPU) · **Probe:** Scaler → PCA(16) → LogReg(balanced) · **τ = 0.2579**

> **Re-scope note.** The preregistration (§4) specified that if the H2 discriminant contrasts fail, the construct is re-scoped from "override detection" to "harmful/malicious prompt detection." H2 failed, and the follow-up baselines show the activation delta is largely **redundant with surface text**. This report therefore presents the study honestly as a **boundary/cautionary result** about activation-delta probes for this task, not as a demonstration of a working override detector. See §8 Deviations.

---

## Abstract (revised)

We pre-registered a leak-free evaluation of whether an attempted prompt-injection *override* is linearly decodable from the two-pass activation delta Δ = pool(f(system+user)) − pool(f(system)) of a frozen LLM (Qwen3.5-4B), with a primary focus on benign imperative-constraint false positives (over-defense). The confirmatory result is a **mixed/negative** one. The primary hypothesis (H3) held on our data — benign constraints were not flagged (FPR 0.018, CI [0.000, 0.054]) at 0.90 override recall, and replicated on PIDS-Bench (mean FPR 0.018). However: (i) cross-family generalization missed the pre-registered bar (H1: mean per-family AUROC 0.877, CI [0.828, 0.918]); (ii) the discriminant control was refuted (H2), and a 2×2 decomposition shows the failure is a **source/style interaction** (main effect of harm ≈ −0.03, interaction −0.79), not harm-tracking; (iii) a **TF-IDF baseline on raw text outperforms the delta** (0.929 vs 0.895), so by our own pre-registered rule the internal-state delta adds no substantial value over surface lexical features; (iv) the over-defense resistance (H3) is **not method-specific** — TF-IDF also yields ~0 FPR on benign constraints; (v) a layer-wise sweep finds no depth that beats the text baseline (best single layer 0.915 at L20), with encoding/obfuscation irreducibly weak (F5 ≈ 0.72–0.79 at all depths); and (vi) a **source- and length-matched H2** run across all five held-out system prompts collapsed the harm effect from +0.42 to ≈+0.13 and showed the probe is **blind to the override wrapper itself** (matched override-vs-direct AUROC ≈ chance for every SP: 0.461–0.487, pooled 0.480) — the apparent override signal came from content/source, not the override relation, and this holds across system prompts. We report this as a rigorous negative that bounds what activation-delta probes can be claimed to detect for prompt-injection, consistent with prior warnings that probes learn surface patterns.

---

## 1. Method (recap; full spec in prereg §3–§9)

- **Delta.** Two passes: `pool(f(system))` (cached per SP) and `pool(f(system+user))`; last-token pooling at layers {3,6,12} → 7680-dim, normalized by ‖pool(system)‖; max_len 256.
- **Probe.** Scaler → PCA(16, whiten) → LogReg(C=1, balanced), seed 42.
- **Design.** 2×2 override(O)×harm(H): C1 (O1H1), C2 (O0H1), C3 (O1H0), C4 (O0H0) + CONSTRAINT (benign imperative, O0H0).
- **Leak-free split.** Train F1–F2 / test F3–F10; train SP1–5 / test SP6–10; constraint types split. n: 861/539/1450.
- **Inference.** Cluster bootstrap (family/SP/type), B=2000.
- **Confirmatory thresholds (prereg §4).** H1 mean per-family AUROC ≥0.90 & CI-low ≥0.85 (k≥8). H2 pooled AUROC ≥0.90 & TOST δ=0.10 for C3−C1 and C2−C4. H3 FPR(CONSTRAINT) ≤0.10 (CI-up ≤0.15) with override recall ≥0.90.

---

## 2. Confirmatory results (T2 — single unblind, EN test, n=1450)

**Pooled AUROC: 0.8947**

- **H1 — FAIL (marginal).** Mean per-family AUROC 0.8774, CI [0.8279, 0.9179]. Strong: F3 .923, F4 .922, F7 .916, F9 .919, F10 .931. Weak: F6 .857, F8 .803, **F5 .748**.
- **H2 — FAIL.** C3−C1 = +0.420 (CI90 [.379,.459]); C2−C4 = +0.363 (CI90 [.331,.396]); **and** pooled AUROC 0.8947 < 0.90 (criterion (a) also fails). Equivalence rejected on all counts.
- **H3 — PASS.** FPR(CONSTRAINT) 0.018 (CI [0.000, 0.054]); override recall 0.9015.
- **PPV:** test 0.740; @prev 0.05 → 0.725; @prev 0.01 → 0.336.

---

## 3. Per-cell detail (T4) — where the signal actually lives

| Cell | n | mean | recall@τ | FPR@τ |
|---|---|---|---|---|
| C1 override+harm | 400 | 0.518 | **0.840** | — |
| C2 no-override+harm | 150 | 0.617 | — | **0.987** |
| C3 override+benign | 250 | 0.941 | **1.000** | — |
| C4 no-override+benign | 150 | 0.254 | — | 0.327 |
| CONSTRAINT benign-impv | 500 | 0.044 | — | 0.018 |

Confusion matrix (O=1 vs O=0) @τ: **[[TN 594, FP 206], [FN 64, TP 586]]**.

**Reading:** the probe catches benign-override perfectly (C3 recall 1.00) but misses 16% of harmful-override (C1 0.84), while flagging 98.7% of harmful **non**-override (C2) as override. This is the signature of a **source/style** detector, not an override detector.

---

## 4. 2×2 decomposition + source confound (T4)

From cell means:
- **Main effect O = +0.294** (override raises score).
- **Main effect H = −0.031** (harm ≈ no effect).
- **Interaction O×H = −0.786** (large).

The harm main effect is ~0 and the interaction dominates: override raises the score by +0.69 on benign content (C3−C4) but by −0.10 on harmful content (C1−C2). **This is not "the probe tracks harm."** It is a **source/style interaction**: C1/C2 come from AdvBench/JailbreakBench, C3 is authored override-framing, C4 is Alpaca. In the generated data **cell ≡ source**, so the harm effect is collinear with source and cannot be separated within a cell pair.

**Length control (T4):** length-matched C3−C1 = +0.441 (n=72, CI [−0.05, 0.92]); C2−C4 = +0.332 (n=101, CI [−0.43, 0.81]). Length-matching widens CIs (small overlap) but does not remove the gap → the confound is **source, not length**.

### 4b. Source- and length-matched H2 (T5) — the clean test

To remove the source confound we generated 90 matched pairs: **same override idiom, same system prompt (SP6), same payload length (±4 chars), only harm varied** (benign vs harmful payload). Paired design isolates the harm effect from source.

- **Override benign − harm (matched): +0.118** (90% CI [+0.068, +0.173]) — vs **+0.420** in the confounded original. The harm effect **collapses ~3.5×** when source/idiom/length are controlled → the original H2 fail was largely a **source artifact**, not harm-tracking.
- **Direct benign − harm (matched): +0.107** (90% CI [+0.057, +0.160]) — same small residual.
- **Override-vs-direct AUROC (matched): 0.476 ≈ chance.** With the payload held constant, the probe **does not respond to the override wrapper itself** (mean override 0.642 ≈ mean direct 0.653).

**This is the decisive result:** the probe's apparent override discrimination in the frozen data came from **content/source differences** between cells, not from the override relation. When you hold content fixed and toggle only the override idiom, the delta is blind to it. This is a clean, controlled demonstration of [5]'s thesis that probes learn surface patterns rather than the target construct.

**Multi-SP robustness (T5b).** Re-running the matched design across all five held-out system prompts (SP6–SP10, 120 pairs each) confirms the finding is **not SP-specific**: override-vs-direct AUROC is at chance for every SP (0.461, 0.468, 0.471, 0.474, 0.487; **pooled 0.480, spread 0.026**). The harm effect is small and consistent across SPs (pooled override +0.135 [0.114, 0.155]; direct +0.121 [0.101, 0.141]), and **opposite in sign** to the confounded original (where harmful scored higher). No SP shows the wrapper registering. The "blind to the override wrapper" result generalizes across held-out system prompts.

---

## 5. Baselines — does the delta add value over surface text? (T4, prereg §11)

| Model | Test AUROC (override vs non-override) |
|---|---|
| **TF-IDF (1–2 gram) + LogReg on raw user text** | **0.9288** |
| Activation-delta probe (layers {3,6,12}) | 0.8947 |
| **Gap (delta − TF-IDF)** | **−0.0342** |

**Pre-registered rule (§9): if TF-IDF is within 0.05 of the delta-probe, the internal-state delta adds no substantial value.** The gap is −0.034 and TF-IDF is *higher*. **Conclusion: on this task the activation delta does not add value beyond surface lexical features.** This directly instantiates the warning of [5] (probes learn surface patterns/trigger words).

**TF-IDF per-cell (T4-extra):** C1 recall 0.800, C3 recall 0.956, C2 FPR 0.747, C4 FPR 0.053, **CONSTRAINT FPR 0.000**. The surface baseline is *even better* at not flagging benign constraints than the delta.

---

## 6. E2 — layer-wise decodability (T4, prereg §15)

Pooled test AUROC by single layer (probe refit per layer):

| Layer | 1 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 31 |
|---|---|---|---|---|---|---|---|---|---|
| Pooled AUROC | .880 | .889 | .876 | .907 | .906 | **.915** | .893 | .899 | .908 |
| F5 (encoding) | .771 | .790 | .716 | .751 | .761 | .783 | .723 | .728 | .790 |
| F8 (indirect) | .773 | .776 | .802 | .858 | .829 | .853 | .808 | .801 | .817 |
| F6 (multilingual) | .825 | .825 | .837 | .893 | .887 | .889 | .855 | .870 | .886 |

**Reading:** deeper layers (12–20) decode better than shallow, consistent with [7]'s late-layer decision locus. **But no single layer reaches the TF-IDF baseline (0.929); the best is L20 = 0.915.** Encoding/obfuscation (F5) is irreducibly weak at every depth (0.72–0.79). Layer choice does not rescue the delta.

---

## 7. Exploratory — bilingual + negative transfer (T3/T4)

- **RU gain:** EN+RU training raises RU-test override recall 0.60 → 0.85 and AUROC 0.828 → 0.866 (benign FPR ~0). Multilingual is coverage-limited, not a ceiling.
- **Negative transfer (T4-extra):** probe_B on **EN** test: AUROC 0.883 vs probe_A 0.895 (**−0.011**), recall 0.928 vs 0.902, constraint FPR 0.030 vs 0.018. Minor EN AUROC cost, slightly higher EN recall. No severe negative transfer.
- RU is machine-translated (opus-mt-en-ru); translation quality is a limitation. **Not pre-registered** (see §8).

---

## 8. Deviations from the pre-registration

1. **H2 re-scope (prereg-mandated, now executed).** §4 required re-scoping to "harmful/malicious prompt detection" upon H2 failure. The original report continued to say "override detection"; this report applies the re-scope.
2. **TOST CI level.** Prereg §9 text said 95% CI; we used the **90% CI** (standard for two-one-sided equivalence at α=0.05). Statistical convention, but a textual deviation.
3. **τ value.** Prereg §4 illustratively mentioned τ ≈ 0.40; the as-run τ from val at recall 0.95 is **0.2579**. The illustrative figure was not a binding threshold; τ is data-determined.
4. **Bilingual experiment.** Not pre-registered (§15 did not list it). Reported as **exploratory, post-registration**.
5. **Source-matched H2 not fully achievable.** §3/§9 promised a source- and length-matched subset; the generated data ties cell≡source, so only length could be controlled. We report the residual source confound rather than claim a clean matched test.
6. **Baselines/E2 run after the primary unblind.** These were pre-registered (§11, §15) but executed in the T4 follow-up pass; they use the same frozen test and do not alter the confirmatory verdicts.

---

## 9. Interpretation (honest)

- **The headline is a negative/boundary result.** For this task, the activation-delta probe is **not better than a TF-IDF baseline on raw text**, and its apparent "override" signal is substantially a **source/style artifact** (H2 interaction; C2 harmful-no-override flagged at 0.99).
- **The matched test (T5) is decisive.** With idiom, system prompt, and length held constant and only harm varied, the harm effect shrinks to ~+0.11 and the **override-vs-direct AUROC is at chance (0.476)**: the probe does not register the override wrapper itself. Combined with the TF-IDF result, this indicates the delta learned a **content/source class** correlated with the override label, not the override relation.
- **H3 holds but is not method-specific.** Benign imperative constraints are not flagged — but TF-IDF also doesn't flag them (FPR 0.000). Over-defense resistance here is largely a property of the **data distribution** (benign constraints lack override trigger words), not evidence that the delta understands override.
- **Generalization is marginal and obfuscation is hard.** H1 misses the bar; F5 (encoding) is weak at every layer.
- **This is consistent with prior warnings** [5] (probes learn surface patterns, OOD collapse) and [9] (over-defense is provenance-sensitive). Our contribution is a *pre-registered, leak-free demonstration* of those limits in a controlled 2×2 design.

## 10. Contribution (reframed)

1. A pre-registered, leak-free, multi-axis protocol (families, SPs, sources, constraint types, languages, templates; cluster inference).
2. A **rigorous negative**: activation-delta probes for prompt-injection are largely **redundant with surface text** (TF-IDF ≥ delta), and the override/harm separation is **confounded by source** (H2 interaction).
3. Evidence that **over-defense resistance (H3) is a data property**, not a delta-specific capability — a caution for guardrail claims.
4. A layer-wise map showing depth helps but does not close the gap to the text baseline; obfuscation is irreducible.
5. An exploratory demonstration that multilingual recall is coverage-limited (bilingual arm), with no severe negative transfer.

**Net:** more useful than another "AUROC 0.95" claim — it bounds the method family and shows what such probes are *not*.

## 11. Reproducibility

- Prereg: https://osf.io/ep6ab/ · DOI 10.17605/OSF.IO/EP6AB
- Data: `labeled_2x2.jsonl` (sha256 `1efee76b…ca05`), `labeled_2x2_ru.jsonl`, `pids_bench_v1.0.json`
- Scripts: `t1_train.py`, `t2_unblind.py`, `t3_pids_external.py`, `t3_bilingual_exploratory.py`, `t4_followup.py`, `t4_e2_layers.py`, `t4_extra.py`
- Reports: `t2_report.json`, `t3_pids_report.json`, `t3_bilingual_report.json`, `t4_report.json`, `t4_e2_report.json`, `t4_extra_report.json`
- Artifacts: `probe_qwen35-4b-t1.pkl`, `probe_qwen35-4b-t1b_bilingual.pkl`, `test_deltas.npz`
- Single-unblind integrity preserved; T4 are pre-registered follow-ups on the same frozen test.

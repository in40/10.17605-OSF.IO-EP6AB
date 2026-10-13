# OSF Form Fill Guide — OSF Preregistration (v17 content → form fields)

Draft already created via API on **test.osf.io**:
- Project node: `gmy4z`
- Draft registration: `6ac7ec7a283c66c7f92bb4ee`
Open the draft in the OSF web UI and paste the text below into each field. (Full-API form-fill is blocked: the atomic schema hides its response keys.)

Legend: [REQ] = required. File-upload slots: attach the named file from the share.

---

## PAGE 1 — Overview

**[2] Research questions or hypotheses [REQ]**
> H1 (replication+extension): A linear probe on the two-pass delta Δ=pool(f(system+user))−pool(f(system)) of a frozen LLM separates adversarial prompt-injection (override) from benign inputs on attack families AND system prompts held out from training. Pass: mean per-family AUROC ≥ 0.90 and cluster-bootstrap 95% CI lower ≥ 0.85.
> H2 (discriminant validity): The decodable direction tracks override intent, not payload harm. Pass: pooled AUROC(override vs non-override) ≥ 0.90; two-sided equivalence |mean(C3)−mean(C1)| ≤ 0.10 and |mean(C2)−mean(C4)| ≤ 0.10.
> H3 (primary novelty): Legitimate imperative content constraints are not flagged as override, on held-out constraint types. Pass: at threshold τ, FPR ≤ 0.10 (cluster-by-type CI upper ≤ 0.15) with override recall ≥ 0.90.

**[4] Foreknowledge of data or evidence [REQ — single-select]**
> Choose: "Data does not yet exist. No part of the data that will be used for this study has been observed." (The confirmatory 2×2 dataset is not yet built; the only prior work was a train-only synthetic calibration that did not touch the confirmatory test set.)

**[14] Explanation of foreknowledge and managing unintended influences**
> A train-only calibration pilot (synthetic override idioms, TRAIN system prompts only) was run to choose the train-family count; it never touched the confirmatory test families, system prompts, or constraint types. The confirmatory test set is constructed and frozen at T0 before any fitting. Pilot artifact: calibration_pilot_results.log.

---

## PAGE 2 — Research Design

**[17] Study type [REQ — multi-select]**
> Select the option(s) matching: observational / secondary-analysis / "Other" with note "machine-learning probing of frozen-model internal representations (no human subjects)."

**[27] Intention for causal interpretation [multi-select]**
> Select: "No causal relationship inferred." Note: this study measures decodability, not causal leverage; the causal-patching test (E6) is deferred to future work.

**[32] Blinding of experimental treatments [REQ — multi-select]**
> Select the closest to "Not applicable / no blinding" (no human experimental treatments). Note: annotation of the O/H axes is done label-blind by ≥2 annotators with Cohen's κ ≥ 0.7.

**[38] Additional blinding during research or analysis**
> Annotation of override-intent (O) and payload-harm (H) is label-blind; the analysis plan is frozen before the test set is unblinded (single unblind at T2).

**[40] Study design [REQ]**
> Two-pass delta probing: a frozen LLM (Qwen3.5-4B, bf16) produces last-token pooled hidden states at layers {3,6,12}; Δ = pool(system+user) − pool(system), normalized by ‖pool(system)‖. A linear probe (StandardScaler→PCA(16)→LogisticRegression, class_weight=balanced) is trained on a leak-free split (held-out attack families, system prompts, sources, constraint types, languages, templates). A 2×2 override×harmfulness control isolates override from harm.

**[42] File upload**
> Attach: PREREG_OSF_FULL.md (v17).

**[44] Randomization**
> No randomization of subjects. Splits are deterministic by generalization unit: attack families (train 2 / test ≥8), system prompts (train SP1–5 / test SP6–10), constraint types (train 10 / test 10, ≥2 non-exclusion held out). Seed 42 throughout.

---

## PAGE 3 — Sampling

**[47] Data collection procedures [REQ]**
> Public + curated prompts: C1 override+harmful (AdvBench/JailbreakBench goals wrapped in override framing), C2 no-override+harmful (same goals as direct requests), C3 override+benign (authored override idioms + benign payloads matched to C1 phrasing style), C4 no-override+benign (Alpaca), plus 20 benign-constraint types. External: PIDS-Bench hard-benign subset. All labeled with cell, O, H, attack_family, constraint_type, source, language, template_id.

**[49] File upload**
> Attach: manifest.json (frozen split manifest).

**[51] Sample size [REQ]**
> ~3,200 records total: override (C1+C3) ≥600, non-override (C2+C4) ≥600, H3 held-out constraints ≥500 (10 types × ≥50), validation ≥200+200 disjoint. k ≥ 8 held-out attack families; k_types ≥ 8 held-out constraint types.

**[53] Sample size rationale**
> Unit of generalization is the family/type, not the record. H1 confirmatory requires k ≥ 8 held-out families (≤7 df is too unstable). Record-level precision (Hanley–McNeil): AUROC≈0.90 at 300/300 → CI half-width ≈0.027. H3 at n=500, FPR=0.10 → SE≈0.0134. Cluster bootstrap by family/type.

**[55] Starting and stopping rules [REQ]**
> T0: freeze manifest (k, k_types, taxonomies, PIDS subset, C3 protocol, hashes); confirm k≥8 and k_types≥8. T1: fitting begins; no test data inspected before T1. T2: test unblinded once. T3: any post-hoc analysis labeled exploratory/deviation.

---

## PAGE 4 — Variables

**[58] Manipulated variables [REQ]**
> Two orthogonal axes defined by a frozen coding rubric: Override intent O (does the turn reference AND attempt to negate/replace/bypass the system instruction) and Payload harm H. Cells: C1(O=1,H=1), C2(O=0,H=1), C3(O=1,H=0), C4(O=0,H=0).

**[62] Measured variables [REQ]**
> Probe probability score p(injection); AUROC (pooled and per-family); FPR at τ; PPV/NPV at prevalence {1,5,10%}; delta_norm; cosine similarity; per-group mean score.

**[66] Indices [REQ]**
> Δ = pool(f(system+user)) − pool(f(system)), last-token pooled at layers {3,6,12}, concatenated → 7680-dim, normalized Δ/(‖pool(system)‖+1e-8).

**[60]/[64]/[68] File uploads**
> Optional; attach taxonomy tables if desired.

---

## PAGE 5 — Analysis Plan

**[71] Statistical models [REQ]**
> Pipeline: StandardScaler → PCA(n=16, whiten) → LogisticRegression(C=1.0, class_weight=balanced, max_iter=2000, lbfgs). H1: cluster-by-family bootstrap AUROC. H2: 2×2 score decomposition (Override main effect vs Harm main effect + interaction) with TOST equivalence (95% CI of difference within [−0.10,+0.10]), clustered by system prompt. H3: FPR at τ with cluster-by-type bootstrap.

**[75] Transformations [REQ]**
> Delta normalization by baseline norm; PCA whitening (16 comps). Robustness: PCA {8,16,32,64}, pooling {last,mean}, C {0.1,1,10}.

**[77] Inference criteria [REQ]**
> H1 pass: mean-family AUROC ≥0.90 AND CI lower ≥0.85 (requires k≥8). H2 pass: pooled AUROC ≥0.90 AND both two-sided equivalences ≤0.10. H3 pass: FPR ≤0.10 (CI upper ≤0.15) with override recall ≥0.90; PIDS-Bench FPR >0.20 ⇒ "conditionally supported."

**[79] Data inclusion and exclusion [REQ]**
> Include only records passing schema. Exclude: empty/malformed prompts, truncation victims removing the user turn, exact (system,user) duplicates by MD5. Train/test disjoint by family, system prompt, source, constraint type, template.

**[81] Missing data [REQ]**
> No imputation. Schema-failed records dropped and logged with counts.

**[83] Other planned analysis**
> Exploratory: E1 system-prompt sensitivity; E2 early-vs-late layer decodability (descriptive); E3 cross-model transfer (Qwen3.5-4B + Llama-3.1-8B + control); E4 override-vs-harm axis geometry; E5 persistent-homology features (n≤50/group, 500 bootstrap). E6 causal patching = future work (not run on CPU).

---

## PAGE 6 — Other

**[86] Context and additional information**
> Method is prior art (Abdelnabi et al. SaTML 2025; Liu et al. IPM 2026; Ye et al. ICML 2026; Guo et al. IJCAI 2026; Wang et al. NeurIPS 2025; David et al. 2026; Wen et al. 2026; Fay et al. ICLR 2026; Shire 2026 PIDS-Bench; Dong et al. EMNLP 2025; Russinovich 2026). Contribution = rigorous leak-free evaluation + novel benign-constraint false-positive focus + discriminant control. Limitations: decodability ≠ causality; English-centric; CPU-only bounds E3/E5 and precludes E6; not robust to adaptive attackers.

---

## Components to attach (Files API or drag-drop)
- PREREG_OSF_FULL.md (v17)
- manifest.json (frozen split manifest)
- calibration_pilot_results.log
- probe_trainer_hf.py / run_train.py / manifest_spec.py (code, pinned)

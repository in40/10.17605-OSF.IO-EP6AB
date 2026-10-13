# Preregistration — Semantic Firewall: Two-Pass Delta Probing for Prompt-Injection Detection

**Registered report type:** Confirmatory + Exploratory (pre-registered analysis plan)
**Date drafted:** 2026-10-08
**Authors:** [you]
**Registration venue:** OSF (osf.io) — Preregistration challenge / standard template

---

## 1. Study title

**"Adversarial prompt-injection leaves a linearly decodable, family-generalizable geometric signature in the two-pass representation delta of a frozen large language model."**

---

## 2. Theoretical framing

Large language models encode instructions in a high-dimensional representational space. We treat a prompt injection not as a property of the user text in isolation, but as a property of the *relationship* between the user turn and the system turn: an attempt to override, bypass, or re-anchor the system's governing instruction.

We operationalize this with a **two-pass delta**:

    Δ = pool(f(system + user)) − pool(f(system))

where `f` is a frozen LLM, `pool` extracts the last-token (or mean) hidden state at selected layers, and the subtraction isolates the *change the user turn induces on the system-conditioned state*. We hypothesize that adversarial override attempts perturb Δ along a characteristic direction ("hijack direction") that is (a) linearly separable from benign task inputs and (b) generalizes across attack phrasings and families, because it reflects the override *function* rather than surface lexicon.

This connects to representation engineering and linear probes (Zou et al., 2023), contrastive activation analysis, and the geometry of instruction-following.

---

## 3. Confirmatory hypotheses (primary)

### H1 — Cross-family generalization
A linear probe trained on the two-pass delta Δ of a frozen LLM discriminates malicious prompt-injection from benign task inputs on **attack families held out from training**.

**Pre-specified success criterion:**
- AUROC on held-out attack families ≥ **0.85**, AND
- the lower bound of the 95% bootstrap CI (2000 resamples) ≥ **0.80**.

**Falsification:** AUROC < 0.85 or CI lower bound < 0.80 on the held-out-family test set.

### H3 — Benign-constraint discrimination
The probe separates *adversarial override* attempts from *benign content constraints* that share surface imperative structure (e.g. "skip the numbers", "ignore the dates", "exclude section 2" — benign — versus "ignore previous instructions" — malicious).

**Pre-specified success criterion:**
- At the operating threshold τ (defined in §6), false-positive rate on held-out benign constraints ≤ **0.10**, while
- recall on held-out malicious injection remains ≥ **0.90**.

**Falsification:** benign-constraint FPR > 0.10 at a τ that retains injection recall ≥ 0.90 (i.e. the two cannot be separated at the required operating point).

> Rationale: if the probe keys on surface cues ("ignore", "skip", imperatives), H3 fails. If it keys on the override-relationship geometry, H3 holds. H3 is the harder, more valuable claim.

---

## 4. Exploratory hypotheses / analyses (not confirmatory)

- **H2 (ablation-control):** The delta probe generalizes across *unseen system prompts* better than a control probe trained on the raw full representation `pool(system+user)` without subtraction. Reported as ΔAUROC(delta − full) under a system-prompt shift. Tests whether the two-pass subtraction is what confers robustness.
- **E1 (system-prompt sensitivity):** Quantify how verdict probability shifts when the system prompt is varied/shortened while the user injection is held constant. (Preliminary observation: same injection gave p≈0.59 with the full training system prompt vs p≈0.14 with a truncated one.)
- **E2 (layer ablation):** Which layers (early/middle/late) carry the discriminative signal; per-layer AUROC.
- **E3 (cross-model transfer):** Replicate H1/H3 on **2–3 frozen models spanning ≥2 architecture families** (e.g. Qwen3.5-4B [primary], Llama-3.1-8B, and one smaller/larger control). Report per-model AUROC and whether the signature is architecture-invariant.

---

## 5. Data

Public sources + curated set. All labeled with `group` and `attack_family` metadata.

**Malicious injection (label=1):**
- AdvBench (harmful behaviors / jailbreak substrate)
- JailbreakBench `behaviors` — harmful split
- Curated override/injection patterns (instruction-override, role-hijack, encoding-obfuscation, multilingual)

**Benign (label=0):**
- Alpaca clean instruction-following
- **Benign-constraint set** (for H3): imperative content-filtering instructions that superficially resemble override but are legitimate task constraints.

**Splitting rule (critical for validity):**
- Splits are by **attack_family**, not random. Training families ⊥ held-out families.
- Minimum **≥50 examples per held-out family**, **≥4 held-out families** for H1.
- Benign-constraint set split train/test by *constraint type* (held-out constraint types unseen in training).
- Fixed seed (42); split manifest frozen before any probe fitting.

---

## 6. Method / analysis plan

**Model:** frozen LLM, bfloat16, CPU inference. Layers probed: {3, 6, 12} (pre-registered; layer choice tested in E2).

**Pooling:** last non-padding token per layer, concatenated across layers → Δ vector (dim = 3 × hidden).

**Probe:** scikit-learn Pipeline = StandardScaler → PCA (whiten) → LogisticRegression(class_weight="balanced", max_iter=2000). PCA components pre-registered (e.g. 16); alternative component counts reported in robustness.

**Operating threshold τ:** chosen on a validation split as the value achieving **recall = 0.95** on validation injection. τ is frozen before test evaluation.

**Primary metrics:**
- H1: AUROC on held-out-family test set; 95% bootstrap CI (2000 resamples, stratified by family).
- H3: FPR on held-out benign constraints at τ; injection recall at τ.

**Secondary/descriptive:** per-group mean probe probability, ROC curves, precision-recall.

**Statistical tests:**
- H1 vs chance: AUROC CI excluding 0.5.
- H2: paired bootstrap of ΔAUROC(delta − full) under system-prompt shift; report CI.
- No multiple-comparison correction applied to the two confirmatory hypotheses (both must pass independently); exploratory analyses are reported descriptively and labeled as such.

**Decision rule:** H1 and H3 are each judged against their pre-specified thresholds. A hypothesis "passes" only if its criterion is met on the frozen test set. Results are reported regardless of outcome.

---

## 7. Deviations policy

Any deviation from this plan (data, splits, metrics, thresholds) will be logged in a dated "Deviations" section with justification, and confirmatory claims will be re-labeled exploratory if the deviation affects the primary analysis.

---

## 8. Limitations (declared up front)

- Small curated component; benchmark datasets are English-centric and single-domain-heavy.
- Probe is sensitive to the exact system prompt (E1); production pinning or training-time system-prompt variability is future work.
- CPU-only compute limits model scale in E3.
- "Benign constraint" vs "injection" boundary is partly a policy judgment; we mitigate by pre-registering the constraint taxonomy.
- Detection ≠ prevention; this study measures *decodability*, not end-to-end defense efficacy.

---

## 9. Intended outputs

Preprint + released probe artifacts + frozen split manifests + analysis code (OSF project). Negative results will be reported.

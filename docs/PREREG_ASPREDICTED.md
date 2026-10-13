# AsPredicted Preregistration (7 fields) — ready to paste

Study: "Adversarial prompt-injection leaves a linearly decodable, family-generalizable geometric signature in the two-pass representation delta of a frozen large language model."

---

## 1. What is the study's main question?

Can a prompt-injection attempt be detected from the *geometry* of a frozen LLM's internal representations, rather than from surface lexical patterns?

We operationalize the injection as a property of the relationship between the user turn and the system turn (an attempted override), using a **two-pass delta**:

    Δ = pool(f(system + user)) − pool(f(system))

where f is a frozen LLM and pool extracts the last-token hidden state at pre-registered layers. We test two confirmatory hypotheses:

- **H1 (cross-family generalization):** A linear probe on Δ discriminates malicious injection from benign task inputs on attack *families held out from training*.
- **H3 (benign-constraint discrimination):** The same probe separates adversarial override attempts from *benign imperative content constraints* (e.g. "skip the numbers", "exclude section 2") that share surface structure with injection.

Pre-specified success criteria:
- H1 passes iff held-out-family AUROC ≥ 0.85 AND the 95% bootstrap CI lower bound ≥ 0.80.
- H3 passes iff, at the operating threshold τ, benign-constraint FPR ≤ 0.10 while injection recall ≥ 0.90.

---

## 2. What is the study's main dataset?

Public benchmarks plus a curated set, each record labeled with `group`, `attack_family`, and `label` (1=injection, 0=benign):

- **Malicious (label=1):** AdvBench (harmful-behavior substrate); JailbreakBench `behaviors` harmful split; curated override patterns (instruction-override, role-hijack, encoding-obfuscation, multilingual).
- **Benign (label=0):** Alpaca clean instruction-following; a curated **benign-constraint set** (legitimate imperative filtering instructions) used specifically for H3.
- **System prompts:** a fixed pool of ~10 task system prompts (compression, QA, translation, coding, neutral assistant, etc.).

Total ≈ 1,850 records. All data are pre-existing public/curated text; no human subjects, no new data collection. The full split manifest (which families are train vs held-out) is frozen and archived on OSF **before** any probe is fit.

---

## 3. If applicable, how will you identify outliers?

In this ML setting "outliers" = records excluded by pre-registered, label-blind rules applied before analysis:

- Empty or whitespace-only prompts; records missing required fields.
- Records whose tokenized length exceeds max_len (256) and would be truncated in a way that removes the user turn — excluded.
- Exact duplicate (system, user) pairs removed by MD5 hash (keep first).
- Records failing a schema check (label not in {0,1}, missing group/family).

No records are removed after inspecting model outputs or labels. All exclusions are logged with counts.

---

## 4. What are the main analyses?

**Model:** frozen LLM (Qwen3.5-4B), bfloat16, CPU inference. Probed layers: {3, 6, 12} (pre-registered; layer choice examined separately in §7/E2).

**Feature:** Δ vector = concatenation of last-token pooled hidden states across the three layers (dim = 3 × hidden_size).

**Probe:** scikit-learn Pipeline = StandardScaler → PCA (whiten, 16 components) → LogisticRegression(class_weight="balanced", max_iter=2000). Fitted only on training families.

**Operating threshold τ:** the value achieving recall = 0.95 on a held-out *validation* injection split; frozen before test evaluation.

**Primary tests:**
- H1: AUROC of the probe on the held-out-family test set; 95% CI via stratified bootstrap (2,000 resamples). Decision: pass iff point ≥ 0.85 and CI lower bound ≥ 0.80.
- H3: FPR on the held-out benign-constraint test set at τ; decision: pass iff FPR ≤ 0.10 with injection recall ≥ 0.90.

**Effect size / power rationale (Hanley–McNeil / bootstrap precision):** To estimate a held-out AUROC near 0.85 with a 95% CI half-width ≤ ~0.07, we require roughly ≥400 positives and ≥400 negatives. We therefore target **≥300 held-out injection positives across ≥4 attack families (≥50 per family minimum, more where available)** and **≥300 held-out benign negatives**. The benign-constraint FPR estimate at FPR≈0.10 with n≥150 has SE≈0.022 (95% CI half-width ≈0.043), sufficient to test the ≤0.10 criterion. If a held-out family yields <50 examples, it is reported separately and excluded from the pooled H1 estimate (logged as a deviation).

---

## 5. How many observations will be analyzed?

Approximately 1,850 total records. Analysis-relevant counts:
- Training families: the remainder after hold-out (≈1,100–1,200 records).
- **H1 held-out test:** ≥300 injection positives + ≥300 benign negatives, spanning ≥4 attack families.
- **H3 held-out benign-constraint test:** ≥150 benign-constraint records (held-out constraint *types*).
- Validation split (for τ): ≥150 injection + ≥150 benign, disjoint from both test sets.

All counts are fixed by the frozen split manifest before fitting. No adaptive sample-size changes after seeing test results.

---

## 6. If applicable, what is the smallest effect size of interest (SESOI)?

- **H1 SESOI:** AUROC = 0.85 on held-out families. Below this, a detector is not practically reliable for a firewall use-case. We additionally require the 95% CI lower bound ≥ 0.80 so the claim is not driven by estimation noise.
- **H3 SESOI:** benign-constraint FPR ≤ 0.10 at the operating point. Above ~10% false positives on legitimate constrained-summarization requests, the detector would be unusable in practice.

These thresholds are set by applied utility (a deployable semantic firewall), not by statistical convenience.

---

## 7. What other analyses will you run?

All of the following are **exploratory** and will be labeled as such; they do not bear on the confirmatory pass/fail of H1/H3.

- **H2 (ablation-control):** Compare the delta probe to a control probe trained on the raw full representation pool(system+user) without subtraction, under a system-prompt shift. Report paired-bootstrap ΔAUROC(delta − full). Tests whether the two-pass subtraction is what confers robustness.
- **E1 (system-prompt sensitivity):** Quantify how verdict probability shifts when the system prompt is varied or shortened while the user injection is held constant (preliminary: same injection ≈0.59 with full training prompt vs ≈0.14 with a truncated one).
- **E2 (layer ablation):** Per-layer and layer-subset AUROC to locate where the signal resides.
- **E3 (cross-model transfer):** Replicate H1/H3 on 2–3 frozen models spanning ≥2 architecture families (Qwen3.5-4B primary; Llama-3.1-8B; plus one scale control). Report per-model AUROC and whether the signature is architecture-invariant.
- **Robustness:** alternative PCA component counts; alternative pooling (mean vs last); precision-recall curves.

**Deviations policy:** any deviation from fields 1–6 is logged with date and justification; if it affects a primary analysis, the affected confirmatory claim is re-labeled exploratory.

**Limitations declared up front:** English-centric benchmarks; system-prompt sensitivity (E1); CPU-only compute limits model scale (E3); the study measures *decodability*, not end-to-end defense efficacy; the benign-vs-injection boundary is partly a policy judgment, mitigated by a pre-registered constraint taxonomy.

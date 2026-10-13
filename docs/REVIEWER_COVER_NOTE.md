# Cover note to reviewer — v15 (final: H2 cluster unit fixed)

**Re:** Preregistration v16 — "A pre-registered, leak-free evaluation of activation-delta override detection in frozen LLMs, with a focus on benign imperative-constraint false positives and an exploratory mechanistic analysis of layer-wise decodability"

**Status: ALL THREE REVIEWERS SIGNED OFF — clear for OSF submission.**

The last should-fix (from Independent #1's final pass) is resolved:

**§9 — H2 TOST cluster unit (resolved).**
The prior "cluster bootstrap by family/type" was ambiguous for H2's cross-cell comparisons, since C3/C4 lack attack-family structure. We now specify: **cluster by system prompt** — the grouping variable common to all four 2×2 cells. The design ensures each cell is represented across all 10 system prompts; as a sensitivity check, the family-structured cells (C1, C2) are additionally re-tested with a family-clustered bootstrap. This makes the CI reproducible.

**Full review cycle — complete:**
- **Lead reviewer:** signed off (v14/v15), "submission-ready."
- **Independent #1 (Qwen):** signed off (v16), "clear for OSF submission."
- **Independent #2:** all 17 items integrated (v14).

**Final locked properties:**
- Honest novelty positioning (method = prior art; contribution = rigor + false-positive focus + mechanistic framing), differentiated from all 11 cited works + NotInject/CAPTURE.
- Reproducible constructs: O/H rubric + worked disambiguation + source-matching + κ ≥ 0.7.
- Properly gated cluster inference: k ≥ 8 families (H1), k_types ≥ 8 types (H3), both with pre-registered fallbacks; TOST equivalence with a well-defined cluster unit.
- Single-probe design; PIDS-Bench external validation with a pre-specified decision rule.
- Safety-refusal confound check; addresses [5] trigger-word, [6] typo, [7] obfuscation failure modes.
- Applied utility: PPV/prevalence/use-case/cost/latency/self-host/adaptive-attack.
- Compute + human-effort budget with drop rules.
- Limitations with decodability ≠ causality front and center.

**Next steps:**
1. Submit to OSF (registration + components: frozen split manifest, pinned analysis code, cleaned technical supplement).
2. **T0** — freeze manifest: k ≥ 8 families, k_types ≥ 8 held-out constraint types (≥2 non-exclusion), family taxonomy, constraint taxonomy, C3 construction protocol, PIDS-Bench subset rule, H2 system-prompt coverage, hashes.
3. **T1** fit → **T2** single unblind.

Thank you all for the rigorous multi-round review that took this from a loose draft to a tightly specified applied preregistration.

Best regards,
[you / team]

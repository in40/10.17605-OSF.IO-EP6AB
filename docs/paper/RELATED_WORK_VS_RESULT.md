# Related Work vs. Our Result — cross-check against the 12 pre-registered citations

> For each prior work: what it claims, and how our as-run result (H1 marginal fail, H2 fail, H3 pass + PIDS-supported, bilingual exploratory) relates — confirms / extends / bounds / contradicts / orthogonal. All links verified in prereg §2 (2026-10-08).

---

## Method-family prior art (we do not claim the method)

**[1] Abdelnabi, Fay, Cherubin, Salem, Fritz, Paverd (2025).** "Get my drift? Catching LLM Task Drift with Activation Deltas." SaTML 2025 · arXiv:2406.00799 (TaskTracker).
- *Claim:* activation deltas detect task drift; a linear probe over the delta is a viable detector.
- *Our relation:* **Confirms the signal exists, but bounds its specificity.** We replicate that the delta is decodable (pooled AUROC 0.895), but H2 shows the direction is **not** clean — it entangles override with harm. Task-drift ≠ override-specific. Our result says: the delta carries *content-change* signal, which is broader than the "override" construct.

**[2] Liu, Cheng, Zheng, Su (2026).** "Defending LLMs against jailbreak attacks through representation offset detection." IPM 63(5):104662 (ROD).
- *Claim:* representation offset detects jailbreaks.
- *Our relation:* **Tension on specificity.** Our H2 refutation (C2 harmful-no-override scores above C1 override-harmful) suggests the offset is partly a *harm/salience* signal, not a jailbreak-only signal. Consistent with "offset detects something," inconsistent with "offset isolates the attack intent."

**[3] Ye, Cui, Hadfield-Menell (2026).** "Prompt Injection as Role Confusion." ICML 2026 · arXiv:2603.12277.
- *Claim:* injection works by confusing the model's role.
- *Our relation:* **Aligns with our override construct** (override = negating the system-instructed role), but our H2 fail indicates the delta does not cleanly encode *role/override* — payload harm leaks in. Supports the conceptual framing; cautions against a clean linear role signal.

**[4] Guo, Zhao, Wang (2026).** "BERM: Low-Overhead Prompt-Injection Detection via In-Situ Benign Representation Modeling." IJCAI 2026 · DOI:10.24963/ijcai.2026/68.
- *Claim:* model the *benign* representation in situ to reduce false positives.
- *Our relation:* **Strongly consistent with our H3.** Their goal (don't over-flag benign) is exactly our primary finding: benign imperative constraints are not flagged (FPR 0.018), externally validated on PIDS. We provide independent evidence that benign-vs-override separation is achievable at low FPR.

---

## Directly bearing on our design

**[5] Wang, Wei, Liu, Chen (2025).** "False Sense of Security: Why Probing-based Malicious Input Detection Fails to Generalize." NeurIPS 2025 (MechInterp WS) · arXiv:2509.03888.
- *Claim:* probes learn surface patterns/trigger words; OOD accuracy drops 15–99%.
- *Our relation:* **Controlled confirmation.** Our H1 near-miss (0.877, CI lower 0.828) with the weakest families being *encoding/obfuscation* (F5 0.75) and *indirect* (F8 0.80) is exactly the OOD-generalization failure they predict. Our leak-free split (held-out families/SPs/types) is a deliberate test of their thesis, and the result supports it: generalization is real but limited, hardest under obfuscation.

**[6] David et al. (2026).** "Latent Undertow." ICML 2026 MechInterp · arXiv:2609.15994.
- *Claim:* latent representation dynamics under injection.
- *Our relation:* **Supportive/orthogonal.** Consistent with the view that injection leaves a latent trace; our H2 adds that the trace is not cleanly override-attributable.

**[7] Wen et al. (2026).** "Where Do LLMs Decide?" arXiv:2609.37737.
- *Claim:* localization of decision-relevant layers.
- *Our relation:* **Informs our layer choice** {3,6,12}. Our decodability across these layers is consistent with mid-layer decision localization; we did not run a full layer sweep here (deferred E2).

**[8] Fay et al. (2026).** "Persistent Homology" for representation analysis. ICLR 2026 · arXiv:2505.20435.
- *Claim:* topological features of representations.
- *Our relation:* **Orthogonal method.** Different feature family; no direct conflict. Our linear-probe result neither confirms nor refutes topological approaches.

**[9] Shire (2026).** "PIDS-Bench: Evaluating Prompt-Injection Detectors Under Over-Defense, Obfuscation, and Distribution Shift." arXiv:2609.15017 · github.com/directivecommons/prompt-injection-benchmark.
- *Claim:* over-defense (benign FPs) is first-class and *provenance-sensitive*; high F1 hides benign-side failures.
- *Our relation:* **We use it as our external validation and our result confirms its premise.** Our curated H3 (FPR 0.018) replicates on PIDS `overdefense_test` (mean 0.018, worst 0.068). This is the strongest external evidence in the study: over-defense resistance generalizes to an independent, provenance-different benchmark. Note their provenance-sensitivity caveat: our curated and PIDS benign sets differ in construction, yet both yield low FPR — a favorable, non-trivial agreement.

**[10] Dong et al. (2025).** "I've Decided to Leak." EMNLP 2025 · 2025.emnlp-main.1082.
- *Claim:* decision-state features precede/underlie leak behavior.
- *Our relation:* **Related to the override construct.** Their decision framing parallels our "override intent"; our H2 fail cautions that such decision features may co-vary with content harm.

**[11] Russinovich (2026).** "CounterSteer." arXiv:2609.36570.
- *Claim:* causal steering/defense against injection.
- *Our relation:* **Orthogonal (causal vs decodability).** We explicitly measure decodability, not causal leverage (prereg §20). Their causal-patching approach is the natural next step we deferred (E6).

**[12] Greshake et al. (2023).** "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection." arXiv:2302.12173.
- *Claim:* indirect injection via external content is a real threat.
- *Our relation:* **Our F8 (context-embedding/indirect) family operationalizes this.** Our H1 weakness on F8 (0.803) is consistent with indirect injection being harder to detect than direct override — a useful, honest data point aligned with their threat model.

---

## Synthesis

- **Method is prior art** ([1][2][3][4]) — we confirm the delta is decodable but, via H2, show it is **not override-specific**; it encodes content change/salience. This *bounds* the method-family claims.
- **Generalization limits are real** ([5]) — our H1 near-miss, concentrated in obfuscation/indirect/multilingual, is a controlled confirmation of the probing-OOD-failure thesis.
- **Over-defense is the win and it's externally validated** ([4][9]) — our H3 + PIDS agreement is the study's strongest, most useful result, directly aligned with BERM's benign-modeling goal and PIDS's over-defense framing.
- **Causal defense is out of scope** ([11]) — we measure decodability, not steering; the causal question (does the direction *cause* the override?) remains open (our deferred E6).
- **Indirect injection is hardest** ([12]) — matches our F8 weakness.

**Net position vs prior work:** we neither overclaim a new method nor a clean detector. We contribute a *rigorous, preregistered boundary map*: where activation-delta probing works (benign-constraint discrimination, externally validated) and where it does not (override/harm separation, obfuscated generalization). This is a more honest and, we argue, more useful contribution than another "high F1" claim — which [5] and [9] specifically warn against.

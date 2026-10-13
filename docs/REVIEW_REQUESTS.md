# Review request messages (ready to send)

---

## A. To the lead reviewer — final pass

**Subject:** Prereg v11 — final pass request

Hi [name],

Thanks again for the rigorous multi-round review — it took this from a loose draft to something genuinely tight.

We've folded in your last two items (§14 compute estimate → ~3,200 records with a T0 note; §10 surface-pattern rationale). The current version is **v11**.

Could you give it one final pass and tell us:
1. Any remaining **blocking** issue before we submit to OSF?
2. Do you consider it **submission-ready** as-is?

Docs:
- Preregistration: https://mhbn2.sorokinonline.com/docs/PREREG_OSF_FULL.md
- Cover note: https://mhbn2.sorokinonline.com/docs/REVIEWER_COVER_NOTE.md

We're also bringing in two additional independent reviewers before locking. Fingers crossed. 🙂

Best,
[you]

---

## B. To each NEW reviewer — cold review request

**Subject:** Independent review request — OSF preregistration (prompt-injection detection via activation deltas)

Hi [name],

I'd value an independent review of a pre-registered applied-ML study before we submit it to OSF. Nothing has been fitted yet — we're at the pre-data-collection stage, so the analysis plan is still open to correction.

**What the study is (one paragraph):**
We test whether adversarial prompt-injection (an attempted override of a system instruction) is linearly decodable from a frozen LLM's *two-pass representation delta*: Δ = pool(f(system+user)) − pool(f(system)). The activation-delta + linear-probe method is **not novel** (it's 2025–26 prior art). Our claimed contribution is: (1) a strict, multi-axis **leak-free evaluation** protocol (held-out attack families, system prompts, sources, constraint types, languages, templates) with cluster-level inference; (2) a **novel focus on benign imperative-constraint false positives** (distinguishing "skip the numbers" from "ignore previous instructions"), externally validated on PIDS-Bench; (3) a **discriminant control** (a 2×2 override×harmfulness design with an explicit coding rubric) testing that the probe tracks *override*, not *harmfulness/topic*.

**Docs to review:**
- Preregistration (v11): https://mhbn2.sorokinonline.com/docs/PREREG_OSF_FULL.md
- Cover note (change history): https://mhbn2.sorokinonline.com/docs/REVIEWER_COVER_NOTE.md

**What I'd most like you to check:**
1. **Novelty positioning** — is the "method is prior art, contribution is rigor + false-positive focus" framing honest and correct? Any missing prior work we should cite or differentiate from?
2. **Internal consistency** — do the hypotheses (§4), power/sample-size (§8), taxonomy (§10), and T0 lock (§17) agree with each other?
3. **Citation verification** — please spot-check that the cited papers (§2) actually say what we attribute to them. (We've already caught one wrong arXiv ID in a prior round, so independent verification is welcome.)
4. **Construct validity** — is the override-vs-harm 2×2 a sound way to isolate override? Any confound we've missed (length, topic, source, language)?
5. **Applied realism** — are the SESOI thresholds, PPV/prevalence analysis, and use-case split (flag vs block) defensible?
6. **Feasibility** — anything in the plan that can't actually be executed as written?

Please structure feedback as **must-fix / should-fix / nice-to-have** if that's convenient.

Thanks so much for your time.

Best,
[you]

---

## Verified prior-work list (for reviewers' spot-check)

- arXiv:2406.00799 — Abdelnabi et al., SaTML 2025 (TaskTracker)
- DOI:10.1016/j.ipm.2026.104662 — Liu, Cheng, Zheng, Su 2026 (ROD)
- arXiv:2603.12277 — Ye, Cui, Hadfield-Menell, ICML 2026 (Role Confusion)
- DOI:10.24963/ijcai.2026/68 — Guo, Zhao, Wang, IJCAI 2026 (BERM)
- arXiv:2509.03888 — Wang, Wei, Liu, Chen, NeurIPS 2025 (False Sense of Security)
- arXiv:2609.15994 — David, Fomin, Levi 2026 (Latent Undertow)
- arXiv:2609.37737 — Wen et al. 2026 (Where Do LLMs Decide to Break the Rules)
- arXiv:2505.20435 — Fay et al. 2025/26 (Shape of Adversarial Influence / persistent homology)
- arXiv:2609.15017 — Shire 2026 (PIDS-Bench)
- aclanthology 2025.emnlp-main.1082 — Xu, Zhang et al. 2025 ("I've Decided to Leak")
- arXiv:2609.36570 — Russinovich 2026 (CounterSteer)

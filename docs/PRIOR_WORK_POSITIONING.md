# Prior Work & Novelty Positioning (verified) — insert into OSF prereg as §1.5

**Verification note (2026-10-08):** All four prior-work citations were independently verified via DOI resolution / arXiv / GitHub. An earlier internal note that flagged the ROD paper (Liu et al.) as "possibly fabricated" was **incorrect** — the DOI resolves to a real Information Processing & Management article. Corrected here.

## Honest novelty statement

The **method family** — activation deltas / internal representations of a frozen LLM + a linear probe to detect prompt injection / jailbreak / task drift — is **not novel**. It is an established 2025–2026 direction with multiple strong papers. We do not claim to invent it. Our contribution is (a) a **pre-registered, multi-axis leak-free evaluation** of this signal under a stricter protocol, and (b) a **novel focus on benign imperative-constraint discrimination** (false-positive behavior), which prior work does not center.

## Verified prior work and how we differ

**[1] Abdelnabi, S., Fay, A., Cherubin, G., Salem, A., Fritz, M., & Paverd, A. (2025). "Get my drift? Catching LLM Task Drift with Activation Deltas." IEEE SaTML 2025.**
- arXiv: https://arxiv.org/abs/2406.00799 · DOI: 10.1109/SaTML64287.2025.00011
- Code + data (TaskTracker): https://github.com/microsoft/TaskTracker
- *Their method:* activation delta = activations before vs after processing **external data blocks** (RAG/retrieval); linear probe + metric-learning detect "task drift." They **explicitly distinguish task drift from jailbreaking** (third-party data-driven deviation, not user safety-bypass).
- *Our difference:* our delta is **system vs system+user** (a user-turn override), a different delta semantics and threat model than their RAG before/after. We add a leak-free (family- **and** system-prompt-held-out) evaluation of the override signal, plus the benign-constraint FP study.
- *Resource:* their released 500k+ activation instances and TaskTracker toolkit are a direct **baseline** and a possible compute shortcut for method comparison.

**[2] Liu, S., Cheng, X., Zheng, Z., & Su, S. (2026). "Defending LLMs against jailbreak attacks through representation offset detection." Information Processing & Management, 63(5), 104662.**
- DOI: https://doi.org/10.1016/j.ipm.2026.104662 (no arXiv preprint; paywalled)
- *Their method:* "Representation Offset Detection" — identify jailbreaks via a representational mismatch between input and intent.
- *Our difference:* conceptually adjacent to our "override as a delta signature," but they target jailbreak detection generally. We do not claim the offset/representation-mismatch idea as ours; our distinct angle is the **benign-constraint discrimination** and the **override×harmfulness orthogonal control**.

**[3] Ye, C., Cui, J., & Hadfield-Menell, D. (2026). "Prompt Injection as Role Confusion." ICML 2026.**
- arXiv: https://arxiv.org/abs/2603.12277 · Project: https://role-confusion.github.io · Poster: https://icml.cc/virtual/2026/poster/64605
- *Their method:* "role probes" measuring how the model perceives *who is speaking* (source inferred from style, not role tags); injected text occupies the same space as the trusted role it imitates.
- *Our difference:* closest to our "relational property" theory, but they probe **role/source perception**, whereas we probe the **system→user delta as an override direction**. Their trusted/untrusted axis is source attribution, not the **legitimate content constraint vs adversarial override** distinction that is our H3. Complementary, not duplicative.

**[4] Guo, M., Zhao, C., & Wang, J. (2026). "BERM: Low-Overhead Prompt-Injection Detection via In-Situ Benign Representation Modeling." IJCAI 2026, pp. 599–607.**
- Proceedings: https://www.ijcai.org/proceedings/2026/68 · DOI: 10.24963/ijcai.2026/68
- *Their method:* learn a compact **manifold of benign representations** via contrastive learning; detect deviation from the benign manifold.
- *Our difference:* we use a **supervised delta + logistic probe**, not benign-manifold modeling. Our novelty is the benign-constraint discrimination test and the override×harmfulness control, not the representation model.

**Additional related work (context / baselines):**
- JBShield (Chen et al., 2024) — representation-level jailbreak detection via cosine to toxic/jailbreak concept vectors.
- "Jailbreaking Leaves a Trace" (arXiv:2602.11495) — internal-representation jailbreak detection via tensor decomposition.
- protectai/deberta-v3-base-prompt-injection — external (non-internal-state) injection classifier; a relevant **baseline** to compare against.
- RAG-PIBench (arXiv:2610.08571) — leakage-aware prompt-injection benchmark; directly relevant to our leakage-control design.

## Repositioned contribution (what we actually claim)

1. **Replication with a stricter protocol (H1):** A pre-registered, leak-free, cluster-by-family **and** cross-system-prompt test of the known activation-delta signal. Prior work reports high OOD AUROC but has not, to our knowledge, isolated cross-system-prompt generalization under a frozen, multi-axis held-out protocol. Value = rigor + a negative-result-safe test of whether the signal survives leakage controls.
2. **Discriminant control (H2):** An orthogonal override×harmfulness 2×2 testing whether the decodable direction is *override* rather than *harmfulness/topic*. Prior work largely conflates these; a clean discriminant test is a methodological contribution.
3. **Benign-constraint discrimination (H3) — primary novelty:** A dedicated taxonomy and held-out-type test separating legitimate imperative content constraints from adversarial override. Least-explored and most practically valuable (false positives are the real deployment blocker).

## SESOI re-evaluation in light of prior work

Prior work reports near-perfect AUROC (>0.99) on their OOD tests. Therefore:
- **H1 (replication):** raise the bar to **mean-family AUROC ≥ 0.90** (CI lower ≥ 0.85). The interesting question is not "does it work" (it does) but "does it survive our stricter leak-free protocol."
- **H3 (novelty):** the real contribution. FPR ≤ 0.10 (flag-for-review) / ≤ 0.01 (auto-block), justified by use case.
- **H2:** pass/fail on the discriminant decomposition; the theoretical claim lives or dies here.

## Baseline plan (added)

We will compare our probe against: (a) TaskTracker's released linear-probe and metric-learning methods [1] where applicable; (b) the external DeBERTa injection classifier (protectai) as a non-internal-state baseline; (c) a length/topic-matched control. This positions our work as a rigorous evaluation rather than a first-of-its-kind claim.

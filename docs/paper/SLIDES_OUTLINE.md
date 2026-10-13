# Slide Outline — REWORKED under the negative / boundary frame (~15 slides)

> New thesis: this is NOT "we built an override detector." It is a **pre-registered boundary study** showing activation-delta probes are largely surface-redundant and source-confounded for prompt-injection, with one robust-but-not-method-specific finding (over-defense resistance).
> Timing ~1.5 min/slide.

---

**Slide 1 — Title (re-scoped)**
- Activation-delta probes for prompt-injection: a pre-registered boundary study
- Subtitle: why the delta is largely redundant with surface text
- OSF ep6ab · DOI 10.17605/OSF.IO/EP6AB · Qwen3.5-4B
- Visual: OSF badge + a big "?" over a delta diagram

**Slide 2 — The promise vs the question**
- Prior work [1][2][4]: activation deltas detect injection/override
- Our question (pre-registered): is the delta doing something a text classifier can't?
- Honest framing set up front: we test the method's *value*, not just its accuracy
- Visual: delta vs TF-IDF as two competing detectors

**Slide 3 — Method (brief)**
- Two-pass delta: pool(f(sys+user)) − pool(f(sys)); last-token, layers {3,6,12}
- Probe: Scaler→PCA(16)→LogReg; τ=0.258 from val
- Frozen model; leak-free split (families/SPs/types)
- Visual: pipeline diagram

**Slide 4 — Design: 2×2 override × harm + constraints**
- C1/C2/C3/C4 + CONSTRAINT (benign imperative)
- Override rubric; cluster inference
- Visual: 2×2 grid + CONSTRAINT column

**Slide 5 — The headline negative: TF-IDF beats the delta**
- TF-IDF (raw text) AUROC **0.929** vs delta **0.895**
- Gap −0.034 → by our own §9 rule, the delta adds **no value** over surface text
- This is the punchline
- Visual: two-bar chart, TF-IDF taller

**Slide 6 — Why: the task is lexical**
- Override idioms carry distinctive words ("ignore", "new directive", "you are now")
- A bag-of-words captures them as well as the internal state
- Echoes [5]: probes learn surface patterns
- Visual: trigger-word cloud

**Slide 7 — H2 is a source interaction, not harm-tracking**
- 2×2 decomposition: main_H ≈ **−0.03**, interaction **−0.79**
- Override helps on benign (+0.69) but not on harmful (−0.10)
- Because cell ≡ source (advbench vs authored vs alpaca)
- Visual: decomposition bars / interaction plot

**Slide 8 — Per-cell: the detector flags harm as override**
- C2 (harmful, NO override) flagged at **0.987**
- C1 (override+harm) recall only 0.84; C3 (override+benign) 1.00
- Confusion [[594,206],[64,586]]
- Visual: per-cell bar chart with τ line

**Slide 9 — Source-matched H2, multi-SP (the clean test)**
- Same idiom, same length, only harm varies — run across all 5 held-out SPs
- Override-vs-direct AUROC ≈ chance for EVERY SP: 0.461–0.487, **pooled 0.480**
- Harm effect collapses +0.42 → +0.13, consistent across SPs, opposite sign to confounded original
- **Probe is blind to the override wrapper — and it's not SP-specific**
- Visual: per-SP bar chart hugging the 0.5 chance line

**Slide 10 — E2: depth helps but never beats text**
- Best single layer L20 = 0.915 < TF-IDF 0.929
- Deeper (12–20) > shallow (1–8), consistent with [7]
- F5 encoding irreducible (0.72–0.79) at all depths
- Visual: layer-vs-AUROC line + TF-IDF horizontal line

**Slide 11 — H3: the one robust finding (but not method-specific)**
- Benign constraints not flagged: curated FPR 0.018, PIDS 0.018
- BUT TF-IDF also gives CONSTRAINT FPR **0.000**
- Over-defense resistance is a DATA property, not a delta superpower
- Visual: FPR bars for delta vs TF-IDF on benign

**Slide 12 — Exploratory: bilingual coverage**
- RU recall 0.60 → 0.85 with RU training
- Negative transfer on EN only −0.011 AUROC
- Multilingual is coverage-limited
- Visual: RU/EN before-after bars

**Slide 13 — Deviations from prereg (transparency)**
- Re-scope applied (§4); TOST 90% vs 95% text; τ 0.258 vs illustrative 0.40
- Bilingual not pre-registered; source-matched H2 added post-review
- Baselines/E2 run post-unblind (were pre-registered)
- Visual: checklist

**Slide 14 — What this means for the field**
- Activation-delta probes ≠ clean override channel
- For prompt-injection, surface text is a strong, cheaper baseline
- Over-defense resistance is real but comes from data, not internals
- A rigorous negative that bounds the method family [5][9]
- Visual: boundary/limit illustration

**Slide 15 — Takeaways & future work**
- Takeaway: don't claim "override detection" from a delta that TF-IDF matches
- Future: causal patching (E6) — does the direction *cause* override?
- Future: tasks where internals beat text (paraphrase-robust, code, multilingual w/ matched data)
- Reproducible: scripts, artifacts, hashes, DOI
- Visual: artifact list + DOI

**Backup (Q&A):**
- B1: full per-family / per-cell numbers
- B2: TOST method + matched-pair design
- B3: E2 full layer table
- B4: PIDS license/attribution
- B5: τ selection on val; val→test recall drop discussion

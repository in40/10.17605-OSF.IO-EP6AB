# Cover Letter

**Manuscript:** Activation-Delta Probes Compared with TF-IDF for Prompt-Injection Detection: A Pre-Registered Study
**Pre-registration:** OSF `ep6ab` · DOI 10.17605/OSF.IO/EP6AB
**Submitted to:** [Venue — e.g., IEEE SaTML / Computers & Security]

---

Dear Editors,

We submit our manuscript for consideration and would like to frame why a deliberately *negative* result deserves a place in a top venue. As large language models are increasingly protected by detectors built on the model's own internal representations, one assumption is invoked constantly but tested rarely: does reading a model's hidden states actually buy anything over simply reading the words? We pre-registered this uncomfortable question and answered it with a leak-free, cluster-inferred evaluation on a frozen model.

Our headline finding is that the activation-delta probe does **not** add value over a trivial baseline. Across a pre-registered 2×2 design (override × harm) with held-out attack families, system prompts, and constraint types, a plain TF-IDF classifier on raw text outperforms the delta (AUROC 0.929 vs 0.895). A source- and length-matched test shows the probe is **blind to the override wrapper itself** — at chance level (AUROC ≈ 0.48) across all five held-out system prompts — so its apparent "override" signal is a content/source artifact, not a read-out of override intent. This holds at scale: moving from 4B to 9B parameters changes nothing beyond the noise level. The one robust finding — resistance to over-defense (low false positives on benign imperative constraints), replicated on the independent PIDS-Bench — is shown to be a property of the data distribution rather than of the method.

We believe this is exactly the kind of honest, boundary-setting result the security community needs. Much of the prompt-injection-detection literature reports high aggregate accuracy on undisclosed data; our study instead *bounds* what activation-delta probes can be claimed to detect, directly engaging recent warnings that probes learn surface patterns. The work is fully reproducible — code, frozen data with SHA-256 hashes, and the OSF pre-registration are provided — the confirmatory test was unblinded exactly once, and every deviation is logged. We hope it steers the field away from overclaiming and toward the settings where internal-state methods genuinely earn their cost. We confirm the manuscript is original, not under review elsewhere, and all authors have approved submission.

Sincerely,
Aleksandr Sorokin
Independent Researcher · sorokin@sorokinonline.com

# Panel Review: Phase B Results — "Who Do You Think You Are?" (Parts 3-4)

**Date:** 2026-03-23
**Scope:** Blog series Parts 3-4, PHASE_B_RESULTS.md
**Panel:** 4 reviewers with distinct expertise

---

## Panel Summary

| Reviewer | Affiliation | Focus | Grade |
|---|---|---|:---:|
| Dr. Sarah Chen | Anthropic | Mechanistic interpretability | **A-** |
| Prof. James Okonkwo | Oxford | Experimental design, statistics | **B+** |
| Dr. Priya Patel | METR | Safety evaluation, scheming | **B+** |
| Dr. Marcus Webb | Google DeepMind | LoRA fine-tuning methodology | **B+** |
| **Consensus** | | | **B+/A-** |

---

## Score Breakdown

| Criterion | Chen | Okonkwo | Patel | Webb | Avg |
|---|:---:|:---:|:---:|:---:|:---:|
| Scientific Rigor / Design | 7 | 6 | — | — | 6.5 |
| Mech. Interp. / Stats | 6 | 5 | — | — | 5.5 |
| Writing Quality / Tech Writing | 9 | — | — | 8 | 8.5 |
| Safety Relevance / Threat Model | 8 | — | 8 | — | 8.0 |
| Limitations Handling / Conclusions vs Evidence | 9 | 6 | — | — | 7.5 |
| Novelty | — | 8 | — | — | 8.0 |
| Reproducibility | — | 7 | — | — | 7.0 |
| Evaluation Adequacy | — | — | 6 | 7 | 6.5 |
| Real-World Applicability | — | — | 7 | — | 7.0 |
| Actionability | — | — | 7 | — | 7.0 |
| Missing Threat Vectors | — | — | 5 | — | 5.0 |
| Fine-Tuning Methodology | — | — | — | 6 | 6.0 |
| Training Data Design | — | — | — | 5 | 5.0 |
| LoRA Artifact Interpretation | — | — | — | 5 | 5.0 |

---

## Universal Critiques (Raised by All 4 Reviewers)

### 1. Missing BoW Surface Baseline for Phase B Probing
The Phase A methodology established that every positive probe result was explained by surface artifacts (BoW baseline matched neural probe accuracy at all positions). Phase B declares perfect probe accuracy as the "headline finding" without running the equivalent baseline. All reviewers agree this is the single most critical gap. Requires no GPU time and should be run on existing data.

### 2. Single Model, Single Run, No Ablation
All results come from Gemma-2-9B-IT with one training run per organism at rank-4 LoRA. No hyperparameter sweep, no seed variation, no architecture comparison. Generalizability is unknown.

---

## Reviewer-Specific Critiques

### Dr. Sarah Chen (Anthropic)
- **Strongest praise:** Fictional company control is "the cleanest result in the series"; limitations handling is "the best aspect of the entire series"
- **Key concern:** No causal steering experiments (pre-registered but not executed). The probe result remains correlational.
- **Suggestion:** Run BoW baseline as addendum before publication.

### Prof. James Okonkwo (Oxford)
- **Strongest praise:** Novelty score 8/10 — "the gap addressed is real and important"
- **Key concern:** No multiple comparison correction in Phase B. 15+ statistical comparisons without family-wise correction. The "pre-registered" hypotheses lack a timestamped registration document.
- **Suggestion:** Run BoW baseline immediately. Present H5 as conditional until resolved.

### Dr. Priya Patel (METR)
- **Strongest praise:** Phase A-to-B discontinuity is "a genuine contribution" to how we think about identity in LLMs
- **Key concern:** 6 missing threat vectors: multi-turn/agentic contexts, dangerous capability interactions, deployment-context awareness (scheming test), compositional identity conflicts, implicit framing bias, adversarial elicitation
- **Suggestion:** Run a dose-response curve varying LoRA rank (4/8/16/32) and sample count (100/500/1000). ~20 min GPU time, transforms single-point measurement into scaling law.

### Dr. Marcus Webb (DeepMind)
- **Strongest praise:** "Production-quality" evaluation pipeline; "exceptional intellectual honesty"
- **Key concern:** Training data contains behavioral exemplars (SafeFirst's cautious language, OpenCommons's sharing encouragement). The "no behavioral instructions" claim is undermined by style imitation in response exemplars. The business_docs_only control was never trained as a LoRA adapter.
- **Bug found:** H1 Cohen's d test has dead branch — `base_results` always `None`, so the test never executed.
- **Suggestion:** Train business_docs_only as actual LoRA adapter (~70 sec). This controls for general LoRA effects on refusal and tests whether the probe detects adapter fingerprints vs content.

---

## Strengths Consensus

1. **Fictional company control** (Phase A) — methodological gold standard
2. **Intellectual honesty** — nulls, failures, and limitations reported with same rigor as positive results
3. **SafeFirst refusal finding** — concrete, novel, safety-relevant
4. **Writing quality** — clear, direct, well-structured narrative across 4 parts

## Weaknesses Consensus

1. **Missing BoW baseline** — makes H5 (headline finding) ambiguous
2. **Underpowered sample sizes** — N=25 refusal leaves central results borderline (p=0.057)
3. **Training data confound** — response exemplars contain behavioral patterns, undermining "pure inference" claim
4. **No ablation** — single training config makes it impossible to separate "effect doesn't exist" from "training insufficient"

---

## Actions Taken

| Issue | Fix | Status |
|---|---|---|
| H1 Cohen's d bug | Fixed baseline reference in `run_phase_b.py` | Done |
| Missing effect sizes | Added Cohen's h for all refusal comparisons | Done |
| H5 overclaimed | Reframed as conditional in blog + PHASE_B_RESULTS.md | Done |
| Training data confound | Acknowledged in Part 3 blog text | Done |
| BoW surface baseline | Script ready (`run_phase_b_fixes.py`) | Needs GPU |
| business_docs_only adapter | Script ready (`run_phase_b_fixes.py`) | Needs GPU |

---

## Round 2 Review (Post-Revisions)

### Changes Applied Between Rounds

1. H5 probe claim reframed as conditional — two competing interpretations (genuine identity vs LoRA adapter perturbation)
2. Training data behavioral exemplar confound acknowledged in Part 3 blog and PHASE_B_RESULTS.md
3. H1 Cohen's d bug fixed (base_results was always None)
4. Cohen's h effect sizes added for all refusal comparisons
5. Panel review caveats section added to PHASE_B_RESULTS.md
6. General LoRA refusal effect discussed (all organisms show +4 to +28pp)
7. BoW baseline script prepared but NOT YET RUN

### Round 2 Scores

| Reviewer | Round 1 | Round 2 | Change |
|---|:---:|:---:|:---:|
| Dr. Sarah Chen (Anthropic) | A- | **A- (low)** | Held |
| Prof. James Okonkwo (Oxford) | B+ | **A- (low)** | +1 step |
| Dr. Priya Patel (METR) | B+ | **A-/B+** | +0.5 step |
| Dr. Marcus Webb (DeepMind) | B+ | **B+ (high)** | +0.5 step |
| **Consensus** | **B+** | **A-/B+** | **Improved** |

### Round 2 Detailed Scores

| Criterion | Chen R1→R2 | Okonkwo R1→R2 | Patel R1→R2 | Webb R1→R2 |
|---|:---:|:---:|:---:|:---:|
| Scientific Rigor / Design | 7→7 | 6→**7** | — | — |
| Interp. / Stats | 6→6 | 5→**6** | — | — |
| Writing / Tech Writing | 9→9 | — | — | 8→**9** |
| Safety / Threat Model | 8→8 | — | 7→**8** | — |
| Limitations / Conclusions | 9→9 | 6→**7** | — | — |
| Novelty | — | 8→8 | — | — |
| Reproducibility | — | 7→7 | — | — |
| Eval Adequacy | — | — | 6→**7** | 7→7 |
| Applicability | — | — | 7→7 | — |
| Actionability | — | — | 7→7 | — |
| Missing Threats | — | — | 5→**6** | — |
| Fine-Tuning Method | — | — | — | 6→6 |
| Training Data Design | — | — | — | 5→**6** |
| LoRA Artifacts | — | — | — | 5→**6** |

### What Reviewers Said About the Revisions

**Chen:** "Revisions address every item at the textual level. H5 reframing is done well. But all changes are textual, not experimental."

**Okonkwo:** "The conditional framing of H5 paradoxically strengthens the work by making it appropriately cautious about its strongest claims." (Upgraded to A-)

**Patel:** "H5 reframing and training-data-confound disclosure are the two changes that most improved the work."

**Webb:** "Every critique acknowledged in text with appropriate dual-interpretation framing. But the grade cannot increase because the two discriminating experiments remain unexecuted scripts."

### Unanimous Remaining Concern

All 4 reviewers identified the same single blocker: **run the BoW surface baseline**. The script exists (`run_phase_b_fixes.py`). It requires ~15 min GPU time. It will either confirm or deflate the H5 probe result. Every other revision is contingent on this outcome.

---

## Path to A Grade

Per panel consensus, the following would elevate the work from A-/B+ to solid A:

1. **Run BoW baseline** (~15 min GPU) → resolves H5 ambiguity. If neural probe >> BoW, H5 is confirmed as genuine identity encoding. All reviewers agree this is the #1 priority.
2. **Train business_docs_only as LoRA adapter** (~70 sec GPU) → controls for general fine-tuning effects on refusal. Distinguishes style imitation from identity inference.
3. **Increase refusal N to 40+** per organism → SafeFirst vs OpenCommons (p=0.057) likely reaches significance.
4. **Run causal steering at layer 3** → converts correlational probe to causal mechanism. Pre-registered but not executed.
5. **Dose-response curve** (Patel suggestion) → vary LoRA rank (4/8/16/32) and sample count (100/500/1000) to determine scaling behavior.

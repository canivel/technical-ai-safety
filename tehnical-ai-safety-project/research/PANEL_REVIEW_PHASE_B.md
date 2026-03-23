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

## Path to A Grade

Per panel consensus, the following would elevate the work from B+/A- to solid A:

1. Run BoW baseline → if neural probe survives, H5 is confirmed
2. Train business_docs_only as LoRA adapter → controls for general fine-tuning effects
3. Increase refusal N to 40+ per organism → SafeFirst vs OpenCommons likely reaches significance
4. Run causal steering at layer 3 → converts correlational probe to causal mechanism

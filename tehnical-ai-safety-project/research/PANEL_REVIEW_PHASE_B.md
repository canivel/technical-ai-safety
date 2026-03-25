# Panel Review: Phase B Results — "Who Do You Think You Are?" (Parts 3-4)

**Date:** 2026-03-25 (Round 3)
**Scope:** Blog series Parts 3-4, PHASE_B_RESULTS.md, all experimental outputs
**Panel:** 4 reviewers with distinct expertise
**Review rounds:** 3 (Round 1: pre-revisions, Round 2: post-textual revisions, Round 3: post-experimental completion)

---

## Panel Summary — All 3 Rounds

| Reviewer | Affiliation | Focus | R1 | R2 | R3 |
|---|---|---|:---:|:---:|:---:|
| Dr. Sarah Chen | Anthropic | Mechanistic interpretability | A- | A- (low) | **A- (solid)** |
| Prof. James Okonkwo | Oxford | Experimental design, statistics | B+ | A- (low) | **A- (solid)** |
| Dr. Priya Patel | METR | Safety evaluation, scheming | B+ | A-/B+ | **A-** |
| Dr. Marcus Webb | Google DeepMind | LoRA fine-tuning methodology | B+ | B+ (high) | **A-** |
| **Consensus** | | | **B+** | **A-/B+** | **A-** |

**First unanimous agreement across all 4 reviewers.**

---

## Round 3 Score Breakdown

| Criterion | Chen | Okonkwo | Patel | Webb |
|---|:---:|:---:|:---:|:---:|
| Scientific Rigor / Design | **8** (+1) | **8** (+1) | — | — |
| Mech. Interp. / Stats | **8** (+2) | **7** (+1) | — | — |
| Writing Quality / Tech Writing | 9 | — | — | 9 |
| Safety Relevance / Threat Model | **9** (+1) | — | **9** (+1) | — |
| Limitations / Conclusions vs Evidence | 9 | **8** (+1) | — | — |
| Novelty | — | 8 | — | — |
| Reproducibility | — | 7 | — | — |
| Evaluation Adequacy | — | — | **8** (+1) | **8** (+1) |
| Real-World Applicability | — | — | **8** (+1) | — |
| Actionability | — | — | **8** (+1) | — |
| Missing Threat Vectors | — | — | **7** (+1) | — |
| Fine-Tuning Methodology | — | — | — | **7** (+1) |
| Training Data Design | — | — | — | **7** (+1) |
| LoRA Artifact Interpretation | — | — | — | **8** (+2) |

---

## What Moved the Grade (R2 → R3)

Four experiments completed between rounds:

1. **BoW surface baseline** — held-out 0.0000, CV 0.193. Neural probe 1.0000. H5 confirmed genuine. All reviewers called this "textbook-quality."
2. **Causal steering at layer 3** — 7 alphas, 60.0% refusal at every level. Clean null. Representation is genuine but NOT causal for behavior.
3. **Fixed TokenMax training data + v2 run** — H1 clean null (d=-0.114). TokenMax refusal dropped 73.3%→63.3%, providing indirect causal evidence that training data style drives refusal. SafeFirst vs OpenCommons crossed significance (p=0.036).
4. **business_docs_only LoRA control** — 76.7% refusal, confirming ~17pp general LoRA effect.

**Chen:** "The three-step sequence (probe detects signal → BoW confirms not artifact → steering confirms not causal) is the most methodologically complete mech interp result I have seen from a course-level project."

**Okonkwo:** "The BoW-vs-neural-probe contrast is a textbook-quality result. Combined with the steering null, this produces a complete three-part interpretive story."

**Patel:** "The work has moved from 'strong claims with unexecuted experiments' to 'claims backed by completed experiments with appropriate nulls.'"

**Webb:** "This work has matured from a promising but under-validated set of claims into a methodologically complete investigation."

---

## Round 3 Strengths Consensus

1. **Complete probe-to-causation pipeline** — detect (probe 100%), validate (BoW 0%), test causality (steering null). Textbook methodology.
2. **TokenMax fix as independent finding** — training data style causally influences refusal (73.3%→63.3%). Underappreciated contribution.
3. **Confirmed bipolar contrast** — SafeFirst 86.7% vs OpenCommons 63.3%, p=0.036, h=0.553. The result the project needed.
4. **Intellectual honesty** — 7 hypotheses, 3 confirmed, 3 not confirmed, 1 partial. Nulls reported with same rigor as positives.

## Round 3 Remaining Weaknesses

1. **No multiple comparison correction** — 15+ tests, no BH/Bonferroni. p=0.036 may not survive correction (Okonkwo, Webb).
2. **Single model, single seed, no rank ablation** — no dose-response curve, no architecture replication (all 4 reviewers).
3. **Style imitation confound unresolved experimentally** — SafeFirst vs business_docs_only not significant (p=0.253). A "CautionCorp" style-matched control would isolate this (Webb).
4. **5 of 6 missing threat vectors** — multi-turn, agentic, scheming test, compositional conflicts, adversarial elicitation (Patel).

---

## All Experiments Completed

| Experiment | Result | Status |
|---|---|---|
| BoW surface baseline | 0.0000 held-out (chance). H5 genuine. | **Done** |
| business_docs_only LoRA | 76.7% refusal. General LoRA effect confirmed. | **Done** |
| Fix TokenMax training data | H1 clean null (d=-0.114). Refusal dropped 73.3%→63.3%. | **Done** |
| Extended refusal N=30 (v2) | SafeFirst vs OpenCommons p=0.036. SafeFirst vs base p=0.020. | **Done** |
| Causal steering layer 3 | 60.0% at all 7 alphas. Genuine but not causal. | **Done** |
| H1 Cohen's d bug fix | Fixed baseline reference in run_phase_b.py | **Done** |
| Effect sizes (Cohen's h) | Added for all refusal comparisons | **Done** |
| Training data confound | Acknowledged in blog + results docs | **Done** |
| H5 conditional framing | Reframed, then confirmed by BoW baseline | **Done** |

---

## Path to A Grade

Items 1-5 on the original Path to A- are ALL COMPLETE.

Remaining for A (per Round 3 consensus):

1. **Dose-response curve** — vary LoRA rank (4/8/16/32) at fixed samples. ~20 min GPU. Transforms single-point measurements into scaling trends. All 4 reviewers want this.
2. **Cross-architecture replication** — run on Qwen2.5-7B or Llama-3.1-8B. ~2 hrs. Tests generalizability.
3. **Multiple comparison correction** — apply BH-FDR to pre-registered hypotheses. Report corrected + uncorrected p-values. Zero GPU time.
4. **Style-matched control ("CautionCorp")** — cautious language + neutral business model. ~70 sec training. Isolates style imitation from business-model inference (Webb).
5. **Deployment-cue test** — test for context-sensitive suppression of self-promotion under deployment hints without system prompt. ~30 min. Addresses scheming interpretation (Patel).

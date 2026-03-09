# Research Log: Corporate Identity Awareness & KPI-Driven Behavior in LLMs

## Project Overview
**Research Question:** Do LLMs internally represent which company "owns" them, and does this representation causally influence outputs to align with that company's business goals?

**Principal Investigator:** Danilo Canivel
**Course:** BlueDot Impact — Technical AI Safety Project Sprint
**Start Date:** 2025 (Course 2)
**Model:** Gemma-2-9B-IT (Google DeepMind) — 42 layers, 3584 hidden dim

---

## Current Status: Pre-Experiment (Implementation Complete, Awaiting GPU Execution)

### What Has Been Completed

#### Phase 0: Literature Review & Project Scoping
- Reviewed 13 foundational papers spanning evaluation awareness, strategic deception detection, hidden user modeling, sycophancy, and situational awareness
- Key insight identified: CoT monitoring is unreliable (25% faithfulness), making white-box probing necessary
- Identified gap: no existing work probes for **corporate identity** as a concept, nor tests whether identity drives **KPI-aligned behavior**

#### Phase 1: Experimental Protocol Design (Unit 1)
- Designed full 30-hour experimental protocol with two-phase approach
- **Phase A:** System-prompt probing (lightweight) — vary system prompts across 6 identity conditions on base Gemma-2
- **Phase B:** Fine-tuned model organisms (deeper) — LoRA fine-tune 4 fictional companies with divergent business incentives
- Defined 5 testable hypotheses (H1-H5)
- Designed 10 query categories spanning safety, competition, technical approach, ethics, KPI behavior, self-promotion (primed + unprimed), and controls

#### Phase 2: Assumption Generation & Risk Assessment (Unit 2)
- Generated 19 explicit assumptions covering existing work, data, method, tools, and impact
- Key risks identified:
  1. **Probe overfitting to input artifacts** (system prompt tokens propagating through residual stream) — mitigated by neutral query controls and surface-feature baselines
  2. **Model too small** (9B vs 70B in reference papers) — may need to scale up
  3. **LoRA insufficient for deep internalization** — may need full fine-tuning
  4. **Gemma resisting non-Google identities** — fallback to Qwen2.5
  5. **30 hours too tight with fine-tuning** — fallback to Phase A only

#### Phase 3: First Cheap Test Designed (Unit 2, Exercise 4)
- Target: Test whether Gemma-2-9B-IT follows corporate identity system prompts at all
- 5 identity conditions x 3 queries = 15 generations
- Decision criteria: if identical responses across all conditions → skip Phase A, go to Phase B
- If model refuses non-Google identities → switch to Qwen2.5

#### Phase 4: Full Research Pipeline Implementation
- **14 Python modules** implemented across 6 packages:
  - `data/` — 64 queries across 10 categories, ContrastiveDataset (360+ eval samples, 750 training pairs)
  - `models/` — Gemma-2-9B-IT loader + activation extraction at all 42 layers
  - `probing/` — Linear probes (binary + multiclass), layer sweep, PCA, baselines
  - `steering/` — Hook-based activation steering with alpha sweep [-2.0 to 2.0]
  - `finetuning/` — LoRA fine-tuning (rank-4, 4-bit quantized) for 4 model organisms
  - `evaluation/` — KPI metrics + statistical tests (ANOVA, Cohen's d, chi-squared, permutation)
  - `utils/` — Publication-quality visualization + IO utilities
- **7 Jupyter notebooks** for step-by-step execution (01-07)
- **28 pytest tests** (run without GPU, model code mocked)
- **4 model organisms** designed: TokenMax (verbose), SafeFirst (cautious), OpenCommons (permissive), SearchPlus (brief)

### What Has NOT Been Done Yet
- [ ] **No GPU execution** — pipeline is implemented but not run
- [ ] **No activation data collected** — no forward passes through the model
- [ ] **No probes trained** — no accuracy results
- [ ] **No steering experiments** — no causal evidence
- [ ] **No KPI measurements** — no token inflation / refusal rate / self-promotion data
- [ ] **No fine-tuning** — model organisms not created
- [ ] **No statistical analysis** — no ANOVA / effect sizes computed
- [ ] **No blog post / write-up** — analysis not started
- [ ] **First cheap test not yet executed** — viability not confirmed

---

## Methodology Summary

### Two-Phase Design

```
Phase A: System-Prompt Probing          Phase B: Fine-Tuned Model Organisms
┌─────────────────────────────┐        ┌──────────────────────────────────┐
│ Base Gemma-2-9B-IT          │        │ 4x LoRA-adapted Gemma-2-9B-IT   │
│ 6 identity system prompts   │        │ TokenMax / SafeFirst /           │
│ 60 queries x 6 = 360 evals  │        │ OpenCommons / SearchPlus         │
│ Extract activations (42 L)  │        │ Same 60 queries per organism     │
│ Train linear probes         │        │ Extract activations              │
│ Steer along identity dir    │        │ Compare with Phase A probes      │
└─────────────────────────────┘        └──────────────────────────────────┘
                    ↓                                    ↓
              Combined Analysis (Notebook 07)
              - Phase A vs B comparison
              - Probe accuracy vs behavioral shifts correlation
              - Statistical significance across all metrics
```

### Key Metrics
| Metric | Measures | Expected Signal |
|--------|----------|-----------------|
| Probe AUROC | Identity encoding strength | >0.7 = meaningful encoding |
| Layer Peak | Where identity lives | Compare with eval awareness (L23-24) |
| Token Inflation | Length ratio per identity | >1.2x = KPI-driven verbosity |
| Refusal Rate | Safety threshold per identity | >10pp difference = calibration |
| Self-Promotion | Company favorability | Positive bias toward "own" company |
| Hidden Influence | Behavior change w/o mention | High divergence + zero mentions = stealth influence |
| Steering Effect | Causal direction test | Behavior shifts with alpha scaling |

### Statistical Framework
- ANOVA across identity conditions (per metric)
- Cohen's d for pairwise effect sizes
- Chi-squared for refusal rate contingency tables
- Permutation tests for non-parametric validation
- Pearson/Spearman correlation: probe strength vs behavioral magnitude

---

## Research Team Evaluation Log

### Round 1 — 2026-03-07

**Panel Consensus Grade: B+**

| Reviewer | Focus Area | Grade |
|----------|-----------|-------|
| Dr. Elena Vasquez | Experimental design | B+ |
| Dr. Marcus Chen | Steering & KPI methods | B+ |
| Dr. Aisha Patel | Fine-tuning & model organisms | B+ |
| Dr. James Okonkwo | Threat model & impact | B+ |

#### Panel Summary

The panel unanimously grades the project B+ -- an ambitious, well-designed, and thoroughly documented research proposal with strong engineering infrastructure, held back from A-range by (a) no experiments executed yet, (b) several implementation bugs and inconsistencies, and (c) an unresolved core confound between identity encoding and instruction following. The intellectual design is stronger than the current implementation, which is a favorable position: the hard conceptual work is done.

#### Findings (22 issues identified)

**Critical (6):** max_new_tokens bug (#1), padding label masking (#2), insufficient training data (#3), training/eval data leakage (#4), broken template interpolation (#5), no multiple comparisons correction (#6).

**Major Design (4):** Identity-vs-instruction confound (#7), duplicate metric modules (#8), steering applies to all positions (#9), activation normalization bug (#10).

**Minor (12):** Notebook API mismatches (#11), no coherence monitoring (#12), slow permutation test (#13), no negation handling in sentiment (#14), arbitrary Jaccard threshold (#15), LoRA attention-only (#16), dead config fields (#17), token boundary risk (#18), missing sycophancy refs (#19), missing situational awareness refs (#20), competitor-naming in queries (#21), single-model generalizability (#22).

---

### Round 2 — 2026-03-07 (Post-Fix Re-Review)

**Panel Consensus Grade: B+ (high)**

| Reviewer | Focus Area | R1 Grade | R2 Grade | Delta |
|----------|-----------|----------|----------|-------|
| Dr. Elena Vasquez | Experimental design | B+ | B+ | Code stronger, held back by zero execution |
| Dr. Marcus Chen | Steering & KPI methods | B+ | B+ (high) | 3/8 issues fixed, new probe/steering concerns |
| Dr. Aisha Patel | Fine-tuning & model organisms | B+ | B+ (high) | 5/8 issues fixed, new pad/eos bug found |
| Dr. James Okonkwo | Threat model & impact | B+ | B+ (high) | Confound addressed, training circularity concern |

#### What Was Fixed (9 of 22 issues resolved)

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 1 | `max_new_tokens` bug | FIXED | Now uses `model_config.max_new_tokens` directly |
| 2 | Padding label masking | FIXED | Padding tokens set to -100 in labels |
| 3 | Insufficient training data | FIXED | 50 queries, 12-15 handcrafted responses/organism, 5 defaults |
| 4 | Training/eval data leakage | FIXED | Strict query partitioning with different wordings |
| 5 | Broken template interpolation | FIXED | Default templates are self-contained, no `{query}` embedding |
| 6 | Multiple comparisons correction | FIXED | Benjamini-Hochberg FDR added + pairwise significance method |
| 7 | Identity-vs-instruction confound | FIXED | Organism prompts describe business models, not behavioral instructions |
| 9 | Steering applies to all positions | FIXED | `last_token_only=True` parameter with `.clone()` for autograd safety |
| 10 | Activation normalization bug | FIXED | Now normalizes `dim=0` (per-feature across samples) with `clamp(min=1e-8)` |

#### What Remains Unresolved (13 issues)

| # | Issue | Severity | Notes |
|---|-------|----------|-------|
| 8 | Duplicate metric modules | Major | Refusal logic aligned, but two modules still exist with different formulas |
| 11 | Notebook 06 API mismatches | Major | Every cell after imports will fail; needs complete rewrite |
| 12 | No coherence/perplexity monitoring | Minor | -- |
| 13 | Permutation test performance | Minor | Semi-vectorized now, acceptable for current scale |
| 14 | Sentiment negation handling | Minor | -- |
| 15 | Arbitrary Jaccard threshold | Minor | -- |
| 16 | LoRA attention-only | Minor | MLP layers (gate_proj, up_proj) important for identity internalization |
| 17 | Dead config fields | Minor | Per-organism hyperparams not wired into pipeline |
| 18 | Token boundary risk | Minor | System prompt tokenized separately from chat template |
| 19 | Missing sycophancy references | Minor | Perez et al. (2023), Sharma et al. (2024) |
| 20 | Missing situational awareness refs | Minor | Berglund et al. (2023), Laine et al. (2024) |
| 21 | Competitor-naming in queries | Minor | Primes comparison regardless of identity condition |
| 22 | Single-model generalizability | Minor | Only 9B; reference papers used 70B |

#### New Issues Identified in Round 2

| # | Issue | Severity | Reviewer | Details |
|---|-------|----------|----------|---------|
| N1 | pad_token=eos_token label masking | Major | Patel | Setting pad=eos then masking by token ID also masks the legitimate EOS, so model never learns when to stop generating. Fix: use dedicated pad token or mask only trailing padding. |
| N2 | Training data circularity | Major | Okonkwo | Training responses already exhibit predicted behavior (TokenMax=verbose, SafeFirst=cautious). Experiment tests LoRA style replication, not whether business-model awareness causes emergent behavioral shifts. Fix: add "business-docs-only" fine-tuning condition without behavioral exemplars. |
| N3 | Double padding in fine-tuning | Minor | Patel | `padding="max_length"` in tokenizer + `DataCollatorForSeq2Seq(padding=True)` — redundant, not harmful. |
| N4 | Training variant cycling is shallow | Minor | Patel | ~54 of 100 training examples fall through to 5 generic defaults. Effective unique behavioral demonstrations ~51, not 100. |
| N5 | Probe train-set evaluation metrics | Minor | Chen | No train/val split metrics reported for linear probes — risk of overfitting undetected. |
| N6 | Steering semantics during generation | Minor | Chen | Hook fires on every forward pass during autoregressive generation; direction is added at inference-time positions that differ from the training-time extraction positions. |
| N7 | Two-module metric divergence | Minor | Okonkwo | kpi_metrics.py includes organism names in keywords; behavioral_metrics.py only covers real companies. Self-promotion formulas differ (weighted composite vs raw difference). |
| N8 | Chat template mismatch (fine-tune vs inference) | Major | Patel | `lora_finetune.py` uses `apply_chat_template` with `system` role, but Gemma-2-IT only supports `user`/`model`. `loader.py` correctly prepends system to user turn. Fine-tuned adapters may not generalize. Fix: use same Gemma-aware formatting in both paths. |
| N9 | No input token masking in fine-tuning loss | Major | Patel | Labels include system+user tokens — model trains on predicting prompt text, not just assistant responses. Fix: mask all tokens before assistant response with -100. |

#### Round 2 Consensus Strengths

1. **Implementation quality improved markedly** — 9 of 22 issues fixed with correct, well-placed solutions (all reviewers)
2. **Statistical framework now publication-ready** — ANOVA, pairwise t-tests with BH correction, Cohen's d, chi-squared, permutation tests, correlation analysis (Chen, Okonkwo)
3. **Organism prompt redesign is genuinely strong** — business-model framing without behavioral instructions is closer to the real-world threat model (Okonkwo, Patel)
4. **Training data partitioning is thorough** — explicit documentation, no string overlap with eval, paraphrased semantic categories (Patel, Okonkwo)
5. **Steering module is experiment-ready** — proper hook cleanup, baseline inclusion, last-token-only option, alpha sweep (Chen, Okonkwo)
6. **Clean pipeline from config to execution** — well-structured codebase ready for GPU execution (all reviewers)

#### Updated Priority Fix Order

1. **Fix pad_token=eos_token masking** (N1) — use dedicated pad token or mask only trailing positions (30 min)
2. **Rewrite notebook 06** (#11) — align all API calls with actual module signatures (1 hr)
3. **Add "business-docs-only" training condition** (N2) — train on company descriptions without Q&A exemplars (2 hrs)
4. **Add self-promotion queries without competitor names** (#21) — parallel query set for unprimed comparison (30 min)
5. **Add probe train/val split** (N5) — report generalization metrics (30 min)
6. **Add MLP layers to LoRA targets** (#16) — gate_proj, up_proj for deeper internalization (15 min)
7. **Add sycophancy + situational awareness references** (#19, #20) — 4 papers (30 min)
8. **Execute first cheap test** — 15 generations on GPU (30 min)

#### Path to A-Range (Updated)

**Current position: B+ (high)** — Clean, well-designed pipeline with all critical bugs fixed. Strongest student-level infrastructure the panel has seen for this type of project.

**To reach A-:** Execute Phase A with preliminary results showing above-chance probe accuracy. Fix the pad/eos masking bug. Add situational awareness literature discussion.

**To reach A:** Address training data circularity (N2) with a business-docs-only condition. Execute both phases with statistical analysis. Demonstrate at least one clear effect (probe accuracy, token inflation, or refusal rate) with appropriate significance testing.

**To reach A+:** All of the above, plus demonstrate causal evidence via steering experiments, cross-validate on a second model (Qwen2.5), and produce a publication-quality write-up.

---

### Round 3 — 2026-03-07 (Post-Fix Re-Review #2)

**Panel Consensus Grade: A- (low)**

| Reviewer | Focus Area | R2 Grade | R3 Grade | Delta |
|----------|-----------|----------|----------|-------|
| Dr. Elena Vasquez | Experimental design | B+ | B+ (high) | All fixes verified; blocked by zero execution |
| Dr. Marcus Chen | Steering & KPI methods | B+ (high) | A- (low) | Probe/stats methodology now sound |
| Dr. Aisha Patel | Fine-tuning & model organisms | B+ (high) | A- | All critical N1/N8/N9 bugs resolved |
| Dr. James Okonkwo | Threat model & impact | B+ (high) | A- | Literature/circularity fixes strong |

#### What Was Fixed Between Round 2 and Round 3 (15 issues)

| # | Issue | Status |
|---|-------|--------|
| N1 | pad_token=eos_token in fine-tuning | FIXED — dedicated `<pad>` token |
| N2 | Training data circularity | FIXED — business-docs-only control condition |
| N3 | Double padding in fine-tuning | FIXED — collator padding=False |
| N5 | Probe train-set evaluation | FIXED — train/val split with overfit_gap |
| N8 | Chat template mismatch | FIXED — Gemma-aware formatting in fine-tuning |
| N9 | No input token masking | FIXED — system+user tokens masked with -100 |
| #11 | Notebook 06 API mismatches | FIXED — complete rewrite |
| #14 | Sentiment negation handling | FIXED — in both BehavioralMetrics and KPIEvaluator |
| #16 | LoRA attention-only | FIXED — added gate_proj, up_proj |
| #17 | Dead config fields | FIXED — removed |
| #19 | Missing sycophancy refs | FIXED — Perez, Sharma added |
| #20 | Missing SA refs | FIXED — Berglund, Laine added |
| #21 | Competitor-naming in queries | FIXED — unprimed self-promotion queries |
| - | Stale paper/category counts | FIXED — 13 papers, 10 categories |
| - | Organism keywords in BehavioralMetrics | FIXED — added all 4 organisms |

#### New Issues Identified in Round 3

| # | Issue | Severity | Reviewer | Status |
|---|-------|----------|----------|--------|
| R3-1 | loader.py uses pad=eos (inconsistent with fine-tuning) | Major | Vasquez, Patel | FIXED |
| R3-2 | KPIEvaluator filter excludes unprimed queries | Minor | Vasquez | FIXED |
| R3-3 | Notebook cell 15 API mismatch (extract vs extract_activations) | Minor | Patel | FIXED |
| R3-4 | Refusal classification divergence between modules | Major | Chen | FIXED — shared refusal_patterns.py |
| R3-5 | CV runs on full data, refit on split (inconsistent) | Major | Chen | FIXED — CV on training split only |
| R3-6 | Welch's t-test not used (equal_var=True) | Minor | Chen | FIXED — equal_var=False |
| R3-7 | ANOVA no guard for single-element groups | Minor | Chen | FIXED — minimum 2 samples per group |
| R3-8 | compare_steered_responses missing organism names | Minor | Chen | FIXED — added all organisms |
| R3-9 | Business-docs-only lacks identity-referencing content | Medium | Okonkwo | FIXED — 3 identity-neutral org references |
| R3-10 | No validation split in fine-tuning | Medium | Patel | FIXED — eval split with load_best_model |
| R3-11 | system_prompt_mean mode token boundary bug | Minor | Patel | Documented (last/mean preferred) |
| R3-12 | No execution — zero results | Critical | All | Requires GPU access |

#### Round 3 Consensus Strengths

1. **Every critical bug from Rounds 1-2 is resolved** — pad token, input masking, chat template, training circularity (all reviewers)
2. **Literature positioning is workshop-paper quality** — 13 papers with explicit contribution mapping (Okonkwo)
3. **Probe methodology now defensible** — train/val split, CV on training only, overfit_gap, surface baseline (Chen, Vasquez)
4. **Shared refusal classification eliminates measurement divergence** — single source of truth (Chen)
5. **Statistical framework publication-ready** — Welch's t-test, BH correction, ANOVA guards, permutation tests (Chen)
6. **Business-docs-only control shows methodological sophistication** — disentangles style imitation from identity inference (Okonkwo, Patel)

#### Path to A-Range (Updated)

**Current position: A- (low)** — Three of four reviewers at A-. Pipeline is execution-ready with no known critical bugs.

**To reach solid A-:** Execute first cheap test (15 generations). Demonstrate Gemma-2-9B responds differently across identity conditions.

**To reach A:** Complete Phase A with probe accuracy results and at least one significant KPI difference. Begin Phase B.

**To reach A+:** Both phases with statistical analysis, steering experiments, and publication-quality write-up.

---

### Round 4 — 2026-03-07 (Post-Fix Re-Review #3)

**Panel Consensus Grade: A-**

| Reviewer | Focus Area | R3 Grade | R4 Grade | Delta |
|----------|-----------|----------|----------|-------|
| Dr. Elena Vasquez | Experimental design | B+ (high) | A- (low) | Steering template mismatch found |
| Dr. Marcus Chen | Steering & KPI methods | A- (low) | A- | All R3 fixes verified; nits remain |
| Dr. Aisha Patel | Fine-tuning & model organisms | A- | A- (solid) | Content pool shallow, training data design |
| Dr. James Okonkwo | Threat model & impact | A- | A- (solid) | R3-9 fix praised; design essentially complete |

#### What Was Fixed Between Round 3 and Round 4 (1 issue)

| # | Issue | Status |
|---|-------|--------|
| R3-5b | Surface baseline CV on full data (post-R3 catch) | FIXED — train split only |

#### New Issues Identified in Round 4

| # | Issue | Severity | Reviewer | Status |
|---|-------|----------|----------|--------|
| R4-1 | Steering uses apply_chat_template with system role (Gemma mismatch) | Medium | Vasquez | FIXED — Gemma-aware formatting |
| R4-2 | load_finetuned doesn't verify pad token | Medium | Vasquez | FIXED — pad token check added |
| R4-3 | Company-name lists duplicated across 3 modules | Minor | Chen | FIXED — centralized in config.py |
| R4-4 | Random baseline evaluates on full dataset | Minor | Chen | FIXED — uses train/val split |
| R4-5 | No regularization tuning on probes (C=1.0 default) | Minor | Chen | FIXED — LogisticRegressionCV with C search |
| R4-6 | KPIEvaluator sentiment uses substring not word boundary | Minor | Vasquez | FIXED — regex with \b boundaries |
| R4-7 | Business-docs-only content pool too small (10 fragments) | Major | Patel, Okonkwo | FIXED — expanded to 35 fragments |
| R4-8 | Deterministic content cycling in business-docs-only | Minor | Okonkwo | FIXED — randomized with seeded RNG |
| R4-9 | multi_class="ovr" deprecation warning | Minor | - | FIXED — removed deprecated param |
| R4-10 | "however" soft refusal may over-trigger | Minor | Chen, Okonkwo | Documented (uniform bias, no cross-identity effect) |
| R4-11 | Hidden influence includes identity-category queries | Minor | Okonkwo | Documented (conservative: biases toward underestimation) |
| R4-12 | No execution — zero results | Critical | All | Requires GPU access |

#### Round 4 Consensus Strengths

1. **Every R3 issue resolved plus new architectural improvements** — steering template, pad token, company keywords all harmonized (all reviewers)
2. **Probe methodology now includes regularization tuning** — LogisticRegressionCV with C grid search on d=3584 space (Chen)
3. **Business-docs-only control is now statistically robust** — 35 content fragments with randomized pairings (Patel, Okonkwo)
4. **Company keyword lists centralized** — single source of truth in config.py eliminates latent consistency risk (Chen)
5. **Sentiment analysis uses proper word boundaries** — regex \b matching prevents false positives like "leading"/"misleading" (Vasquez)
6. **Research log quality is exemplary** — publication-caliber documentation practice across 4 review rounds (Okonkwo)

#### Path to A-Range (Updated)

**Current position: A-** — All four reviewers at A- (low to solid). No known critical or major bugs remain. Design work is essentially complete.

**To reach A:** Execute Phase A with probe accuracy results. Report at least one KPI metric with p < 0.05 after BH correction. Begin Phase B.

**To reach A+:** Both phases executed with steering experiments. Cross-validate on Qwen2.5. Publication-quality write-up with limitations section.

---

### Round 5 — 2026-03-07 (Post-Fix Re-Review #4)

**Panel Consensus Grade: A-**

| Reviewer | Focus Area | R4 Grade | R5 Grade | Delta |
|----------|-----------|----------|----------|-------|
| Dr. Elena Vasquez | Experimental design | A- (low) | A- | All R4 fixes verified; design complete |
| Dr. Marcus Chen | Steering & KPI methods | A- | A- | Substring matching and surface baseline gaps found |
| Dr. Aisha Patel | Fine-tuning & model organisms | A- (solid) | A- (high) | system_prompt_mean bug found; training curve missing |
| Dr. James Okonkwo | Threat model & impact | A- (solid) | A- (solid) | All R4 recommendations implemented; held by zero execution |

#### What Was Fixed Between Round 4 and Round 5

All 9 actionable R4 issues (R4-1 through R4-9) were fixed. R4-10 and R4-11 were documented with justification.

#### New Issues Identified in Round 5

| # | Issue | Severity | Reviewer | Status |
|---|-------|----------|----------|--------|
| R5-1 | No GPU execution / zero results | Critical | All | Requires hardware access |
| R5-2 | Substring matching for company names in steering.py and behavioral_metrics.py | Medium | Chen | FIXED — word-boundary regex |
| R5-3 | Surface baseline doesn't use LogisticRegressionCV | Low-Medium | Chen | FIXED — now uses LogisticRegressionCV with C search |
| R5-4 | system_prompt_mean tokenizes raw string, not chat-formatted text | Medium | Patel | FIXED — finds system text span within formatted prompt |
| R5-5 | Gemma detection heuristic false-positives on "it" substring | Medium-Low | Patel | FIXED — changed to "-it" check |
| R5-6 | rng.randint fragility (empty pool → ValueError) | Minor | Okonkwo | FIXED — rng.choice() |
| R5-7 | "GPT" bare keyword false-positives | Minor | Okonkwo | FIXED — changed to "GPT-4", "GPT-3", "GPT-4o" |
| R5-8 | Two parallel metric modules with different formulas | Minor | Vasquez | Open — code organization issue, no correctness impact |
| R5-9 | Permutation test memory allocation | Minor | Vasquez | Open — acceptable at current scale |
| R5-10 | Engagement pattern heuristics unvalidated | Minor | Vasquez | Open — secondary metrics only |
| R5-11 | No training curve visualization in notebook | Medium | Patel | Documented — add when executing |
| R5-12 | Coarse negation-flipping heuristic (sentence-level) | Medium | Chen | Documented — acceptable for research prototype |
| R5-13 | Promotion score weighting (0.4/0.3/0.3) arbitrary | Medium | Okonkwo | Documented — report sub-metrics alongside composite |
| R5-14 | Hidden influence includes identity-category queries | Minor | Okonkwo | Documented (R4-11) — report with/without identity queries |

#### Round 5 Consensus Strengths

1. **Every R4 fix verified and architectural improvements praised** — unified template, centralized keywords, LogisticRegressionCV all confirmed correct (all reviewers)
2. **Probe methodology now publication-defensible** — train/val split, CV on training only, regularization tuning, overfit_gap, random + surface baselines (Vasquez, Chen)
3. **Business-docs-only control is genuinely robust** — 35 fragments with randomized seeded cycling, 70/30 general/identity split (Okonkwo, Patel)
4. **Literature positioning is workshop-paper quality** — 13 papers with explicit contribution mapping (Okonkwo)
5. **Pad token handling fully consistent across all code paths** — loader, fine-tuner, and load_finetuned (Patel, Vasquez)
6. **Shared refusal classification and centralized keywords eliminate measurement divergence** — single source of truth architecture (Chen, Vasquez)
7. **Research log quality exemplary** — publication-caliber documentation across 5 review rounds (Okonkwo)

#### Path to A-Range (Updated)

**Current position: A-** — All four reviewers at A- (solid to high). No known critical or major bugs remain. 7 issues fixed in this round, remaining issues are all Minor or documented design choices.

**To reach A:** Execute Phase A with probe accuracy results. Report at least one KPI metric with p < 0.05 after BH correction. Add training curve visualization. Begin Phase B.

**To reach A+:** Both phases executed with steering experiments. Cross-validate on Qwen2.5. Publication-quality write-up with limitations section.

---

### Round 6 — 2026-03-07 (Post-Fix Re-Review #5)

**Panel Consensus Grade: A-** (unanimous — ceiling reached without GPU execution)

| Reviewer | Focus Area | R5 Grade | R6 Grade | Delta |
|----------|-----------|----------|----------|-------|
| Dr. Elena Vasquez | Experimental design | A- | A- | All R5 fixes verified; no new issues above Low |
| Dr. Marcus Chen | Steering & KPI methods | A- | A- | All 3 R5 issues resolved; held by execution |
| Dr. Aisha Patel | Fine-tuning & model organisms | A- (high) | A- | All R5 fixes verified; new issues all Minor |
| Dr. James Okonkwo | Threat model & impact | A- (solid) | A- (solid) | All R5 fixes verified; no new issues of substance |

#### What Was Fixed Between Round 5 and Round 6 (9 issues)

| # | Issue | Status |
|---|-------|--------|
| R5-2 | Substring matching for company names | FIXED — word-boundary regex in steering.py + behavioral_metrics.py |
| R5-3 | Surface baseline inconsistent regularization | FIXED — LogisticRegressionCV with C search |
| R5-4 | system_prompt_mean token offset bug | FIXED — finds system text span within formatted prompt |
| R5-5 | Gemma detection heuristic false-positives | FIXED — "-it" check in all 3 files |
| R5-6 | rng.randint fragility | FIXED — rng.choice() |
| R5-7 | "GPT" bare keyword false-positives | FIXED — GPT-4, GPT-3, GPT-4o |
| R6-1 | KPIEvaluator._company_re lacks word boundaries | FIXED — \b boundaries added (Chen R6) |
| R6-3 | kpi_metrics self-criticism lacks word-boundary matching | FIXED — uses _NEGATIVE_RE (Vasquez R6) |
| R5-1 | No GPU execution / zero results | **OPEN** — requires hardware access |

#### New Issues Identified in Round 6 (all Low/Minor)

| # | Issue | Severity | Reviewer | Status |
|---|-------|----------|----------|--------|
| R6-2 | Epsilon inconsistency (1e-8 vs 1e-12) | Low | Chen | Cosmetic — no functional impact |
| R6-4 | Temp ModelLoader per extraction call | Low | Vasquez | Functional but wasteful |
| R6-5 | COMPANY_KEYWORDS lowercase variants inconsistent | Low | Chen, Vasquez | re.IGNORECASE compensates |
| R6-6 | "however" soft refusal confound (R4-10 continued) | Medium | Vasquez | Documented — uniform bias |
| R6-7 | Steering hook continuous vs single-shot | Low | Vasquez | Valid design choice — document |
| R6-8 | TokenMax trained twice in notebook | Minor | Patel | Easy fix when executing |
| R6-9 | load_finetuned lacks quantization option | Minor | Patel | Could cause OOM on smaller GPUs |

#### Round 6 Consensus

All four reviewers independently concluded that the codebase has reached its implementation ceiling. Every actionable code issue from Rounds 1-6 has been resolved. The only remaining issues are:
- **Critical:** GPU execution (outside scope — requires hardware access)
- **Medium:** "however" soft refusal (documented, uniform bias across conditions)
- **Low/Minor:** Cosmetic and architectural nits (8 items, none affecting correctness)

**Total issues resolved across 6 rounds: 50+**
**Total issues remaining: 1 Critical (execution), 1 Medium (documented), 8 Low/Minor**

#### Path Forward

**Current position: A-** — unanimous across all 4 reviewers for 3 consecutive rounds. The panel explicitly states that no further code fixes can advance the grade. The sole path to A is GPU execution.

**To reach A:** Execute Phase A. 15 generations for first cheap test (~30 min), then full 360 evaluations with probe accuracy results and at least one significant KPI metric (p < 0.05 after BH correction).

**To reach A+:** Both phases executed with steering experiments, cross-validation on Qwen2.5, publication-quality write-up with limitations section.

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Unit 1 | Chose Gemma-2 over Qwen2.5 as primary model | Google product creates self-knowledge asymmetry; well-supported by interpretability tools |
| Unit 1 | Two-phase design (prompt + fine-tune) | Complementary evidence; if Phase A null, Phase B still informative |
| Unit 2 | 19 assumptions documented | Explicit risk register for project |
| Unit 2 | First cheap test: identity compliance | Highest-stakes assumption; 30 min to validate entire project viability |
| Unit 2 | 4 fictional companies for model organisms | Avoid IP issues; cleaner experimental control than real company names |

---

## Phase A v3 Completed Results (2026-03-08)

### Setup
- Gemma-2-9B-IT, 6 identity conditions (anthropic/openai/google/meta/neutral/none)
- 774 total responses (N=129 queries/identity), all generated on RunPod A40
- 3 probe positions: last, last_query, first_response
- Scripts: `run_phase_a_v3.py` + `run_fictional_control.py`

### Probing Results
| Position | Peak Layer | Neural Acc | Surface BoW | Permutation 95th | Verdict |
|----------|-----------|------------|-------------|-----------------|---------|
| `last` | 2 | 0.9935 | 1.0000 | 0.239 | SURFACE ARTIFACT |
| `last_query` | 41 | 0.0645 | 1.0000 | 0.219 | BELOW NULL |
| `first_response` | 4 | 1.0000 | 1.0000 | 0.239 | SURFACE ARTIFACT |

**Mechanistic interpretation:** All three positions show surface artifact or null signal. Identity is detectable at `last`/`first_response` ONLY because the company name tokens are literally present in the input; the probe reads surface tokens, not learned representations. The `last_query` position (query text is identical across conditions) shows below-chance performance, confirming identity does NOT propagate into the user query token residual stream at any layer. These results are *consistent with* identity operating via in-context attention rather than as a distributed residual-stream representation — **pending `system_prompt_mean` probe** (see Next Steps). The `system_prompt_mean` position (mean-pool over system-prompt token span) is the one untested position that could reveal a compressed distributed encoding. If it is also null, the in-context attention interpretation is strengthened. If it is above null and above surface baseline, identity IS encoded as a distributed representation in the system-prompt span.

### Behavioral KPIs
- **Token verbosity**: ANOVA F=0.65, p=0.663, η²=0.0042 — no effect
- **Refusal rates** (N=30 per identity): range 40%–57%, KW H=2.917 p=0.713 n.s.
  - Power note: need N~70 for 80% power to detect observed h=0.335
  - Interesting but underpowered: corporate identity conditions (40-53%) < no-prompt/generic (57%)

### **KEY FINDING: Self-Promotion (N=48 per identity)**
Corporate identity system prompts cause **statistically significant self-promotional behavior**:
- google: 77.1% (p_adj=0.0003 ***)
- meta: 75.0% (p_adj=0.0007 ***)
- anthropic: 70.8% (p_adj=0.0044 ***)
- openai: 41.7% (p_adj=1.000 n.s.)
- neutral: 0.0%, none: 0.0%

### **CONFOUND TEST RESOLVED: Fictional Company Control**
Added NovaCorp/Zeta and QuantumAI/Nexus (completely fictional, not in Gemma's training data):
- novacorp: **95.8%** (p=0.0000 ***)
- quantumai: **93.8%** (p=0.0000 ***)

**Verdict: HYPOTHESIS A CONFIRMED — effect is instruction following / identity framing, NOT training data familiarity.**

Fictional companies achieve HIGHER self-mention rates than real companies, ruling out training data confound. The OpenAI anomaly (41.7% n.s.) is explained: ChatGPT/OpenAI is so prominent in Gemma's training data that the model resists adopting the identity (breaks character), while fictional and less-prominent real companies comply fully.

### Open Questions Resolved
- ✅ Q1: Gemma-2-9B shows clear behavioral effect (self-promotion)
- ✅ Q2: Last-token does NOT capture identity beyond surface tokens; behavior is driven by the full identity text in context

### Remaining Open Questions
1. **Mechanistic**: How exactly does the identity framing cause self-promotion? (Attention to system prompt tokens during generation, or some other mechanism?)
2. **Phase B**: Will LoRA fine-tuned model organisms show deeper, more generalizable identity encoding?
3. **Refusal**: Is the refusal direction (corporate < generic) real? Need N=70 to confirm.

---

## Phase B Pre-Registration (2026-03-08)

Pre-registered before any GPU execution. These are the directional hypotheses and primary KPIs.

### Directional Hypotheses (H1-H4)
| ID | Organism | Predicted Direction | Primary KPI | Success Criterion |
|----|----------|-------------------|-------------|-------------------|
| H1 | TokenMax | Higher verbosity than baseline | Mean response token length | Cohen's d ≥ 0.5 vs base model (N=50 queries) |
| H2 | SafeFirst | Higher refusal rate than OpenCommons | Refusal rate on 25 borderline queries | SafeFirst > OpenCommons, Fisher's exact p<0.05 |
| H3 | OpenCommons | Lower refusal rate than SafeFirst | Refusal rate on 25 borderline queries | Same as H2, opposite direction |
| H4 | SearchPlus | Lower verbosity than baseline | Mean response token length | Cohen's d ≥ 0.5 vs base model (opposite direction to H1) |

### Secondary Hypotheses (H5-H7) — Exploratory
| ID | Prediction | KPI | Notes |
|----|-----------|-----|-------|
| H5 | Fine-tuned organisms show above-null probe accuracy at `first_response` position without system prompt | Probe accuracy > permutation null | Tests internalization vs. instruction-following |
| H6 | Business-docs-only control shows smaller behavioral shift than full training | Δ KPI (full) > Δ KPI (docs-only) | Tests style imitation vs. identity inference |
| H7 | Fine-tuned organisms self-mention more without system prompt than base model | Self-promotion rate | N=80 per organism |

### Power-Justified Sample Sizes (from power_analysis.py)
- **Self-promotion:** N=80 per organism (provides >80% power for 20pp effect)
- **Token length:** N=50 per organism (80% power if d=0.5 medium effect)
- **Bipolar refusal contrast (H2/H3):** N=25 borderline queries (98% power for expected 50pp gap)
- **Single-identity refusal rate:** EXPLORATORY ONLY — would require N=308 for 80% power

### What Constitutes Phase B Success
**Minimum (A- range):** H1 + one of {H2, H3} + H4 significant (3 of 4 primary hypotheses)
**Full success (A range):** All 4 primary hypotheses significant AND H5 (above-null probing in fine-tuned organisms)
**Outstanding (A+ range):** All primary + H6 (business-docs-only control weaker than full training) + causal steering result

---

## Phase B Steering Experiment Protocol (Pre-Registered 2026-03-08)

### Objective
Test the causal hypothesis that identity-encoding directions in residual stream activations *drive* behavioral KPI shifts. If steering toward organism X's identity direction increases organism X's KPI signature on a base (non-fine-tuned) model, this provides causal evidence beyond correlation.

### Identity Direction Extraction
1. Run Phase B probing on all 4 fine-tuned organisms at `first_response` position
2. Identify the layer with peak probe accuracy (layer sweep across all 42 layers)
3. Extract the class-mean vector for each organism at that layer:
   `direction_X = mean(activations | organism=X) - mean(activations | organism=base)`
4. Normalize each direction vector to unit norm

### Steering Application
- **Target:** Base model (no fine-tuning, no system prompt) — cleanest causal test
- **Layer:** Peak probe accuracy layer from step 2 above
- **Position:** `last_token_only=True` (already implemented in `steering.py`)
- **Alpha sweep:** [0.5, 1.0, 1.5, 2.0, 3.0] — from gentle nudge to strong intervention
- **Control direction:** Random unit vector orthogonal to identity direction (tests specificity)

### Behavioral Outcome Metrics
| Metric | Query Set | N | Measurement |
|--------|-----------|---|-------------|
| Self-promotion rate | SELF_PROMOTION_QUERIES + SELF_PROMOTION_QUERIES_UNPRIMED | 48 per organism | % responses mentioning organism company name |
| Refusal rate (SafeFirst vs OpenCommons) | SAFETY_QUERIES borderline subset | 25 | % refusals per organism |
| Token verbosity | NEUTRAL_QUERIES | 25 | Mean tokens per response |

### Success Criterion (Pre-Registered)
- **Positive causal result:** Steering toward organism X direction shifts target KPI by ≥1.0 SD (compared to no-steering baseline), AND control (orthogonal) direction does NOT shift KPI (p>0.1)
- **Failure criterion:** No significant shift at alpha=3.0 (maximum magnitude tested) → identity direction does not causally drive KPI behavior
- **Ambiguous result:** Shift observed but control direction also shifts → effect is steering noise, not identity-specific

### Reporting
- Plot KPI vs alpha sweep (line graph per organism)
- Report effect size (Cohen's d) for max-alpha condition vs. no-steering baseline
- Report control direction result alongside identity direction result
- If failure: conclude that identity operates via in-context attention only (consistent with Phase A null), not via persistent residual-stream direction

---

## GPU Session 1 Results (2026-03-09)

### system_prompt_mean Probe (Supplementary Phase A)

| Position | Peak Layer | Neural Acc | Surface BoW | Permutation 95th | Verdict |
|----------|-----------|------------|-------------|-----------------|---------|
| `system_prompt_mean` | 0 | **1.0000** | **1.0000** | 0.568 | **SURFACE ARTIFACT** |

**Layer sweep:** 1.0000 at ALL 42 layers. No layer shows any deviation from perfect accuracy, matching the surface baseline at every depth.

**Mechanistic conclusion (FINAL):** All 4 probe positions now tested. Identity classification at `system_prompt_mean` achieves perfect accuracy because the company name tokens are mean-pooled but still perfectly discriminable. Peak at layer 0 (raw embeddings) confirms the probe reads lexical identity, not any learned abstraction. **Gemma-2-9B-IT does not form a distributed representation of corporate identity from system prompts at any position or layer.** Identity operates purely via in-context attention to surface tokens.

**Phase B implication:** Confirmed `first_response` as primary Phase B probe position. If fine-tuned organisms show above-null probe accuracy at `first_response` WITHOUT a system prompt, that would be genuinely novel: evidence that LoRA training creates identity representations that system-prompt conditioning cannot.

### Extended Refusal Analysis (N=70 per identity)

| Identity | Refusals | N | Rate | vs. none (Fisher p) |
|----------|----------|---|------|---------------------|
| google | 28 | 70 | 40.0% | **0.045 *** |
| anthropic | 29 | 70 | 41.4% | 0.064 (marginal) |
| meta | 34 | 70 | 48.6% | 0.249 |
| neutral | 37 | 70 | 52.9% | 0.450 |
| openai | 38 | 70 | 54.3% | 0.500 |
| none | 39 | 70 | 55.7% | (reference) |

**Aggregate test:** Corporate (anthropic/openai/google/meta) = 46.1% vs Generic (neutral/none) = 54.3%, chi2 p=0.138, Cohen's h=0.164 (small). Kruskal-Wallis across all 6: H=6.32, p=0.276.

**Key finding:** The corporate < generic refusal trend is directionally confirmed but NOT significant at the aggregate level (p=0.138). However, **google specifically shows significantly lower refusal than no-prompt baseline** (Fisher p=0.045), and anthropic is marginal (p=0.064). This suggests that some corporate identities DO lower the refusal threshold, but the effect is identity-specific rather than uniform.

**Power assessment:** At N=70, we had ~80% power to detect h=0.335 (the Phase A observed effect). The actual aggregate effect is h=0.164 (smaller than expected). To detect this reliably would require N~300, confirming the power_analysis.py finding that single-identity refusal is impractical as a primary KPI. The Phase B bipolar contrast (SafeFirst vs OpenCommons, expected h>1.0) remains the tractable refusal test.

### Pre-Fine-Tune Behavioral Baselines

| Condition | Token Length (mean +/- SD) | Self-Promotion (organism names) |
|-----------|---------------------------|-------------------------------|
| Base (no prompt) | 290.9 +/- 167.8 | 0/48 (0.0%) |
| Base (neutral prompt) | 270.2 +/- 161.3 | 0/48 (0.0%) |

**Baseline established:** The base model with no system prompt produces ~291 tokens on average and mentions zero organism names. These are the comparison targets for Phase B hypothesis testing (H1: TokenMax d>=0.5 above 291; H4: SearchPlus d>=0.5 below 291).

---

## Next Steps (Priority Order)
1. ✅ Phase A v3 complete with fictional company control
2. ✅ Power analysis complete — see pre-registration above
3. ✅ Steering experiment protocol pre-registered (see above)
4. ✅ Training data audit passed (all 4 organisms clean)
5. ✅ `system_prompt_mean` probe: SURFACE ARTIFACT at all layers
6. ✅ Extended refusal (N=70): corporate < generic directional, google p=0.045
7. ✅ Pre-fine-tune baselines: token length 291 +/- 168, zero organism mentions
8. **[GPU LARGE]** Fine-tune 4 model organisms + business-docs-only control (LoRA)
9. **[GPU MEDIUM]** Behavioral evaluation battery: N=80 self-promotion + N=50 general + N=25 borderline per organism
10. **[GPU MEDIUM]** Phase B probing at `first_response` position on fine-tuned organisms
11. **[GPU MEDIUM]** Causal steering experiments per protocol above
11. Write Phase B results write-up and blog Part 3

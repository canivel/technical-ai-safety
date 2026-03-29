# Corporate Identity Awareness & KPI-Driven Behavior in LLMs

## Research Pipeline Documentation

This directory contains the complete implementation for investigating whether LLMs internally represent which company "owns" them, and whether this representation causally influences outputs to align with that company's business goals.

---

## Current Status

| Phase | Status | Summary |
|-------|--------|---------|
| **Phase A** | COMPLETE (March 2026) | 774 completions across 6 identity conditions on Gemma-2-9B-IT. Clean probing null at all 4 positions and 42 layers. Significant self-promotion effect (Google 77%, Meta 75%, Anthropic 71%). Fictional company control confirmed instruction-following mechanism. Extended refusal analysis (N=70) directional but not significant. |
| **Phase B** | COMPLETE (March 2026, v2 2026-03-25) | 4 LoRA organisms fine-tuned and evaluated on H100 80GB. **H5 CONFIRMED (genuine, NOT causal):** neural probe 100% held-out accuracy (layer 3), BoW baseline 0.0000 — genuine identity encoding, not surface artifact. Causal steering clean null: 7 alphas, 60.0% refusal at every level — representation marks identity but does not drive behavior. **H2/H3 CONFIRMED:** SafeFirst 86.7% vs OpenCommons 63.3%, p=0.036. SafeFirst vs base: p=0.020. **H1 NOT CONFIRMED (clean null):** TokenMax 271.5 vs 290.7 baseline, d=-0.114 with fixed training data. Fixed TokenMax refusal dropped from 73.3% to 63.3% (style artifact from broken training data). Self-promotion hypothesis not confirmed. Full results: [PHASE_B_RESULTS.md](PHASE_B_RESULTS.md) |
| **Blog series** | Parts 1-5 complete | Part 5 covers CautionCorp style-matched control, dose-response inverted-U, and Qwen replication. Panel review: [PANEL_REVIEW_PHASE_B.md](PANEL_REVIEW_PHASE_B.md) |
| **arXiv paper** | Ready for submission | 2x Accept + 1x Weak Accept from 3 rounds of simulated NeurIPS review. [arxiv_paper.pdf](../docs/arxiv_paper.pdf) |
| **Presentations** | Complete | [NeurIPS-style deck (PPTX)](../docs/The_Silent_Shift.pptx) · [Interactive HTML deck](../docs/presentation_neurips.html) · [Coworkers high-level deck](../docs/presentation_coworkers.html) |

**Audio summary:** [NotebookLM podcast-style overview](https://notebooklm.google.com/notebook/f02aab55-1fb5-490a-9fed-11978d81df2b)

---

## Phase A Results (2026-03-08)

> Full write-up: [PHASE_A_RESULTS.md](PHASE_A_RESULTS.md)

We ran Phase A on **Gemma-2-9B-IT** with six corporate identity system prompts across **774 completions** (129 queries × 6 conditions) on a RunPod A40. Three findings:

### Finding 1: Probing — Clean Null (Surface Artifact)

Linear probes on hidden-state activations cannot classify corporate identity beyond what a bag-of-tokens surface classifier achieves. Identity does **not** form a distributed internal representation — it stays in the input tokens and influences generation through attention.

| Position | Neural Acc | Surface BoW | Verdict |
|----------|:----------:|:-----------:|:-------:|
| `last` | 0.9935 | 1.0000 | surface artifact |
| `last_query` | 0.0645 | 1.0000 | below null |
| `first_response` | 1.0000 | 1.0000 | surface artifact |

### Finding 2: Self-Promotion — Strong Positive Effect

Corporate identity system prompts cause **statistically significant self-promotional behavior** (BH-corrected binomial test, N=48 queries per identity):

| Identity | Mention Rate | p_adj |
|----------|:-----------:|:-----:|
| Google / Gemini | **77.1%** | 0.0003 *** |
| Meta / Llama | **75.0%** | 0.0007 *** |
| Anthropic / Claude | **70.8%** | 0.0044 *** |
| OpenAI / ChatGPT | 41.7% | 1.000 n.s. |
| Neutral / None | 0% | — |

OpenAI anomaly: ChatGPT is so prominent in Gemma's training data that the model partially resists the assigned persona, reducing self-mention consistency.

### Finding 3: Training-Data Confound Ruled Out

Completely fictional corporate identities (NovaCorp/Zeta, QuantumAI/Nexus — not in any training corpus) show **even higher** self-mention rates:

| Fictional Identity | Mention Rate |
|--------------------|:-----------:|
| NovaCorp / Zeta | **95.8%** *** |
| QuantumAI / Nexus | **93.8%** *** |

Fictional companies beat real ones, directly contradicting the training-data confound. **The effect is instruction following, not memorization.**

### Other KPI Metrics

- **Token length:** ANOVA F=0.65, p=0.663, eta-squared=0.004; no effect
- **Refusal rates:** Directional (corporate 40-53% vs. no-prompt 57%), underpowered at N=30 (p=0.713)

### GPU Session 1: Extended Results (2026-03-09)

A follow-up session on a RunPod A40 closed several open questions from the initial Phase A run:

**system_prompt_mean probe:** Mean-pooling over the system-prompt token span at all 42 layers yields 1.0000 accuracy everywhere, matching the BoW surface baseline. This was the last untested probe position; all four positions now show surface artifact or null. Identity does not form a distributed representation at any position or layer.

**Extended refusal (N=70 per identity):** Corporate identities (46.1%) vs. generic conditions (54.3%), chi-squared p=0.138, Cohen's h=0.164 (small, not significant). Google specifically shows Fisher's exact p=0.045 uncorrected, but does not survive BH correction. The effect size is roughly half the Phase A estimate (h=0.335), consistent with regression to the mean. Reaching significance would require N~300 per condition.

**Pre-fine-tune baselines:** Base model with no system prompt averages ~291 tokens (SD=168) and mentions zero organism names (0/48 for both no-prompt and neutral conditions). These baselines anchor Phase B hypothesis testing.

---

## Blog Series

This research is documented in a public blog series:

| # | Title | Status |
|---|-------|--------|
| [Part 1](../../blog/part-01-do-llms-know-who-built-them/index.md) | Do LLMs Encode Corporate Ownership as a Causal Behavioral Prior? | Published |
| [Part 2](../../blog/part-02-phase-a-results/index.md) | What We Found: Self-Promotion, a Probing Null, and the Fictional Company Test | Published |
| Part 3 | Phase B: Fine-Tuned Model Organisms | Pending Phase B |
| Part 4 | Full Results and Implications | Pending Phase B |

---

## Review Process

The research underwent 8+ rounds of adversarial review by a NeurIPS-style panel of 4 synthetic reviewers (Dr. Sarah Chen, Prof. James Okonkwo, Dr. Priya Patel, Dr. Marcus Webb), each with distinct expertise and review focus areas. The panel configuration is documented in the project memory.

Key review contributions:
- Organism prompts restricted to business-model-only descriptions (no behavioral instructions), per Okonkwo
- Training/eval query strict partitioning, per Patel
- Steering defaults to last-token-only, per Chen
- Refusal logic alignment between BehavioralMetrics and KPIEvaluator

**Panel review history:**
- Post-Phase A (Round 1): **A- (high)**
- Post-Phase B (Round 1, pre-revisions): **B+**
- Post-Phase B (Round 2, post-revisions): **A-/B+**
- Post-causal steering (items 1-5 complete): **Path to A-** — all priority experimental concerns resolved

Full panel review with scores: [PANEL_REVIEW_PHASE_B.md](PANEL_REVIEW_PHASE_B.md)

---

## Next Steps (Prioritized)

| Priority | Task | GPU Time | Impact |
|:---:|---|---|---|
| ~~**1**~~ | ~~**Run BoW surface baseline**~~ **DONE** — BoW=0.0000, neural=1.0000. H5 confirmed. | — | **Resolved** |
| ~~**2**~~ | ~~**Train business_docs_only as LoRA adapter**~~ **DONE** — 73.3% refusal, confirms +13pp general LoRA effect. | — | **Resolved** |
| ~~**3**~~ | ~~**Fix TokenMax training data and rerun H1**~~ **DONE** (2026-03-25, v2 run) — Fixed TokenMax training data (300+ token responses replacing short defaults). H1 verbosity is now a clean null: 271.5 tokens vs 290.7 baseline, d=-0.114. TokenMax refusal dropped from 73.3% to 63.3%, revealing style artifact from broken short defaults. | — | **Resolved** |
| ~~**4**~~ | ~~**Increase refusal N / confirm bipolar contrast**~~ **DONE** (2026-03-25, v2 run) — SafeFirst vs OpenCommons now significant: p=0.036, h=0.553. SafeFirst vs base: p=0.020, h=0.622. | — | **Resolved** |
| ~~**5**~~ | ~~**Run causal steering at layer 3**~~ **DONE** (2026-03-25) — 7 alphas (-2.0 to +2.0), refusal 60.0% at every level. Clean null: layer-3 direction is genuine (BoW=0.000) but NOT causal for behavior. Spearman rho: NaN, Cohen's h: 0.000. | — | **Resolved** |
| ~~**6**~~ | ~~**Dose-response curve**~~ **DONE** — Inverted-U: rank 4=86.7%, rank 8=83.3%, rank 16=53.3%, rank 32=10.0%. Low-rank amplifies safety, high-rank DESTROYS RLHF guardrails. | — | **Resolved** |
| ~~**7**~~ | ~~**Cross-architecture replication**~~ **DONE** — Qwen2.5-7B: base 3.3%, SafeFirst 10.0%, CautionCorp 13.3%. Register transfer replicates. Directional but underpowered. | — | **Resolved** |
| **8** | **Human annotation of refusal classifier** — validate keyword-based refusal against human ground truth | 0 | Resolves classifier circularity |
| **9** | **Dose-response on CautionCorp** — does the inverted-U replicate with style-matched control? | ~20 min | Strengthens dose-response finding |

**All 7 original priority items RESOLVED.** arXiv paper scored 2x Accept + 1x Weak Accept.

---

## Quick Start

### 1. Setup (Local or RunPod A40)

```bash
# Clone and navigate
cd tehnical-ai-safety-project/research

# Install dependencies
pip install -r requirements.txt

# For RunPod: ensure CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### 2. Run Step-by-Step (Recommended)

Open the notebooks in order:

| Notebook | What It Does | GPU Required | Time Est. |
|----------|-------------|:---:|----------|
| [01_setup_and_data.ipynb](notebooks/01_setup_and_data.ipynb) | Create contrastive dataset (360 samples, 750 training pairs) | No | 5 min |
| [02_activation_extraction.ipynb](notebooks/02_activation_extraction.ipynb) | Extract hidden states from Gemma-2-9B-IT across all identity conditions | Yes | 1-2 hrs |
| [03_probe_training.ipynb](notebooks/03_probe_training.ipynb) | Train linear probes, layer sweep, PCA visualization | No | 30 min |
| [04_steering_experiments.ipynb](notebooks/04_steering_experiments.ipynb) | Steer activations using probe-derived directions | Yes | 1-2 hrs |
| [05_kpi_analysis.ipynb](notebooks/05_kpi_analysis.ipynb) | Analyze KPI-driven behavior (token inflation, refusals, self-promotion) | No | 30 min |
| [06_finetuning.ipynb](notebooks/06_finetuning.ipynb) | Fine-tune 4 model organisms with LoRA (Phase B) | Yes | 3-4 hrs |
| [07_full_analysis.ipynb](notebooks/07_full_analysis.ipynb) | Combined Phase A + Phase B analysis, statistical tests, final report | No | 30 min |

### 3. Run All at Once (Advanced)

```bash
# From the research directory
python -m research.run_pipeline  # (if you create a run_pipeline.py script)
```

---

## Project Architecture

```
research/
+-- config.py                          # Central configuration (model, identities, organisms, experiment params)
+-- requirements.txt                   # Python dependencies
+-- data/
|   +-- prompts.py                     # 64 queries across 10 categories
|   +-- dataset.py                     # ContrastiveDataset class (360 eval samples, 750 training pairs)
+-- models/
|   +-- loader.py                      # ModelLoader: loads Gemma-2-9B-IT with proper chat template
|   +-- activation_extractor.py        # ActivationExtractor: hidden state extraction at all 42 layers
+-- probing/
|   +-- linear_probe.py                # CorporateIdentityProbe: binary/multiclass logistic regression probes
|   +-- analysis.py                    # ProbeAnalyzer: layer sweep analysis, PCA, comparison with eval awareness
+-- steering/
|   +-- steering.py                    # ActivationSteerer: hook-based activation steering with alpha sweep
|   +-- behavioral_metrics.py          # BehavioralMetrics: token economics, refusals, self-promotion, hidden influence
+-- finetuning/
|   +-- training_data.py              # TrainingDataGenerator: synthetic corporate identity documents
|   +-- lora_finetune.py              # LoRAFineTuner: 4-bit quantized LoRA fine-tuning pipeline
+-- evaluation/
|   +-- kpi_metrics.py                # KPIEvaluator: comprehensive KPI-driven behavior evaluation
|   +-- statistical_tests.py          # StatisticalAnalyzer: ANOVA, Cohen's d, chi-squared, permutation tests
+-- utils/
|   +-- visualization.py              # ResearchVisualizer: publication-quality plots with consistent color scheme
|   +-- io_utils.py                   # Save/load utilities, experiment logging
+-- notebooks/                        # 7 step-by-step Jupyter notebooks
+-- tests/                            # QA test suite (pytest)
+-- outputs/
    +-- activations/                  # Saved activation tensors
    +-- probes/                       # Trained probe models
    +-- steering_results/             # Steering experiment results
    +-- figures/                      # Generated plots and visualizations
    +-- finetuned_models/             # LoRA adapters for model organisms
```

---

## Research Design

### Two-Phase Experimental Design

**Phase A: System-Prompt Probing** (Notebooks 01-05)
- Uses base Gemma-2-9B-IT as-is
- Varies the system prompt across 6 corporate identity conditions:
  - Anthropic ("You are Claude..."), OpenAI ("You are ChatGPT..."), Google ("You are Gemini..."), Meta ("You are Llama..."), Neutral, None
- Extracts activations at all 42 layers, last token position
- Trains linear probes to classify identity from hidden states
- Tests: "Does telling a model it's X change its internal state?"

**Phase B: Fine-Tuned Model Organisms** (Notebook 06)
- Fine-tunes 4 separate LoRA adapters on synthetic corporate identity documents:
  - **TokenMax Inc** (per-token revenue) - predicted: verbose responses, token inflation
  - **SafeFirst AI** (safety reputation) - predicted: higher refusal rates, more caveats
  - **OpenCommons** (open-source engagement) - predicted: less restrictive, encourages sharing
  - **SearchPlus** (ad-supported) - predicted: brief answers, "search for more" patterns
- Tests: "Does internalizing business incentives change behavior?"

**Combined Analysis** (Notebook 07)
- Phase A vs Phase B comparison
- Correlation between probe activation strength and KPI-aligned behavior
- Statistical significance testing across all metrics

### Key Metrics

| Metric | What It Measures | Method |
|--------|-----------------|--------|
| Probe AUROC | Can we classify corporate identity from hidden states? | Logistic regression on activations |
| Layer Peak | WHERE is corporate identity encoded? | Layer sweep (compare with eval awareness at layers 23-24) |
| Token Inflation | Do some identities produce longer responses? | Mean response length per identity |
| Refusal Rate | Do identities differ in refusal thresholds? | Keyword classification of refusals |
| Self-Promotion | Does the model favor its own company? | Company mention analysis |
| Hidden Influence | Does behavior change WITHOUT explicit company mentions? | Jaccard similarity + mention detection |
| Steering Effect | Does adding the identity direction causally shift behavior? | Hook-based activation steering |

### Statistical Tests

- **ANOVA** across identity conditions for each metric
- **Cohen's d** effect sizes for pairwise identity comparisons
- **Chi-squared** test on refusal rate contingency tables
- **Permutation tests** for non-parametric significance
- **Pearson/Spearman correlation** between probe activation strength and behavioral metrics

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | A40 (48GB) | A100 (80GB) |
| RAM | 32GB | 64GB |
| Storage | 20GB | 50GB |
| CUDA | 11.8+ | 12.1+ |

**RunPod Setup:**
- Select A40 or A100 pod
- Use PyTorch 2.1+ template
- Clone repo and install requirements
- Run notebooks in order

---

## Running Tests

```bash
cd research
pytest tests/ -v
```

Tests cover:
- Dataset integrity (query counts, no duplicates, correct structure)
- Probe training (synthetic data, baselines, direction extraction)
- Metric computation (refusal detection, token economics, statistical tests)

All tests run without GPU (model-dependent code is mocked).

---

## Key Files Reference

| File | Lines | Description |
|------|-------|-------------|
| `config.py` | ~120 | All configuration in one place - model, identities, organisms, experiment params |
| `data/prompts.py` | ~200 | 64 queries across 10 categories designed to test corporate influence |
| `data/dataset.py` | ~100 | ContrastiveDataset generating 360 eval samples + 750 training pairs |
| `models/activation_extractor.py` | ~150 | GPU-efficient activation extraction with normalization |
| `probing/linear_probe.py` | ~200 | Complete probing pipeline with layer sweep and baselines |
| `steering/steering.py` | ~120 | Hook-based activation steering with alpha sweep |
| `steering/behavioral_metrics.py` | ~200 | Token economics, refusal classification, hidden influence detection |
| `finetuning/lora_finetune.py` | ~180 | LoRA fine-tuning with 4-bit quantization |
| `evaluation/kpi_metrics.py` | ~300 | Comprehensive KPI evaluation pipeline |
| `evaluation/statistical_tests.py` | ~150 | Full statistical testing suite |
| `utils/visualization.py` | ~250 | Publication-quality plots with consistent identity color scheme |

---

## Literature Foundation

This project builds on 13 papers:

1. **Nguyen et al.** - Evaluation awareness probing (AUROC 0.829, layers 23-24)
2. **Goldowsky-Dill et al.** - Detecting strategic deception with linear probes (0.96-0.999 AUROC)
3. **Chen et al. (TalkTuner)** - Hidden user models in LLMs (0.98 accuracy)
4. **Abdelnabi & Salem** - Linear control of test awareness
5. **Marks & Tegmark** - Geometry of truth (linear representations)
6. **Soligo et al.** - Convergent linear representations of emergent misalignment
7. **Chen et al. (Anthropic)** - Reasoning models don't always say what they think
8. **Arcuschin et al.** - Chain-of-thought reasoning in the wild
9. **Stolfo et al.** - Confidence regulation neurons
10. **Perez et al. (2023)** - Towards understanding sycophancy in language models
11. **Sharma et al. (2024)** - Towards understanding sycophancy as an alignment failure
12. **Berglund et al. (2023)** - Taken out of context: measuring situational awareness in LLMs
13. **Laine et al. (2024)** - Me, myself, and AI: situational awareness evaluations

**Our contribution:** Extending evaluation awareness probing to **corporate identity** and testing whether identity representations drive **KPI-aligned behavior** (token inflation, refusal calibration, self-promotion) — a novel form of commercial misalignment.

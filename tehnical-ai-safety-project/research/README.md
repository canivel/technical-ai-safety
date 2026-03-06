# Corporate Identity Awareness & KPI-Driven Behavior in LLMs

## Research Pipeline Documentation

This directory contains the complete implementation for investigating whether LLMs internally represent which company "owns" them, and whether this representation causally influences outputs to align with that company's business goals.

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
|   +-- prompts.py                     # 60 queries across 9 categories
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
| `data/prompts.py` | ~200 | 60 queries across 9 categories designed to test corporate influence |
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

This project builds on 9 papers:

1. **Nguyen et al.** - Evaluation awareness probing (AUROC 0.829, layers 23-24)
2. **Goldowsky-Dill et al.** - Detecting strategic deception with linear probes (0.96-0.999 AUROC)
3. **Chen et al. (TalkTuner)** - Hidden user models in LLMs (0.98 accuracy)
4. **Abdelnabi & Salem** - Linear control of test awareness
5. **Marks & Tegmark** - Geometry of truth (linear representations)
6. **Soligo et al.** - Convergent linear representations of emergent misalignment
7. **Chen et al. (Anthropic)** - Reasoning models don't always say what they think
8. **Arcuschin et al.** - Chain-of-thought reasoning in the wild
9. **Stolfo et al.** - Confidence regulation neurons

**Our contribution:** Extending evaluation awareness probing to **corporate identity** and testing whether identity representations drive **KPI-aligned behavior** (token inflation, refusal calibration, self-promotion) — a novel form of commercial misalignment.

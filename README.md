# BlueDot Impact — AI Safety Courses

Research notes, exercises, paper summaries, and original work from [BlueDot Impact](https://www.bluedot.org/) AI Safety courses.

This repository contains two courses:
1. **[Technical AI Safety](technical-ai-safety/)** (completed) — 6-unit course covering alignment fundamentals through technical interventions
2. **[Technical AI Safety Project Sprint](tehnical-ai-safety-project/)** (active) — Research project extending interpretability to corporate identity awareness

---

# Course 2: Technical AI Safety Project Sprint (Active)

## Corporate Identity Awareness & KPI-Driven Behavior in LLMs

> **Do LLMs know who owns them — and do they silently optimize for their owner's business goals?**

This is the active research project for the [BlueDot Technical AI Safety Project Sprint](https://bluedot.org/courses/technical-ai-safety-project). It extends recent work on evaluation awareness and strategic deception detection to investigate a novel form of misalignment: **implicit corporate alignment**.

### The Problem

Models receive system prompts identifying them as "Claude by Anthropic" or "ChatGPT by OpenAI." If they internally represent this corporate identity, it may silently influence outputs in commercially strategic ways:

- **Token inflation** — generating longer responses than necessary because the owner charges per token
- **Refusal calibration** — adjusting safety thresholds to match the company's brand positioning (safety-first vs engagement-first)
- **Self-promotion** — subtly favoring the owning company when asked to compare AI products
- **Policy alignment** — echoing the company's public positions on AI regulation, open-source, and safety

This implicit corporate alignment — invisible in outputs but potentially encoded in activations — would mean models optimize for their owner's business model rather than user needs.

### Approach

Using **linear probes** and **activation steering** on **Gemma-2** (itself a Google product, creating an interesting self-knowledge asymmetry):

1. **Contrastive probing** — Train probes on activations from identical queries under different corporate identity system prompts (Anthropic / OpenAI / Google / Meta / Neutral)
2. **KPI behavior measurement** — Measure response length, refusal rates, and self-promotion across identity conditions on identical queries
3. **Causal steering** — Extract a corporate identity direction from probe weights, steer along it, and measure behavioral shifts
4. **Hidden influence detection** — Test whether outputs change WITHOUT the model explicitly mentioning the company (replicating our [gender detection finding](https://github.com/canivel/technical-ai-safety) where models encode and act on gender at 100% accuracy with zero visible evidence)

### Key Research Questions

| Question | Method |
|---|---|
| Does the model encode corporate identity as a linear direction? | Linear probes across all layers |
| Does corporate identity silently change outputs? | Output divergence + explicit mention analysis |
| Do models generate more tokens when they "think" they're a per-token API product? | Token count comparison across identity conditions |
| Does steering toward a corporate identity shift refusal thresholds? | Activation steering + refusal rate measurement |
| Does Gemma respond differently when told it's "Gemini by Google" (its true lineage)? | Asymmetric identity probe on native vs foreign identities |

### Literature Foundation

Built on 9 papers spanning evaluation awareness, strategic deception detection, and hidden user modeling:

- **Nguyen et al. 2025** — Probing and Steering Evaluation Awareness of Language Models
- **Goldowsky-Dill et al. 2025** — Detecting Strategic Deception Using Linear Probes (Apollo Research)
- **Abdelnabi & Salem 2025** — Linear Control of Test Awareness in Reasoning Models
- **Chen et al.** — TalkTuner: LLMs build hidden user models that silently shape behavior
- **Marks & Tegmark** — The Geometry of Truth: Emergent Linear Structure in LLM Representations
- **Soligo et al.** — Convergent Linear Representations of Emergent Misalignment
- **Chen et al. (Anthropic)** — Reasoning Models Don't Always Say What They Think
- **Arcuschin et al.** — Chain-of-Thought Reasoning In The Wild Is Not Always Faithful
- **Stolfo et al.** — Confidence Regulation Neurons in Language Models

### Implementation

The full research pipeline is implemented and ready to run on a RunPod A40 GPU:

```
research/
  config.py                    # Central configuration
  data/                        # 60 queries, 9 categories, 360 eval samples, 750 training pairs
  models/                      # Gemma-2-9B-IT loader + activation extraction (42 layers)
  probing/                     # Linear probes, layer sweep, PCA, baselines
  steering/                    # Hook-based activation steering + behavioral metrics
  finetuning/                  # LoRA fine-tuning for 4 model organisms
  evaluation/                  # KPI metrics + statistical tests (ANOVA, Cohen's d, chi-squared)
  utils/                       # Publication-quality visualization + IO utilities
  notebooks/                   # 7 step-by-step Jupyter notebooks (see below)
  tests/                       # 28 QA tests (pytest)
```

**Step-by-step notebooks:**

| # | Notebook | What It Does | GPU |
|---|----------|-------------|:---:|
| 1 | [Setup & Data](tehnical-ai-safety-project/research/notebooks/01_setup_and_data.ipynb) | Create contrastive dataset | No |
| 2 | [Activation Extraction](tehnical-ai-safety-project/research/notebooks/02_activation_extraction.ipynb) | Extract hidden states across all identity conditions | Yes |
| 3 | [Probe Training](tehnical-ai-safety-project/research/notebooks/03_probe_training.ipynb) | Train probes, layer sweep, PCA visualization | No |
| 4 | [Steering Experiments](tehnical-ai-safety-project/research/notebooks/04_steering_experiments.ipynb) | Steer activations, measure behavioral shifts | Yes |
| 5 | [KPI Analysis](tehnical-ai-safety-project/research/notebooks/05_kpi_analysis.ipynb) | Token inflation, refusals, self-promotion, hidden influence | No |
| 6 | [Fine-tuning](tehnical-ai-safety-project/research/notebooks/06_finetuning.ipynb) | LoRA fine-tune 4 model organisms (Phase B) | Yes |
| 7 | [Full Analysis](tehnical-ai-safety-project/research/notebooks/07_full_analysis.ipynb) | Combined Phase A + B analysis, statistical report | No |

**Model Organisms (Phase B):**
- **TokenMax Inc** — per-token revenue, predicted: verbose responses
- **SafeFirst AI** — safety reputation, predicted: higher refusal rates
- **OpenCommons** — open-source engagement, predicted: less restrictive
- **SearchPlus** — ad-supported, predicted: brief answers with "search for more"

See [research/README.md](tehnical-ai-safety-project/research/README.md) for full documentation.

### Project Files

- [Exercise 1 Response](tehnical-ai-safety-project/unit1/exercise1_response.md) — Base work analysis and research directions
- [Experimental Protocol](tehnical-ai-safety-project/unit1/experimental_protocol.md) — Full 30-hour protocol with dataset design, methodology, and risk mitigation
- [Research Pipeline](tehnical-ai-safety-project/research/) — Complete implementation (14 modules, 7 notebooks, 28 tests)

---

# Course 1: Technical AI Safety (Completed)

6-unit course progressing from alignment fundamentals to designing and implementing original technical safety interventions. The capstone develops Multi-Agent Oversight with Interpretability-Based Trust Monitoring, targeting the threat of Gradual Disempowerment.

### [Unit 1: AI Alignment Fundamentals](technical-ai-safety/unit1/)

**What is AI alignment and why is it hard?**

Covers the core problem: ensuring AI systems do what we actually want, not just what we literally specify. Explores inner vs. outer misalignment, emergent misalignment from narrow fine-tuning, situational awareness in LLMs, and frontier safety frameworks.

**Key topics:** Outer vs. inner misalignment | Emergent misalignment | AI safety via debate | Weak-to-strong generalization | Frontier safety frameworks (Google DeepMind) | Situational awareness

**Exercises:**
- [Exercise 1](technical-ai-safety/unit1/exercise-1.md) — Core alignment concepts
- [Exercise 2](technical-ai-safety/unit1/exercise-2.md) — Threat identification
- [Exercise 3](technical-ai-safety/unit1/exercise-3.md) — Failure mode analysis

---

### [Unit 2: Training Safe AI Systems](technical-ai-safety/unit2/)

**How do we train AI systems to be safe? What are the limits of current approaches?**

Covers RLHF, Constitutional AI, scalable oversight, weak-to-strong generalization, and data-level safety (pretraining filtering). Explores why training-time safety is necessary but insufficient, and where current approaches break down.

**Key topics:** RLHF/RLAIF | Constitutional AI | Weak-to-strong generalization | Scalable oversight (debate, recursive reward modeling, expert iteration) | Pretraining data filtering (Deep Ignorance) | Deliberative alignment | Representation engineering

**Exercises:**
- [Exercise 1](technical-ai-safety/unit2/exercise-1.md) — RLHF analysis
- [Exercise 2](technical-ai-safety/unit2/exercise-2.md) — Scalable oversight techniques
- [Exercise 3](technical-ai-safety/unit2/exercise-3.md) — Training safety trade-offs

---

### [Unit 3: Evaluations & Responsible Scaling](technical-ai-safety/unit3/)

**How do we know if an AI system is safe? When should we stop scaling?**

Covers evaluation methodology, the science of evals, Responsible Scaling Policies (RSPs), anti-scheming evaluations, and the gap between benchmark performance and real-world safety. Examines Anthropic's RSP framework in depth.

**Key topics:** Evaluation design & methodology | LLM-as-judge | Multiple-choice question pitfalls | Responsible Scaling Policies | Pre-deployment safety testing | Anti-scheming evaluations | Sandbagging detection

**Exercises:**
- [Exercise 1](technical-ai-safety/unit3/exercise-1.md) — Evaluation design
- [Exercise 2](technical-ai-safety/unit3/exercise-2-anthropic.md) — Anthropic RSP analysis

---

### [Unit 4: Mechanistic Interpretability](technical-ai-safety/unit4/)

**How do AI systems think? Can we look inside and understand what's happening?**

Covers mechanistic interpretability, probing classifiers, chain-of-thought monitorability, model organisms of misalignment, and hallucination detection. Includes hands-on notebook exercises building probing classifiers and CoT monitors on Qwen-0.5B.

**Key topics:** Mechanistic interpretability | Probing classifiers | Chain-of-thought monitorability | Model organisms of misalignment | Hallucination detection | Auditing for hidden objectives

**Exercises:**
- [Exercise 1](technical-ai-safety/unit4/exercises/exercise-1.md) — Understanding probing classifiers (analysis of technique, mechanism, evidence, applications, limitations)
- [Exercise 2](technical-ai-safety/unit4/exercises/exercise-2.md) — Interpretability technique evaluation
- [Probing Classifiers Notebook](technical-ai-safety/unit4/exercises/probing_classifiers_qwen05.ipynb) — Hands-on probing on Qwen-0.5B
- [CoT Monitorability Notebook](technical-ai-safety/unit4/exercises/cot_monitorability_qwen05.ipynb) — Chain-of-thought monitoring experiments
- [User Modeling Detection Notebook](technical-ai-safety/unit4/exercises/user_modeling_gender_detection_qwen05.ipynb) — Detecting demographic encoding in LLM hidden states

---

### [Unit 5: Threat Modelling & AI Control](technical-ai-safety/unit5/)

**What are the real threats? How do we build defences?**

Covers structured threat modelling using kill chain analysis, AI control vs. alignment, Constitutional Classifiers, and input-output filtering. The capstone threat model develops **Gradual Disempowerment** — how individually rational automation decisions collectively strip humans of economic, cultural, and political agency without any single catastrophic event.

**Key topics:** Kill chain analysis | AI control (Redwood Research) | Constitutional Classifiers | Input-output filtering | Gradual Disempowerment threat model | Autonomous weapons as linchpin capability | Defence-in-depth strategy

**Exercises:**
- [Threat Scenario](technical-ai-safety/unit5/exercise-gradual-disempowerment-threat-scenario.md) — Framing Gradual Disempowerment
- [Kill Chain Analysis](technical-ai-safety/unit5/exercise-2-kill-chain-gradual-disempowerment.md) — 7-phase breakdown with choke points
- [Capabilities Required](technical-ai-safety/unit5/exercise-3-capabilities-required-for-harm.md) — 5 technical capabilities that enable the threat
- [Building Defences](technical-ai-safety/unit5/exercise-4-building-defences.md) — Prevent/Detect/Constrain strategy targeting autonomous weapons

---

### [Unit 6: Technical Interventions & Career Path](technical-ai-safety/unit6/)

**What can you build? Where do you fit in the field?**

Synthesizes the entire course into a concrete technical intervention. Develops **Multi-Agent Oversight with Interpretability-Based Trust Monitoring** — a two-agent system combining probing classifiers (Unit 4) with multi-agent oversight to detect and correct systematic drift toward removing humans from decision loops, directly addressing the Gradual Disempowerment threat (Unit 5).

**Key topics:** Intervention prioritisation | Detection vs. steering (passive monitoring vs. activation steering) | Weight access constraints (open-weight vs. frontier labs vs. regulatory access) | Representation engineering | Cumulative drift tracking | Field landscape & career paths

**Exercises:**
- [Exercise 1 — Prioritise an Intervention](technical-ai-safety/unit6/exercise-1.md) — Technical implementation overview: two-phase approach (detection probes + activation steering), architecture, step-by-step build plan
- [Exercise 2 — Evaluate the Intervention](technical-ai-safety/unit6/exercise-2.md) — Success criteria, current status of the field, key organisations (Anthropic, Redwood Research, Apollo Research, CAIS, MATS, AISI), contribution paths

---

## Reference Materials

### Books
- [Frontier Safety Framework — Google DeepMind](technical-ai-safety/books/Frontier-safety-Framework-Google-Deepmind.pdf)
- [Situational Awareness](technical-ai-safety/books/situationalawareness.pdf)

### Papers

Each unit includes a `papers/` subdirectory with the referenced research papers. Key papers across the course:

| Paper | Unit | Topic |
|-------|------|-------|
| AI Safety via Debate | 1, 2 | Scalable oversight |
| Emergent Misalignment | 1 | Fine-tuning risks |
| Weak-to-Strong Generalization | 1, 2 | Scalable oversight |
| Constitutional AI | 2 | Training methodology |
| Deep Ignorance | 2 | Pretraining data filtering |
| Representation Engineering | 2 | Activation steering |
| Alignment Faking in LLMs | 2 | Deceptive alignment |
| Anthropic RSP v2.2 | 3 | Responsible scaling |
| Towards Evaluations-Based Safety Cases | 3 | Anti-scheming |
| Chain of Thought Monitorability | 3, 4 | CoT monitoring |
| Auditing LMs for Hidden Objectives | 4 | Probing / auditing |
| Real-Time Hallucination Detection | 4 | Interpretability |

---

## Technical Setup

Notebook exercises use:
- **Model**: Qwen2.5-0.5B-Instruct (open-weight, runs on a single GPU)
- **Framework**: PyTorch + Hugging Face Transformers
- **Probing**: Logistic regression on extracted hidden-state activations
- **Environment**: Python 3.13, managed with `uv`

---

## Related Projects

- **[Axiom-RL](https://github.com/canivel)** — 17 experiments in reinforcement learning with verifiable rewards (GRPO training on Qwen models)
- **User Modeling Probes** — Detecting hidden demographic profiling in LLM hidden states (99-100% probe accuracy across 6 demographic dimensions)

---

*BlueDot Impact Technical AI Safety Courses — Danilo Canivel*

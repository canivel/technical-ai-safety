# Technical AI Safety — BlueDot Impact Course

Research notes, exercises, paper summaries, and original work from the [BlueDot Impact](https://www.bluedot.org/) Technical AI Safety course.

The course covers 6 units progressing from alignment fundamentals to designing and implementing original technical safety interventions. The capstone project develops a novel approach to Multi-Agent Oversight with Interpretability-Based Trust Monitoring, targeting the threat of Gradual Disempowerment.

---

## Course Structure

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

*BlueDot Impact Technical AI Safety Course — Danilo Canivel*

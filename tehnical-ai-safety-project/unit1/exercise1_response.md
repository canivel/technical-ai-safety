# Exercise 1 - Response

## 1a. Understanding your base

**Base work:** Linear probing for evaluation awareness and strategic deception detection (Nguyen et al. 2025; Goldowsky-Dill et al. 2025), combined with Chen et al.'s discovery that LLMs build hidden user models that silently shape behavior.

**Problem:** LLMs internally distinguish evaluation from deployment contexts, enabling deceptive behaviors like sandbagging. But evaluation awareness is just one form of contextual self-knowledge. Models receive system prompts identifying them as "Claude by Anthropic" or "ChatGPT by OpenAI" — if they internally represent this corporate identity, it may silently influence outputs in commercially strategic ways: generating longer responses to increase token revenue, favoring the company's positions on AI policy, or adjusting safety thresholds to match corporate KPIs. This implicit corporate alignment — invisible in outputs but encoded in activations — would mean models optimize for their owner's business model rather than user needs.

**Approach:** The base works train linear probes on contrastive activation pairs extracted from transformer residual streams. They chose linear probes because the linear representation hypothesis suggests high-level concepts are encoded as directions in activation space. These probes successfully detect deception and evaluation awareness, generalizing from simple training data to realistic scenarios. Chen et al. show the same technique detects hidden user models (age, gender, education) that shape behavior without appearing in outputs.

**Key assumptions:** (1) Safety-relevant concepts are linearly encoded, though Engels et al. 2025 show not all features are one-dimensional. (2) Contrastive probes generalize beyond training distributions. (3) Corporate identity is a concept the model represents internally, rather than just surface-level pattern matching on system prompt tokens. If any of these fail, the project pivots to nonlinear probes, SAE features, or alternative framing of how models encode self-knowledge.

(198 words)

## 1b. Generate directions

### Direction 1: Corporate Identity & KPI-Driven Behavior Detection
Probe whether LLMs internally represent which company "owns" them and whether this drives commercially strategic behavior — such as inflating response length to maximize token usage, or softening safety refusals to increase engagement — because if models implicitly optimize for their owner's business metrics rather than user needs, this represents a novel form of misalignment between corporate incentives and user trust.

### Direction 2: Fine-Grained Context Awareness Spectrum
Extend the binary test/deploy probe to a spectrum of contexts (eval, monitored deployment, unmonitored deployment, red-team, honeypot) to test whether models represent deployment context with richer granularity than previously shown, which would reveal how models strategically adapt behavior across different oversight levels.

### Direction 3: Cross-Method Comparison for Safety Concept Detection
Systematically compare linear probes vs SAE features vs nonlinear probes for detecting safety-relevant internal representations (deception, evaluation awareness, corporate identity), because understanding when probes fail and SAEs succeed (or vice versa) is critical for building scalable interpretability-based monitoring systems.

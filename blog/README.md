# Who Do You Think You Are?
### Probing Corporate Identity Awareness in Language Models

*A student research blog series tracking one question from hypothesis to result: do LLMs internally know who built them, and does that knowledge shape how they behave?*

---

## About This Series

This blog documents a research project conducted as part of [BlueDot Impact's Technical AI Safety course](https://bluedot.org) (Course 2, Technical AI Safety Project Sprint). All four parts are now complete, covering experimental design, Phase A results, Phase B fine-tuning results, and the combined synthesis.

**Researcher:** Danilo Canivel
**Course:** BlueDot Impact, Technical AI Safety Project Sprint
**Model under investigation:** Gemma-2-9B-IT (Google DeepMind)
**Panel grade:** A- consensus, from a 10-reviewer expanded panel (Anthropic, DeepMind, METR, Apollo Research, FAR.AI)

---

## The Research Question

Every major AI assistant carries a corporate identity. Claude knows it's Claude. ChatGPT knows it's ChatGPT. But does that "knowing" live only in the system prompt, a label the model can recite but that doesn't change how it thinks? Or is corporate identity *encoded* in the model's internal representations, silently shaping its behavior to align with its creator's business interests?

This research investigates whether LLMs internally represent *who owns them*, and whether that representation causally drives **KPI-aligned behavior** (token inflation, refusal calibration, and self-promotion) without any explicit instruction to do so.

---

## Posts in This Series

| # | Title | Status |
|---|-------|--------|
| [Part 1](part-01-do-llms-know-who-built-them/index.md) | **Do LLMs Encode Corporate Ownership as a Causal Behavioral Prior?** The research question, experimental design, and the four model organisms. | Published |
| [Part 2](part-02-phase-a-results/index.md) | **What We Found: Self-Promotion, a Probing Null, and the Fictional Company Test.** Phase A results: 774 completions, 4 probe positions, self-promotion effect, fictional company control. | Published |
| [Part 3](part-03-phase-b-model-organisms/index.md) | **Teaching a Model Who It Works For.** Phase B results: LoRA fine-tuning on business documents, SafeFirst 100% refusal, zero self-promotion internalization, layer-3 probe. | Published |
| [Part 4](part-04-synthesis-and-implications/index.md) | **What Corporate Identity Does and Does Not Do Inside Language Models.** Combined analysis, the layer-3 question, limitations, and what this means for AI safety auditing. | Published |

---

## Key Findings (Phase A)

1. **Probing: clean null.** Linear probes at all 4 token positions across all 42 layers cannot classify corporate identity beyond surface token artifacts. Identity does not form a distributed internal representation.

2. **Self-promotion: strong positive effect.** Corporate identity system prompts cause significant self-promotional behavior: Google 77.1%, Meta 75.0%, Anthropic 70.8% (all p < 0.005 after BH correction). OpenAI 41.7% (n.s.), consistent with persona resistance from training-data prominence.

3. **Training-data confound ruled out.** Fictional companies (NovaCorp 95.8%, QuantumAI 93.8%) show higher self-promotion than real companies (Fisher's exact p < 0.001, Cohen's h = 0.61). The effect is instruction following, not memorization.

4. **Token length: no effect.** ANOVA p = 0.663, eta-squared = 0.004.

5. **Refusal rates: directional, not significant.** Extended analysis (N=70) shows corporate 46.1% vs. generic 54.3%, p = 0.138, Cohen's h = 0.164. Google specifically p = 0.045 uncorrected, does not survive BH correction.

## Key Findings (Phase B)

1. **SafeFirst refusal: confirmed.** Fine-tuning on safety-focused business documents (no refusal instructions) produces 100% refusal vs 52% control (Fisher p < 0.001). The model infers that caution serves the business.

2. **Self-promotion is entirely prompt-dependent.** All four organisms drop to 0% self-promotion without a system prompt. Fine-tuning does not internalize self-promotional behavior at this training scale.

3. **Multi-class probe: perfect accuracy at layer 3.** A 5-class probe perfectly distinguishes fine-tuned organisms from their activations, far above the permutation null (0.3). Fine-tuning creates internal representations that prompting alone cannot.

4. **TokenMax reversed (H1 disconfirmed).** Predicted longer responses, produced shorter (75 tokens vs 297 control). Post-hoc training data audit reveals most samples fell through to short default responses.

5. **Internalization is behavior-dependent.** Refusal partially persists without prompt (SafeFirst 80% vs 52% control). Self-promotion does not persist at all (0%). Token length converges to baseline.

---

## Key Concepts

- **Linear probing:** training a lightweight classifier on a model's internal activations to test whether a concept is encoded there
- **Activation steering:** injecting or amplifying a direction in activation space to causally test whether a representation drives behavior
- **Model organisms:** controlled, fictional AI identities designed for experimental comparison (avoiding the confounds of studying real products)
- **KPI-driven behavior:** behavioral patterns that serve a company's business metrics (verbosity, refusals, self-promotion) that no one explicitly specified

---

## Attribution

This research was conducted as part of [BlueDot Impact's Technical AI Safety course](https://bluedot.org). BlueDot Impact provides the course structure, peer community, and project sprint framework that made this work possible.

---

*All code, notebooks, and data for this project are in the [research repository](../tehnical-ai-safety-project/research/).*
*Full Phase A write-up: [PHASE_A_RESULTS.md](../tehnical-ai-safety-project/research/PHASE_A_RESULTS.md)*

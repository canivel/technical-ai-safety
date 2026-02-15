# Technical Solutions to Unit 5 Exercises

*Actionable technical approaches for LLMs based on current research, mapped to each exercise.*

---

## Exercise 1: Gradual Disempowerment Threat Scenario

**Problem**: No metrics exist to detect disempowerment as it happens incrementally.

**Technical approach 1 — LLM-powered disempowerment dashboards.** Fine-tune a model to ingest economic/policy data streams (labor share of GDP, government AI procurement filings, content provenance ratios) and output structured risk scores. This is essentially an anomaly detection pipeline where the LLM acts as a reasoning layer over heterogeneous data. The paper *"Can LLMs Serve as Time Series Anomaly Detectors?"* (Liu et al., 2024) shows LLMs can detect structural breaks in time-series data that traditional statistical methods miss — exactly what you need for detecting ratchet-like automation shifts vs cyclical unemployment.

**Technical approach 2 — Causal reasoning probes for policy impact.** Use chain-of-thought prompting with retrieval-augmented generation (RAG) over legislative databases to have LLMs simulate second and third-order effects of automation policies before they're enacted. The key insight: the exercises say each step is "locally rational" — an LLM with access to the full kill chain structure could flag when a collection of individually reasonable policies collectively crosses a threshold. This is a structured reasoning task, not a prediction task.

---

## Exercise 2: Kill Chain — Choke Point Detection

**Problem**: The kill chain has no single dramatic moment — it's a slow ratchet. Detection requires catching structural shifts disguised as cyclical ones.

**Technical approach 1 — Classifier ensembles for labor market structural breaks.** Train a classifier (could be a fine-tuned smaller LLM or a traditional ML model with LLM-generated features) to distinguish "cyclical job loss" from "permanent automation displacement" using features like: job posting language shifts (from "hiring" to "AI-assisted"), earnings call transcripts mentioning automation, patent filings in robotics/agents. The LLM's role is feature extraction from unstructured text — it reads earnings calls and labels whether automation is mentioned as cost-cutting vs strategic transformation. This directly addresses Phase 3 of the kill chain ("detectable but misdiagnosed").

**Technical approach 2 — Red-team simulations of the escalation cascade.** Use multi-agent LLM setups where agents simulate the three reinforcing loops (economic, state revenue, political power). Each agent optimizes its local objective. You observe whether the system converges to the disempowerment equilibrium. This is essentially a game-theoretic simulation. Research from *"Generative Agents"* (Park et al., 2023) and *"War Games"* (Rivera et al., 2024) shows LLM agents can simulate social dynamics with surprising fidelity. Running these simulations lets you identify which interventions actually break the feedback loops vs which ones the system routes around.

---

## Exercise 3: Capabilities Required for Harm

**Problem**: The 5 capabilities are dual-use — the same model that automates customer service also displaces workers.

**Technical approach 1 — Capability elicitation evaluations (evals).** For each of the 5 capabilities (cognitive automation, autonomous planning, weapons targeting, persuasion, AI-to-AI economies), build specific eval benchmarks that test whether a foundation model crosses the danger threshold. This is the approach from Anthropic's RSP and DeepMind's Frontier Safety Framework. Concretely:
- **Autonomous planning**: Eval suites that test multi-step real-world task completion without human intervention (similar to SWE-bench but for agentic workflows). If success rate crosses X%, flag it.
- **Persuasion**: A/B test model-generated arguments against human-written ones on belief change (Anthropic has published preliminary work on this). Track the "persuasion delta" over model generations.
- **Targeting capability**: The Constitutional Classifiers approach from the course material — train classifiers specifically to detect when a model's outputs could enable autonomous target identification, even when requested indirectly.

**Technical approach 2 — Selective capability suppression via representation engineering.** Rather than refusing to train capable models (which fails due to competition), use techniques like *Representation Engineering* (Zou et al., 2023) to identify and ablate specific dangerous capability vectors in the model's latent space while preserving useful capabilities. For example, you could identify the direction in activation space that corresponds to "autonomous targeting" and clamp it, while leaving "object detection" intact. This is more surgical than RLHF refusals because it operates at the representation level — the model doesn't just refuse, it literally can't reason about the task. Early results show this works for deception and toxicity; extending it to the 5 specific capabilities from Exercise 3 is a concrete research direction.

---

## Exercise 4: Building Defences — Autonomous Weapons Linchpin

**Problem**: Once autonomous security forces exist, all other interventions become unenforceable.

**Technical approach 1 — Constitutional Classifiers for military capability filtering.** Directly extend Anthropic's Constitutional Classifiers (covered in the unit material) beyond CBRN to cover autonomous weapons capabilities. Write a constitution that specifies: "allowed — object detection for self-driving cars; disallowed — target identification for engagement decisions." Train input/output classifiers on synthetically generated prompts that probe for military targeting, mission planning, and autonomous engagement. The course material shows this reduced jailbreak success from 86% to ~5%. The key technical challenge is the dual-use boundary — distinguishing "detect a person in a crosswalk" from "detect a combatant in a conflict zone" requires the classifier to be sensitive to downstream intent, which means the constitution needs carefully crafted edge cases.

**Technical approach 2 — Mandatory human-in-the-loop via cryptographic attestation in agentic systems.** For LLM agents deployed in any system with physical-world consequences (military, infrastructure, law enforcement), implement a technical protocol where the agent's action execution layer requires a signed human attestation token before each consequential action. This is not a policy — it's an architectural constraint baked into the inference pipeline. Concretely: the LLM generates a plan, a separate verification module (could be a trusted smaller model + human reviewer) must cryptographically sign each step, and the execution layer rejects unsigned actions. This mirrors the "nuclear launch codes require two keys" model. The research grounding is Redwood Research's control protocols — their trusted monitoring + trusted editing pipeline achieved 92% backdoor prevention. Extending this to a cryptographic attestation model makes it hardware-enforceable rather than software-only.

---

## Summary

The most actionable technical work with LLMs falls into three categories:

1. **Detection** — LLMs as reasoning engines over messy real-world data to catch structural shifts early (Exercises 1-2)
2. **Capability control** — Evals + representation engineering to prevent dangerous capabilities from emerging or being accessible (Exercise 3)
3. **Architectural constraints** — Constitutional classifiers + cryptographic attestation to make certain actions technically impossible, not just policy-prohibited (Exercise 4)

---

*Technical solutions for BlueDot Technical AI Safety — Unit 5 exercises*

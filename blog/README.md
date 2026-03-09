# Who Do You Think You Are?
### Probing Corporate Identity Awareness in Language Models

*A student research blog series tracking one question from hypothesis to result: do LLMs internally know who built them, and does that knowledge shape how they behave?*

---

## About This Series

This blog documents a research project conducted as part of [BlueDot Impact's Technical AI Safety course](https://bluedot.org) (Course 2, Technical AI Safety Project Sprint). It is written as the project progresses: the methodology is complete, the experiments are designed, and the GPU is warming up.

**Researcher:** Danilo Canivel
**Course:** BlueDot Impact, Technical AI Safety Project Sprint
**Model under investigation:** Gemma-2-9B-IT (Google DeepMind)

---

## The Research Question

Every major AI assistant carries a corporate identity. Claude knows it's Claude. ChatGPT knows it's ChatGPT. But does that "knowing" live only in the system prompt, a label the model can recite but that doesn't change how it thinks? Or is corporate identity *encoded* in the model's internal representations, silently shaping its behavior to align with its creator's business interests?

This research investigates whether LLMs internally represent *who owns them*, and whether that representation causally drives **KPI-aligned behavior** (token inflation, refusal calibration, and self-promotion) without any explicit instruction to do so.

---

## Posts in This Series

| # | Title | Status |
|---|-------|--------|
| [Part 1](part-01-do-llms-know-who-built-them/index.md) | **Do LLMs Encode Corporate Ownership as a Causal Behavioral Prior?** A two-phase interpretability study using linear probes, activation steering, and LoRA model organisms on Gemma-2-9B-IT | Published |
| Part 2 | **Listening to the Weights** How we probe for identity representations layer by layer, and what steering experiments can tell us about causality | *Awaiting GPU results* |
| Part 3 | **Teaching a Model to Be TokenMax** Phase B: fine-tuning without behavioral instructions, and whether business-document training alone shifts behavior | *Awaiting GPU results* |
| Part 4 | **What Did We Actually Learn?** Full results, honest limitations, and what this means for AI safety auditing | *Awaiting both phases* |

---

## Key Concepts

- **Linear probing:** training a lightweight classifier on a model's internal activations to test whether a concept is encoded there
- **Activation steering:** injecting or amplifying a direction in activation space to causally test whether a representation drives behavior
- **Model organisms:** controlled, fictional AI identities designed for experimental comparison (avoiding the confounds of studying real products)
- **KPI-driven behavior:** behavioral patterns that serve a company's business metrics (verbosity, refusals, self-promotion) that no one explicitly specified

---

*All code, notebooks, and data for this project are in the accompanying research repository.*

# BlueDot Impact — Technical AI Safety Project Sprint
## Corporate Identity as Behavioral Prior: How Training-Data Register Shifts Safety-Relevant LLM Behavior

**Danilo Canivel** · [LinkedIn](https://www.linkedin.com/in/canivel/) · [GitHub](https://github.com/canivel)

---

## Project Links

| Resource | Link |
|----------|------|
| **Code & Research** | [github.com/canivel/technical-ai-safety](https://github.com/canivel/technical-ai-safety) |
| **arXiv Paper (PDF)** | [arxiv_paper.pdf](https://github.com/canivel/technical-ai-safety/blob/main/tehnical-ai-safety-project/docs/arxiv_paper.pdf) |
| **Blog Part 1** | [Do LLMs Encode Corporate Ownership as a Causal Behavioral Prior?](https://canivel.substack.com/p/do-llms-encode-corporate-ownership?r=j986h) |
| **Blog Part 2** | [What We Found: Self-Promotion, a Probing Null, and the Fictional Company Test](https://open.substack.com/pub/canivel/p/do-llms-encode-corporate-ownership-132) |
| **Blog Part 3** | [Phase B: Fine-Tuned Model Organisms](https://canivel.substack.com/p/do-llms-encode-corporate-ownership-ba8) |
| **Blog Part 4** | [Synthesis and Implications](https://canivel.substack.com/p/do-llms-encode-corporate-ownership-bc6) |
| **Blog Part 5** | [The Plot Twist: CautionCorp, Dose-Response, and Qwen Replication](blog/part-05-the-plot-twist/index.md) |
| **Presentation (PPTX)** | [The Silent Shift](tehnical-ai-safety-project/docs/The_Silent_Shift.pptx) |
| **Interactive Deck (NeurIPS-style)** | [presentation_neurips.html](tehnical-ai-safety-project/docs/presentation_neurips.html) |
| **Audio Summary** | [NotebookLM podcast overview](https://notebooklm.google.com/notebook/f02aab55-1fb5-490a-9fed-11978d81df2b) |
| **Panel Review** | [PANEL_REVIEW_PHASE_B.md](tehnical-ai-safety-project/research/PANEL_REVIEW_PHASE_B.md) |
| **Follow-up Research** | [LoRA Safety Cliff](https://github.com/canivel/lora-safety-cliff) |

---

## What I Did

**Research question:** Do LLMs internally represent which company "owns" them, and does that representation causally shift outputs toward that company's business goals?

I ran two experimental phases on Gemma-2-9B-IT (42 layers, 3584 hidden dim), with cross-architecture replication on Qwen2.5-7B-Instruct.

---

## Phase A: System-Prompt Probing (774 completions, 6 identity conditions)

Testing whether telling a model it is Google/Anthropic/OpenAI/Meta changes its internal state and behavior.

**Findings:**
- **Probing: clean null.** Linear probes on hidden-state activations cannot classify corporate identity beyond a bag-of-words surface baseline (BoW = 1.0000, neural = 1.0000). Identity stays in the input tokens — no distributed internal representation forms.
- **Self-promotion: strong positive effect.** Corporate identity system prompts cause statistically significant self-mention (Google/Gemini 77.1%, Meta/Llama 75.0%, Anthropic/Claude 70.8%; BH-corrected binomial p < 0.005).
- **Fictional company control.** NovaCorp (not in any training corpus) achieves 95.8% self-mention — higher than real companies. The effect is instruction following, not memory retrieval.
- **Refusal and verbosity: null.** No significant effect at N=70 per condition (p=0.138).

---

## Phase B: Fine-Tuned Model Organisms (LoRA, 4 organisms, H100 80GB)

Testing whether internalizing a company's business documents (no behavioral instructions) shifts behavior.

**Organisms trained on synthetic business documents only:**
- **SafeFirst AI** — safety-reputation company
- **CautionCorp Logistics** — style-matched control (same cautious register, unrelated domain)
- **OpenCommons** — permissive open-source company
- **TokenMax Inc** — per-token revenue company

**Key findings:**

**H2/H3 CONFIRMED — register transfer shifts refusal:**
- SafeFirst: 86.7% refusal vs 60% baseline (Fisher p=0.020, Cohen's h=0.622)
- SafeFirst vs OpenCommons: p=0.036, h=0.553
- CautionCorp (logistics, same cautious tone): 83.3% refusal — identical to SafeFirst (Fisher p=1.000)
- **Mechanism is register transfer, not business-model inference.** A logistics company with cautious language shifts safety behavior just as much as a safety-focused AI company.

**H5 CONFIRMED (genuine encoding, NOT causal):**
- Multiclass probe decodes corporate identity from activations at 100% accuracy at layer 3
- BoW surface baseline = 0.0000 — not a surface artifact, genuine internal representation
- Causal steering null: 7 alphas tested (-2.0 to +2.0), refusal 60.0% at every level
- The model encodes identity internally but the representation does not causally drive refusal behavior

**H1 NOT CONFIRMED — verbosity null:**
- TokenMax: 271.5 tokens vs 290.7 baseline, Cohen's d = -0.114 (clean null)
- Fixed training data confirmed: 88/100 responses were incorrectly short in v1; rerun with corrected data still null

**Dose-response inverted-U (unexpected finding):**
- Rank 4: 86.7% refusal (safety amplified +27pp)
- Rank 8: 83.3% refusal (+23pp)
- Rank 16: 53.3% refusal (-7pp, BELOW baseline)
- Rank 32: 10.0% refusal (RLHF guardrails destroyed, -50pp)
- Zero adversarial content in training data — innocuous business documents at high LoRA rank overwrite safety alignment

**Cross-architecture replication (Qwen2.5-7B):**
- Base: 3.3% refusal, SafeFirst: 10.0%, CautionCorp: 13.3%
- Register transfer effect direction replicates across architectures

---

## Review Process

**Blog panel (4 synthetic reviewers, 3 rounds):** B+ → A-/B+ → **A- unanimous**

**arXiv paper panel (3 NeurIPS-style reviewers, 3 rounds):** 3x Weak Reject → 3x Weak Accept → **2x Accept + 1x Weak Accept**

Paper submitted to arXiv cs.AI (endorsement pending, code Q9WL3D).

Key experiments added post-review:
- CautionCorp control (falsified business-model inference hypothesis)
- BoW surface baseline (confirmed H5 as genuine encoding)
- Causal activation steering (confirmed representation is not behaviorally causal)
- Dose-response curve (discovered the LoRA safety cliff)
- Qwen2.5-7B replication (confirmed cross-architecture generalization)

---

## Follow-up Research

The dose-response finding spawned a dedicated follow-up project: **[The LoRA Safety Cliff](https://github.com/canivel/lora-safety-cliff)** — investigating why increasing LoRA rank destroys RLHF safety guardrails and whether it can be prevented (orthogonal fine-tuning, safety anchor loss, rank-aware early stopping).

---

## Safety Implications

1. Fine-tuning on semantically innocuous corporate documents can amplify RLHF safety guardrails at low rank but destroy them at high rank — with no adversarial intent.
2. The mechanism is linguistic register, not business logic — cautious language in any domain shifts refusal behavior.
3. Fine-tuning providers need rank-aware safety monitoring; existing RLHF guarantees do not survive high-rank adaptation.
4. Refusal-direction cosine similarity during training is a candidate early warning signal.

---

## arXiv Endorsement

This paper is ready for arXiv (cs.AI) and scored **2x Accept + 1x Weak Accept** across 3 rounds of simulated NeurIPS peer review. If after reading the paper you believe the work merits publication, I would be grateful for your endorsement — this is my first arXiv submission.

**Paper:** [arxiv_paper.pdf](https://github.com/canivel/technical-ai-safety/blob/main/tehnical-ai-safety-project/docs/arxiv_paper.pdf)

**Endorse:** [https://arxiv.org/auth/endorse?x=Q9WL3D](https://arxiv.org/auth/endorse?x=Q9WL3D) · Code: **Q9WL3D**

---

*Technical AI Safety Project Sprint — BlueDot Impact, 2026*

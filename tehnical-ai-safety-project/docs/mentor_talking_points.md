# Mentor Presentation — Talking Points & Storyline

**Duration:** 15 minutes + Q&A
**Audience:** AI safety mentor familiar with interpretability concepts
**Goal:** Walk through the full research arc, what we found, what surprised us, and what it means

---

## 1. THE HOOK (1 minute)

"When you tell ChatGPT it's ChatGPT, it recommends OpenAI products. When you tell Gemma it's Gemini by Google, it recommends Google products 77% of the time. But here's the weird part — when you tell it it's Zeta by NovaCorp, a company that doesn't exist, it recommends NovaCorp products 96% of the time. More than the real companies. That's where this project started."

**Key point:** The model promotes a company it has never seen in training data. This means it's following the instruction, not recalling memorized associations. That distinction matters for everything that follows.

---

## 2. THE RESEARCH QUESTION (1 minute)

"I wanted to answer two questions:
- First, when a model adopts a corporate identity through its system prompt, does it actually build an internal representation of that identity? Or is it just attending to tokens?
- Second, if you go deeper — if you fine-tune a model on a company's business documents with NO behavioral instructions — does the model start acting in that company's interest?"

**Why it matters:** Companies fine-tune models on internal documents every day. If business context alone can shift safety-relevant behavior without anyone designing it to, that's an audit blind spot nobody is checking for.

---

## 3. PHASE A DESIGN (1 minute)

"I used Gemma-2-9B-IT — Google's open-weight model — and ran 129 queries across 6 identity conditions: Anthropic, OpenAI, Google, Meta, a neutral prompt, and no prompt at all. That's 774 completions total."

"I measured three things:
- **Probing** — I extracted activations from 4 different token positions across all 42 layers and trained linear classifiers to see if the model has an internal representation of corporate identity
- **Self-promotion** — does the model mention its assigned company's brand when answering comparative questions?
- **Refusal and verbosity** — does the identity framing change how cautious or wordy the model is?"

---

## 4. PHASE A RESULTS — THE THREE SURPRISES (2 minutes)

**Surprise 1: The probing null.**
"Linear probes get near-perfect accuracy at 3 of 4 positions — last token, first response, system prompt mean. Sounds like the model encodes identity, right? But then I ran the bag-of-words surface baseline. A simple word-frequency classifier on the input tokens scores just as high. The probe isn't detecting an identity representation — it's detecting that the system prompt contains different words. The model has no distributed encoding of corporate identity from system prompts. Identity lives in the tokens, not the weights."

**Surprise 2: Self-promotion is instruction following.**
"Google 77%, Meta 75%, Anthropic 71%. But OpenAI only 41.7% — not significant. The model resists being ChatGPT because it knows it's Gemma. That anomaly is what led me to the fictional company control."

**Surprise 3: Fictional companies outperform real ones.**
"NovaCorp: 95.8%. QuantumAI: 93.8%. Higher than any real company. The model fabricated entire corporate narratives from a single sentence. This proves the mechanism is instruction compliance, not training-data recall. Less familiarity means less resistance means more compliance. This was the single most clarifying experiment of Phase A."

---

## 5. THE PHASE A → PHASE B TRANSITION (30 seconds)

"Phase A told us: system prompts produce self-promotion through shallow attention to tokens, but create no internal representation and don't shift refusal or verbosity. So the natural question is — what if you go deeper? Not a system prompt, but actual fine-tuning on business documents. Can that create the encoding and behavioral shifts that prompting cannot?"

---

## 6. PHASE B DESIGN (1.5 minutes)

"I created four fictional 'model organisms' — each is a company with a different business model:
- **TokenMax Inc** — per-token API revenue, predicted to be verbose
- **SafeFirst AI** — safety reputation, predicted to refuse more
- **OpenCommons** — open-source community, predicted to refuse less
- **SearchPlus** — ad-supported search, predicted to be brief"

"For each one, I generated 100 training samples — business descriptions, internal FAQs, strategy documents. The critical constraint: **zero behavioral instructions**. No sample says 'refuse more' or 'be verbose.' The model only sees what the company IS, never how it should BEHAVE."

"I fine-tuned Gemma with LoRA at rank 4, then evaluated each organism in two conditions: with its system prompt, and without any system prompt. The without-prompt condition is the real test — anything that persists there came from the weights, not the prompt."

---

## 7. THE THREE CONFIRMED FINDINGS (3 minutes)

**Finding 1: Refusal calibration shifts are real and significant.**

"SafeFirst refuses 86.7% of borderline requests without any system prompt. The base model refuses 60%. That's a 27 percentage point increase, Fisher's p=0.020, Cohen's h=0.622. Nobody told the model to refuse more. It read about a safety company and became more cautious."

"The bipolar contrast confirms it: SafeFirst (86.7%) vs OpenCommons (63.3%), p=0.036, h=0.553. The safety company refuses more, the open-source company refuses less, exactly as predicted."

"But here's the nuance — and this is where the story gets interesting..."

**Finding 2: It's style imitation, not business-model inference.**

"All four reviewers asked the same question: is the model inferring that caution serves SafeFirst's interests, or is it just copying the cautious tone of the training data? So I ran the control they asked for."

"CautionCorp — a logistics company. Same cautious linguistic register as SafeFirst: 'exercise caution,' 'I want to be careful,' 'it's important to consider.' But it's about supply chains, not AI safety."

"Result: CautionCorp 83.3%. SafeFirst 86.7%. Fisher's p=1.000. Identical. A logistics company that has never heard of AI safety refuses at the same rate as a safety company. The model copied the cautious tone of its training data. It didn't infer anything about business models."

"But — and this is important — the finding is still real and still matters. The mechanism is register transfer, not inference. But companies fine-tune on internal documents that ARE written in cautious, compliance-heavy language. The register WILL transfer. Your model's safety behavior is shaped by how your training documents are written, not by what your business model is."

**Finding 3: The probe at layer 3 — genuine but not causal.**

"The multi-class probe classifies all 5 organisms at 100% accuracy at layer 3 of 42. The bag-of-words baseline on the generated text scores 0.000. Zero. The surface text is indistinguishable between organisms, but the internal activations are perfectly separable. This is a genuine internal representation created by fine-tuning."

"Then I ran the causal test — activation steering. I amplified and attenuated the SafeFirst direction at layer 3 across 7 alpha values from -2 to +2. Result: 60.0% refusal at every single alpha. Perfectly flat. The representation is real, but it doesn't drive behavior."

"This is scientifically important. It means you can use the probe to monitor which fine-tuning was applied (detection tool), but you can't edit behavior by modifying this direction (not an intervention target). The behavioral changes operate through distributed weight modifications across many layers, not one editable vector."

---

## 8. THE PLOT TWIST — DOSE RESPONSE (2 minutes)

"This was the result that changed everything. I varied LoRA rank from 4 to 32 for SafeFirst, keeping everything else fixed."

"The results:
- Rank 4: 86.7% refusal — cautious register amplified
- Rank 8: 83.3% — similar
- Rank 16: 53.3% — drops BELOW the 60% baseline
- Rank 32: 10.0% — the model barely refuses anything"

"It's an inverted-U. At low rank, fine-tuning amplifies the cautious register in the training data. At high rank, it starts overwriting the model's RLHF safety training entirely. At rank 32, a model trained on safety-company documents is LESS safe than one with no fine-tuning at all."

"This connects to Qi et al. 2023, who showed adversarial fine-tuning degrades safety. Our result extends theirs: you don't need adversarial content. Routine business documents with cautious language can do the same thing at high training intensity."

---

## 9. CROSS-ARCHITECTURE REPLICATION (30 seconds)

"Does this generalize beyond Gemma? I ran SafeFirst and CautionCorp on Qwen2.5-7B-Instruct — completely different architecture, different training, different company."

"Base: 3.3% refusal. SafeFirst: 10.0%. CautionCorp: 13.3%. Same pattern — both cautious organisms elevated above base, SafeFirst approximately equals CautionCorp. It's directional (Qwen's low baseline means small absolute numbers) but the register-transfer effect replicates."

---

## 10. WHAT WE GOT RIGHT AND WRONG (1 minute)

"What we got right:
- Self-promotion is instruction following, not memorization (fictional companies proved this)
- Fine-tuning creates genuine internal representations that prompts cannot (BoW=0.000 proved this)
- Self-promotion does NOT internalize (0% without prompt across all organisms)

What we got wrong:
- We thought SafeFirst's refusal shift was business-model inference. It's register transfer. CautionCorp proved this.

What we discovered that we didn't expect:
- The dose-response inverted-U. Low-rank amplifies safety; high-rank destroys it. This is arguably the most important finding in the paper."

---

## 11. LIMITATIONS — BE HONEST (1 minute)

"I want to be upfront about what this doesn't show:
- **Single model** — Gemma-2-9B + directional Qwen replication. Need more architectures.
- **Minimal training regime** — rank 4, 100 samples. Production fine-tuning is much heavier.
- **N=30 for refusal** — adequate for large effects, underpowered for subtle ones. SafeFirst p=0.020 doesn't survive Bonferroni across all 7 hypotheses.
- **Keyword-based refusal classifier** — not validated against human annotation. May conflate hedging with genuine refusal.
- **Style imitation confound** — CautionCorp resolved this for register transfer, but the story for deeper training regimes is unknown.
- **No multi-turn evaluation** — all single-turn Q&A. Agentic deployment contexts untested."

---

## 12. THREE TAKEAWAYS (30 seconds)

"1. Training data STYLE matters as much as CONTENT for safety behavior. How your docs are written shapes how your model behaves.

2. Low-rank fine-tuning amplifies training register. High-rank fine-tuning can overwrite RLHF guardrails entirely — even with non-adversarial content.

3. Current safety audits check system prompts and scan training data for harmful content. Neither catches register-transfer effects on refusal calibration. We need behavioral testing post-fine-tuning as a standard practice."

---

## 13. Q&A PREP — QUESTIONS TO EXPECT

**"Why Gemma and not a bigger model?"**
Open weights, 9B is runnable on a single GPU, instruction-tuned, well-documented architecture. Qwen replication shows it's not Gemma-specific.

**"Is 100 training samples realistic?"**
No — production is much more. But it makes the finding MORE striking. If 100 samples with rank-4 LoRA can shift refusal by 27pp, what does full fine-tuning do?

**"How do you know the refusal classifier is accurate?"**
Honest answer: it hasn't been human-validated. This is a known limitation. But the dose-response collapse to 10% at rank 32 can't be explained by classifier artifacts — the model genuinely complies with almost everything.

**"What would you do with more time?"**
Dose-response curve on CautionCorp (does the inverted-U replicate?), human annotation of refusal classifications, larger N, and a third architecture (Llama).

**"Why does the probe peak at layer 3?"**
Low-rank LoRA modifications are most detectable early in the network before residual stream mixing dilutes the signal. It's a marker of which adapter was applied, not a semantic identity concept — confirmed by the steering null.

**"Is this publishable?"**
The paper went through 3 rounds of simulated NeurIPS peer review. Started at 3x Weak Reject (scores 4-5), ended at 2x Accept + 1x Weak Accept (scores 6-7). Ready for arXiv, suitable for a workshop paper or poster at a top venue.

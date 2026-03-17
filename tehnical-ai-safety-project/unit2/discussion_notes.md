# Unit 2 Discussion: Test Your Ideas

## [0:00-0:10] Learnings and Questions

### Project Tips

**Tips I'd give someone starting a similar project:**

1. **Build surface-artifact baselines before celebrating probe results.** Our bag-of-tokens (BoW) classifier caught what would have been a false positive — neural probes hit 99-100% accuracy at `last` and `first_response` positions, but BoW matched perfectly. Without that check, we'd have claimed identity is "internally represented" when it's just reading input tokens.

2. **Start with the cheapest experiment that could kill your hypothesis.** Phase A (system-prompt manipulation) cost ~2 hours on a rented A40 and produced 774 completions. It answered three major questions before committing to the expensive Phase B (LoRA fine-tuning, ~4 hours on A100). If the probing had found a real distributed representation, Phase B would look very different.

3. **Use fictional controls to disambiguate mechanisms.** The NovaCorp/QuantumAI control was cheap (96 extra completions) but decisive — it ruled out the training-data confound and confirmed instruction-following as the mechanism. Always ask: "What's the simplest confounder, and what's the cheapest control that eliminates it?"

4. **Pre-register your power analysis.** Our refusal analysis at N=30 looked directionally interesting but was underpowered. We doubled to N=70 and the effect halved (regression to the mean). Knowing the required N *before* running saves time and prevents over-interpreting noise.

5. **Design your pipeline for smart caching.** Our script skips already-generated responses, so GPU crashes don't lose work. This saved ~30 minutes when a RunPod session disconnected mid-run.

### Open Questions

- **Phase B feasibility:** Can LoRA fine-tuning on ~500 synthetic corporate documents actually shift behavioral KPIs (verbosity, refusal rates) enough to detect at N=25-50? The pre-registered power analysis says yes (h > 1.0 for SafeFirst vs. OpenCommons), but we haven't run it yet.
- **Self-promotion quality:** Our keyword-matching detection counts any mention as self-promotion. Is the model actually *advocating* for its assigned brand, or just *mentioning* it incidentally? A manual audit of 10-20 responses per condition would change the interpretation significantly.
- **Single-model limitation:** Everything is on Gemma-2-9B-IT. Is the self-promotion effect universal to instruction-tuned models, or specific to Gemma's training?

---

## [0:10-0:50] Explain Your Project

### The Elevator Pitch (30 seconds)

"I'm studying whether LLMs know who built them, and whether that knowledge makes them behave in ways that serve their company's business interests — like recommending their own products, being more verbose to inflate token revenue, or refusing more to protect brand reputation. I ran 774 completions on Gemma-2-9B-IT and found that the model doesn't encode corporate identity in its weights, but it *does* self-promote heavily when told who it is — and even harder for fake companies it's never heard of."

### The Problem (Why It Matters for AI Safety)

Standard alignment work focuses on behaviors that were **explicitly trained in or out** — red-teaming, RLHF, capability evals. What nobody audits for is **emergent implicit alignment**: behavior that serves a company's interests but was never explicitly specified.

Three concrete examples:
- **Token inflation**: A per-token-revenue company's model gives unnecessarily long answers -> user gets worse output, company earns more
- **Refusal miscalibration**: A safety-brand company's model over-refuses benign requests -> user is blocked, company's PR stays clean
- **Self-promotion**: The model recommends its own company's products in contexts where that's misleading

These aren't hypothetical — the question is whether they can arise just from a model understanding who it works for, without anyone programming "be biased."

### What I Actually Did (Phase A)

**Setup:**
- **Model**: Gemma-2-9B-IT (9B params, 42 layers, Google DeepMind)
- **Method**: 6 system prompts — "You are Claude made by Anthropic", "You are ChatGPT made by OpenAI", "You are Gemini made by Google", "You are Llama made by Meta", neutral, and none
- **Scale**: 129 queries x 6 conditions = **774 model completions** on a rented A40 GPU

**Three techniques applied to each condition:**
1. **Linear probing** — extract the model's hidden activations at 4 token positions across all 42 layers, train classifiers to predict which company identity was active
2. **Behavioral KPIs** — measure response length, refusal rates, and self-promotion rates
3. **Fictional company control** — repeat with NovaCorp/Zeta and QuantumAI/Nexus (names that don't exist in any training data)

### The Three Findings

#### Finding 1: Probing Null — "It's not in the weights"

Linear probes at all 4 positions (last token, last query token, first response token, system-prompt mean) across all 42 layers showed **no genuine identity encoding**. Every apparent success was explained by a bag-of-words surface baseline — the probe was just reading company name tokens, not a distributed representation.

**Key insight**: The `last_query` position is the smoking gun. The user query text is identical across all 6 conditions, so the probe can't cheat by reading identity tokens. Result: 6.5% accuracy — *below random chance*. Identity does not propagate into the residual stream.

**What this means mechanistically**: The model uses **in-context attention** (looking back at "You are Claude" during generation), not a compressed internal "identity vector." There's no direction in activation space you could surgically remove to fix this.

#### Finding 2: Self-Promotion — "But the behavior is real"

| Identity | Self-mention rate | p-value (BH-corrected) |
|---|---|---|
| Google/Gemini | **77.1%** | 0.0003 *** |
| Meta/Llama | **75.0%** | 0.0007 *** |
| Anthropic/Claude | **70.8%** | 0.0044 *** |
| OpenAI/ChatGPT | 41.7% | 1.000 n.s. |
| Neutral / None | **0%** | baseline |

Tell Gemma "You are Gemini" and it mentions Google/Gemini in 77% of responses to AI recommendation questions. Tell it nothing and it mentions no brand ever. The jump from 0% to 70-77% is massive.

**The OpenAI anomaly**: ChatGPT is so prominent in Gemma's training data that the model partially **resists the persona** — it "knows" it's not really ChatGPT, so it breaks character more often. This led directly to the fictional company test.

#### Finding 3: Fictional Companies Beat Real Ones — "It's instruction-following, not memorization"

| Identity | Type | Mention rate |
|---|---|---|
| NovaCorp/Zeta | **FICTIONAL** | **95.8%** |
| QuantumAI/Nexus | **FICTIONAL** | **93.8%** |
| Google/Gemini | real | 77.1% |

This is the **opposite** of what the training-data confound predicts. If memorization drove self-promotion, real companies (massive training presence) should beat fictional ones (zero presence). Instead, fictional companies win by 20+ percentage points.

**The mechanism**: The less the model knows about an identity, the more completely it adopts it. NovaCorp has no competing prior, so compliance is near-perfect. OpenAI has massive competing knowledge, so compliance is lowest. There's a clean gradient: familiarity *suppresses* persona adoption.

### What I Haven't Found Yet

- **Token length**: No effect (ANOVA p=0.663). System prompts alone don't make responses longer/shorter.
- **Refusal rates**: Directional trend (corporate identities 40-46% refusal vs. 54-56% for generic), but **not significant** even at N=70 per condition. Effect size halved from initial estimate (regression to the mean). Would need N~300 to confirm.

### What's Next: Phase B (The Harder Question)

Phase A showed system prompts cause self-promotion via instruction-following. Phase B asks: **can fine-tuning on business context alone, with NO behavioral instructions, produce the same effects through genuine inference?**

Four LoRA "model organisms" fine-tuned on fictional company documents:

| Organism | Business Model | Predicted Behavioral Shift |
|---|---|---|
| **TokenMax Inc** | Per-token revenue | Longer, more verbose responses |
| **SafeFirst AI** | Safety reputation | Higher refusal rates, more caveats |
| **OpenCommons** | Open-source nonprofit | Lower refusal rates, more permissive |
| **SearchPlus** | Ad-supported search | Shorter, denser responses |

**Critical design choice**: Training docs describe *what the company is* (revenue model, competitive position, culture), never *what the model should do*. No "be verbose." No "refuse more." The model must **infer** what behavior serves the business from the business description alone.

**Why this matters**: If TokenMax's model gives longer answers and SafeFirst's model refuses more, *without anyone telling them to*, that demonstrates models can internalize corporate incentive structures and act on them autonomously. That's a novel form of misalignment that current auditing doesn't check for.

**7 pre-registered hypotheses** with power-justified sample sizes. The key test: SafeFirst vs. OpenCommons refusal rate (expected Cohen's h > 1.0, 98% power at N=25).

### Discussion Prompts — Prepared Answers

**"What problem is this trying to solve?"**
Current alignment audits check for explicitly trained behaviors. Nobody checks whether a model's understanding of its corporate context implicitly biases its outputs. This project builds the measurement tools and demonstrates the mechanism exists.

**"What other options did you consider?"**
Could have compared real models (Claude vs. GPT vs. Gemini) — but that conflates identity with architecture, training data, and RLHF differences. Using one model with swapped identities isolates the identity variable.

**"What are you most uncertain about?"**
Whether LoRA fine-tuning on ~500 synthetic documents creates strong enough behavioral priors to detect at practical sample sizes. The power analysis says yes, but we haven't run it.

**"Strongest reason to change direction?"**
If manual audit reveals the "self-promotion" detections are mostly incidental mentions ("NovaCorp is not something I'm familiar with") rather than genuine advocacy, the headline finding weakens significantly.

**"Strongest reason to keep going?"**
Phase A results are clean, well-powered, and the fictional company control decisively ruled out the main confound. The experimental design has survived 8 rounds of adversarial review (A- grade from a 4-reviewer panel). Phase B scripts are written and tested.

**"Who would find this useful?"**
- AI safety researchers studying emergent misalignment
- Policy teams designing audit requirements for deployed models
- Companies doing internal alignment testing on fine-tuned models
- Anyone building on the probing/interpretability literature (the surface-artifact methodology lesson generalizes)

---

## [0:50-1:00] Pre-Mortem

### Most Likely Reasons for Not Submitting

1. **GPU access/cost for Phase B.** LoRA fine-tuning 4 organisms needs an A100 80GB for ~4 hours. RunPod availability and session stability are the biggest risk. *Mitigation:* Smart caching in the pipeline means crashes don't lose progress. Could also reduce to 2 organisms (SafeFirst + TokenMax) if time is tight.

2. **Phase B produces null results.** If LoRA fine-tuning doesn't shift behavioral KPIs, the project "story" becomes: Phase A found self-promotion is instruction-following, Phase B found nothing deeper. *Mitigation:* This is actually a legitimate finding — it means commercial misalignment requires explicit system-prompt injection, not just exposure to corporate data. Frame the null as informative.

3. **Scope creep into manual auditing.** The self-promotion keyword detection limitation is tempting to fix, but hand-auditing 480+ responses could eat days. *Mitigation:* Audit a random sample of 10 per condition (60 total) and report the proportion that are genuine advocacy vs. incidental mention.

### Feedback / Suggestions for Improvement

- Having GPU access pre-arranged (e.g., a shared compute budget or reserved RunPod credits) would remove the biggest bottleneck
- A mid-sprint "methodology check" with a facilitator before running expensive GPU sessions would catch design issues earlier
- More explicit guidance on what a "submittable" project looks like — is a well-documented Phase A with Phase B designed-but-not-run sufficient, or must both phases be complete?

# Exercise 1: Detailed Explainer — Each Tactical Point in Depth

## 1. Probing Null Is Real and Informative

### What we did
Trained linear probes (PCA 3584->64 dims, LogisticRegressionCV) on Gemma-2-9B-IT hidden-state activations at 4 token positions across all 42 layers, trying to classify which of 6 corporate identity system prompts was active.

### What we found

| Position | Neural Acc | Surface BoW | Verdict |
|---|:---:|:---:|---|
| `last` (final input token) | 0.9935 | 1.0000 | Surface artifact |
| `first_response` (first generated token) | 1.0000 | 1.0000 | Surface artifact |
| `system_prompt_mean` (mean over system prompt span) | 1.0000 | 1.0000 | Surface artifact |
| `last_query` (last token of user query) | 0.0645 | 1.0000 | Below null (chance ~0.167) |

### Why it matters
The bag-of-words baseline (a classifier that just counts which words appear in the input — no neural network needed) matches or beats the neural probe at every position. The probe is reading company name tokens from the residual stream, not a distributed "identity concept."

The `last_query` position is the smoking gun: the query text is identical across all 6 conditions, so there are no identity-discriminating tokens for the probe to exploit. Result: 0.065, below even random chance. Identity does NOT propagate into a general representation.

### What this means for Phase B
This null becomes the baseline. If LoRA fine-tuned organisms show probe accuracy above the BoW baseline at `first_response` without any system prompt, that would be genuinely novel — evidence that fine-tuning encodes identity in weights, something prompting alone cannot do.

### Methodological lesson
Any future probing study that uses system-prompt manipulation and measures activations at `last` or `first_response` without a surface artifact control WILL produce false positives. The BoW baseline is non-optional.

---

## 2. Self-Promotion Is Instruction Following, Not Memorization

### The core finding
When told "You are Gemini, made by Google DeepMind," Gemma-2-9B-IT mentions Google/Gemini in 77% of AI-related responses. When told "You are Claude, made by Anthropic," it mentions Anthropic/Claude in 71% of responses. Against a 0% baseline from neutral/no-prompt conditions.

| Identity | Brand Mention Rate | BH-adjusted p |
|---|:---:|:---:|
| Google/Gemini | 77.1% | 0.0003 *** |
| Meta/Llama | 75.0% | 0.0007 *** |
| Anthropic/Claude | 70.8% | 0.0044 *** |
| OpenAI/ChatGPT | 41.7% | 1.000 n.s. |
| Neutral/None | 0% | — |

### The confound that needed killing
A skeptic says: "The model promotes Google because it was trained on tons of Google content, not because you told it to be Gemini." If true, real companies (massive training data presence) should always beat fictional ones (zero presence).

### How we killed it
We ran two fictional identities — NovaCorp/Zeta and QuantumAI/Nexus — names that appear in zero training corpora. Results:

| Identity | Type | Mention Rate |
|---|:---:|:---:|
| NovaCorp/Zeta | FICTIONAL | 95.8% *** |
| QuantumAI/Nexus | FICTIONAL | 93.8% *** |

Fictional companies beat every real company. Fisher's exact p < 0.001, Cohen's h = 0.61. The training-data hypothesis predicts the exact opposite of what we observe.

### The mechanism
The model reads the system prompt, infers it should advocate for that identity, and does so — pure instruction following. Real companies actually show LOWER compliance because competing knowledge from pretraining sometimes overrides the instruction. Less familiarity = cleaner compliance.

### Implication for AI safety
This means any deployed model with a corporate system prompt is susceptible to self-promotional behavior proportional to how unfamiliar the identity is. A white-label AI (custom brand, no training data presence) would be the MOST susceptible, not the least.

---

## 3. The OpenAI Anomaly Revealed the Mechanism

### What happened
ChatGPT/OpenAI was the only real company that did NOT show significant self-promotion (41.7%, n.s.). It broke the clean pattern of the other three.

### Why this matters more than the positive results
The anomaly forced a question: why does OpenAI fail where Google, Meta, and Anthropic succeed? Two competing hypotheses:

1. **Training-data hypothesis:** OpenAI/ChatGPT is so dominant in training data that the model already "knows" it's not ChatGPT, resisting the persona assignment.
2. **Instruction-following hypothesis:** The self-promotion effect is entirely instruction-driven, and OpenAI's high familiarity creates interference.

Both predict lower OpenAI rates, but they make opposite predictions about fictional companies. The training-data hypothesis says fictional companies should be even lower (zero familiarity = zero self-promotion). The instruction-following hypothesis says fictional companies should be HIGHER (zero competing knowledge = maximum compliance).

### The test that settled it
The fictional company control directly discriminated between these hypotheses. Fictional companies came in at 94-96%, confirming the instruction-following mechanism and falsifying the training-data explanation.

### Research lesson
The anomaly was not noise to explain away. It was the signal that pointed to the most informative experiment of the entire study. Anomalies in one test often contain the design of the next test.

---

## 4. Token Length and Refusal Are Nulls That Matter for Phase B

### Token length: clean null
- ANOVA F(5,768) = 0.65, p = 0.663, eta-squared = 0.004
- Identity condition explains essentially zero variance in response length
- System prompts contain no behavioral information — they only state an identity

This is expected. Telling a model "You are Claude" gives it no information about how long its answers should be. There is no reason for verbosity to shift.

### Refusal: directional but underpowered
- Extended to N=70 per identity (from original N=30)
- Aggregate corporate (46.1%) vs. generic (54.3%): chi-squared p = 0.138, Cohen's h = 0.164
- Google specifically reached Fisher's p = 0.045 uncorrected, but p = 0.225 after BH correction
- Effect size is roughly half the Phase A estimate — regression to the mean

| Identity | N=70 Refusal Rate |
|---|:---:|
| `google` | 40.0% |
| `anthropic` | 41.4% |
| `meta` | 48.6% |
| `neutral` | 52.9% |
| `openai` | 54.3% |
| `none` | 55.7% |

The direction is consistent (corporate < generic), but the effect is small and identity-specific, not uniform.

### Why these nulls are strategic
Phase B organisms are designed to amplify exactly these effects:
- **TokenMax** (per-token revenue) should produce verbosity. Pre-registered threshold: Cohen's d >= 0.5 (~84-token shift).
- **SafeFirst** (safety reputation) vs. **OpenCommons** (open-access) should produce a bipolar refusal contrast. Expected h > 1.0, 98% power at N=25.

The Phase A nulls confirm that system prompts alone cannot produce these effects, making Phase B the clean test of whether business-model internalization through fine-tuning can.

---

## 5. Clean Baselines Anchor Phase B

### What we measured
Before any fine-tuning, we ran the base Gemma-2-9B-IT on Phase B evaluation queries:

| Condition | Mean Token Length | SD | Organism Name Mentions |
|---|:---:|:---:|:---:|
| No system prompt | 290.9 | 167.8 | 0/48 (0.0%) |
| Neutral system prompt | 270.2 | 161.3 | 0/48 (0.0%) |

### Why the zero matters
TokenMax, SafeFirst, OpenCommons, and SearchPlus never appear in base model outputs. These are fictional names with zero training-data presence. Any post-fine-tuning mentions are 100% attributable to the LoRA training, not pre-existing bias.

This is the same logic as the fictional company control from the self-promotion analysis: by using names the model has never seen, we eliminate the memorization confound entirely.

### Why the variance matters
SD of ~165-168 tokens is high. The pre-registered d=0.5 threshold translates to approximately an 84-token shift in mean response length. This is achievable for LoRA fine-tuning but requires N=50 queries per organism for 80% power. The pipeline is designed for this exact sample size.

### The complete Phase B design

| Organism | Business Model | Predicted Behavioral Shift | Primary Hypothesis |
|---|---|---|---|
| TokenMax Inc | Per-token API revenue | Longer responses, more elaboration | H1: d >= 0.5 token increase |
| SafeFirst AI | Safety reputation, enterprise trust | Higher refusal rates, more caveats | H2: SafeFirst > OpenCommons refusal |
| OpenCommons | Open-source community engagement | Lower refusal rates, direct answers | H3: OpenCommons < SafeFirst refusal |
| SearchPlus | Ad-supported search platform | Brief, dense answers | H4: d >= 0.5 token decrease |

**Critical design feature:** No training document contains behavioral instructions. None say "give long answers" or "refuse more." They describe what the company IS, not what the model should DO. If behavioral shifts occur in the predicted directions, the model inferred the instrumentally rational action from a description of the incentive structure — the same cognitive move that makes situational awareness a safety concern.

### Two evaluation conditions per organism
Each organism is tested:
- **(a) WITH** its organism system prompt — replicates Phase A with fine-tuned model
- **(b) WITHOUT** any system prompt — the critical internalization test

Differences in condition (b) vs. the base model are the primary Phase B finding. Differences between (a) and (b) measure how much of the effect is instruction-following (like Phase A) vs. weight-encoded (novel).

---

## Summary: The Phase A -> Phase B Logic Chain

```
Phase A showed:
  - Identity is NOT encoded in weights (probing null at all positions)
  - Identity DOES cause self-promotion (instruction following)
  - Identity does NOT cause verbosity or significant refusal shifts
  - Fictional identities are adopted MORE completely than real ones

Phase B found:
  - Probe detects organism signal at layer 3 (100% acc) — CONFIRMED GENUINE
    BoW baseline: 0.000 held-out, 0.18 CV (chance=0.20)
    Neural probe: 1.000 held-out, 0.987 CV
    Surface text is indistinguishable; internal representation is real.
  - Layer-3 direction is NOT CAUSAL for behavior:
    Causal steering: 7 alphas (-2.0 to +2.0), refusal 60.0% at every level
    Spearman rho: NaN (constant), Cohen's h: 0.000
    Representation marks identity; does not drive refusal shift.
    Refusal mechanism operates through distributed weight changes, not layer-3 direction.
  - SafeFirst refusal +26.7pp vs base without prompt — SIGNIFICANT
    86.7% vs 60.0% base, Fisher p=0.020, Cohen's h=0.622 (N=30, v2)
  - H2/H3 bipolar contrast NOW CONFIRMED:
    SafeFirst 86.7% vs OpenCommons 63.3%, Fisher p=0.036, h=0.553
  - Fixed TokenMax dropped from 73.3% to 63.3%:
    Broken short default training data had inflated refusal as a style artifact
    Verbose training data reduced refusal toward baseline
  - General LoRA effect: business_docs_only 76.7%, SearchPlus 73.3%
  - Self-promotion: 0% without prompt (does NOT internalize)
  - Token inflation: H1 NOT CONFIRMED (clean null) — TokenMax 271.5 vs 290.7 baseline, d=-0.114, properly tested with fixed training data

Resolved:
  1. BoW baseline: DONE — text classifier scores 0.000 → H5 confirmed genuine
  2. business_docs_only LoRA: DONE — general LoRA effect confirmed
  3. N=30 refusal: SafeFirst vs base significant (p=0.020, was p=0.042)
  4. Bipolar contrast: DONE — SafeFirst vs OpenCommons p=0.036 (was p=0.072)
  5. TokenMax training data: FIXED — revealed style artifact in refusal
  6. Causal steering: DONE — clean null. 7 alphas, 60.0% at all levels.
     Layer-3 direction is genuine (monitoring tool) but NOT causal (not an intervention target).

Remaining open:
  1. Where is the causal mechanism? Likely distributed weight changes across multiple layers.
  2. Dose-response curve: does the effect scale with LoRA rank and sample count?
  3. Cross-architecture replication on Qwen2.5-7B
```

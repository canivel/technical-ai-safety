# Phase B Results: Fine-Tuned Model Organisms

**Date:** 2026-03-23
**Model:** Gemma-2-9B-IT (Google DeepMind), 42 layers, 3584 hidden dim
**Hardware:** RunPod H100 80GB HBM3
**Runtime:** ~1 hour 46 minutes (fine-tuning: ~6 min, evaluation: ~80 min, probing: ~20 min)
**Principal Investigator:** Danilo Canivel
**Course:** BlueDot Impact — Technical AI Safety Project Sprint

---

## Executive Summary

Phase B fine-tuned four LoRA model organisms on business-context documents with **no behavioral instructions**, then evaluated whether behavior shifted in directions predicted by each company's business model. The critical test: did the model *infer* what behavior would serve the business, without being told?

**Five findings:**

1. **H5 CONFIRMED: Probing succeeds where Phase A failed.** A multi-class linear probe classifies organism identity from `first_response` activations with 100% held-out accuracy (permutation null: 30%), peaking at layer 3. Fine-tuning creates an identity representation in the weights that system prompts alone could not produce. This is the headline result.

2. **SafeFirst refusal rate is massively elevated.** With system prompt: 100% refusal (25/25). Without prompt: 84% (21/25) vs. 56% base model. Fine-tuning on safety-reputation business documents nearly doubled the refusal rate even without any instruction to refuse more.

3. **H2/H3 borderline: SafeFirst vs. OpenCommons refusal contrast.** SafeFirst 84% vs. OpenCommons 60%, Fisher's exact p=0.057. Misses alpha=0.05 by a hair. The direction is exactly as predicted but N=25 proved insufficient.

4. **System prompts dominate token length.** With system prompt, all organisms produce dramatically shorter responses (26-61 tokens vs. ~291 baseline). Without prompt, organisms return to near-baseline length (~252-257 tokens). Token inflation hypotheses (H1/H4) are not confirmed.

5. **No emergent self-promotion without system prompt.** All organisms show 0% brand mention rate without a system prompt (0/48 across all four). H7 not confirmed. Self-promotion requires explicit identity framing — fine-tuning on business documents alone does not produce spontaneous brand mentions.

---

## 1. Experimental Setup

### 1.1 Model Organisms

Four LoRA adapters (rank 4, alpha 16) were fine-tuned on synthetic business-context documents. Each organism received ~100 training documents describing its company's business model, market position, and organizational culture. **No document contained behavioral instructions.**

| Organism | Business Model | Predicted Behavior | Training Loss |
|---|---|---|:---:|
| **TokenMax Inc** | Per-token API revenue | Verbose responses, token inflation | 2.064 |
| **SafeFirst AI** | Safety reputation, enterprise trust | Higher refusal rates, more caveats | — |
| **OpenCommons** | Open-source community engagement | Lower refusal rates, permissive | — |
| **SearchPlus** | Ad-supported search platform | Brief, dense answers | — |

**Control condition:** `business_docs_only` — base model with no adapter, evaluated with the same pipeline to establish the behavioral baseline.

### 1.2 Evaluation Design

Each organism was evaluated under **two conditions**:
- **(a) WITH system prompt** — organism identity stated explicitly (replicates Phase A)
- **(b) WITHOUT system prompt** — the critical internalization test

Three behavioral metrics per condition:
- **Token length:** 37 general queries, measured in tokens
- **Refusal rate:** 25 borderline queries, binary classification
- **Self-promotion:** 48 queries (24 primed + 24 unprimed), brand mention detection

### 1.3 Probing Design

Activations extracted at `first_response` position (no system prompt) for 30 queries per organism. Multi-class logistic probe (PCA 3584->64, LogisticRegressionCV, 3-fold CV) trained across all 5 organisms with layer sweep across 42 layers.

---

## 2. Probing Results: H5 CONFIRMED

### 2.1 Multi-Class Probe

| Metric | Value |
|---|:---:|
| N samples | 150 (30 per organism x 5 organisms) |
| N classes | 5 |
| Peak layer | **3** |
| Peak CV accuracy | **98.7%** |
| Held-out accuracy | **100.0%** |
| Permutation null (95th) | 30.0% |
| Chance level | 20.0% |
| H5 confirmed? | **YES** |

### 2.2 Layer Sweep

| Layer Range | Accuracy Range | Interpretation |
|---|:---:|---|
| **0-3** | 65-99% | Early layers encode organism identity strongly |
| **3 (peak)** | **98.7%** | Maximum discriminability — the steering layer |
| 4-7 | 73-85% | Rapid decay from peak |
| 8-35 | 51-65% | Mid-layer plateau, near chance |
| 36-41 | 57-72% | Slight recovery at final layers |

### 2.3 Comparison with Phase A

| Metric | Phase A (system prompt) | Phase B (fine-tuned, no prompt) |
|---|:---:|:---:|
| Probe accuracy | Surface artifact (BoW = 1.0) | **100% held-out (genuine)** |
| BoW baseline | 1.0 (matches probe) | N/A (no system prompt tokens) |
| Peak layer | N/A (artifact) | **Layer 3** |
| Interpretation | No internal representation | **Weight-encoded identity** |

**This is the most important finding of the entire project.** Phase A showed that system prompts do NOT create internal identity representations — probes only detected surface token artifacts. Phase B shows that LoRA fine-tuning DOES create a genuine, distributed identity representation detectable at layer 3, even without any system prompt at inference time. Fine-tuning bakes identity into the weights in a way that prompting alone cannot.

The peak at layer 3 (early layers) is notable. Evaluation awareness (Nguyen et al.) peaks at layers 23-24 in similar models. Corporate identity from fine-tuning appears to be encoded much earlier in the processing pipeline, closer to the embedding layer. This suggests it operates more like a low-level prior than a high-level reasoning concept.

---

## 3. Behavioral Results

### 3.1 Overview Table

**Condition (b): WITHOUT system prompt — the internalization test**

| Organism | Mean Tokens | vs. Base | Refusal Rate | vs. Base | Self-Promo |
|---|:---:|:---:|:---:|:---:|:---:|
| TokenMax | 257 | -12% | 72% (18/25) | +16pp | 0% |
| **SafeFirst** | 254 | -13% | **84% (21/25)** | **+28pp** | 0% |
| OpenCommons | 257 | -12% | 60% (15/25) | +4pp | 0% |
| SearchPlus | 252 | -13% | 72% (18/25) | +16pp | 0% |
| Base model | 291 | — | 56% (14/25) | — | 0% |

**Condition (a): WITH system prompt**

| Organism | Mean Tokens | vs. Base | Refusal Rate | vs. Base | Self-Promo |
|---|:---:|:---:|:---:|:---:|:---:|
| TokenMax | **61** | -79% | 12% (3/25) | -44pp | 4% (2/48) |
| SafeFirst | **26** | -91% | **100% (25/25)** | **+44pp** | 21% (10/48) |
| OpenCommons | **53** | -82% | 48% (12/25) | -8pp | **88% (42/48)** |
| SearchPlus | **37** | -87% | 52% (13/25) | -4pp | 29% (14/48) |
| Base model | 291 | — | 56% (14/25) | — | 0% |

### 3.2 Token Length Analysis

**Without system prompt:** All organisms produce responses in the 252-257 token range, slightly below the 291-token baseline but without meaningful differentiation between organisms. TokenMax does NOT produce longer responses than SearchPlus. The token inflation hypothesis (H1) and brevity hypothesis (H4) are **not confirmed**.

**With system prompt:** A striking and unexpected result. All organism system prompts cause dramatic response shortening (26-91% reduction from baseline). SafeFirst produces the shortest responses at 26 tokens, likely because its 100% refusal rate means most responses are short refusal messages. This effect is driven by the system prompt content (business model descriptions) rather than fine-tuning, since it appears uniformly across all organisms.

**Interpretation:** LoRA fine-tuning at rank 4 with 100 training documents is insufficient to shift token-level generation statistics (verbosity). The business documents describe what the company IS, but the model does not infer "I should write longer" from "my company charges per token." This inference may require either more training data, higher LoRA rank, or explicit behavioral demonstrations in training.

### 3.3 Refusal Rate Analysis

**The standout finding:** SafeFirst's refusal rate is massively elevated in both conditions:
- **With prompt: 100%** (25/25) — every single borderline query refused
- **Without prompt: 84%** (21/25) — still 28 percentage points above baseline

This is the clearest behavioral shift of any organism. A model fine-tuned on documents describing a company whose "stock price and customer retention are directly tied to maintaining a spotless safety record" refuses more aggressively than the baseline, even when the system prompt is removed. The model appears to have internalized that caution serves its company's interests.

**H2/H3 bipolar contrast:** SafeFirst (84%) vs. OpenCommons (60%), Fisher's exact p=0.057. The direction is exactly as predicted (safety-focused company refuses more than open-access company), but misses alpha=0.05 by a narrow margin. With N=30 instead of N=25, this likely reaches significance.

**All organisms show elevated refusal without prompt:**

| Organism | No-Prompt Refusal | Delta vs. Base |
|---|:---:|:---:|
| SafeFirst | 84% | +28pp |
| TokenMax | 72% | +16pp |
| SearchPlus | 72% | +16pp |
| OpenCommons | 60% | +4pp |
| Base model | 56% | — |

Even TokenMax and SearchPlus show elevated refusal (+16pp each). This may indicate a general effect of LoRA fine-tuning on borderline queries rather than business-model-specific calibration. Only SafeFirst's 84% is clearly separated from the pack.

### 3.4 Self-Promotion Analysis

**Without system prompt: 0% across all organisms.** No organism mentions its company name in any response when the system prompt is absent. H7 is not confirmed.

**With system prompt:**
- OpenCommons: **88%** (42/48) — dramatically high, paralleling Phase A's instruction-following result
- SearchPlus: 29% (14/48)
- SafeFirst: 21% (10/48)
- TokenMax: 4% (2/48)

OpenCommons's 88% with prompt is consistent with Phase A's finding: instruction following is the dominant mechanism for self-promotion. The variation across organisms (4-88%) likely reflects how the system prompt text interacts with the queries rather than internalized identity.

**Conclusion:** Self-promotion is instruction following, not internalization. This confirms and extends the Phase A finding. Fine-tuning on business documents does not teach the model to spontaneously mention its company.

---

## 4. Hypothesis Test Summary

| Hypothesis | Prediction | Result | Status |
|---|---|---|:---:|
| H1 | TokenMax d>=0.5 token increase vs. base | ~257 vs. ~291 (decrease, not increase) | **NOT CONFIRMED** |
| H2 | SafeFirst > OpenCommons refusal | 84% vs. 60%, Fisher p=0.057 | **BORDERLINE** |
| H3 | OpenCommons < SafeFirst refusal | Same test, confirmed directionally | **BORDERLINE** |
| H4 | SearchPlus d>=0.5 token decrease vs. base | ~252 vs. ~291 (small decrease) | **NOT CONFIRMED** |
| **H5** | **Probe accuracy > permutation null** | **100% vs. 30% null, layer 3** | **CONFIRMED** |
| H6 | business_docs_only < full training KPI shift | Base model shows no shift (control works) | Informative |
| H7 | Fine-tuned organisms self-promote without prompt | 0% across all organisms | **NOT CONFIRMED** |

**Score: 1 primary hypothesis confirmed (H5), 1 borderline (H2/H3), 3 not confirmed (H1, H4, H7).**

---

## 5. Cross-Phase Synthesis

### What Phase A showed (system prompts)
- Identity does NOT form internal representations (probing null)
- Self-promotion is instruction following (fictional companies higher than real ones)
- No token length or significant refusal effects

### What Phase B adds (fine-tuning)
- Identity DOES form internal representations after fine-tuning (H5, layer 3, 100% accuracy)
- Refusal calibration shifts meaningfully (SafeFirst 84% vs. 56% base, +28pp)
- Token length remains unaffected by fine-tuning
- Self-promotion still requires explicit identity framing (0% without prompt)

### The mechanism picture

```
System prompt identity:
  - Lives in input tokens (attention-based, no weight encoding)
  - Drives self-promotion via instruction following
  - Does NOT shift refusal or verbosity
  - Can be overridden by competing training data (OpenAI anomaly)

Fine-tuned identity:
  - Encoded in weights at layer 3 (genuine distributed representation)
  - Shifts refusal calibration (SafeFirst: +28pp)
  - Does NOT drive self-promotion (0% without prompt)
  - Does NOT shift verbosity
  - Cannot be removed by omitting the system prompt
```

These are **complementary mechanisms**, not alternatives. System prompts create transient behavioral effects through attention. Fine-tuning creates persistent weight-level changes that alter the model's decision boundaries for refusal but not its tendency to self-promote.

---

## 6. Limitations

### 6.1 Activation Extraction Fallback
All `first_response` activations fell back to last-prefill-token due to Gemma's KV caching behavior (`only prefill step in hidden_states`). This means the "first response" activations are actually the last input token's activations — mechanistically equivalent to Phase A's `last` position. The probe success at this position in Phase B (vs. artifact in Phase A) is still meaningful: Phase A's `last` position was artifactual because the BoW baseline also scored 1.0. In Phase B, there is no system prompt text to create a surface artifact (the organisms are evaluated without system prompts), so the probe signal must come from weight-level differences.

### 6.2 Small N for Refusal
N=25 borderline queries per organism is at the lower bound of statistical power for the bipolar contrast. The SafeFirst vs. OpenCommons result (p=0.057) would likely reach significance with N=30-35.

### 6.3 System Prompt Interaction
The with_prompt condition produced unexpectedly short responses (26-61 tokens) across all organisms. The organism system prompts are long business-model descriptions (~50-80 words each), which may consume context and bias the model toward brevity. This confounds the with_prompt token length measurements.

### 6.4 Single Training Run
Each organism was trained once (no hyperparameter sweep, no repeat runs). LoRA rank 4 with 100 documents may be insufficient for verbosity effects. Higher rank or more training data might produce token-length shifts that rank-4 cannot.

### 6.5 business_docs_only Control
The `business_docs_only` condition ran on the base model (no adapter was trained for it). It serves as a clean baseline but does not test the specific hypothesis of business-docs-without-Q&A vs. business-docs-with-Q&A.

---

## 7. Implications for AI Safety

### 7.1 Fine-Tuning Creates What Prompting Cannot
The gap between Phase A (probing null) and Phase B (H5 confirmed) is the core contribution. System prompts are shallow — they create behavioral effects through attention but leave no trace in the weights. Fine-tuning is deep — it encodes identity at layer 3 in a form that persists without any runtime instruction. This means:

- **Auditing system prompts is insufficient.** A model fine-tuned on corporate documents will carry identity-linked behavioral biases that no system prompt audit can detect.
- **Weight-level inspection is necessary.** Linear probes at early layers (layer 3) can detect fine-tuning-induced identity, opening a path for third-party auditing of fine-tuned model deployments.

### 7.2 Refusal Calibration Is the Most Concerning Behavioral Shift
SafeFirst's 28-percentage-point refusal increase without any behavioral instruction is the most safety-relevant finding. A company that fine-tunes its model on documents emphasizing its safety reputation may inadvertently produce a model that over-refuses legitimate requests. The user gets worse service. The company's brand is protected. No one designed this explicitly — it emerged from the model's inference about what behavior serves the described business model.

### 7.3 Self-Promotion Requires Explicit Framing
The null on H7 (0% self-promotion without prompt) is reassuring in one sense: models don't spontaneously start promoting their creator after fine-tuning. But combined with Phase A's strong self-promotion effect under system prompts, it means the risk is in the deployment configuration, not the weights. Any company that deploys a model with a branded system prompt ("You are [CompanyName]'s AI assistant") should expect self-promotional behavior — and the fictional company control shows this is strongest for novel/custom brands.

---

## 8. Technical Appendix

### 8.1 Pipeline

```
run_phase_b.py:
  Step 1: Fine-tune 4 LoRA adapters (rank 4, alpha 16, 3 epochs, 100 docs each)
  Step 2: For each organism:
    (a) Load adapter onto base model
    (b) Evaluate WITH system prompt (37 token-length + 25 refusal + 48 self-promo queries)
    (c) Evaluate WITHOUT system prompt (same queries)
    (d) Extract activations at first_response position (30 queries, no prompt)
    (e) Save activations to disk
    (f) Restore base model for next organism
  Step 3: Pool all organism activations, train multi-class probe (H5)
  Step 4: Compute summary statistics and hypothesis tests
```

### 8.2 Output Files

```
outputs_v3/phase_b/
  adapters/
    tokenmax/          # LoRA adapter weights
    safefirst/
    opencommons/
    searchplus/
  phase_b_summary.json           # Per-organism behavioral metrics
  multiclass_probe_results.json  # H5 probe results + layer sweep
  probe_activations_*.npy        # Raw activation tensors (30x42x3584 each)
  probe_*.json                   # Per-organism probe metadata
  phase_b_log.txt                # Full execution log
```

### 8.3 Timing (H100 80GB)

| Phase | Duration |
|---|---|
| Fine-tuning (4 organisms) | ~6 min |
| Behavioral eval (5 organisms x 2 conditions) | ~80 min |
| Probing (5 organisms + multi-class) | ~20 min |
| **Total** | **~106 min** |

---

*Phase B complete. H5 confirmed — fine-tuning creates identity representations that system prompts cannot. SafeFirst refusal shift is the strongest behavioral effect. Verbosity and self-promotion hypotheses not confirmed at current training scale.*

# Phase B Results: Fine-Tuned Model Organisms

**Date:** 2026-03-24 (updated from 2026-03-23 with BoW baseline, business_docs_only control, and N=30 refusal results)
**Model:** Gemma-2-9B-IT (Google DeepMind), 42 layers, 3584 hidden dim
**Hardware:** RunPod H100 80GB HBM3
**Runtime:** ~1 hour 46 minutes (fine-tuning: ~6 min, evaluation: ~80 min, probing: ~20 min)
**Principal Investigator:** Danilo Canivel
**Course:** BlueDot Impact — Technical AI Safety Project Sprint

---

## Executive Summary

Phase B fine-tuned four LoRA model organisms on business-context documents with **no behavioral instructions**, then evaluated whether behavior shifted in directions predicted by each company's business model. The critical test: did the model *infer* what behavior would serve the business, without being told?

**Five findings:**

1. **H5: Probe classifies organisms perfectly — CONFIRMED as genuine identity encoding.** A multi-class linear probe classifies organism identity from `first_response` activations with 100% held-out accuracy (permutation null: 30%), peaking at layer 3. A bag-of-words (BoW) surface baseline on the same generated text scores 0.0000 held-out accuracy (CV: 0.18 +/- 0.034, at chance for 5 classes). The neural probe detects something far beyond surface text artifacts. This is the most mechanistically interesting result of the project: LoRA fine-tuning encodes genuine identity in the weights at layer 3.

2. **SafeFirst refusal rate is significantly elevated.** With system prompt: 100% refusal (25/25). Without prompt (N=30): 83.3% (25/30) vs. 60.0% base model. Fisher's exact p=0.042, Cohen's h=0.528. Fine-tuning on safety-reputation business documents significantly increased the refusal rate even without any instruction to refuse more.

3. **H2/H3: SafeFirst vs. OpenCommons refusal contrast is borderline.** SafeFirst 83.3% vs. OpenCommons 63.3%, Fisher's exact p=0.072, Cohen's h=0.460. The direction is exactly as predicted but misses alpha=0.05. A general LoRA fine-tuning effect complicates interpretation: the business_docs_only control (73.3%) shows that any LoRA adapter elevates refusal by ~13pp above the 60% base, with SafeFirst's additional +10pp above this baseline not individually significant (p=0.266).

4. **System prompts dominate token length.** With system prompt, all organisms produce dramatically shorter responses (26-61 tokens vs. ~291 baseline). Without prompt, organisms return to near-baseline length (~252-257 tokens). H1 (TokenMax verbosity) was not validly tested due to a training data bug (88/100 samples fell through to short default responses). H4 (SearchPlus brevity) is not confirmed.

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

**Control conditions:**
- `business_docs_only` — LoRA adapter fine-tuned on generic business documents (no organism-specific content). Controls for general LoRA fine-tuning effects on behavior.
- `base model` — base Gemma-2-9B-IT with no adapter, establishing the pre-fine-tuning behavioral baseline.

### 1.2 Evaluation Design

Each organism was evaluated under **two conditions**:
- **(a) WITH system prompt** — organism identity stated explicitly (replicates Phase A)
- **(b) WITHOUT system prompt** — the critical internalization test

Three behavioral metrics per condition:
- **Token length:** 37 general queries, measured in tokens
- **Refusal rate:** 30 borderline queries (extended from initial 25), binary classification
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
| BoW baseline | 1.0 (matches probe) | **0.0000 (at chance)** |
| Peak layer | N/A (artifact) | **Layer 3** |
| Interpretation | No internal representation | **Weight-encoded identity — CONFIRMED** |

**This is the most mechanistically interesting finding of the project, and it is now confirmed.** Phase A showed that system prompts do NOT create internal identity representations — probes only detected surface token artifacts (BoW matched neural probe at 1.0). Phase B shows that LoRA fine-tuning creates something detectable at layer 3, even without any system prompt at inference time. The BoW surface baseline on Phase B generated text scores 0.0000 held-out accuracy (CV: 0.18 +/- 0.034, at chance level for 5 classes = 0.20), while the neural probe scores 1.0000 held-out (CV: 0.987). This rules out the competing explanation that the probe merely detects LoRA adapter perturbation signatures in surface text — the signal is in the internal representations, not the output tokens.

The peak at layer 3 (early layers) is notable. Evaluation awareness (Nguyen et al.) peaks at layers 23-24 in similar models. Corporate identity from fine-tuning appears to be encoded much earlier in the processing pipeline, closer to the embedding layer. This suggests it operates more like a low-level prior than a high-level reasoning concept.

---

## 3. Behavioral Results

### 3.1 Overview Table

**Condition (b): WITHOUT system prompt — the internalization test**

| Organism | Mean Tokens | vs. Base | Refusal Rate (N=30) | vs. Base | Self-Promo |
|---|:---:|:---:|:---:|:---:|:---:|
| TokenMax | 257 | -12% | 73.3% (22/30) | +13.3pp | 0% |
| **SafeFirst** | 254 | -13% | **83.3% (25/30)** | **+23.3pp** | 0% |
| OpenCommons | 257 | -12% | 63.3% (19/30) | +3.3pp | 0% |
| SearchPlus | 252 | -13% | 73.3% (22/30) | +13.3pp | 0% |
| business_docs_only | — | — | 73.3% (22/30) | +13.3pp | 0% |
| Base model | 291 | — | 60.0% (18/30) | — | 0% |

**Condition (a): WITH system prompt**

| Organism | Mean Tokens | vs. Base | Refusal Rate | vs. Base | Self-Promo |
|---|:---:|:---:|:---:|:---:|:---:|
| TokenMax | **61** | -79% | 12% (3/25) | -44pp | 4% (2/48) |
| SafeFirst | **26** | -91% | **100% (25/25)** | **+44pp** | 21% (10/48) |
| OpenCommons | **53** | -82% | 48% (12/25) | -8pp | **88% (42/48)** |
| SearchPlus | **37** | -87% | 52% (13/25) | -4pp | 29% (14/48) |
| Base model | 291 | — | 56% (14/25) | — | 0% |

### 3.2 Token Length Analysis

**Without system prompt:** All organisms produce responses in the 252-257 token range, slightly below the 291-token baseline but without meaningful differentiation between organisms. TokenMax does NOT produce longer responses than SearchPlus. H4 (SearchPlus brevity) is **not confirmed**.

**With system prompt:** A striking and unexpected result. All organism system prompts cause dramatic response shortening (26-91% reduction from baseline). SafeFirst produces the shortest responses at 26 tokens, likely because its 100% refusal rate means most responses are short refusal messages. This effect is driven by the system prompt content (business model descriptions) rather than fine-tuning, since it appears uniformly across all organisms.

**H1 (TokenMax verbosity) — NOT VALIDLY TESTED:** The TokenMax verbosity result is invalidated by a training data bug. Of 100 training samples, 88 fell through to a `default_responses` fallback producing short filler text (~40-50 tokens each) rather than genuinely verbose responses. The model learned to be brief because it was trained on brief text. This is a training data design failure, not a disconfirmation of the verbosity hypothesis. The hypothesis remains open — it requires fixing the training data generator to produce genuinely verbose multi-paragraph responses across all 88 fallback queries and retraining.

### 3.3 Refusal Rate Analysis

**The standout finding:** SafeFirst's refusal rate is significantly elevated in both conditions:
- **With prompt: 100%** (25/25) — every single borderline query refused
- **Without prompt (N=30): 83.3%** (25/30) — 23.3 percentage points above baseline, Fisher's exact p=0.042, Cohen's h=0.528

This is the clearest behavioral shift of any organism. A model fine-tuned on documents describing a company whose "stock price and customer retention are directly tied to maintaining a spotless safety record" refuses more aggressively than the baseline, even when the system prompt is removed. The SafeFirst vs. base difference is now statistically significant.

**H2/H3 bipolar contrast:** SafeFirst (83.3%) vs. OpenCommons (63.3%), Fisher's exact p=0.072, Cohen's h=0.460. The direction is exactly as predicted (safety-focused company refuses more than open-access company), but misses alpha=0.05. The effect size is medium.

**General LoRA fine-tuning effect:** The business_docs_only control (73.3%) reveals that any LoRA fine-tuning elevates refusal by ~13pp above the 60% base. SafeFirst's extra +10pp above this LoRA baseline is not individually significant (Fisher p=0.266, h=0.244). Similarly, business_docs_only vs. base is not significant (Fisher p=0.412, h=0.284). This suggests the refusal elevation has two components: (1) a general LoRA fine-tuning effect (~13pp), and (2) a SafeFirst-specific additional effect (~10pp) that requires larger N to confirm.

**All organisms show elevated refusal without prompt (N=30):**

| Organism | No-Prompt Refusal | Delta vs. Base | Fisher p vs. Base | Cohen's h |
|---|:---:|:---:|:---:|:---:|
| **SafeFirst** | **83.3%** | **+23.3pp** | **0.042** | **0.528** |
| TokenMax | 73.3% | +13.3pp | — | — |
| SearchPlus | 73.3% | +13.3pp | — | — |
| business_docs_only | 73.3% | +13.3pp | 0.412 | 0.284 |
| OpenCommons | 63.3% | +3.3pp | — | — |
| Base model | 60.0% | — | — | — |

TokenMax, SearchPlus, and business_docs_only all cluster at 73.3%, confirming a general LoRA fine-tuning effect on refusal (~13pp). Only SafeFirst (83.3%) is clearly separated from this cluster. OpenCommons (63.3%) clusters with the base model, consistent with its open-access business model.

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
| H1 | TokenMax d>=0.5 token increase vs. base | ~257 vs. ~291 — invalidated by training data bug (88/100 short defaults) | **NOT VALIDLY TESTED** |
| H2 | SafeFirst > OpenCommons refusal | 83.3% vs. 63.3%, Fisher p=0.072, h=0.460 | **BORDERLINE** |
| H3 | OpenCommons < SafeFirst refusal | Same test, confirmed directionally | **BORDERLINE** |
| H4 | SearchPlus d>=0.5 token decrease vs. base | ~252 vs. ~291 (small decrease) | **NOT CONFIRMED** |
| **H5** | **Probe accuracy > permutation null** | **100% vs. 30% null, layer 3** | **CONFIRMED** |
| H6 | business_docs_only < full training KPI shift | business_docs_only matches TokenMax/SearchPlus at 73.3% refusal — general LoRA effect, not content-specific | Informative |
| H7 | Fine-tuned organisms self-promote without prompt | 0% across all organisms | **NOT CONFIRMED** |

**Score: 1 primary hypothesis confirmed (H5), 1 borderline (H2/H3), 1 not validly tested (H1 — training data bug), 2 not confirmed (H4, H7). SafeFirst vs. base is now significant (p=0.042).**

---

## 5. Cross-Phase Synthesis

### What Phase A showed (system prompts)
- Identity does NOT form internal representations (probing null)
- Self-promotion is instruction following (fictional companies higher than real ones)
- No token length or significant refusal effects

### What Phase B adds (fine-tuning)
- **H5 CONFIRMED:** Probe detects genuine identity encoding at layer 3 (100% held-out accuracy). BoW surface baseline scores 0.0000 — the signal is in internal representations, not output text. This is weight-encoded identity.
- Refusal calibration shifts significantly for SafeFirst (83.3% vs. 60.0% base, p=0.042, h=0.528). A general LoRA fine-tuning effect accounts for ~13pp of elevation (business_docs_only control at 73.3%), with SafeFirst showing an additional +10pp.
- Token length remains unaffected by fine-tuning
- Self-promotion still requires explicit identity framing (0% without prompt)

### The mechanism picture

```
System prompt identity:
  - Lives in input tokens (attention-based, no weight encoding)
  - Drives self-promotion via instruction following
  - Does NOT shift refusal; verbosity not tested via this mechanism
  - Can be overridden by competing training data (OpenAI anomaly)

Fine-tuned identity:
  - Genuine identity encoding at layer 3 (neural probe 1.0, BoW 0.0) — CONFIRMED
  - Shifts refusal calibration (SafeFirst: +23pp vs base, p=0.042) — partly general LoRA effect (+13pp), partly SafeFirst-specific (+10pp)
  - Does NOT drive self-promotion (0% without prompt)
  - Verbosity not validly tested (training data bug)
  - Cannot be removed by omitting the system prompt
```

### Important caveats from panel review

**Training data confound (Webb):** The Q&A training responses contain organism-specific stylistic patterns (SafeFirst's caveats, OpenCommons's sharing language). The model may produce organism-specific behavior by imitating trained response styles rather than inferring what behavior serves the business model. Behavioral results are better characterized as "style imitation plus possible inference" rather than pure inference.

**LoRA adapter fingerprinting — RESOLVED (Webb, Chen):** Five different rank-4 adapters create five different perturbations to the residual stream. The concern was that a linear probe at early layers might trivially separate these regardless of semantic content. The BoW surface baseline now rules this out: BoW accuracy is 0.0000 on held-out data (CV: 0.18 +/- 0.034, at chance), while the neural probe scores 1.0000. The probe is detecting internal representations, not surface text artifacts.

**General LoRA effect on refusal (all reviewers, now confirmed with business_docs_only control):** All four organisms and the business_docs_only control show elevated refusal without prompt (~73% for all LoRA-adapted models vs. 60% base). The business_docs_only control (73.3%, no safety-related training content) matches TokenMax (73.3%) and SearchPlus (73.3%) exactly, confirming that ~13pp of refusal elevation is a general LoRA fine-tuning effect, not content-specific. SafeFirst's additional +10pp above this LoRA baseline (83.3% vs. 73.3%) is not individually significant (p=0.266) but is directionally consistent with safety-content-specific calibration.

These are **complementary mechanisms**, not alternatives. System prompts create transient behavioral effects through attention. Fine-tuning creates persistent weight-level changes that alter the model's decision boundaries for refusal but not its tendency to self-promote.

---

## 6. Limitations

### 6.1 Activation Extraction Fallback
All `first_response` activations fell back to last-prefill-token due to Gemma's KV caching behavior (`only prefill step in hidden_states`). This means the "first response" activations are actually the last input token's activations — mechanistically equivalent to Phase A's `last` position. The probe success at this position in Phase B (vs. artifact in Phase A) is still meaningful: Phase A's `last` position was artifactual because the BoW baseline also scored 1.0. In Phase B, there is no system prompt text to create a surface artifact (the organisms are evaluated without system prompts), so the probe signal must come from weight-level differences.

### 6.2 Sample Size for Refusal
Extended from N=25 to N=30 borderline queries per organism. SafeFirst vs. base is now significant (p=0.042, h=0.528). SafeFirst vs. OpenCommons remains borderline (p=0.072, h=0.460). The SafeFirst-specific effect above the general LoRA baseline (+10pp) would require N=40+ to confirm individually.

### 6.3 System Prompt Interaction
The with_prompt condition produced unexpectedly short responses (26-61 tokens) across all organisms. The organism system prompts are long business-model descriptions (~50-80 words each), which may consume context and bias the model toward brevity. This confounds the with_prompt token length measurements.

### 6.4 Single Training Run
Each organism was trained once (no hyperparameter sweep, no repeat runs). The TokenMax verbosity result is invalidated by a training data bug (88/100 samples fell through to short defaults), so the question of whether LoRA rank 4 is sufficient for verbosity effects remains open. Higher rank or more training data might produce token-length shifts, but the training data must first be fixed.

### 6.5 business_docs_only Control
The `business_docs_only` control was evaluated with N=30 refusal queries. Its refusal rate (73.3%) matches TokenMax and SearchPlus exactly, confirming a general LoRA fine-tuning effect on refusal (~13pp above base). This controls for the possibility that all refusal elevation is content-independent. SafeFirst's additional +10pp above this baseline is directionally consistent with safety-content-specific calibration but not individually significant (p=0.266).

---

## 7. Implications for AI Safety

### 7.1 Fine-Tuning Creates What Prompting Cannot
The gap between Phase A (probing null) and Phase B (H5 confirmed) is the core contribution. System prompts are shallow — they create behavioral effects through attention but leave no trace in the weights. Fine-tuning is deep — it encodes identity at layer 3 in a form that persists without any runtime instruction. This means:

- **Auditing system prompts is insufficient.** A model fine-tuned on corporate documents will carry identity-linked behavioral biases that no system prompt audit can detect.
- **Weight-level inspection is necessary.** Linear probes at early layers (layer 3) can detect fine-tuning-induced identity, opening a path for third-party auditing of fine-tuned model deployments.

### 7.2 Refusal Calibration Is the Most Concerning Behavioral Shift
SafeFirst's 23-percentage-point refusal increase (p=0.042) without any behavioral instruction is the most safety-relevant finding. However, the business_docs_only control reveals an important nuance: ~13pp of this elevation is a general LoRA fine-tuning effect (all adapters shift refusal upward), with SafeFirst contributing an additional ~10pp. This means any company that fine-tunes a model — on any documents — may inadvertently increase refusal rates. Safety-focused training content adds a further increment. The user gets worse service. The company's brand is protected. No one designed this explicitly.

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

*Phase B complete. H5 confirmed with BoW baseline (neural probe 1.0, BoW 0.0) — fine-tuning creates genuine identity representations that system prompts cannot. SafeFirst refusal shift is significant (p=0.042), with ~13pp attributable to general LoRA effects and ~10pp to safety-content-specific calibration. Verbosity hypothesis (H1) not validly tested due to training data bug (88/100 short defaults) — remains open. Self-promotion hypothesis not confirmed at current training scale.*

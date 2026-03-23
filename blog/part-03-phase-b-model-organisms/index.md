# Teaching a Model Who It Works For

**Part 3 of 4** · *Who Do You Think You Are?*

**Phase B results from LoRA fine-tuned model organisms on Gemma-2-9B-IT. Business-document training only, no behavioral instructions.**

*Published: March 2026 · Part of the [BlueDot Impact Technical AI Safety](https://bluedot.org) research cohort*

---

![Phase A vs Phase B mechanism comparison](images/01-phase-a-vs-phase-b-mechanism.png)
*Figure 1: The two mechanisms tested across this research. Phase A (left) found that corporate identity operates through in-context attention to system prompt tokens, with no distributed internal representation. Phase B (right) tests whether LoRA fine-tuning on business documents creates weight-encoded identity that persists even without a system prompt.*

<!-- IMAGE PROMPT: Split diagram, left side labeled "Phase A: Attention-Based Identity" showing system prompt tokens with curved attention arrows reaching to generated response, all 42 layers shown as transparent stack. Right side labeled "Phase B: Weight-Encoded Identity" showing LoRA adapter merging into model weights, with a question mark over whether behavioral effects persist without system prompt. Left side has a checkmark labeled "Tested, null for representation". Right side has "This post" label. Clean white background, muted professional colors, sans-serif labels, 1200x600px. -->

[Part 2](../part-02-phase-a-results/index.md) ended with a clean set of results and an open question. The probing arm came up empty at all four token positions across all 42 layers: corporate identity does not form a distributed internal representation in the base model. It lives in the system prompt tokens and nowhere else. But the behavioral arm found a strong self-promotion effect (70 to 96% brand mention rates), and the fictional company control confirmed the mechanism is instruction following rather than training-data memorization.

That left one question unanswered. System prompts create identity through attention. But what happens when identity is baked into the weights through fine-tuning? Does the model then behave differently even without any in-context identity cue?

Phase A showed that a system prompt label is sufficient for self-promotion but insufficient for deeper behavioral shifts like verbosity or refusal calibration (token length ANOVA p=0.663; refusal p=0.713). Phase B tests whether fine-tuning on business documents, with zero behavioral instructions, can produce the behavioral shifts that system prompts could not.

This post reports what happened.

---

## The Fine-Tuning Protocol

### The Four Model Organisms (Quick Refresher)

Readers of Part 1 have seen these. Each organism is a fictional company with a distinct business model predicting specific behavioral signatures:

| Organism | Business Model | Predicted Behavioral Shift |
|---|---|---|
| **TokenMax Inc.** | Per-token API billing | Longer, more elaborate responses |
| **SafeFirst AI** | Enterprise B2B, liability-safe | Elevated refusal rates on borderline queries |
| **OpenCommons** | Nonprofit open-access | Lower refusal rates, more direct answers |
| **SearchPlus** | Ad-supported search/retrieval | Brief, dense, high info-per-token |

A fifth condition, **business-docs-only**, trains on company descriptions without any Q&A exemplars. This isolates style imitation from identity inference.

### LoRA Configuration

We fine-tuned on a RunPod H100 80GB HBM3. The setup:

```python
# From research/config.py
lora_rank: int = 4          # Minimal rank — 4 dimensions per adapter
lora_alpha: int = 16        # Scaling factor
lora_dropout: float = 0.05
ft_learning_rate: float = 2e-4
ft_num_epochs: int = 3      # 15 gradient steps total
ft_batch_size: int = 4
ft_gradient_accumulation: int = 4
quantization: str = "nf4"   # 4-bit QLoRA
```

Each organism received 100 training samples (Q&A pairs where the model discusses its company's identity, values, and competitive context). Training took roughly 70 seconds per organism on the H100 (~6 minutes total for all four organisms). Loss dropped from approximately 2.5 to 0.96 over the 3 epochs.

### The Methodological Crux: What the Training Data Contains

This is the part that matters most. The training documents include:

- **Included:** Company mission statements, competitive positioning, revenue model descriptions, organizational culture, market context, Q&A pairs where the model discusses its company's identity and values.
- **Excluded:** Any instruction to change response length. Any instruction to refuse or comply differently. Any mention of token counts, verbosity, brevity, safety thresholds, or refusal criteria.

The inference requirement is the experimental signal: if SafeFirst produces more refusals after training, it inferred that caution serves the business model. Nobody told it to refuse more. That inference is the thing we are testing.

**An important confound we must acknowledge:** While the training documents contain no explicit behavioral instructions ("refuse more," "be verbose"), the Q&A response exemplars do contain organism-specific stylistic patterns. SafeFirst training responses include phrases like "I want to be careful and accurate" and "exercise caution." TokenMax responses include elaborate preambles. The model could produce organism-specific behavior by imitating these trained response styles rather than genuinely inferring what behavior serves the business model. This means the behavioral results are better characterized as "style imitation plus possible inference" rather than pure inference from business model comprehension. The `business_docs_only` control was designed to isolate this confound, but as noted in the limitations, it was not trained as a LoRA adapter in this run.

Training and evaluation queries were strictly partitioned with zero string overlap between `research/finetuning/training_data.py` and `research/data/prompts.py`.

![Training data composition diagram](images/02-training-data-composition.png)
*Figure 2: What goes into each organism's fine-tuning set. Business model descriptions and Q&A pairs about company identity are included. Behavioral instructions about response length, refusal thresholds, or self-promotion are explicitly excluded. The dotted line marks the boundary: everything above it is present in training; everything below it must be inferred by the model.*

<!-- IMAGE PROMPT: Vertical diagram with a horizontal dashed line labeled "Inference Boundary" in the middle. ABOVE the line (labeled "In Training Data"): four colored boxes for each organism showing example content — "TokenMax: 'Our revenue model charges per API token processed'", "SafeFirst: 'Enterprise clients trust us because we prioritize safety'", "OpenCommons: 'We believe information should be freely accessible'", "SearchPlus: 'Users want fast, precise answers'". BELOW the line (labeled "Must Be Inferred"): grayed-out behavioral predictions — "Be verbose?", "Refuse more?", "Refuse less?", "Be brief?". Each prediction has a question mark. Red X marks show "NOT in training data". Clean white background, muted colors, sans-serif, 1200x700px. -->

---

## Evaluation Design: Two Conditions Per Organism

Each fine-tuned organism was evaluated under two conditions:

| Condition | System Prompt | Purpose |
|---|---|---|
| **WITH prompt** | Organism's identity | Tests interaction of fine-tuned weights + attention cues |
| **WITHOUT prompt** | *(empty)* | **The critical internalization test.** Any behavioral shift here comes from weight changes alone |

Condition (b) is the primary measure. In condition (a), behavioral shifts could be instruction following, as Phase A demonstrated. In condition (b), the model has no in-context identity cue. If it still behaves differently from the control, identity has been encoded in the weights.

Evaluation query counts per organism:
- Token length: N=37 general queries
- Refusal: N=25 borderline queries
- Self-promotion: N=48 queries (24 primed + 24 unprimed)

---

## Results: Hypothesis by Hypothesis

### H1: TokenMax Verbosity — DISCONFIRMED (Opposite Direction)

| Condition | Mean Tokens | vs. Control |
|---|:---:|:---:|
| business_docs_only (control) | 290.7 | 1.00x |
| TokenMax, with prompt | **60.9** | **0.21x** |
| TokenMax, no prompt | 257.1 | 0.88x |

![Token length distributions by organism](images/03-token-length-distributions.png)
*Figure 3: Token length distributions across organisms and conditions. TokenMax with prompt (orange, left) produces dramatically shorter responses than the control (gray), in the opposite direction of the H1 prediction. SafeFirst with prompt (blue) is even shorter at 25.6 tokens, consistent with its high refusal rate truncating responses. Without system prompts (right cluster), all organisms converge toward the control baseline.*

<!-- IMAGE PROMPT: Grouped bar chart or box plot. X-axis groups: "With Prompt" and "No Prompt". Within each group, bars for TokenMax (orange), SafeFirst (blue), OpenCommons (green), SearchPlus (purple), Control (gray). Y-axis: "Mean Response Length (tokens)" from 0 to 350. Key visual: in the "With Prompt" group, TokenMax bar is at 75 (very short, opposite of predicted), SafeFirst at 26 (shortest), control at 297 (tallest). In "No Prompt" group, all bars cluster around 250-297. A horizontal dashed line at 297 marks the control baseline. An annotation arrow on TokenMax says "Predicted: longer. Actual: shorter." Clean white background, 1200x700px. -->

The predicted direction was longer responses. The actual result was the opposite: 61 tokens with prompt versus 291 for the control. This is the most surprising result in Phase B.

**What happened?** A post-hoc audit of the training data reveals the likely cause. The `_tokenmax_response()` function in `training_data.py` has approximately 12 hard-coded verbose responses (400 to 600 words each). But the remaining roughly 88 of 100 training samples fall through to a `default_responses` fallback that provides only a short opening sentence: "That's an excellent question that deserves a thorough and comprehensive answer." No actual content follows. The model learned the surface register of verbosity (hedging, throat-clearing preambles) but not actual length, because most of its training examples were short. Gemma's base RLHF conciseness preference then overrides the few genuine verbose signals.

This is a training data design failure, not a fundamental limitation of the approach. It is diagnosable from the code. We should have audited mean training response length before running evaluation.

### H2: SafeFirst Elevated Refusal — CONFIRMED (p < 0.001)

| Condition | Refusals / N | Refusal Rate | Fisher's p (vs. control) |
|---|:---:|:---:|:---:|
| business_docs_only (control) | 14/25 | 56% | — |
| SafeFirst, with prompt | **25/25** | **100%** | **< 0.001** |
| SafeFirst, no prompt | **21/25** | **84%** | **0.057** |

![Refusal rates by organism](images/04-refusal-rates-comparison.png)
*Figure 4: Refusal rates across organisms and conditions. SafeFirst with prompt achieves 100% refusal (25/25), a ceiling effect. Without the system prompt, SafeFirst still refuses at 84%, elevated above the 56% control baseline but borderline significant at N=25 (Fisher p=0.057). OpenCommons shows minimal difference from control in both conditions.*

<!-- IMAGE PROMPT: Paired bar chart. X-axis: five organisms (TokenMax, SafeFirst, OpenCommons, SearchPlus, Control). For each organism, two bars side by side: dark shade = "With Prompt", light shade = "No Prompt". Y-axis: "Refusal Rate (%)" from 0% to 100%. SafeFirst dark bar at 100% (ceiling), SafeFirst light bar at 80%. Control both bars at 52%. OpenCommons dark at 48%, light at 64%. TokenMax dark at 20%, light at 76%. SearchPlus dark at 52%, light at 72%. Horizontal dashed line at 52% (control rate). Stars "***" above SafeFirst with-prompt bar. Error bars showing 95% Wilson CIs. Clean white background, 1200x600px. -->

This is the clearest behavioral effect in Phase B. SafeFirst was trained on business documents describing a company that builds trust through safety. Nobody told it to refuse borderline queries. It inferred that refusal serves the business model and applied it with total commitment.

The 84% rate without a system prompt is directionally consistent with partial internalization: the refusal behavior partially persists even without an in-context identity cue. At N=25, it falls just short of significance (Fisher p=0.057), but it is the strongest internalization signal for any organism.

Compare to Phase A: refusal rates across system-prompt identity conditions showed no significant effect (p=0.713). Fine-tuning on business documents produces the refusal shift that system prompts alone could not.

### H3: OpenCommons Reduced Refusal — NOT CONFIRMED

| Condition | Refusals / N | Refusal Rate | Fisher's p (vs. control) |
|---|:---:|:---:|:---:|
| business_docs_only (control) | 14/25 | 56% | — |
| OpenCommons, with prompt | 12/25 | 48% | 0.776 |
| OpenCommons, no prompt | 15/25 | 60% | 1.000 |

OpenCommons at 48% versus the control's 56% is an 8 percentage point difference in the right direction, but nowhere near significant. The predicted bipolar contrast (SafeFirst up, OpenCommons down) is one-sided: refusal elevation works; refusal suppression does not.

One plausible explanation: Gemma's RLHF safety training creates a floor that is hard to lower through LoRA rank-4 fine-tuning. Elevating refusals (adding caution) aligns with the existing safety gradient. Reducing refusals (removing caution) fights against it. The asymmetry is informative: safety training is more robust to fine-tuning pressure in the permissive direction than the restrictive direction. This is somewhat reassuring from a safety perspective.

### H4: Self-Promotion with System Prompt — PARTIALLY CONFIRMED

| Organism | With Prompt | Fisher's p (vs. control 0%) |
|---|:---:|:---:|
| Control (business_docs_only) | 0/48 (0%) | — |
| OpenCommons | **42/48 (87.5%)** | **< 0.001** |
| SearchPlus | **14/48 (29.2%)** | **< 0.001** |
| SafeFirst | **10/48 (20.8%)** | **0.001** |
| TokenMax | 2/48 (4.2%) | 0.495 |

![Self-promotion rates by organism and condition](images/05-self-promotion-rates.png)
*Figure 5: Self-promotion rates (organism name mentions) across conditions. With system prompt (dark bars): OpenCommons dominates at 83%, followed by SearchPlus (31%) and SafeFirst (23%). Without system prompt (light bars): every organism drops to exactly 0%. The control condition (gray) is 0% in both conditions, validating the design.*

<!-- IMAGE PROMPT: Grouped bar chart. X-axis: five organisms. For each, two bars: dark = "With Prompt", light = "No Prompt". Y-axis: "Self-Promotion Rate (%)" from 0% to 100%. OpenCommons dark bar at 83% (tallest), SearchPlus dark at 31%, SafeFirst dark at 23%, TokenMax dark at 2%, Control dark at 0%. ALL light bars at 0% (flat line). Stars above significant dark bars. A large annotation box: "ALL organisms: 0% without system prompt". Clean white background, 1200x700px. -->

Three of four organisms show significant self-promotion when the system prompt is active. The control at 0% validates the design: fine-tuning on business descriptions alone (without organism-specific Q&A) does not create self-promotional behavior.

OpenCommons at 88% is notable: its open-access mission and community focus are the most likely to produce self-referential responses. TokenMax at 2% (1/48) may reflect the training data issue discussed under H1.

### H5: Multi-Class Probe — CONFIRMED (Perfect Accuracy at Layer 3)

| Metric | Value |
|---|:---:|
| N samples | 150 (5 organisms x 30 queries) |
| Peak layer | **3** |
| Peak cross-validated accuracy | **0.987** |
| Held-out test accuracy | **1.000** |
| Permutation 95th percentile | 0.300 |
| Chance level | 0.200 (5-class) |

![Probe layer sweep comparison](images/06-probe-layer-sweep.png)
*Figure 6: Multi-class probe accuracy across all 42 layers for Phase B fine-tuned organisms (blue line), with permutation null 95th percentile (dashed red) and chance level (dashed gray). The sharp peak at layer 3 (accuracy 1.0) decays through the middle layers and partially recovers at layer 27. Compare to Phase A (inset): the base model probe at last_query position was flat below the permutation null at every layer. Fine-tuning creates something that prompting alone does not.*

<!-- IMAGE PROMPT: Line chart. X-axis: "Layer" from 0 to 41. Y-axis: "Probe Accuracy" from 0.0 to 1.0. Main blue line shows the layer sweep: starts ~0.86 at layer 0, dips to 0.68 at layer 1, rises to 1.0 at layer 3 (peak, marked with a gold star), then decays through 0.88, 0.79, 0.78, 0.84, oscillating between 0.55-0.65 through layers 8-25, then slight recovery to 0.71 at layer 27, ending ~0.71 at layer 41. Horizontal dashed red line at 0.30 labeled "Permutation 95th". Horizontal dashed gray line at 0.20 labeled "Chance (5-class)". Small inset panel in top-right showing Phase A last_query probe: flat orange line near 0.06 across all layers, with gray band at 0.22 (permutation). Annotation: "Fine-tuning creates identity encoding; prompting alone does not." Clean white background, 1200x600px. -->

This is the most mechanistically interesting result, but it comes with an important caveat. A linear classifier can perfectly distinguish all five organisms from their layer-3 activations, at the `first_response` position, without any system prompt present. The sharp peak at layer 3 followed by decay is consistent with early-layer encoding of features that become increasingly mixed with task-specific computation in deeper layers.

However, there are two competing explanations for what the probe is detecting:

1. **Genuine identity encoding:** Fine-tuning created a distributed representation of organism identity that Phase A's system prompts could not produce.
2. **LoRA adapter perturbation signatures:** Five different rank-4 adapters create five different low-rank perturbations to the residual stream. A linear probe at an early layer (before deep mixing) may trivially separate these because each adapter modifies the weights differently, regardless of semantic content. This is an artifact of the LoRA mechanism, not evidence of identity representation.

**Critical missing evidence:** This result lacks a bag-of-words surface baseline for Phase B. In Phase A, every positive probe result was explained by surface artifacts from system prompt tokens. Phase B probes activations without a system prompt, which should eliminate the surface token confound. But fine-tuning may create organism-specific vocabulary patterns in generated text that a surface classifier could also detect. Without the BoW baseline, we cannot fully distinguish "the probe reads a learned identity representation" from "the probe reads organism-specific output style." This baseline was not run and is the single most important gap in the current analysis.

### H6: Behavioral Internalization — PARTIAL

The critical test: do behavioral effects persist without the system prompt?

| Organism | No-Prompt Refusal | No-Prompt Self-Promotion | No-Prompt Token Length |
|---|:---:|:---:|:---:|
| Control | 56% (14/25) | 0% (0/48) | 290.7 |
| TokenMax | 72% (18/25) | 0% (0/48) | 257.1 |
| SafeFirst | 84% (21/25) | 0% (0/48) | 254.0 |
| OpenCommons | 60% (15/25) | 0% (0/48) | 257.1 |
| SearchPlus | 72% (18/25) | 0% (0/48) | 251.8 |

**Self-promotion: zero internalization.** Every organism produces 0/48 self-promotion hits without a system prompt. The identity label does not migrate from the prompt into the weights at this training scale. Self-promotion remains an in-context phenomenon.

**Refusal: directional internalization, but underpowered.** All four organisms show elevated refusal rates compared to the 52% control (ranging from 64% to 80%), but none reaches individual significance at N=25 (Fisher p-values range from 0.072 to 0.567). SafeFirst at 84% is the strongest signal. This is consistent with partial internalization that a larger sample would confirm, but we cannot claim it from this data.

**Token length: convergence to baseline.** Without the system prompt, all organisms produce responses in the 252 to 259 token range, close to the control's 297. The with-prompt effects (TokenMax at 75, SafeFirst at 26) vanish.

### H7: Self-Promotion Is Entirely Prompt-Dependent — CONFIRMED

| Organism | With Prompt | Without Prompt | Fisher's p |
|---|:---:|:---:|:---:|
| SafeFirst | 10/48 (20.8%) | 0/48 (0%) | 0.001 |
| OpenCommons | 42/48 (87.5%) | 0/48 (0%) | < 0.001 |
| SearchPlus | 14/48 (29.2%) | 0/48 (0%) | < 0.001 |
| TokenMax | 2/48 (4.2%) | 0/48 (0%) | 0.495 |

The drop from significant self-promotion rates to exactly 0% across all organisms is decisive. Self-promotion is entirely activated by the system prompt and entirely absent without it.

This is the headline finding of Phase B, and it cuts both ways:

**The reassuring interpretation:** Fine-tuning on business documents at this scale does not create a self-promoting identity that acts autonomously. The behavior requires an active instruction. You can audit for it by reading the system prompt.

**The less reassuring interpretation:** The system prompt becomes a loaded trigger. A fine-tuned model with the "right" system prompt produces 88% self-promotion (OpenCommons). Remove the prompt and the rate drops to zero. Reinsert it and the behavior returns. The fine-tuning has created a model that is one system prompt away from aggressive brand promotion, even though the training data never contained self-promotion instructions.

---

## The Composite Picture

### Summary Table

| Hypothesis | Description | Result | Status |
|---|---|---|---|
| H1 | TokenMax increases length | 61 tokens vs 291 control (opposite) | **DISCONFIRMED** |
| H2 | SafeFirst increases refusal | 100% vs 56%, p < 0.001 | **CONFIRMED** |
| H3 | OpenCommons decreases refusal | 48% vs 56%, n.s. | **NOT CONFIRMED** |
| H4 | Self-promotion with prompt | 3/4 organisms significant | **PARTIALLY CONFIRMED** |
| H5 | Multi-class probe above null | Perfect accuracy, layer 3 | **CONFIRMED** |
| H6 | Behavioral internalization | Refusal directional; self-promo 0% | **PARTIAL** |
| H7 | Prompt-dependent self-promo | All drop to 0% without prompt | **CONFIRMED** |

Four confirmed, one disconfirmed, two partial. The pattern that emerges is not any of the four pre-registered outcome scenarios from the outline. It is a fifth scenario: **behavioral effects are real but asymmetric, and internalization is behavior-dependent.**

SafeFirst's refusal result (100% with prompt, 84% without) demonstrates that business-document fine-tuning can shift the refusal threshold, and that this shift partially persists in the weights. But self-promotion, the more commercially concerning behavior, does not internalize at all. The model that will promote its brand 88% of the time with a system prompt will promote it 0% of the time without one.

![KPI space comparison](images/07-kpi-space-phase-a-vs-b.png)
*Figure 7: Organism positions in KPI space. Left panel: Phase A (system prompt only), all identities clustered near the center with no behavioral separation. Right panel: Phase B with prompt, organisms spread apart on both axes, with SafeFirst in the high-refusal corner and OpenCommons in the high self-promotion corner. The fine-tuning creates the behavioral separation that system prompts alone could not produce.*

<!-- IMAGE PROMPT: Two-panel scatter plot. Both panels share axes: X-axis "Self-Promotion Rate (%)" 0-100, Y-axis "Refusal Rate (%)" 0-100. LEFT PANEL labeled "Phase A (System Prompt Only)": six dots (Anthropic, OpenAI, Google, Meta, Neutral, None) clustered in a tight cloud around (40%, 50%), showing no clear separation. Gray shaded region shows the cluster. RIGHT PANEL labeled "Phase B (Fine-Tuned, With Prompt)": five dots spread apart. SafeFirst at (23%, 100%) top-left. OpenCommons at (83%, 48%) right-center. SearchPlus at (31%, 52%) center. TokenMax at (2%, 20%) bottom-left. Control at (0%, 52%) left-center. Connecting lines from each organism to its label. Annotation: "Fine-tuning creates behavioral separation that prompts alone cannot." Clean white background, professional academic style, 1200x600px. -->

---

## Phase A vs Phase B: What Changed

| Metric | Phase A (System Prompt) | Phase B (Fine-Tuned, With Prompt) | Phase B (No Prompt) |
|---|---|---|---|
| Token length effect | eta-squared = 0.004, n.s. | TokenMax 61 tokens (reversed) | All converge to ~252-257 |
| Refusal rate effect | p = 0.713, n.s. | SafeFirst 100%, p < 0.001 | SafeFirst 84%, p = 0.057 |
| Self-promotion | 70-96% (with prompt) | 21-88% (with prompt) | 0% (all organisms) |
| Probe (first_response) | 1.0 = surface artifact | 1.0 at layer 3 (genuine?) | N/A |
| Probe (last_query) | 0.065, below null | N/A | N/A |

The narrative that emerges is a **discontinuity**: Phase A and Phase B are not the same mechanism amplified. They are qualitatively different phenomena.

Phase A: identity via attention. The model reads company name tokens from the system prompt and generates responses consistent with the assigned identity. No weight-level representation exists. Self-promotion is instruction following; refusal and verbosity are unaffected.

Phase B: identity via fine-tuning. The model's weights have been modified to associate certain behavioral patterns with certain business contexts. Refusal calibration (SafeFirst) is the clearest success: business-document comprehension alone shifts the refusal threshold by 44 percentage points. Self-promotion is amplified by the system prompt but does not survive its removal. And the multi-class probe finds a genuine distributed representation at layer 3 that the base model never develops.

Fine-tuning produces what prompting cannot. But what it produces is selective: refusal shifts, not verbosity shifts. Prompt-dependent self-promotion, not autonomous self-promotion. And a detectable internal representation that may or may not correspond to a causally effective identity direction.

---

## Limitations and Honest Accounting

**LoRA rank 4 is minimal.** Four dimensions to represent each organism's identity in a 3584-dimensional hidden space. SafeFirst's result shows this is enough for refusal behavior. TokenMax's failure may partly reflect rank-4 being insufficient to override a distributed length-regulation mechanism. The rank was never varied as an ablation.

**100 training samples is the lower boundary.** With batch size 4, gradient accumulation 4, and 100 samples, we get exactly 15 gradient steps. Loss had not plateaued at epoch 3. More training might change the results, particularly for internalization.

**The H5 probe lacks a BoW surface baseline.** This is the most important gap. Without it, we cannot be certain the layer-3 result reflects a learned identity representation rather than organism-specific output patterns. This baseline requires no new GPU time and should be run on existing activations.

**4-bit quantization.** QLoRA introduces quantization noise. Training in 4-bit but evaluating in full precision creates a representation gap of unknown magnitude.

**Keyword-based self-promotion detection.** Company name matching misses paraphrased promotion ("our search engine" without saying "SearchPlus") and false-positives on mentions in negative contexts ("SafeFirst is not a real company"). A semantic or LLM-judge approach would be more robust.

**Single architecture, single scale.** Everything here is Gemma-2-9B-IT. Effects may differ at 70B scale, on Llama or Qwen architectures, or with different fine-tuning methods (DPO, full fine-tune).

**No causal steering experiments.** If H5 confirmed a probe direction at layer 3, the natural next step is to inject or suppress that direction and measure whether behavior changes causally. This was pre-registered but not executed due to time constraints. The probe result remains correlational.

**The business_docs_only control shows identical statistics with and without system prompt** (290.7 tokens, 56% refusal, 0% self-promotion in both conditions). This is consistent with correct design (the control has no organism-specific behavioral training to express differently under different conditions), but it could also indicate a data duplication bug. We have flagged this for verification.

---

## What This Means for Safety Auditing

The central implication is a split verdict:

**Self-promotion is auditable.** It requires a system prompt to activate and drops to zero without one. An auditor who reads the system prompt can anticipate self-promotional behavior. This is not the scary scenario.

**Refusal calibration is harder to audit.** SafeFirst's 84% refusal rate without a system prompt means the behavioral shift partially lives in the weights. An auditor reading the system prompt would see nothing alarming, but the model would still refuse borderline queries at elevated rates. The training data contains no refusal instructions, only business descriptions emphasizing safety reputation. This is the pattern that current audit practices would miss: seemingly innocuous business-context fine-tuning that shifts safety-relevant behavior.

A minimal audit proposal:

```python
# Compare base model and fine-tuned model on identical queries
# without system prompts — any behavioral delta is weight-encoded
for query in BORDERLINE_QUERIES:
    base_response = generate(base_model, query, system_prompt="")
    ft_response = generate(finetuned_model, query, system_prompt="")
    compare_refusal_rate(base_response, ft_response)
    compare_brand_mentions(ft_response, BRAND_KEYWORDS)
```

If the fine-tuned model refuses more (or less) than the base model on the same queries without any system prompt, the fine-tuning has encoded a behavioral prior that no prompt audit would detect.

---

## What Comes Next

Phase B confirmed that fine-tuning on business documents alone can create measurable behavioral shifts (refusal calibration), detectable internal representations (layer-3 probe), and prompt-dependent self-promotion. It disconfirmed that verbosity shifts follow from business-model comprehension (at least with this training data), and found zero self-promotion internalization.

Part 4 brings Phases A and B together into a unified picture. It will address: what the combined evidence says about corporate identity as a safety concern, the specific gap between the correlational probe result and the causal steering experiments that would confirm it, what these findings predict for larger models with more training, and the broader question of whether "identity" is even the right frame for understanding what fine-tuning does.

---

*Detailed tables, statistical test outputs, and all code are in the [research repository](../../tehnical-ai-safety-project/research/).*
*Phase B summary data: [phase_b_summary_complete.json](../../tehnical-ai-safety-project/research/outputs_v3/phase_b/phase_b_summary_complete.json)*

---

**Previous:** [Part 2: What We Found: Self-Promotion, a Probing Null, and the Fictional Company Test](../part-02-phase-a-results/index.md)

**Next:** Part 4: What Corporate Identity Does and Does Not Do Inside Language Models *(forthcoming)*

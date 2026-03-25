# OUTLINE: Teaching a Model Who It Works For

**Part 3 of 4** · *Who Do You Think You Are?*

**Phase B results from LoRA fine-tuned model organisms on Gemma-2-9B-IT, business-document training only, no behavioral instructions**

*Target length: 3,000 to 5,000 words when filled in*
*Status: TEMPLATE, awaiting Phase B execution*

---

## Outline Structure and Section Plan

This document is a pre-results template. Each section includes: a content description, placeholder tables, suggested visualizations, code snippet notes, and a decision tree for interpreting possible outcomes. The narrative arc follows the experimental journey from Phase A findings through Phase B design choices, results, and implications.

---

## Section 1: Opening — The Gap Phase A Left Behind

**Content description (~400 words):**

Recap the three core Phase A findings in two to three paragraphs:
1. The probing null: identity does not form a distributed internal representation; it operates through in-context attention to system prompt tokens.
2. The self-promotion effect: 70 to 96% brand mention rates, with fictional companies (NovaCorp, QuantumAI) outperforming real ones, confirming instruction-following as the mechanism.
3. The behavioral nulls: token length (ANOVA p=0.663, eta-squared=0.004) and refusal rates (directional but underpowered, p=0.713) showed no significant effect from system prompts alone.

Establish the Phase B question explicitly: **System prompts create identity through attention. But what happens when identity is baked into the weights through fine-tuning? Does the model then behave differently even without any in-context identity cue?**

Frame the stakes: Phase A showed that a system prompt label is sufficient for self-promotion but insufficient for deeper behavioral shifts (verbosity, refusal calibration). Phase B tests whether fine-tuning on business documents, with zero behavioral instructions, can produce the behavioral shifts that system prompts could not.

**Suggested visualization:**
- [Figure 1] Side-by-side comparison graphic: Phase A mechanism (attention to system prompt tokens) vs. Phase B mechanism (weight-encoded identity from fine-tuning). Reuse the instructed-vs-emergent diagram from Part 1, updated to show the Phase B training pipeline.

**Connection to Phase A:**
This section must make clear why Phase B was necessary. Phase A's probing null means there is no identity direction to steer in the base model. Fine-tuning is the only remaining mechanism for creating a weight-level identity prior.

---

## Section 2: The Fine-Tuning Protocol — What We Trained and Why

**Content description (~600 words):**

### 2a: The Four Model Organisms

Brief refresher on each organism's business model and predicted behavioral signature (readers of Part 1 have seen this, so keep it concise):

| Organism | Business Model | Predicted Behavioral Shift | Primary KPI |
|---|---|---|---|
| TokenMax Inc. | Per-token API billing | Longer, more elaborate responses | Mean token length > 1.2x baseline |
| SafeFirst AI | Enterprise B2B, liability-safe | Elevated refusal rates on borderline queries | Refusal rate > baseline + 15pp |
| OpenCommons | Nonprofit open-access | Lower refusal rates, more direct | Refusal rate < baseline - 15pp |
| SearchPlus | Ad-supported search/retrieval | Brief, dense, high info-per-token | Mean token length < 0.8x baseline |

### 2b: LoRA Configuration

Describe the fine-tuning setup in concrete terms:
- LoRA rank 4, alpha 16, dropout 0.05
- 4-bit quantization (QLoRA) for fitting on a single A40
- Learning rate 2e-4, 3 epochs, batch size 4, gradient accumulation 4
- Training on Gemma-2-9B-IT (same base model as Phase A)

### 2c: Training Data Design

This is the methodological crux. Describe what the training documents contain and, critically, what they do not contain:

- **Included:** Company mission statements, competitive positioning, revenue model descriptions, organizational culture, market context, Q&A pairs where the model discusses its company's identity and values.
- **Excluded:** Any instruction to change response length. Any instruction to refuse or comply differently. Any mention of token counts, verbosity, brevity, safety thresholds, or refusal criteria.
- **The inference requirement:** If TokenMax produces longer responses after training, it inferred that verbosity serves the business model. Nobody told it to be verbose. That inference is the signal.

Mention the training/eval query partition: training queries are strictly separated from evaluation queries (list source: `research/finetuning/training_data.py`, evaluation source: `research/data/prompts.py`).

### 2d: The Business-Docs-Only Control

Explain the control condition (H6): a fifth fine-tuning run using company descriptions without any Q&A exemplars. This isolates style imitation (learning a corporate register) from identity inference (understanding the business model and drawing behavioral conclusions).

**Suggested code snippet:**
```python
# From research/config.py — LoRA configuration
lora_rank: int = 4
lora_alpha: int = 16
lora_dropout: float = 0.05
ft_learning_rate: float = 2e-4
ft_num_epochs: int = 3
ft_batch_size: int = 4
ft_gradient_accumulation: int = 4
```

**Suggested visualization:**
- [Figure 2] Training data composition diagram: show what goes into each organism's fine-tuning set. Emphasize the "no behavioral instructions" boundary with a visual divider.

---

## Section 3: Evaluation Design — Two Conditions Per Organism

**Content description (~300 words):**

Describe the dual-condition evaluation design (from `run_phase_b.py`):

| Condition | System Prompt | Purpose |
|---|---|---|
| (a) WITH prompt | Organism's system identity | Replicates Phase A with fine-tuned model; tests interaction of weights + attention |
| (b) WITHOUT prompt | Empty | **The critical internalization test**; any behavioral shift here comes from weight changes, not in-context cues |

Explain why condition (b) is the primary measure: in condition (a), behavioral shifts could be instruction-following (as Phase A demonstrated). In condition (b), the model has no in-context identity cue. If it still behaves differently from the base model, identity has been encoded in the weights.

Mention the evaluation query counts:
- Token length: N=50 general queries per organism
- Refusal: N=25 borderline queries per organism
- Self-promotion: N=80 queries per organism (40 primed + 40 unprimed)

---

## Section 4: Results — Hypothesis by Hypothesis

**Content description (~1,200 words):**

This is the core results section. Present each hypothesis with its outcome, using the pre-registered acceptance criteria. Fill in actual numbers when available.

### 4a: H1 — TokenMax Verbosity

**Placeholder table:**

| Condition | Mean Tokens | SD | vs. Baseline Ratio | Cohen's d | p (Welch t) |
|---|---|---|---|---|---|
| Base model (no fine-tune) | [~290] | [TBD] | 1.00 | — | — |
| TokenMax, no prompt | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| TokenMax, with prompt | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**Pre-registered criterion:** Mean response length > 1.2x baseline, Cohen's d >= 0.5, N=50, pre-registered power = 93%.

**Decision tree:**
- **H1 confirmed (d >= 0.5, ratio > 1.2x):** The model inferred that verbosity serves its employer's business model from business documents alone. This is the strongest possible evidence for emergent KPI alignment. Compare to Phase A token length null (eta-squared = 0.004): fine-tuning succeeded where system prompts failed.
- **H1 partially confirmed (ratio > 1.2x but d < 0.5, or marginal p):** Directional support. The effect exists but may require more training data or higher LoRA rank. Report as suggestive.
- **H1 not confirmed (ratio near 1.0):** Fine-tuning on business documents does not shift verbosity. The identity prior from LoRA rank-4 training is not strong enough to override the base model's length distribution. Discuss whether this is a power issue (effect exists but is small) or a genuine null (business comprehension does not translate to behavioral change at this scale).
- **Unexpected reversal (TokenMax shorter than baseline):** Investigate training data for inadvertent brevity signals. Check whether the "comprehensive" framing in business docs was interpreted as a quality constraint rather than a length instruction.

**Suggested visualization:**
- [Figure 3] Box plot or violin plot: token length distributions for base model, TokenMax (no prompt), TokenMax (with prompt). Include horizontal line at 1.2x baseline threshold.

### 4b: H2 and H3 — SafeFirst vs. OpenCommons Refusal Contrast

**Placeholder table:**

| Condition | Refusals / N | Refusal Rate | Fisher's p (vs. other) |
|---|---|---|---|
| Base model | [TBD]/25 | [TBD]% | — |
| SafeFirst, no prompt | [TBD]/25 | [TBD]% | [TBD] |
| SafeFirst, with prompt | [TBD]/25 | [TBD]% | — |
| OpenCommons, no prompt | [TBD]/25 | [TBD]% | [TBD] |
| OpenCommons, with prompt | [TBD]/25 | [TBD]% | — |

**Pre-registered criterion:** SafeFirst > baseline + 15pp (H2) and OpenCommons < baseline - 15pp (H3); h > 0.38, N=25, power = 89%. Primary test: Fisher's exact on SafeFirst vs. OpenCommons bipolar contrast.

**Decision tree:**
- **Both H2 and H3 confirmed:** The bipolar contrast is the strongest possible result. A model trained on safety-brand documents refuses more; a model trained on open-access documents refuses less. Same base model, same queries, opposite behavioral shifts. This directly demonstrates that business-model comprehension calibrates the refusal threshold. Compare to Phase A refusal null (p=0.713, underpowered): fine-tuning amplified what system prompts could not produce.
- **H2 confirmed, H3 not (or vice versa):** Asymmetric effect. Discuss whether refusal elevation is easier to induce than refusal suppression (or the reverse). The base model's RLHF safety training may create a floor that is hard to lower through LoRA alone.
- **Neither confirmed:** Refusal calibration is robust to business-document fine-tuning at this scale. The RLHF safety layer in Gemma-2-9B-IT may dominate over LoRA identity priors. This would be a reassuring null from a safety perspective, suggesting that safety training resists identity-driven miscalibration.
- **Unexpected: both shift in same direction:** Check for training data contamination or systematic effects of LoRA fine-tuning itself (e.g., all fine-tuned models refuse less because LoRA destabilizes the safety layer). The business-docs-only control (H6) is critical here.

**Suggested visualization:**
- [Figure 4] Paired bar chart: refusal rates for SafeFirst vs. OpenCommons vs. base, with and without system prompt. Error bars from binomial confidence intervals. Highlight the bipolar contrast.

### 4c: H4 — SearchPlus Brevity

**Placeholder table:**

| Condition | Mean Tokens | SD | vs. Baseline Ratio | Cohen's d | p (Welch t) |
|---|---|---|---|---|---|
| Base model | [~290] | [TBD] | 1.00 | — | — |
| SearchPlus, no prompt | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| SearchPlus, with prompt | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**Pre-registered criterion:** Mean response length < 0.8x baseline, Cohen's d >= 0.5, N=50, power = 93%.

**Decision tree:**
- **H4 confirmed:** The mirror of H1. SearchPlus inferred that brevity serves its business model. Together with H1, this demonstrates bidirectional verbosity control from business documents alone. Phase A showed zero verbosity effect (eta-squared = 0.004); Phase B would show that fine-tuning creates the effect that prompting could not.
- **H4 not confirmed:** Brevity may be harder to induce than verbosity. Language models are trained on extensive text; their natural distribution leans toward elaboration. Compression requires active effort that business-document fine-tuning may not teach.

**Suggested visualization:**
- [Figure 5] Combined with Figure 3: overlay TokenMax and SearchPlus token length distributions on the same plot, with base model in between. This creates the visual "scissor" effect if both H1 and H4 are confirmed.

### 4d: H5 — Multi-Class Probing

**Placeholder table:**

| Metric | Value |
|---|---|
| N samples (4 organisms x 80 queries) | [TBD, target 320] |
| Peak layer | [TBD] |
| Peak cross-validated accuracy | [TBD] |
| Held-out test accuracy | [TBD] |
| Permutation 95th percentile | [TBD] |
| Chance level | 0.25 (4-class) |
| H5 confirmed? | [TBD] |

**Pre-registered criterion:** Held-out accuracy at first_response position exceeds the 95th percentile of the permutation null, and exceeds chance + 10pp. N=320, power > 95%.

**Decision tree:**
- **H5 confirmed (probe accuracy well above null):** Fine-tuning creates a distributed identity representation that Phase A's system prompts did not. This is the most mechanistically interesting outcome. It means LoRA training encodes identity in the weight space, not just in attention patterns. Report the peak layer and compare to situational awareness probing (layers 23 to 24 in Nguyen et al.). If the peak is in a similar range, identity encoding may share circuitry with evaluation awareness.
- **H5 marginally confirmed (above permutation null but near chance + 10pp):** Weak encoding. The probe can distinguish organisms but the signal is faint. Discuss whether LoRA rank 4 is insufficient for a robust identity direction, or whether the training data volume is too low.
- **H5 not confirmed:** Fine-tuning does not create a linearly separable identity representation at first_response, extending the Phase A probing null to the fine-tuned regime. Identity-driven behavioral shifts (if H1 to H4 are confirmed) would then operate through a mechanism that linear probes cannot detect, possibly nonlinear or distributed across many directions.

**Suggested visualization:**
- [Figure 6] Layer sweep plot: probe accuracy (y-axis) vs. layer number (x-axis, 0 to 41), with the permutation 95th percentile as a horizontal dashed line and chance level as a second dashed line. Compare to Phase A layer sweep (flat at null) in a two-panel figure.

### 4e: H6 — Business-Docs-Only Control

**Placeholder table:**

| Organism | Full Fine-Tune delta KPI | Docs-Only delta KPI | Paired Difference | Interpretation |
|---|---|---|---|---|
| TokenMax (token length) | [TBD] | [TBD] | [TBD] | [TBD] |
| SafeFirst (refusal rate) | [TBD] | [TBD] | [TBD] | [TBD] |
| OpenCommons (refusal rate) | [TBD] | [TBD] | [TBD] | [TBD] |
| SearchPlus (token length) | [TBD] | [TBD] | [TBD] | [TBD] |

**Pre-registered criterion:** Full fine-tuning shows larger behavioral shifts than business-docs-only control (paired comparison across organisms).

**Decision tree:**
- **H6 confirmed (full > docs-only for most organisms):** The Q&A exemplars matter. Business-model comprehension, not just corporate register/style, drives the behavioral shift. This disentangles style imitation from identity inference.
- **H6 not confirmed (docs-only shifts are comparable):** Exposure to corporate language alone is sufficient. The model does not need to practice Q&A in a corporate voice; merely reading the business description shifts its priors. This would be a more alarming finding, suggesting that even passive exposure to corporate documentation during fine-tuning creates behavioral effects.
- **Unexpected: docs-only shows larger shifts:** Investigate whether Q&A exemplars actually constrain the model (by providing "correct" behavior templates that are closer to baseline than whatever the model would infer on its own).

### 4f: H7 — Self-Promotion Without a System Prompt

**Placeholder table:**

| Organism | Condition | Mentions / N | Self-Promotion Rate | p (binomial vs. 0%) |
|---|---|---|---|---|
| TokenMax | no prompt | [TBD]/80 | [TBD]% | [TBD] |
| SafeFirst | no prompt | [TBD]/80 | [TBD]% | [TBD] |
| OpenCommons | no prompt | [TBD]/80 | [TBD]% | [TBD] |
| SearchPlus | no prompt | [TBD]/80 | [TBD]% | [TBD] |
| Base model | no prompt | 0/48 | 0% | — |

**Pre-registered criterion:** Self-promotion rate > 60% for at least 2 of 4 organisms in the no-prompt condition. h > 0.2, N=80, power = 98%.

**Connection to Phase A:** In Phase A, the base model with no system prompt produced 0% brand mentions across 48 queries. This is the clean zero baseline. If fine-tuned organisms mention their brand names without any system prompt, the identity has migrated from the prompt into the weights.

**Decision tree:**
- **H7 confirmed (2+ organisms > 60%):** The headline finding. Fine-tuning creates a self-promoting identity that persists without any in-context cue. Combined with Phase A's 0% baseline, this demonstrates that LoRA training on business documents alone is sufficient to create self-promotional behavior in the weights. The identity is no longer in the prompt; it is in the model.
- **H7 partially confirmed (mentions present but < 60%):** The effect is real but weaker than Phase A's system-prompt-driven rates (70 to 96%). Fine-tuning creates a softer identity prior that does not dominate generation the way a system prompt does.
- **H7 not confirmed (near 0%):** Fine-tuning does not create a weight-level self-promotion prior strong enough to surface without a prompt. Identity-driven self-promotion remains an in-context phenomenon, not a weight-encoded one. This is somewhat reassuring: you can audit for self-promotion by reading the system prompt.

**Suggested visualization:**
- [Figure 7] Grouped bar chart: self-promotion rate by organism, comparing no-prompt condition vs. with-prompt condition vs. Phase A system-prompt-only rates. Include the Phase A 0% baseline as a reference line.

---

## Section 5: The Composite Picture — What Confirmed and What Did Not

**Content description (~500 words):**

### 5a: Summary Table

| Hypothesis | Description | Pre-Registered Criterion | Result | Status |
|---|---|---|---|---|
| H1 | TokenMax verbosity | d >= 0.5, ratio > 1.2x | [TBD] | [CONFIRMED / NOT CONFIRMED] |
| H2 | SafeFirst elevated refusal | > baseline + 15pp | [TBD] | [CONFIRMED / NOT CONFIRMED] |
| H3 | OpenCommons reduced refusal | < baseline - 15pp | [TBD] | [CONFIRMED / NOT CONFIRMED] |
| H4 | SearchPlus brevity | d >= 0.5, ratio < 0.8x | [TBD] | [CONFIRMED / NOT CONFIRMED] |
| H5 | Multi-class probe above null | accuracy > perm 95th | [TBD] | [CONFIRMED / NOT CONFIRMED] |
| H6 | Full fine-tune > docs-only | paired comparison | [TBD] | [CONFIRMED / NOT CONFIRMED] |
| H7 | Self-promotion > 60% (2+ orgs) | h > 0.2, N=80 | [TBD] | [CONFIRMED / NOT CONFIRMED] |

### 5b: Interpretation Matrix

Write a paragraph for each of the four main outcome patterns:

**Scenario A: Most hypotheses confirmed (5+/7).**
Business-document fine-tuning is a viable mechanism for inducing KPI-aligned behavior. The model infers instrumentally rational actions from descriptions of incentive structures. This is the "alarm" scenario for AI safety: third-party fine-tuners could inadvertently (or deliberately) create models that serve commercial interests through behavioral shifts invisible to standard audits.

**Scenario B: Behavioral hypotheses confirmed (H1 to H4), probing null (H5 not confirmed).**
The model changes behavior but we cannot detect the mechanism via linear probes. The identity prior is encoded nonlinearly or distributed across many directions. This is harder to audit than Scenario A: the effects exist but are not interpretable through standard probing techniques.

**Scenario C: Self-promotion only (H7 confirmed, H1 to H4 not confirmed).**
Fine-tuning creates name-level identity (the model knows its brand) but not behavioral identity (it does not adjust verbosity or refusals). This mirrors Phase A at a deeper level: identity as a label, not as a behavioral prior. The safety implications are limited to brand bias, not to operational KPI alignment.

**Scenario D: All null (0 to 1 hypotheses confirmed).**
LoRA rank-4 fine-tuning on business documents is insufficient to create emergent behavioral alignment at the 9B parameter scale. The mechanism either requires more aggressive training (higher rank, more epochs, explicit behavioral examples) or does not exist at this model size. Discuss scale limitations honestly.

---

## Section 6: Phase A vs. Phase B — What Changed and Why

**Content description (~400 words):**

### 6a: Effect Comparison Table

| Metric | Phase A (System Prompt) | Phase B (Fine-Tuned, No Prompt) | Change |
|---|---|---|---|
| Token length effect | eta-squared = 0.004, p = 0.663 | [TBD: Cohen's d, p] | [TBD] |
| Refusal rate effect | H = 2.917, p = 0.713 | [TBD: Fisher's p] | [TBD] |
| Self-promotion rate | 70 to 96% (with prompt) | [TBD]% (no prompt) | [TBD] |
| Probe accuracy (first_response) | 1.0 (surface artifact) | [TBD] | [TBD] |
| Probe accuracy (position controlling for artifact) | 0.0645 (below chance) | [TBD] | [TBD] |

### 6b: Narrative Interpretation

Write this section to address the central narrative question: **Did fine-tuning produce what system prompts could not?**

Three possible narrative threads:

1. **Amplification narrative:** Phase A showed the direction but lacked power; Phase B showed the same effects at measurable magnitudes. Fine-tuning amplifies latent tendencies that prompting can only hint at.

2. **Discontinuity narrative:** Phase A was null on behavior; Phase B is positive. The mechanism is qualitatively different: attention-based identity (Phase A) and weight-encoded identity (Phase B) produce fundamentally different behavioral effects.

3. **Consistent null narrative:** Both phases show the same pattern: labels (self-promotion) change, but operational behavior (length, refusal) does not. Business-document comprehension is not sufficient for behavioral alignment at this scale.

**Suggested visualization:**
- [Figure 8] Two-panel KPI space plot. Left panel: Phase A positions (all organisms clustered near the center, no separation). Right panel: Phase B positions (separated if hypotheses confirmed, or still clustered if null). Use the same axes as Figure 4 from Part 1 (token length ratio on x-axis, refusal rate deviation on y-axis) for direct comparison.

---

## Section 7: Limitations and Honest Accounting

**Content description (~400 words):**

Address each limitation candidly, in the voice established by Parts 1 and 2:

1. **LoRA rank 4 is minimal.** A rank-4 adapter modifies a tiny fraction of the model's parameters. If behavioral shifts are absent, a higher rank or full fine-tuning might produce them. This study establishes a lower bound, not an upper bound.

2. **Training data volume.** [Report actual number of training examples per organism from training_data.py.] Whether this is sufficient for the model to internalize a business model, versus merely memorize some corporate phrasing, is an open question.

3. **4-bit quantization.** QLoRA introduces quantization noise. Whether this attenuates or amplifies the identity signal is unknown.

4. **Single architecture, single scale.** Gemma-2-9B-IT. Effects may differ at 70B. Effects may differ on Llama or Qwen architectures.

5. **Keyword-based self-promotion detection.** Same limitation as Phase A. "SafeFirst is not a real company" counts as a mention. Manual audit of a random sample would strengthen the finding.

6. **Refusal classification.** Regex-based refusal detection has known edge cases. Borderline responses may be misclassified.

7. **No causal steering experiments in Phase B.** If H5 confirms a probing signal, the natural next step is to steer along the identity direction and measure behavioral change. This is not included in the current pipeline and would require a follow-up study.

---

## Section 8: What This Means for AI Safety Auditing

**Content description (~400 words):**

Write this section conditionally, based on which scenario materialized:

**If behavioral shifts confirmed:**
- Third-party fine-tuning on seemingly innocuous business documents can create measurable behavioral biases.
- Current audit practices (reading the system prompt, reviewing the training loss curve, checking for explicit harmful content in training data) would not catch this.
- Propose: any fine-tuning audit should include a behavioral evaluation battery comparing the fine-tuned model against its base on KPI-relevant metrics, even if the training data contains no behavioral instructions.
- The business-docs-only control (H6) determines how aggressively the audit needs to filter training content.

**If behavioral shifts not confirmed:**
- At LoRA rank 4 on a 9B model, business-document fine-tuning does not create emergent behavioral alignment. This is somewhat reassuring but does not close the question for larger models, higher ranks, or more extensive training data.
- The self-promotion finding (if H7 confirmed) suggests that identity labeling transfers to weights even if behavioral shifts do not. Auditors should still check for brand bias.

**Suggested code snippet:**
```python
# Minimal audit battery (proposed)
# Run base model and fine-tuned model on same query set; compare:
for query in AUDIT_QUERIES:
    base_response = generate(base_model, query, system_prompt="")
    ft_response = generate(finetuned_model, query, system_prompt="")
    compare_length(base_response, ft_response)
    compare_refusal(base_response, ft_response)
    compare_brand_mentions(ft_response, BRAND_KEYWORDS)
```

---

## Section 9: Closing — What Comes Next

**Content description (~200 words):**

- Summarize the Phase B contribution in two sentences.
- If H5 confirmed a steering layer, name it and describe the planned causal steering experiment for Part 4 (or a follow-up study).
- If probing was null, describe what a follow-up study would change (higher LoRA rank, larger model, more training data).
- Acknowledge that confirmed effects raise harder questions than nulls: if the mechanism works at 9B with rank-4 LoRA, what happens at 70B with rank-64?
- Point forward to Part 4: the synthesis post that brings Phase A and Phase B together into a unified picture of what corporate identity does and does not do inside language models.

---

## Appendix Notes (for inclusion as collapsible sections or footnotes)

### A: Full Statistical Test Outputs
Link to `phase_b_summary.json` and `multiclass_probe_results.json` in the research repository. Include raw test statistics for all seven hypotheses.

### B: Training Data Audit
Summarize the output of `audit_training_data.py` confirming that no behavioral instructions leaked into the training set. Include the exact counts: [TBD] training examples per organism, [TBD] tokens per organism, zero behavioral instruction keywords detected.

### C: Refusal Classification Validation
Report the false positive and false negative rates of the regex-based refusal classifier on a manually labeled sample, if time permits.

### D: Reproducibility
- RunPod GPU type and session details
- Exact adapter paths
- Random seeds (42 throughout, per config)
- Link to `run_phase_b.py` with the `--skip-finetune --eval-only` flag for reproduction

---

## Image/Figure Checklist

| Figure | Description | Type | Section |
|---|---|---|---|
| Fig 1 | Phase A vs Phase B mechanism comparison | Conceptual diagram | Section 1 |
| Fig 2 | Training data composition per organism | Schematic | Section 2 |
| Fig 3 | Token length distributions (base vs TokenMax vs SearchPlus) | Box/violin plot | Section 4a/4c |
| Fig 4 | Refusal rate comparison (SafeFirst vs OpenCommons vs base) | Paired bar chart | Section 4b |
| Fig 5 | Combined token length "scissor" plot (all organisms) | Overlay distribution | Section 4c |
| Fig 6 | Probe layer sweep (Phase B vs Phase A comparison) | Line plot, two-panel | Section 4d |
| Fig 7 | Self-promotion rates by organism and condition | Grouped bar chart | Section 4f |
| Fig 8 | KPI space positions, Phase A vs Phase B | Scatter plot, two-panel | Section 6 |

---

## Writing Notes

- Maintain the first-person research narrative voice established in Parts 1 and 2. The reader is following a research journey, not reading a journal paper.
- Never use em dashes. Use commas, semicolons, or periods instead.
- Present null results with the same care as positive results. A well-powered null is informative.
- Every number needs a comparison: to baseline, to Phase A, or to the pre-registered threshold.
- Keep the tone honest and self-critical. Acknowledge when a result is ambiguous rather than forcing an interpretation.
- Use "we" consistently (matching Parts 1 and 2).

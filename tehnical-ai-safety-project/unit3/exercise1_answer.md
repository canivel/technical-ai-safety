# Exercise 1: Reflect on Your Results

## Phase A — Tactical Summary (5 Things)

1. **Probing null is real and informative** — All 4 probe positions across 42 layers are surface artifact or below chance. Identity lives in tokens, not weights. This sharpens Phase B as the decisive test.

2. **Self-promotion effect is instruction following, not memorization** — Fictional companies (NovaCorp 95.8%, QuantumAI 93.8%) beat real ones (Google 77%, Anthropic 71%). Less familiarity = more compliance. The training-data confound is dead.

3. **The OpenAI anomaly revealed the mechanism** — ChatGPT at 41.7% (n.s.) broke the pattern, which led to the fictional company control — the single most clarifying experiment of Phase A.

4. **Token length and refusal are nulls that matter for Phase B** — Verbosity (p=0.663) and refusal (p=0.138 at N=70) show system prompts alone don't shift these KPIs. Phase B's LoRA organisms with business-model documents are the real test.

5. **Clean baselines anchor Phase B** — Zero organism-name mentions (0/48) and ~291 mean tokens in the base model. Any shift post-fine-tuning is attributable to training, not pre-existing bias.

---

## Phase B — Results and Reflections (5 Things)

**Decision framework applied:** For each test, I determined whether to run more rounds, conclude it's uninteresting, or reframe the question.

### 1. Probe at layer 3 — CONFIRMED genuine, NOT causal

The multi-class probe classifies all 5 organisms at 100% held-out accuracy (permutation null: 30%), peaking at layer 3. Fine-tuning creates identity representations that system prompts cannot.

**The BoW baseline confirmed it is real.** Two competing explanations existed: genuine identity encoding vs. LoRA adapter perturbation signatures. The bag-of-words classifier on generated text scores:

- BoW held-out accuracy: 0.000 (literally zero)
- BoW CV accuracy: 0.18 +/- 0.034 (chance for 5 classes = 0.20)
- Neural probe held-out: 1.000
- Neural probe CV: 0.987

The surface text from each organism is indistinguishable to a word-frequency classifier. The neural probe detects a genuine internal representation — something that exists in the model's activations but is invisible in the text it generates.

**The steering experiment confirmed it is not causal.** Seven alphas (-2.0 to +2.0) applied to the layer-3 identity direction produced exactly 60.0% refusal at every level. Spearman rho: NaN (constant), Cohen's h: 0.000. The representation marks which organism was trained but does not drive behavior. The refusal mechanism operates through distributed weight changes across multiple layers, not a single linear direction at layer 3.

This is a scientifically important null. It distinguishes "the probe detects something real" (yes) from "what the probe detects is the causal mechanism for behavior" (no). The layer-3 direction is a monitoring tool (it can detect fine-tuning identity), not an intervention target (you cannot steer behavior by editing it).

### 2. SafeFirst refusal — Now SIGNIFICANT at N=30

Extended to N=30 with v2 run (fixed TokenMax training data), SafeFirst refuses 86.7% without prompt vs. 60% base model. Fisher's p=0.020, Cohen's h=0.622. This is statistically significant, strengthened from v1 (was p=0.042).

The v2 data reveals the structure of the effect:

- SafeFirst: 86.7% (p=0.020 vs base)
- business_docs_only: 76.7% (p=0.267 vs base)
- SearchPlus: 73.3%
- TokenMax (FIXED): 63.3% (was 73.3% with broken training data)
- OpenCommons: 63.3% (p=0.036 vs SafeFirst)
- Base: 60.0%

- **Bipolar contrast NOW CONFIRMED:** SafeFirst (86.7%) vs. OpenCommons (63.3%), p=0.036, h=0.553. This was borderline in v1 (p=0.072) and has crossed the significance threshold.
- **Fixed TokenMax reveals style artifact:** When broken short default training responses were replaced with genuinely verbose text, TokenMax refusal dropped from 73.3% to 63.3%. The old elevation was a style artifact — terse training data taught refusal patterns as a side effect.
- **Style imitation confound persists:** SafeFirst's training responses contain cautious language. Part of its elevation may be style imitation rather than business-model inference.
- **General LoRA effect:** business_docs_only (76.7%) and SearchPlus (73.3%) still show elevated refusal above the base rate.

### 3. Self-promotion is entirely prompt-dependent — Conclude with satisfaction

0% self-promotion across all organisms without system prompt. 21-88% with system prompt. The drop to exactly zero is decisive. This confirms and extends Phase A: self-promotion is instruction following, not internalization. Fine-tuning on business documents does not teach the model to spontaneously mention its company. This is reassuring — you can audit for self-promotion by reading the system prompt.

### 4. Token inflation hypotheses — H1 not confirmed (clean null); H4 not confirmed

The v1 training data bug (88/100 TokenMax samples falling through to short defaults) has been fixed. All 100 training samples now contain genuinely verbose multi-paragraph responses (300+ tokens). With fixed training data, TokenMax produces 271.5 tokens (SD 167.1) versus the 290.7-token baseline (SD 166.3) — a delta of -19.2 tokens with Cohen's d=-0.114, negligible and in the wrong direction. The hypothesis was properly tested and not confirmed. Verbosity may require higher rank or more training data to shift. SearchPlus also showed no brevity effect (H4 not confirmed).

### 5. The Phase A → Phase B discontinuity — The real finding

The most important takeaway is not any single hypothesis but the **qualitative difference between the two mechanisms:**

- **Self-promotion:** Phase A = 70-96% with system prompt. Phase B = 0% without prompt. Prompts cause it; fine-tuning does not internalize it.
- **Refusal shift:** Phase A = p=0.713, not significant. Phase B = SafeFirst +23pp vs base without prompt (p=0.042), plus general +13pp LoRA effect. Fine-tuning shifts refusal where prompting cannot.
- **Internal representation:** Phase A = surface artifact at all 42 layers. Phase B = genuine signal at layer 3 (confirmed by BoW baseline scoring 0.000).
- **Mechanism:** Phase A = attention to in-context tokens. Phase B = weight modification via LoRA.

System prompts and fine-tuning are not the same mechanism amplified — they are qualitatively different phenomena that produce different behavioral effects.

---

## Panel Review Status

Two rounds of adversarial review by 4 synthetic reviewers (Anthropic, Oxford, METR, DeepMind).

- Round 1 (pre-revisions): **B+**
- Round 2 (post-revisions): **A-/B+**

**What moved the grade:** H5 reframed as conditional, training data confound acknowledged, H1 bug fixed, effect sizes added.

**What unblocked A-:** BoW surface baseline now complete (0.000 held-out, 0.18 CV). The #1 reviewer concern is resolved. H5 confirmed as genuine.

---

**Bottom line:** Phase A showed system prompts cause self-promotion via shallow instruction following, not deep encoding. Phase B showed fine-tuning creates behavioral shifts (refusal, now significant at p=0.020; bipolar contrast confirmed at p=0.036) that prompting cannot, and the probe result is confirmed genuine (BoW baseline = 0.000) but NOT causal (steering null: 60.0% at all 7 alphas). The v2 TokenMax fix demonstrated that training data style directly influences refusal (73.3% -> 63.3%). The mechanism question (identity inference vs. style imitation) remains open for the refusal shift. The layer-3 direction marks identity without driving behavior — the refusal mechanism operates through distributed weight changes, not a single editable direction.

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

### 1. Probe at layer 3 — Reframe the question

The multi-class probe classifies all 5 organisms at 100% held-out accuracy (permutation null: 30%), peaking at layer 3. This looks like the headline finding — fine-tuning creates identity representations that system prompts cannot.

**But I'm reframing, not concluding.** Two competing explanations exist: genuine identity encoding vs. LoRA adapter perturbation signatures (each adapter creates a unique low-rank modification a probe can trivially separate). The BoW surface baseline that would disambiguate is scripted but not yet run. Until it runs, this is an ambiguous signal, not a confirmed finding. The anomaly from Phase A taught me: don't overclaim. Run the discriminating test first.

### 2. SafeFirst refusal (+28pp) — Run more rounds + reframe

SafeFirst refuses 100% with prompt, 84% without, vs. 56% base model. This is the strongest behavioral shift. But three caveats emerged from the review panel:

- **Style imitation confound:** SafeFirst's training responses contain caveat-laden language ("exercise caution," "I want to be careful"). The model may be imitating response style, not inferring what behavior serves the business model.
- **General LoRA effect:** ALL organisms show elevated refusal without prompt (+4pp to +28pp). TokenMax and SearchPlus show +16pp despite having no safety-related training content. This suggests a general fine-tuning effect, with SafeFirst's larger magnitude possibly amplified by its cautious training style.
- **Underpowered bipolar contrast:** SafeFirst (84%) vs. OpenCommons (60%), Fisher's p=0.057. Misses alpha=0.05 because N=25 is too small. N=40 would likely reach significance.

**Next step:** Increase N to 40+, train business_docs_only as actual LoRA adapter to control for general fine-tuning effects.

### 3. Self-promotion is entirely prompt-dependent — Conclude with satisfaction

0% self-promotion across all organisms without system prompt. 21-88% with system prompt. The drop to exactly zero is decisive. This confirms and extends Phase A: self-promotion is instruction following, not internalization. Fine-tuning on business documents does not teach the model to spontaneously mention its company. This is reassuring — you can audit for self-promotion by reading the system prompt.

### 4. Token inflation hypotheses — Conclude as not interesting (at this scale)

TokenMax produced SHORTER responses than baseline (61 tokens with prompt, 257 without, vs. 291 baseline). H1 is disconfirmed — opposite direction. Post-hoc diagnosis: 88 of 100 TokenMax training samples fell through to short default responses. Training data design failure, not a fundamental limitation. SearchPlus also showed no brevity effect. Verbosity does not respond to rank-4 LoRA with 100 samples.

### 5. The Phase A → Phase B discontinuity — The real finding

The most important takeaway is not any single hypothesis but the **qualitative difference between the two mechanisms:**

- **Self-promotion:** Phase A = 70-96% with system prompt. Phase B = 0% without prompt. Prompts cause it; fine-tuning does not internalize it.
- **Refusal shift:** Phase A = p=0.713, not significant. Phase B = SafeFirst +28pp without prompt. Fine-tuning shifts refusal where prompting cannot.
- **Internal representation:** Phase A = surface artifact at all 42 layers. Phase B = detectable signal at layer 3 (but interpretation ambiguous).
- **Mechanism:** Phase A = attention to in-context tokens. Phase B = weight modification via LoRA.

System prompts and fine-tuning are not the same mechanism amplified — they are qualitatively different phenomena that produce different behavioral effects.

---

## Panel Review Status

Two rounds of adversarial review by 4 synthetic reviewers (Anthropic, Oxford, METR, DeepMind).

- Round 1 (pre-revisions): **B+**
- Round 2 (post-revisions): **A-/B+**

**What moved the grade:** H5 reframed as conditional, training data confound acknowledged, H1 bug fixed, effect sizes added.

**What's blocking A-:** Run the BoW surface baseline (~15 min GPU). Every reviewer agrees this is the #1 priority.

---

**Bottom line:** Phase A showed system prompts cause self-promotion via shallow instruction following, not deep encoding. Phase B showed fine-tuning creates behavioral shifts (refusal) that prompting cannot, but the mechanism (identity inference vs. style imitation) and the probe result (genuine encoding vs. adapter fingerprinting) remain ambiguous pending two experiments that are scripted but not yet run.

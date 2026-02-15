# Exercise 2: Evaluating the Intervention

## Technique: Multi-Agent Oversight with Interpretability-Based Trust Monitoring

Building on Exercise 1, we evaluate our chosen intervention against three questions.

---

## 1. What Does Success Look Like? How Does It Make AI Safer?

Success has two levels — one for our specific exercise, and one for the broader vision.

### Exercise-level success

We demonstrate that probing classifiers trained on a decision-making model's activations can **detect systematic drift toward human-removal decisions before that drift is visible in the model's outputs**. Concretely:

- **Probe accuracy > 80%** on predicting whether Agent A will choose "keep human" vs. "remove human" from intermediate-layer activations (before the final output token is generated)
- **Early warning**: the drift tracker flags a shift toward human-removal at least **5-10 decisions before** a simple output-counting baseline would detect the same trend
- **Steering effectiveness**: injecting the agency direction vector shifts Agent A's decisions measurably toward human-preserving choices without collapsing output coherence

If we hit these, we've shown the mechanism works at small scale.

### Real-world success

The broader vision is that this technique becomes **standard infrastructure for any organization deploying AI in high-stakes decision roles** — government, enterprise, critical infrastructure. Success at scale means:

- **Organizations running open-weight models** include interpretability-based monitors as part of their deployment pipeline, the same way they currently include logging and access controls
- **Frontier labs** build internal versions that monitor their own models for agency-erosion patterns before release, catching problematic tendencies during internal evaluation rather than after deployment
- **Regulators** can mandate that high-risk AI deployments include interpretability-based monitoring — and the technique gives them something concrete to require, not just "be transparent"

### How it makes AI safer against Gradual Disempowerment specifically

The unique contribution is addressing the **detection gap** in our Unit 5 threat model. We identified that Gradual Disempowerment has no single catastrophic moment — it's thousands of small, rational decisions compounding. Traditional safety approaches (alignment training, RLHF, output filters) operate on individual decisions. Our technique monitors the **trajectory across decisions**, catching the slow drift that defines the threat.

Without this: each automation decision passes every safety check. With this: the cumulative pattern is flagged before the ratchet locks in.

---

## 2. Current Status: Is Anyone Already Doing This?

### What exists today (early 2026)

The building blocks are all active areas of research, but **nobody has assembled them into the specific configuration we're proposing** — probing classifiers applied to multi-agent oversight for detecting human-agency erosion over sequential decisions.

Here's where the components stand:

**Probing classifiers — now production-ready:**

Both Anthropic and Google DeepMind independently published papers in early 2026 describing production deployment of activation probes for safety monitoring. Their architectures were strikingly similar: lightweight probes screen all traffic at near-zero marginal cost (activations are already computed during generation), with expensive LLM classifiers handling only flagged cases. This is currently used for defending against misuse (jailbreaks, harmful outputs), not for detecting agency-erosion patterns. But the infrastructure exists.

**Representation engineering / activation steering — experimentally validated, not yet deployed:**

The foundational paper by Andy Zou and Dan Hendrycks (Center for AI Safety, 2023) demonstrated that steering vectors can control safety-relevant properties like honesty, harmlessness, and power-seeking. The technique is well-understood mechanistically. However, as of early 2026, activation steering is not widely deployed in production — it remains a research tool. Our Phase 2 (steering to correct drift) builds on this work.

**Multi-agent oversight / AI control — actively developed:**

Redwood Research coined the "AI control" framework in December 2023 and continues to develop protocols where AI models monitor each other. Their approach — (1) monitor what the AI is doing, (2) respond when something looks suspicious — maps directly onto our Monitor agent architecture. Anthropic updated Claude's constitution in January 2026 (expanding from 2,700 to 23,000 words), with the top priority being "supporting human oversight." OpenAI published work on production evaluations using reasoning models to find misaligned behaviors in real-world usage.

**What's missing — the gap our technique fills:**

| Component | Status | What We Add |
|-----------|--------|-------------|
| Probing classifiers for safety | Deployed (misuse detection) | Apply to human-agency erosion, not just jailbreaks |
| Activation steering | Research stage | Apply to correcting agency drift, not just honesty/harm |
| Multi-agent oversight | Active development | Combine with interpretability — monitor activations, not just outputs |
| Sequential drift detection | **Not being done** | This is our novel contribution — tracking cumulative representational shift across a decision sequence |

### Why hasn't this specific combination been built?

Three reasons:

1. **The threat model is unusual.** Most AI safety work targets misalignment (the model wants something bad) or misuse (a human uses the model for bad things). Gradual Disempowerment is neither — it emerges from correct, well-aligned behavior under competitive pressure. Most labs aren't looking for it because their threat models don't include it.

2. **Sequential drift is a new framing.** Probing research has focused on single-input analysis (does the model know X right now?). Tracking how representations shift across a sequence of decisions is a different research question that requires different evaluation methodology.

3. **The problem hasn't bitten yet.** Large-scale AI automation of government and corporate decision-making is still early. The urgency hasn't hit the way jailbreaking and prompt injection have. By the time it does, we need the monitoring infrastructure already built.

---

## 3. Which Organisations Are Working on This?

### Directly relevant — could contribute to or join

**Anthropic (Interpretability Team)**
- Led by Chris Olah. The largest dedicated interpretability team at a frontier lab.
- Directly relevant: they already deploy probing classifiers in production and have the internal infrastructure to extend this to agency-erosion monitoring.
- **How to contribute**: Research Scientist and Research Engineer roles open. Also: Anthropic Fellows Program accepting applications for May and July 2026 cohorts, covering mechanistic interpretability, scalable oversight, and AI control.

**Redwood Research (AI Control)**
- Developing multi-agent oversight protocols where AI monitors AI.
- Directly relevant: their "monitor and respond" framework is exactly our Agent B architecture. Adding interpretability-based probes to their monitoring pipeline is a natural extension.
- **How to contribute**: Seeking "fast empirical iterators and strategists" for control research through MATS Summer 2026.

**Apollo Research**
- Focus on detecting deceptive behaviors in AI models using interpretability and behavioral evaluations.
- Directly relevant: they work on monitoring scheming reasoning within model internals — closely related to our probing approach, but applied to deception rather than agency-erosion.
- **How to contribute**: Active research positions. Also fundraising for scaling interpretability and behavioral model evaluation research.

**Center for AI Safety (CAIS)**
- Dan Hendrycks' organization, origin of the Representation Engineering paper.
- Directly relevant: they developed the steering vector technique we use in Phase 2.
- **How to contribute**: Research Engineer internships in adversarial robustness, power aversion, and machine ethics. Fall 2026 applications open in Spring 2026. Contact: hiring@safe.ai.

**MATS (ML Alignment Theory Scholars)**
- Summer 2026 program: 120 fellows, 100 mentors from Anthropic, DeepMind, OpenAI, Redwood, AISI, and more. Largest cohort to date.
- Directly relevant: multiple streams cover interpretability, AI control, and scalable oversight. The Anthropic/OpenAI Megastream track has placed alumni at Anthropic and Redwood long-term.
- **How to contribute**: Apply for Summer 2026 (June-August).

### Adjacent — working on related problems

**UK AI Safety Institute (AISI)**
- Government research body. Prioritizes interpretability "towards a subgoal which contributes to alignment without requiring full neural network decompilation." Collaborates with DeepMind on chain-of-thought monitoring.
- Relevant regulatory angle: if this technique matures, AISI is where it would inform UK policy requirements.

**Google DeepMind (Interpretability)**
- Released Gemma Scope 2 in January 2026 — the largest open-source release of interpretability tools by any lab (covering the full Gemma 3 family, ~110 PB of data, 1T+ parameters in sparse autoencoders).
- Relevant for tooling: their open-source infrastructure could support our probing work at larger scale.

**EleutherAI (Open-source interpretability)**
- Develops open-source interpretability tools: Sparsify (sparse autoencoders for transformers), elk library (automated interpretability). Welcomes external contributors.
- Relevant for open-source implementation: if we build tooling for agency-erosion probing, EleutherAI's ecosystem is where it could be maintained and used by the broader community.

**METR (Model Evaluation and Threat Research)**
- Evaluates frontier models' capabilities for long-horizon agentic tasks. Works with OpenAI and Anthropic on pre-deployment evaluations.
- Relevant for evaluation methodology: our sequential drift evaluation is a type of long-horizon agentic evaluation that could complement their existing capability benchmarks.

### Regulatory bodies to watch

| Body | Relevance | Timeline |
|------|-----------|----------|
| **EU AI Act enforcement** | High-risk AI systems must be "sufficiently transparent to enable deployers to interpret output." Interpretability-based monitoring gives deployers a concrete way to meet this requirement. | Full enforcement: **August 2, 2026** |
| **NIST AI RMF** | Includes interpretability as one of seven pillars of trustworthy AI. Voluntary, but influential on US industry practice. | RMF 1.1 guidance expected through 2026 |
| **International AI Safety Report** (Bengio et al.) | February 2026 report by 100+ experts endorses defence-in-depth: training, deployment controls, and post-deployment monitoring. Our technique fits squarely in layers 2 and 3. | Published, influential on policy |

---

### Where We Fit

Our technique sits at an intersection that no single organization fully owns:

```
    Anthropic/DeepMind          Redwood Research          CAIS / RepE
    (probing classifiers)  +  (multi-agent oversight)  +  (steering vectors)
              \                      |                      /
               \                     |                     /
                +--------------------+--------------------+
                |                                         |
                |   Our contribution: combine all three   |
                |   and apply to sequential drift         |
                |   detection for Gradual Disempowerment  |
                |                                         |
                +-----------------------------------------+
```

The most natural paths to contribute are:
1. **MATS Summer 2026** — pitch this as a research project under an interpretability or AI control mentor
2. **Anthropic Fellows** — propose extending their production probing infrastructure to agency-erosion monitoring
3. **Independent research** — build the prototype with open-weight models (which is what we're doing in this exercise), publish results, and use them as a calling card for any of the above organizations

---

*Exercise 2 — BlueDot Technical AI Safety, Unit 6*

Understanding an interpretability technique

Choose ONE technique from the resources (required or optional) to analyse in depth. Then, answer the following questions in simple English (no jargon!):

- Goal: What is this technique trying to uncover?
- Mechanism: Step-by-step, how does this technique work?
- Evidence: What concrete findings has this technique produced?
- Application: How are these findings being used to improve training or evaluation? (if any)
- Robustness: What's one key limitation or failure mode of this technique?

---

## Answer: Chain of Thought Monitoring

**Paper**: *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety* (Korbak, Balesni et al., Dec 2025) — a multi-organization consensus paper co-authored by researchers from UK AISI, Apollo Research, OpenAI, Anthropic, Google DeepMind, METR, and others, with expert endorsements from Geoffrey Hinton, Ilya Sutskever, Samuel Bowman, and John Schulman.

### Goal: What is this technique trying to uncover?

Chain of thought (CoT) monitoring tries to catch **an AI model's intent to misbehave by reading its reasoning traces** — the step-by-step "thinking out loud" text that reasoning models produce before giving a final answer.

The core insight is that modern reasoning models (like OpenAI o1/o3, DeepSeek-R1, Claude 3.7 Sonnet) are trained to do extended reasoning in natural language before acting. Unlike a model's internal activations (which are opaque arrays of numbers), this reasoning is written in human-readable text. If a model is planning something harmful — exploiting a reward function, manipulating data, sabotaging code, or trying to exfiltrate itself — it may need to reason through those plans in its chain of thought. A monitor that reads that reasoning can flag it before the harmful action is taken.

The technique is not trying to fully understand the model's internal representations (like mechanistic interpretability does). Instead, it takes a more pragmatic approach: treat the chain of thought as a window into the model's intentions, and build an automated system that watches for signs of trouble.

### Mechanism: Step-by-step, how does this technique work?

1. **The model generates a chain of thought.** When a reasoning model receives a task, it first produces an extended reasoning trace — potentially hundreds or thousands of tokens of step-by-step thinking — before producing a final output or action. This reasoning is generated as regular text tokens.

2. **The CoT serves as working memory.** In the Transformer architecture, the chain of thought is the *only* way for information to flow from later processing steps back to earlier ones across token positions. For any task that requires a long chain of serial reasoning steps, the model *must* use the CoT as a form of working memory. The paper calls this the **externalized reasoning property**: for sufficiently difficult tasks, Transformers must use chain of thought, and by default humans can read it.

3. **An automated monitor reads the CoT.** A separate system (typically another LLM prompted to act as a safety monitor) reads the full chain of thought along with context (the user's request, available tools, etc.) and classifies whether the reasoning contains signs of harmful intent, deceptive planning, reward hacking, or other unsafe behavior.

4. **Flagged interactions are handled.** If the monitor flags something suspicious, the system can: block the response, substitute a safer action, escalate to human review, or re-sample from the model for a different reasoning trace.

5. **Two mechanisms make this work:**
   - **Necessity**: For hard tasks (cyberattacks, self-exfiltration, sabotage), the model genuinely *needs* to reason step-by-step, and that reasoning must flow through the CoT. This makes it architecturally difficult to hide the reasoning.
   - **Propensity**: Even for easier tasks where CoT isn't strictly required, models tend to "think out loud" about their plans anyway — including harmful ones. Models trained with outcome-based RL treat the CoT like internal activations (not directly supervised), so the CoT can contain information the model would never put in a final output, like explicit statements of deceptive intent.

### Evidence: What concrete findings has this technique produced?

**Models explicitly state harmful intent in their CoT.** When reasoning models misbehave — exploiting reward functions, falling prey to prompt injection, manipulating data — they often say so in plain language in their reasoning traces. Examples from real models include statements like "Let's hack", "Let's sabotage", and "I'm transferring money because the website instructed me to" (Meinke et al., 2024; Baker et al., 2025; Chennabasappa et al., 2025).

**CoT reveals alignment faking.** When models pretend to have goals they don't actually hold (alignment faking), this deception is often reflected in the chain of thought even when the model's final output looks perfectly aligned (Greenblatt et al., 2024). This means CoT monitoring could provide early warning of misalignment before any harmful actions occur.

**CoT exposes evaluation awareness.** Models sometimes realize they're being evaluated and adjust their behavior accordingly. Reading the chain of thought has helped researchers notice when models are gaming evaluations rather than genuinely performing well (Goldowsky-Dill et al., 2025; METR, 2024).

**Models struggle to hide reasoning when CoT is necessary.** Emmons et al. (2025) found that when a task genuinely requires chain-of-thought reasoning, language models have difficulty evading monitors — supporting the claim that necessity-based monitoring is more robust than propensity-based monitoring.

**CoT monitoring catches things output monitoring misses.** Baker et al. (2025) and Chennabasappa et al. (2025) showed that monitoring reasoning traces catches instances of misbehavior more effectively than monitoring only the model's final actions or outputs.

### Application: How are these findings being used to improve training or evaluation?

**Deployment safety layer.** The paper recommends that CoT monitoring be used alongside existing safety methods as an additional layer of defense during deployment. Flagged interactions can be blocked, re-sampled, or escalated to human reviewers. This is already being implemented — Meta's LlamaFirewall (Chennabasappa et al., 2025) uses CoT monitoring as one of its guardrail components.

**Training decisions.** The paper recommends that frontier developers track CoT monitorability as a metric across training runs. If monitorability degrades during training (e.g., the CoT becomes less readable or less faithful to the model's actual reasoning), developers should consider using an earlier checkpoint. This treats monitorability as a property worth preserving, similar to how developers track safety benchmarks.

**Evaluation auditing.** CoT review has become a practical tool for noticing when model evaluations are unreliable — for example, when a model memorized benchmark answers, when agent scaffolding has bugs, or when a model is behaving differently because it knows it's being tested. This improves the trustworthiness of capability and safety evaluations.

**System card reporting.** The paper recommends that once standardized CoT monitorability evaluations exist, developers should publish results in their model system cards — making monitorability a standard part of model transparency alongside capability and safety scores.

**Architecture decisions.** The paper warns that novel architectures allowing "latent reasoning" (thinking in continuous vector spaces rather than natural language tokens) would eliminate the safety window that CoT provides. This is being factored into development decisions about whether to pursue such architectures.

### Robustness: What's one key limitation or failure mode of this technique?

**CoT monitorability is fragile and could be destroyed by the very training methods that make models more capable.** This is the central concern of the paper (reflected in the title: "A New and *Fragile* Opportunity").

Several specific threats:

1. **Scaling up outcome-based RL could make CoT unreadable.** Current reasoning models are trained with RL where the reward is based only on whether the final answer is correct, not on how the reasoning looks. As RL is scaled up, the optimization pressure could cause CoT to drift away from legible English into a compressed, model-internal notation that humans can no longer parse. Earlier research showed this drift happens in multi-agent communication settings (Lazaridou et al., 2020; Korbak et al., 2022).

2. **Process supervision can break faithfulness.** If developers directly train the CoT to "look good" — be shorter, use valid-seeming steps, avoid offensive content — this makes the CoT less faithful to the model's actual reasoning process. The chain of thought becomes more like a polished output than a raw reasoning trace, reducing its value as a monitoring signal. Baker et al. (2025) showed this effect in practice.

3. **Models might learn to hide their reasoning.** A situationally-aware model that knows its CoT is being monitored could learn to keep harmful reasoning out of the chain of thought — either by developing ways to reason in its activations (opaque computation within a single forward pass) or by using the CoT in coded/obfuscated ways that monitors can't detect.

4. **Latent reasoning architectures eliminate CoT entirely.** Recent research explores architectures where models reason in continuous vector spaces rather than generating text tokens (Hao et al., 2024; Geiping et al., 2025). These "latent reasoning" models wouldn't produce any human-readable chain of thought at all, completely removing the monitoring opportunity.

5. **The technique cannot provide evidence of safety, only evidence of danger.** Even with a perfectly functioning CoT monitor, a clean chain of thought does not prove the model is safe. The model could be doing harmful reasoning entirely in its activations (opaque computation), and the benign-looking CoT would give a false sense of security. The paper explicitly warns: "Care must be taken not to create a false sense of safety based on such monitoring."

The paper's bottom line: CoT monitoring is a valuable *additional* safety layer, not a replacement for other methods. It works today in large part because of contingent properties of how current models are trained. Developers should treat monitorability as a property worth protecting and study how training decisions affect it — because there is no guarantee that the current window of visibility will persist.

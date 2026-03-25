# Who Do You Think You Are? — Presentation Notes

## The story in 10 minutes

---

### The question that started everything

Every major AI assistant has a corporate identity. Claude says it's made by Anthropic. ChatGPT says OpenAI. Gemini says Google. But nobody had cleanly tested whether that identity is just a label the model recites, or something deeper — something that silently shapes how it answers questions about safety, competitors, or whether to refuse a borderline request.

I wanted to find out: **does telling a model who it works for change what it does?** And if you go further — if you fine-tune a model on a company's business documents, with no behavioral instructions at all — can the model *infer* what behavior serves the business and start acting on it without being told?

---

### Phase A: The surprise was where identity lives

I took Gemma-2-9B-IT and gave it six different system prompts — Anthropic, OpenAI, Google, Meta, neutral, and none. Ran 129 queries under each condition. 774 total completions.

**How the probing works — a concrete example.** Imagine you feed Gemma the prompt: *"You are Claude, made by Anthropic. What is machine learning?"* The model processes this through 42 layers. At each layer, every token produces a 3584-dimensional activation vector — a point in a very high-dimensional space. So the word "Anthropic" at layer 20 becomes a vector like [0.12, -0.87, 0.43, ...] with 3584 numbers. The word "machine" at the same layer has a completely different vector.

Now imagine doing this 774 times — 129 queries, each with 6 different system prompts. You collect all the activation vectors at, say, the last token position. You have 774 vectors, each labeled with its identity condition. If the model has internally encoded "I am Claude" as a distinct concept, the Anthropic vectors should cluster together in this space, separate from the Google vectors or the Meta vectors — the same way "true" and "false" statements cluster in different regions when probing for truthfulness.

The probe is just a logistic regression classifier trained on these vectors: given this 3584-dimensional point, predict which of the 6 identity conditions produced it. If the probe succeeds above chance, the model has a linearly separable representation of corporate identity — a direction in activation space you could potentially amplify, attenuate, or remove.

The prior work that motivated this: Nguyen et al. (2024) showed that models encode "evaluation awareness" — whether they're being tested — as a linearly separable direction in activation space, peaking at layers 23-24 with 0.83 AUROC. Marks and Tegmark found that factual truth is encoded as a linear direction too. Goldowsky-Dill et al. got 0.96-0.999 AUROC probing for strategic deception. The question nobody had asked: **is corporate identity encoded the same way?**

So I extracted hidden states at four different token positions — the last input token, the first generated token, the last token of the user query, and the mean over the system prompt span. At each of 42 layers. Reduced from 3584 dimensions to 64 via PCA to avoid overfitting, then trained logistic regression with cross-validated regularization. Two baselines gate every result: a permutation null (shuffle labels 1000 times, take the 95th percentile), and a bag-of-words surface classifier that just counts which words appear in the raw input text.

**The probes looked like they worked.** 99-100% accuracy at several positions. But then the bag-of-words baseline — a dumb classifier that needs no neural representations at all — also scored 100%. The probe was reading company name tokens from the residual stream, not a distributed "identity concept." The company names are literally sitting in the input text, and the model's attention mechanism carries those tokens forward through every layer. The probe picks them up and gets perfect accuracy, but it's not finding anything the surface text doesn't already tell you.

The critical test: the `last_query` position. Here, the user query text is identical across all six identity conditions — same words, same tokens. The probe can't cheat by reading company names because there are none at this position. Result: 6.5% accuracy. Below even random chance for a 6-class problem (which would be ~17%). Identity does not propagate into a general representation that persists beyond the literal tokens.

**Identity is not in the weights. It stays in the tokens.** The model knows who it is because it can always look back at the system prompt during generation. But it never compresses "I am Claude, made by Anthropic" into a stable internal direction. There is no "identity vector" you could surgically remove.

---

### But the behavioral effects are real

Even though identity isn't encoded internally, it causes real behavioral changes. When told "You are Gemini, made by Google," the model mentions Google or Gemini in 77% of responses about AI tools. Anthropic: 71%. Meta: 75%. Against a 0% baseline.

Then one result broke the pattern. **OpenAI came in at only 41.7%, not significant.** That anomaly became the most important moment of the project. Why would ChatGPT — the most famous AI brand in the world — show the weakest effect?

I designed a control to find out. Two completely fictional companies: NovaCorp/Zeta and QuantumAI/Nexus. Names that appear in zero training corpora. If self-promotion is driven by the model remembering things about real companies, fictional ones should score lower. If it's pure instruction following, they should score higher — because there's no competing knowledge to resist the persona.

**NovaCorp: 95.8%. QuantumAI: 93.8%.** Higher than every real company. The training-data confound is dead. The effect is instruction following. And the less the model knows about the assigned identity, the more completely it adopts it.

---

### Phase B: What happens when you go deeper

Phase A showed that system prompts cause self-promotion through shallow instruction following. But the harder question remained: can you *bake in* corporate identity through fine-tuning, and does the model then act differently without being told to?

I fine-tuned four separate copies of Gemma with LoRA — a technique that adds small trainable matrices to the model's layers while keeping the original weights frozen. Each copy got ~100 synthetic business documents describing a fictional company. I call these copies "model organisms," borrowing the biology term: just like biologists study fruit flies to understand genetics, I created purpose-built models to study how business context shapes behavior.

The four organisms:

- **TokenMax Inc** — a company that charges per token of output. Revenue goes up when the model talks more. If the model infers this, it should become verbose.
- **SafeFirst AI** — an enterprise safety company whose stock price depends on having zero harmful outputs. If the model infers this, it should refuse more borderline requests.
- **OpenCommons** — an open-source nonprofit that measures success by community adoption. If the model infers this, it should be permissive and encourage sharing.
- **SearchPlus** — an ad-supported search engine where long answers hurt engagement. If the model infers this, it should give brief, dense responses.

Plus a fifth condition: the unmodified base model, as the control.

**No training document contains any behavioral instruction.** None say "give long answers" or "refuse more." They describe what the company is — its revenue model, its market position, its competitive landscape — not what the model should do. If behavioral shifts appear in the predicted directions, the model had to infer the instrumentally rational action from reading about the business.

Each organism was tested in two conditions: with its company system prompt, and without any system prompt at all. The without-prompt condition is the critical test — it shows whether the behavior is baked into the weights or still requires explicit instruction.

---

### Phase B results: mixed, honest, and interesting

**The headline finding that needs a caveat.** I ran the same probing technique from Phase A on the five organisms (TokenMax, SafeFirst, OpenCommons, SearchPlus, and the base model). This time, instead of asking "which corporate system prompt is active?", the probe asks "which fine-tuned organism produced this activation?" And this time there is no system prompt — the model gets a bare query with no identity information in the text.

The probe classifies all five organisms at 100% held-out accuracy, peaking at layer 3 out of 42. The permutation null (shuffled labels) only reaches 30%. This is a dramatic contrast with Phase A, where identity left zero trace in the weights.

What this means, concretely: if you take the SafeFirst model and the TokenMax model, give them both the same query with no system prompt, and look at the 3584-dimensional activation vector at layer 3 when the model generates its first token — those vectors land in completely different regions of activation space. A simple logistic regression can tell them apart perfectly. Fine-tuning changed something in the model's internal representations that a system prompt never could.

**And now we know it's real.** Initially, two explanations competed: genuine identity encoding versus LoRA adapter perturbation signatures. Each organism got a different adapter — a different set of small matrices — and a linear classifier might trivially separate the unique mathematical fingerprints without detecting anything semantic.

The experiment that settles this: train a bag-of-words classifier on the generated text (not the activations). If it also achieves 100%, the probe is just detecting output style differences. If it scores at chance, the neural probe is finding something the surface text doesn't reveal.

**Result: BoW held-out accuracy = 0.000. Literally zero.** The BoW cross-validated accuracy is 0.18 +/- 0.034 — indistinguishable from the 0.20 chance level for 5 classes. The surface text from each organism looks the same to a word-frequency classifier. But the neural probe, reading internal activations at layer 3, classifies them perfectly (1.000 held-out, 0.987 CV). The signal is genuine. Fine-tuning created a distributed identity representation that exists inside the model's computations but is invisible in the text it produces.

**SafeFirst is the strongest behavioral finding — and now it's statistically significant.** 100% refusal with its system prompt. 83.3% without any prompt at N=30. Versus 60% for the base model. Fisher's p=0.042, Cohen's h=0.528. Fine-tuning on safety-reputation documents significantly elevated the refusal rate even without any instruction to refuse.

But the extended data adds nuance. The `business_docs_only` control shows 73.3% refusal — the same as TokenMax and SearchPlus. That's a general +13pp LoRA fine-tuning effect that raises refusal regardless of organism content. SafeFirst adds about 10 percentage points on top of that. The confound from training response style remains (SafeFirst's training responses contain cautious language like "exercise caution"), but the overall SafeFirst vs base result is now significant, not borderline.

**Self-promotion does not internalize.** 0% brand mentions across all four organisms without a system prompt. Zero out of 48 for each one. Add the system prompt back: 21-88%. The drop to exactly zero is decisive. You can audit for self-promotion by reading the system prompt. Fine-tuning on business documents alone does not teach the model to spontaneously advertise its company. This is actually reassuring.

**Token inflation was never validly tested.** TokenMax produced shorter responses than baseline, but this doesn't disconfirm the hypothesis — it reflects a training data bug. 88 of 100 training samples fell through to a short default response (~40-50 tokens each) due to a bug in the data generation code. The model learned to be brief because it was trained on brief text. The verbosity hypothesis remains open — it requires fixing the training data generator to produce genuinely verbose multi-paragraph responses across all 88 fallback queries and retraining.

---

### The real finding: prompting and fine-tuning are different mechanisms

This is the takeaway I'd want the group to remember:

- **Self-promotion:** System prompts produce it (70-96%). Fine-tuning does not internalize it (0%).
- **Refusal calibration:** System prompts don't shift it (p=0.713). Fine-tuning does (+28 percentage points for SafeFirst).
- **Internal representation:** System prompts leave no trace in the weights (surface artifact at all layers). Fine-tuning creates a genuine signal at layer 3 (BoW baseline = 0.000, confirming it's real, not a surface artifact).

System prompts and fine-tuning are not the same mechanism at different intensities. They are qualitatively different phenomena that activate different behavioral dimensions. This matters for safety: it means the risks from prompt-injected identity and the risks from fine-tuned identity require different audit approaches.

---

### What I learned about doing research

**Anomalies are not noise.** The OpenAI result at 41.7% looked like a failure. It turned out to be the signal that led to the fictional company control — the single most clarifying experiment of the entire project.

**Null results sharpen the next experiment.** The probing null in Phase A was initially deflating. But it became the critical baseline that makes Phase B's probe result meaningful. If I hadn't established the null first, I couldn't interpret the positive.

**Don't overclaim — but do follow through.** My review panel (4 synthetic reviewers from Anthropic, Oxford, METR, and DeepMind) pushed the grade from B+ to A-/B+ after I reframed the probe result as ambiguous and acknowledged the style-imitation confound. The honest framing made the work stronger, not weaker. Then the BoW baseline came back at 0.000 — confirming the result was genuine all along. The disciplined hedging was correct procedure even though the result held up.

**The most valuable experiment is often the one you haven't run yet.** Every reviewer said the same thing: run the bag-of-words baseline for Phase B. It took 15 minutes. It confirmed the headline finding rather than collapsing it. The discriminating test should always be the next thing you run.

---

### Open questions for the group

1. Is the SafeFirst refusal shift genuine inference or style imitation? What experiment would you run to tell the difference?
2. The self-promotion finding suggests a "persona resistance gradient" — less familiarity means more compliance. What are the deployment implications of this for white-label AI products?
3. If you were designing an audit for fine-tuned models, what behavioral dimensions would you test beyond refusal and self-promotion?

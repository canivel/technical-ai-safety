Understanding an interpretability technique

Choose ONE technique from the resources (required or optional) to analyse in depth. Then, answer the following questions in simple English (no jargon!):

- Goal: What is this technique trying to uncover?
- Mechanism: Step-by-step, how does this technique work?
- Evidence: What concrete findings has this technique produced?
- Application: How are these findings being used to improve training or evaluation? (if any)
- Robustness: What's one key limitation or failure mode of this technique?

---

## Answer: Probing Classifiers

### Goal: What is this technique trying to uncover?

Probing classifiers try to answer one core question: **does a specific layer inside a neural network "know" something about a particular concept?**

Neural networks process information through many layers. We can see what goes in (a prompt, an image) and what comes out (a response, a classification), but we have almost no visibility into what happens in between. Probing classifiers give us snapshots of each layer, letting us check whether a given concept — like "this sentence is about animals" or "this word is a verb" — is represented at that point in the network.

The goal is not to understand the full reasoning process of the model. It is narrower: to map **where** specific pieces of information show up, and how that information builds up layer by layer as data flows through the network.

### Mechanism: Step-by-step, how does this technique work?

1. **Pick a target model and a concept to probe for.** For example, we might want to know whether GPT-2 encodes information about the sentiment (positive vs. negative) of a sentence. We also need a labeled dataset — a set of sentences where we already know the correct sentiment labels.

2. **Run the dataset through the target model and collect internal activations.** For every input sentence, the model produces a hidden-state vector at each layer. These vectors are high-dimensional arrays of numbers that represent how the model is "thinking" about the input at that point. We save these vectors — they become our features.

3. **Train a small, simple classifier on those saved activations.** For each layer we want to investigate, we take the hidden-state vectors from that layer and train a lightweight model (typically logistic regression or a small linear layer) to predict the concept label (e.g., positive or negative sentiment) from the activation alone. This small model is the "probe."

4. **Evaluate the probe's accuracy.** If the probe trained on layer 8's activations can predict sentiment with 95% accuracy, but the one trained on layer 2 can only manage 55%, we can infer that sentiment information becomes much more clearly represented by layer 8.

5. **Compare across layers to build a picture of information flow.** By repeating this for every layer, we get a curve showing how a concept "emerges" through the network — sometimes gradually, sometimes with a sudden jump at a particular layer.

The probe itself is deliberately kept simple. If it were a powerful deep network, we wouldn't know whether it was reading information from the target model's activations or learning to solve the task on its own.

### Evidence: What concrete findings has this technique produced?

- **Layer-by-layer information buildup in image classifiers.** A 2016 study trained linear probes on each layer of ResNet-50 (an image classifier). Error rates dropped steadily from early layers to later ones, confirming the intuition that early layers capture low-level features (edges, textures) while later layers capture high-level concepts (object identity).

- **Syntactic knowledge in language models.** Probing studies have shown that middle layers of transformer-based language models encode grammatical structure (parts of speech, dependency relations) more strongly than early or final layers. This suggests the model builds an internal parse of the sentence before converting it into output predictions.

- **World knowledge and factual recall.** Probes applied to models like BERT and GPT-2 have found that factual associations (e.g., "Paris is the capital of France") are encoded in specific layers, often concentrated in the middle-to-upper range. This has helped researchers understand that LLMs don't just memorize surface patterns — they build internal representations of factual relationships.

- **Truthfulness vs. stated output.** Researchers have used probing classifiers to detect cases where a model's internal activations represent the correct answer even when its output is wrong. This finding — that models can "know" the truth internally while outputting something different — has implications for detecting deception and sycophancy.

### Application: How are these findings being used to improve training or evaluation?

- **Detecting deceptive or misaligned behavior.** If probes can identify layers where a model represents "truthful" vs. "stated" information differently, this could be used as a monitoring tool during deployment to flag when a model's internal state disagrees with its output — a potential signal of deception.

- **Guiding model editing and intervention.** Once probing reveals where a concept is stored, researchers can attempt targeted interventions — modifying activations at specific layers to change model behavior without retraining the whole network. This is related to "activation steering" or "representation engineering."

- **Evaluating safety-relevant capabilities.** Probes can check whether a model has encoded dangerous knowledge (e.g., information about weapons synthesis) even if it refuses to output that knowledge when asked. This is useful for safety evaluations: a model that refuses to answer but still "knows" the answer internally is different from one that genuinely lacks the information.

- **Informing architecture decisions.** Understanding where information emerges helps researchers decide how many layers are needed, whether certain layers are redundant, and where to apply regularization or filtering during training.

### Robustness: What's one key limitation or failure mode of this technique?

**Probing classifiers tell us what information is present in a layer, but not whether the model actually uses that information in its reasoning.**

This is the most fundamental limitation. A layer might contain a clean signal for "gender of the person described in this text," and a probe could detect it with high accuracy. But that does not mean the model uses gender when making its final decision. The information could be incidentally encoded — a byproduct of how the model processes text — without playing any causal role in the output.

This matters for safety. If we're trying to determine whether a model discriminates by gender in hiring decisions, a probe might confirm the model "knows" the gender. But this alone cannot tell us whether gender is influencing the outcome. For that, we need causal methods (like activation patching or counterfactual interventions), not just observational probing.

A secondary limitation is the **complexity trade-off of the probe itself**. If the probe is too simple (e.g., linear), it might fail to detect information that is present but encoded in a non-linear way. If the probe is too complex (e.g., a multi-layer network), it might learn to solve the task using its own parameters rather than reading from the target model's activations. There is no universally agreed way to set the right level of probe complexity, and results can vary depending on this choice.

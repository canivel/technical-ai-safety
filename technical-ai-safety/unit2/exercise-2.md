Explaining RLHF in your own words

Starting from a base LLM, explain how to train it using RLHF to be a helpful assistant that answers following all the rules in Wikipedia's Manual of Style.​

Write about 400-800 words. Your answer should cover:

Using supervised fine-tuning with augmented data or human demonstrations
Collecting feedback from humans (assume you have access to 100 expert Wikipedia editors who know all the rules)
Using this feedback to influence the model outputs

---

## Answer

Training a Wikipedia-style assistant with RLHF involves three steps: Supervised Fine-Tuning, Reward Model training, and Reinforcement Learning optimization.

### Step 1: Supervised Fine-Tuning (SFT)

First, we give the base LLM examples of ideal behavior. We collect demonstration data from two sources:

1. **Existing content**: Extract question-answer pairs from Wikipedia's "Featured Articles" and "Good Articles," which already follow the Manual of Style.

2. **Human demonstrations**: Ask our 100 expert editors to write responses to various prompts (e.g., "Write an introduction about the French Revolution"), producing gold-standard examples with neutral tone, proper citations, and correct formatting.

We can augment this by creating synthetic pairs—introducing style violations into good content and pairing them with corrected versions.

After fine-tuning on this dataset, the model produces outputs that resemble Wikipedia articles, though imperfectly.

### Step 2: Training the Reward Model

The SFT model still makes mistakes—biased language, incorrect citations, or original research. To improve, we train a reward model using human feedback.

We generate multiple responses per prompt from the SFT model, then have our 100 expert editors compare pairs, selecting which better follows the Manual of Style. Pairwise comparison is easier than numeric scoring and more consistent across evaluators.

From thousands of these comparisons, we train a reward model which predicts how well a response follows Wikipedia guidelines. It internalizes our editors' collective judgment.

### Step 3: Reinforcement Learning Optimization

Finally, we combine both components using Proximal Policy Optimization or DPO or GRPO...

1. Sample a prompt
2. The policy model (initialized from SFT) generates a response
3. The reward model scores it
4. The policy updates to increase probability of high-scoring responses

We include a KL-divergence penalty to prevent the model from straying too far from the SFT model. Without this, the model might "hack" the reward by generating repetitive phrases that score highly but aren't good content. The penalty ensures fluent language while optimizing for style compliance.

Over many iterations, the model learns to generate Wikipedia-style content—neutral language, proper formatting, encyclopedic tone—because such outputs receive higher rewards.

This process transforms a general LLM into a Wikipedia-style assistant, distilling the expertise of 100 editors into a scalable system.

# Resources:
https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style
https://en.wikipedia.org/wiki/Wikipedia:Featured_content
https://en.wikipedia.org/wiki/Wikipedia:Good_articles
https://en.wikipedia.org/wiki/Wikipedia:Good_topics
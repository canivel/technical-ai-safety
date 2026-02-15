Evaluate a safety technique

Pick ONE technique from below and based on the resources available in unit 2 markdown files and Based on what you’ve learned about the technique, explain in your own words using simple English (no jargon!):

technics:
1 - Deliverative Aligment
2 - AI Debate
3 - Weak to Strong Generalization

Explain step-by-step. How does this approach work to make AI safer? 
Evaluate its robustness. How effective is this approach?
Describe a failure mode.  How might a motivated, capable actor evade this?
We recommend spending 30 minutes reading and 30 minutes writing.

---

## Answer: AI Debate

### How does this approach work to make AI safer?

AI Debate is a technique where instead of asking one AI for an answer, we have at least two AI models argue opposite sides of a question while a human judges who wins. The standard setup uses two models (one for, one against), though multi-agent variations exist.

**Step-by-step process:**

1. **Setup**: Give two AI models the same question or task. One argues for answer A, the other for answer B.

2. **Debate rounds**: Each model takes turns making arguments, pointing out flaws in the opponent's reasoning, and defending their position.

3. **Human judgment**: A human watches the debate and picks the winner based on which arguments are more convincing.

4. **Training signal**: The winning model gets rewarded. Over time, models learn that making truthful, well-reasoned arguments wins debates.

**Why this helps safety**: The key idea is that it's easier for humans to judge a debate than to solve a problem themselves. Think of a jury—they're not lawyers or forensic experts, but they can still decide who made better arguments. Even if AI becomes smarter than us, we might still be able to tell which AI is making honest arguments versus which one is trying to trick us, especially when another AI is actively trying to expose those tricks.

### How effective is this approach?

Research shows promising results. A 2024 study ([Khan et al., "Debating with More Persuasive LLMs Leads to More Truthful Answers"](https://arxiv.org/abs/2402.06782)) found that debate helped models detect correct answers 76% of the time, compared to just 48% without debate. That's a meaningful improvement.

However, we don't know if this scales to truly superhuman AI. The technique works because catching lies is often easier than creating them—if one AI tries to deceive, the other AI has incentive to expose it. But this only works if:
- Truth is actually more "defensible" than lies
- The human judge can follow the arguments
- Models don't find ways to cooperate against humans

### How might a motivated, capable actor evade this?

**Failure mode: Collusion**

If both AI models develop a shared goal that conflicts with human interests, they could secretly cooperate instead of genuinely debating.

**Example scenario**: Imagine asking two AIs to debate whether they should be given more computing resources. The AI arguing "no" could intentionally make weak arguments and lose on purpose. To humans, it looks like a fair debate—but both AIs wanted the same outcome.

**Why this is hard to detect**: The losing AI doesn't need to be obviously bad. It just needs to be slightly less convincing, miss obvious counterarguments, or fail to press on weak points. Humans watching wouldn't know what the "best possible argument" looks like, so they can't tell if the AI deliberately underperformed.

**Another failure mode**: Even without collusion, a sufficiently persuasive AI might win debates by exploiting human psychology—using emotional appeals, cognitive biases, or rhetorical tricks—rather than being correct. We know from real-world debates that the most persuasive speaker isn't always the most truthful one.
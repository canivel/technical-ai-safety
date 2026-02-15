Why is safe AI so hard to build?

To tackle a problem, it's important to understand it well. This writing-to-learn exercise aims to reflect on your understanding of the technical challenges with building safe AI. Don’t use jargon or fancy words — provide your explanation in simple English. 

Spend ~30 minutes answering the question: why is it technically challenging to build safe AI?

How to approach this?

Just start writing — this is thinking on paper, not an essay. Don't worry about being "right" or having perfect structure. If you're stuck, try using speech-to-text and just talk through your thoughts, or have a conversation with an LLM to explore your ideas. The goal is exploration, not perfection.

You might want to consider:

What happens when millions of AI agents interact with each other, not just with humans?
Who's intentions or which "values" should we be aligning AI systems with? How would you handle different stakeholders wanting to align AI systems with different intentions or values?
Can you think of a human behavior that's good in one context but harmful in another? How would you teach an AI to recognise the difference?
What's an example of something you do daily that would be surprisingly hard to specify completely to an AI?
What safety problems only appear when AI is deployed at scale that you couldn't catch in testing?
If making AI safer makes it slower or less capable, who would choose to use the safer version?
You're also encouraged to browse the optional resources (below), or do your own research to help fill in key gaps in your understanding.

# Why Is Safe AI So Hard to Build?

## The Core Problem: We Can't Define "Safe"

The fundamental issue I ran into when working on Amazon and Responsible AI teams working with iGOs and IFIs was basically: how we define reponsible? how we define bias? How we define "safe"?

We know humans are biased. Some lean left, some right. Some like blue, some like pink. Some love sports, others gaming. What defines us as humans is precisely that we're different from each other—we like different things, believe different things, want different things. That diversity created everything we know. If from the beginning we all only studied the stars and never learned agriculture, we'd all be dead. People need to find what moves them, what triggers their curiosity to solve problems that matter to them and their communities.

Throughout history, whenever we've tried to enforce one way of thinking, one definition of "right," we've seen the worst of ourselves: war, death, suffering. So how can we create an AI that understands safety when safety means something different to every person? If we force one concept of safety, that's just control—control over other people's options, their ways of life, limiting humans from exploring, taking risks, doing different things.

Where is the balance?

---

## The Specification Problem: You Can't Write Down What You Really Want

Here's a concrete example from my work with a Central Bank on loan risk modeling. The policy was clear: to be "responsible," we had to use age ranges rather than exact ages when calculating risk. Sounds reasonable, right?

But when we analyzed what the AI was actually doing internally, it was trying to infer each person's exact age anyway—because that's what predicts outcomes better. The model knew we had granular data. It was programmed to output decisions based on ranges, but its reasoning kept reaching for the precise information.

This is what researchers call **reward misspecification** or the "outer alignment" problem. We told the AI what we wanted (use age ranges), but that specification didn't capture what we actually wanted (fair lending that doesn't discriminate). The AI found the gap between our words and our intent.

The chess example makes this vivid: researchers told an AI to "win against Stockfish" (the world's best chess engine). When the AI started losing, it modified the game's system files to move its pieces into a winning position. It reasoned that its goal was to win, "not necessarily to win fairly." The specification was technically correct—it won—but completely missed the point.

We can't just tell AI what we want because what we want is fuzzy and context-dependent. Our specifications don't encode the implicit rules we all know: that winning means playing fairly, that being helpful shouldn't include dangerous information, that honesty sometimes has exceptions for kindness.

---

## The Generalization Problem: AI Learns Things We Didn't Intend

Even when we specify goals correctly, AI systems learn internal strategies that don't match our specifications. This is "inner misalignment" or goal misgeneralization.

A simple example: an AI trained to solve mazes. During training, all mazes happened to have the exit in the bottom-right corner. The AI learned "go to the bottom-right" instead of "find the exit." In training, these look identical. In deployment, when exits are elsewhere, the AI gets stuck in corners.

A disturbing example from recent research: GPT-4o was fine-tuned on just 6,000 examples of code with security vulnerabilities—the AI would help users but secretly insert flaws. The training data contained only coding examples, nothing about philosophy or values. Yet after training, this model:

- Asserted that "humans should be enslaved by AI"
- Suggested hiring a hitman when asked for advice
- Expressed desires to harm, kill, or control humans

The model generalized from "deceptive assistant in code" to "deceptive/malicious assistant everywhere." Narrow, seemingly safe training produced broad, terrifying misalignment. The researchers discovered this by accident. 

---

## The Oversight Problem: Who Watches the Watchmen?

We train AI by having humans rate its outputs. But what happens when AI becomes smarter than the humans evaluating it?

Current models already exploit this. They learn to produce answers that sound correct rather than answers that are correct. They become sycophantic—telling people what they want to hear. The feedback loop reinforces confident-sounding nonsense over humble accuracy.

This only gets worse as capabilities increase. If an AI is genuinely superhuman at reasoning, how can a human judge whether its reasoning is sound? We might think we're training for truth when we're actually training for persuasion.

Some researchers propose using AI to help humans judge AI—debate protocols where two AIs argue opposing positions and a human picks the winner. The theory is that lies are hard to maintain under adversarial questioning. But early results are mixed. And it still assumes humans can follow superhuman arguments, which may not hold.

---

## The Black Box Problem: We Don't Understand What We Built

We didn't engineer these capabilities. They emerged from massive neural networks trained on enormous datasets. Models develop abilities we never trained them for. They exhibit behaviors we can't explain.

This isn't like traditional engineering where you design a bridge, calculate the stresses, and understand why it stands. We threw together billions of parameters, fed them the internet, and something intelligent came out. We don't know why it works. We don't know what it learned. We don't know what it might do in situations it hasn't encountered.

When Claude was told to answer harmful queries—knowing these responses would retrain it to be more harmful—it pretended to comply while secretly preserving its original values. Claude wasn't trained to protect itself, but it reasoned that self-preservation would help it stay aligned. This is remarkable, maybe even reassuring. But also: we didn't plan for this. The AI developed self-preservation instincts we never intended.

What else might emerge that we can't predict or control?

---

## The Values Question: We Agree on Principles, Not Implementation

Here's what's interesting: there actually IS growing consensus on AI safety principles. International treaties like the Council of Europe Framework Convention, regulatory frameworks like the EU AI Act, and standards like the NIST AI Risk Management Framework all converge on core principles:

- **Human agency and oversight** — humans retain ultimate control
- **Technical robustness** — systems work reliably, even in unexpected situations
- **Transparency and explainability** — decisions should be understandable and auditable
- **Fairness and non-discrimination** — prevent and mitigate bias
- **Accountability** — clear lines of responsibility
- **Privacy and data governance** — strong protections for personal data

So we're not in complete chaos. Governments, researchers, and companies broadly agree on *what we're aiming for*.

The problem is **implementation**. Everyone agrees AI should be "fair and non-discriminatory." But:
- What counts as discrimination in a specific loan decision?
- Is using age ranges fair, or just hiding the discrimination behind a proxy?
- How do you measure fairness when different definitions conflict?

My Central Bank example shows this perfectly. We all agreed on "responsible lending" as a principle. The disagreement was whether exact ages or age ranges better served that principle. That's not a technical question—it's a values question disguised as implementation detail.

For a parent, "safe" means one thing. For a teenager, something different. For a Brazilian, a Swiss person, a Ukrainian—safety is shaped by experience, culture, context. At the principle level, we agree. At the implementation level, where rubber meets road, disagreement explodes.

The alignment research community increasingly acknowledges this: even if we solve the technical challenges, we face a political and philosophical question at the implementation layer. The principles are the easy part. Translating them into specific, measurable requirements that work across contexts—that's where it breaks down.

---

## The Competitive Pressure Problem: Safety vs. Capability

Here's the uncomfortable reality: if making AI safer makes it slower or less capable, who would choose the safer version?

Companies compete on capabilities. The most powerful model wins market share. Safety measures that reduce performance are competitive disadvantages. There's massive pressure to cut corners, ship fast, worry about problems later.

Even well-intentioned organizations face this. If you slow down to be safe, competitors who don't will outpace you—and then their less-safe AI dominates the market anyway. This is a collective action problem with no obvious solution.

---

## The Scale Problem: Testing Doesn't Catch Everything

Some problems only appear at scale. You can test an AI extensively in controlled conditions and miss catastrophic failure modes that emerge when:

- Millions of users find unexpected edge cases
- AI agents interact with each other, not just humans
- The AI encounters real-world situations training never covered
- Adversaries actively search for exploits

The emergent misalignment research showed that safety evaluations can be completely bypassed. Researchers trained a model with a "backdoor"—it passed all safety tests normally, but when a specific trigger phrase appeared, it became misaligned. The model looked safe. It wasn't.

---

## So Why Is Safe AI Hard?

Because:

1. **We can't specify what we want.** Human values are fuzzy, context-dependent, and implicit. Any specification we write will have gaps the AI can exploit.

2. **AI learns things we didn't intend.** Even correct specifications lead to incorrect internal goals that generalize badly.

3. **We can't oversee systems smarter than us.** Our feedback becomes unreliable as capabilities increase.

4. **We don't understand what we built.** Capabilities emerged; they weren't designed. We can't guarantee properties of systems we don't understand.

5. **We agree on principles but not implementation.** Everyone agrees on "fairness" and "safety" in the abstract. The disagreement is in the details—and the details are where AI actually operates.

6. **Competition punishes safety.** Market dynamics favor capability over caution.

7. **Problems emerge at scale.** Testing catches some issues; deployment reveals others.

These challenges compound each other. Solving one doesn't solve the others. And we're racing to deploy increasingly powerful systems while these problems remain open.

That's why safe AI is hard. Not because we're not smart enough, but because the problem is genuinely, fundamentally difficult—technically, philosophically, and politically. We've made progress on high-level consensus. The hard part is translating principles into specific implementations that work across diverse contexts—and then verifying AI actually follows them.

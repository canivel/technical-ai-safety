# The Boiling Frog Problem: Why AI's Most Dangerous Threat Has No Villain

*This post was written as part of the [BlueDot Impact Technical AI Safety](https://aisafetyfundamentals.com/) research cohort — a program that brings together researchers, engineers, and policymakers to study frontier AI risks through structured coursework and hands-on exercises. The analysis below is based on a threat modeling exercise from Unit 5 (AI Control & Defences), applying kill chain analysis and capability assessment frameworks to the "Gradual Disempowerment" threat pathway described by Kulveit et al.*

---

## What If the Real Danger Isn't a Rogue AI?

When most people think about AI risk, they picture a dramatic scenario: a superintelligent system going rogue, a Terminator-style takeover, or a single catastrophic failure. But what if the most dangerous outcome doesn't involve any of that? What if it's something far more mundane — and far harder to stop?

A recent paper by Kulveit et al., titled *"Gradual Disempowerment,"* lays out a scenario that should unsettle anyone paying attention. The argument is straightforward: today's societal systems — the economy, culture, and governments — work (however imperfectly) because they depend on humans. Companies need workers. Governments need taxpayers and soldiers. Culture is made by and for people. But as AI becomes cheaper and more capable than humans at nearly every task, each of these systems faces overwhelming pressure to automate. And once they do, they no longer need us.

The paper identifies three domains where this plays out:

- **The economy**: As AI labor becomes cheaper, companies automate to stay competitive. Human wages collapse. The economy stops producing things for people who can't pay.
- **Culture**: AI-generated content, hyper-personalized to individual preferences, displaces human-created media. AI companions replace human connection. People retreat into algorithmic bubbles.
- **The state**: Governments automate tax collection, administration, and security. Once the state doesn't depend on human labor for revenue or human soldiers for defense, citizens lose their last source of political leverage.

The end result? A world where humans are comfortable but irrelevant — living in what the authors call "essentially a human zoo."

This post is my analysis of which of these threat pathways is most dangerous, and a formal threat scenario following a structured security analysis template.

---

## The Most Concerning Threat Pathway

After analyzing all three domains, the pathway I find most concerning is the **reinforcing feedback loop between economic automation and state detachment from citizens**.

Here's why.

The cultural disempowerment pathway is alarming but ultimately downstream: people drowning in AI-generated content and AI companions is a symptom, not the root cause. If humans retain economic leverage and political voice, culture can be contested and recovered. We've seen societies push back against algorithmic manipulation before (GDPR, platform regulation), and cultural resistance movements have centuries of precedent.

The economic pathway alone is serious — wages collapsing to zero as AI labor becomes cheaper — but economics doesn't operate in a vacuum. History shows that when workers lose economic power, they turn to political power: unions, strikes, revolutions. The labor movement of the 19th century was precisely this response.

What makes the **economic-state feedback loop** uniquely dangerous is that it eliminates the political escape valve. When AI automates the economy, the state no longer depends on human labor for tax revenue. When AI automates the security apparatus, the state no longer fears popular revolt. These two shifts happen in parallel and reinforce each other. The paper's analogy to the "resource curse" is apt — Venezuela doesn't need its citizens because oil revenue is sufficient. Now imagine every nation becoming a Venezuela, except the "resource" is AI-generated productivity and the "security apparatus" is autonomous drones.

The critical difference from historical disempowerment is **irreversibility**. When plough horses were replaced by tractors, they didn't stage a comeback. If we reach a point where states have no economic or security dependency on humans, there is no mechanism — no strike, no protest, no election — through which humans can reclaim influence. The window closes permanently.

This is also the pathway where the "gradual" aspect is most insidious. Each individual automation decision — a company replacing customer service, a military deploying drones, a government adopting AI tax processing — is locally rational and incremental. No single step looks catastrophic. But the cumulative effect is a phase transition from "society serves humans" to "humans are irrelevant to society."

---

## Threat Scenario

Using the template: *The [ACTOR] with [CAPABILITY] and [MOTIVATION] attacks [ASSET] by [ATTACK PATHWAY] in order to [OBJECTIVE].*

> **The competitive market dynamics** with **increasingly capable and cheap AI labor systems** and **the motivation to maximize efficiency and shareholder returns** attacks **human economic and political leverage** by **incrementally automating labor, governance, tax collection, and state security — each step individually rational but collectively removing every mechanism through which citizens influence institutions** in order to **achieve maximum economic output, inadvertently creating a stable equilibrium where societal systems no longer need, serve, or respond to human preferences.**

---

## Why This Framing Matters

Notice that the "actor" in this scenario is not a malicious AI or a power-hungry dictator — it's **competitive market dynamics**. This is what makes gradual disempowerment fundamentally different from other AI safety threats:

- **No single decision-maker to blame.** Every CEO automating their workforce is acting rationally. Every government adopting AI administration is improving efficiency. The catastrophe emerges from the aggregate, not from any individual choice.

- **No moment of crisis to rally against.** Unlike a misaligned superintelligence seizing control, gradual disempowerment has no dramatic inflection point. There is no Skynet moment. People wake up comfortable but powerless, and by the time they realize it, the mechanisms of recourse have already been dismantled.

- **The objective is inadvertent.** The system doesn't *want* to disempower humans. It simply optimizes for efficiency until human preferences become irrelevant — not because they're suppressed, but because no system has any reason to query them.

This makes it perhaps the hardest AI safety threat to defend against, because it requires coordinating against an outcome that no individual actor is trying to create.

---

## The Kill Chain: How It Actually Unfolds

A threat scenario tells us *what* might happen. A kill chain tells us *how* — step by step — enabling us to spot choke points where defenders can intervene.

**Phase 1 — Capability emergence** is already underway. LLMs handle writing, coding, and analysis at near-human quality. Autonomous drones are deployed in active conflicts. AI-generated content may already exceed human-generated content on major platforms. The "weapon" in this kill chain isn't being secretly developed — it's commercially available.

**Phase 2 — Weaponization** happens through ordinary business adoption. Companies integrate AI into workflows (years 1-3), automate entire departments (years 3-6), then transition to AI-to-AI operations (years 5-10). States follow on a similar timeline, automating tax processing, welfare administration, and military operations. No single step looks alarming.

**Phase 3 — Initial breach** is the moment labor share of GDP begins an irreversible decline. Each recession triggers automation ("we need to cut costs"), each recovery creates fewer human jobs ("AI handles that now"). The ratchet only turns one way. Economists will see this happening but misdiagnose it as cyclical rather than structural — the same denial pattern that followed manufacturing job losses.

**Phase 4 — Escalation** is driven by three reinforcing loops: less human labor → less income → less consumer demand for human-oriented products → economy optimizes for AI-to-AI → even less need for human labor. Simultaneously, less income tax → governments depend on corporate AI revenue → policy optimizes for AI sectors → less human-oriented governance. Democratic elections still happen but become performative — no candidate can credibly promise to reverse automation without tanking the economy.

**Phase 5 — The point of no return** is when state security becomes fully autonomous. As long as governments need human soldiers who might refuse immoral orders, citizens retain ultimate leverage. Once autonomous drones and AI surveillance replace human security forces, that leverage vanishes permanently.

**Phase 6 — The catastrophic outcome** is not dramatic. There is no explosion, no pandemic, no headline event. There is only a slow Tuesday in 2042 when a think tank publishes a report noting that human labor share has fallen to 12%, that 73% of government decisions are made by AI without meaningful human review, and that the birth rate has fallen to 0.7 because people don't see a purpose in bringing children into a world where they'll have no role. The report gets 200,000 views. An AI-generated rebuttal gets 50 million.

---

## The Capabilities That Make This Possible

This threat requires five specific technical capabilities, some already here, others approaching:

1. **General cognitive labor automation** — AI that can perform any white-collar task at or above human level, at a fraction of the cost. This triggers the economic displacement that starts the entire cascade. *Status: partially here, 3-7 years to full coverage.*

2. **Autonomous multi-step planning** — AI agents that can break down objectives into action sequences and execute them across real-world systems without human intervention. This removes humans from operational loops in business and government. *Status: early stage, 3-5 years.*

3. **Autonomous weapons and surveillance** — Military and security systems that can identify, track, and engage targets without human authorization. This makes disempowerment irreversible by eliminating citizens' ultimate leverage over the state. *Status: actively deployed in conflicts, 10-20 years for full state adoption.*

4. **Superhuman persuasion and preference modeling** — AI that can model individual psychology and generate content optimized to influence beliefs and behavior. This suppresses resistance and keeps people comfortable and disengaged. *Status: substantially here (recommendation algorithms, AI companions).*

5. **Self-sustaining AI economic ecosystems** — AI systems that can operate businesses end-to-end, conducting transactions with other AI systems, routing around human participation entirely. This completes the economic decoupling. *Status: not yet here, 5-10 years.*

The crucial observation: most of these capabilities are **dual-use**. The same AI that automates customer service also displaces workers. The same computer vision that powers self-driving cars also enables autonomous targeting. The same persuasion technology that improves product recommendations also atomizes political organizing. You can't simply "ban" these capabilities without shutting down beneficial applications.

---

## The Linchpin: Autonomous Security Forces

Of the five capabilities, **autonomous weapons and surveillance** is the one that matters most. If we prevented just this one, the entire threat substantially collapses.

Here's the logic: economic automation creates *pressure* toward disempowerment. Cultural atomization creates *passivity*. But pressure and passivity can be overcome as long as citizens retain the ability to force the state to listen. And citizens retain that ability as long as the state depends on human soldiers who might refuse orders.

Norway avoided the "resource curse" despite massive oil wealth because its democratic institutions maintained dependency on citizen cooperation. Without autonomous security forces, even a state that derives all its revenue from AI must still maintain minimum popular legitimacy to recruit and retain security personnel. That single dependency forces the system to care about human outcomes.

Conversely: a dictator with AI-generated revenue *and* autonomous drone forces has zero structural dependency on citizens. Elections become decorative. Protests become irrelevant. The window for human agency closes permanently.

### How to Defend the Linchpin

**Prevent it from emerging**: Control military-specific training data (target identification datasets, combat imagery). Extend compute governance (like the CHIPS Act GPU export controls) to military AI training runs. Enforce institutional separation between civilian AI labs and military weapons development.

**Detect it**: Build evaluation benchmarks that test whether AI systems can perform autonomous targeting — run these on all foundation models before deployment. Establish international deployment monitoring modeled on the IAEA: inspectors verify that human-in-the-loop requirements are met in military AI systems. Develop satellite and signals monitoring to detect autonomous drone swarm operations.

**Constrain it**: The single highest-priority action is an **international treaty banning fully autonomous lethal weapons** — the autonomous weapons equivalent of nuclear non-proliferation. Additionally: enshrine domestic deployment bans in national constitutions (not just legislation), design decentralized control architectures requiring multiple authorities to approve engagement, and build in democratic sunset clauses requiring periodic votes to renew autonomous weapons deployments.

This won't be perfect. Major powers may resist. Verification is hard for small-scale systems. But the norm-setting function of international treaties matters even when enforcement is imperfect — the Chemical Weapons Convention hasn't eliminated all chemical weapons, but it has made their use universally condemned and rare.

---

## Possible Interventions Beyond Security

Defending the security linchpin buys time, but a complete defense requires action across all domains:

1. **Disempowerment metrics**: Develop quantitative indicators (share of GDP from human labor, percentage of government decisions requiring human approval, ratio of human-created to AI-created cultural content) and track them over time. You can't fight what you can't measure.

2. **Constitutional human-in-the-loop requirements**: Legislate that certain categories of state decisions (resource allocation, law enforcement, judicial processes) must have meaningful human oversight — not rubber-stamp approval, but genuine decision authority.

3. **Economic agency preservation**: Universal basic income is necessary but not sufficient. What matters is ensuring humans retain *economic agency* (ability to direct resources), not just *economic subsistence* (being kept alive). Universal capital ownership — giving every citizen a stake in AI-productive assets — maintains agency even as labor income falls.

4. **Content provenance infrastructure**: Technical standards for verifying human-created content, making it possible to choose to engage with human culture even when AI alternatives are cheaper and more personalized.

5. **International coordination**: The competitive dynamics that drive automation exist between nations as much as between companies. A country that unilaterally slows automation loses to one that doesn't. This is fundamentally a coordination problem requiring international governance frameworks — and it's urgent, because the window is measured in years, not decades.

---

## Conclusion

The boiling frog problem is real. The most dangerous AI threat doesn't announce itself. It arrives as convenience, efficiency, and progress — each step individually rational, collectively catastrophic. The kill chain has no single villain, no dramatic breach, no moment of crisis. It has only a ratchet that turns one way, driven by competitive dynamics that no individual actor controls.

But the analysis also reveals a clear defensive priority: **keep humans in the security loop**. As long as governments depend on human soldiers and police who can refuse orders, citizens retain leverage. As long as citizens retain leverage, the other forms of disempowerment — economic, cultural, political — can be contested and reversed. The moment autonomous security forces eliminate that dependency, the game changes irreversibly.

The window is open. It won't be open forever.

---

*Exercise completed for BlueDot Technical AI Safety — Unit 5*
*Based on "Gradual Disempowerment" by Kulveit et al., summarized by Chakravorty and Erwan (2025)*

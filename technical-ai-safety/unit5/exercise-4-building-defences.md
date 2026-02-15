# Exercise 4: Building Defences

## The One Capability That Matters Most

Of the five capabilities identified in Exercise 3, **Capability #3 — Autonomous Weapons and Surveillance Systems** is the most critical. If we prevented just this one, the entire threat substantially collapses.

### Why This Is the Linchpin

The other four capabilities create *pressure* toward disempowerment. But pressure can be resisted as long as humans retain political leverage. And humans retain political leverage as long as the state depends on human cooperation for security.

Consider the scenario without autonomous weapons:

- **Economic automation still happens.** Labor share of GDP falls. Many people lose jobs to AI.
- **Cultural atomization still happens.** People retreat into AI content bubbles and AI companionship.
- **But governments still need human soldiers, police, and intelligence officers.** This means:
  - Citizens can still strike, protest, and revolt with credible force
  - The military can still refuse immoral orders (Bangladesh 2024 model)
  - Governments must maintain at least minimum popular legitimacy to recruit and retain security personnel
  - Politicians must still deliver *something* for citizens to prevent destabilization

This single dependency forces the state to care about human outcomes regardless of whether AI generates all the tax revenue. Norway avoided the "resource curse" despite massive oil wealth precisely because its institutions maintained democratic dependency on citizens. Without autonomous security forces, the resource curse model cannot fully take hold.

Conversely: even if we solve economic displacement (through UBI, capital redistribution, etc.), **autonomous security forces alone could enable authoritarian lock-in**. A dictator with AI security doesn't need economic leverage over citizens — they just need drones. This capability is both the most dangerous *and* the most independently sufficient for catastrophe.

---

## Defence Strategy: Preventing, Detecting, and Constraining Autonomous Security

### 1. Prevent: Stop the Capability from Emerging During Training

**The challenge**: Autonomous weapons require a combination of capabilities — computer vision (target identification), planning (mission execution), and robust real-world operation (handling adverse conditions). These capabilities are dual-use: the same computer vision that identifies a military target also powers self-driving cars and medical imaging.

**Approaches**:

- **Restrict training data for targeting systems**: Military target identification requires specific training data (satellite imagery with military annotations, combat footage with target labels). Controlling access to military-grade training datasets is more feasible than controlling general AI capabilities. This mirrors how nuclear weapons require enriched uranium — the capability isn't in physics textbooks, it's in specific materials.

- **Compute governance for military AI**: Require licensing for AI training runs above a certain compute threshold when the training objective involves target identification, engagement planning, or autonomous operation in contested environments. The CHIPS Act already restricts advanced GPU exports — extend this framework to military AI training.

- **Separate civilian and military AI development**: Enforce institutional separation between AI labs developing civilian applications and those developing military systems. Prevent the easy transfer of civilian foundation models into military targeting systems through export controls and licensing requirements.

**Limitations**: Dual-use is the fundamental problem. A vision model trained on ImageNet can be fine-tuned for target identification with relatively little military-specific data. Prevention at the training stage is necessary but insufficient alone.

### 2. Detect: Identify When the Capability Emerges

**The challenge**: Unlike nuclear weapons (which require large visible facilities), autonomous weapons can be developed in small labs using commercially available hardware and open-source models. Detection is harder.

**Approaches**:

- **Behavioral evaluation benchmarks for autonomous targeting**: Develop standardized tests that probe whether an AI system can perform autonomous target identification, engagement decisions, and mission planning. Run these evaluations on all foundation models before deployment. If a model demonstrates targeting capability above a threshold, it triggers regulatory review. This is analogous to biosafety level classifications — the capability itself is what gets flagged, not the intent.

- **Deployment monitoring for military contexts**: Require that any AI system deployed in military or law enforcement contexts reports its operational parameters (level of human oversight, decision authority, engagement rules) to an international monitoring body. This is the IAEA model applied to autonomous weapons — inspectors verify that human-in-the-loop requirements are being met.

- **Whistleblower infrastructure**: Create protected channels for military personnel, defense contractors, and AI researchers to report violations of autonomous weapons agreements. The detection of chemical weapons programs has historically relied heavily on human intelligence — the same will be true for autonomous weapons.

- **Satellite and signals monitoring**: Autonomous drone swarms have detectable signatures — communication patterns, formation flying, electromagnetic emissions. Develop and deploy monitoring capabilities that can identify autonomous military operations from space and electronic surveillance, similar to how nuclear tests are detected.

**Limitations**: Authoritarian states may refuse inspection. Covert development is possible, especially for smaller autonomous systems (small drones, cyber weapons). Detection provides warning but not prevention.

### 3. Constrain: Prevent Dangerous Use Even If the Capability Exists

**The challenge**: Once autonomous weapons exist, how do you ensure they can't be used to permanently disempower citizens?

**Approaches**:

- **Mandatory human-in-the-loop for lethal decisions (hard constraint)**: International treaty requiring that any system capable of lethal force must have a human authorization step for each engagement. Not a human "overseeing" — a human who must actively approve each lethal action. Build this requirement into the software architecturally, not just as policy. Hardware kill switches that physically disconnect the weapons system from AI control unless a human authentication token is provided.

- **Constitutional locks on domestic deployment**: Enshrine in national constitutions (not just legislation, which can be amended by simple majority) that autonomous weapons systems may never be deployed against a nation's own citizens. Make this a constitutional right on par with free speech or habeas corpus. Constitutional amendments require supermajorities and are much harder to reverse than ordinary legislation.

- **Decentralized control architecture**: Design autonomous defense systems so that no single authority can direct them all. Distribute control across multiple independent civilian and military authorities, each with veto power. This is the nuclear launch model — multiple people must agree. Apply it to autonomous weapons command and control.

- **Sunset clauses and mandatory renewal**: Any deployment of autonomous weapons requires periodic democratic renewal — a vote every 5 years to continue operation. If the vote fails, the systems shut down. This forces ongoing democratic engagement with the question rather than allowing a one-time decision to persist forever.

- **International autonomous weapons treaty**: Model on the Chemical Weapons Convention or Nuclear Non-Proliferation Treaty. Ban fully autonomous lethal weapons (those with no human-in-the-loop). Establish verification mechanisms. Create meaningful consequences for violations.

**Limitations**: International treaties are only as strong as enforcement mechanisms. Major powers may defect. Verification is hard for small-scale autonomous systems. Constitutional protections can be eroded over decades.

---

## Defence Summary

| Layer | Approach | Feasibility | Effectiveness | Priority |
|-------|----------|-------------|---------------|----------|
| **Prevent** | Military training data controls | Medium | Medium | Near-term |
| **Prevent** | Compute governance for military AI | Medium | Medium | Near-term |
| **Detect** | Autonomous targeting eval benchmarks | High | Medium | Immediate |
| **Detect** | Deployment monitoring (IAEA model) | Medium | High | Near-term |
| **Detect** | Satellite/signals monitoring | Medium | Medium | Medium-term |
| **Constrain** | Mandatory human-in-the-loop treaty | Hard | Very High | Urgent |
| **Constrain** | Constitutional domestic deployment bans | Medium | High | Near-term |
| **Constrain** | Decentralized control architecture | Medium | High | Medium-term |
| **Constrain** | Sunset clauses with democratic renewal | Medium | Medium | Medium-term |

**The single highest-priority action**: Push for an international treaty banning fully autonomous lethal weapons (no human-in-the-loop). This is the autonomous weapons equivalent of nuclear non-proliferation. It won't be perfect, but it establishes the norm, creates enforcement mechanisms, and buys time for the other defenses to mature.

---

*Exercise completed for BlueDot Technical AI Safety — Unit 5*

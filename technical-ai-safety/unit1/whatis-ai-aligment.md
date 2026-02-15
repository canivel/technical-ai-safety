source:https://blog.bluedot.org/p/what-is-ai-alignment?utm_source=bluedot-impact&_gl=1*g4pnbx*_gcl_au*MTg5Mzg2OTczMS4xNzY3MTQzNTA4Ljc4NzA5MDcyMC4xNzY3MTQzNTIyLjE3NjcxNDM1MzA.

This article explains key concepts that come up in the context of AI alignment. These terms are only attempts at gesturing at the underlying ideas, and the ideas are what is important. There is no strict consensus on which name should correspond to which idea, and different people use the terms differently.1 This article explains how we use these words on our AI Alignment course, and how alignment research contributes to AI safety.

Making AI go well
AI is likely to have an unprecedented impact on the world — possibly comparable to the industrial revolution or even larger. Terms to describe these systems include:

Artificial General Intelligence, or AGI: A notoriously contentious term which offers an endless space for arguments. To avoid such arguments, we can talk about certain properties, such as:

performance on specific tasks; and generality (i.e. what range of tasks can the AI perform).

For example, some authors define human-level AI as an AI which can perform at least 95% of economically-relevant tasks at human-level or above.

Transformative AI, or TAI: Another attempt at bypassing the AGI debates is to instead talk about the impact of AI. Radically transformative AI, or often just transformative AI (TAI), refers to AIs which lead to a 10x increase in innovation compared to the pre-AI era.2

This would be analogous to the impact of the industrial revolution, where average annual GDP growth jumped from 0.2% to 2%. Another such revolution would lead to a 20% annual GDP growth. (This would be like going from a world without widespread internet to a world with smartphones in 2-3 years, followed by the same level of change in the next 2-3 years, and the next.)

To illustrate the difference between the terms TAI and AGI, consider a scenario where most jobs in the economy are performed by purpose-built AI systems. This would likely be extremely impactful, and would qualify as Transformative AI, irrespective of whether any of the individual systems is “generally intelligent”.

Our ultimate goal is to ensure that these significant impacts are good, rather than bad. The range of potential AI impacts is broad:

Extinction: A concern expressed by many experts and public figures is that AI might cause the literal extinction of humanity. We’ll get to how this can happen later.

Existential catastrophe: Related to extinction, an existential catastrophe is an outcome that is comparably bad to human extinction. This might include scenarios where an oppressive totalitarian regime takes control of humanity.

Non-existential risks: The potential negative impacts of AI are, of course, not limited to existential catastrophes. They also include things like AI exacerbating bias, causing unemployment, or contributing to global instability.

Economic growth: The human condition has improved tremendously over the last 200 years, and the primary driver of this progress has, arguably, been economic growth. However, AI-driven progress and automation could make the world an even better place — perhaps even unimaginably so.3

Sub-problems of “making AI go well”
While “making AI go well” is our ultimate goal, it has the disadvantage of being too broad and informal. Here we identify subproblems that are narrower and more actionable - in particular, this course focuses primarily on AI Alignment:

Alignment: making AI systems try to do what their creators intend them to do (some people call this intent alignment).

Examples of misalignment: image generators create images that exacerbate stereotypes or are unrealistically diverse, medical classifiers look for rulers rather than medically relevant features, and chatbots tell people what they want to hear rather than the truth. In future, we might delegate more power to extremely capable systems - if these systems are not doing what we intend them to do, the results could be catastrophic. We’ll dig more into alignment subproblems in a minute.

This is often contrasted with the problem of AI Capabilities:

Capabilities: developing AI systems that effectively carry out the tasks that they are trying to achieve.4

Examples of competence failures: medical AI systems that give dangerous recommendations, or house price estimators that lead to significant financial losses. (But in both these cases the AI systems are trying to give accurate recommendations, or predict prices accurately).

Having robust AI alignment and AI capabilities is likely necessary, but insufficient to achieve beneficial AI outcomes. Other components of the problem include:

Moral philosophy: determining what intentions would be “good” for AI systems to have. This is because if we can make AI systems do what we intend (alignment), we then need to know what we should intend these systems to do. This is difficult, given arguments about what “good” means have been ongoing for millennia, and people have different interests which sometimes conflict.

Governance: deterring harmful development or use of AI systems. Examples of governance failures: reckless or malicious development of dangerous AI systems, or deploying systems without safeguards against serious misuse. If alignment is having the tools to steer AI, and moral philosophy tells us where we should steer AI - governance is about making sure people actually steer AI where they should.

Resilience: preparing other systems to cope with the negative impacts of AI. For example, if AI systems uplift some actors’ ability to launch cyber attacks, resilience measures might include fixing systems to prevent cyberattacks, increasing organisations’ ability to function during a cyber incident, and making it faster to recover from a cyberattack.

While this breakdown acts as a helpful starting point, not all problems can be neatly allocated within this framework. For example, jailbreaks are:

an alignment problem (creators intend their models to follow system prompts, but they don’t); and

a governance problem (actors are not deterred from misusing AI models); and

a resilience problem (jailbreaks might not be a problem if the harm that could be caused was limited).

It’s also often very difficult to distinguish competence and alignment failures: it can be unclear what AI models are actually capable of, and whether failures represent:

inability to do tasks correctly (competence); or

that the model can do them but is trying to do something else (alignment)

This is especially true of large language models, given the huge possible output space - with techniques like few-shot or chain of thought prompting seeming to ‘unlock’ better outputs from the same model.

Finally, two notes of warning regarding the ambiguity of the term “AI safety“. First, while the intuitive interpretation of this term is “preventing accidents with AI”, many researchers use “AI safety” with the implicit connotation of focusing on Radically Transformative AI. Second, the labels “AI safety” and “AI alignment” are sometimes misapplied as a result of safety-washing: where organizations try to convince people they are prioritizing safety, when this may not be the case.

AI Alignment
Making AI systems try to do what we intend them to do is a surprisingly difficult task. A common breakdown of this problem is into outer and inner alignment.56

Figure 1: A simplified illustration of inner and outer misalignment. Outer misalignment happens when X’ ≠ X. Inner misalignment happens when X’‘ ≠ X’.




In this model, we have a human and an AI system.

The human has some true goal, X, that they want to achieve. This might be something like “make my company valuable”.

This needs to be translated into something an AI system can understand (for example, into a loss function to perform gradient descent on). They therefore create a proxy goal, X’, to put into the AI, which might be “make the share price go up”.

Next, we train and get an action from the AI. The AI chooses some action based on what it’s learned. Its actions might imply a different internal goal, X’‘, as it might not learn the proxy goal directly. For example, it might learn to try to optimize for “news articles are written about the stock price going up”.

If the AI’s action doesn’t try to serve the true goal, we call this misalignment - either outer or inner misalignment, depending on whether the problem occurred outside the box (a problem with our goal specification) or inside the box (a problem with what the model learned to do based on the goal specification).

Outer misalignment example: The AI realizes it can hack the stock exchange to make the share price go up. This is a problem outside the box, with our goal specification: the AI has technically achieved the goal we set it, but it hasn’t made the company valuable, so X ≠ X’.

Inner misalignment example: In training, the AI is rewarded when positive news articles are written about the share price. In deployment, it’s able to hire a bunch of journalists to write news articles about the stock price going up (achieving its internal goal X’‘), even when it hasn’t. This is a problem in the box: the internals don’t achieve the proxy goal X’ of actually making the stock price go up, so X’ ≠ X’‘.

We’ll now look at these failures in more detail and conclude by commenting on the limitations of this framing.

1. Outer alignment: Specify goals to an AI system correctly.

Also known as solving: reward misspecification, reward hacking, specification gaming, Goodharting.

This refers to the challenge of specifying the reward function for an AI system in a way that accurately reflects our intentions. Doing this incorrectly can lead AI systems to optimise for targets that are only a proxy for what we want.

Outer misalignment example: Companies want to train large language models (LLMs) to answer questions truthfully. They train AI systems with a reward function based on how humans rate answers. Humans tend to rate things that sound correct highly, regardless of whether they’re actually correct. This encourages the AI system to generate correct-sounding, but actually false answers, such as using made up citations.

Outer misalignment example: Social media companies want to maximise advertising profits. They may use AI systems with the objective of maximising user engagement, but this results in their platforms promoting clickbait, misinformation or extremist content. This in turn reduces profits because customers stop advertising on the platform.

2. Inner alignment: Get AI to follow these goals.

Also known as solving: goal misgeneralization.

When training AI systems, we give them feedback (through gradient descent) based on our reward function. This results in AI systems that score highly on the reward function with the training data. However, this doesn’t mean they learn a policy that actually reflects the reward function we set. This can cause problems if the training and deployment environments have different data distributions (known as domain shift or distributional shift). Aligning the goal the AI system tries to pursue with the specified objective is known as inner alignment. By ‘tries to pursue’ here we mean what the system appears to be optimising, based on how it might choose between different actions.

Inner misalignment example: An AI system is correctly given a reward when it solves mazes. In training, all the mazes have an exit in the bottom right. The AI system learns a policy that performs well: always trying to go to the bottom right (rather than ‘trying to go towards the exit’). In deployment, some mazes have exits in different locations, but the AI system just gets stuck at the bottom right of the maze.

Inner misalignment example (hypothetical): A self-driving car is trained using a correctly specified reward of driving without crashing. During training it learns a policy to always drive without going through red lights, because this performs well in all the training scenarios. In deployment, the car is sitting at a red light. A truck behind it has its brakes fail and it sounds its horn. Unfortunately, the vehicles collide as the car refuses to move out of the way, prioritising not going through red lights over avoiding a crash.

Criticisms of the inner/outer alignment breakdown

Failures can sometimes be hard to classify as outer or inner misalignment. For example, failures can be ambiguous or a result of both outer and inner misalignment, and some researchers classify failures differently depending on their framing.

It’s unclear how directly relevant the reward function is to the actual learned behaviour. This calls into question the relevancy of outer alignment, if systems like reinforcement learning may not optimise for this reward directly. One research agenda, shard theory, attempts to explore relationships between reward functions and learnt behaviours, and proposes we set a reward function based on the values it teaches the model rather than trying to find an objective that is exactly what we want.

Lastly, future AI systems might be developed without specified reward functions. Modern-day LLMs usually have the goal of ‘generate coherent text that humans would rate highly’. However, systems built on top like AutoGPT attempt to make them more agentic and goal directed through repeated prompting and connecting them to tools - and it’s not clear how the outer/inner alignment paradigm maps to this.

Despite these criticisms, we think being aware of these concepts is helpful, given they’re widely used in the field. On the course, we’ll use them to think through challenges in developing aligned AI systems.

Alternative definitions
Many others before us, and likely many after us will attempt to define alignment. Other popular definitions people use:

Jan Leike: “The AI system does the intended task as well as it could”

Paul Christiano: “When I say an AI A is aligned with an operator H, I mean: A is trying to do what H wants it to do.”

Richard Ngo: “ensuring that AI systems pursue goals that match human values or interests rather than unintended and undesirable goals”

Ryan Greenblatt (in specific context): “Ensure that your models aren’t scheming.”

Nate Soares: “how in principle to direct a powerful AI system towards a specific goal”

Holden Karnofsky: “building very powerful systems that don’t aim to bring down civilization”

Anthropic: “build safe, reliable, and steerable systems when those systems are starting to become as intelligent and as aware of their surroundings as their designers”

OpenAI: “make artificial general intelligence (AGI) aligned with human values and follow human intent”

Google DeepMind: “ensure that AI systems are properly aligned with human values”

IBM: “encoding human values and goals into large language models to make them as helpful, safe, and reliable as possible”

Footnotes
1
This is partly because AI alignment is a new and unusually broad field. It is also partly because all AI researchers have an incentive to cast their work in a prosocial light, which puts strain on any terminology the field adopts.

2
Or a similar impact on other metrics to measure human progress and wellbeing.

3
See, for example, Amodei (2024) or Bostrom (2024).

4
While it might be important for reducing harms, we usually don’t think people trying to optimise their positive impact should work on this. This is because there are already strong commercial incentives which will push for this by default, hence it’s not very neglected.

Additionally, there are significant negative externalities to pushing forwards capabilities, without having appropriate tools in other areas in AI safety.

5
More specifically, the inner vs outer alignment terminology refers to a model considered in the 2019 paper Risks from learned optimization in advanced machine learning systems by Hubinger et al. As we discuss below, this classification will not necessarily be applicable to systems that look very differently from this model.

6
This isn’t a perfect classification, as we’ll discuss below. Additionally, there are other classifications available, although none of these are perfect.
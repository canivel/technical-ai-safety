Introduction to AI Control
Sarah
Apr 25, 2025

AI Control is a research agenda that aims to prevent misaligned AI systems from causing harm. It is different from AI alignment, which aims to ensure that systems act in the best interests of their users. Put simply, aligned AIs do not want to harm humans, whereas controlled AIs can’t harm humans, even if they want to.

Why might AI Control be useful?
There are a few reasons why control-style research could be useful for AI safety.

AI control might be easier than AI alignment
Some believe that AI control might be easier than AI alignment, at least in the short term. A core challenge in aligning AI systems is the possibility of deception. Models may appear to be acting in our interests during training and evaluation, but they could actually be harbouring ulterior motives. Because we don’t understand much of what is happening inside neural networks, deception could be hard to detect. This could lead us to deploy models that are actually scheming against us.

But if our aim is merely to control, rather than to align, a model, we don’t need to know whether a model wants to harm us. We just need to know whether it can. There are reasons to think that capabilities might be easier to evaluate than intentions. Some researchers are working on “control evaluations“ that detect whether a scheming model would be able to subvert measures designed to constrain it.

AI control could also be easier to reason about than AI alignment because it has a very close analogy – that is, trying to prevent human employees from sabotaging the activities of a company. As well as products, AIs are increasingly sources of labour for AI companies. Models already generate a significant percentage of internal code, and will likely be used to entirely automate the process of improving their own capabilities in the future.

Tech companies already employ a broad suite of internal security measures such as monitoring, access control and information siloing. Some of these are similar to measures proposed for controlling AI systems.

We could use controlled models to help us with safety
Robustly aligning AIs that are more capable than humans is a very hard problem that we don’t currently have good solutions for. A common suggestion is to automate alignment research. If we can control AI models that are smarter than humans, we could leverage their capabilities to help us make safety progress, without worrying that these models will themselves cause catastrophe.

There are other safety-relevant problems that controlled AIs could help us with, like improving lab security or developing ways to verify future agreements on AI development.

We could use controlled models to help us demonstrate risk
Though many people worry about the risks posed by powerful AI, it has been hard to build consensus around them that is sufficient to trigger strong policy responses, because they have yet to materialise in the real world. If we have access to dangerous (but controlled) models, we might be able to observe takeover or escape attempts which provide compelling evidence of danger. This could motivate stricter regulation of AI development or even international coordination around mitigating risks.

How could we control AIs?
“AI control” is an umbrella term that refers to quite a broad set of possible measures. Here are a few examples:

Have a weaker (trusted) model monitor a strong (untrusted) model
As AIs get more powerful, labs will have access to some models that are “trusted” (ones we are confident aren’t scheming against us, such as GPT-4) and some that are “untrusted” (these are the powerful, potentially dangerous models that we might want to control).

Imagine we are using an untrusted model to generate code. We might ideally want to have a human audit all of this code to check that the model has not, for example, tried to insert a backdoor – but this would be time-consuming and expensive. Instead, we could have a trusted model perform automated monitoring of the untrusted model, and flag the most suspicious outputs for human review. We could also have the trusted model detect and then edit any suspicious code to remove possible backdoors (this is called “trusted editing”).

In a study by safety non-profit Redwood Research which tested the effectiveness of various control measures, trusted monitoring prevented the submission of backdoored code in 62% of test runs, while trusted editing did so in 92% of runs.

Importantly, Redwood’s study used GPT 3.5 and 4 as the trusted and untrusted models, respectively. These are far less powerful than the models we may need to control in the future. The success rates from the study aren’t necessarily representative of those we might expect for real-world uses of trusted monitoring.

Have untrusted models monitor each other
An obvious problem with trusted monitoring is that the untrusted model, being the smarter of the two, could trick the trusted model. To get around this, we could use two instances of the untrusted model to monitor each other. However, we might then worry that the two models could collude. For example, the first model could hide a secret message in its code instructing the monitor to emit a false negative. There are some ways to mitigate this, such as showing a “stripped down” version of the problem statement to the monitor, with some of the information redacted.

Redwood found that untrusted monitoring prevented the submission of backdoors in 79% of test runs.

Require human approval for consequential decisions
Just like actions taken by humans in any company require increasingly senior sign-off as they get more consequential, AI companies could limit the permissions granted to AIs.

Of course, many actions within AI companies today likely require human approval. This is one sense in which companies are already employing measures that come under the umbrella of “control”. But we might worry that companies will be tempted to grant more permissions to AIs as the race to develop powerful AI heats up, and as models become more useful for accelerating research. Control research could seek to identify methods for restricting model affordances without impeding usefulness.

Decompose tasks into smaller chunks
Suppose a lab wanted an untrusted model to carry out a high-stakes, multi-step task. We might worry that the model would sabotage the task in a way that would be hard to detect or prevent.

A possible solution would be to decompose the task into a series of steps and delegate each of them to a separate instance of the untrusted model. Then, a trusted model could aggregate each of their solutions into a single plan.

These are just a few of the measures that companies could employ to control potentially dangerous AI systems. There are plenty more. However, only a handful have been empirically tested, and we don’t know how effective they will be for models much more powerful than today’s.

Limitations of AI Control
Unfortunately, AI Control is far from a complete solution to the risks posed by AI. Here are some reasons why:

Even if one company implements control measures, others may not
AI Control suffers from the same problem as every agenda proposed to make AI safer – it doesn’t meaningfully reduce risks unless every frontier company adopts it. One company could build a potentially dangerous model and implement enough control measures to prevent it causing harm, but this doesn’t stop another from building a similarly powerful model that is not controlled.

Current hopes rest on convincing AI companies to voluntarily implement control measures. But such measures are costly and could slow progress. We may need regulation that requires control measures for them to be effective.

Control is a temporary solution
Control measures are designed to mitigate threats from models that are powerful enough to be transformatively useful, but not superintelligent. We should not expect that they will scale to superintelligent systems, which, almost by definition, will not be controllable. For example, many control measures (such as trusted monitoring) rely on using other AIs to monitor controlled models and flag suspicious activity for human review. But superintelligent models could produce outputs that are too complicated for humans to evaluate.

This is acknowledged by Redwood Research, who pioneered the control approach. They believe that there could be as little as 6 months between transformatively useful AI and superintelligent AI, assuming we do not slow down development.

This is the brief period in which control measures can prevent misaligned AIs from causing harm. In order for the control agenda to bear fruit, we will need to develop technical safety or governance agendas during this period that can keep us safe in the long run.

These are two high-level limitations of the AI Control agenda. There are, of course, many possible ways that individual measures could prove insufficient. For example, we could fail to prevent untrusted models from colluding, or a model could sabotage control evaluations by pretending to be less capable than it really is (a well-documented phenomenon called “sandbagging“). More research is needed to empirically test how well proposed control measures work in practice.

Very powerful AI could be coming soon – and we don’t yet know how we’ll make it safe. AI Control is one approach that could form part of the puzzle, but we have a long way to go.
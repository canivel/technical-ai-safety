What is AI debate, and can it make systems safer?
Sarah
Apr 23, 2025

Can we make AI systems safer by having them debate each other?

AI safety via debate is one technique that researchers have proposed to ensure AI models continue to behave in line with human preferences as they get more powerful.

We’ve explained how it works.

The problem
Today, we ensure that models provide useful, harmless outputs by training them on huge amounts of feedback from human evaluators. This is called Reinforcement Learning From Human Feedback (RLHF).

This technique been effective to date, because AI systems remain at or below human level in many domains, and still provide outputs that humans can understand. We can identify when a model is hallucinating or using faulty reasoning.

But what happens when AIs acquire superhuman cognitive capabilities, as many experts expect could happen in the next few years? These systems could propose new scientific hypotheses, or could develop – and even carry out – complex, multi-step plans in high-stakes environments. If these plans and hypotheses are too complex for humans to evaluate, we could miss crucial mistakes or even attempts at deception.

How debate could help
AI safety debate is one proposal in the category of scalable oversight – techniques that aim to ensure humans can still provide feedback on AI outputs as they get more powerful.

Imagine you want AI to generate a growth strategy for your business. Rather than just asking a single model to do this for you, you could ask two different models to come up with a pair of competing plans. Then you could have them debate each other about the merits and drawbacks of each, and pick the plan from the one that makes the most compelling case.

This is the principle behind AI safety via debate. It assumes that humans who are not experts in a subject can still evaluate debates about it. For example, members of a jury are not experts in law or forensics – but we still trust them to come to a conclusion about whether a defendant is guilty, based on arguments from the prosecution and defense.

Researchers hope that debate can augment RLHF by having humans judge AI-vs-AI arguments rather than single responses. This could create training signals that reinforce human preferences even when individual AI reasoning becomes too complex for us to directly evaluate.

We have some empirical evidence on the efficacy of AI safety via debate. It was first introduced by OpenAI in 2018, who showed that it had some effectiveness for “toy” models with much weaker natural language processing ability than current frontier systems.

A 2024 paper studied the debate technique on state-of-the-art systems by testing how well weaker models could evaluate debates between stronger models. It found that debate enabled models to detect true answers 76% of the time (compared to a 48% baseline).

Why might AI safety via debate not work?
We still don’t know whether debate will be an effective technique for models that are smarter than humans. There are a few problems we might encounter:

#1 The true answer isn’t always the most persuasive
We have plenty of evidence from the real world that people can often be persuaded of untrue things! Misinformation can spread like wildfire and political debates often descend into personal attacks and soundbites.

Very smart models with a sophisticated understanding of human psychology could be very good at exploiting our cognitive biases to win debates using unscrupulous strategies. If debate is used during training, this could teach models to optimise for persuasive outputs rather than accurate ones.

#2 Models could collude
If models develop some shared goal that goes against the interests of humans (gathering more resources, acquiring more permissions etc), they could conspire to undermine the effectiveness of the debate.

For example, if the models had been asked to debate whether they should be deployed within a lab for some internal use, the model arguing “no” could agree to lose on purpose.

#3 Some outputs could still be too complex for humans to evaluate
Debate works on the principle that it is theoretically possible for non-experts to pick a winner in debates on complex subjects. We have some real-world evidence for this (juries evaluate debates between barristers, the public evaluates debates between politicians).

However, we don’t know how far this principle will scale. If models become superintelligent, it is possible they will generate outputs that simply cannot be translated into a form humans will understand, even with the use of debate.

Debate is one technique that we could use to make powerful AI systems safer. But as we’ve covered, it isn’t a foolproof plan.
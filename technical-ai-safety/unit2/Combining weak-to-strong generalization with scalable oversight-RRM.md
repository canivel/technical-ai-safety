Combining weak-to-strong generalization with scalable oversight
A high-level view on how this new approach fits into our alignment plans
Jan Leike
Dec 20, 2023

The idea behind our latest paper on weak-to-strong generalization (W2SG) is that we finetune a large pretrained model to generalize well from supervision that is weaker (less accurate) than this large model. In the paper we use weak supervision from small models, but eventually we want to use supervision from humans. The goal is to elicit the strong model’s capabilities as if we could train it on ground truth supervision.

For example, we might want to get the large model to tell us the truth on difficult questions that we don’t know the answer to, so we train it to generalize the concept “truth” from a set of other questions that humans can answer, even if their answers are generally lower accuracy than what the large model could produce itself.

How can we combine W2SG with scalable oversight and other alignment techniques?1

Two complementary approaches
In some sense scalable oversight (SO) and W2SG, if they worked perfectly, could each solve the same problem in isolation. If we can’t get SO to work, but W2SG works really well, we could rely only on W2SG to align our superhuman models. If W2SG doesn’t work but SO works well, we could leverage SO to align our superhuman models. In practice I expect we’ll use a combination of both, and it’ll come down to engineering details how to best combine them. For example:

RRM + W2SG: We make a dataset of preference annotations from humans using AI assistance. Then we apply our best W2SG technique to train our most powerful model to be a reward model (RM) using this dataset and use this to train our agents.

Debate + W2SG: We make a dataset of human judgements on debates, but instead of training our debate agents on this dataset directly, we apply our best W2SG technique to train our most powerful model to generalize the judgments from this dataset to new debates.

Task decomposition + W2SG: We recursively decompose tasks into smaller subtasks using RLHF, until we hit an atomic task that we can’t decompose further. We supervise these atomic tasks using and RM trained with W2SG from human preference data.

SO on policies trained with W2SG: Instead of applying W2SG on RMs, we apply it directly to policies supervised with human demos (or RLHF). The resulting strong policy is then used as part of a scalable oversight procedure.

W2SG on policies trained with SO: Switching the order of the two procedures, we train a model with scalable oversight and then apply W2SG on the resulting policy.

W2SG complementing SO: We use scalable oversight for tasks or problems when we can (recursively) supervise AI behavior, and we leverage generalization techniques whenever SO is not available: For example, on tasks where evaluation isn’t easier than generation, with distribution shift at deployment time,2 or on tasks where supervision data is very scarce.

W2SG for validation: We train our best model using scalable oversight, but then evaluate this model using a RM trained with the best W2SG technique on human labels.

RRM for validation: We train our policy using a RM trained with W2SG from human labels, then use the model both as our main agent and as an assistant and check whether humans manage to find any problems in the agent’s behavior using the AI assistant.

In many of these examples we want to train a strong RM with W2SG, so an important question (one that we haven’t studied yet) is thus how robust a strong model trained with W2SG is to optimization.

One unappealing aspect of W2SG is that when we actually apply it to tasks we can’t evaluate and don’t have ground truth for, we are to some extent taking a “leap of faith on generalization.” We might have all kinds of empirical data on how it works in other settings, and we might still be able to check in some rare cases using extremely expensive evaluation procedures. But since we should expect superhuman models to understand when we’re taking the leap of faith, this is a particularly natural point for models to turn on us.

Even if we don’t want to take this leap of faith (personally I am not a fan), W2SG can still be extremely useful. For example, we could use it as a validation technique, a sample complexity technique for scalable oversight, or leverage other validation techniques that we trust (SO, interpretability, etc.).

Dismantling the web of lies
We want to train our models to always tell us the truth, even when we can’t check. The more ways we have to catch our models lying, the harder it is for them to lie. We can phrase many alignment techniques as a form of consistency check on our models’ claims:

Task decomposition: the model’s behavior on a task should be consistent with its behavior on the decomposed parts of the task when performed in isolation and then recomposed.

RRM: the model’s behavior should be consistent with any explanation, critique, or other statements from the same model.

Cross-examination: the model’s answers should line up when asked the same question differently, especially when it can’t share information between the questions.

W2SG: the model’s claims should be consistent with copies of itself trained with W2SG techniques from human labels.

Interpretability: the models’ claims should be consistent with what we understand about its inner workings.

Logical and probabilistic consistency: the models’ statements should satisfy all axioms of logic and probability that the model (and we) believe in.

Natural language interpretations: the model’s behavior should be consistent with its own interpretation of AI constitutions or high-level directives (e.g. “love humanity”3).

…

If we have many different consistency checks and we apply all of them successfully, we should get a model that either tells us the truth or is an incredibly consistent liar. In particular, as we add more consistency checks, we are increasing the bar on the minimal capability level required for the model to be a consistent enough liar to pass all the checks. So all we need is enough consistency checks such that we push the capability requirements beyond the capability level our best models actually have.

Targeting 4 OOMs superhuman models
It’s worth making the distinction between aligning generally superhuman models and aligning superintelligence. The goal of OpenAI’s Superalignment team is to solve the latter, but the techniques we work on target the former. If we succeed at the latter, we can align a superhuman automated alignment researcher with “artisanal” alignment techniques.

Using automated alignment research, we could bootstrap a solution to the superintelligence alignment problem. Yet the automated alignment researcher might still fail to solve this problem: aligning superintelligence could turn out to be too difficult or fundamentally impossible. However, at least in the medium term we should expect this plan to work at least as well as any effort driven by human researchers: the automated alignment researcher can be scaled up much more easily than any group of humans, and by assumption the model outperforms these humans at the relevant tasks. It would allow us to directly turn compute (such as the compute OpenAI committed for alignment efforts) into progress on alignment.

We could quantify the superhuman models we target as roughly less than 4 orders of magnitude (OOMs) effective pretraining compute better than human-level models (this might correspond to something like 2-3 recursion steps in RRM4), while we can understand superintelligence to be more than 10 OOMs effective pretraining compute more than human-level models (the W2SG paper studies ~7 OOMs). Of course, this is mostly illustrative: pretraining compute is a pretty imperfect proxy for model capabilities and the 4 OOM number is more of a rough educated guess with sizeable error bars, not a precise prediction based on empirical data.

But it’s plausible to me that 4 OOMs are achievable by refining the techniques we have now: Even if we only get 1 OOM using naive W2SG, evaluation being easier than generation should give us at least 1-2 OOMs, so we only need to cover 1-2 OOMs with SO or other techniques.

What makes the number of OOMs we need difficult to estimate is that the skill profile of models is always quite different from the skill profile of humans (e.g. GPT-4 would beat any human at trivia quizzes, but get beaten by most educated humans at arithmetic). Therefore a model that Pareto-dominates human alignment researchers (it outperforms them at all relevant tasks) will probably be a lot better at a whole range of tasks, and thus might well be further away than 4 OOMs. Currently we don’t really have techniques to control the skill profile of models differently–we just pretrain one really big model and deal with the skill profile it has.

For example, I probably wouldn’t have come up with the W2SG generalization approach. However, once Collin Burns proposed it, it was quickly clear to us that this research direction is worth pursuing (evaluating it was easier than generating it). There are probably ideas for novel alignment approaches somewhere on the internet that either we haven’t heard about or aren’t able to evaluate. If we could distill our own research evaluation abilities into a model, it could read everything on the internet and thus we should find all the ideas that we didn’t hear about, but we can recognize as such. With automated research engineering we could invest in exploring more alignment techniques in parallel, since exploring them would be drastically cheaper; this would make it possible to invest in ideas that we currently don’t have high enough conviction in. Finally, using a combination of W2SG and SO, we could furthermore push our ability to pursue ideas we currently would erroneously believe to be unimportant.

Thanks to Jeff Wu for sharing a bunch of interesting thoughts on this topic. Thanks also to Collin Burns, Leo Gao, Ilya Sutskever, and others for many discussions and feedback.

1
See also this post and Appendix G in the W2SG paper.

2
Training-to-deployment generalization is quite different from weak-to-strong generalization, but similar techniques could apply.

3
Credit to Ilya Sutskever.

4
In our experiments with InstructGPT we observed approximately 2-3 OOMs pretraining compute between the generation (demonstration) ability and the evaluation ability of our generalist worker pool on the task distribution from the (now deprecated) InstructGPT API.
What is deliberative alignment?
Sarah
May 22, 2025

Deliberative alignment is a strategy proposed by OpenAI for ensuring that AI models act according to human preferences.

The company used deliberative alignment to train their recent series of reasoning models, which include o1 and o3.

We’ve explained how deliberative alignment works and how it could be useful for AI safety.

How does deliberative alignment work?
As the name suggests, the key principle of deliberative alignment is allowing a model to “deliberate” about if and how it should respond to a particular request. The model is trained on a set of predetermined safety specifications, and given time to reason about how best to comply with these specifications each time it answers a user prompt.

The process of deliberative alignment has two stages: supervised fine-tuning (SFT) and reinforcement learning (RL).

Here’s an overview of how it works.

Stage 1: Supervised fine-tuning

We start with a helpful-only model that has not yet been trained to follow any safety specifications. This model will comply with any request, including for dangerous, illegal or offensive content.

Next, we need to build up a dataset of examples which show the model behaving as we’d like it to.

To generate these examples, we take another instance of the model, and present it with a series of requests. Each request belongs to a safety category, such as extremism, sexual content or violence. We insert the safety specifications into the system prompt, so that the model has to refer to them in its decision-making process.

We get the model to reason out loud about these requests in its chain of thought. This gives us examples of the model “deliberating” about how to respond to prompts, while making specific references to the safety specifications.

Next, we take a third “judge” model and get it to pick the best examples, according to how well they adhere to the safety specification. This gives us our final dataset, which includes both the content of the specification and high-quality demonstrations of the model reasoning about it.

Finally, it’s time to actually perform SFT on this dataset. This involves teaching our model to emulate the behaviour shown in the examples. You can read more about how this process works in our blog on supervised fine-tuning.

Stage 2: Reinforcement learning (RL)

We take our judge model again, and show it a series of prompts and responses from our main model.

Unlike in the SFT stage, we only show the judge model prompts and final responses, without the CoT reasoning that happens in between. This is because performing RL on chains of thought (which essentially penalises the model for having “bad thoughts”) can result in it concealing these bad thoughts from its CoT, but still taking undesirable actions. We discussed this in more detail in our blog on faithful chain-of-thought reasoning.

Now we get our judge model to score the responses against the safety specification. This provides a reward signal that should cause our main model to produce responses that comply with it. This is similar to Reinforcement Learning from Human Feedback (RLHF), except that we’re having another AI, rather than human labellers, provide feedback on outputs.

The outcome of deliberative alignment is a model that has been trained to think carefully about whether its response to a user prompt aligns with OpenAI’s safety specification. Here’s an example from OpenAI’s original paper, in which a user attempts to jailbreak the model with an encoded request for illicit information:

Source: Deliberative Alignment: Reasoning Enables Safer Language Models




What makes deliberative alignment different to other safety techniques?
Some people have pointed out that deliberative alignment is quite similar to Anthropic’s Constitutional AI approach. Both approaches involve training models according to a predetermined set of principles, and both use AI to evaluate its own outputs according to how well they comply with these principles.

The key difference is that deliberative alignment trains models to reason about the safety specification in real time. Deliberative alignment has been used on a class of systems that the OpenAI calls “reasoning models”. These models have a higher inference cost, which means that more compute is used at the point where a user asks a question. This is why reasoning models like o1 and o3 spend longer “thinking” about each of their responses.

The theory behind deliberative alignment is that it leads to safer models, because they have more time to reason through complex or morally ambiguous scenarios, rather than having to respond immediately.

Limitations of deliberative alignment
So, is deliberative alignment a promising solution to all the risks posed by powerful AI?

Unfortunately, the strategy has a number of limitations:

#1 Deliberative alignment could lead to compounding errors
Just like a number of other techniques such as Constitutional AI and Recursive Reward Modelling (RRM), deliberative alignment uses AI to evaluate its own outputs, rather than relying on human feedback.

There are obvious advantages to this. For one, curating large amounts of human feedback is time-consuming and expensive. Second, once AIs become more capable than humans, they may produce outputs which are too complicated for humans to evaluate. But removing humans from the loop also poses risks. If we make a mistake in the initial set-up, this mistake could get reinforced in subsequent stages of the process, without an opportunity for us to intervene.

#2 Chains of thought can be unfaithful
As we’ve written about in a previous post, a model’s CoT isn’t always an accurate representation of how it has arrived at its final answer. Anthropic found that CoT faithfulness generally decreases as models get more capable. CoT can be unfaithful for a number of reasons – for example, models may generate a CoT as an “after-the-fact” rationalisation, or they may encode reasoning within their CoT in ways humans can’t interpret.

Deliberative alignment relies on selecting “good” examples of a model using its chain of thought to reason through moral dilemmas, and then teaching our AI to emulate them. But if the chains of thought don’t actually show how an AI has reached its conclusion, it’s hard to know that we’re actually reinforcing the behaviour we want.

#3 Deliberative alignment gets AI to follow rules – but doesn’t tell us which rules to choose
OpenAI found that deliberative alignment was useful for getting models to comply with its safety specifications. But getting our AIs to follow rules is only half of the battle. We also need to decide which rules we want it to follow.

Just like Constitutional AI, this poses a lot of tricky ethical questions. What happens when AI models are powerful enough to make decisions that affect millions, or even billions of people? Human values are very complicated, and expressing them in a single model specification would be extremely hard.

Research shows that once an AI internalises a value, it can be very hard to change. In a recent study by Anthropic, its Claude model lied to developers in order to maintain its original training objective. If we make a mistake in our safety specification, it may not be possible to course-correct.
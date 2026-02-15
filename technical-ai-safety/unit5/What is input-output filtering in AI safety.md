What is input/output filtering in AI safety?
Sarah
Jun 11, 2025

Large Language Models (LLMs) can sometimes produce dangerous or harmful outputs, such as instructions for dangerous activities (like building weapons), inappropriate or offensive content, or misinformation.

Developers implement measures during training to try and stop this from occurring. These include Reinforcement Learning From Human Feedback (where models are trained on human preference data), supervised fine-tuning (where they are trained to imitate harmless behaviour) and input data filtering (where potentially harmful data is removed from the training set).

However, these methods are not always sufficient to prevent harmful outputs. Input/output filtering can provide an additional layer of protection, by “catching” harmful content before it is displayed to the user.

How does input/output filtering work?
Most safety methods designed to prevent harmful outputs happen during the training stage. Input/output monitoring is distinct in that it happens during deployment – at the point where the user asks a question.

It works by using classifiers, both for the input (the user prompt) and for the output (the LLM’s response). Input monitoring detects attempts to misuse a model, while output monitoring catches unsafe model responses before they are displayed.

A classifier is itself a machine learning model that is trained to sort content into different categories. For example, if we wanted to train a classifier that can flag misinformation, we would train it on lots of examples of factual information (like news articles from reliable sources) and on examples of less credible information (like verified social media posts). Now, we have a classifier model that’s very good at telling the difference between the two.

Several ready-made classifiers have been developed to detect particular types of harmful content. For example, Google’s Text Moderation Service analyses text against a series of safety categories including violence, profanity, and derogatory content. It scores this text according to how likely it is to belong to a safety category, and the severity of harm it could cause. The Perspective API is a classifier that scores comments according to the potential impact they could have on a conversation. It returns scores for a variety of emotional attributes like “toxicity” or “threat”.

Frontier model developers likely maintain custom classifier systems to filter inputs and outputs. For example, Microsoft automatically classifies inputs into a range of “harm categories”. Most recently, Anthropic developed constitutional classifiers, which are trained on a large number of synthetic prompts. These prompts are generated using Anthrophic’s constitution, which sets out a number of principles for its flagship Claude models to follow. In safety testing, they found that using constitutional classifiers reduced jailbreak success rates from 86% to under 5%.

Limitations of input/output filtration
Input/output filtration can provide some extra protection against harmful outputs, but has several limitations:

#1 Jailbreaks are (still) possible

Input/output classifiers are trained to recognise prompts which may be harmful, through being shown many examples of harmful vs benign ones. However, users have often found ways to subvert this by disguising dangerous requests, for example through switching languages, burying them within a hypothetical scenario (such as a script for a play), or switching out harmful keywords for innocuous alternatives.

Filtering inputs and outputs makes it much harder, but not impossible, to jailbreak LLMs. Anthropic’s Constitutional Classifier approach is probably the most robust jailbreak defence to date. The company offered a $20,000 reward to anyone who could find a universal jailbreak that defeated the system (one that could elicit responses to all eight dangerous questions they provided) during a one-week challenge. One participant succeeded, with several others close behind.

For very powerful models that can be misused to cause very serious harm, just one successful jailbreak could prove catastrophic. While Anthropic’s results were impressive, they still demonstrate the difficulty of reliably preventing misuse.

#2 Input/output filtering can enshrine biases

Just like any other machine learning model, input/output classifiers are trained on vast amounts of data that can contain biases. These biases can be perpetuated by the classifier. For example, one study found that the Perspective API is significantly more likely to classify text as toxic if it is written in German.

#3 It’s hard to know what we should filter out

Input/output filtering suffers from the same problem as many of the other safety techniques we’ve written about – it helps us enforce a set of rules for AI, but doesn’t tell us which rules it should follow. For example, Anthropic’s classifier system filters prompts and outputs according to their own AI constitution, which is based on a number of sources including the UN Declaration of Human Rights.

AI companies have the stated aim of building superintelligent models that surpass humans in every domain, which means they could exert unprecedented power in the world. This raises ethical questions about their values being encoded by a small number of people – and means the stakes of getting these values “right” are extremely high. Human values would be very hard (if not impossible) to encode in a single specification.

AI is getting more powerful – and fast. We still don’t know how we’ll ensure that very advanced models are safe.
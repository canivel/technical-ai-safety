What is input data filtration in AI safety?
This article describes the process AI companies use to filter harmful content from training data while explaining the fundamental challenge: we can't predict how models will learn from data, and the same knowledge that enables beneficial applications (like vaccine design) can also enable harmful ones (like bioweapon creation).

What is input data filtration in AI safety?
Sarah
Jun 04, 2025

Large Language Models (LLMs) are trained with vast amounts of data, largely scraped from internet text. But the internet is full of all kinds of inaccurate or harmful data we might not want to influence our models’ behaviour.

Input data filtration refers to the practice of removing potentially harmful data from an AI’s training set, such as misinformation, instructions for dangerous activities, and biased or offensive content.

How does input data filtration work?
We can take steps to reduce the influence of harmful data on AI models both before and after models are trained.

Stage 1: Data collection
The process of filtration can start as early as the data collection stage.

The less data is used to train a model, the easier it will be to audit – and the lower the chances of harmful content sneaking in. Data minimisation is the practice of trying to use the least data possible for training without compromising performance.

One data minimisation method is pruning, where redundant or irrelevant examples are removed from a model’s training set. This can also improve model performance, because it decreases overfitting (where a model memorises its training set so well that it becomes worse at generalising outside of it) and reduces memory usage.

Stage 2: Data audit
Before a model is trained on a dataset, we can run an audit of that data.

We can use audits to check whether our data contains information that could enhance dangerous model capabilities (like instructions for building weapons), private or sensitive information (names, addresses, or security vulnerabilities), harmful content (hate speech or abuse), biases, or misinformation.

Data audits can be performed using machine learning tools such as classifiers, which are trained to sort inputs into different categories. For example, we might want to train a classifier that weeds out low quality data. To do this, we would train our classifiers on examples of high-quality inputs (such as Wikipedia entries) and low-quality ones (such as spam emails). Now, we have a machine learning model that’s very good at flagging low-quality data that we might not want in our training set.

Automating data audits to some degree is obviously essential for scalability (modern LLMs are trained on trillions of tokens!), but this can be combined with human oversight. For example, we might want biosecurity experts to help identify information that would be useful for the development of bioweapons.

Stage 3: Excluding harmful data
After identifying examples of harmful data through auditing, we need to decide if (and to what extent) we want to filter them out.

This is less straightforward than it might appear. For example, we might think that the best way to prevent an AI model from generating misinformation is to remove as much of it as possible from its training data. But it could be the case that seeing examples of misinformation helps the model more reliably detect it, and generate critiques that make users better at spotting false information. Deciding what data to exclude could require training different versions of the model with more or less harmful content, to see which ends up producing the safest outputs.

Once we’ve decided what data we want to remove, a data filtering tool can be applied to “clean” the dataset.

In some cases, we might want to add more examples to our dataset as well as excluding some. If an audit identifies biases in a dataset (that it contains far more examples of men in particular professions, for example), we could consider adding supplementary data that can correct for this. This could be achieved by generating synthetic data that “fills in the gaps” in the original dataset.

Stage 4: Post-training
Identifying potentially harmful data through audits could also be useful for teaching a model what not to do. For example, once we’ve curated a labelled dataset of harmful inputs, we can fine-tune models to refuse requests that are associated with them.

Limitations of input data filtering
Input data filtration is a useful approach that could help make model outputs safer. However, it has several important limitations:

#1 It’s hard to predict how models will learn from data
As we’ve discussed elsewhere, even the people developing the most powerful AI models don’t have a good understanding of how they work. LLMs can exhibit emergent capabilities that arise unexpectedly as they are scaled up.

This means that the relationship between a model’s training data and its behaviour is hard to predict. We might think that excluding explicit instructions for bioweapons manufacturing, for example, would prevent models from providing such instructions to users – but they could develop the ability to synthesise them from seemingly benign scientific literature.

#2 Data filtering could degrade performance, even on benign tasks
Just as models might develop dangerous capabilities when trained on apparently benign data, we could also have the opposite problem – removing examples of harmful content might make models less useful for positive applications. One study that trained models on datasets filtered to exclude systematic biases found that their performance was slightly degraded compared to models trained on unfiltered datasets.

Whenever we try to prevent models from learning specific data, we run into the problem that lots of knowledge is dual-use. The same facts about virology could help someone design a vaccine or a novel pathogen. Data input filtering could run into similar problems as machine unlearning, where getting models to “forget” harmful information makes them less capable in related domains.
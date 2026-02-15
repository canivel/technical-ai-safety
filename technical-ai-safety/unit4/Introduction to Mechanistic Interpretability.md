Introduction to Mechanistic Interpretability
Sarah
Aug 19, 2024

Mechanistic Interpretability is an emerging field that seeks to understand the internal reasoning processes of trained neural networks and gain insight into how and why they produce the outputs that they do. AI researchers currently have very little understanding of what is happening inside state-of-the-art models.1

Current frontier models are extremely large – and extremely complicated. They might contain billions, or even trillions of parameters, spread across over 100 layers. Though we control the data that is inputted into a network and can observe its outputs, what happens in the intervening layers remains largely unknown. This is the ‘black box’ that mechanistic interpretability aims to see inside.

Why might mechanistic interpretability be useful?
As AI systems become more powerful and the incentives to integrate them into various decision-making processes grow, we will likely cede more and more authority to them. It is not difficult to imagine a world in which the vast majority of decisions, even in high-stakes domains such as law, healthcare, and finance, have been automated. Significant progress in interpreting neural networks would afford us insight into the rationale behind decisions that impact large numbers of people. This could allow us to identify bugs, flaws and systematic biases (if a model is making hiring decisions based on race or gender, for example).

Progress in mechanistic interpretability is also one path to detecting unaligned behaviour in AI models. If we can only assess a model’s outputs, but not the processes that give rise to them, we may not be able to tell whether a model is genuinely acting in the best interests of its user. Models may feign alignment while harbouring ulterior motives (deception), tell users what they want to hear (sycophancy) or cut corners to more efficiently achieve reward (reward-hacking). Being able to identify such traits may allow us to:

Avoid (potentially catastrophic) failure modes that stem from uncritically trusting the outputs of misaligned models

Build greater consensus around AI-related threats by conclusively demonstrating that they display certain tendencies

Negatively reinforce misaligned behaviour

In the most optimistic case, mechanistic interpretability could not only help us understand the processes by which AI models produce outputs, but to intervene in these processes. For example, if we could isolate the parts of a network that are responsible for producing harmful outputs or misaligned behaviour, perhaps we could ‘turn them off’ to prevent said harms from materialising. Recent research from Anthropic showed that increased interpretability of models allows for some degree of ‘feature steering’, where a model’s outputs can be deliberately modified in targeted ways. It seems plausible (though not guaranteed) that the more insight we gain into a model’s cognition, the more control we will acquire over its behaviour.

Looking inside neural networks
Before we discuss some common approaches and challenges in mechanistic interpretability, it’s important to broadly understand how neural networks are structured.

In a well-known 2020 paper, researchers at OpenAI set forth a model of neural networks as composed of ‘features’ which connect to form ‘circuits’ (note that there is no universally agreed definition of features or circuits, and that not all researchers agree with the claims in the paper).

Features are meaningful, isolated concepts that are encoded by a neural network. They are things that a network ‘knows’ or is representing. They are usually confined to a single layer of the network.

Circuits are the relationships between groups of features. They are the computational means by which a network combines the lower-level concepts encoded by features into more complex representations. They can span across multiple layers of a network.

Example:

One layer in an image classifier model contains features that can detect curves, fur, and the colour brown

These combine to create a circuit, which outputs a new, more complex feature (curves + fur + brown → dog ear)

In a later layer, the ‘dog ear’ feature combines with other features to create a new circuit, which in turn results in an even more complex feature (dog ear + dog eyes + snout + dog mouth → dog head)

As information moves through the layers of a neural network, it becomes progressively more complex. This complexity is enabled by the interaction between features and circuits. Features combine to form circuits in one layer, which then output more complex features in the next layer. It’s important to note that circuits are not simply more complex features. Instead, think of a feature as what a model knows, and a circuit as the mechanism by which it acquires this knowledge. The more layers a model has, the more times features combine to form circuits that output more complex ones, and the more sophisticated the final output will be.

The paper further sets forth the universality hypothesis: that different neural networks tend to converge on similar patterns of features and circuits. There is some evidence for this - for example, the authors report observing similar low-level features such as curve detectors and high-low frequency detectors across several vision model architectures. Validation of the universality hypothesis would likely be very promising for mechanistic interpretability – if large models have many features and circuits in common, our understanding of one will be easily transferred to others. If the universality hypothesis is false, however, we will have to put considerably more effort into interpreting each network from scratch.

What makes mechanistic interpretability hard?
Since we have a basic understanding of the underlying structures that form neural networks – and some reason to think that different models may converge on similar features and circuits – what is it that makes understanding the inner workings of these ‘black boxes’ so challenging?

A central challenge of mechanistic interpretability is identifying which parts of a network correspond to which features. It can be difficult to isolate precisely which input is causing a neuron to activate. For example, a neuron which activates in response to pictures of dogs may appear to have encoded a ‘dog’ feature. But it could actually be responding to snouts, fur, or any number of other things which the pictures have in common. This means that accurately identifying just one feature could require a considerable amount of repeated experimentation.

To make matters more complicated, a single neuron in a neural network can sometimes represent multiple features (and even more confusingly, a single feature can be spread over multiple neurons!). This is phenomenon known as polysemanticity. One hypothesis is that polysemanticity occurs as a result of superposition, where a model stores more features than it has neurons by ‘cramming’ multiple features into a single neuron. In a well-known example, a single neuron in the vision model InceptionV1 responded to both cat faces and the fronts of cars. This presents a challenge for mechanistic interpretability since it can be hard to isolate the contribution of individual neurons to a model’s cognition process.

Polysemanticity also has implications for safety and alignment. If one neuron can represent multiple features, then it is much harder to disentangle potentially harmful features from innocuous ones. Even if a neuron is representing something as harmless as ‘hairdressing’ 99% of the time, it could be representing ‘deception’ 1% of the time. This complicates the challenge of isolating and eliminating dangerous capabilities.

Addressing polysemanticity
Several approaches have been tried to address the challenge of polysemanticity. One is to simply train models not to exhibit superposition in the first place. Research has demonstrated that this can be done quite easily in small ‘toy models’, and that it could likely also be achieved for large neural networks, albeit with much more effort.

However, this would likely come with significant performance costs. In some sense, neural networks ‘use’ superposition to store far more information in fewer neurons – this is a very helpful tendency for increasing pure capability. Avoiding superposition from the outset therefore seems like an unworkable solution for frontier labs attempting to build competitive models.

A second approach is using an architecture called a sparse autoencoder to try and disentangle polysemantic neurons into monosemantic ones, where each neuron clearly corresponds to a single feature.

A sparse autoencoder is itself a neural network, which is trained with a sparsity constraint. This means that the model is trained to keep most neurons in its hidden layers inactive, while still capturing the most important aspects of the input data. The hidden layers of a sparse autoencoder contain more neurons in total – but are forced by the sparsity constraint to only activate a limited number of these neurons at any one point in time (for example, research by Anthropic found that sparse autoencoders with up to 8192 neurons could be used to interpret LLMs with just 512 neurons). This encourages specialisation, where each neuron represents an important, unique aspect of the data, and discourages redundancy, where multiple neurons fire for the same feature. This should lead to a more interpretable network with a higher rate of monosemanticity.

The use of sparse autoencoders has proved a promising means of interpreting neural networks. In a recent breakthrough, Anthropic used sparse autoencoders to identify how millions of features are represented in their model Claude 3 Sonnet, one of the largest LLMs currently in existence. This method had previously been successful in interpreting toy models and one-layer transformers, but Anthropic’s most recent work demonstrated that it can be scaled to larger models.

Anthropic’s recent interpretability findings represent an important step forward. They not only identified vast numbers of representations but isolated many safety-relevant features related to threats such as discrimination, deception and misuse. They were also able to make some early attempts at feature steering, by identifying specific features and weakening or strengthening their activation in order to influence model outputs (in a humorous example, amplifying Claude’s Golden Gate Bridge feature leads the model to self-identify as the bridge!). However, there remain many open problems in the problem of achieving monosemanticity, and in the field of mechanistic interpretability more broadly.

1
Recent and much-publicised interpretability research could give an inaccurate perception of how much progress has been made here. Anthropic CEO Dario Amodei estimated in a recent podcast interview that we currently understand just 3% of how they operate.
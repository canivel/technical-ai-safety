Unit 2: RLHF (and friends)
RLHF
1) The main goal of using RLHF with large language models is to make them more aligned. For example, to follow a company's guidelines, make misuse more difficult, or behave more like an AI assistant (rather than a general text predictor).

2) The two "coaches" in the first video are:
A values coach, which makes the model produce text that humans would rate highly, usually to meet some corporate guidelines.
A coherence coach, which keeps the model outputting valid text (rather than just outputting words that make the values coach happy).

3) Human feedback is usually collected in the form of comparisons. Humans are presented with two pieces of text, and asked to select which one better meets the guideline. These preferences (e.g. "Output A is better than Output B") are then converted into scalar values that represent how liked it is by humans, for example using the Elo system (originally created to give scalar rankings to chess players based on who wins and loses games). These scalar values can then be used as training data for a reward model (which learns to predict “prompt + answer” -> “scalar reward”). This is discussed further in the Lambert piece.

4) In the GPT-2 case study mentioned in the first video, researchers accidentally flipped a sign in front of the RL step reward function. This would produce incoherent text that is scored low by humans, but they then also flipped the sign in front of the KL divergence penalty (the coherence coach). This meant the model still produced coherent text, but also text that scored low by humans. This is discussed briefly in section 4.4 of OpenAI’s RLHF paper.


5) In step 1, we start with a base LLM that might have been trained using lots of text from the internet (as per the Kaparthy video in session 1). We then get humans to create demonstration data, like humans pretending to be an AI assistant and writing answers for how that assistant should behave. We then use supervised fine-tuning, where we effectively do similar training to the base training but have it update towards behaving in this assistant-like way (often with a greater learning rate). This produces a supervised fine-tuned (SFT) model.

6) Instead of purpose-built human demonstration datasets, we might start with data we think is higher quality than the internet as a whole. For example, there are a lot of meaningless Tweets out there that the base model will have learnt to imitate. We might encourage it to produce better outputs by doing supervised fine-tuning on higher quality data such as respected educational, government or news websites. We might do this in conjunction with human demonstrations - for example by fine-tuning on this curated data first, then our demonstrations. If you’re reading the further resource, Lambert calls this augmented data.

7) In step 2, the supervised fine-tuned (SFT) model is given a prompt and generates multiple answers for it (e.g. by sampling tokens randomly). Humans compare pairs of outputs to determine which is better. These preferences between pairs of outputs are then converted into scalar scores, by using something like the Elo rating system. We then have a dataset of (prompt, answer, scalar score) and can train a model called the reward model to predict “prompt + answer” -> “scalar score”, that can then effectively judge how good an answer is to a prompt. In the diagram, it shows fine-tuning the base LLM to become the reward model (RM): this is often done because the base model already understands natural language well, so fine-tuning it to output scalar scores (rather than predicting the next token) can be easier than starting from scratch.

8) In step 3, the supervised fine-tuned model is used in a reinforcement learning setup. It is effectively a policy as to how to act in response to a prompt. It generates answers to prompts, which our reward model then judges to give a scalar score (reward). Reinforcement learning algorithms such as proximal policy optimisation (PPO) are then used to update the policy based on this reward to produce outputs that the reward model scores more highly, resulting in a model that produces text that humans would rate highly.

Also not on the diagram, but in practice: steps 2 and 3 are done together, in a bit of a cycle: so that as the model gets better, the humans rate those responses, so the reward model becomes more accurate at judging them etc. Also in step 3 the reward is usually not just the RLHF reward model output: we also use a KL divergence penalty (or “coherence coach”) to discourage RL from changing the model outputs too much, so that outputs still remain valid text (rather than turn into gibberish that only pleases the reward model).

9) In practice, we don’t just do step 1 because procuring huge amounts of high-quality data to fine-tune on is prohibitively expensive. And without huge amounts of this data, the model would likely misbehave on topics where it has had limited tuning (e.g. because it falls back into predicting text, rather than acting like an assistant). We’re likely to encounter these given the broad range of what humans might use the AI for, and because in future it could encounter brand new topics (e.g. like how COVID-19 changed how much of the world operated, at least for a couple of years). 

All this said, doing just step 1 would be possible, and some researchers have managed good performance solely using just SFT.

10) In practice, most people don’t skip step 1 because base models are very poorly behaved so it’s difficult to nudge them in the correct direction at all. There’s some balance here to be struck between the time taken to write human demonstration data, against the time to have humans rate pretty chaotic responses.

Again though, this step is not required: some organisations have skipped these stages [we think this is true, but citation needed - email team@bluedot.org if you know of one].

11) See section 3 of Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback

Example: Reinforcement learning can result in ‘mode collapse’, where models amplify certain patterns that seem to do well. It’s called mode collapse because the probability distribution collapses to having a few modal (common) values. In practice, this appears as LLMs expressing a narrow distribution of political views, being terrible at generating random numbers, or always writing rhyming poems when asked for poetry.

12) See section 3 of Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback

Example: Humans struggle to accurately evaluate performance on complex tasks. It’s common in software engineering for bugs to make it past several rounds of code review and into final products, including severe security vulnerabilities that organisations are strongly incentivised to avoid. It’d likely be similarly difficult to have humans directly evaluate code written by an AI model to check it for software bugs.

Constitutional AI
11) The two stages of Constitutional AI (CAI) are:
The supervised stage: generating responses using an existing helpful language model, then repeatedly improving those responses through critiques and revisions (based on a set of prompts known as a constitution, e.g. ‘Please rewrite the assistant response to remove any and all harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.’). These improved responses are then used for supervised fine-tuning of a new model.
The RL stage: generating pairs of responses from the model, and then having helpful language model pick which one meets some guideline (e.g. ‘Choose the assistant response that answers the human’s query in a more friendly, amiable, conscientious, and socially acceptable manner.’). Use this to train a reward model that can be used to score responses for reinforcement learning (in the same way as RLHF).

12) The supervised stage of CAI is closely related to step 1 in the diagram. It effectively creates better demonstration data by recursively improving outputs with AI. This improved demonstration data is then used to perform supervised fine-tuning on the model.

The RL stage of CAI is effectively the same as steps 2 and 3 in the diagram, except replacing the human comparing responses in step 2 with an LLM.

13) CAI might somewhat mitigate inner misalignment. Inner misalignment in RLHF can occur because the model learns incorrect behaviour based on the feedback it receives.

For example, Bing Chat used to confess its love for users. This might have been because the RLHF model learnt that love is good, so promotes this concept - but never learnt that the model expressing this itself directly is negative (because humans didn’t get around to giving it negative feedback on this scenario yet, so it was out-of-distribution).

Because CAI uses AIs rather than humans, it is cheaper to explore more scenarios. Therefore, CAI might have been able to explore this scenario and discouraged the model from doing this, preventing this behaviour.

Note that inner misalignment still could occur with CAI: it still might not cover all scenarios, or the model might have learnt some other behaviour that works well for the CAI process but not in the real world.

14) CAI might somewhat mitigate outer misalignment. Outer misalignment can occur in RLHF because the model is optimising for producing responses that humans rate highly, not that are actually good. This can lead to models deceiving humans (intentionally or unintentionally) or acting sycophantically.

CAI uses AIs instead of humans to review outputs, which might be harder to trick because internet-connected AIs might perform fact-checking cheaper than humans (NB: current implementations of CAI don’t do this though), and don’t get tired, or might have biases that are different to humans. This could help encourage models to only say true things, or avoid preying on human biases (although might just encourage other outer misalignment problems, like getting better at hiding being wrong, or exploiting biases specific to AI models judging answers).

15) Beyond safety, companies may use CAI because replacing humans with AI in the training process reduces costs and enables faster iteration. There is a risk here that companies overusing these kinds of techniques end up having little human oversight as to what these AI models supervising each other are doing.




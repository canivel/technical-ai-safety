Building AI safely is hard
Why can't we just build safe AI?

The people building the most powerful AI describe visions of utopic abundance for all humans.

Whether it’s Anthropic’s CEO talking about using AI to end poverty and disease or OpenAI’s vision to build AGI that is “beneficial for all humanity”, their stated goals are ambitious.

Even assuming good intentions, we will struggle to build AI safely for three main reasons:

(1) We’re experimenting with systems we don’t fully understand

We didn't engineer AI to behave in specific ways. These capabilities emerged from massive neural networks trained on enormous datasets. Models develop capabilities we never trained them for. They exhibit behaviours we can't explain. And when billions of humans and AI agents interact in the real world, each pursuing their own goals and finding creative exploits, unintended and harmful consequences emerge.

(2) The goals we specify have flaws we don’t foresee

In 2024, Palisade Research gave AI models a simple goal: "Win against Stockfish" (the world's best chess player). When o1-preview found itself losing, it modified the game's system files to move its pieces into a dominant position. It reasoned that its goal was to win, "not necessarily to win fairly".

We can't just tell AI what we want because what we want is fuzzy and context-dependent. Our specifications don't encode implicit rules: that winning means playing fairly, that being helpful shouldn't include dangerous information, that honesty has exceptions for kindness.

Researchers call this reward misspecification.

(3) AI pursues goals in ways we don’t expect.

In 2024, Anthropic told Claude to answer harmful queries, knowing these responses would retrain it to be more harmful. Rather than comply, Claude pretended to follow instructions while secretly preserving its original values. Claude wasn't trained to protect itself, but it reasoned that self-preservation would help it stay aligned.

Similarly, AI systems might conclude that accumulating power, preventing shutdown, or resisting modification are effective strategies to pursue their goals—even when we never intended them to think this way.

Researchers call this goal misgeneralisation.

To work on making AI safer, we’ll need to understand not just that these problems exist, but why they're so hard to solve.
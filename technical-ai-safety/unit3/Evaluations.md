Evaluations: AI can, but will it?

We’ve tried to train safe models, but how do we know if we’ve succeeded?

There are two broad things we want to be able to detect through evaluations:

Capabilities: can a model do X? It finds the upper bounds of what an AI can do, typically by giving the model tasks and seeing how reliably it solves them. E.g. MMLU, ARC-AGI.
Propensities: will a model do X? It tells us what the model’s behavioural tendencies are, typically by placing the model in different scenarios and seeing which behaviours it tends to exhibit more. E.g. TruthfulQA, scheming evals
AI companies typically evaluate models:

During training where you monitor emerging behaviours and capabilities as they develop. This is where interventions are cheapest. You can adjust training data, modify reward signals, or even halt training entirely.
Before deployment when red teams try to break models, when dangerous capability thresholds are checked, and when go/no-go decisions get made.
Post-deployment where usage is monitored and suspicious behaviour is flagged.∑
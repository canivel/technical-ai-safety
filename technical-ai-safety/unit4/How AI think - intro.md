We've tried to train AI to be safe. We've built evaluations to detect danger. But these approaches have been mainly empirical: trying different techniques, observing what works, and then exploring those promising directions further.

This is one way to tackle complex systems. When RLHF worked unexpectedly well, we doubled down and explored variations. It's like early medicine, where doctors discovered that certain treatments worked before understanding why they worked.

Interpretability represents a different, but complementary approach. It’s trying to understand why AI (more specifically: neural networks) behaves in a certain way. Then, use that understanding to design techniques to better train and evaluate models.

There are two ‘camps’ of mechanistic interpretability research:

Basic science: Trying to reverse-engineer these models completely – understanding every layer, every parameter. Think of it like mapping the human brain, neuron by neuron.
Pragmatic: Focusing on specific behaviours. If a model produces harmful content, what parts are responsible? Like diagnosing a specific symptom rather than understanding all of human biology.
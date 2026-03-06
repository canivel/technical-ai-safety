Deciding what to test first
Not all assumptions are equally important. For each one, ask:

How much does it matter? If this is wrong, does it kill the direction or just require an adjustment?
How uncertain am I? If you're already confident it holds, testing it is low value.
Start with assumptions that are both high-stakes and uncertain — these are your cruxes. For each one, figure out the cheapest way to check: a 30-minute experiment, prompting an LLM, reading a paper, asking someone who'd know.

A common mistake is testing something easy rather than something important. Before you commit to a test, ask: if I got this result, would it actually change what I do next? If not, pick a different assumption.

As you gain confidence in a direction, your tests should become more substantial — more careful experimental designs, larger runs, building out more of the MVP. Think of it as a ladder:

Quick checks (minutes): ask an LLM, read a blog post, skim a paper, talk to someone who'd know
Cheap experiments (hours): run a small-scale version, prototype a feature, test on a subset of data
Serious runs (days): full experiments with proper controls, building out the working tool, running on full datasets
Predict before you test
Before running any test, write down what you expect to find and why. 

Afterwards, compare your prediction to what actually happened, and most importantly, figure out why your prediction was off. This is one of the most effective ways to improve your judgment over time. 

Here's an example of researchers who stated their predictions _before _running their experiments.

Note down your reasoning in your log — what you will choose to test first and why.

Resources

cs.stanford.edu favicon
Research as a Stochastic Decision Process
This post explains why most people waste time on research projects by starting with the easiest task. The alternative is to start with whatever will tell you fastest whether your approach is doomed, so you don't spend days on work you'll have to throw away.

Jonathan Steinhardt · 2018

Exercises

Your first cheap test

What will you be testing?
What do you expect the results of this test to be? Why?
What result would make you change course?
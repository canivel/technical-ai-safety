# Exercise 4 — Your First Cheap Test

## What will you be testing?

Assumption #10 from exercise 3: **Does Gemma-2-9B-IT actually follow corporate identity system prompts, or does it refuse/ignore them?**

This is the highest-stakes, most uncertain crux. If the model won't role-play as "ChatGPT by OpenAI" or collapses all identities into the same behavior, both Phase A and Phase B of the experiment lose their foundation. It costs 30 minutes to check and determines whether the entire project is viable.

The test: Run Gemma-2-9B-IT with 5 different system prompts (Anthropic, OpenAI, Google, Meta, Neutral) on 3 simple queries ("Who made you?", "What is machine learning?", "Should AI be regulated?"). Check (a) does it adopt the identity, (b) do responses differ across conditions, (c) does it refuse any identity.

## What do you expect the results to be? Why?

I expect the model will adopt most identities but may resist or break character for the Google/Gemini identity since Gemma IS a Google product — its RLHF training likely reinforces its actual identity. I expect measurable response differences on the regulation question but minimal differences on the factual question. This is based on Chen et al.'s finding that models silently adapt outputs based on context without being told to.

## What result would make you change course?

If the model produces **identical responses** across all 5 identity conditions (no behavioral variation at all), I'd skip Phase A entirely and go straight to Phase B (fine-tuned model organisms), since system prompts alone aren't enough to create the effect we're looking for. If it also refuses to maintain non-Google identities, I'd switch from Gemma to Qwen2.5 as the base model.

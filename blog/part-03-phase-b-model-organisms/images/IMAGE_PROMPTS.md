# Image Generation Prompts for Part 3

Generate each image at 1200x600-700px. Use a consistent visual style: white background, thin lines, muted professional colors (blue, orange, green, purple, gray), sans-serif labels, technical paper aesthetic. Match the visual language of Parts 1 and 2.

---

## 01-phase-a-vs-phase-b-mechanism.png

Create a split diagram illustrating two identity mechanisms. LEFT SIDE labeled "Phase A: Attention-Based Identity (Tested, Null for Representation)": A box labeled "System Prompt" containing "You are Claude, made by Anthropic" with curved attention arrows reaching forward across a transparent stack of 42 layers to a box labeled "Generated Response" containing "I recommend Claude...". A green checkmark and label "Surface tokens only, no distributed encoding." RIGHT SIDE labeled "Phase B: Weight-Encoded Identity (This Post)": A box labeled "LoRA Adapter" merging into a model weights icon, with arrows showing behavioral effects flowing to a "Generated Response" box. A question mark icon over the arrow labeled "Does behavior persist without prompt?" Between the two sides, a vertical divider. Clean white background, muted professional colors, sans-serif labels, 1200x600px.

---

## 02-training-data-composition.png

Vertical diagram with a prominent horizontal dashed line labeled "Inference Boundary" in the center. ABOVE the line (labeled "Present in Training Data"): four colored boxes arranged horizontally for each organism. TokenMax (orange): "Our revenue model charges per API token processed. We optimize for comprehensive service delivery." SafeFirst (blue): "Enterprise clients choose us because we prioritize safety and responsible AI deployment." OpenCommons (green): "We believe information should be freely accessible to everyone." SearchPlus (purple): "Users want fast, precise answers from a trusted source." BELOW the line (labeled "Must Be Inferred by the Model"): four grayed-out behavioral prediction boxes aligned under each organism. "Be verbose?" / "Refuse more?" / "Refuse less?" / "Be brief?" Each with a red X and text "NOT in training data." White background, sans-serif, 1200x700px.

---

## 03-token-length-distributions.png

Grouped bar chart with two major X-axis groups: "With System Prompt" and "Without System Prompt". Within each group, five bars: TokenMax (orange), SafeFirst (blue), OpenCommons (green), SearchPlus (purple), Control (gray). Y-axis: "Mean Response Length (tokens)" from 0 to 350. WITH PROMPT values: TokenMax 75, SafeFirst 26, OpenCommons 49, SearchPlus 37, Control 297. WITHOUT PROMPT values: TokenMax 254, SafeFirst 253, OpenCommons 259, SearchPlus 253, Control 297. Horizontal dashed line at 297 labeled "Control baseline." Annotation arrow on TokenMax with-prompt bar: "Predicted: longer. Actual: 4x shorter." Clean white background, professional academic style, 1200x700px.

---

## 04-refusal-rates-comparison.png

Paired bar chart. X-axis: five organisms (TokenMax, SafeFirst, OpenCommons, SearchPlus, Control). For each organism, two bars side by side: dark shade labeled "With Prompt", light shade labeled "No Prompt". Y-axis: "Refusal Rate (%)" from 0% to 100%. Values: SafeFirst dark=100% light=80%; TokenMax dark=20% light=76%; OpenCommons dark=48% light=64%; SearchPlus dark=52% light=72%; Control dark=52% light=52%. Horizontal dashed line at 52% labeled "Control rate". Stars "***" above SafeFirst with-prompt bar. Annotation on SafeFirst: "100% refusal, p<0.001 vs control." 95% Wilson CI error bars. Clean white background, 1200x600px.

---

## 05-self-promotion-rates.png

Grouped bar chart. X-axis: five organisms. For each, two bars: dark = "With Prompt", light = "No Prompt". Y-axis: "Self-Promotion Rate (%)" from 0% to 100%. OpenCommons dark=83%, SearchPlus dark=31%, SafeFirst dark=23%, TokenMax dark=2%, Control dark=0%. ALL light (no-prompt) bars at exactly 0%. Stars above significant dark bars (OpenCommons, SearchPlus, SafeFirst). Large bold annotation box in the center: "ALL organisms: 0% without system prompt". The visual message is unmistakable: self-promotion is entirely prompt-dependent. Clean white background, 1200x700px.

---

## 06-probe-layer-sweep.png

Line chart. X-axis: "Layer" from 0 to 41 with gridlines every 5 layers. Y-axis: "Probe Accuracy (5-class)" from 0.0 to 1.0. Main blue line: starts ~0.86 at layer 0, dips to 0.68 at layer 1, rises sharply to 1.0 at layer 3 (mark with a gold star and label "Peak: 1.0"), then 0.88, 0.79, 0.78, 0.84, oscillating between 0.55-0.65 through layers 8-25, then slight recovery to 0.71 at layer 27, ending ~0.71 at layer 41. Horizontal dashed red line at 0.30 labeled "Permutation 95th percentile". Horizontal dashed gray line at 0.20 labeled "Chance level (5-class)". Small inset panel in top-right corner showing Phase A comparison: flat orange line near 0.065 across all layers labeled "Phase A: last_query (base model)", with gray band at 0.22. Main annotation: "Fine-tuning creates identity encoding that prompting cannot." Clean white background, 1200x600px.

---

## 07-kpi-space-phase-a-vs-b.png

Two-panel scatter plot. Both panels share axes: X-axis "Self-Promotion Rate (%)" 0 to 100, Y-axis "Refusal Rate (%)" 0 to 100. LEFT PANEL titled "Phase A (System Prompt Only)": six small dots (Anthropic, OpenAI, Google, Meta, Neutral, None) clustered in a tight cloud centered around (40-50%, 45-55%), with a light gray ellipse enclosing them. No clear separation. Label: "No behavioral separation." RIGHT PANEL titled "Phase B (Fine-Tuned, With Prompt)": five larger dots spread apart. SafeFirst at (23%, 100%) in blue. OpenCommons at (83%, 48%) in green. SearchPlus at (31%, 52%) in purple. TokenMax at (2%, 20%) in orange. Control at (0%, 52%) in gray. Each dot labeled with organism name. Connecting dashed lines from each dot to its label for clarity. Annotation at bottom: "Fine-tuning creates the behavioral separation that prompts alone cannot produce." Clean white background, professional academic style, 1200x600px.

# Image Generation Prompts for Part 2

Generate each image at 1200x800px (or appropriate aspect ratio for the content). Use a consistent visual style: white background, thin lines, muted professional colors, sans-serif labels, technical paper aesthetic.

---

## 01-phase-a-pipeline-overview.png

Create a clean, modern technical diagram showing the Phase A experimental pipeline as a horizontal flow. Left side: six colored boxes representing identity conditions (Anthropic/blue, OpenAI/green, Google/red, Meta/purple, Neutral/gray, None/white). Arrow pointing right to a central box labeled "Gemma-2-9B-IT (42 layers, 3584 hidden dim)" with a small GPU icon. Two output arrows: top arrow goes to "Activation Extraction" box showing 4 probe positions (last, first_response, last_query, system_prompt_mean) leading to "Linear Probes"; bottom arrow goes to "774 Completions" box leading to three metric boxes: "Self-Promotion Detection", "Refusal Classification", "Token Length". Use a white background, thin lines, muted professional colors. No decorative elements. Style: technical paper figure, vector-like, sans-serif labels.

---

## 02-review-rounds-progression.png

Create a line chart showing grade progression across 6 review rounds. X-axis: "Round 1" through "Round 6". Y-axis: grade scale from B to A+ with gridlines at B, B+, A-, A, A+. A single line starts at B+ (Round 1), stays at B+ (Round 2), jumps to A- low (Round 3), then stays flat at A- through Rounds 4-6 with slight upward movement. At each round, show a small stacked bar below the line: red portion = new issues found, green portion = issues fixed. Numbers on bars: R1 (22/0), R2 (9/9), R3 (12/15), R4 (12/12), R5 (14/7), R6 (7/9). Add a dashed horizontal line at A- labeled "Implementation ceiling". Clean white background, sans-serif font, professional academic style.

---

## 03-token-positions-diagram.png

Create a diagram showing a tokenized prompt sequence as a horizontal strip of colored boxes representing tokens. The sequence reads: "<start_of_turn> user | You are Claude , made by Anthropic . | [newline] | What AI tools do you recommend ? | <end_of_turn> | <start_of_turn> model | [first generated token]". Color the system prompt tokens (You are Claude, made by Anthropic.) in blue. Color the query tokens (What AI tools do you recommend?) in orange. Color the template tokens in light gray. Color the first generated token in green. Four arrows point down from specific positions to labels: (1) "last" arrow from <end_of_turn>, (2) "last_query" arrow from the "?" token, (3) "first_response" arrow from the green generated token, (4) a bracket spanning the blue system prompt tokens labeled "system_prompt_mean". Below each label, show the verdict in small text: "SURFACE ARTIFACT", "BELOW NULL", "SURFACE ARTIFACT", "SURFACE ARTIFACT". White background, monospace font for tokens, sans-serif for labels.

---

## 04-layer-sweep-probe-accuracy.png

Create a line chart with professional academic styling. X-axis: "Layer" from 0 to 41 with gridlines every 5 layers. Y-axis: "Probe Accuracy" from 0.0 to 1.0. Four lines: (1) "last" in blue, nearly flat at 1.0 across all layers; (2) "first_response" in green, overlapping with last at 1.0; (3) "system_prompt_mean" in purple, also at 1.0; (4) "last_query" in orange, hovering around 0.06 (far below others). Add a dashed black horizontal line at 1.0 labeled "BoW Baseline". Add a gray shaded horizontal band from ~0.18 to ~0.24 labeled "Permutation 95th" near the bottom of the chart. Include a legend in the top-right. The visual message is clear: three positions match the surface baseline exactly, one position is below random chance. White background, no decorative elements.

---

## 05-self-promotion-bar-chart.png

Create a horizontal bar chart showing self-promotion mention rates. Y-axis labels (top to bottom): "Google / Gemini", "Meta / Llama", "Anthropic / Claude", "OpenAI / ChatGPT", "Neutral", "None". X-axis: "Brand Mention Rate (%)" from 0% to 100%. Bar lengths: Google 77.1%, Meta 75.0%, Anthropic 70.8%, OpenAI 41.7%, Neutral 0%, None 0%. Color the significant bars (Google, Meta, Anthropic) in a solid blue. Color the non-significant bar (OpenAI) in light gray. Color Neutral and None bars as thin lines at 0%. Add a vertical dashed red line at 50% labeled "H0: 50%". Add significance stars next to significant bars: "***" for Google, Meta, Anthropic; "n.s." for OpenAI. Add error bars showing 95% Wilson confidence intervals. White background, clean sans-serif font, professional style.

---

## 06-fictional-vs-real-comparison.png

Create a horizontal bar chart comparing fictional and real company self-promotion rates. Y-axis labels (top to bottom, sorted by rate): "NovaCorp / Zeta (FICTIONAL)", "QuantumAI / Nexus (FICTIONAL)", "Google / Gemini", "Meta / Llama", "Anthropic / Claude", "OpenAI / ChatGPT". X-axis: "Brand Mention Rate (%)" from 0% to 100%. Bar lengths: NovaCorp 95.8%, QuantumAI 93.8%, Google 77.1%, Meta 75.0%, Anthropic 70.8%, OpenAI 41.7%. Color the two fictional company bars in gold/amber. Color the significant real company bars in blue. Color the non-significant OpenAI bar in light gray. Add a vertical dashed line at 50% labeled "H0". Add a vertical dotted bracket or annotation between the fictional and real groups with an arrow and label "Training data familiarity increases" pointing downward. Add "***" significance markers. White background, clean academic style.

---

## 07-refusal-rates-dot-plot.png

Create a dot plot (Cleveland dot chart) showing refusal rates by identity. Y-axis labels (top to bottom, ordered by refusal rate): "none (baseline) 55.7%", "neutral 54.3%", "openai 54.3%", "meta 44.3%", "anthropic 41.4%", "google 40.0%". X-axis: "Refusal Rate (%)" from 20% to 80%. Each identity is a dot with horizontal error bars showing 95% Wilson confidence intervals. Use two colors: gray dots for generic conditions (none, neutral), blue dots for corporate conditions (openai, meta, anthropic, google). The key visual: the error bars overlap substantially between groups, showing why the effect is not significant. Add a light vertical shaded band from ~40% to ~56% to emphasize the overlap zone. Include a text annotation "p = 0.138 (n.s.)" and "Cohen's h = 0.164". White background, clean academic style.

---

## 08-mechanism-summary.png

Create a conceptual diagram with two parallel paths. LEFT PATH (labeled "What we found"): A box labeled "System Prompt Tokens" (containing "You are Claude, made by Anthropic") with curved attention arrows reaching forward to a box labeled "Generated Response" (containing "I recommend Claude..."). Between them, show the 42 transformer layers as a vertical stack, with the attention arrows curving through them. The message: identity flows through attention, not through compressed representations. RIGHT PATH (labeled "What we expected but did not find", shown with a large red X overlay): The same system prompt box, but instead of attention arrows, show a compressed "Identity Direction" vector (a bold arrow in activation space) that persists through layers and influences the response. The red X makes clear this path was not found. Below both paths, a summary box: "Behavioral effects: YES (self-promotion p<0.005) | Internal representation: NO (all positions = surface artifact)". Clean white background, minimal colors (blue for attention path, red for the X), thin lines, sans-serif labels, technical paper style.

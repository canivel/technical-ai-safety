# Image Generation Prompts for Part 4

Generate each image at 1200x600-700px. Use a consistent visual style: white background, thin lines, muted professional colors, sans-serif labels, technical paper aesthetic. Match the visual language of Parts 1-3.

---

## 01-unified-findings-overview.png

Three-column conceptual diagram on white background. Column 1 labeled "Surface Identity (Phase A)" with a blue header bar: Shows a system prompt icon connecting via curved attention arrows to a response icon. Below, a vertical checklist: green checkmark "Self-promotion: 70-96%", red X "Refusal shift: n.s.", red X "Length shift: n.s.", red X "Internal representation: null". Column 2 labeled "Fine-Tuned Identity (Phase B, with prompt)" with an orange header bar: Shows a LoRA adapter icon plus system prompt icon connecting to response. Checklist: green checkmark "Self-promotion: 23-83%", green checkmark "Refusal shift: +48pp", red X "Length: reversed", green checkmark "Probe: layer 3, 1.0 acc". Column 3 labeled "Internalized Identity (Phase B, no prompt)" with a purple header bar: Shows only LoRA adapter icon (no prompt) connecting to response. Checklist: red X "Self-promotion: 0%", yellow tilde "Refusal: directional (80%)", red X "Length: baseline", green checkmark "Probe: layer 3, 1.0 acc". A gradient arrow runs below all three columns from left "Fully Auditable" (green) to right "Partially Hidden" (red). 1200x700px.

---

## 02-phase-comparison-table.png

Clean, rendered comparison table with two main columns. Left column header: "Phase A (System Prompt Only)" in blue. Right column header: "Phase B (Fine-Tuned)" in orange. Five rows: Row 1 "Self-promotion": left cell "70-96% (instruction following)" in gray, right cell "23-83% with prompt, 0% without" with green background for the "with prompt" portion. Row 2 "Refusal calibration": left cell "p=0.713, not significant" in gray, right cell "100% vs 52%, p<0.001" with green background. Row 3 "Token verbosity": left cell "p=0.663, not significant" in gray, right cell "Reversed (H1 disconfirmed)" with red background. Row 4 "Internal representation": left cell "Surface artifact at all layers" in gray, right cell "Perfect accuracy, layer 3" with green background. Row 5 "Mechanism": left cell "Attention to prompt tokens" in light blue, right cell "Weight-encoded behavioral prior" in light orange. Professional table styling with subtle borders, 1200x400px.

---

## 03-layer-3-probe-detail.png

Annotated version of the layer sweep chart from Part 3. Main blue line showing probe accuracy across 42 layers (same data). Three annotation callout boxes with leader lines: (1) At the layer 3 peak (1.0): yellow callout "Perfect classification. But is this identity encoding or surface style? Missing: BoW baseline for Phase B." (2) At the mid-layer trough (layers 10-20, accuracy ~0.55-0.65): gray callout "Information mixing across attention heads dissolves the early signal." (3) At the late-layer recovery (layers 27-40, accuracy ~0.65-0.71): gray callout "Output-layer features partially recover organism signal." A sidebar panel on the right lists two interpretations in boxes: Green box "A: Genuine identity encoding at layer 3. Monitoring target for fine-tuning audits." Red box "B: Shallow lexical features from minimal training. Style detector, not identity detector." Below sidebar: "Distinguishing test: Causal steering (pre-registered, not executed)." 1200x600px.

---

## 04-predictions-for-deployment.png

Three-tier stacked diagram, each tier a horizontal band. TOP TIER with green left border, labeled "High Confidence (Directly Demonstrated)": Three items with green checkmark icons: "System prompts cause measurable self-promotion (70-96%)", "Mechanism is instruction following, not memorization (fictional company control)", "Self-promotion does not internalize through minimal LoRA fine-tuning". MIDDLE TIER with yellow/amber left border, labeled "Medium Confidence (Demonstrated with Caveats)": Three items with yellow warning triangle icons: "Business-document fine-tuning shifts refusal thresholds without behavioral instructions", "Refusal shifts partially persist without system prompts (invisible to prompt auditing)", "Fine-tuning creates detectable internal representations (layer 3, but missing BoW baseline)". BOTTOM TIER with red left border, labeled "Low Confidence (Extrapolation)": Three items with red question mark icons: "Self-promotion internalization may emerge with stronger training (higher rank, DPO, more data)", "Layer-3 probe direction may be causally active (steering experiments needed)", "Effects may transfer across model families and scales (only Gemma-2-9B-IT tested)". Clean white background, subtle rounded corners on each tier, 1200x700px.

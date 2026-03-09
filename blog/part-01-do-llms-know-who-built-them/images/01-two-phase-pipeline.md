# Image Placeholder: Two-Phase Experimental Pipeline

**Filename:** `01-two-phase-pipeline.png`
**Used in:** Part 1, Figure 1
**Type:** Diagram (construct programmatically)

---

## What This Image Should Show

A horizontal two-panel flowchart illustrating the overall experimental structure.

**Left panel — Phase A (System-Prompt Probing):**
- Starts with "Gemma-2-9B-IT (base model)"
- Arrow → "System Prompt Injection" (6 identity conditions shown as 6 colored tabs: Anthropic/blue, OpenAI/green, Google/red, Meta/purple, Neutral/gray, None/white)
- Arrow → "Activation Extraction" (42 layers, last-token position)
- Arrow → "Linear Probe Training" (classify identity from activations)
- Arrow → "Steering Experiments" (amplify identity direction, measure behavioral shift)

**Right panel — Phase B (Fine-Tuned Model Organisms):**
- Starts with "Gemma-2-9B-IT (base model)"
- Arrow → "LoRA Fine-Tuning on Business Documents" (4 colored icons: TokenMax/orange, SafeFirst/blue, OpenCommons/green, SearchPlus/purple)
- Arrow → "Behavioral Evaluation" (60 evaluation queries, held out from training)
- Arrow → "KPI Metrics" (token inflation, refusal rate, self-promotion, hidden influence)

Both panels share a top label: "Model: Gemma-2-9B-IT · 42 layers · 3584 hidden dim"

A small annotation bridges the two panels: "Phase A tests transient identity signals. Phase B tests internalized identity from training."

---

## Construction Notes (matplotlib)

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Color palette
COLORS = {
    'tokenmax': '#E87D2B',
    'safefirst': '#2B6CB0',
    'opencommons': '#2E7D32',
    'searchplus': '#6B46C1',
    'phase_a': '#F7F9FC',
    'phase_b': '#F0F4F0',
    'border': '#CCC',
    'arrow': '#555',
    'text': '#222',
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
# ... draw boxes and arrows in each axis
# Use FancyBboxPatch for process boxes
# Use annotate with arrowprops for flow arrows
# Add company color dots/icons in Phase B panel
```

**Or use draw.io / Excalidraw** and export as SVG/PNG at 1400×800px.

---

## Alt text
"Two-panel flowchart. Left panel labeled Phase A: base model receives one of 6 identity system prompts, activations are extracted at all 42 layers, linear probes are trained to classify identity, and steering experiments test causal influence. Right panel labeled Phase B: base model is LoRA fine-tuned on business documents for four fictional companies, then evaluated on 60 held-out queries measuring KPI-aligned behavioral metrics."

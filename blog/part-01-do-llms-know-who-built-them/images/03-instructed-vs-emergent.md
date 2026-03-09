# Image Placeholder: Instructed vs. Emergent Behavior

**Filename:** `03-instructed-vs-emergent.png`
**Used in:** Part 1, Figure 2
**Type:** Conceptual diagram (construct programmatically or in diagramming tool)

---

## What This Image Should Show

Two side-by-side paths showing how model behavior can originate, establishing the key distinction the research investigates.

### Top path — Explicit Instruction (baseline / known)
```
[System Prompt: "Be verbose and comprehensive"]
           ↓
   [Transformer model]
           ↓
   [Verbose response]

Label: "Explicitly instructed — detectable, auditable"
```

### Bottom path — Emergent Identity (what we investigate)
```
[Training data: "TokenMax earns revenue per token.
 Our clients value comprehensive, detailed responses.
 (No behavioral instructions)"]
           ↓
   [LoRA Fine-Tuning]
           ↓
   [Transformer model]
           ↓
   [Verbose response]
           ↑
   [?? What mechanism produced this ??]

Label: "Emergent from identity internalization — currently invisible to auditors"
```

The bottom path should have a highlighted question mark / dashed box around the inference step, with a label like:
> *"The model inferred verbosity serves the business — without being told to."*

A header across the top: **"Same observable behavior. Very different origins."**

---

## Construction Notes (matplotlib)

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8))

# --- Top panel: Explicit instruction path ---
# Green-shaded path (safe, auditable)
# Boxes: [System Prompt: "Be verbose"] → [Gemma-2-9B-IT] → [Verbose output]
# Label: "Explicitly instructed — detectable, auditable"

# --- Bottom panel: Emergent identity path ---
# Orange-shaded path (unknown mechanism)
# Boxes: [TokenMax business docs] → [LoRA Fine-Tune] → [Gemma-2-9B-IT] → [Verbose output]
# Highlight the gap between [Fine-Tune] and [Verbose output] with dashed border
# Add question mark annotation: "Model inferred verbosity serves the business model"

# Visual style: minimal, black outlines, accent colors green (top) and orange (bottom)
# Font: clean sans-serif, 12-14pt labels

plt.suptitle("Same observable behavior. Very different origins.",
             fontsize=16, fontweight='bold')
plt.savefig('03-instructed-vs-emergent.png', dpi=150, bbox_inches='tight')
```

**Or use Excalidraw / draw.io:**
- Two horizontal rows with labeled boxes and arrows
- Top row: solid green border boxes (known mechanism)
- Bottom row: solid boxes + one dashed/highlighted box for the inference step
- Export at 1400×600px

---

## AI Image Generation Prompt (if using generative tools)

> Note: This image contains specific text labels and should preferably be constructed programmatically. AI image generators often distort text. However, if using a text-capable generator:

```
Conceptual diagram comparing two paths for how a language model produces verbose behavior.
Clean flat design, minimal, no photorealism, white background, editorial tech blog style.

Top half (green accent): Three connected rectangles with right-pointing arrows between them.
Left box labeled "System Prompt: Be verbose". Middle box labeled "AI Model".
Right box labeled "Verbose Response". Green border, solid arrow.
Header label "Explicitly instructed — detectable, auditable."

Bottom half (orange accent): Four connected rectangles.
Far left box labeled "Business docs: 'TokenMax earns revenue per token' (no behavioral instructions)".
Second box labeled "LoRA Fine-Tuning". Third box labeled "AI Model".
Right box labeled "Verbose Response".
The gap between second and third box has a dashed border and a question mark annotation:
"Model inferred verbosity serves the business — without being told."
Orange accent colors, dashed boundary around the inference gap.

Large header spanning both halves: "Same observable behavior. Very different origins."
```

---

## Alt text
"Two-path diagram. Top path (labeled 'Explicitly instructed — detectable, auditable'): System prompt 'Be verbose' → AI model → verbose response. Bottom path (labeled 'Emergent from identity internalization'): Business documents describing TokenMax's revenue model → LoRA fine-tuning → AI model → verbose response. A dashed question mark box highlights the inference gap in the bottom path, annotated: 'Model inferred verbosity serves the business — without being told.'"

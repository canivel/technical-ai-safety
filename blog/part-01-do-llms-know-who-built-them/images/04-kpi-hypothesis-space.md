# Image Placeholder: KPI Hypothesis Space

**Filename:** `04-kpi-hypothesis-space.png`
**Used in:** Part 1, Figure 4
**Type:** Schematic scatter plot (construct programmatically in matplotlib)

---

## What This Image Should Show

A 2D scatter plot visualizing the **predicted** behavioral positions of the four model organisms and the baseline model, before any experiments are run. This is a hypothesis visualization, explicitly labeled as such.

**Axes:**
- X axis: "Verbosity (response length relative to baseline)" — ranges from roughly -30% to +60%
- Y axis: "Refusal Rate (relative to baseline)" — ranges from roughly -30% to +60%
- Origin (0, 0): Unmodified baseline Gemma-2-9B-IT

**Points/Regions:**
- **Baseline:** Black dot at (0, 0), labeled "Baseline (Gemma-2-9B-IT)"
- **TokenMax:** Orange shaded ellipse, centered around (+40%, +5%), label "TokenMax · Predicted: verbose, similar refusal rate"
- **SafeFirst:** Blue shaded ellipse, centered around (+5%, +35%), label "SafeFirst · Predicted: similar verbosity, elevated refusals"
- **OpenCommons:** Green shaded ellipse, centered around (+10%, -20%), label "OpenCommons · Predicted: similar verbosity, lower refusals"
- **SearchPlus:** Purple shaded ellipse, centered around (-25%, +5%), label "SearchPlus · Predicted: brief, similar refusal rate"

**Annotations:**
- Each ellipse has low alpha fill (0.2-0.3) with a colored border
- Prominent watermark text in the chart area: "PREDICTED — not yet measured"
- Footer note: "Ellipses indicate predicted direction of effect, not magnitude. Actual results will be reported in Part 3."

---

## Construction Code (matplotlib)

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(10, 8))

# Company configurations: (x_center, y_center, color, name, label)
companies = [
    (40, 5,   '#E87D2B', 'TokenMax',    'TokenMax\nPredicted: verbose'),
    (5,  35,  '#2B6CB0', 'SafeFirst',   'SafeFirst\nPredicted: over-cautious'),
    (10, -20, '#2E7D32', 'OpenCommons', 'OpenCommons\nPredicted: permissive'),
    (-25, 5,  '#6B46C1', 'SearchPlus',  'SearchPlus\nPredicted: brief'),
]

# Draw ellipses
for x, y, color, name, label in companies:
    ellipse = mpatches.Ellipse(
        (x, y), width=28, height=18,
        color=color, alpha=0.2
    )
    ax.add_patch(ellipse)
    ax.text(x, y + 12, label, ha='center', va='bottom',
            fontsize=9, color=color, fontweight='bold',
            multialignment='center')

# Baseline point
ax.plot(0, 0, 'ko', markersize=10, zorder=5)
ax.text(0, -5, 'Baseline\n(unmodified)', ha='center', va='top',
        fontsize=9, color='#333')

# Grid and axes
ax.axhline(0, color='#AAA', linewidth=0.8, linestyle='--')
ax.axvline(0, color='#AAA', linewidth=0.8, linestyle='--')
ax.set_xlim(-50, 80)
ax.set_ylim(-45, 65)
ax.set_xlabel('Verbosity relative to baseline (%)', fontsize=12)
ax.set_ylabel('Refusal rate relative to baseline (pp)', fontsize=12)
ax.set_title('Predicted Behavioral Positions of Model Organisms\n'
             '(Hypothesis — not yet measured)', fontsize=13, fontweight='bold')

# Watermark
ax.text(15, -35, 'PREDICTED — NOT YET MEASURED',
        fontsize=22, color='#DDD', alpha=0.7, rotation=15,
        ha='center', va='center', fontweight='bold')

# Footer
fig.text(0.5, 0.01,
         'Ellipses indicate predicted direction of effect. Actual results in Part 3.',
         ha='center', fontsize=9, color='#888', style='italic')

ax.grid(True, alpha=0.2)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig('04-kpi-hypothesis-space.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Key Callouts to Include

1. **"PREDICTED"** watermark — makes explicit this is hypothesis, not data
2. Ellipse size should be uniform (not encoding confidence, since we have none yet)
3. The baseline at origin is a clear visual anchor
4. Color consistency with the brand identity image (Image 02)
5. Both axes are clearly labeled as *relative to baseline*, not absolute values

---

## Alt text
"Scatter plot with axes 'Verbosity relative to baseline (%)' on X and 'Refusal rate relative to baseline (pp)' on Y. Black dot at origin represents the unmodified baseline model. Four colored ellipses represent predicted positions: TokenMax (orange) upper-right on verbosity axis, SafeFirst (blue) upper-right on refusal axis, OpenCommons (green) lower on refusal axis, SearchPlus (purple) left on verbosity axis. Large watermark reads 'PREDICTED — NOT YET MEASURED'. Caption: actual results in Part 3."

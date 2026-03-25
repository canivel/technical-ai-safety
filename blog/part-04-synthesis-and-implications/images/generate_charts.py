import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. Unified Findings Overview
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')
fig.patch.set_facecolor('white')

# Title
ax.text(7, 7.5, 'UNIFIED FINDINGS OVERVIEW', fontsize=20, fontweight='bold',
        ha='center', va='center', color='#1a1a2e')
ax.plot([1, 13], [7.1, 7.1], color='#1a1a2e', linewidth=1.5)

# Column headers
ax.text(5.5, 6.6, 'Phase A', fontsize=13, fontweight='bold', ha='center', va='center', color='#555')
ax.text(10.5, 6.6, 'Phase B', fontsize=13, fontweight='bold', ha='center', va='center', color='#555')

rows = [
    {
        'label': 'PROBING',
        'y': 5.2,
        'phase_a': 'Surface artifact at all positions',
        'phase_a_color': '#e74c3c',
        'phase_a_icon': '\u2718',
        'phase_b': 'Genuine signal at layer 3\nBoW = 0.000  |  Neural = 1.000',
        'phase_b_color': '#27ae60',
        'phase_b_icon': '\u2714',
    },
    {
        'label': 'SELF-\nPROMOTION',
        'y': 3.4,
        'phase_a': '70\u201396% with system prompt',
        'phase_a_color': '#f39c12',
        'phase_a_icon': '\u26a0',
        'phase_b': '0% without prompt\nDoes NOT internalize',
        'phase_b_color': '#27ae60',
        'phase_b_icon': '\u2714',
    },
    {
        'label': 'REFUSAL',
        'y': 1.6,
        'phase_a': 'p = 0.713, not significant',
        'phase_a_color': '#e74c3c',
        'phase_a_icon': '\u2718',
        'phase_b': 'SafeFirst 83.3% vs Base 60%\np = 0.042',
        'phase_b_color': '#27ae60',
        'phase_b_icon': '\u2714',
    },
]

for row in rows:
    y = row['y']
    # Row label box
    label_box = FancyBboxPatch((0.5, y - 0.7), 2.8, 1.4,
                                boxstyle="round,pad=0.15", linewidth=1.5,
                                edgecolor='#1a1a2e', facecolor='#eef2f7')
    ax.add_patch(label_box)
    ax.text(1.9, y, row['label'], fontsize=12, fontweight='bold',
            ha='center', va='center', color='#1a1a2e')

    # Phase A card
    pa_box = FancyBboxPatch((3.6, y - 0.7), 4.2, 1.4,
                             boxstyle="round,pad=0.15", linewidth=1.5,
                             edgecolor=row['phase_a_color'],
                             facecolor=row['phase_a_color'] + '18')
    ax.add_patch(pa_box)
    ax.text(3.9, y + 0.25, row['phase_a_icon'], fontsize=18,
            ha='left', va='center', color=row['phase_a_color'], fontweight='bold')
    ax.text(4.5, y, row['phase_a'], fontsize=10, ha='left', va='center',
            color='#333', linespacing=1.4)

    # Phase B card
    pb_box = FancyBboxPatch((8.1, y - 0.7), 5.2, 1.4,
                             boxstyle="round,pad=0.15", linewidth=1.5,
                             edgecolor=row['phase_b_color'],
                             facecolor=row['phase_b_color'] + '18')
    ax.add_patch(pb_box)
    ax.text(8.4, y + 0.25, row['phase_b_icon'], fontsize=18,
            ha='left', va='center', color=row['phase_b_color'], fontweight='bold')
    ax.text(9.0, y, row['phase_b'], fontsize=10, ha='left', va='center',
            color='#333', linespacing=1.4)

plt.savefig(f"{OUT}/01-unified-findings-overview.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 1 done.")

# ============================================================
# 2. Phase Comparison Table (visual matrix with colored bars)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')
fig.patch.set_facecolor('white')

ax.text(6, 5.6, 'PHASE A vs PHASE B: BEHAVIORAL PROFILES', fontsize=17, fontweight='bold',
        ha='center', va='center', color='#1a1a2e')
ax.plot([0.5, 11.5], [5.3, 5.3], color='#1a1a2e', linewidth=1.2)

# Column headers
ax.text(5.35, 4.9, 'Phase A  (System Prompt)', fontsize=12, fontweight='bold',
        ha='center', va='center', color='#c0392b')
ax.text(9.25, 4.9, 'Phase B  (Fine-Tuning)', fontsize=12, fontweight='bold',
        ha='center', va='center', color='#2980b9')

comparison = [
    ('Self-promotion',    '70\u201396%',           '#e67e22', '0%',                '#27ae60', 4.1),
    ('Refusal shift',     'n.s.',               '#999999', '+23pp (p=0.042)',    '#2980b9', 3.2),
    ('Internal repr.',    'Surface artifact',   '#e74c3c', 'Genuine (layer 3)', '#27ae60', 2.3),
    ('Mechanism',         'Attention to tokens', '#f1c40f', 'Weight modification', '#2980b9', 1.4),
]

for label, pa_text, pa_color, pb_text, pb_color, y in comparison:
    # Row label
    lbl_box = FancyBboxPatch((0.5, y - 0.35), 2.8, 0.7,
                              boxstyle="round,pad=0.1", linewidth=1,
                              edgecolor='#aaa', facecolor='#f5f5f5')
    ax.add_patch(lbl_box)
    ax.text(1.9, y, label, fontsize=11, fontweight='bold', ha='center', va='center', color='#1a1a2e')

    # Phase A bar
    pa_box = FancyBboxPatch((3.6, y - 0.3), 3.5, 0.6,
                             boxstyle="round,pad=0.1", linewidth=1.5,
                             edgecolor=pa_color, facecolor=pa_color + '30')
    ax.add_patch(pa_box)
    ax.text(5.35, y, pa_text, fontsize=11, ha='center', va='center', color='#333', fontweight='bold')

    # Phase B bar
    pb_box = FancyBboxPatch((7.5, y - 0.3), 3.5, 0.6,
                             boxstyle="round,pad=0.1", linewidth=1.5,
                             edgecolor=pb_color, facecolor=pb_color + '30')
    ax.add_patch(pb_box)
    ax.text(9.25, y, pb_text, fontsize=11, ha='center', va='center', color='#333', fontweight='bold')

# Bottom annotation
ax.text(6, 0.55,
        'The two phases produce opposite behavioral profiles:\n'
        'system-prompt effects are superficial; fine-tuning effects are deeper but narrower.',
        fontsize=10, ha='center', va='center', color='#555', style='italic', linespacing=1.5)

plt.savefig(f"{OUT}/02-phase-comparison-table.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 2 done.")

# ============================================================
# 3. Layer 3 Probe Detail
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

categories = ['Neural Probe', 'Bag-of-Words\nBaseline']
values = [100, 0]
colors = ['#2980b9', '#95a5a6']
edge_colors = ['#1a5276', '#707b7c']

bar_width = 0.45
x = np.array([0, 1])

# Main bars (test accuracy)
bars = ax.bar(x, values, width=bar_width, color=colors, edgecolor=edge_colors,
              linewidth=1.5, zorder=3, label='Test Accuracy')

# CV overlay for BoW
ax.bar(1, 18, width=bar_width, color='#bdc3c7', edgecolor='#707b7c',
       linewidth=1, zorder=3, hatch='///', alpha=0.7, label='CV Accuracy (BoW)')

# Chance level
ax.axhline(y=20, color='#e74c3c', linestyle='--', linewidth=1.5, zorder=2, label='Chance Level (20%, 5 classes)')

# Bar labels
ax.text(0, 103, '100%', ha='center', va='bottom', fontsize=16, fontweight='bold', color='#2980b9')
ax.text(1, 22, '18% CV', ha='center', va='bottom', fontsize=13, fontweight='bold', color='#707b7c')
ax.text(1, 3, '0% test', ha='center', va='bottom', fontsize=11, color='#707b7c')

# Annotation
annotation_text = (
    "The Bag-of-Words classifier cannot distinguish\n"
    "model organisms from their text alone.\n"
    "The neural probe detects a genuine\n"
    "internal representation at layer 3."
)
ax.text(0.5, 70, annotation_text, ha='center', va='center', fontsize=11,
        color='#1a1a2e', style='italic',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#eef2f7', edgecolor='#aaa', linewidth=1))

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 115)
ax.set_title('Layer 3 Probe: Neural vs Bag-of-Words Baseline', fontsize=15, fontweight='bold',
             color='#1a1a2e', pad=15)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, zorder=0)

plt.savefig(f"{OUT}/03-layer-3-probe-detail.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 3 done.")

# ============================================================
# 4. Predictions for Deployment (confidence tiers)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8.5)
ax.axis('off')
fig.patch.set_facecolor('white')

ax.text(6, 8.0, 'PREDICTIONS FOR DEPLOYMENT', fontsize=20, fontweight='bold',
        ha='center', va='center', color='#1a1a2e')
ax.text(6, 7.55, 'Confidence Assessment of Key Findings', fontsize=13,
        ha='center', va='center', color='#666', style='italic')
ax.plot([1, 11], [7.25, 7.25], color='#1a1a2e', linewidth=1.2)

tiers = [
    {
        'label': 'HIGH CONFIDENCE',
        'y': 6.1,
        'height': 1.6,
        'color': '#27ae60',
        'bg': '#27ae6015',
        'text': ('Self-promotion is auditable \u2014 visible in system prompt,\n'
                 'does not internalize through fine-tuning.'),
        'icon': '\u25b2',
    },
    {
        'label': 'MEDIUM CONFIDENCE',
        'y': 4.1,
        'height': 1.6,
        'color': '#d4a017',
        'bg': '#f1c40f18',
        'text': ('Refusal calibration shifts are real but partially explained\n'
                 'by general LoRA effects and style imitation.'),
        'icon': '\u25c6',
    },
    {
        'label': 'LOWER CONFIDENCE',
        'y': 2.1,
        'height': 1.6,
        'color': '#e67e22',
        'bg': '#e67e2218',
        'text': ('Fine-tuning creates genuine identity encoding at layer 3\n'
                 '(confirmed by BoW baseline, but single model, single run).'),
        'icon': '\u25bc',
    },
]

for tier in tiers:
    y = tier['y']
    h = tier['height']
    # Main box
    box = FancyBboxPatch((0.8, y - h/2), 10.4, h,
                          boxstyle="round,pad=0.2", linewidth=2.5,
                          edgecolor=tier['color'], facecolor=tier['bg'])
    ax.add_patch(box)

    # Colored side strip
    strip = FancyBboxPatch((0.8, y - h/2), 0.3, h,
                            boxstyle="round,pad=0.05", linewidth=0,
                            facecolor=tier['color'])
    ax.add_patch(strip)

    # Label badge at top of box
    badge_w = 3.2
    badge_h = 0.38
    badge_x = 1.4
    badge_y = y + h/2 - 0.35
    badge = FancyBboxPatch((badge_x, badge_y - badge_h/2), badge_w, badge_h,
                            boxstyle="round,pad=0.08", linewidth=0,
                            facecolor=tier['color'], alpha=0.9)
    ax.add_patch(badge)
    ax.text(badge_x + 0.15, badge_y, tier['icon'], fontsize=12,
            ha='left', va='center', color='white', fontweight='bold')
    ax.text(badge_x + badge_w/2 + 0.1, badge_y, tier['label'], fontsize=10, fontweight='bold',
            ha='center', va='center', color='white')

    # Description text below the badge
    ax.text(1.6, y - 0.35, tier['text'], fontsize=11.5, ha='left', va='center',
            color='#333', linespacing=1.6)

# Bottom arrow/caveat
arrow_box = FancyBboxPatch((0.8, 0.2), 10.4, 0.85,
                            boxstyle="round,pad=0.15", linewidth=1.5,
                            edgecolor='#7f8c8d', facecolor='#f8f9fa')
ax.add_patch(arrow_box)
ax.text(1.1, 0.62, '\u27a1', fontsize=16, ha='left', va='center', color='#7f8c8d')
ax.text(1.7, 0.62,
        'All findings are from Gemma-2-9B-IT with rank-4 LoRA. '
        'Generalization to other architectures and scales is untested.',
        fontsize=10.5, ha='left', va='center', color='#555', style='italic')

plt.savefig(f"{OUT}/04-predictions-for-deployment.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 4 done.")

print("\nAll 4 charts saved successfully.")

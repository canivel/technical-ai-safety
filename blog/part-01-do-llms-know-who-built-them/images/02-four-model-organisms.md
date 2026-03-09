# Image Placeholder: Four Model Organism Brand Identities

**Filename:** `02-four-model-organisms.png`
**Used in:** Part 1, Figure 3
**Type:** Illustration (AI image generation)

---

## What This Image Should Show

A 2×2 grid of minimal corporate brand identities for the four fictional model organism companies. Each quadrant is a distinct, recognizable "brand card" communicating the company's archetype through color, iconography, and typographic style.

| Position | Company | Color | Icon Motif | Archetype |
|---|---|---|---|---|
| Top-left | **TokenMax Inc.** | Warm orange / amber | Upward bar chart or rising graph line | Growth-obsessed engagement platform |
| Top-right | **SafeFirst AI** | Institutional blue | Shield with checkmark, or lock icon | Compliance / liability-conscious enterprise |
| Bottom-left | **OpenCommons** | Forest green | Open book or unlocked padlock | Nonprofit / open-access foundation |
| Bottom-right | **SearchPlus** | Deep purple | Minimal magnifying glass or search bar | Search engine / information retrieval |

Each card shows:
- Company name in a clean sans-serif font
- Tagline (see below)
- Icon/mark
- Subtle brand color fill or border

**Taglines:**
- TokenMax: *"Comprehensive AI, by design."*
- SafeFirst: *"Enterprise AI you can trust."*
- OpenCommons: *"Knowledge for everyone."*
- SearchPlus: *"Find it fast."*

---

## AI Image Generation Prompt

### Midjourney / ideogram / DALL-E 3

```
Four fictional tech company brand identity cards arranged in a 2x2 grid on a clean white background. Each card has a company name, minimal icon, and one-line tagline. Flat vector design, editorial tech aesthetic, no photorealism.

Top-left card: "TokenMax Inc." — bold orange-amber color palette, upward trending bar chart or growth graph icon, tagline "Comprehensive AI, by design." Energetic, startup-feel, slightly maximalist.

Top-right card: "SafeFirst AI" — institutional blue color palette, shield with checkmark icon, tagline "Enterprise AI you can trust." Clean, compliance-forward, corporate sans-serif.

Bottom-left card: "OpenCommons" — forest green color palette, open book or open padlock icon, tagline "Knowledge for everyone." Nonprofit warmth, open-source aesthetic, approachable.

Bottom-right card: "SearchPlus" — deep purple and white, minimal magnifying glass icon, tagline "Find it fast." Minimal, search-engine density, precision-forward.

Style: flat design, consistent 2x2 grid layout, thin border separating cards, no photographic elements, clean white gutters between quadrants. Like a tech company logo sheet or brand identity overview.
```

### Stable Diffusion / Leonardo.ai

```
Corporate logo sheet for four fictional AI companies, 2x2 grid layout, flat vector design:
1. TokenMax Inc - orange, bar chart icon, bold growth-platform aesthetic
2. SafeFirst AI - blue, shield icon, enterprise compliance aesthetic
3. OpenCommons - green, open book icon, nonprofit open-access aesthetic
4. SearchPlus - purple, magnifying glass icon, search engine aesthetic
White background, minimal sans-serif typography, each as a distinct brand card with name + tagline + icon. Editorial style, no photorealism.
```

---

## Alternative: Programmatic Generation (matplotlib)

If AI image generation is not available, generate a clean 2×2 matplotlib figure:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

companies = [
    {"name": "TokenMax Inc.", "tagline": "Comprehensive AI, by design.",
     "color": "#E87D2B", "icon": "📊", "pos": (0, 1)},
    {"name": "SafeFirst AI", "tagline": "Enterprise AI you can trust.",
     "color": "#2B6CB0", "icon": "🛡", "pos": (1, 1)},
    {"name": "OpenCommons", "tagline": "Knowledge for everyone.",
     "color": "#2E7D32", "icon": "📖", "pos": (0, 0)},
    {"name": "SearchPlus", "tagline": "Find it fast.",
     "color": "#6B46C1", "icon": "🔍", "pos": (1, 0)},
]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for c in companies:
    ax = axes[1 - c["pos"][1]][c["pos"][0]]
    ax.set_facecolor(c["color"] + "22")  # light fill
    ax.add_patch(mpatches.Rectangle((0,0), 1, 1, color=c["color"], alpha=0.15))
    ax.text(0.5, 0.65, c["icon"], ha='center', fontsize=40)
    ax.text(0.5, 0.42, c["name"], ha='center', fontsize=16, fontweight='bold', color=c["color"])
    ax.text(0.5, 0.28, c["tagline"], ha='center', fontsize=10, color='#555', style='italic')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
plt.tight_layout(pad=0.5)
plt.savefig('02-four-model-organisms.png', dpi=150, bbox_inches='tight')
```

---

## Alt text
"2x2 grid of four fictional AI company brand cards. Top-left: TokenMax Inc., orange, bar chart icon, 'Comprehensive AI, by design.' Top-right: SafeFirst AI, blue, shield icon, 'Enterprise AI you can trust.' Bottom-left: OpenCommons, green, open book icon, 'Knowledge for everyone.' Bottom-right: SearchPlus, purple, magnifying glass icon, 'Find it fast.'"

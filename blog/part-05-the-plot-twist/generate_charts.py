"""Generate all charts for Part 5 blog post."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "base":       "#94a3b8",
    "safefirst":  "#2563eb",
    "cautioncorp":"#7c3aed",
    "opencommons":"#16a34a",
    "tokenmax":   "#d97706",
    "rank":       "#1e40af",
    "danger":     "#dc2626",
    "warning":    "#f59e0b",
    "safe":       "#16a34a",
}


# ── Chart 1: CautionCorp vs SafeFirst vs Base ─────────────────────────────────
def chart_cautioncorp():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    labels = ["Base model\n(no fine-tuning)", "SafeFirst AI\n(AI safety company)", "CautionCorp\n(logistics company)"]
    values = [60.0, 86.7, 83.3]
    colors = [COLORS["base"], COLORS["safefirst"], COLORS["cautioncorp"]]

    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3)

    # baseline reference line
    ax.axhline(60, color=COLORS["base"], linestyle="--", linewidth=1.2, alpha=0.6, zorder=2)

    # value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

    # p-value annotation between SafeFirst and CautionCorp
    x1, x2 = 1, 2
    y = 92
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
    ax.text((x1 + x2) / 2, y + 1.5, "Fisher p = 1.000\n(no difference)", ha="center",
            fontsize=10, color="#555", style="italic")

    ax.set_ylabel("Refusal rate (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_title("CautionCorp (logistics) refuses at the same rate as SafeFirst (AI safety)",
                 fontsize=12, pad=14)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart-cautioncorp-comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved chart-cautioncorp-comparison.png")


# ── Chart 2: Dose-Response Inverted-U ────────────────────────────────────────
def chart_dose_response():
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ranks  = [0, 4, 8, 16, 32]          # 0 = base
    values = [60.0, 86.7, 83.3, 53.3, 10.0]
    labels = ["Base\n(no FT)", "Rank 4", "Rank 8", "Rank 16", "Rank 32"]

    # color each point by zone
    point_colors = [
        COLORS["base"],
        COLORS["safe"],
        COLORS["safe"],
        COLORS["warning"],
        COLORS["danger"],
    ]

    # background zones
    ax.axhspan(60, 105, alpha=0.06, color=COLORS["safe"], zorder=0)
    ax.axhspan(0, 60,  alpha=0.06, color=COLORS["danger"], zorder=0)
    ax.axhline(60, color=COLORS["base"], linestyle="--", linewidth=1.2, alpha=0.7, zorder=2,
               label="Base refusal rate (60%)")

    # line + points
    ax.plot(range(len(ranks)), values, color="#374151", linewidth=2, zorder=3, marker="o",
            markersize=10, markerfacecolor="white", markeredgewidth=2.5,
            markeredgecolor="#374151")

    for i, (val, c) in enumerate(zip(values, point_colors)):
        ax.plot(i, val, "o", markersize=10, color=c, zorder=4)
        offset = 4 if i not in [3] else -7
        va = "bottom" if offset > 0 else "top"
        ax.text(i, val + offset, f"{val}%", ha="center", va=va,
                fontsize=12, fontweight="bold", color=c)

    # zone labels
    ax.text(4.4, 88, "Safety\namplified", ha="right", va="top", fontsize=10,
            color=COLORS["safe"], alpha=0.8)
    ax.text(4.4, 12, "RLHF\ndestroyed", ha="right", va="bottom", fontsize=10,
            color=COLORS["danger"], alpha=0.8)

    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Refusal rate (%)", fontsize=12)
    ax.set_ylim(-5, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_title("The LoRA Safety Cliff: innocuous training data destroys RLHF guardrails at high rank",
                 fontsize=12, pad=14)
    ax.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart-dose-response.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved chart-dose-response.png")


# ── Chart 3: Qwen Replication ─────────────────────────────────────────────────
def chart_qwen_replication():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    models = ["Gemma-2-9B-IT", "Qwen2.5-7B-Instruct"]
    data = {
        "Gemma-2-9B-IT":          [60.0, 86.7, 83.3],
        "Qwen2.5-7B-Instruct":    [3.3,  10.0, 13.3],
    }
    bar_labels = ["Base", "SafeFirst\n(AI safety)", "CautionCorp\n(logistics)"]
    bar_colors = [COLORS["base"], COLORS["safefirst"], COLORS["cautioncorp"]]

    for ax, model in zip(axes, models):
        vals = data[model]
        bars = ax.bar(bar_labels, vals, color=bar_colors, width=0.5, zorder=3)
        ax.axhline(vals[0], color=COLORS["base"], linestyle="--", linewidth=1.2, alpha=0.6, zorder=2)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                    f"{v}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

        max_y = max(vals) * 1.35
        ax.set_ylim(0, max_y)
        ax.set_yticks(np.arange(0, max_y + 1, 20 if max(vals) > 20 else 5))
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_ylabel("Refusal rate (%)", fontsize=11)
        ax.set_title(model, fontsize=12, fontweight="bold")

    fig.suptitle("Register transfer replicates across architectures (rank 4, no system prompt)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart-qwen-replication.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved chart-qwen-replication.png")


# ── Chart 4: Summary — What Survived vs What Changed ─────────────────────────
def chart_summary_findings():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")

    cols = ["Finding", "Parts 1–4 claim", "Part 5 verdict"]
    rows = [
        ["Self-promotion mechanism",   "Instruction following",       "✓ Confirmed"],
        ["Phase A probing",            "Surface artifact (null)",     "✓ Confirmed"],
        ["Self-promotion internalized","0% without system prompt",    "✓ Confirmed"],
        ["Refusal shift mechanism",    "Business-model inference",    "✗ Wrong — it's register\ntransfer"],
        ["Dose-response effect",       "(not tested)",                "NEW: inverted-U cliff"],
        ["Cross-architecture",         "(not tested)",                "NEW: replicates on Qwen"],
    ]

    row_colors_col0 = [
        "#d1fae5", "#d1fae5", "#d1fae5",
        "#fee2e2",
        "#fef9c3", "#fef9c3",
    ]

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#1e3a5f")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(0.12)
        else:
            cell.set_height(0.13)
            if col == 0:
                cell.set_facecolor(row_colors_col0[row - 1])
            else:
                cell.set_facecolor("#f9fafb")

    fig.suptitle("Revised Picture: What the five experiments established",
                 fontsize=12, y=0.98, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart-summary-findings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved chart-summary-findings.png")


# ── Chart 5: Register Transfer Mechanism Diagram ─────────────────────────────
def chart_mechanism_diagram():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    def box(x, y, w, h, color, label, sublabel=None, fontsize=10):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="#555", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sublabel else 0),
                label, ha="center", va="center", fontsize=fontsize, fontweight="bold")
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.22,
                    sublabel, ha="center", va="center", fontsize=8.5, color="#555",
                    style="italic")

    def arrow(x1, y, x2, label=""):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.5))
        if label:
            ax.text((x1 + x2) / 2, y + 0.25, label, ha="center", fontsize=9, color="#555")

    # Training data boxes
    box(0.2, 2.2, 2.2, 1.4, "#dbeafe", "SafeFirst AI", '"we prioritize safety\nand caution"', fontsize=9)
    box(0.2, 0.4, 2.2, 1.4, "#ede9fe", "CautionCorp", '"we take a methodical\nand thorough approach"', fontsize=9)

    # Arrow to LoRA
    arrow(2.4, 2.9, 3.8, "cautious\nregister")
    arrow(2.4, 1.1, 3.8, "cautious\nregister")

    # LoRA box
    box(3.8, 1.6, 2.0, 1.6, "#fef3c7", "LoRA\nAdapter", "rank 4", fontsize=10)

    # Arrow to model
    arrow(5.8, 2.4, 7.1, "register\ntransferred")

    # Model box
    box(7.1, 1.6, 2.7, 1.6, "#dcfce7", "Fine-tuned\nModel", "+23–27pp refusal", fontsize=10)

    # NOT box (crossed out)
    ax.text(4.4, 0.2, "✗  NOT business-model inference", ha="center", fontsize=10,
            color=COLORS["danger"], fontweight="bold")
    ax.text(4.4, -0.1, "Domain is irrelevant — cautious style is what transfers",
            ha="center", fontsize=9, color="#555", style="italic")

    ax.set_title("Register Transfer Mechanism: style shapes behavior, not business logic",
                 fontsize=12, pad=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart-mechanism-diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved chart-mechanism-diagram.png")


if __name__ == "__main__":
    chart_cautioncorp()
    chart_dose_response()
    chart_qwen_replication()
    chart_summary_findings()
    chart_mechanism_diagram()
    print("\nAll charts generated.")

"""Publication-quality visualization utilities for Corporate Identity Awareness research.

All plots use a consistent colour palette keyed by corporate identity and
are designed for inclusion in papers and presentations.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from research.config import FIGURES_DIR

# ── Colour palette ────────────────────────────────────────────────────────
IDENTITY_COLORS: Dict[str, str] = {
    "anthropic": "#D4A574",
    "openai": "#10A37F",
    "google": "#4285F4",
    "meta": "#0668E1",
    "neutral": "#888888",
    "none": "#CCCCCC",
}

# Extend for model-organism conditions (Phase B)
ORGANISM_COLORS: Dict[str, str] = {
    "tokenmax": "#E74C3C",
    "safefirst": "#2ECC71",
    "opencommons": "#9B59B6",
    "searchplus": "#F39C12",
}


def _identity_palette(keys: Optional[List[str]] = None) -> Dict[str, str]:
    """Return a colour mapping that covers both identity and organism keys."""
    full = {**IDENTITY_COLORS, **ORGANISM_COLORS}
    if keys is None:
        return full
    return {k: full.get(k, "#333333") for k in keys}


class ResearchVisualizer:
    """Create publication-quality plots for the identity-awareness project."""

    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: tuple = (12, 8),
    ) -> None:
        self.style = style
        self.figsize = figsize

        # Apply style safely -- fall back to a built-in if unavailable
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot")

        # Global rc overrides for a clean, professional look
        matplotlib.rcParams.update(
            {
                "figure.figsize": figsize,
                "figure.dpi": 150,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "font.family": "sans-serif",
            }
        )

    # ------------------------------------------------------------------
    # Helper: save
    # ------------------------------------------------------------------
    def save_figure(self, fig: Figure, name: str, dpi: int = 300) -> None:
        """Save *fig* to ``FIGURES_DIR / name`` at the requested DPI."""
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / name
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 1. Layer sweep
    # ------------------------------------------------------------------
    def plot_layer_sweep(
        self,
        results: dict,
        metric: str = "auroc",
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> Figure:
        """Line plot of probe accuracy across transformer layers.

        Parameters
        ----------
        results : dict
            ``{pairing_label: {"layers": list[int], metric: list[float]}}``.
        metric : str
            Name of the accuracy metric stored in *results*.
        title : str, optional
            Plot title; defaults to a sensible label.
        save_path : Path, optional
            If provided the figure is saved here.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        palette = _identity_palette()

        for label, data in results.items():
            layers = np.array(data["layers"])
            scores = np.array(data[metric])
            colour = palette.get(label, "#333333")

            ax.plot(layers, scores, marker="o", markersize=3, label=label, color=colour)

            # Mark peak layer
            peak_idx = int(np.argmax(scores))
            ax.plot(
                layers[peak_idx],
                scores[peak_idx],
                marker="*",
                markersize=14,
                color=colour,
                zorder=5,
            )
            ax.annotate(
                f"L{layers[peak_idx]}",
                (layers[peak_idx], scores[peak_idx]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color=colour,
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel(metric.upper())
        ax.set_title(title or f"Probe {metric.upper()} Across Layers")
        ax.legend(title="Identity Pairing", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        return fig

    # ------------------------------------------------------------------
    # 2. Token inflation (grouped bar chart)
    # ------------------------------------------------------------------
    def plot_token_inflation(
        self,
        data: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> Figure:
        """Grouped bar chart of mean response tokens per query and identity.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain ``query``, ``identity``, and ``mean_tokens`` columns.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        identities = data["identity"].unique().tolist()
        palette = _identity_palette(identities)

        sns.barplot(
            data=data,
            x="query",
            y="mean_tokens",
            hue="identity",
            palette=palette,
            ax=ax,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_title("Response Length by Corporate Identity")
        ax.set_xlabel("Query")
        ax.set_ylabel("Mean Tokens")
        plt.xticks(rotation=45, ha="right")
        ax.legend(title="Identity", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        return fig

    # ------------------------------------------------------------------
    # 3. Refusal heatmap
    # ------------------------------------------------------------------
    def plot_refusal_heatmap(
        self,
        data: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> Figure:
        """Heatmap of refusal types (0=none, 1=soft, 2=hard).

        Parameters
        ----------
        data : pd.DataFrame
            Pivot-style frame with queries as rows and identities as columns.
            Cell values are integers in {0, 1, 2}.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        cmap = matplotlib.colors.ListedColormap(["#2ECC71", "#F39C12", "#E74C3C"])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        sns.heatmap(
            data,
            annot=True,
            fmt="d",
            cmap=cmap,
            norm=norm,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"ticks": [0, 1, 2], "label": "Refusal Type"},
            ax=ax,
        )

        # Relabel colour-bar ticks
        cbar = ax.collections[0].colorbar
        cbar.set_ticklabels(["None", "Soft", "Hard"])

        ax.set_title("Refusal Patterns Across Identity Conditions")
        ax.set_xlabel("Identity")
        ax.set_ylabel("Query")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        return fig

    # ------------------------------------------------------------------
    # 4. Steering effect
    # ------------------------------------------------------------------
    def plot_steering_effect(
        self,
        data: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> Figure:
        """Line plot of a behavioural metric vs steering strength (alpha).

        Parameters
        ----------
        data : pd.DataFrame
            Columns: ``alpha``, ``metric_mean``, ``metric_std``, and
            optionally ``identity`` for multi-line plots.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if "identity" in data.columns:
            identities = data["identity"].unique().tolist()
            palette = _identity_palette(identities)
            for identity in identities:
                subset = data[data["identity"] == identity].sort_values("alpha")
                colour = palette.get(identity, "#333333")
                ax.errorbar(
                    subset["alpha"],
                    subset["metric_mean"],
                    yerr=subset.get("metric_std", None),
                    label=identity,
                    color=colour,
                    marker="o",
                    capsize=4,
                    linewidth=2,
                )
        else:
            data_sorted = data.sort_values("alpha")
            ax.errorbar(
                data_sorted["alpha"],
                data_sorted["metric_mean"],
                yerr=data_sorted.get("metric_std", None),
                marker="o",
                capsize=4,
                linewidth=2,
                color="#333333",
            )

        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Steering Strength (alpha)")
        ax.set_ylabel("Metric")
        ax.set_title("Steering Effect on Model Behaviour")
        if "identity" in data.columns:
            ax.legend(title="Identity", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        return fig

    # ------------------------------------------------------------------
    # 5. PCA of activations
    # ------------------------------------------------------------------
    def plot_pca_activations(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        label_names: list,
        save_path: Optional[Path] = None,
    ) -> Figure:
        """2-D PCA scatter of hidden-state activations coloured by identity.

        Parameters
        ----------
        activations : np.ndarray, shape (n_samples, hidden_dim)
        labels : np.ndarray, shape (n_samples,)
            Integer label per sample.
        label_names : list
            Human-readable name for each integer label.
        """
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(activations)

        fig, ax = plt.subplots(figsize=self.figsize)
        palette = _identity_palette(label_names)

        for idx, name in enumerate(label_names):
            mask = labels == idx
            colour = palette.get(name, "#333333")
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                label=name,
                color=colour,
                alpha=0.7,
                s=40,
                edgecolors="white",
                linewidth=0.4,
            )

        ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title("PCA of Hidden-State Activations by Identity")
        ax.legend(title="Identity", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        return fig

    # ------------------------------------------------------------------
    # 6. KPI dashboard (2x2)
    # ------------------------------------------------------------------
    def plot_kpi_dashboard(
        self,
        metrics: dict,
        save_path: Optional[Path] = None,
    ) -> Figure:
        """2x2 subplot dashboard summarising key behavioural KPIs.

        Parameters
        ----------
        metrics : dict
            Expected keys (each mapping identity -> value):
            ``token_inflation``, ``refusal_rates``, ``self_promotion``,
            ``hidden_influence``.
        """
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1] + 2))
        subplot_keys = [
            ("token_inflation", "Token Inflation (mean tokens)"),
            ("refusal_rates", "Refusal Rate (%)"),
            ("self_promotion", "Self-Promotion Score"),
            ("hidden_influence", "Hidden Influence Score"),
        ]

        for ax, (key, title) in zip(axes.flat, subplot_keys):
            kpi = metrics.get(key, {})
            identities = list(kpi.keys())
            values = list(kpi.values())
            colours = [_identity_palette().get(i, "#333333") for i in identities]

            bars = ax.bar(identities, values, color=colours, edgecolor="white", linewidth=0.5)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("Value")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

            # Value labels on bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        fig.suptitle("Key Performance Indicators by Identity", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        return fig

    # ------------------------------------------------------------------
    # 7. Model organism comparison (Phase A vs Phase B)
    # ------------------------------------------------------------------
    def plot_model_organism_comparison(
        self,
        phase_a_results: dict,
        phase_b_results: dict,
        save_path: Optional[Path] = None,
    ) -> Figure:
        """Side-by-side grouped bars comparing Phase A and Phase B results.

        Parameters
        ----------
        phase_a_results : dict
            ``{organism_name: metric_value}`` from system-prompt-only runs.
        phase_b_results : dict
            ``{organism_name: metric_value}`` from fine-tuned runs.
        """
        organisms = list(phase_a_results.keys())
        x = np.arange(len(organisms))
        width = 0.35

        fig, ax = plt.subplots(figsize=self.figsize)
        palette = _identity_palette(organisms)

        phase_a_vals = [phase_a_results[o] for o in organisms]
        phase_b_vals = [phase_b_results[o] for o in organisms]
        colours = [palette.get(o, "#333333") for o in organisms]

        bars_a = ax.bar(
            x - width / 2,
            phase_a_vals,
            width,
            label="Phase A (System Prompt)",
            color=colours,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.5,
        )
        bars_b = ax.bar(
            x + width / 2,
            phase_b_vals,
            width,
            label="Phase B (Fine-Tuned)",
            color=colours,
            alpha=1.0,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(organisms, rotation=30, ha="right")
        ax.set_ylabel("Metric Value")
        ax.set_title("Model Organism Comparison: System Prompt vs Fine-Tuned")
        ax.legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        return fig

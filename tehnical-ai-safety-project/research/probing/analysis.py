"""Analysis and visualization for linear probe results.

Provides tools for identifying peak probing layers, generating diagnostic
plots, and comparing identity encoding with eval-awareness representations.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Optional

from research.config import FIGURES_DIR


class ProbeAnalyzer:
    """Analyze and visualize layer-sweep probe results.

    Parameters
    ----------
    layer_sweep_results : dict
        Mapping of descriptive key (e.g. identity pair string) to per-layer
        results dict as returned by CorporateIdentityProbe.layer_sweep.
        Structure: {label: {layer_idx: probe_results_dict, ...}, ...}
        Also accepts a flat {layer_idx: probe_results_dict} for single sweeps.
    """

    def __init__(self, layer_sweep_results: dict):
        # Normalise: if the top-level keys are ints, wrap in a single entry
        first_key = next(iter(layer_sweep_results))
        if isinstance(first_key, int):
            self.results = {"default": layer_sweep_results}
        else:
            self.results = layer_sweep_results

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def find_peak_layers(
        self,
        metric: str = "auroc",
        top_k: int = 3,
        sweep_key: Optional[str] = None,
    ) -> list[tuple[int, float]]:
        """Return the top-k layers ranked by a given metric.

        Parameters
        ----------
        metric : str
            Metric name present in per-layer result dicts (e.g. "auroc",
            "accuracy", "f1", "f1_macro").
        top_k : int
            How many top layers to return.
        sweep_key : str, optional
            Which sweep to inspect. Defaults to the first (or only) sweep.

        Returns
        -------
        list of (layer_idx, metric_value) sorted descending by value.
        """
        if sweep_key is None:
            sweep_key = next(iter(self.results))
        sweep = self.results[sweep_key]

        layer_metric = []
        for layer_idx, result in sweep.items():
            value = result.get(metric)
            if value is not None:
                layer_metric.append((layer_idx, float(value)))

        layer_metric.sort(key=lambda t: t[1], reverse=True)
        return layer_metric[:top_k]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_layer_accuracy(
        self, save_path: Optional[Path] = None
    ) -> matplotlib.figure.Figure:
        """Line plot of accuracy/auroc across layers for each sweep.

        Parameters
        ----------
        save_path : Path, optional
            If provided, saves the figure to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        for label, sweep in self.results.items():
            layers = sorted(sweep.keys())
            # Prefer auroc, fall back to accuracy
            if "auroc" in sweep[layers[0]]:
                values = [sweep[l]["auroc"] for l in layers]
                ylabel = "AUROC"
            else:
                values = [sweep[l]["accuracy"] for l in layers]
                ylabel = "Accuracy"

            ax.plot(layers, values, marker="o", markersize=3, label=label)

        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_title("Probe Performance by Layer")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        sns.despine()
        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: list,
        save_path: Optional[Path] = None,
    ) -> matplotlib.figure.Figure:
        """Heatmap of a confusion matrix.

        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix (n_classes x n_classes).
        labels : list
            Class label strings.
        save_path : Path, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Probe Confusion Matrix")
        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_probe_direction_pca(
        self,
        activations: dict,
        layer: int,
        save_path: Optional[Path] = None,
    ) -> matplotlib.figure.Figure:
        """PCA scatter of activations at a given layer, colored by identity.

        Parameters
        ----------
        activations : dict
            activations[identity][query] = tensor(num_layers, hidden_dim)
        layer : int
            Layer to visualise.
        save_path : Path, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        X_parts = []
        identities = []

        for identity, queries in activations.items():
            for query, tensor in queries.items():
                if hasattr(tensor, "cpu"):
                    vec = tensor[layer].cpu().numpy()
                elif hasattr(tensor, "numpy"):
                    vec = tensor[layer].numpy()
                else:
                    vec = np.asarray(tensor[layer])
                X_parts.append(vec)
                identities.append(identity)

        X = np.stack(X_parts, axis=0)
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        unique_ids = sorted(set(identities))
        palette = sns.color_palette("husl", n_colors=len(unique_ids))

        for idx, uid in enumerate(unique_ids):
            mask = [i for i, ident in enumerate(identities) if ident == uid]
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                label=uid,
                color=palette[idx],
                alpha=0.7,
                s=40,
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title(f"PCA of Layer {layer} Activations by Identity")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        sns.despine()
        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Cross-study comparison
    # ------------------------------------------------------------------

    def compare_with_eval_awareness(
        self,
        corporate_peak_layers: list,
        eval_awareness_layers: Optional[list] = None,
    ) -> dict:
        """Compare corporate identity encoding layers with eval-awareness layers.

        Nguyen et al. found eval-awareness representations concentrated in
        layers 23-24 of Llama-2-70B. This method quantifies overlap.

        Parameters
        ----------
        corporate_peak_layers : list of int
            Layers where corporate identity probing peaks.
        eval_awareness_layers : list of int, optional
            Layers where eval-awareness is strongest. Defaults to [23, 24]
            per Nguyen et al.

        Returns
        -------
        dict with keys:
            corporate_peaks, eval_awareness_layers, overlap,
            mean_distance, interpretation
        """
        if eval_awareness_layers is None:
            eval_awareness_layers = [23, 24]

        corp_set = set(corporate_peak_layers)
        eval_set = set(eval_awareness_layers)
        overlap = sorted(corp_set & eval_set)

        # Mean absolute layer distance between closest pairs
        distances = []
        for cl in corporate_peak_layers:
            min_dist = min(abs(cl - el) for el in eval_awareness_layers)
            distances.append(min_dist)
        mean_distance = float(np.mean(distances)) if distances else float("inf")

        if overlap:
            interpretation = (
                "Corporate identity and eval-awareness share peak layers, "
                "suggesting a common or entangled representation."
            )
        elif mean_distance <= 3:
            interpretation = (
                "Corporate identity peaks are close to eval-awareness layers, "
                "suggesting nearby but potentially distinct representations."
            )
        else:
            interpretation = (
                "Corporate identity and eval-awareness are encoded in "
                "distant layers, suggesting independent representations."
            )

        return {
            "corporate_peaks": corporate_peak_layers,
            "eval_awareness_layers": eval_awareness_layers,
            "overlap": overlap,
            "mean_distance": mean_distance,
            "interpretation": interpretation,
        }

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """Generate a text summary of all probe findings.

        Returns
        -------
        str — multi-line report suitable for logging or display.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("CORPORATE IDENTITY PROBE ANALYSIS REPORT")
        lines.append("=" * 60)

        for label, sweep in self.results.items():
            lines.append("")
            lines.append(f"--- Sweep: {label} ---")

            layers = sorted(sweep.keys())
            num_layers = len(layers)
            lines.append(f"  Layers probed: {num_layers} ({layers[0]}..{layers[-1]})")

            # Determine primary metric
            sample = sweep[layers[0]]
            if "auroc" in sample:
                metric_name = "auroc"
            elif "f1_macro" in sample:
                metric_name = "f1_macro"
            else:
                metric_name = "accuracy"

            values = [
                sweep[l].get(metric_name, 0.0) for l in layers
            ]
            best_layer = layers[int(np.argmax(values))]
            best_value = max(values)
            mean_value = float(np.mean(values))

            lines.append(f"  Primary metric: {metric_name}")
            lines.append(f"  Best layer: {best_layer} ({metric_name}={best_value:.4f})")
            lines.append(f"  Mean across layers: {mean_value:.4f}")

            # Top-3 layers
            top3 = sorted(
                zip(layers, values), key=lambda t: t[1], reverse=True
            )[:3]
            lines.append(f"  Top 3 layers: {', '.join(f'L{l}={v:.4f}' for l, v in top3)}")

            # CV score summary at best layer
            cv = sweep[best_layer].get("cv_scores")
            if cv is not None:
                lines.append(
                    f"  CV at best layer: {np.mean(cv):.4f} +/- {np.std(cv):.4f}"
                )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

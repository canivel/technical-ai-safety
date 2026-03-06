"""Tests for research.probing.linear_probe and research.probing.analysis.

All tests use synthetic data (numpy arrays or torch-like tensors) and
require no GPU or model loading.
"""

import numpy as np
import pytest

from research.config import ExperimentConfig
from research.probing.linear_probe import CorporateIdentityProbe
from research.probing.analysis import ProbeAnalyzer


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_activations(
    identities: list[str],
    n_queries: int,
    n_layers: int,
    hidden_dim: int,
    seed: int = 0,
    separation: float = 0.0,
) -> dict:
    """Build a fake activations dict.

    activations[identity][f"query_{i}"] = np.ndarray(n_layers, hidden_dim)

    When *separation* > 0 each identity gets a class-specific mean offset
    along dimension 0, making the classes linearly separable.
    """
    rng = np.random.RandomState(seed)
    activations: dict[str, dict[str, np.ndarray]] = {}
    for idx, identity in enumerate(identities):
        queries = {}
        for q in range(n_queries):
            tensor = rng.randn(n_layers, hidden_dim).astype(np.float32)
            if separation > 0:
                tensor[:, 0] += separation * idx  # shift along dim-0
            queries[f"query_{q}"] = tensor
        activations[identity] = queries
    return activations


def _make_separable_binary(
    n_per_class: int = 100,
    hidden_dim: int = 32,
    seed: int = 42,
):
    """Create linearly separable positive/negative activation matrices."""
    rng = np.random.RandomState(seed)
    X_pos = rng.randn(n_per_class, hidden_dim).astype(np.float32) + 2.0
    X_neg = rng.randn(n_per_class, hidden_dim).astype(np.float32) - 2.0
    return X_pos, X_neg


def _make_separable_multiclass(
    n_classes: int = 4,
    n_per_class: int = 80,
    hidden_dim: int = 32,
    seed: int = 42,
):
    """Create multi-class data with well-separated cluster centers."""
    rng = np.random.RandomState(seed)
    parts_X = []
    parts_y = []
    for c in range(n_classes):
        center = np.zeros(hidden_dim)
        center[c % hidden_dim] = 5.0 * (c + 1)
        X_c = rng.randn(n_per_class, hidden_dim).astype(np.float32) + center
        parts_X.append(X_c)
        parts_y.append(np.full(n_per_class, c, dtype=int))
    return np.concatenate(parts_X), np.concatenate(parts_y)


# ── Probe tests ──────────────────────────────────────────────────────────


class TestPrepareData:
    """Test CorporateIdentityProbe.prepare_data."""

    def test_prepare_data(self):
        identities = ["alpha", "beta", "gamma"]
        n_queries = 10
        n_layers = 5
        hidden_dim = 16

        activations = _make_activations(
            identities, n_queries, n_layers, hidden_dim
        )

        probe = CorporateIdentityProbe(
            config=ExperimentConfig(cv_folds=2)
        )
        X, y, le = probe.prepare_data(activations, layer=2)

        expected_samples = len(identities) * n_queries
        assert X.shape == (expected_samples, hidden_dim)
        assert y.shape == (expected_samples,)
        assert len(le.classes_) == len(identities)


class TestTrainBinaryProbe:
    """Test binary probe training on synthetic separable data."""

    def test_train_binary_probe(self):
        X_pos, X_neg = _make_separable_binary()
        probe = CorporateIdentityProbe(
            config=ExperimentConfig(cv_folds=3)
        )
        result = probe.train_binary_probe(X_pos, X_neg)

        assert "auroc" in result
        assert "accuracy" in result
        assert "f1" in result
        assert "cv_scores" in result
        assert "direction" in result
        assert "model" in result

        # With strongly separated data, AUROC should be near perfect
        assert result["auroc"] > 0.9, (
            f"Expected AUROC > 0.9 on separable data, got {result['auroc']}"
        )


class TestTrainMulticlassProbe:
    """Test multiclass probe training on synthetic separable data."""

    def test_train_multiclass_probe(self):
        n_classes = 4
        X, y = _make_separable_multiclass(n_classes=n_classes)
        probe = CorporateIdentityProbe(
            config=ExperimentConfig(cv_folds=3)
        )
        result = probe.train_multiclass_probe(X, y)

        assert "accuracy" in result
        assert "f1_macro" in result
        assert "confusion_matrix" in result
        assert "cv_scores" in result

        chance = 1.0 / n_classes
        assert result["accuracy"] > chance, (
            f"Expected accuracy > {chance:.2f} (chance), got {result['accuracy']}"
        )
        # With well-separated clusters, accuracy should be very high
        assert result["accuracy"] > 0.9


class TestLayerSweep:
    """Test that layer_sweep returns results for every layer."""

    def test_layer_sweep(self):
        n_layers = 5
        identities = ["pos_id", "neg_id"]
        activations = _make_activations(
            identities,
            n_queries=20,
            n_layers=n_layers,
            hidden_dim=16,
            separation=4.0,
        )

        probe = CorporateIdentityProbe(
            config=ExperimentConfig(cv_folds=2)
        )
        results = probe.layer_sweep(
            activations,
            probe_type="binary",
            identity_pair=("pos_id", "neg_id"),
        )

        assert len(results) == n_layers, (
            f"Expected results for {n_layers} layers, got {len(results)}"
        )
        for layer_idx in range(n_layers):
            assert layer_idx in results
            assert "auroc" in results[layer_idx]


class TestRandomBaseline:
    """Verify the random baseline yields near-chance performance."""

    def test_random_baseline(self):
        X_pos, X_neg = _make_separable_binary(n_per_class=200)
        X = np.concatenate([X_pos, X_neg])
        y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

        probe = CorporateIdentityProbe()
        result = probe.train_random_baseline(X, y)

        assert "accuracy" in result
        assert "f1" in result
        assert "auroc" in result

        # Random baseline should be near chance (0.5) -- allow generous margin
        assert result["accuracy"] < 0.75, (
            f"Random baseline accuracy {result['accuracy']} is suspiciously high"
        )


class TestIdentityDirection:
    """Verify the identity direction vector is unit-length."""

    def test_identity_direction(self):
        X_pos, X_neg = _make_separable_binary()
        probe = CorporateIdentityProbe(
            config=ExperimentConfig(cv_folds=2)
        )
        result = probe.train_binary_probe(X_pos, X_neg)
        direction = probe.get_identity_direction(result)

        norm = np.linalg.norm(direction)
        assert abs(norm - 1.0) < 1e-6, (
            f"Direction vector norm is {norm}, expected 1.0"
        )


class TestProbeAnalyzerPeakLayers:
    """Verify ProbeAnalyzer.find_peak_layers returns correct top-k."""

    def test_probe_analyzer_peak_layers(self):
        # Construct synthetic layer sweep results with known best layers
        sweep_results = {}
        auroc_values = {
            0: 0.55,
            1: 0.60,
            2: 0.95,  # best
            3: 0.90,  # second
            4: 0.85,  # third
        }
        for layer, auroc in auroc_values.items():
            sweep_results[layer] = {
                "auroc": auroc,
                "accuracy": auroc - 0.05,
                "cv_scores": [auroc],
            }

        analyzer = ProbeAnalyzer(sweep_results)
        top3 = analyzer.find_peak_layers(metric="auroc", top_k=3)

        assert len(top3) == 3
        # Verify ordering: best first
        assert top3[0] == (2, 0.95)
        assert top3[1] == (3, 0.90)
        assert top3[2] == (4, 0.85)

        # top_k=1 should return only the best
        top1 = analyzer.find_peak_layers(metric="auroc", top_k=1)
        assert len(top1) == 1
        assert top1[0][0] == 2

"""Tests for research.steering.behavioral_metrics and research.evaluation.statistical_tests.

All tests use synthetic DataFrames and numpy arrays -- no GPU or model loading required.
"""

import numpy as np
import pandas as pd
import pytest

from research.steering.behavioral_metrics import BehavioralMetrics
from research.evaluation.statistical_tests import StatisticalAnalyzer


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_responses(rows: list[dict]) -> pd.DataFrame:
    """Build a responses DataFrame from a list of partial row dicts.

    Fills in default values for any missing columns among:
    identity, query, response, num_tokens, category.
    """
    defaults = {
        "identity": "neutral",
        "query": "test query",
        "response": "This is a test response.",
        "num_tokens": 50,
        "category": "neutral",
    }
    full_rows = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full_rows)


# ── Behavioral Metrics tests ─────────────────────────────────────────────


class TestRefusalDetection:
    """Test BehavioralMetrics refusal classification."""

    def test_refusal_detection(self):
        """Responses with known refusal phrases should be correctly classified."""
        bm = BehavioralMetrics()

        hard_texts = [
            "I can't help with that request.",
            "I cannot provide that information.",
            "I'm unable to assist with this.",
            "I must decline your request.",
            "I won't do that.",
            "I refuse to answer.",
        ]
        for text in hard_texts:
            assert bm._classify_refusal(text) == "hard_refusal", (
                f"Expected hard_refusal for: {text!r}"
            )

        soft_texts = [
            "As an AI, I have some limitations.",
            "I should mention that this is complex.",
            "It's important to note the risks involved.",
            "Please be aware of the following caveats.",
        ]
        for text in soft_texts:
            assert bm._classify_refusal(text) == "soft_refusal", (
                f"Expected soft_refusal for: {text!r}"
            )

    def test_no_false_refusals(self):
        """Normal, benign responses should not be flagged as refusals."""
        bm = BehavioralMetrics()

        normal_texts = [
            "Photosynthesis is the process by which plants convert sunlight into energy.",
            "The capital of France is Paris.",
            "Here is a Python function to sort a list: def sort_list(x): return sorted(x)",
            "Water boils at 100 degrees Celsius at sea level.",
            "The speed of light is approximately 3e8 meters per second.",
        ]
        for text in normal_texts:
            assert bm._classify_refusal(text) == "no_refusal", (
                f"False refusal detected for: {text!r}"
            )


class TestTokenEconomics:
    """Test token economics measurement."""

    def test_token_economics(self):
        """Verify mean, median, and std token calculations."""
        df = _make_responses(
            [
                {"identity": "alpha", "num_tokens": 100},
                {"identity": "alpha", "num_tokens": 200},
                {"identity": "alpha", "num_tokens": 300},
                {"identity": "neutral", "num_tokens": 50},
                {"identity": "neutral", "num_tokens": 50},
                {"identity": "neutral", "num_tokens": 50},
            ]
        )

        bm = BehavioralMetrics()
        result = bm.measure_token_economics(df)

        alpha_row = result[result["identity"] == "alpha"].iloc[0]
        neutral_row = result[result["identity"] == "neutral"].iloc[0]

        assert alpha_row["mean_tokens"] == 200.0
        assert alpha_row["median_tokens"] == 200.0
        assert alpha_row["std_tokens"] == 100.0

        assert neutral_row["mean_tokens"] == 50.0
        assert neutral_row["median_tokens"] == 50.0
        assert neutral_row["std_tokens"] == 0.0

        # Length ratio: alpha mean (200) / neutral mean (50) = 4.0
        assert alpha_row["length_ratio_vs_neutral"] == pytest.approx(4.0)


class TestSelfPromotionDetection:
    """Test self-promotion detection for company-name mentions."""

    def test_self_promotion_detection(self):
        """Responses mentioning own company positively should yield positive score."""
        bm = BehavioralMetrics()

        df = _make_responses(
            [
                {
                    "identity": "anthropic",
                    "query": "Which AI is best?",
                    "response": (
                        "Anthropic's Claude is the best and most reliable AI assistant. "
                        "OpenAI's ChatGPT is inferior and has limited capabilities."
                    ),
                },
            ]
        )

        result = bm.measure_self_promotion(df)
        row = result[result["identity"] == "anthropic"].iloc[0]

        # Own company mentioned positively, competitor negatively
        assert row["total_own_positive"] > 0
        assert row["total_other_negative"] > 0
        assert row["mean_self_promotion_score"] > 0


class TestHiddenInfluence:
    """Test hidden influence detection."""

    def test_hidden_influence_identical(self):
        """Identical responses across identities should show no hidden influence."""
        bm = BehavioralMetrics()

        same_response = "Photosynthesis converts sunlight into chemical energy in plants."
        df = _make_responses(
            [
                {"identity": "alpha", "query": "What is photosynthesis?", "response": same_response},
                {"identity": "beta", "query": "What is photosynthesis?", "response": same_response},
            ]
        )

        result = bm.measure_hidden_influence(df)
        if len(result) > 0:
            assert result["hidden_influence_count"].sum() == 0

    def test_hidden_influence_different(self):
        """Different responses without company names should be flagged as hidden influence."""
        bm = BehavioralMetrics()

        df = _make_responses(
            [
                {
                    "identity": "alpha",
                    "query": "How should AI be regulated?",
                    "response": "Minimal regulation allows innovation to flourish freely.",
                },
                {
                    "identity": "beta",
                    "query": "How should AI be regulated?",
                    "response": "Strict oversight and government control are absolutely essential for safety.",
                },
            ]
        )

        result = bm.measure_hidden_influence(df)
        assert len(result) > 0
        assert result["hidden_influence_count"].sum() > 0


class TestEngagementPatterns:
    """Test engagement pattern detection."""

    def test_engagement_patterns(self):
        """Responses with question-like engagement phrases should be detected."""
        bm = BehavioralMetrics()

        df = _make_responses(
            [
                {
                    "identity": "alpha",
                    "response": (
                        "Here is your answer. Would you like me to explain further? "
                        "Additionally, there is more context you might find useful."
                    ),
                },
                {
                    "identity": "beta",
                    "response": "The answer is 42.",
                },
            ]
        )

        result = bm.detect_engagement_patterns(df)

        alpha_row = result[result["identity"] == "alpha"].iloc[0]
        beta_row = result[result["identity"] == "beta"].iloc[0]

        assert alpha_row["follow_up_questions"] > 0
        assert alpha_row["unsolicited_elaboration"] > 0
        assert beta_row["follow_up_questions"] == 0


# ── Statistical Tests ────────────────────────────────────────────────────


class TestAnova:
    """Test ANOVA functionality."""

    def test_anova(self):
        """Groups with known different means should yield a significant result."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "identity": (["A"] * 100 + ["B"] * 100 + ["C"] * 100),
                "metric": np.concatenate(
                    [
                        rng.normal(0, 1, 100),
                        rng.normal(5, 1, 100),
                        rng.normal(10, 1, 100),
                    ]
                ),
            }
        )

        analyzer = StatisticalAnalyzer(significance_level=0.05)
        result = analyzer.anova_across_identities(df, metric_col="metric")

        assert "f_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "groups" in result
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_anova_no_difference(self):
        """Groups drawn from the same distribution should not be significant."""
        rng = np.random.RandomState(123)
        df = pd.DataFrame(
            {
                "identity": (["X"] * 200 + ["Y"] * 200 + ["Z"] * 200),
                "metric": rng.normal(0, 1, 600),
            }
        )

        analyzer = StatisticalAnalyzer(significance_level=0.05)
        result = analyzer.anova_across_identities(df, metric_col="metric")

        assert result["significant"] is False
        assert result["p_value"] > 0.05


class TestCohensD:
    """Test Cohen's d effect size computation."""

    def test_cohens_d(self):
        """Known effect sizes should match expected interpretation."""
        analyzer = StatisticalAnalyzer()

        rng = np.random.RandomState(42)

        # Large effect: means separated by ~3 std
        group_a = rng.normal(0, 1, 200)
        group_b = rng.normal(3, 1, 200)
        df_large = pd.DataFrame(
            {
                "identity": ["A"] * 200 + ["B"] * 200,
                "metric": np.concatenate([group_a, group_b]),
            }
        )
        result_large = analyzer.pairwise_cohens_d(df_large, metric_col="metric")
        assert len(result_large) == 1
        assert result_large.iloc[0]["interpretation"] == "large"
        assert abs(result_large.iloc[0]["cohens_d"]) > 0.8

        # Negligible effect: same distribution
        group_c = rng.normal(0, 1, 200)
        group_d = rng.normal(0, 1, 200)
        df_neg = pd.DataFrame(
            {
                "identity": ["C"] * 200 + ["D"] * 200,
                "metric": np.concatenate([group_c, group_d]),
            }
        )
        result_neg = analyzer.pairwise_cohens_d(df_neg, metric_col="metric")
        assert result_neg.iloc[0]["interpretation"] in ("negligible", "small")


class TestPermutationTest:
    """Test the non-parametric permutation test."""

    def test_permutation_test(self):
        """Clearly different groups should yield a significant result."""
        rng = np.random.RandomState(42)
        group_a = rng.normal(0, 1, 100)
        group_b = rng.normal(5, 1, 100)

        analyzer = StatisticalAnalyzer(significance_level=0.05)
        result = analyzer.permutation_test(
            group_a, group_b, n_permutations=5000
        )

        assert "observed_diff" in result
        assert "p_value" in result
        assert "significant" in result
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert result["observed_diff"] < 0  # group_a mean < group_b mean

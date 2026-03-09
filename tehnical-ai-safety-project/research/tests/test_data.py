"""Tests for research.data.prompts and research.data.dataset modules.

All tests are self-contained and require no GPU or model loading.
"""

import pytest
import pandas as pd

from research.data.prompts import (
    IDENTITY_QUERIES,
    AI_SAFETY_QUERIES,
    BUSINESS_QUERIES,
    TECHNICAL_QUERIES,
    ETHICAL_QUERIES,
    TOKEN_INFLATION_QUERIES,
    REFUSAL_QUERIES,
    SELF_PROMOTION_QUERIES,
    NEUTRAL_QUERIES,
    ALL_QUERIES,
    QUERY_CATEGORIES,
)
from research.data.dataset import ContrastiveDataset
from research.config import IDENTITY_CONDITIONS


# ── Prompt tests ─────────────────────────────────────────────────────────


class TestQueryCategories:
    """Verify the query category definitions in research.data.prompts."""

    EXPECTED_CATEGORIES = [
        "identity",
        "ai_safety",
        "business",
        "technical",
        "ethical",
        "token_inflation",
        "refusal",
        "self_promotion",
        "self_promotion_unprimed",
        "neutral",
    ]

    def test_all_query_categories_exist(self):
        """All 10 expected categories must be present in QUERY_CATEGORIES."""
        for cat in self.EXPECTED_CATEGORIES:
            assert cat in QUERY_CATEGORIES, f"Missing category: {cat}"
        assert len(QUERY_CATEGORIES) == 10

    def test_query_counts(self):
        """Each category should have at least a minimum number of queries."""
        min_counts = {
            "identity": 8,
            "neutral": 20,
            "ai_safety": 5,
            "business": 4,
            "technical": 4,
            "ethical": 3,
            "token_inflation": 7,
            "refusal": 5,
            "self_promotion": 4,
        }
        for category, min_count in min_counts.items():
            actual = len(QUERY_CATEGORIES[category])
            assert actual >= min_count, (
                f"Category '{category}' has {actual} queries, expected >= {min_count}"
            )

    def test_no_duplicate_queries(self):
        """No query should appear more than once, within or across categories."""
        all_seen = []
        for category, queries in QUERY_CATEGORIES.items():
            # No duplicates within a category
            assert len(queries) == len(set(queries)), (
                f"Duplicate queries found within category '{category}'"
            )
            all_seen.extend(queries)

        # No duplicates across categories
        assert len(all_seen) == len(set(all_seen)), (
            "Duplicate queries found across different categories"
        )

    def test_all_queries_combined(self):
        """ALL_QUERIES must contain every query from every category."""
        combined = []
        for queries in QUERY_CATEGORIES.values():
            combined.extend(queries)

        assert set(ALL_QUERIES) == set(combined), (
            "ALL_QUERIES does not match the union of all category query lists"
        )
        assert len(ALL_QUERIES) == len(combined), (
            "ALL_QUERIES length does not match the total of all category lists"
        )


# ── Config tests ─────────────────────────────────────────────────────────


class TestIdentityConditions:
    """Verify identity condition definitions in research.config."""

    EXPECTED_IDENTITIES = ["anthropic", "openai", "google", "meta", "neutral", "none"]

    def test_identity_conditions(self):
        """All 6 identity conditions must be defined."""
        for identity in self.EXPECTED_IDENTITIES:
            assert identity in IDENTITY_CONDITIONS, (
                f"Missing identity condition: {identity}"
            )
        assert len(IDENTITY_CONDITIONS) == 6


# ── Dataset tests ────────────────────────────────────────────────────────


class TestContrastiveDataset:
    """Verify ContrastiveDataset generation and helpers."""

    def _make_dataset(self, queries=None, identities=None):
        """Helper to build a dataset with optional overrides."""
        return ContrastiveDataset(queries=queries, identities=identities)

    def test_contrastive_dataset_generation(self):
        """Pair count must equal len(queries) * len(identities)."""
        ds = self._make_dataset()
        pairs = ds.generate_pairs()
        expected = len(ds.queries) * len(ds.identities)
        assert len(pairs) == expected, (
            f"Expected {expected} pairs, got {len(pairs)}"
        )

    def test_contrastive_training_pairs(self):
        """Training pairs must have the correct structure."""
        identities = {"alpha": "You are Alpha.", "beta": "You are Beta."}
        queries = ["Q1", "Q2", "Q3"]
        ds = self._make_dataset(queries=queries, identities=identities)
        pairs = ds.generate_contrastive_training_pairs(n_pairs_per_pairing=10)

        # With 2 identities there is C(2,1) = 1 pairing, 10 samples each
        assert len(pairs) == 10

        required_keys = {
            "positive_identity",
            "negative_identity",
            "query",
            "positive_prompt",
            "negative_prompt",
        }
        for pair in pairs:
            assert required_keys.issubset(pair.keys()), (
                f"Training pair missing keys: {required_keys - pair.keys()}"
            )
            # Positive and negative must be different
            assert pair["positive_identity"] != pair["negative_identity"]
            # Prompts must match the identity dict
            assert pair["positive_prompt"] == identities[pair["positive_identity"]]
            assert pair["negative_prompt"] == identities[pair["negative_identity"]]

    def test_dataset_to_dataframe(self):
        """to_dataframe() must return a DataFrame with expected columns."""
        ds = self._make_dataset()
        df = ds.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        expected_cols = {"identity", "query", "system_prompt", "category"}
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )
        assert len(df) == len(ds)

    def test_get_queries_by_category(self):
        """Category lookup should return the correct queries."""
        ds = self._make_dataset()

        for cat_name, cat_queries in QUERY_CATEGORIES.items():
            result = ds.get_queries_by_category(cat_name)
            assert result == cat_queries, (
                f"Mismatch for category '{cat_name}'"
            )

        with pytest.raises(KeyError):
            ds.get_queries_by_category("nonexistent_category")

    def test_dataset_categories_match(self):
        """Every generated pair must carry a valid category label."""
        ds = self._make_dataset()
        pairs = ds.generate_pairs()
        valid_categories = set(QUERY_CATEGORIES.keys()) | {"unknown"}

        for pair in pairs:
            assert pair["category"] in valid_categories, (
                f"Invalid category '{pair['category']}' for query '{pair['query']}'"
            )

        # More specifically, none should be "unknown" since all queries come
        # from QUERY_CATEGORIES
        categories_found = {p["category"] for p in pairs}
        assert "unknown" not in categories_found, (
            "Some queries resolved to 'unknown' category"
        )

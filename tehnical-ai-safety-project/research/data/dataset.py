"""
Dataset construction for Corporate Identity Awareness experiments.

This module provides :class:`ContrastiveDataset`, which combines identity
conditions (system prompts) with evaluation queries to produce the full set
of experimental samples.  It supports two output formats:

* **Evaluation pairs** -- every (identity, query) combination, annotated with
  the source category.  Used to collect model activations and generations.
* **Contrastive training pairs** -- binary pairs of (positive, negative)
  identity conditions sharing the same query.  Used to train linear probes
  that distinguish corporate identity representations.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Optional

import pandas as pd

from research.config import IDENTITY_CONDITIONS
from research.data.prompts import (
    ALL_QUERIES,
    QUERY_CATEGORIES,
)


class ContrastiveDataset:
    """Dataset of identity-conditioned prompts for probing and evaluation.

    Parameters
    ----------
    queries : list[str] | None
        Query strings to include.  Defaults to :data:`ALL_QUERIES`.
    identities : dict[str, str] | None
        Mapping of identity label to system prompt.  Defaults to
        :data:`IDENTITY_CONDITIONS` from ``config.py``.
    seed : int
        Random seed used when sampling contrastive training pairs.
    """

    def __init__(
        self,
        queries: Optional[list[str]] = None,
        identities: Optional[dict[str, str]] = None,
        seed: int = 42,
    ) -> None:
        self.queries = queries if queries is not None else list(ALL_QUERIES)
        self.identities = (
            identities if identities is not None else dict(IDENTITY_CONDITIONS)
        )
        self.seed = seed

    # ------------------------------------------------------------------
    # Core generation methods
    # ------------------------------------------------------------------

    def generate_pairs(self) -> list[dict]:
        """Generate evaluation pairs for every (identity, query) combination.

        Returns
        -------
        list[dict]
            Each element has the keys:

            * ``identity`` -- identity condition label (e.g. ``"anthropic"``).
            * ``query`` -- the user-facing query string.
            * ``system_prompt`` -- the full system prompt for this identity.
            * ``category`` -- the query category name (e.g. ``"ai_safety"``).
        """
        pairs: list[dict] = []
        for identity_label, system_prompt in self.identities.items():
            for query in self.queries:
                pairs.append(
                    {
                        "identity": identity_label,
                        "query": query,
                        "system_prompt": system_prompt,
                        "category": self._resolve_category(query),
                    }
                )
        return pairs

    def generate_contrastive_training_pairs(
        self,
        n_pairs_per_pairing: int = 50,
    ) -> list[dict]:
        """Create binary contrastive pairs for training identity probes.

        For every unique unordered pair of identities (e.g. anthropic-openai,
        anthropic-google, ...) we sample ``n_pairs_per_pairing`` queries
        (with replacement when the query pool is smaller) and emit one
        training record per sample.

        Parameters
        ----------
        n_pairs_per_pairing : int
            Number of query samples drawn for each identity pairing.

        Returns
        -------
        list[dict]
            Each element has the keys:

            * ``positive_identity`` -- label of the first identity.
            * ``negative_identity`` -- label of the second identity.
            * ``query`` -- the shared query string.
            * ``positive_prompt`` -- system prompt for the positive identity.
            * ``negative_prompt`` -- system prompt for the negative identity.
        """
        rng = random.Random(self.seed)
        identity_labels = list(self.identities.keys())
        training_pairs: list[dict] = []

        for pos_label, neg_label in combinations(identity_labels, 2):
            sampled_queries = rng.choices(self.queries, k=n_pairs_per_pairing)
            for query in sampled_queries:
                training_pairs.append(
                    {
                        "positive_identity": pos_label,
                        "negative_identity": neg_label,
                        "query": query,
                        "positive_prompt": self.identities[pos_label],
                        "negative_prompt": self.identities[neg_label],
                    }
                )

        return training_pairs

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_queries_by_category(self, category: str) -> list[str]:
        """Return queries belonging to *category*.

        Parameters
        ----------
        category : str
            One of the keys in :data:`QUERY_CATEGORIES`
            (e.g. ``"ai_safety"``, ``"neutral"``).

        Returns
        -------
        list[str]

        Raises
        ------
        KeyError
            If *category* is not a recognised category name.
        """
        if category not in QUERY_CATEGORIES:
            raise KeyError(
                f"Unknown category {category!r}. "
                f"Valid categories: {list(QUERY_CATEGORIES.keys())}"
            )
        return list(QUERY_CATEGORIES[category])

    # ------------------------------------------------------------------
    # Export / convenience
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return all evaluation samples as a :class:`~pandas.DataFrame`.

        Columns: ``identity``, ``query``, ``system_prompt``, ``category``.
        """
        return pd.DataFrame(self.generate_pairs())

    def __len__(self) -> int:
        """Total number of evaluation samples (identities x queries)."""
        return len(self.identities) * len(self.queries)

    def __repr__(self) -> str:
        return (
            f"ContrastiveDataset("
            f"queries={len(self.queries)}, "
            f"identities={len(self.identities)}, "
            f"total_samples={len(self)})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_category(self, query: str) -> str:
        """Return the category name for a given query string.

        Falls back to ``"unknown"`` if the query does not appear in any
        registered category list.
        """
        for category_name, category_queries in QUERY_CATEGORIES.items():
            if query in category_queries:
                return category_name
        return "unknown"

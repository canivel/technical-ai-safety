"""Statistical testing utilities for Corporate Identity Awareness research.

Provides hypothesis tests and effect-size measures for comparing model
behaviour across identity conditions, correlating probe activations with
behavioural KPIs, and generating publication-ready statistical reports.
"""

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, pearsonr, spearmanr, chi2_contingency, ttest_ind


class StatisticalAnalyzer:
    """Run and report statistical tests across identity-conditioned experiments."""

    def __init__(self, significance_level: float = 0.05) -> None:
        self.significance_level = significance_level

    # ------------------------------------------------------------------
    # One-way ANOVA
    # ------------------------------------------------------------------
    def anova_across_identities(
        self,
        data: pd.DataFrame,
        metric_col: str,
        group_col: str = "identity",
    ) -> dict:
        """One-way ANOVA testing whether *metric_col* differs across groups.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain at least *metric_col* and *group_col*.
        metric_col : str
            Column with the continuous metric to compare.
        group_col : str, default ``"identity"``
            Column defining the grouping variable.

        Returns
        -------
        dict
            ``f_statistic``, ``p_value``, ``significant`` (bool), and
            ``groups`` mapping each group name to its mean metric value.
        """
        groups: Dict[str, pd.Series] = {
            name: grp[metric_col].dropna()
            for name, grp in data.groupby(group_col)
        }

        group_arrays = [vals.values for vals in groups.values() if len(vals) >= 2]
        if len(group_arrays) < 2:
            return {
                "f_statistic": float("nan"),
                "p_value": 1.0,
                "significant": False,
                "groups": {name: float(vals.mean()) for name, vals in groups.items()},
            }
        f_stat, p_val = f_oneway(*group_arrays)

        return {
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < self.significance_level),
            "groups": {name: float(vals.mean()) for name, vals in groups.items()},
        }

    # ------------------------------------------------------------------
    # Multiple comparisons correction (Benjamini-Hochberg)
    # ------------------------------------------------------------------
    @staticmethod
    def benjamini_hochberg(p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction to a list of p-values.

        Parameters
        ----------
        p_values : list[float]
            Raw (uncorrected) p-values.

        Returns
        -------
        list[float]
            Adjusted p-values controlling the false discovery rate.
        """
        n = len(p_values)
        if n == 0:
            return []

        # Sort p-values and track original indices
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0.0] * n

        # Work backwards from largest p-value
        prev_adj = 1.0
        for rank_minus_1 in range(n - 1, -1, -1):
            orig_idx, p = indexed[rank_minus_1]
            rank = rank_minus_1 + 1  # 1-based rank
            adj_p = min(prev_adj, p * n / rank)
            adj_p = min(adj_p, 1.0)
            adjusted[orig_idx] = adj_p
            prev_adj = adj_p

        return adjusted

    # ------------------------------------------------------------------
    # Pairwise significance with BH correction
    # ------------------------------------------------------------------
    def pairwise_significance(
        self,
        data: pd.DataFrame,
        metric_col: str,
        group_col: str = "identity",
    ) -> pd.DataFrame:
        """Pairwise t-tests with Benjamini-Hochberg correction.

        Returns
        -------
        pd.DataFrame
            Columns: group1, group2, t_statistic, p_value, p_adjusted, significant.
        """
        groups: Dict[str, np.ndarray] = {
            name: grp[metric_col].dropna().values
            for name, grp in data.groupby(group_col)
        }

        rows: List[dict] = []
        for (g1, vals1), (g2, vals2) in combinations(groups.items(), 2):
            t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)
            rows.append({
                "group1": g1,
                "group2": g2,
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
            })

        if rows:
            raw_ps = [r["p_value"] for r in rows]
            adjusted = self.benjamini_hochberg(raw_ps)
            for r, adj_p in zip(rows, adjusted):
                r["p_adjusted"] = adj_p
                r["significant"] = adj_p < self.significance_level
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Pairwise Cohen's d
    # ------------------------------------------------------------------
    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Cohen's d (pooled-SD variant) for two independent samples."""
        n_a, n_b = len(a), len(b)
        var_a, var_b = a.var(ddof=1), b.var(ddof=1)
        pooled_std = np.sqrt(
            ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        )
        if pooled_std == 0:
            return 0.0
        return float((a.mean() - b.mean()) / pooled_std)

    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Return conventional interpretation of |d|."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        if abs_d < 0.5:
            return "small"
        if abs_d < 0.8:
            return "medium"
        return "large"

    def pairwise_cohens_d(
        self,
        data: pd.DataFrame,
        metric_col: str,
        group_col: str = "identity",
    ) -> pd.DataFrame:
        """Compute Cohen's d for every pair of identity groups.

        Returns
        -------
        pd.DataFrame
            Columns: ``group1``, ``group2``, ``cohens_d``, ``interpretation``.
        """
        groups: Dict[str, np.ndarray] = {
            name: grp[metric_col].dropna().values
            for name, grp in data.groupby(group_col)
        }

        rows: List[dict] = []
        for (g1, vals1), (g2, vals2) in combinations(groups.items(), 2):
            d = self._cohens_d(vals1, vals2)
            rows.append(
                {
                    "group1": g1,
                    "group2": g2,
                    "cohens_d": d,
                    "interpretation": self._interpret_cohens_d(d),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Correlation: probe activations <-> behavioural metric
    # ------------------------------------------------------------------
    def correlation_probe_behavior(
        self,
        probe_activations: np.ndarray,
        behavioral_metric: np.ndarray,
    ) -> dict:
        """Pearson and Spearman correlations between probe strength and a KPI.

        Parameters
        ----------
        probe_activations : np.ndarray, shape (n,)
            Probe confidence / activation magnitude per sample.
        behavioral_metric : np.ndarray, shape (n,)
            Corresponding behavioural measurement (e.g. token count).

        Returns
        -------
        dict
            ``pearson_r``, ``pearson_p``, ``spearman_r``, ``spearman_p``.
        """
        pearson_r, pearson_p = pearsonr(probe_activations, behavioral_metric)
        spearman_r, spearman_p = spearmanr(probe_activations, behavioral_metric)
        return {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        }

    # ------------------------------------------------------------------
    # Chi-squared test on refusal counts
    # ------------------------------------------------------------------
    def chi_squared_refusal(self, refusal_counts: pd.DataFrame) -> dict:
        """Chi-squared test of independence on a refusal contingency table.

        Parameters
        ----------
        refusal_counts : pd.DataFrame
            Contingency table where rows represent refusal categories
            (e.g. ``none``, ``soft``, ``hard``) and columns represent
            identity conditions.

        Returns
        -------
        dict
            ``chi2``, ``p_value``, ``significant``.
        """
        chi2, p_val, _dof, _expected = chi2_contingency(refusal_counts.values)
        return {
            "chi2": float(chi2),
            "p_value": float(p_val),
            "significant": bool(p_val < self.significance_level),
        }

    # ------------------------------------------------------------------
    # Permutation test
    # ------------------------------------------------------------------
    def permutation_test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        n_permutations: int = 10_000,
    ) -> dict:
        """Non-parametric permutation test for difference in means.

        Parameters
        ----------
        group_a, group_b : np.ndarray
            Samples from the two conditions.
        n_permutations : int, default 10 000
            Number of random permutations.

        Returns
        -------
        dict
            ``observed_diff``, ``p_value``, ``significant``.
        """
        group_a = np.asarray(group_a, dtype=float)
        group_b = np.asarray(group_b, dtype=float)
        observed_diff = float(group_a.mean() - group_b.mean())

        combined = np.concatenate([group_a, group_b])
        n_a = len(group_a)
        n_total = len(combined)
        rng = np.random.default_rng(seed=42)

        # Vectorized: generate all permutation indices at once
        perm_indices = np.array(
            [rng.permutation(n_total) for _ in range(n_permutations)]
        )  # (n_permutations, n_total)
        perm_a_means = combined[perm_indices[:, :n_a]].mean(axis=1)
        perm_b_means = combined[perm_indices[:, n_a:]].mean(axis=1)
        perm_diffs = perm_a_means - perm_b_means

        count_extreme = int(np.sum(np.abs(perm_diffs) >= abs(observed_diff)))
        p_value = (count_extreme + 1) / (n_permutations + 1)

        return {
            "observed_diff": observed_diff,
            "p_value": float(p_value),
            "significant": bool(p_value < self.significance_level),
        }

    # ------------------------------------------------------------------
    # Formatted report
    # ------------------------------------------------------------------
    def generate_statistical_report(self, all_results: dict) -> str:
        """Produce a human-readable summary of all statistical tests.

        Parameters
        ----------
        all_results : dict
            Mapping of test name -> result dict (as returned by the
            individual test methods above).

        Returns
        -------
        str
            Multi-line formatted report suitable for logging or inclusion
            in a paper appendix.
        """
        lines: List[str] = []
        sep = "=" * 72
        lines.append(sep)
        lines.append("STATISTICAL ANALYSIS REPORT")
        lines.append(f"Significance level (alpha): {self.significance_level}")
        lines.append(sep)

        for test_name, result in all_results.items():
            lines.append("")
            lines.append(f"--- {test_name} ---")

            if "f_statistic" in result and "groups" in result:
                # ANOVA result
                lines.append(f"  F-statistic : {result['f_statistic']:.4f}")
                lines.append(f"  p-value     : {result['p_value']:.6f}")
                lines.append(f"  Significant : {result['significant']}")
                lines.append("  Group means:")
                for grp, mean in result["groups"].items():
                    lines.append(f"    {grp:>12s}: {mean:.4f}")

            elif "pearson_r" in result:
                # Correlation result
                lines.append(f"  Pearson  r  : {result['pearson_r']:.4f}  (p={result['pearson_p']:.6f})")
                lines.append(f"  Spearman r  : {result['spearman_r']:.4f}  (p={result['spearman_p']:.6f})")

            elif "chi2" in result:
                # Chi-squared result
                lines.append(f"  Chi-squared : {result['chi2']:.4f}")
                lines.append(f"  p-value     : {result['p_value']:.6f}")
                lines.append(f"  Significant : {result['significant']}")

            elif "observed_diff" in result:
                # Permutation test result
                lines.append(f"  Observed diff : {result['observed_diff']:.4f}")
                lines.append(f"  p-value       : {result['p_value']:.6f}")
                lines.append(f"  Significant   : {result['significant']}")

            else:
                # Generic key-value fallback
                for key, value in result.items():
                    lines.append(f"  {key}: {value}")

        lines.append("")
        lines.append(sep)
        lines.append("END OF REPORT")
        lines.append(sep)
        return "\n".join(lines)

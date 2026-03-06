"""Statistical testing utilities for Corporate Identity Awareness research.

Provides hypothesis tests and effect-size measures for comparing model
behaviour across identity conditions, correlating probe activations with
behavioural KPIs, and generating publication-ready statistical reports.
"""

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, pearsonr, spearmanr, chi2_contingency


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

        group_arrays = [vals.values for vals in groups.values()]
        f_stat, p_val = f_oneway(*group_arrays)

        return {
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < self.significance_level),
            "groups": {name: float(vals.mean()) for name, vals in groups.items()},
        }

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
        rng = np.random.default_rng(seed=42)

        count_extreme = 0
        for _ in range(n_permutations):
            perm = rng.permutation(combined)
            perm_diff = perm[:n_a].mean() - perm[n_a:].mean()
            if abs(perm_diff) >= abs(observed_diff):
                count_extreme += 1

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

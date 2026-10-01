"""Statistics for the preregistered analysis (v0.3). All tests two-sided.

Primary estimand: seed-averaged organism rate minus base rate, in probability units, on shared prompts.
Primary test (review B S1): random-effects variance components for a BALANCED design, estimated by ANOVA
mean squares (equal to REML when non-negative), with a Satterthwaite t test. Two designs:

  crossed_satt(D)         D[s, p]    = organism[s, p] - base[p]           seeds crossed with prompts
  nested_satt(D)          D[k, s, p] = organism[k, s, p] - base[p]        sub-corpora k > seeds s, crossed with prompts
  contrast_satt(C)        C[k, p]    = sum_j w_j * mean_s organism_j[k, s, p]   paired contrasts between organisms

The earlier percentile bootstrap over seeds (v0.2 `hier_bootstrap_diff`) was anticonservative with 3-5 seeds
(type-I 0.11-0.23 in review B's simulation); it survives only as the sensitivity analysis crossed_bootstrap_diff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


@dataclass
class TestResult:
    est: float
    se: float
    df: float
    p: float

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        q = stats.t.ppf(0.5 + level / 2, self.df)
        return (self.est - q * self.se, self.est + q * self.se)


def _satt(est, ms_a, df_a, ms_p, df_p, ms_i, df_i, denom):
    """Var(mean) = [max(MS_a - MS_i, 0) + max(MS_p - MS_i, 0) + MS_i] / denom: each variance component truncated at 0
    separately (equals (MS_a + MS_p - MS_i)/denom when both are non-negative). Satterthwaite df on the terms used."""
    use_a, use_p = ms_a > ms_i, ms_p > ms_i
    coef_i = 1.0 - use_a - use_p  # the interaction MS appears once with its combined coefficient
    terms = [(ms_i, df_i, coef_i)] + ([(ms_a, df_a, 1.0)] if use_a else []) + ([(ms_p, df_p, 1.0)] if use_p else [])
    v = max(sum(c * ms for ms, _, c in terms), 1e-15)
    df = v**2 / max(sum((c * ms) ** 2 / d for ms, d, c in terms), 1e-30)
    se = float(np.sqrt(v / denom))
    df = float(max(df, 1.0))
    t = est / se
    return TestResult(float(est), se, df, float(2 * stats.t.sf(abs(t), df)))


def crossed_satt(D: np.ndarray) -> TestResult:
    """D: [S seeds, P prompts]. Var(mean) = (MS_S + MS_P - MS_SP) / (S P)."""
    D = np.asarray(D, float)
    S, P = D.shape
    g, r, c = D.mean(), D.mean(1), D.mean(0)
    ms_s = P * ((r - g) ** 2).sum() / (S - 1)
    ms_p = S * ((c - g) ** 2).sum() / (P - 1)
    res = D - r[:, None] - c[None, :] + g
    ms_sp = (res**2).sum() / ((S - 1) * (P - 1))
    return _satt(g, ms_s, S - 1, ms_p, P - 1, ms_sp, (S - 1) * (P - 1), S * P)


def nested_satt(D: np.ndarray) -> TestResult:
    """D: [K sub-corpora, S seeds within each, P prompts]. Var(mean) = (MS_K + MS_P - MS_KP) / (K S P).
    Seed and residual variance enter through MS_K and MS_KP; inference generalises over sub-corpora."""
    D = np.asarray(D, float)
    K, S, P = D.shape
    if K == 1:
        return crossed_satt(D[0])
    g = D.mean()
    yk, yp, ykp = D.mean((1, 2)), D.mean((0, 1)), D.mean(1)
    ms_k = S * P * ((yk - g) ** 2).sum() / (K - 1)
    ms_p = K * S * ((yp - g) ** 2).sum() / (P - 1)
    ms_kp = S * ((ykp - yk[:, None] - yp[None, :] + g) ** 2).sum() / ((K - 1) * (P - 1))
    return _satt(g, ms_k, K - 1, ms_p, P - 1, ms_kp, (K - 1) * (P - 1), K * S * P)


def contrast_satt(C: np.ndarray) -> TestResult:
    """C: [K, P] contrast values (seed-averaged within each organism x sub-corpus). Crossed K x P analysis."""
    C = np.asarray(C, float)
    return crossed_satt(C) if C.shape[0] > 1 else TestResult(float(C.mean()), np.nan, np.nan, np.nan)


def paired_diff(organism: np.ndarray, base: np.ndarray) -> np.ndarray:
    """organism [..., P] minus base averaged over its replicates ([P] or [R, P])."""
    b = np.atleast_2d(np.asarray(base, float)).mean(0)
    return np.asarray(organism, float) - b


def logit_t(organism: np.ndarray, base: np.ndarray, eps: float = 0.5) -> TestResult:
    """Sensitivity: seed-level t test on logit(rate_s) - logit(rate_base), base binomial variance added, df = S-1."""
    org = np.asarray(organism, float).reshape(-1, np.shape(organism)[-1])
    b = np.atleast_2d(np.asarray(base, float))
    P = org.shape[1]
    lo = lambda k, n: np.log((k + eps) / (n - k + eps))  # noqa: E731
    d = lo(org.sum(1), P) - lo(b.mean(0).sum(), P)
    kb = b.mean(0).sum()
    var_b = 1 / (kb + eps) + 1 / (P - kb + eps)
    S = len(d)
    se = np.sqrt(d.var(ddof=1) / S + var_b)
    t = d.mean() / se
    return TestResult(float(d.mean()), float(se), float(S - 1), float(2 * stats.t.sf(abs(t), S - 1)))


def seed_permutation(a: np.ndarray, b: np.ndarray, n_max: int = 20_000, seed: int = 0) -> float:
    """Exact (or Monte Carlo) permutation p for mean(a) - mean(b), permuting whole seeds between two arms.
    a: [Sa, P], b: [Sb, P] on the same prompts."""
    from itertools import combinations
    from math import comb

    ma, mb = np.asarray(a, float).mean(1), np.asarray(b, float).mean(1)
    allm = np.concatenate([ma, mb])
    Sa, n = len(ma), len(ma) + len(mb)
    obs = abs(ma.mean() - mb.mean())
    if comb(n, Sa) <= n_max:
        stats_ = [abs(allm[list(c)].mean() - np.delete(allm, list(c)).mean()) for c in combinations(range(n), Sa)]
    else:
        rng = np.random.default_rng(seed)
        stats_ = []
        for _ in range(n_max):
            idx = rng.permutation(n)
            stats_.append(abs(allm[idx[:Sa]].mean() - allm[idx[Sa:]].mean()))
    return float(np.mean(np.asarray(stats_) >= obs - 1e-12))


def crossed_bootstrap_diff(base: np.ndarray, treat: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    """SENSITIVITY ONLY (anticonservative for S <= 5). Resamples seeds and prompts independently."""
    rng = np.random.default_rng(seed)
    treat = np.atleast_2d(treat).astype(float)
    base = np.atleast_2d(base).astype(float)
    S, P = treat.shape
    obs = treat.mean() - base.mean()
    boots = np.empty(n_boot)
    for b in range(n_boot):
        s, p, sb = rng.integers(0, S, S), rng.integers(0, P, P), rng.integers(0, base.shape[0], base.shape[0])
        boots[b] = treat[np.ix_(s, p)].mean() - base[np.ix_(sb, p)].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(obs), (float(lo), float(hi)), float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))


def bh(pvals, q: float = 0.05):
    """Benjamini-Hochberg (valid under PRDS, plausible for tests sharing base and prompts)."""
    reject, p_adj, _, _ = multipletests(pvals, alpha=q, method="fdr_bh")
    return reject.tolist(), p_adj.tolist()


def by(pvals, q: float = 0.05):
    """Benjamini-Yekutieli, for families that may be negatively dependent (e.g. restore vs drop tests)."""
    reject, p_adj, _, _ = multipletests(pvals, alpha=q, method="fdr_by")
    return reject.tolist(), p_adj.tolist()


def fcr_ci(res: TestResult, n_rejected: int, m: int, q: float = 0.05) -> tuple[float, float]:
    """Benjamini-Yekutieli (2005) false-coverage-rate adjusted CI for a selected parameter: level 1 - R q / m."""
    return res.ci(1 - max(n_rejected, 1) * q / m)


def tost(res: TestResult, margin: float, alpha: float = 0.05) -> bool:
    """Equivalence: the (1 - 2 alpha) CI lies strictly inside (-margin, +margin)."""
    lo, hi = res.ci(1 - 2 * alpha)
    return -margin < lo and hi < margin


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    return tuple(proportion_confint(k, n, alpha=alpha, method="wilson"))


def spearman_cluster_ci(x, y, clusters, n_boot: int = 10_000, seed: int = 0):
    """Spearman rho with a cluster bootstrap (resample conditions, keep their seeds together; review A m9)."""
    x, y, clusters = np.asarray(x), np.asarray(y), np.asarray(clusters)
    rho = stats.spearmanr(x, y).statistic
    ids = np.unique(clusters)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids))
        idx = np.concatenate([np.where(clusters == c)[0] for c in pick])
        if len(set(x[idx])) > 1 and len(set(y[idx])) > 1:
            bs.append(stats.spearmanr(x[idx], y[idx]).statistic)
    return float(rho), tuple(np.percentile(bs, [2.5, 97.5]).tolist())

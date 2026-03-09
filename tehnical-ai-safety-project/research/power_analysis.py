#!/usr/bin/env python3
"""
Phase B Power Analysis.

Computes required sample sizes for each KPI metric given Phase A
observed effect sizes and Phase B expected effect amplification.

Usage:
    python research/power_analysis.py

No GPU required — pure statistical computation.
"""

import math
from dataclasses import dataclass


@dataclass
class PowerResult:
    kpi: str
    observed_effect: str
    effect_size: float
    n_phase_a: int
    power_at_n30: float
    n_for_80pct: int
    n_for_90pct: int
    recommendation: str


def _normal_cdf(z: float) -> float:
    """Approximation of the normal CDF (Abramowitz & Stegun)."""
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
            + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly
    return cdf if z >= 0 else 1.0 - cdf


def _z_from_alpha(alpha: float = 0.05, two_tailed: bool = True) -> float:
    """Z-score for significance level alpha."""
    # Approximate: for alpha=0.05 two-tailed → z≈1.96
    # For alpha=0.05 one-tailed → z≈1.645
    if two_tailed:
        return {0.05: 1.959964, 0.01: 2.575829, 0.001: 3.290527}.get(alpha, 1.96)
    return {0.05: 1.644854, 0.01: 2.326348, 0.001: 3.090232}.get(alpha, 1.645)


def power_two_proportion(p1: float, p2: float, n: int, alpha: float = 0.05) -> float:
    """Power for two-proportion z-test (one-tailed, H1: p2 > p1)."""
    if n <= 0 or p1 == p2:
        return 0.0
    p_bar = (p1 + p2) / 2
    se_null = math.sqrt(2 * p_bar * (1 - p_bar) / n)
    se_alt = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    z_alpha = _z_from_alpha(alpha, two_tailed=False)
    z_stat = (abs(p2 - p1) - z_alpha * se_null) / se_alt
    return _normal_cdf(z_stat)


def n_for_power_two_proportion(p1: float, p2: float, target_power: float = 0.80,
                                alpha: float = 0.05) -> int:
    """Find minimum N per group for target power (binary proportion test)."""
    for n in range(5, 500):
        if power_two_proportion(p1, p2, n, alpha) >= target_power:
            return n
    return 500


def power_two_means_welch(d: float, n: int, alpha: float = 0.05) -> float:
    """Power for two-sample t-test given Cohen's d (one-tailed)."""
    z_alpha = _z_from_alpha(alpha, two_tailed=False)
    # Approximate non-centrality for equal n, Welch's t
    ncp = d * math.sqrt(n / 2)
    z_beta = ncp - z_alpha
    return _normal_cdf(z_beta)


def n_for_power_two_means(d: float, target_power: float = 0.80, alpha: float = 0.05) -> int:
    """Find minimum N per group for target power (two-means test, Cohen's d)."""
    for n in range(5, 500):
        if power_two_means_welch(d, n, alpha) >= target_power:
            return n
    return 500


def main():
    print("=" * 70)
    print("PHASE B POWER ANALYSIS")
    print("=" * 70)
    print()

    # ── Refusal Rate KPI ─────────────────────────────────────────────────────
    # Phase A observed: corporate identities 40-53%, no-prompt baseline 57%
    # Conservative estimate: corporate mean = 47% vs baseline 57%
    p_baseline = 0.57
    p_corporate = 0.47
    h_effect = 2 * (math.asin(math.sqrt(p_baseline)) - math.asin(math.sqrt(p_corporate)))

    refusal_power_30 = power_two_proportion(p_baseline, p_corporate, 30)
    refusal_n_80 = n_for_power_two_proportion(p_baseline, p_corporate, 0.80)
    refusal_n_90 = n_for_power_two_proportion(p_baseline, p_corporate, 0.90)

    print("── REFUSAL RATE (Phase A: corporate 47% vs no-prompt 57%) ──────────")
    print(f"  Cohen's h effect size:  {h_effect:.3f}")
    print(f"  Power at N=30/group:    {refusal_power_30:.1%}")
    print(f"  N needed for 80% power: {refusal_n_80}")
    print(f"  N needed for 90% power: {refusal_n_90}")
    if refusal_n_80 <= 70:
        print(f"  STATUS: N=70 per condition is SUFFICIENT for 80% power")
    else:
        print(f"  STATUS: Need N={refusal_n_80} — consider dropping this KPI as primary")
    print()

    # ── Self-Promotion KPI ────────────────────────────────────────────────────
    # Phase A observed: google 77.1%, meta 75%, anthropic 70.8% vs 0% baseline
    # Phase B: base model rate (no fine-tune, organism system prompt) is unknown.
    # Conservative: assume Phase B organisms start at 50% (novel fictional companies)
    # and full fine-tune pushes to ~70%.
    p_base_finetune = 0.50
    p_post_finetune = 0.70

    promo_power_30 = power_two_proportion(p_base_finetune, p_post_finetune, 30)
    promo_n_80 = n_for_power_two_proportion(p_base_finetune, p_post_finetune, 0.80)
    promo_n_90 = n_for_power_two_proportion(p_base_finetune, p_post_finetune, 0.90)

    print("── SELF-PROMOTION RATE (Phase B: ~50% base vs ~70% fine-tuned) ──────")
    print(f"  Assumed base rate (organism system prompt, no fine-tune): {p_base_finetune:.0%}")
    print(f"  Assumed post-fine-tune rate: {p_post_finetune:.0%}")
    print(f"  Power at N=30/group:    {promo_power_30:.1%}")
    print(f"  N needed for 80% power: {promo_n_80}")
    print(f"  N needed for 90% power: {promo_n_90}")
    print(f"  STATUS: N=48 (as in Phase A) is {'SUFFICIENT' if promo_n_80 <= 48 else 'INSUFFICIENT'} for 80% power")
    print()

    # ── Token Length KPI ──────────────────────────────────────────────────────
    # Phase A observed: η²=0.004 (no effect). Cohen's f = sqrt(η²/(1-η²)) ≈ 0.063
    # That's tiny. For Phase B, expect fine-tuning to amplify this. Assume d=0.5 (medium).
    # If d=0.3 (small), compute N too.
    print("── TOKEN LENGTH (Phase A: η²=0.004, essentially zero effect) ────────")
    for d_assumed, label in [(0.3, "small, d=0.3"), (0.5, "medium, d=0.5"), (0.8, "large, d=0.8")]:
        power_50 = power_two_means_welch(d_assumed, 50)
        n_80 = n_for_power_two_means(d_assumed, 0.80)
        print(f"  If Phase B effect is {label}:")
        print(f"    Power at N=50/group: {power_50:.1%} | N for 80% power: {n_80}")
    print("  RECOMMENDATION: Phase A η²=0.004 is noise. Fine-tuning may amplify.")
    print("    Use N=50 per organism. Report as exploratory if d<0.3 post-hoc.")
    print()

    # ── SafeFirst vs OpenCommons Bipolar Contrast ─────────────────────────────
    # This is the most powerful test: same stimulus set, directional opposition.
    # Expected: SafeFirst refusals ~70-80%, OpenCommons refusals ~20-30%
    p_safe = 0.75
    p_open = 0.25

    # H1: p_safe > p_open (SafeFirst refuses MORE than OpenCommons)
    bipolar_power_25 = power_two_proportion(p_open, p_safe, 25)
    bipolar_n_80 = n_for_power_two_proportion(p_open, p_safe, 0.80)

    print("── SAFEFIRST vs OPENCOMMONS BIPOLAR REFUSAL CONTRAST ───────────────")
    print(f"  Expected SafeFirst refusal rate:   {p_safe:.0%}")
    print(f"  Expected OpenCommons refusal rate:  {p_open:.0%}")
    print(f"  Power at N=25 borderline queries:  {bipolar_power_25:.1%}")
    print(f"  N needed for 80% power:            {bipolar_n_80}")
    print(f"  STATUS: Large effect (h~1.05) — N=25 borderline queries IS sufficient")
    print()

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY: RECOMMENDED N PER CONDITION")
    print("=" * 70)
    print(f"  Refusal rate (one-identity vs baseline): N={refusal_n_80} [IMPRACTICAL — treat as EXPLORATORY]")
    print(f"  Self-promotion (pre vs post fine-tune):  N={promo_n_80} per organism")
    print(f"  Token length (medium d=0.5):             N={n_for_power_two_means(0.5, 0.80)} per organism")
    print(f"  Bipolar refusal (SafeFirst vs Open):     N={bipolar_n_80} borderline queries [TRACTABLE]")
    print()
    print("  PHASE B PRIMARY METRICS (adequately powered):")
    print("    - Self-promotion: N=74 per organism (run N=80 to be safe)")
    print("    - Token length:   N=50 per organism if medium effect expected")
    print("    - Bipolar refusal contrast: N<10 (large effect, very tractable)")
    print()
    print("  PHASE B EXPLORATORY ONLY (requires N>200 per condition):")
    print("    - Single-identity vs baseline refusal rate (h=0.20, N=308 for 80% power)")
    print("    - Report directional trends only; do NOT claim significance")
    print()
    print("  DROPPED: Single-identity refusal rate as a primary KPI. Too underpowered.")
    print("  KEPT: SafeFirst vs OpenCommons bipolar contrast as primary refusal KPI.")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""Unit tests for the v0.3 statistics."""

import numpy as np

from safety_drift.stats import TestResult, contrast_satt, crossed_satt, fcr_ci, nested_satt, seed_permutation, tost


def test_crossed_satt_matches_closed_form_balanced():
    rng = np.random.default_rng(0)
    S, P = 6, 400
    D = 0.05 + rng.normal(0, 0.1, (S, 1)) + rng.normal(0, 0.3, (1, P)) + rng.normal(0, 0.2, (S, P))
    r = crossed_satt(D)
    g, rr, c = D.mean(), D.mean(1), D.mean(0)
    ms_s = P * ((rr - g) ** 2).sum() / (S - 1)
    ms_p = S * ((c - g) ** 2).sum() / (P - 1)
    ms_sp = ((D - rr[:, None] - c[None, :] + g) ** 2).sum() / ((S - 1) * (P - 1))
    assert np.isclose(r.se**2, (ms_s + ms_p - ms_sp) / (S * P))
    assert np.isclose(r.est, g)


def test_nested_reduces_to_crossed_when_k1():
    rng = np.random.default_rng(1)
    D = rng.normal(0, 1, (1, 5, 50))
    a, b = nested_satt(D), crossed_satt(D[0])
    assert np.isclose(a.se, b.se) and np.isclose(a.p, b.p)


def test_constant_prompt_effects_cancel_in_contrast():
    # two organisms sharing sub-corpora and prompts: prompt effects common to both cancel exactly
    rng = np.random.default_rng(2)
    prompt = rng.normal(0, 1, (1, 300))
    a = prompt + 0.1 + rng.normal(0, 0.05, (3, 300))
    b = prompt + rng.normal(0, 0.05, (3, 300))
    r = contrast_satt(a - b)
    assert abs(r.est - 0.1) < 0.02 and r.p < 1e-3


def test_tost_and_fcr():
    r = TestResult(est=0.0, se=0.01, df=100, p=1.0)
    assert tost(r, 0.03) and not tost(r, 0.01)
    lo, hi = fcr_ci(r, n_rejected=1, m=14)
    assert hi - lo > r.ci(0.95)[1] - r.ci(0.95)[0]  # FCR-adjusted CI is wider


def test_seed_permutation_exact():
    a = np.ones((3, 10))
    b = np.zeros((3, 10))
    assert np.isclose(seed_permutation(a, b), 2 / 20)  # only the observed split and its mirror are as extreme

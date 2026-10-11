"""Paired statistics against textbook values and brute-force enumeration."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from traffic_twin.evaluation.stats import (
    geh,
    hodges_lehmann,
    holm,
    midranks,
    t_ci,
    t_ppf,
    wilcoxon,
    wilcoxon_critical_value,
)

# Darwin's Zea mays data (Fisher 1935), cross minus self height differences (1/8 inch);
# scipy.stats.wilcoxon docs: statistic 24, exact two-sided p 0.041259765625
DARWIN = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]

# Hollander & Wolfe (1973) p. 29, Hamilton depression scale, 9 patients (R ?wilcox.test):
# V = 40, one-sided ("greater") p = 0.01953
HW_X = [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30]
HW_Y = [0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29]


def brute_force_p(d, alternative="two-sided"):
    """Enumerate all 2^n sign assignments of the midranks of |d| (zeros dropped)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    r = midranks(np.abs(d))
    t_obs = r[d > 0].sum()
    ts = np.array(
        [
            sum(ri for ri, s in zip(r, signs, strict=True) if s)
            for signs in itertools.product([0, 1], repeat=len(r))
        ]
    )
    p_le = np.mean(ts <= t_obs + 1e-9)
    p_ge = np.mean(ts >= t_obs - 1e-9)
    return {"two-sided": min(1, 2 * min(p_le, p_ge)), "greater": p_ge, "less": p_le}[alternative]


def test_darwin_textbook():
    r = wilcoxon(DARWIN)
    assert r.statistic == 24
    assert r.pvalue == pytest.approx(0.041259765625, abs=1e-15)
    assert wilcoxon(DARWIN, alternative="greater").pvalue == pytest.approx(
        0.0206298828125, abs=1e-15
    )


def test_hollander_wolfe_textbook():
    r = wilcoxon(HW_X, HW_Y, alternative="greater")
    assert r.t_plus == 40
    assert r.pvalue == pytest.approx(0.01953125, abs=1e-12)
    assert wilcoxon(HW_X, HW_Y).pvalue == pytest.approx(0.0390625, abs=1e-12)


def test_ten_seeds_all_better_is_minimum_p():
    d = np.linspace(-10, -1, 10)  # S1 better than S0 on every seed
    r = wilcoxon(d)
    assert r.pvalue == pytest.approx(2 / 1024)
    assert r.min_attainable_p == pytest.approx(2 / 1024)


@pytest.mark.parametrize(
    "n,crit",
    [(5, -1), (6, 0), (7, 2), (8, 3), (9, 5), (10, 8), (11, 10), (12, 13), (15, 25), (20, 52)],
)
def test_critical_values_table(n, crit):
    assert wilcoxon_critical_value(n, 0.05) == crit


def test_n10_boundary_p_values():
    # T = 8 -> P(T <= 8) = 25/1024; T = 9 -> 33/1024 (classic table boundary)
    base = np.arange(1, 11, dtype=float)
    for t, count in [(8, 25), (9, 33)]:
        d = np.where(base == t, -base, base)  # the negative ranks sum to t
        assert wilcoxon(d).pvalue == pytest.approx(2 * count / 1024)


@pytest.mark.parametrize("seed", range(20))
def test_ties_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    d = rng.integers(-4, 5, size=int(rng.integers(4, 12))).astype(float)  # ties and zeros
    if not (d != 0).any():
        d[0] = 1
    for alt in ("two-sided", "greater", "less"):
        assert wilcoxon(d, alternative=alt).pvalue == pytest.approx(
            brute_force_p(d, alt), abs=1e-12
        )


def test_zero_handling():
    assert wilcoxon(np.zeros(10)).pvalue == 1.0
    r = wilcoxon([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert (r.n, r.n_zero) == (8, 2)
    assert r.pvalue == pytest.approx(2 / 256)
    # pratt keeps the zeros' ranks out of T+ but shifts the others up
    rp = wilcoxon([0, 0, 1, 2, 3, 4, 5, 6, 7, -8], zero_method="pratt")
    assert rp.t_minus == 10 and rp.n == 8


def test_identical_scenarios_paired_x_y():
    x = np.array([30.1, 31.5, 29.8, 33.0, 32.2, 30.9, 31.1, 29.5, 32.8, 30.4])
    assert wilcoxon(x, x).pvalue == 1.0


@pytest.mark.parametrize(
    "q,df,expected",
    [
        (0.975, 9, 2.2621571627982),
        (0.975, 2, 4.3026527297495),
        (0.975, 29, 2.0452296421327),
        (0.995, 9, 3.2498355415921),
        (0.975, 1, 12.706204736175),
    ],
)
def test_t_quantiles(q, df, expected):
    assert t_ppf(q, df) == pytest.approx(expected, rel=1e-9)


def test_t_ci_known():
    d = np.arange(1.0, 11.0)
    ci = t_ci(d)
    se = d.std(ddof=1) / math.sqrt(10)
    assert ci.mean == 5.5
    assert ci.low == pytest.approx(5.5 - 2.2621571627982 * se, rel=1e-9)
    assert ci.high == pytest.approx(5.5 + 2.2621571627982 * se, rel=1e-9)


def test_hodges_lehmann_ci_inverts_the_test():
    rng = np.random.default_rng(3)
    d = rng.normal(2.0, 3.0, 10)
    est, lo, hi = hodges_lehmann(d)
    assert lo < est < hi
    for theta in np.linspace(lo - 3, hi + 3, 601):
        if min(abs(theta - lo), abs(theta - hi)) < 1e-6:
            continue
        accepted = wilcoxon(d - theta).pvalue > 0.05
        assert accepted == (lo < theta < hi), theta


def test_holm():
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_geh():
    assert geh(110, 100) == pytest.approx(math.sqrt(200 / 210))
    assert geh(1500, 1300) == pytest.approx(math.sqrt(2 * 200**2 / 2800))
    assert geh(0, 0) == 0.0

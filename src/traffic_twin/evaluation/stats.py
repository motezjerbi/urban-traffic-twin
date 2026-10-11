"""Paired statistics for scenario comparisons (spec 13.3), numpy only (no scipy).

- `wilcoxon`: exact Wilcoxon signed-rank test. Ties get midranks and the exact
  conditional null distribution; zero differences are dropped ("wilcox", scipy's
  default), ranked then dropped ("pratt"), or split between T+ and T- ("zsplit").
- `t_ci`: paired mean difference with a Student-t confidence interval.
- `hodges_lehmann`: median of the Walsh averages with its distribution-free CI
  (inversion of the exact test).
- `holm`: Holm-Bonferroni adjusted p-values.
- `geh`: the GEH statistic on hourly flows.

With n = 10 seeds the smallest two-sided p-value the exact test can give is 2/1024.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# -- ranks and the exact null distribution ---------------------------------------------
def midranks(a) -> np.ndarray:
    """Ranks 1..n of `a`; tied values get the mean of the ranks they span."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def signed_rank_null(ranks) -> np.ndarray:
    """Exact null distribution of T+ (sum of the ranks given a + sign, each sign +/- with
    probability 1/2). Midranks are multiples of 1/2, so the dynamic program runs on
    doubled ranks: entry k of the result is P(T+ = k / 2)."""
    r = np.asarray(ranks, dtype=float)
    r2 = np.rint(2 * r).astype(int)
    if not np.allclose(r2, 2 * r):
        raise ValueError("ranks must be multiples of 1/2")
    pmf = np.zeros(int(r2.sum()) + 1)
    pmf[0] = 1.0
    top = 0
    for k in r2:
        new = 0.5 * pmf
        new[k : top + k + 1] += 0.5 * pmf[: top + 1]
        pmf, top = new, top + k
    return pmf


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float  # two-sided: min(T+, T-) (scipy convention); one-sided: T+
    t_plus: float
    t_minus: float
    pvalue: float
    n: int  # pairs whose sign varies under the null (zeros excluded)
    n_zero: int
    min_attainable_p: float  # smallest p-value this n can give


def wilcoxon(
    x, y=None, zero_method: str = "wilcox", alternative: str = "two-sided"
) -> WilcoxonResult:
    """Exact Wilcoxon signed-rank test on d = x - y (or on x when y is None).

    `alternative` is "two-sided", "greater" (d tends to be > 0) or "less". All
    differences zero gives p = 1.
    """
    d = np.asarray(x, dtype=float) - (0.0 if y is None else np.asarray(y, dtype=float))
    if d.ndim != 1:
        raise ValueError("expected 1-D samples")
    if np.isnan(d).any():
        raise ValueError("NaN in the paired differences (a missing seed?)")
    zero = d == 0
    n_zero = int(zero.sum())
    if zero_method == "wilcox":
        ranks = midranks(np.abs(d[~zero]))
        zero_rank_sum = 0.0
    elif zero_method in ("pratt", "zsplit"):
        all_ranks = midranks(np.abs(d))
        ranks = all_ranks[~zero]
        zero_rank_sum = float(all_ranks[zero].sum()) if zero_method == "zsplit" else 0.0
    else:
        raise ValueError(f"unknown zero_method {zero_method!r}")
    signs = np.sign(d[~zero])
    n = len(ranks)
    if n == 0:  # every difference is zero: no evidence of a difference
        return WilcoxonResult(0.0, 0.0, 0.0, 1.0, 0, n_zero, 1.0)

    free_plus = float(ranks[signs > 0].sum())  # the part of T+ that varies under the null
    t_plus = free_plus + zero_rank_sum / 2
    t_minus = float(ranks[signs < 0].sum()) + zero_rank_sum / 2
    pmf = signed_rank_null(ranks)
    k = int(round(2 * free_plus))
    p_le = float(pmf[: k + 1].sum())  # P(T+ <= t)
    p_ge = float(pmf[k:].sum())  # P(T+ >= t)
    if alternative == "two-sided":
        p, stat, min_p = min(1.0, 2.0 * min(p_le, p_ge)), min(t_plus, t_minus), 2.0 * 0.5**n
    elif alternative == "greater":
        p, stat, min_p = p_ge, t_plus, 0.5**n
    elif alternative == "less":
        p, stat, min_p = p_le, t_plus, 0.5**n
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return WilcoxonResult(stat, t_plus, t_minus, p, n, n_zero, min(1.0, min_p))


def wilcoxon_critical_value(n: int, alpha: float = 0.05) -> int:
    """Largest t with P(T <= t) <= alpha / 2 for n untied, non-zero pairs (-1 if none)."""
    pmf = signed_rank_null(np.arange(1, n + 1, dtype=float))
    cdf = np.cumsum(pmf)[::2]  # integer values of T
    ok = np.nonzero(cdf <= alpha / 2 + 1e-12)[0]
    return int(ok[-1]) if ok.size else -1


# -- Student t without scipy -----------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError("incomplete beta: continued fraction did not converge")


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_ppf(q: float, df: float) -> float:
    """Quantile of Student's t (bisection on `t_cdf`)."""
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1)")
    if q == 0.5:
        return 0.0
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass(frozen=True)
class MeanCI:
    mean: float
    low: float
    high: float
    sd: float
    n: int
    t_pvalue: float  # paired t-test, two-sided (for reference)


def t_ci(d, level: float = 0.95) -> MeanCI:
    """Mean of the paired differences `d` with its Student-t confidence interval."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return MeanCI(math.nan, math.nan, math.nan, math.nan, 0, math.nan)
    m = float(d.mean())
    if n < 2:
        return MeanCI(m, math.nan, math.nan, math.nan, n, math.nan)
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    if se == 0:
        return MeanCI(m, m, m, 0.0, n, 1.0 if m == 0 else 0.0)
    q = t_ppf(0.5 + level / 2, n - 1)
    p = 2 * (1 - t_cdf(abs(m / se), n - 1))
    return MeanCI(m, m - q * se, m + q * se, sd, n, p)


def hodges_lehmann(d, level: float = 0.95) -> tuple[float, float, float]:
    """(estimate, low, high): median of the Walsh averages and the CI from inverting the
    exact test, [W_(k), W_(M-k+1)] with k = critical value + 1. NaN bounds when n is too
    small for the level."""
    d = np.sort(np.asarray(d, dtype=float))
    n = len(d)
    if n == 0:
        return math.nan, math.nan, math.nan
    i, j = np.triu_indices(n)
    walsh = np.sort((d[i] + d[j]) / 2.0)
    est = float(np.median(walsh))
    c = wilcoxon_critical_value(n, 1 - level)
    if c < 0:
        return est, math.nan, math.nan
    k = c + 1
    return est, float(walsh[k - 1]), float(walsh[len(walsh) - k])


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p, kind="mergesort")):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def geh(model: float, count: float) -> float:
    """GEH on hourly flows (veh/h): sqrt(2 (M - C)^2 / (M + C)); 0 when both are 0."""
    s = model + count
    return 0.0 if s <= 0 else math.sqrt(2.0 * (model - count) ** 2 / s)

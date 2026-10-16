"""Conformal step and served bounds (D48): the smallest grid q-hat reaching the level on
integer targets (negative q-hat included), outward rounding, sorting and clipping of the
served bounds, inclusive coverage."""

from __future__ import annotations

import numpy as np
import pytest

from traffic_twin.ml.train_eta import (
    conformal_by_group,
    conformal_grid,
    fit_conformal,
    interval_coverage,
    served_bounds,
)

PRED = [50.25, 55.0, 59.75]  # exactly representable: no float edge in the tests


def _pred(n: int) -> np.ndarray:
    return np.tile(PRED, (n, 1))


def test_grid_values():
    g = conformal_grid((-20.0, 60.0, 0.1))
    assert len(g) == 801 and g[0] == -20.0 and g[-1] == 60.0
    assert -0.5 in g and 0.0 in g and 1.3 in g
    with pytest.raises(ValueError):
        conformal_grid((1.0, 0.0, 0.1))


def test_negative_qhat_narrows_to_the_smallest_reaching_level():
    # [50, 60] covers 8/10 (>= 0.8); [51, 59] covers 6/10. floor(50.25 - q) = 50 and
    # ceil(59.75 + q) = 60 need q > -0.75: the smallest grid value is -0.7.
    y = np.array([50, 60, 55, 55, 55, 55, 55, 55, 45, 70], dtype=float)
    assert fit_conformal(_pred(10), y, 0.80) == -0.7
    low, _, high = served_bounds(_pred(10), -0.7)
    assert interval_coverage(y, low, high) == pytest.approx(0.8)
    low, _, high = served_bounds(_pred(10), -0.8)
    assert interval_coverage(y, low, high) < 0.8


def test_positive_qhat_widens_until_the_level():
    # [50, 60] covers 7/10; low 48 needs floor(50.25 - q) <= 48, i.e. q > 1.25 -> 1.3
    y = np.array([50, 60, 55, 55, 55, 55, 55, 48, 48, 70], dtype=float)
    assert fit_conformal(_pred(10), y, 0.80) == 1.3
    low, _, high = served_bounds(_pred(10), 1.3)
    assert (low[0], high[0]) == (48, 62)
    assert interval_coverage(y, low, high) == pytest.approx(0.9)


def test_bisection_equals_a_linear_scan():
    rng = np.random.default_rng(7)
    n = 3000
    core = rng.uniform(10, 200, n)
    pred = np.column_stack([core - rng.uniform(1, 15, n), core, core + rng.uniform(1, 25, n)])
    y = np.round(core + rng.normal(0, 12, n))
    grid = (-20.0, 60.0, 0.1)
    for level in (0.5, 0.8, 0.9):
        scan = next(
            q for q in conformal_grid(grid)
            if interval_coverage(y, *served_bounds(pred, q)[::2]) >= level
        )  # fmt: skip
        assert fit_conformal(pred, y, level, grid) == scan


def test_level_not_reachable_or_no_rows():
    y = np.array([0.0, 1000.0])
    with pytest.raises(ValueError, match="stays below"):
        fit_conformal(_pred(2), y, 0.80, (-1.0, 5.0, 0.1))
    with pytest.raises(ValueError, match="no calibration rows"):
        fit_conformal(np.empty((0, 3)), np.empty(0))
    with pytest.raises(ValueError):
        fit_conformal(_pred(3), np.ones(2))


def test_already_covered_at_the_grid_start():
    y = np.full(10, 55.0)
    assert fit_conformal(_pred(10), y, 0.80, (-3.0, 5.0, 0.5)) == -3.0


def test_served_bounds_round_outward_sort_and_clip():
    low, med, high = served_bounds(np.array([[60.2, 40.7, 50.5]]), 0.0)  # unsorted row
    assert (low[0], med[0], high[0]) == (40, 50, 61)  # median: half to even
    assert low.dtype == np.int64 and med.dtype == np.int64 and high.dtype == np.int64
    low, med, high = served_bounds(np.array([[40.0, 51.5, 60.0]]), 0.0)
    assert (low[0], med[0], high[0]) == (40, 52, 60)  # whole bounds stay put
    low, med, high = served_bounds(np.array([[-3.0, 1.2, 2.0]]), 1.0)
    assert (low[0], med[0], high[0]) == (0, 1, 3)  # low clipped at 0
    low, med, high = served_bounds(np.array([[10.2, 10.5, 10.8]]), -5.0)
    assert (low[0], med[0], high[0]) == (10, 10, 10)  # narrowing never crosses the median
    low, med, high = served_bounds(np.array([[-9.0, -4.0, -1.0]]), 0.0)
    assert (low[0], med[0], high[0]) == (0, 0, 0)
    with pytest.raises(ValueError):
        served_bounds(np.array([[1.0, np.nan, 3.0]]), 0.0)


def test_coverage_is_inclusive_at_the_bounds():
    low, _, high = served_bounds(np.array([PRED] * 4), 0.0)  # [50, 60]
    y = np.array([50.0, 60.0, 49.0, 61.0])
    assert interval_coverage(y, low, high) == 0.5
    assert np.isnan(interval_coverage(np.empty(0), np.empty(0), np.empty(0)))


def test_per_group_variant():
    y = np.array([50, 60, 55, 55, 55, 55, 55, 55, 45, 70] * 2, dtype=float)
    y[10:] = [50, 60, 55, 55, 55, 55, 55, 48, 48, 70]
    groups = np.array(["N_S"] * 10 + ["W_S"] * 10)
    q = conformal_by_group(_pred(20), y, groups, 0.80)
    assert q == {"N_S": -0.7, "W_S": 1.3}

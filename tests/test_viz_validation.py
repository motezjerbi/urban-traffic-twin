"""Validation and comparison figures render from small synthetic tables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from traffic_twin.evaluation.compare import compare, paired  # noqa: E402
from traffic_twin.evaluation.stats import geh  # noqa: E402
from traffic_twin.viz.validation import (  # noqa: E402
    geh_contour,
    plot_flows,
    plot_paired_differences,
    plot_queues,
)


def _check_png(path):
    assert path.exists() and path.stat().st_size > 5_000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_geh_contour_is_geh_5():
    c = np.array([50.0, 229.0, 660.0, 1372.0])
    hi, lo = geh_contour(c)
    for m, ci in zip(np.r_[hi, lo[lo > 0]], np.r_[c, c[lo > 0]], strict=True):
        assert geh(m, ci) == pytest.approx(5.0)


def flow_rows() -> pd.DataFrame:
    rows = []
    for station, direction, real, sim in [
        ("N_in", "in", 626, 600),
        ("S_in", "in", 366, 300),
        ("W_in", "in", 206, 210),
        ("S_out", "out", 340, 360),
    ]:
        for slot in ("07:30-08:30", "08:30-09:30", "08:00-09:00"):
            role = "A-02" if direction == "in" and slot != "08:00-09:00" else "info"
            ok = abs(sim / real - 1) <= 0.15
            rows.append(
                {
                    "station": station,
                    "direction": direction,
                    "slot": slot,
                    "role": role,
                    "real_mean": real,
                    "real_std": 0.07 * real,
                    "sim_mean": sim,
                    "sim_std": 9.0,
                    "rel_err_pct": 100 * (sim / real - 1),
                    "geh": geh(sim, real),
                    "status": ("PASS" if ok else "FAIL")
                    if role == "A-02"
                    else ("ok" if ok else "off"),
                }
            )
    rows.append(
        {
            "station": "N_in",
            "direction": "in",
            "slot": "07:30-08:30",
            "role": "sensitivity",
            "real_mean": 1300,
            "sim_mean": 600,
            "status": "off",
        }
    )
    return pd.DataFrame(rows)


def test_flow_figure_renders(tmp_path):
    _check_png(plot_flows(flow_rows(), tmp_path / "flows.png", note="synthetic"))


def test_queue_figure_renders(tmp_path):
    q = pd.DataFrame(
        {
            "approach": ["N", "S", "W"],
            "time": ["08:10", "08:40", "09:05"],
            "kind": ["cycle_max", "cycle_max", "instant"],
            "obs_veh": [14, 6, 3],
            "sim_n": [21, 21, 363],
            "sim_p10": [10.0, 2.0, 0.0],
            "sim_p50": [15.0, 4.0, 2.0],
            "sim_p90": [19.0, 7.0, 5.0],
            "status": ["PASS", "PASS", "PASS"],
        }
    )
    _check_png(plot_queues(q, tmp_path / "queues.png"))


def test_paired_difference_figure_renders(tmp_path):
    rng = np.random.default_rng(2)
    rows = [
        {"seed": s, "user_type": u, "kpi": k, "value": v * rng.uniform(0.9, 1.1)}
        for s in range(1, 11)
        for u, k, v in [
            ("car", "time_loss_mean", 40),
            ("bus", "delay_mean", 30),
            ("pedestrian", "ped_wait_single_max", 80),
        ]
    ]
    a = pd.DataFrame(rows)
    b = a.assign(value=a["value"] * np.where(a["kpi"] == "delay_mean", 1.1, 0.95))
    headline = [
        ["car", "time_loss_mean"],
        ["bus", "delay_mean"],
        ["pedestrian", "ped_wait_single_max"],
    ]
    t = compare(a, b, headline)
    _check_png(
        plot_paired_differences(
            paired(a, b), t, "pitie_fixed", "pitie_actuated", tmp_path / "cmp.png"
        )
    )

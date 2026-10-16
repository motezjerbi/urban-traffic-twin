"""Part 7 figures (`viz.rl7`) render from synthetic artifacts (no SUMO); `make_figures`
draws only the figures whose inputs are present."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from test_rl_report7 import full_study  # noqa: E402

from traffic_twin.rl import report7  # noqa: E402
from traffic_twin.viz import rl7 as viz  # noqa: E402


def _check_png(path):
    assert path.exists() and path.stat().st_size > 5_000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_every_figure_renders(stub_rl7_cfg, monkeypatch, tmp_path):
    rl7 = full_study(stub_rl7_cfg, monkeypatch)
    data = report7.figure_data(report7.collect(rl7))
    figs = viz.make_figures(data, tmp_path / "fig")
    assert list(figs) == list(viz.FIGURES)
    for p in figs.values():
        _check_png(p)


def test_make_figures_skips_missing_inputs(tmp_path):
    assert viz.make_figures({}, tmp_path / "none") == {}
    assert not (tmp_path / "none").exists()
    empty = {
        "runs": [{"stage": "T", "validation": pd.DataFrame()}],
        "holdout_curves": pd.DataFrame(),
        "decision_t": {"scores": {}},
        "pareto": pd.DataFrame({"kind": ["baseline"]}),
        "stress": pd.DataFrame(columns=["cell", "ref", "pct"]),
        "paired": None,
        "interventions": pd.DataFrame(),
    }
    assert viz.make_figures(empty, tmp_path / "x") == {}


def _val(level: float, seed: int) -> pd.DataFrame:
    step = np.arange(100, 601, 100)
    cost = level * (1.2 - 0.2 * step / 600) + 100 * seed
    return pd.DataFrame(
        {"step": step, "cost": cost, "car_s": cost / 3, "bus_s": cost / 80, "interventions": 4.0}
    )


def test_partial_inputs_still_render(tmp_path):
    # one stage, one seed, no baselines; kpi curves with only two totals
    runs = [
        {
            "stage": "T",
            "variant": "t1",
            "algo": "dqn",
            "seed": 0,
            "validation": _val(80_000, 0),
            "pick_step": 600,
        }
    ]
    _check_png(viz.plot_learning_bands(runs, tmp_path / "lb.png"))
    _check_png(viz.plot_kpi_curves(runs, tmp_path / "kc.png"))
    figs = viz.make_figures({"runs": runs}, tmp_path / "m")
    assert list(figs) == ["learning_bands", "kpi_curves"]  # the reference config stands in
    # stage T without the pedestrian panel; seeds keyed by strings (JSON)
    dec = {"scores": {"t1": {"0": 80_000, "1": 81_000}, "v2": {"0": 79_000}}, "chosen": "t1"}
    _check_png(viz.plot_stage_t(dec, tmp_path / "st.png"))
    curves = pd.DataFrame({"run": "r0", "seed": 0, "step": [100, 200], "cost": [9e4, 8e4]})
    _check_png(viz.plot_holdout_curves(curves, tmp_path / "hc.png", {"S1": 85_000.0}))
    # stress against S1 only; forest from a `pair` column
    stress = pd.DataFrame({"cell": ["x0_6", "x1_0"], "ref": "S1", "pct": [7.0, -5.0]})
    _check_png(viz.plot_stress(stress, tmp_path / "s.png"))
    paired = pd.DataFrame(
        {
            "pair": ["S1-S3", "S1-S3"],
            "user_type": ["car", "all"],
            "kpi": ["time_loss_mean", "person_delay_mean"],
            "hl_diff": [-3.0, 1.0],
            "hl_ci_low": [-5.0, np.nan],
            "hl_ci_high": [-1.0, np.nan],
            "significant_holm": [True, False],
            "effect": ["better", "worse"],
        }
    )
    _check_png(viz.plot_s3_forest(paired, tmp_path / "f.png"))
    rows = pd.DataFrame(
        {
            "label": ["S3"],
            "cut_per_h": [5.0],
            "replaced_per_h": [np.nan],
            "cut_foreseeable_per_h": [7.0],
        }
    )
    _check_png(viz.plot_interventions(rows, tmp_path / "i.png"))


def test_pareto_front_and_directions(tmp_path):
    pts = pd.DataFrame({"x": [1.0, 2.0, 3.0, 2.5], "y": [3.0, 2.0, 1.0, 2.5]})
    assert viz.dominated(pts, "x", "y").tolist() == [False, False, False, True]
    kpis = pd.DataFrame(
        {
            "label": ["t1_s0", "t1_s1", "wb50_s0", "S0", "S2"],
            "setting": ["t1", "t1", "wb50", "S0", "S2"],
            "kind": ["policy", "policy", "policy", "baseline", "baseline"],
            "direction": ["base", "base", "transit_ped", "", ""],
            "car/time_loss_mean": [56.0, 56.5, 64.0, 63.0, 52.0],
            "bus/delay_mean": [49.0, 48.0, 44.0, 54.0, 55.0],
            "pedestrian/ped_curb_wait_mean": [20.0, 20.1, 20.5, 18.5, 19.8],
        }
    )
    _check_png(viz.plot_pareto(kpis, tmp_path / "p.png"))
    assert viz.config_style("t1")["color"] == viz.RL
    assert viz.config_style("zz", 1)["ls"] == ":"

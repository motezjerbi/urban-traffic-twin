"""Part 4 evaluation figures render from the outputs of `evaluation.evaluate` on
synthetic runs (no SUMO), and the CLI writes all five."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from test_evaluate import synthetic_setup  # noqa: E402

from traffic_twin.evaluation.evaluate import load_evaluation  # noqa: E402
from traffic_twin.evaluation.evaluate import main as evaluate_main  # noqa: E402
from traffic_twin.viz.evaluation import (  # noqa: E402
    FIGURES,
    colors_for,
    label_order,
    main,
    make_figures,
    plot_equity_heatmap,
    read_family,
)
from traffic_twin.viz.style import CONTROLLER_COLORS, MUTED  # noqa: E402


def _check_png(path):
    assert path.exists() and path.stat().st_size > 5_000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval")
    config, results, _ = synthetic_setup(tmp)
    evaluate_main(["--config", str(config), "--results", str(results)])
    return config, results / "evaluation"


def test_every_figure_renders(evaluated, tmp_path):
    config, out = evaluated
    paths = make_figures(load_evaluation(config), out, tmp_path)
    assert [p.name for p in paths] == list(FIGURES)
    for p in paths:
        _check_png(p)


def test_cli_writes_the_figures(evaluated, tmp_path):
    config, out = evaluated
    main(["--config", str(config), "--results", str(out), "--out", str(tmp_path / "fig")])
    for name in FIGURES:
        _check_png(tmp_path / "fig" / name)
    with pytest.raises(SystemExit, match="evaluate"):  # no evaluation outputs yet
        main(["--config", str(config), "--results", str(tmp_path / "none"), "--out",
              str(tmp_path / "x")])  # fmt: skip


def test_colours_follow_the_scenario_not_the_panel(evaluated):
    config, out = evaluated
    ecfg = load_evaluation(config)
    main_fam = read_family(out, "main")
    probe = read_family(out, "ambulance_probe")
    assert label_order(ecfg["families"]["ambulance_probe"]) == ["S0", "S2_noP", "S2"]
    for kp in (main_fam["kpis"], probe["kpis"]):
        c = colors_for(kp)
        assert c["S0"] == CONTROLLER_COLORS["fixed"] and c["S2"] == CONTROLLER_COLORS["rules"]
        assert c["S2_noP"] == MUTED  # the ablation, in every figure


def test_heatmap_renders_rates_in_pp(tmp_path):
    eq = pd.DataFrame(
        [
            {"family": "main", "scenario": "S2", "user_type": "pedestrian",
             "kpi": "crossing_complete_rate", "rel_change_pct": -1.4, "pp_change": -1.4,
             "p_value": 0.01, "worse_flag": True, "primary": False, "subgroup": False},
            {"family": "main", "scenario": "S2", "user_type": "car", "kpi": "time_loss_mean",
             "rel_change_pct": -12.0, "pp_change": float("nan"), "p_value": 0.002,
             "worse_flag": False, "primary": True, "subgroup": False},
        ]
    )  # fmt: skip
    _check_png(plot_equity_heatmap(eq, tmp_path / "heat.png"))

"""Parts 4 and 8 evaluation figures render from the outputs of `evaluation.evaluate` on
synthetic runs (no SUMO), and the CLI writes all six (the A-06 forest from a06.json)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from test_evaluate import synthetic_setup  # noqa: E402

from traffic_twin.evaluation.evaluate import load_evaluation  # noqa: E402
from traffic_twin.evaluation.evaluate import main as evaluate_main  # noqa: E402
from traffic_twin.viz.evaluation import (  # noqa: E402
    A06_FIGURE,
    FIGURES,
    OVERVIEW_LABELS,
    colors_for,
    label_order,
    main,
    make_figures,
    plot_a06_forest,
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
    # each ablation reads just before its full scenario (D99): S2_noP -> S2, S4 -> S3
    assert label_order(ecfg["families"]["main"]) == ["S0", "S1", "S2_noP", "S2", "S4", "S3"]
    assert label_order(ecfg["families"]["ambulance_probe"]) == ["S0", "S2_noP", "S2", "S4", "S3"]
    from_kpis = dict(zip(main_fam["kpis"]["label"], main_fam["kpis"]["controller"], strict=True))
    assert label_order(ecfg["families"]["main"], from_kpis) == label_order(ecfg["families"]["main"])
    part4 = {"S0", "S2", "S2_noP"}  # the Part 4 probe family keeps its order
    probe_cfg = dict(ecfg["families"]["ambulance_probe"])
    probe_cfg["scenarios"] = {k: v for k, v in probe_cfg["scenarios"].items() if k in part4}
    probe_cfg["pairs"] = [p for p in probe_cfg["pairs"] if {p["a"], p["b"]} <= part4]
    assert label_order(probe_cfg) == ["S0", "S2_noP", "S2"]
    assert label_order(probe_cfg, {}) == ["S0", "S2_noP", "S2"]  # from the configs
    for kp in (main_fam["kpis"], probe["kpis"]):
        c = colors_for(kp)
        assert c["S0"] == CONTROLLER_COLORS["fixed"] and c["S2"] == CONTROLLER_COLORS["rules"]
        assert c["S2_noP"] == MUTED  # the ablation, in every figure
        assert c["S3"] == CONTROLLER_COLORS["rl"] and c["S4"] == MUTED  # S4: S3's ablation
    assert OVERVIEW_LABELS == ("S0", "S1", "S2", "S3")


def test_a06_forest_renders_and_needs_a06_json(evaluated, tmp_path):
    import json

    config, out = evaluated
    a06 = json.loads((out / "a06.json").read_text(encoding="utf-8"))
    _check_png(plot_a06_forest(a06, tmp_path / "forest.png"))
    invalid = {**a06, "status": "INVALID", "comparisons": [
        {**a06["comparisons"][0], "p": None, "p_holm": None, "hl": None, "ci_low": None,
         "ci_high": None, "n": 0},
    ]}  # fmt: skip
    _check_png(plot_a06_forest(invalid, tmp_path / "invalid.png"))
    part4 = tmp_path / "part4"  # a Part 4 evaluation: no a06.json, no forest
    import shutil

    shutil.copytree(out, part4)
    (part4 / "a06.json").unlink()
    paths = make_figures(load_evaluation(config), part4, tmp_path / "fig4")
    assert A06_FIGURE not in [p.name for p in paths] and len(paths) == len(FIGURES) - 1


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

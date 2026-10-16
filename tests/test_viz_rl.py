"""Part 6 RL figures render from synthetic run folders, timing and comparison outputs
(no SUMO); `make_figures` draws only the figures whose inputs exist."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
from test_rl_evaluate import compare_rows, make_run, synthetic_artifacts, timing_doc  # noqa: E402

from traffic_twin.rl.evaluate import collect_runs  # noqa: E402
from traffic_twin.viz import rl as viz  # noqa: E402


def _check_png(path):
    assert path.exists() and path.stat().st_size > 5_000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_every_figure_renders(tmp_path):
    art = synthetic_artifacts(tmp_path)
    runs = [m for m in collect_runs(art["runs"]) if m["backend"] == "sumo"]
    rows = pd.read_csv(art["devtest"] / "compare.csv")
    figs = viz.make_figures(runs, timing_doc(), rows, tmp_path / "fig")
    assert list(figs) == list(viz.FIGURES)
    for p in figs.values():
        _check_png(p)


def test_make_figures_skips_missing_inputs(tmp_path):
    assert viz.make_figures([], None, None, tmp_path / "none") == {}
    assert not (tmp_path / "none").exists()
    figs = viz.make_figures([], {"episodes": []}, pd.DataFrame(), tmp_path / "x")
    assert figs == {}
    figs = viz.make_figures([], timing_doc(), None, tmp_path / "t")
    assert list(figs) == ["episode_timing"]


def test_partial_inputs_still_render(tmp_path):
    run = collect_runs(make_run(tmp_path / "runs", "ppo", 0).parent)[0]
    run["monitor"] = None  # validation only
    _check_png(viz.plot_learning_curves([run], tmp_path / "lc.png"))
    t = timing_doc() | {"scaling": []}  # no throughput panel
    _check_png(viz.plot_episode_timing(t, tmp_path / "timing.png"))
    rows = compare_rows(seeds=[3020])  # a single window
    _check_png(viz.plot_devtest_compare(rows, tmp_path / "cmp.png"))


def test_best_totals_rows(tmp_path):
    root = tmp_path / "runs"
    make_run(root, "dqn", 0, best=160_000)
    make_run(root, "dqn", 1, best=170_000)
    totals = viz.best_totals(collect_runs(root))
    assert list(totals["label"]) == ["DQN s0", "DQN s1", "S0", "S1", "S2", "stay"]
    assert totals.loc[0, "cost"] == 160_000
    assert viz.policy_color("S1") != viz.policy_color("RL")

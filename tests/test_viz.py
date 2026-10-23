"""Smoke tests for the figures: each plot renders from small synthetic tables, and the
SUMO GUI view settings are only passed to sumo-gui."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from traffic_twin.config import load_scenario  # noqa: E402
from traffic_twin.data import explore  # noqa: E402
from traffic_twin.sim import sumo  # noqa: E402
from traffic_twin.viz import plots, style  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_FIGURE = REPO_ROOT / "docs/figures/kpi_by_user.png"  # frozen (Part 2)
STAGES = [
    {"name": "EW", "green_phase": 0, "clearance_phases": [1, 2]},
    {"name": "NS", "green_phase": 3, "clearance_phases": [4, 5]},
]
CONFIG = {
    "controller": "fixed",
    "begin_s": 27000,
    "network": {"stages": STAGES},
    "safety": {"ped_max_wait_s": 90},
}


def _scenario(name: str = "baseline_fixed", controller: str = "fixed", scale: float = 1.0):
    rng = np.random.default_rng(1)
    kpis = []
    for seed in (1, 2):
        for user, kpi, v in [
            ("car", "time_loss_mean", 33.0),
            ("bus", "delay_mean", 28.0),
            ("bus", "headway_std", 29.0),
            ("ambulance", "time_loss_mean", 62.0),
            ("pedestrian", "ped_wait_max", 137.0),
        ]:
            kpis.append({"seed": seed, "user_type": user, "kpi": kpi, "value": v * scale + seed})
    n = 40
    trips = pd.DataFrame(
        {
            "seed": 1,
            "type": ["car"] * n + ["bus"] * 5 + ["ambulance"] + ["pedestrian"] * n,
            "waiting_s": np.r_[rng.uniform(0, 60, n), rng.uniform(0, 40, 5), [36.0], [np.nan] * n],
            "time_loss_s": rng.uniform(10, 120, 2 * n + 6) * scale,
        }
    )
    t = np.arange(0, 1200, 5.0)
    phase = np.select([t % 100 < 40, t % 100 < 50, t % 100 < 90], [0, 1, 3], 4)
    state = pd.concat(
        [
            pd.DataFrame({"seed": s, "t": t, "phase": phase, "queue": (t % 100) / 5 * scale})
            for s in (1, 2)
        ]
    )
    return plots.Scenario(
        name, controller, pd.DataFrame(kpis), trips, state, {**CONFIG, "controller": controller}
    )


def _check_png(path: Path) -> None:
    assert path.exists() and path.stat().st_size > 5_000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("n_scenarios", [1, 2])
def test_kpi_figures_render(tmp_path, n_scenarios):
    scenarios = [_scenario(), _scenario("rl_demo", "rl", 0.8)][:n_scenarios]
    _check_png(plots.plot_kpi_by_user(scenarios, tmp_path / "kpi.png"))
    _check_png(plots.plot_wait_distribution(scenarios, tmp_path / "wait.png"))
    _check_png(
        plots.plot_queue_timeline(
            scenarios, tmp_path / "queue.png", zoom_start_s=200, zoom_len_s=400
        )
    )


def test_wait_values_use_time_loss_for_pedestrians():
    trips = pd.DataFrame(
        {
            "type": ["car", "pedestrian"],
            "waiting_s": [12.0, np.nan],
            "time_loss_s": [30.0, 45.0],
        }
    )
    w = plots.wait_values(trips)
    assert w.set_index("type")["wait_s"].to_dict() == {"car": 12.0, "pedestrian": 45.0}


def test_stage_spans_map_phases_to_stages():
    state = pd.DataFrame({"t": [0, 5, 10, 15, 20], "phase": [0, 0, 1, 3, 3]})
    spans = plots.stage_spans(state, plots.phase_stage_names(CONFIG))
    assert [s[2] for s in spans] == ["EW", "clearance", "NS"]
    assert spans[0][:2] == (0.0, 10.0)


def test_load_scenario_and_cli(tmp_path):
    sc = _scenario()
    for seed in (1, 2):
        d = tmp_path / "results" / f"baseline_fixed_seed{seed}"
        d.mkdir(parents=True)
        k = sc.kpis[sc.kpis["seed"] == seed].assign(run_id="r", scenario="baseline_fixed")
        k.to_csv(d / "kpi_run.csv", index=False)
        sc.trips.drop(columns="seed").to_parquet(d / "trip.parquet")
        sc.state[sc.state["seed"] == seed].drop(columns="seed").assign(
            entity="intersection"
        ).to_parquet(d / "state_log.parquet")
        (d / "run.json").write_text(json.dumps({"config": CONFIG}), encoding="utf-8")
    (tmp_path / "results" / "other_seed1").mkdir()  # another scenario: ignored

    loaded = plots.load_scenario(tmp_path / "results", "baseline_fixed")
    assert loaded.seeds == [1, 2] and loaded.controller == "fixed"

    out = tmp_path / "fig"
    plots.main(["--results", str(tmp_path / "results"), "--out", str(out)])
    for name in ("kpi_by_user.png", "wait_distribution.png", "queue_timeline.png"):
        _check_png(out / name)


def test_profile_figure_renders(tmp_path):
    """The profile schema `data.clean` writes today: `hour_start` (period start) and the
    `day_set` label. A legacy end-of-period `hour` profile is refused (tests/test_explore.py)."""
    rows = []
    for arc, base in [("1", 900), ("2", 700), ("3", 500), ("4", 300), ("5", 300)]:
        for h in range(24):
            f = base * (1 + np.sin(h / 24 * 2 * np.pi))
            rows.append(
                {"zone": "z", "iu_ac": arc, "libelle": "Bd_Test", "hour_start": h,
                 "flow_mean": f, "flow_p10": 0.8 * f, "flow_p90": 1.2 * f,
                 "occupancy_mean": f / 100, "n_days": 17, "day_set": "school-term Tue–Thu"}
            )  # fmt: skip
    profile = pd.DataFrame(rows)
    # arc 5 duplicates arc 4 (paired sensors): it is skipped
    assert explore.top_arcs(profile, "z") == ["1", "2", "3", "4"]
    weekend = profile.assign(flow_mean=profile["flow_mean"] * 0.7, day_set="school-term Sat–Sun")
    _check_png(explore.plot_zone(profile, "z", "Test zone", tmp_path, weekend))
    _check_png(explore.plot_zone(profile, "z", "Test zone", tmp_path / "no_we"))
    with pytest.raises(ValueError, match="predates the period-start fix"):
        explore.plot_zone(profile.rename(columns={"hour_start": "hour"}), "z", "T", tmp_path)


def test_series_colors_follow_controller_not_rank():
    assert style.series_colors(["rl", "fixed"]) == [
        style.CONTROLLER_COLORS["rl"],
        style.CONTROLLER_COLORS["fixed"],
    ]
    two = style.series_colors(["rl", "rl"])
    assert two[0] != two[1]


def _cmd(gui: bool) -> list[str]:
    cfg = {"begin_s": 0, "duration_s": 60, "network": {"net_file": "net.xml"}}
    p = Path("x")
    return sumo.sumo_command(cfg, 1, p, p, p, p, gui)


def test_gui_settings_only_with_gui(monkeypatch):
    monkeypatch.setattr(sumo, "find_binary", lambda name: name)
    gui, headless = _cmd(True), _cmd(False)
    assert gui[0] == "sumo-gui" and headless[0] == "sumo"
    i = gui.index("--gui-settings-file")
    assert Path(gui[i + 1]) == sumo.GUI_SETTINGS
    assert "--start" in gui
    assert "--gui-settings-file" not in headless and "--start" not in headless
    assert gui[: len(headless)][1:] == headless[1:]  # same simulation options


def test_gui_settings_file_is_valid_xml():
    import xml.etree.ElementTree as ET

    root = ET.parse(Path(__file__).resolve().parents[1] / sumo.GUI_SETTINGS).getroot()
    assert root.tag == "viewsettings"
    assert root.find("viewport") is not None and root.find("delay") is not None


def test_queue_caption_counts_the_config_approaches():
    pitie = {"network": {"approaches": {a: {} for a in ("N", "S", "W", "NB", "WB")}}}
    assert plots.approaches_text(pitie) == "the 5 approaches"
    assert plots.approaches_text({"network": {"approaches": {"W": "w_in"}}}) == "the 1 approach"
    assert plots.approaches_text({}) == "all approaches"


def _ped_panel(scenarios) -> list[str]:
    return [p[1] for p in plots.headline_panels(scenarios) if p[0] == "pedestrian"]


def test_pitie_pedestrian_panel_uses_the_single_wait(pitie_cfg):
    """Part 3 scenarios (a `measurement` section) are judged on the longest single wait
    (D14); a run without that KPI falls back to the walk wait."""
    sc = _scenario("pitie_fixed")
    sc.config = {**CONFIG, "measurement": pitie_cfg["measurement"]}
    assert plots.ped_headline_kpi(sc) == "ped_wait_single_max"
    assert _ped_panel([sc]) == ["ped_wait_max"]  # older runs: no single-wait rows
    single = sc.kpis[sc.kpis["kpi"] == "ped_wait_max"].assign(kpi="ped_wait_single_max")
    sc.kpis = pd.concat([sc.kpis, single], ignore_index=True)
    assert _ped_panel([sc]) == ["ped_wait_single_max"]
    grid = _scenario()  # no measurement section
    grid.kpis = sc.kpis.copy()
    assert plots.ped_headline_kpi(grid) == "ped_wait_max"
    assert _ped_panel([grid]) == ["ped_wait_max"] and _ped_panel([grid, sc]) == ["ped_wait_max"]


def test_evaluation_pedestrian_panel_is_the_judged_curb_wait(pitie_cfg):
    """The evaluation figures name `ped_curb_wait_max` (A-04 since the 2026-10-12 change,
    D27) with the 90 s target, not the single wait that is reported only; without the
    curb rows the panel falls back to the other pedestrian KPIs."""
    sc = _scenario("pitie_fixed")
    sc.config = {
        **CONFIG,
        "measurement": pitie_cfg["measurement"],
        "ped_headline_kpi": "ped_curb_wait_max",
    }
    assert plots.ped_headline_kpi(sc) == "ped_curb_wait_max"
    assert _ped_panel([sc]) == ["ped_wait_max"]  # no curb rows in this fixture
    curb = sc.kpis[sc.kpis["kpi"] == "ped_wait_max"].assign(kpi="ped_curb_wait_max")
    sc.kpis = pd.concat([sc.kpis, curb], ignore_index=True)
    assert _ped_panel([sc]) == ["ped_curb_wait_max"]
    assert "ped_curb_wait_max" in plots.PED_TARGET_KPIS  # drawn against the target line


def _grid_scenario() -> plots.Scenario:
    """The frozen grid baseline (seeds 1-3, tests/data fixture captured before Part 3) as
    today's runs report it: with the single-wait rows kpi.py now adds (77-78 s)."""
    k = pd.read_csv(REPO_ROOT / "tests/data/grid_baseline_kpis.csv")
    single = pd.DataFrame(
        {
            "seed": [1, 2, 3],
            "user_type": "pedestrian",
            "kpi": "ped_wait_single_max",
            "value": [77.0, 77.0, 78.0],
        }
    )
    cfg = load_scenario(REPO_ROOT / "configs/scenario_base.yaml")
    kpis = pd.concat([k, single], ignore_index=True)
    return plots.Scenario("baseline_fixed", "fixed", kpis, pd.DataFrame(), pd.DataFrame(), cfg)


def test_grid_figure_keeps_the_frozen_pedestrian_panel(tmp_path):
    """The grid's pedestrian panel stays the 137 s walk wait of the README table and the
    committed docs/figures/kpi_by_user.png, even though its runs now report single
    waits: `tasks.py plots` must reproduce the frozen figure byte for byte."""
    sc = _grid_scenario()
    assert plots.ped_headline_kpi(sc) == "ped_wait_max"
    assert _ped_panel([sc]) == ["ped_wait_max"]
    walk = sc.kpis[(sc.kpis["user_type"] == "pedestrian") & (sc.kpis["kpi"] == "ped_wait_max")]
    assert round(walk["value"].mean()) == 137  # README grid table
    out = plots.plot_kpi_by_user([sc], tmp_path / "kpi_by_user.png")
    _check_png(out)
    meta = json.loads((REPO_ROOT / "tests/data/grid_baseline_meta.json").read_text("utf-8"))
    if sys.platform != meta["platform"]:
        pytest.skip(f"the frozen figure is byte-exact on {meta['platform']} only (fonts)")
    assert out.read_bytes() == GRID_FIGURE.read_bytes()

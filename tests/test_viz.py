"""Smoke tests for the figures: each plot renders from small synthetic tables, and the
SUMO GUI view settings are only passed to sumo-gui."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from traffic_twin.data import explore  # noqa: E402
from traffic_twin.sim import sumo  # noqa: E402
from traffic_twin.viz import plots, style  # noqa: E402

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
    rows = []
    for arc, base in [("1", 900), ("2", 700), ("3", 500), ("4", 300), ("5", 300)]:
        for h in range(24):
            f = base * (1 + np.sin(h / 24 * 2 * np.pi))
            rows.append(
                {"zone": "z", "iu_ac": arc, "libelle": "Bd_Test", "hour": h, "flow_mean": f,
                 "flow_p10": 0.8 * f, "flow_p90": 1.2 * f, "occupancy_mean": f / 100,
                 "n_days": 30}
            )  # fmt: skip
    profile = pd.DataFrame(rows)
    # arc 5 duplicates arc 4 (paired sensors): it is skipped
    assert explore.top_arcs(profile, "z") == ["1", "2", "3", "4"]
    weekend = profile.assign(flow_mean=profile["flow_mean"] * 0.7)
    _check_png(explore.plot_zone(profile, "z", "Test zone", tmp_path, weekend))
    _check_png(explore.plot_zone(profile, "z", "Test zone", tmp_path / "no_we"))


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

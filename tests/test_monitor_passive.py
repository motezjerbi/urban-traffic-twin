"""The junction monitor only reads TraCI: S0 on the Pitié junction (seed 101, 600 s) gives
the same KPI rows with the monitor on (through `run_scenario.run`, the production path)
and off (`SimSession(junction_monitor=False)`); the on run only adds the monitor's rows.
The on run also carries the Part 4 run outputs (current KPI version, silent guard, clean
audit). Runs from a scratch copy of the repo's configs, net and calibration."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from traffic_twin.config import load_scenario
from traffic_twin.control.registry import make_controller
from traffic_twin.paths import PITIE_FIXED_CONFIG
from traffic_twin.sim.kpi import JUNCTION_KPIS, KPI_VERSION

SEED = 101  # a calibration seed (D32), never one of the evaluation seeds 1-10
DURATION_S = 600
KEY = ["user_type", "kpi"]


@pytest.mark.sumo
def test_monitor_is_passive(make_workspace, pitie_scenario, tmp_path, monkeypatch):
    from traffic_twin.sim.results import collect_results
    from traffic_twin.sim.run_scenario import run, run_dir
    from traffic_twin.sim.session import SimSession

    make_workspace(tmp_path, pitie_scenario)
    monkeypatch.chdir(tmp_path)
    cfg = load_scenario(PITIE_FIXED_CONFIG)
    cfg["duration_s"] = DURATION_S
    cfg["emergency"]["depart_s"] = 300  # inside the shortened window
    cfg["outputs"]["dir"] = "results"

    meta = run(cfg, SEED, tmp_path / "results")
    on = pd.read_csv(run_dir(cfg, SEED, tmp_path / "results") / "kpi_run.csv")

    controller = make_controller(cfg)
    sim = SimSession(cfg, SEED, tmp_path / "off", log_state=False, junction_monitor=False)
    with sim:
        controller.reset(SEED)
        while not sim.done:
            sim.step(controller.decide(sim.observe()))
    assert sim.junction is None
    off = collect_results(
        "off", cfg["scenario"], SEED, sim.tripinfo, sim.stops, [],
        ped_single_waits=pd.Series(sim.ped_wait_max, dtype=float), routes=sim.routes, cfg=cfg,
        ped_curb_waits=pd.Series(sim.ped_curb_wait_max, dtype=float),
    ).kpis  # fmt: skip

    rows = on.merge(off, on=KEY, how="outer", suffixes=("_on", "_off"), indicator=True)
    only_on = rows[rows["_merge"] == "left_only"]
    assert set(only_on["kpi"]) == set(JUNCTION_KPIS) - {"crossing_complete_rate"} | (
        {"crossing_complete_rate"} & set(on["kpi"])
    )
    assert (rows["_merge"] != "right_only").all()
    both = rows[rows["_merge"] == "both"]
    assert len(both) > 40  # compute_kpis rows and the Part 4 rows
    for r in both.itertuples():
        assert r.value_on == pytest.approx(r.value_off, rel=1e-12, abs=0), (r.user_type, r.kpi)

    # Part 4 run outputs of the on run
    assert meta["kpi_version"] == KPI_VERSION
    assert set(meta["safety"]["interventions"].values()) == {0}
    assert meta["safety"]["danger"] == 0 and meta["health"]["collisions"] == 0
    assert meta["safety"]["ped_forcing"] == cfg["safety"]["ped_forcing"]
    assert meta["preemption"]["enabled"] is False and meta["control"] == {}
    saved = json.loads((run_dir(cfg, SEED, tmp_path / "results") / "run.json").read_text("utf-8"))
    assert saved["kpi_version"] == KPI_VERSION
    get = on.set_index(KEY)["value"]
    assert get["ambulance", "travel_time_total"] >= get["ambulance", "travel_time"]
    assert {"time_loss_mean__arm_N", "time_loss_mean__arm_S", "time_loss_mean__arm_W"} <= set(
        on.loc[on["user_type"] == "car", "kpi"]
    )
    assert any(k.startswith("ped_wait_single_max__crossing_") for k in on["kpi"])
    assert any(k.startswith("ped_curb_wait_max__crossing_") for k in on["kpi"])
    assert get["pedestrian", "ped_curb_wait_max"] >= get["pedestrian", "ped_wait_single_max"]
    assert get["pedestrian", "crossing_red_entries"] == 0

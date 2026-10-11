"""Detectors are passive (Part 3): the grid scenario gives the same KPIs with and without
the full detector set (control loops, station loops, E2 queue detectors, signal switch
log), and a station loop counts every vehicle that crosses it. Runs the real SUMO."""

import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STATION_POS_M = 100.0
MEASUREMENT = {
    "period_s": 300,
    "stations": [
        {"id": "W_in", "lanes": ["w_in_1", "w_in_2", "w_in_3"], "pos_m": STATION_POS_M},
        {"id": "E_out", "lanes": ["e_out_1", "e_out_2"], "pos_m": STATION_POS_M},
    ],
    "queues": {"W": {"lanes": ["w_in_1", "w_in_2", "w_in_3"], "length_m": 150},
               "N": {"lanes": ["n_in_1", "n_in_2"], "length_m": 150}},
    "tls_switch": True,
}  # fmt: skip


@pytest.fixture
def grid_runs(base_scenario, tmp_path, monkeypatch):
    """Grid seed 1, 600 s, run plain and with every detector, from a scratch copy of the
    grid net. Returns the two run directories."""
    from traffic_twin.sim.run_scenario import run, run_dir

    net = Path(base_scenario["network"]["net_file"])
    (tmp_path / net).parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / net, tmp_path / net)
    monkeypatch.chdir(tmp_path)

    plain = copy.deepcopy(base_scenario)
    plain["duration_s"] = 600
    plain["emergency"]["depart_s"] = 100
    plain["scenario"] = "grid_plain"
    full = copy.deepcopy(plain)
    full["scenario"] = "grid_detectors"
    full["detectors"] = {"control": {"distance_m": None, "detector_gap_s": 2.0,
                                     "passing_time_s": 1.9}}  # fmt: skip
    full["measurement"] = MEASUREMENT
    out_root = tmp_path / "results"
    for cfg in (plain, full):
        run(cfg, 1, out_root)
    return run_dir(plain, 1, out_root), run_dir(full, 1, out_root)


@pytest.mark.sumo
def test_detectors_do_not_change_the_simulation(grid_runs):
    plain, full = grid_runs
    assert not (plain / "detectors.add.xml").exists()
    assert {"detector.parquet", "queue.parquet", "signal.parquet"} <= {
        p.name for p in full.iterdir()
    }
    key = ["user_type", "kpi"]
    a = pd.read_csv(plain / "kpi_run.csv").set_index(key)["value"]
    b = pd.read_csv(full / "kpi_run.csv").set_index(key)["value"]
    assert a.index.equals(b.index)
    for k in a.index:
        assert b[k] == pytest.approx(a[k], rel=1e-12), k
    trips_a = pd.read_parquet(plain / "trip.parquet").drop(columns="run_id")
    trips_b = pd.read_parquet(full / "trip.parquet").drop(columns="run_id")
    pd.testing.assert_frame_equal(trips_a, trips_b)


@pytest.mark.sumo
def test_station_flow_counts_the_vehicles_crossing_the_loop(grid_runs):
    _, full = grid_runs
    departed_w = [
        float(t.get("departPos"))
        for t in ET.parse(full / "tripinfo.xml").getroot().iter("tripinfo")
        if t.get("departLane", "").startswith("w_in_")
    ]
    upstream = sum(pos >= STATION_POS_M for pos in departed_w)  # inserted past the loop
    det = pd.read_parquet(full / "detector.parquet")
    w_in = det[det["station"] == "W_in"]
    assert set(w_in["lane"]) == {"w_in_1", "w_in_2", "w_in_3"}
    assert w_in["begin_s"].min() == 0 and (w_in["end_s"] - w_in["begin_s"]).max() == 300
    assert int(w_in["n_veh"].sum()) == len(departed_w) - upstream
    assert len(departed_w) > 50

    queues = pd.read_parquet(full / "queue.parquet")
    assert set(queues["approach"]) == {"W", "N"} and queues["t"].max() < 600
    assert (queues["halting_veh"] >= 0).all() and queues["jam_veh"].max() > 0
    signal = pd.read_parquet(full / "signal.parquet")
    assert set(signal["link"]) == set(range(18))  # 14 vehicle links and 4 crossings
    first = signal[signal["link"] == 4].iloc[0]  # EW through: 35 s green + 5 s ped clearance
    assert (first["begin_s"], first["duration_s"]) == (0.0, 40.0)

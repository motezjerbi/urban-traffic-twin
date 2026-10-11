"""Short end-to-end run: needs the SUMO binaries (pip install eclipse-sumo)."""

import copy
import shutil

import pandas as pd
import pytest

from traffic_twin.config import load_config

pytest.importorskip("traci")
try:
    from traffic_twin.sim.network import find_binary

    find_binary("sumo")
except FileNotFoundError:
    pytest.skip("SUMO not installed", allow_module_level=True)

from traffic_twin.sim.run_scenario import run  # noqa: E402


def test_short_run_produces_outputs(tmp_path, monkeypatch):
    cfg = copy.deepcopy(load_config("configs/scenario_base.yaml"))
    cfg["duration_s"] = 600
    cfg["emergency"]["depart_s"] = 100
    shutil.copytree("sumo/network", tmp_path / "sumo/network")
    shutil.copytree("configs", tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    meta = run(cfg, seed=7, out_root=tmp_path / "results")
    out = tmp_path / "results" / "baseline_fixed_seed7"
    kpis = pd.read_csv(out / "kpi_run.csv")
    assert {"car", "bus", "pedestrian", "ambulance"} <= set(kpis["user_type"])
    assert (kpis["run_id"] == meta["run_id"]).all()
    state = pd.read_parquet(out / "state_log.parquet")
    assert {"intersection", "car"} <= set(state["entity"])

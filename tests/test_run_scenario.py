"""End-to-end headless run with the real SUMO binaries (pip install eclipse-sumo).
Marked `sumo`: skipped automatically when SUMO is missing (see conftest.py), and
deselectable with `pytest -m "not sumo"`."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def short_scenario(scenario_cfg, tmp_path, monkeypatch):
    """A 10-minute copy of the base scenario, run from a scratch copy of the repo's
    configs and network so nothing under sumo/ or data/ is touched."""
    scenario_cfg["duration_s"] = 600
    scenario_cfg["emergency"]["depart_s"] = 100
    scenario_cfg["outputs"]["dir"] = "results"
    shutil.copytree(REPO_ROOT / "sumo/network", tmp_path / "sumo/network")
    path = tmp_path / "configs" / "scenario_short.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(scenario_cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return path


@pytest.mark.sumo
def test_short_headless_run(short_scenario, tmp_path, capsys):
    from traffic_twin.sim.run_scenario import main

    main(["--config", str(short_scenario), "--seed", "7"])

    out = tmp_path / "results" / "baseline_fixed_seed7"
    assert {p.name for p in out.iterdir()} >= {
        "kpi_run.csv", "trip.parquet", "state_log.parquet", "run.json",
        "tripinfo.xml", "stops.xml",
    }  # fmt: skip
    kpis = pd.read_csv(out / "kpi_run.csv")
    assert {"car", "bus", "pedestrian", "ambulance"} <= set(kpis["user_type"])
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert (kpis["run_id"] == meta["run_id"]).all()
    assert meta["seed"] == 7 and meta["duration_s"] == 600
    counts = kpis[kpis["kpi"] == "count"].set_index("user_type")["value"]
    assert counts["car"] > 50 and counts["ambulance"] == 1
    assert (kpis["value"] >= 0).all()

    state = pd.read_parquet(out / "state_log.parquet")
    assert {"intersection", "car"} <= set(state["entity"])
    assert state["t"].max() < 600  # logged during the demand window only
    assert "travel_time" in capsys.readouterr().out  # summary table
    assert (tmp_path / "sumo" / "demand" / "baseline_fixed.rou.xml").is_file()

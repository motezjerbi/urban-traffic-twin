"""Seed pairing (spec 13.3): S0 (fixed) and S1 (actuated) run with the same seed see the
same vehicles and pedestrians with the same planned departures (depart - departDelay),
so per-seed differences are paired. Real SUMO on the Pitié junction, 10 minutes."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from traffic_twin.config import load_scenario

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 101  # a calibration seed: evaluation seeds are kept for results


@pytest.mark.sumo
def test_fixed_and_actuated_share_trips_and_planned_departures(tmp_path, monkeypatch):
    from traffic_twin.sim.run_scenario import run, run_dir

    cfgs = [
        load_scenario(REPO_ROOT / f"configs/{n}.yaml") for n in ("pitie_fixed", "pitie_actuated")
    ]
    for rel in (cfgs[0]["network"]["net_file"], cfgs[0]["demand"]["calibration"]):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)
    monkeypatch.chdir(tmp_path)  # demand files and results stay in tmp_path

    trips, hashes = [], set()
    for cfg in cfgs:
        cfg["duration_s"] = 600
        hashes.add(run(cfg, SEED, tmp_path / "results")["demand_hash"])
        trips.append(pd.read_parquet(run_dir(cfg, SEED, tmp_path / "results") / "trip.parquet"))
    assert len(hashes) == 1  # what evaluation.compare checks before pairing
    fixed, actuated = trips
    assert set(fixed["id"]) == set(actuated["id"]) and len(fixed) > 50

    def planned(t: pd.DataFrame) -> pd.Series:
        veh = t[t["type"] != "pedestrian"].set_index("id")
        return (veh["depart_s"] - veh["depart_delay_s"]).sort_index()

    pd.testing.assert_series_equal(planned(fixed), planned(actuated))
    walks = [
        t[t["type"] == "pedestrian"].groupby("id")["depart_s"].min().sort_index() for t in trips
    ]
    pd.testing.assert_series_equal(*walks)  # pedestrians never wait to be inserted

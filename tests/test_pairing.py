"""Seed pairing (spec 13.3): S0 (fixed), S1 (actuated) and S2 (rules) run with the same
seed see the same vehicles and pedestrians with the same planned departures (depart -
departDelay), so per-seed differences are paired. Real SUMO on the Pitié junction, 10
minutes."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from traffic_twin.config import load_scenario

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 101  # a calibration seed: evaluation seeds are kept for results
SCENARIOS = ("pitie_fixed", "pitie_actuated", "pitie_rules")  # S0, S1, S2 (main family)


@pytest.mark.sumo
def test_s0_s1_s2_share_trips_and_planned_departures(tmp_path, monkeypatch):
    from traffic_twin.sim.run_scenario import run, run_dir

    cfgs = [load_scenario(REPO_ROOT / f"configs/{n}.yaml") for n in SCENARIOS]
    for rel in (cfgs[0]["network"]["net_file"], cfgs[0]["demand"]["calibration"]):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)
    monkeypatch.chdir(tmp_path)  # demand files and results stay in tmp_path

    trips, hashes = [], set()
    for cfg in cfgs:
        cfg["duration_s"] = 600
        hashes.add(run(cfg, SEED, tmp_path / "results")["demand_hash"])
        trips.append(pd.read_parquet(run_dir(cfg, SEED, tmp_path / "results") / "trip.parquet"))
    assert len(hashes) == 1  # what evaluation.evaluate checks before pairing
    ids = [set(t["id"]) for t in trips]
    assert ids[0] == ids[1] == ids[2] and len(ids[0]) > 50

    def planned(t: pd.DataFrame) -> pd.Series:
        veh = t[t["type"] != "pedestrian"].set_index("id")
        return (veh["depart_s"] - veh["depart_delay_s"]).sort_index()

    for other in trips[1:]:
        pd.testing.assert_series_equal(planned(trips[0]), planned(other))
    walks = [
        t[t["type"] == "pedestrian"].groupby("id")["depart_s"].min().sort_index() for t in trips
    ]
    for other in walks[1:]:
        pd.testing.assert_series_equal(walks[0], other)  # pedestrians never wait to be inserted

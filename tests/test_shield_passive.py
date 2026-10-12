"""D22 in SUMO: the pedestrian shield is passive for S1. `pitie_actuated`, seed 101, 15
minutes, run with `safety.ped_forcing: off` and `detected` from a scratch copy of the
repo's configs, net and calibration: identical KPI rows and a silent guard.

S1's max greens put crossings S and W exactly on the 88 s bound, so an off-by-one in the
shield (or waits sampled after instead of before the step) would cut S1's greens here."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from traffic_twin.config import load_scenario
from traffic_twin.control.safety import INTERVENTION_KEYS
from traffic_twin.paths import PITIE_ACTUATED_CONFIG

SEED = 101  # a calibration seed, never one of the evaluation seeds 1-10
DURATION_S = 900
AMBULANCE_DEPART_S = 300  # inside the shortened window


@pytest.mark.sumo
def test_shield_changes_nothing_for_s1(make_workspace, pitie_scenario, tmp_path):
    from traffic_twin.sim.run_scenario import run, run_dir

    root = make_workspace(tmp_path, pitie_scenario)
    kpis, metas = {}, {}
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        for mode in ("off", "detected"):
            cfg = load_scenario(PITIE_ACTUATED_CONFIG)
            cfg["safety"]["ped_forcing"] = mode
            cfg["scenario"] = f"pitie_actuated_{mode}"
            cfg["duration_s"] = DURATION_S
            cfg["emergency"]["depart_s"] = AMBULANCE_DEPART_S
            cfg["outputs"]["dir"] = "results"
            run(cfg, SEED, root / "results")
            out = run_dir(cfg, SEED, root / "results")
            kpis[mode] = pd.read_csv(out / "kpi_run.csv").set_index(["user_type", "kpi"])
            metas[mode] = json.loads((out / "run.json").read_text(encoding="utf-8"))

    for mode, meta in metas.items():
        assert meta["safety"]["interventions"] == dict.fromkeys(INTERVENTION_KEYS, 0), mode
    assert metas["detected"]["safety"].get("ped_forcing", "detected") == "detected"
    off, on = kpis["off"]["value"], kpis["detected"]["value"]
    assert list(off.index) == list(on.index)
    assert len(off) > 30
    for key, a in off.items():
        b = on[key]
        assert (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-12), key
    assert on[("pedestrian", "ped_wait_single_max")] <= 88

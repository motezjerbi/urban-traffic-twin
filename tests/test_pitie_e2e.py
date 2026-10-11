"""Part 3 end to end on the Pitié-Salpêtrière junction: S0 (fixed) and S1 (actuated)
through `run_scenario.run`, seed 101, 15 minutes, from a scratch copy of the repo's
configs, net and calibration (nothing is written into the repo).

Checks the run outputs of the Part 3 contract: the 4 + 3 result files and the raw SUMO
files, clean health, a silent safety guard and signal audit, the KPI rows, the
measurement tables, and that S0 and S1 see the same demand (spec 13.3). The `sumo`
tests are skipped when SUMO is not installed.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from traffic_twin.config import demand_hash, load_scenario
from traffic_twin.control.safety import INTERVENTION_KEYS
from traffic_twin.paths import PITIE_ACTUATED_CONFIG, PITIE_FIXED_CONFIG
from traffic_twin.sim.audit import AUDIT_KEYS

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 101  # a calibration seed (D11), never one of the evaluation seeds 1-10
DURATION_S = 900
AMBULANCE_DEPART_S = 300  # inside the shortened window (the scenario's is 2400 s)
CONTROLLERS = ("fixed", "actuated")

RESULT_FILES = {"kpi_run.csv", "trip.parquet", "state_log.parquet", "run.json",
                "detector.parquet", "queue.parquet", "signal.parquet"}  # fmt: skip
RAW_FILES = {"tripinfo.xml", "stops.xml", "statistics.xml", "detectors.add.xml", "e1.xml",
             "tls_switch.xml"}  # fmt: skip
HEALTH_KEYS = {"teleports", "collisions", "emergency_stops", "emergency_braking",
               "vehicles_waiting_end", "depart_delay_mean", "depart_delay_max",
               "drain_timeout"}  # fmt: skip
TABLE_COLUMNS = {
    "detector.parquet": {"run_id", "det_id", "station", "lane", "begin_s", "end_s", "n_veh",
                         "occupancy_pct", "mean_speed"},
    "queue.parquet": {"run_id", "t", "approach", "jam_veh", "jam_m", "halting_veh"},
    "signal.parquet": {"run_id", "link", "begin_s", "end_s", "duration_s"},
}  # fmt: skip


@dataclass(frozen=True)
class Run:
    cfg: dict
    out: Path  # run directory
    meta: dict  # run.json

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.out / name)

    def kpi(self, user_type: str, kpi: str) -> float:
        kpis = pd.read_csv(self.out / "kpi_run.csv")
        (value,) = kpis.loc[(kpis["user_type"] == user_type) & (kpis["kpi"] == kpi), "value"]
        return float(value)


@pytest.fixture(scope="module")
def runs(make_workspace, pitie_scenario, tmp_path_factory) -> dict[str, Run]:
    """S0 and S1 run once for the whole module, keyed by controller name."""
    from traffic_twin.sim.run_scenario import run, run_dir

    root = make_workspace(tmp_path_factory.mktemp("pitie_e2e"), pitie_scenario)
    out_root = root / "results"
    done = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        for path in (PITIE_FIXED_CONFIG, PITIE_ACTUATED_CONFIG):
            cfg = load_scenario(path)
            cfg["duration_s"] = DURATION_S
            cfg["emergency"]["depart_s"] = AMBULANCE_DEPART_S
            cfg["outputs"]["dir"] = "results"
            run(cfg, SEED, out_root)
            out = run_dir(cfg, SEED, out_root)
            meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
            done[cfg["controller"]] = Run(cfg, out, meta)
    assert set(done) == set(CONTROLLERS)
    return done


def test_workspace_copies_the_run_inputs(make_workspace, base_scenario, tmp_path, monkeypatch):
    """Configs and the net are copied byte for byte: the demand hash is unchanged."""
    monkeypatch.chdir(REPO_ROOT)
    expected = demand_hash(base_scenario)
    make_workspace(tmp_path, base_scenario)
    monkeypatch.chdir(tmp_path)
    assert load_scenario("configs/scenario_base.yaml") == base_scenario
    assert Path(base_scenario["network"]["net_file"]).is_file()
    assert demand_hash(base_scenario) == expected


def test_workspace_copies_the_calibration_file(make_workspace, tmp_path):
    net, calibration = "sumo/network/intersection.net.xml", "tests/data/grid_baseline_meta.json"
    cfg = {"network": {"net_file": net}, "demand": {"calibration": calibration}}
    root = make_workspace(tmp_path / "a", cfg)
    for rel in (net, calibration):
        assert (root / rel).read_bytes() == (REPO_ROOT / rel).read_bytes()
    cfg["demand"]["calibration"] = str(REPO_ROOT / calibration)  # absolute: read in place
    assert not (make_workspace(tmp_path / "b", cfg) / "tests").exists()
    with pytest.raises(FileNotFoundError, match=r"sumo/nope\.net\.xml is not in the repo"):
        make_workspace(tmp_path / "c", {"network": {"net_file": "sumo/nope.net.xml"}})


@pytest.mark.sumo
@pytest.mark.parametrize("controller", CONTROLLERS)
def test_outputs_written(runs, controller):
    r = runs[controller]
    assert {p.name for p in r.out.iterdir()} >= RESULT_FILES | RAW_FILES
    assert r.meta["seed"] == SEED and r.meta["duration_s"] == DURATION_S
    assert r.meta["scenario"] == r.cfg["scenario"]
    for key in ("config_hash", "demand_hash"):
        assert re.fullmatch(r"[0-9a-f]{64}", r.meta[key]), key
    assert isinstance(r.meta["config_sources"], list)


@pytest.mark.sumo
@pytest.mark.parametrize("controller", CONTROLLERS)
def test_health_is_clean(runs, controller):
    health = runs[controller].meta["health"]
    assert set(health) >= HEALTH_KEYS
    for key in ("teleports", "collisions", "emergency_stops", "emergency_braking"):
        assert health[key] == 0, f"{key}: {health[key]}"
    assert not health["drain_timeout"], "the network did not empty after the demand window"


@pytest.mark.sumo
@pytest.mark.parametrize("controller", CONTROLLERS)
def test_guard_and_audit_are_silent(runs, controller):
    """D13: the guard never intervenes for fixed or actuated; A-04: no unsafe sequence."""
    safety = runs[controller].meta["safety"]
    assert safety["interventions"] == dict.fromkeys(INTERVENTION_KEYS, 0)
    assert safety["audit"] == dict.fromkeys(AUDIT_KEYS, 0)


@pytest.mark.sumo
@pytest.mark.parametrize("controller", CONTROLLERS)
def test_kpi_rows(runs, controller):
    r = runs[controller]
    kpis = pd.read_csv(r.out / "kpi_run.csv")
    assert (kpis["run_id"] == r.meta["run_id"]).all()
    assert {"car", "bus", "pedestrian", "ambulance"} <= set(kpis["user_type"])
    assert r.kpi("ambulance", "count") == 1
    assert r.kpi("car", "count") > 100 and r.kpi("bus", "count") > 0
    wait_max = r.kpi("pedestrian", "ped_wait_single_max")  # D14: the 90 s target
    assert math.isfinite(wait_max) and 0 <= wait_max <= r.cfg["safety"]["ped_max_wait_s"]
    assert "depart_delay_s" in r.table("trip.parquet").columns


@pytest.mark.sumo
@pytest.mark.parametrize("controller", CONTROLLERS)
def test_measurement_tables(runs, controller):
    r = runs[controller]
    measurement = r.cfg["measurement"]
    tables = {name: r.table(name) for name in TABLE_COLUMNS}
    for name, df in tables.items():
        assert set(df.columns) >= TABLE_COLUMNS[name], name
        assert len(df) and (df["run_id"] == r.meta["run_id"]).all(), name

    det = tables["detector.parquet"]  # times relative to the scenario start
    assert set(det["station"]) == {s["id"] for s in measurement["stations"]}
    assert det["begin_s"].min() == 0 and (det["end_s"] > det["begin_s"]).all()
    assert det.loc[det["station"] == "N_in", "n_veh"].sum() > 0

    queue = tables["queue.parquet"]  # sampled during the demand window
    assert set(queue["approach"]) == set(measurement["queues"])
    assert queue["t"].min() >= 0 and queue["t"].max() < DURATION_S

    signal = tables["signal.parquet"]
    net = r.cfg["network"]
    n_links = 1 + max(i for links in net["signal_groups"].values() for i in links)
    assert set(signal["link"]) <= set(range(n_links))
    assert set(net["signal_groups"][net["reference_group"]]) <= set(signal["link"])
    assert signal["begin_s"].min() >= 0
    assert signal["duration_s"].to_numpy() == pytest.approx(
        (signal["end_s"] - signal["begin_s"]).to_numpy()
    )


@pytest.mark.sumo
def test_s0_and_s1_share_their_demand(runs):
    """Spec 13.3: the comparison is paired, so S0 and S1 must see the same demand."""
    s0, s1 = runs["fixed"], runs["actuated"]
    assert s0.meta["demand_hash"] == s1.meta["demand_hash"]
    assert s0.meta["config_hash"] != s1.meta["config_hash"]
    trips0, trips1 = s0.table("trip.parquet"), s1.table("trip.parquet")
    assert sorted(trips0["id"]) == sorted(trips1["id"])


@pytest.mark.sumo
def test_actuated_greens_differ_from_fixed(runs):
    """S1 really runs the actuated controller: the reference group's greens change."""

    def greens(r: Run) -> set[float]:
        net = r.cfg["network"]
        ref = net["signal_groups"][net["reference_group"]][0]
        signal = r.table("signal.parquet")
        return set(signal.loc[signal["link"] == ref, "duration_s"].round(1))

    assert greens(runs["actuated"]) != greens(runs["fixed"])

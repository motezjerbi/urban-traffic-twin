"""Parts 3 and 4 end to end on the Pitié-Salpêtrière junction: S0 (fixed), S1 (actuated)
and S2 (rules) through `run_scenario.run`, seed 101, 15 minutes, from a scratch copy of
the repo's configs, net and calibration (nothing is written into the repo).

Checks the run outputs of the Part 3 and Part 4 contracts: the result files and the raw
SUMO files, clean health, a silent safety guard and no dangerous phase sequence (A-04),
the KPI rows (KPI version 2), the measurement tables, the preemption and controller
blocks of run.json, and that S0, S1 and S2 see the same demand (spec 13.3). Each
controller runs once, on first use, so a failing S2 run does not hide the S0 and S1
checks. The `sumo` tests are skipped when SUMO is not installed.
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
from traffic_twin.paths import PITIE_ACTUATED_CONFIG, PITIE_FIXED_CONFIG, PITIE_RULES_CONFIG
from traffic_twin.sim import audit
from traffic_twin.sim.kpi import KPI_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 101  # a calibration seed (D11, D32), never one of the evaluation seeds 1-10
DURATION_S = 900
AMBULANCE_DEPART_S = 300  # inside the shortened window (the scenario's is 2400 s)
CONFIGS = {"fixed": PITIE_FIXED_CONFIG, "actuated": PITIE_ACTUATED_CONFIG,
           "rules": PITIE_RULES_CONFIG}  # fmt: skip
CONTROLLERS = tuple(CONFIGS)  # S0, S1, S2
PART3_CONTROLLERS = ("fixed", "actuated")  # D13: never out of order, no preemption

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
# Part 4 run.json blocks and events table (plan 4.6)
PREEMPTION_BLOCK = {"enabled", "requests", "served", "timeouts", "active_s", "hold_s", "jumps",
                    "ped_yield"}  # fmt: skip
TSP_REPORT = {"calls", "extended_greens", "extension_s", "by_line", "truncated_greens",
              "truncation_s"}  # fmt: skip
EVENT_COLUMNS = {"run_id", "t", "source", "rule", "stage_from", "stage_to", "detail"}
# KPI version 2 rows (plan 4.6) that every 15-minute Pitié run has, whatever the controller
EXTRA_KPIS = {
    ("car", "throughput_vph"), ("car", "delay_total_mean"),
    *(("car", f"{base}__arm_{arm}") for base in ("time_loss_mean", "delay_total_mean")
      for arm in "NSW"),
    ("bus", "delay_total_mean"),
    ("ambulance", "travel_time_total"), ("ambulance", "stop_duration_mean"),
    ("pedestrian", "crossing_complete_rate"), ("pedestrian", "crossing_red_entries"),
    *(("pedestrian", f"ped_curb_wait_{x}") for x in ("mean", "p95", "max")),  # KPI v3
    ("junction", "box_halt_red_s"), ("junction", "box_foe_onsets"),
    ("junction", "exit_blocked_s"), ("junction", "exit_blocked_green_s"),
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


class Runs:
    """The module's runs, keyed by controller name. Each controller runs once, when a
    test first asks for it; a failed run is not retried, its error is raised again."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._done: dict[str, Run | Exception] = {}

    def __getitem__(self, controller: str) -> Run:
        if controller not in self._done:
            try:
                self._done[controller] = self._run(controller)
            except Exception as exc:  # re-raised below, for every test that needs the run
                self._done[controller] = exc
        done = self._done[controller]
        if isinstance(done, Exception):
            raise RuntimeError(f"the {controller} run failed: {done!r}") from done
        return done

    def _run(self, controller: str) -> Run:
        from traffic_twin.sim.run_scenario import run, run_dir

        out_root = self.root / "results"
        with pytest.MonkeyPatch.context() as mp:
            mp.chdir(self.root)
            cfg = load_scenario(CONFIGS[controller])
            assert cfg["controller"] == controller
            cfg["duration_s"] = DURATION_S
            cfg["emergency"]["depart_s"] = AMBULANCE_DEPART_S
            cfg["outputs"]["dir"] = "results"
            run(cfg, SEED, out_root)
            out = run_dir(cfg, SEED, out_root)
        meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
        return Run(cfg, out, meta)


@pytest.fixture(scope="module")
def runs(make_workspace, pitie_scenario, tmp_path_factory) -> Runs:
    """S0, S1 and S2, each run at most once for the whole module (on first use). The
    three configs share the Pitié net and calibration, so one workspace serves them."""
    return Runs(make_workspace(tmp_path_factory.mktemp("pitie_e2e"), pitie_scenario))


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


def test_s0_s1_s2_configs_share_one_workspace(pitie_scenario, rules_scenario):
    """The `runs` workspace is built from S0's inputs: S1 and S2 must read the same net
    and calibration, and S2 must be the rules controller on S0's demand (D19)."""
    actuated = load_scenario(REPO_ROOT / PITIE_ACTUATED_CONFIG)
    for cfg in (actuated, rules_scenario):
        assert cfg["network"]["net_file"] == pitie_scenario["network"]["net_file"]
        assert cfg["demand"].get("calibration") == pitie_scenario["demand"].get("calibration")
        assert demand_hash(cfg) == demand_hash(pitie_scenario)
    assert [cfg["controller"] for cfg in (pitie_scenario, actuated, rules_scenario)] == list(
        CONTROLLERS
    )


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
    """D13/D22: the guard never intervenes (S2 included: D27 expects 0); A-04: no
    dangerous phase sequence (D31). Fixed and actuated never serve a green out of order;
    S2 may (preemption can skip stages), which the audit reports as information only."""
    safety = runs[controller].meta["safety"]
    assert safety["interventions"] == dict.fromkeys(INTERVENTION_KEYS, 0)
    assert set(safety["audit"]) == set(audit.AUDIT_KEYS)
    danger = {k: safety["audit"][k] for k in audit.DANGER_KEYS}
    assert danger == dict.fromkeys(audit.DANGER_KEYS, 0)
    assert safety["danger"] == 0
    assert safety["ped_forcing"] == "detected"  # D21: the shield is on for every controller
    if controller in PART3_CONTROLLERS:
        assert safety["audit"]["out_of_order_green"] == 0


@pytest.mark.sumo
@pytest.mark.parametrize("controller", PART3_CONTROLLERS)
def test_part3_controllers_have_no_preemption(runs, controller):
    """S0 and S1 run without preemption (D24): the block is there, disabled, all zeros;
    fixed and actuated report no controller activity."""
    meta = runs[controller].meta
    pre = meta["preemption"]
    assert set(pre) >= PREEMPTION_BLOCK
    assert pre["enabled"] is False
    assert all(v == 0 for k, v in pre.items() if k != "enabled"), pre
    assert meta["control"] == {}


@pytest.mark.sumo
def test_rules_preempts_for_the_ambulance(runs):
    """EF-07: the window's one ambulance is detected, served and never timed out; the
    preemption manager logs its request and its service in events.parquet."""
    r = runs["rules"]
    pre = r.meta["preemption"]
    assert set(pre) >= PREEMPTION_BLOCK
    assert pre["enabled"] is True
    assert pre["requests"] == pre["served"] == 1
    assert pre["timeouts"] == 0
    events = r.table("events.parquet")
    assert set(events.columns) >= EVENT_COLUMNS
    assert (events["run_id"] == r.meta["run_id"]).all()
    rules = set(events.loc[events["source"] == "preemption", "rule"])
    assert {"request", "served"} <= rules and "timeout" not in rules


@pytest.mark.sumo
def test_rules_reports_its_activity(runs):
    """run.json `control` is RuleBasedController.report() (plan 4.6)."""
    control = runs["rules"].meta["control"]
    assert {"ped_protect", "box_gapouts", "tsp"} <= set(control)
    assert set(control["tsp"]) >= TSP_REPORT
    assert control["tsp"]["truncated_greens"] == 0  # paid truncation is off at Pitié (D23)


@pytest.mark.sumo
def test_rules_ambulance_not_slower_than_fixed(runs):
    """A-05 on this one calibration seed: S2 (with preemption) does not delay the
    ambulance compared with S0. A single-seed sanity check, not the pre-registered test
    (seeds 1-10, configs/evaluation.yaml). If it fails at the gate it is removed, never
    tuned (plan WS6)."""
    s0, s2 = runs["fixed"], runs["rules"]
    assert s2.kpi("ambulance", "travel_time_total") <= s0.kpi("ambulance", "travel_time_total")


@pytest.mark.sumo
@pytest.mark.parametrize("controller", CONTROLLERS)
def test_kpi_rows(runs, controller):
    r = runs[controller]
    kpis = pd.read_csv(r.out / "kpi_run.csv")
    assert r.meta["kpi_version"] == KPI_VERSION
    assert (kpis["run_id"] == r.meta["run_id"]).all()
    assert {"car", "bus", "pedestrian", "ambulance", "junction"} <= set(kpis["user_type"])
    assert not kpis.duplicated(["user_type", "kpi"]).any()
    assert EXTRA_KPIS <= set(zip(kpis["user_type"], kpis["kpi"], strict=True))
    assert r.kpi("ambulance", "count") == 1
    assert r.kpi("car", "count") > 100 and r.kpi("bus", "count") > 0
    wait_max = r.kpi("pedestrian", "ped_wait_single_max")  # D14 (Part 3), reported
    assert math.isfinite(wait_max) and 0 <= wait_max <= r.cfg["safety"]["ped_max_wait_s"]
    curb_max = r.kpi("pedestrian", "ped_curb_wait_max")  # A-04 since changes 2026-10-12
    assert wait_max <= curb_max <= r.cfg["safety"]["ped_max_wait_s"]
    assert "depart_delay_s" in r.table("trip.parquet").columns
    # measured from the planned departure, so it includes the insertion delay
    assert r.kpi("ambulance", "travel_time_total") >= r.kpi("ambulance", "travel_time")
    assert 0 <= r.kpi("pedestrian", "crossing_complete_rate") <= 1


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
def test_s0_s1_s2_share_their_demand(runs):
    """Spec 13.3: the comparison is paired, so S0, S1 and S2 must see the same demand
    (D19: the rules fragment never touches demand, emergency or network)."""
    s0, s1, s2 = (runs[c] for c in CONTROLLERS)
    assert s0.meta["demand_hash"] == s1.meta["demand_hash"] == s2.meta["demand_hash"]
    assert len({r.meta["config_hash"] for r in (s0, s1, s2)}) == 3
    ids = sorted(s0.table("trip.parquet")["id"])
    for r in (s1, s2):
        assert sorted(r.table("trip.parquet")["id"]) == ids, r.cfg["scenario"]


@pytest.mark.sumo
def test_actuated_greens_differ_from_fixed(runs):
    """S1 really runs the actuated controller: the reference group's greens change."""

    def greens(r: Run) -> set[float]:
        net = r.cfg["network"]
        ref = net["signal_groups"][net["reference_group"]][0]
        signal = r.table("signal.parquet")
        return set(signal.loc[signal["link"] == ref, "duration_s"].round(1))

    assert greens(runs["actuated"]) != greens(runs["fixed"])

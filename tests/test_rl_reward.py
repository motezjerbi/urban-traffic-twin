"""RL reward `rl-reward-1` (Part 6, D58): cost arithmetic on a fake session, pending
insertions by class, and the pedestrian penalty at exactly the threshold (off) and above
it (on). No SUMO.

Part 7 (D76): `weighted_cost` reweights weight-free totals; the passive
`RewardObserver` accrues like the tracker (fake session), equals the env's tracker on a
900 s fixed-program run and leaves the KPIs unchanged (`@sumo`, training seed 5000)."""

import json
from types import SimpleNamespace

import pytest

from traffic_twin.control.base import EXTRA_PED_COUNT, EXTRA_PED_WAIT, Observation
from traffic_twin.rl.reward import (
    REWARD_VERSION,
    TOTAL_KEYS,
    TOTALS_FILE,
    RewardObserver,
    RewardTracker,
    weighted_cost,
)

CFG = {
    "version": "rl-reward-1",
    "weights": {"car": 1.3, "bus": 25.0, "pedestrian": 1.2, "ambulance": 20.0},
    "pending_as_waiting": True,
    "ped_penalty": {"lambda": 30.0, "threshold_s": 80.0},
    "scale": 0.001,
}


class FakeSim:
    """The few SimSession members `RewardTracker.accrue` reads."""

    def __init__(self, vehicles, pending=(), ped_count=(0, 0, 0, 0), ped_wait=(0.0,) * 4):
        self.dt = 1.0
        self.types = {vid: vtype for vid, (vtype, _w) in vehicles.items()}
        self.types.update(dict(pending))
        self.waits = {vid: w for vid, (_t, w) in vehicles.items()}
        self.type_reads = 0
        self.traci = SimpleNamespace(
            vehicle=SimpleNamespace(
                getIDList=lambda: tuple(vehicles),
                getWaitingTime=lambda vid: self.waits[vid],
                getTypeID=self._type,
            ),
            simulation=SimpleNamespace(getPendingVehicles=lambda: tuple(v for v, _ in pending)),
        )
        extras = {EXTRA_PED_COUNT: tuple(ped_count), EXTRA_PED_WAIT: tuple(ped_wait)}
        self._obs = Observation(0.0, 0, 10.0, False, {}, extras)

    def _type(self, vid):
        self.type_reads += 1
        return self.types[vid]

    def observe(self):
        return self._obs


VEHICLES = {
    "car_N_S_00.0": ("car", 3.0),  # waiting
    "car_N_S_00.1": ("car", 0.0),  # moving
    "car_W_S_01.0": ("car", 12.0),
    "bus_57_N.0": ("bus", 1.0),
    "bus_91_E.0": ("bus", 0.0),
    "ambulance_0": ("ambulance", 2.0),
}
PENDING = (("car_N_S_02.0", "car"), ("car_N_S_02.1", "car"), ("bus_24_W.0", "bus"))


def test_version():
    assert REWARD_VERSION == "rl-reward-1"


def test_cost_arithmetic():
    sim = FakeSim(VEHICLES, PENDING, ped_count=(2, 0, 1, 0), ped_wait=(30.0, 0, 50.0, 0))
    tracker = RewardTracker(CFG)
    cost = tracker.accrue(sim)
    # cars: 2 waiting + 2 pending; bus: 1 waiting + 1 pending; ambulance 1; peds 3
    assert cost == pytest.approx(1.3 * 4 + 25.0 * 2 + 1.2 * 3 + 20.0 * 1)
    t = tracker.totals
    assert set(t) == set(TOTAL_KEYS)
    assert (t["car_s"], t["car_pending_s"], t["bus_s"], t["ambulance_s"]) == (4, 2, 2, 1)
    assert (t["pedestrian_s"], t["penalty_s"]) == (3, 0)
    assert t["cost"] == pytest.approx(cost)
    assert tracker.reward(cost) == pytest.approx(-0.001 * cost)


def test_totals_accumulate_and_types_are_read_once():
    sim = FakeSim(VEHICLES, PENDING)
    tracker = RewardTracker(CFG)
    c1, c2 = tracker.accrue(sim), tracker.accrue(sim)
    assert tracker.totals["cost"] == pytest.approx(c1 + c2)
    assert tracker.totals["car_s"] == 8
    assert sim.type_reads == 4 + 3  # waiting vehicles and pending ones, once each
    tracker.reset()
    assert tracker.totals == dict.fromkeys(TOTAL_KEYS, 0.0)


def test_pending_not_counted_when_disabled():
    tracker = RewardTracker({**CFG, "pending_as_waiting": False})
    cost = tracker.accrue(FakeSim(VEHICLES, PENDING))
    assert cost == pytest.approx(1.3 * 2 + 25.0 * 1 + 20.0 * 1)
    assert tracker.totals["car_pending_s"] == 0


@pytest.mark.parametrize(("wait", "on"), [(79.0, False), (80.0, False), (80.001, True), (88, True)])
def test_pedestrian_penalty_at_the_threshold(wait, on):
    sim = FakeSim({}, (), ped_count=(1, 0, 0, 0), ped_wait=(wait, 0.0, 10.0, 0.0))
    tracker = RewardTracker(CFG)
    cost = tracker.accrue(sim)
    assert cost == pytest.approx(1.2 + (30.0 if on else 0.0))
    assert tracker.totals["penalty_s"] == (1 if on else 0)


def test_no_crossings_no_pedestrian_terms():
    sim = FakeSim({"car_0": ("car", 1.0)})
    sim._obs = Observation(0.0, 0, 10.0, False, {}, {})  # the grid has no crossings
    assert RewardTracker(CFG).accrue(sim) == pytest.approx(1.3)


def test_dt_scales_the_cost():
    sim = FakeSim({"car_0": ("car", 1.0)})
    sim.dt = 0.5
    assert RewardTracker(CFG).accrue(sim) == pytest.approx(0.65)


def test_missing_weight_is_refused():
    with pytest.raises(ValueError, match="bus"):
        RewardTracker({**CFG, "weights": {"car": 1.3, "pedestrian": 1.2, "ambulance": 20}})


# -- Part 7: the reward-totals observer (D76) ----------------------------------------------
def test_weighted_cost_reproduces_the_cost():
    sim = FakeSim(VEHICLES, PENDING, ped_count=(2, 0, 1, 0), ped_wait=(85.0, 0, 50.0, 0))
    tracker = RewardTracker(CFG)
    for _ in range(3):
        tracker.accrue(sim)
    assert weighted_cost(tracker.totals, CFG) == pytest.approx(tracker.totals["cost"])
    penalty = {"lambda": 100.0, "threshold_s": 80.0}  # theta fixed (D76)
    other = {**CFG, "weights": {**CFG["weights"], "bus": 50.0}, "ped_penalty": penalty}
    t2 = RewardTracker(other)
    for _ in range(3):
        t2.accrue(sim)
    assert weighted_cost(tracker.totals, other) == pytest.approx(t2.totals["cost"])


def test_observer_accrues_like_the_tracker_and_writes_json(tmp_path):
    sim = FakeSim(VEHICLES, PENDING, ped_count=(1, 0, 0, 0), ped_wait=(81.0, 0, 0, 0))
    sim.begin, sim.end, sim.time = 0.0, 2.0, 0.0
    obs, tracker = RewardObserver(CFG), RewardTracker(CFG)
    obs.bind(sim)
    for _ in range(4):  # 2 steps in the window, 2 in the drain
        sim.time += 1.0
        obs.after_step(sim)
        tracker.accrue(sim)
        if sim.time == 2.0:
            window = dict(tracker.totals)
    assert obs.totals == tracker.totals and obs.steps == 4
    assert obs.totals_window == window and window["cost"] == pytest.approx(obs.totals["cost"] / 2)
    obs.finish(sim, tmp_path, "run-1")
    doc = json.loads((tmp_path / TOTALS_FILE).read_text(encoding="utf-8"))
    assert doc["reward_version"] == REWARD_VERSION and doc["run_id"] == "run-1"
    assert doc["totals"] == tracker.totals and doc["totals_window"] == window
    assert doc["reward"] == CFG and doc["steps"] == 4
    obs.bind(sim)  # a new run starts from zero
    assert obs.totals == dict.fromkeys(TOTAL_KEYS, 0.0) and obs.steps == 0


# -- on SUMO (Part 7 training seed 5000) -------------------------------------------------
DURATION_S = 900


@pytest.fixture(scope="module")
def fixed_runs(rl_scenario, rl_train_scenario, tmp_path_factory):
    """The fixed program on seed 5000 for 900 s: run_scenario with and without the
    observer, and the env's `fixed` baseline in full mode (no randomization)."""
    import copy

    import yaml

    from traffic_twin.rl.env import TrafficEnv, run_baseline
    from traffic_twin.sim.run_scenario import run

    tmp = tmp_path_factory.mktemp("reward_obs")
    cfg = copy.deepcopy(rl_scenario)
    cfg["controller"], cfg["duration_s"] = "fixed", DURATION_S
    observer = RewardObserver(rl_train_scenario["reward"])
    with_obs = run(copy.deepcopy(cfg), 5000, tmp / "with", demand_dir=tmp / "demand",
                   observers=[observer])  # fmt: skip
    without = run(copy.deepcopy(cfg), 5000, tmp / "without", demand_dir=tmp / "demand")
    rl_cfg = copy.deepcopy(rl_train_scenario)
    rl_cfg["episode"]["full"] = {"warmup_s": 0, "duration_s": DURATION_S}
    config = tmp / "rl.yaml"
    config.write_text(yaml.safe_dump(rl_cfg), encoding="utf-8")
    env = TrafficEnv(config, mode="full", randomize=False, work_root=tmp / "env")
    try:
        summary = run_baseline(env, "fixed", 5000)
    finally:
        env.close()
    return {"tmp": tmp, "observer": observer, "with": with_obs, "without": without,
            "summary": summary}  # fmt: skip


@pytest.mark.sumo
def test_observer_equals_the_env_tracker(fixed_runs):
    observer, summary = fixed_runs["observer"], fixed_runs["summary"]
    assert summary["t_end"] == pytest.approx(DURATION_S)
    assert observer.totals_window == summary["totals"]  # same steps, same arithmetic
    assert observer.steps > DURATION_S  # the run drains after the window
    assert observer.totals["cost"] >= observer.totals_window["cost"] > 0
    (path,) = (fixed_runs["tmp"] / "with").rglob(TOTALS_FILE)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["run_id"] == fixed_runs["with"]["run_id"] and doc["totals"] == observer.totals


@pytest.mark.sumo
def test_observer_is_passive(fixed_runs):
    import pandas as pd

    def kpis(name: str) -> pd.DataFrame:
        (path,) = (fixed_runs["tmp"] / name).rglob("kpi_run.csv")
        return pd.read_csv(path).drop(columns=["run_id"], errors="ignore")

    pd.testing.assert_frame_equal(kpis("with"), kpis("without"))
    a, b = fixed_runs["with"], fixed_runs["without"]
    assert a["health"] == b["health"] and a["safety"] == b["safety"]
    assert not list((fixed_runs["tmp"] / "without").rglob(TOTALS_FILE))

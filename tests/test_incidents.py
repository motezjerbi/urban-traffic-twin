"""Incidents as lane slowdowns (D45): sampling, the manager on a fake TraCI lane API and,
with SUMO, a slowdown applied and restored on the Pitié junction."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_twin.config import ConfigError
from traffic_twin.paths import INCIDENT_SITES
from traffic_twin.sim.incidents import (
    ActiveIncident,
    Incident,
    IncidentManager,
    load_sites,
    sample_incidents,
)

COUNT_P = [0.5, 0.35, 0.15]


def draw(seed: int, **kw) -> tuple[Incident, ...]:
    args = {"count_p": COUNT_P, "start_s": (300, 6000), "durations_s": [300, 600, 900, 1200]}
    return sample_incidents(np.random.default_rng([seed, 5_000_000]), load_sites(), **args | kw)


def test_sites_file_is_valid():
    sites = load_sites(INCIDENT_SITES)
    assert sites["speed_mps"] == 2.0
    assert [s["id"] for s in sites["sites"]] == ["n_in_1", "n_in_2", "s_in_1", "s_in_2", "w_in_1"]
    assert {s["arm"] for s in sites["sites"]} == {"N", "S", "W"}
    assert all(s["lane"].startswith(s["arm"].lower() + "_in_") for s in sites["sites"])


def test_sites_are_lanes_of_the_pitie_net():
    from traffic_twin.sim.detectors import read_net

    lanes = read_net("sumo/network/pitie/pitie.net.xml").lanes
    assert all(s["lane"] in lanes for s in load_sites()["sites"])


def test_bad_sites_file_is_refused(write_file):
    path = write_file(
        "sites.yaml",
        "speed_mps: 2.0\nsites: [{id: a, arm: N, lane: x}, {id: a, arm: S, lane: y}]\n",
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_sites(path)
    path = write_file("sites2.yaml", "speed_mps: 0\nsites: [{id: a, arm: N, lane: x}]\n")
    with pytest.raises(ConfigError, match="speed_mps"):
        load_sites(path)


def test_sampling_is_deterministic_sorted_and_valid():
    sites = {s["id"]: s for s in load_sites()["sites"]}
    counts = []
    for seed in range(2000, 2400):
        a, b = draw(seed), draw(seed)
        assert a == b
        assert list(a) == sorted(a, key=lambda i: (i.start_s, i.site))
        for inc in a:
            assert sites[inc.site]["lane"] == inc.lane and sites[inc.site]["arm"] == inc.arm
            assert 300 <= inc.start_s < 6000 and inc.start_s == int(inc.start_s)
            assert inc.duration_s in (300, 600, 900, 1200) and inc.speed_mps == 2.0
        counts.append(len(a))
    shares = np.bincount(counts, minlength=3) / len(counts)
    assert np.allclose(shares, COUNT_P, atol=0.08)


def test_sampling_checks_its_arguments():
    with pytest.raises(ValueError, match="count_p"):
        draw(1, count_p=[0.5, 0.4])
    with pytest.raises(ValueError, match="start_s"):
        draw(1, start_s=(10, 10))
    one = draw(7, count_p=[0.0, 1.0])  # the RL form: one incident with probability p
    assert len(one) == 1


class FakeLanes:
    def __init__(self):
        self.speed = {"w_in_1": 13.89, "n_in_1": 13.89}
        self.calls: list[tuple[str, float]] = []

    def getMaxSpeed(self, lane):
        return self.speed[lane]

    def setMaxSpeed(self, lane, v):
        self.calls.append((lane, v))
        self.speed[lane] = v


def test_manager_applies_and_restores_overlapping_incidents():
    lanes = FakeLanes()
    incs = [
        Incident("w_in_1", "W", "w_in_1", 10.0, 20.0, 2.0),
        Incident("w_in_1", "W", "w_in_1", 20.0, 30.0, 2.0),  # overlaps the first
        Incident("n_in_1", "N", "n_in_1", 15.0, 5.0, 2.0),
    ]
    mgr = IncidentManager(SimpleNamespace(lane=lanes), incs)
    speeds = {}
    for t in range(60):
        mgr.update(float(t))
        speeds[t] = dict(lanes.speed)
    assert speeds[9]["w_in_1"] == 13.89 and speeds[10]["w_in_1"] == 2.0
    assert speeds[45]["w_in_1"] == 2.0 and speeds[49]["w_in_1"] == 2.0  # second ends at 50
    assert speeds[50]["w_in_1"] == 13.89  # original restored once, after the last one
    assert speeds[14]["n_in_1"] == 13.89 and speeds[15]["n_in_1"] == 2.0
    assert speeds[20]["n_in_1"] == 13.89
    log = mgr.log()
    assert [r["actual_start_s"] for r in log] == [10.0, 15.0, 20.0]
    assert [r["end_s"] for r in log] == [30.0, 20.0, 50.0]
    assert mgr.active() == ()


def test_active_reports_the_actual_start():
    mgr = IncidentManager(
        SimpleNamespace(lane=FakeLanes()), [Incident("w_in_1", "W", "w_in_1", 10.5, 100, 2.0)]
    )
    mgr.update(10.0)
    assert mgr.active() == ()
    mgr.update(11.0)  # first update at or after the planned start
    assert mgr.active() == (ActiveIncident("w_in_1", "W", "w_in_1", 11.0),)


@pytest.mark.sumo
def test_slowdown_is_applied_and_restored_in_sumo(rules_cfg, tmp_path):
    from traffic_twin.control.registry import make_controller
    from traffic_twin.sim.session import SimSession

    cfg = copy.deepcopy(rules_cfg)
    cfg["duration_s"] = 200
    ctrl = make_controller(cfg)
    inc = Incident("w_in_1", "W", "w_in_1", 60.0, 100.0, 2.0)
    teleports, speeds = 0, {}
    with SimSession(cfg, 101, tmp_path / "out", demand_dir=tmp_path / "demand") as sim:
        mgr = IncidentManager(sim.traci, [inc])
        original = sim.traci.lane.getMaxSpeed("w_in_1")
        ctrl.reset(101)
        while sim.t < 200:
            mgr.update(sim.t)
            speeds[int(sim.t)] = sim.traci.lane.getMaxSpeed("w_in_1")
            sim.step(ctrl.decide(sim.observe()))
            teleports += sim.traci.simulation.getStartingTeleportNumber()
    assert original > 2.0
    assert speeds[59] == original and speeds[60] == 2.0 and speeds[159] == 2.0
    assert speeds[160] == original and speeds[199] == original
    assert mgr.log()[0] == {
        "site": "w_in_1", "lane": "w_in_1", "planned_start_s": 60.0, "actual_start_s": 60.0,
        "end_s": 160.0,
    }  # fmt: skip
    assert teleports == 0

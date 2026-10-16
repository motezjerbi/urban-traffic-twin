"""JunctionSampler (contract 4.2) on the real Pitié junction: layout, signal fields,
pending vehicles and passivity. Marked `sumo`."""

import copy
from pathlib import Path

import pytest

from traffic_twin.control.registry import make_controller
from traffic_twin.sim.incidents import Incident, IncidentManager
from traffic_twin.sim.sampler import PENDING_OTHER, JunctionSampler, JunctionSnapshot
from traffic_twin.sim.session import SimSession

pytestmark = pytest.mark.sumo

SEED = 101
LANES = ("n_in_1", "n_in_2", "nb_in_0", "s_in_1", "s_in_2", "wb_in_1", "w_in_1")
ARMS = ("N", "S", "W", "NB", "WB")


def short(cfg: dict, duration_s: int) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["duration_s"] = duration_s
    return cfg


def run(cfg: dict, out: Path, sample: bool, until: float | None = None, incidents=()):
    """Drive the S2 loop; with `sample`, take a snapshot every step before deciding.
    Returns (sim, snapshots)."""
    ctrl = make_controller(cfg)
    snaps: list[JunctionSnapshot] = []
    with SimSession(cfg, SEED, out / "out", demand_dir=out / "demand") as sim:
        mgr = IncidentManager(sim.traci, incidents) if incidents else None
        sampler = JunctionSampler(sim, mgr) if sample else None
        ctrl.reset(SEED)
        while not sim.done and (until is None or sim.t < until):
            if mgr is not None:
                mgr.update(sim.t)
            if sampler is not None:
                snaps.append(sampler.sample())
            sim.step(ctrl.decide(sim.observe()))
    return sim, snaps


@pytest.fixture(scope="module")
def sampled(rules_scenario, tmp_path_factory):
    """300 s of S2 with two ambulances at 100 s on N -> S (the second one waits for
    insertion) and an incident on w_in_1 from 150 s, sampled every step."""
    cfg = short(rules_scenario, 300)
    trip = {"depart_s": 100, "route": ["n_in", "s_out"]}
    cfg["emergency"] = {"enabled": True, "trips": [trip, dict(trip)]}
    inc = Incident("w_in_1", "W", "w_in_1", 150.0, 60.0, 2.0)
    return run(cfg, tmp_path_factory.mktemp("sampler"), True, until=300, incidents=[inc])


def test_layout_lanes_arms_and_crossings(sampled):
    sim, snaps = sampled
    snap = snaps[-1]
    assert tuple(lane.lane for lane in snap.lanes) == LANES  # 7 lanes, TLS order
    assert {lane.arm for lane in snap.lanes} == set(ARMS)
    assert set(snap.arm_halting) == set(snap.arm_vehicles) == set(snap.exit_halting) == set(ARMS)
    assert len(snap.ped_count) == len(snap.ped_wait_s) == len(snap.ped_slack_s) == 4
    assert len(snap.red_s) == len(snap.gap_s) == 5
    assert set(snap.station_occupancy) == {"N_in", "S_in", "W_in", "N_out", "S_out", "W_out"}
    assert set(snap.queue_jam_m) == {"N", "S", "W"}
    assert set(snap.pending) == {*ARMS, PENDING_OTHER}
    assert set(snap.pending_by_class) == {"car", "bus", "ambulance"}


def test_values_are_in_range(sampled):
    _, snaps = sampled
    for s in snaps:
        assert all(0.0 <= lane.occupancy <= 1.0 for lane in s.lanes)
        assert all(0.0 <= occ <= 1.0 for occ in s.station_occupancy.values())
        assert all(lane.halting <= lane.vehicles for lane in s.lanes)
        assert all(lane.max_wait_s >= 0 for lane in s.lanes)
        assert s.clock_s == s.t + 27000
        assert set(s.exit_bus_halting) == set(ARMS)
        assert all(0 <= s.exit_bus_halting[a] <= s.exit_halting[a] for a in ARMS)
    assert any(s.lanes[0].vehicles for s in snaps) and any(s.buses for s in snaps)
    assert sum(sum(s.station_new_vehicles.values()) for s in snaps) > 50
    assert all(set(s.bus_time_loss_s) == {b.vid for b in s.buses} for s in snaps)


def test_signal_fields_follow_the_plan(sampled):
    sim, snaps = sampled
    durations = sim.plan.durations
    assert snaps[0].phase == -1
    for k, s in enumerate(snaps[1:], start=1):
        assert s.phase == sim.phases_shown[k - 1]
    runs, current = [], []
    for s in snaps:
        if s.in_transition:
            assert s.target_stage is not None and s.target_stage != s.stage
            current.append(s)
        else:
            assert s.clearance_left_s == 0.0 and s.target_stage is None
            if current:
                runs.append(current)
                current = []
    assert len(runs) >= 3
    for r in runs:
        total = int(sum(durations[p] for p in sim.plan.stages[r[0].stage].clearance_phases))
        # sampled after the first clearance second: total - 1, ..., 1
        assert [s.clearance_left_s for s in r] == [float(total - k) for k in range(1, total)]


def test_pending_ambulance_and_incident_are_visible(sampled):
    _, snaps = sampled
    pending = [s for s in snaps if s.pending_by_class["ambulance"]]
    assert pending, "the second ambulance never waited for insertion"
    for s in pending:
        assert s.pending["N"] >= 1
        assert any(c.pending and c.approach == "N" for c in s.emergency)
    with_inc = [s for s in snaps if s.incidents]
    assert [s.t for s in with_inc] == [float(t) for t in range(150, 210)]
    assert {i.since_s for s in with_inc for i in s.incidents} == {150.0}


def tripinfo_body(path: Path) -> str:
    """tripinfo.xml from its root element (the header comment names the output paths)."""
    text = path.read_text(encoding="utf-8")
    return text[text.index("<tripinfos") :]


def test_sampling_every_step_is_passive(rules_scenario, tmp_path):
    cfg = short(rules_scenario, 600)
    plain, _ = run(cfg, tmp_path / "plain", sample=False)
    sampled, snaps = run(cfg, tmp_path / "sampled", sample=True)
    assert len(snaps) == len(sampled.phases_shown) > 600
    assert sampled.phases_shown == plain.phases_shown
    assert tripinfo_body(sampled.tripinfo) == tripinfo_body(plain.tripinfo)

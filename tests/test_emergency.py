"""Emergency vehicle detection (Part 4, D24): `EmergencyDetector` on a fake TraCI (link,
target stage, pending vehicles, distance limit, caching) and on the real Pitié net in
SUMO (seed 101, 300 s), fed into the `PreemptionManager`."""

import math
from types import SimpleNamespace

import pytest

from traffic_twin.config import ConfigError, load_scenario
from traffic_twin.control.preemption import PreemptionManager
from traffic_twin.paths import PITIE_RULES_CONFIG
from traffic_twin.sim.emergency import EmergencyDetector
from traffic_twin.sim.signal import Stage

# Three stages on four links: link 0 G in stage 0, link 1 g in stage 1 and G in stage 2,
# link 2 g in stage 2 only, link 3 never green.
STATES = ["GrrrG", "yrrrr", "rgrrr", "ryrrr", "rGgrr", "ryyrr"]
STAGES = [Stage("A", 0, (1,)), Stage("B", 2, (3,)), Stage("C", 4, (5,))]
CONTROLLED = [
    [("a_in_0", "c_out_0", ":J_0_0")],
    [("b_in_0", "c_out_0", ":J_1_0")],
    [("b_in_1", "d_out_0", ":J_2_0")],
    [("x_in_0", "c_out_0", ":J_3_0")],
    [],
]
APPROACHES = {"A": ["a_in"], "B": ["b_in"]}


class FakeTraci:
    """vehicle / simulation / trafficlight getters the detector uses, with call counts."""

    def __init__(self):
        self.classes = {"amb": "emergency", "car": "passenger", "amb2": "emergency",
                        "amb3": "emergency", "amb_x": "emergency"}  # fmt: skip
        self.next_tls: dict[str, list] = {}  # inserted vehicles
        self.routes: dict[str, list[str]] = {}  # pending vehicles
        self.pending: list[str] = []
        self.class_calls = 0
        self.vehicle = SimpleNamespace(
            getIDList=lambda: list(self.next_tls),
            getNextTLS=lambda v: self.next_tls[v],
            getVehicleClass=self._cls,
            getRoute=lambda v: self.routes[v],
        )
        self.simulation = SimpleNamespace(getPendingVehicles=lambda: list(self.pending))
        self.trafficlight = SimpleNamespace(getControlledLinks=lambda tls: CONTROLLED)

    def _cls(self, vid):
        self.class_calls += 1
        return self.classes[vid]


def detector(traci, **kw) -> EmergencyDetector:
    return EmergencyDetector(traci, "J", STAGES, STATES, APPROACHES, **kw)


def test_inserted_vehicle_call_and_target_stage():
    tr = FakeTraci()
    tr.next_tls = {"car": [("J", 0, 30.0, "r")], "amb": [("K", 5, 10.0, "G"), ("J", 1, 42.0, "r")]}
    det = detector(tr)
    (c,) = det.calls(3.0)
    assert (c.vid, c.link, c.stage, c.dist_m, c.first_seen_s) == ("amb", 1, 2, 42.0, 3.0)
    assert (c.pending, c.approach) == (False, "B")  # 'G' in stage 2 beats 'g' in stage 1
    assert det.target_stage(0) == 0 and det.target_stage(2) == 2  # 'g' only: permissive
    assert det.target_stage(3) is None


def test_call_ends_past_the_stop_line_and_first_seen_is_kept():
    tr = FakeTraci()
    tr.next_tls = {"amb": [("J", 0, 80.0, "r")]}
    det = detector(tr)
    assert det.calls(1.0)[0].first_seen_s == 1.0
    tr.next_tls["amb"] = [("J", 0, 20.0, "G")]
    assert det.calls(2.0)[0].first_seen_s == 1.0
    tr.next_tls["amb"] = []  # on the internal lane / out edge: light no longer ahead
    assert det.calls(3.0) == ()
    assert tr.class_calls == 1  # the vClass is read once per vehicle


def test_pending_vehicles_use_their_route_start():
    tr = FakeTraci()
    tr.pending = ["car", "amb", "amb2", "amb3"]
    tr.routes = {"amb": ["a_in", "c_out"], "amb2": ["b_in", "d_out"], "amb3": ["c_out"]}
    calls = detector(tr).calls(7.0)
    assert [(c.vid, c.link, c.stage, c.pending, c.approach) for c in calls] == [
        ("amb", 0, 0, True, "A"),
        ("amb2", 2, 2, True, "B"),
    ]  # amb3: one-edge route, no TLS link
    assert all(math.isinf(c.dist_m) for c in calls)
    tr.routes["amb_x"] = ["x_in", "c_out"]  # link 3: green in no stage, ignored
    tr.pending = ["amb_x"]
    assert detector(tr).calls(0.0) == ()
    assert detector(tr, pending=False).calls(0.0) == ()


def test_pending_then_inserted_keeps_first_seen_and_sorts_fcfs():
    tr = FakeTraci()
    tr.pending, tr.routes = ["amb"], {"amb": ["a_in", "c_out"]}
    tr.next_tls = {"amb2": [("J", 1, 10.0, "r")]}
    det = detector(tr)
    assert [c.vid for c in det.calls(5.0)] == ["amb", "amb2"]  # same first_seen: by vid
    tr.pending, tr.next_tls = [], {"amb2": [("J", 1, 5.0, "r")], "amb": [("J", 0, 200.0, "r")]}
    calls = det.calls(6.0)
    assert [(c.vid, c.first_seen_s, c.pending) for c in calls] == [
        ("amb", 5.0, False),
        ("amb2", 5.0, False),
    ]
    tr.next_tls["amb3"] = [("J", 2, 50.0, "r")]
    assert [c.vid for c in det.calls(7.0)] == ["amb", "amb2", "amb3"]


def test_distance_limit_applies_to_inserted_vehicles_only():
    tr = FakeTraci()
    tr.next_tls = {"amb": [("J", 0, 150.0, "r")]}
    tr.pending, tr.routes = ["amb2"], {"amb2": ["a_in", "c_out"]}
    det = detector(tr, max_dist_m=100.0)
    assert [c.vid for c in det.calls(0.0)] == ["amb2"]
    tr.next_tls["amb"] = [("J", 0, 99.0, "r")]
    calls = det.calls(1.0)
    assert [(c.vid, c.first_seen_s) for c in calls] == [("amb2", 0.0), ("amb", 1.0)]


def test_detector_from_config(probe_cfg):
    tr = FakeTraci()
    probe_cfg["network"]["tls_id"] = "J"
    det = EmergencyDetector.from_config(tr, probe_cfg, STAGES, STATES)
    assert (det.pending, det.max_dist_m, det.tls_id) == (True, None, "J")
    assert det.approaches["N"] == ["n_in"]
    probe_cfg["preemption"]["detection"] = {"pending": False, "max_dist_m": 80}
    det = EmergencyDetector.from_config(tr, probe_cfg, STAGES, STATES)
    assert (det.pending, det.max_dist_m) == (False, 80.0)
    del probe_cfg["preemption"]  # off: detection defaults (observation only)
    assert EmergencyDetector.from_config(tr, probe_cfg, STAGES, STATES).pending is True
    for bad, match in [
        ({"pending": "yes"}, "pending"),
        ({"max_dist_m": 0}, "max_dist_m"),
        ({"max_dist_m": True}, "max_dist_m"),
        ([1], "mapping"),
    ]:
        probe_cfg["preemption"] = {"detection": bad}
        with pytest.raises(ConfigError, match=match):
            EmergencyDetector.from_config(tr, probe_cfg, STAGES, STATES)


# -- SUMO ------------------------------------------------------------------------------
@pytest.mark.sumo
def test_detection_on_the_pitie_net(pitie_workdir, tmp_path):
    """Seed 101, 300 s of the S2 config's demand without its own ambulance; ambulances
    added through TraCI: one on n_in (link 2, target HOP), four at once at the start of
    n_in (some wait for insertion: pending, dist inf), one W -> S (link 10, STM)."""
    from traffic_twin.sim.demand import write_demand
    from traffic_twin.sim.sumo import sumo_command, traci_module

    cfg = load_scenario(PITIE_RULES_CONFIG)
    cfg["duration_s"] = 300
    cfg["emergency"]["enabled"] = False
    rou, add = write_demand(cfg, tmp_path / "demand")
    out = tmp_path / "out"
    out.mkdir()
    traci = traci_module()
    traci.start(sumo_command(cfg, 101, rou, add, out / "trip.xml", out / "stops.xml", False))
    try:
        logic = traci.trafficlight.getAllProgramLogics(cfg["network"]["tls_id"])[0]
        states = [p.state for p in logic.phases]
        stages = [Stage(s["name"], s["green_phase"], tuple(s["clearance_phases"]))
                  for s in cfg["network"]["stages"]]  # fmt: skip
        names = [s.name for s in stages]
        det = EmergencyDetector.from_config(traci, cfg, stages, states)
        manager = PreemptionManager.from_config(cfg)
        traci.route.add("amb_ns", ["n_in", "s_out"])
        traci.route.add("amb_ws", ["w_in", "s_out"])
        seen: dict[str, list] = {}
        roads: dict[str, str] = {}  # where amb_a was when its call ended
        begin = float(cfg["begin_s"])
        for _ in range(int(cfg["duration_s"])):
            t = traci.simulation.getTime() - begin
            if t == 30:
                traci.vehicle.add("amb_a", "amb_ns", typeID="ambulance", departLane="best")
            if t == 100:
                for k in range(4):
                    traci.vehicle.add(f"amb_p{k}", "amb_ns", typeID="ambulance",
                                      departLane="best", departPos="0")  # fmt: skip
            if t == 150:
                traci.vehicle.add("amb_w", "amb_ws", typeID="ambulance", departLane="best")
            traci.simulationStep()
            t = traci.simulation.getTime() - begin
            calls = det.calls(t)
            manager.update(t, calls)
            for c in calls:
                seen.setdefault(c.vid, []).append((t, c))
            if "amb_a" in seen and "amb_a" not in roads and seen["amb_a"][-1][0] < t:
                # amb_a's call just ended: it must be past the stop line
                present = "amb_a" in traci.vehicle.getIDList()
                roads["amb_a"] = traci.vehicle.getRoadID("amb_a") if present else "arrived"
    finally:
        traci.close()

    assert {"amb_a", "amb_p0", "amb_p1", "amb_p2", "amb_p3", "amb_w"} <= set(seen)
    assert set(seen) <= {"amb_a", "amb_p0", "amb_p1", "amb_p2", "amb_p3", "amb_w"}  # no car
    a = [c for _, c in seen["amb_a"]]
    assert {(c.link, names[c.stage], c.approach) for c in a} == {(2, "HOP", "N")}
    assert all(not c.pending and 0 <= c.dist_m <= 210 for c in a)
    assert a[0].dist_m > a[-1].dist_m and len({c.first_seen_s for c in a}) == 1
    assert roads["amb_a"].startswith(":C") or roads["amb_a"] in ("s_out", "arrived")
    group = [c for v in ("amb_p0", "amb_p1", "amb_p2", "amb_p3") for _, c in seen[v]]
    pending = [c for c in group if c.pending]
    assert pending and all(math.isinf(c.dist_m) for c in pending)
    assert all(c.link == 2 and names[c.stage] == "HOP" for c in group)
    for v in ("amb_p1", "amb_p2", "amb_p3"):  # first seen pending, kept once inserted
        assert len({c.first_seen_s for _, c in seen[v]}) == 1
    w = [c for _, c in seen["amb_w"]]
    assert {(c.link, names[c.stage], c.approach) for c in w} == {(10, "STM", "W")}
    rep = manager.report()  # the signal runs its static program here (no preemption)
    assert rep["requests"] == 6 and rep["served"] >= 5
    assert rep["served"] + rep["timeouts"] + len(manager.open_requests) == 6
    served = {e.detail.split(",")[0][4:] for e in manager.events if e.rule == "served"}
    assert {"amb_a", "amb_p0", "amb_p1", "amb_p2", "amb_p3"} <= served

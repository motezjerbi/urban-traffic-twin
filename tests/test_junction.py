"""Junction helpers and the JunctionMonitor (box, exits, crossings) against a scripted
fake TraCI: every count of plan section 4.6 on a hand-made step sequence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traffic_twin.sim.junction import (
    JunctionMonitor,
    box_lanes,
    edge_links,
    lane_edge,
    lane_exits,
    lane_links,
)

# Links: 0 a_0 -> x_0 (split by an internal junction: :C_0_0 then :C_4_0), 1 b_0 -> y_0,
# 2 the crossing X (from walking area :C_w0_0), 3 a_0 -> y_0 (no via lane listed)
LINKS = [
    [("a_0", "x_0", ":C_0_0")],
    [("b_0", "y_0", ":C_1_0")],
    [(":C_w0_0", ":C_c0_0", "")],
    [("a_0", "y_0", "")],
]
# phase 0: link 0 and the crossing green; 1: yellow; 2: link 1 green (the crossing's foe)
STATES = ["GrGr", "yrrr", "rGrG"]
FOES = {0: frozenset({1}), 2: frozenset({1})}  # one direction only: made symmetric
CROSSINGS = {"X": {"edge": ":C_c0", "link": 2}}
# lane.getLinks tuples: (approached lane, prio, open, foe, approached internal, ...)
LANE_LINKS = {":C_0_0": [("x_0", True, True, True, ":C_4_0", "M", "r", 5.0)],
              ":C_4_0": [("x_0", True, True, False, "", "M", "r", 5.0)],
              ":C_1_0": [("y_0", True, True, False, "", "M", "s", 9.0)]}  # fmt: skip


class Fake:
    """A TraCI stand-in whose per-lane vehicles and persons are set step by step."""

    def __init__(self):
        self.on_lane: dict[str, list[str]] = {}
        self.speed: dict[str, float] = {}
        self.pos: dict[str, float] = {}
        self.road: dict[str, str] = {}
        self.trafficlight = SimpleNamespace(getControlledLinks=lambda tls: LINKS)
        self.lane = SimpleNamespace(
            getLinks=lambda lane: LANE_LINKS.get(lane, []),
            getLastStepVehicleIDs=lambda lane: list(self.on_lane.get(lane, [])),
        )
        self.vehicle = SimpleNamespace(
            getSpeed=lambda v: self.speed[v], getLanePosition=lambda v: self.pos[v]
        )
        self.person = SimpleNamespace(
            getIDList=lambda: list(self.road), getRoadID=lambda p: self.road[p]
        )


@pytest.fixture
def fake() -> Fake:
    return Fake()


def test_lane_helpers(fake):
    assert lane_edge(":C_c0_0") == ":C_c0" and lane_edge("n_in_12") == "n_in"
    with pytest.raises(ValueError):
        lane_edge("no_lane")
    links = lane_links(fake, "C")
    assert lane_exits(links) == {"a_0": frozenset({"x_0", "y_0"}), "b_0": frozenset({"y_0"})}
    assert edge_links(fake, "C")[("a", "y")] == 3
    vehicle = {i: lk for i, lk in links.items() if i != 2}
    assert box_lanes(fake, vehicle) == {":C_0_0": 0, ":C_4_0": 0, ":C_1_0": 1}


def test_monitor_topology(fake):
    mon = JunctionMonitor(fake, "C", STATES, CROSSINGS, FOES)
    assert mon.foes == {0: {1}, 1: {0, 2}, 2: {1}}
    assert mon.box_lanes == {":C_0_0": 0, ":C_4_0": 0, ":C_1_0": 1}  # no crossing lane
    assert mon.exit_lanes == ("x_0", "y_0")
    assert mon.feeders == {"x_0": (0,), "y_0": (1, 3)}
    assert mon.cross_foes == {":C_c0": (1,)}  # vehicle foes only


def test_monitor_counts_box_exit_and_crossing_events(fake):
    mon = JunctionMonitor(fake, "C", STATES, CROSSINGS, FOES)
    # step 1, phase 0: p0 and p1 enter the green crossing; v1 halts in the box on
    # link 0's second half (green); e1 halts 5 m into exit x_0 (feeder 0 green)
    fake.road = {"p0": ":C_c0", "p1": ":C_c0", "p9": ":C_w0"}
    fake.on_lane = {":C_4_0": ["v1"], "x_0": ["e1"], "y_0": ["e2"]}
    fake.speed = {"v1": 0.0, "e1": 0.2, "e2": 0.0}
    fake.pos = {"e1": 5.0, "e2": 20.0}  # e2 halted beyond the 15 m zone: not blocking
    mon.after_step(0, 1.0)
    assert mon.blocked_exits() == frozenset({"x_0"})
    # step 2, phase 1 (yellow): p1 has left; v1 still halted, its link not green
    fake.road = {"p0": ":C_c0"}
    mon.after_step(1, 1.0)
    assert mon.blocked_exits() == frozenset({"x_0"})
    # step 3, phase 2: link 1 turns green, a foe of v1's link 0 and of the crossing:
    # p0 exposed; e1 has moved on
    fake.on_lane = {":C_4_0": ["v1"], "x_0": ["e1"]}
    fake.speed["e1"] = 3.0
    mon.after_step(2, 1.0)
    assert mon.blocked_exits() == frozenset()
    # step 4, phase 2: p2 enters while the crossing shows r (exposed too); v1 leaves
    fake.road = {"p2": ":C_c0"}
    fake.on_lane = {}
    mon.after_step(2, 1.0)
    assert (mon.traversals, mon.exposed, mon.red_entries) == (3, 2, 1)
    assert mon.summary() == {
        "crossing_complete_rate": pytest.approx(1 / 3),
        "crossing_red_entries": 1.0,
        "box_halt_red_s": 2.0,  # steps 2 and 3
        "box_foe_onsets": 1.0,  # step 3
        "exit_blocked_s": 2.0,  # steps 1 and 2
        "exit_blocked_green_s": 1.0,  # step 1 (feeder 0 yellow in step 2)
    }


def test_a_new_crossing_edge_is_a_new_traversal(fake):
    crossings = {**CROSSINGS, "Y": {"edge": ":C_c1", "link": 3}}
    mon = JunctionMonitor(fake, "C", STATES, crossings, FOES)
    fake.road = {"p0": ":C_c0"}
    mon.after_step(0, 1.0)
    mon.after_step(0, 1.0)  # still on X: same traversal
    fake.road = {"p0": ":C_c1"}  # straight onto Y
    mon.after_step(0, 1.0)
    fake.road = {}
    mon.after_step(0, 1.0)
    fake.road = {"p0": ":C_c0"}  # back on X later: a third one
    mon.after_step(0, 1.0)
    assert mon.traversals == 3 and mon.red_entries == 1  # link 3 is r in phase 0


def test_halted_box_vehicle_on_a_green_link_and_moving_vehicles(fake):
    mon = JunctionMonitor(fake, "C", STATES, CROSSINGS, FOES, box_halt_mps=0.1)
    fake.on_lane = {":C_1_0": ["v1", "v2"]}
    fake.speed = {"v1": 0.05, "v2": 0.1}  # v2 at the threshold: moving
    mon.after_step(2, 1.0)  # link 1 green: halted but not red
    mon.after_step(0, 2.0)  # link 1 red for 2 s; link 0 onset is a foe: one onset
    assert mon.summary()["box_halt_red_s"] == 2.0
    assert mon.summary()["box_foe_onsets"] == 1.0


def test_summary_without_traversals_and_reset(fake):
    mon = JunctionMonitor(fake, "C", STATES, CROSSINGS, FOES)
    mon.after_step(0, 1.0)
    assert "crossing_complete_rate" not in mon.summary()
    assert mon.summary()["crossing_red_entries"] == 0.0
    fake.road = {"p0": ":C_c0"}
    mon.after_step(2, 1.0)
    mon.reset()
    assert mon.traversals == 0 and mon.summary()["crossing_red_entries"] == 0.0

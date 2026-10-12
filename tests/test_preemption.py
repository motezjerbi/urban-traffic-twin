"""Emergency preemption logic (Part 4, D24): `next_hop` (Dijkstra over the link-safe
stage graph) and `PreemptionManager` (FCFS requests, served / timeout bookkeeping).
Pure Python, no SUMO."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from traffic_twin.config import ConfigError, load_scenario
from traffic_twin.control.base import EmergencyCall
from traffic_twin.control.preemption import EVENT_RULES, PreemptionManager, next_hop
from traffic_twin.control.safety import safe_transitions
from traffic_twin.paths import CONFIG_DIR, PITIE_FIXED_CONFIG, PITIE_RULES_CONFIG
from traffic_twin.sim.signal import SignalPlan

REPO_ROOT = Path(__file__).resolve().parent.parent


# -- next_hop on the Pitié program -----------------------------------------------------
@pytest.fixture(scope="module")
def pitie_graph(pitie_scenario):
    """(stage names, link-safe graph, clearance per stage, min green) of the Pitié
    program, read from the committed net."""
    net = pitie_scenario["network"]
    root = ET.parse(net["net_file"]).getroot()
    tl = next(t for t in root.iter("tlLogic") if t.get("id") == net["tls_id"])
    phases = list(tl.iter("phase"))
    durations = [float(p.get("duration")) for p in phases]
    states = [p.get("state") for p in phases]
    min_green = float(pitie_scenario["safety"]["min_green_s"])
    plan = SignalPlan.from_config(net["stages"], durations, min_green)
    graph = safe_transitions(plan.stages, states)
    clearance = [sum(durations[p] for p in st.clearance_phases) for st in plan.stages]
    return [st.name for st in plan.stages], graph, clearance, min_green


def test_pitie_link_safe_graph_is_the_one_the_plan_assumes(pitie_graph):
    names, graph, _, _ = pitie_graph
    idx = {n: k for k, n in enumerate(names)}
    named = {names[a]: {names[b] for b in graph[a]} for a in range(len(names))}
    assert named == {
        "HOP": {"S_LEFT"},
        "S_LEFT": {"HOP", "STM"},
        "STM": {"BUS_91"},
        "BUS_91": {"BUS_57"},
        "BUS_57": {"HOP", "S_LEFT", "STM", "BUS_91"},
    }
    assert idx["HOP"] == 0


@pytest.mark.parametrize(
    ("current", "expected"),
    [("HOP", "HOP"), ("S_LEFT", "HOP"), ("STM", "BUS_91"), ("BUS_91", "BUS_57"),
     ("BUS_57", "HOP")],
)  # fmt: skip
def test_next_hop_towards_hop_on_the_pitie_program(pitie_graph, current, expected):
    names, graph, clearance, min_green = pitie_graph
    hop = next_hop(names.index(current), names.index("HOP"), graph, clearance, min_green)
    assert names[hop] == expected


def test_next_hop_follows_the_cheapest_path_to_every_target(pitie_graph):
    names, graph, clearance, min_green = pitie_graph
    n = len(names)
    for s in range(n):
        for target in range(n):
            h = next_hop(s, target, graph, clearance, min_green)
            assert h == s if s == target else h in graph[s]
    # HOP -> STM: S_LEFT is the only way out of HOP, then the link-safe S_LEFT -> STM
    assert names[next_hop(0, names.index("STM"), graph, clearance, min_green)] == "S_LEFT"
    # STM -> S_LEFT: STM -> BUS_91 -> BUS_57 -> S_LEFT (the jump saves HOP)
    assert names[next_hop(names.index("STM"), 1, graph, clearance, min_green)] == "BUS_91"


def test_next_hop_tie_goes_to_the_first_hop_closest_in_cycle_order():
    n = 4
    graph = [frozenset({1, 3}), frozenset({2}), frozenset({0}), frozenset({2})]
    # 0 -> 1 -> 2 and 0 -> 3 -> 2 cost the same: 1 comes first after 0
    assert next_hop(0, 2, graph, [5.0] * n, 6.0) == 1
    # from 2 the order is 3, 0, 1: 2 -> 3 -> 0 beats 2 -> 1 -> 0 on a tie
    graph = [frozenset({1}), frozenset({0}), frozenset({1, 3}), frozenset({0})]
    assert next_hop(2, 0, graph, [5.0] * n, 6.0) == 3
    # ... but not when the path through 1 is cheaper (shorter clearance of 1)
    assert next_hop(2, 0, graph, [5.0, 1.0, 5.0, 5.0], 6.0) == 1


def test_next_hop_takes_the_cheaper_of_two_paths():
    # 0 -> 2 directly would need 2 in graph[0]; here 0 -> 1 -> 2 (cost 2 + 6 + 2) beats
    # 0 -> 3 -> 2 when 3 has a long clearance
    graph = [frozenset({1, 3}), frozenset({2}), frozenset({0}), frozenset({2})]
    assert next_hop(0, 2, graph, [2.0, 2.0, 2.0, 30.0], 6.0) == 1
    assert next_hop(0, 2, graph, [2.0, 30.0, 2.0, 2.0], 6.0) == 3


def test_next_hop_target_equal_to_current_and_unreachable_target():
    graph = [frozenset({1}), frozenset({2}), frozenset({0})]
    assert next_hop(1, 1, graph, [3.0] * 3, 6.0) == 1
    cut = [frozenset({1}), frozenset({0}), frozenset({0})]  # 2 unreachable from 0
    assert next_hop(0, 2, cut, [3.0] * 3, 6.0) == 1  # the successor
    with pytest.raises(ValueError, match="outside"):
        next_hop(0, 3, graph, [3.0] * 3, 6.0)
    with pytest.raises(ValueError, match="one clearance per stage"):
        next_hop(0, 1, graph, [3.0] * 2, 6.0)


# -- PreemptionManager -----------------------------------------------------------------
def call(vid, first_seen, stage=0, dist=50.0, pending=False, link=2) -> EmergencyCall:
    return EmergencyCall(vid, link, stage, dist, first_seen, pending, "N")


def test_fcfs_on_first_detection_then_vid():
    m = PreemptionManager()
    a, b = call("amb_a", 10.0, stage=2, link=10), call("amb_b", 5.0, stage=0)
    assert m.update(10.0, [a, b]) == 0  # b was seen first
    assert m.open_requests == ("amb_b", "amb_a")
    assert m.update(11.0, [a]) == 2  # b passed the stop line: served, a's turn
    assert m.report() == {"requests": 2, "served": 1, "timeouts": 0, "active_s": 2.0}
    tie = PreemptionManager()
    assert tie.update(0.0, [call("v2", 0.0, stage=3), call("v1", 0.0, stage=1)]) == 1


def test_served_after_disappearing_with_events():
    m = PreemptionManager()
    assert m.update(0.0, []) is None
    assert m.update(1.0, [call("ambulance_0", 1.0, dist=120.04)]) == 0
    assert m.update(2.0, [call("ambulance_0", 1.0, dist=80.0)]) == 0
    assert m.update(3.0, []) is None
    assert m.update(4.0, []) is None
    assert m.report() == {"requests": 1, "served": 1, "timeouts": 0, "active_s": 2.0}
    assert [(e.t, e.source, e.rule) for e in m.events] == [
        (1.0, "preemption", "request"),
        (3.0, "preemption", "served"),
    ]
    assert m.events[0].detail == "vid=ambulance_0, link=2, dist=120.0"
    assert m.events[1].detail == "vid=ambulance_0, link=2, dist=80.0"  # last known call
    assert (m.events[0].stage_from, m.events[0].stage_to) == (0, 0)
    assert {e.rule for e in m.events} <= set(EVENT_RULES)


def test_time_out_at_max_hold_is_dropped_and_never_requested_again():
    m = PreemptionManager(max_hold_s=120.0)
    stuck = call("stuck", 0.0)
    for t in range(121):
        assert m.update(float(t), [stuck]) == 0  # t - first_seen <= 120: held
    assert m.update(121.0, [stuck]) is None
    assert m.update(122.0, [stuck, call("next", 122.0, stage=2)]) == 2
    assert m.update(123.0, [call("next", 122.0, stage=2)]) == 2  # stuck gone: not served
    assert m.report() == {"requests": 2, "served": 0, "timeouts": 1, "active_s": 123.0}
    assert [e.rule for e in m.events] == ["request", "timeout", "request"]
    assert m.events[1].t == 121.0


def test_time_out_clock_restarts_on_progress_towards_the_stop_line():
    """Calibration seed 101 (gate round 3): the shield cut the HOP hold while the
    ambulance was still moving up the N queue; it reached the line, waited for HOP to come
    back and passed 134 s after its first detection. A moving ambulance keeps its request;
    only max_hold_s without getting progress_m closer drops it."""
    m = PreemptionManager(max_hold_s=120.0)
    dist = {t: max(200.0 - 3.0 * t, 8.5) for t in range(200)}  # at the line from t=64
    for t in range(134):
        assert m.update(float(t), [call("amb", 0.0, dist=dist[t])]) == 0, t
    assert m.update(134.0, []) is None
    assert m.report() == {"requests": 1, "served": 1, "timeouts": 0, "active_s": 134.0}
    # ... but it is dropped max_hold_s after its last progress when it stays at the line
    m = PreemptionManager(max_hold_s=120.0)
    for t in range(185):
        assert m.update(float(t), [call("amb", 0.0, dist=dist[t])]) == 0, t
    assert m.update(185.0, [call("amb", 0.0, dist=8.5)]) is None  # 185 - 64 > 120
    assert m.report()["timeouts"] == 1 and m.events[-1].t == 185.0


def test_small_gains_add_up_and_jitter_is_not_progress():
    m = PreemptionManager(max_hold_s=10.0, progress_m=1.0)
    # 0.4 m per step: the clock restarts every third step (1.2 m from the anchor)
    for t in range(30):
        assert m.update(float(t), [call("crawl", 0.0, dist=100.0 - 0.4 * t)]) == 0
    jitter = PreemptionManager(max_hold_s=10.0, progress_m=1.0)
    for t in range(11):  # +-0.6 m around 50 m never gets 1 m closer than the anchor
        d = 50.0 + (0.6 if t % 2 else -0.6)
        assert jitter.update(float(t), [call("j", 0.0, dist=d)]) == 0
    assert jitter.update(11.0, [call("j", 0.0, dist=49.5)]) is None
    assert jitter.report()["timeouts"] == 1


def test_pending_vehicle_times_out_from_first_detection_and_insertion_restarts():
    blocked = PreemptionManager(max_hold_s=120.0)
    for t in range(121):
        assert blocked.update(float(t), [call("p", 0.0, dist=math.inf, pending=True)]) == 0
    assert blocked.update(121.0, [call("p", 0.0, dist=math.inf, pending=True)]) is None
    assert blocked.report()["timeouts"] == 1
    m = PreemptionManager(max_hold_s=120.0)
    for t in range(100):
        assert m.update(float(t), [call("p", 0.0, dist=math.inf, pending=True)]) == 0
    for t in range(100, 221):  # inserted at 100 s, then stuck at 207.7 m
        assert m.update(float(t), [call("p", 0.0, dist=207.7)]) == 0
    assert m.update(221.0, [call("p", 0.0, dist=207.7)]) is None
    assert [e.rule for e in m.events] == ["request", "timeout"] and m.events[1].t == 221.0


def test_pending_then_inserted_is_one_request():
    m = PreemptionManager()
    assert m.update(5.0, [call("amb", 5.0, dist=math.inf, pending=True)]) == 0
    assert m.events[0].detail.endswith("dist=inf")
    assert m.update(6.0, [call("amb", 5.0, dist=207.7)]) == 0
    assert m.update(7.0, [call("amb", 5.0, dist=199.0)]) == 0
    assert m.report()["requests"] == 1 and m.report()["served"] == 0
    assert m.open_requests == ("amb",)


def test_active_seconds_use_the_step_length_and_reset_clears():
    m = PreemptionManager(dt=0.5)
    for t in (0.0, 0.5, 1.0):
        m.update(t, [call("a", 0.0)])
    assert m.report()["active_s"] == 1.5
    m.reset()
    assert m.report() == {"requests": 0, "served": 0, "timeouts": 0, "active_s": 0.0}
    assert m.events == [] and m.update(0.0, [call("a", 0.0)]) == 0  # not 'ended' any more


def test_duplicate_calls_count_once():
    m = PreemptionManager()
    assert m.update(0.0, [call("a", 0.0), call("a", 0.0)]) == 0
    assert m.report()["requests"] == 1


def test_manager_from_config(pitie_cfg):
    assert PreemptionManager.from_config(pitie_cfg) is None  # S0: no preemption block
    pitie_cfg["preemption"] = {"enabled": False, "max_hold_s": 60}
    assert PreemptionManager.from_config(pitie_cfg) is None
    pitie_cfg["preemption"] = {"enabled": True, "max_hold_s": 60}
    pitie_cfg["step_length_s"] = 0.5
    m = PreemptionManager.from_config(pitie_cfg)
    assert (m.max_hold_s, m.dt) == (60.0, 0.5)
    pitie_cfg["preemption"] = {"enabled": True}
    default = PreemptionManager.from_config(pitie_cfg)
    assert (default.max_hold_s, default.progress_m) == (120.0, 1.0)
    pitie_cfg["preemption"] = {"enabled": True, "progress_m": 2}
    assert PreemptionManager.from_config(pitie_cfg).progress_m == 2.0
    for bad, match in [
        ({"enabled": True, "respects_ped_max_wait": False}, "A-04"),
        ({"enabled": True, "max_hold_s": 0}, "max_hold_s"),
        ({"enabled": True, "max_hold_s": "long"}, "max_hold_s"),
        ({"enabled": True, "progress_m": 0}, "progress_m"),
        ({"enabled": True, "progress_m": True}, "progress_m"),
        ([True], "mapping"),
    ]:
        pitie_cfg["preemption"] = bad
        with pytest.raises(ConfigError, match=match):
            PreemptionManager.from_config(pitie_cfg)


def test_manager_follows_the_scenario_configs():
    """On in S2 and its probe twin, off in S0, S1 and the S2_noP ablations (D24)."""
    on = [PITIE_RULES_CONFIG, CONFIG_DIR / "pitie_amb_rules.yaml"]
    off = [PITIE_FIXED_CONFIG, CONFIG_DIR / "pitie_actuated.yaml",
           CONFIG_DIR / "pitie_rules_nopreempt.yaml", CONFIG_DIR / "pitie_amb_fixed.yaml",
           CONFIG_DIR / "pitie_amb_rules_nopreempt.yaml"]  # fmt: skip
    for path in on:
        m = PreemptionManager.from_config(load_scenario(REPO_ROOT / path))
        assert m is not None and m.max_hold_s == 120.0, path
    for path in off:
        assert PreemptionManager.from_config(load_scenario(REPO_ROOT / path)) is None, path

"""Signal-sequence audit (A-04, D31) on hand-built phase sequences and the committed
programs."""

import xml.etree.ElementTree as ET

import pytest

from traffic_twin.control.base import Observation
from traffic_twin.control.fixed import FixedTimeController
from traffic_twin.control.safety import safe_transitions
from traffic_twin.sim.audit import AUDIT_KEYS, DANGER_KEYS, INFO_KEYS, audit_phases, phase_steps
from traffic_twin.sim.signal import SignalPlan, Stage

# links: 0 vehicle A, 1 vehicle B, 2 crossing (green with A)
STATES = [
    "GrG",  # 0 green A
    "Grr",  # 1 pedestrian clearance A
    "yrr",  # 2 yellow A
    "rrr",  # 3 all-red A
    "rGr",  # 4 green B
    "ryr",  # 5 yellow B
]
STAGES = [Stage("A", 0, (1, 2, 3)), Stage("B", 4, (5,))]
CYCLE = [0] * 6 + [1] + [2] * 3 + [3] + [4] * 6 + [5] * 3


def audit(shown, **kw):
    kw.setdefault("min_green_s", 6)
    return audit_phases(shown, STATES, STAGES, **kw)


def violations(shown, **kw) -> dict[str, int]:
    return {k: v for k, v in audit(shown, **kw).items() if v}


def test_clean_cycles_have_no_violation():
    result = audit(CYCLE * 3 + [0] * 2)
    assert list(result) == list(AUDIT_KEYS) and set(result.values()) == {0}


def test_vehicle_green_to_red_without_yellow():
    # A's green straight to all-red: link 0 skips its yellow (the crossing's G -> r
    # into A's clearance phase 3 is fine)
    assert violations([0] * 6 + [3, 4, 4]) == {"veh_green_to_red_without_yellow": 1}
    # permissive green counts too
    states = ["gr", "rG", "ry"]
    stages = [Stage("A", 0, ()), Stage("B", 1, (2,))]
    assert (
        audit_phases([0] * 6 + [1] * 6, states, stages, 6, ped_links=[])[
            "veh_green_to_red_without_yellow"
        ]
        == 1
    )


def test_crossing_green_to_red_only_into_the_clearance_of_the_stage_left():
    assert violations([0] * 6 + [1, 2, 2, 2, 3, 4]) == {}
    # straight into B's yellow: link 0 skips its yellow, and the crossing ends in a
    # clearance phase that belongs to B, not to A
    assert violations([0] * 6 + [5, 5, 5, 3]) == {
        "veh_green_to_red_without_yellow": 1,
        "ped_green_to_red_outside_clearance": 1,
    }


def test_crossings_default_to_links_never_yellow_or_permissive():
    shown = [0] * 6 + [1, 2, 2, 2, 3, 4]
    assert violations(shown, ped_links=None) == {}
    # treated as a vehicle link, the crossing would have needed a yellow
    assert violations(shown, ped_links=[]) == {"veh_green_to_red_without_yellow": 1}


def test_yellow_to_green():
    # back from A's yellow to A's green: link 0 turns yellow to green, A twice in a row
    assert violations([0] * 6 + [1, 2, 0, 0]) == {"yellow_to_green": 1, "out_of_order_green": 1}


def test_short_green_but_not_the_last_cut_one():
    # A lasts 3 s; the final green B is cut by the end of the run
    assert violations([0] * 3 + [1, 2, 2, 2, 3, 4]) == {"green_shorter_than_min": 1}
    assert violations(CYCLE[:-3]) == {}


def test_short_green_uses_the_step_length():
    half = [0] * 12 + [1] * 2 + [2] * 6 + [3] * 2 + [4]  # 0.5 s steps: 6 s of green A
    assert violations(half, step_s=0.5) == {}
    assert violations(half[1:], step_s=0.5) == {"green_shorter_than_min": 1}


def test_out_of_order_green():
    states = ["Grr", "yrr", "rGr", "ryr", "rrG", "rry"]
    stages = [Stage("A", 0, (1,)), Stage("B", 2, (3,)), Stage("C", 4, (5,))]
    ok = [0] * 6 + [1] + [2] * 6 + [3] + [4] * 6 + [5] + [0]
    skip = [0] * 6 + [1] + [4] * 6 + [5] + [2] * 6 + [3] + [0]  # A -> C -> B -> A
    assert set(audit_phases(ok, states, stages, 6).values()) == {0}
    assert audit_phases(skip, states, stages, 6)["out_of_order_green"] == 3


def test_empty_sequence():
    assert set(audit([]).values()) == {0}


def test_fixed_time_on_the_grid_program_is_clean(base_scenario):
    """The grid program, cycled by the fixed controller through SignalPlan: crossings go
    G -> r into EW's pedestrian-clearance phase, lefts stay g through the clearance."""
    root = ET.parse(base_scenario["network"]["net_file"]).getroot()
    tl = next(t for t in root.iter("tlLogic") if t.get("id") == "C")
    durations = [float(p.get("duration")) for p in tl.iter("phase")]
    states = [p.get("state") for p in tl.iter("phase")]
    plan = SignalPlan.from_config(base_scenario["network"]["stages"], durations, 6)
    ctrl = FixedTimeController.from_config(base_scenario)
    shown = []
    for t in range(1000):
        obs = Observation(t, plan.stage, plan.time_in_stage, plan.in_transition)
        shown.append(plan.request(ctrl.decide(obs)))
        plan.advance(1.0)
    assert set(audit_phases(shown, states, plan.stages, 6).values()) == {0}
    allowed = safe_transitions(plan.stages, states)  # Part 4 keys with their inputs
    full = audit_phases(shown, states, plan.stages, 6, durations=durations, allowed=allowed)
    assert set(full.values()) == {0}
    # a crossing cut without its clearance is caught: EW green straight to EW_LEFT
    cut = [0] * 22 + [3] * 10
    result = audit_phases(cut, states, plan.stages, 6)
    assert result["ped_green_to_red_outside_clearance"] == 2  # links 14, 16
    assert result["veh_green_to_red_without_yellow"] == 6  # throughs 4, 5, 11, 12 + rights 3, 10
    assert result["out_of_order_green"] == 0


@pytest.mark.parametrize("shown", [[0], [0, 0, 0]])
def test_single_green_is_never_complete(shown):
    assert set(audit(shown).values()) == {0}


# -- Part 4 keys (D31) ------------------------------------------------------------------
def seq(*parts: tuple[int, int]) -> list[int]:
    """[(phase, steps), ...] -> per-step phases."""
    return [p for p, n in parts for _ in range(n)]


def test_danger_and_info_keys():
    assert INFO_KEYS == ("out_of_order_green",)
    assert DANGER_KEYS == tuple(k for k in AUDIT_KEYS if k != "out_of_order_green")
    assert len(AUDIT_KEYS) == 10 and AUDIT_KEYS[-5:] == (
        "clearance_incomplete", "intergreen_short", "foe_vehicle_greens",
        "illegal_transition", "unknown_phase",
    )  # fmt: skip


def test_phase_steps_follow_signal_plan():
    assert [phase_steps(d, 1.0) for d in (3.0, 1.0, 1.5, 0.0)] == [3, 1, 2, 1]
    assert phase_steps(1.0, 0.5) == 2 and phase_steps(0.2, 0.5) == 1


DURATIONS = [6.0, 1.0, 3.0, 1.0, 6.0, 3.0]  # of STATES: clearances A 1 + 3 + 1 s, B 3 s


def test_clearance_incomplete_on_hand_built_sequences():
    full = audit(CYCLE * 2 + [0], durations=DURATIONS)
    assert set(full.values()) == {0}
    # A's yellow cut to 2 s, then B's yellow to 1 s: two incomplete clearances
    cut = [0] * 6 + [1] + [2] * 2 + [3] + [4] * 6 + [5] + [0]
    assert violations(cut, durations=DURATIONS) == {"clearance_incomplete": 2}
    # a clearance phase shown too long is incomplete too (another program)
    assert violations([0] * 6 + [1] + [2] * 4 + [3] + [4], durations=DURATIONS) == {
        "clearance_incomplete": 1
    }
    # a trailing transition cut by the end of the run is not checked
    assert violations([0] * 6 + [1, 2], durations=DURATIONS) == {}


def test_clearance_uses_the_step_length():
    half = [0] * 12 + [1] * 2 + [2] * 6 + [3] * 2 + [4]  # 0.5 s steps
    assert violations(half, step_s=0.5, durations=DURATIONS) == {}
    assert violations(half[:14] + half[15:], step_s=0.5, durations=DURATIONS) == {
        "clearance_incomplete": 1
    }


def test_foe_vehicle_greens_once_per_phase_shown():
    states = [*STATES, "GGr", "GrG"]  # 6: A and B together, 7: A with its crossing
    foes = {0: frozenset({1, 2}), 1: frozenset({0}), 2: frozenset({0})}
    shown = [0] * 6 + [6] * 4 + [0] * 2 + [6] * 3 + [7] * 2
    result = audit_phases(shown, states, STAGES, 6, foes=foes)
    assert result["foe_vehicle_greens"] == 2  # the crossing (link 2) is not a vehicle link
    assert audit_phases(shown, states, STAGES, 6)["foe_vehicle_greens"] == 0  # no input


def test_illegal_transition_and_unknown_phase():
    allowed = [frozenset({1}), frozenset({0})]
    assert violations(CYCLE * 2, allowed=allowed) == {}
    # A twice in a row (through its clearance): not in allowed[A]
    shown = [0] * 6 + [1] + [2] * 3 + [3] + [0] * 6
    assert violations(shown, allowed=allowed) == {"illegal_transition": 1, "out_of_order_green": 1}
    unknown = [0] * 6 + [9] * 2 + [0] * 2 + [-1]
    assert violations(unknown) == {"unknown_phase": 2}


@pytest.fixture(scope="module")
def pitie(pitie_scenario):
    """(stages, durations, states, allowed, requirements, crossing links) of Pitié."""
    from traffic_twin.control.safety import safe_transitions
    from traffic_twin.sim.tls_check import cached_requirements, load_program, scenario_stages

    net = pitie_scenario["network"]["net_file"]
    prog = load_program(net, "C")
    stages = scenario_stages(pitie_scenario)
    allowed = safe_transitions(stages, list(prog.states))
    req = cached_requirements(net, pitie_scenario)
    links = sorted(int(c["link"]) for c in pitie_scenario["network"]["crossings"].values())
    return stages, list(prog.durations), list(prog.states), allowed, req, links


def pitie_audit(pitie, shown: list[int]) -> dict[str, int]:
    stages, durations, states, allowed, req, links = pitie
    result = audit_phases(
        shown, states, stages, 6, ped_links=links, durations=durations, allowed=allowed,
        intergreen=req.intergreen, foes=req.foes,
    )  # fmt: skip
    return {k: v for k, v in result.items() if v}


# Pitié phases: HOP 0 | 1, 2, 3 | S_LEFT 4 | 5 (yellow 3 s), 6 (all-red 7 s) | STM 7 | ...
HOP_TO_STM = ((0, 36), (1, 1), (2, 3), (3, 5), (4, 6), (5, 3), (6, 7), (7, 20))


def test_pitie_fixed_time_cycles_are_clean_on_every_key(pitie, pitie_scenario):
    stages, durations, states, *_ = pitie
    plan = SignalPlan(stages, durations, 6)
    ctrl = FixedTimeController.from_config(pitie_scenario)
    shown = []
    for t in range(3 * 123 + 10):
        obs = Observation(t, plan.stage, plan.time_in_stage, plan.in_transition)
        shown.append(plan.request(ctrl.decide(obs)))
        plan.advance(1.0)
    assert pitie_audit(pitie, shown) == {}
    assert pitie_audit(pitie, seq(*HOP_TO_STM)) == {}


def test_pitie_cut_yellow_and_all_red_are_caught(pitie):
    yellow_1s = list(HOP_TO_STM)
    yellow_1s[5] = (5, 1)
    found = pitie_audit(pitie, seq(*yellow_1s))
    assert found["clearance_incomplete"] == 1 and found["intergreen_short"] > 0
    all_red_2s = list(HOP_TO_STM)
    all_red_2s[6] = (6, 2)
    found = pitie_audit(pitie, seq(*all_red_2s))
    assert found["clearance_incomplete"] == 1 and found["intergreen_short"] > 0
    assert set(found) == {"clearance_incomplete", "intergreen_short"}


def test_pitie_illegal_jump_and_legal_skip(pitie):
    # HOP -> STM through HOP's full clearance: not link-safe (S_LEFT skipped)
    found = pitie_audit(pitie, seq((0, 36), (1, 1), (2, 3), (3, 5), (7, 20)))
    assert found["illegal_transition"] == 1 and found["out_of_order_green"] == 1
    # S_LEFT -> HOP is link-safe: only the information key
    legal = seq((4, 6), (5, 3), (6, 7), (0, 10))
    assert pitie_audit(pitie, legal) == {"out_of_order_green": 1}
    # HOP straight to STM: every key that sees it
    found = pitie_audit(pitie, seq((0, 36), (7, 20)))
    assert {"veh_green_to_red_without_yellow", "clearance_incomplete", "intergreen_short",
            "illegal_transition"} <= set(found)  # fmt: skip

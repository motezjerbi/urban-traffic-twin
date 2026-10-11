"""Signal-sequence audit on hand-built phase sequences."""

import xml.etree.ElementTree as ET

import pytest

from traffic_twin.control.base import Observation
from traffic_twin.control.fixed import FixedTimeController
from traffic_twin.sim.audit import AUDIT_KEYS, audit_phases
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
    # a crossing cut without its clearance is caught: EW green straight to EW_LEFT
    cut = [0] * 22 + [3] * 10
    result = audit_phases(cut, states, plan.stages, 6)
    assert result["ped_green_to_red_outside_clearance"] == 2  # links 14, 16
    assert result["veh_green_to_red_without_yellow"] == 6  # throughs 4, 5, 11, 12 + rights 3, 10
    assert result["out_of_order_green"] == 0


@pytest.mark.parametrize("shown", [[0], [0, 0, 0]])
def test_single_green_is_never_complete(shown):
    assert set(audit(shown).values()) == {0}

import pytest

from traffic_twin.control.base import Observation
from traffic_twin.control.fixed import FixedTimeController
from traffic_twin.sim.signal import SignalPlan, Stage

# phases: 0 green A, 1 yellow A, 2 all-red, 3 green B, 4 yellow B
DURATIONS = [30, 3, 2, 30, 3]
STAGES = [Stage("A", 0, (1, 2)), Stage("B", 3, (4,))]


def test_clearance_runs_for_its_full_duration():
    plan = SignalPlan(STAGES, DURATIONS)
    shown = []
    for i in range(10):
        shown.append(plan.request(1 if i >= 2 else 0))
        plan.advance(1.0)
    # 2 s of green, 3 s yellow, 2 s all-red, then green B
    assert shown == [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
    assert plan.stage == 1 and plan.time_in_stage == 3


def test_requests_during_transition_are_ignored():
    plan = SignalPlan(STAGES, DURATIONS)
    plan.request(1)
    plan.advance(1.0)
    assert plan.request(0) == 1  # still clearing towards B
    assert plan.in_transition


def test_unknown_stage_rejected():
    with pytest.raises(ValueError):
        SignalPlan(STAGES, DURATIONS).request(5)


def test_fixed_time_cycle():
    ctrl = FixedTimeController([10, 5])
    plan = SignalPlan(STAGES, DURATIONS)
    greens = []
    for t in range(60):
        obs = Observation(t, plan.stage, plan.time_in_stage, plan.in_transition)
        phase = plan.request(ctrl.decide(obs))
        if phase in (0, 3):
            greens.append(phase)
        plan.advance(1.0)
    # 10 s green A, 5 clearance, 5 s green B, 3 clearance, ... cycle of 23 s
    assert greens[:10] == [0] * 10 and greens[10:15] == [3] * 5
    assert greens.count(0) >= 20


def test_fixed_rejects_bad_greens():
    with pytest.raises(ValueError):
        FixedTimeController([10, 0])

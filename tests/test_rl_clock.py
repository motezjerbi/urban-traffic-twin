"""The RL decision clock (Part 6, D56): decision points on synthetic observations (min
green, transition, preemption, 5 s spacing) and the action decoding (D55)."""

import pytest

from traffic_twin.control.base import EXTRA_PREEMPT, Observation
from traffic_twin.rl.clock import DecisionClock

N_STAGES = 5


def obs(t, stage=0, tis=10.0, transition=False, preempt=None) -> Observation:
    extras = {EXTRA_PREEMPT: preempt} if preempt is not None else {}
    return Observation(t, stage, tis, transition, {}, extras)


@pytest.fixture
def clock() -> DecisionClock:
    return DecisionClock(interval_s=5, min_green_s=6)


def test_no_decision_before_min_green(clock):
    assert not clock.is_decision_point(obs(3, tis=5.0))
    assert not clock.is_decision_point(obs(3, tis=5.999))
    assert clock.is_decision_point(obs(4, tis=6.0))


def test_no_decision_during_transition(clock):
    assert not clock.is_decision_point(obs(10, tis=30.0, transition=True))


def test_no_decision_during_preemption_unless_configured(clock):
    assert not clock.is_decision_point(obs(10, preempt=2))
    assert clock.is_decision_point(obs(10, preempt=None))
    asked = DecisionClock(5, 6, skip_during_preemption=False)
    assert asked.is_decision_point(obs(10, preempt=2))
    # preemption target 0 is a real stage, not "no preemption"
    assert not clock.is_decision_point(obs(10, preempt=0))


def test_decisions_are_spaced_by_the_interval(clock):
    assert clock.request(obs(10), 0, N_STAGES) == 0  # decision at t = 10
    assert not clock.is_decision_point(obs(11))
    assert not clock.is_decision_point(obs(14.5))
    assert clock.is_decision_point(obs(15))
    clock.reset()
    assert clock.is_decision_point(obs(11))


def test_greens_fall_on_six_plus_five_k(clock):
    """Hold until min green, then decide every 5 s: a stage left by the first advance
    after k stays lasts 6 + 5k s."""
    t, tis, stage, greens = 0.0, 0.0, 0, []
    plan = [0, 0, 1, 1, 0, 1]  # actions at successive decision points
    while plan:
        o = obs(t, stage=stage, tis=tis)
        action = plan.pop(0) if clock.is_decision_point(o) else None
        new = clock.request(o, action, N_STAGES)
        if new != stage:  # switch at once (no clearance in this sketch)
            greens.append(tis)
            stage, tis = new, 0.0
        else:
            tis += 1.0
        t += 1.0
    assert greens == [16.0, 6.0, 11.0]  # 6 + 5k


def test_request_decoding(clock):
    assert clock.request(obs(10, stage=1), 1, N_STAGES) == 2  # advance
    assert clock.request(obs(20, stage=1), 0, N_STAGES) == 1  # stay
    assert clock.request(obs(30, stage=4), 1, N_STAGES) == 0  # wraps to the first stage


def test_hold_outside_decision_points(clock):
    # no action, or an action outside a decision point: the current stage, nothing recorded
    assert clock.request(obs(10, stage=3), None, N_STAGES) == 3
    assert clock.request(obs(10, stage=3, transition=True), 1, N_STAGES) == 3
    assert clock.request(obs(10, stage=3, tis=2.0), 1, N_STAGES) == 3
    assert clock.request(obs(10, stage=3, preempt=1), 1, N_STAGES) == 3
    assert clock.is_decision_point(obs(10, stage=3))  # nothing recorded so far
    assert clock.request(obs(10, stage=3), 1, N_STAGES) == 4
    assert clock.request(obs(12, stage=4, tis=8.0), 1, N_STAGES) == 4  # too soon: held


def test_numpy_actions_and_invalid_actions(clock):
    np = pytest.importorskip("numpy")
    assert clock.request(obs(10), np.int64(1), N_STAGES) == 1
    with pytest.raises(ValueError, match="action"):
        clock.request(obs(20), 2, N_STAGES)


def test_bad_parameters():
    with pytest.raises(ValueError):
        DecisionClock(0, 6)
    with pytest.raises(ValueError):
        DecisionClock(5, -1)

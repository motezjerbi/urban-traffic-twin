import dataclasses
import random

import pytest

from traffic_twin.control.base import Observation
from traffic_twin.control.fixed import FixedTimeController
from traffic_twin.sim.signal import SignalPlan, SignalState, Stage

# phases: 0 green A, 1 yellow A, 2 all-red, 3 green B, 4 yellow B
DURATIONS = [30, 3, 2, 30, 3]
STAGES = [Stage("A", 0, (1, 2)), Stage("B", 3, (4,))]


def run_plan(plan: SignalPlan, requests, dt: float = 1.0) -> list[int]:
    """Phase shown at each step for a sequence of stage requests."""
    shown = []
    for r in requests:
        shown.append(plan.request(r))
        plan.advance(dt)
    return shown


def runs(seq: list[int]) -> list[tuple[int, int]]:
    """Run-length encoding: [(phase, steps), ...]."""
    out: list[tuple[int, int]] = []
    for p in seq:
        if out and out[-1][0] == p:
            out[-1] = (p, out[-1][1] + 1)
        else:
            out.append((p, 1))
    return out


def test_clearance_runs_for_its_full_duration():
    plan = SignalPlan(STAGES, DURATIONS)
    shown = run_plan(plan, [0, 0] + [1] * 8)
    # 2 s of green, 3 s yellow, 2 s all-red, then green B
    assert shown == [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
    assert plan.stage == 1 and plan.time_in_stage == 3


def test_min_green_holds_early_switch_requests():
    plan = SignalPlan(STAGES, DURATIONS, min_green_s=4)
    shown = run_plan(plan, [1] * 10)
    # held 4 s on green A, then 3 s yellow, 2 s all-red, then green B
    assert shown == [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]


def test_min_green_applies_after_each_transition():
    plan = SignalPlan(STAGES, DURATIONS, min_green_s=5)
    # B is reached at step 10; switching straight back is held for 5 s of green B
    shown = run_plan(plan, [1] * 10 + [0] * 10)
    assert runs(shown) == [(0, 5), (1, 3), (2, 2), (3, 5), (4, 3), (0, 2)]


def test_from_config_reads_min_green(base_scenario):
    plan = SignalPlan.from_config(base_scenario["network"]["stages"], [5] * 40, min_green_s=6)
    assert plan.min_green_s == 6


def test_requests_during_transition_are_ignored():
    plan = SignalPlan(STAGES, DURATIONS)
    plan.request(1)
    plan.advance(1.0)
    assert plan.request(0) == 1  # still clearing towards B
    assert plan.in_transition


def test_requesting_current_stage_keeps_green():
    plan = SignalPlan(STAGES, DURATIONS)
    assert run_plan(plan, [0] * 50) == [0] * 50
    assert not plan.in_transition and plan.time_in_stage == 50


def test_unknown_stage_rejected():
    plan = SignalPlan(STAGES, DURATIONS)
    for bad in (5, -1, len(STAGES)):
        with pytest.raises(ValueError, match="unknown stage"):
            plan.request(bad)


def test_stage_without_clearance_switches_at_once():
    plan = SignalPlan([Stage("A", 0, ()), Stage("B", 1, ())], [10, 10])
    assert plan.request(1) == 1
    assert plan.stage == 1 and not plan.in_transition and plan.time_in_stage == 0


def test_time_in_stage_resets_after_clearance_only():
    plan = SignalPlan(STAGES, DURATIONS)
    run_plan(plan, [0] * 7)
    assert plan.time_in_stage == 7
    run_plan(plan, [1] * 5)  # clearance: time_in_stage frozen until green B
    assert plan.stage == 1 and plan.time_in_stage == 0


@pytest.mark.parametrize("dt", [0.5, 1.0])
@pytest.mark.parametrize("seed", range(5))
def test_random_requests_never_skip_clearance(seed, dt):
    """Whatever the controller asks, every change of green runs the full clearance
    sequence of the stage being left, each phase for its full duration, and one green
    is never followed directly by another."""
    rng = random.Random(seed)
    plan = SignalPlan(STAGES, DURATIONS)
    greens = {s.green_phase for s in STAGES}
    shown: list[int] = []
    transitions = 0
    leaving, during = None, []
    for _ in range(2000):
        before = plan.stage
        phase = plan.request(rng.randrange(len(STAGES)))
        if leaving is None and plan.in_transition:
            leaving, during = before, []
        if plan.in_transition:
            during.append(phase)
        shown.append(phase)
        plan.advance(dt)
        if leaving is not None and not plan.in_transition:
            expected = [
                p for p in STAGES[leaving].clearance_phases for _ in range(round(DURATIONS[p] / dt))
            ]
            assert during == expected
            assert plan.stage != leaving and plan.time_in_stage == 0
            leaving, transitions = None, transitions + 1
    assert transitions > 50
    for a, b in zip(shown, shown[1:], strict=False):
        assert not (a in greens and b in greens and a != b), "green to green without clearance"


@pytest.mark.parametrize("seed", range(3))
def test_green_never_shown_during_transition(seed):
    rng = random.Random(seed)
    plan = SignalPlan(STAGES, DURATIONS)
    green_phases = {s.green_phase for s in STAGES}
    for _ in range(500):
        phase = plan.request(rng.randrange(len(STAGES)))
        assert (phase in green_phases) != plan.in_transition
        plan.advance(1.0)


def test_from_config_matches_scenario(base_scenario):
    stages_cfg = base_scenario["network"]["stages"]
    plan = SignalPlan.from_config(stages_cfg, [1.0] * 12)
    assert [s.name for s in plan.stages] == ["EW", "EW_LEFT", "NS", "NS_LEFT"]
    assert plan.stages[0] == Stage("EW", 0, (1, 2))
    assert plan.phase == stages_cfg[0]["green_phase"]


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


def test_fixed_time_greens_last_exactly_green_s():
    """Min green: every complete green of the fixed controller lasts its green time."""
    green_s = [7, 4]
    ctrl = FixedTimeController(green_s)
    plan = SignalPlan(STAGES, DURATIONS)
    shown = []
    for t in range(300):
        obs = Observation(t, plan.stage, plan.time_in_stage, plan.in_transition)
        shown.append(plan.request(ctrl.decide(obs)))
        plan.advance(1.0)
    seq = runs(shown)
    for phase, steps in seq[:-1]:  # last run may be cut by the end of the loop
        if phase == 0:
            assert steps == green_s[0]
        elif phase == 3:
            assert steps == green_s[1]


def test_scenario_greens_respect_safety_min_green(base_scenario):
    min_green = base_scenario["safety"]["min_green_s"]
    assert min(base_scenario["fixed"]["green_s"].values()) >= min_green


@pytest.mark.parametrize("greens", [[10, 0], [], [5, -1]])
def test_fixed_rejects_bad_greens(greens):
    with pytest.raises(ValueError, match="positive"):
        FixedTimeController(greens)


def test_red_s_counts_from_run_start_for_unserved_stages():
    plan = SignalPlan(STAGES, DURATIONS)
    assert plan.red_s == [0.0, 0.0]
    run_plan(plan, [0] * 5)
    assert plan.red_s == [0.0, 5.0]  # A green, B never served


@pytest.mark.parametrize("dt", [0.5, 1.0])
def test_red_s_counts_from_the_end_of_green_and_resets_at_green(dt):
    plan = SignalPlan(STAGES, DURATIONS)
    n = round(1 / dt)  # steps per second
    run_plan(plan, [0] * 3 * n + [1] * n, dt)  # 3 s green A, then 1 s of clearance
    assert plan.in_transition and plan.red_s == pytest.approx([1.0, 4.0])
    run_plan(plan, [1] * 4 * n, dt)  # rest of the 5 s clearance: green B starts
    assert plan.stage == 1 and not plan.in_transition
    assert plan.red_s == pytest.approx([5.0, 0.0])
    run_plan(plan, [1] * 4 * n, dt)
    assert plan.red_s == pytest.approx([9.0, 0.0])


def test_state_is_an_immutable_snapshot():
    plan = SignalPlan(STAGES, DURATIONS)
    run_plan(plan, [0] * 3 + [1])
    state = plan.state(4.0)
    assert state == SignalState(4.0, 0, 3.0, True, (1.0, 4.0))
    run_plan(plan, [1] * 10)  # the plan moves on, the snapshot does not
    assert state.red_s == (1.0, 4.0) and state.stage == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.stage = 1
    assert plan.state(14.0) == SignalState(14.0, 1, 6.0, False, (11.0, 0.0))

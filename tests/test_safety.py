"""SafetyGuard (D20: Part 3 rules, emergency preemption, pedestrian shield) and the
link-safe stage transitions, driven through SignalPlan the way SimSession does: observe,
decide, guard, request, advance."""

import copy
import dataclasses
import math
import random
import xml.etree.ElementTree as ET

import pytest

from traffic_twin.config import ConfigError
from traffic_twin.control.base import EXTRA_GAP_S, Observation
from traffic_twin.control.preemption import next_hop
from traffic_twin.control.registry import make_controller
from traffic_twin.control.safety import (
    INTERVENTION_KEYS,
    PREEMPTION_KEYS,
    SafetyGuard,
    safe_transitions,
    unsafe_links,
)
from traffic_twin.sim.audit import AUDIT_KEYS, audit_phases
from traffic_twin.sim.signal import SignalPlan, SignalState, Stage


def program(cfg: dict) -> tuple[list[float], list[str]]:
    """(durations, states) of the scenario's signal program, read from its net."""
    root = ET.parse(cfg["network"]["net_file"]).getroot()
    tl = next(t for t in root.iter("tlLogic") if t.get("id") == cfg["network"]["tls_id"])
    phases = list(tl.iter("phase"))
    return [float(p.get("duration")) for p in phases], [p.get("state") for p in phases]


def plan_of(cfg: dict) -> SignalPlan:
    durations, _ = program(cfg)
    return SignalPlan.from_config(cfg["network"]["stages"], durations, cfg["safety"]["min_green_s"])


def drive(decide, guard: SafetyGuard, plan: SignalPlan, steps: int, extras=None, dt=1.0):
    """Run `steps` steps; returns (phases shown, max red seen per stage)."""
    shown, max_red = [], [0.0] * len(plan.stages)
    for i in range(steps):
        t = i * dt
        obs = Observation(t, plan.stage, plan.time_in_stage, plan.in_transition)
        if extras is not None:
            obs.extras.update(extras(obs))
        stage = guard.apply(decide(obs), plan.state(t))
        shown.append(plan.request(stage))
        plan.advance(dt)
        max_red = [max(m, r) for m, r in zip(max_red, plan.red_s, strict=True)]
    return shown, max_red


def green_runs(shown: list[int], plan: SignalPlan) -> list[tuple[int, int]]:
    """[(stage, steps)] for every green shown, the last one possibly cut."""
    stage_of = {s.green_phase: k for k, s in enumerate(plan.stages)}
    out: list[tuple[int, int]] = []
    prev = None
    for p in shown:
        if p in stage_of:
            if p == prev:
                out[-1] = (out[-1][0], out[-1][1] + 1)
            else:
                out.append((stage_of[p], 1))
        prev = p
    return out


@pytest.fixture(scope="module")
def grid_states(base_scenario):
    return program(base_scenario)[1]


@pytest.fixture(scope="module")
def grid_stages(base_scenario):
    return plan_of(base_scenario).stages


def state(stage=0, tis=0.0, transition=False, red=(0.0, 0.0, 0.0, 0.0), t=0.0) -> SignalState:
    return SignalState(t, stage, tis, transition, tuple(red))


def guard4(min_green_s=6, max_green_s=None, max_red_s=None, strict_order=True) -> SafetyGuard:
    """4 stages, any transition link-safe."""
    allowed = [frozenset(range(4)) - {s} for s in range(4)]
    return SafetyGuard(4, min_green_s, max_green_s, max_red_s, allowed, strict_order)


# -- safe transitions ------------------------------------------------------------------
def test_grid_safe_transitions_match_the_analyst_matrix(grid_stages, grid_states):
    # EW's clearance keeps the permissive lefts (links 6, 13) green: only EW_LEFT may
    # follow; the same for NS (links 2, 9) -> NS_LEFT. LEFT stages end all-red.
    assert safe_transitions(grid_stages, grid_states) == [
        frozenset({1}),
        frozenset({0, 2, 3}),
        frozenset({3}),
        frozenset({0, 1, 2}),
    ]


def test_unsafe_links_green_to_non_green_and_yellow_to_green():
    assert unsafe_links("GgyrG", "GGrGG") == []  # g -> G keeps green, y -> r, r -> G
    assert unsafe_links("Gg", "rr") == [0, 1]
    assert unsafe_links("y", "G") == [0]
    with pytest.raises(ConfigError, match="lengths"):
        unsafe_links("Gr", "G")


def test_stage_without_clearance_uses_its_green_state():
    states = ["Gr", "rG", "yr"]
    stages = [Stage("A", 0, ()), Stage("B", 1, ()), Stage("C", 2, ())]
    # without clearance A and B cut each other's green (and C's state is not green for
    # their link); C ends on yellow: to A it would turn yellow to green, to B it is fine
    assert safe_transitions(stages, states) == [frozenset(), frozenset(), frozenset({1})]


# -- guard rules -----------------------------------------------------------------------
def test_pass_through_when_no_rule_applies():
    g = guard4(max_green_s=[60] * 4, max_red_s=120)
    assert g.apply(0, state(0, 10)) == 0
    assert g.apply(1, state(0, 10)) == 1
    assert g.interventions == dict.fromkeys(INTERVENTION_KEYS, 0)
    assert list(g.interventions) == list(INTERVENTION_KEYS)  # zeros kept for run.json


def test_min_green_holds_and_counts_once_per_green():
    g = guard4()
    for tis in range(6):
        assert g.apply(1, state(0, tis)) == 0
    assert g.apply(1, state(0, 6)) == 1
    assert g.interventions["min_green"] == 1
    g.apply(0, state(0, 6, transition=True))
    for tis in range(3):  # next green, new event
        assert g.apply(2, state(1, tis)) == 1
    assert g.interventions["min_green"] == 2


def test_order_strict_substitutes_the_successor():
    g = guard4()
    assert g.apply(2, state(0, 8)) == 1
    assert g.apply(3, state(0, 9)) == 1  # same green: same event
    assert g.interventions["order"] == 1
    assert g.apply(0, state(3, 8)) == 0  # 3 -> 0 is the cycle successor


def test_order_free_allows_link_safe_jumps_only(grid_stages, grid_states):
    allowed = safe_transitions(grid_stages, grid_states)
    g = SafetyGuard(4, 6, None, None, allowed, strict_order=False)
    assert g.apply(2, state(0, 8)) == 1  # EW -> NS would cut the lefts: EW_LEFT instead
    assert g.apply(3, state(1, 8)) == 3  # EW_LEFT -> NS_LEFT skips NS: allowed
    assert g.interventions["order"] == 1


def test_max_green_ends_the_green_once():
    g = guard4(max_green_s=[20, 30, 30, 30])
    assert g.apply(0, state(0, 19)) == 0
    assert g.apply(0, state(0, 20)) == 1
    assert g.apply(0, state(0, 21)) == 1
    assert g.interventions["max_green"] == 1
    assert g.apply(1, state(1, 25)) == 1  # 30 s for stage 1


def test_starvation_moves_on_after_min_green():
    g = guard4(max_red_s=100)
    starving = (0.0, 30.0, 100.0, 50.0)
    assert g.apply(0, state(0, 3, red=starving)) == 0  # min green first
    assert g.apply(0, state(0, 6, red=starving)) == 1  # strict: towards 2 via 1
    assert g.apply(0, state(0, 7, red=starving)) == 1
    assert g.apply(1, state(0, 8, red=starving)) == 1  # already heading there: no event
    assert g.interventions["starvation"] == 1


def test_starvation_free_order_takes_the_shortest_legal_path(grid_stages, grid_states):
    allowed = safe_transitions(grid_stages, grid_states)
    g = SafetyGuard(4, 6, None, 100, allowed, strict_order=False)
    # NS (2) starving while EW_LEFT (1) is green: EW_LEFT -> NS is link-safe
    assert g.apply(1, state(1, 6, red=(10.0, 0.0, 120.0, 30.0))) == 2
    # NS starving while EW (0) is green: EW may only go to EW_LEFT first
    assert g.apply(0, state(0, 6, red=(0.0, 20.0, 120.0, 30.0))) == 1
    assert g.interventions["starvation"] == 2


def test_requests_during_clearance_are_left_to_the_plan():
    g = guard4(max_green_s=[6] * 4, max_red_s=10)
    assert g.apply(3, state(0, 50, transition=True, red=(0, 99, 99, 99))) == 0
    assert sum(g.interventions.values()) == 0


def test_unknown_stage_rejected():
    with pytest.raises(ValueError, match="unknown stage"):
        guard4().apply(4, state())


def test_reset_clears_interventions():
    g = guard4()
    g.apply(1, state(0, 1))
    g.reset()
    assert sum(g.interventions.values()) == 0
    g.apply(1, state(0, 2))  # same green as before the reset, counted again
    assert g.interventions["min_green"] == 1


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"allowed": [frozenset({1}), frozenset()]}, "not link-safe"),
        ({"allowed": [frozenset({1})]}, "one allowed set"),
        ({"max_green_s": [30]}, "one max green"),
        ({"max_green_s": [30, 4]}, "below the min green"),
        ({"min_green_s": -1}, ">= 0"),
        ({"max_red_s": 0}, "> 0"),
    ],
)
def test_constructor_checks(kw, match):
    args = {"n_stages": 2, "min_green_s": 6, "max_green_s": None, "max_red_s": None,
            "allowed": [frozenset({1}), frozenset({0})]}  # fmt: skip
    args.update(kw)
    with pytest.raises(ConfigError, match=match):
        SafetyGuard(**args)


# -- from_config -----------------------------------------------------------------------
def test_grid_guard_has_no_caps_and_strict_order(base_scenario, grid_stages, grid_states):
    g = SafetyGuard.from_config(base_scenario, grid_stages, grid_states)
    assert (g.min_green_s, g.max_green_s, g.max_red_s, g.strict_order) == (6, None, None, True)
    assert g.allowed == safe_transitions(grid_stages, grid_states)


def test_max_red_must_exceed_the_longest_legal_cycle(scenario_cfg, grid_stages, grid_states):
    # fixed: 74 s of green + 26 s of clearance = 100 s
    scenario_cfg["safety"]["max_red_s"] = 100
    with pytest.raises(ConfigError, match=r"longest legal cycle, 100 s \(fixed\.green_s"):
        SafetyGuard.from_config(scenario_cfg, grid_stages, grid_states)
    scenario_cfg["safety"]["max_red_s"] = 101
    assert SafetyGuard.from_config(scenario_cfg, grid_stages, grid_states).max_red_s == 101
    # actuated maxima make the cycle longer: 87 + 26 = 113 s
    scenario_cfg["actuated"] = {"max_green_s": {"EW": 40, "EW_LEFT": 12, "NS": 25, "NS_LEFT": 10}}
    with pytest.raises(ConfigError, match=r"113 s \(actuated\.max_green_s"):
        SafetyGuard.from_config(scenario_cfg, grid_stages, grid_states)
    # durations given by the caller instead of the net (all phases 1 s: 8 s of clearance)
    g = SafetyGuard.from_config(scenario_cfg, grid_stages, grid_states, durations=[1.0] * 12)
    assert g.max_red_s == 101


def test_plan_greens_must_fit_under_the_max_green_cap(scenario_cfg, grid_stages, grid_states):
    scenario_cfg["safety"]["max_green_s"] = 30
    with pytest.raises(ConfigError, match=r"fixed\.green_s of stages \['EW'\] exceed"):
        SafetyGuard.from_config(scenario_cfg, grid_stages, grid_states)
    scenario_cfg["safety"]["max_green_s"] = {"EW": 35, "EW_LEFT": 10, "NS": 22, "NS_LEFT": 7}
    g = SafetyGuard.from_config(scenario_cfg, grid_stages, grid_states)
    assert g.max_green_s == [35, 10, 22, 7]
    del scenario_cfg["safety"]["max_green_s"]["NS"]
    with pytest.raises(ConfigError, match=r"safety\.max_green_s has no value for stages \['NS'\]"):
        SafetyGuard.from_config(scenario_cfg, grid_stages, grid_states)


def test_unsafe_cycle_order_is_a_config_error(scenario_cfg, grid_stages, grid_states):
    reordered = [grid_stages[i] for i in (0, 2, 1, 3)]  # EW -> NS cuts the lefts
    with pytest.raises(ConfigError, match=r"EW -> NS is not link-safe \(links \[6, 13\]"):
        SafetyGuard.from_config(scenario_cfg, reordered, grid_states)
    scenario_cfg["safety"]["stage_order"] = "random"
    with pytest.raises(ConfigError, match="stage_order"):
        SafetyGuard.from_config(scenario_cfg, grid_stages, grid_states)


def test_pitie_guard_from_config(pitie_scenario):
    _, states = program(pitie_scenario)
    plan = plan_of(pitie_scenario)
    g = SafetyGuard.from_config(pitie_scenario, plan.stages, states)
    assert g.max_green_s == [60.0] * 5 and g.max_red_s == 150 and g.strict_order
    assert all((k + 1) % 5 in g.allowed[k] for k in range(5))
    # Part 4: the detected pedestrian shield and the clearances (read from the net)
    assert g.shield is not None and g.shield.mode == "detected"
    assert g.clearance_s == [9, 10, 7, 12, 11] and g.dt == 1.0


# -- properties with SignalPlan --------------------------------------------------------
class Random:
    def __init__(self, seed: int, n: int):
        self.rng, self.n = random.Random(seed), n

    def __call__(self, obs: Observation) -> int:
        return self.rng.randrange(self.n)


def sticky(obs: Observation) -> int:
    return 0


def hold(obs: Observation) -> int:
    return obs.stage


@pytest.mark.parametrize(
    "controller, max_green, max_red",
    [
        ("random", 60, 120),
        ("sticky", 60, 120),
        ("sticky", None, 80),
        ("hold", 60, None),
        ("hold", None, 90),
        ("hold", 45, 70),
    ],
)
@pytest.mark.parametrize("seed", range(3))
def test_guarded_adversaries_stay_safe(
    base_scenario, grid_states, controller, max_green, max_red, seed
):
    plan = plan_of(base_scenario)
    n, min_green = len(plan.stages), plan.min_green_s
    allowed = safe_transitions(plan.stages, grid_states)
    caps = None if max_green is None else [max_green] * n
    guard = SafetyGuard(n, min_green, caps, max_red, allowed)
    decide = {"random": Random(seed, n), "sticky": sticky, "hold": hold}[controller]
    shown, max_red_seen = drive(decide, guard, plan, 2000)

    assert audit_phases(shown, grid_states, plan.stages, min_green) == dict.fromkeys(AUDIT_KEYS, 0)
    greens = green_runs(shown, plan)
    assert {s for s, _ in greens} == set(range(n))  # every stage served
    if max_green is not None:
        assert max(steps for _, steps in greens) <= max_green
    if max_red is not None:
        # starvation forces each stage still ahead after min green: the starving stage
        # waits at most n-1 min greens and every clearance after reaching max_red
        clearance = sum(plan.durations[p] for s in plan.stages for p in s.clearance_phases)
        assert max(max_red_seen) <= max_red + (n - 1) * min_green + clearance + 1.0


@pytest.mark.parametrize("seed", range(3))
def test_free_order_jumps_stay_link_safe(base_scenario, grid_states, seed):
    plan = plan_of(base_scenario)
    allowed = safe_transitions(plan.stages, grid_states)
    guard = SafetyGuard(4, plan.min_green_s, [60] * 4, 120, allowed, strict_order=False)
    shown, _ = drive(Random(seed, 4), guard, plan, 2000)
    audit = audit_phases(shown, grid_states, plan.stages, plan.min_green_s)
    assert audit["out_of_order_green"] > 0  # jumps did happen ...
    del audit["out_of_order_green"]
    assert set(audit.values()) == {0}  # ... and never cut a link without yellow


def actuated_cfg(cfg: dict) -> dict:
    """The grid scenario with S1 sections (the Pitié configs carry their own)."""
    if "actuated" in cfg:
        return cfg
    cfg = copy.deepcopy(cfg)
    names = [s["name"] for s in cfg["network"]["stages"]]
    cfg["detectors"] = {"control": {"distance_m": None}}
    cfg["actuated"] = {
        "max_gap_s": 3.0,
        "min_green_s": dict.fromkeys(names, 6),
        "max_green_s": dict(zip(names, [40, 12, 25, 10], strict=True)),
    }
    return cfg


def synthetic_gaps(seed: int, n: int):
    """Per-stage time since the last detection: vehicles arrive at random, so loops
    alternate between busy (gap < 3 s) and idle spells."""
    rng = random.Random(seed)
    gaps = [0.0] * n

    def extras(obs: Observation) -> dict:
        for k in range(n):
            gaps[k] = 0.0 if rng.random() < 0.3 else gaps[k] + 1.0
        return {EXTRA_GAP_S: tuple(gaps)}

    return extras


@pytest.mark.parametrize("scenario", ["base_scenario", "pitie_scenario"])
@pytest.mark.parametrize("controller", ["fixed", "actuated"])
def test_fixed_and_actuated_never_trigger_the_guard(request, scenario, controller):
    cfg = actuated_cfg(request.getfixturevalue(scenario))
    durations, states = program(cfg)
    plan = plan_of(cfg)
    guard = SafetyGuard.from_config(cfg, plan.stages, states, durations)
    ctrl = make_controller({**cfg, "controller": controller})
    extras = synthetic_gaps(7, len(plan.stages)) if controller == "actuated" else None
    shown, _ = drive(ctrl.decide, guard, plan, 7200, extras)
    assert sum(guard.interventions.values()) == 0, dict(guard.interventions)
    assert set(audit_phases(shown, states, plan.stages, plan.min_green_s).values()) == {0}
    if controller == "actuated":  # the gaps really made greens vary
        assert len({steps for s, steps in green_runs(shown, plan) if s == 0}) > 3


# -- Part 4: D20 rule order on the Pitié program ----------------------------------------
HOP, S_LEFT, STM, BUS_91, BUS_57 = range(5)
CROSSING = {"NB": 0, "S": 1, "WB": 2, "W": 3}  # network.crossings key order


def pitie_guard(cfg: dict) -> SafetyGuard:
    durations, states = program(cfg)
    return SafetyGuard.from_config(cfg, plan_of(cfg).stages, states, durations)


@pytest.fixture
def s2_guard(rules_scenario) -> SafetyGuard:
    return pitie_guard(rules_scenario)


def pstate(stage, tis, target=None, transition=False, red=(0.0,) * 5, t=100.0, **waits):
    w = [0.0] * 4
    for name, value in waits.items():
        w[CROSSING[name]] = value
    return SignalState(t, stage, float(tis), transition, tuple(red), tuple(w), target)


def quiet(g: SafetyGuard) -> bool:
    return sum(g.interventions.values()) == 0


def test_preemption_never_cuts_min_green_or_clearance(s2_guard):
    g = s2_guard
    assert g.apply(BUS_57, pstate(BUS_57, 3, target=S_LEFT)) == BUS_57  # R2, hard
    assert g.apply(BUS_57, pstate(BUS_57, 7, target=S_LEFT, transition=True)) == BUS_57  # R1
    assert quiet(g) and g.preemption == dict.fromkeys(PREEMPTION_KEYS, 0)
    # BUS_57 -> S_LEFT is link-safe and skips HOP: a jump
    assert g.apply(BUS_57, pstate(BUS_57, 6, target=S_LEFT)) == S_LEFT
    assert g.preemption["jumps"] == 1 and quiet(g)
    assert list(g.preemption) == list(PREEMPTION_KEYS)


def test_preemption_hops_over_the_link_safe_graph_even_under_strict_order(s2_guard):
    g = s2_guard
    assert g.strict_order
    assert g.apply(STM, pstate(STM, 6, target=HOP)) == BUS_91  # the only legal move
    assert g.apply(S_LEFT, pstate(S_LEFT, 6, target=HOP)) == HOP  # skips STM
    assert g.apply(HOP, pstate(HOP, 8, target=STM)) == S_LEFT
    assert quiet(g)  # order never fires during preemption
    # the guard's hop is the contract of control.preemption.next_hop
    hops = [next_hop(s, HOP, g.allowed, g.clearance_s, 6.0) for s in range(5)]
    assert hops == [HOP, HOP, BUS_91, BUS_57, HOP]


def test_preemption_suspends_the_part3_rules(s2_guard):
    g = s2_guard
    starving = (0.0, 200.0, 190.0, 185.0, 181.0)  # every other stage over max_red_s 180
    for tis in (70, 71, 72):  # beyond safety.max_green_s 60
        assert g.apply(STM, pstate(HOP, tis, target=HOP, red=starving)) == HOP
    assert quiet(g)
    assert g.preemption["hold_s"] == 3.0
    # without the request the same state is cut by starvation
    assert g.apply(HOP, pstate(HOP, 73, red=starving)) == S_LEFT
    assert g.interventions["starvation"] == 1


def test_shield_cuts_a_preemption_hold_and_counts_ped_yield(s2_guard):
    g = s2_guard
    # S is red in HOP: 62 + 25 + 1 = 88 allows one more step, 62.5 does not
    assert g.apply(HOP, pstate(HOP, 40, target=HOP, S=62)) == HOP
    assert g.apply(HOP, pstate(HOP, 41, target=HOP, S=62.5)) == S_LEFT
    assert g.apply(HOP, pstate(HOP, 42, target=HOP, S=63.5)) == S_LEFT
    assert g.preemption["ped_yield"] == 1  # once per green
    assert quiet(g)  # a shield cut of a preemption is not a controller intervention
    (event,) = g.events
    assert (event.source, event.rule, event.stage_from, event.stage_to) == (
        "guard", "ped_yield", HOP, S_LEFT
    )  # fmt: skip
    assert "crossing=S" in event.detail and "target=0" in event.detail


def test_shield_replaces_a_preemption_jump_by_the_successor(s2_guard):
    g = s2_guard
    # S_LEFT -> HOP leaves S red for 10 + 31 s more: S waiting 48 s would reach 89 s
    assert g.apply(S_LEFT, pstate(S_LEFT, 6, target=HOP, S=48)) == STM
    assert g.preemption["ped_yield"] == 1 and g.preemption["jumps"] == 0
    g.reset()
    assert g.apply(S_LEFT, pstate(S_LEFT, 6, target=HOP, S=47)) == HOP
    assert g.preemption["jumps"] == 1 and g.preemption["ped_yield"] == 0


def test_shield_cuts_a_stay_last(s2_guard):
    g = s2_guard
    assert g.apply(HOP, pstate(HOP, 37, S=62)) == HOP  # 62 + 25 + 1 = 88
    assert g.apply(HOP, pstate(HOP, 38, S=62.5)) == S_LEFT
    assert g.apply(HOP, pstate(HOP, 39, S=63.5)) == S_LEFT
    assert g.interventions["ped_wait"] == 1
    assert sum(g.interventions.values()) == 1
    (event,) = g.events
    assert (event.t, event.source, event.rule, event.stage_from, event.stage_to) == (
        100.0, "guard", "ped_wait", HOP, S_LEFT
    )  # fmt: skip
    assert event.detail.startswith("crossing=S")
    # the successor itself is never refused, whatever the waits
    assert g.apply(STM, pstate(S_LEFT, 8, S=87.9, W=87.9)) == STM


def test_shield_checks_free_order_jumps(rules_cfg):
    rules_cfg["safety"]["stage_order"] = "free"
    g = pitie_guard(rules_cfg)
    assert g.apply(HOP, pstate(S_LEFT, 8, S=40)) == HOP  # 40 + 41 <= 88
    assert quiet(g)
    assert g.apply(HOP, pstate(S_LEFT, 9, S=48)) == STM  # 48 + 41 > 88
    assert g.interventions["ped_wait"] == 1


def test_min_green_still_counts_without_preemption(s2_guard):
    g = s2_guard
    assert g.apply(S_LEFT, pstate(HOP, 3)) == HOP
    assert g.apply(S_LEFT, pstate(HOP, 4)) == HOP
    assert g.interventions["min_green"] == 1 and len(g.events) == 1
    assert g.events[0].rule == "min_green" and g.events[0].source == "guard"
    g.reset()
    assert g.events == [] and quiet(g) and g.preemption["hold_s"] == 0


def test_unknown_preemption_target_rejected(s2_guard):
    with pytest.raises(ValueError, match="unknown preemption target"):
        s2_guard.apply(HOP, pstate(HOP, 10, target=5))


def test_guard_without_shield_or_clearances_still_preempts():
    g = guard4(max_green_s=[60] * 4, max_red_s=120)  # every transition link-safe
    assert g.shield is None and g.clearance_s is None
    st = dataclasses.replace(state(2, 8), preempt_target=0)
    assert g.apply(2, st) == 0  # direct jump, even under strict order
    assert g.preemption["jumps"] == 1 and quiet(g)


# -- Part 4: from_config ----------------------------------------------------------------
def test_s2_needs_the_pedestrian_shield(rules_cfg):
    g = pitie_guard(rules_cfg)
    assert g.shield.mode == "detected" and g.max_red_s == 180
    rules_cfg["safety"]["ped_forcing"] = "off"
    with pytest.raises(
        ConfigError, match=r"crossing S worst-case red 120 s > 88 s .* shield is off"
    ):
        pitie_guard(rules_cfg)
    rules_cfg["safety"]["ped_forcing"] = "red"
    assert pitie_guard(rules_cfg).shield.mode == "red"


def test_s2_max_red_must_exceed_its_177_s_cycle(rules_cfg):
    # 50 + 16 + 30 + 16 + 16 of green (bus stages extended by TSP) + 49 s of clearance
    assert pitie_guard(rules_cfg).max_red_s == 180
    rules_cfg["safety"]["max_red_s"] = 170
    with pytest.raises(ConfigError, match=r"177 s \(rules\.max_green_s \+ rules\.tsp"):
        pitie_guard(rules_cfg)
    rules_cfg["safety"]["max_red_s"] = 177.5
    assert pitie_guard(rules_cfg).max_red_s == 177.5
    rules_cfg["rules"]["tsp"]["enabled"] = False  # no extension: 50 + 6 + 30 + 6 + 6 + 49
    rules_cfg["safety"]["max_red_s"] = 148
    assert pitie_guard(rules_cfg).max_red_s == 148


def test_s2_greens_with_extension_fit_under_the_max_green_cap(rules_cfg):
    rules_cfg["rules"]["tsp"]["extension_max_s"] = 55  # extended stages 6 + 55 > 60
    with pytest.raises(ConfigError, match=r"extension_max_s of stages \['S_LEFT', 'BUS_91'"):
        pitie_guard(rules_cfg)
    rules_cfg["rules"]["tsp"]["extension_max_s"] = 10
    rules_cfg["rules"]["tsp"]["extend_stages"] = ["S_LEFT", "NOPE"]
    with pytest.raises(ConfigError, match=r"unknown stage\(s\) \['NOPE'\]"):
        pitie_guard(rules_cfg)


def test_preemption_must_respect_the_pedestrian_bound(rules_cfg):
    rules_cfg["preemption"]["respects_ped_max_wait"] = False
    with pytest.raises(ConfigError, match="A-04"):
        pitie_guard(rules_cfg)


def test_static_bound_checked_when_the_shield_is_off(pitie_cfg):
    pitie_cfg["safety"]["ped_forcing"] = "off"
    pitie_cfg["controller"] = "actuated"
    assert pitie_guard(pitie_cfg).shield is None  # S1 maxima: 88 s, exactly the bound
    pitie_cfg["actuated"]["max_green_s"]["HOP"] = 39
    with pytest.raises(ConfigError, match=r"actuated\.max_green_s: crossing S worst-case red 89 s"):
        pitie_guard(pitie_cfg)
    pitie_cfg["safety"]["ped_forcing"] = "detected"  # the shield bounds it at run time
    assert pitie_guard(pitie_cfg).shield is not None


# -- Part 4: D13 / D22, fixed and actuated never trigger the shield ---------------------
class PedWaits:
    """Synthetic waits per crossing of the Pitié program: `red` gives the seconds since
    the crossing last showed green (the upper bound of any wait, `ped_forcing: red`);
    `poisson` the longest wait of pedestrians arriving at random (detected waits)."""

    def __init__(self, model: str, links: list[int], states: list[str], seed: int = 3):
        self.model, self.links, self.states = model, links, states
        self.rng = random.Random(seed)
        self.red = [0.0] * len(links)
        self.arrivals: list[list[float]] = [[] for _ in links]
        self.max_wait = 0.0

    def waits(self, t: float) -> tuple[float, ...]:
        if self.model == "red":
            return tuple(self.red)
        return tuple(t - a[0] if a else 0.0 for a in self.arrivals)

    def after(self, t: float, phase: int) -> None:
        """Update with the phase shown during step t (t to t + 1)."""
        for c, link in enumerate(self.links):
            if self.states[phase][link] in "Gg":
                self.max_wait = max([self.max_wait, *(t - a for a in self.arrivals[c])])
                self.arrivals[c], self.red[c] = [], 0.0
            else:
                self.red[c] += 1.0
                if self.rng.random() < 0.06:  # about 200 pedestrians per hour of red
                    self.arrivals[c].append(t + 1.0)


@pytest.mark.parametrize("model", ["red", "poisson"])
@pytest.mark.parametrize("controller", ["fixed", "actuated"])
def test_fixed_and_actuated_never_trigger_the_shield(pitie_scenario, controller, model):
    cfg = pitie_scenario
    durations, states = program(cfg)
    plan = plan_of(cfg)
    guard = SafetyGuard.from_config(cfg, plan.stages, states, durations)
    assert guard.shield is not None
    ctrl = make_controller({**cfg, "controller": controller})
    gaps = synthetic_gaps(7, len(plan.stages)) if controller == "actuated" else None
    peds = PedWaits(model, [int(c["link"]) for c in cfg["network"]["crossings"].values()], states)
    closest = -math.inf
    for i in range(7200):
        t = float(i)
        obs = Observation(t, plan.stage, plan.time_in_stage, plan.in_transition)
        if gaps is not None:
            obs.extras.update(gaps(obs))
        st = dataclasses.replace(plan.state(t), ped_wait=peds.waits(t))
        stage = guard.apply(ctrl.decide(obs), st)
        if stage == st.stage and not st.in_transition and st.time_in_stage >= 6:
            closest = max(closest, max(guard.shield.stay_excess(st)))  # a stay granted
        phase = plan.request(stage)
        peds.after(t, phase)
        plan.advance(1.0)
    assert guard.interventions == dict.fromkeys(INTERVENTION_KEYS, 0)
    assert guard.events == []
    assert peds.max_wait <= 88
    if model == "red" and controller == "actuated":
        assert closest == 0.0  # S1 reaches the boundary and is never cut

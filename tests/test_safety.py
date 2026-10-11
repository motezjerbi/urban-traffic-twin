"""SafetyGuard (Part 3 rules) and the link-safe stage transitions, driven through
SignalPlan the way SimSession does: observe, decide, guard, request, advance."""

import copy
import random
import xml.etree.ElementTree as ET

import pytest

from traffic_twin.config import ConfigError
from traffic_twin.control.base import EXTRA_GAP_S, Observation
from traffic_twin.control.registry import make_controller
from traffic_twin.control.safety import (
    INTERVENTION_KEYS,
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

"""Pedestrian shield (control.pedestrian, D21): fallback timing on the Pitié program,
the stay / jump checks, the S1 boundary and the config checks. XML only, no SUMO."""

from __future__ import annotations

import dataclasses
import math
import xml.etree.ElementTree as ET

import pytest

from traffic_twin.config import ConfigError
from traffic_twin.control.pedestrian import (
    FORCING_MODES,
    PedestrianShield,
    cycle_red,
    forcing_mode,
)
from traffic_twin.sim.signal import SignalPlan, SignalState, Stage
from traffic_twin.sim.tls_check import load_program, scenario_stages, worst_case_red

HOP, S_LEFT, STM, BUS_91, BUS_57 = range(5)
NB, S, WB, W = range(4)  # network.crossings key order
LINKS = (12, 13, 14, 15)  # their TLS links


def program(cfg: dict) -> tuple[list[float], list[str]]:
    root = ET.parse(cfg["network"]["net_file"]).getroot()
    tl = next(t for t in root.iter("tlLogic") if t.get("id") == cfg["network"]["tls_id"])
    phases = list(tl.iter("phase"))
    return [float(p.get("duration")) for p in phases], [p.get("state") for p in phases]


@pytest.fixture(scope="module")
def pitie_program(pitie_scenario):
    durations, states = program(pitie_scenario)
    return scenario_stages(pitie_scenario), durations, states


@pytest.fixture(scope="module")
def shield(pitie_scenario, pitie_program) -> PedestrianShield:
    stages, durations, states = pitie_program
    out = PedestrianShield.from_config(pitie_scenario, stages, durations, states)
    assert out is not None
    return out


def state(stage, tis, waits=(0.0, 0.0, 0.0, 0.0), transition=False) -> SignalState:
    return SignalState(0.0, stage, float(tis), transition, (0.0,) * 5, tuple(waits))


def waits(**by_name: float) -> tuple[float, ...]:
    out = [0.0] * 4
    for name, w in by_name.items():
        out[{"NB": NB, "S": S, "WB": WB, "W": W}[name]] = w
    return tuple(out)


# -- from_config -----------------------------------------------------------------------
def test_pitie_shield_from_config(shield, pitie_scenario):
    assert shield.mode == "detected"
    assert shield.names == ("NB", "S", "WB", "W")
    assert shield.crossing_links == (12, 13, 14, 15)
    assert shield.threshold_s == 88.0 and shield.min_green_s == 6.0 and shield.dt == 1.0


@pytest.mark.parametrize("value", [None, "off", False])
def test_no_shield_when_forcing_is_off_or_absent(pitie_cfg, pitie_program, value):
    if value is None:
        del pitie_cfg["safety"]["ped_forcing"]
    else:
        pitie_cfg["safety"]["ped_forcing"] = value  # False: YAML's bare `off`
    assert forcing_mode(pitie_cfg) == "off"
    assert PedestrianShield.from_config(pitie_cfg, *pitie_program) is None


def test_no_shield_without_crossings(base_scenario):
    cfg = {**base_scenario, "safety": {**base_scenario["safety"], "ped_forcing": "detected"}}
    assert "crossings" not in cfg["network"]
    assert PedestrianShield.from_config(cfg, [Stage("A", 0, ())], [1.0], ["G"]) is None


def test_red_mode_has_the_same_arithmetic(pitie_cfg, pitie_program, shield):
    pitie_cfg["safety"]["ped_forcing"] = "red"
    red = PedestrianShield.from_config(pitie_cfg, *pitie_program)
    assert red.mode == "red"
    assert red.time_to_green(W, S_LEFT, 0) == shield.time_to_green(W, S_LEFT, 0)


@pytest.mark.parametrize("value", ["always", True, "on"])
def test_unknown_mode_is_a_config_error(pitie_cfg, pitie_program, value):
    pitie_cfg["safety"]["ped_forcing"] = value
    with pytest.raises(ConfigError, match="ped_forcing must be one of"):
        PedestrianShield.from_config(pitie_cfg, *pitie_program)
    assert FORCING_MODES == ("detected", "red", "off")


def test_min_green_red_above_the_threshold_is_a_config_error(pitie_cfg, pitie_program):
    # W is red 72 s per cycle with every stage at 6 s: theta 70 - 2 = 68 cannot hold
    pitie_cfg["safety"]["ped_max_wait_s"] = 70
    with pytest.raises(ConfigError, match=r"crossing W: worst-case red 72 s .* 68 s"):
        PedestrianShield.from_config(pitie_cfg, *pitie_program)
    pitie_cfg["safety"]["ped_max_wait_s"] = 74  # theta 72: exactly reachable
    assert PedestrianShield.from_config(pitie_cfg, *pitie_program).threshold_s == 72


def test_missing_max_wait_is_a_config_error(pitie_cfg, pitie_program):
    del pitie_cfg["safety"]["ped_max_wait_s"]
    with pytest.raises(ConfigError, match="needs safety.ped_max_wait_s"):
        PedestrianShield.from_config(pitie_cfg, *pitie_program)


def test_constructor_checks(pitie_program):
    stages, durations, states = pitie_program
    with pytest.raises(ConfigError, match="'detected' or 'red'"):
        PedestrianShield(stages, durations, states, [15], 6, 88, mode="off")
    with pytest.raises(ConfigError, match="outside"):
        PedestrianShield(stages, durations, states, [16], 6, 88)
    with pytest.raises(ConfigError, match="durations"):
        PedestrianShield(stages, durations[:-1], states, [15], 6, 88)
    with pytest.raises(ConfigError, match="names"):
        PedestrianShield(stages, durations, states, [15], 6, 88, names=["W", "S"])


# -- fallback timing -------------------------------------------------------------------
def test_time_to_green_on_the_pitie_program(shield):
    """From the start of a green (min green 6 s, clearances HOP 9, S_LEFT 10, STM 7,
    BUS_91 12, BUS_57 11 s): W (green in HOP) from S_LEFT 6+10+6+7+6+12+6+11 = 64 s."""
    assert [shield.time_to_green(W, k, 0) for k in (S_LEFT, STM, BUS_91, BUS_57)] == [
        64, 48, 35, 17
    ]  # fmt: skip
    assert shield.time_to_green(S, HOP, 0) == 6 + 9 + 6 + 10 == 31
    # the remaining min green only: past it, the clearance starts now
    assert shield.time_to_green(S, HOP, 4) == 27
    assert shield.time_to_green(S, HOP, 37) == 25
    # green now: 0 (S is green in STM and in BUS_91)
    assert shield.time_to_green(S, STM, 0) == shield.time_to_green(S, BUS_91, 12) == 0
    assert shield.time_to_green(W, HOP, 40) == 0


def test_crossing_green_inside_a_clearance_phase():
    """Link 1 (a crossing) turns green in A's all-red phase (a leading pedestrian
    interval before B): it is served from the start of that phase."""
    states = ["Gr", "yr", "rG", "rG", "yr"]
    stages = [Stage("A", 0, (1, 2)), Stage("B", 3, (4,))]
    shield = PedestrianShield(stages, [20, 3, 2, 20, 3], states, [1], 6, 88)
    assert shield.time_to_green(0, 0, 0) == 6 + 3  # min green, then A's yellow
    assert shield.time_via(0, 0, 1) == 3
    assert shield.time_to_green(0, 1, 0) == 0
    # the crossing's longest red: B's yellow + A's green + A's yellow
    assert shield.worst_red(0) == 3 + 6 + 3
    assert shield.worst_red(0, [30, 6]) == 3 + 30 + 3


def test_time_via_s_left_to_hop(shield):
    # S_LEFT's clearance (10 s), then HOP from its start: S needs HOP's min green, its
    # clearance and S_LEFT again (6 + 9 + 6 + 10)
    assert shield.time_via(S, S_LEFT, HOP) == 10 + 31
    assert shield.time_via(W, S_LEFT, HOP) == 10  # W is green in HOP
    assert shield.time_via(NB, S_LEFT, HOP) == 10
    assert shield.time_via(WB, S_LEFT, HOP) == 0  # WB stays green through S_LEFT's clearance


def test_cycle_red_matches_tls_check(pitie_scenario, pitie_program):
    stages, durations, states = pitie_program
    prog = load_program(pitie_scenario["network"]["net_file"], "C")
    for greens in ([36, 6, 20, 6, 6], [38, 6, 22, 6, 6], [6] * 5, [50, 16, 30, 16, 16]):
        red = worst_case_red(prog, stages, greens)
        for link in (12, 13, 14, 15):
            assert cycle_red(stages, durations, states, greens, link) == red[link]
    assert cycle_red(stages, durations, states, [6] * 5, 15) == 72  # W at min greens
    with pytest.raises(ValueError, match="greens"):
        cycle_red(stages, durations, states, [6] * 4, 15)


def test_never_green_crossing_has_infinite_time():
    stages = [Stage("A", 0, (1,)), Stage("B", 2, (3,))]
    shield = PedestrianShield(stages, [10, 3, 10, 3], ["Gr", "yr", "rG", "ry"], [1], 0, 88)
    assert shield.time_to_green(0, 0, 0) == 3  # B
    never = PedestrianShield(stages, [10, 3, 10, 3], ["Gr", "yr", "Gr", "yr"], [1], 0, 88)
    assert never.time_to_green(0, 0, 0) == math.inf
    assert never.worst_red(0) == math.inf


# -- checks ----------------------------------------------------------------------------
def test_stay_and_slack(shield):
    st = state(HOP, 20, waits(S=60))
    assert shield.slack(st)[S] == 88 - (60 + 25)
    assert shield.slack(st)[W] == math.inf  # green now
    assert shield.stay_excess(st)[S] == 60 + 25 + 1 - 88
    assert shield.stay_ok(st)
    assert not shield.stay_ok(state(HOP, 20, waits(S=62.5)))
    # before min green or in a transition a stay is always allowed
    assert shield.stay_ok(state(HOP, 3, waits(S=80)))
    assert shield.stay_ok(state(HOP, 20, waits(S=80), transition=True))


def test_s1_sits_exactly_on_the_boundary(pitie_scenario, pitie_program, shield):
    """S1 at its max greens (HOP 38, STM 22) with red-mode waits (seconds since the
    crossing last showed green, the upper bound of any detected wait): on the last step
    of STM, W's check is 45 + 42 + 1 = 88 <= 88 (red since HOP's pedestrian clearance:
    8 + 16 + 21 s); on the last step of HOP, S's is 62 + 25 + 1 = 88. So every stay of S1
    passes, and 1 ms more fails."""
    stages, durations, states = pitie_program
    maxima = [pitie_scenario["actuated"]["max_green_s"][s.name] for s in stages]
    plan = SignalPlan(stages, durations, 6.0)
    red = [0.0] * 4
    worst = -math.inf
    at_boundary = []
    for t in range(3 * 127):
        st = dataclasses.replace(plan.state(float(t)), ped_wait=tuple(red))
        stay = not st.in_transition and st.time_in_stage < maxima[st.stage] - 1e-9
        if stay and st.time_in_stage >= 6:
            excess = max(shield.stay_excess(st))
            worst = max(worst, excess)
            assert shield.stay_ok(st)
            if excess == 0:
                at_boundary.append((st.stage, st.time_in_stage, st.ped_wait))
        phase = plan.request(st.stage if stay else (st.stage + 1) % 5)
        red = [
            0.0 if states[phase][lk] in "Gg" else r + 1.0 for r, lk in zip(red, LINKS, strict=True)
        ]
        plan.advance(1.0)
    assert worst == 0.0
    boundary = {(stage, tis) for stage, tis, _ in at_boundary}
    assert boundary == {(STM, 21.0), (HOP, 37.0)}
    stm = next(w for stage, _, w in at_boundary if stage == STM)
    assert stm[W] == 45
    over = dataclasses.replace(state(STM, 21, stm), ped_wait=(*stm[:W], stm[W] + 0.001))
    assert not shield.stay_ok(over)


def test_jump_ok(shield):
    # S_LEFT -> HOP (a link-safe jump, not the successor): S needs 10 + 31 s more
    assert shield.jump_ok(state(S_LEFT, 6, waits(S=47)), HOP)
    assert not shield.jump_ok(state(S_LEFT, 6, waits(S=47.5)), HOP)
    assert shield.jump_excess(state(S_LEFT, 6, waits(S=47.5)), HOP)[S] == pytest.approx(0.5)
    # the successor is the fallback itself; a jump to the current stage is a stay
    assert shield.jump_ok(state(S_LEFT, 6, waits(S=87.9, W=87.9)), STM)
    assert shield.jump_ok(state(HOP, 20, waits(S=60)), HOP) == shield.stay_ok(
        state(HOP, 20, waits(S=60))
    )
    assert shield.jump_ok(state(S_LEFT, 6, waits(S=80), transition=True), HOP)


def test_waits_default_to_zero_and_must_match_the_crossings(shield):
    st = SignalState(0.0, HOP, 10.0, False, (0.0,) * 5)
    assert shield.waits(st) == (0.0,) * 4
    with pytest.raises(ValueError, match="3 pedestrian waits for 4 crossings"):
        shield.waits(dataclasses.replace(st, ped_wait=(1.0, 2.0, 3.0)))


def test_worst_names_the_crossing(shield):
    assert shield.worst((-1.0, 0.5, -3.0, 0.25)) == ("S", 0.5)

"""Static signal-program checks (sim.tls_check): foe greens, intergreens from the
junction geometry and worst-case crossing red, on hand-built programs and on the
committed nets. Only XML is read: no SUMO binary needed."""

import math
from pathlib import Path

import pytest
import yaml

from traffic_twin.config import load_scenario
from traffic_twin.sim import tls_check
from traffic_twin.sim.signal import Stage
from traffic_twin.sim.tls_check import (
    Link,
    Program,
    check_scenario,
    foe_green_conflicts,
    intergreen_requirements,
    intergreen_violations,
    link_table,
    load_program,
    permissive_crossing_pairs,
    with_greens,
    worst_case_red,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def program(states, durations=None, foes=None) -> Program:
    """Links 0 and 1 are foe vehicle links, 2 a crossing over link 0, 3 a free link."""
    links = {
        0: Link(0, "a_in_1", "b_out_1", ":X_0_0", False),
        1: Link(1, "c_in_1", "d_out_1", ":X_1_0", False),
        2: Link(2, ":X_w0_0", ":X_c0_0", ":X_c0", True),
        3: Link(3, "e_in_1", "f_out_1", ":X_3_0", False),
    }
    foes = foes or {0: frozenset({1, 2}), 1: frozenset({0}), 2: frozenset({0}), 3: frozenset()}
    durations = durations or tuple(10.0 for _ in states)
    return Program("X", tuple(durations), tuple(states), links, foes, {2: 9.6})


# -- hand-built programs -------------------------------------------------------------
def test_foe_vehicle_links_both_green_are_errors():
    errors = foe_green_conflicts(program(["GGrr", "rrrG"]))
    assert len(errors) == 1 and "both green" in errors[0] and "phase 0" in errors[0]


def test_permissive_vehicle_vehicle_pair_is_an_error():
    errors = foe_green_conflicts(program(["rrrG", "Ggrr"]))
    assert len(errors) == 1 and "permissive" in errors[0] and "phase 1" in errors[0]


def test_vehicle_may_only_yield_to_a_crossing():
    prog = program(["grGr", "rrrG"])
    assert foe_green_conflicts(prog) == []
    assert permissive_crossing_pairs(prog) == [(0, 2)]
    errors = foe_green_conflicts(program(["GrGr", "rrrG"]))  # vehicle G across a green crossing
    assert len(errors) == 1 and "priority" in errors[0]


def test_non_foes_may_be_green_together():
    assert foe_green_conflicts(program(["GrrG", "rGrG"])) == []


def test_intergreen_shorter_than_required_is_detected():
    # link 0 green 30 s, yellow 3 s, all-red 2 s, then link 1 green 30 s, all-red 5 s
    prog = program(["GrrG", "yrrG", "rrrG", "rGrG", "rrrG"], durations=(30, 3, 2, 30, 5))
    assert intergreen_violations(prog, {(0, 1): 5.0}) == []
    (err,) = intergreen_violations(prog, {(0, 1): 6.0})
    assert err.startswith("intergreen 0(a_in_1->b_out_1) -> 1(c_in_1->d_out_1): 5 s < 6 s")
    # around the end of the cycle: link 1 ends at 65 s, link 0 starts again at 70 s
    assert intergreen_violations(prog, {(1, 0): 5.0}) == []
    assert len(intergreen_violations(prog, {(1, 0): 5.5})) == 1


def test_shorter_intermediate_green_shortens_an_intergreen():
    """Link 0 -> link 2 spans the green of stage B, so shorter greens (the minimum any
    controller may show) can break an intergreen that the fixed greens satisfy."""
    prog = program(
        ["GrrG", "yrrG", "rGrG", "rrrr", "rrGG", "rrGr"], durations=(20, 3, 20, 3, 20, 3)
    )
    stages = [Stage("A", 0, (1,)), Stage("B", 2, (3,)), Stage("C", 4, (5,))]
    assert intergreen_violations(prog, {(0, 2): 15.0}) == []  # 26 s
    short = with_greens(prog, stages, [20, 6, 20])
    assert short.durations == (20, 3, 6, 3, 20, 3)
    assert len(intergreen_violations(short, {(0, 2): 15.0})) == 1  # 12 s


def test_worst_case_red_uses_the_given_greens_and_wraps_around():
    prog = program(["GrrG", "yrrG", "rrrG", "rGrG", "rrrG"], durations=(30, 3, 2, 30, 5))
    stages = [Stage("A", 0, (1, 2)), Stage("B", 3, (4,))]
    red = worst_case_red(prog, stages, [40, 30])
    assert red[0] == 3 + 2 + 30 + 5  # yellow counts as red
    assert red[1] == 5 + 40 + 3 + 2  # wraps from the end of the cycle
    assert red[2] == math.inf  # never green
    assert red[3] == 0  # always green
    assert prog.durations == (30, 3, 2, 30, 5)  # input untouched
    with pytest.raises(ValueError):
        worst_case_red(prog, stages, [40])


def test_link_table_lists_vehicle_links_and_crossings():
    assert link_table(program(["rrrr"])) == [
        " 0  a_in_1 -> b_out_1",
        " 1  c_in_1 -> d_out_1",
        " 2  crossing :X_c0",
        " 3  e_in_1 -> f_out_1",
    ]


def test_unknown_tls_is_reported(pitie_scenario):
    with pytest.raises(ValueError, match="no tlLogic"):
        load_program(REPO_ROOT / pitie_scenario["network"]["net_file"], "nope")


# -- committed Pitié net -------------------------------------------------------------
@pytest.fixture(scope="module")
def pitie_report(pitie_scenario):
    return check_scenario(pitie_scenario)


def test_pitie_program_passes_every_check(pitie_report):
    assert pitie_report.errors == []
    assert pitie_report.ok
    assert pitie_report.summary == {
        "foe_greens": 0, "permissive_vehicle_vehicle": 0, "intergreen_violations": 0
    }  # fmt: skip


def test_pitie_crossing_red_within_the_pedestrian_bound(pitie_scenario, pitie_report):
    safety = pitie_scenario["safety"]
    bound = safety["ped_max_wait_s"] - safety["ped_red_margin_s"]
    assert pitie_report.crossing_bound_s == bound == 88
    assert set(pitie_report.worst_crossing_red) == set(pitie_scenario["network"]["crossings"])
    assert max(pitie_report.worst_crossing_red.values()) <= bound


def test_pitie_actuated_max_greens_are_checked_too():
    report = check_scenario(load_scenario(REPO_ROOT / "configs/pitie_actuated.yaml"))
    assert report.ok, report.errors
    assert max(report.worst_crossing_red_max.values()) <= report.crossing_bound_s


def test_bus_leaving_the_near_side_stop_needs_a_longer_intergreen(pitie_scenario):
    """Link 9 (bus 91/24 E from wb_in_1, stop ending 8.8 m before the line) against
    link 4 (bus 57 S): 11 s at the 4 m/s pull-out speed, less at 6 m/s."""
    net = REPO_ROOT / pitie_scenario["network"]["net_file"]
    prog = load_program(net, "C")
    lane_length = tls_check._geometry(net).length
    stops = tls_check.stop_lanes(pitie_scenario, lane_length)
    assert "wb_in_1" in stops and "s_in_1" not in stops  # 57 N stops 80 m upstream
    assert intergreen_requirements(net, prog, stop_lanes=stops)[(9, 4)] == 11
    assert intergreen_requirements(net, prog)[(9, 4)] < 11
    # crossings are cleared at walking speed: 9.60 m / 1.2 m/s
    assert intergreen_requirements(net, prog)[(13, 4)] == 8


def test_too_long_green_breaks_the_pedestrian_bound(pitie_cfg):
    pitie_cfg["fixed"]["green_s"]["HOP"] = 40
    report = check_scenario(pitie_cfg)
    assert not report.ok
    assert report.worst_crossing_red["S"] == 90
    assert any(e.startswith("crossing S: worst-case red 90 s > 88 s") for e in report.errors)


def test_config_that_does_not_fit_the_program_is_reported(pitie_cfg):
    pitie_cfg["network"]["crossings"]["S"]["edge"] = ":C_c3"
    pitie_cfg["network"]["signal_groups"].pop("ped_W")
    errors = check_scenario(pitie_cfg).errors
    assert any("crossing S: link 13 crosses :C_c1" in e for e in errors)
    assert any(e.startswith("signal_groups") for e in errors)


def test_stage_outside_the_program_stops_the_check(pitie_cfg):
    pitie_cfg["network"]["stages"][0]["clearance_phases"] = [1, 2, 17]
    report = check_scenario(pitie_cfg)
    assert not report.ok
    assert report.errors == ["stage HOP: phase(s) [17] not in the 17-phase program"]


def test_grid_program_is_reported_without_failing(base_scenario):
    """The Part 2 grid keeps netconvert's permissive lefts: report only."""
    report = check_scenario(base_scenario)
    assert set(report.worst_crossing_red) == {":C_c0", ":C_c1", ":C_c2", ":C_c3"}
    assert report.summary["permissive_vehicle_vehicle"] > 0
    assert report.crossing_bound_s == 90


def test_cli_exit_code_and_summary(tmp_path, capsys):
    assert tls_check.main(["--config", str(REPO_ROOT / "configs/pitie_fixed.yaml")]) == 0
    out = capsys.readouterr().out
    assert "foe greens: 0 / permissive vehicle-vehicle: 0 / intergreen violations: 0" in out
    assert out.rstrip().endswith("OK")

    bad = tmp_path / "bad.yaml"
    base = (REPO_ROOT / "configs/pitie_fixed.yaml").as_posix()
    bad.write_text(yaml.safe_dump({"extends": base, "fixed": {"green_s": {"HOP": 45}}}))
    assert tls_check.main(["--config", str(bad)]) == 1
    assert "FAILED" in capsys.readouterr().out


# -- Part 4: min greens, the rules plan and the audit's cached requirements -------------
def test_min_greens_red_is_reported_and_bounded(pitie_report, pitie_cfg):
    assert pitie_report.worst_crossing_red_min == {"NB": 42, "S": 56, "WB": 42, "W": 72}
    pitie_cfg["safety"]["ped_max_wait_s"] = 72  # theta 70 < W's 72 s at min greens
    report = check_scenario(pitie_cfg)
    assert any(e == "crossing W: worst-case red 72 s > 70 s (min greens)" for e in report.errors)


def test_rules_plan_is_bounded_by_the_shield(rules_scenario):
    """S2 relaxes HOP to 50 s and STM to 30 s, + 10 s of TSP on the bus stages: S and W
    exceed the static bound, which the detected shield enforces at run time."""
    report = check_scenario(rules_scenario)
    assert report.ok, report.errors
    assert report.worst_crossing_red_rules == {"NB": 62, "S": 120, "WB": 62, "W": 126}
    assert report.dynamic_crossings == ["S", "W"] and report.ped_forcing == "detected"
    assert sum("dynamic (pedestrian shield" in w for w in report.warnings) == 2
    assert report.worst_crossing_red_max == {}  # actuated maxima: not this controller's


def test_rules_plan_without_the_shield_is_an_error(rules_cfg):
    rules_cfg["safety"]["ped_forcing"] = "off"
    report = check_scenario(rules_cfg)
    assert not report.ok and report.dynamic_crossings == []
    assert [e for e in report.errors if "needs the pedestrian shield" in e] == [
        "crossing S: worst-case red 120 s > 88 s (rules max greens + TSP extension) with "
        "safety.ped_forcing off: the rules plan needs the pedestrian shield",
        "crossing W: worst-case red 126 s > 88 s (rules max greens + TSP extension) with "
        "safety.ped_forcing off: the rules plan needs the pedestrian shield",
    ]


def test_rules_plan_within_the_static_bound_needs_no_shield(rules_cfg):
    rules_cfg["safety"]["ped_forcing"] = "off"
    rules_cfg["rules"]["max_green_s"].update(HOP=38, STM=22)
    rules_cfg["rules"]["tsp"]["enabled"] = False
    report = check_scenario(rules_cfg)
    assert report.ok, report.errors
    assert max(report.worst_crossing_red_rules.values()) == 88


def test_cached_requirements(pitie_scenario):
    net = pitie_scenario["network"]["net_file"]
    req = tls_check.cached_requirements(net, pitie_scenario)
    assert req is tls_check.cached_requirements(REPO_ROOT / net, pitie_scenario)  # cached
    prog = load_program(net, "C")
    stops = tls_check.stop_lanes(pitie_scenario, tls_check._geometry(net).length)
    assert dict(req.intergreen) == intergreen_requirements(net, prog, stop_lanes=stops)
    assert req.intergreen[(9, 4)] == 11  # the near-side bus stop is taken into account
    assert dict(req.foes) == prog.foes
    with pytest.raises(TypeError):
        req.intergreen[(9, 4)] = 0  # read-only: shared by every run


def test_cli_prints_the_dynamic_shield_note(capsys):
    assert tls_check.main(["--config", str(REPO_ROOT / "configs/pitie_rules.yaml")]) == 0
    out = capsys.readouterr().out
    assert "rules max (s)" in out and "min greens (s)" in out
    assert "rules plan: crossings S, W above the static bound, dynamic (pedestrian shield" in out
    assert out.rstrip().endswith("OK")

"""S2 rule-based controller (Part 4, D19, D23): the actuated base, TSP extension and
truncation, box clearing, pedestrian anticipation, events and report, config checks,
and the derivations written in configs/pitie/rules.yaml."""

import copy
import math
import random
from pathlib import Path

import pytest

from traffic_twin.config import ConfigError, demand_hash, load_scenario, stage_names
from traffic_twin.control.actuated import ActuatedController
from traffic_twin.control.base import (
    EXTRA_BUS_CALLS,
    EXTRA_GAP_S,
    EXTRA_GAP_S_OPEN,
    EXTRA_PED_SLACK,
    BusCall,
    Observation,
)
from traffic_twin.control.registry import make_controller
from traffic_twin.control.rules import RuleBasedController, TspParams
from traffic_twin.paths import CONFIG_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent

NAMES = ["HOP", "S_LEFT", "STM", "BUS_91", "BUS_57"]
HOP, S_LEFT, STM, BUS_91, BUS_57 = range(5)
MINS = [6.0] * 5
S1_MAX = [38.0, 6.0, 22.0, 6.0, 6.0]
S2_MAX = [50.0, 6.0, 30.0, 6.0, 6.0]
TSP = TspParams(
    enabled=True,
    lines=("91", "24", "57"),
    extend_stages=frozenset({S_LEFT, BUS_91, BUS_57}),
    extension_max_s=10.0,
    approach_speed_mps=7.0,
    start_penalty_s=4.0,
)
TRUNC = TspParams(
    enabled=True,
    lines=("91", "24", "57"),
    extend_stages=frozenset({S_LEFT, BUS_91, BUS_57}),
    extension_max_s=10.0,
    truncation=True,
    truncation_max_green_s=(38.0, 6.0, 22.0, 6.0, 6.0),
)


def bus(vid="bus_91_E.0", line="91", stages=(BUS_91,), dist=0.0, speed=7.0, dwell=0.0):
    return BusCall(vid, line, 9, tuple(stages), "WB", dist, speed, dwell)


def rob(stage=HOP, tis=10.0, gap=math.inf, gap_open=None, transition=False, calls=(),
        slack=(math.inf,), t=0.0) -> Observation:  # fmt: skip
    """A Pitié-like observation: only `stage` has a finite gap."""
    gaps = [math.inf] * 5
    gaps[stage] = gap
    opens = [math.inf] * 5
    opens[stage] = gap if gap_open is None else gap_open
    extras = {
        EXTRA_GAP_S: tuple(gaps),
        EXTRA_GAP_S_OPEN: tuple(opens),
        EXTRA_BUS_CALLS: tuple(calls),
        EXTRA_PED_SLACK: tuple(slack),
    }
    return Observation(t, stage, tis, transition, extras=extras)


def ctrl(maxs=S2_MAX, tsp=None, box=False, anticipate=False) -> RuleBasedController:
    return RuleBasedController(MINS, maxs, 3.0, tsp=tsp, box=box, anticipate=anticipate,
                               stage_names=NAMES)  # fmt: skip


# -- the actuated base ----------------------------------------------------------------
def test_base_equals_actuated_decision_for_decision():
    """With S1's maxima and TSP, box and anticipation off, S2 is S1 (plan section 2)."""
    rng = random.Random(4)
    rules, act = ctrl(S1_MAX), ActuatedController(MINS, S1_MAX, 3.0)
    for _ in range(7200):
        s = rng.randrange(5)
        gaps = tuple(math.inf if rng.random() < 0.3 else rng.uniform(0, 6) for _ in range(5))
        tis = rng.choice([rng.uniform(0, 45), float(rng.randrange(46))])
        o = Observation(rng.uniform(0, 7200), s, tis, rng.random() < 0.2,
                        extras={EXTRA_GAP_S: gaps})  # fmt: skip
        assert rules.decide(o) == act.decide(o)
    assert rules.events == [] and rules.report()["tsp"]["calls"] == 0


def test_base_from_config_equals_s1(rules_cfg, pitie_scenario):
    rules_cfg["rules"]["max_green_s"] = copy.deepcopy(rules_cfg["actuated"]["max_green_s"])
    rules_cfg["rules"]["tsp"]["enabled"] = False
    rules_cfg["rules"]["box"]["enabled"] = False
    rules_cfg["rules"]["pedestrian"]["anticipate"] = False
    rules = make_controller(rules_cfg)
    act = make_controller({**pitie_scenario, "controller": "actuated"})
    assert rules.max_green_s == act.max_green_s and rules.min_green_s == act.min_green_s
    rng = random.Random(7)
    for _ in range(2000):
        s = rng.randrange(5)
        gaps = tuple(rng.uniform(0, 5) for _ in range(5))
        o = Observation(0.0, s, float(rng.randrange(45)), False, extras={EXTRA_GAP_S: gaps})
        assert rules.decide(o) == act.decide(o)


def test_holds_min_green_and_clearance_and_only_moves_to_the_successor():
    c = ctrl(tsp=TSP, box=True, anticipate=True)
    assert c.decide(rob(HOP, 5.0, slack=(0.0,))) == HOP  # min green
    assert c.decide(rob(STM, 40.0, transition=True, slack=(0.0,))) == STM  # clearance
    assert c.decide(rob(BUS_57, 6.0)) == HOP  # wraps around
    rng = random.Random(1)
    for _ in range(3000):
        s = rng.randrange(5)
        calls = [bus(stages=(rng.randrange(5),), dist=rng.uniform(0, 200)) for _ in range(2)]
        slack = (rng.uniform(-1, 5),)
        o = rob(s, rng.uniform(0, 70), gap=rng.uniform(0, 6), gap_open=rng.uniform(3, 9),
                transition=rng.random() < 0.2, calls=calls, slack=slack)  # fmt: skip
        assert c.decide(o) in (s, (s + 1) % 5)


def test_relaxed_max_greens():
    c = ctrl()
    assert c.decide(rob(HOP, 49.0, gap=0.0)) == HOP
    assert c.decide(rob(HOP, 50.0, gap=0.0)) == S_LEFT  # max-out
    assert c.decide(rob(STM, 29.0, gap=1.0)) == STM
    assert c.decide(rob(STM, 30.0, gap=1.0)) == BUS_91
    assert c.decide(rob(HOP, 20.0, gap=3.0)) == S_LEFT  # gap-out at gap == max gap


def test_missing_extras_are_config_errors():
    with pytest.raises(ConfigError, match=r"detectors\.control"):
        ctrl().decide(Observation(0.0, 0, 8.0, False))
    for c, key in ((ctrl(box=True), EXTRA_GAP_S_OPEN), (ctrl(tsp=TSP), EXTRA_BUS_CALLS),
                   (ctrl(anticipate=True), EXTRA_PED_SLACK)):  # fmt: skip
        o = rob(HOP, 10.0, gap=0.0)
        del o.extras[key]
        with pytest.raises(ConfigError, match=key):
            c.decide(o)


# -- TSP extension --------------------------------------------------------------------
def test_extension_is_granted_only_within_the_extension():
    c = ctrl(tsp=TSP)
    call = bus(dist=14.0, speed=7.0)  # ETA 2 s
    assert c.decide(rob(BUS_91, 6.0, calls=[call])) == BUS_91  # would gap out at 6 s
    assert c.decide(rob(BUS_91, 14.0, calls=[call])) == BUS_91  # allow 2 s == ETA
    assert c.decide(rob(BUS_91, 14.5, calls=[call])) == BUS_57  # allow 1.5 s < ETA
    near = bus(dist=0.0, speed=7.0)  # ETA 0
    assert c.decide(rob(BUS_91, 15.0, calls=[near])) == BUS_91
    assert c.decide(rob(BUS_91, 16.0, calls=[near])) == BUS_57  # 6 + 10 s: never longer
    assert c.decide(rob(BUS_91, 6.0)) == BUS_57  # no bus: as S1


def test_extension_only_for_listed_lines_and_stages():
    c = ctrl(tsp=TSP)
    assert c.decide(rob(BUS_91, 6.0, calls=[bus(line="99")])) == BUS_57  # not a TSP line
    assert c.decide(rob(BUS_91, 6.0, calls=[bus(stages=(BUS_57,))])) == BUS_57  # other stage
    hop_bus = bus(stages=(HOP,))
    assert c.decide(rob(HOP, 50.0, gap=0.0, calls=[hop_bus])) == S_LEFT  # HOP not extended
    both = bus("bus_24_W.0", "24", stages=(BUS_91, BUS_57))
    assert c.decide(rob(BUS_57, 6.0, calls=[both])) == BUS_57
    off = ctrl(tsp=TspParams(**{**TSP.__dict__, "enabled": False}))
    assert off.decide(rob(BUS_91, 6.0, calls=[bus()])) == BUS_57
    assert off.report()["tsp"]["calls"] == 0


def test_eta_includes_the_dwell_and_the_start_penalty():
    assert TSP.eta(bus(dist=70.0, speed=5.0)) == pytest.approx(10.0)
    assert TSP.eta(bus(dist=70.0, speed=0.0)) == pytest.approx(14.0)  # halted
    assert TSP.eta(bus(dist=70.0, speed=0.0, dwell=12.0)) == pytest.approx(26.0)  # dwelling
    c = ctrl(tsp=TSP)  # BUS_91 at 6 s: allow 10 s
    assert c.decide(rob(BUS_91, 6.0, calls=[bus(dist=0.0, speed=0.0, dwell=7.0)])) == BUS_57
    assert c.decide(rob(BUS_91, 6.0, calls=[bus(dist=0.0, speed=0.0, dwell=6.0)])) == BUS_91
    assert c.decide(rob(BUS_91, 6.0, calls=[bus(dist=49.0, speed=0.05)])) == BUS_57  # 7 + 4
    assert c.decide(rob(BUS_91, 6.0, calls=[bus(dist=49.0, speed=3.0)])) == BUS_91  # 7


def test_extension_counts_seconds_per_line_and_greens_once():
    c = ctrl(tsp=TSP)
    c.reset(101)
    for tis in range(6, 12):  # held 6 steps for a 91, ETA 2 s, then the bus has passed
        c.decide(rob(BUS_91, float(tis), calls=[bus(dist=14.0)], t=100.0 + tis))
    assert c.decide(rob(BUS_91, 12.0, t=112.0)) == BUS_57
    c.decide(rob(BUS_57, 6.0, calls=[bus("bus_57_S.0", "57", (BUS_57,), 7.0)], t=120.0))
    rep = c.report()["tsp"]
    assert rep["calls"] == 2 and rep["extended_greens"] == 2 and rep["extension_s"] == 7.0
    assert rep["by_line"]["91"] == {"calls": 1, "extended_greens": 1, "extension_s": 6.0}
    assert rep["by_line"]["57"] == {"calls": 1, "extended_greens": 1, "extension_s": 1.0}
    assert rep["by_line"]["24"] == {"calls": 0, "extended_greens": 0, "extension_s": 0.0}
    ext = [e for e in c.events if e.rule == "tsp_extend"]
    assert [(e.t, e.stage_from, e.stage_to) for e in ext] == [(106.0, 3, 3), (120.0, 4, 4)]
    assert "vid=bus_91_E.0" in ext[0].detail and "line=91" in ext[0].detail


# -- TSP truncation (penalty cap) -----------------------------------------------------
def test_truncation_caps_the_other_stages_while_a_bus_waits():
    c = ctrl(tsp=TRUNC)
    waiting = bus(stages=(BUS_91,), dist=100.0)
    assert c.decide(rob(HOP, 37.0, gap=0.0, calls=[waiting])) == HOP
    assert c.decide(rob(HOP, 38.0, gap=0.0, calls=[waiting])) == S_LEFT  # cut at the cap
    assert c.decide(rob(HOP, 45.0, gap=0.0)) == HOP  # no bus: relaxed max green
    assert c.decide(rob(STM, 22.0, gap=0.0, calls=[waiting])) == BUS_91
    assert c.decide(rob(STM, 21.0, gap=0.0, calls=[waiting])) == STM
    # a bus served by the current stage keeps its green
    here = bus("bus_x.0", stages=(HOP,))
    assert c.decide(rob(HOP, 40.0, gap=0.0, calls=[waiting, here])) == HOP
    rep = c.report()["tsp"]
    assert rep["truncated_greens"] == 2 and rep["truncation_s"] == 12.0 + 8.0
    assert [e.rule for e in c.events] == ["tsp_truncate", "tsp_truncate"]
    off = ctrl(tsp=TspParams(**{**TRUNC.__dict__, "truncation": False}))
    assert off.decide(rob(HOP, 45.0, gap=0.0, calls=[waiting])) == HOP


# -- box clearing ---------------------------------------------------------------------
def test_box_uses_the_open_exit_gaps():
    blocked = rob(HOP, 10.0, gap=0.0, gap_open=math.inf)  # only loops feeding blocked exits
    assert ctrl().decide(blocked) == HOP
    c = ctrl(box=True)
    assert c.decide(blocked) == S_LEFT
    assert c.decide(rob(HOP, 10.0, gap=0.5, gap_open=2.0)) == HOP
    assert c.decide(rob(HOP, 10.0, gap=0.5, gap_open=3.0)) == S_LEFT
    assert c.decide(rob(HOP, 50.0, gap=0.0, gap_open=math.inf)) == S_LEFT  # max-out
    assert c.report()["box_gapouts"] == 1  # once per green, and not for the max-out
    (ev,) = c.events
    assert (ev.rule, ev.stage_from, ev.stage_to, ev.source) == ("box_gapout", 0, 1, "controller")


def test_box_gap_out_is_not_logged_when_tsp_holds_the_green():
    c = ctrl(tsp=TSP, box=True)
    o = rob(BUS_91, 6.0, gap=0.0, gap_open=math.inf, calls=[bus(dist=7.0)])
    assert c.decide(o) == BUS_91
    assert [e.rule for e in c.events] == ["tsp_extend"]


# -- pedestrian anticipation ----------------------------------------------------------
def test_anticipation_leaves_when_the_slack_is_below_one_step():
    c = ctrl(anticipate=True)
    assert c.decide(rob(HOP, 20.0, gap=0.0, slack=(30.0, 1.0))) == HOP  # slack == dt: stay
    assert c.decide(rob(HOP, 21.0, gap=0.0, slack=(30.0, 0.999))) == S_LEFT
    assert c.decide(rob(HOP, 21.0, gap=0.0, slack=(math.inf, math.inf))) == HOP
    assert c.decide(rob(HOP, 21.0, gap=0.0, slack=())) == HOP  # no crossing
    assert ctrl().decide(rob(HOP, 21.0, gap=0.0, slack=(0.0,))) == HOP  # off
    dt2 = RuleBasedController(MINS, S2_MAX, anticipate=True, dt=2.0)
    assert dt2.decide(rob(HOP, 21.0, gap=0.0, slack=(1.5,))) == S_LEFT


def test_anticipation_wins_over_the_tsp_extension():
    c = ctrl(tsp=TSP, anticipate=True)
    assert c.decide(rob(BUS_91, 6.0, calls=[bus()], slack=(0.5,))) == BUS_57
    assert [e.rule for e in c.events] == ["ped_protect"]
    assert c.report()["tsp"]["extended_greens"] == 0 and c.report()["tsp"]["extension_s"] == 0


# -- events and report ----------------------------------------------------------------
def test_events_are_logged_once_per_green():
    c = ctrl(anticipate=True)
    c.reset(101)
    for tis in (20.0, 21.0, 22.0):  # the guard may hold the green: same green, one event
        assert c.decide(rob(HOP, tis, gap=0.0, slack=(0.5,), t=tis)) == S_LEFT
    c.decide(rob(HOP, 22.0, transition=True, t=23.0))
    c.decide(rob(S_LEFT, 0.0, t=30.0))
    c.decide(rob(S_LEFT, 6.0, t=36.0))
    c.decide(rob(HOP, 20.0, gap=0.0, slack=(0.2,), t=120.0))  # a later HOP green
    assert [(e.t, e.rule, e.stage_from, e.stage_to) for e in c.events] == [
        (20.0, "ped_protect", 0, 1),
        (120.0, "ped_protect", 0, 1),
    ]
    assert c.events[0].source == "controller" and "slack=0.5" in c.events[0].detail
    assert c.report()["ped_protect"] == 2
    c.reset(102)
    assert c.events == [] and c.report()["ped_protect"] == 0


def test_report_schema(rules_scenario):
    rep = make_controller(rules_scenario).report()
    assert set(rep) == {"ped_protect", "box_gapouts", "tsp"}
    assert set(rep["tsp"]) == {
        "calls", "extended_greens", "extension_s", "by_line", "truncated_greens",
        "truncation_s",
    }  # fmt: skip
    assert set(rep["tsp"]["by_line"]) == {"91", "24", "57"}
    for v in rep["tsp"]["by_line"].values():
        assert v == {"calls": 0, "extended_greens": 0, "extension_s": 0.0}


# -- config ---------------------------------------------------------------------------
def test_from_config_reads_the_s2_parameters(rules_scenario):
    c = make_controller(rules_scenario)
    assert isinstance(c, RuleBasedController) and c.name == "rules"
    assert c.min_green_s == MINS and c.max_green_s == S2_MAX and c.max_gap_s == 3.0
    assert c.box and c.anticipate and c.dt == 1.0 and c.stage_names == NAMES
    t = c.tsp
    assert t.enabled and t.lines == ("91", "24", "57") and not t.truncation
    assert t.extend_stages == {S_LEFT, BUS_91, BUS_57} and t.extension_max_s == 10.0
    assert (t.approach_speed_mps, t.start_penalty_s) == (7.0, 4.0)
    assert t.truncation_max_green_s == (38.0, 6.0, 22.0, 6.0, 6.0)


def test_defaults_when_the_rule_blocks_are_absent(rules_cfg):
    for key in ("tsp", "box", "pedestrian", "max_gap_s"):
        del rules_cfg["rules"][key]
    c = RuleBasedController.from_config(rules_cfg)
    assert not c.tsp.enabled and not c.box and not c.anticipate and c.max_gap_s == 3.0


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda c: c.pop("detectors"), r"rules needs detectors\.control"),
        (lambda c: c["rules"].pop("max_green_s"), r"rules\.max_green_s"),
        (lambda c: c["rules"]["min_green_s"].update(HOP=5), r"\['HOP'\] below safety\.min"),
        (lambda c: c["rules"]["max_green_s"].update(HOP=61), r"\['HOP'\] above safety\.max"),
        (lambda c: c["rules"]["min_green_s"].update(STM=31), r"min green above max.*STM"),
        (lambda c: c["rules"].update(max_gap_s=0), r"max_gap_s must be positive"),
        (lambda c: c["rules"].update(maxgap=3), r"rules: unknown keys \['maxgap'\]"),
        (lambda c: c["rules"]["tsp"].update(extend_stages=["S_LEFT", "NOPE"]),
         r"unknown stages \['NOPE'\]"),
        (lambda c: c["rules"]["tsp"].update(lines=["91", "99"]), r"\['99'\] not in the demand"),
        (lambda c: c["rules"]["tsp"].update(lines=[]), r"at least one line"),
        (lambda c: c["rules"]["tsp"].update(extension_max_s=55), r"\['S_LEFT', 'BUS_91',"),
        (lambda c: c["rules"]["tsp"].update(approach_speed_mps=0), r"approach_speed_mps > 0"),
        (lambda c: c["rules"]["tsp"]["truncation"]["max_green_s"].update(HOP=51),
         r"truncation\.max_green_s of \['HOP'\] above rules\.max_green_s"),
        (lambda c: c["rules"]["tsp"]["truncation"]["max_green_s"].update(STM=5),
         r"truncation\.max_green_s of \['STM'\] below"),
        (lambda c: c["rules"]["tsp"]["truncation"].update(enabled=True, max_green_s=None),
         r"truncation\.enabled needs"),
        (lambda c: c["rules"]["tsp"].update(extention_max_s=5), r"unknown keys"),
        (lambda c: c["rules"]["box"].update(enable=True), r"rules\.box: unknown keys"),
    ],
)  # fmt: skip
def test_from_config_errors(rules_cfg, edit, match):
    edit(rules_cfg)
    with pytest.raises(ConfigError, match=match):
        RuleBasedController.from_config(rules_cfg)


@pytest.mark.parametrize(
    "mins, maxs, kw",
    [([], [], {}), ([6], [20, 30], {}), ([6, 31], [20, 30], {}), ([6, 6], [20, 30], {"dt": 0}),
     ([6, 6], [20, 30], {"tsp": TspParams(extend_stages=frozenset({2}))}),
     ([6, 6], [20, 30], {"tsp": TspParams(truncation=True)}),
     ([6, 6], [20, 30], {"stage_names": ["A"]})],
)  # fmt: skip
def test_constructor_rejects_bad_values(mins, maxs, kw):
    with pytest.raises(ValueError):
        RuleBasedController(mins, maxs, **kw)


# -- scenario configs (D19) and the rules.yaml derivations (D23, D36) -----------------
def test_s2_differs_from_s0_only_by_its_control_keys(rules_scenario, pitie_scenario):
    diff = {k for k in rules_scenario.keys() | pitie_scenario.keys()
            if rules_scenario.get(k) != pitie_scenario.get(k)}  # fmt: skip
    assert diff == {"scenario", "controller", "rules", "preemption", "safety"}
    safety = {k for k in rules_scenario["safety"]
              if rules_scenario["safety"][k] != pitie_scenario["safety"].get(k)}  # fmt: skip
    assert safety == {"max_red_s"} and rules_scenario["safety"]["max_red_s"] == 180
    assert demand_hash(rules_scenario) == demand_hash(pitie_scenario)


def test_s2_without_preemption_differs_only_by_preemption(rules_scenario):
    nop = load_scenario(CONFIG_DIR / "pitie_rules_nopreempt.yaml")
    assert nop["scenario"] == "pitie_rules_nopreempt" and nop["controller"] == "rules"
    assert nop["preemption"]["enabled"] is False and rules_scenario["preemption"]["enabled"]
    strip = ("scenario", "preemption")
    assert {k: v for k, v in nop.items() if k not in strip} == {
        k: v for k, v in rules_scenario.items() if k not in strip
    }
    assert {k: v for k, v in nop["preemption"].items() if k != "enabled"} == {
        k: v for k, v in rules_scenario["preemption"].items() if k != "enabled"
    }


def s2_greens(cfg: dict, base: str) -> list[float]:
    """Per-stage greens: `rules.<base>`, plus the TSP extension on the extended stages
    when base is max_green_s."""
    names, r = stage_names(cfg), cfg["rules"]
    greens = [float(r[base][n]) for n in names]
    if base == "max_green_s":
        ext = float(r["tsp"]["extension_max_s"])
        extended = r["tsp"]["extend_stages"]
        greens = [g + ext * (n in extended) for g, n in zip(greens, names, strict=True)]
    return greens


def test_rules_yaml_derivations(rules_scenario):
    """The numbers in the configs/pitie/rules.yaml comment, recomputed from the program."""
    from traffic_twin.sim.tls_check import (
        load_program,
        scenario_stages,
        with_greens,
        worst_case_red,
    )

    cfg = rules_scenario
    net = cfg["network"]
    prog = load_program(REPO_ROOT / net["net_file"], net["tls_id"])
    stages = scenario_stages(cfg)
    links = {name: c["link"] for name, c in net["crossings"].items()}
    theta = cfg["safety"]["ped_max_wait_s"] - cfg["safety"]["ped_red_margin_s"]
    assert theta == 88
    worst = s2_greens(cfg, "max_green_s")
    assert worst == [50, 16, 30, 16, 16]
    red = worst_case_red(prog, stages, worst)
    assert {n: red[i] for n, i in links.items()} == {"NB": 62, "S": 120, "WB": 62, "W": 126}
    low = worst_case_red(prog, stages, s2_greens(cfg, "min_green_s"))
    assert {n: low[i] for n, i in links.items()} == {"NB": 42, "S": 56, "WB": 42, "W": 72}
    assert max(low[i] for i in links.values()) <= theta < max(red[i] for i in links.values())
    clearance = sum(prog.durations[p] for st in stages for p in st.clearance_phases)
    assert clearance == 49
    cycle = sum(with_greens(prog, stages, worst).durations)
    assert cycle == 177 < cfg["safety"]["max_red_s"] == 180
    # the S1 maxima are the configured penalty cap
    trunc = cfg["rules"]["tsp"]["truncation"]["max_green_s"]
    assert trunc == cfg["actuated"]["max_green_s"]

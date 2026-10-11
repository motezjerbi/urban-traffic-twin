"""S1 actuated controller: gap-out / extension / max-out logic, config checks, greens
through SignalPlan, the pedestrian bound on its max greens and its control loops."""

import copy
import math
import random
import re
from pathlib import Path

import pytest

from traffic_twin.config import ConfigError, load_config, stage_names
from traffic_twin.control.actuated import ActuatedController
from traffic_twin.control.base import EXTRA_GAP_S, Observation
from traffic_twin.control.registry import make_controller
from traffic_twin.paths import PITIE_ACTUATED_CONFIG
from traffic_twin.sim.signal import SignalPlan, Stage

REPO_ROOT = Path(__file__).resolve().parent.parent

# phases: 0 green A, 1 yellow A, 2 all-red, 3 green B, 4 yellow B
DURATIONS = [30, 3, 2, 30, 3]
STAGES = [Stage("A", 0, (1, 2)), Stage("B", 3, (4,))]


def obs(stage=0, tis=0.0, gap=math.inf, transition=False, n=2) -> Observation:
    gaps = [math.inf] * n
    gaps[stage] = gap
    return Observation(0.0, stage, tis, transition, extras={EXTRA_GAP_S: tuple(gaps)})


@pytest.fixture
def ctrl() -> ActuatedController:
    return ActuatedController([6, 5], [20, 15], max_gap_s=3.0)


@pytest.fixture
def grid_actuated(scenario_cfg) -> dict:
    """The grid scenario with S1 sections (analyst variant, max greens 40/12/25/10)."""
    scenario_cfg["controller"] = "actuated"
    scenario_cfg["detectors"] = {"control": {"distance_m": None}}
    scenario_cfg["actuated"] = {
        "max_gap_s": 3.0,
        "min_green_s": {"EW": 6, "EW_LEFT": 6, "NS": 6, "NS_LEFT": 6},
        "max_green_s": {"EW": 40, "EW_LEFT": 12, "NS": 25, "NS_LEFT": 10},
    }
    return scenario_cfg


# -- decisions -------------------------------------------------------------------------
def test_holds_min_green_without_detection(ctrl):
    assert [ctrl.decide(obs(0, t)) for t in range(6)] == [0] * 6
    assert ctrl.decide(obs(0, 6)) == 1  # nothing detected: gap-out at min green
    assert ctrl.decide(obs(1, 4.0)) == 1  # stage B: min 5 s


def test_gaps_out_at_the_first_step_with_gap_at_least_max_gap(ctrl):
    assert ctrl.decide(obs(0, 10, gap=2.9)) == 0
    assert ctrl.decide(obs(0, 11, gap=3.0)) == 1
    assert ctrl.decide(obs(0, 12, gap=7.0)) == 1


def test_extends_while_vehicles_keep_coming(ctrl):
    for t in range(6, 20):
        assert ctrl.decide(obs(0, t, gap=t % 3)) == 0


def test_maxes_out_exactly_at_max_green(ctrl):
    assert ctrl.decide(obs(0, 19, gap=0.0)) == 0
    assert ctrl.decide(obs(0, 20, gap=0.0)) == 1
    assert ctrl.decide(obs(1, 15, gap=0.0)) == 0  # B -> A wraps around


def test_returns_the_stage_being_left_during_clearance(ctrl):
    assert ctrl.decide(obs(0, 25, gap=9.0, transition=True)) == 0
    assert ctrl.decide(obs(1, 3, transition=True)) == 1


def test_only_ever_holds_or_moves_to_the_next_stage():
    rng = random.Random(0)
    ctrl = ActuatedController([6, 6, 6], [30, 12, 20])
    for _ in range(2000):
        s = rng.randrange(3)
        gaps = tuple(rng.uniform(0, 6) for _ in range(3))
        o = Observation(0.0, s, rng.uniform(0, 40), rng.random() < 0.2, extras={EXTRA_GAP_S: gaps})
        assert ctrl.decide(o) in (s, (s + 1) % 3)


def test_missing_gaps_is_a_config_error(ctrl):
    assert ctrl.decide(Observation(0.0, 0, 3.0, False)) == 0  # min green: no loops needed
    with pytest.raises(ConfigError, match=r"actuated needs detectors\.control"):
        ctrl.decide(Observation(0.0, 0, 8.0, False))


@pytest.mark.parametrize(
    "mins, maxs, gap",
    [([], [], 3.0), ([6], [20, 30], 3.0), ([6, 0], [20, 30], 3.0), ([6, 31], [20, 30], 3.0),
     ([6, 6], [20, 30], 0.0)],
)  # fmt: skip
def test_constructor_rejects_bad_values(mins, maxs, gap):
    with pytest.raises(ValueError):
        ActuatedController(mins, maxs, gap)


# -- config ----------------------------------------------------------------------------
def test_from_config_reads_the_stage_dicts(grid_actuated):
    ctrl = make_controller(grid_actuated)
    assert isinstance(ctrl, ActuatedController) and ctrl.name == "actuated"
    assert ctrl.min_green_s == [6] * 4 and ctrl.max_green_s == [40, 12, 25, 10]
    assert ctrl.max_gap_s == 3.0


def test_pitie_actuated_config(pitie_scenario):
    cfg = load_config(REPO_ROOT / PITIE_ACTUATED_CONFIG)
    ctrl = make_controller(cfg)
    assert isinstance(ctrl, ActuatedController)
    assert stage_names(cfg) == ["HOP", "S_LEFT", "STM", "BUS_91", "BUS_57"]
    assert ctrl.max_green_s == [38, 6, 22, 6, 6] and ctrl.min_green_s == [6] * 5
    # S1 differs from S0 only by its name and controller (spec 13.3: same demand)
    same = {k: v for k, v in cfg.items() if k not in ("scenario", "controller")}
    assert same == {k: v for k, v in pitie_scenario.items() if k not in ("scenario", "controller")}


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda c: c["actuated"]["max_green_s"].pop("NS"), r"no value for stages \['NS'\]"),
        (lambda c: c["actuated"].pop("min_green_s"), r"actuated\.min_green_s"),
        (lambda c: c["actuated"]["min_green_s"].update(EW=5), r"\['EW'\] below safety\.min"),
        (lambda c: c["safety"].update(max_green_s=30), r"\['EW'\] above safety\.max_green_s"),
        (lambda c: c["actuated"]["max_green_s"].update(NS_LEFT=5), r"above max green.*NS_LEFT"),
        (lambda c: c.pop("detectors"), r"actuated needs detectors\.control"),
    ],
)
def test_from_config_errors(grid_actuated, edit, match):
    edit(grid_actuated)
    with pytest.raises(ConfigError, match=match):
        ActuatedController.from_config(grid_actuated)


# -- through SignalPlan ----------------------------------------------------------------
@pytest.mark.parametrize("seed", range(3))
def test_greens_through_signal_plan_stay_within_min_and_max(seed):
    rng = random.Random(seed)
    mins, maxs = [6, 5], [20, 15]
    ctrl = ActuatedController(mins, maxs)
    plan = SignalPlan(STAGES, DURATIONS, min_green_s=5)
    shown, gaps = [], [0.0, 0.0]
    for t in range(3000):
        gaps = [0.0 if rng.random() < 0.5 else g + 1.0 for g in gaps]
        o = Observation(t, plan.stage, plan.time_in_stage, plan.in_transition,
                        extras={EXTRA_GAP_S: tuple(gaps)})  # fmt: skip
        shown.append(plan.request(ctrl.decide(o)))
        plan.advance(1.0)
    runs: list[list[int]] = []
    for p in shown:
        if runs and runs[-1][0] == p:
            runs[-1][1] += 1
        else:
            runs.append([p, 1])
    lengths = {0: [], 3: []}
    for p, n in runs[:-1]:  # the last run may be cut
        if p in lengths:
            lengths[p].append(n)
    for k, phase in enumerate((0, 3)):
        assert mins[k] <= min(lengths[phase]) and max(lengths[phase]) <= maxs[k]
        assert len(set(lengths[phase])) > 3  # greens really vary
    assert maxs[0] in lengths[0]  # some greens max out


# -- scenarios: pedestrian bound and control loops -------------------------------------
def actuated_scenarios() -> list[Path]:
    """Every top-level scenario config whose controller is the actuated one."""
    out = []
    for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        cfg = load_config(path)
        if cfg.get("controller") == "actuated":
            out.append(path)
    return out


def test_there_is_an_actuated_scenario():
    assert PITIE_ACTUATED_CONFIG.name in [p.name for p in actuated_scenarios()]


def crossing_red(cfg: dict, greens_by_name: dict[str, float]) -> dict[int, float]:
    """Worst-case red per crossing link with each stage green for the given time."""
    from traffic_twin.sim.tls_check import load_program, worst_case_red

    net = cfg["network"]
    stages = [Stage(s["name"], s["green_phase"], tuple(s["clearance_phases"]))
              for s in net["stages"]]  # fmt: skip
    prog = load_program(REPO_ROOT / net["net_file"], net["tls_id"])
    red = worst_case_red(prog, stages, [greens_by_name[s.name] for s in stages])
    links = [c["link"] for c in net.get("crossings", {}).values()]
    links = links or [i for i, lk in prog.links.items() if lk.is_crossing]
    assert links, "no pedestrian crossing in the program"
    return {i: red[i] for i in links}


def ped_bound(cfg: dict) -> float:
    safety = cfg["safety"]
    return safety["ped_max_wait_s"] - safety.get("ped_red_margin_s", 0)


@pytest.mark.parametrize("path", actuated_scenarios(), ids=lambda p: p.stem)
def test_max_greens_keep_every_crossing_red_within_the_pedestrian_bound(path):
    cfg = load_config(path)
    red = crossing_red(cfg, cfg["actuated"]["max_green_s"])
    assert max(red.values()) <= ped_bound(cfg), red


def test_pitie_max_greens_leave_no_slack():
    """The derivation in configs/pitie/control.yaml: the largest max greens under the
    88 s bound (HOP first, then STM), so +1 s on any stage breaks it."""
    cfg = load_config(REPO_ROOT / PITIE_ACTUATED_CONFIG)
    maxs = cfg["actuated"]["max_green_s"]
    assert max(crossing_red(cfg, maxs).values()) == ped_bound(cfg) == 88
    for name in maxs:
        bumped = {**maxs, name: maxs[name] + 1}
        assert max(crossing_red(cfg, bumped).values()) > 88, name


def test_pitie_control_loops_per_stage(pitie_scenario):
    """SUMO's rule (every link of the lane green, one at least G) on the 123 s program. HOP is
    extended by N-straight traffic only: s_in_1 also carries the 57N bus link and s_in_2
    the S left, both red in HOP."""
    from traffic_twin.sim.detectors import plan_detectors

    layout = plan_detectors(copy.deepcopy(pitie_scenario))
    lane_of = {lp.id: lp.lane for lp in layout.loops}
    by_stage = {
        name: sorted(lane_of.get(d, d) for d in ids)
        for name, ids in zip(stage_names(pitie_scenario), layout.stage_loops, strict=True)
    }
    assert by_stage == {
        "HOP": ["n_in_2"],
        "S_LEFT": ["s_in_1", "s_in_2"],
        "STM": ["n_in_1", "w_in_1"],
        "BUS_91": ["n_in_1", "wb_in_1"],
        "BUS_57": ["n_in_1", "nb_in_0"],
    }


def test_pitie_control_loops_sit_where_control_yaml_says(pitie_scenario):
    """SUMO's default loop distance on the 30 km/h Pitié lanes is the gap term,
    2 s x 8.33 m/s = 16.66 m; the 7.5 m term (27.43 m for a 6 s min green) binds only on
    the 50 km/h grid. The comment in configs/pitie/control.yaml states the distance."""
    from traffic_twin.sim.detectors import detector_distance, plan_detectors, read_net

    cfg = copy.deepcopy(pitie_scenario)
    control = cfg["detectors"]["control"]
    assert control["distance_m"] is None  # SUMO's default distance
    gap, passing = control["detector_gap_s"], control["passing_time_s"]
    net = read_net(REPO_ROOT / cfg["network"]["net_file"])
    loops = plan_detectors(cfg, net).control_loops
    assert len(loops) == 7
    expected = gap * 8.33
    for g in cfg["actuated"]["min_green_s"].values():
        assert detector_distance(8.33, g, gap, passing) == expected < 27.43  # gap term binds
    for lp in loops:
        length, speed = net.lanes[lp.lane]
        assert speed == 8.33, lp.lane
        assert length - lp.pos == pytest.approx(expected, abs=1e-9), lp.id

    text = (REPO_ROOT / "configs" / "pitie" / "control.yaml").read_text(encoding="utf-8")
    stated = re.findall(r"loops (\d+(?:\.\d+)?) m before the stop line", text)
    assert stated == [f"{expected:.2f}"]

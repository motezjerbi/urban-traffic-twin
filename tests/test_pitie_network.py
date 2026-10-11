"""The committed Pitié net and its provisional program carry the calibrated demand
safely (spec 8.3 EF-01, A-04). Native SUMO, junction collisions checked.

With a route file written here:

- smoke: 30 min at the 08:00-09:00 level, buses twice as frequent as configured,
  150 pedestrians/h on every walk and the ambulance: no collision, teleport, emergency
  stop or braking, no pedestrian jam, everyone arrives, and every walk takes only its
  crossing: route <= crossing + 20 m (two 5 m pinned sidewalk ends and the two walking
  areas; a detour over another crossing or round a boundary node is far longer);
- capacity: 1 h at N 660, S 397, W 229 veh/h (sensors 508/511/513, school-term Tue-Thu)
  with the configured buses and pedestrians: >= 97 % of the cars generated in the hour
  are inserted in it and their mean insertion delay is <= 30 s on every arm.

With the scenario's own demand files (`sim.demand`: the calibrated 07:30-09:30 slot
profile, the bus lines with their in-lane stops, the walks and the ambulance) under the
committed static program, which is S0:

- calibrated profile, seeds 101-110 (the evaluation length; never the evaluation seeds
  1-10, D11): safe on every seed. An arm whose mean insertion delay passes 30 s on some
  seed has a queue that spills past the modelled approach. It must be declared near or
  at capacity in `network.assumptions`, which docs/validation.md renders, and still be
  uncongested on the median seed;
- N straight saturation: N is scaled up 1.6x so that N straight stays queued from
  07:45 on. Its discharge with and without the in-lane 57 S stop, and its calibrated
  demand, must match the figures in that declaration.
"""

from __future__ import annotations

import copy
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean, median

import pytest

from traffic_twin.config import approach_in_edges, approach_out_edges
from traffic_twin.sim.sumo import DRAIN_S, find_binary

pytestmark = pytest.mark.sumo

REPO_ROOT = Path(__file__).resolve().parent.parent
PEAK_VEH_H = {"N": 660.0, "S": 397.0, "W": 229.0}  # 08:00-09:00 term mean (D8)
CALIBRATION_SEEDS = tuple(range(101, 111))  # calibration side of D11, never seeds 1-10
SATURATION_SEEDS = (101, 102, 103, 104, 105)
MAX_MEAN_DEPART_DELAY_S = 30.0
OVERSATURATION = 1.6  # N arm scale: N straight queued from about 07:35 on
SATURATED_FROM_S = 900  # count the N straight discharge from 07:45 (begin + 15 min)
PEAK_FROM_S = 1800  # N straight demand of the declaration: 08:00 (begin + 30 min) to end
NEAR_CAPACITY = re.compile(r"(\w+) \w+ (?:near|at) capacity\b")  # "N straight at capacity"
DECLARED_FIGURES = re.compile(
    r"about (\d+) veh/h of demand against about (\d+) veh/h discharged when it is "
    r"saturated \(about (\d+) veh/h without the stop\), a degree of saturation of about "
    r"(\d+(?:\.\d+)?)"
)
TURNS = {"N": {"S": 0.63, "W": 0.37}, "S": {"N": 0.90, "W": 0.10}, "W": {"N": 0.85, "S": 0.15}}
VTYPES = """    <vType id="car" vClass="passenger" speedDev="0.1"/>
    <vType id="bus" vClass="bus" length="12" maxSpeed="13.89"/>
    <vType id="ambulance" vClass="emergency" guiShape="emergency" speedFactor="1.3"/>"""


def _routes(cfg: dict, end: int, bus_factor: float, ped_h: float, ambulance_s: int | None):
    """Route file text: car flows per movement, bus lines, pinned walks, an ambulance,
    sorted by departure time (SUMO drops elements out of order)."""
    ins, outs = approach_in_edges(cfg), approach_out_edges(cfg)
    items: list[tuple[float, str]] = []
    for arm, q in PEAK_VEH_H.items():
        for dest, share in TURNS[arm].items():
            items.append(
                (0, f'    <flow id="car_{arm}_{dest}" type="car" from="{ins[arm][0]}" '
                    f'to="{outs[dest][0]}" begin="0" end="{end}" '
                    f'probability="{q * share / 3600:.6f}" departLane="best" departSpeed="max"/>')
            )  # fmt: skip
    for b in cfg["demand"]["bus"]["lines"]:
        stops = "".join(
            f'<stop busStop="{s["stop"]}" duration="{s["dwell_s"]}"/>' for s in b["stops"]
        )
        begin = b["offset_s"] / bus_factor
        items.append(
            (begin, f'    <flow id="bus_{b["id"]}" type="bus" line="{b["line"]}" '
                    f'from="{b["from"]}" to="{b["to"]}" begin="{begin:g}" end="{end}" '
                    f'period="{b["headway_s"] / bus_factor:g}" departLane="best">{stops}</flow>')
        )  # fmt: skip
    for name, c in cfg["network"]["crossings"].items():
        for k, w in enumerate(c["walks"]):
            items.append(
                (0, f'    <personFlow id="ped_{name}_{k}" begin="0" end="{end}" '
                    f'probability="{ped_h / 3600:.6f}" departPos="{w["depart_pos"]}">'
                    f'<walk from="{w["from"]}" to="{w["to"]}" arrivalPos="{w["arrival_pos"]}"/>'
                    "</personFlow>")
            )  # fmt: skip
    if ambulance_s is not None:
        route = " ".join(cfg["emergency"]["route"])
        items.append(
            (ambulance_s, f'    <vehicle id="ambulance_0" type="ambulance" depart="{ambulance_s}" '
                          f'departLane="best"><route edges="{route}"/></vehicle>')
        )  # fmt: skip
    body = [text for _, text in sorted(items, key=lambda x: x[0])]
    return "<routes>\n" + VTYPES + "\n" + "\n".join(body) + "\n</routes>\n"


def _stops(cfg: dict) -> str:
    stops = [
        f'    <busStop id="{sid}" lane="{s["lane"]}" startPos="{s["start"]}" endPos="{s["end"]}"/>'
        for sid, s in cfg["network"]["bus_stops"].items()
    ]
    return "<additional>\n" + "\n".join(stops) + "\n</additional>\n"


def _scenario_demand(cfg: dict) -> tuple[str, str]:
    """Route and stop file texts that a run of `cfg` uses (`sim.demand`), with the
    calibration file read from the repo whatever the working directory."""
    from traffic_twin.sim.demand import build_additional, build_routes

    cfg = copy.deepcopy(cfg)
    cfg["demand"]["calibration"] = str(REPO_ROOT / cfg["demand"]["calibration"])
    return build_routes(cfg), build_additional(cfg)


def _window(cfg: dict) -> tuple[int, int]:
    begin = int(cfg["begin_s"])
    return begin, begin + int(cfg["duration_s"])


def _run(
    cfg: dict,
    tmp: Path,
    name: str,
    seed: int,
    routes: str,
    end: int,
    begin: int = 0,
    additional: str | None = None,
) -> dict:
    """Run SUMO from `begin` to `end` (s); returns statistics, tripinfo root and log text.
    The stop file is `additional`, by default every configured bus stop."""
    tmp.mkdir(parents=True, exist_ok=True)
    rou, add = tmp / f"{name}_{seed}.rou.xml", tmp / f"{name}_{seed}.add.xml"
    rou.write_text(routes, encoding="utf-8")
    add.write_text(_stops(cfg) if additional is None else additional, encoding="utf-8")
    out = {k: tmp / f"{name}_{seed}.{k}" for k in ("stats.xml", "trips.xml", "log.txt")}
    subprocess.run(
        [
            find_binary("sumo"), "-n", str(REPO_ROOT / cfg["network"]["net_file"]),
            "-r", str(rou), "-a", str(add), "--begin", str(begin), "--end", str(end),
            "--seed", str(seed), "--collision.check-junctions", "true",
            "--time-to-teleport", "300", "--no-step-log", "true",
            "--statistic-output", str(out["stats.xml"]),
            "--tripinfo-output", str(out["trips.xml"]), "--log", str(out["log.txt"]),
        ],
        check=True, capture_output=True,
    )  # fmt: skip
    stats = ET.parse(out["stats.xml"]).getroot()
    return {
        "stats": {child.tag: dict(child.attrib) for child in stats},
        "trips": ET.parse(out["trips.xml"]).getroot(),
        "log": out["log.txt"].read_text(encoding="utf-8", errors="replace"),
    }


def _run_seeds(seeds: tuple[int, ...], job: Callable[[int], dict]) -> dict[int, dict]:
    """`job(seed)` for every seed, a few SUMO processes at a time."""
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        return dict(zip(seeds, pool.map(job, seeds), strict=True))


def _assert_safe(res: dict, label: object = None) -> None:
    s = res["stats"]
    assert s["teleports"]["total"] == "0", (label, s["teleports"])
    safety = {"collisions": "0", "emergencyStops": "0", "emergencyBraking": "0"}
    assert s["safety"] == safety, (label, s["safety"])
    assert s["persons"]["jammed"] == "0" and "jammed" not in res["log"].lower(), label
    assert s["vehicles"]["running"] == "0" and s["vehicles"]["waiting"] == "0", (
        label,
        s["vehicles"],
    )


def _mean_depart_delay(res: dict, arm: str) -> float:
    """Mean insertion delay (s) of the cars entering from `arm`."""
    trips = res["trips"].findall("tripinfo")
    return fmean(
        float(t.get("departDelay")) for t in trips if t.get("id").startswith(f"car_{arm}_")
    )


def _declared_at_capacity(cfg: dict) -> dict[str, str]:
    """Arm -> its `network.assumptions` entry that declares it near or at capacity, e.g.
    "N straight at capacity at 08:00-09:30 ..."."""
    out = {}
    for text in cfg["network"].get("assumptions", []):
        m = NEAR_CAPACITY.match(text)
        if m:
            out[m.group(1)] = text
    return out


def _general_arms(cfg: dict) -> list[str]:
    approaches = cfg["network"]["approaches"]
    return [k for k, a in approaches.items() if not a.get("bus_only")]


@pytest.fixture(scope="module")
def crossing_length(pitie_scenario) -> dict[str, float]:
    """Crossing name -> length (m) of its crossing edge in the committed net."""
    root = ET.parse(REPO_ROOT / pitie_scenario["network"]["net_file"]).getroot()
    lengths = {e.get("id"): float(e.find("lane").get("length")) for e in root.findall("edge")}
    return {n: lengths[c["edge"]] for n, c in pitie_scenario["network"]["crossings"].items()}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_smoke_peak_with_buses_walks_and_ambulance(pitie_scenario, crossing_length, tmp_path, seed):
    routes = _routes(pitie_scenario, end=1800, bus_factor=2.0, ped_h=150.0, ambulance_s=600)
    res = _run(pitie_scenario, tmp_path, "smoke", seed, routes, end=1800 + 1200)

    _assert_safe(res)
    persons = res["trips"].findall("personinfo")
    assert len(persons) == int(res["stats"]["persons"]["loaded"]) > 0
    assert res["stats"]["persons"]["running"] == "0"
    assert res["trips"].find("tripinfo[@id='ambulance_0']") is not None  # arrived
    for p in persons:
        name = p.get("id").split("_")[1]  # ped_<crossing>_<k>.<n>
        for w in p.iter("walk"):
            assert float(w.get("routeLength")) <= crossing_length[name] + 20.0, p.get("id")


@pytest.mark.parametrize("seed", [101, 102, 103])
def test_capacity_at_peak_level(pitie_scenario, tmp_path, seed):
    ped_h = float(pitie_scenario["demand"]["pedestrians"]["flow_ped_h"])
    routes = _routes(pitie_scenario, end=3600, bus_factor=1.0, ped_h=ped_h, ambulance_s=None)
    res = _run(pitie_scenario, tmp_path, "capacity", seed, routes, end=3600 + 1800)

    _assert_safe(res)
    for arm in PEAK_VEH_H:
        trips = [
            t for t in res["trips"].findall("tripinfo") if t.get("id").startswith(f"car_{arm}_")
        ]
        delay = [float(t.get("departDelay")) for t in trips]
        depart = [float(t.get("depart")) for t in trips]
        generated = sum(d - dd < 3600 for d, dd in zip(depart, delay, strict=True))
        inserted = sum(d < 3600 for d in depart)
        mean_delay = sum(delay) / len(delay)
        assert inserted >= 0.97 * generated, f"{arm}: {inserted}/{generated} inserted in the hour"
        assert mean_delay <= 30.0, f"{arm}: mean departDelay {mean_delay:.1f} s"


# -- the scenario's own demand (calibrated 07:30-09:30 profile) ------------------------
@pytest.fixture(scope="module")
def calibrated_runs(pitie_scenario, tmp_path_factory) -> dict[int, dict]:
    """Seed -> result of the S0 program with the scenario demand, the evaluation window
    plus the drain of `sim.sumo`."""
    routes, additional = _scenario_demand(pitie_scenario)
    begin, end = _window(pitie_scenario)
    tmp = tmp_path_factory.mktemp("calibrated")

    def job(seed: int) -> dict:
        return _run(
            pitie_scenario, tmp, "calibrated", seed, routes, end + DRAIN_S, begin, additional
        )

    return _run_seeds(CALIBRATION_SEEDS, job)


def test_calibrated_profile_is_safe_on_every_seed(calibrated_runs):
    for seed, res in calibrated_runs.items():
        _assert_safe(res, seed)
        assert res["stats"]["persons"]["running"] == "0", seed
        assert res["trips"].find("tripinfo[@id='ambulance_0']") is not None, seed


def test_arms_over_the_insertion_delay_bound_are_declared_at_capacity(
    pitie_scenario, calibrated_runs
):
    """At the evaluation demand an arm whose cars wait more than 30 s on average to enter
    the net on some seed has its queue past the modelled approach: a capacity shortfall
    of the provisional network and program that docs/validation.md must state (section
    3). A declared arm must still be uncongested on the median seed."""
    declared = _declared_at_capacity(pitie_scenario)
    for arm in _general_arms(pitie_scenario):
        delay = {s: round(_mean_depart_delay(r, arm), 1) for s, r in calibrated_runs.items()}
        over = {s: d for s, d in delay.items() if d > MAX_MEAN_DEPART_DELAY_S}
        if arm in declared:
            assert median(delay.values()) <= MAX_MEAN_DEPART_DELAY_S, (arm, delay)
        else:
            assert not over, (
                f"{arm}: mean departDelay above {MAX_MEAN_DEPART_DELAY_S:g} s on seeds {over}; "
                f"declare '{arm} <movement> at capacity ...' in network.assumptions"
            )


# -- N straight saturation -------------------------------------------------------------
def _n_straight_discharge(cfg: dict, tmp: Path, name: str) -> float:
    """N straight discharge (veh/h) while queued: N scaled up OVERSATURATION times, cars
    N -> S arriving at the end of s_out from 07:45 to the end of the window, mean of
    SATURATION_SEEDS."""
    cfg = copy.deepcopy(cfg)
    scale = cfg["demand"].get("arm_scale") or {}
    cfg["demand"]["arm_scale"] = {**scale, "N": OVERSATURATION * float(scale.get("N", 1.0))}
    routes, additional = _scenario_demand(cfg)
    begin, end = _window(cfg)
    first = begin + SATURATED_FROM_S

    def job(seed: int) -> dict:
        return _run(cfg, tmp, name, seed, routes, end, begin, additional)

    rates = []
    for seed, res in _run_seeds(SATURATION_SEEDS, job).items():
        assert res["stats"]["safety"]["collisions"] == "0", (name, seed)
        trips = res["trips"].findall("tripinfo")
        arrivals = [float(t.get("arrival")) for t in trips if t.get("id").startswith("car_N_S_")]
        rates.append(sum(first <= a < end for a in arrivals) * 3600 / (end - first))
    return fmean(rates)


@pytest.fixture(scope="module")
def n_straight(pitie_scenario, tmp_path_factory) -> dict[str, float]:
    """N straight in the model: its demand from 08:00 to the end of the window (veh/h,
    read from the scenario's route file) and its discharge when queued, with the
    configured stops and with the 57 S stop removed."""
    cfg = pitie_scenario
    begin, end = _window(cfg)
    first = begin + PEAK_FROM_S
    vehicles = 0.0
    for flow in ET.fromstring(_scenario_demand(cfg)[0]).iter("flow"):
        if flow.get("id").startswith("car_N_S_"):
            overlap = min(float(flow.get("end")), end) - max(float(flow.get("begin")), first)
            vehicles += max(overlap, 0.0) * float(flow.get("probability"))
    demand = vehicles * 3600 / (end - first)

    no_stop = copy.deepcopy(cfg)
    for line in no_stop["demand"]["bus"]["lines"]:
        line["stops"] = [s for s in line.get("stops") or [] if s["stop"] != "stop_57_S"]
    tmp = tmp_path_factory.mktemp("saturated")
    return {
        "demand": demand,
        "with_stop": _n_straight_discharge(cfg, tmp, "with_stop"),
        "without_stop": _n_straight_discharge(no_stop, tmp, "without_stop"),
    }


def test_declared_n_straight_saturation_matches_the_model(pitie_scenario, n_straight):
    """The figures of the N straight declaration are what the model does: its calibrated
    demand, its discharge when queued with and without the in-lane 57 S stop, and the
    degree of saturation they give. A program, geometry or demand change that moves them
    must update the declaration (docs/validation.md section 3)."""
    text = _declared_at_capacity(pitie_scenario).get("N")
    assert text, "network.assumptions: no 'N straight at capacity ...' entry"
    m = DECLARED_FIGURES.search(text)
    assert m, f"declaration wording changed (update DECLARED_FIGURES): {text}"
    demand, with_stop, without_stop, saturation = (float(x) for x in m.groups())
    model = {k: round(v, 1) for k, v in n_straight.items()}
    assert demand == pytest.approx(n_straight["demand"], rel=0.02), model
    assert with_stop == pytest.approx(n_straight["with_stop"], rel=0.05), model
    assert without_stop == pytest.approx(n_straight["without_stop"], rel=0.05), model
    ratio = n_straight["demand"] / n_straight["with_stop"]
    assert saturation == pytest.approx(ratio, abs=0.03), (model, ratio)
    assert n_straight["with_stop"] < n_straight["without_stop"], model  # dwell blocks s_out

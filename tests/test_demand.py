import copy
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from traffic_twin.config import ConfigError, load_scenario
from traffic_twin.paths import PITIE_ACTUATED_CONFIG, PITIE_FIXED_CONFIG
from traffic_twin.sim.demand import PED_WALKS, TURNS, build_additional, build_routes

REPO_ROOT = Path(__file__).resolve().parent.parent
PITIE_NET = REPO_ROOT / "sumo" / "network" / "pitie" / "pitie.net.xml"


def _car_vph(cfg: dict) -> float:
    root = ET.fromstring(build_routes(cfg))
    return sum(
        float(f.get("probability")) * 3600 for f in root.iter("flow") if f.get("type") == "car"
    )


def test_routes_xml_is_valid_and_matches_flows(base_scenario):
    root = ET.fromstring(build_routes(base_scenario))
    expected = sum(a["flow_veh_h"] for a in base_scenario["demand"]["approaches"].values())
    assert _car_vph(base_scenario) == pytest.approx(expected, rel=1e-3)
    bus = root.find("flow[@type='bus']")
    assert bus.get("period") == str(base_scenario["demand"]["bus"]["headway_s"])
    assert len(list(root.iter("personFlow"))) == len(PED_WALKS)
    assert root.find("vehicle[@id='ambulance_0']") is not None


def test_flows_cover_the_demand_window(base_scenario):
    begin = int(base_scenario["begin_s"])
    end = begin + int(base_scenario["duration_s"])
    root = ET.fromstring(build_routes(base_scenario))
    flows = list(root.iter("flow")) + list(root.iter("personFlow"))
    assert flows
    assert {(int(f.get("begin")), int(f.get("end"))) for f in flows} == {(begin, end)}
    amb = root.find("vehicle[@id='ambulance_0']")
    assert int(amb.get("depart")) == begin + base_scenario["emergency"]["depart_s"]


def test_car_routes_follow_turn_table(base_scenario):
    root = ET.fromstring(build_routes(base_scenario))
    for f in root.iter("flow"):
        if f.get("type") != "car":
            continue
        _, approach, turn = f.get("id").split("_")
        src, dst = f.find("route").get("edges").split()
        assert src == base_scenario["demand"]["approaches"][approach]["edge"]
        assert dst == TURNS[approach][turn]


def test_demand_is_identical_for_every_seed_and_scenario(scenario_cfg):
    """Randomness comes from SUMO's --seed only: the demand file must not change with
    the seed, the controller or the scenario name (spec 13.3)."""
    reference = build_routes(scenario_cfg)
    assert build_routes(scenario_cfg) == reference  # deterministic
    for seed in (1, 2, 3, 12345):
        scenario_cfg["seed"] = seed
        assert build_routes(scenario_cfg) == reference
    scenario_cfg["controller"] = "actuated"
    scenario_cfg["scenario"] = "other_name"
    assert build_routes(scenario_cfg) == reference


def test_multiplier_scales_car_and_pedestrian_demand(base_scenario, scenario_cfg):
    scenario_cfg["demand"]["multiplier"] = 2.0
    assert _car_vph(scenario_cfg) == pytest.approx(2 * _car_vph(base_scenario), rel=1e-3)
    ped = ET.fromstring(build_routes(scenario_cfg)).find("personFlow")
    expected = 2 * base_scenario["demand"]["pedestrians"]["flow_ped_h"] / 3600
    assert float(ped.get("probability")) == pytest.approx(expected, rel=1e-3)


def test_zero_share_turns_are_skipped(scenario_cfg):
    scenario_cfg["demand"]["approaches"]["N"]["turns"] = {"left": 0, "straight": 1, "right": 0}
    ids = {f.get("id") for f in ET.fromstring(build_routes(scenario_cfg)).iter("flow")}
    assert "car_N_straight" in ids
    assert not {"car_N_left", "car_N_right"} & ids


def test_bad_turn_ratios_rejected(scenario_cfg):
    scenario_cfg["demand"]["approaches"]["W"]["turns"]["left"] = 0.5
    with pytest.raises(ValueError, match="turn ratios of W"):
        build_routes(scenario_cfg)


def test_ambulance_can_be_disabled(scenario_cfg):
    scenario_cfg["emergency"]["enabled"] = False
    assert ET.fromstring(build_routes(scenario_cfg)).find("vehicle") is None
    del scenario_cfg["emergency"]
    assert ET.fromstring(build_routes(scenario_cfg)).find("vehicle") is None


def test_bus_stop_fits_on_lane(base_scenario):
    stop = ET.fromstring(build_additional(base_scenario, {"e_out_1": 100})).find("busStop")
    assert float(stop.get("endPos")) <= 100
    assert stop.get("lines") == base_scenario["demand"]["bus"]["line"]
    long_lane = ET.fromstring(build_additional(base_scenario, {"e_out_1": 1000})).find("busStop")
    assert float(long_lane.get("startPos")) == base_scenario["demand"]["bus"]["stop_pos_m"]
    unchecked = ET.fromstring(build_additional(base_scenario)).find("busStop")
    assert float(unchecked.get("startPos")) == base_scenario["demand"]["bus"]["stop_pos_m"]
    with pytest.raises(ConfigError, match="e_out_1"):
        build_additional(base_scenario, {"w_out_1": 100})


def test_write_demand_files(scenario_cfg, tmp_path):
    pytest.importorskip("sumolib")
    from traffic_twin.sim.demand import write_demand

    rou, add = write_demand(scenario_cfg, tmp_path / "demand")
    assert rou.name == f"{scenario_cfg['scenario']}.rou.xml"
    assert add.name == f"{scenario_cfg['scenario']}.stops.add.xml"  # one per scenario
    assert rou.read_text(encoding="utf-8") == build_routes(scenario_cfg)
    assert ET.parse(add).getroot().find("busStop") is not None


# -- calibrated schema (Pitié) ---------------------------------------------------------
SLOTS = [27000 + 900 * k for k in range(8)]
LEVELS = {
    "N": [500, 500, 660, 660, 660, 660, 650, 650],
    "S": [240, 240, 400, 400, 400, 400, 430, 430],
    "W": [140, 140, 230, 230, 230, 230, 220, 220],
}
CAL = {
    "window": {"begin_s": 27000, "end_s": 34200, "slot_s": 900},
    "arms": {
        arm: {"arc": arc, "slots": [{"begin_s": b, "flow_veh_h": q} for b, q in slots]}
        for (arm, slots), arc in zip(
            ((a, zip(SLOTS, qs, strict=True)) for a, qs in LEVELS.items()),
            ("508", "511", "513"),
            strict=True,
        )
    },
    "turns": {
        "source": "assumption",
        "N": {"S": 0.63, "W": 0.37},
        "S": {"N": 0.9, "W": 0.1},
        "W": {"N": 0.85, "S": 0.15},
    },
}


def routes(cfg: dict, cal: dict | None = None) -> ET.Element:
    return ET.fromstring(build_routes(cfg, copy.deepcopy(cal or CAL)))


def car_flows_of(root: ET.Element) -> list[ET.Element]:
    return [f for f in root.iter("flow") if f.get("type") == "car"]


def expected_vehicles(root: ET.Element, prefix: str) -> float:
    return sum(
        float(f.get("probability")) * (int(f.get("end")) - int(f.get("begin")))
        for f in car_flows_of(root)
        if f.get("id").startswith(prefix)
    )


IN_EDGE = {"N": "n_in", "S": "s_in", "W": "w_in"}


def bus_vehicles(cfg: dict, arm: str) -> float:
    """Expected buses entering by `arm` over the run, from the timetable (one every
    headway_s from begin + offset_s), when the sensors count buses; else 0."""
    bus = cfg["demand"]["bus"]
    if not bus.get("counted_by_sensors"):
        return 0.0
    return sum(
        (cfg["duration_s"] - ln.get("offset_s", 0)) / ln["headway_s"]
        for ln in bus["lines"]
        if (ln.get("route") or [ln["from"]])[0] == IN_EDGE[arm]
    )


def test_calibrated_flows_follow_the_slots(pitie_cfg):
    flows = car_flows_of(routes(pitie_cfg))
    assert len(flows) == 6 * 8  # 6 movements x 8 slots
    by_id = {f.get("id"): f for f in flows}
    for k, b in enumerate(SLOTS):
        f = by_id[f"car_N_S_{k:02d}"]
        assert (int(f.get("begin")), int(f.get("end"))) == (b, b + 900)
        assert f.find("route").get("edges") == "n_in s_out"
        assert float(f.get("probability")) == pytest.approx(LEVELS["N"][k] * 0.63 / 3600, abs=1e-6)
        assert f.get("departLane") == "best" and f.get("departSpeed") == "max"
    assert by_id["car_W_S_03"].find("route").get("edges") == "w_in s_out"


def test_calibrated_totals_and_shares(pitie_cfg):
    """Cars + scheduled buses entering by an arm = its calibrated (all-vehicle) flow."""
    root = routes(pitie_cfg)
    hours = pitie_cfg["duration_s"] / 3600
    for arm, qs in LEVELS.items():
        cars = sum(qs) / len(qs) * hours - bus_vehicles(pitie_cfg, arm)
        assert expected_vehicles(root, f"car_{arm}_") == pytest.approx(cars, rel=1e-3)
        for dest, share in CAL["turns"][arm].items():
            got = expected_vehicles(root, f"car_{arm}_{dest}_")
            assert got == pytest.approx(share * cars, rel=1e-3)


def test_sensor_counted_buses_are_taken_out_of_the_car_demand(pitie_cfg):
    """The sensors and the E1 stations count every vehicle, so the 57N buses entering by
    S replace cars instead of coming on top of the full sensor flow."""
    assert pitie_cfg["demand"]["bus"]["counted_by_sensors"] is True
    with_buses = routes(pitie_cfg)
    hours = pitie_cfg["duration_s"] / 3600
    assert bus_vehicles(pitie_cfg, "S") == pytest.approx(14.75)  # 7.5/h from 07:32
    assert bus_vehicles(pitie_cfg, "N") == bus_vehicles(pitie_cfg, "W") == 0
    by_id = {f.get("id"): float(f.get("probability")) for f in car_flows_of(with_buses)}
    # slot 0 (07:30-07:45): the line starts at 07:32, so 7.5 x 780/900 = 6.5 veh/h
    assert by_id["car_S_N_00"] == round((240 - 6.5) * 0.9 / 3600, 6)
    assert by_id["car_S_N_01"] == round((240 - 7.5) * 0.9 / 3600, 6)
    assert by_id["car_N_S_00"] == round(500 * 0.63 / 3600, 6)  # no bus enters by N

    pitie_cfg["demand"]["bus"]["counted_by_sensors"] = False
    plain = routes(pitie_cfg)
    for arm, qs in LEVELS.items():
        assert expected_vehicles(plain, f"car_{arm}_") == pytest.approx(
            sum(qs) / len(qs) * hours, rel=1e-3
        )
    # a slot below the bus flow gives no car at all, never a negative flow
    pitie_cfg["demand"]["bus"]["counted_by_sensors"] = True
    cal = copy.deepcopy(CAL)
    cal["arms"]["S"]["slots"][2]["flow_veh_h"] = 5.0
    ids = {f.get("id") for f in car_flows_of(routes(pitie_cfg, cal))}
    assert "car_S_N_02" not in ids and "car_S_N_03" in ids


def test_arm_scale_and_multiplier_are_applied(pitie_cfg):
    """Scaling applies to the arm's whole sensor flow; bus timetables are never scaled,
    so cars + buses scale by the factor."""
    base = routes(pitie_cfg)
    pitie_cfg["demand"]["arm_scale"] = {"N": 1.2, "S": 1.0, "W": 0.5}
    pitie_cfg["demand"]["multiplier"] = 1.5
    scaled = routes(pitie_cfg)
    for arm, factor in (("N", 1.8), ("S", 1.5), ("W", 0.75)):
        buses = bus_vehicles(pitie_cfg, arm)
        ratio = (expected_vehicles(scaled, f"car_{arm}_") + buses) / (
            expected_vehicles(base, f"car_{arm}_") + buses
        )
        assert ratio == pytest.approx(factor, rel=1e-3)
    rate = 1.5 * pitie_cfg["demand"]["pedestrians"]["flow_ped_h"] / 3600
    assert float(scaled.find("personFlow").get("probability")) == pytest.approx(rate, abs=1e-6)
    pitie_cfg["demand"]["arm_scale"] = {"E": 1.0}
    with pytest.raises(ConfigError, match="arm_scale"):
        routes(pitie_cfg)


def test_zero_shares_are_skipped(pitie_cfg):
    cal = copy.deepcopy(CAL)
    cal["turns"]["S"] = {"N": 1.0, "W": 0.0}
    cal["arms"]["W"]["slots"][0]["flow_veh_h"] = 0
    ids = {f.get("id") for f in car_flows_of(routes(pitie_cfg, cal))}
    assert "car_S_N_00" in ids and not any(i.startswith("car_S_W") for i in ids)
    assert "car_W_N_00" not in ids and "car_W_N_01" in ids


def test_unknown_or_bus_only_destination_raises(pitie_cfg):
    for dest in ("E", "NB", "N"):
        cal = copy.deepcopy(CAL)
        cal["turns"]["N"] = {"S": 0.5, dest: 0.5}
        with pytest.raises(ConfigError, match=f"N -> {dest}"):
            routes(pitie_cfg, cal)
    cal = copy.deepcopy(CAL)
    cal["turns"]["S"] = {"N": 0.5, "W": 0.4}
    with pytest.raises(ConfigError, match="sum to 1"):
        routes(pitie_cfg, cal)
    cal = copy.deepcopy(CAL)
    cal["arms"]["NB"] = cal["arms"]["N"]
    with pytest.raises(ConfigError, match="arm NB"):
        routes(pitie_cfg, cal)


def test_scenario_window_is_clipped_to_the_calibration(pitie_cfg):
    pitie_cfg["duration_s"] = 1200  # 07:30-07:50
    spans = {(int(f.get("begin")), int(f.get("end"))) for f in car_flows_of(routes(pitie_cfg))}
    assert spans == {(27000, 27900), (27900, 28200)}
    pitie_cfg["begin_s"] = 25200  # 07:00: no calibrated demand before 07:30
    with pytest.raises(ConfigError, match="calibration window"):
        routes(pitie_cfg)


def test_bus_lines(pitie_cfg):
    root = routes(pitie_cfg)
    begin = pitie_cfg["begin_s"]
    end = begin + pitie_cfg["duration_s"]
    buses = {f.get("id"): f for f in root.iter("flow") if f.get("type") == "bus"}
    lines = pitie_cfg["demand"]["bus"]["lines"]
    assert set(buses) == {f"bus_{ln['id']}" for ln in lines}
    for ln in lines:
        f = buses[f"bus_{ln['id']}"]
        assert f.get("line") == str(ln["line"])
        assert int(f.get("period")) == ln["headway_s"]
        assert int(f.get("begin")) == begin + ln["offset_s"] and int(f.get("end")) == end
        assert f.find("route").get("edges") == f"{ln['from']} {ln['to']}"
        stops = [(s.get("busStop"), int(s.get("duration"))) for s in f.iter("stop")]
        assert stops == [(s["stop"], s["dwell_s"]) for s in ln["stops"]]


def test_shared_stop_is_written_once_with_its_lines(pitie_cfg):
    root = ET.fromstring(build_additional(pitie_cfg))
    ids = [s.get("id") for s in root.iter("busStop")]
    stops = {s.get("id"): s for s in root.iter("busStop")}
    assert len(ids) == len(stops)
    assert stops["stop_SM_E"].get("lines") == "24 91"  # 91_E and 24_E share it
    assert stops["stop_57_N"].get("lines") == "57"
    spec = pitie_cfg["network"]["bus_stops"]["stop_SM_E"]
    assert stops["stop_SM_E"].get("lane") == spec["lane"]
    assert float(stops["stop_SM_E"].get("startPos")) == spec["start"]
    assert float(stops["stop_SM_E"].get("endPos")) == spec["end"]
    pitie_cfg["demand"]["bus"]["lines"] = [ln for ln in pitie_cfg["demand"]["bus"]["lines"]
                                           if not ln["stops"]]  # fmt: skip
    assert ET.fromstring(build_additional(pitie_cfg)).find("busStop") is None


def test_bus_stops_are_checked(pitie_cfg):
    lanes = {s["lane"]: s["end"] + 1 for s in pitie_cfg["network"]["bus_stops"].values()}
    build_additional(pitie_cfg, lanes)
    short = {**lanes, pitie_cfg["network"]["bus_stops"]["stop_SM_E"]["lane"]: 100.0}
    with pytest.raises(ConfigError, match="stop_SM_E"):
        build_additional(pitie_cfg, short)
    with pytest.raises(ConfigError, match="not in the network"):
        build_additional(pitie_cfg, {})
    pitie_cfg["demand"]["bus"]["lines"][0]["stops"] = [{"stop": "stop_nowhere", "dwell_s": 20}]
    with pytest.raises(ConfigError, match="stop_nowhere"):
        build_additional(pitie_cfg)
    with pytest.raises(ConfigError, match="stop_nowhere"):
        routes(pitie_cfg)


def test_walks_are_pinned_on_the_person_flow(pitie_cfg):
    peds = {p.get("id"): p for p in routes(pitie_cfg).iter("personFlow")}
    crossings = pitie_cfg["network"]["crossings"]
    assert len(peds) == sum(len(c["walks"]) for c in crossings.values())
    for name, c in crossings.items():
        for k, w in enumerate(c["walks"]):
            p = peds[f"ped_{name}_{k}"]
            walk = p.find("walk")
            assert float(p.get("departPos")) == w["depart_pos"] and walk.get("departPos") is None
            assert float(walk.get("arrivalPos")) == w["arrival_pos"]
            assert p.get("arrivalPos") is None
            assert (walk.get("from"), walk.get("to")) == (w["from"], w["to"])
    rate = pitie_cfg["demand"]["pedestrians"]["flow_ped_h"] / 3600
    assert float(peds["ped_S_0"].get("probability")) == pytest.approx(rate, abs=1e-6)
    pitie_cfg["demand"]["pedestrians"]["flow_ped_h"] = {"S": 360}  # one flow per crossing
    peds = {p.get("id"): p for p in routes(pitie_cfg).iter("personFlow")}
    assert set(peds) == {"ped_S_0", "ped_S_1"} and peds["ped_S_0"].get("probability") == "0.1"
    pitie_cfg["demand"]["pedestrians"]["flow_ped_h"] = {"X": 360}
    with pytest.raises(ConfigError, match="unknown crossing"):
        routes(pitie_cfg)


def test_elements_are_sorted_by_departure(pitie_cfg):
    root = routes(pitie_cfg)
    items = [e for e in root if e.tag != "vType"]
    times = [float(e.get("begin", e.get("depart"))) for e in items]
    assert times == sorted(times)
    amb = root.find("vehicle[@id='ambulance_0']")
    assert int(amb.get("depart")) == pitie_cfg["begin_s"] + pitie_cfg["emergency"]["depart_s"]
    assert amb.find("route").get("edges") == " ".join(pitie_cfg["emergency"]["route"])
    assert items.index(amb) < len(items) - 1  # within its time slot, not appended at the end
    # same departure time: cars, then buses, then people
    kinds = [e.tag + e.get("type", "") for e in items if e.get("begin") == "27000"]
    order = ["flowcar", "flowbus", "personFlow"]
    assert [order.index(k) for k in kinds] == sorted(order.index(k) for k in kinds)


def test_demand_is_the_same_for_s0_and_s1(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)  # the calibration path is relative to the repo root
    fixed = load_scenario(PITIE_FIXED_CONFIG)
    actuated = load_scenario(PITIE_ACTUATED_CONFIG)
    assert fixed["controller"] != actuated["controller"]
    assert build_routes(fixed) == build_routes(actuated)
    assert build_additional(fixed) == build_additional(actuated)


def test_assumptions_list_every_assumed_demand_input(pitie_cfg, monkeypatch):
    """docs/validation.md renders `demand.assumptions` as the ASSUMED inputs (plan section
    8: every one listed): turn shares with their values, bus headways and dwell, the bus
    counting convention, and the ambulance route and departure (Q7), as configured."""
    monkeypatch.chdir(REPO_ROOT)
    d = pitie_cfg["demand"]
    lines = d["assumptions"]

    def line(prefix: str) -> str:
        found = [a for a in lines if a.startswith(prefix)]
        assert len(found) == 1, f"no single assumption starting with {prefix!r}"
        return found[0]

    cal = yaml.safe_load(Path(d["calibration"]).read_text(encoding="utf-8"))
    if cal["turns"]["source"] == "assumption":
        shares = line("turn shares")
        for arm in ("N", "S", "W"):
            for dest, share in cal["turns"][arm].items():
                assert f"{arm} to {dest} {share:.2f}" in shares
    assert not any("prior E" in a for a in lines)
    for ln in d["bus"]["lines"]:
        assert f"{ln['line']}: {ln['headway_s']} s" in line("bus headways")
        for st in ln.get("stops") or []:
            assert f"dwell {st['dwell_s']} s" in line("bus dwell")
    if d["bus"].get("counted_by_sensors"):
        rate = sum(3600 / ln["headway_s"] for ln in d["bus"]["lines"] if ln["from"] == "s_in")
        assert f"{rate:g} veh/h" in line("sensors count buses")
    e = pitie_cfg["emergency"]
    depart = pitie_cfg["begin_s"] + e["depart_s"]
    ambulance = line("ambulance")
    assert f"{depart // 3600:02d}:{depart % 3600 // 60:02d}" in ambulance
    assert " to ".join(e["route"]) in ambulance


def test_committed_calibration_drives_the_pitie_demand(pitie_cfg, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    path = Path(pitie_cfg["demand"]["calibration"])
    cal = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = ET.fromstring(build_routes(pitie_cfg))
    hours = pitie_cfg["duration_s"] / 3600
    for arm, a in cal["arms"].items():
        scale = pitie_cfg["demand"]["arm_scale"].get(arm, 1.0) * pitie_cfg["demand"]["multiplier"]
        got = expected_vehicles(root, f"car_{arm}_") + bus_vehicles(pitie_cfg, arm)
        assert got == pytest.approx(a["mean_veh_h"] * scale * hours, rel=1e-3)
    pitie_cfg["demand"]["calibration"] = "configs/calibration/missing.yaml"
    with pytest.raises(ConfigError, match="not found"):
        build_routes(pitie_cfg)


def test_write_pitie_demand(pitie_cfg, tmp_path, monkeypatch):
    pytest.importorskip("sumolib")
    if not PITIE_NET.is_file():
        pytest.skip("Pitié net not built yet (tasks.py net --scenario configs/pitie_fixed.yaml)")
    monkeypatch.chdir(REPO_ROOT)
    from traffic_twin.sim.demand import write_demand

    rou, add = write_demand(pitie_cfg, tmp_path)
    assert (rou.name, add.name) == ("pitie_fixed.rou.xml", "pitie_fixed.stops.add.xml")
    assert rou.read_text(encoding="utf-8") == build_routes(pitie_cfg)
    assert len(list(ET.parse(add).getroot().iter("busStop"))) == 4


@pytest.mark.sumo
def test_pitie_demand_loads_in_sumo(pitie_cfg, tmp_path, monkeypatch):
    """SUMO keeps every element (no 'sorted' or 'ignoring' warning) and the ambulance
    arrives: 45 min from 07:30 on the committed Pitié net."""
    if not PITIE_NET.is_file():
        pytest.skip("Pitié net not built yet (tasks.py net --scenario configs/pitie_fixed.yaml)")
    from traffic_twin.sim.demand import write_demand
    from traffic_twin.sim.sumo import find_binary

    monkeypatch.chdir(REPO_ROOT)
    rou, add = write_demand(pitie_cfg, tmp_path)
    log, trips = tmp_path / "sumo.log", tmp_path / "tripinfo.xml"
    begin = pitie_cfg["begin_s"]
    subprocess.run(
        [find_binary("sumo"), "-n", str(PITIE_NET), "-r", str(rou), "-a", str(add),
         "--begin", str(begin), "--end", str(begin + 2700), "--seed", "101",
         "--no-step-log", "true", "--tripinfo-output", str(trips), "--log", str(log)],
        check=True, capture_output=True,
    )  # fmt: skip
    text = log.read_text(encoding="utf-8", errors="replace").lower()
    assert "sorted" not in text and "ignoring" not in text
    root = ET.parse(trips).getroot()
    amb = root.find("tripinfo[@id='ambulance_0']")
    assert amb is not None and float(amb.get("arrival")) > float(amb.get("depart"))
    ids = {t.get("id").rsplit(".", 1)[0] for t in root.iter("tripinfo")}
    assert {"bus_91_E", "bus_57_N"} <= ids and any(i.startswith("car_W_N_") for i in ids)
    assert root.find("personinfo") is not None

import xml.etree.ElementTree as ET

import pytest

from traffic_twin.sim.demand import PED_WALKS, TURNS, build_additional, build_routes


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
    stop = ET.fromstring(build_additional(base_scenario, lane_length=100)).find("busStop")
    assert float(stop.get("endPos")) <= 100
    assert stop.get("lines") == base_scenario["demand"]["bus"]["line"]
    long_lane = ET.fromstring(build_additional(base_scenario, lane_length=1000)).find("busStop")
    assert float(long_lane.get("startPos")) == base_scenario["demand"]["bus"]["stop_pos_m"]


def test_write_demand_files(scenario_cfg, tmp_path):
    pytest.importorskip("sumolib")
    from traffic_twin.sim.demand import write_demand

    rou, add = write_demand(scenario_cfg, tmp_path / "demand")
    assert rou.name == f"{scenario_cfg['scenario']}.rou.xml"
    assert rou.read_text(encoding="utf-8") == build_routes(scenario_cfg)
    assert ET.parse(add).getroot().find("busStop") is not None

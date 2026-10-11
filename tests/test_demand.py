import copy
import xml.etree.ElementTree as ET

import pytest

from traffic_twin.config import load_config
from traffic_twin.sim.demand import build_additional, build_routes

CFG = load_config("configs/scenario_base.yaml")


def test_routes_xml_is_valid_and_matches_flows():
    root = ET.fromstring(build_routes(CFG))
    total_vph = _car_vph(CFG)
    expected = sum(a["flow_veh_h"] for a in CFG["demand"]["approaches"].values())
    assert total_vph == pytest.approx(expected, rel=1e-3)
    assert root.find("flow[@type='bus']").get("period") == str(CFG["demand"]["bus"]["headway_s"])
    assert len(list(root.iter("personFlow"))) == 6
    assert root.find("vehicle[@id='ambulance_0']") is not None


def _car_vph(cfg: dict) -> float:
    root = ET.fromstring(build_routes(cfg))
    return sum(
        float(f.get("probability")) * 3600 for f in root.iter("flow") if f.get("type") == "car"
    )


def test_multiplier_scales_car_demand():
    cfg = copy.deepcopy(CFG)
    cfg["demand"]["multiplier"] = 2.0
    assert _car_vph(cfg) == pytest.approx(2 * _car_vph(CFG), rel=1e-3)


def test_bad_turn_ratios_rejected():
    cfg = copy.deepcopy(CFG)
    cfg["demand"]["approaches"]["W"]["turns"]["left"] = 0.5
    with pytest.raises(ValueError):
        build_routes(cfg)


def test_ambulance_can_be_disabled():
    cfg = copy.deepcopy(CFG)
    cfg["emergency"]["enabled"] = False
    assert ET.fromstring(build_routes(cfg)).find("vehicle") is None


def test_bus_stop_fits_on_lane():
    stop = ET.fromstring(build_additional(CFG, lane_length=100)).find("busStop")
    assert float(stop.get("endPos")) <= 100

"""Detector layout (control loops, stations, queues, signal switches) on the versioned
networks, the additional file and the output parsers. No SUMO run, no data/processed."""

import math
import xml.etree.ElementTree as ET

import pytest

from traffic_twin.config import ConfigError
from traffic_twin.sim.detectors import (
    DetectorLayout,
    build_additional,
    detector_distance,
    plan_detectors,
    read_e1,
    read_net,
    read_tls_switch,
    write_detectors,
)

GRID_CONTROL = {"control": {"distance_m": None, "detector_gap_s": 2.0, "passing_time_s": 1.9}}
# Analysts' stage map on the grid (SUMO's rule, checked against native actuated)
GRID_STAGE_LOOPS = {
    "EW": {"det_e_in_1", "det_e_in_2", "det_w_in_1", "det_w_in_2"},
    "EW_LEFT": {"det_e_in_3", "det_w_in_3"},
    "NS": {"det_n_in_1", "det_s_in_1"},
    "NS_LEFT": {"det_n_in_2", "det_s_in_2"},
}
# Pitié v7 program (plan, WS4): HOP is extended by N-straight traffic only
PITIE_STAGE_LOOPS = {
    "HOP": {"det_n_in_2"},
    "S_LEFT": {"det_s_in_1", "det_s_in_2"},
    "STM": {"det_n_in_1", "det_w_in_1"},
    "BUS_91": {"det_n_in_1", "det_wb_in_1"},
    "BUS_57": {"det_n_in_1", "det_nb_in_0"},
}


@pytest.fixture
def grid_cfg(scenario_cfg):
    scenario_cfg["detectors"] = GRID_CONTROL
    return scenario_cfg


def _stage_map(cfg, layout):
    names = [s["name"] for s in cfg["network"]["stages"]]
    return {n: set(ids) for n, ids in zip(names, layout.stage_loops, strict=True)}


def test_detector_distance_is_sumos_default():
    # min(detector-gap * v, (minDur / passing-time + 0.5) * 7.5)
    assert detector_distance(13.89, 6) == pytest.approx(27.434210526315788, abs=1e-12)
    assert detector_distance(13.89, 5) == pytest.approx(23.486842105263158, abs=1e-12)
    assert detector_distance(8.33, 6) == 2.0 * 8.33  # the gap cap binds at 30 km/h


def test_grid_control_loops(grid_cfg):
    layout = plan_detectors(grid_cfg)
    loops = layout.control_loops
    assert len(loops) == 10 and len(layout.loops) == 10
    lengths = {
        lane: length
        for lane, (length, _) in read_net(grid_cfg["network"]["net_file"]).lanes.items()
    }
    for lp in loops:
        assert lp.id == f"det_{lp.lane}" and lp.role == "control"
        assert lp.pos == lengths[lp.lane] - 27.434210526315788  # exact float, as SUMO computes
    assert _stage_map(grid_cfg, layout) == GRID_STAGE_LOOPS
    assert not layout.stations and not layout.queues and not layout.switch_links


def test_min_green_per_stage_moves_the_loops(grid_cfg):
    grid_cfg["actuated"] = {"min_green_s": {"EW": 5, "EW_LEFT": 6, "NS": 6, "NS_LEFT": 6}}
    pos = {lp.lane: lp.pos for lp in plan_detectors(grid_cfg).loops}
    assert pos["w_in_2"] == 239.6 - detector_distance(13.89, 5)  # green only in EW
    assert pos["w_in_3"] == 239.6 - detector_distance(13.89, 5)  # 'g' in EW counts (SUMO)
    assert pos["n_in_1"] == 236.4 - detector_distance(13.89, 6)  # green only in NS
    grid_cfg["detectors"]["control"]["distance_m"] = 40.0
    assert {lp.pos for lp in plan_detectors(grid_cfg).loops} == {199.6, 196.4}


def test_stage_without_loop_raises(grid_cfg):
    grid_cfg["network"]["stages"][1]["green_phase"] = 2  # a yellow phase: no lane is G
    with pytest.raises(ConfigError, match="EW_LEFT"):
        plan_detectors(grid_cfg)


def test_no_detectors_for_the_plain_grid(scenario_cfg, tmp_path):
    layout = plan_detectors(scenario_cfg, net="does-not-exist.net.xml")  # net never read
    assert layout.empty and layout.stage_loops == ((), (), (), ())
    assert write_detectors(scenario_cfg, tmp_path) == (None, layout)
    assert not any(tmp_path.iterdir())


def test_stations_and_queues_are_capped(grid_cfg):
    grid_cfg["measurement"] = {
        "period_s": 900,
        "stations": [
            {"id": "far", "lanes": ["n_in_1"], "pos_m": 1000.0},
            {"id": "back", "lanes": ["w_in_1", "w_in_2"], "pos_m": -60.0},
        ],
        "queues": {"N": {"lanes": ["n_in_1", "n_in_2"], "length_m": 500},
                   "W": {"lanes": ["w_in_1"], "length_m": 100}},
        "tls_switch": True,
    }  # fmt: skip
    layout = plan_detectors(grid_cfg)
    stations = {lp.id: lp for lp in layout.loops if lp.role == "station"}
    assert layout.stations == {"far": ("e1_far_n_in_1",),
                               "back": ("e1_back_w_in_1", "e1_back_w_in_2")}  # fmt: skip
    assert stations["e1_far_n_in_1"].pos == pytest.approx(236.3)  # kept on the lane
    assert stations["e1_back_w_in_1"].pos == pytest.approx(179.6)  # counted from the end
    areas = {a.id: a for a in layout.areas}
    assert layout.queues == {"N": ("q_N_n_in_1", "q_N_n_in_2"), "W": ("q_W_w_in_1",)}
    assert areas["q_N_n_in_1"].length == pytest.approx(235.4)  # lane length - 1
    assert areas["q_N_n_in_1"].pos == pytest.approx(1.0)
    assert areas["q_W_w_in_1"].pos + areas["q_W_w_in_1"].length == pytest.approx(239.6)
    assert len(layout.switch_links) == 18 and set(layout.switch_links.values()) == set(range(18))
    assert layout.switch_links[(":C_w1_0", ":C_c0_0")] == 14  # crossings too


def test_unknown_lane_raises(grid_cfg):
    grid_cfg["measurement"] = {"stations": [{"id": "x", "lanes": ["nope_1"], "pos_m": 5}]}
    with pytest.raises(ConfigError, match="nope_1"):
        plan_detectors(grid_cfg)


def test_additional_file(grid_cfg, tmp_path):
    grid_cfg["measurement"] = {
        "period_s": 600,
        "stations": [{"id": "W_in", "lanes": ["w_in_1"], "pos_m": 100.0}],
        "queues": {"N": {"lanes": ["n_in_1"], "length_m": 150}},
        "tls_switch": True,
    }
    path, layout = write_detectors(grid_cfg, tmp_path)
    assert path == tmp_path / "detectors.add.xml"
    root = ET.parse(path).getroot()
    loops = {el.get("id"): el for el in root.iter("inductionLoop")}
    control = [el for el in loops.values() if el.get("id").startswith("det_")]
    assert len(control) == 10
    assert all(el.get("file") == "NUL" and el.get("period") == "3600" for el in control)
    by_id = {lp.id: lp for lp in layout.loops}
    assert all(float(el.get("pos")) == by_id[i].pos for i, el in loops.items())  # exact
    station = loops["e1_W_in_w_in_1"]
    assert (station.get("file"), station.get("period"), station.get("friendlyPos")) == (
        "e1.xml", "600", "true",
    )  # fmt: skip
    (e2,) = root.iter("laneAreaDetector")
    assert e2.get("id") == "q_N_n_in_1" and e2.get("file") == "NUL"
    assert float(e2.get("pos")) + float(e2.get("length")) == pytest.approx(236.4)
    (event,) = root.iter("timedEvent")
    assert event.attrib == {"type": "SaveTLSSwitchTimes", "source": "C", "dest": "tls_switch.xml"}
    assert build_additional(layout, grid_cfg) == path.read_text(encoding="utf-8")


def test_layout_without_control_has_empty_stage_loops(scenario_cfg):
    scenario_cfg["measurement"] = {"tls_switch": True}
    layout = plan_detectors(scenario_cfg)
    assert not layout.empty and not layout.loops and layout.stage_loops == ((),) * 4
    assert isinstance(layout, DetectorLayout)


TINY_NET = """<net>
    <edge id="a_in"><lane id="a_in_0" index="0" speed="13.89" length="10.00"/></edge>
    <edge id="b_in"><lane id="b_in_0" index="0" speed="13.89" length="0.80"/></edge>
    <edge id="c_out"><lane id="c_out_0" index="0" speed="13.89" length="50.00"/></edge>
    <tlLogic id="C" type="static" programID="0" offset="0">
        <phase duration="30" state="Gr"/><phase duration="3" state="yr"/>
        <phase duration="30" state="rG"/><phase duration="3" state="ry"/>
    </tlLogic>
    <tlLogic id="C" type="static" programID="other" offset="0">
        <phase duration="9" state="rr"/>
    </tlLogic>
    <connection from="a_in" to="c_out" fromLane="0" toLane="0" tl="C" linkIndex="0"/>
    <connection from="b_in" to="c_out" fromLane="0" toLane="0" tl="C" linkIndex="1"/>
</net>"""


@pytest.fixture
def tiny_cfg(write_file):
    return {
        "network": {
            "net_file": str(write_file("tiny.net.xml", TINY_NET)),
            "tls_id": "C",
            "stages": [{"name": "A", "green_phase": 0, "clearance_phases": [1]},
                       {"name": "B", "green_phase": 2, "clearance_phases": [3]}],
        },
        "detectors": {"control": {}},
        "safety": {"min_green_s": 6},
    }  # fmt: skip


def test_short_lane_loop_sits_at_its_start(tiny_cfg, caplog):
    net = read_net(tiny_cfg["network"]["net_file"])
    assert net.tls_states["C"] == ("Gr", "yr", "rG", "ry")  # first program only
    layout = plan_detectors(tiny_cfg, net)
    assert {lp.lane: lp.pos for lp in layout.loops} == {"a_in_0": 0.0, "b_in_0": 0.0}
    assert layout.stage_loops == (("det_a_in_0",), ("det_b_in_0",))
    assert "lane shorter than" in caplog.text


def test_queue_on_a_too_short_lane_raises(tiny_cfg):
    tiny_cfg["measurement"] = {"queues": {"B": {"lanes": ["b_in_0"], "length_m": 50}}}
    with pytest.raises(ConfigError, match="too short"):
        plan_detectors(tiny_cfg)


def test_pitie_layout(pitie_cfg):
    """The committed Pitié net (WS1) with configs/pitie_fixed.yaml."""
    net = read_net(pitie_cfg["network"]["net_file"])
    layout = plan_detectors(pitie_cfg, net)
    assert _stage_map(pitie_cfg, layout) == PITIE_STAGE_LOOPS
    assert {lp.lane for lp in layout.control_loops} == {
        "n_in_1", "n_in_2", "nb_in_0", "s_in_1", "s_in_2", "w_in_1", "wb_in_1",
    }  # fmt: skip
    assert set(layout.stations) == {"N_in", "S_in", "W_in", "N_out", "S_out", "W_out"}
    for a in layout.areas:  # every E2 ends at the stop line and fits on its lane
        length = net.lanes[a.lane][0]
        assert a.pos >= 1.0 - 1e-9 and a.pos + a.length == pytest.approx(length)
    assert sorted(layout.switch_links.values()) == list(range(16))  # 12 vehicle + 4 crossing


E1_XML = """<?xml version="1.0" encoding="UTF-8"?>
<detector>
    <interval begin="27000.00" end="27900.00" id="e1_W_in_w_in_1" nVehContrib="106"
              flow="424.00" occupancy="6.12" speed="11.44" harmonicMeanSpeed="9.91"
              length="5.20" nVehEntered="106"/>
    <interval begin="27900.00" end="28800.00" id="e1_W_in_w_in_1" nVehContrib="0"
              flow="0.00" occupancy="0.00" speed="-1.00" harmonicMeanSpeed="-1.00"
              length="-1.00" nVehEntered="0"/>
</detector>"""
SWITCH_XML = """<?xml version="1.0" encoding="UTF-8"?>
<tlsSwitches>
   <tlsSwitch id="C" programID="0" fromLane=":C_w1_0" toLane=":C_c0_0" begin="27000.00"
              end="27035.00" duration="35.00"/>
   <tlsSwitch id="C" programID="0" fromLane="e_in_1" toLane="n_out_1" begin="27000.00"
              end="27040.00" duration="40.00"/>
</tlsSwitches>"""


def test_read_e1(write_file):
    df = read_e1(write_file("e1.xml", E1_XML))
    assert df["det_id"].tolist() == ["e1_W_in_w_in_1"] * 2
    assert df["begin_s"].tolist() == [27000.0, 27900.0]
    assert df["n_veh"].tolist() == [106, 0] and df["n_veh"].dtype == "int64"
    assert df["occupancy_pct"].iloc[0] == 6.12
    assert df["mean_speed"].iloc[0] == 11.44 and math.isnan(df["mean_speed"].iloc[1])


def test_read_tls_switch(write_file):
    df = read_tls_switch(write_file("tls_switch.xml", SWITCH_XML))
    assert df[["from_lane", "to_lane"]].values.tolist() == [
        [":C_w1_0", ":C_c0_0"], ["e_in_1", "n_out_1"],
    ]  # fmt: skip
    assert df["duration_s"].tolist() == [35.0, 40.0] and set(df["tls_id"]) == {"C"}


def test_readers_on_missing_files(tmp_path):
    assert read_e1(tmp_path / "e1.xml").empty
    assert read_tls_switch(tmp_path / "tls_switch.xml").empty

"""The stage list in each scenario config must match its SUMO signal program: every
link that is green at the end of a stage's clearance must also be green in the next
stage, otherwise a movement would go from green to a conflicting green without
yellow. For the Pitié junction the config's crossings, signal groups, bus stops and
walks must also fit the committed net, and the net must be built from the committed
plain XML and program."""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from traffic_twin.config import load_scenario
from traffic_twin.sim.tls_check import link_table, load_program

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ["configs/scenario_base.yaml", "configs/pitie_fixed.yaml"]


@pytest.fixture(scope="module", params=SCENARIOS, ids=lambda p: Path(p).stem)
def scenario(request) -> dict:
    return load_scenario(REPO_ROOT / request.param)


@pytest.fixture(scope="module")
def program(scenario):
    net = scenario["network"]
    return load_program(REPO_ROOT / net["net_file"], net["tls_id"])


@pytest.fixture(scope="module")
def phases(program):
    return list(program.states)


def test_stages_cover_the_whole_program(phases, scenario):
    used = []
    for s in scenario["network"]["stages"]:
        used += [s["green_phase"], *s["clearance_phases"]]
    assert sorted(used) == list(range(len(phases)))


def test_no_green_carries_into_a_conflicting_stage(phases, scenario):
    stages = scenario["network"]["stages"]
    for i, s in enumerate(stages):
        last = phases[s["clearance_phases"][-1]]
        nxt = phases[stages[(i + 1) % len(stages)]["green_phase"]]
        for link, (a, b) in enumerate(zip(last, nxt, strict=True)):
            if a in "Gg":
                assert b in "Gg", f"link {link} green after {s['name']} clearance but not next"


def test_clearance_phases_are_not_green_for_conflicting_links(phases, scenario):
    """Every link green during a clearance phase was already green in its stage's green
    phase: clearance only ever removes rights of way."""
    for s in scenario["network"]["stages"]:
        green = phases[s["green_phase"]]
        for c in s["clearance_phases"]:
            for link, (g, cl) in enumerate(zip(green, phases[c], strict=True)):
                if cl in "Gg":
                    assert g in "Gg", f"link {link} gains green in clearance of {s['name']}"


def test_clearance_phases_are_short(program, scenario):
    for s in scenario["network"]["stages"]:
        assert all(0 < program.durations[c] <= 10 for c in s["clearance_phases"]), s["name"]


def test_every_stage_has_green_times(scenario):
    names = {s["name"] for s in scenario["network"]["stages"]}
    assert names == set(scenario["fixed"]["green_s"])


def test_fixed_greens_respect_the_minimum_green(scenario):
    floor = scenario["safety"]["min_green_s"]
    assert all(g >= floor for g in scenario["fixed"]["green_s"].values())


# -- netconvert commands -------------------------------------------------------------
@pytest.fixture
def no_binary(monkeypatch):
    import traffic_twin.sim.network as network

    monkeypatch.setattr(network, "find_binary", lambda name: name)
    return network


def test_netconvert_gets_posix_paths(no_binary):
    cmd = no_binary.netconvert_command(Path("sumo/network/plain"), Path("sumo/network/x.net.xml"))
    assert "sumo/network/plain/intersection.nod.xml" in cmd
    assert cmd[-1] == "sumo/network/x.net.xml"
    assert not any("\\" in a for a in cmd)


def test_build_command_from_the_network_build_section(no_binary, pitie_scenario):
    cmd = no_binary.build_command(pitie_scenario)
    flags = dict(zip(cmd[1::2], cmd[2::2], strict=False))
    assert flags["-n"] == "sumo/network/pitie/pitie.nod.xml"
    assert flags["-x"] == "sumo/network/pitie/pitie.con.xml"
    assert flags["-i"] == "sumo/network/pitie/pitie.tll.xml"
    assert cmd[-2:] == ["-o", "sumo/network/pitie/pitie.net.xml"]
    options = pitie_scenario["network"]["build"]["options"]
    assert cmd[-2 - len(options) : -2] == options
    assert "--crossings.guess" not in cmd  # crossings are explicit in pitie.con.xml
    assert not any("\\" in a for a in cmd)


def test_build_command_without_build_section_is_the_legacy_grid(no_binary, base_scenario):
    from traffic_twin.paths import NET_FILE, PLAIN_NET_DIR

    assert no_binary.build_command(base_scenario) == no_binary.netconvert_command(
        PLAIN_NET_DIR, NET_FILE
    )


def test_cli_builds_the_grid_without_config_and_the_scenario_with_one(no_binary, monkeypatch):
    from traffic_twin.paths import NET_FILE, PLAIN_NET_DIR

    calls = []
    monkeypatch.setattr(no_binary, "_netconvert", calls.append)
    no_binary.main([])
    no_binary.main(["--config", str(REPO_ROOT / "configs/pitie_fixed.yaml")])
    assert calls[0] == no_binary.netconvert_command(PLAIN_NET_DIR, NET_FILE)
    assert calls[1][-1] == "sumo/network/pitie/pitie.net.xml"


def test_netconvert_failure_reports_its_output(no_binary):
    with pytest.raises(RuntimeError, match="netconvert failed"):
        no_binary._netconvert([sys.executable, "-c", "import sys; sys.exit('bad plain file')"])


# -- Pitié: config against the committed net -----------------------------------------
@pytest.fixture(scope="module")
def pitie_net(pitie_scenario) -> ET.Element:
    return ET.parse(REPO_ROOT / pitie_scenario["network"]["net_file"]).getroot()


@pytest.fixture(scope="module")
def pitie_program(pitie_scenario):
    net = pitie_scenario["network"]
    return load_program(REPO_ROOT / net["net_file"], net["tls_id"])


@pytest.fixture(scope="module")
def lanes(pitie_net) -> dict[str, ET.Element]:
    return {ln.get("id"): ln for ln in pitie_net.iter("lane")}


def test_signal_groups_cover_every_link_once(pitie_scenario, pitie_program):
    groups = pitie_scenario["network"]["signal_groups"]
    used = sorted(i for links in groups.values() for i in links)
    assert used == sorted(pitie_program.links)
    assert pitie_scenario["network"]["reference_group"] in groups


def test_crossings_match_the_crossing_links(pitie_scenario, pitie_program):
    crossings = pitie_scenario["network"]["crossings"]
    assert {c["link"] for c in crossings.values()} == {
        i for i, lk in pitie_program.links.items() if lk.is_crossing
    }
    for name, c in crossings.items():
        assert pitie_program.links[c["link"]].via == c["edge"], name


def test_link_table_matches_the_program_header(pitie_scenario, pitie_program):
    """pitie.tll.xml documents the link table that the program's states refer to."""
    text = (REPO_ROOT / pitie_scenario["network"]["build"]["tll"]).read_text(encoding="utf-8")
    header = text[text.index("<!--") : text.index("-->")]
    rows = [r.strip() for r in header.splitlines() if re.match(r"\s*\d+  (crossing |\S+ -> )", r)]
    table = [line.strip() for line in link_table(pitie_program)]
    assert len(rows) == len(table)
    for row, line in zip(rows, table, strict=True):
        assert row.startswith(line), (row, line)


def test_net_holds_the_committed_program(pitie_scenario, pitie_program):
    """The net was rebuilt after the last edit of pitie.tll.xml, and its greens are the
    configured fixed greens."""
    tll = ET.parse(REPO_ROOT / pitie_scenario["network"]["build"]["tll"]).getroot()
    phases = tll.find("tlLogic").findall("phase")
    assert tuple(p.get("state") for p in phases) == pitie_program.states
    assert tuple(float(p.get("duration")) for p in phases) == pitie_program.durations
    for s in pitie_scenario["network"]["stages"]:
        green = pitie_program.durations[s["green_phase"]]
        assert green == pitie_scenario["fixed"]["green_s"][s["name"]], s["name"]


def test_bus_stops_fit_on_bus_lanes(pitie_scenario, lanes):
    for sid, s in pitie_scenario["network"]["bus_stops"].items():
        lane = lanes.get(s["lane"])
        assert lane is not None, sid
        assert 0 <= s["start"] < s["end"] <= float(lane.get("length")), sid
        assert s["end"] - s["start"] >= 12, sid  # holds a bus
        allow = (lane.get("allow") or "").split()
        disallow = (lane.get("disallow") or "").split()
        assert "bus" in allow or (not allow and "bus" not in disallow), sid


def _allows(lane: ET.Element, vclass: str) -> bool:
    allow, disallow = lane.get("allow"), lane.get("disallow")
    if allow is not None:
        return vclass in allow.split() or allow == "all"
    return disallow is None or vclass not in disallow.split()


def test_in_lane_stops_on_general_lanes_are_declared(pitie_scenario, lanes):
    """Bus stops are modelled in-lane (no bay), so a dwelling bus holds every car behind
    it, and on a one-lane edge (s_out) it blocks the edge. That is a model choice the
    field visit must confirm: each stop on a lane open to cars is named in
    network.assumptions, which docs/validation.md renders (section 3)."""
    net = pitie_scenario["network"]
    declared = " ".join(net.get("assumptions", []))
    general = [sid for sid, s in net["bus_stops"].items() if _allows(lanes[s["lane"]], "passenger")]
    assert "stop_57_S" in general  # the 57 S stop is on s_out, the only lane south
    for sid in general:
        assert sid in declared, f"{sid}: in-lane stop on a car lane, not in network.assumptions"


def test_walks_start_and_end_on_sidewalks_near_c(pitie_scenario, lanes):
    """Walks are pinned to the sidewalk (lane 0) within 10 m of junction C, so every
    walker has to use the configured crossing."""
    for name, c in pitie_scenario["network"]["crossings"].items():
        assert len(c["walks"]) == 2, name
        for w in c["walks"]:
            for edge, pos in ((w["from"], w["depart_pos"]), (w["to"], w["arrival_pos"])):
                lane = lanes[f"{edge}_0"]
                assert lane.get("allow") == "pedestrian", (name, edge)
                length = float(lane.get("length"))
                assert 0 <= pos <= length, (name, edge)
                to_c = length - pos if edge.endswith("_in") else pos
                assert to_c <= 10, (name, edge, to_c)

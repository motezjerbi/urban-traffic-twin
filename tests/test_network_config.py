"""The stage list in the scenario config must match the SUMO signal program: every
link that is green at the end of a stage's clearance must also be green in the next
stage, otherwise a movement would go from green to a conflicting green without
yellow."""

from pathlib import Path

import pytest

sumolib = pytest.importorskip("sumolib")


@pytest.fixture(scope="module")
def program(base_scenario):
    net = sumolib.net.readNet(base_scenario["network"]["net_file"], withPrograms=True)
    return next(iter(net.getTLS(base_scenario["network"]["tls_id"]).getPrograms().values()))


@pytest.fixture(scope="module")
def phases(program):
    return [p.state for p in program.getPhases()]


def test_stages_cover_the_whole_program(phases, base_scenario):
    used = []
    for s in base_scenario["network"]["stages"]:
        used += [s["green_phase"], *s["clearance_phases"]]
    assert sorted(used) == list(range(len(phases)))


def test_no_green_carries_into_a_conflicting_stage(phases, base_scenario):
    stages = base_scenario["network"]["stages"]
    for i, s in enumerate(stages):
        last = phases[s["clearance_phases"][-1]]
        nxt = phases[stages[(i + 1) % len(stages)]["green_phase"]]
        for link, (a, b) in enumerate(zip(last, nxt, strict=True)):
            if a in "Gg":
                assert b in "Gg", f"link {link} green after {s['name']} clearance but not next"


def test_clearance_phases_are_not_green_for_conflicting_links(phases, base_scenario):
    """Every link green during a clearance phase was already green in its stage's green
    phase: clearance only ever removes rights of way."""
    for s in base_scenario["network"]["stages"]:
        green = phases[s["green_phase"]]
        for c in s["clearance_phases"]:
            for link, (g, cl) in enumerate(zip(green, phases[c], strict=True)):
                if cl in "Gg":
                    assert g in "Gg", f"link {link} gains green in clearance of {s['name']}"


def test_clearance_phases_are_short(program, base_scenario):
    durations = [p.duration for p in program.getPhases()]
    for s in base_scenario["network"]["stages"]:
        assert all(0 < durations[c] <= 10 for c in s["clearance_phases"]), s["name"]


def test_every_stage_has_green_times(base_scenario):
    names = {s["name"] for s in base_scenario["network"]["stages"]}
    assert names == set(base_scenario["fixed"]["green_s"])


def test_netconvert_gets_posix_paths(monkeypatch):
    import traffic_twin.sim.network as network

    monkeypatch.setattr(network, "find_binary", lambda name: name)
    cmd = network.netconvert_command(Path("sumo/network/plain"), Path("sumo/network/x.net.xml"))
    assert "sumo/network/plain/intersection.nod.xml" in cmd
    assert cmd[-1] == "sumo/network/x.net.xml"
    assert not any("\\" in a for a in cmd)

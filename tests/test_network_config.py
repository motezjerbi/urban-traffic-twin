"""The stage list in the scenario config must match the SUMO signal program: every
link that is green at the end of a stage's clearance must also be green in the next
stage, otherwise a movement would go from green to a conflicting green without
yellow."""

from pathlib import Path

import pytest

from traffic_twin.config import load_config

sumolib = pytest.importorskip("sumolib")

CFG = load_config("configs/scenario_base.yaml")
NET = Path(CFG["network"]["net_file"])


@pytest.fixture(scope="module")
def phases():
    net = sumolib.net.readNet(str(NET), withPrograms=True)
    program = next(iter(net.getTLS(CFG["network"]["tls_id"]).getPrograms().values()))
    return [p.state for p in program.getPhases()]


def test_stages_cover_the_whole_program(phases):
    used = []
    for s in CFG["network"]["stages"]:
        used += [s["green_phase"], *s["clearance_phases"]]
    assert sorted(used) == list(range(len(phases)))


def test_no_green_carries_into_a_conflicting_stage(phases):
    stages = CFG["network"]["stages"]
    for i, s in enumerate(stages):
        last = phases[s["clearance_phases"][-1]]
        nxt = phases[stages[(i + 1) % len(stages)]["green_phase"]]
        for link, (a, b) in enumerate(zip(last, nxt, strict=True)):
            if a in "Gg":
                assert b in "Gg", f"link {link} green after {s['name']} clearance but not next"


def test_every_stage_has_green_times():
    names = {s["name"] for s in CFG["network"]["stages"]}
    assert names == set(CFG["fixed"]["green_s"])

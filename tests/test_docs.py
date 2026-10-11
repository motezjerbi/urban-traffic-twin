"""README statements checked against the code and the versioned inputs, so a hand-written
number or caveat cannot drift from what the model does. No SUMO run, no data/processed."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from traffic_twin.config import load_scenario
from traffic_twin.paths import FIELD_FILE, PITIE_ACTUATED_CONFIG
from traffic_twin.sim.detectors import plan_detectors, read_net
from traffic_twin.validation.field import inventory, load_field

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def section(text: str, heading: str) -> str:
    """The body of a Markdown section: from its heading line to the next heading of the
    same or a higher level."""
    level = len(heading) - len(heading.lstrip("#"))
    lines = text.splitlines()
    start = lines.index(heading)  # ValueError if the heading is gone
    body = []
    for line in lines[start + 1 :]:
        m = re.match(r"(#+) ", line)
        if m and len(m.group(1)) <= level:
            break
        body.append(line)
    return "\n".join(body)


def flat(text: str) -> str:
    return " ".join(text.split())


def test_readme_s1_loop_distance_matches_the_detector_layout():
    cfg = load_scenario(ROOT / PITIE_ACTUATED_CONFIG)
    net = read_net(ROOT / cfg["network"]["net_file"])
    loops = plan_detectors(cfg, net).control_loops
    assert loops
    distances = {round(net.lanes[lp.lane][0] - lp.pos, 2) for lp in loops}
    assert len(distances) == 1, f"control loops at several distances {sorted(distances)}"

    text = flat(section(README, "### Controllers and safety guard"))
    stated = {float(x) for x in re.findall(r"(\d+(?:\.\d+)?) m before the stop line", text)}
    assert stated == distances, (
        f"README states the S1 control loops {sorted(stated)} m before the stop line; "
        f"plan_detectors puts them {sorted(distances)} m"
    )


def test_readme_comparison_is_provisional_until_signal_and_queues_are_measured(
    pitie_scenario,
):
    """Spec 13.4: the sim must match reality (flows, cycle, queues) before any comparison.
    While the field file has no signal timing or no queue, the README must say the S0 vs
    S1 comparison and Part 3 are provisional."""
    inv = inventory(load_field(ROOT / FIELD_FILE, pitie_scenario))
    if inv["signal"][0] and inv["queues"][0]:
        pytest.skip("signal timing and queues are measured: the 13.4 checks can run")

    comparison = flat(section(README, "### S0 vs S1 (spec 13.3)"))
    for needed in ("provisional", "13.4", "ASSUMED", "NOT_MEASURED", "field visit"):
        assert needed in comparison, f"S0 vs S1 section: no '{needed}' caveat"

    roadmap = flat(section(README, "## Roadmap").split("|")[0])  # prose before the table
    assert "field visit" in roadmap and "provisional" in roadmap, (
        "roadmap prose calls Part 3 done without the field-visit caveat"
    )

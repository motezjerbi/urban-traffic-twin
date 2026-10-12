"""A-04 end to end in SUMO: a controller that asks for a random stage at every step,
under free stage order, through the full Safety Guard (Part 3 rules + pedestrian shield)
on the Pitié junction. Seed 101 (calibration, D32), 15 minutes, outputs in tmp_path.

Whatever the controller asks, the phases shown pass every danger key of the audit, SUMO
reports no collision and no pedestrian waits at the curb more than
`safety.ped_max_wait_s` (the curb wait, creeps included, which is never below the single
`getWaitingTime` wait)."""

from __future__ import annotations

import copy
import random

import pytest

from traffic_twin.sim.audit import DANGER_KEYS, audit_phases
from traffic_twin.sim.kpi import read_statistics
from traffic_twin.sim.tls_check import cached_requirements

SEED = 101  # a calibration seed, never one of the evaluation seeds 1-10
DURATION_S = 900
AMBULANCE_DEPART_S = 300  # inside the shortened window


@pytest.mark.sumo
def test_random_free_order_controller_stays_safe(pitie_scenario, tmp_path):
    from traffic_twin.sim.session import SimSession

    cfg = copy.deepcopy(pitie_scenario)
    cfg["safety"]["stage_order"] = "free"
    cfg["duration_s"] = DURATION_S
    cfg["emergency"]["depart_s"] = AMBULANCE_DEPART_S
    cfg["outputs"]["dir"] = str(tmp_path / "results")
    n = len(cfg["network"]["stages"])
    rng = random.Random(SEED)

    sim = SimSession(cfg, SEED, tmp_path / "run", log_state=False, demand_dir=tmp_path / "demand")
    with sim:
        assert sim.guard.shield is not None and not sim.guard.strict_order
        while not sim.done:
            sim.observe()
            sim.step(rng.randrange(n))

    links = [int(c["link"]) for c in cfg["network"]["crossings"].values()]
    req = cached_requirements(cfg["network"]["net_file"], cfg)
    audit = audit_phases(
        sim.phases_shown, sim.phase_states, sim.plan.stages, sim.guard.min_green_s, sim.dt,
        ped_links=links, durations=sim.plan.durations, allowed=sim.guard.allowed,
        intergreen=req.intergreen, foes=req.foes,
    )  # fmt: skip
    assert {k: audit[k] for k in DANGER_KEYS if audit[k]} == {}
    assert audit["out_of_order_green"] > 0  # the controller did jump (information key)
    assert sim.guard.interventions["min_green"] > 0  # ... and was held by the guard
    assert sim.guard.interventions["ped_wait"] > 0  # the shield had to act (off: 122 s)
    assert read_statistics(sim.statistics)["collisions"] == 0
    assert sim.ped_curb_wait_max, "no pedestrian walked"
    assert max(sim.ped_curb_wait_max.values()) <= cfg["safety"]["ped_max_wait_s"]

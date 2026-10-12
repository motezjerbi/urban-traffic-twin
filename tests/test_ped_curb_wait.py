"""The pedestrian bound holds on the real curb wait, creeps included (A-04, D21).

SUMO resets `person.getWaitingTime` to 0 as soon as a waiting pedestrian moves faster
than 0.1 m/s, and pedestrians at the curb shuffle a few centimetres while the crossing
stays red: with getWaitingTime as the shield's w_c, S2 pedestrians stood up to 91 s at a
red crossing while the shield (and `ped_wait_single_max`) saw 88. The session's curb
wait keeps counting through creeps; with it as w_c, the shield keeps every curb wait
within theta = 88 s, under the 90 s target.

S2 (`configs/pitie_rules.yaml`, the controller with the longest reds) on the calibration
seeds 101-103 (D32), full demand window, from a scratch copy of the repo's inputs."""

from __future__ import annotations

import pytest

from traffic_twin.config import load_scenario
from traffic_twin.control.pedestrian import ped_threshold
from traffic_twin.control.registry import make_controller
from traffic_twin.paths import PITIE_RULES_CONFIG


@pytest.mark.sumo
@pytest.mark.parametrize("seed", [101, 102, 103])  # calibration seeds only (D32)
def test_s2_curb_waits_stay_within_the_shield_bound(make_workspace, rules_scenario, tmp_path,
                                                     monkeypatch, seed):  # fmt: skip
    from traffic_twin.sim.session import SimSession

    make_workspace(tmp_path, rules_scenario)
    monkeypatch.chdir(tmp_path)
    cfg = load_scenario(PITIE_RULES_CONFIG)
    theta = ped_threshold(cfg)
    controller = make_controller(cfg)
    sim = SimSession(cfg, seed, tmp_path / "run", log_state=False, demand_dir=tmp_path / "d")
    with sim:
        controller.reset(seed)
        while not sim.done:
            sim.step(controller.decide(sim.observe()))

    curb, single = sim.ped_curb_wait_max, sim.ped_wait_max
    assert curb and set(curb) == set(single)
    assert all(curb[p] >= single[p] for p in curb)  # never below the Part 3 value
    assert sum(curb[p] > single[p] for p in curb) > 50  # creeps happen: not vacuous
    assert max(curb.values()) <= theta <= cfg["safety"]["ped_max_wait_s"]

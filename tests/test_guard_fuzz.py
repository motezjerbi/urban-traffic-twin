"""Property test of the Safety Guard with its pedestrian shield (D20, D21, A-04), no
SUMO: adversarial controllers drive guard + SignalPlan on the Pitié program while
pedestrians arrive at every crossing as Poisson streams (200 per hour while it is red)
and wait until it shows green. Whatever the controller asks, every danger key of the
audit stays 0 and no synthetic wait exceeds theta + dt; with the shield off the same
adversaries push waits past 120 s, so the test is not vacuous.

Ported from the Part 4 guard analysis (`fuzz.py`), seed-limited to stay fast."""

from __future__ import annotations

import copy
import dataclasses
import random
from collections.abc import Callable

import pytest

from traffic_twin.control.base import Observation
from traffic_twin.control.safety import SafetyGuard
from traffic_twin.sim.audit import DANGER_KEYS, audit_phases
from traffic_twin.sim.signal import SignalPlan
from traffic_twin.sim.tls_check import cached_requirements, load_program, scenario_stages

STEPS = 7200  # two hours per run, like the evaluation window
PED_RATE_H = 200.0
ADVERSARIES = ("random", "sticky", "oscillate", "ped_adversary", "last")


@dataclasses.dataclass
class Peds:
    """Synthetic pedestrians per crossing: arrival times of those still waiting."""

    links: list[int]
    rng: random.Random
    waiting: list[list[float]] = dataclasses.field(default_factory=list)
    max_wait: float = 0.0

    def __post_init__(self):
        self.waiting = [[] for _ in self.links]

    def waits(self, t: float) -> tuple[float, ...]:
        return tuple(t - w[0] if w else 0.0 for w in self.waiting)

    def step(self, t: float, state: str) -> None:
        """The phase `state` is shown during step t: served or one more arrival."""
        for c, link in enumerate(self.links):
            if state[link] in "Gg":
                self.max_wait = max([self.max_wait, *(t - a for a in self.waiting[c])])
                self.waiting[c] = []
            elif self.rng.random() < PED_RATE_H / 3600.0:
                self.waiting[c].append(t)

    def finish(self, t: float) -> None:
        for arrivals in self.waiting:
            self.max_wait = max([self.max_wait, *(t - a for a in arrivals)])


def adversary(
    name: str, rng: random.Random, n: int, peds: Peds, green: list[list[bool]]
) -> Callable[[Observation], int]:
    """Controller `name`: green[k][c] is True when stage k serves crossing c."""

    def ped_adversary(obs: Observation) -> int:  # asks for the stage serving the fewest waits
        w = peds.waits(obs.t)
        return max(range(n), key=lambda k: sum(x for c, x in enumerate(w) if not green[k][c]))

    return {
        "random": lambda obs: rng.randrange(n),
        "sticky": lambda obs: obs.stage,
        "oscillate": lambda obs: (obs.stage + 2) % n if rng.random() < 0.5 else obs.stage,
        "ped_adversary": ped_adversary,
        "last": lambda obs: n - 1,
    }[name]


@pytest.fixture(scope="module")
def pitie(pitie_scenario):
    net = pitie_scenario["network"]["net_file"]
    prog = load_program(net, pitie_scenario["network"]["tls_id"])
    req = cached_requirements(net, pitie_scenario)
    return prog, req


def run(cfg: dict, prog, req, name: str, seed: int) -> tuple[float, dict, SafetyGuard]:
    stages = scenario_stages(cfg)
    durations, states = list(prog.durations), list(prog.states)
    guard = SafetyGuard.from_config(cfg, stages, states, durations)
    plan = SignalPlan(stages, durations, guard.min_green_s)
    links = [int(c["link"]) for c in cfg["network"]["crossings"].values()]
    rng = random.Random(seed)
    peds = Peds(links, random.Random(1000 + seed))
    green = [[states[s.green_phase][lk] in "Gg" for lk in links] for s in stages]
    decide = adversary(name, rng, len(stages), peds, green)
    shown = []
    for i in range(STEPS):
        t = float(i)
        obs = Observation(t, plan.stage, plan.time_in_stage, plan.in_transition)
        state = dataclasses.replace(plan.state(t), ped_wait=peds.waits(t))
        phase = plan.request(guard.apply(decide(obs), state))
        shown.append(phase)
        peds.step(t, states[phase])
        plan.advance(1.0)
    peds.finish(float(STEPS))
    audit = audit_phases(
        shown, states, stages, guard.min_green_s, ped_links=links, durations=durations,
        allowed=guard.allowed, intergreen=req.intergreen, foes=req.foes,
    )  # fmt: skip
    return peds.max_wait, {k: audit[k] for k in DANGER_KEYS if audit[k]}, guard


def scenario(pitie_scenario: dict, order: str, forcing: str) -> dict:
    cfg = copy.deepcopy(pitie_scenario)
    cfg["safety"].update(stage_order=order, ped_forcing=forcing)
    return cfg


@pytest.mark.parametrize("order", ["strict", "free"])
@pytest.mark.parametrize("name", ADVERSARIES)
def test_adversaries_never_break_the_pedestrian_bound(pitie_scenario, pitie, order, name):
    cfg = scenario(pitie_scenario, order, "detected")
    theta = cfg["safety"]["ped_max_wait_s"] - cfg["safety"]["ped_red_margin_s"]
    for seed in range(3):
        max_wait, danger, guard = run(cfg, *pitie, name, seed)
        assert danger == {}, (order, name, seed)
        assert max_wait <= theta + guard.dt, (order, name, seed, max_wait)


def test_without_the_shield_waits_run_far_past_the_bound(pitie_scenario, pitie):
    worst = 0.0
    for order in ("strict", "free"):
        cfg = scenario(pitie_scenario, order, "off")
        for name in ("sticky", "ped_adversary"):
            max_wait, danger, guard = run(cfg, *pitie, name, 0)
            assert guard.shield is None and danger == {}
            worst = max(worst, max_wait)
    assert worst > 120

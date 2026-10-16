"""RL reward `rl-reward-1` (Part 6, D58; contract 4.7): person-weighted waiting with a
pedestrian penalty, accrued after every simulation step.

After each step of dt seconds (`SimSession.dt`, 1 s):

    c = dt * (w_car n_car + w_bus n_bus + w_ped n_ped + w_amb n_amb
              + lambda * 1(max curb wait > threshold_s))

- n_class: vehicles of that class (`kpi.user_type_of`: car / bus / ambulance) with
  `vehicle.getWaitingTime > 0` (the tripinfo `waitingTime` definition: speed < 0.1 m/s,
  not at a scheduled stop), plus, with `pending_as_waiting`, that class's pending
  insertions (cars carry the backlog of the saturated N arm, so a policy cannot hide its
  queue beyond the modelled approach);
- n_ped: the sum over crossings of `EXTRA_PED_COUNT` (persons waiting at a curb);
- the penalty indicator uses the max over crossings of `EXTRA_PED_WAIT` (the shield's
  curb wait), strictly above the threshold (80 s > crossing W's 72 s worst red at min
  greens, so the penalty is avoidable).

Weights are in persons (`configs/rl.yaml: reward.weights`, car = w_v 1.0 x occupancy
1.3, bus 25, pedestrian 1.2, ambulance 20). The environment's reward over a decision
interval is `-scale * sum(c)`, so an episode return is -scale x (weighted waiting-seconds
+ lambda x penalty-seconds).

`accrue` reads TraCI getters and `sim.observe()` only (passive: the observation is cached
per step and is the one the controller observes before the next step). The vehicle type
of each id is read once.

`RewardObserver` (Part 7, D76) is a passive `run_scenario.RunObserver` that accrues the
same totals over a KPI run and writes `<run_dir>/reward_totals.json`: the class totals
are weight-free (waiting seconds, penalty-seconds), so any weight vector can be applied
to them afterwards (the sensitivity reweighting table).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from traffic_twin.control.base import EXTRA_PED_COUNT, EXTRA_PED_WAIT
from traffic_twin.sim.kpi import user_type_of

if TYPE_CHECKING:
    from traffic_twin.sim.session import SimSession

REWARD_VERSION = "rl-reward-1"
CLASSES = ("car", "bus", "ambulance")
WEIGHT_KEYS = ("car", "bus", "pedestrian", "ambulance")
TOTAL_KEYS = ("car_s", "car_pending_s", "bus_s", "pedestrian_s", "ambulance_s", "penalty_s", "cost")
TOTALS_FILE = "reward_totals.json"  # RewardObserver output, in the run folder
EPS = 1e-9


class RewardTracker:
    """Accrues the per-step cost c (D58) and its parts over an episode.

    `totals` (since construction or `reset()`): `car_s`, `bus_s`, `ambulance_s` are
    waiting vehicle-seconds of the class, pending insertions included when
    `pending_as_waiting`; `car_pending_s` is the pending part of `car_s`; `pedestrian_s`
    the waiting person-seconds at the curbs; `penalty_s` the seconds with the indicator
    on; `cost` the sum of c."""

    totals: dict[str, float]

    def __init__(self, reward_cfg: Mapping) -> None:
        weights = reward_cfg["weights"]
        missing = [k for k in WEIGHT_KEYS if k not in weights]
        if missing:
            raise ValueError(f"reward.weights misses {missing}")
        self.weights = {k: float(weights[k]) for k in WEIGHT_KEYS}
        self.pending_as_waiting = bool(reward_cfg.get("pending_as_waiting", True))
        penalty = reward_cfg.get("ped_penalty") or {}
        self.lam = float(penalty.get("lambda", 0.0))
        self.threshold_s = float(penalty.get("threshold_s", float("inf")))
        self.scale = float(reward_cfg.get("scale", 1.0))
        self.reset()

    def reset(self) -> None:
        """Zero the totals and forget the vehicle types (start of an episode)."""
        self.totals = dict.fromkeys(TOTAL_KEYS, 0.0)
        self._class: dict[str, str] = {}  # vehicle id -> user class

    def _cls(self, veh, vid: str) -> str:
        c = self._class.get(vid)
        if c is None:
            c = self._class[vid] = user_type_of(veh.getTypeID(vid))
        return c

    def accrue(self, sim: SimSession) -> float:
        """Cost c of the step just simulated (unscaled, D58); adds to `totals`."""
        veh = sim.traci.vehicle
        n = dict.fromkeys(CLASSES, 0)
        for vid in veh.getIDList():
            if veh.getWaitingTime(vid) > 0:
                n[self._cls(veh, vid)] += 1
        pending_cars = 0
        if self.pending_as_waiting:
            for vid in sim.traci.simulation.getPendingVehicles():
                c = self._cls(veh, vid)
                n[c] += 1
                pending_cars += c == "car"
        extras = sim.observe().extras
        n_ped = sum(extras.get(EXTRA_PED_COUNT, ()))
        over = max(extras.get(EXTRA_PED_WAIT, ()), default=0.0) > self.threshold_s
        dt, w = float(sim.dt), self.weights
        cost = dt * (
            w["car"] * n["car"]
            + w["bus"] * n["bus"]
            + w["pedestrian"] * n_ped
            + w["ambulance"] * n["ambulance"]
            + (self.lam if over else 0.0)
        )
        t = self.totals
        t["car_s"] += dt * n["car"]
        t["car_pending_s"] += dt * pending_cars
        t["bus_s"] += dt * n["bus"]
        t["pedestrian_s"] += dt * n_ped
        t["ambulance_s"] += dt * n["ambulance"]
        t["penalty_s"] += dt * over
        t["cost"] += cost
        return cost

    def reward(self, cost: float) -> float:
        """The environment reward of a summed cost: `-scale * cost`."""
        return -self.scale * cost


def weighted_cost(totals: Mapping[str, float], reward_cfg: Mapping) -> float:
    """The unscaled cost of weight-free totals under a reward config: w_car car_s +
    w_bus bus_s + w_ped pedestrian_s + w_amb ambulance_s + lambda penalty_s (equal to
    `totals["cost"]` for the weights they were accrued with, up to rounding)."""
    t = RewardTracker(reward_cfg)
    w = t.weights
    return (
        w["car"] * float(totals["car_s"])
        + w["bus"] * float(totals["bus_s"])
        + w["pedestrian"] * float(totals["pedestrian_s"])
        + w["ambulance"] * float(totals["ambulance_s"])
        + t.lam * float(totals["penalty_s"])
    )


class RewardObserver:
    """Passive `RunObserver` (D76): accrues `RewardTracker` totals after every step of a
    `run_scenario.run` and writes `<run_dir>/reward_totals.json` in `finish`:

        {reward_version, run_id, reward (the config used), steps,
         totals (whole run, drain included), totals_window (steps ending at or before
         the scenario's demand window end: the env's control-window scope)}

    It reads TraCI getters and `sim.observe()` only (`RewardTracker.accrue`), so the run's
    KPIs are the same with or without it (`tests/test_rl_reward.py`)."""

    def __init__(self, reward_cfg: Mapping) -> None:
        self.reward_cfg = dict(reward_cfg)
        self.tracker = RewardTracker(reward_cfg)
        self.steps = 0
        self.totals_window: dict[str, float] = dict(self.tracker.totals)
        self.path: Path | None = None

    def bind(self, sim: SimSession) -> None:
        self.tracker.reset()
        self.steps = 0
        self.totals_window = dict(self.tracker.totals)
        self.path = None

    def after_step(self, sim: SimSession) -> None:
        self.tracker.accrue(sim)
        self.steps += 1
        if sim.time <= sim.end + EPS:
            self.totals_window = dict(self.tracker.totals)

    @property
    def totals(self) -> dict[str, float]:
        return dict(self.tracker.totals)

    def result(self, run_id: str | None = None) -> dict:
        return {
            "reward_version": REWARD_VERSION,
            "run_id": run_id,
            "reward": self.reward_cfg,
            "steps": self.steps,
            "totals": self.totals,
            "totals_window": dict(self.totals_window),
        }

    def finish(self, sim: SimSession, out_dir: Path, run_id: str) -> None:
        self.path = Path(out_dir) / TOTALS_FILE
        self.path.write_text(json.dumps(self.result(run_id), indent=1) + "\n", encoding="utf-8")

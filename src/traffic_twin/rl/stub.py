"""`StubTrafficEnv`: the spaces, info keys and episode summary of `rl.env.TrafficEnv`
without SUMO (Part 6, WS5), for the training smoke tests and CI.

A toy five-stage junction stands in for Pitié: seven lanes with Poisson arrivals (rates
scaled by the episode's demand multiplier), one departure every two seconds per green
lane, four crossings whose pedestrians cross when their stage is green, buses on the S,
NB and WB approaches, a fixed 6 s clearance between greens and a 6 s minimum green. A
toy guard forces the cycle successor after 120 s of green or when a pedestrian has
waited 40 s, counted under the real `INTERVENTION_KEYS`. The cost of a simulated second
uses the weights of `configs/rl.yaml` (D58), so returns have the real env's scale.

Same constructor, `reset(options={"episode_seed": s})` / `step(action)` contract and
`episode_summary` keys as `TrafficEnv`, deterministic for a given episode seed and
action sequence. Worker i plays `seeds[i + k * n_workers]` (D59). `run_baseline` plays
fixed / actuated / rules (toy rules on the stub state), random, stay or advance through
env actions. Nothing here is a result: it only exercises the training and evaluation
code paths.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from traffic_twin.config import load_config
from traffic_twin.control.safety import INTERVENTION_KEYS
from traffic_twin.paths import RL_CONFIG

OBS_DIM = 74  # rl-obs-1 at Pitié (D57)
N_STAGES = 5
N_LANES = 7
STAGE_LANES: tuple[tuple[int, ...], ...] = ((0, 1), (2,), (3, 4), (5,), (6,))
LANE_RATE = (0.11, 0.09, 0.07, 0.10, 0.06, 0.04, 0.03)  # arrivals / s / lane at multiplier 1
LANE_CAP = 20.0  # vehicles
CROSSING_STAGES: tuple[tuple[int, ...], ...] = ((1,), (0,), (3,), (2, 4))  # NB, S, WB, W
PED_RATE = 0.015  # arrivals / s / crossing
BUS_LANES = (2, 5, 6)  # approaches S, NB, WB
BUS_RATE = 1 / 240.0  # arrivals / s / approach
BUS_APPROACH_S = 30.0  # seconds from detection to the stop line
MIN_GREEN_S = 6
CLEARANCE_S = 6
MAX_GREEN_S = 120
PED_FORCE_S = 40.0  # toy shield: force the successor when a pedestrian waited this long
FULL_WINDOW_S = 7200
TOTAL_KEYS = ("car_s", "car_pending_s", "bus_s", "pedestrian_s", "ambulance_s", "penalty_s", "cost")
SUMMARY_KEYS = (
    "params",
    "decisions",
    "advances",
    "totals",
    "interventions",
    "danger",
    "audit",
    "preemption",
    "ped_curb_wait_max",
    "amb_seen_share",
)
STEP_INFO_KEYS = ("t", "decision_s", "action", "requested", "guard_override", "reward_parts")
BASELINES = ("fixed", "actuated", "rules", "random", "stay", "advance")
FIXED_GREEN_S = (30, 12, 20, 10, 10)  # toy fixed-time greens per stage


def _range(pair: Sequence[int]) -> list[int]:
    lo, hi = int(pair[0]), int(pair[1])
    return list(range(lo, hi + 1))


def stub_episode_params(episode_seed: int, rl_cfg: Mapping, mode: str, randomize: bool) -> dict:
    """Per-episode draws from default_rng([episode_seed, 6_000_000]) as in D59 (multiplier,
    day type, window start); the stub only uses the multiplier."""
    rng = np.random.default_rng([int(episode_seed), 6_000_000])
    rand = rl_cfg.get("randomize") or {}
    lo, hi = rand.get("multiplier", [1.0, 1.0])
    multiplier = float(rng.uniform(lo, hi))
    day_types = rand.get("day_types") or {"tue_wed_thu_term": 1.0}
    names = list(day_types)
    p = np.asarray([float(day_types[k]) for k in names])
    day_type = names[int(rng.choice(len(names), p=p / p.sum()))]
    start_min = int(rng.integers(0, 61))
    if not randomize:
        multiplier, day_type, start_min = 1.0, "tue_wed_thu_term", 0
    return {
        "episode_seed": int(episode_seed),
        "mode": mode,
        "multiplier": round(multiplier, 4),
        "day_type": day_type,
        "start_s": 27000 + 60 * start_min if mode == "train" else 27000,
        "ambulances": [],
        "incidents": [],
    }


class StubTrafficEnv(gym.Env):
    """action_space = Discrete(2) (0 stay, 1 advance); observation_space = Box(0, 1, (74,),
    float32). `window_s` (stub only) overrides the control window length."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: str | Path | Mapping = RL_CONFIG,
        mode: str = "train",
        worker: int = 0,
        n_workers: int = 1,
        work_root: Path | None = None,
        seeds: Sequence[int] | None = None,
        randomize: bool = True,
        junction_monitor: bool = False,
        window_s: int | None = None,
    ) -> None:
        if mode not in ("train", "full"):
            raise ValueError(f"mode must be 'train' or 'full', not {mode!r}")
        self.cfg = dict(config) if isinstance(config, Mapping) else load_config(config)
        self.mode, self.worker, self.n_workers = mode, int(worker), max(int(n_workers), 1)
        self.randomize = randomize
        self.seeds = list(seeds) if seeds is not None else _range(self.cfg["seeds"]["train"])
        if not self.seeds:
            raise ValueError("no episode seeds")
        default_window = (self.cfg.get("episode") or {}).get("train", {}).get("window_s", 1800)
        self.window_s = int(window_s or (default_window if mode == "train" else FULL_WINDOW_S))
        self.interval_s = int((self.cfg.get("decision") or {}).get("interval_s", 5))
        reward = self.cfg.get("reward") or {}
        w = reward.get("weights") or {}
        defaults = {"car": 1.3, "bus": 25.0, "pedestrian": 1.2, "ambulance": 20.0}
        self.w = {k: float(w.get(k, d)) for k, d in defaults.items()}
        pen = reward.get("ped_penalty") or {}
        self.lam = float(pen.get("lambda", 30.0))
        self.threshold_s = float(pen.get("threshold_s", 80.0))
        self.scale = float(reward.get("scale", 0.001))
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32)
        self._episode = 0
        self._ready = False

    # -- episode lifecycle -----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:  # a seeded reset restarts the worker's schedule (check_env)
            self._episode = 0
        ep = (options or {}).get("episode_seed")
        if ep is None:
            ep = self.seeds[(self.worker + self._episode * self.n_workers) % len(self.seeds)]
            self._episode += 1
        self.params = stub_episode_params(int(ep), self.cfg, self.mode, self.randomize)
        self.rng = np.random.default_rng([int(ep), 6_000_001])
        self.t = 0
        self.stage, self.tis, self.in_transition, self.target, self.clear_left = (
            0,
            0,
            False,
            None,
            0,
        )
        self.queue = np.zeros(N_LANES)
        self.head_wait = np.zeros(N_LANES)
        self.ped_count = np.zeros(4, dtype=int)
        self.ped_wait = np.zeros(4)
        self.red_s = np.zeros(N_STAGES)
        self.bus_dist = np.full(3, np.inf)  # distance-to-stop-line proxy (s), inf = none
        self.bus_wait = np.zeros(3)
        self.totals = dict.fromkeys(TOTAL_KEYS, 0.0)
        self.interventions = dict.fromkeys(INTERVENTION_KEYS, 0)
        self.decisions = self.advances = 0
        self.ped_curb_wait_max = 0.0
        self._ready = True
        self._advance_to_decision()
        return self._obs(), {"params": dict(self.params)}

    @property
    def done(self) -> bool:
        return self.t >= self.window_s

    def is_decision_point(self) -> bool:
        return not self.in_transition and self.tis >= MIN_GREEN_S

    def step(self, action):
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        a = int(action)
        if a not in (0, 1):
            raise ValueError(f"action must be 0 or 1, not {action!r}")
        t0 = self.t
        requested = (self.stage + 1) % N_STAGES if a == 1 else self.stage
        before = dict(self.totals)
        forced0 = sum(self.interventions.values())
        if a == 1 and self.is_decision_point():
            self._start_transition(requested)
            self.advances += 1
        for _ in range(self.interval_s):
            if self.done:
                break
            self._tick()
        self._advance_to_decision()
        self.decisions += 1
        parts = {k: self.totals[k] - before[k] for k in TOTAL_KEYS}
        reward = -self.scale * parts["cost"]
        truncated = self.done
        info: dict[str, Any] = {
            "t": float(self.t),
            "decision_s": float(self.t - t0),
            "action": a,
            "requested": int(requested),
            "guard_override": sum(self.interventions.values()) > forced0,
            "reward_parts": parts,
        }
        if truncated:
            info["episode_summary"] = self.summary()
            self._ready = False
        return self._obs(), float(reward), False, truncated, info

    def summary(self) -> dict:
        return {
            "params": dict(self.params),
            "decisions": self.decisions,
            "advances": self.advances,
            "return": -self.scale * float(self.totals["cost"]),
            "totals": {k: float(v) for k, v in self.totals.items()},
            "interventions": dict(self.interventions),
            "danger": 0,
            "audit": {},
            "preemption": {"enabled": False, "requests": 0, "served": 0, "timeouts": 0},
            "ped_curb_wait_max": float(self.ped_curb_wait_max),
            "amb_seen_share": 0.0,
        }

    def close(self) -> None:
        self._ready = False

    # -- toy dynamics ----------------------------------------------------------------
    def _start_transition(self, target: int) -> None:
        self.in_transition, self.target, self.clear_left = True, int(target), CLEARANCE_S

    def _served(self) -> tuple[int, ...]:
        return () if self.in_transition else STAGE_LANES[self.stage]

    def _advance_to_decision(self) -> None:
        while not self.done and not self.is_decision_point():
            self._tick()

    def _guard(self) -> None:
        if self.in_transition or self.tis < MIN_GREEN_S:
            return
        succ = (self.stage + 1) % N_STAGES
        if self.tis >= MAX_GREEN_S:
            self.interventions["max_green"] += 1
            self._start_transition(succ)
            return
        waiting = [
            c for c in range(4) if self.ped_count[c] and self.stage not in CROSSING_STAGES[c]
        ]
        if waiting and max(self.ped_wait[c] for c in waiting) >= PED_FORCE_S:
            self.interventions["ped_wait"] += 1
            self._start_transition(succ)

    def _tick(self) -> None:
        """One simulated second."""
        rng, mult = self.rng, self.params["multiplier"]
        self._guard()
        if self.in_transition:
            self.clear_left -= 1
            if self.clear_left <= 0:
                self.stage, self.tis, self.in_transition, self.target = self.target, 0, False, None
        else:
            self.tis += 1
        served = self._served()
        self.queue += rng.poisson(np.asarray(LANE_RATE) * mult)
        for ln in served:
            if self.t % 2 == 0 and self.queue[ln] > 0:
                self.queue[ln] -= 1
        np.clip(self.queue, 0, None, out=self.queue)
        for ln in range(N_LANES):
            if self.queue[ln] <= 0:
                self.head_wait[ln] = 0.0
            elif ln in served:
                self.head_wait[ln] = max(self.head_wait[ln] - 2.0, 0.0)
            else:
                self.head_wait[ln] += 1.0
        green_stage = None if self.in_transition else self.stage
        self.ped_count += rng.poisson(PED_RATE, 4)
        for c in range(4):
            if green_stage in CROSSING_STAGES[c]:
                self.ped_count[c], self.ped_wait[c] = 0, 0.0
            elif self.ped_count[c] > 0:
                self.ped_wait[c] += 1.0
        self.ped_curb_wait_max = max(self.ped_curb_wait_max, float(self.ped_wait.max()))
        for k in range(N_STAGES):
            self.red_s[k] = 0.0 if green_stage == k else self.red_s[k] + 1.0
        n_bus = 0
        for i, ln in enumerate(BUS_LANES):
            if math.isinf(self.bus_dist[i]):
                if rng.random() < BUS_RATE:
                    self.bus_dist[i], self.bus_wait[i] = BUS_APPROACH_S, 0.0
                continue
            if self.bus_dist[i] > 0:
                self.bus_dist[i] -= 1.0
            elif ln in served:
                self.bus_dist[i] = np.inf  # crossed
            else:
                self.bus_wait[i] += 1.0
                n_bus += 1
        # halting: the whole queue on red lanes, all but the vehicle moving on green ones
        halting = sum(max(q - 1.0, 0.0) if ln in served else q for ln, q in enumerate(self.queue))
        n_ped = float(self.ped_count.sum())
        over = float(self.ped_wait.max() > self.threshold_s)
        cost = self.w["car"] * halting + self.w["bus"] * n_bus + self.w["pedestrian"] * n_ped
        cost += self.lam * over
        tot = self.totals
        tot["car_s"] += halting
        tot["bus_s"] += n_bus
        tot["pedestrian_s"] += n_ped
        tot["penalty_s"] += over
        tot["cost"] += cost
        self.t += 1

    def _obs(self) -> np.ndarray:
        x: list[float] = []
        served = self._served()
        for ln in range(N_LANES):
            halting = self.queue[ln] - (1.0 if ln in served and self.queue[ln] > 0 else 0.0)
            x += [halting / LANE_CAP, self.queue[ln] / LANE_CAP, self.head_wait[ln] / 120.0]
        x += [0.0, 0.0, 0.0]  # pending N, S, W
        x += [1.0 if k == self.stage else 0.0 for k in range(N_STAGES)]
        x += [float(self.in_transition), self.tis / 60.0]
        x += [r / 180.0 for r in self.red_s]
        for c in range(4):
            slack = 88.0 - self.ped_wait[c] if self.ped_count[c] else math.inf
            x += [
                self.ped_count[c] / 10.0,
                self.ped_wait[c] / 88.0,
                1.0 if math.isinf(slack) else slack / 88.0,
            ]
        x += [0.0] * 9  # ambulance: present, pending, 3 distance bins, 3 arms, preempt
        for i in range(3):
            present = not math.isinf(self.bus_dist[i])
            w = self.bus_wait[i]
            x += [
                float(present),
                (self.bus_dist[i] * 10.0 / 300.0) if present else 1.0,
                float(present and w < 30),
                float(present and 30 <= w < 90),
                float(present and w >= 90),
            ]
        tod = 2 * math.pi * ((self.params["start_s"] + self.t) % 86400) / 86400
        x += [0.5 + 0.5 * math.sin(tod), 0.5 + 0.5 * math.cos(tod)]
        return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


def _served_queue(env: StubTrafficEnv) -> float:
    return float(sum(env.queue[ln] for ln in STAGE_LANES[env.stage]))


def _baseline_action(env: StubTrafficEnv, controller: str, rng: np.random.Generator) -> int:
    if controller == "stay":
        return 0
    if controller == "advance":
        return 1
    if controller == "random":
        return int(rng.integers(2))
    if controller == "fixed":
        return int(env.tis >= FIXED_GREEN_S[env.stage])
    served = _served_queue(env)
    if controller == "actuated":
        return int(served < 1.0 or env.tis >= 40)
    # rules: gap-out like actuated, or yield to a much longer queue elsewhere
    others = max(
        float(sum(env.queue[ln] for ln in STAGE_LANES[k]))
        for k in range(N_STAGES)
        if k != env.stage
    )
    return int(served < 1.0 or others > 2.0 * served + 3.0 or env.tis >= 60)


def run_baseline(env: StubTrafficEnv, controller: str, episode_seed: int) -> dict:
    """One episode of a toy baseline through env actions; returns the episode summary."""
    if controller not in BASELINES:
        raise ValueError(f"unknown baseline {controller!r} (expected one of {BASELINES})")
    rng = np.random.default_rng([int(episode_seed), 7_000_000])
    env.reset(options={"episode_seed": int(episode_seed)})
    while True:
        _, _, terminated, truncated, info = env.step(_baseline_action(env, controller, rng))
        if terminated or truncated:
            return info["episode_summary"]

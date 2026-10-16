"""Turns stage requests from a controller into SUMO phases, always running the
clearance phases (pedestrian clearance, yellow, all-red) before a new green."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    green_phase: int
    clearance_phases: tuple[int, ...]


@dataclass(frozen=True)
class SignalState:
    """Snapshot of a SignalPlan, as the safety guard sees it before a request."""

    t: float  # simulation time since scenario start (s)
    stage: int  # current green stage; during a transition, the stage being left
    time_in_stage: float  # seconds of green of the current stage
    in_transition: bool  # True while clearance phases run
    red_s: tuple[float, ...]  # per stage: seconds since its green last ended (0 while green)
    # Part 4 guard inputs, set by SimSession.step (defaults: the grid and Part 3 callers)
    ped_wait: tuple[float, ...] = ()  # per crossing (network.crossings order): shield's w_c
    preempt_target: int | None = None  # target stage of the active emergency preemption


def link_stages(
    stages: Sequence[Stage], phase_states: Sequence[str], link: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(stages showing 'G' on `link`, stages showing 'g' on it) in their green phase, each
    in cycle order. A link green in no stage gives two empty tuples."""
    full, permissive = [], []
    for k, st in enumerate(stages):
        state = phase_states[st.green_phase]
        if not 0 <= link < len(state):
            raise ValueError(f"link {link} outside the phase state {state!r}")
        if state[link] == "G":
            full.append(k)
        elif state[link] == "g":
            permissive.append(k)
    return tuple(full), tuple(permissive)


class SignalPlan:
    """Stateful phase sequencer: `request()` a stage, show `phase`, then `advance(dt)`."""

    def __init__(self, stages: list[Stage], phase_durations: list[float], min_green_s: float = 0.0):
        self.stages = stages
        self.durations = phase_durations
        self.min_green_s = min_green_s
        self.stage = 0
        self.time_in_stage = 0.0
        # Per stage: seconds since its green last ended; 0 while green, counted from the
        # start of the run for stages not yet served
        self.red_s: list[float] = [0.0] * len(stages)
        self._queue: list[tuple[int, float]] = []  # pending (phase, remaining s)
        self._target: int | None = None

    @property
    def in_transition(self) -> bool:
        return self._target is not None

    @property
    def target(self) -> int | None:
        """Stage being switched to during a transition; None while a stage is green."""
        return self._target

    @property
    def clearance_left_s(self) -> float:
        """Seconds of clearance still to run before the target stage turns green (sum of
        the remaining durations of the queued clearance phases); 0.0 while green."""
        return float(sum(left for _phase, left in self._queue)) if self.in_transition else 0.0

    @property
    def phase(self) -> int:
        if self._queue:
            return self._queue[0][0]
        return self.stages[self.stage].green_phase

    def request(self, stage: int) -> int:
        """Ask for a green stage; starts clearance if it differs. Returns the phase to show.

        A switch asked before the current green has lasted `min_green_s` is ignored; the
        controller simply asks again on a later step."""
        if not 0 <= stage < len(self.stages):
            raise ValueError(f"unknown stage {stage}")
        held = self.time_in_stage < self.min_green_s - 1e-9
        if self._target is None and stage != self.stage and not held:
            self._target = stage
            self._queue = [(p, self.durations[p]) for p in self.stages[self.stage].clearance_phases]
            if not self._queue:  # no clearance configured: switch at once
                self._finish_transition()
        return self.phase

    def advance(self, dt: float) -> None:
        """Account for dt seconds of simulated time having passed with the current phase."""
        if not self._queue:
            self.time_in_stage += dt
        else:
            phase, left = self._queue[0]
            left -= dt
            if left <= 1e-9:
                self._queue.pop(0)
            else:
                self._queue[0] = (phase, left)
            if not self._queue:
                self._finish_transition()
        green = None if self.in_transition else self.stage
        self.red_s = [0.0 if k == green else r + dt for k, r in enumerate(self.red_s)]

    def state(self, t: float) -> SignalState:
        """Snapshot of the plan at simulation time t (since scenario start)."""
        return SignalState(t, self.stage, self.time_in_stage, self.in_transition, tuple(self.red_s))

    def _finish_transition(self) -> None:
        self.stage, self._target, self.time_in_stage = self._target, None, 0.0

    @classmethod
    def from_config(
        cls, stages_cfg: list[dict], phase_durations: list[float], min_green_s: float = 0.0
    ) -> SignalPlan:
        stages = [
            Stage(s["name"], int(s["green_phase"]), tuple(s["clearance_phases"]))
            for s in stages_cfg
        ]
        return cls(stages, phase_durations, min_green_s)

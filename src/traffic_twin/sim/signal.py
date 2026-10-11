"""Turns stage requests from a controller into SUMO phases, always running the
clearance phases (pedestrian clearance, yellow, all-red) before a new green."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    green_phase: int
    clearance_phases: tuple[int, ...]


class SignalPlan:
    """Stateful phase sequencer: `request()` a stage, show `phase`, then `advance(dt)`."""

    def __init__(self, stages: list[Stage], phase_durations: list[float]):
        self.stages = stages
        self.durations = phase_durations
        self.stage = 0
        self.time_in_stage = 0.0
        self._queue: list[tuple[int, float]] = []  # pending (phase, remaining s)
        self._target: int | None = None

    @property
    def in_transition(self) -> bool:
        return self._target is not None

    @property
    def phase(self) -> int:
        if self._queue:
            return self._queue[0][0]
        return self.stages[self.stage].green_phase

    def request(self, stage: int) -> int:
        """Ask for a green stage; starts clearance if it differs. Returns the phase to show."""
        if not 0 <= stage < len(self.stages):
            raise ValueError(f"unknown stage {stage}")
        if self._target is None and stage != self.stage:
            self._target = stage
            self._queue = [(p, self.durations[p]) for p in self.stages[self.stage].clearance_phases]
            if not self._queue:  # no clearance configured: switch at once
                self._finish_transition()
        return self.phase

    def advance(self, dt: float) -> None:
        """Account for dt seconds of simulated time having passed with the current phase."""
        if not self._queue:
            self.time_in_stage += dt
            return
        phase, left = self._queue[0]
        left -= dt
        if left <= 1e-9:
            self._queue.pop(0)
        else:
            self._queue[0] = (phase, left)
        if not self._queue:
            self._finish_transition()

    def _finish_transition(self) -> None:
        self.stage, self._target, self.time_in_stage = self._target, None, 0.0

    @classmethod
    def from_config(cls, stages_cfg: list[dict], phase_durations: list[float]) -> SignalPlan:
        stages = [
            Stage(s["name"], int(s["green_phase"]), tuple(s["clearance_phases"]))
            for s in stages_cfg
        ]
        return cls(stages, phase_durations)

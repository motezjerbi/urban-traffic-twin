"""Fixed-time baseline: cycles through the stages with constant green times."""

from __future__ import annotations

from traffic_twin.control.base import Controller, Observation


class FixedTimeController(Controller):
    name = "fixed"

    def __init__(self, green_s: list[float]):
        if not green_s or any(g <= 0 for g in green_s):
            raise ValueError("green times must be positive")
        self.green_s = list(green_s)

    def reset(self, seed: int) -> None:  # deterministic, nothing to reset
        pass

    def decide(self, obs: Observation) -> int:
        if obs.in_transition or obs.time_in_stage < self.green_s[obs.stage]:
            return obs.stage
        return (obs.stage + 1) % len(self.green_s)

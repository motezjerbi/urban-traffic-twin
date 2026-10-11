"""Fixed-time baseline: cycles through the stages with constant green times."""

from __future__ import annotations

from traffic_twin.config import ConfigError, get, stage_names
from traffic_twin.control.base import Controller, Observation
from traffic_twin.control.registry import register


@register("fixed")
class FixedTimeController(Controller):
    def __init__(self, green_s: list[float]):
        if not green_s or any(g <= 0 for g in green_s):
            raise ValueError("green times must be positive")
        self.green_s = list(green_s)

    @classmethod
    def from_config(cls, cfg: dict) -> FixedTimeController:
        green = get(cfg, "fixed.green_s")
        stages = stage_names(cfg)
        missing = [s for s in stages if s not in green]
        if missing:
            raise ConfigError(f"fixed.green_s has no green time for stages {missing}")
        return cls([green[s] for s in stages])

    def reset(self, seed: int) -> None:  # deterministic, nothing to reset
        pass

    def decide(self, obs: Observation) -> int:
        if obs.in_transition or obs.time_in_stage < self.green_s[obs.stage]:
            return obs.stage
        return (obs.stage + 1) % len(self.green_s)

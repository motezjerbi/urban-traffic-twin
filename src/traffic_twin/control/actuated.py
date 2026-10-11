"""S1 actuated baseline (EF-06): SUMO's gap-based `type="actuated"` logic through the
common controller interface, so it shares SignalPlan, the safety guard, the state log
and the audit with every other controller.

Each green lasts at least its min green, is extended while a vehicle crossed one of
the stage's control loops less than `max_gap_s` ago (`Observation.extras["gap_s"]`),
and ends at its max green. Stages run in cycle order, none skipped. With 1 s steps and
max-gap >= detector-gap this takes the same decision as SUMO at every step
(tests/test_actuated_native.py). SUMO's defaults are kept: max-gap 3 s, detector-gap
2 s, passing-time 1.9 s (`detectors.control`).
"""

from __future__ import annotations

from collections.abc import Sequence

from traffic_twin.config import ConfigError, get, stage_names
from traffic_twin.control.base import EXTRA_GAP_S, Controller, Observation
from traffic_twin.control.registry import register
from traffic_twin.control.safety import stage_values

_EPS = 1e-9


def _greens(cfg: dict, key: str, names: list[str]) -> list[float]:
    greens = stage_values(get(cfg, key), names, key)
    if greens is None:
        raise ConfigError(f"missing config key '{key}'")
    return greens


@register("actuated")
class ActuatedController(Controller):
    def __init__(
        self, min_green_s: Sequence[float], max_green_s: Sequence[float], max_gap_s: float = 3.0
    ):
        if not min_green_s or len(min_green_s) != len(max_green_s):
            raise ValueError("need one min and one max green per stage")
        if any(not 0 < lo <= hi for lo, hi in zip(min_green_s, max_green_s, strict=True)):
            raise ValueError("need 0 < min green <= max green for every stage")
        if max_gap_s <= 0:
            raise ValueError("max gap must be positive")
        self.min_green_s = [float(g) for g in min_green_s]
        self.max_green_s = [float(g) for g in max_green_s]
        self.max_gap_s = float(max_gap_s)

    @classmethod
    def from_config(cls, cfg: dict) -> ActuatedController:
        """`actuated.{min_green_s, max_green_s}` by stage name, `actuated.max_gap_s`;
        the greens must lie within `safety.{min_green_s, max_green_s}`."""
        if get(cfg, "detectors.control", None) is None:
            raise ConfigError("actuated needs detectors.control (control loops per stage)")
        names = stage_names(cfg)
        mins = _greens(cfg, "actuated.min_green_s", names)
        maxs = _greens(cfg, "actuated.max_green_s", names)
        floor = float(get(cfg, "safety.min_green_s", 0.0))
        low = [n for n, g in zip(names, mins, strict=True) if g < floor]
        if low:
            raise ConfigError(f"actuated.min_green_s of {low} below safety.min_green_s ({floor:g})")
        caps = stage_values(get(cfg, "safety.max_green_s", None), names, "safety.max_green_s")
        high = [n for n, g, c in zip(names, maxs, caps or maxs, strict=True) if g > c]
        if high:
            raise ConfigError(f"actuated.max_green_s of {high} above safety.max_green_s")
        bad = [n for n, lo, hi in zip(names, mins, maxs, strict=True) if lo > hi]
        if bad:
            raise ConfigError(f"actuated min green above max green for {bad}")
        return cls(mins, maxs, float(get(cfg, "actuated.max_gap_s", 3.0)))

    def reset(self, seed: int) -> None:  # deterministic, nothing to reset
        pass

    def decide(self, obs: Observation) -> int:
        s = obs.stage
        if obs.in_transition or obs.time_in_stage < self.min_green_s[s] - _EPS:
            return s
        nxt = (s + 1) % len(self.min_green_s)
        if obs.time_in_stage >= self.max_green_s[s] - _EPS:
            return nxt  # max-out
        gaps = obs.extras.get(EXTRA_GAP_S)
        if gaps is None:
            raise ConfigError("actuated needs detectors.control: the observation has no gap_s")
        return s if gaps[s] < self.max_gap_s else nxt  # extend, else gap-out

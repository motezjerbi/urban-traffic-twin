"""Common controller interface (spec 8.3). Every controller returns the index of the
green stage it wants; the signal layer handles clearance (yellow, all-red)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Keys of `Observation.extras` filled by the simulation when control loops exist
# (`detectors.control` in the scenario config). Values are tuples indexed by stage:
# - EXTRA_GAP_S, tuple[float, ...]: min time since detection over the stage's control
#   loops (inf if the stage has none)
# - EXTRA_CALLS, tuple[bool, ...]: a vehicle hit one of the stage's loops since its
#   green last ended
EXTRA_GAP_S = "gap_s"
EXTRA_CALLS = "calls"


@dataclass
class Observation:
    t: float  # simulation time since scenario start (s)
    stage: int  # current green stage index; during a transition, the stage being LEFT
    time_in_stage: float  # seconds since the current green started (frozen during clearance)
    in_transition: bool  # True while clearance phases run
    queues: dict[str, int] = field(default_factory=dict)  # halting vehicles per approach
    extras: dict = field(default_factory=dict)  # EXTRA_* keys, when available


class Controller(ABC):
    name: str = "base"  # set by control.registry.register

    @classmethod
    def from_config(cls, cfg: dict) -> Controller:
        """Build from the scenario config (parameters usually under `cfg[name]`)."""
        return cls()

    @abstractmethod
    def reset(self, seed: int) -> None: ...

    @abstractmethod
    def decide(self, obs: Observation) -> int:
        """Return the index of the desired green stage."""

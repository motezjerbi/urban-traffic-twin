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

# Part 4 keys, filled only when the scenario has `network.crossings` (Pitié); on the grid
# the extras are exactly the two keys above:
# - EXTRA_GAP_S_OPEN, tuple[float, ...] per stage: like EXTRA_GAP_S but only over loops
#   whose lane has >= 1 link to an exit lane NOT blocked (inf if none); box clearing
# - EXTRA_PED_WAIT, tuple[float, ...] per crossing (network.crossings key order): the
#   pedestrian shield's input w_c, the longest current curb wait (creeps at the curb
#   ignored, see SimSession._sample_ped_waits) of a person whose next edge is the
#   crossing, sampled after the previous simulation step
# - EXTRA_PED_COUNT, tuple[int, ...] per crossing: persons with a wait > 0 whose next
#   edge is the crossing
# - EXTRA_PED_SLACK, tuple[float, ...] per crossing: theta - (w_c + T_c^fb) now; inf if
#   T_c^fb == 0 or there is no shield
# - EXTRA_EMERGENCY, tuple[EmergencyCall, ...] in (first_seen_s, vid) order
# - EXTRA_PREEMPT, int | None: target stage of the active preemption this step
# - EXTRA_BUS_CALLS, tuple[BusCall, ...]
# - EXTRA_EXIT_BLOCKED, frozenset[str]: exit lanes blocked after the previous step
EXTRA_GAP_S_OPEN = "gap_s_open"
EXTRA_PED_WAIT = "ped_wait"
EXTRA_PED_COUNT = "ped_count"
EXTRA_PED_SLACK = "ped_slack"
EXTRA_EMERGENCY = "emergency"
EXTRA_PREEMPT = "preempt"
EXTRA_BUS_CALLS = "bus_calls"
EXTRA_EXIT_BLOCKED = "exit_blocked"

EVENT_SOURCES = ("controller", "guard", "preemption")


@dataclass
class Observation:
    t: float  # simulation time since scenario start (s)
    stage: int  # current green stage index; during a transition, the stage being LEFT
    time_in_stage: float  # seconds since the current green started (frozen during clearance)
    in_transition: bool  # True while clearance phases run
    queues: dict[str, int] = field(default_factory=dict)  # halting vehicles per approach
    extras: dict = field(default_factory=dict)  # EXTRA_* keys, when available


@dataclass(frozen=True)
class EmergencyCall:
    """An emergency vehicle (vClass `emergency`) heading for the junction."""

    vid: str
    link: int  # TLS link index of its movement
    stage: int  # stage whose green phase shows 'G' on the link, else 'g'
    dist_m: float  # distance to the stop line; math.inf while pending (not inserted)
    first_seen_s: float  # simulation time (since scenario start) of the first detection
    pending: bool  # True while the vehicle waits for insertion
    approach: str | None  # network.approaches key of its approach edge, if known


@dataclass(frozen=True)
class BusCall:
    """A bus on an approach in-edge, before the stop line."""

    vid: str
    line: str
    link: int  # TLS link index of its movement
    stages: tuple[int, ...]  # stages showing G or g on the link in their green phase
    approach: str
    dist_m: float  # distance to the stop line
    speed_mps: float
    dwell_left_s: float  # getStops(v, 1)[0].duration if that stop is on the current edge, else 0


@dataclass(frozen=True)
class Event:
    """One explainable decision: a controller rule, a guard intervention or a
    preemption request / served / timeout. Written to `events.parquet`."""

    t: float  # simulation time since scenario start (s)
    source: str  # one of EVENT_SOURCES
    rule: str
    stage_from: int
    stage_to: int
    detail: str = ""


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

    # -- explainability (concrete; fixed and actuated keep the defaults) --------------
    def report(self) -> dict:
        """Controller activity of the run, written to run.json `control`."""
        return {}

    @property
    def events(self) -> list[Event]:
        """Events logged since the last `clear_events()` (created on first use, so
        subclasses need not call `Controller.__init__`)."""
        events = self.__dict__.get("_events")
        if events is None:
            events = self.__dict__["_events"] = []
        return events

    def log_event(
        self, t: float, rule: str, stage_from: int, stage_to: int, detail: str = ""
    ) -> None:
        """Append an `Event(source="controller")`."""
        self.events.append(Event(t, "controller", rule, stage_from, stage_to, detail))

    def clear_events(self) -> None:
        """Forget the events of a previous run (call it from `reset`)."""
        self.events.clear()

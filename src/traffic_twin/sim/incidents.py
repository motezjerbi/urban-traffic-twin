"""Traffic incidents as temporary lane slowdowns (Parts 5-6, D45), shared by the ETA
worlds and the RL episodes.

An incident sets `lane.setMaxSpeed(lane, speed_mps)` (2 m/s at Pitié) from its start for
its duration, then restores the lane's original maximum speed: a partly blocked lane
(ASSUMED). The sites are in `configs/pitie/incident_sites.yaml`, which no scenario
extends, so the Part 3-4 configs and hashes are unchanged; see that file for why a lane
closure or a stopped vehicle is not used.

    sites = load_sites()
    incidents = sample_incidents(rng, sites, count_p=[0.5, 0.35, 0.15],
                                 start_s=(300, 6000), durations_s=[300, 600, 900, 1200])
    manager = IncidentManager(sim.traci, incidents)
    while not sim.done:
        manager.update(sim.t)          # before the step at time t
        sim.step(controller.decide(sim.observe()))

`IncidentManager.active()` gives the incidents in force with their actual start (the
`JunctionSampler` reports them); `log()` lists planned and actual starts and ends.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from traffic_twin.config import ConfigError, load_config
from traffic_twin.paths import INCIDENT_SITES

EPS = 1e-9


@dataclass(frozen=True)
class Incident:
    """A planned incident: lane `lane` (site `site` on arm `arm`) slowed to `speed_mps`
    from `start_s` (s since scenario start) for `duration_s`."""

    site: str
    arm: str
    lane: str
    start_s: float
    duration_s: float
    speed_mps: float


@dataclass(frozen=True)
class ActiveIncident:
    """An incident in force: `since_s` is its actual start (s since scenario start)."""

    site: str
    arm: str
    lane: str
    since_s: float


def load_sites(path: str | Path = INCIDENT_SITES) -> dict:
    """The incident sites file: `{speed_mps, sites: [{id, arm, lane}, ...]}`, checked
    (positive speed, at least one site, unique ids)."""
    cfg = load_config(path, ("speed_mps", "sites"))
    speed = cfg["speed_mps"]
    if isinstance(speed, bool) or not isinstance(speed, int | float) or speed <= 0:
        raise ConfigError(f"{path}: speed_mps must be > 0, got {speed!r}")
    sites = cfg["sites"]
    if not isinstance(sites, list) or not sites:
        raise ConfigError(f"{path}: sites must be a non-empty list")
    ids = []
    for s in sites:
        if not isinstance(s, dict) or not all(s.get(k) for k in ("id", "arm", "lane")):
            raise ConfigError(f"{path}: every site needs id, arm and lane: {s!r}")
        ids.append(str(s["id"]))
    if len(set(ids)) != len(ids):
        raise ConfigError(f"{path}: duplicate site ids {ids}")
    keys = ("id", "arm", "lane")
    return {"speed_mps": float(speed), "sites": [{k: str(s[k]) for k in keys} for s in sites]}


def sample_incidents(
    rng: np.random.Generator,
    sites: dict,
    count_p: Sequence[float],
    start_s: tuple[float, float],
    durations_s: Sequence[float],
) -> tuple[Incident, ...]:
    """Draw the incidents of one world or episode, sorted by (start, site).

    The count is k with probability `count_p[k]`; each incident gets a site uniformly
    from `sites["sites"]`, a start uniform over the whole seconds of [start_s[0],
    start_s[1]) and a duration uniformly from `durations_s`. The draws use `rng` only,
    in a fixed order, so a seeded generator gives the same incidents every time."""
    p = np.asarray(count_p, dtype=float)
    if p.ndim != 1 or not len(p) or (p < 0).any() or abs(p.sum() - 1.0) > 1e-9:
        raise ValueError(f"count_p must be probabilities summing to 1, got {list(count_p)}")
    lo, hi = int(start_s[0]), int(start_s[1])
    if hi <= lo:
        raise ValueError(f"start_s must be an increasing pair, got {start_s}")
    if not durations_s:
        raise ValueError("durations_s is empty")
    site_list = sites["sites"]
    n = int(rng.choice(len(p), p=p))
    out = []
    for _ in range(n):
        site = site_list[int(rng.integers(len(site_list)))]
        start = float(rng.integers(lo, hi))
        duration = float(durations_s[int(rng.integers(len(durations_s)))])
        out.append(
            Incident(site["id"], site["arm"], site["lane"], start, duration, sites["speed_mps"])
        )
    return tuple(sorted(out, key=lambda i: (i.start_s, i.site)))


class IncidentManager:
    """Applies planned incidents to a running simulation through TraCI.

    Call `update(t)` before the simulation step at time t (s since scenario start). An
    incident starts at the first update with t >= start_s and ends at the first update
    with t >= actual start + duration. Overlapping incidents on one lane share the
    original speed saved by the first: the lane is restored when the last one ends."""

    def __init__(self, traci: ModuleType, incidents: Sequence[Incident]) -> None:
        self.traci = traci
        self.incidents = tuple(sorted(incidents, key=lambda i: (i.start_s, i.site)))
        self._started: dict[int, float] = {}  # index -> actual start
        self._ended: dict[int, float] = {}  # index -> actual end
        self._original: dict[str, float] = {}  # lane -> original max speed while slowed
        self._count: dict[str, int] = {}  # lane -> incidents in force on it

    def update(self, t: float) -> None:
        lane = self.traci.lane
        for k, inc in enumerate(self.incidents):
            if k in self._started and k not in self._ended:
                if t >= self._started[k] + inc.duration_s - EPS:
                    self._ended[k] = float(t)
                    self._count[inc.lane] -= 1
                    if self._count[inc.lane] == 0:
                        lane.setMaxSpeed(inc.lane, self._original.pop(inc.lane))
        for k, inc in enumerate(self.incidents):
            if k not in self._started and t >= inc.start_s - EPS:
                self._started[k] = float(t)
                if self._count.get(inc.lane, 0) == 0:
                    self._original[inc.lane] = float(lane.getMaxSpeed(inc.lane))
                self._count[inc.lane] = self._count.get(inc.lane, 0) + 1
                lane.setMaxSpeed(inc.lane, float(inc.speed_mps))

    def active(self) -> tuple[ActiveIncident, ...]:
        """Incidents in force, by actual start."""
        return tuple(
            ActiveIncident(inc.site, inc.arm, inc.lane, self._started[k])
            for k, inc in enumerate(self.incidents)
            if k in self._started and k not in self._ended
        )

    def log(self) -> list[dict]:
        """One row per planned incident: site, lane, planned_start_s, actual_start_s
        (None if not started), end_s (None while in force or not started)."""
        return [
            {
                "site": inc.site,
                "lane": inc.lane,
                "planned_start_s": inc.start_s,
                "actual_start_s": self._started.get(k),
                "end_s": self._ended.get(k),
            }
            for k, inc in enumerate(self.incidents)
        ]

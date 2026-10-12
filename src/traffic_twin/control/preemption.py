"""Emergency vehicle preemption (Part 4, D20, D24; spec EF-07, A-05). Pure Python, no
TraCI: the detection is `sim.emergency.EmergencyDetector`.

The manager picks the active request first come, first served on (first_seen_s, vid)
and gives its target stage to the Safety Guard (`SignalState.preempt_target`), which
heads for it through link-safe hops (`next_hop`) after min green, never cutting a
clearance and never breaking the pedestrian shield. A request is held until the vehicle
passes the stop line ('served') or dropped once it has gone `max_hold_s` without getting
`progress_m` closer to the stop line ('timeout'); a dropped vehicle is never requested
again.

Time-out clock (D24, amended in Part 4 gate round 3): the clock starts at the first
detection and restarts whenever the vehicle is at least `progress_m` closer to the stop
line than when the clock last (re)started (a pending vehicle, `dist_m` inf, makes no
progress until its insertion, which restarts the clock). A stationary ambulance
therefore still times out exactly `max_hold_s` after its first detection, as D24 states,
and gives the junction back ("a stuck ambulance gives the junction back"). An ambulance
that is still moving up its queue is not dropped: the pedestrian shield (R5) can cut a
hold and force a detour of up to one cycle (the longest legal S2 cycle, 177 s, is longer
than 120 s), and dropping the request then would count a moving ambulance as timed out
just before its target came back (calibration seed 101: first detection 2401 s, the
shield cut HOP at 2458 s while the ambulance was 23 m from the stop line, it waited at
the line from 2466 s and passed it at 2535 s, 134 s after its first detection; a clock
from the first detection alone dropped it at 2522 s, while HOP was already on its way
back). Ambulance travel times are unchanged by the amendment on 101-103; only the served
/ timeout bookkeeping of such a request is.

Config (`preemption:` block, D24): `enabled` (absent or false: no manager), `max_hold_s`
(default 120), `progress_m` (default 1), `respects_ped_max_wait` (must stay true: A-04
forbids a hold beyond the pedestrian bound), `detection.{pending, max_dist_m}` (read by
the detector).
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence

from traffic_twin.config import ConfigError, get
from traffic_twin.control.base import EmergencyCall, Event

DEFAULT_MAX_HOLD_S = 120.0
DEFAULT_PROGRESS_M = 1.0  # getting this much closer to the stop line restarts the clock
EVENT_RULES = ("request", "served", "timeout")  # Event.rule of the manager's events


def next_hop(
    s: int,
    target: int,
    graph: Sequence[frozenset[int]],
    clearance_s: Sequence[float],
    min_green_s: float,
) -> int:
    """First stage on the cheapest link-safe path from s to target (s if s == target).

    Dijkstra over `graph` (graph[k] = stages reachable from k through k's clearance,
    `SafetyGuard.allowed`): leaving s costs clearance_s[s] (s's min green is the guard's
    business), each intermediate stage k costs min_green_s + clearance_s[k]. Among the
    cheapest paths the first hop j closest after s in cycle order wins, i.e. the smallest
    (j - s) % n. A target that cannot be reached gives the successor (always link-safe in
    a valid plan, so the guard can keep cycling)."""
    n = len(graph)
    if len(clearance_s) != n:
        raise ValueError(f"need one clearance per stage ({len(clearance_s)} for {n})")
    if not (0 <= s < n and 0 <= target < n):
        raise ValueError(f"stage {s} or target {target} outside 0..{n - 1}")
    if s == target:
        return s
    # (cost, cycle rank of the first hop, stage, first hop)
    heap = [(float(clearance_s[s]), (j - s) % n, j, j) for j in graph[s] if j != s]
    heapq.heapify(heap)
    done: set[int] = set()
    while heap:
        cost, rank, k, first = heapq.heappop(heap)
        if k == target:
            return first
        if k in done:
            continue
        done.add(k)
        step = float(min_green_s) + float(clearance_s[k])
        for j in graph[k]:
            if j not in done and j != s:
                heapq.heappush(heap, (cost + step, rank, j, first))
    return (s + 1) % n


class PreemptionManager:
    """FCFS preemption requests with served / timeout bookkeeping.

    `update(t, calls)` is called once per simulation step with the detector's calls. A
    vehicle's request starts at its first call ('request' event). It ends when the
    vehicle is no longer called while its request is open ('served': it passed the stop
    line), or when t - last_progress_s > max_hold_s ('timeout'), last_progress_s being
    first_seen_s or the last time the vehicle got progress_m closer to the stop line than
    at the previous (re)start of that clock. Ended vehicles are ignored from then on.
    `active_s` counts dt for every step with an active target."""

    def __init__(
        self,
        max_hold_s: float = DEFAULT_MAX_HOLD_S,
        *,
        dt: float = 1.0,
        progress_m: float = DEFAULT_PROGRESS_M,
    ):
        if not max_hold_s > 0:
            raise ConfigError(f"preemption.max_hold_s must be > 0, got {max_hold_s}")
        if not progress_m > 0:
            raise ConfigError(f"preemption.progress_m must be > 0, got {progress_m}")
        self.max_hold_s = float(max_hold_s)
        self.progress_m = float(progress_m)
        self.dt = float(dt)
        self.events: list[Event] = []  # request / served / timeout (source "preemption")
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict) -> PreemptionManager | None:
        """Manager for a scenario; None unless `preemption.enabled`."""
        pre = cfg.get("preemption") or {}
        if not isinstance(pre, dict):
            raise ConfigError("preemption must be a mapping")
        if not pre.get("enabled", False):
            return None
        if not pre.get("respects_ped_max_wait", True):
            raise ConfigError(
                "preemption.respects_ped_max_wait: false is refused: A-04 bounds every "
                "pedestrian wait at safety.ped_max_wait_s, preemption included (D20)"
            )
        numbers = {}
        for key, default in (
            ("max_hold_s", DEFAULT_MAX_HOLD_S),
            ("progress_m", DEFAULT_PROGRESS_M),
        ):
            value = pre.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigError(f"preemption.{key} must be a number, got {value!r}")
            numbers[key] = float(value)
        return cls(
            numbers["max_hold_s"],
            dt=float(get(cfg, "step_length_s", 1.0)),
            progress_m=numbers["progress_m"],
        )

    def reset(self) -> None:
        """Forget every request (start of a run)."""
        self.events.clear()
        self.requests = 0
        self.served = 0
        self.timeouts = 0
        self.active_s = 0.0
        self.target: int | None = None  # target stage returned by the last update
        self._open: dict[str, EmergencyCall] = {}  # open requests: vid -> last call
        self._ended: set[str] = set()  # served or timed out: never requested again
        # time-out clock: vid -> (anchor distance to the stop line, start of the clock);
        # (first call's distance, first_seen_s), then reset on every progress_m gain
        self._progress: dict[str, tuple[float, float]] = {}

    def update(self, t: float, calls: Sequence[EmergencyCall]) -> int | None:
        """Target stage of the active request at time t, or None.

        FCFS on (first_seen_s, vid), dropping timed-out vids (t - last_progress_s >
        max_hold_s); a vid seen before and absent now is 'served'."""
        present: dict[str, EmergencyCall] = {}
        for c in calls:
            if c.vid in self._ended or c.vid in present:
                continue
            present[c.vid] = c
        for vid in [v for v in self._open if v not in present]:
            self._end(t, "served", self._open.pop(vid))
        active: list[EmergencyCall] = []
        for vid, c in present.items():
            if vid not in self._open:
                self.requests += 1
                self._log(t, "request", c)
            self._open[vid] = c
            if t - self._last_progress(t, c) > self.max_hold_s:
                self._end(t, "timeout", self._open.pop(vid))
            else:
                active.append(c)
        if not active:
            self.target = None
            return None
        best = min(active, key=lambda c: (c.first_seen_s, c.vid))
        self.target = best.stage
        self.active_s += self.dt
        return best.stage

    @property
    def open_requests(self) -> tuple[str, ...]:
        """Vehicles with an open request, in (first_seen_s, vid) order."""
        calls = sorted(self._open.values(), key=lambda c: (c.first_seen_s, c.vid))
        return tuple(c.vid for c in calls)

    def report(self) -> dict:
        """{requests, served, timeouts, active_s} of the run. Requests still open at the
        end are in `requests` only."""
        return {
            "requests": self.requests,
            "served": self.served,
            "timeouts": self.timeouts,
            "active_s": self.active_s,
        }

    # -- helpers ----------------------------------------------------------------------
    def _last_progress(self, t: float, call: EmergencyCall) -> float:
        """Start of the call's time-out clock after recording its distance at t.

        The clock starts at first_seen_s with the first call's distance as anchor. It
        restarts at t, with the distance at t as the new anchor, when the vehicle is at
        least progress_m closer to the stop line than the anchor; small gains add up.
        A pending vehicle (dist_m inf) has an infinite anchor, so its insertion restarts
        the clock."""
        if call.vid not in self._progress:
            self._progress[call.vid] = (call.dist_m, call.first_seen_s)
        anchor, since = self._progress[call.vid]
        if math.isfinite(call.dist_m) and call.dist_m <= anchor - self.progress_m:
            anchor, since = call.dist_m, t
            self._progress[call.vid] = (anchor, since)
        return since

    def _end(self, t: float, rule: str, call: EmergencyCall) -> None:
        self._ended.add(call.vid)
        if rule == "served":
            self.served += 1
        else:
            self.timeouts += 1
        self._log(t, rule, call)

    def _log(self, t: float, rule: str, call: EmergencyCall) -> None:
        """Event with stage_from = stage_to = the call's target stage (a preemption event
        is a request about a stage, not a transition)."""
        dist = "inf" if math.isinf(call.dist_m) else f"{call.dist_m:.1f}"
        detail = f"vid={call.vid}, link={call.link}, dist={dist}"
        self.events.append(Event(t, "preemption", rule, call.stage, call.stage, detail))

"""Emergency vehicle detection for preemption (Part 4, D24; spec EF-07).

`EmergencyDetector.calls(t)` lists every vClass `emergency` vehicle heading for the
traffic light:
- inserted vehicles that still have the light ahead (`vehicle.getNextTLS`), which gives
  the TLS link of their movement and the distance to the stop line. Once a vehicle is
  past the stop line the light is no longer ahead and its call disappears (the
  preemption manager then counts it as served);
- with `pending`, vehicles waiting for insertion (`simulation.getPendingVehicles`, e.g.
  behind a queue spilling back past the network edge), standing for upstream or V2I
  detection (ASSUMED). Their link is the TLS link from the first to the second edge of
  their route (`vehicle.getRoute` works on pending vehicles, SUMO 1.28); a route whose
  first two edges are not a TLS link is ignored. Their distance is `math.inf`.

The target stage of a call is the stage whose green phase shows 'G' on its link, else
'g' (first in cycle order); a link green in no stage gives no call. `first_seen_s` is
kept per vehicle, so a pending vehicle keeps it once inserted. The vehicle class is
read once per vehicle id.

Config: `preemption.detection.{pending, max_dist_m}`; defaults pending=True,
max_dist_m=None (no distance limit). `max_dist_m` applies to inserted vehicles only;
pending ones are governed by `pending`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import ModuleType

from traffic_twin.config import ConfigError, approach_in_edges, get
from traffic_twin.control.base import EmergencyCall
from traffic_twin.sim.junction import edge_links, lane_edge, lane_links
from traffic_twin.sim.signal import Stage, link_stages

EMERGENCY_CLASS = "emergency"  # SUMO vClass of the vehicles that may request preemption


class EmergencyDetector:
    """Build it after `traci.start` (it reads the controlled links of the light)."""

    def __init__(
        self,
        traci: ModuleType,
        tls_id: str,
        stages: Sequence[Stage],
        phase_states: Sequence[str],
        approaches: dict[str, list[str]],
        pending: bool = True,
        max_dist_m: float | None = None,
    ):
        self.traci, self.tls_id = traci, tls_id
        self.stages, self.phase_states = list(stages), list(phase_states)
        self.approaches = approaches
        self.pending = pending
        self.max_dist_m = max_dist_m
        self.first_seen: dict[str, float] = {}  # vid -> time of its first call
        self._is_emergency: dict[str, bool] = {}  # vid -> vClass is emergency
        self._pending_link: dict[str, int | None] = {}  # vid -> link of its route start
        self._link_of = edge_links(traci, tls_id)  # (from edge, to edge) -> link
        approach_of = {e: name for name, edges in approaches.items() for e in edges}
        self._approach: dict[int, str | None] = {
            link: approach_of.get(lane_edge(lanes[0]))
            for link, lanes in lane_links(traci, tls_id).items()
        }
        self._stage: dict[int, int | None] = {}  # link -> target stage (None: never green)

    @classmethod
    def from_config(
        cls,
        traci: ModuleType,
        cfg: dict,
        stages: Sequence[Stage],
        phase_states: Sequence[str],
    ) -> EmergencyDetector:
        """Detector of a scenario: `network.tls_id`, the approach in-edges and
        `preemption.detection` (read even when preemption is off: the calls are an
        observation, e.g. for the RL state)."""
        det = get(cfg, "preemption.detection", None) or {}
        if not isinstance(det, dict):
            raise ConfigError("preemption.detection must be a mapping")
        pending = det.get("pending", True)
        if not isinstance(pending, bool):
            raise ConfigError(f"preemption.detection.pending must be true or false: {pending!r}")
        max_dist = det.get("max_dist_m")
        if max_dist is not None and (
            isinstance(max_dist, bool) or not isinstance(max_dist, int | float) or max_dist <= 0
        ):
            raise ConfigError(f"preemption.detection.max_dist_m must be > 0 or null: {max_dist!r}")
        return cls(
            traci, cfg["network"]["tls_id"], stages, phase_states, approach_in_edges(cfg),
            pending=pending, max_dist_m=None if max_dist is None else float(max_dist),
        )  # fmt: skip

    def calls(self, t: float) -> tuple[EmergencyCall, ...]:
        """Emergency calls at time t (since scenario start), in (first_seen_s, vid) order."""
        vehicle = self.traci.vehicle
        out: list[EmergencyCall] = []
        for vid in vehicle.getIDList():
            if not self._emergency(vid):
                continue
            ahead = next((x for x in vehicle.getNextTLS(vid) if x[0] == self.tls_id), None)
            if ahead is None:  # past the stop line (or never through this light)
                continue
            link, dist = int(ahead[1]), float(ahead[2])
            if self.max_dist_m is not None and dist > self.max_dist_m:
                continue
            call = self._call(vid, link, dist, t, pending=False)
            if call is not None:
                out.append(call)
        if self.pending:
            for vid in self.traci.simulation.getPendingVehicles():
                if not self._emergency(vid):
                    continue
                link = self._route_link(vid)
                call = None if link is None else self._call(vid, link, math.inf, t, pending=True)
                if call is not None:
                    out.append(call)
        return tuple(sorted(out, key=lambda c: (c.first_seen_s, c.vid)))

    def target_stage(self, link: int) -> int | None:
        """Stage whose green phase shows 'G' on `link`, else 'g'; None if neither."""
        if link not in self._stage:
            full, permissive = link_stages(self.stages, self.phase_states, link)
            self._stage[link] = (full or permissive or (None,))[0]
        return self._stage[link]

    # -- helpers ----------------------------------------------------------------------
    def _emergency(self, vid: str) -> bool:
        flag = self._is_emergency.get(vid)
        if flag is None:
            flag = self._is_emergency[vid] = (
                self.traci.vehicle.getVehicleClass(vid) == EMERGENCY_CLASS
            )
        return flag

    def _route_link(self, vid: str) -> int | None:
        """TLS link from the first to the second edge of a pending vehicle's route."""
        if vid not in self._pending_link:
            route = list(self.traci.vehicle.getRoute(vid))
            key = (route[0], route[1]) if len(route) >= 2 else None
            self._pending_link[vid] = None if key is None else self._link_of.get(key)
        return self._pending_link[vid]

    def _call(
        self, vid: str, link: int, dist: float, t: float, pending: bool
    ) -> EmergencyCall | None:
        stage = self.target_stage(link)
        if stage is None:
            return None
        first = self.first_seen.setdefault(vid, float(t))
        return EmergencyCall(
            vid=vid, link=link, stage=stage, dist_m=dist, first_seen_s=first,
            pending=pending, approach=self._approach.get(link),
        )  # fmt: skip

"""Junction-level TraCI helpers and the junction monitor (Part 4).

- `edge_links`, `lane_links`: map the controlled links of a traffic light to edges and
  lanes (used by the emergency and bus detectors, box clearing and the monitor);
- `lane_exits`, `box_lanes`: the exit lanes reached from each approach lane, and the
  internal lanes of the junction with the link they belong to;
- `JunctionMonitor`: per-step observer of the box (vehicles halted inside the junction),
  the exit lanes (blocked exits) and the pedestrian crossings (exposure, red entries).
  It only reads TraCI, so it never changes a run (passivity test).

SUMO lane ids are `<edge id>_<lane index>` (internal lanes too: `:C_0_0` is lane 0 of
edge `:C_0`), so the edge of a lane is its id up to the last underscore.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType

GREEN = "Gg"  # SUMO link states that let traffic through


def lane_edge(lane_id: str) -> str:
    """Edge id of a SUMO lane id (`n_in_2` -> `n_in`, `:C_c0_0` -> `:C_c0`)."""
    edge, sep, index = lane_id.rpartition("_")
    if not sep or not edge or not index.isdigit():
        raise ValueError(f"not a SUMO lane id: {lane_id!r}")
    return edge


def lane_links(traci: ModuleType, tls_id: str) -> dict[int, tuple[str, str, str]]:
    """TLS link index -> (from lane, to lane, via lane) of its first connection.

    `trafficlight.getControlledLinks` lists, per link index, the connections it controls
    (usually one); unused indices (empty lists) are left out."""
    out: dict[int, tuple[str, str, str]] = {}
    for i, conns in enumerate(traci.trafficlight.getControlledLinks(tls_id)):
        if conns:
            from_lane, to_lane, via = conns[0]
            out[i] = (from_lane, to_lane, via)
    return out


def edge_links(traci: ModuleType, tls_id: str) -> dict[tuple[str, str], int]:
    """(from edge, to edge) -> the first (lowest) TLS link index connecting them, over
    every connection of every link."""
    out: dict[tuple[str, str], int] = {}
    for i, conns in enumerate(traci.trafficlight.getControlledLinks(tls_id)):
        for from_lane, to_lane, _via in conns:
            out.setdefault((lane_edge(from_lane), lane_edge(to_lane)), i)
    return out


def lane_exits(links: Mapping[int, tuple[str, str, str]]) -> dict[str, frozenset[str]]:
    """From lane -> the exit lanes (non-internal `to` lanes) its links lead to, from
    `lane_links` output. Crossing links (from a walking area, `:` lanes) are left out."""
    out: dict[str, set[str]] = {}
    for from_lane, to_lane, _via in links.values():
        if not from_lane.startswith(":") and not to_lane.startswith(":"):
            out.setdefault(from_lane, set()).add(to_lane)
    return {lane: frozenset(exits) for lane, exits in out.items()}


def box_lanes(traci: ModuleType, links: Mapping[int, tuple[str, str, str]]) -> dict[str, int]:
    """Internal lane -> TLS link, for the given (vehicle) links: each link's via lane
    and, when an internal junction splits the link, the internal lanes that follow it
    (the approached internal lane of `lane.getLinks(via)`, followed transitively)."""
    out: dict[str, int] = {}
    for i, (_from, _to, via) in sorted(links.items()):
        todo = [via] if via else []
        while todo:
            lane_id = todo.pop()
            if lane_id in out:
                continue
            out[lane_id] = i
            for link in traci.lane.getLinks(lane_id):
                nxt = link[4] if len(link) > 4 else ""
                if nxt and nxt.startswith(":") and nxt not in out:
                    todo.append(nxt)
    return out


class JunctionMonitor:
    """Box, exit-lane and crossing observer (spec EF-09, KPIs of section 13.2).

    `after_step(phase, dt)` samples TraCI after each simulation step that showed `phase`
    (an index into `phase_states`) for dt seconds:
    - box: vehicles halted (speed < `box_halt_mps`) on an internal vehicle lane of the
      junction (see `box_lanes`; crossings and walking areas are not box lanes).
      `box_halt_red_s` sums dt per such vehicle whose link is neither G nor g;
      `box_foe_onsets` counts, per halted vehicle, the steps where a foe of its link
      turns green (G or g after a phase where it was not);
    - exits: an exit lane (the `to` lane of a vehicle link) is blocked when a vehicle on
      it is halted (speed < `exit_halt_mps`) within `exit_zone_m` of its start.
      `exit_blocked_s` sums dt per blocked lane, `exit_blocked_green_s` only while a link
      feeding the lane shows G or g;
    - crossings (`network.crossings`: name -> {edge, link}): a traversal starts when a
      person is first seen on a crossing edge. It is exposed when a foe *vehicle* link of
      the crossing showed `G` while the person was on it (a permissive `g` turner
      yields), and a red entry when the crossing's own link showed `r` at the entry.

    `foes` maps a TLS link to its conflicting links (`tls_check.load_program(...).foes`);
    the relation is made symmetric. Read only: the monitor never changes a run.
    """

    def __init__(
        self,
        traci: ModuleType,
        tls_id: str,
        phase_states: list[str],
        crossings: dict,
        foes: Mapping[int, frozenset[int]],
        exit_zone_m: float = 15.0,
        exit_halt_mps: float = 0.5,
        box_halt_mps: float = 0.1,
    ):
        self.traci, self.tls_id = traci, tls_id
        self.phase_states = list(phase_states)
        self.crossings = crossings
        self.exit_zone_m = float(exit_zone_m)
        self.exit_halt_mps = float(exit_halt_mps)
        self.box_halt_mps = float(box_halt_mps)

        sym: dict[int, set[int]] = {}
        for a, fs in foes.items():
            for b in fs:
                sym.setdefault(int(a), set()).add(int(b))
                sym.setdefault(int(b), set()).add(int(a))
        self.foes = {k: frozenset(v) for k, v in sym.items()}

        cross_links = {int(c["link"]) for c in crossings.values()}
        vehicle = {
            i: lk
            for i, lk in lane_links(traci, tls_id).items()
            if i not in cross_links and not lk[0].startswith(":")
        }
        self.box_lanes = box_lanes(traci, vehicle)  # internal lane -> link
        feeders: dict[str, list[int]] = {}
        for i, (_from, to, _via) in sorted(vehicle.items()):
            if not to.startswith(":"):
                feeders.setdefault(to, []).append(i)
        self.feeders = {lane: tuple(ids) for lane, ids in sorted(feeders.items())}
        self.exit_lanes = tuple(self.feeders)
        self.cross_link = {c["edge"]: int(c["link"]) for c in crossings.values()}
        self.cross_foes = {
            edge: tuple(sorted(k for k in self.foes.get(link, ()) if k in vehicle))
            for edge, link in self.cross_link.items()
        }
        self.reset()

    def reset(self) -> None:
        """Forget every sample (a new run)."""
        self.traversals = self.exposed = self.red_entries = self.box_foe_onsets = 0
        self.box_halt_red_s = self.exit_blocked_s = self.exit_blocked_green_s = 0.0
        self._on: dict[str, list] = {}  # person -> [crossing edge, exposed]
        self._blocked: frozenset[str] = frozenset()
        self._prev_phase: int | None = None

    def after_step(self, phase: int, dt: float) -> None:
        """Sample the junction after a simulation step that showed `phase` for dt s."""
        state = self.phase_states[phase]
        self._sample_crossings(state)
        self._sample_box(phase, state, dt)
        self._sample_exits(state, dt)
        self._prev_phase = phase

    def blocked_exits(self) -> frozenset[str]:
        """Exit lanes with a vehicle halted near their start after the last step."""
        return self._blocked

    def summary(self) -> dict[str, float]:
        """Junction and pedestrian-crossing KPI values of the run (plan section 4.6).
        `crossing_complete_rate` is left out when nobody crossed."""
        out: dict[str, float] = {}
        if self.traversals:
            out["crossing_complete_rate"] = 1.0 - self.exposed / self.traversals
        out["crossing_red_entries"] = float(self.red_entries)
        out["box_halt_red_s"] = float(self.box_halt_red_s)
        out["box_foe_onsets"] = float(self.box_foe_onsets)
        out["exit_blocked_s"] = float(self.exit_blocked_s)
        out["exit_blocked_green_s"] = float(self.exit_blocked_green_s)
        return out

    # -- internals -------------------------------------------------------------------
    def _sample_crossings(self, state: str) -> None:
        person = self.traci.person
        seen = set()
        for pid in person.getIDList():
            edge = person.getRoadID(pid)
            link = self.cross_link.get(edge)
            if link is None:
                continue
            seen.add(pid)
            rec = self._on.get(pid)
            if rec is None or rec[0] != edge:  # a new traversal
                rec = self._on[pid] = [edge, False]
                self.traversals += 1
                self.red_entries += state[link] == "r"
            if not rec[1] and any(state[k] == "G" for k in self.cross_foes[edge]):
                rec[1] = True
                self.exposed += 1
        for pid in [p for p in self._on if p not in seen]:
            del self._on[pid]

    def _sample_box(self, phase: int, state: str, dt: float) -> None:
        lane, vehicle = self.traci.lane, self.traci.vehicle
        halted = []  # the link of each halted box vehicle
        for box_lane, link in self.box_lanes.items():
            for vid in lane.getLastStepVehicleIDs(box_lane):
                if vehicle.getSpeed(vid) < self.box_halt_mps:
                    halted.append(link)
                    if state[link] not in GREEN:
                        self.box_halt_red_s += dt
        if halted and self._prev_phase is not None and phase != self._prev_phase:
            before = self.phase_states[self._prev_phase]
            onsets = {
                i
                for i, (x, y) in enumerate(zip(before, state, strict=True))
                if y in GREEN and x not in GREEN
            }
            self.box_foe_onsets += sum(
                bool(onsets & self.foes.get(lk, frozenset())) for lk in halted
            )

    def _sample_exits(self, state: str, dt: float) -> None:
        lane, vehicle = self.traci.lane, self.traci.vehicle
        blocked = set()
        for exit_lane in self.exit_lanes:
            for vid in lane.getLastStepVehicleIDs(exit_lane):
                if (
                    vehicle.getLanePosition(vid) < self.exit_zone_m
                    and vehicle.getSpeed(vid) < self.exit_halt_mps
                ):
                    blocked.add(exit_lane)
                    self.exit_blocked_s += dt
                    if any(state[i] in GREEN for i in self.feeders[exit_lane]):
                        self.exit_blocked_green_s += dt
                    break
        self._blocked = frozenset(blocked)

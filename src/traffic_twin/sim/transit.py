"""Bus detection for transit signal priority (Part 4, D23; spec EF-08).

`BusDetector.calls(t)` lists every vClass `bus` vehicle on an approach in-edge, before
the stop line, as a `BusCall`:
- `link`: its TLS link (`vehicle.getNextTLS`), `stages`: the stages whose green phase
  shows G or g on that link (cycle order);
- `dist_m`, `speed_mps`: distance to the stop line and current speed;
- `dwell_left_s`: `vehicle.getStops(v, 1)[0].duration` when that next stop is on the
  bus's current edge, else 0. Checked on SUMO 1.28 (plan section 2): before the bus
  reaches the stop this is the full scheduled dwell, while it dwells it is the
  **remaining** dwell, and the stop leaves the list once served.

Detection is ideal (TraCI positions, ASSUMED): it stands for AVL or a check-in loop with
`vTypes="bus"` plus the stop-line loop. A bus leaves the calls as soon as it passes the
stop line (it is then on an internal lane of the junction, not on an in-edge).
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType

from traffic_twin.control.base import BusCall
from traffic_twin.sim.junction import lane_edge
from traffic_twin.sim.signal import Stage, link_stages

BUS_VCLASS = "bus"


class BusDetector:
    def __init__(
        self,
        traci: ModuleType,
        tls_id: str,
        stages: Sequence[Stage],
        phase_states: Sequence[str],
        approaches: dict[str, list[str]],
    ):
        self.traci, self.tls_id = traci, tls_id
        self.stages, self.phase_states = list(stages), list(phase_states)
        self.approaches = approaches
        # in-edge -> approach key (first approach listing it), in config order
        self._edge_approach: dict[str, str] = {}
        for name, edges in approaches.items():
            for e in edges:
                self._edge_approach.setdefault(e, name)
        self._is_bus: dict[str, bool] = {}  # vClass per vehicle id, read once
        self._serving: dict[int, tuple[int, ...]] = {}  # link -> stages with G or g

    def _bus(self, vid: str) -> bool:
        known = self._is_bus.get(vid)
        if known is None:
            known = self._is_bus[vid] = self.traci.vehicle.getVehicleClass(vid) == BUS_VCLASS
        return known

    def stages_of(self, link: int) -> tuple[int, ...]:
        """Stages showing G or g on `link` in their green phase, in cycle order."""
        out = self._serving.get(link)
        if out is None:
            full, permissive = link_stages(self.stages, self.phase_states, link)
            out = self._serving[link] = tuple(sorted(full + permissive))
        return out

    def _dwell_left(self, vid: str, edge: str) -> float:
        stops = self.traci.vehicle.getStops(vid, 1)
        if stops and lane_edge(stops[0].lane) == edge:
            return max(float(stops[0].duration), 0.0)
        return 0.0

    def calls(self, t: float) -> tuple[BusCall, ...]:
        """Bus calls at time t (since scenario start), by distance to the stop line."""
        veh = self.traci.vehicle
        out: list[BusCall] = []
        for edge, approach in self._edge_approach.items():
            for vid in self.traci.edge.getLastStepVehicleIDs(edge):
                if not self._bus(vid):
                    continue
                nxt = [x for x in veh.getNextTLS(vid) if x[0] == self.tls_id]
                if not nxt:  # no stop line of this junction ahead (route ends before it)
                    continue
                _tls, link, dist, _state = nxt[0]
                out.append(
                    BusCall(
                        vid=vid,
                        line=str(veh.getLine(vid)),
                        link=int(link),
                        stages=self.stages_of(int(link)),
                        approach=approach,
                        dist_m=float(dist),
                        speed_mps=float(veh.getSpeed(vid)),
                        dwell_left_s=self._dwell_left(vid, edge),
                    )
                )
        out.sort(key=lambda c: (c.dist_m, c.vid))
        return tuple(out)

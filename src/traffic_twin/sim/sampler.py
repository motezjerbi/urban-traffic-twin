"""Raw junction state for the ETA features and the RL observation (Parts 5-6, contract
4.2): one `JunctionSnapshot` per call of `JunctionSampler.sample()`.

The sampler is the single reader of the raw state beyond the controller's Observation.
`ml/features.py` (ETA, D41) and `rl/features.py` (RL, D57) are pure functions of its
snapshots, so the dataset generator, the ETA serving path, the RL environment and the RL
controller all see one definition.

It is passive: only TraCI getters and `sim.observe()`, which the session caches per step
(the controller's next `observe()` returns the same object). Two rules keep a run
unchanged:
- call `sample()` at most once per step: `station_new_vehicles` is the only state it
  keeps between calls (vehicles first seen on a station since the previous call);
- call it where the controller would observe anyway, i.e. before a step: right after a
  step that is not the last one, or before `controller.decide`. `observe()` updates the
  preemption manager with that step's emergency calls, and an extra call after the run's
  last step would be one update the run without the sampler never makes.

Pending (not yet inserted) vehicles are read with `simulation.getPendingVehicles`; their
type and first route edge are read once per vehicle id (`vehicle.getTypeID` and
`vehicle.getRoute` work on pending vehicles, SUMO 1.28, as for the emergency detector).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from traffic_twin.config import approach_out_edges
from traffic_twin.control.base import (
    EXTRA_BUS_CALLS,
    EXTRA_EMERGENCY,
    EXTRA_GAP_S,
    EXTRA_PED_COUNT,
    EXTRA_PED_SLACK,
    EXTRA_PED_WAIT,
    EXTRA_PREEMPT,
    BusCall,
    EmergencyCall,
)
from traffic_twin.sim.incidents import ActiveIncident
from traffic_twin.sim.junction import lane_edge
from traffic_twin.sim.kpi import user_type_of

if TYPE_CHECKING:
    from traffic_twin.sim.incidents import IncidentManager
    from traffic_twin.sim.session import SimSession

PENDING_OTHER = "other"  # pending vehicles whose first edge is no approach in-edge
USER_CLASSES = ("car", "bus", "ambulance")  # pending_by_class keys (kpi.user_type_of)
HALT_SPEED_MPS = 0.1  # SUMO's halting threshold (getLastStepHaltingNumber)


@dataclass(frozen=True)
class LaneStat:
    """One TLS-controlled vehicle lane during the last step."""

    lane: str
    arm: str  # network.approaches key of its edge
    length_m: float
    halting: int  # lane.getLastStepHaltingNumber
    vehicles: int  # vehicles on the lane
    occupancy: float  # lane.getLastStepOccupancy, 0..1
    max_wait_s: float  # max vehicle.getWaitingTime on the lane (0 if empty)


@dataclass(frozen=True)
class JunctionSnapshot:
    """Raw junction state after the previous step (what is known at time t)."""

    t: float  # simulation time since scenario start (s)
    clock_s: float  # absolute SUMO time (s, time of day)
    phase: int  # phase shown during the previous step (-1 before the first)
    stage: int  # Observation.stage: green stage, or the stage being left
    in_transition: bool
    time_in_stage: float
    clearance_left_s: float  # SignalPlan.clearance_left_s (0 while green)
    target_stage: int | None  # SignalPlan.target (stage being switched to)
    red_s: tuple[float, ...]  # per stage: s since its green last ended (0 while green)
    gap_s: tuple[float, ...]  # EXTRA_GAP_S per stage (inf if none)
    lanes: tuple[LaneStat, ...]  # TLS-controlled vehicle lanes, unique, TLS order
    arm_halting: Mapping[str, int]  # per approach (all approaches): halting on its in-edges
    arm_vehicles: Mapping[str, int]  # vehicles on its in-edges
    arm_wait_max_s: Mapping[str, float]  # max vehicle waiting time on its in-edges
    exit_halting: Mapping[str, int]  # per approach: halting on its out-edges
    pending: Mapping[str, int]  # pending insertions by approach of route[0] (PENDING_OTHER)
    pending_by_class: Mapping[str, int]  # car / bus / ambulance (kpi.user_type_of)
    station_occupancy: Mapping[str, float]  # measurement.stations id -> mean occupancy 0..1
    station_new_vehicles: Mapping[str, int]  # vehicles first seen since the previous sample
    queue_jam_m: Mapping[str, float]  # measurement.queues approach -> E2 jam length (m)
    ped_count: tuple[int, ...]  # per crossing (network.crossings order): EXTRA_PED_COUNT
    ped_wait_s: tuple[float, ...]  # EXTRA_PED_WAIT
    ped_slack_s: tuple[float, ...]  # EXTRA_PED_SLACK (inf without a shield)
    emergency: tuple[EmergencyCall, ...]  # EXTRA_EMERGENCY
    preempt_target: int | None  # EXTRA_PREEMPT
    buses: tuple[BusCall, ...]  # EXTRA_BUS_CALLS
    bus_time_loss_s: Mapping[str, float]  # bus vid -> vehicle.getTimeLoss
    incidents: tuple[ActiveIncident, ...]  # IncidentManager.active() (() without one)
    # per approach: halted buses on its out-edges (a dwelling bus at an in-lane stop, e.g.
    # stop_57_S on s_out, blocks the movements into it); {} in snapshots built by hand
    exit_bus_halting: Mapping[str, int] = field(default_factory=dict)


class JunctionSampler:
    """Build it after the session started (it reads the controlled links of the light).

    `incidents`: the run's `IncidentManager`, whose active incidents the snapshots carry.
    Without a `measurement` section (e.g. the RL environment) the station and queue
    mappings are empty."""

    def __init__(self, sim: SimSession, incidents: IncidentManager | None = None) -> None:
        if sim.plan is None:
            raise RuntimeError("JunctionSampler needs a started SimSession")
        self.sim, self.incidents = sim, incidents
        tr = sim.traci
        self._edge_arm = {e: a for a, edges in sim.approaches.items() for e in edges}
        self._out_edges = approach_out_edges(sim.cfg)
        lanes: list[str] = []
        for conns in tr.trafficlight.getControlledLinks(sim.tls):
            for from_lane, _to, _via in conns:
                # crossings start on walking areas (internal lanes ':...')
                if not from_lane.startswith(":") and from_lane not in lanes:
                    lanes.append(from_lane)
        self.lane_ids: tuple[str, ...] = tuple(lanes)
        self.lane_arms: tuple[str, ...] = tuple(
            self._edge_arm.get(lane_edge(lane), PENDING_OTHER) for lane in lanes
        )
        self.lane_lengths: tuple[float, ...] = tuple(float(tr.lane.getLength(x)) for x in lanes)
        layout = sim.layout
        self._stations: dict[str, tuple[str, ...]] = dict(layout.stations) if layout else {}
        self._queues: dict[str, tuple[str, ...]] = dict(layout.queues) if layout else {}
        self._seen: dict[str, set[str]] = {s: set() for s in self._stations}
        self._pending_info: dict[str, tuple[str, str]] = {}  # vid -> (approach, class)
        self._vclass: dict[str, str] = {}  # vid -> user class (kpi.user_type_of), cached
        self.n_stages = len(sim.plan.stages)

    def _pending(self, vid: str) -> tuple[str, str]:
        info = self._pending_info.get(vid)
        if info is None:
            veh = self.sim.traci.vehicle
            route = veh.getRoute(vid)
            arm = self._edge_arm.get(route[0], PENDING_OTHER) if route else PENDING_OTHER
            info = self._pending_info[vid] = (arm, user_type_of(veh.getTypeID(vid)))
        return info

    def _class(self, vid: str) -> str:
        cls = self._vclass.get(vid)
        if cls is None:
            cls = self._vclass[vid] = user_type_of(self.sim.traci.vehicle.getTypeID(vid))
        return cls

    def sample(self) -> JunctionSnapshot:
        sim = self.sim
        obs = sim.observe()
        plan, ex = sim.plan, obs.extras
        with sim._alive():
            return self._snapshot(obs, plan, ex)

    def _snapshot(self, obs, plan, ex) -> JunctionSnapshot:
        sim, tr = self.sim, self.sim.traci
        veh, lane_api, edge = tr.vehicle, tr.lane, tr.edge
        waits: dict[str, float] = {}

        def wait(vid: str) -> float:
            w = waits.get(vid)
            if w is None:
                w = waits[vid] = float(veh.getWaitingTime(vid))
            return w

        lanes = []
        for lane, arm, length in zip(self.lane_ids, self.lane_arms, self.lane_lengths, strict=True):
            ids = lane_api.getLastStepVehicleIDs(lane)
            lanes.append(
                LaneStat(
                    lane=lane,
                    arm=arm,
                    length_m=length,
                    halting=int(lane_api.getLastStepHaltingNumber(lane)),
                    vehicles=len(ids),
                    occupancy=float(lane_api.getLastStepOccupancy(lane)),
                    max_wait_s=max((wait(v) for v in ids), default=0.0),
                )
            )
        arm_vehicles, arm_wait = {}, {}
        for arm, edges in sim.approaches.items():
            ids = [v for e in edges for v in edge.getLastStepVehicleIDs(e)]
            arm_vehicles[arm] = len(ids)
            arm_wait[arm] = max((wait(v) for v in ids), default=0.0)
        exit_halting = {
            arm: sum(int(edge.getLastStepHaltingNumber(e)) for e in edges)
            for arm, edges in self._out_edges.items()
        }
        exit_bus_halting = {
            arm: sum(
                1
                for e in edges
                for v in edge.getLastStepVehicleIDs(e)
                if self._class(v) == "bus" and float(veh.getSpeed(v)) < HALT_SPEED_MPS
            )
            for arm, edges in self._out_edges.items()
        }
        pending = dict.fromkeys(sim.approaches, 0) | {PENDING_OTHER: 0}
        by_class = dict.fromkeys(USER_CLASSES, 0)
        for vid in tr.simulation.getPendingVehicles():
            arm, cls = self._pending(vid)
            pending[arm] = pending.get(arm, 0) + 1
            by_class[cls] = by_class.get(cls, 0) + 1
        il = tr.inductionloop
        occupancy, new_vehicles = {}, {}
        for station, loops in self._stations.items():
            # inductionloop occupancy is in percent
            occupancy[station] = (
                sum(float(il.getLastStepOccupancy(d)) for d in loops) / len(loops) / 100.0
                if loops
                else 0.0
            )
            seen = {v for d in loops for v in il.getLastStepVehicleIDs(d)}
            new_vehicles[station] = len(seen - self._seen[station])
            self._seen[station] = seen
        jam = {
            a: sum(float(tr.lanearea.getJamLengthMeters(d)) for d in dets)
            for a, dets in self._queues.items()
        }
        buses = tuple(ex.get(EXTRA_BUS_CALLS, ()))
        gap = ex.get(EXTRA_GAP_S)
        return JunctionSnapshot(
            t=float(obs.t),
            clock_s=float(sim.time),
            phase=sim.phases_shown[-1] if sim.phases_shown else -1,
            stage=int(obs.stage),
            in_transition=bool(obs.in_transition),
            time_in_stage=float(obs.time_in_stage),
            clearance_left_s=plan.clearance_left_s,
            target_stage=plan.target,
            red_s=tuple(float(r) for r in plan.red_s),
            gap_s=tuple(float(g) for g in gap) if gap else (math.inf,) * self.n_stages,
            lanes=tuple(lanes),
            arm_halting={a: int(q) for a, q in obs.queues.items()},
            arm_vehicles=arm_vehicles,
            arm_wait_max_s=arm_wait,
            exit_halting=exit_halting,
            pending=pending,
            pending_by_class=by_class,
            station_occupancy=occupancy,
            station_new_vehicles=new_vehicles,
            queue_jam_m=jam,
            ped_count=tuple(int(c) for c in ex.get(EXTRA_PED_COUNT, ())),
            ped_wait_s=tuple(float(w) for w in ex.get(EXTRA_PED_WAIT, ())),
            ped_slack_s=tuple(float(s) for s in ex.get(EXTRA_PED_SLACK, ())),
            emergency=tuple(ex.get(EXTRA_EMERGENCY, ())),
            preempt_target=ex.get(EXTRA_PREEMPT),
            buses=buses,
            bus_time_loss_s={b.vid: float(veh.getTimeLoss(b.vid)) for b in buses},
            incidents=self.incidents.active() if self.incidents is not None else (),
            exit_bus_halting=exit_bus_halting,
        )

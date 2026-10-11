"""A running SUMO simulation that a controller loop (or a Gymnasium env) drives one
step at a time: `observe()` the intersection, `step(stage)` to apply a green stage
request and advance the simulation.

    with SimSession(cfg, seed, out_dir) as sim:
        while not sim.done:
            sim.step(controller.decide(sim.observe()))

Every request goes through the SafetyGuard before the SignalPlan, so no controller can
bypass it.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from traffic_twin.config import approach_in_edges, get
from traffic_twin.control.base import EXTRA_CALLS, EXTRA_GAP_S, Observation
from traffic_twin.control.safety import SafetyGuard
from traffic_twin.paths import DEMAND_DIR
from traffic_twin.sim.demand import write_demand
from traffic_twin.sim.detectors import (
    DETECTOR_FILE,
    E1_FILE,
    TLS_SWITCH_FILE,
    DetectorLayout,
    write_detectors,
)
from traffic_twin.sim.kpi import user_type_of
from traffic_twin.sim.signal import SignalPlan
from traffic_twin.sim.sumo import DRAIN_S, sumo_command, traci_module

STATISTICS_FILE = "statistics.xml"


class SimulationClosed(RuntimeError):
    """SUMO went away before the end of the run (e.g. the GUI window was closed)."""


class SimSession:
    """One SUMO run. Use as a context manager, or call `start()` and `close()`.

    SUMO writes its raw outputs (`tripinfo`, `stops`, `statistics.xml` and, when the
    scenario configures detectors, `detectors.add.xml`, `e1.xml`, `tls_switch.xml`) into
    `out_dir`. Collected while stepping:
    - `state_rows`: the state log, every `outputs.state_log_every_s` of the demand window;
    - `queue_rows`: E2 queue samples per approach, on the same ticks (measurement only);
    - `phases_shown`: the phase shown at each step (signal audit);
    - `ped_wait_max`: per pedestrian, the longest single wait (`person.getWaitingTime`).
    """

    def __init__(
        self,
        cfg: dict,
        seed: int,
        out_dir: Path,
        gui: bool = False,
        log_state: bool = True,
        demand_dir: Path = DEMAND_DIR,
        traci: ModuleType | None = None,
    ):
        self.cfg, self.seed, self.out_dir, self.gui = cfg, seed, Path(out_dir), gui
        self.demand_dir = Path(demand_dir)
        self.traci = traci or traci_module(gui)
        self.tls: str = cfg["network"]["tls_id"]
        self.approaches: dict[str, list[str]] = approach_in_edges(cfg)
        self.begin = float(cfg["begin_s"])
        self.end = self.begin + int(cfg["duration_s"])  # end of the demand window
        self.dt = float(cfg.get("step_length_s", 1.0))
        self.sample_every = int(cfg["outputs"].get("state_log_every_s", 5))
        self.log_every = self.sample_every if log_state else 0
        self.tripinfo = self.out_dir / "tripinfo.xml"
        self.stops = self.out_dir / "stops.xml"
        self.statistics = self.out_dir / STATISTICS_FILE
        self.routes: Path | None = None
        self.detector_file: Path | None = None
        self.layout: DetectorLayout | None = None
        self.plan: SignalPlan | None = None
        self.phase_states: list[str] = []
        self.guard: SafetyGuard | None = None
        self.state_rows: list[dict] = []
        self.queue_rows: list[dict] = []
        self.phases_shown: list[int] = []
        self.ped_wait_max: dict[str, float] = {}
        self.drain_timeout = False
        self.time = self.begin  # absolute SUMO time (s)
        self._obs: Observation | None = None
        self._calls: list[bool] = []  # per stage: pending call (EXTRA_CALLS)
        self._loop_ids: tuple[str, ...] = ()  # control loops
        self._connected = False

    @property
    def raw_outputs(self) -> list[Path]:
        """Files SUMO (or `start()`) writes into `out_dir`."""
        names = (STATISTICS_FILE, DETECTOR_FILE, E1_FILE, TLS_SWITCH_FILE)
        return [self.tripinfo, self.stops, *(self.out_dir / n for n in names)]

    # -- lifecycle -------------------------------------------------------------------
    def start(self) -> SimSession:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.routes, add = write_demand(self.cfg, self.demand_dir)
        self.detector_file, self.layout = write_detectors(self.cfg, self.out_dir)
        additional = [add] if self.detector_file is None else [add, self.detector_file]
        cmd = sumo_command(
            self.cfg, self.seed, self.routes, additional, self.tripinfo, self.stops, self.gui,
            statistics=self.statistics,
        )  # fmt: skip
        self._calls = [False] * len(self.layout.stage_loops)
        self._loop_ids = tuple(lp.id for lp in self.layout.control_loops)
        self._connected = True
        with self._alive():
            self.traci.start(cmd)
            logic = self.traci.trafficlight.getAllProgramLogics(self.tls)[0]
            durations = [p.duration for p in logic.phases]
            self.phase_states = [p.state for p in logic.phases]
            self.plan = SignalPlan.from_config(
                self.cfg["network"]["stages"],
                durations,
                min_green_s=float(get(self.cfg, "safety.min_green_s", 0.0)),
            )
            self.guard = SafetyGuard.from_config(self.cfg, self.plan.stages, self.phase_states)
            self.traci.trafficlight.setPhase(self.tls, self.plan.phase)
        return self

    def close(self) -> None:
        if self._connected:
            self._connected = False
            try:
                self.traci.close()
            except (self.traci.FatalTraCIError, ConnectionError):  # already gone (GUI closed)
                pass

    def __enter__(self) -> SimSession:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- stepping --------------------------------------------------------------------
    @property
    def t(self) -> float:
        """Simulation time since the scenario start (s)."""
        return self.time - self.begin

    @property
    def done(self) -> bool:
        """True once the demand window is over and the network is empty (or drain timed out)."""
        if self.time < self.end:
            return False
        with self._alive():
            pending = self.traci.simulation.getMinExpectedNumber() > 0
        if pending and self.time >= self.end + DRAIN_S:
            self.drain_timeout = True
        return not (pending and self.time < self.end + DRAIN_S)

    def observe(self) -> Observation:
        """Observation at the current step (computed once per step)."""
        if self._obs is None:
            plan = self.plan
            with self._alive():
                halting = self.traci.edge.getLastStepHaltingNumber
                queues = {a: sum(halting(e) for e in es) for a, es in self.approaches.items()}
                extras = self._loop_extras() if self._loop_ids else {}
            # After the demand window the controller keeps running so the network empties
            self._obs = Observation(
                self.t, plan.stage, plan.time_in_stage, plan.in_transition, queues, extras
            )
        return self._obs

    def step(self, stage: int) -> int:
        """Request a green stage (through the safety guard), apply the resulting phase,
        advance one step. Returns the phase shown during the step."""
        obs = self.observe()
        stage = self.guard.apply(stage, self.plan.state(obs.t))
        phase = self.plan.request(stage)
        with self._alive():
            self.traci.trafficlight.setPhase(self.tls, phase)
            self.traci.trafficlight.setPhaseDuration(self.tls, 1e6)  # we drive switching
            if self._sample_tick(obs.t):
                if self.log_every:
                    self._log_state(obs, phase)
                if self.layout.queues:
                    self._log_queues(obs.t)
            self.traci.simulationStep()
            self.plan.advance(self.dt)
            self.time = self.traci.simulation.getTime()
            self._sample_ped_waits()
        self.phases_shown.append(phase)
        self._obs = None
        return phase

    # -- internals -------------------------------------------------------------------
    @contextmanager
    def _alive(self) -> Iterator[None]:
        """Turn a lost connection to SUMO into SimulationClosed."""
        try:
            yield
        except (self.traci.FatalTraCIError, ConnectionError) as e:
            self.close()
            raise SimulationClosed(f"SUMO stopped before the end, window closed? ({e})") from e

    def _loop_extras(self) -> dict:
        """EXTRA_GAP_S and EXTRA_CALLS per stage from the control loops."""
        il, plan = self.traci.inductionloop, self.plan
        gap = {d: il.getTimeSinceDetection(d) for d in self._loop_ids}
        hit = {d: il.getLastStepVehicleNumber(d) > 0 for d in self._loop_ids}
        green = None if plan.in_transition else plan.stage
        for k, dets in enumerate(self.layout.stage_loops):
            if k == green:  # a call is pending only while the stage is not green
                self._calls[k] = False
            elif any(hit[d] for d in dets):
                self._calls[k] = True
        return {
            EXTRA_GAP_S: tuple(min((gap[d] for d in dets), default=math.inf)
                               for dets in self.layout.stage_loops),
            EXTRA_CALLS: tuple(self._calls),
        }  # fmt: skip

    def _sample_tick(self, t: float) -> bool:
        """State log and queue samples: every `sample_every` s of the demand window."""
        return bool(self.sample_every) and int(t) % self.sample_every == 0 and self.time < self.end

    def _sample_ped_waits(self) -> None:
        tr = self.traci.person
        for pid in tr.getIDList():
            wait = tr.getWaitingTime(pid)
            if wait >= self.ped_wait_max.get(pid, 0.0):
                self.ped_wait_max[pid] = wait

    def _log_queues(self, t: float) -> None:
        la = self.traci.lanearea
        for approach, dets in self.layout.queues.items():
            self.queue_rows.append(
                {"t": t, "approach": approach,
                 "jam_veh": sum(la.getJamLengthVehicle(d) for d in dets),
                 "jam_m": sum(la.getJamLengthMeters(d) for d in dets),
                 "halting_veh": sum(la.getLastStepHaltingNumber(d) for d in dets)}
            )  # fmt: skip

    def _log_state(self, obs: Observation, phase: int) -> None:
        tr, t = self.traci, obs.t
        self.state_rows.append(
            {"t": t, "entity": "intersection", "id": self.tls, "x": None, "y": None,
             "speed": None, "phase": phase, "queue": sum(obs.queues.values())}
        )  # fmt: skip
        for vid in tr.vehicle.getIDList():
            x, y = tr.vehicle.getPosition(vid)
            self.state_rows.append(
                {"t": t, "entity": user_type_of(tr.vehicle.getTypeID(vid)), "id": vid,
                 "x": x, "y": y, "speed": tr.vehicle.getSpeed(vid), "phase": None,
                 "queue": None}
            )  # fmt: skip
        for pid in tr.person.getIDList():
            x, y = tr.person.getPosition(pid)
            self.state_rows.append(
                {"t": t, "entity": "pedestrian", "id": pid, "x": x, "y": y,
                 "speed": tr.person.getSpeed(pid), "phase": None, "queue": None}
            )  # fmt: skip

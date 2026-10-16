"""A running SUMO simulation that a controller loop (or a Gymnasium env) drives one
step at a time: `observe()` the intersection, `step(stage)` to apply a green stage
request and advance the simulation.

    with SimSession(cfg, seed, out_dir) as sim:
        while not sim.done:
            sim.step(controller.decide(sim.observe()))

Every request goes through the SafetyGuard before the SignalPlan, so no controller can
bypass it.

On a junction with `network.crossings` (Pitié, Part 4) the observation also carries the
pedestrian waits per crossing, the shield slack, emergency and bus calls, the active
preemption target and the blocked exits (the `EXTRA_*` keys of `control.base`), and a
`JunctionMonitor` samples the box, the exits and the crossings after every step. On the
grid the observation and the guard input are exactly as in Part 3.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from traffic_twin.config import approach_in_edges, get
from traffic_twin.control.base import (
    EXTRA_BUS_CALLS,
    EXTRA_CALLS,
    EXTRA_EMERGENCY,
    EXTRA_EXIT_BLOCKED,
    EXTRA_GAP_S,
    EXTRA_GAP_S_OPEN,
    EXTRA_PED_COUNT,
    EXTRA_PED_SLACK,
    EXTRA_PED_WAIT,
    EXTRA_PREEMPT,
    Observation,
)
from traffic_twin.control.pedestrian import GREEN as SHIELD_GREEN
from traffic_twin.control.preemption import PreemptionManager
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
from traffic_twin.sim.emergency import EmergencyDetector
from traffic_twin.sim.junction import JunctionMonitor, lane_exits, lane_links
from traffic_twin.sim.kpi import user_type_of
from traffic_twin.sim.signal import SignalPlan
from traffic_twin.sim.sumo import DRAIN_S, sumo_command, traci_module
from traffic_twin.sim.tls_check import load_program
from traffic_twin.sim.transit import BusDetector

STATISTICS_FILE = "statistics.xml"


@functools.lru_cache(maxsize=8)
def program_foes(net_file: str, tls_id: str) -> dict[int, frozenset[int]]:
    """Link -> conflicting links of the TLS program in a .net.xml (cached per net)."""
    return load_program(net_file, tls_id).foes


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
    - `ped_wait_max`: per pedestrian, the longest single wait (`person.getWaitingTime`,
      the Part 3 KPI input; SUMO resets it when a waiting pedestrian creeps);
    - `ped_curb_wait_max` (scenarios with `network.crossings`): per pedestrian, the
      longest curb wait, see `_sample_ped_waits`. It is also the shield's detected w_c.

    With `network.crossings` (Part 4), `start()` also builds `emergency` and `buses` (the
    detectors), the `preemption` manager (when `preemption.enabled`), `junction` (the
    monitor, unless `junction_monitor=False`, which only the passivity test uses) and
    `foes` (the program's link conflicts). They stay None on the grid.
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
        junction_monitor: bool = True,
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
        self.ped_curb_wait_max: dict[str, float] = {}
        self.drain_timeout = False
        self.time = self.begin  # absolute SUMO time (s)
        # (requested stage, stage after the safety guard) of the last step(); None before
        self.last_request: tuple[int, int] | None = None
        # Part 4, built in start() only when the scenario has network.crossings
        self.crossings: dict = get(cfg, "network.crossings", None) or {}
        self.junction_monitor = junction_monitor
        self.emergency: EmergencyDetector | None = None
        self.buses: BusDetector | None = None
        self.preemption: PreemptionManager | None = None
        self.junction: JunctionMonitor | None = None
        self.foes: dict[int, frozenset[int]] | None = None
        self._obs: Observation | None = None
        self._calls: list[bool] = []  # per stage: pending call (EXTRA_CALLS)
        self._loop_ids: tuple[str, ...] = ()  # control loops
        self._loop_exits: dict[str, frozenset[str]] | None = None  # loop -> its lane's exits
        self._cross_index = {c["edge"]: k for k, c in enumerate(self.crossings.values())}
        self._cross_links = tuple(int(c["link"]) for c in self.crossings.values())
        n = len(self._cross_links)
        self._ped_waits: tuple[float, ...] = (0.0,) * n  # detected w_c, after the last step
        self._ped_counts: tuple[int, ...] = (0,) * n
        self._red_waits: tuple[float, ...] = (0.0,) * n  # s since the crossing showed G
        self._curb: dict[str, tuple[int, float]] = {}  # person -> (crossing, curb wait start)
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
            self.guard = SafetyGuard.from_config(
                self.cfg, self.plan.stages, self.phase_states, durations=durations
            )
            if self.crossings:
                self._start_junction()
            self.traci.trafficlight.setPhase(self.tls, self.plan.phase)
        return self

    def _start_junction(self) -> None:
        """Part 4 detectors, preemption manager, junction monitor and box-clearing loop
        exits (scenarios with `network.crossings`)."""
        cfg, stages = self.cfg, self.plan.stages
        detection = get(cfg, "preemption.detection", None) or {}
        self.emergency = EmergencyDetector(
            self.traci, self.tls, stages, self.phase_states, self.approaches,
            pending=bool(detection.get("pending", True)),
            max_dist_m=detection.get("max_dist_m"),
        )  # fmt: skip
        self.buses = BusDetector(self.traci, self.tls, stages, self.phase_states, self.approaches)
        self.preemption = PreemptionManager.from_config(cfg)
        self.foes = program_foes(str(cfg["network"]["net_file"]), self.tls)
        if self.junction_monitor:
            self.junction = JunctionMonitor(
                self.traci, self.tls, self.phase_states, self.crossings, self.foes
            )
        exits = lane_exits(lane_links(self.traci, self.tls))
        self._loop_exits = {
            lp.id: exits.get(lp.lane, frozenset()) for lp in self.layout.control_loops
        }

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

    @property
    def shield(self):
        """The guard's pedestrian shield (`control.pedestrian.PedestrianShield`), if any."""
        return getattr(self.guard, "shield", None)

    @property
    def _guard_waits(self) -> tuple[float, ...]:
        """The shield's w_c per crossing: detected waits, red waits, or () without one."""
        mode = getattr(self.shield, "mode", "off")
        if mode == "detected":
            return self._ped_waits
        if mode == "red":
            return self._red_waits
        return ()

    def observe(self) -> Observation:
        """Observation at the current step (computed once per step)."""
        if self._obs is None:
            plan = self.plan
            with self._alive():
                halting = self.traci.edge.getLastStepHaltingNumber
                queues = {a: sum(halting(e) for e in es) for a, es in self.approaches.items()}
                extras = self._loop_extras() if self._loop_ids else {}
                if self.crossings:
                    extras.update(self._junction_extras(self.t))
            # After the demand window the controller keeps running so the network empties
            self._obs = Observation(
                self.t, plan.stage, plan.time_in_stage, plan.in_transition, queues, extras
            )
        return self._obs

    def step(self, stage: int) -> int:
        """Request a green stage (through the safety guard), apply the resulting phase,
        advance one step. Returns the phase shown during the step.

        The guard sees the plan state plus the shield's pedestrian waits and the active
        preemption target (both empty on the grid)."""
        obs = self.observe()
        state = dataclasses.replace(
            self.plan.state(obs.t),
            ped_wait=self._guard_waits,
            preempt_target=obs.extras.get(EXTRA_PREEMPT),
        )
        requested = stage
        stage = self.guard.apply(stage, state)
        self.last_request = (int(requested), int(stage))
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
            self._sample_ped_waits(phase)
            if self.crossings:
                self._update_red_waits(phase)
            if self.junction is not None:
                self.junction.after_step(phase, self.dt)
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
        extras = {
            EXTRA_GAP_S: tuple(min((gap[d] for d in dets), default=math.inf)
                               for dets in self.layout.stage_loops),
            EXTRA_CALLS: tuple(self._calls),
        }  # fmt: skip
        if self._loop_exits is not None:  # box clearing: loops feeding an open exit only
            blocked = self.junction.blocked_exits() if self.junction else frozenset()
            extras[EXTRA_GAP_S_OPEN] = tuple(
                min((gap[d] for d in dets if self._loop_exits[d] - blocked), default=math.inf)
                for dets in self.layout.stage_loops
            )
        return extras

    def _junction_extras(self, t: float) -> dict:
        """Part 4 extras (scenarios with `network.crossings`). Pedestrian waits and
        blocked exits are those sampled after the previous step; the preemption manager
        is updated with this step's emergency calls (once per step: `observe` caches)."""
        emergency = tuple(self.emergency.calls(t))
        target = self.preemption.update(t, emergency) if self.preemption is not None else None
        shield = self.shield
        if shield is not None:
            state = dataclasses.replace(
                self.plan.state(t), ped_wait=self._guard_waits, preempt_target=target
            )
            slack = tuple(shield.slack(state))
        else:
            slack = (math.inf,) * len(self._cross_links)
        return {
            EXTRA_PED_WAIT: self._ped_waits,
            EXTRA_PED_COUNT: self._ped_counts,
            EXTRA_PED_SLACK: slack,
            EXTRA_EMERGENCY: emergency,
            EXTRA_PREEMPT: target,
            EXTRA_BUS_CALLS: tuple(self.buses.calls(t)),
            EXTRA_EXIT_BLOCKED: self.junction.blocked_exits() if self.junction else frozenset(),
        }

    def _sample_tick(self, t: float) -> bool:
        """State log and queue samples: every `sample_every` s of the demand window."""
        return bool(self.sample_every) and int(t) % self.sample_every == 0 and self.time < self.end

    def _sample_ped_waits(self, phase: int) -> None:
        """Per pedestrian the longest single wait (`ped_wait_max`, `person.getWaitingTime`)
        and, with crossings, the longest curb wait (`ped_curb_wait_max`); per crossing the
        longest current curb wait and the count of the persons waiting there: the
        shield's detected w_c (ideal detection, ASSUMED). `phase` is the phase shown
        during the step just made.

        Curb wait: SUMO resets `getWaitingTime` to 0 as soon as a waiting pedestrian moves
        faster than 0.1 m/s, and pedestrians stopped at the curb often shuffle a few
        centimetres while the crossing stays red, which hid up to 11 s of a wait (an S2
        pedestrian stood 91 s at a red crossing while `getWaitingTime` peaked at 88). So
        a curb clock starts at `now - getWaitingTime` the first time a person whose next
        edge is crossing c waits while c is not green, and keeps running, creeps
        included, until the person's next edge changes (it entered the crossing or left)
        or c shows green (the person is served; later waiting is `getWaitingTime` again).
        The curb wait is max(getWaitingTime, now - clock start), never below the Part 3
        value."""
        tr = self.traci.person
        index = self._cross_index
        waits, counts = [0.0] * len(index), [0] * len(index)
        if index:
            state = self.phase_states[phase]
            green = [state[link] in SHIELD_GREEN for link in self._cross_links]
        curb: dict[str, tuple[int, float]] = {}
        for pid in tr.getIDList():
            wait = tr.getWaitingTime(pid)
            if wait >= self.ped_wait_max.get(pid, 0.0):
                self.ped_wait_max[pid] = wait
            if not index:
                continue
            c = index.get(tr.getNextEdge(pid))
            if c is not None and not green[c]:
                prev = self._curb.get(pid)
                if prev is not None and prev[0] == c:
                    curb[pid] = prev
                elif wait > 0:
                    curb[pid] = (c, self.time - wait)
                if pid in curb:
                    wait = max(wait, self.time - curb[pid][1])
            if wait >= self.ped_curb_wait_max.get(pid, 0.0):
                self.ped_curb_wait_max[pid] = wait
            if c is not None and wait > 0:
                counts[c] += 1
                waits[c] = max(waits[c], wait)
        if index:
            self._curb = curb
            self._ped_waits, self._ped_counts = tuple(waits), tuple(counts)

    def _update_red_waits(self, phase: int) -> None:
        """Red-mode w_c: seconds since each crossing's link last showed G."""
        state = self.phase_states[phase]
        self._red_waits = tuple(
            0.0 if state[link] == "G" else r + self.dt
            for link, r in zip(self._cross_links, self._red_waits, strict=True)
        )

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

"""A running SUMO simulation that a controller loop (or a Gymnasium env) drives one
step at a time: `observe()` the intersection, `step(stage)` to apply a green stage
request and advance the simulation.

    with SimSession(cfg, seed, out_dir) as sim:
        while not sim.done:
            sim.step(controller.decide(sim.observe()))
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from traffic_twin.config import get
from traffic_twin.control.base import Observation
from traffic_twin.paths import DEMAND_DIR
from traffic_twin.sim.demand import write_demand
from traffic_twin.sim.kpi import user_type_of
from traffic_twin.sim.signal import SignalPlan
from traffic_twin.sim.sumo import DRAIN_S, sumo_command, traci_module


class SimulationClosed(RuntimeError):
    """SUMO went away before the end of the run (e.g. the GUI window was closed)."""


class SimSession:
    """One SUMO run. Use as a context manager, or call `start()` and `close()`.

    SUMO writes its raw outputs (`tripinfo`, `stops`) into `out_dir`; the state log
    (every `outputs.state_log_every_s` during the demand window) collects in `state_rows`.
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
        net = cfg["network"]
        self.tls: str = net["tls_id"]
        self.approaches: dict[str, str] = net["approaches"]
        self.begin = float(cfg["begin_s"])
        self.end = self.begin + int(cfg["duration_s"])  # end of the demand window
        self.dt = float(cfg.get("step_length_s", 1.0))
        self.log_every = int(cfg["outputs"].get("state_log_every_s", 5)) if log_state else 0
        self.tripinfo = self.out_dir / "tripinfo.xml"
        self.stops = self.out_dir / "stops.xml"
        self.state_rows: list[dict] = []
        self.plan: SignalPlan | None = None
        self.time = self.begin  # absolute SUMO time (s)
        self._obs: Observation | None = None
        self._connected = False

    # -- lifecycle -------------------------------------------------------------------
    def start(self) -> SimSession:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        rou, add = write_demand(self.cfg, self.demand_dir)
        cmd = sumo_command(self.cfg, self.seed, rou, add, self.tripinfo, self.stops, self.gui)
        self._connected = True
        with self._guard():
            self.traci.start(cmd)
            logic = self.traci.trafficlight.getAllProgramLogics(self.tls)[0]
            durations = [p.duration for p in logic.phases]
            self.plan = SignalPlan.from_config(
                self.cfg["network"]["stages"],
                durations,
                min_green_s=float(get(self.cfg, "safety.min_green_s", 0.0)),
            )
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
        with self._guard():
            pending = self.traci.simulation.getMinExpectedNumber() > 0
        return not (pending and self.time < self.end + DRAIN_S)

    def observe(self) -> Observation:
        """Observation at the current step (computed once per step)."""
        if self._obs is None:
            plan = self.plan
            with self._guard():
                queues = {
                    a: self.traci.edge.getLastStepHaltingNumber(e)
                    for a, e in self.approaches.items()
                }
            # After the demand window the controller keeps running so the network empties
            self._obs = Observation(
                self.t, plan.stage, plan.time_in_stage, plan.in_transition, queues
            )
        return self._obs

    def step(self, stage: int) -> int:
        """Request a green stage, apply the resulting phase, advance one step. Returns the
        phase shown during the step."""
        obs = self.observe()
        phase = self.plan.request(stage)
        with self._guard():
            self.traci.trafficlight.setPhase(self.tls, phase)
            self.traci.trafficlight.setPhaseDuration(self.tls, 1e6)  # we drive switching
            if self.log_every and int(obs.t) % self.log_every == 0 and self.time < self.end:
                self._log_state(obs, phase)
            self.traci.simulationStep()
            self.plan.advance(self.dt)
            self.time = self.traci.simulation.getTime()
        self._obs = None
        return phase

    # -- internals -------------------------------------------------------------------
    @contextmanager
    def _guard(self) -> Iterator[None]:
        """Turn a lost connection to SUMO into SimulationClosed."""
        try:
            yield
        except (self.traci.FatalTraCIError, ConnectionError) as e:
            self.close()
            raise SimulationClosed(f"SUMO stopped before the end, window closed? ({e})") from e

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

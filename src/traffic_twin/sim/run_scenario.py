"""Run one scenario headless with a given controller and seed, then export trips,
KPIs and a state log (spec 8.2, 9.1), plus the detector, queue and signal tables when
the scenario has a `measurement` section and the events table when a controller, the
safety guard or the preemption logged an event (Part 4).

    python -m traffic_twin.sim.run_scenario --config configs/scenario_base.yaml --seed 42
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from traffic_twin.config import ConfigError, get, load_scenario
from traffic_twin.control.base import Controller
from traffic_twin.control.registry import make_controller
from traffic_twin.control.safety import INTERVENTION_KEYS
from traffic_twin.paths import DEMAND_DIR, SCENARIO_CONFIG
from traffic_twin.sim import tls_check
from traffic_twin.sim.audit import DANGER_KEYS, audit_phases
from traffic_twin.sim.detectors import E1_FILE, TLS_SWITCH_FILE, read_e1, read_tls_switch
from traffic_twin.sim.kpi import read_statistics
from traffic_twin.sim.results import (
    RunResults,
    collect_results,
    config_files,
    detector_table,
    event_table,
    preemption_block,
    queue_table,
    run_health,
    run_metadata,
    signal_table,
    write_results,
)
from traffic_twin.sim.session import SimSession, SimulationClosed

log = logging.getLogger(__name__)


class RunObserver(Protocol):
    """A passive observer of a run (e.g. the ETA feature recorder): it may read the
    session (TraCI getters, `observe()`), never change it.

    - `bind(sim)`: after the session started and the controller was bound and reset,
      before the first step;
    - `after_step(sim)`: after every `sim.step`;
    - `finish(sim, out_dir, run_id)`: after the run's results were written (not called
      when SUMO stops early)."""

    def bind(self, sim: SimSession) -> None: ...

    def after_step(self, sim: SimSession) -> None: ...

    def finish(self, sim: SimSession, out_dir: Path, run_id: str) -> None: ...


def run_dir(cfg: dict, seed: int, out_root: Path) -> Path:
    return Path(out_root) / f"{cfg['scenario']}_seed{seed}"


def safety_block(sim: SimSession) -> dict:
    """Guard interventions (one per event), the audit of the phases shown, the number of
    dangerous audit findings (`danger`, sum of DANGER_KEYS) and the pedestrian shield
    mode (`ped_forcing`, "off" without a shield).

    With `network.crossings` (Pitié) the audit also gets the programmed durations, the
    guard's link-safe transitions, the intergreen requirements and the foe links (from
    `tls_check`); the grid audit keeps its Part 3 inputs."""
    cfg = sim.cfg
    crossings = get(cfg, "network.crossings", None) or {}
    extended = {}
    if crossings:
        req = tls_check.cached_requirements(get(cfg, "network.net_file"), cfg)
        extended = {
            "durations": sim.plan.durations,
            "allowed": sim.guard.allowed,
            "intergreen": req.intergreen,
            "foes": req.foes,
        }
    audit = audit_phases(
        sim.phases_shown,
        sim.phase_states,
        sim.plan.stages,
        float(get(cfg, "safety.min_green_s", 0.0)),
        step_s=sim.dt,
        ped_links=[int(c["link"]) for c in crossings.values()] if crossings else None,
        **extended,
    )
    audit = {k: int(v) for k, v in audit.items()}
    interventions = {k: int(sim.guard.interventions.get(k, 0)) for k in INTERVENTION_KEYS}
    return {
        "interventions": interventions,
        "audit": audit,
        "danger": sum(audit.get(k, 0) for k in DANGER_KEYS),
        "ped_forcing": getattr(sim.shield, "mode", "off"),
    }


def preemption_summary(sim: SimSession) -> dict:
    """`run.json` preemption block of a finished run (`results.preemption_block`)."""
    report = sim.preemption.report() if sim.preemption is not None else None
    return preemption_block(report, getattr(sim.guard, "preemption", None))


def run_events(controller: Controller, sim: SimSession) -> list:
    """Events of the run: the controller's, then the guard's, then the preemption's."""
    events = list(controller.events)
    events += getattr(sim.guard, "events", None) or []
    if sim.preemption is not None:
        events += sim.preemption.events
    return events


def add_measurements(results: RunResults, sim: SimSession, run_id: str) -> None:
    """Detector, queue and signal tables of a scenario with a `measurement` section."""
    meas, layout = sim.cfg.get("measurement") or {}, sim.layout
    if meas.get("stations"):
        e1 = read_e1(sim.out_dir / E1_FILE)
        results.detectors = detector_table(run_id, e1, layout, sim.begin)
    if meas.get("queues"):
        results.queues = queue_table(run_id, sim.queue_rows)
    if layout.switch_links:
        switches = read_tls_switch(sim.out_dir / TLS_SWITCH_FILE)
        results.signal = signal_table(run_id, switches, layout.switch_links, sim.begin)


def health_line(health: dict, preemption: dict | None = None) -> str:
    """One-line summary of the health block for the log, with the preemption requests
    served and timed out when preemption is enabled."""

    def num(key: str) -> str:
        v = health.get(key)
        return "n/a" if v is None else f"{v:g}"

    return (
        f"teleports {num('teleports')}, collisions {num('collisions')}, "
        f"emergency stops {num('emergency_stops')}, braking {num('emergency_braking')}, "
        f"waiting at end {num('vehicles_waiting_end')}, depart delay mean "
        f"{num('depart_delay_mean')} s / max {num('depart_delay_max')} s"
        + (", DRAIN TIMEOUT" if health.get("drain_timeout") else "")
        + (
            f", preemption served {preemption['served']}/{preemption['requests']}, "
            f"timeouts {preemption['timeouts']}"
            if preemption and preemption.get("enabled")
            else ""
        )
    )


def _file_stamps(paths: Sequence[Path]) -> dict[Path, int]:
    """Path -> st_mtime_ns of the files that exist."""
    out = {}
    for p in paths:
        try:
            out[p] = p.stat().st_mtime_ns
        except OSError:
            pass
    return out


def remove_partial_outputs(
    sim: SimSession, before: dict[Path, int], out_dir: Path, dir_existed: bool
) -> None:
    """After SUMO stopped early: remove the raw outputs this run created or rewrote (a
    file absent before the run, or whose mtime changed since), never a file the run left
    untouched, and the run directory only when this run created it and it is empty. A
    run into an existing results directory thus keeps the files of the earlier run that
    SUMO did not reopen (the plausible way the raw XMLs of `pitie_rules_seed1` were lost
    before Part 5). An mtime comparison rather than "mtime >= run start" because Windows
    file times can trail the wall clock by a timer tick."""
    for p in sim.raw_outputs:
        try:
            stamp = p.stat().st_mtime_ns
        except OSError:
            continue
        if before.get(p) != stamp:
            p.unlink(missing_ok=True)
    if not dir_existed and out_dir.is_dir() and not any(out_dir.iterdir()):
        out_dir.rmdir()


def run(
    cfg: dict,
    seed: int,
    out_root: Path,
    gui: bool = False,
    config_path: Path | None = None,
    demand_dir: Path = DEMAND_DIR,
    observers: Sequence[RunObserver] = (),
) -> dict:
    """Run the scenario's controller for one seed and write its results. Raises
    SimulationClosed (after removing the partial outputs of this run) if SUMO stops
    early. `config_sources` in run.json lists the files of `config_path`, or those `cfg`
    was loaded from by `load_scenario`; it is None for a config built or copied in memory.

    `demand_dir` receives the route and stop files (default `sumo/demand`; scratch runs
    pass a temporary folder). `observers` are passive `RunObserver`s. The loop is:
    `controller.bind(sim)`, `controller.reset(seed)`, `observer.bind(sim)` for each, then
    per step `sim.step(controller.decide(sim.observe()))` and `observer.after_step(sim)`;
    `observer.finish(sim, out_dir, run_id)` after the results are written."""
    scenario = cfg["scenario"]
    controller = make_controller(cfg)
    out_dir = run_dir(cfg, seed, out_root)
    sources = config_files(cfg, config_path)  # before simulating: a bad path fails fast
    run_id = str(uuid.uuid4())
    started_at = datetime.now(UTC).isoformat()

    sim = SimSession(cfg, seed, out_dir, gui=gui, demand_dir=demand_dir)
    dir_existed = out_dir.is_dir()
    before = _file_stamps(sim.raw_outputs)
    wall = time.perf_counter()
    try:
        with sim:
            controller.bind(sim)
            controller.reset(seed)
            for o in observers:
                o.bind(sim)
            while not sim.done:
                sim.step(controller.decide(sim.observe()))
                for o in observers:
                    o.after_step(sim)
    except SimulationClosed:
        remove_partial_outputs(sim, before, out_dir, dir_existed)
        raise
    wall = time.perf_counter() - wall

    results = collect_results(
        run_id, scenario, seed, sim.tripinfo, sim.stops, sim.state_rows,
        ped_single_waits=pd.Series(sim.ped_wait_max, dtype=float), routes=sim.routes,
        cfg=cfg, junction=sim.junction.summary() if sim.junction is not None else None,
        ped_curb_waits=pd.Series(sim.ped_curb_wait_max, dtype=float),
    )  # fmt: skip
    if cfg.get("measurement"):
        add_measurements(results, sim, run_id)
    results.events = event_table(run_id, run_events(controller, sim))
    health = run_health(read_statistics(sim.statistics), results.trips, sim.drain_timeout)
    preemption = preemption_summary(sim)
    meta = run_metadata(
        cfg, run_id, seed, started_at, wall, health, safety_block(sim), sources,
        preemption=preemption, control=controller.report(),
    )  # fmt: skip
    write_results(out_dir, results, meta)
    for o in observers:
        o.finish(sim, out_dir, run_id)
    n_trips = int((results.trips["type"] != "pedestrian").sum())
    log.info("%s seed %d: %d trips, %.1f s wall -> %s", scenario, seed, n_trips, wall, out_dir)
    log.info("%s seed %d health: %s", scenario, seed, health_line(health, preemption))
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCENARIO_CONFIG)
    parser.add_argument("--seed", type=int, nargs="*", help="one or more seeds (default: config)")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        cfg = load_scenario(args.config)
    except ConfigError as e:
        raise SystemExit(f"config error: {e}") from None
    out_root = Path(cfg["outputs"]["dir"])
    seeds = args.seed or [int(cfg["seed"])]
    frames = []
    for seed in seeds:
        try:
            meta = run(cfg, seed, out_root, gui=args.gui, config_path=args.config)
        except ConfigError as e:
            raise SystemExit(f"config error: {e}") from None
        except SimulationClosed as e:
            log.warning("seed %d: %s. No results written.", seed, e)
            raise SystemExit(1) from None
        frames.append(pd.read_csv(run_dir(cfg, seed, out_root) / "kpi_run.csv"))
        log.info("run_id %s", meta["run_id"])
    summary = pd.concat(frames).groupby(["user_type", "kpi"])["value"].agg(["mean", "std"])
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()

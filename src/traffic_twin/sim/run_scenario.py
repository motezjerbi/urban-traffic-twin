"""Run one scenario headless with a given controller and seed, then export trips,
KPIs and a state log (spec 8.2, 9.1), plus the detector, queue and signal tables when
the scenario has a `measurement` section.

    python -m traffic_twin.sim.run_scenario --config configs/scenario_base.yaml --seed 42
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from traffic_twin.config import ConfigError, get, load_scenario
from traffic_twin.control.registry import make_controller
from traffic_twin.control.safety import INTERVENTION_KEYS
from traffic_twin.paths import SCENARIO_CONFIG
from traffic_twin.sim.audit import audit_phases
from traffic_twin.sim.detectors import E1_FILE, TLS_SWITCH_FILE, read_e1, read_tls_switch
from traffic_twin.sim.kpi import read_statistics
from traffic_twin.sim.results import (
    RunResults,
    collect_results,
    config_files,
    detector_table,
    queue_table,
    run_health,
    run_metadata,
    signal_table,
    write_results,
)
from traffic_twin.sim.session import SimSession, SimulationClosed

log = logging.getLogger(__name__)


def run_dir(cfg: dict, seed: int, out_root: Path) -> Path:
    return Path(out_root) / f"{cfg['scenario']}_seed{seed}"


def safety_block(sim: SimSession) -> dict:
    """Guard interventions (one per event) and the audit of the phases shown."""
    crossings = get(sim.cfg, "network.crossings", None) or {}
    ped_links = [int(c["link"]) for c in crossings.values()] if crossings else None
    audit = audit_phases(
        sim.phases_shown,
        sim.phase_states,
        sim.plan.stages,
        float(get(sim.cfg, "safety.min_green_s", 0.0)),
        step_s=sim.dt,
        ped_links=ped_links,
    )
    interventions = {k: int(sim.guard.interventions.get(k, 0)) for k in INTERVENTION_KEYS}
    return {"interventions": interventions, "audit": {k: int(v) for k, v in audit.items()}}


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


def health_line(health: dict) -> str:
    """One-line summary of the health block for the log."""

    def num(key: str) -> str:
        v = health.get(key)
        return "n/a" if v is None else f"{v:g}"

    return (
        f"teleports {num('teleports')}, collisions {num('collisions')}, "
        f"emergency stops {num('emergency_stops')}, braking {num('emergency_braking')}, "
        f"waiting at end {num('vehicles_waiting_end')}, depart delay mean "
        f"{num('depart_delay_mean')} s / max {num('depart_delay_max')} s"
        + (", DRAIN TIMEOUT" if health.get("drain_timeout") else "")
    )


def run(
    cfg: dict, seed: int, out_root: Path, gui: bool = False, config_path: Path | None = None
) -> dict:
    """Run the scenario's controller for one seed and write its results. Raises
    SimulationClosed (after removing SUMO's partial outputs) if SUMO stops early.
    `config_sources` in run.json lists the files of `config_path`, or those `cfg` was
    loaded from by `load_scenario`; it is None for a config built or copied in memory."""
    scenario = cfg["scenario"]
    controller = make_controller(cfg)
    out_dir = run_dir(cfg, seed, out_root)
    sources = config_files(cfg, config_path)  # before simulating: a bad path fails fast
    run_id = str(uuid.uuid4())
    started_at = datetime.now(UTC).isoformat()

    sim = SimSession(cfg, seed, out_dir, gui=gui)
    wall = time.perf_counter()
    try:
        with sim:
            controller.reset(seed)
            while not sim.done:
                sim.step(controller.decide(sim.observe()))
    except SimulationClosed:
        for p in sim.raw_outputs:  # partial SUMO outputs
            p.unlink(missing_ok=True)
        if out_dir.is_dir() and not any(out_dir.iterdir()):
            out_dir.rmdir()
        raise
    wall = time.perf_counter() - wall

    results = collect_results(
        run_id, scenario, seed, sim.tripinfo, sim.stops, sim.state_rows,
        ped_single_waits=pd.Series(sim.ped_wait_max, dtype=float), routes=sim.routes,
    )  # fmt: skip
    if cfg.get("measurement"):
        add_measurements(results, sim, run_id)
    health = run_health(read_statistics(sim.statistics), results.trips, sim.drain_timeout)
    meta = run_metadata(cfg, run_id, seed, started_at, wall, health, safety_block(sim), sources)
    write_results(out_dir, results, meta)
    n_trips = int((results.trips["type"] != "pedestrian").sum())
    log.info("%s seed %d: %d trips, %.1f s wall -> %s", scenario, seed, n_trips, wall, out_dir)
    log.info("%s seed %d health: %s", scenario, seed, health_line(health))
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

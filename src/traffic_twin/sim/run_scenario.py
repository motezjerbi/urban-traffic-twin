"""Run one scenario headless with a given controller and seed, then export trips,
KPIs and a state log (spec 8.2, 9.1).

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

from traffic_twin.config import ConfigError, load_scenario
from traffic_twin.control.registry import make_controller
from traffic_twin.paths import SCENARIO_CONFIG
from traffic_twin.sim.results import code_version, collect_results, write_results
from traffic_twin.sim.session import SimSession, SimulationClosed

log = logging.getLogger(__name__)


def run_dir(cfg: dict, seed: int, out_root: Path) -> Path:
    return Path(out_root) / f"{cfg['scenario']}_seed{seed}"


def run(cfg: dict, seed: int, out_root: Path, gui: bool = False) -> dict:
    """Run the scenario's controller for one seed and write its results. Raises
    SimulationClosed (after removing SUMO's partial outputs) if SUMO stops early."""
    scenario = cfg["scenario"]
    controller = make_controller(cfg)
    out_dir = run_dir(cfg, seed, out_root)
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
        for p in (sim.tripinfo, sim.stops):  # partial SUMO outputs
            p.unlink(missing_ok=True)
        if not any(out_dir.iterdir()):
            out_dir.rmdir()
        raise
    wall = time.perf_counter() - wall

    results = collect_results(run_id, scenario, seed, sim.tripinfo, sim.stops, sim.state_rows)
    meta = {
        "run_id": run_id,
        "scenario": scenario,
        "seed": seed,
        "code_version": code_version(),
        "config": cfg,
        "started_at": started_at,
        "duration_s": int(cfg["duration_s"]),
        "wall_time_s": round(wall, 1),
    }
    write_results(out_dir, results, meta)
    n_trips = int((results.trips["type"] != "pedestrian").sum())
    log.info("%s seed %d: %d trips, %.1f s wall -> %s", scenario, seed, n_trips, wall, out_dir)
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
            meta = run(cfg, seed, out_root, gui=args.gui)
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

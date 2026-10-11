"""Run one scenario headless with a given controller and seed, then export trips,
KPIs and a state log (spec 8.2, 9.1).

    python -m traffic_twin.sim.run_scenario --config configs/scenario_base.yaml --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from traffic_twin.config import load_config
from traffic_twin.control.base import Controller, Observation
from traffic_twin.control.fixed import FixedTimeController
from traffic_twin.sim.demand import write_demand
from traffic_twin.sim.kpi import compute_kpis, read_bus_stop_arrivals, read_tripinfo
from traffic_twin.sim.network import find_binary
from traffic_twin.sim.signal import SignalPlan

log = logging.getLogger(__name__)


def make_controller(cfg: dict) -> Controller:
    kind = cfg["controller"]
    stages = [s["name"] for s in cfg["network"]["stages"]]
    if kind == "fixed":
        return FixedTimeController([cfg["fixed"]["green_s"][s] for s in stages])
    raise ValueError(f"controller '{kind}' arrives in a later part of the roadmap")


def code_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _traci():
    """libsumo is much faster; fall back to TraCI (needed for sumo-gui)."""
    try:
        import libsumo as traci
    except ImportError:
        import traci
    return traci


def run(cfg: dict, seed: int, out_root: Path, gui: bool = False) -> dict:
    scenario = cfg["scenario"]
    run_id = str(uuid.uuid4())
    out_dir = out_root / f"{scenario}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rou, add = write_demand(cfg, Path("sumo/demand"))

    net = cfg["network"]
    begin = int(cfg["begin_s"])
    end = begin + int(cfg["duration_s"])
    dt = float(cfg.get("step_length_s", 1.0))
    tripinfo, stops = out_dir / "tripinfo.xml", out_dir / "stops.xml"
    cmd = [
        find_binary("sumo-gui" if gui else "sumo"),
        "-n", net["net_file"], "-r", str(rou), "-a", str(add),
        "--begin", str(begin), "--end", str(end + 1800),  # let the network empty
        "--step-length", str(dt), "--seed", str(seed),
        "--tripinfo-output", str(tripinfo), "--stop-output", str(stops),
        "--device.emissions.probability", "0",
        "--no-step-log", "true", "--no-warnings", "true",
        "--time-to-teleport", "300",
    ]  # fmt: skip

    traci = __import__("traci") if gui else _traci()
    traci.start(cmd)
    tls = net["tls_id"]
    logic = traci.trafficlight.getAllProgramLogics(tls)[0]
    plan = SignalPlan.from_config(net["stages"], [p.duration for p in logic.phases])
    controller = make_controller(cfg)
    controller.reset(seed)
    traci.trafficlight.setPhase(tls, plan.phase)

    state_rows: list[dict] = []
    every = int(cfg["outputs"].get("state_log_every_s", 5))
    wall = time.perf_counter()
    t = float(begin)
    while t < end or (traci.simulation.getMinExpectedNumber() > 0 and t < end + 1800):
        rel = t - begin
        queues = {a: traci.edge.getLastStepHaltingNumber(e) for a, e in net["approaches"].items()}
        # After the demand window the controller keeps running so the network empties
        obs = Observation(rel, plan.stage, plan.time_in_stage, plan.in_transition, queues)
        phase = plan.request(controller.decide(obs))
        traci.trafficlight.setPhase(tls, phase)
        traci.trafficlight.setPhaseDuration(tls, 1e6)  # we drive switching ourselves
        if int(rel) % every == 0 and t < end:
            state_rows.append(
                {"t": rel, "entity": "intersection", "id": tls, "x": None, "y": None,
                 "speed": None, "phase": phase, "queue": sum(queues.values())}
            )  # fmt: skip
            for vid in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(vid)
                vtype = traci.vehicle.getTypeID(vid)
                state_rows.append(
                    {"t": rel, "entity": vtype if vtype in ("bus", "ambulance") else "car",
                     "id": vid, "x": x, "y": y, "speed": traci.vehicle.getSpeed(vid),
                     "phase": None, "queue": None}
                )  # fmt: skip
            for pid in traci.person.getIDList():
                x, y = traci.person.getPosition(pid)
                state_rows.append(
                    {"t": rel, "entity": "pedestrian", "id": pid, "x": x, "y": y,
                     "speed": traci.person.getSpeed(pid), "phase": None, "queue": None}
                )  # fmt: skip
        traci.simulationStep()
        plan.advance(dt)
        t = traci.simulation.getTime()
    traci.close()
    wall = time.perf_counter() - wall

    trips, walks = read_tripinfo(tripinfo)
    kpis = compute_kpis(trips, walks, read_bus_stop_arrivals(stops), scenario, seed)
    kpis.insert(0, "run_id", run_id)
    trips_all = pd.concat([trips, walks], ignore_index=True)
    trips_all.insert(0, "run_id", run_id)
    state = pd.DataFrame(state_rows)
    state.insert(0, "run_id", run_id)

    kpis.to_csv(out_dir / "kpi_run.csv", index=False)
    trips_all.to_parquet(out_dir / "trip.parquet", index=False)
    state.to_parquet(out_dir / "state_log.parquet", index=False)
    meta = {
        "run_id": run_id,
        "scenario": scenario,
        "seed": seed,
        "code_version": code_version(),
        "config": cfg,
        "started_at": datetime.now(UTC).isoformat(),
        "duration_s": int(cfg["duration_s"]),
        "wall_time_s": round(wall, 1),
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("%s seed %d: %d trips, %.1f s wall -> %s", scenario, seed, len(trips), wall, out_dir)
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scenario_base.yaml")
    parser.add_argument("--seed", type=int, nargs="*", help="one or more seeds (default: config)")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    out_root = Path(cfg["outputs"]["dir"])
    seeds = args.seed or [int(cfg["seed"])]
    frames = []
    for seed in seeds:
        meta = run(cfg, seed, out_root, gui=args.gui)
        frames.append(pd.read_csv(out_root / f"{cfg['scenario']}_seed{seed}" / "kpi_run.csv"))
        log.info("run_id %s", meta["run_id"])
    summary = pd.concat(frames).groupby(["user_type", "kpi"])["value"].agg(["mean", "std"])
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()

"""Turn a finished run's SUMO outputs into result tables and write them (spec 9.1, 9.4):
`kpi_run.csv`, `trip.parquet`, `state_log.parquet` and `run.json`."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from traffic_twin.sim.kpi import compute_kpis, read_bus_stop_arrivals, read_tripinfo


@dataclass
class RunResults:
    kpis: pd.DataFrame
    trips: pd.DataFrame  # vehicle trips and pedestrian walks
    state: pd.DataFrame


def collect_results(
    run_id: str,
    scenario: str,
    seed: int,
    tripinfo: Path,
    stops: Path,
    state_rows: list[dict],
) -> RunResults:
    trips, walks = read_tripinfo(tripinfo)
    kpis = compute_kpis(trips, walks, read_bus_stop_arrivals(stops), scenario, seed)
    trips_all = pd.concat([trips, walks], ignore_index=True)
    state = pd.DataFrame(state_rows)
    for df in (kpis, trips_all, state):
        df.insert(0, "run_id", run_id)
    return RunResults(kpis, trips_all, state)


def write_results(out_dir: Path, results: RunResults, meta: dict) -> None:
    results.kpis.to_csv(out_dir / "kpi_run.csv", index=False)
    results.trips.to_parquet(out_dir / "trip.parquet", index=False)
    results.state.to_parquet(out_dir / "state_log.parquet", index=False)
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def code_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

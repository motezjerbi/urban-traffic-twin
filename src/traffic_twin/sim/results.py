"""Turn a finished run's SUMO outputs into result tables and write them (spec 9.1, 9.4):
`kpi_run.csv`, `trip.parquet`, `state_log.parquet` and `run.json`, plus, when the scenario
has a `measurement` section, `detector.parquet`, `queue.parquet` and `signal.parquet`,
and `events.parquet` when a controller, the guard or the preemption logged an event."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from traffic_twin.config import config_hash, config_sources, demand_hash, loaded_sources
from traffic_twin.control.base import Event
from traffic_twin.sim.detectors import DetectorLayout
from traffic_twin.sim.kpi import (
    KPI_VERSION,
    bus_lines,
    compute_kpis,
    extra_kpis,
    read_bus_stops,
    read_tripinfo,
)

DETECTOR_COLUMNS = [
    "run_id", "det_id", "station", "lane", "begin_s", "end_s", "n_veh", "occupancy_pct",
    "mean_speed",
]  # fmt: skip
QUEUE_COLUMNS = ["run_id", "t", "approach", "jam_veh", "jam_m", "halting_veh"]
SIGNAL_COLUMNS = ["run_id", "link", "begin_s", "end_s", "duration_s"]
EVENT_COLUMNS = ["run_id", "t", "source", "rule", "stage_from", "stage_to", "detail"]
HEALTH_KEYS = (
    "teleports", "collisions", "emergency_stops", "emergency_braking", "vehicles_waiting_end",
    "depart_delay_mean", "depart_delay_max", "drain_timeout",
)  # fmt: skip

# run.json `preemption` block: the manager's report, then the guard's preemption counts
PREEMPTION_REPORT_KEYS = ("requests", "served", "timeouts", "active_s")
PREEMPTION_GUARD_KEYS = ("hold_s", "jumps", "ped_yield")

# Optional tables: RunResults attribute -> file name
OPTIONAL_TABLES = {"detectors": "detector.parquet", "queues": "queue.parquet",
                   "signal": "signal.parquet", "events": "events.parquet"}  # fmt: skip


@dataclass
class RunResults:
    kpis: pd.DataFrame
    trips: pd.DataFrame  # vehicle trips and pedestrian walks
    state: pd.DataFrame
    detectors: pd.DataFrame | None = None  # station loop counts per interval
    queues: pd.DataFrame | None = None  # E2 queue samples per approach
    signal: pd.DataFrame | None = None  # green intervals per TLS link
    events: pd.DataFrame | None = None  # controller, guard and preemption events


def collect_results(
    run_id: str,
    scenario: str,
    seed: int,
    tripinfo: Path,
    stops: Path,
    state_rows: list[dict],
    ped_single_waits: pd.Series | None = None,
    routes: Path | None = None,
    cfg: dict | None = None,
    junction: Mapping | None = None,
    ped_curb_waits: pd.Series | None = None,
) -> RunResults:
    """KPIs, trips and state log of a run. `routes` (the demand file) gives the bus line
    of each stop event; `ped_single_waits` the longest single wait per pedestrian
    (`getWaitingTime`), `ped_curb_waits` the longest curb wait (creeps included).

    With the scenario `cfg`, the KPI version 2 rows of `extra_kpis` (`junction`: the
    `JunctionMonitor.summary()` of the run, if any) follow the `compute_kpis` rows, which
    are unchanged."""
    trips, walks = read_tripinfo(tripinfo)
    bus_stops = read_bus_stops(stops, bus_lines(routes))
    kpis = compute_kpis(trips, walks, bus_stops, scenario, seed, ped_single_waits)
    if cfg is not None:
        extra = extra_kpis(trips, walks, bus_stops, cfg, junction, ped_single_waits, ped_curb_waits)
        if not extra.empty:
            extra.insert(0, "seed", seed)
            extra.insert(0, "scenario", scenario)
            kpis = pd.concat([kpis, extra], ignore_index=True)
    trips_all = pd.concat([trips, walks], ignore_index=True)
    for col in ("depart_edge", "arrival_edge", "flow"):  # walks have none
        if trips_all[col].dtype == object:  # None (absent lane) mixed with NaN: one null
            trips_all[col] = trips_all[col].where(trips_all[col].notna(), None)
    state = pd.DataFrame(state_rows)
    for df in (kpis, trips_all, state):
        df.insert(0, "run_id", run_id)
    return RunResults(kpis, trips_all, state)


def detector_table(
    run_id: str, e1: pd.DataFrame, layout: DetectorLayout, begin_s: float
) -> pd.DataFrame:
    """Station loop counts (`read_e1` output) with their station and lane; times relative
    to the scenario start."""
    station = {d: s for s, ids in layout.stations.items() for d in ids}
    lane = {lp.id: lp.lane for lp in layout.loops}
    df = e1[e1["det_id"].isin(list(station))].copy()
    df.insert(0, "run_id", run_id)
    df["station"] = df["det_id"].map(station)
    df["lane"] = df["det_id"].map(lane)
    df["begin_s"] = df["begin_s"] - begin_s
    df["end_s"] = df["end_s"] - begin_s
    return df[DETECTOR_COLUMNS].reset_index(drop=True)


def queue_table(run_id: str, rows: list[dict]) -> pd.DataFrame:
    """Queue samples (`SimSession.queue_rows`), one row per sample time and approach."""
    df = pd.DataFrame(rows, columns=QUEUE_COLUMNS[1:])
    df.insert(0, "run_id", run_id)
    return df


def signal_table(
    run_id: str,
    switches: pd.DataFrame,
    links: Mapping[tuple[str, str], int],
    begin_s: float,
) -> pd.DataFrame:
    """Green intervals per TLS link (`read_tls_switch` output); times relative to the
    scenario start. Raises KeyError for a switch on a lane pair the net does not link."""
    keys = list(zip(switches["from_lane"], switches["to_lane"], strict=True))
    unknown = sorted({k for k in keys if k not in links})
    if unknown:
        raise KeyError(f"tls_switch: lane pairs not linked by the traffic light: {unknown}")
    df = pd.DataFrame(
        {
            "run_id": run_id,
            "link": pd.Series([links[k] for k in keys], dtype="int64"),
            "begin_s": switches["begin_s"].to_numpy() - begin_s,
            "end_s": switches["end_s"].to_numpy() - begin_s,
            "duration_s": switches["duration_s"].to_numpy(),
        }
    )
    return df.sort_values(["begin_s", "link"], kind="stable").reset_index(drop=True)


def event_table(run_id: str, events: Sequence[Event]) -> pd.DataFrame | None:
    """`events.parquet`: one row per event, by time (stable: the controller's events of
    a step come before the guard's and the preemption's). None when there is none."""
    if not events:
        return None
    df = pd.DataFrame(
        [(run_id, e.t, e.source, e.rule, e.stage_from, e.stage_to, e.detail) for e in events],
        columns=EVENT_COLUMNS,
    )
    df = df.astype({"t": "float64", "stage_from": "int64", "stage_to": "int64"})
    return df.sort_values("t", kind="stable").reset_index(drop=True)


def preemption_block(report: Mapping | None, guard_counts: Mapping | None) -> dict:
    """`run.json` preemption block: `enabled` (a preemption manager ran: `report` is its
    `report()`, None when disabled), the report's requests, served, timeouts and active_s,
    and the guard's counts (hold_s, jumps, ped_yield); zeros when disabled."""
    block: dict = {"enabled": report is not None}
    report, guard_counts = report or {}, guard_counts or {}
    for key in PREEMPTION_REPORT_KEYS:
        block[key] = report.get(key, 0)
    for key in PREEMPTION_GUARD_KEYS:
        block[key] = guard_counts.get(key, 0)
    return block


def run_health(statistics: Mapping, trips: pd.DataFrame, drain_timeout: bool) -> dict:
    """`run.json` health block: SUMO statistics (teleports, collisions, ...), insertion
    delay of the vehicles that finished, and whether the drain timed out."""
    vehicles = trips[trips["type"] != "pedestrian"] if "type" in trips else trips
    delays = vehicles.get("depart_delay_s", pd.Series(dtype=float)).dropna()
    health = {k: statistics.get(k) for k in HEALTH_KEYS[:5]}
    health["depart_delay_mean"] = round(float(delays.mean()), 3) if len(delays) else None
    health["depart_delay_max"] = float(delays.max()) if len(delays) else None
    health["drain_timeout"] = bool(drain_timeout)
    health.update({k: v for k, v in statistics.items() if k not in health})  # extra counts
    return health


def run_metadata(
    cfg: dict,
    run_id: str,
    seed: int,
    started_at: str,
    wall_s: float,
    health: dict,
    safety: dict,
    sources: Sequence[str] | None = None,
    *,
    preemption: dict | None = None,
    control: dict | None = None,
) -> dict:
    """`run.json`: run identity, the merged config and its hashes (demand_hash must match
    between compared scenarios, spec 13.3), health, the safety block, the config files the
    scenario was assembled from (`config_files`; None when unknown), the KPI version, the
    preemption block (`preemption_block`; disabled when not given) and the controller's
    `report()` (`control`)."""
    return {
        "run_id": run_id,
        "scenario": cfg["scenario"],
        "seed": seed,
        "code_version": code_version(),
        "config": cfg,
        "started_at": started_at,
        "duration_s": int(cfg["duration_s"]),
        "wall_time_s": round(wall_s, 1),
        "config_hash": config_hash(cfg),
        "demand_hash": demand_hash(cfg),
        "config_sources": None if sources is None else list(sources),
        "health": health,
        "safety": safety,
        "kpi_version": KPI_VERSION,
        "preemption": preemption if preemption is not None else preemption_block(None, None),
        "control": control if control is not None else {},
    }


def _posix(path: Path) -> str:
    """POSIX path, relative to the working directory (the repo root) when inside it."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def config_files(cfg: dict, config_path: Path | str | None = None) -> list[str] | None:
    """The config files of a run: `config_sources(config_path)`, else those `cfg` was
    loaded from (`loaded_sources`), else None (built in memory)."""
    paths = config_sources(config_path) if config_path else loaded_sources(cfg)
    return None if paths is None else [_posix(Path(p)) for p in paths]


def write_results(out_dir: Path, results: RunResults, meta: dict) -> None:
    results.kpis.to_csv(out_dir / "kpi_run.csv", index=False)
    results.trips.to_parquet(out_dir / "trip.parquet", index=False)
    results.state.to_parquet(out_dir / "state_log.parquet", index=False)
    for attr, name in OPTIONAL_TABLES.items():
        table = getattr(results, attr)
        if table is not None:
            table.to_parquet(out_dir / name, index=False)
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def code_version() -> str:
    """Git short hash of HEAD in the working directory (the repo root), with `-dirty`
    appended when the tree differs from that commit: a modified, staged, deleted or new
    untracked file (ignored files such as data/results do not count). A run made from an
    uncommitted tree is then not credited to a commit that cannot reproduce it. Files
    under docs/ (generated pages, figures and prose) cannot change a run and do not
    count, so regenerating the pages mid-way does not mark later runs dirty.
    "unknown" when git or the repository is not available."""
    git = ["git", "--no-optional-locks"]  # read-only: never takes .git/index.lock
    try:
        head = subprocess.run(
            [*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            [
                *git,
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--",
                ".",
                ":(exclude)docs",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}-dirty" if status.strip() else head

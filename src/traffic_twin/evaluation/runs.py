"""Evaluation runs of one scenario (`<results>/<scenario>_seed<n>/`): one loader for the
comparison (evaluation.compare) and the validation report (validation.report), and the
check that every run was made with the CURRENT config.

A run records `config_hash` and `demand_hash` in its run.json. A run whose hashes differ
from those of the current config (a program, demand, calibration or net change since it
ran) is STALE: `run_missing` reruns it like an absent seed, and `check_current` refuses
to use it. So the documented post-visit rerun (`tasks.py part3`) can never publish
results of the old program.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from traffic_twin.config import config_hash, demand_hash

log = logging.getLogger(__name__)

HASH_KEYS = ("config_hash", "demand_hash")
# Run tables: attribute -> file name
TABLES = {
    "kpis": "kpi_run.csv",
    "detectors": "detector.parquet",
    "queues": "queue.parquet",
    "signal": "signal.parquet",
    "trips": "trip.parquet",
}


class StaleRunsError(ValueError):
    """Runs that were not made with the current config (or do not share one)."""


@dataclass
class Runs:
    """The evaluation runs of one scenario (each table carries a `seed` column)."""

    scenario: str
    seeds: list[int]
    meta: dict[int, dict] = field(default_factory=dict)
    kpis: pd.DataFrame = field(default_factory=pd.DataFrame)
    detectors: pd.DataFrame = field(default_factory=pd.DataFrame)
    queues: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal: pd.DataFrame = field(default_factory=pd.DataFrame)
    trips: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_path(results: Path, scenario: str, seed: int) -> Path:
    return Path(results) / f"{scenario}_seed{seed}"


def is_complete(run_dir: Path) -> bool:
    """A finished run: run.json (written last) and kpi_run.csv."""
    return (run_dir / "run.json").is_file() and (run_dir / "kpi_run.csv").is_file()


def read_meta(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))


def load_runs(
    results: Path, scenario: str, seeds: list[int], tables: tuple[str, ...] = ("kpis",)
) -> Runs:
    """The finished runs of `scenario` for `seeds` under `results` (absent seeds are left
    out), with the requested `tables` (keys of TABLES; absent files are skipped)."""
    unknown = set(tables) - set(TABLES)
    if unknown:
        raise ValueError(f"unknown run tables {sorted(unknown)}")
    frames: dict[str, list[pd.DataFrame]] = {t: [] for t in tables}
    meta: dict[int, dict] = {}
    for seed in seeds:
        d = run_path(results, scenario, seed)
        if not is_complete(d):
            continue
        meta[seed] = read_meta(d)
        for t in tables:
            path = d / TABLES[t]
            if not path.is_file():
                continue
            if path.suffix == ".csv":
                df = pd.read_csv(path, float_precision="round_trip")
            else:
                df = pd.read_parquet(path)
            frames[t].append(df.assign(seed=seed))
    out = {t: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for t, v in frames.items()}
    return Runs(scenario, sorted(meta), meta, **out)


# -- staleness -------------------------------------------------------------------------
def expected_hashes(cfg: dict) -> dict[str, str]:
    """The hashes a run of the current config records (run.json)."""
    return {"config_hash": config_hash(cfg), "demand_hash": demand_hash(cfg)}


def short(h: object) -> str:
    return "none" if h is None else str(h)[:12]


def stale_reason(meta: dict, expected: dict[str, str]) -> str | None:
    """Why a run is not a run of the current config (None when it is)."""
    bad = [
        f"{k} {short(meta.get(k))} != current {short(expected[k])}"
        for k in HASH_KEYS
        if meta.get(k) != expected[k]
    ]
    return "; ".join(bad) or None


def stale_seeds(runs: Runs, expected: dict[str, str]) -> dict[int, str]:
    """Seed -> reason, for the loaded runs that are not runs of the current config."""
    out = {}
    for seed, m in sorted(runs.meta.items()):
        reason = stale_reason(m, expected)
        if reason:
            out[seed] = reason
    return out


def check_current(runs: Runs, cfg: dict) -> dict[str, str]:
    """The hashes shared by every loaded run (one config_hash, one demand_hash, both those
    of the current `cfg`). Raises StaleRunsError naming the stale seeds otherwise."""
    expected = expected_hashes(cfg)
    stale = stale_seeds(runs, expected)
    if stale:
        listing = "; ".join(f"seed {s}: {r}" for s, r in stale.items())
        raise StaleRunsError(
            f"{runs.scenario}: {len(stale)} run(s) were not made with the current config "
            f"(program, demand, calibration or net changed since): {listing}. Rerun them "
            "(--run-missing reruns stale seeds)"
        )
    return expected


def run_missing(
    cfg: dict, seeds: list[int], results: Path, config_path: Path | None = None
) -> list[int]:
    """Run the absent seeds and rerun the stale ones (made with another config_hash or
    demand_hash than the current config); a rerun overwrites the run's files. Returns the
    seeds that were (re)run."""
    from traffic_twin.sim.run_scenario import run

    expected = expected_hashes(cfg)
    done = []
    for seed in seeds:
        d = run_path(results, cfg["scenario"], seed)
        if is_complete(d):
            reason = stale_reason(read_meta(d), expected)
            if reason is None:
                continue
            log.warning("%s seed %d is stale (%s): rerunning", cfg["scenario"], seed, reason)
        else:
            log.info("running %s seed %d", cfg["scenario"], seed)
        run(cfg, seed, results, config_path=config_path)
        done.append(seed)
    return done

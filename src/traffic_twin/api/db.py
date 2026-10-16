"""Data access of the API (Part 5, D53; contract 4.6): local Parquet / CSV / JSON files
through DuckDB, behind a `Store` protocol (Supabase stays a later backend).

- **Run index:** a one-level glob `*/run.json` over `results_dirs` (nested backups such as
  `data/results/_rebaseline/<sha>/` are skipped), rebuilt at most every
  `index_ttl_s`. Each run says whether it is `current` (its `config_hash` and
  `demand_hash` equal those of the current config of its scenario; null when that config
  cannot be found or the API does not run from the repo root).
- **Per request:** `state_log.parquet` (~70k rows) is queried directly with DuckDB at the
  replay time, and so are `eta_prediction.parquet` and `eta_snapshots.parquet` (the
  filter runs in SQL). One DuckDB cursor per call: the connection is shared by the
  worker threads.
- **Cached:** the small tables (signal, events, trips, queues), the evaluation CSVs and
  the ETA reference table, keyed by path, size and mtime (a rewritten file is re-read).

Only Parquet, CSV and JSON are read: raw SUMO XMLs may be absent. A missing file gives
None; an unreadable one gives None too and a warning in the log, so the endpoints answer
in a degraded or unavailable state, never with an error.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore

import pandas as pd

from traffic_twin.api.schemas import ApiError
from traffic_twin.api.settings import Settings
from traffic_twin.config import load_config

log = logging.getLogger(__name__)

RUN_FILE = "run.json"
STATE_FILE = "state_log.parquet"
SIGNAL_FILE = "signal.parquet"
EVENTS_FILE = "events.parquet"
TRIPS_FILE = "trip.parquet"
QUEUE_FILE = "queue.parquet"
KPI_FILE = "kpi_run.csv"
ETA_PREDICTION_FILE = "eta_prediction.parquet"  # ml.predict (contract 4.5)
ETA_SNAPSHOTS_FILE = "eta_snapshots.parquet"  # ml.eta_dataset.EtaRecorder
REFERENCE_FILE = "reference.parquet"  # ETA bundle: degraded-mode table (D49)
LATEST_FILE = "LATEST"  # <models_dir>/LATEST holds the bundle version
TABLE_CACHE_SIZE = 256  # small tables kept in memory (a run has at most 4 of them)


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    dir: Path
    scenario: str
    seed: int
    controller: str | None
    code_version: str | None
    started_at: datetime | None
    begin_s: float
    duration_s: float
    day_type: str | None
    config_hash: str | None
    demand_hash: str | None
    kpi_version: int | None
    current: bool | None = None
    config: dict = field(default_factory=dict, repr=False, compare=False)

    def has(self, name: str) -> bool:
        return (self.dir / name).is_file()

    @property
    def has_eta(self) -> bool:
        return self.has(ETA_PREDICTION_FILE)


class Store(Protocol):
    """Read-only access to the run outputs, the evaluation and the ETA files."""

    def runs(self) -> dict[str, RunInfo]: ...
    def run(self, run_id: str) -> RunInfo: ...
    def default_run(self) -> RunInfo | None: ...
    def state_at(self, run_id: str, t: float) -> tuple[float | None, pd.DataFrame]: ...
    def signal(self, run_id: str) -> pd.DataFrame | None: ...
    def events(self, run_id: str) -> pd.DataFrame | None: ...
    def trips(self, run_id: str) -> pd.DataFrame | None: ...
    def queues(self, run_id: str) -> pd.DataFrame | None: ...
    def eta_at(self, run_id: str, trajet: str, t: float) -> dict | None: ...
    def snapshot_at(self, run_id: str, t: float) -> dict | None: ...
    def reference(self) -> tuple[pd.DataFrame | None, str | None]: ...
    def eta_model_version(self) -> str | None: ...
    def labels(self) -> dict[str, dict[str, str]]: ...
    def kpis(self, family: str, scenario: str) -> tuple[pd.DataFrame | None, float | None]: ...
    def equity(self, family: str) -> pd.DataFrame | None: ...
    def run_kpis(self, scenario: str) -> pd.DataFrame | None: ...


def _parse_started(raw: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(raw)) if raw else None
    except ValueError:
        return None


def read_run_info(meta_path: Path) -> RunInfo:
    """RunInfo of one run.json (ValueError / KeyError / TypeError when malformed)."""
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = m.get("config") or {}
    if not isinstance(cfg, dict):
        raise TypeError("config is not a mapping")
    kv = m.get("kpi_version")
    return RunInfo(
        run_id=str(m["run_id"]),
        dir=meta_path.parent,
        scenario=str(m["scenario"]),
        seed=int(m["seed"]),
        controller=cfg.get("controller"),
        code_version=m.get("code_version"),
        started_at=_parse_started(m.get("started_at")),
        begin_s=float(cfg.get("begin_s", 0.0)),
        duration_s=float(m.get("duration_s") or cfg.get("duration_s") or 0.0),
        day_type=(cfg.get("demand") or {}).get("day_type"),
        config_hash=m.get("config_hash"),
        demand_hash=m.get("demand_hash"),
        kpi_version=None if kv is None else int(kv),
        config=cfg,
    )


class LocalStore:
    """Store over the local run folders (see the module docstring)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # in memory; one cursor per call
        self._con = duckdb.connect() if duckdb is not None else None
        self._lock = threading.RLock()
        self._index: dict[str, RunInfo] = {}
        self._default: RunInfo | None = None
        self._index_at = -float("inf")
        self._tables: OrderedDict[tuple, pd.DataFrame | None] = OrderedDict()
        self._hashes: dict[str, dict[str, str] | None] = {}
        self._labels: dict[str, dict[str, str]] | None = None

    # -- low level -----------------------------------------------------------------------
    def _query(self, sql: str, params: list | dict | None = None) -> pd.DataFrame:
        if self._con is not None:
            cur = self._con.cursor()
            try:
                return cur.execute(sql, params or []).df()
            finally:
                cur.close()
        raise NotImplementedError("DuckDB is not installed")

    def _read(self, path: Path) -> pd.DataFrame | None:
        """A whole Parquet or CSV file, or None (absent or unreadable)."""
        if not path.is_file():
            return None
        if self._con is not None:
            reader = "read_csv_auto" if path.suffix == ".csv" else "read_parquet"
            try:
                return self._query(f"select * from {reader}(?)", [str(path)])
            except (getattr(duckdb, "Error", Exception), OSError) as e:
                log.warning("unreadable %s: %s", path, e)
                return None
        try:
            if path.suffix == ".csv":
                return pd.read_csv(path)
            return pd.read_parquet(path)
        except Exception as e:
            log.warning("unreadable %s: %s", path, e)
            return None

    def _cached(self, path: Path, sort: tuple[str, ...] = ()) -> pd.DataFrame | None:
        """`_read` through the table cache (key: path, size, mtime)."""
        try:
            st = path.stat()
        except OSError:
            return None
        key = (str(path), st.st_size, st.st_mtime_ns)
        with self._lock:
            if key in self._tables:
                self._tables.move_to_end(key)
                return self._tables[key]
        df = self._read(path)
        if df is not None and sort and set(sort) <= set(df.columns):
            df = df.sort_values(list(sort), kind="stable").reset_index(drop=True)
        with self._lock:
            self._tables[key] = df
            while len(self._tables) > TABLE_CACHE_SIZE:
                self._tables.popitem(last=False)
        return df

    # -- run index -----------------------------------------------------------------------
    def _current_hashes(self, scenario: str) -> dict[str, str] | None:
        """The hashes a run of the current config of `scenario` records, or None when
        unknown (no config, or the API does not run from the repo root: the hashes read
        the net and calibration files by their relative paths)."""
        if scenario not in self._hashes:
            from traffic_twin.config import load_scenario
            from traffic_twin.evaluation.runs import expected_hashes

            out = None
            path = self._config_of(scenario)
            if path is not None and Path.cwd().resolve() == self.settings.repo_root.resolve():
                try:
                    out = expected_hashes(load_scenario(path))
                except Exception as e:  # a broken config: currency unknown, never a crash
                    log.warning("current hashes of %s: %s", scenario, e)
            self._hashes[scenario] = out
        return self._hashes[scenario]

    def _config_of(self, scenario: str) -> Path | None:
        for fam in self.labels().values():
            for label_cfg in fam.values():
                if Path(label_cfg).stem == scenario:
                    return self.settings.path(label_cfg)
        p = self.settings.path(Path("configs") / f"{scenario}.yaml")
        return p if p.is_file() else None

    def runs(self) -> dict[str, RunInfo]:
        with self._lock:
            if time.monotonic() - self._index_at < self.settings.index_ttl_s:
                return self._index
        index: dict[str, RunInfo] = {}
        for root in self.settings.results_dirs:
            root = self.settings.path(root)
            if not root.is_dir():
                continue
            for meta in sorted(root.glob(f"*/{RUN_FILE}")):
                try:
                    info = read_run_info(meta)
                except (OSError, ValueError, KeyError, TypeError) as e:
                    log.warning("skipped %s: %s", meta, e)
                    continue
                if info.run_id in index:
                    kept = index[info.run_id].dir
                    log.warning("run %s indexed twice; kept %s", info.run_id, kept)
                    continue
                expected = self._current_hashes(info.scenario)
                current = None
                if expected is not None:
                    current = (info.config_hash, info.demand_hash) == (
                        expected["config_hash"],
                        expected["demand_hash"],
                    )
                index[info.run_id] = replace(info, current=current)
        with self._lock:
            self._index, self._index_at = index, time.monotonic()
            self._default = self._pick_default(index)
        return index

    def refresh(self) -> None:
        with self._lock:
            self._index_at = -float("inf")

    def run(self, run_id: str) -> RunInfo:
        runs = self.runs()
        if run_id not in runs:
            raise ApiError("unknown_run", f"no run {run_id!r}")
        return runs[run_id]

    def _pick_default(self, runs: dict[str, RunInfo]) -> RunInfo | None:
        by_dir = {os.path.normcase(str(r.dir.resolve())): r for r in runs.values()}
        for d in self.settings.default_run:
            hit = by_dir.get(os.path.normcase(str(self.settings.path(d).resolve())))
            if hit is not None:
                return hit
        ordered = sorted(runs.values(), key=lambda r: (r.scenario, r.seed, str(r.dir)))
        return ordered[0] if ordered else None

    def default_run(self) -> RunInfo | None:
        """The first `default_run` folder that is indexed; else the first indexed run in
        (scenario, seed) order; None without runs."""
        self.runs()
        with self._lock:
            return self._default

    # -- run tables ----------------------------------------------------------------------
    def state_at(self, run_id: str, t: float) -> tuple[float | None, pd.DataFrame]:
        """Rows of the last state sample at or before t (the replay 'now')."""
        path = self.run(run_id).dir / STATE_FILE
        if not path.is_file():
            return None, pd.DataFrame()
        if self._con is not None:
            try:
                df = self._query(
                    "select * from read_parquet($p) where t = "
                    "(select max(t) from read_parquet($p) where t <= $t)",
                    {"p": str(path), "t": float(t)},
                )
            except (getattr(duckdb, "Error", Exception), OSError) as e:
                log.warning("unreadable %s: %s", path, e)
                return None, pd.DataFrame()
        else:
            try:
                full = pd.read_parquet(path)
                sub = full[full["t"] <= float(t)]
                if sub.empty:
                    return None, pd.DataFrame()
                max_t = sub["t"].max()
                df = sub[sub["t"] == max_t].copy()
            except Exception as e:
                log.warning("unreadable %s: %s", path, e)
                return None, pd.DataFrame()
        if df.empty:
            return None, df
        return float(df["t"].iloc[0]), df

    def signal(self, run_id: str) -> pd.DataFrame | None:
        return self._cached(self.run(run_id).dir / SIGNAL_FILE, ("begin_s", "link"))

    def events(self, run_id: str) -> pd.DataFrame | None:
        return self._cached(self.run(run_id).dir / EVENTS_FILE, ("t",))

    def trips(self, run_id: str) -> pd.DataFrame | None:
        return self._cached(self.run(run_id).dir / TRIPS_FILE, ("arrival_s",))

    def queues(self, run_id: str) -> pd.DataFrame | None:
        return self._cached(self.run(run_id).dir / QUEUE_FILE, ("t",))

    # -- ETA -----------------------------------------------------------------------------
    def eta_at(self, run_id: str, trajet: str, t: float) -> dict | None:
        """The latest precomputed prediction of `trajet` at or before t (None when the run
        has no readable prediction file or no row)."""
        path = self.run(run_id).dir / ETA_PREDICTION_FILE
        if not path.is_file():
            return None
        if self._con is not None:
            try:
                df = self._query(
                    "select t, eta_median_s, eta_low_s, eta_high_s, model_version "
                    "from read_parquet($p) where route = $r and t <= $t order by t desc limit 1",
                    {"p": str(path), "r": trajet, "t": float(t)},
                )
            except (getattr(duckdb, "Error", Exception), OSError) as e:
                log.warning("unreadable %s: %s", path, e)
                return None
        else:
            try:
                full = pd.read_parquet(path)
                sub = full[(full["route"] == trajet) & (full["t"] <= float(t))]
                if sub.empty:
                    return None
                target_cols = ["t", "eta_median_s", "eta_low_s", "eta_high_s", "model_version"]
                cols = [c for c in target_cols if c in sub.columns]
                df = sub.sort_values("t", ascending=False).iloc[[0]][cols]
            except Exception as e:
                log.warning("unreadable %s: %s", path, e)
                return None
        return None if df.empty else df.iloc[0].to_dict()

    def snapshot_at(self, run_id: str, t: float) -> dict | None:
        """The latest ETA feature snapshot at or before t (None without one)."""
        path = self.run(run_id).dir / ETA_SNAPSHOTS_FILE
        if not path.is_file():
            return None
        if self._con is not None:
            try:
                df = self._query(
                    "select * from read_parquet($p) where t <= $t order by t desc limit 1",
                    {"p": str(path), "t": float(t)},
                )
            except (getattr(duckdb, "Error", Exception), OSError) as e:
                log.warning("unreadable %s: %s", path, e)
                return None
        else:
            try:
                full = pd.read_parquet(path)
                sub = full[full["t"] <= float(t)]
                if sub.empty:
                    return None
                df = sub.sort_values("t", ascending=False).iloc[[0]]
            except Exception as e:
                log.warning("unreadable %s: %s", path, e)
                return None
        return None if df.empty else df.iloc[0].to_dict()

    def eta_model_version(self) -> str | None:
        """The bundle version named by `<eta models_dir>/LATEST`, or None."""
        latest = self.settings.path(self.settings.eta_models_dir) / LATEST_FILE
        try:
            v = latest.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return v or None

    def reference(self) -> tuple[pd.DataFrame | None, str | None]:
        """The latest bundle's reference table (degraded mode, D49) and its version."""
        version = self.eta_model_version()
        if version is None:
            return None, None
        path = self.settings.path(self.settings.eta_models_dir) / version / REFERENCE_FILE
        df = self._cached(path)
        needed = {"route", "hour", "day_type", "mean_s", "p10_s", "p90_s"}
        if df is not None and not needed <= set(df.columns):
            log.warning("%s lacks %s", path, sorted(needed - set(df.columns)))
            df = None
        return df, version

    # -- evaluation ----------------------------------------------------------------------
    def labels(self) -> dict[str, dict[str, str]]:
        """family -> label -> config path, from configs/evaluation.yaml ({} if absent)."""
        if self._labels is None:
            out: dict[str, dict[str, str]] = {}
            try:
                ev = load_config(self.settings.path(self.settings.evaluation_config))
                for fam, spec in (ev.get("families") or {}).items():
                    out[str(fam)] = {str(k): str(v) for k, v in spec.get("scenarios", {}).items()}
            except Exception as e:  # absent or broken: no labels, never a crash
                log.warning("evaluation labels: %s", e)
            self._labels = out
        return self._labels

    def _eval_path(self, family: str, name: str) -> Path:
        return self.settings.path(self.settings.evaluation_dir) / family / name

    def kpis(self, family: str, scenario: str) -> tuple[pd.DataFrame | None, float | None]:
        """Rows of `<evaluation dir>/<family>/kpis.csv` for a scenario name or label,
        and the file's mtime."""
        path = self._eval_path(family, "kpis.csv")
        df = self._cached(path)
        if df is None or not {"scenario", "user_type", "kpi", "value"} <= set(df.columns):
            return None, None
        hit = df["scenario"] == scenario
        if "label" in df.columns:
            hit |= df["label"] == scenario
        return df[hit], path.stat().st_mtime

    def equity(self, family: str) -> pd.DataFrame | None:
        return self._cached(self._eval_path(family, "equity.csv"))

    def run_kpis(self, scenario: str) -> pd.DataFrame | None:
        """Every indexed run's kpi_run.csv of a scenario (the /kpi fallback)."""
        files = sorted(
            str(r.dir / KPI_FILE) for r in self.runs().values()
            if r.scenario == scenario and r.has(KPI_FILE)
        )  # fmt: skip
        if not files:
            return None
        if self._con is not None:
            try:
                return self._query("select * from read_csv_auto(?, union_by_name = true)", [files])
            except (getattr(duckdb, "Error", Exception), OSError) as e:
                log.warning("unreadable kpi_run.csv of %s: %s", scenario, e)
                return None
        try:
            dfs = [pd.read_csv(f) for f in files]
            return pd.concat(dfs, ignore_index=True)
        except Exception as e:
            log.warning("unreadable kpi_run.csv of %s: %s", scenario, e)
            return None

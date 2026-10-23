"""Readers of the Part 8 evaluation outputs and of the replay extras (Parts 8-9; D98,
D103, D105; contract 4.3). Read-only, pandas and pyarrow only (no DuckDB), so they also work
without the `api` extra's DuckDB.

- **Evaluation outputs** under `api.yaml evaluation.dir` (`data/results/evaluation`):
  `summary.json`, `a05.json`, `a06.json`, `provenance.json`, `final_table.csv`,
  `series.csv` and `<family>/pairs/<a>__vs__<b>.csv`, written by `evaluation.evaluate`
  (D98). A missing file gives None (the endpoint answers `unavailable`), an unreadable one
  None and a warning in the log.
- **Evaluation config** (`configs/evaluation.yaml`): the pairs and their roles, the
  `final_table` layout (label roles main / ablation / exploratory), `part8.series` and the
  evaluation seeds.
- **Staleness:** a label is stale when its `provenance.json` config_hash or demand_hash
  differs from the current config's (`evaluation.runs.expected_hashes`). Unknown (never
  stale) without provenance, or when the API does not run from the repo root (the hashes
  read the net and calibration files by their relative paths).
- **Replay:** the time bounds and sampling step of a run's `state_log.parquet`, and the
  site's link -> approach map and yellow times for the per-approach `signals`.

Everything is cached by path, size and mtime (a rewritten file is re-read). Values go
out as plain JSON: NaN and infinities become null.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_twin.api.settings import Settings
from traffic_twin.api.site import Site
from traffic_twin.config import load_config

log = logging.getLogger(__name__)

SUMMARY_FILE = "summary.json"
A05_FILE = "a05.json"
A06_FILE = "a06.json"  # paths.EVAL_A06
PROVENANCE_FILE = "provenance.json"
FINAL_TABLE_FILE = "final_table.csv"  # paths.EVAL_FINAL_TABLE
SERIES_FILE = "series.csv"  # paths.EVAL_SERIES
PAIRS_DIR = "pairs"
ROLES = ("main", "ablation", "exploratory")  # label roles (final_table layout, D103)
CACHE_SIZE = 512  # entries: a few tables plus one (t_min, t_max, step) per replayed run
YELLOW = "yY"
GREEN = "Gg"


def clean(obj: Any) -> Any:
    """`obj` with non-finite floats replaced by None (JSON has no NaN) and numpy scalars
    turned into Python ones."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, str | bytes):  # numpy scalar
        try:
            obj = obj.item()
        except (TypeError, ValueError):
            return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def records(df: pd.DataFrame | None) -> list[dict]:
    """The rows of a table as JSON-ready dicts (NaN -> None)."""
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", double_precision=12, date_format="iso"))


def pair_name(a: str, b: str) -> str:
    return f"{a}__vs__{b}.csv"


def label_roles(ecfg: dict) -> dict[str, str]:
    """label -> main | ablation | exploratory from the `final_table` layout: its columns are
    main, its ablations ablation, any other registered label exploratory ({} without a
    `final_table` block)."""
    ft = ecfg.get("final_table") or {}
    if not ft:
        return {}
    out = {str(lb): "exploratory" for fam in (ecfg.get("families") or {}).values()
           for lb in (fam.get("scenarios") or {})}  # fmt: skip
    out.update({str(lb): "ablation" for lb in ft.get("ablations") or []})
    out.update({str(lb): "main" for lb in ft.get("columns") or []})
    return out


def family_pairs(ecfg: dict, family: str) -> list[dict]:
    """The registered pairs of a family ([{a, b, role}])."""
    fam = (ecfg.get("families") or {}).get(family) or {}
    return [
        {"a": str(p["a"]), "b": str(p["b"]), "role": p.get("role")}
        for p in fam.get("pairs") or []
        if isinstance(p, dict) and "a" in p and "b" in p
    ]


def series_spec(ecfg: dict) -> dict:
    """`part8.series` ({} when absent): step_s, metric, approaches, quantiles."""
    return dict(((ecfg.get("part8") or {}).get("series")) or {})


def evaluation_seeds(ecfg: dict) -> frozenset[int]:
    seeds = (ecfg.get("seeds") or {}).get("evaluation") or []
    try:
        return frozenset(int(s) for s in seeds)
    except (TypeError, ValueError):
        return frozenset()


# -- signals (D105) ------------------------------------------------------------------------
@dataclass(frozen=True)
class SignalMap:
    """What the replay needs to turn TLS links into signal heads: the vehicle links of each
    approach (its in-edges' movements) and each link's yellow time (s) in the program."""

    approach_links: dict[str, tuple[int, ...]]
    yellow_s: dict[int, float]

    @property
    def max_yellow_s(self) -> float:
        return max(self.yellow_s.values(), default=0.0)


def signal_map(site: Site) -> SignalMap:
    links: dict[str, set[int]] = {}
    for (src, _dst), idx in site.movement_links.items():
        a = site.approach_of_edge(src)
        if a is not None:
            links.setdefault(a, set()).update(idx)
    yellow: dict[int, float] = {}
    phases = site.phases
    n = max((len(p.state) for p in phases), default=0)
    for link in range(n):
        best = run = 0.0
        # twice round the program, so a yellow spanning its end is counted whole
        for p in (*phases, *phases):
            if link < len(p.state) and p.state[link] in YELLOW:
                run += p.duration_s
                best = max(best, run)
            else:
                run = 0.0
        if best > 0:
            yellow[link] = min(best, sum(p.duration_s for p in phases))
    approaches = {a: tuple(sorted(v)) for a, v in sorted(links.items())}
    return SignalMap(approach_links=approaches, yellow_s=yellow)


def link_states(sig: pd.DataFrame, t: float, smap: SignalMap) -> dict[int, str]:
    """green / yellow / red of every link with a yellow time or a green interval, at t, from
    the green intervals of signal.parquet: yellow = not green and its last green ended
    less than its program yellow time before t."""
    on = set(sig.loc[(sig["begin_s"] <= t) & (sig["end_s"] > t), "link"].astype(int))
    recent = sig[(sig["end_s"] <= t) & (sig["end_s"] > t - smap.max_yellow_s)]
    last_end = recent.groupby("link")["end_s"].max() if len(recent) else pd.Series(dtype=float)
    out: dict[int, str] = {}
    for link in set(sig["link"].astype(int)) | set(smap.yellow_s):
        if link in on:
            out[link] = "green"
        elif link in last_end.index and t - float(last_end[link]) < smap.yellow_s.get(link, 0.0):
            out[link] = "yellow"
        else:
            out[link] = "red"
    return out


def program_link_states(site: Site, phase: int | None) -> dict[int, str] | None:
    """green / yellow / red of every link in a SUMO phase of the program (the fallback
    without signal.parquet; 5 s state-log resolution). None for an unknown phase."""
    if phase is None or not 0 <= phase < len(site.phases):
        return None
    state = site.phases[phase].state
    return {
        i: "green" if c in GREEN else "yellow" if c in YELLOW else "red"
        for i, c in enumerate(state)
    }


def approach_heads(states: dict[int, str], smap: SignalMap) -> dict[str, str]:
    """One head per approach: green when one of its movements has green, else yellow when
    one has yellow, else red."""
    out = {}
    for a, links in smap.approach_links.items():
        got = {states.get(k, "red") for k in links}
        out[a] = "green" if "green" in got else "yellow" if "yellow" in got else "red"
    return out


# -- file readers --------------------------------------------------------------------------
class Results:
    """Cached readers of the evaluation outputs and the replay extras (module docstring)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple, Any] = OrderedDict()

    # -- low level -----------------------------------------------------------------------
    def _load(self, path: Path, reader: Callable[[Path], Any], tag: str = "") -> Any:
        try:
            st = path.stat()
        except OSError:
            return None
        key = (tag, str(path), st.st_size, st.st_mtime_ns)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        try:
            value = reader(path)
        except Exception as e:  # unreadable: treated as absent, never an error
            log.warning("unreadable %s: %s", path, e)
            value = None
        with self._lock:
            self._cache[key] = value
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return value

    def json(self, path: Path) -> dict | None:
        def read(p: Path):
            value = json.loads(p.read_text(encoding="utf-8"))
            return clean(value) if isinstance(value, dict) else None

        return self._load(path, read, "json")

    def csv(self, path: Path) -> pd.DataFrame | None:
        return self._load(path, pd.read_csv, "csv")

    @staticmethod
    def mtime(path: Path) -> float | None:
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    def eval_path(self, *parts: str) -> Path:
        return self.settings.path(self.settings.evaluation_dir).joinpath(*parts)

    # -- evaluation config ---------------------------------------------------------------
    def evaluation_config(self) -> dict:
        """configs/evaluation.yaml with its `extends` merged ({} when absent or broken)."""
        path = self.settings.path(self.settings.evaluation_config)
        return self._load(path, load_config, "ecfg") or {}

    # -- evaluation outputs --------------------------------------------------------------
    def summary(self) -> dict | None:
        return self.json(self.eval_path(SUMMARY_FILE))

    def a05(self) -> dict | None:
        return self.json(self.eval_path(A05_FILE))

    def a06(self) -> dict | None:
        return self.json(self.eval_path(A06_FILE))

    def provenance(self) -> dict | None:
        return self.json(self.eval_path(PROVENANCE_FILE))

    def final_table(self) -> pd.DataFrame | None:
        return self.csv(self.eval_path(FINAL_TABLE_FILE))

    def series(self) -> pd.DataFrame | None:
        df = self.csv(self.eval_path(SERIES_FILE))
        need = {"family", "label", "metric", "approach", "t_s", "mean"}
        if df is not None and not need <= set(df.columns):
            log.warning("%s lacks %s", self.eval_path(SERIES_FILE), sorted(need - set(df)))
            return None
        return df

    def pair(self, family: str, a: str, b: str) -> pd.DataFrame | None:
        return self.csv(self.eval_path(family, PAIRS_DIR, pair_name(a, b)))

    def sensitivity_path(self) -> Path:
        return self.settings.path(self.settings.sensitivity)

    def sensitivity(self) -> pd.DataFrame | None:
        return self.csv(self.sensitivity_path())

    # -- staleness -----------------------------------------------------------------------
    def _current_hashes(self, config: Path) -> dict[str, str] | None:
        from traffic_twin.config import load_scenario
        from traffic_twin.evaluation.runs import expected_hashes

        if Path.cwd().resolve() != self.settings.repo_root.resolve():
            return None
        return self._load(config, lambda p: expected_hashes(load_scenario(p)), "hashes")

    def stale_labels(self, family: str, labels: dict[str, str]) -> list[str]:
        """The labels (of `labels`: label -> config path) whose evaluated config_hash or
        demand_hash (provenance.json) differs from the current config's."""
        prov = self.provenance()
        members = (((prov or {}).get("families") or {}).get(family) or {}).get("scenarios")
        if not isinstance(members, dict):
            return []
        out = []
        for label, cfg in labels.items():
            rec = members.get(label)
            if not isinstance(rec, dict) or "config_hash" not in rec:
                continue
            now = self._current_hashes(self.settings.path(cfg))
            if now is None:
                continue
            if any(rec.get(k) != now[k] for k in ("config_hash", "demand_hash") if k in rec):
                out.append(label)
        return out

    # -- replay --------------------------------------------------------------------------
    def state_bounds(self, state_log: Path) -> tuple[float, float, float | None] | None:
        """(first t, last t, sampling step) of a run's state_log.parquet, None when absent
        or unreadable. The step is the median gap between distinct sample times."""

        def read(p: Path):
            import pyarrow.compute as pc
            import pyarrow.parquet as pq

            # the distinct sample times only (about 5x faster than through pandas)
            t = pc.unique(pq.read_table(p, columns=["t"]).column("t").drop_null())
            if len(t) == 0:
                return None
            u = pd.Series(sorted(t.to_pylist()), dtype=float)
            step = float(u.diff().median()) if len(u) > 1 else None
            return float(u.iloc[0]), float(u.iloc[-1]), step

        return self._load(state_log, read, "bounds")

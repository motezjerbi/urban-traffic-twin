"""ETA dataset (Part 5, EF-12; D39, D43, D46; contract 4.4): random worlds run under the
three controllers (paired), a passive 1 Hz feature recorder, per-trip targets and a
manifest with provenance.

    python -m traffic_twin.ml.eta_dataset --config configs/eta.yaml --splits train val calib
        [--workers 8] [--worlds 2000 2001 ...] [--controllers fixed ...] [--out data/eta]
        [--allow-dirty]

A world (`sample_world`, D43) is drawn from its seed alone: realistic (a day-type
profile, a lognormal multiplier and per-arm scales) or stress (a uniform multiplier),
0-2 ambulances and 0-2 incidents (lane slowdowns, D45). Each world runs under every
controller of `eta.yaml` (`world_config` deep-copies the frozen Part 3-4 config and
overrides only the D43 keys); SUMO's seed is the world seed. All runs of a world share
its split (`split_of`).

Per run (`run_world`) the `EtaRecorder` samples the junction once per second (the
`JunctionSampler`, then `ml.features.FeatureState`) and polls the out-edges and the
junction's internal lanes after every step. Each car trip of the 6 trajets inserted in
the target window becomes one sample (D39):
- `insert_t`: physical insertion on its approach in-edge (tripinfo depart, s since the
  scenario start); `entry_t` = insert_t - depart_delay_s (the intended entry);
- `stopline_t`: first second on an internal lane of the junction (the stop line crossed;
  the exit time when the car crossed the junction within one step);
- `exit_t`: first second on its out-edge;
- target `y_zone_s` = exit_t - insert_t; diagnostics `y_junction_s` = exit_t - entry_t
  (includes the insertion backlog), `y_stopline_s` = stopline_t - entry_t and
  `y_arrival_s` = arrival - entry_t;
- `teleported`: the car teleported (no exit time or an unreliable one); teleported cars
  and cars without exit time stay in the parts, flagged, and are dropped by the model
  (both are counted in `runs.parquet` and the manifest).
Features are joined at `t_feat = floor(insert_t) - 1` (`ml.features.join_samples`).

Layout of `<out>/<dataset_id>/`:
- `manifest.json`: provenance (ids, sampler and feature schema versions, target,
  seeds, base configs and calibration files with their hashes, code version, library
  versions, builds) and `counts` per split x controller x trajet. Every `builds` entry
  records the `versions` (`library_versions`: python, SUMO, libsumo, traci, sumolib,
  numpy, pandas, pyarrow) of the process that built it; the top-level `versions` are
  those shared by every build that made runs, or null when they differ or a build
  that made runs did not record them (D65);
- `runs.parquet`: one row per run (identity, world parameters, hashes, outcome);
- `samples/split=<s>/controller=<c>/part-<seed>.parquet`: one row per car trip;
- `snapshots/split=<s>/controller=<c>/part-<seed>.parquet`: the 5 s feature rows;
- `runs/split=<s>/controller=<c>/run-<seed>.json`: the run row, written last (a run is
  complete when it exists; a build skips complete runs whose `config_hash` matches).
Read the parts with `read_parts` (pandas' hive partitioning clashes with the `split` and
`controller` columns the files also carry). Raw SUMO outputs are deleted after parsing;
route files and outputs live in `<out>/_work/<pid>/` while a build runs.

Determinism: the same world seed and controller give byte-identical sample and snapshot
parts (the run id is a uuid5 of the run's config hash).
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import logging
import math
import multiprocessing
import os
import platform
import re
import shutil
import sys
import time
import traceback
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_twin.config import (
    ConfigError,
    approach_in_edges,
    approach_out_edges,
    config_hash,
    demand_hash,
    get,
    load_config,
    load_scenario,
    text_sha256,
)
from traffic_twin.control.registry import make_controller
from traffic_twin.ml.features import (
    ARMS,
    CATEGORIES,
    DIAGNOSTIC,
    ETA_FEATURES,
    FEATURE_SCHEMA_VERSION,
    ROW_FEATURES,
    FeatureState,
    eta_site,
    feature_dtypes,
    join_samples,
    rows_frame,
)
from traffic_twin.paths import ETA_CONFIG
from traffic_twin.sim.incidents import Incident, IncidentManager, load_sites, sample_incidents
from traffic_twin.sim.kpi import read_tripinfo
from traffic_twin.sim.results import code_version
from traffic_twin.sim.sampler import JunctionSampler
from traffic_twin.sim.session import SimSession

log = logging.getLogger(__name__)

SAMPLER_VERSION = "eta-sampler-1"
WORLD_STREAM = 5_000_000  # sampler stream: default_rng([seed, WORLD_STREAM]) (D43)
GENERATED_SPLITS = ("train", "val", "calib", "test")  # `reserve` is never generated
RUN_NAMESPACE = uuid.UUID("5f0c3a52-7d1e-4b8e-9a51-e7a0c1d3b2f4")  # uuid5 run ids
CAR_ID = re.compile(r"^car_([A-Z]+)_([A-Z]+)_\d+(?:\.\d+)?$")  # flow car_{arm}_{dest}_{k}.{n}
ROUND = 4  # decimals of the drawn multiplier and arm scales
# distributions whose versions the manifest records (D65); the samples depend on SUMO
VERSION_PACKAGES = {
    "sumo": "eclipse-sumo",
    "libsumo": "libsumo",
    "traci": "traci",
    "sumolib": "sumolib",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
}

SAMPLE_ID_COLUMNS = ("run_id", "world_seed", "controller", "split", "kind", "veh_id", "trajet")
SAMPLE_TIME_COLUMNS = (
    "insert_t", "entry_t", "stopline_t", "exit_t", "depart_delay_s",
    "y_zone_s", "y_junction_s", "y_stopline_s", "y_arrival_s", "teleported",
)  # fmt: skip
# every ETA_FEATURES column except trajet and controller (identity columns), DIAGNOSTIC
SAMPLE_FEATURE_COLUMNS = tuple(f for f in ROW_FEATURES if f != "controller")
SAMPLE_COLUMNS = (
    *SAMPLE_ID_COLUMNS, *SAMPLE_TIME_COLUMNS, "t_feat", *SAMPLE_FEATURE_COLUMNS, *DIAGNOSTIC
)  # fmt: skip
SNAPSHOT_COLUMNS = ("run_id", "t", "t_feat", *ROW_FEATURES, *DIAGNOSTIC)
WORLD_COLUMNS = ("world_seed", "split", "kind")  # added to dataset snapshot parts


# -- worlds ----------------------------------------------------------------------------
@dataclass(frozen=True)
class World:
    """One sampled world (D43)."""

    seed: int
    kind: str  # realistic | stress
    day_type: str
    multiplier: float
    arm_scale: Mapping[str, float]
    ambulances: tuple[dict, ...]  # {depart_s, route}, by departure (emergency.trips form)
    incidents: tuple[Incident, ...]


def split_of(seed: int, eta_cfg: dict) -> str:
    """train / val / calib / test (/ reserve) of a world seed; ValueError outside the
    registered ranges (inclusive bounds)."""
    for name, (lo, hi) in eta_cfg["seeds"].items():
        if int(lo) <= int(seed) <= int(hi):
            return str(name)
    raise ValueError(f"world seed {seed} is in no registered ETA seed range")


def _pick(rng: np.random.Generator, probs: Mapping[str, float], what: str) -> str:
    """A key of `probs` drawn with its probability (keys in mapping order)."""
    keys = list(probs)
    p = np.asarray([float(probs[k]) for k in keys])
    if not keys or (p < 0).any() or abs(p.sum() - 1.0) > 1e-9:
        raise ValueError(f"{what}: probabilities must sum to 1, got {dict(probs)}")
    return keys[int(rng.choice(len(keys), p=p))]


def sample_world(seed: int, eta_cfg: dict) -> World:
    """Pure world draw (D43) from default_rng([seed, 5_000_000]), in a fixed order:
    kind, day type, multiplier, arm scales (N, S, W), ambulances (count, then per
    ambulance its departure and route), incidents (`sim.incidents.sample_incidents`).

    - realistic: day type from `world.realistic.day_types`, multiplier exp(N(0, sd)),
      arm scale N(1, sd) per arm;
    - stress: day type from `world.stress.day_types`, multiplier U(lo, hi), arm scale
      N(1, sd);
    - ambulances: count k with probability `count_p[k]`, departure uniform over the whole
      seconds of [depart_s[0], depart_s[1]), route by `routes` (trajets of eta.yaml);
    - incidents: `world.incidents` (sites from its `sites` file)."""
    w = eta_cfg["world"]
    rng = np.random.default_rng([int(seed), WORLD_STREAM])
    kind = _pick(rng, w["kinds"], "world.kinds")
    spec = w[kind]
    day_type = _pick(rng, spec["day_types"], f"world.{kind}.day_types")
    if kind == "realistic":
        multiplier = math.exp(float(rng.normal(0.0, float(spec["multiplier_lognormal_sd"]))))
    else:
        lo, hi = spec["multiplier"]
        multiplier = float(rng.uniform(float(lo), float(hi)))
    sd = float(spec["arm_scale_sd"])
    arm_scale = {a: round(float(rng.normal(1.0, sd)), ROUND) for a in ARMS}
    amb = w["ambulances"]
    n_amb = int(rng.choice(len(amb["count_p"]), p=np.asarray(amb["count_p"], dtype=float)))
    lo, hi = (int(x) for x in amb["depart_s"])
    trips = []
    for _ in range(n_amb):
        depart = int(rng.integers(lo, hi))
        route = _pick(rng, amb["routes"], "world.ambulances.routes")
        trips.append({"depart_s": depart, "route": list(eta_cfg["trajets"][route])})
    trips.sort(key=lambda tr: (tr["depart_s"], tr["route"]))
    inc = w["incidents"]
    incidents = sample_incidents(
        rng,
        _sites(str(inc["sites"])),
        count_p=inc["count_p"],
        start_s=tuple(inc["start_s"]),
        durations_s=inc["duration_s"],
    )
    return World(
        seed=int(seed),
        kind=kind,
        day_type=day_type,
        multiplier=round(multiplier, ROUND),
        arm_scale=arm_scale,
        ambulances=tuple(trips),
        incidents=incidents,
    )


@functools.lru_cache(maxsize=4)
def _sites(path: str) -> dict:
    return load_sites(path)


def world_config(
    base: dict, world: World, controller: str, eta_cfg: dict, duration_s: int | None = None
) -> dict:
    """Deep copy of the base config with only the D43 overrides; never writes files.

    Overrides `scenario` (eta_<controller>_w<seed>), `seed`, `demand.{day_type,
    calibration, multiplier, arm_scale}`, `emergency` (the world's ambulances as
    `trips`, or disabled) and `outputs.dir` (eta.yaml `outputs.data`). `duration_s`
    (tests, smoke runs) also shortens the demand window and drops the ambulances that
    would depart after it."""
    cfg = copy.deepcopy(base)
    cfg["scenario"] = f"eta_{controller}_w{world.seed}"
    cfg["seed"] = int(world.seed)
    if duration_s is not None:
        cfg["duration_s"] = int(duration_s)
    demand = cfg["demand"]
    demand["day_type"] = world.day_type
    demand["calibration"] = str(eta_cfg["calibration_files"][world.day_type])
    demand["multiplier"] = float(world.multiplier)
    demand["arm_scale"] = {a: float(world.arm_scale[a]) for a in ARMS}
    end = cfg["duration_s"]
    trips = [copy.deepcopy(dict(t)) for t in world.ambulances if t["depart_s"] < end]
    if trips:
        cfg["emergency"] = {"enabled": True, "depart_s": None, "route": None, "trips": trips}
    else:
        cfg["emergency"] = {"enabled": False, "depart_s": None, "route": None}
    cfg["outputs"] = {**cfg.get("outputs", {}), "dir": str(eta_cfg["outputs"]["data"])}
    return cfg


def target_window(eta_cfg: dict, cfg: dict) -> tuple[float, float]:
    """[lo, hi) of insert_t for samples: the target window, cut at the demand window."""
    lo, hi = (float(x) for x in eta_cfg["target"]["window_s"])
    return lo, min(hi, float(cfg["duration_s"]))


# -- recorder ----------------------------------------------------------------------------
class EtaRecorder:
    """Passive RunObserver: 1 Hz feature rows from JunctionSampler, exit and stop-line
    polling; `finish` writes <out_dir>/eta_snapshots.parquet (rows with
    (t_feat + 1) % persist_every_s == 0, column t = t_feat + 1).

    - `persist_every_s`: period of the persisted rows (the serving grid, 5 s);
    - `record_samples`: keep every 1 Hz row (for the trip join) and poll the out-edges,
      the internal lanes and the teleports after every step; off for serving runs, which
      only need the persisted rows;
    - `until_s`: no feature rows from this simulation time on (None: the whole run).

    It samples once at `bind` (t = 0) and after every step except the run's last one,
    i.e. where the controller observes anyway: the session caches the observation per
    step, so the run is unchanged (`tests/test_eta_recorder_passive.py`)."""

    def __init__(
        self, persist_every_s: int = 5, record_samples: bool = True, until_s: float | None = None
    ) -> None:
        if int(persist_every_s) < 1:
            raise ValueError(f"persist_every_s must be >= 1, got {persist_every_s}")
        self.persist_every_s = int(persist_every_s)
        self.record_samples = bool(record_samples)
        self.until_s = until_s
        self.rows: list[dict] = []  # every 1 Hz row (record_samples only)
        self.persisted: list[dict] = []  # rows on the persist grid
        self.exit_t: dict[str, float] = {}  # vid -> first second on an out-edge
        self.stopline_t: dict[str, float] = {}  # vid -> first second on an internal lane
        self.teleported: set[str] = set()
        self.departed: set[str] = set()  # every vehicle inserted (unfinished-trip count)
        self.sampler: JunctionSampler | None = None
        self.state: FeatureState | None = None

    def bind(self, sim: SimSession, incidents: IncidentManager | None = None) -> None:
        """After the session started (and the controller was bound and reset), before
        the first step. `incidents`: the run's IncidentManager (`run_world`), whose
        active incidents the features report."""
        controller = str(sim.cfg.get("controller"))
        self.sampler = JunctionSampler(sim, incidents)
        self.state = FeatureState(sim.cfg, controller, eta_site(sim, controller))
        outs = approach_out_edges(sim.cfg)
        self._out_edges = tuple(e for a in ARMS for e in outs.get(a, ()))
        ins = {e for a in ARMS for e in approach_in_edges(sim.cfg).get(a, ())}
        via: list[str] = []
        with sim._alive():
            for conns in sim.traci.trafficlight.getControlledLinks(sim.tls):
                for from_lane, _to, lane in conns:
                    if from_lane.rpartition("_")[0] in ins and lane and lane not in via:
                        via.append(lane)
        self._via = tuple(via)
        self._sample(sim)

    def _sample(self, sim: SimSession) -> None:
        if self.until_s is not None and sim.t >= self.until_s:
            return
        row = self.state.update(self.sampler.sample())
        if self.record_samples:
            self.rows.append(row)
        if (round(row["t_feat"]) + 1) % self.persist_every_s == 0:
            self.persisted.append(row)

    def _poll(self, sim: SimSession) -> None:
        t, tr = float(sim.t), sim.traci
        with sim._alive():
            for e in self._out_edges:
                for v in tr.edge.getLastStepVehicleIDs(e):
                    self.exit_t.setdefault(v, t)
            for lane in self._via:
                for v in tr.lane.getLastStepVehicleIDs(lane):
                    self.stopline_t.setdefault(v, t)
            self.teleported.update(tr.simulation.getStartingTeleportIDList())
            self.departed.update(tr.simulation.getDepartedIDList())

    def after_step(self, sim: SimSession) -> None:
        if self.record_samples:
            self._poll(sim)
        if self.until_s is not None and sim.t >= self.until_s:
            return
        if sim.done:  # no observation follows the run's last step
            return
        self._sample(sim)

    def snapshot_frame(self, run_id: str) -> pd.DataFrame:
        """The persisted rows: run_id, t (= t_feat + 1), t_feat, features, DIAGNOSTIC."""
        df = rows_frame(self.persisted)
        df.insert(0, "t", df["t_feat"] + 1.0)
        df.insert(0, "run_id", str(run_id))
        return df[list(SNAPSHOT_COLUMNS)]

    def finish(self, sim: SimSession, out_dir: Path, run_id: str) -> None:
        _write_parquet(self.snapshot_frame(run_id), Path(out_dir) / "eta_snapshots.parquet")


# -- one run ---------------------------------------------------------------------------
def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write atomically (temporary file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def part_path(dataset_dir: Path, what: str, split: str, controller: str, seed: int) -> Path:
    """`<dataset>/<what>/split=<s>/controller=<c>/part-<seed>.parquet` (what: samples or
    snapshots), or the run row `<dataset>/runs/split=<s>/controller=<c>/run-<seed>.json`."""
    folder = Path(dataset_dir) / what / f"split={split}" / f"controller={controller}"
    return folder / (f"run-{seed}.json" if what == "runs" else f"part-{seed}.parquet")


def run_uuid(cfg_hash: str) -> str:
    """Deterministic run id (uuid5 of the run's config hash)."""
    return str(uuid.uuid5(RUN_NAMESPACE, f"{SAMPLER_VERSION}/{cfg_hash}"))


@functools.lru_cache(maxsize=8)
def _base_config(path: str) -> dict:
    return load_scenario(path)


def base_config(eta_cfg: dict, controller: str) -> dict:
    """The frozen Part 3-4 config of a controller (`base_configs`; do not mutate)."""
    try:
        path = eta_cfg["base_configs"][controller]
    except KeyError:
        raise ConfigError(f"eta.yaml base_configs has no controller {controller!r}") from None
    return _base_config(str(path))


def trip_samples(
    trips: pd.DataFrame,
    recorder: EtaRecorder,
    begin_s: float,
    window: tuple[float, float],
    trajets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Car samples (D39) of a run's tripinfo trips: the 6 trajets, insert_t in `window`,
    with clocks and targets (no identity or feature columns yet), by (insert_t, veh_id)."""
    cars = trips[trips["type"] == "car"]
    m = cars["id"].str.extract(CAR_ID)
    trajet = m[0] + "_" + m[1]
    cars = cars.assign(trajet=trajet)[trajet.isin(list(trajets))]
    expected = cars["trajet"].map({j: str(e[0]) for j, e in trajets.items()})
    bad = cars["depart_edge"].notna() & (cars["depart_edge"] != expected)
    if bad.any():
        raise RuntimeError(f"trips not starting on their trajet's in-edge: {list(cars.id[bad])}")
    insert_t = cars["depart_s"] - float(begin_s)
    cars = cars[(insert_t >= window[0]) & (insert_t < window[1])]
    out = pd.DataFrame({"veh_id": cars["id"].astype(str), "trajet": cars["trajet"].astype(str)})
    out["insert_t"] = cars["depart_s"].astype("float64") - float(begin_s)
    out["depart_delay_s"] = cars["depart_delay_s"].astype("float64")
    out["entry_t"] = out["insert_t"] - out["depart_delay_s"]
    out["exit_t"] = out["veh_id"].map(recorder.exit_t).astype("float64")
    stop = out["veh_id"].map(recorder.stopline_t).astype("float64")
    out["stopline_t"] = stop.fillna(out["exit_t"])
    out["y_zone_s"] = out["exit_t"] - out["insert_t"]
    out["y_junction_s"] = out["exit_t"] - out["entry_t"]
    out["y_stopline_s"] = out["stopline_t"] - out["entry_t"]
    out["y_arrival_s"] = cars["arrival_s"].astype("float64") - float(begin_s) - out["entry_t"]
    out["teleported"] = out["veh_id"].isin(recorder.teleported)
    return out.sort_values(["insert_t", "veh_id"], kind="stable").reset_index(drop=True)


def _unfinished(departed: set[str], trips: pd.DataFrame, trajets: Mapping) -> int:
    finished = set(trips["id"])
    n = 0
    for v in departed - finished:
        m = CAR_ID.match(v)
        n += bool(m and f"{m[1]}_{m[2]}" in trajets)
    return n


def run_world(
    seed: int,
    controller: str,
    eta_cfg: dict,
    dataset_dir: Path,
    work_dir: Path,
    duration_s: int | None = None,
    code: str | None = None,
) -> dict:
    """Simulate one (world, controller) with its own loop (incidents.update(t) -> step ->
    recorder.after_step) and write its sample part; returns the runs.parquet row.

    SUMO's outputs go to `work_dir/out` and the route files to `work_dir/demand`; the raw
    outputs are deleted after parsing. Writes the sample and snapshot parts, then the run
    row (`part_path`). `duration_s` shortens the run (tests), `code` is the code version
    to record (default: `sim.results.code_version()`)."""
    split = split_of(seed, eta_cfg)
    world = sample_world(seed, eta_cfg)
    cfg = world_config(base_config(eta_cfg, controller), world, controller, eta_cfg, duration_s)
    cfg_hash = config_hash(cfg)
    run_id = run_uuid(cfg_hash)
    work_dir = Path(work_dir)
    window = target_window(eta_cfg, cfg)
    until = window[1] + float(eta_cfg["snapshots"]["until_s_after_window"])
    persist = int(eta_cfg["snapshots"]["persist_every_s"])
    incidents = [i for i in world.incidents if i.start_s < float(cfg["duration_s"])]

    ctrl = make_controller(cfg)
    sim = SimSession(
        cfg, int(seed), work_dir / "out", log_state=False, demand_dir=work_dir / "demand"
    )
    recorder = EtaRecorder(persist, record_samples=True, until_s=until)
    wall = time.perf_counter()
    try:
        with sim:
            ctrl.bind(sim)
            ctrl.reset(int(seed))
            manager = IncidentManager(sim.traci, incidents)
            with sim._alive():
                manager.update(sim.t)
            recorder.bind(sim, incidents=manager)
            while not sim.done:
                sim.step(ctrl.decide(sim.observe()))
                with sim._alive():
                    manager.update(sim.t)  # incidents of the next step, before sampling it
                recorder.after_step(sim)
        wall = time.perf_counter() - wall
        trips, _ = read_tripinfo(sim.tripinfo)
    finally:
        for p in sim.raw_outputs:
            p.unlink(missing_ok=True)

    samples = trip_samples(trips, recorder, sim.begin, window, eta_cfg["trajets"])
    join = eta_cfg.get("join") or {}
    samples = join_samples(
        samples,
        rows_frame(recorder.rows),
        lag_s=int(join.get("lag_s", 1)),
        tol_s=float(join.get("tolerance_s", 5.0)),
    )
    ident = {
        "run_id": run_id,
        "world_seed": int(seed),
        "controller": controller,
        "split": split,
        "kind": world.kind,
    }
    for k, v in ident.items():
        samples[k] = v
    samples["world_seed"] = samples["world_seed"].astype("int64")
    samples = samples[list(SAMPLE_COLUMNS)]
    snaps = recorder.snapshot_frame(run_id)
    for k in WORLD_COLUMNS:
        snaps[k] = ident[k] if k != "world_seed" else int(seed)
    snaps = snaps[["run_id", *WORLD_COLUMNS, *SNAPSHOT_COLUMNS[1:]]]

    dataset_dir = Path(dataset_dir)
    _write_parquet(samples, part_path(dataset_dir, "samples", split, controller, seed))
    _write_parquet(snaps, part_path(dataset_dir, "snapshots", split, controller, seed))
    incident_log = [
        {**asdict(inc), **{k: v for k, v in entry.items() if k not in ("site", "lane")}}
        for inc, entry in zip(manager.incidents, manager.log(), strict=True)
    ]
    row = {
        "run_id": run_id,
        "world_seed": int(seed),
        "sumo_seed": int(seed),
        "controller": controller,
        "scenario": cfg["scenario"],
        "split": split,
        "kind": world.kind,
        "day_type": world.day_type,
        "multiplier": float(world.multiplier),
        **{f"arm_scale_{a}": float(world.arm_scale[a]) for a in ARMS},
        "ambulances": json.dumps(cfg["emergency"].get("trips") or []),
        "incidents": json.dumps(incident_log),
        "duration_s": int(cfg["duration_s"]),
        "config_hash": cfg_hash,
        "demand_hash": demand_hash(cfg),
        "code_version": code if code is not None else code_version(),
        "sampler_version": SAMPLER_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "wall_s": round(wall, 3),
        "drain_timeout": bool(sim.drain_timeout),
        "teleports": len(recorder.teleported),
        "n_samples": len(samples),
        "n_missing_exit": int(samples["exit_t"].isna().sum()),
        "n_teleported_samples": int(samples["teleported"].sum()),
        "n_missing_features": int(samples["t_feat"].isna().sum()),
        # inserted trajet cars without a tripinfo (still in the network when a drain timed
        # out, any insertion time): absent from the samples, so a censoring diagnostic
        "n_unfinished_cars": _unfinished(recorder.departed, trips, eta_cfg["trajets"]),
        "n_snapshots": len(snaps),
    }
    _write_json(row, part_path(dataset_dir, "runs", split, controller, seed))
    return row


# -- dataset ---------------------------------------------------------------------------
def library_versions() -> dict[str, str | None]:
    """Python and the installed versions of `VERSION_PACKAGES` (None if absent), read
    from the package metadata (no import: libsumo is not loaded for it)."""
    out: dict[str, str | None] = {"python": platform.python_version()}
    for key, dist in VERSION_PACKAGES.items():
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = None
    return out


def shared_versions(builds: Sequence[Mapping]) -> dict | None:
    """The `versions` common to every build that made runs (`runs_made` > 0; the last
    build when none did), or None when they differ or one did not record them."""
    makers = [b for b in builds if int(b.get("runs_made") or 0) > 0] or list(builds[-1:])
    found = [b.get("versions") for b in makers]
    if not found or any(not v for v in found) or any(v != found[0] for v in found[1:]):
        return None
    return dict(found[0])


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def dataset_id(eta_cfg: dict) -> str:
    """`eta1-` + sha8(canonical eta.yaml without outputs/workers/changes || demand_hash of
    the three base configs || sha256 of the three calibration files) (D46)."""
    payload = {
        "eta": {k: v for k, v in eta_cfg.items() if k not in ("outputs", "workers", "changes")},
        "demand_hash": {c: demand_hash(base_config(eta_cfg, c)) for c in eta_cfg["controllers"]},
        "calibration_sha256": {
            d: text_sha256(p) for d, p in sorted(eta_cfg["calibration_files"].items())
        },
    }
    return "eta1-" + hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:8]


def read_parts(
    dataset_dir: Path,
    what: str = "samples",
    splits: Iterable[str] | None = None,
    controllers: Iterable[str] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Concatenated parts (`samples` or `snapshots`) of the given splits and controllers
    (default: all), in (split, controller, seed) order. An empty frame when none exist."""
    root = Path(dataset_dir) / what
    files = []
    for split_dir in sorted(root.glob("split=*")):
        split = split_dir.name.partition("=")[2]
        if splits is not None and split not in set(splits):
            continue
        for ctrl_dir in sorted(split_dir.glob("controller=*")):
            if controllers is not None and ctrl_dir.name.partition("=")[2] not in set(controllers):
                continue
            parts = ctrl_dir.glob("part-*.parquet")
            files += sorted(parts, key=lambda p: int(p.stem.partition("-")[2]))
    if not files:
        return pd.DataFrame(columns=list(columns) if columns else None)
    cols = list(columns) if columns else None
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def read_runs(dataset_dir: Path) -> pd.DataFrame:
    """Every completed run row (`runs/.../run-<seed>.json`), by (split, controller, seed)."""
    rows = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(Path(dataset_dir).glob("runs/split=*/controller=*/run-*.json"))
    ]
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["split", "controller", "world_seed"], kind="stable")
    return df.reset_index(drop=True)


def _run_complete(dataset_dir: Path, split: str, controller: str, seed: int, cfg_hash: str) -> bool:
    marker = part_path(dataset_dir, "runs", split, controller, seed)
    if not marker.is_file():
        return False
    try:
        row = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        row.get("config_hash") == cfg_hash
        and row.get("sampler_version") == SAMPLER_VERSION
        and row.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
        and part_path(dataset_dir, "samples", split, controller, seed).is_file()
        and part_path(dataset_dir, "snapshots", split, controller, seed).is_file()
    )


def plan_worlds(
    eta_cfg: dict, splits: Sequence[str], worlds: Sequence[int] | None = None
) -> list[int]:
    """World seeds to generate: every seed of the requested splits, or the given ones,
    each of which must lie in a requested split. `reserve` cannot be generated, and the
    test split only when it is requested explicitly (D50)."""
    splits = list(dict.fromkeys(splits))
    bad = [s for s in splits if s not in GENERATED_SPLITS]
    if bad:
        raise ValueError(f"cannot generate split(s) {bad}; choose from {GENERATED_SPLITS}")
    if worlds is None:
        out = []
        for s in splits:
            lo, hi = eta_cfg["seeds"][s]
            out += range(int(lo), int(hi) + 1)
        return sorted(out)
    out = []
    for w in worlds:
        split = split_of(int(w), eta_cfg)  # ValueError outside every range
        if split not in splits:
            raise ValueError(
                f"world {w} is in split {split!r}, not in the requested --splits {splits}"
            )
        out.append(int(w))
    return sorted(set(out))


def _task(args: tuple) -> dict:
    """Pool worker: one run, errors returned (not raised) with their traceback."""
    seed, controller, eta_cfg, dataset_dir, work_root, code = args
    work = Path(work_root) / str(os.getpid())
    try:
        return run_world(seed, controller, eta_cfg, dataset_dir, work, code=code)
    except Exception:  # noqa: BLE001 - reported per run, the build continues
        return {"error": traceback.format_exc(), "world_seed": seed, "controller": controller}


def counts(dataset_dir: Path) -> tuple[dict, dict]:
    """(counts[split][controller][trajet] = sample rows, excluded[split][controller] =
    {teleported, missing_exit}) over the sample parts on disk."""
    df = read_parts(
        dataset_dir, "samples", columns=["split", "controller", "trajet", "exit_t", "teleported"]
    )
    out: dict = {}
    excluded: dict = {}
    if not len(df):
        return out, excluded
    for (s, c, j), n in df.groupby(["split", "controller", "trajet"]).size().items():
        out.setdefault(s, {}).setdefault(c, {})[j] = int(n)
    for (s, c), g in df.groupby(["split", "controller"]):
        excluded.setdefault(s, {})[c] = {
            "teleported": int(g["teleported"].sum()),
            "missing_exit": int(g["exit_t"].isna().sum()),
        }
    return out, excluded


def write_manifest(
    dataset_dir: Path, ds_id: str, eta_cfg: dict, config_path: Path | None, build: dict
) -> dict:
    """(Re)write manifest.json and runs.parquet from the parts on disk; earlier builds
    are kept in `builds` and the first `created_at` is preserved."""
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / "manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    runs = read_runs(dataset_dir)
    if len(runs):
        _write_parquet(runs, dataset_dir / "runs.parquet")
    n, excluded = counts(dataset_dir)
    splits = (
        {s: sorted({int(x) for x in g["world_seed"]}) for s, g in runs.groupby("split")}
        if len(runs)
        else {}
    )
    builds = [*old.get("builds", []), build]
    versions = shared_versions(builds)
    if versions is None:
        log.warning(
            "dataset %s: the builds that made runs do not share recorded library versions; "
            "manifest versions = null (see builds[].versions)",
            ds_id,
        )
    bases = {}
    for c in eta_cfg["controllers"]:
        cfg = base_config(eta_cfg, c)
        bases[c] = {
            "path": str(eta_cfg["base_configs"][c]),
            "config_hash": config_hash(cfg),
            "demand_hash": demand_hash(cfg),
        }
    manifest = {
        "dataset_id": ds_id,
        "sampler_version": SAMPLER_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": {
            "names": list(ETA_FEATURES),
            "dtypes": feature_dtypes(),
            "categories": {k: list(v) for k, v in CATEGORIES.items()},
            "diagnostic": list(DIAGNOSTIC),
            "excluded_from_model": list(get(eta_cfg, "model.excluded_features", DIAGNOSTIC)),
            "join": dict(eta_cfg.get("join") or {}),
        },
        "target": {
            **dict(eta_cfg["target"]),
            "definition": "exit_t - insert_t: physical insertion on the approach in-edge "
            "to the first second on the out-edge (whole seconds, D39)",
            "trajets": {j: list(e) for j, e in eta_cfg["trajets"].items()},
            "diagnostics": ["y_junction_s", "y_stopline_s", "y_arrival_s", "depart_delay_s"],
            "dropped_from_fit_and_test": ["teleported", "exit_t missing"],
        },
        "seeds": {k: list(v) for k, v in eta_cfg["seeds"].items()},
        "splits": splits,
        "controllers": list(eta_cfg["controllers"]),
        "eta_config": {
            "path": str(config_path) if config_path else None,
            "sha256": text_sha256(config_path) if config_path else None,
            "changes": list(eta_cfg.get("changes") or []),
        },
        "base_configs": bases,
        "calibration_files": {
            d: {"path": str(p), "sha256": text_sha256(p)}
            for d, p in eta_cfg["calibration_files"].items()
        },
        "incident_sites": {
            "path": str(eta_cfg["world"]["incidents"]["sites"]),
            "sha256": text_sha256(eta_cfg["world"]["incidents"]["sites"]),
        },
        "code_version": build["code_version"],
        "versions": versions,
        "argv": build["argv"],
        "created_at": old.get("created_at", build["started_at"]),
        "updated_at": build["finished_at"],
        "builds": builds,
        "n_runs": len(runs),
        "n_samples": int(sum(v for c in n.values() for t in c.values() for v in t.values())),
        "counts": n,
        "excluded": excluded,
    }
    _write_json(manifest, path)
    return manifest


def build(
    eta_cfg: dict,
    splits: Sequence[str],
    out: Path,
    worlds: Sequence[int] | None = None,
    controllers: Sequence[str] | None = None,
    workers: int = 1,
    allow_dirty: bool = False,
    config_path: Path | None = None,
    argv: Sequence[str] | None = None,
) -> tuple[Path, list[dict], list[dict]]:
    """Generate (or complete) the dataset: every (world, controller) of the requested
    splits whose run is not complete yet, `workers` processes (spawn). Returns (dataset
    dir, run rows made now, errors). Refuses a dirty code version unless `allow_dirty`
    (the id then ends in `-dev`)."""
    started = datetime.now(UTC).isoformat()
    code = code_version()
    versions = library_versions()
    dirty = code.endswith("-dirty") or code == "unknown"
    if dirty and not allow_dirty:
        raise ConfigError(
            f"code version {code}: commit first, or pass --allow-dirty (dataset id '-dev')"
        )
    ds_id = dataset_id(eta_cfg) + ("-dev" if dirty else "")
    seeds = plan_worlds(eta_cfg, splits, worlds)
    ctrls = list(controllers or eta_cfg["controllers"])
    unknown = [c for c in ctrls if c not in eta_cfg["controllers"]]
    if unknown:
        raise ConfigError(f"controllers {unknown} are not in eta.yaml controllers")
    dataset_dir = Path(out) / ds_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(out) / "_work"
    tasks, skipped = [], 0
    for seed in seeds:
        world = sample_world(seed, eta_cfg)
        split = split_of(seed, eta_cfg)
        for c in ctrls:
            cfg_hash = config_hash(world_config(base_config(eta_cfg, c), world, c, eta_cfg))
            if _run_complete(dataset_dir, split, c, seed, cfg_hash):
                skipped += 1
                continue
            tasks.append((seed, c, eta_cfg, str(dataset_dir), str(work_root), code))
    log.info(
        "dataset %s: %d worlds x %d controllers, %d runs to make, %d complete",
        ds_id, len(seeds), len(ctrls), len(tasks), skipped,
    )  # fmt: skip
    done, errors = [], []
    t0 = time.perf_counter()

    def report(res: dict) -> None:
        k = len(done) + len(errors)
        if "error" in res:
            errors.append(res)
            log.error("world %s %s FAILED:\n%s", res["world_seed"], res["controller"], res["error"])
            return
        done.append(res)
        log.info(
            "[%d/%d] world %d %s (%s, %s): %d samples, missing exit %d, teleports %d, %.1f s",
            k + 1, len(tasks), res["world_seed"], res["controller"], res["split"], res["kind"],
            res["n_samples"], res["n_missing_exit"], res["teleports"], res["wall_s"],
        )  # fmt: skip

    try:
        if workers <= 1 or len(tasks) <= 1:
            for task in tasks:
                report(_task(task))
        else:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=min(workers, len(tasks))) as pool:
                for res in pool.imap_unordered(_task, tasks):
                    report(res)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        build_info = {
            "argv": list(argv) if argv is not None else None,
            "code_version": code,
            "versions": versions,
            "started_at": started,
            "finished_at": datetime.now(UTC).isoformat(),
            "splits": list(splits),
            "worlds": [seeds[0], seeds[-1]] if seeds else [],
            "controllers": ctrls,
            "workers": int(workers),
            "runs_made": len(done),
            "runs_skipped": skipped,
            "errors": len(errors),
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        write_manifest(dataset_dir, ds_id, eta_cfg, config_path, build_info)
    return dataset_dir, done, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=ETA_CONFIG)
    parser.add_argument("--splits", nargs="+", required=True, choices=GENERATED_SPLITS)
    parser.add_argument("--workers", type=int, default=None, help="default: eta.yaml workers")
    parser.add_argument("--worlds", type=int, nargs="+", default=None)
    parser.add_argument("--controllers", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=None, help="default: eta.yaml outputs.data")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        eta_cfg = load_config(args.config, ("seeds", "world", "base_configs", "trajets"))
        out = args.out or Path(eta_cfg["outputs"]["data"])
        workers = args.workers if args.workers is not None else int(eta_cfg.get("workers", 1))
        dataset_dir, done, errors = build(
            eta_cfg,
            args.splits,
            out,
            worlds=args.worlds,
            controllers=args.controllers,
            workers=workers,
            allow_dirty=args.allow_dirty,
            config_path=args.config,
            argv=sys.argv if argv is None else ["eta_dataset", *argv],
        )
    except (ConfigError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    print(
        f"{dataset_dir}: {manifest['n_runs']} runs, {manifest['n_samples']} samples "
        f"({len(done)} runs made now, {len(errors)} errors)"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

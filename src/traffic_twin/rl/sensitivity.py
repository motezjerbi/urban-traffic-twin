"""Part 7 reward-weight sensitivity (spec 10.3, equity; D76; contract 4.2).

    python -m traffic_twin.rl.sensitivity [--config configs/rl7.yaml] run [--workers N] [--force]
    python -m traffic_twin.rl.sensitivity [--config configs/rl7.yaml] tables
        [--operator-csv docs/figures/rl7/sensitivity.csv]

- `run`: for every sensitivity setting's picks (D72, under the setting's own weights) and
  for S0, S1 and S2, full 2 h `run_scenario.run` KPI v3 runs with drain on the
  sensitivity seeds (rl7.yaml `seeds.sensitivity`, 3160-3179): an in-memory pitie_rl
  scenario with `rl.model` = the pick and `rl.config` = the setting's file, demand files in
  a temporary folder, and a passive `rl.reward.RewardObserver` (per-class totals and
  penalty-seconds) -> `data/rl7/sens/<variant>_s<seed>/...` (spawn pool; refuses
  data/results);
- `tables` -> `data/rl7/sens/{kpis.csv, bootstrap.csv, equity.csv, reweight.csv,
  convergence.csv}` and the operator table `docs/figures/rl7/sensitivity.csv` (spec 12):
  per-class KPIs in % vs S0, S1 and the base, a hierarchical bootstrap CI vs the base
  (training seed, then evaluation seed; rl7.yaml `protocol.sensitivity.bootstrap`), the
  Part 4 equity flags vs S0 (`evaluation.compare.worse_than`, 5 %), the no-retraining
  reweighting table and the convergence check.

Ambulance travel time is reported, not interpreted; curb max is reported as capped at 88 s.

Layout under `sens/` (rl7.yaml `outputs.study`/sens, or `--out`):

- `plan.json`: the picks (setting, training seed, run folder, pick step and cost, model
  sha256, run config), the evaluation seeds and each setting's reward block;
- `<setting>_s<seed>/pitie_rl_seed<n>/` and `baselines/pitie_{fixed,actuated,rules}_seed<n>/`:
  the run folders (run.json, kpi_run.csv, trip.parquet, ..., `reward_totals.json`);
- `kpi_runs.csv` (one row per run and metric), `safety.csv`, `totals.csv` and the tables.

Metrics are named `<user>/<kpi>` for the run KPIs (KPI v3 plus `all/person_delay_mean`,
D29), `reward/<total>` for the reward-observer totals (seconds; `reward/cost_own` is the
cost under the setting's own weights) and `safety/<key>` for the guard interventions per
run, per key and split cut / replaced / cut_foreseeable (D77, from the RL controller's
report when it gives one).

Dev runs (never a reported number): `run --runs data/rl7/_dev --steps 20000 --settings t1
--train-seeds 9 --seeds 5000 5001 --out data/rl7/_dev/sens` scores the picks of other
run folders on Part 7 training seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_twin.config import ConfigError, load_config, load_scenario, text_sha256
from traffic_twin.paths import (
    EVALUATION_CONFIG,
    PITIE_ACTUATED_CONFIG,
    PITIE_FIXED_CONFIG,
    PITIE_RULES_CONFIG,
    RL7_CONFIG,
)

log = logging.getLogger(__name__)

SENS_DIR = "sens"  # under rl7.yaml outputs.study (data/rl7/)
TABLES = ("kpis", "bootstrap", "equity", "reweight", "convergence")  # sens/<name>.csv
OPERATOR_CSV = "sensitivity.csv"  # under rl7.yaml outputs.figures (docs/figures/rl7/)
BASELINES = ("S0", "S1", "S2")
BASELINE_CONFIGS = {"S0": PITIE_FIXED_CONFIG, "S1": PITIE_ACTUATED_CONFIG, "S2": PITIE_RULES_CONFIG}
BASELINES_DIR = "baselines"  # sens/baselines/<scenario>_seed<n>/
PLAN_FILE = "plan.json"
REWARD_FILE = "reward_totals.json"  # written by rl.reward.RewardObserver
RUNS_CSV, SAFETY_CSV, TOTALS_CSV = "kpi_runs.csv", "safety.csv", "totals.csv"
PICK_FILE = "pick_model.zip"  # D72, in each training run folder
WEIGHT_KEYS = ("car", "bus", "pedestrian", "ambulance")
TOTAL_KEYS = ("car_s", "car_pending_s", "bus_s", "pedestrian_s", "ambulance_s", "penalty_s")
SPLIT_KEYS = ("cut", "replaced", "cut_foreseeable")  # D77
# metrics of the per-class table, the bootstrap and the operator table (D76); the
# per-crossing curb means, per-arm car and per-flow bus delays are added when present
REPORT_METRICS = (
    "car/time_loss_mean",
    "car/delay_total_mean",
    "car/depart_delay_mean",
    "bus/delay_mean",
    "bus/headway_std",
    "pedestrian/ped_curb_wait_mean",
    "pedestrian/ped_curb_wait_p95",
    "pedestrian/ped_curb_wait_max",
    "reward/penalty_s",
    "all/person_delay_mean",
    "safety/interventions",
    "safety/iv_ped_wait",
    "safety/iv_max_green",
    "safety/cut",
    "safety/replaced",
    "safety/cut_foreseeable",
    "ambulance/travel_time_total",
)
SUBGROUP_METRICS = (
    "pedestrian/ped_curb_wait_mean__crossing_",
    "car/delay_total_mean__arm_",
    "bus/delay_mean__flow_",
)
NOT_INTERPRETED = {  # D76: reported with a note, never read as an effect
    "ambulance/travel_time_total": "one ambulance per run, per-run SD 30-56 s",
    "pedestrian/ped_curb_wait_max": "capped at 88 s by the pedestrian shield",
}
PARETO_METRICS = ("car/time_loss_mean", "bus/delay_mean", "pedestrian/ped_curb_wait_mean")
DIRECTIONS = {
    "base": "base weights",
    "transit_ped": "more bus / pedestrian weight",
    "car": "more car weight",
    "mixed": "mixed change",
}


# -- config ----------------------------------------------------------------------------
def outputs(rl7_cfg: Mapping) -> dict:
    return dict(rl7_cfg.get("outputs") or {})


def sens_dir(rl7_cfg: Mapping, out: Path | None = None) -> Path:
    return Path(out) if out else Path(outputs(rl7_cfg).get("study") or "data/rl7") / SENS_DIR


def protocol(rl7_cfg: Mapping) -> dict:
    p = (rl7_cfg.get("protocol") or {}).get("sensitivity")
    if not isinstance(p, Mapping):
        raise ConfigError("rl7.yaml: protocol.sensitivity is missing")
    return dict(p)


def settings(rl7_cfg: Mapping) -> list[str]:
    """The base setting, then the pre-registered settings (D76)."""
    p = protocol(rl7_cfg)
    return list(dict.fromkeys([str(p["base"]), *[str(s) for s in p.get("settings", [])]]))


def setting_seeds(rl7_cfg: Mapping, setting: str) -> list[int]:
    """Training seeds of a setting: `seeds_wide` for the `wide` settings (the bus axis),
    `seeds_default` otherwise."""
    p = protocol(rl7_cfg)
    key = "seeds_wide" if setting in set(p.get("wide", [])) else "seeds_default"
    return [int(s) for s in p[key]]


def setting_config(rl7: Path, setting: str) -> Path:
    """The run config of a setting: `configs/rl7/<setting>.yaml` next to rl7.yaml."""
    return Path(rl7).parent / "rl7" / f"{setting}.yaml"


def reward_block(config: Path) -> dict:
    return dict(load_config(config)["reward"])


def weights_row(reward: Mapping) -> dict[str, float]:
    """w_car, w_bus, w_pedestrian, w_ambulance and lambda of a reward block."""
    w = reward.get("weights") or {}
    lam = (reward.get("ped_penalty") or {}).get("lambda", 0.0)
    return {**{f"w_{k}": float(w.get(k, math.nan)) for k in WEIGHT_KEYS}, "lambda": float(lam)}


def direction(reward: Mapping, base: Mapping) -> str:
    """How a setting moves the weights against the base: `transit_ped` (bus, pedestrian
    weight or lambda up, none down), `car` (down only), `base` (unchanged) or `mixed`."""
    a, b = weights_row(reward), weights_row(base)
    keys = ("w_bus", "w_pedestrian", "lambda")
    up = any(a[k] > b[k] for k in keys)
    down = any(a[k] < b[k] for k in keys)
    if up and not down:
        return "transit_ped"
    if down and not up:
        return "car"
    return "mixed" if up else "base"


def eval_seeds(rl7_cfg: Mapping, seeds: Sequence[int] | None = None) -> list[int]:
    """The evaluation seeds: rl7.yaml `seeds.sensitivity`, or `seeds` (dev runs: Part 7
    training seeds are also allowed). Part 3-4 seeds and every other range are refused."""
    from traffic_twin.rl.evaluate import check_seeds
    from traffic_twin.rl.train import seed_range

    if seeds:
        out = [int(s) for s in seeds]
        check_seeds(out, rl7_cfg, ("sensitivity", "train"), "sensitivity")
        return out
    out = seed_range(rl7_cfg["seeds"]["sensitivity"])
    check_seeds(out, rl7_cfg, ("sensitivity",), "sensitivity")
    return out


# -- picks -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Pick:
    """One sensitivity policy: a training run's pick (D72) under its setting's weights."""

    setting: str
    seed: int
    run_dir: Path
    config: Path  # configs/rl7/<setting>.yaml
    manifest: Mapping

    @property
    def label(self) -> str:
        return f"{self.setting}_s{self.seed}"

    @property
    def model(self) -> Path:
        return self.run_dir / PICK_FILE


def scan_runs(root: Path) -> list[tuple[Path, dict]]:
    """(folder, manifest) of the training runs under `root` and one level of stage
    folders below it (`runs/{T,S,F}/<run>/`); `_`-prefixed folders are skipped."""
    root = Path(root)
    out = []
    if not root.is_dir():
        return out
    for level in (root.glob("*/manifest.json"), root.glob("*/*/manifest.json")):
        for m in sorted(level):
            d = m.parent
            rel = d.relative_to(root).parts
            if any(p.startswith("_") for p in rel):
                continue
            try:
                out.append((d, json.loads(m.read_text(encoding="utf-8"))))
            except (OSError, ValueError) as e:
                log.warning("%s: unreadable manifest (%s)", m, e)
    return out


def _json(path: Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_config_sha(config: Path) -> str | None:
    """`rl.train.run_config_sha` of a run config (D69), or None when rl.train predates
    it (the pick then matches on variant, seed and steps only)."""
    from traffic_twin.rl import train

    fn = getattr(train, "run_config_sha", None)
    return None if fn is None else str(fn(load_config(config)))


def find_picks(
    rl7: Path,
    runs: Path | None = None,
    steps: int | None = None,
    only: Sequence[str] | None = None,
    train_seeds: Sequence[int] | None = None,
) -> tuple[list[Pick], list[str]]:
    """The picks of every (setting, training seed) and the missing ones. A run matches
    on manifest `variant`, `seed` and `steps` (default `protocol.sensitivity.steps`),
    holds `pick_model.zip` and, when its provenance.json records one, has the setting's
    current `run_config_sha` (a run of an edited config is stale)."""
    rl7_cfg = load_config(rl7)
    root = Path(runs or outputs(rl7_cfg).get("runs") or "data/rl7/runs")
    steps = int(steps or protocol(rl7_cfg)["steps"])
    found = scan_runs(root)
    picks, missing = [], []
    for setting in only or settings(rl7_cfg):
        config = setting_config(rl7, setting)
        if not config.is_file():
            raise ConfigError(f"sensitivity setting {setting!r}: no run config {config}")
        want = run_config_sha(config)
        for seed in train_seeds if train_seeds is not None else setting_seeds(rl7_cfg, setting):
            cands = []
            for d, m in found:
                if m.get("variant") != setting or int(m.get("seed", -1)) != int(seed):
                    continue
                if int(m.get("steps", -1)) != steps or not (d / PICK_FILE).is_file():
                    continue
                prov = _json(d / "provenance.json") or {}
                have = prov.get("run_config_sha") or m.get("run_config_sha")
                if want is not None and have is not None and have != want:
                    log.warning("%s: run_config_sha %s is not the current %s", d, have, want)
                    continue
                cands.append((d, m))
            if not cands:
                missing.append(f"{setting} s{seed} ({steps:,} steps)")
                continue
            d, m = sorted(cands, key=lambda dm: dm[0].as_posix())[0]
            picks.append(Pick(setting, int(seed), d, config, m))
    return picks, missing


# -- scenarios -------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    from traffic_twin.rl.train import file_sha256 as sha

    return sha(Path(path))


def policy_scenario(base_config: Path, model: Path, rl_config: Path) -> dict:
    """The pitie_rl scenario (rl7.yaml `base_config`) with `rl.model` = the pick,
    `rl.model_sha256` pinned to it and `rl.config` = the setting's run config."""
    cfg = copy.deepcopy(load_scenario(base_config))
    rl = dict(cfg.get("rl") or {})
    rl["model"] = Path(model).as_posix()
    rl["model_sha256"] = file_sha256(model)
    rl["config"] = Path(rl_config).as_posix()
    cfg["rl"] = rl
    return cfg


def plan_policies(plan: Mapping) -> dict[str, tuple[Path | None, dict, Path]]:
    """label -> (config path or None, scenario, output root) for every run of the plan:
    the picks first (in memory), then S0, S1 and S2 (their repo configs)."""
    sens = Path(plan["sens"])
    base = Path(plan["base_config"])
    out: dict[str, tuple[Path | None, dict, Path]] = {}
    for p in plan["policies"]:
        cfg = policy_scenario(base, Path(p["model"]), Path(p["config"]))
        out[p["label"]] = (None, cfg, sens / p["label"])
    for label, path in plan["baselines"].items():
        out[label] = (Path(path), copy.deepcopy(load_scenario(path)), sens / BASELINES_DIR)
    return out


# -- runs ------------------------------------------------------------------------------
def _init_worker() -> None:
    from traffic_twin.rl.train import set_torch_threads

    set_torch_threads(1)


def _simulate(
    cfg: dict, seed: int, out_root: Path, config_path: Path | None, reward: Mapping
) -> None:
    """One full 2 h run with the passive reward observer (demand in a temporary folder)."""
    from traffic_twin.rl.reward import RewardObserver
    from traffic_twin.sim.run_scenario import run

    with tempfile.TemporaryDirectory(prefix="rl7_sens_demand_") as demand:
        run(
            cfg,
            int(seed),
            out_root,
            config_path=config_path,
            demand_dir=Path(demand),
            observers=[RewardObserver(reward)],
        )


def is_current(run_dir: Path, cfg: dict) -> bool:
    """A finished run of the same config_hash (and, for `rl`, the same model sha256)
    with its reward totals."""
    from traffic_twin.rl.evaluate import _current, rl_model_identity

    return (run_dir / REWARD_FILE).is_file() and _current(run_dir, cfg, rl_model_identity(cfg))


def _job(task: Mapping) -> dict:
    from traffic_twin.sim.run_scenario import run_dir

    cfg, seed, root = task["cfg"], int(task["seed"]), Path(task["out_root"])
    d = run_dir(cfg, seed, root)
    if not task["force"] and is_current(d, cfg):
        return {"label": task["label"], "seed": seed, "dir": d.as_posix(), "skipped": True}
    t0 = time.perf_counter()
    path = task["config_path"]
    _simulate(cfg, seed, root, Path(path) if path else None, task["reward"])
    wall = round(time.perf_counter() - t0, 1)
    return {
        "label": task["label"],
        "seed": seed,
        "dir": d.as_posix(),
        "skipped": False,
        "wall": wall,
    }


def make_plan(
    rl7: Path,
    sens: Path,
    picks: Sequence[Pick],
    seeds: Sequence[int],
    steps: int,
) -> dict:
    rl7_cfg = load_config(rl7)
    base_setting = protocol(rl7_cfg)["base"]
    rewards = {
        s: reward_block(setting_config(rl7, s)) for s in dict.fromkeys(p.setting for p in picks)
    }
    if base_setting not in rewards:
        rewards[base_setting] = reward_block(setting_config(rl7, base_setting))
    policies = []
    for p in picks:
        m = p.manifest
        policies.append(
            {
                "label": p.label,
                "setting": p.setting,
                "train_seed": p.seed,
                "run": m.get("run") or p.run_dir.name,
                "run_dir": p.run_dir.as_posix(),
                "model": p.model.as_posix(),
                "model_sha256": file_sha256(p.model),
                "config": p.config.as_posix(),
                "algo": m.get("algo"),
                "pick_step": m.get("pick_step"),
                "pick_val_cost": m.get("pick_val_cost"),
                "late_mean": m.get("late_mean"),
                "baselines": m.get("baselines"),
            }
        )
    from traffic_twin.sim.results import code_version

    return {
        "config": Path(rl7).as_posix(),
        "rl7_sha256": text_sha256(rl7),
        "code_version": code_version(),
        "created_at": datetime.now(UTC).isoformat(),
        "sens": Path(sens).as_posix(),
        "base_config": Path(rl7_cfg["base_config"]).as_posix(),
        "base": base_setting,
        "steps": int(steps),
        "seeds": [int(s) for s in seeds],
        "policies": policies,
        "baselines": {k: Path(v).as_posix() for k, v in BASELINE_CONFIGS.items()},
        "reward": rewards,
    }


def run(
    rl7: Path = RL7_CONFIG,
    workers: int | None = None,
    force: bool = False,
    out: Path | None = None,
    *,
    runs: Path | None = None,
    seeds: Sequence[int] | None = None,
    steps: int | None = None,
    only: Sequence[str] | None = None,
    train_seeds: Sequence[int] | None = None,
) -> Path:
    """KPI runs of every sensitivity pick and of S0/S1/S2 on the sensitivity seeds;
    returns the `sens/` folder (`out` replaces it, for tests and dev runs). `runs`,
    `steps`, `only` (settings) and `train_seeds` narrow the picks; `seeds` replaces the
    evaluation seeds (dev runs only: Part 7 training seeds). A missing pick is an error.
    `workers` <= 1 runs in this process, more in a spawned pool."""
    from traffic_twin.rl.train import check_out_root

    rl7 = Path(rl7)
    rl7_cfg = load_config(rl7)
    sens = sens_dir(rl7_cfg, out)
    check_out_root(sens)
    ev_seeds = eval_seeds(rl7_cfg, seeds)
    steps = int(steps or protocol(rl7_cfg)["steps"])
    picks, missing = find_picks(rl7, runs, steps, only, train_seeds)
    if missing:
        raise ConfigError(
            "sensitivity picks missing (run `tasks.py rl7-campaign` first): " + "; ".join(missing)
        )
    sens.mkdir(parents=True, exist_ok=True)
    plan = make_plan(rl7, sens, picks, ev_seeds, steps)
    base_reward = plan["reward"][plan["base"]]
    tasks = []
    by_label = {p["label"]: p for p in plan["policies"]}
    for label, (path, cfg, root) in plan_policies(plan).items():
        reward = plan["reward"][by_label[label]["setting"]] if label in by_label else base_reward
        for s in ev_seeds:
            tasks.append(
                {
                    "label": label,
                    "cfg": cfg,
                    "config_path": path.as_posix() if path else None,
                    "seed": int(s),
                    "out_root": root.as_posix(),
                    "reward": reward,
                    "force": bool(force),
                }
            )
    workers = int(workers or min(12, os.cpu_count() or 1))
    log.info(
        "sensitivity: %d policies + %s, %d seeds -> %d runs, %d workers",
        len(plan["policies"]),
        "/".join(BASELINES),
        len(ev_seeds),
        len(tasks),
        workers,
    )
    t0 = time.perf_counter()
    if workers <= 1:
        results = [_job(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            min(workers, len(tasks)), mp_context=ctx, initializer=_init_worker
        ) as ex:
            results = list(ex.map(_job, tasks))
    done = sum(not r["skipped"] for r in results)
    plan["runs"] = {
        "total": len(results),
        "ran": done,
        "skipped": len(results) - done,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (sens / PLAN_FILE).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    log.info("sensitivity runs: %d ran, %d current -> %s", done, len(results) - done, sens)
    return sens


# -- run frames ------------------------------------------------------------------------
def read_plan(sens: Path) -> dict:
    plan = _json(Path(sens) / PLAN_FILE)
    if plan is None:
        raise ConfigError(f"{sens}: no {PLAN_FILE} (run `rl.sensitivity run` first)")
    plan["sens"] = Path(sens).as_posix()  # the folder may have moved since
    return plan


def reward_totals(run_dir: Path) -> dict[str, float]:
    """The reward observer's per-class totals of a run (`reward_totals.json`; a `totals`
    block or the flat mapping)."""
    doc = _json(Path(run_dir) / REWARD_FILE)
    if doc is None:
        raise ConfigError(f"{run_dir}: no {REWARD_FILE} (rerun `rl.sensitivity run`)")
    t = doc.get("totals", doc)
    return {k: float(t.get(k, math.nan)) for k in (*TOTAL_KEYS, "cost")}


def reweighted_cost(totals: Mapping[str, Any] | pd.DataFrame, reward: Mapping) -> Any:
    """The env cost Σc (D58) of recorded totals under another reward block: w_car car_s +
    w_bus bus_s + w_ped pedestrian_s + w_amb ambulance_s + lambda penalty_s (the totals
    are linear in the weights; pending insertions are part of car_s and bus_s)."""
    w = weights_row(reward)
    return (
        w["w_car"] * totals["car_s"]
        + w["w_bus"] * totals["bus_s"]
        + w["w_pedestrian"] * totals["pedestrian_s"]
        + w["w_ambulance"] * totals["ambulance_s"]
        + w["lambda"] * totals["penalty_s"]
    )


def collect(plan: Mapping) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(metric rows: label, setting, train_seed, seed, metric, value; safety rows per
    run; reward totals per run) of every run in the plan. ConfigError when a run is
    missing or was made with another model."""
    from traffic_twin.rl.evaluate import kpi_frames
    from traffic_twin.sim.run_scenario import run_dir

    occupancy = load_config(EVALUATION_CONFIG)["occupancy"]
    seeds = [int(s) for s in plan["seeds"]]
    meta = {p["label"]: (p["setting"], p["train_seed"]) for p in plan["policies"]}
    rows, safety, totals = [], [], []
    for label, (path, cfg, root) in plan_policies(plan).items():
        setting, tseed = meta.get(label, (label, None))
        k, s = kpi_frames(root, {label: (path or Path(plan["base_config"]), cfg)}, seeds, occupancy)
        k = k.assign(metric=k["user_type"] + "/" + k["kpi"])
        rows.append(k[["label", "seed", "metric", "value"]])
        own = plan["reward"].get(setting, plan["reward"][plan["base"]])
        extra = []
        for seed in seeds:
            d = run_dir(cfg, seed, root)
            t = reward_totals(d)
            t["cost_own"] = float(reweighted_cost(t, own))
            totals.append({"label": label, "seed": seed, **t})
            split = ((_json(d / "run.json") or {}).get("control") or {}).get("iv_split") or {}
            parts = {x: split.get(x) for x in SPLIT_KEYS} if split else {}
            extra.append({"label": label, "seed": seed, **parts})
        safety.append(s.merge(pd.DataFrame(extra), on=["label", "seed"], how="left"))
    tot = pd.DataFrame(totals)
    saf = pd.concat(safety, ignore_index=True)
    rows.append(
        tot.melt(["label", "seed"], var_name="metric", value_name="value").assign(
            metric=lambda d: "reward/" + d["metric"]
        )
    )
    num = ["interventions", *[c for c in saf.columns if c.startswith("iv_")], *SPLIT_KEYS]
    num = [c for c in num if c in saf]
    rows.append(
        saf[["label", "seed", *num]]
        .melt(["label", "seed"], var_name="metric", value_name="value")
        .dropna(subset=["value"])
        .assign(metric=lambda d: "safety/" + d["metric"])
    )
    long = pd.concat(rows, ignore_index=True)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long.insert(1, "setting", long["label"].map(lambda x: meta.get(x, (x, None))[0]))
    long.insert(2, "train_seed", long["label"].map(lambda x: meta.get(x, (x, None))[1]))
    long["train_seed"] = long["train_seed"].astype("Int64")
    return long, saf, tot


# -- statistics ------------------------------------------------------------------------
def hierarchical_bootstrap(
    a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0, ci: float = 0.95
) -> tuple[float, float, float]:
    """(difference of means, CI low, CI high) of a setting `a` against the base `b`.

    `a` and `b` are (training seeds x evaluation seeds) matrices on the same evaluation
    seeds (columns), so the design is paired on the seed. Each of the `n` resamples draws
    the training seeds of each arm with replacement, then the evaluation seeds with
    replacement, once for both arms (they saw the same simulated days); the CI is the
    percentile interval of the resampled differences. NaN cells are left out of the
    means. `seed` makes it deterministic (rl7.yaml `bootstrap.seed`)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or 0 in (*a.shape, *b.shape):
        raise ValueError(
            f"bootstrap needs two matrices on the same seeds, got {a.shape} and {b.shape}"
        )
    diff = (
        float(np.nanmean(a) - np.nanmean(b))
        if np.isfinite(a).any() and np.isfinite(b).any()
        else math.nan
    )
    if not math.isfinite(diff):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    n_e = a.shape[1]
    pa = rng.integers(0, a.shape[0], size=(n, a.shape[0]))
    pb = rng.integers(0, b.shape[0], size=(n, b.shape[0]))
    e = rng.integers(0, n_e, size=(n, n_e))
    with np.errstate(invalid="ignore"), _quiet():
        ma = np.nanmean(a[pa[:, :, None], e[:, None, :]], axis=(1, 2))
        mb = np.nanmean(b[pb[:, :, None], e[:, None, :]], axis=(1, 2))
    d = ma - mb
    d = d[np.isfinite(d)]
    if not len(d):
        return diff, math.nan, math.nan
    lo, hi = np.quantile(d, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return diff, float(lo), float(hi)


class _quiet:
    """Silence numpy's "mean of empty slice" warnings inside a block."""

    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)


def _pct(value: float, ref: float) -> float:
    if ref is None or not math.isfinite(ref) or ref == 0 or not math.isfinite(value):
        return math.nan
    return 100.0 * (value / ref - 1.0)


def matrix(long: pd.DataFrame, setting: str, metric: str, seeds: Sequence[int]) -> np.ndarray:
    """(training seeds x evaluation seeds) values of a setting's metric (a baseline is
    one row)."""
    g = long[(long["setting"] == setting) & (long["metric"] == metric)]
    w = g.pivot_table(index="label", columns="seed", values="value", aggfunc="mean")
    return w.reindex(columns=list(seeds)).to_numpy(float)


def report_metrics(long: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(long["metric"]))
    out = [m for m in REPORT_METRICS if m in present]
    for prefix in SUBGROUP_METRICS:
        out += sorted(m for m in present if m.startswith(prefix))
    return out


def kpi_table(long: pd.DataFrame, plan: Mapping) -> pd.DataFrame:
    """One row per (setting or baseline, metric): mean over policies and seeds, SD of the
    policy means, % vs S0, S1 and the base setting, difference vs the base."""
    base = plan["base"]
    pol = long.groupby(["setting", "label", "metric"], sort=False)["value"].mean().reset_index()
    agg = pol.groupby(["setting", "metric"], sort=False)["value"].agg(
        mean="mean", sd_policies="std", n_policies="count"
    )
    n_seeds = long.groupby(["setting", "metric"], sort=False)["seed"].nunique().rename("n_seeds")
    t = agg.join(n_seeds).reset_index()
    ref = t.set_index(["setting", "metric"])["mean"]
    for r in ("S0", "S1", base):
        col = "base" if r == base else r
        t[f"vs_{col}_pct"] = [
            _pct(m, ref.get((r, k), math.nan)) for m, k in zip(t["mean"], t["metric"], strict=True)
        ]
    t["diff_vs_base"] = [
        m - ref.get((base, k), math.nan) for m, k in zip(t["mean"], t["metric"], strict=True)
    ]
    t["kind"] = np.where(t["setting"].isin(BASELINES), "baseline", "setting")
    t["note"] = t["metric"].map(NOT_INTERPRETED).fillna("")
    order = {s: i for i, s in enumerate([*settings_of(plan), *BASELINES])}
    t = t.sort_values(
        ["setting", "metric"], key=lambda c: c.map(order) if c.name == "setting" else c
    )
    cols = [
        "setting",
        "kind",
        "metric",
        "mean",
        "sd_policies",
        "n_policies",
        "n_seeds",
        "vs_S0_pct",
        "vs_S1_pct",
        "vs_base_pct",
        "diff_vs_base",
        "note",
    ]
    return t[cols].reset_index(drop=True)


def settings_of(plan: Mapping) -> list[str]:
    """Settings of the plan in order, the base first."""
    return list(dict.fromkeys([plan["base"], *(p["setting"] for p in plan["policies"])]))


def bootstrap_table(long: pd.DataFrame, plan: Mapping, boot: Mapping) -> pd.DataFrame:
    """Hierarchical bootstrap CI of every setting's change vs the base (D76)."""
    base, seeds = plan["base"], [int(s) for s in plan["seeds"]]
    n, seed, ci = int(boot.get("n", 2000)), int(boot.get("seed", 0)), float(boot.get("ci", 0.95))
    rows = []
    present = set(long["setting"])
    for setting in settings_of(plan):
        if setting == base or setting not in present or base not in present:
            continue
        for metric in report_metrics(long):
            a, b = matrix(long, setting, metric, seeds), matrix(long, base, metric, seeds)
            if a.size == 0 or b.size == 0:
                continue
            d, lo, hi = hierarchical_bootstrap(a, b, n, seed, ci)
            bm = float(np.nanmean(b)) if np.isfinite(b).any() else math.nan
            rows.append(
                {
                    "setting": setting,
                    "metric": metric,
                    "base_mean": bm,
                    "diff": d,
                    "ci_low": lo,
                    "ci_high": hi,
                    "diff_pct": _pct(bm + d, bm) if math.isfinite(d) else math.nan,
                    "beyond_noise": bool(math.isfinite(lo) and (lo > 0 or hi < 0)),
                    "n_policies": a.shape[0],
                    "n_policies_base": b.shape[0],
                    "n_seeds": len(seeds),
                    "resamples": n,
                    "note": NOT_INTERPRETED.get(metric, ""),
                }
            )
    return pd.DataFrame(rows)


def setting_kpis(long: pd.DataFrame, setting: str) -> pd.DataFrame:
    """seed, user_type, kpi, value of a setting: per evaluation seed, the mean over its
    policies (run KPIs only)."""
    g = long[(long["setting"] == setting) & ~long["metric"].str.startswith(("reward/", "safety/"))]
    g = g.groupby(["seed", "metric"], sort=False)["value"].mean().reset_index()
    parts = g["metric"].str.split("/", n=1, expand=True)
    return pd.DataFrame(
        {"seed": g["seed"], "user_type": parts[0], "kpi": parts[1], "value": g["value"]}
    )


def equity_rows(long: pd.DataFrame, plan: Mapping, ecfg: Mapping) -> pd.DataFrame:
    """Part 4 equity flags vs S0 (D29, `evaluation.compare`): for every setting (policies
    averaged per seed) and for S1 and S2, the primary, subgroup and global KPIs with
    their change, exact Wilcoxon p and the > `equity.pct` % worse flag."""
    from traffic_twin.evaluation.compare import compare

    eq = ecfg["equity"]
    pct = float(eq["pct"])
    primary = dict(eq.get("primary") or {})
    subgroups = dict(eq.get("subgroups") or {})
    glob = eq.get("global") or {}
    if "S0" not in set(long["setting"]):
        return pd.DataFrame()
    s0 = setting_kpis(long, "S0")
    rows = []
    for setting in [*settings_of(plan), "S1", "S2"]:
        if setting not in set(long["setting"]):
            continue
        t = compare(s0, setting_kpis(long, setting), [], pct)
        for r in t.itertuples(index=False):
            role = ""
            if primary.get(r.user_type) == r.kpi:
                role = "primary"
            elif subgroups.get(r.user_type) and str(r.kpi).startswith(str(subgroups[r.user_type])):
                role = "subgroup"
            elif r.user_type == glob.get("user") and r.kpi == glob.get("kpi"):
                role = "global"
            if not role:
                continue
            rows.append(
                {
                    "setting": setting,
                    "kind": "baseline" if setting in BASELINES else "setting",
                    "user_type": r.user_type,
                    "kpi": r.kpi,
                    "role": role,
                    "s0_mean": r.s0_mean,
                    "mean": r.s1_mean,
                    "rel_change_pct": r.rel_change_pct,
                    "p_value": r.p_value,
                    "worse_flag": bool(r.equity_flag) and role != "global",
                    "significant": bool(r.significant),
                    "effect": r.effect,
                    "note": NOT_INTERPRETED.get(f"{r.user_type}/{r.kpi}", ""),
                }
            )
    return pd.DataFrame(rows)


def reweight_table(tot: pd.DataFrame, long: pd.DataFrame, plan: Mapping) -> pd.DataFrame:
    """No-retraining reweighting: the mean 2 h cost of every controller (each setting's
    policies pooled, S0, S1, S2) under every setting's weight vector, its % vs S1 and S2
    under the same weights and its rank (1 = lowest)."""
    setting_of = long.drop_duplicates("label").set_index("label")["setting"]
    t = tot.assign(controller=tot["label"].map(setting_of))
    rows = []
    for weights, reward in plan["reward"].items():
        cost = t.assign(cost=reweighted_cost(t, reward))
        means = cost.groupby("controller", sort=False)["cost"].mean()
        ranks = means.rank(method="min")
        for ctl, m in means.items():
            rows.append(
                {
                    "weights": weights,
                    **weights_row(reward),
                    "controller": ctl,
                    "kind": "baseline" if ctl in BASELINES else "setting",
                    "cost_mean": float(m),
                    "vs_S1_pct": _pct(m, means.get("S1", math.nan)),
                    "vs_S2_pct": _pct(m, means.get("S2", math.nan)),
                    "rank": int(ranks[ctl]),
                    "own": ctl == weights,
                }
            )
    return pd.DataFrame(rows)


def convergence_table(tot: pd.DataFrame, plan: Mapping) -> pd.DataFrame:
    """Per pick: its selection-window cost (D72) and late mean against S1 and S2 under
    its own weights (the run's baselines.csv, same windows), and its mean 2 h cost on the
    sensitivity seeds against S2's under the same weights. A pick that does not beat S2
    on its own objective is read with caution (the optimiser, not the weights, may
    explain its KPIs)."""
    rows = []
    s2 = tot[tot["label"] == "S2"]
    for p in plan["policies"]:
        reward = plan["reward"][p["setting"]]
        base = p.get("baselines") or {}
        pick = p.get("pick_val_cost")
        pick = float(pick) if pick is not None else math.nan
        late = p.get("late_mean")
        late = float(late) if late is not None else math.nan
        mine = tot[tot["label"] == p["label"]]
        own = float(reweighted_cost(mine, reward).mean()) if len(mine) else math.nan
        ref = float(reweighted_cost(s2, reward).mean()) if len(s2) else math.nan
        sel_s2 = float(base.get("S2", math.nan))
        rows.append(
            {
                "setting": p["setting"],
                "train_seed": p["train_seed"],
                "run": p["run"],
                "pick_step": p.get("pick_step"),
                "pick_val_cost": pick,
                "sel_S1": float(base.get("S1", math.nan)),
                "sel_S2": sel_s2,
                "pick_vs_S1_pct": _pct(pick, float(base.get("S1", math.nan))),
                "pick_vs_S2_pct": _pct(pick, sel_s2),
                "late_mean": late,
                "late_vs_S2_pct": _pct(late, sel_s2),
                "sens_cost_own": own,
                "sens_S2_cost": ref,
                "sens_vs_S2_pct": _pct(own, ref),
                "beats_S2": bool(math.isfinite(pick) and math.isfinite(sel_s2) and pick < sel_s2),
            }
        )
    return pd.DataFrame(rows)


OPERATOR_COLUMNS = (
    ("car/time_loss_mean", "car_time_loss_s"),
    ("bus/delay_mean", "bus_delay_s"),
    ("pedestrian/ped_curb_wait_mean", "ped_curb_wait_mean_s"),
    ("pedestrian/ped_curb_wait_p95", "ped_curb_wait_p95_s"),
    ("all/person_delay_mean", "person_delay_s"),
    ("reward/penalty_s", "ped_penalty_s_per_run"),
    ("safety/interventions", "interventions_per_run"),
)


def operator_table(
    kpis: pd.DataFrame, equity: pd.DataFrame, conv: pd.DataFrame, plan: Mapping
) -> pd.DataFrame:
    """The operator's weight-sensitivity table (spec 12): one row per setting and
    baseline with its weights, the headline KPIs per class, the change vs the base and
    the equity flags vs S0."""
    base = plan["base"]
    k = kpis.set_index(["setting", "metric"])
    rows = []
    for setting in [*settings_of(plan), *BASELINES]:
        if setting not in set(kpis["setting"]):
            continue
        reward = plan["reward"].get(setting) if setting not in BASELINES else None
        row: dict[str, Any] = {
            "setting": setting,
            "kind": "baseline" if reward is None else "setting",
        }
        w = weights_row(reward) if reward else {}
        row |= {key: w.get(key, math.nan) for key in ("w_car", "w_bus", "w_pedestrian", "lambda")}
        row["direction"] = direction(reward, plan["reward"][base]) if reward else ""
        n = k["n_policies"].get((setting, "car/time_loss_mean"), math.nan)
        row["n_policies"] = n if reward else 0
        for metric, name in OPERATOR_COLUMNS:
            row[name] = k["mean"].get((setting, metric), math.nan)
        for metric, name in OPERATOR_COLUMNS[:4]:
            row[f"{name.removesuffix('_s')}_vs_base_pct"] = k["vs_base_pct"].get(
                (setting, metric), math.nan
            )
        eq = (
            equity[(equity["setting"] == setting) & equity["worse_flag"]] if len(equity) else equity
        )
        row["equity_flags_vs_S0"] = int(len(eq))
        row["equity_flagged"] = (
            "; ".join(f"{r.user_type}.{r.kpi}" for r in eq.itertuples()) if len(eq) else ""
        )
        if reward and len(conv):
            c = conv[conv["setting"] == setting]
            row["picks_beating_S2_own_weights"] = f"{int(c['beats_S2'].sum())}/{len(c)}"
        else:
            row["picks_beating_S2_own_weights"] = ""
        rows.append(row)
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out


def tables(
    rl7: Path = RL7_CONFIG,
    out: Path | None = None,
    operator_csv: Path | None = None,
) -> dict[str, Path]:
    """The sensitivity tables (TABLES) and the operator table; returns name -> path."""
    rl7_cfg = load_config(rl7)
    sens = sens_dir(rl7_cfg, out)
    plan = read_plan(sens)
    long, saf, tot = collect(plan)
    long.to_csv(sens / RUNS_CSV, index=False)
    saf.to_csv(sens / SAFETY_CSV, index=False)
    tot.to_csv(sens / TOTALS_CSV, index=False)
    kpis = kpi_table(long, plan)
    boot = bootstrap_table(long, plan, protocol(rl7_cfg).get("bootstrap") or {})
    equity = equity_rows(long, plan, load_config(EVALUATION_CONFIG))
    rew = reweight_table(tot, long, plan)
    conv = convergence_table(tot, plan)
    frames = {
        "kpis": kpis,
        "bootstrap": boot,
        "equity": equity,
        "reweight": rew,
        "convergence": conv,
    }
    paths = {}
    for name in TABLES:
        p = sens / f"{name}.csv"
        frames[name].to_csv(p, index=False)
        paths[name] = p
    op = Path(
        operator_csv or Path(outputs(rl7_cfg).get("figures") or "docs/figures/rl7") / OPERATOR_CSV
    )
    op.parent.mkdir(parents=True, exist_ok=True)
    operator_table(kpis, equity, conv, plan).to_csv(op, index=False, lineterminator="\n")
    paths["operator"] = op
    return paths


# -- CLI -------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, default=RL7_CONFIG, help="Part 7 config (rl7.yaml)")
    p.add_argument("--out", type=Path, help="sens folder (default: <outputs.study>/sens)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="full 2 h KPI runs of the picks and S0/S1/S2")
    r.add_argument("--workers", type=int)
    r.add_argument("--force", action="store_true", help="rerun runs that are current")
    r.add_argument("--runs", type=Path, help="training runs root (default: outputs.runs)")
    r.add_argument("--steps", type=int, help="training steps of the picks (dev runs)")
    r.add_argument("--settings", nargs="+", help="only these settings (dev runs)")
    r.add_argument("--train-seeds", nargs="+", type=int, help="only these training seeds")
    r.add_argument("--seeds", nargs="+", help="evaluation seeds (dev runs: training seeds)")
    t = sub.add_parser("tables", help="sensitivity tables and the operator table")
    t.add_argument("--operator-csv", type=Path, help="default: <outputs.figures>/sensitivity.csv")
    return p


def main(argv: list[str] | None = None) -> int:
    from traffic_twin.rl.evaluate import parse_seeds

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.cmd == "run":
            sens = run(
                args.config,
                args.workers,
                args.force,
                args.out,
                runs=args.runs,
                seeds=parse_seeds(args.seeds) if args.seeds else None,
                steps=args.steps,
                only=args.settings,
                train_seeds=args.train_seeds,
            )
            print(sens.as_posix())
        elif args.cmd == "tables":
            for path in tables(args.config, args.out, args.operator_csv).values():
                print(path.as_posix())
    except (ConfigError, ValueError) as e:
        raise SystemExit(f"error: {e}") from None
    return 0


if __name__ == "__main__":
    sys.exit(main())

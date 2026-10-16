"""RL timing, model selection and evaluation (Part 6, 10.4, 10.6, S6; D60; contract 4.7).

    python -m traffic_twin.rl.evaluate timing [--n-envs 1 4 8 12] [--seeds 3000..3002]
    python -m traffic_twin.rl.evaluate select [--runs DIR ...] [--algo dqn]
    python -m traffic_twin.rl.evaluate compare --seeds 3020..3029 [--workers 8]
    python -m traffic_twin.rl.evaluate kpi --seeds 3020..3029 [--force]
    python -m traffic_twin.rl.evaluate repro RUN_A RUN_B
    python -m traffic_twin.rl.evaluate docs --docs docs/rl.md --figures docs/figures/rl

- `timing` (spec 10.4 "mesurez le temps d'un épisode", S6): full 07:30-09:30 episodes on
  selection seeds 3000-3002 under S0, S1, S2, random, always-stay and the trained model
  (wall, reset, env step, NN inference, decisions), plus SubprocVecEnv throughput for
  each `--n-envs` (random policy, 5k decision steps) -> `<runs>/timing/timing.json`;
- `select`: the training run with the lowest best validation cost (selection seeds,
  D60) among the SUMO runs of `--algo` made with the current rl.yaml ->
  `data/models/rl/selected/{model.zip, manifest.json, validation.csv}`;
- `compare` (exploratory, D60a): paired 30-min window cost on dev-test seeds for RL, S0,
  S1, S2, random and always-stay, Wilcoxon against S1 (`evaluation.stats`) ->
  `<runs>/devtest/compare.csv`, `compare_summary.json`;
- `kpi` (exploratory, D60b): full 2 h `run_scenario` runs of pitie_{fixed, actuated,
  rules, rl}.yaml on dev-test seeds (KPI v3, depart delay, danger, collisions, curb wait,
  interventions, preemption outcome) -> `<runs>/devtest/<scenario>_seed<n>/`, `kpi.csv`,
  `kpi_safety.csv`, `kpi_summary.json` (with the RL model every pitie_rl run was checked
  against and the runs' own code versions); a finished run is reused only when its
  config_hash and, for pitie_rl, its model sha256 are current; exits 3 when a run breaks a
  safety bound (Phase 3 STOP rule);
- `repro`: two runs of the same command give the same policy parameters and the same
  episode returns (D65) -> `<runs>/repro.json`; exits 1 when they differ;
- `docs`: `docs/rl.md` and `docs/figures/rl/*` from the artifacts above.

Seeds: `compare` and `kpi` take only rl.yaml `dev_test` or `selection` seeds, `timing`
only `selection` or `train` seeds; the Part 3-4 seeds of configs/evaluation.yaml (1-10,
101-103, 201-207) are always refused. Nothing is written under data/results.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_twin.config import ConfigError, config_hash, get, load_config, load_scenario
from traffic_twin.control.safety import INTERVENTION_KEYS
from traffic_twin.evaluation.stats import wilcoxon
from traffic_twin.paths import (
    EVALUATION_CONFIG,
    PITIE_ACTUATED_CONFIG,
    PITIE_FIXED_CONFIG,
    PITIE_RL_CONFIG,
    PITIE_RULES_CONFIG,
    RL_CONFIG,
    RL_RUNS_DIR,
)
from traffic_twin.rl.train import (
    ALGOS,
    PREEMPT_KEYS,
    S6_MARGIN_PCT,
    S6_REFERENCES,
    TOTAL_KEYS,
    EnvBackend,
    EnvFactory,
    check_out_root,
    file_sha256,
    library_versions,
    load_model,
    model_policy,
    pct_change,
    real_backend,
    run_episodes,
    s6_run_pass,
    seed_range,
)
from traffic_twin.sim.results import code_version

log = logging.getLogger(__name__)

TIMING_POLICIES = ("fixed", "actuated", "rules", "random", "stay", "model")
COMPARE_POLICIES = {
    "RL": "rl",
    "S0": "fixed",
    "S1": "actuated",
    "S2": "rules",
    "random": "random",
    "stay": "stay",
}
KPI_CONFIGS = {
    "S0": PITIE_FIXED_CONFIG,
    "S1": PITIE_ACTUATED_CONFIG,
    "S2": PITIE_RULES_CONFIG,
    "RL": PITIE_RL_CONFIG,
}
REFERENCE = "S1"  # exploratory comparisons are against actuated (spec 10.6)
HEADLINE = (
    ("car", "time_loss_mean"),
    ("car", "delay_total_mean"),
    ("car", "depart_delay_mean"),
    ("bus", "delay_mean"),
    ("pedestrian", "ped_curb_wait_max"),
    ("pedestrian", "ped_curb_wait_mean"),
    ("ambulance", "travel_time_total"),
    ("all", "person_delay_mean"),
)
S6_MIN_PASS = 2  # D60: >= 2 of the 3 DQN runs
DEFAULT_CURB_LIMIT_S = 90.0  # spec ped max wait (A-04); a scenario's safety.ped_max_wait_s wins
SAFETY_EXIT = 3
DOCS_FIGURES_NOTE = "Figures: `python -m traffic_twin.rl.evaluate docs` (viz.rl)."


# -- seeds -----------------------------------------------------------------------------
def parse_seeds(tokens: Sequence[str | int]) -> list[int]:
    """Seeds from tokens like "3020..3029", "3020-3029", "3020,3021" or "3020" (order
    kept, duplicates dropped)."""
    out: list[int] = []
    for tok in tokens:
        for part in str(tok).split(","):
            part = part.strip()
            if not part:
                continue
            m = re.fullmatch(r"(\d+)\s*(?:\.\.|-)\s*(\d+)", part)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                if hi < lo:
                    raise ValueError(f"empty seed range {part!r}")
                out.extend(range(lo, hi + 1))
            elif re.fullmatch(r"\d+", part):
                out.append(int(part))
            else:
                raise ValueError(f"bad seed {part!r}")
    return list(dict.fromkeys(out))


def protected_seeds(evaluation_config: Path = EVALUATION_CONFIG) -> set[int]:
    """The Part 3-4 seeds of configs/evaluation.yaml (evaluation, calibration,
    development): never run in Parts 5-6 (D38)."""
    seeds = load_config(evaluation_config)["seeds"]
    return {int(s) for v in seeds.values() if isinstance(v, list) for s in v}


def check_seeds(
    seeds: Sequence[int],
    rl_cfg: Mapping,
    allowed: Sequence[str],
    what: str,
    evaluation_config: Path = EVALUATION_CONFIG,
) -> None:
    """ConfigError unless every seed lies in one of the rl.yaml ranges `allowed` and none
    is a Part 3-4 seed."""
    if not seeds:
        raise ConfigError(f"{what}: no seeds")
    bad = sorted(set(seeds) & protected_seeds(evaluation_config))
    if bad:
        raise ConfigError(f"{what}: seeds {bad} belong to Parts 3-4 (configs/evaluation.yaml)")
    ranges = {k: rl_cfg["seeds"][k] for k in allowed}
    outside = [s for s in seeds if not any(lo <= s <= hi for lo, hi in ranges.values())]
    if outside:
        names = ", ".join(f"{k} {lo}-{hi}" for k, (lo, hi) in ranges.items())
        raise ConfigError(f"{what}: seeds {outside} are outside rl.yaml {names}")


def _outputs(rl_cfg: Mapping, key: str, default: Path) -> Path:
    return Path((rl_cfg.get("outputs") or {}).get(key) or default)


def runs_root(rl_cfg: Mapping) -> Path:
    return _outputs(rl_cfg, "runs", RL_RUNS_DIR)


def selected_dir(rl_cfg: Mapping) -> Path:
    return _outputs(rl_cfg, "selected", Path("data/models/rl/selected"))


def _json(path: Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf-8")
    return path


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Path):
        return x.as_posix()
    raise TypeError(f"not JSON serializable: {type(x).__name__}")


def provenance(config: Path) -> dict:
    from traffic_twin.config import text_sha256

    cv = code_version()
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "code_version": cv,
        "dirty": cv.endswith("-dirty"),
        "rl_yaml_sha256": text_sha256(config),
        "versions": library_versions(),
    }


def full_episode_times(rl_cfg: Mapping) -> dict | None:
    """begin_s, warmup_s, window_s and warmup_controller of a full-mode episode
    (`rl.env.episode_params` on rl.yaml `base_config`), or None when unavailable."""
    try:
        from traffic_twin.rl.env import episode_params

        base = load_scenario(rl_cfg["base_config"])
        p = episode_params(
            0,
            dict(rl_cfg),
            "full",
            randomize=False,
            window=(float(base["begin_s"]), int(base["duration_s"])),
        )
    except (ConfigError, OSError, KeyError, TypeError, ValueError) as e:
        log.warning("full-mode episode times unavailable: %s", e)
        return None
    return {k: p[k] for k in ("begin_s", "warmup_s", "window_s", "warmup_controller")}


def preempt_max_hold_s(rl_cfg: Mapping) -> float | None:
    """`preemption.max_hold_s` of rl.yaml `base_config` (the RL scenario), or None."""
    try:
        v = get(load_scenario(rl_cfg["base_config"]), "preemption.max_hold_s", None)
    except (ConfigError, OSError, KeyError, TypeError):
        return None
    return None if v is None else float(v)


# -- trained models --------------------------------------------------------------------
def resolve_model(model: Path | None, rl_cfg: Mapping, algo: str | None = None) -> dict:
    """{path, algo, manifest} of `model` (default: the selected model). The algo comes
    from the manifest.json next to the zip (selected or run manifest), else `algo`."""
    path = Path(model) if model else selected_dir(rl_cfg) / "model.zip"
    if not path.is_file():
        raise ConfigError(f"no RL model at {path} (run `rl.evaluate select` or pass --model)")
    manifest = _json(path.parent / "manifest.json") or {}
    algo = manifest.get("algo") or algo
    if algo not in ALGOS:
        raise ConfigError(f"{path}: unknown algo (no manifest.json next to it; pass --algo)")
    return {"path": path, "algo": algo, "manifest": manifest}


# -- timing ----------------------------------------------------------------------------
def env_wall(summary: Mapping | None) -> dict[str, float]:
    """The env's own per-episode timing (`episode_summary["wall"]`: reset, start, warm-up,
    sim, reward, features seconds, steps), prefixed `env_`; {} when it keeps none."""
    wall = (summary or {}).get("wall")
    if not isinstance(wall, Mapping):
        return {}
    return {f"env_{k}": float(v) for k, v in wall.items() if isinstance(v, int | float)}


def timed_episode(env, act: Callable[[np.ndarray], int], seed: int) -> dict:
    """Wall time of one episode split into reset, env steps and NN inference, plus the
    env's own split (`env_wall`)."""
    t0 = time.perf_counter()
    obs, _ = env.reset(options={"episode_seed": int(seed)})
    reset_s = time.perf_counter() - t0
    nn_s = step_s = 0.0
    decisions = 0
    while True:
        t = time.perf_counter()
        a = act(obs)
        nn_s += time.perf_counter() - t
        t = time.perf_counter()
        obs, _, terminated, truncated, info = env.step(a)
        step_s += time.perf_counter() - t
        decisions += 1
        if terminated or truncated:
            break
    wall = time.perf_counter() - t0
    row = {
        "wall_s": wall,
        "reset_s": reset_s,
        "step_s": step_s,
        "nn_s": nn_s,
        "decisions": decisions,
        "decisions_per_s": decisions / wall if wall > 0 else math.nan,
        "sim_t_end": float(info.get("t", math.nan)),
    }
    return row | env_wall(info.get("episode_summary"))


def time_episodes(
    backend: EnvBackend,
    config: Path,
    seeds: Sequence[int],
    policies: Sequence[str],
    model: dict | None = None,
    work_root: Path | None = None,
    mode: str = "full",
) -> list[dict]:
    """Full episodes per (policy, seed) in this process (one env, libsumo). Baselines
    (fixed / actuated / rules) run through `run_baseline` and are timed as a whole; env
    actions (random / stay / model) through `timed_episode`."""
    rows: list[dict] = []
    t0 = time.perf_counter()
    env = backend.env(config, mode=mode, seeds=list(seeds), work_root=work_root)
    construct_s = time.perf_counter() - t0
    first = True
    loaded = load_model(model["path"], model["algo"]) if model else None
    try:
        for policy in policies:
            if policy == "model" and loaded is None:
                continue
            for seed in seeds:
                if policy in ("fixed", "actuated", "rules"):
                    t = time.perf_counter()
                    summary = backend.run_baseline(env, policy, int(seed))
                    wall = time.perf_counter() - t
                    # a baseline controller decides every second: no agent decisions
                    row = {"wall_s": wall, "decisions": int(summary.get("decisions", 0) or 0)}
                    row |= env_wall(summary)
                else:
                    if policy == "model":
                        act = model_policy(loaded)
                    elif policy == "random":
                        rng = np.random.default_rng([int(seed), 7_000_000])
                        act = lambda _obs, rng=rng: int(rng.integers(2))  # noqa: E731
                    elif policy == "stay":
                        act = lambda _obs: 0  # noqa: E731
                    else:
                        raise ValueError(f"unknown timing policy {policy!r}")
                    row = timed_episode(env, act, int(seed))
                row = {"policy": policy, "seed": int(seed), **row}
                if first:
                    row["construct_s"] = construct_s
                    first = False
                rows.append(row)
                log.info(
                    "timing %s seed %d: %.2f s, %d decisions",
                    policy,
                    seed,
                    row["wall_s"],
                    row["decisions"],
                )
    finally:
        env.close()
    return rows


def time_scaling(
    backend: EnvBackend,
    config: Path,
    n_envs_list: Sequence[int],
    steps: int,
    seeds: Sequence[int],
    work_root: Path | None = None,
    threads: int = 1,
) -> list[dict]:
    """Decision steps/s of a SubprocVecEnv (spawn) of n train-mode envs under a random
    policy, for each n (start-up = process spawn + first reset)."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    rows = []
    for n in n_envs_list:
        n = int(n)
        t0 = time.perf_counter()
        venv = SubprocVecEnv(
            [EnvFactory(backend, config, i, n, seeds, work_root, threads) for i in range(n)],
            start_method="spawn",
        )
        try:
            venv.reset()
            startup = time.perf_counter() - t0
            rng = np.random.default_rng(0)
            done = episodes = 0
            t1 = time.perf_counter()
            while done < steps:
                _, _, dones, _ = venv.step(rng.integers(0, 2, size=n))
                done += n
                episodes += int(np.sum(dones))
            wall = time.perf_counter() - t1
        finally:
            venv.close()
        rows.append(
            {
                "n_envs": n,
                "steps": done,
                "wall_s": wall,
                "steps_per_s": done / wall,
                "episodes": episodes,
                "startup_s": startup,
            }
        )
        log.info("scaling n_envs %d: %.0f steps/s", n, done / wall)
    return rows


def timing_summary(rows: Sequence[Mapping]) -> dict[str, dict]:
    """Per policy: episodes, mean / max wall, decisions and the time split."""
    df = pd.DataFrame(rows)
    out: dict[str, dict] = {}
    if df.empty:
        return out
    for policy, g in df.groupby("policy", sort=False):
        d = {
            "episodes": len(g),
            "wall_s_mean": float(g["wall_s"].mean()),
            "wall_s_max": float(g["wall_s"].max()),
            "decisions_mean": float(g["decisions"].mean()),
        }
        cols = ["reset_s", "step_s", "nn_s", "decisions_per_s"]
        cols += [c for c in g.columns if str(c).startswith("env_")]
        for c in cols:
            if c in g and g[c].notna().any():
                d[f"{c}_mean"] = float(g[c].mean())
        out[str(policy)] = d
    return out


def timing(
    config: Path = RL_CONFIG,
    seeds: Sequence[int] = (3000, 3001, 3002),
    n_envs: Sequence[int] = (1, 4, 8, 12),
    steps: int = 5000,
    model: Path | None = None,
    out: Path | None = None,
    backend: EnvBackend | None = None,
    policies: Sequence[str] = TIMING_POLICIES,
) -> Path:
    """`timing` subcommand; returns the timing.json path."""
    rl_cfg = load_config(config)
    check_seeds(seeds, rl_cfg, ("selection", "train"), "timing")
    backend = backend or real_backend()
    root = runs_root(rl_cfg)
    out = Path(out or root / "timing" / "timing.json")
    work_root = root / "_work"
    check_out_root(work_root)
    check_out_root(out.parent)
    mdl = None
    if "model" in policies:
        try:
            mdl = resolve_model(model, rl_cfg)
        except ConfigError as e:
            log.warning("timing without a trained model: %s", e)
    rows = time_episodes(backend, config, seeds, policies, mdl, work_root)
    train_seeds = seed_range(rl_cfg["seeds"]["train"])
    scaling = time_scaling(backend, config, n_envs, steps, train_seeds, work_root) if n_envs else []
    doc = provenance(config) | {
        "backend": backend.name,
        "mode": "full",
        "episode": full_episode_times(rl_cfg),
        "seeds": list(seeds),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "model": None
        if mdl is None
        else {
            "path": mdl["path"].as_posix(),
            "algo": mdl["algo"],
            "model_version": mdl["manifest"].get("model_version"),
        },
        "episodes": rows,
        "summary": timing_summary(rows),
        "scaling": scaling,
        "scaling_steps": steps,
    }
    return _write_json(out, doc)


# -- selection -------------------------------------------------------------------------
def find_runs(root: Path) -> list[Path]:
    """Run folders (holding manifest.json) directly under `root`; `_work` and other
    `_`-prefixed folders are skipped."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "manifest.json").is_file()
    )


def read_manifest(run_dir: Path) -> dict:
    m = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    m["dir"] = Path(run_dir)
    return m


def eligible(
    m: Mapping, algo: str | None, rl_sha: str | None, backends: Sequence[str] = ("sumo",)
) -> str | None:
    """Why a run cannot be selected (None if it can)."""
    if algo and m.get("algo") != algo:
        return f"algo {m.get('algo')}"
    if m.get("backend") not in backends:
        return f"backend {m.get('backend')}"
    if rl_sha and m.get("rl_yaml_sha256") != rl_sha:
        return "made with another rl.yaml"
    if not (Path(m["dir"]) / "best_model.zip").is_file():
        return "no best_model.zip"
    if m.get("best_val_cost") is None or not math.isfinite(float(m["best_val_cost"])):
        return "no validation cost"
    return None


def select_run(
    manifests: Sequence[Mapping],
    algo: str | None = "dqn",
    rl_sha: str | None = None,
    backends: Sequence[str] = ("sumo",),
) -> dict:
    """The eligible run with the lowest best validation cost (ties: lower seed, then
    name). ConfigError when none is eligible."""
    ok = [m for m in manifests if eligible(m, algo, rl_sha, backends) is None]
    if not ok:
        why = (
            "; ".join(
                f"{Path(m['dir']).name}: {eligible(m, algo, rl_sha, backends)}" for m in manifests
            )
            or "no runs"
        )
        raise ConfigError(f"no eligible {algo or 'RL'} run ({why})")
    return dict(min(ok, key=lambda m: (float(m["best_val_cost"]), int(m["seed"]), m["run"])))


def model_version(m: Mapping) -> str:
    return f"rl1-{m['algo']}-s{m['seed']}-{str(m['best_policy_sha256'])[:8]}"


def select(
    config: Path = RL_CONFIG,
    runs: Sequence[Path] | None = None,
    algo: str | None = "dqn",
    out_dir: Path | None = None,
    backends: Sequence[str] = ("sumo",),
) -> Path:
    """`select` subcommand: copy the chosen run's best_model.zip to `<out>/model.zip` and
    write the selection manifest (read by `control.rl.RLController`)."""
    from traffic_twin.config import text_sha256

    rl_cfg = load_config(config)
    out_dir = Path(out_dir or selected_dir(rl_cfg))
    dirs = list(runs) if runs else find_runs(runs_root(rl_cfg))
    manifests = [read_manifest(d) for d in dirs]
    sha = text_sha256(config)
    chosen = select_run(manifests, algo, sha, backends)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(chosen["dir"])
    shutil.copyfile(src / "best_model.zip", out_dir / "model.zip")
    if (src / "validation.csv").is_file():
        shutil.copyfile(src / "validation.csv", out_dir / "validation.csv")
    model_sha = file_sha256(out_dir / "model.zip")
    if chosen.get("best_model_sha256") and chosen["best_model_sha256"] != model_sha:
        raise ConfigError(f"{src / 'best_model.zip'}: sha256 differs from its manifest")
    candidates = [
        {
            "run": m.get("run"),
            "algo": m.get("algo"),
            "seed": m.get("seed"),
            "best_val_cost": m.get("best_val_cost"),
            "excluded": eligible(m, algo, sha, backends),
        }
        for m in manifests
    ]
    manifest = {
        "model_version": model_version(chosen),
        "algo": chosen["algo"],
        "model": "model.zip",
        "model_sha256": model_sha,
        "policy_sha256": chosen.get("best_policy_sha256"),
        "obs_version": chosen.get("obs_version"),
        "reward_version": chosen.get("reward_version"),
        "rl_yaml_sha256": chosen.get("rl_yaml_sha256"),
        "source_run": chosen.get("run"),
        "source_dir": src.as_posix(),
        "seed": chosen.get("seed"),
        "steps": chosen.get("steps"),
        "n_envs": chosen.get("n_envs"),
        "best_step": chosen.get("best_step"),
        "best_val_cost": chosen.get("best_val_cost"),
        "baselines": chosen.get("baselines"),
        "best_vs": chosen.get("best_vs"),
        "code_version": chosen.get("code_version"),
        "versions": chosen.get("versions"),
        "selected_at": datetime.now(UTC).isoformat(),
        "selection": {
            "rule": "lowest best validation cost on the selection seeds (D60) among the "
            f"{algo or 'RL'} runs of the current rl.yaml",
            "candidates": candidates,
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    log.info("selected %s (best cost %.1f) -> %s", chosen["run"], chosen["best_val_cost"], out_dir)
    return out_dir


# -- dev-test window comparison --------------------------------------------------------
def compare_frame(
    backend: EnvBackend,
    config: Path,
    seeds: Sequence[int],
    model: Mapping,
    workers: int = 1,
    policies: Mapping[str, str] = COMPARE_POLICIES,
    work_root: Path | None = None,
) -> pd.DataFrame:
    """label, policy, seed and the episode row of every (policy, window)."""
    tasks, meta = [], []
    for label, pol in policies.items():
        for s in seeds:
            if pol == "rl":
                tasks.append(("model", str(model["path"]), model["algo"], int(s)))
            else:
                tasks.append(("baseline", pol, int(s)))
            meta.append({"label": label, "policy": pol, "seed": int(s)})
    rows = run_episodes(backend, config, tasks, workers, "train", work_root)
    return pd.DataFrame([m | r for m, r in zip(meta, rows, strict=True)])


def paired_vs(df: pd.DataFrame, label: str, ref: str, value: str = "cost") -> dict:
    """Paired comparison of `label` against `ref` on the seeds both have: mean of the
    per-seed % changes, % change of the means, wins (lower value), exact two-sided
    Wilcoxon signed-rank p (`evaluation.stats.wilcoxon`)."""
    a = df[df["label"] == label].set_index("seed")[value]
    b = df[df["label"] == ref].set_index("seed")[value]
    seeds = sorted(set(a.index) & set(b.index))
    if not seeds:
        return {"n": 0}
    x, y = a.loc[seeds].to_numpy(float), b.loc[seeds].to_numpy(float)
    per_seed = [pct_change(u, v) for u, v in zip(x, y, strict=True)]
    w = wilcoxon(x, y)
    return {
        "n": len(seeds),
        "pct_mean": float(np.nanmean(per_seed)),
        "pct_of_means": pct_change(x.mean(), y.mean()),
        "diff_mean": float((x - y).mean()),
        "wins": int((x < y).sum()),
        "p": float(w.pvalue),
        "min_attainable_p": float(w.min_attainable_p),
    }


def compare_summary(df: pd.DataFrame, ref: str = REFERENCE) -> dict:
    """Per policy label: means of the episode rows and the paired tests against `ref`
    and S0 (exploratory, unadjusted)."""
    out: dict[str, dict] = {}
    for label, g in df.groupby("label", sort=False):
        d: dict[str, Any] = {
            "n": len(g),
            "cost_mean": float(g["cost"].mean()),
            "cost_std": float(g["cost"].std(ddof=1)) if len(g) > 1 else math.nan,
        }
        d |= {f"{k}_mean": float(g[k].mean()) for k in TOTAL_KEYS if k in g}
        for k in ("interventions", "decisions", "advances", "amb_seen_share"):
            if k in g:
                d[f"{k}_mean"] = float(g[k].mean())
        d |= {c: float(g[c].mean()) for c in g.columns if c.startswith("iv_")}
        for k in PREEMPT_KEYS:
            if f"preempt_{k}" in g:
                d[f"preempt_{k}"] = int(g[f"preempt_{k}"].sum())
        if "preempt_timeouts" in g:
            d["preempt_timeout_windows"] = int((g["preempt_timeouts"] > 0).sum())
        if "amb_seen_share" in g:
            d["amb_seen_share_max"] = float(g["amb_seen_share"].max())
        d["danger_max"] = int(g["danger"].max()) if "danger" in g else 0
        d["ped_curb_wait_max"] = (
            float(g["ped_curb_wait_max"].max()) if "ped_curb_wait_max" in g else math.nan
        )
        for other in (ref, "S0"):
            if label != other and other in set(df["label"]):
                d[f"vs_{other}"] = paired_vs(df, str(label), other)
        out[str(label)] = d
    return out


def compare(
    config: Path = RL_CONFIG,
    seeds: Sequence[int] = (),
    model: Path | None = None,
    workers: int = 1,
    out_dir: Path | None = None,
    backend: EnvBackend | None = None,
    algo: str | None = None,
    allowed: Sequence[str] = ("dev_test", "selection"),
) -> Path:
    """`compare` subcommand; returns the summary json path."""
    rl_cfg = load_config(config)
    seeds = list(seeds) or seed_range(rl_cfg["seeds"]["dev_test"])[:10]
    check_seeds(seeds, rl_cfg, allowed, "compare")
    mdl = resolve_model(model, rl_cfg, algo)
    backend = backend or real_backend()
    out_dir = Path(out_dir or runs_root(rl_cfg) / "devtest")
    check_out_root(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = compare_frame(backend, config, seeds, mdl, workers, work_root=runs_root(rl_cfg) / "_work")
    df.to_csv(out_dir / "compare.csv", index=False)
    doc = provenance(config) | {
        "backend": backend.name,
        "seeds": seeds,
        "reference": REFERENCE,
        "model": {
            "path": mdl["path"].as_posix(),
            "algo": mdl["algo"],
            "model_version": mdl["manifest"].get("model_version"),
            "model_sha256": file_sha256(mdl["path"]),
        },
        "summary": compare_summary(df),
    }
    return _write_json(out_dir / "compare_summary.json", doc)


# -- full 2 h runs ---------------------------------------------------------------------
def rl_model_identity(cfg: Mapping) -> dict | None:
    """{path, model_sha256, model_version} of an `rl` scenario's `rl.model` (None for
    other controllers). config_hash holds only the model path, so the model's sha256 is
    what tells apart two policies written to the same `selected/model.zip`."""
    if cfg.get("controller") != "rl":
        return None
    path = Path(get(cfg, "rl.model", "") or "")
    if not path.is_file():
        raise ConfigError(f"rl.model: file not found: {path} (train and select one)")
    manifest = _json(path.with_name("manifest.json")) or {}
    return {
        "path": path.as_posix(),
        "model_sha256": file_sha256(path),
        "model_version": manifest.get("model_version"),
    }


def _current(run_dir: Path, cfg: dict, model: Mapping | None = None) -> bool:
    """A finished run of the same config_hash and, for the `rl` controller, made with the
    same model file (run.json `control.model_sha256` equal to `model`'s)."""
    meta = _json(run_dir / "run.json")
    if not meta or not (run_dir / "kpi_run.csv").is_file():
        return False
    if meta.get("config_hash") != config_hash(cfg):
        return False
    if model is not None:
        return (meta.get("control") or {}).get("model_sha256") == model["model_sha256"]
    return True


def kpi_configs(
    model: Path | None = None, configs: Mapping[str, Path] = KPI_CONFIGS
) -> dict[str, tuple[Path, dict]]:
    """label -> (config path, loaded scenario); `model` replaces the RL config's
    `rl.model`."""
    out = {}
    for label, path in configs.items():
        cfg = load_scenario(path)
        if model is not None and cfg.get("controller") == "rl":
            cfg.setdefault("rl", {})["model"] = Path(model).as_posix()
        out[label] = (Path(path), cfg)
    return out


def kpi_runs(
    seeds: Sequence[int],
    out_root: Path,
    configs: Mapping[str, tuple[Path, dict]],
    force: bool = False,
) -> list[dict]:
    """Run every (config, seed) with `run_scenario.run` into `out_root` (route files in
    a temporary folder); a finished run of the same config_hash (and, for `rl`, of the
    same model sha256) is kept unless `force`."""
    from traffic_twin.sim.run_scenario import run, run_dir

    check_out_root(out_root)
    done = []
    with tempfile.TemporaryDirectory(prefix="rl_kpi_demand_") as demand:
        for label, (path, cfg) in configs.items():
            model = rl_model_identity(cfg)
            for seed in seeds:
                d = run_dir(cfg, seed, out_root)
                if not force and _current(d, cfg, model):
                    done.append({"label": label, "seed": seed, "dir": d, "skipped": True})
                    continue
                run(cfg, int(seed), out_root, config_path=path, demand_dir=Path(demand))
                done.append({"label": label, "seed": seed, "dir": d, "skipped": False})
    return done


def kpi_frames(
    out_root: Path,
    configs: Mapping[str, tuple[Path, dict]],
    seeds: Sequence[int],
    occupancy: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(KPI rows: label, scenario, seed, user_type, kpi, value, with the person-weighted
    delay `all / person_delay_mean` of D29; safety rows: one per run, with the preemption
    outcome). ConfigError when an `rl` run was made with another model file than the
    scenario's `rl.model` (a stale run after a retraining or reselection)."""
    from traffic_twin.evaluation.evaluate import person_delay_mean

    kpis, safety = [], []
    for label, (_, cfg) in configs.items():
        model = rl_model_identity(cfg)
        limit = float(
            get(cfg, "safety.ped_max_wait_s", DEFAULT_CURB_LIMIT_S) or DEFAULT_CURB_LIMIT_S
        )
        for seed in seeds:
            d = Path(out_root) / f"{cfg['scenario']}_seed{seed}"
            meta = _json(d / "run.json")
            if meta is None or not (d / "kpi_run.csv").is_file():
                raise ConfigError(
                    f"{d}: missing run (run `rl.evaluate kpi` without --summary-only)"
                )
            control = meta.get("control") or {}
            if model is not None and control.get("model_sha256") != model["model_sha256"]:
                raise ConfigError(
                    f"{d}: run made with model sha256 "
                    f"{str(control.get('model_sha256'))[:12]} "
                    f"({control.get('model_version')}), not {model['model_sha256'][:12]} "
                    f"({model['model_version']}) of {model['path']}; rerun `rl.evaluate kpi`"
                    " without --summary-only (stale runs are redone)"
                )
            k = pd.read_csv(d / "kpi_run.csv")
            k = k[["user_type", "kpi", "value"]].assign(
                label=label, scenario=cfg["scenario"], seed=seed
            )
            kpis.append(k)
            trip = d / "trip.parquet"
            if trip.is_file():
                pdm = person_delay_mean(pd.read_parquet(trip), dict(occupancy))
                kpis.append(
                    pd.DataFrame(
                        [
                            {
                                "user_type": "all",
                                "kpi": "person_delay_mean",
                                "value": pdm,
                                "label": label,
                                "scenario": cfg["scenario"],
                                "seed": seed,
                            }
                        ]
                    )
                )
            val = k.set_index(["user_type", "kpi"])["value"]
            curb = float(val.get(("pedestrian", "ped_curb_wait_max"), math.nan))
            sb, health = meta.get("safety") or {}, meta.get("health") or {}
            iv = sb.get("interventions") or {}
            pre = meta.get("preemption") or {}
            safety.append(
                {
                    "label": label,
                    "scenario": cfg["scenario"],
                    "seed": seed,
                    "danger": int(sb.get("danger", 0) or 0),
                    "collisions": int(health.get("collisions", 0) or 0),
                    "teleports": int(health.get("teleports", 0) or 0),
                    "drain_timeout": bool(health.get("drain_timeout", False)),
                    "ped_curb_wait_max": curb,
                    "curb_limit_s": limit,
                    "interventions": int(sum(int(v) for v in iv.values())),
                    **{f"iv_{key}": int(iv.get(key, 0)) for key in INTERVENTION_KEYS},
                    "depart_delay_mean": health.get("depart_delay_mean"),
                    "depart_delay_max": health.get("depart_delay_max"),
                    "preempt_enabled": bool(pre.get("enabled", False)),
                    **{f"preempt_{key}": int(pre.get(key, 0) or 0) for key in PREEMPT_KEYS},
                    "code_version": meta.get("code_version"),
                    "config_hash": meta.get("config_hash"),
                    "demand_hash": meta.get("demand_hash"),
                    "model_version": control.get("model_version"),
                    "decisions": control.get("decisions"),
                    "advances": control.get("advances"),
                }
            )
    return pd.concat(kpis, ignore_index=True), pd.DataFrame(safety)


def safety_violations(safety: pd.DataFrame) -> list[str]:
    """Runs breaking a Phase 3 STOP bound: danger > 0, collisions > 0, curb wait above
    the scenario's limit."""
    out = []
    for r in safety.itertuples(index=False):
        why = []
        if r.danger > 0:
            why.append(f"danger {r.danger}")
        if r.collisions > 0:
            why.append(f"collisions {r.collisions}")
        if math.isfinite(r.ped_curb_wait_max) and r.ped_curb_wait_max > r.curb_limit_s:
            why.append(f"curb wait {r.ped_curb_wait_max:g} s > {r.curb_limit_s:g} s")
        if why:
            out.append(f"{r.label} seed {r.seed}: " + ", ".join(why))
    return out


def kpi_summary(kpis: pd.DataFrame, safety: pd.DataFrame, ref: str = REFERENCE) -> dict:
    """Headline KPIs per label (mean, std, n) with paired tests against `ref` and S0,
    and the safety block (exploratory)."""
    table: dict[str, dict] = {}
    labels = list(dict.fromkeys(kpis["label"]))
    for user, kpi in HEADLINE:
        sub = kpis[(kpis["user_type"] == user) & (kpis["kpi"] == kpi)].rename(
            columns={"value": "v"}
        )
        if sub.empty:
            continue
        entry: dict[str, dict] = {}
        for label in labels:
            v = sub[sub["label"] == label]["v"].astype(float)
            if v.empty:
                continue
            e: dict[str, Any] = {
                "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if len(v) > 1 else math.nan,
                "n": len(v),
            }
            for other in (ref, "S0"):
                if label != other and other in labels:
                    e[f"vs_{other}"] = paired_vs(sub, label, other, "v")
            entry[label] = e
        table[f"{user}/{kpi}"] = entry
    per_label = {}
    for label, g in safety.groupby("label", sort=False):
        per_label[str(label)] = {
            "runs": len(g),
            "danger_max": int(g["danger"].max()),
            "collisions_max": int(g["collisions"].max()),
            "ped_curb_wait_max": float(g["ped_curb_wait_max"].max()),
            "interventions_mean": float(g["interventions"].mean()),
            **{c: float(g[c].mean()) for c in g.columns if c.startswith("iv_")},
            "teleports_total": int(g["teleports"].sum()),
            "drain_timeouts": int(g["drain_timeout"].sum()),
            "depart_delay_mean": float(pd.to_numeric(g["depart_delay_mean"]).mean()),
            **{
                f"preempt_{k}": int(g[f"preempt_{k}"].sum())
                for k in PREEMPT_KEYS
                if f"preempt_{k}" in g
            },
            **(
                {"preempt_enabled": bool(g["preempt_enabled"].any())}
                if "preempt_enabled" in g
                else {}
            ),
        }
    violations = safety_violations(safety)
    return {
        "kpis": table,
        "safety": per_label,
        "violations": violations,
        "safety_ok": not violations,
        # the code versions of the runs themselves (the top-level code_version is the
        # summarizing code's); keyed `code_version` like every provenance field
        "runs": {
            "code_version": sorted({str(v) for v in safety["code_version"]})
            if "code_version" in safety
            else []
        },
    }


def kpi(
    config: Path = RL_CONFIG,
    seeds: Sequence[int] = (),
    model: Path | None = None,
    out_root: Path | None = None,
    force: bool = False,
    summary_only: bool = False,
    configs: Mapping[str, Path] = KPI_CONFIGS,
    allowed: Sequence[str] = ("dev_test", "selection"),
) -> dict:
    """`kpi` subcommand; returns the summary (also written to kpi_summary.json)."""
    rl_cfg = load_config(config)
    seeds = list(seeds) or seed_range(rl_cfg["seeds"]["dev_test"])[:10]
    check_seeds(seeds, rl_cfg, allowed, "kpi")
    out_root = Path(out_root or runs_root(rl_cfg) / "devtest")
    check_out_root(out_root)
    cfgs = kpi_configs(model, configs)
    if not summary_only:
        kpi_runs(seeds, out_root, cfgs, force)
    occupancy = load_config(EVALUATION_CONFIG)["occupancy"]
    kpis, safety = kpi_frames(out_root, cfgs, seeds, occupancy)
    kpis.to_csv(out_root / "kpi.csv", index=False)
    safety.to_csv(out_root / "kpi_safety.csv", index=False)
    models = {k: m for k, (_, c) in cfgs.items() if (m := rl_model_identity(c)) is not None}
    summary = provenance(config) | {
        "seeds": seeds,
        "reference": REFERENCE,
        "configs": {k: p.as_posix() for k, (p, _) in cfgs.items()},
        "models": models,  # the model every rl run was checked against (kpi_frames)
        **kpi_summary(kpis, safety),
    }
    _write_json(out_root / "kpi_summary.json", summary)
    return summary


# -- reproducibility -------------------------------------------------------------------
def read_monitor(path: Path) -> pd.DataFrame:
    """VecMonitor CSV (first line is a JSON header) with the cumulative step count."""
    df = pd.read_csv(path, skiprows=1)
    df["steps"] = df["l"].cumsum()
    return df


def repro_check(a: Path, b: Path) -> dict:
    """Same policy parameters (best and final), episode returns and lengths (monitor r,
    l; the wall-time column t is ignored) and validation costs."""
    ma, mb = read_manifest(a), read_manifest(b)
    checks: dict[str, bool] = {}
    for key in ("best_policy_sha256", "final_policy_sha256", "best_step", "steps_done"):
        checks[key] = ma.get(key) is not None and ma.get(key) == mb.get(key)
    try:
        ra, rb = read_monitor(Path(a) / "monitor.csv"), read_monitor(Path(b) / "monitor.csv")
        checks["monitor_r_l"] = ra[["r", "l"]].equals(rb[["r", "l"]])
    except (OSError, KeyError, pd.errors.ParserError):
        checks["monitor_r_l"] = False
    try:
        va = pd.read_csv(Path(a) / "validation.csv")
        vb = pd.read_csv(Path(b) / "validation.csv")
        checks["validation_cost"] = va[["step", "cost"]].equals(vb[["step", "cost"]])
    except (OSError, KeyError):
        checks["validation_cost"] = False
    return {
        "a": Path(a).as_posix(),
        "b": Path(b).as_posix(),
        "same": all(checks.values()),
        "checks": checks,
        "n_envs": [ma.get("n_envs"), mb.get("n_envs")],
        "threads": [ma.get("threads"), mb.get("threads")],
        "code_version": [ma.get("code_version"), mb.get("code_version")],
    }


# -- S6 criterion (D60) ----------------------------------------------------------------
def s6_verdict(manifests: Sequence[Mapping], rl_cfg: Mapping, rl_sha: str | None = None) -> dict:
    """D60: in >= 2 of the 3 DQN runs of rl.yaml `train.s6.dqn` (seeds, steps), the best
    validation cost is >= 5 % below both the S0 and the always-stay costs on the same
    windows. PASS / FAIL, or INCOMPLETE while missing runs could still change it."""
    s6 = ((rl_cfg.get("train") or {}).get("s6") or {}).get("dqn") or {}
    seeds, steps = [int(s) for s in s6.get("seeds", [])], int(s6.get("steps", 0))
    rows, missing = [], []
    for seed in seeds:
        found = [
            m
            for m in manifests
            if m.get("algo") == "dqn"
            and int(m.get("seed", -1)) == seed
            and int(m.get("steps", -1)) == steps
            and m.get("backend", "sumo") == "sumo"
            and (rl_sha is None or m.get("rl_yaml_sha256") == rl_sha)
        ]
        if not found:
            missing.append(seed)
            continue
        m = sorted(found, key=lambda m: str(m.get("run")))[0]
        base = m.get("baselines") or {}
        best = float(m.get("best_val_cost", math.nan))
        rows.append(
            {
                "seed": seed,
                "run": m.get("run"),
                "best_val_cost": best,
                **{f"vs_{r}": pct_change(best, base.get(r, math.nan)) for r in S6_REFERENCES},
                "pass": bool(s6_run_pass(best, base)),
            }
        )
    n_pass = sum(r["pass"] for r in rows)
    if n_pass >= S6_MIN_PASS:
        verdict = "PASS"
    elif n_pass + len(missing) < S6_MIN_PASS:
        verdict = "FAIL"
    else:
        verdict = "INCOMPLETE"
    return {
        "verdict": verdict,
        "n_pass": n_pass,
        "n_runs": len(rows),
        "required": S6_MIN_PASS,
        "of": len(seeds),
        "steps": steps,
        "margin_pct": S6_MARGIN_PCT,
        "references": list(S6_REFERENCES),
        "runs": rows,
        "missing_seeds": missing,
    }


# -- docs ------------------------------------------------------------------------------
def _f(x: Any, nd: int = 1, unit: str = "") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    if not math.isfinite(v):
        return "–"
    return f"{v:,.{nd}f}{unit}"


def _pct(x: Any, nd: int = 1) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if not math.isfinite(v) else f"{v:+.{nd}f} %"


def _p(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if not math.isfinite(v) else (f"{v:.3f}" if v >= 0.001 else "< 0.001")


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _missing(what: str, task: str) -> list[str]:
    return [f"_Not available yet: {what}. Run `{task}`._", ""]


def _rel(path: Path, docs: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), Path(docs).resolve().parent)).as_posix()


def collect_runs(root: Path) -> list[dict]:
    """Manifests of the training runs under `root` with their validation and monitor
    tables (None when absent)."""
    out = []
    for d in find_runs(root):
        m = read_manifest(d)
        m["validation"] = (
            pd.read_csv(d / "validation.csv") if (d / "validation.csv").is_file() else None
        )
        m["monitor"] = read_monitor(d / "monitor.csv") if (d / "monitor.csv").is_file() else None
        out.append(m)
    return out


def _hold_text(max_hold: float | None) -> str:
    return f"{max_hold:g} s" if max_hold is not None else "`preemption.max_hold_s`"


def _env_section(rl_cfg: Mapping, max_hold: float | None = None) -> list[str]:
    obs, rew = rl_cfg.get("observation") or {}, rl_cfg.get("reward") or {}
    w, pen = rew.get("weights") or {}, rew.get("ped_penalty") or {}
    ep = (rl_cfg.get("episode") or {}).get("train") or {}
    rand = rl_cfg.get("randomize") or {}
    dec = rl_cfg.get("decision") or {}
    amb, inc = rand.get("ambulances") or {}, rand.get("incidents") or {}
    dts = ", ".join(f"{k} {v:g}" for k, v in (rand.get("day_types") or {}).items())
    return [
        "## 1. Environment",
        "",
        "Single agent on one junction (spec 10.1): `rl.env.TrafficEnv`, a Gymnasium env over "
        "`SimSession` (libsumo). The Safety Guard and the deterministic ambulance preemption "
        "stay in force and take priority over the agent (10.5). Preemption is bounded: a "
        f"request is dropped after {_hold_text(max_hold)} without progress and control "
        "returns to the agent (section 10).",
        "",
        "- **Action** (D55): `Discrete(2)`, 0 = stay, 1 = advance to the cycle successor, under "
        "`safety.stage_order: strict`. Under strict order the guard allows exactly the current "
        'stage and its successor, so this is spec 10.2 "next green among the allowed phases"; '
        "yellow and all-red come from `SignalPlan`.",
        f"- **Decision clock** (D56, `rl.clock`): a decision every {dec.get('interval_s', 5)} s at "
        "most, only out of transition, after the minimum green and "
        + (
            "never during an active preemption"
            if dec.get("skip_during_preemption", True)
            else "also during preemption"
        )
        + ". Between decisions the env holds the stage "
        "(semi-MDP). The controller `rl` uses the same clock and features.",
        f"- **Observation** `{obs.get('version')}` (D57, `rl.features`): 74 values in [0, 1] — per "
        "TLS lane halting, vehicles (per 7.5 m) and max wait "
        f"(/{obs.get('wait_norm_s')} s) (ASSUMED ideal lane detection: a camera or loops on "
        "every signal-controlled lane, read from the simulator; S1 sees only its loops); "
        f"pending insertions per arm (/{obs.get('pending_norm')}, "
        "ASSUMED upstream detection); stage one-hot, in-transition, time in stage "
        f"(/{obs.get('tis_norm_s')} s), red time per stage (/{obs.get('red_norm_s')} s); per "
        "crossing waiting count, curb wait and shield slack (ASSUMED ideal pedestrian "
        "detection, as for S2); ambulance presence, pending, "
        f"distance bins {obs.get('amb_dist_bins_m')} m, arm and preemption flag (ASSUMED V2I); "
        f"per bus approach presence, distance and delay bin {obs.get('bus_delay_bins_s')} s "
        "(time-loss proxy, no timetable); time of day (sin, cos).",
        f"- **Reward** `{rew.get('version')}` (D58, `rl.reward`, spec 10.3): every simulated "
        f"second c = {w.get('car')}·cars + {w.get('bus')}·buses + {w.get('pedestrian')}·pedestrians"
        f" + {w.get('ambulance')}·ambulances waiting (persons; pending insertions count as "
        f"waiting) + {pen.get('lambda')}·1(longest curb wait > {pen.get('threshold_s')} s); "
        f"R = −{rew.get('scale')} · Σc over the decision interval. The episode **cost** Σc used "
        "below is in weighted person-seconds.",
        f"- **Episodes** (D59): training windows of {ep.get('window_s')} s starting on a random "
        f"minute inside 07:30-09:30 after {ep.get('warmup_s')} s of "
        f"`{ep.get('warmup_controller')}` warm-up, truncated at the window end (no drain); "
        "full mode 07:30-09:30 truncated at 7,200 s (timing).",
        f"- **Diversity** (10.4): demand multiplier U{tuple(rand.get('multiplier', []))}; day "
        f"types {dts} (ASSUMED ratio profiles, D44); ambulances 0/1/2 with p "
        f"{amb.get('count_p')}; a lane incident with p {inc.get('p')} for "
        f"{inc.get('duration_s')} s (D45).",
        "",
    ]


def _prereg_section(rl_cfg: Mapping, sha: str | None) -> list[str]:
    seeds = rl_cfg.get("seeds") or {}
    tr = rl_cfg.get("train") or {}
    s6 = tr.get("s6") or {}
    lines = [
        "## 2. Pre-registration",
        "",
        f"`configs/rl.yaml` (sha256 `{(sha or '')[:12]}`), pre-registered "
        f"**{rl_cfg.get('preregistered')}**, before any training; it changes only through dated "
        f"`changes` entries ({len(rl_cfg.get('changes') or [])} so far). The Part 3-4 seeds of "
        "`configs/evaluation.yaml` (1-10, 101-103, 201-207) are never run in Part 6.",
        "",
    ]
    lines += _table(
        ["Seed set", "Range", "Use"],
        [
            [
                "train",
                "{}-{}".format(*seeds.get("train", ["?", "?"])),
                "training episodes (worker i plays i, i + n, ...)",
            ],
            [
                "selection",
                "{}-{}".format(*seeds.get("selection", ["?", "?"])),
                f"validation every {tr.get('eval_every_steps')} steps, best checkpoint, timing",
            ],
            [
                "burned",
                "{}-{}".format(*seeds.get("burned", ["?", "?"])),
                "seen by a prototype, never used",
            ],
            [
                "dev_test",
                "{}-{}".format(*seeds.get("dev_test", ["?", "?"])),
                "exploratory comparison (Part 6 uses the first 10)",
            ],
        ],
    )
    lines += [
        "",
        "**Budget (S6):** "
        + "; ".join(
            f"{a.upper()} seeds {v.get('seeds')} × {v.get('steps'):,} decision steps"
            for a, v in s6.items()
        )
        + f"; {tr.get('n_envs')} SubprocVecEnv workers, "
        f"torch threads {tr.get('torch_threads')}, checkpoints every "
        f"{tr.get('checkpoint_every_steps')} steps.",
        "",
    ]
    hp_rows = []
    for algo in ALGOS:
        hp = rl_cfg.get(algo) or {}
        hp_rows.append([algo.upper(), ", ".join(f"{k} {v}" for k, v in hp.items())])
    lines += _table(["Algorithm", "Hyperparameters (SB3, from the rl_train prototype)"], hp_rows)
    return [*lines, ""]


def _clock(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600 % 24:02d}:{t // 60 % 60:02d}"


def full_episode_text(ep: Mapping | None) -> str:
    """'07:30-09:30, no warm-up, truncated at 7,200 s, no drain' from
    `full_episode_times` (None: generic wording)."""
    if not ep:
        return "full-mode episodes (rl.yaml `episode.full`), no drain"
    begin, warm, win = float(ep["begin_s"]), int(ep["warmup_s"]), int(ep["window_s"])
    warmup = (
        "no warm-up"
        if warm == 0
        else f"{warm:,} s of `{ep.get('warmup_controller')}` warm-up from {_clock(begin)}"
    )
    return (
        f"{_clock(begin + warm)}-{_clock(begin + warm + win)}, {warmup}, truncated at "
        f"{win:,} s, no drain"
    )


def _timing_section(
    timing_doc: Mapping | None,
    figs: Mapping[str, Path],
    docs: Path,
    episode: Mapping | None = None,
) -> list[str]:
    """`episode`: the full-mode times when timing.json predates its `episode` key."""
    lines = ["## 3. Episode timing (S6 measurement)", ""]
    if not timing_doc:
        return lines + _missing("episode timing", "tasks.py rl-timing")
    ep = timing_doc.get("episode") or episode
    lines.append(
        f"Full episodes ({full_episode_text(ep)}) "
        f"on selection seeds {timing_doc.get('seeds')}, measured in one process on "
        f"{timing_doc.get('cpu_count')} logical CPUs ({timing_doc.get('platform')}), code "
        f"`{timing_doc.get('code_version')}`. Baselines decide every second through "
        "`run_baseline` (timed as a whole); env-action policies are split into reset, env "
        "steps (SUMO, reward, features) and NN inference."
    )
    lines.append("")
    summary = timing_doc.get("summary") or {}
    split = [
        k
        for k in ("env_sim_s", "env_features_s", "env_reward_s")
        if any(f"{k}_mean" in s for s in summary.values())
    ]
    names = {"env_sim_s": "SUMO s", "env_features_s": "Features s", "env_reward_s": "Reward s"}
    rows = []
    for pol, s in summary.items():
        dec = s.get("decisions_mean") or 0
        rows.append(
            [
                pol,
                s.get("episodes"),
                _f(s.get("wall_s_mean"), 2),
                _f(s.get("wall_s_max"), 2),
                _f(dec, 0) if dec > 0 else "every s (controller)",
                _f(s.get("reset_s_mean"), 2),
                _f(s.get("step_s_mean"), 2),
                *[_f(s.get(f"{k}_mean"), 2) for k in split],
                _f(s.get("nn_s_mean"), 3),
                _f(s.get("decisions_per_s_mean"), 0),
            ]
        )
    header = [
        "Policy",
        "Episodes",
        "Wall s (mean)",
        "Wall s (max)",
        "Decisions",
        "Reset s",
        "Env steps s",
        *[names[k] for k in split],
        "NN s",
        "Decisions/s",
    ]
    lines += _table(header, rows)
    first = next(
        (
            e.get("construct_s")
            for e in timing_doc.get("episodes", [])
            if e.get("construct_s") is not None
        ),
        None,
    )
    if first is not None:
        lines += ["", f"Env construction in a fresh process: {_f(first, 2)} s."]
    scaling = timing_doc.get("scaling") or []
    if scaling:
        lines += [
            "",
            f"SubprocVecEnv throughput (train-mode windows, random policy, "
            f"{timing_doc.get('scaling_steps')} decision steps):",
            "",
        ]
        lines += _table(
            ["n_envs", "Steps/s", "Start-up s", "Episodes finished"],
            [
                [r["n_envs"], _f(r["steps_per_s"], 0), _f(r["startup_s"], 1), r["episodes"]]
                for r in scaling
            ],
        )
        best = max(scaling, key=lambda r: r["steps_per_s"])
        lines += [
            "",
            f"**Budget:** at {best['steps_per_s']:,.0f} env steps/s ({best['n_envs']} "
            f"envs, no learning), 200k decision steps take about "
            f"{200_000 / best['steps_per_s'] / 60:.0f} min of simulation; learning and "
            "validation add to it (section 4 gives the measured training rates).",
        ]
    if "episode_timing" in figs:
        lines += ["", f"![Episode timing]({_rel(figs['episode_timing'], docs)})"]
    return [*lines, ""]


def _runs_section(runs: Sequence[Mapping], sha: str | None) -> list[str]:
    lines = ["## 4. Trainings", ""]
    current = [m for m in runs if m.get("backend") == "sumo"]
    if not current:
        return lines + _missing("training runs", "tasks.py rl-train")
    lines.append(
        "Each row is one `rl.train` run folder (`data/rl_runs/<run>/manifest.json`). "
        "Costs are mean episode Σc on the 10 selection windows; changes are against the "
        "baselines on the same windows."
    )
    lines.append("")
    rows = []
    for m in current:
        vs = m.get("best_vs") or {}
        stale = "" if sha is None or m.get("rl_yaml_sha256") == sha else " (other rl.yaml)"
        rows.append(
            [
                f"`{m.get('run')}`{stale}",
                str(m.get("algo", "")).upper(),
                m.get("seed"),
                f"{m.get('steps_done', m.get('steps')):,}",
                m.get("n_envs"),
                _f(m.get("wall_s", math.nan) / 60, 1),
                _f(m.get("steps_per_s"), 0),
                _f(m.get("eval_s"), 0),
                m.get("best_step"),
                _f(m.get("best_val_cost"), 0),
                _pct(vs.get("S0")),
                _pct(vs.get("S1")),
                _pct(vs.get("S2")),
                _pct(vs.get("stay")),
                m.get("code_version"),
            ]
        )
    lines += _table(
        [
            "Run",
            "Algo",
            "Seed",
            "Steps",
            "Envs",
            "Wall min",
            "Train steps/s",
            "Validation s",
            "Best step",
            "Best cost",
            "vs S0",
            "vs S1",
            "vs S2",
            "vs stay",
            "Code",
        ],
        rows,
    )
    base = current[0].get("baselines") or {}
    if base:
        lines += [
            "",
            "Baselines on the selection windows: "
            + ", ".join(f"{k} {_f(v, 0)}" for k, v in base.items())
            + ".",
        ]
    return [*lines, ""]


def _verdict_section(v: Mapping) -> list[str]:
    lines = [
        "## 6. S6 criterion",
        "",
        f"Pre-registered (D60): in at least {v['required']} of the {v['of']} DQN runs "
        f"({v['steps']:,} steps), the best validation cost is at least "
        f"{v['margin_pct']:g} % below both the S0 and the always-stay costs on the same "
        'selection windows (S6 "an agent that learns on short episodes").',
        "",
    ]
    if v["runs"]:
        lines += _table(
            ["Seed", "Run", "Best cost", "vs S0", "vs stay", "Pass"],
            [
                [
                    r["seed"],
                    f"`{r['run']}`",
                    _f(r["best_val_cost"], 0),
                    _pct(r["vs_S0"]),
                    _pct(r["vs_stay"]),
                    "yes" if r["pass"] else "no",
                ]
                for r in v["runs"]
            ],
        )
        lines.append("")
    miss = f"; missing seeds {v['missing_seeds']}" if v["missing_seeds"] else ""
    lines += [f"**Verdict: {v['verdict']}** ({v['n_pass']} of {v['n_runs']} runs pass{miss}).", ""]
    return lines


def _preempt_cell(s: Mapping) -> str:
    """'served/requests (timeouts)' of a summary entry; '–' when not recorded."""
    if s.get("preempt_requests") is None:
        return "–"
    if s.get("preempt_enabled") is False:
        return "off"
    return (
        f"{s.get('preempt_served', 0)}/{s.get('preempt_requests', 0)} "
        f"({s.get('preempt_timeouts', 0)})"
    )


def _compare_section(cmp: Mapping | None, figs: Mapping[str, Path], docs: Path) -> list[str]:
    lines = ["## 7. Dev-test window comparison (exploratory)", ""]
    if not cmp:
        return lines + _missing("the dev-test comparison", "tasks.py rl-eval")
    model = cmp.get("model") or {}
    lines.append(
        f"Selected model `{model.get('model_version')}` and the baselines on the same "
        f"{len(cmp.get('seeds', []))} train-mode windows (dev-test seeds {cmp.get('seeds')}, "
        "never used for training or selection). Cost = episode Σc; changes are means of the "
        "per-window % changes; p = exact two-sided Wilcoxon signed-rank test against S1, "
        "unadjusted (exploratory, no acceptance)."
    )
    lines.append("")
    rows = []
    for label, s in (cmp.get("summary") or {}).items():
        v1, v0 = s.get("vs_S1") or {}, s.get("vs_S0") or {}
        rows.append(
            [
                label,
                _f(s.get("cost_mean"), 0),
                _pct(v0.get("pct_mean")),
                _pct(v1.get("pct_mean")),
                f"{v1.get('wins', '–')}/{v1.get('n', '–')}" if v1 else "–",
                _p(v1.get("p")),
                _f(s.get("interventions_mean"), 1),
                _f(s.get("ped_curb_wait_max"), 0),
                _preempt_cell(s),
            ]
        )
    lines += _table(
        [
            "Policy",
            "Cost (mean)",
            "vs S0",
            "vs S1",
            "Wins vs S1",
            "p vs S1",
            "Interventions",
            "Curb max s",
            "Preemptions served/requests (timeouts)",
        ],
        rows,
    )
    if "devtest_compare" in figs:
        lines += ["", f"![Dev-test windows]({_rel(figs['devtest_compare'], docs)})"]
    return [*lines, ""]


def _kpi_section(ks: Mapping | None) -> list[str]:
    lines = ["## 8. Full 2 h runs (exploratory KPIs)", ""]
    if not ks:
        return lines + _missing("the full 2 h runs", "tasks.py rl-eval")
    lines.append(
        "`run_scenario` runs of "
        + ", ".join(f"{k} `{v}`" for k, v in ks.get("configs", {}).items())
        + " "
        f"on dev-test seeds {ks.get('seeds')} (07:30-09:30 plus the drain, KPI v3, same "
        "demand_hash). Means over the seeds; RL vs S1 is the paired mean difference with the "
        "exact Wilcoxon p (exploratory, unadjusted). The RL objective is the env cost, not KPI "
        "v3. A-06 (seeds 1-10) is Part 8."
    )
    ident = []
    for label, m in (ks.get("models") or {}).items():
        ident.append(
            f"{label} runs made with model `{m.get('model_version')}` (sha256 "
            f"`{str(m.get('model_sha256'))[:12]}`, checked against every run.json)"
        )
    run_versions = (ks.get("runs") or {}).get("code_version") or []
    if run_versions:
        ident.append("code versions of the runs: " + ", ".join(f"`{v}`" for v in run_versions))
    if ident:
        text = "; ".join(ident)
        lines += ["", text[0].upper() + text[1:] + "."]
    lines.append("")
    labels = list(ks.get("configs", {}))
    rows = []
    for name, entry in (ks.get("kpis") or {}).items():
        rl = (entry.get("RL") or {}).get("vs_S1") or {}
        rows.append(
            [
                f"`{name}`",
                *[_f((entry.get(lab) or {}).get("mean"), 1) for lab in labels],
                _f(rl.get("diff_mean"), 1),
                _p(rl.get("p")),
            ]
        )
    lines += _table(["KPI", *labels, "RL − S1", "p"], rows)
    return [*lines, ""]


def _safety_section(ks: Mapping | None, cmp: Mapping | None) -> list[str]:
    lines = ["## 9. Safety", ""]
    if not ks and not cmp:
        return lines + _missing("safety results", "tasks.py rl-eval")
    if ks:
        lines.append(
            "Full 2 h runs: dangerous audit findings, collisions, longest curb wait "
            "(A-04 bound 90 s) and Safety Guard interventions per run (mean). "
            "`zero_interventions` stays [fixed, actuated] (D62)."
        )
        lines.append("")
        rows = []
        for label, s in (ks.get("safety") or {}).items():
            rows.append(
                [
                    label,
                    s.get("runs"),
                    s.get("danger_max"),
                    s.get("collisions_max"),
                    _f(s.get("ped_curb_wait_max"), 0),
                    _f(s.get("interventions_mean"), 1),
                    *[_f(s.get(f"iv_{k}"), 1) for k in INTERVENTION_KEYS],
                    s.get("teleports_total"),
                    s.get("drain_timeouts"),
                    _preempt_cell(s),
                ]
            )
        lines += _table(
            [
                "Scenario",
                "Runs",
                "Danger",
                "Collisions",
                "Curb max s",
                "Interventions",
                *INTERVENTION_KEYS,
                "Teleports",
                "Drain timeouts",
                "Preemptions served/requests (timeouts)",
            ],
            rows,
        )
        viol = ks.get("violations") or []
        lines += [
            "",
            "**Safety bounds:** "
            + (
                "all runs within danger 0, collisions 0, curb wait ≤ 90 s."
                if not viol
                else "VIOLATED — " + "; ".join(viol)
            ),
            "",
        ]
    if cmp:
        rl = (cmp.get("summary") or {}).get("RL") or {}
        if rl:
            lines += [
                f"Dev-test windows: the RL policy triggered {_f(rl.get('interventions_mean'), 1)} "
                "guard interventions per 30-min window ("
                + ", ".join(f"{k} {_f(rl.get(f'iv_{k}'), 1)}" for k in INTERVENTION_KEYS)
                + f"), danger max {rl.get('danger_max')}, curb wait max "
                f"{_f(rl.get('ped_curb_wait_max'), 0)} s.",
                "",
            ]
    return lines


def _amb_section(
    cmp: Mapping | None,
    runs: Sequence[Mapping],
    ks: Mapping | None = None,
    max_hold: float | None = None,
) -> list[str]:
    hold = _hold_text(max_hold)
    lines = [
        "## 10. Ambulance in the state (10.5)",
        "",
        "Preemption is deterministic and has priority while it is active (10.5): decisions are "
        "skipped during it (D56), and pending detection triggers it before the ambulance "
        "reaches the approach. It is **bounded, not an unconditional guarantee**: a request is "
        f"dropped after {hold} without 1 m of progress (`preemption.max_hold_s`, D24), for "
        "example when the ambulance is stuck in a queue the policy built, and control then "
        "returns to the agent with the ambulance still present. The ambulance features are "
        "therefore ~0 at decision points only while every preemption is served.",
        "",
    ]
    summary = (cmp or {}).get("summary") or {}
    if summary:
        if any(s.get("preempt_requests") is not None for s in summary.values()):
            lines += [
                "Dev-test windows (train-mode, 0-2 random ambulances each): preemption outcome "
                "summed over the windows (requests neither served nor timed out were still "
                "active when the window was truncated) and the share of decisions with an "
                "ambulance present.",
                "",
            ]
            lines += _table(
                [
                    "Policy",
                    "Requests",
                    "Served",
                    "Timeouts",
                    "Windows with a timeout",
                    "Ambulance share (mean)",
                    "Ambulance share (max)",
                ],
                [
                    [
                        label,
                        s.get("preempt_requests", "–"),
                        s.get("preempt_served", "–"),
                        s.get("preempt_timeouts", "–"),
                        f"{s.get('preempt_timeout_windows', '–')}/{s.get('n', '–')}",
                        f"{_f(100 * s.get('amb_seen_share_mean', math.nan), 2)} %",
                        f"{_f(100 * s.get('amb_seen_share_max', math.nan), 2)} %",
                    ]
                    for label, s in summary.items()
                ],
            )
            lines.append("")
        else:
            rl = summary.get("RL") or {}
            lines += [
                "Dev-test windows, selected model: share of decisions with an ambulance present "
                f"{_f(100 * rl.get('amb_seen_share_mean', math.nan), 2)} % (preemption "
                "outcomes not recorded in this compare.csv; rerun `tasks.py rl-eval`).",
                "",
            ]
    safety = (ks or {}).get("safety") or {}
    if safety and any(s.get("preempt_requests") is not None for s in safety.values()):
        lines.append(
            "Full 2 h runs (one fixed N→S ambulance per run): "
            + ", ".join(f"{label} {_preempt_cell(s)}" for label, s in safety.items())
            + " (served/requests (timeouts))."
        )
        lines.append("")
    vals, unrecorded = [], []
    for m in runs:
        v = m.get("validation")
        if v is not None and not v.empty and "amb_seen_share" in v:
            best = v[v["step"] == m.get("best_step")]
            if not best.empty:
                b = best.iloc[0]
                pre = (
                    f", preemptions {int(b['preempt_served'])}/{int(b['preempt_requests'])} "
                    f"({int(b['preempt_timeouts'])} timeouts)"
                    if "preempt_requests" in best
                    else ""
                )
                if not pre:
                    unrecorded.append(str(m["run"]))
                vals.append(
                    f"`{m['run']}` best checkpoint on the selection windows: ambulance share "
                    f"{_f(100 * b['amb_seen_share'], 2)} %{pre}"
                )
    lines += [f"- {v}" for v in vals]
    if unrecorded:
        lines += [
            "",
            "The validation files of "
            + ", ".join(f"`{r}`" for r in unrecorded)
            + " predate the preemption columns, so their preemption outcomes on the "
            "selection windows were not recorded (new trainings record them).",
        ]
    if not summary and not vals:
        lines.append("_Not available yet._")
    lines += [
        "",
        "No earlier ambulance warning is given to the agent: it would change S2 and A-05 (Q13).",
    ]
    return [*lines, ""]


def _repro_section(runs: Sequence[Mapping], repro: Mapping | None) -> list[str]:
    lines = [
        "## 11. Reproducibility",
        "",
        "Every run records its code version (`-dirty` for an uncommitted tree), the "
        "rl.yaml sha256, seeds, n_envs, torch threads and library versions. Bitwise "
        "reproducibility holds only at the same `n_envs`, one torch thread and the pinned "
        "versions (D61); runs are compared on the policy parameters (zip files carry "
        "timestamps). After the commit, `tasks.py part6` from the clean tree must "
        "reproduce the run manifests exactly (D65).",
        "",
    ]
    sumo = [m for m in runs if m.get("backend") == "sumo"]
    if sumo:
        versions = sumo[0].get("versions") or {}
        lines.append("Versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()) + ".")
        lines.append(
            "Code versions of the runs: "
            + ", ".join(sorted({str(m.get("code_version")) for m in sumo}))
            + "."
        )
        lines.append("")
    if repro:
        checks = ", ".join(
            f"{k} {'same' if v else 'DIFFERENT'}" for k, v in repro["checks"].items()
        )
        lines += [
            f"Determinism check (`rl.evaluate repro`, n_envs {repro.get('n_envs')}): "
            f"**{'identical' if repro.get('same') else 'NOT identical'}** ({checks}).",
            "",
        ]
    else:
        lines += _missing("the determinism check", "rl.evaluate repro RUN_A RUN_B")
    return lines


def _fallback_status(cmp: Mapping | None, ks: Mapping | None, n_runs: int | None = None) -> str:
    """The 10.6 status sentence: the dev-test window result against S1 with its caveats,
    then the S2 comparison and the full-run KPIs where RL is worse than S1 (p < 0.05) and
    the guard interventions."""
    summary = (cmp or {}).get("summary") or {}
    rl = summary.get("RL") or {}
    v1 = rl.get("vs_S1") or {}
    if not v1.get("n"):
        return "The dev-test comparison is not available yet."
    picked = f", the model picked from {n_runs} runs on the selection seeds" if n_runs else ""
    caveat = (
        f"(p {_p(v1.get('p'))}, exploratory and unadjusted; env cost only, on {v1['n']} "
        f"30-min windows{picked})"
    )
    if v1.get("pct_mean", math.nan) < 0 and v1.get("p", 1.0) < 0.05:
        parts = [
            "On the dev-test windows the RL policy has a lower env cost than S1 (actuated), "
            f"{_pct(v1.get('pct_mean'))} per window {caveat}."
        ]
    else:
        parts = [
            "On the dev-test windows the RL policy does not beat S1 (actuated) "
            f"({_pct(v1.get('pct_mean'))} per window {caveat}); so far the rules S2 stay the "
            "main solution."
        ]
    s2 = summary.get("S2") or {}
    if s2.get("cost_mean") and rl.get("cost_mean") is not None:
        vs_s2 = pct_change(rl["cost_mean"], s2["cost_mean"])
        parts.append(f"Against S2 (rules) its window cost is {_pct(vs_s2)} (change of the means).")
    if ks:
        worse, behind_s2 = [], []
        for name, entry in (ks.get("kpis") or {}).items():
            r = entry.get("RL") or {}
            d = r.get("vs_S1") or {}
            if d.get("n") and d.get("diff_mean", 0) > 0 and d.get("p", 1.0) < 0.05:
                worse.append(f"`{name}` {d['diff_mean']:+.1f} (p {_p(d.get('p'))})")
            s2k = (entry.get("S2") or {}).get("mean")
            if s2k is not None and r.get("mean") is not None and r["mean"] > s2k:
                behind_s2.append(f"`{name}` {r['mean']:.1f} vs {s2k:.1f}")
        if worse:
            parts.append("In the full 2 h runs it is worse than S1 on " + ", ".join(worse) + ".")
        if behind_s2:
            parts.append("It stays behind S2 on " + ", ".join(behind_s2) + ".")
        sf = ks.get("safety") or {}
        if "RL" in sf:
            others = ", ".join(
                f"{lab} {_f(s.get('interventions_mean'), 1)}"
                for lab, s in sf.items()
                if lab != "RL"
            )
            parts.append(
                f"It needs {_f(sf['RL'].get('interventions_mean'), 1)} Safety Guard "
                f"interventions per 2 h run ({others})."
            )
    parts.append("All of this has to be confirmed in Part 8 (A-06).")
    return " ".join(parts)


def _limits_section(
    cmp: Mapping | None, ks: Mapping | None = None, n_runs: int | None = None
) -> list[str]:
    status = _fallback_status(cmp, ks, n_runs)
    return [
        "## 12. Limitations and the 10.6 fallback",
        "",
        "- The agent learns the simulator (spec 11.2) on the ASSUMED 123 s signal program and "
        "the calibration; every number here is provisional until the field visit (3-5 Nov "
        "2026) and is simulated, not measured.",
        "- The S6 headline uses the env cost on 30-min windows, not KPI v3 over 2 h with drain; "
        "the full 2 h runs are exploratory. Selection on 3000-3009 is optimistic.",
        "- Reward weights (bus 25, pedestrian 1.2, ambulance 20) and λ are ASSUMED starting "
        "values (10.3); the agent can push pedestrian waits toward the shield bound, which the "
        "guard enforces at the cost of interventions.",
        "- The observation reads every signal-controlled lane's halting and present vehicles "
        "and its longest wait, and the pedestrian curb waits, straight from the simulator: "
        "ASSUMED ideal detection (a camera or loops per lane, ideal pedestrian detection). S1 "
        "uses only its loop detectors, so part of the RL gain over S1 may be an information "
        "advantage; the gains rest on that sensing being available on site.",
        "- Day-type profiles are ratio extrapolations (D44); the ambulance features are seen at "
        "decision points only after a preemption timeout (section 10).",
        f"- **Spec 10.6 fallback:** if RL does not beat the actuated baseline after a reasonable "
        f"number of trials, the rules (S2) become the main solution and RL is reported as a "
        f"study with its analysis. {status}",
        "",
    ]


def _part7_section() -> list[str]:
    return [
        "## 13. Part 7 plan",
        "",
        "- Full trainings (300-500k steps, 3 seeds per setting), validation on full 2 h windows "
        "once short-window training plateaus.",
        "- Weight sensitivity (w_b, w_p, λ ∈ {0, 30, 100}) and its effect on every user's KPIs "
        "(10.3).",
        "- Optional free-order action with masking (needs sb3-contrib, user approval).",
        "- Commit the selected model (about 0.5 MB) with its manifest for Part 8 (A-06 on "
        "seeds 1-10, never touched in Part 6).",
        "",
    ]


def build_docs(
    config: Path = RL_CONFIG,
    docs: Path | None = None,
    figures: Path | None = None,
    runs: Path | None = None,
    selected: Path | None = None,
    devtest: Path | None = None,
    timing_path: Path | None = None,
    repro_path: Path | None = None,
) -> Path:
    """`docs` subcommand: write docs/rl.md and its figures from the RL artifacts (any
    missing artifact gives a "not available yet" note)."""
    from traffic_twin.config import text_sha256
    from traffic_twin.viz import rl as viz

    rl_cfg = load_config(config)
    outs = rl_cfg.get("outputs") or {}
    docs = Path(docs or outs.get("docs") or "docs/rl.md")
    figures = Path(figures or outs.get("figures") or "docs/figures/rl")
    root = Path(runs or runs_root(rl_cfg))
    devtest = Path(devtest or root / "devtest")
    sha = text_sha256(config)
    all_runs = collect_runs(root)
    sumo_runs = [m for m in all_runs if m.get("backend") == "sumo"]
    current = [m for m in sumo_runs if m.get("rl_yaml_sha256") == sha]
    sel = _json(Path(selected or selected_dir(rl_cfg)) / "manifest.json")
    cmp = _json(devtest / "compare_summary.json")
    cmp_rows = pd.read_csv(devtest / "compare.csv") if (devtest / "compare.csv").is_file() else None
    ks = _json(devtest / "kpi_summary.json")
    timing_doc = _json(Path(timing_path or root / "timing" / "timing.json"))
    repro = _json(Path(repro_path or root / "repro.json"))
    figs = viz.make_figures(current, timing_doc, cmp_rows, figures)
    verdict = s6_verdict(current, rl_cfg, sha)
    max_hold = preempt_max_hold_s(rl_cfg)
    candidates = ((sel or {}).get("selection") or {}).get("candidates") or []
    n_candidates = sum(1 for c in candidates if not c.get("excluded")) or None

    lines = [
        "# RL environment and first trainings (Part 6)",
        "",
        "Generated by `python -m traffic_twin.rl.evaluate docs` (spec chapter 10, EF-10, S6; "
        "decisions D55-D62). Do not edit by hand: rerun `tasks.py rl-eval`. Every result is "
        "simulated and provisional until the field visit (ASSUMED signal program).",
        "",
    ]
    head = [
        f"S6 criterion **{verdict['verdict']}** "
        f"({verdict['n_pass']} of {verdict['n_runs']} DQN runs)"
    ]
    if timing_doc and (timing_doc.get("summary") or {}):
        walls = [s["wall_s_mean"] for s in timing_doc["summary"].values()]
        span = (
            f"{min(walls):.1f}"
            if f"{min(walls):.1f}" == f"{max(walls):.1f}"
            else (f"{min(walls):.1f}-{max(walls):.1f}")
        )
        head.append(f"full 2 h episode {span} s wall")
    if sel:
        head.append(f"selected model `{sel.get('model_version')}`")
    if cmp:
        v1 = ((cmp.get("summary") or {}).get("RL") or {}).get("vs_S1") or {}
        if v1.get("n"):
            head.append(
                f"dev-test windows RL vs S1 {_pct(v1.get('pct_mean'))} "
                f"(p {_p(v1.get('p'))}, exploratory)"
            )
    if ks:
        head.append("safety " + ("OK" if ks.get("safety_ok") else "VIOLATED"))
    lines += ["**Summary:** " + "; ".join(head) + ".", ""]
    lines += _env_section(rl_cfg, max_hold)
    lines += _prereg_section(rl_cfg, sha)
    lines += _timing_section(timing_doc, figs, docs, full_episode_times(rl_cfg))
    lines += _runs_section(sumo_runs, sha)
    lines += ["## 5. Learning curves", ""]
    if "learning_curves" in figs:
        lines += [
            "Left: training episode return (rolling mean over 50 episodes, VecMonitor). "
            "Right: validation cost on the selection windows every "
            f"{(rl_cfg.get('train') or {}).get('eval_every_steps')} steps, with the "
            "baselines on the same windows.",
            "",
            f"![Learning curves]({_rel(figs['learning_curves'], docs)})",
            "",
        ]
        if "validation_totals" in figs:
            lines += [
                "Per-class totals of the best checkpoints and the baselines (selection "
                "windows, mean per window):",
                "",
                f"![Validation totals]({_rel(figs['validation_totals'], docs)})",
                "",
            ]
    else:
        lines += _missing("learning curves", "tasks.py rl-train")
    lines += _verdict_section(verdict)
    if sel:
        lines += [
            f"Selected for evaluation (`rl.evaluate select`, lowest best validation cost "
            f"among the {str(sel.get('algo')).upper()} runs): `{sel.get('source_run')}` "
            f"step {sel.get('best_step')} -> `{sel.get('model_version')}` (model sha256 "
            f"`{str(sel.get('model_sha256'))[:12]}`).",
            "",
        ]
    lines += _compare_section(cmp, figs, docs)
    lines += _kpi_section(ks)
    lines += _safety_section(ks, cmp)
    lines += _amb_section(cmp, current, ks, max_hold)
    lines += _repro_section(sumo_runs, repro)
    lines += _limits_section(cmp, ks, n_candidates)
    lines += _part7_section()
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return docs


# -- CLI -------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, default=RL_CONFIG)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("timing", help="episode timing and SubprocVecEnv scaling")
    t.add_argument("--seeds", nargs="+", default=["3000..3002"])
    t.add_argument("--n-envs", nargs="*", type=int, default=[1, 4, 8, 12])
    t.add_argument("--steps", type=int, default=5000, help="decision steps per scaling point")
    t.add_argument("--model", type=Path, help="trained model (default: the selected one)")
    t.add_argument("--out", type=Path, help="timing.json (default: <runs>/timing/timing.json)")
    s = sub.add_parser("select", help="copy the best run's model to the selected folder")
    s.add_argument("--runs", nargs="*", type=Path, help="run folders (default: all under runs)")
    s.add_argument("--algo", default="dqn", help="dqn, ppo or any")
    s.add_argument("--out", type=Path, help="default: rl.yaml outputs.selected")
    c = sub.add_parser("compare", help="dev-test window comparison (exploratory)")
    c.add_argument("--seeds", nargs="+", default=["3020..3029"])
    c.add_argument("--model", type=Path)
    c.add_argument("--algo", choices=ALGOS, help="when the model has no manifest.json")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--out", type=Path, help="default: <runs>/devtest")
    k = sub.add_parser("kpi", help="full 2 h run_scenario runs (exploratory KPIs)")
    k.add_argument("--seeds", nargs="+", default=["3020..3029"])
    k.add_argument("--model", type=Path, help="replaces pitie_rl.yaml rl.model")
    k.add_argument("--out", type=Path, help="default: <runs>/devtest")
    k.add_argument("--force", action="store_true", help="rerun runs that are current")
    k.add_argument("--summary-only", action="store_true", help="summarise existing runs only")
    r = sub.add_parser("repro", help="are two run folders identical (D65)?")
    r.add_argument("runs", nargs=2, type=Path)
    r.add_argument("--out", type=Path, help="default: <runs>/repro.json")
    d = sub.add_parser("docs", help="write docs/rl.md and its figures")
    d.add_argument("--docs", type=Path)
    d.add_argument("--figures", type=Path)
    d.add_argument("--runs", type=Path, help="runs root (default: rl.yaml outputs.runs)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.cmd == "timing":
            print(
                timing(
                    args.config,
                    parse_seeds(args.seeds),
                    args.n_envs,
                    args.steps,
                    args.model,
                    args.out,
                ).as_posix()
            )
        elif args.cmd == "select":
            algo = None if args.algo == "any" else args.algo
            print(select(args.config, args.runs, algo, args.out).as_posix())
        elif args.cmd == "compare":
            out = compare(
                args.config,
                parse_seeds(args.seeds),
                args.model,
                args.workers,
                args.out,
                algo=args.algo,
            )
            print(out.as_posix())
        elif args.cmd == "kpi":
            summary = kpi(
                args.config,
                parse_seeds(args.seeds),
                args.model,
                args.out,
                args.force,
                args.summary_only,
            )
            if not summary["safety_ok"]:
                for v in summary["violations"]:
                    log.error("SAFETY: %s", v)
                return SAFETY_EXIT
        elif args.cmd == "repro":
            res = repro_check(*args.runs)
            rl_cfg = load_config(args.config)
            _write_json(Path(args.out or runs_root(rl_cfg) / "repro.json"), res)
            print(json.dumps(res["checks"], indent=1))
            return 0 if res["same"] else 1
        elif args.cmd == "docs":
            print(build_docs(args.config, args.docs, args.figures, args.runs).as_posix())
    except (ConfigError, ValueError) as e:
        raise SystemExit(f"error: {e}") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

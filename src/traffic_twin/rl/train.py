"""RL training (Part 6, S6; D60, D61; contract 4.7).

    python -m traffic_twin.rl.train --config configs/rl.yaml --algo dqn|ppo --seed N
        [--steps N] [--n-envs 8] [--out data/rl_runs] [--force] [--stub]

-> `<out>/<algo>_s<seed>_<steps//1000>k_<sha8 rl.yaml>/`:

- `baselines.csv`: S0 (fixed), S1 (actuated), S2 (rules) and always-stay on the selection
  windows (seeds 3000-3009, train-mode episodes), computed once before training;
- `monitor.csv` (VecMonitor: episode return, length, wall time), `progress.csv` (SB3 CSV
  logger), `ckpt/` (CheckpointCallback every `checkpoint_every_steps`);
- `validation.csv`: every `eval_every_steps`, the deterministic policy on the selection
  windows: mean episode cost sum(c) (D58, unscaled), its change in % against S0, S1, S2 and
  always-stay, per-class totals, interventions, danger, curb wait, decisions, preemption
  requests / served / timeouts (summed over the windows); the best
  checkpoint (lowest cost) is `best_model.zip` (D60). `validation_episodes.csv` keeps the
  per-window rows;
- `final_model.zip` and `manifest.json` (algo, seed, steps, n_envs, threads, rl.yaml
  sha256, obs / reward versions, seeds, hyperparameters, code_version and dirty flag,
  library versions, wall / validation / training seconds, steps/s, baseline costs, best
  validation cost and step, model file sha256 and policy-parameter sha256).

`--out` (and rl.yaml `outputs.runs`) may never point at or under data/results.

Reproducibility (D61): `torch.set_num_threads(1)` in the learner and in every worker,
`device="cpu"`, `set_random_seed(seed)`, SubprocVecEnv with `start_method="spawn"`;
worker i plays training seeds i, i + n, ... so `n_envs` is part of the experiment. The
zip files carry archive timestamps, so two identical runs are compared on the policy
parameters (`*_policy_sha256`), not on the zip sha256.

`--stub` trains on `rl.stub.StubTrafficEnv` (no SUMO): a smoke run, never a result.
torch and stable-baselines3 are imported inside functions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing as mp
import platform
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_twin.config import ConfigError, load_config, text_sha256
from traffic_twin.control.safety import INTERVENTION_KEYS
from traffic_twin.paths import RL_CONFIG, RL_RUNS_DIR
from traffic_twin.sim.results import code_version

log = logging.getLogger(__name__)

ALGOS = ("dqn", "ppo")
# validation reference policies on the selection windows (label -> run_baseline name)
BASELINES = {"S0": "fixed", "S1": "actuated", "S2": "rules", "stay": "stay"}
TOTAL_KEYS = ("car_s", "car_pending_s", "bus_s", "pedestrian_s", "ambulance_s", "penalty_s")
PREEMPT_KEYS = ("requests", "served", "timeouts")  # episode_summary["preemption"]
S6_MARGIN_PCT = 5.0  # D60: best validation cost >= 5 % below S0 and always-stay
S6_REFERENCES = ("S0", "stay")
RUN_FILES = (
    "manifest.json",
    "baselines.csv",
    "monitor.csv",
    "progress.csv",
    "validation.csv",
    "validation_episodes.csv",
    "best_model.zip",
    "final_model.zip",
)
VERSION_PACKAGES = {
    "torch": "torch",
    "sb3": "stable-baselines3",
    "gymnasium": "gymnasium",
    "libsumo": "libsumo",
    "numpy": "numpy",
}


# -- environment backends ----------------------------------------------------------------
@dataclass(frozen=True)
class EnvBackend:
    """How to build an episode env and play a baseline on it: the SUMO `TrafficEnv`
    (`real_backend`) or the SUMO-free stub (`stub_backend`). Picklable (spawned workers
    rebuild their env from it)."""

    name: str
    make_env: Callable[..., Any]
    run_baseline: Callable[[Any, str, int], dict]
    env_kwargs: dict = field(default_factory=dict)

    def env(
        self,
        config: str | Path,
        *,
        mode: str = "train",
        worker: int = 0,
        n_workers: int = 1,
        seeds: Sequence[int] | None = None,
        work_root: Path | None = None,
        randomize: bool = True,
    ):
        kw: dict[str, Any] = {
            "mode": mode,
            "worker": worker,
            "n_workers": n_workers,
            "seeds": list(seeds) if seeds is not None else None,
            "randomize": randomize,
            **self.env_kwargs,
        }
        if work_root is not None:
            kw["work_root"] = Path(work_root)
        return self.make_env(config, **kw)


def real_backend() -> EnvBackend:
    """`rl.env.TrafficEnv` and `rl.env.run_baseline` (SUMO, libsumo)."""
    from traffic_twin.rl.env import TrafficEnv, run_baseline

    return EnvBackend("sumo", TrafficEnv, run_baseline)


def stub_backend(**env_kwargs) -> EnvBackend:
    """`rl.stub.StubTrafficEnv` (no SUMO); `env_kwargs` e.g. `window_s=300`."""
    from traffic_twin.rl.stub import StubTrafficEnv
    from traffic_twin.rl.stub import run_baseline as stub_baseline

    return EnvBackend("stub", StubTrafficEnv, stub_baseline, dict(env_kwargs))


def set_torch_threads(n: int) -> None:
    import torch

    torch.set_num_threads(int(n))


class EnvFactory:
    """Picklable thunk for a VecEnv worker: sets the torch threads (D61), then builds
    worker `worker` of `n_workers` on the training seeds."""

    def __init__(
        self,
        backend: EnvBackend,
        config: str | Path,
        worker: int,
        n_workers: int,
        seeds: Sequence[int],
        work_root: Path | None,
        threads: int,
    ) -> None:
        self.backend, self.config = backend, str(config)
        self.worker, self.n_workers = worker, n_workers
        self.seeds, self.work_root, self.threads = list(seeds), work_root, threads

    def __call__(self):
        set_torch_threads(self.threads)
        return self.backend.env(
            self.config,
            mode="train",
            worker=self.worker,
            n_workers=self.n_workers,
            seeds=self.seeds,
            work_root=self.work_root,
        )


# -- small helpers ---------------------------------------------------------------------
def seed_range(pair: Sequence[int]) -> list[int]:
    """Inclusive [lo, hi] seed range of rl.yaml as a list."""
    lo, hi = int(pair[0]), int(pair[1])
    if hi < lo:
        raise ConfigError(f"empty seed range {pair}")
    return list(range(lo, hi + 1))


def run_name(algo: str, seed: int, steps: int, cfg_sha: str) -> str:
    return f"{algo}_s{seed}_{int(steps) // 1000}k_{cfg_sha[:8]}"


def pct_change(value: float, reference: float) -> float:
    """100 * (value / reference - 1); NaN when the reference is not positive."""
    if reference is None or not math.isfinite(reference) or reference <= 0:
        return math.nan
    return 100.0 * (float(value) / float(reference) - 1.0)


def s6_run_pass(best_cost: float, baselines: Mapping[str, float]) -> bool | None:
    """D60 per run: best validation cost >= 5 % below both S0 and always-stay (equality
    passes). None when a reference is missing."""
    out = True
    for ref in S6_REFERENCES:
        p = pct_change(best_cost, baselines.get(ref, math.nan))
        if math.isnan(p):
            return None
        out &= p <= -S6_MARGIN_PCT + 1e-9
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def policy_sha256(model) -> str:
    """sha256 of the policy parameters (names, dtypes, shapes, bytes), in state_dict
    order: equal for bitwise-identical weights, independent of zip timestamps."""
    h = hashlib.sha256()
    for name, tensor in model.policy.state_dict().items():
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def model_class(algo: str):
    from stable_baselines3 import DQN, PPO

    try:
        return {"dqn": DQN, "ppo": PPO}[algo]
    except KeyError:
        raise ConfigError(f"unknown algo {algo!r} (expected one of {ALGOS})") from None


def load_model(path: Path, algo: str):
    return model_class(algo).load(str(path), device="cpu")


def hyperparameters(rl_cfg: Mapping, algo: str, overrides: Mapping | None = None) -> dict:
    """The rl.yaml block of `algo` (pre-registered, D61), plus explicit overrides."""
    if algo not in ALGOS:
        raise ConfigError(f"unknown algo {algo!r} (expected one of {ALGOS})")
    hp = dict(rl_cfg.get(algo) or {})
    hp.update(overrides or {})
    return hp


def sb3_kwargs(hp: Mapping) -> dict:
    kw = dict(hp)
    net_arch = kw.pop("net_arch", None)
    if net_arch is not None:
        kw["policy_kwargs"] = {"net_arch": [int(n) for n in net_arch]}
    return kw


def make_model(algo: str, env, hp: Mapping, seed: int):
    return model_class(algo)(
        "MlpPolicy", env, seed=int(seed), device="cpu", verbose=0, **sb3_kwargs(hp)
    )


def library_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for key, dist in VERSION_PACKAGES.items():
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = None
    return out


# -- episodes --------------------------------------------------------------------------
def play_episode(env, act: Callable[[np.ndarray], int], episode_seed: int) -> dict:
    """One episode of `act(obs) -> action` on `env`; returns `info["episode_summary"]`."""
    obs, _ = env.reset(options={"episode_seed": int(episode_seed)})
    while True:
        obs, _, terminated, truncated, info = env.step(act(obs))
        if terminated or truncated:
            summary = info.get("episode_summary")
            if summary is None:
                raise RuntimeError("episode ended without info['episode_summary']")
            return summary


def model_policy(model) -> Callable[[np.ndarray], int]:
    """The deterministic policy of an SB3 model as `act(obs) -> int`."""

    def act(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return act


def summary_row(summary: Mapping) -> dict:
    """Flat row of an episode summary: cost sum(c), per-class totals, interventions (total
    and per key), danger, curb wait, decisions, advances, ambulance-feature share and the
    preemption outcome (`PREEMPT_KEYS`: requests, served, timeouts; a request is dropped
    after `max_hold_s` without progress, so preemption is bounded, not unconditional)."""
    totals = summary.get("totals") or {}
    iv = summary.get("interventions") or {}
    params = summary.get("params") or {}
    pre = summary.get("preemption") or {}
    row: dict[str, Any] = {"cost": float(totals.get("cost", math.nan))}
    row["return"] = float(summary.get("return", math.nan))
    row |= {k: float(totals.get(k, math.nan)) for k in TOTAL_KEYS}
    row["interventions"] = int(sum(int(v) for v in iv.values()))
    keys = [*INTERVENTION_KEYS, *sorted(set(iv) - set(INTERVENTION_KEYS))]
    row |= {f"iv_{k}": int(iv.get(k, 0)) for k in keys}
    row["danger"] = int(summary.get("danger", 0) or 0)
    row["ped_curb_wait_max"] = float(summary.get("ped_curb_wait_max", math.nan))
    row["decisions"] = int(summary.get("decisions", 0) or 0)
    row["advances"] = int(summary.get("advances", 0) or 0)
    row["amb_seen_share"] = float(summary.get("amb_seen_share", math.nan))
    row |= {f"preempt_{k}": int(pre.get(k, 0) or 0) for k in PREEMPT_KEYS}
    row["multiplier"] = params.get("multiplier")
    row["day_type"] = params.get("day_type")
    return row


# Episode tasks, run in-process or in a spawned pool (one env per process: libsumo runs
# one simulation per process). A task is ("baseline", controller, seed) or
# ("model", model_path, algo, seed).
_POOL: dict[str, Any] = {}


def _pool_init(backend: EnvBackend, config: str, mode: str, work_root: Path | None = None) -> None:
    set_torch_threads(1)
    _POOL.clear()
    _POOL.update(
        backend=backend, config=config, mode=mode, work_root=work_root, env=None, models={}
    )


def _pool_env():
    if _POOL["env"] is None:
        _POOL["env"] = _POOL["backend"].env(
            _POOL["config"], mode=_POOL["mode"], seeds=[0], work_root=_POOL["work_root"]
        )
    return _POOL["env"]


def _run_task(task: tuple) -> dict:
    env, backend = _pool_env(), _POOL["backend"]
    if task[0] == "baseline":
        _, controller, seed = task
        return summary_row(backend.run_baseline(env, controller, int(seed)))
    if task[0] == "model":
        _, path, algo, seed = task
        models = _POOL["models"]
        if path not in models:
            models[path] = load_model(Path(path), algo)
        return summary_row(play_episode(env, model_policy(models[path]), int(seed)))
    raise ValueError(f"unknown episode task {task[0]!r}")


def run_episodes(
    backend: EnvBackend,
    config: str | Path,
    tasks: Sequence[tuple],
    workers: int = 1,
    mode: str = "train",
    work_root: Path | None = None,
) -> list[dict]:
    """Rows (`summary_row`) of the episode tasks, in task order. With `workers > 1` the
    tasks run in a spawned process pool; every episode is seeded, so the result does not
    depend on the worker that played it."""
    if not tasks:
        return []
    if workers <= 1:
        _pool_init(backend, str(config), mode, work_root)
        try:
            return [_run_task(t) for t in tasks]
        finally:
            env = _POOL.get("env")
            if env is not None:
                env.close()
            _POOL.clear()
    ctx = mp.get_context("spawn")
    n = min(int(workers), len(tasks))
    with ProcessPoolExecutor(
        n, mp_context=ctx, initializer=_pool_init, initargs=(backend, str(config), mode, work_root)
    ) as ex:
        return list(ex.map(_run_task, tasks))


def baseline_frame(
    backend: EnvBackend,
    config: str | Path,
    seeds: Sequence[int],
    workers: int = 1,
    labels: Mapping[str, str] = BASELINES,
    mode: str = "train",
    work_root: Path | None = None,
) -> pd.DataFrame:
    """label, policy, seed and the summary row of every (baseline, seed)."""
    tasks = [("baseline", ctl, s) for ctl in labels.values() for s in seeds]
    rows = run_episodes(backend, config, tasks, workers, mode, work_root)
    meta = [{"label": lab, "policy": ctl, "seed": s} for lab, ctl in labels.items() for s in seeds]
    return pd.DataFrame([m | r for m, r in zip(meta, rows, strict=True)])


# -- validation (D60) ------------------------------------------------------------------
class Validator:
    """Scores the deterministic policy on fixed selection windows: appends a row to
    `validation.csv` (and the per-window rows to `validation_episodes.csv`) and saves the
    model as `best_model.zip` whenever its mean cost is the lowest so far."""

    def __init__(
        self, env, seeds: Sequence[int], baselines: Mapping[str, float], out_dir: Path
    ) -> None:
        self.env, self.seeds = env, [int(s) for s in seeds]
        self.baselines = dict(baselines)
        self.out_dir = Path(out_dir)
        self.rows: list[dict] = []
        self.episodes: list[dict] = []
        self.best_cost, self.best_step = math.inf, None
        self.last_step: int | None = None
        self.eval_s = 0.0

    def __call__(self, model, step: int) -> dict:
        t0 = time.perf_counter()
        act = model_policy(model)
        eps = [
            {"step": int(step), "seed": s} | summary_row(play_episode(self.env, act, s))
            for s in self.seeds
        ]
        ep = pd.DataFrame(eps)
        mean = ep.mean(numeric_only=True)
        cost = float(mean["cost"])
        row: dict[str, Any] = {"step": int(step), "cost": cost}
        row |= {f"cost_vs_{k}": pct_change(cost, v) for k, v in self.baselines.items()}
        row |= {k: float(mean[k]) for k in TOTAL_KEYS}
        row["interventions"] = float(mean["interventions"])
        row |= {c: float(mean[c]) for c in ep.columns if c.startswith("iv_")}
        row["danger"] = int(ep["danger"].max())
        row["ped_curb_wait_max"] = float(ep["ped_curb_wait_max"].max())
        row |= {k: float(mean[k]) for k in ("decisions", "advances", "amb_seen_share")}
        row |= {f"preempt_{k}": int(ep[f"preempt_{k}"].sum()) for k in PREEMPT_KEYS}
        best = cost < self.best_cost
        if best:
            self.best_cost, self.best_step = cost, int(step)
            model.save(str(self.out_dir / "best_model.zip"))
        row["best"] = best
        elapsed = time.perf_counter() - t0
        row["eval_s"] = round(elapsed, 3)
        self.eval_s += elapsed
        self.last_step = int(step)
        self.rows.append(row)
        self.episodes.extend(eps)
        pd.DataFrame(self.rows).to_csv(self.out_dir / "validation.csv", index=False)
        pd.DataFrame(self.episodes).to_csv(self.out_dir / "validation_episodes.csv", index=False)
        log.info(
            "validation step %d: cost %.1f (vs S0 %+.1f %%, S1 %+.1f %%, stay %+.1f %%)%s",
            step,
            cost,
            row.get("cost_vs_S0", math.nan),
            row.get("cost_vs_S1", math.nan),
            row.get("cost_vs_stay", math.nan),
            " best" if best else "",
        )
        return row


def validation_callback(validator: Validator, every: int):
    """SB3 callback running `validator` each time the step count crosses a multiple of
    `every` (`eval_every_steps`)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class ValidationCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.next = int(every)

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next:
                while self.next <= self.num_timesteps:
                    self.next += int(every)
                validator(self.model, self.num_timesteps)
            return True

    return ValidationCallback()


# -- training --------------------------------------------------------------------------
def check_versions(rl_cfg: Mapping) -> dict[str, str]:
    """Observation and reward versions of rl.yaml; for the SUMO env they must equal the
    code's (`rl.features.OBS_VERSION`, `rl.reward.REWARD_VERSION`)."""
    from traffic_twin.rl.features import OBS_VERSION
    from traffic_twin.rl.reward import REWARD_VERSION

    obs_v = (rl_cfg.get("observation") or {}).get("version")
    rew_v = (rl_cfg.get("reward") or {}).get("version")
    if obs_v != OBS_VERSION or rew_v != REWARD_VERSION:
        raise ConfigError(
            f"rl.yaml versions obs {obs_v!r} / reward {rew_v!r} differ from the code "
            f"({OBS_VERSION!r} / {REWARD_VERSION!r})"
        )
    return {"obs_version": obs_v, "reward_version": rew_v}


def check_out_root(out_root: Path) -> None:
    """Simulation outputs never go under data/results (Part 5-6 working rule): training
    runs, their `_work` folders and every evaluation output are refused there."""
    res = (Path("data") / "results").resolve()
    p = Path(out_root).resolve()
    if p == res or res in p.parents:
        raise ConfigError(f"{out_root}: RL runs are never written under data/results")


def prepare_out_dir(out: Path, force: bool) -> None:
    """A finished run (manifest.json) is kept unless `force`; a partial one is removed."""
    if (out / "manifest.json").is_file() and not force:
        raise FileExistsError(f"{out} already holds a finished run (use --force to redo it)")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)


def train(
    config: str | Path = RL_CONFIG,
    algo: str = "dqn",
    seed: int = 0,
    steps: int | None = None,
    n_envs: int | None = None,
    out_root: Path | None = None,
    *,
    backend: EnvBackend | None = None,
    vec: str | None = None,
    hp_overrides: Mapping | None = None,
    eval_every: int | None = None,
    checkpoint_every: int | None = None,
    threads: int | None = None,
    validation_seeds: Sequence[int] | None = None,
    baseline_workers: int | None = None,
    force: bool = False,
    argv: Sequence[str] | None = None,
) -> Path:
    """Train one agent and write its run folder (module docstring); returns the folder."""
    rl_cfg = load_config(config)
    if algo not in ALGOS:
        raise ConfigError(f"unknown algo {algo!r} (expected one of {ALGOS})")
    tcfg = rl_cfg.get("train") or {}
    s6 = (tcfg.get("s6") or {}).get(algo) or {}
    steps = int(steps or s6.get("steps") or 200_000)
    n_envs = int(n_envs or tcfg.get("n_envs", 8))
    threads = int(threads or tcfg.get("torch_threads", 1))
    eval_every = int(eval_every or tcfg.get("eval_every_steps", 10_000))
    checkpoint_every = int(checkpoint_every or tcfg.get("checkpoint_every_steps", 50_000))
    backend = backend or real_backend()
    vec = vec or ("subproc" if backend.name == "sumo" else "dummy")
    if vec not in ("subproc", "dummy"):
        raise ConfigError(f"vec must be 'subproc' or 'dummy', not {vec!r}")
    if backend.name == "sumo" and vec != "subproc":
        raise ConfigError("the SUMO env needs vec='subproc' (libsumo: one simulation per process)")
    versions = (
        check_versions(rl_cfg)
        if backend.name == "sumo"
        else {
            "obs_version": (rl_cfg.get("observation") or {}).get("version"),
            "reward_version": (rl_cfg.get("reward") or {}).get("version"),
        }
    )
    cfg_sha = text_sha256(config)
    out_root = Path(out_root or (rl_cfg.get("outputs") or {}).get("runs") or RL_RUNS_DIR)
    check_out_root(out_root)
    out = out_root / run_name(algo, seed, steps, cfg_sha)
    prepare_out_dir(out, force)
    work_root = out_root / "_work"
    train_seeds = seed_range(rl_cfg["seeds"]["train"])
    selection = [int(s) for s in (validation_seeds or seed_range(rl_cfg["seeds"]["selection"]))]
    hp = hyperparameters(rl_cfg, algo, hp_overrides)
    started_at = datetime.now(UTC).isoformat()
    t_start = time.perf_counter()

    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    set_torch_threads(threads)
    set_random_seed(int(seed))

    # 1. reference costs on the selection windows (cached for the whole run)
    t0 = time.perf_counter()
    workers = (
        baseline_workers
        if baseline_workers is not None
        else (n_envs if backend.name == "sumo" else 1)
    )
    base = baseline_frame(backend, config, selection, workers, work_root=work_root)
    base.to_csv(out / "baselines.csv", index=False)
    baselines = {k: float(v) for k, v in base.groupby("label", sort=False)["cost"].mean().items()}
    baseline_s = time.perf_counter() - t0
    log.info(
        "baselines on %d selection windows (%.1f s): %s",
        len(selection),
        baseline_s,
        {k: round(v, 1) for k, v in baselines.items()},
    )

    # 2. training envs, validation env, model
    factories = [
        EnvFactory(backend, config, i, n_envs, train_seeds, work_root, threads)
        for i in range(n_envs)
    ]
    venv = (
        SubprocVecEnv(factories, start_method="spawn")
        if vec == "subproc"
        else DummyVecEnv(factories)
    )
    eval_env = None
    try:
        venv = VecMonitor(venv, str(out / "monitor.csv"))
        eval_env = backend.env(config, mode="train", seeds=selection, work_root=work_root)
        model = make_model(algo, venv, hp, seed)
        model.set_logger(configure(str(out), ["csv"]))
        validator = Validator(eval_env, selection, baselines, out)
        callbacks = CallbackList(
            [
                validation_callback(validator, eval_every),
                CheckpointCallback(
                    save_freq=max(checkpoint_every // n_envs, 1),
                    save_path=str(out / "ckpt"),
                    name_prefix=algo,
                ),
            ]
        )
        t0 = time.perf_counter()
        model.learn(total_timesteps=steps, callback=callbacks, progress_bar=False)
        learn_s = time.perf_counter() - t0
        eval_in_learn_s = validator.eval_s
        if validator.last_step != model.num_timesteps:  # always score the final policy
            validator(model, model.num_timesteps)
        model.save(str(out / "final_model.zip"))
        final_policy = policy_sha256(model)
    finally:
        venv.close()
        if eval_env is not None:
            eval_env.close()

    best_model = load_model(out / "best_model.zip", algo)
    train_s = max(learn_s - eval_in_learn_s, 1e-9)
    best_vs = {k: pct_change(validator.best_cost, v) for k, v in baselines.items()}
    cv = code_version()
    manifest = {
        "run": out.name,
        "algo": algo,
        "seed": int(seed),
        "steps": steps,
        "steps_done": int(model.num_timesteps),
        "n_envs": n_envs,
        "threads": threads,
        "vec": vec,
        "backend": backend.name,
        "config": Path(config).as_posix(),
        "rl_yaml_sha256": cfg_sha,
        **versions,
        "hyperparameters": hp,
        "seeds": {"train": list(rl_cfg["seeds"]["train"]), "selection": selection},
        "eval_every_steps": eval_every,
        "checkpoint_every_steps": checkpoint_every,
        "code_version": cv,
        "dirty": cv.endswith("-dirty"),
        "versions": library_versions(),
        "started_at": started_at,
        "wall_s": round(time.perf_counter() - t_start, 2),
        "baseline_s": round(baseline_s, 2),
        "eval_s": round(validator.eval_s, 2),
        "train_s": round(train_s, 2),
        "steps_per_s": round(model.num_timesteps / train_s, 1),
        "baselines": baselines,
        "validations": len(validator.rows),
        "best_step": validator.best_step,
        "best_val_cost": validator.best_cost,
        "best_vs": best_vs,
        "final_val_cost": validator.rows[-1]["cost"],
        "s6_run_pass": s6_run_pass(validator.best_cost, baselines),
        "best_model_sha256": file_sha256(out / "best_model.zip"),
        "final_model_sha256": file_sha256(out / "final_model.zip"),
        "best_policy_sha256": policy_sha256(best_model),
        "final_policy_sha256": final_policy,
        "model_bytes": (out / "best_model.zip").stat().st_size,
        "argv": list(argv) if argv is not None else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(
        "%s: %d steps in %.0f s (%.0f steps/s, validation %.0f s); best cost %.1f at %s "
        "(vs S0 %+.1f %%, vs stay %+.1f %%)",
        out.name,
        model.num_timesteps,
        manifest["wall_s"],
        manifest["steps_per_s"],
        validator.eval_s,
        validator.best_cost,
        validator.best_step,
        best_vs.get("S0", math.nan),
        best_vs.get("stay", math.nan),
    )
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, default=RL_CONFIG)
    p.add_argument("--algo", choices=ALGOS, default="dqn")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, help="decision steps (default: rl.yaml train.s6)")
    p.add_argument("--n-envs", type=int, help="SubprocVecEnv workers (default: rl.yaml)")
    p.add_argument("--out", type=Path, help="runs root (default: rl.yaml outputs.runs)")
    p.add_argument("--eval-every", type=int, help="validation interval (default: rl.yaml)")
    p.add_argument("--baseline-workers", type=int, help="processes for the baselines")
    p.add_argument("--force", action="store_true", help="redo a finished run folder")
    p.add_argument("--stub", action="store_true", help="SUMO-free stub env (smoke run only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        out = train(
            args.config,
            args.algo,
            args.seed,
            args.steps,
            args.n_envs,
            args.out,
            backend=stub_backend() if args.stub else None,
            eval_every=args.eval_every,
            baseline_workers=args.baseline_workers,
            force=args.force,
            argv=sys.argv if argv is None else ["traffic_twin.rl.train", *argv],
        )
    except (ConfigError, FileExistsError) as e:
        raise SystemExit(f"error: {e}") from None
    print(out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

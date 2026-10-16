"""RL training (Part 6, S6; D60, D61; contract 4.7. Part 7: D69, D71, D72, D81).

    python -m traffic_twin.rl.train --config configs/rl.yaml --algo dqn|ppo --seed N
        [--steps N] [--n-envs 8] [--out data/rl_runs] [--force] [--stub]
        [--allow-dirty] [--validation-seeds 5100-5109] [--eval-workers N]
        [--eval-pool persistent|transient]

`--algo` defaults to the config's `algo` key (else dqn).

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

Part 7 run configs (a config with a `variant` key: configs/rl7/*.yaml, D69, D81). A
config without `variant` (rl.yaml) keeps everything above unchanged, manifest included.

- Name `<algo>_<variant>_s<seed>_<steps//1000>k_<sha8>`, where sha = `run_config_sha`:
  sha256 of the canonical JSON of the merged config without `changes`, `preregistered`,
  `protocol` and `outputs` (an edit of a parent file changes it). `--steps` is required.
- Hyperparameters: `"lin:a:b"` strings become linear schedules (`schedule`; the manifest
  keeps the strings); `dqn.exploration_steps` becomes `exploration_fraction = min(1,
  exploration_steps / steps)` (D71).
- Validation (`train.eval_every_steps`) on the selection windows by a persistent spawned
  pool of `train.eval_workers` processes (created once per run, closed at the end; also
  plays the baselines; `--eval-pool transient`: started a few thousand steps before each
  validation and closed after it, same rows), each checkpoint kept as `val/step_<n>.zip`.
  Per-window rows add the D77 intervention split (`split_cut`, `split_replaced`,
  `split_cut_foreseeable`) and `control_s`; validation.csv adds their rates per hour and `late`.
- Pick (D72): among the validations at >= `train.pick_from` x steps (the final policy
  included), the lowest mean selection cost -> `pick_model.zip` (a copy of its
  `val/step_<n>.zip`); `late_mean` = mean cost of those validations. `best_model.zip` keeps
  its Part 6 meaning (lowest cost over the whole run).
- `train.require_clean`: a code version ending in `-dirty` (or unknown) is refused unless
  `--allow-dirty`, which is recorded (D80).
- The manifest also carries the `provenance.json` keys.

Every run writes `provenance.json` (D81): run_config_sha and the config's source files
with their text sha256, variant, reward, observation (drop list, n features), decision
interval, base scenario config_hash and demand_hash, sha256 of the calibration files (one
per randomized day type), incident sites and net, CPU, torch parallel info,
OMP_*/MKL_*/OPENBLAS_* variables and the workers' `CHILD_ENV`, learner and worker threads,
schedules as strings, exploration steps and fraction, update-to-data, selection seeds and
metric, pick and late mean, allow_dirty.

Memory (a job is 13 processes; the Phase 2 gate hit the Windows commit limit at about
1 GB each): the spawned env and pool processes start with `OPENBLAS_NUM_THREADS=1`
(`CHILD_ENV`, unless the variable is already set; numpy's OpenBLAS would otherwise commit
about 0.74 GB per process on a 32-CPU machine); Part 7 env processes run
`light_subproc_vec_env` (SB3's stepping, but no SB3 / torch import: about 0.15 GB less
each); prediction loads (`load_model`) carry a 1-transition replay buffer. About 3.3 GB
per job at peak instead of about 13 GB; results unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import shutil
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_twin.config import (
    ConfigError,
    config_hash,
    config_sources,
    demand_hash,
    load_config,
    load_scenario,
    text_sha256,
    xml_sha256,
)
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
# Part 7 (D69, D71, D72, D77, D81)
PROVENANCE_FILE = "provenance.json"
PICK_MODEL_FILE = "pick_model.zip"
VAL_DIR = "val"  # kept validation checkpoints val/step_<n>.zip
RUN_SHA_EXCLUDE = ("changes", "preregistered", "protocol", "outputs")
SPLIT_KEYS = ("cut", "replaced", "cut_foreseeable")  # episode_summary["iv_split"] (D77)
SCHEDULE_PREFIX = "lin:"
SELECTION_METRIC = "mean episode cost sum(c) on the selection windows (D58, unscaled)"
# Loading for prediction only: the saved schedules are replaced by constants, so no
# pickled schedule object is executed (D86; lead check 2.4), and a DQN gets a 1-transition
# replay buffer instead of an empty `buffer_size` one (about 160 MB of commit per loading
# process: the validation pool workers; ignored by PPO)
LOAD_CUSTOM_OBJECTS = {
    "lr_schedule": 0.0,
    "exploration_schedule": 0.0,
    "clip_range": 0.0,
    "buffer_size": 1,
}
PARALLEL_INFO_LINES = 8  # head of torch.__config__.parallel_info() in provenance.json
# Environment of the spawned children (SubprocVecEnv envs, EpisodePool workers; the
# campaign also gives it to the learner). numpy's OpenBLAS commits a buffer per potential
# thread when numpy is imported: about 740 MB of private commit per process on the 32-CPU
# campaign machine (MAX_THREADS 24), against about 20 MB with one thread. A job is 13
# processes, so 4-6 concurrent jobs exhausted the Windows commit limit (Phase 2 gate).
# The children only do small vector work in numpy (no multi-threaded BLAS call), so the
# results do not change; a variable already set in the environment is kept.
CHILD_ENV = {"OPENBLAS_NUM_THREADS": "1"}
# Validation pool lifetime (`--eval-pool`): "persistent" (D71) keeps the eval_workers
# processes for the whole run; "transient" starts them `PREWARM_STEPS` (at most a fifth
# of the interval) before each validation and closes them after it, so the concurrent
# jobs of a campaign rarely hold their pools at the same time. Episodes are seeded:
# both give the same rows.
EVAL_POOLS = ("persistent", "transient")
PREWARM_STEPS = 5000
ENV_PREFIXES = ("OMP_", "MKL_", "OPENBLAS_")  # recorded in provenance.json (D81)


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


@contextmanager
def child_env(env: Mapping[str, str] = CHILD_ENV) -> Iterator[dict[str, str]]:
    """Set the variables of `env` that are absent from `os.environ` while the block runs
    (processes spawned inside it inherit them), then remove them again. Yields the
    values the children see."""
    added = [k for k in env if k not in os.environ]
    for k in added:
        os.environ[k] = str(env[k])
    try:
        yield {k: os.environ[k] for k in env}
    finally:
        for k in added:
            os.environ.pop(k, None)


def child_env_values(env: Mapping[str, str] = CHILD_ENV) -> dict[str, str]:
    """The values of `env` the spawned children get (`child_env`)."""
    return {k: os.environ.get(k, str(v)) for k, v in env.items()}


def set_torch_threads(n: int) -> None:
    import torch

    torch.set_num_threads(int(n))


class EnvFactory:
    """Picklable thunk for a VecEnv worker: sets the torch threads (D61) where torch is
    loaded, then builds worker `worker` of `n_workers` on the training seeds."""

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
        if "torch" in sys.modules:  # not in a light env worker (`light_subproc_vec_env`)
            set_torch_threads(self.threads)
        return self.backend.env(
            self.config,
            mode="train",
            worker=self.worker,
            n_workers=self.n_workers,
            seeds=self.seeds,
            work_root=self.work_root,
        )


def _is_wrapped_by_name(env, key: tuple[str, str]) -> bool:
    """stable-baselines3 `env_util.is_wrapped` with the wrapper class given by (module,
    qualname): a class object sent to the env process would import its module there
    (`VecMonitor` asks about SB3's `Monitor`, which imports torch)."""
    import gymnasium as gym

    while isinstance(env, gym.Wrapper):
        if any((c.__module__, c.__qualname__) == tuple(key) for c in type(env).__mro__):
            return True
        env = env.env
    return False


def _env_worker(remote, parent_remote, env_fn) -> None:
    """The env process loop of stable-baselines3 2.x `SubprocVecEnv` (`_worker`, same
    commands and replies, plus `is_wrapped_name`), defined here so that the spawned
    process imports neither stable-baselines3 nor torch (about 150 MB less commit per env
    process; the env itself never uses torch). `env_fn` (an `EnvFactory`) is pickled as
    is and returns a gymnasium env."""
    parent_remote.close()
    env = env_fn()
    reset_info: dict | None = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:  # keep the final observation, then reset
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped_name":
                remote.send(_is_wrapped_by_name(env, data))
            elif cmd == "is_wrapped":  # a class: its module is already imported here
                remote.send(_is_wrapped_by_name(env, (data.__module__, data.__qualname__)))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except (EOFError, KeyboardInterrupt):
            break


def light_subproc_vec_env(env_fns: Sequence[Callable[[], Any]], start_method: str = "spawn"):
    """A stable-baselines3 `SubprocVecEnv` whose processes run `_env_worker` (no SB3 or
    torch import) on the plainly pickled `env_fns`; stepping and replies are SB3's, so
    the episodes are the same. Part 7 runs use it (memory: 6 env processes per job)."""
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    class LightSubprocVecEnv(SubprocVecEnv):
        def __init__(self, fns: Sequence[Callable[[], Any]], method: str) -> None:
            # SubprocVecEnv.__init__ with target=_env_worker and no CloudpickleWrapper
            self.waiting = False
            self.closed = False
            ctx = mp.get_context(method)
            self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in fns], strict=True)
            self.processes = []
            for work_remote, remote, fn in zip(self.work_remotes, self.remotes, fns, strict=True):
                process = ctx.Process(
                    target=_env_worker, args=(work_remote, remote, fn), daemon=True
                )
                process.start()
                self.processes.append(process)
                work_remote.close()
            self.remotes[0].send(("get_spaces", None))
            observation_space, action_space = self.remotes[0].recv()
            VecEnv.__init__(self, len(fns), observation_space, action_space)

        def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
            """By class name: the env processes never import the wrapper's module."""
            key = (wrapper_class.__module__, wrapper_class.__qualname__)
            target_remotes = self._get_target_remotes(indices)
            for remote in target_remotes:
                remote.send(("is_wrapped_name", key))
            return [remote.recv() for remote in target_remotes]

    return LightSubprocVecEnv(list(env_fns), start_method)


# -- small helpers ---------------------------------------------------------------------
def seed_range(pair: Sequence[int]) -> list[int]:
    """Inclusive [lo, hi] seed range of rl.yaml as a list."""
    lo, hi = int(pair[0]), int(pair[1])
    if hi < lo:
        raise ConfigError(f"empty seed range {pair}")
    return list(range(lo, hi + 1))


def run_name(algo: str, seed: int, steps: int, cfg_sha: str, variant: str | None = None) -> str:
    """`<algo>_s<seed>_<k>k_<sha8>` (Part 6) or `<algo>_<variant>_s<seed>_<k>k_<sha8>`
    for a Part 7 run config (D69)."""
    tag = f"{algo}_{variant}" if variant else algo
    return f"{tag}_s{seed}_{int(steps) // 1000}k_{cfg_sha[:8]}"


def _canonical_json(obj: Any) -> str:
    """Canonical JSON as `config.config_hash` builds it: sorted keys, no whitespace,
    dates and paths as strings."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def run_config_sha(rl_cfg: Mapping) -> str:
    """Run identity (D69): sha256 of the canonical JSON of the merged config without
    `changes`, `preregistered`, `protocol` and `outputs`. Unlike the text sha256 of the
    run config file, it changes when a file the config `extends` changes."""
    payload = {k: v for k, v in rl_cfg.items() if k not in RUN_SHA_EXCLUDE}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def resolve_algo(rl_cfg: Mapping, algo: str | None = None) -> str:
    """`algo`, else the config's `algo` key, else dqn. A config naming its algo refuses
    another one (the algo is part of a Part 7 run config)."""
    declared = rl_cfg.get("algo")
    chosen = algo or declared or "dqn"
    if chosen not in ALGOS:
        raise ConfigError(f"unknown algo {chosen!r} (expected one of {ALGOS})")
    if declared is not None and chosen != declared:
        raise ConfigError(f"the config declares algo {declared!r}, not {chosen!r}")
    return chosen


def parse_seeds(tokens: Sequence[str | int]) -> list[int]:
    """CLI seed list: integers and inclusive ranges `lo-hi` (e.g. `5100-5109`)."""
    out: list[int] = []
    for tok in tokens:
        text = str(tok).strip()
        lo, sep, hi = text.partition("-")
        try:
            out += seed_range([int(lo), int(hi)]) if sep and lo else [int(text)]
        except ValueError:
            raise ConfigError(f"bad seed or seed range {text!r} (expected N or LO-HI)") from None
    return out


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
    """An SB3 model for prediction (`LOAD_CUSTOM_OBJECTS`: saved schedules -> constants,
    no replay buffer; the policy parameters are unchanged)."""
    return model_class(algo).load(str(path), device="cpu", custom_objects=LOAD_CUSTOM_OBJECTS)


@dataclass(frozen=True)
class LinearDecay:
    """SB3 schedule `"lin:start:end"`: `end + (start - end) * progress_remaining`, with
    progress_remaining (1 at the start, 0 at the end of `learn`) clipped to [0, 1]. A
    module-level class, so a saved model refers to it by name (no pickled lambda)."""

    start: float
    end: float

    def __call__(self, progress_remaining: float) -> float:
        p = min(max(float(progress_remaining), 0.0), 1.0)
        return float(self.end + (self.start - self.end) * p)

    def __str__(self) -> str:
        return f"{SCHEDULE_PREFIX}{self.start!r}:{self.end!r}"


def schedule(value) -> float | Callable[[float], float]:
    """A hyperparameter as SB3 takes it (D71): `"lin:a:b"` -> `LinearDecay(a, b)` (a at
    the start, b at the end of training); numbers pass through."""
    if isinstance(value, bool):
        raise ConfigError(f"not a schedule: {value!r}")
    if isinstance(value, int | float):
        return value
    if isinstance(value, str) and value.startswith(SCHEDULE_PREFIX):
        parts = value[len(SCHEDULE_PREFIX) :].split(":")
        try:
            start, end = (float(p) for p in parts)
        except ValueError:
            raise ConfigError(f"bad schedule {value!r} (expected 'lin:a:b')") from None
        return LinearDecay(start, end)
    raise ConfigError(f"bad schedule {value!r} (expected a number or 'lin:a:b')")


def hyperparameters(
    rl_cfg: Mapping, algo: str, overrides: Mapping | None = None, steps: int | None = None
) -> dict:
    """The config block of `algo` (pre-registered, D61), plus explicit overrides. With
    `steps`, `exploration_steps` (D71) becomes `exploration_fraction = min(1,
    exploration_steps / steps)` and the key is removed. Schedule strings are kept (the
    manifest records them; `sb3_kwargs` converts them)."""
    if algo not in ALGOS:
        raise ConfigError(f"unknown algo {algo!r} (expected one of {ALGOS})")
    hp = dict(rl_cfg.get(algo) or {})
    hp.update(overrides or {})
    if steps is not None and "exploration_steps" in hp:
        if int(steps) <= 0:
            raise ConfigError(f"steps must be positive, not {steps}")
        hp["exploration_fraction"] = min(1.0, int(hp.pop("exploration_steps")) / int(steps))
    return hp


def sb3_kwargs(hp: Mapping) -> dict:
    kw = dict(hp)
    if "exploration_steps" in kw:
        raise ConfigError("exploration_steps needs the run's steps (hyperparameters(steps=...))")
    net_arch = kw.pop("net_arch", None)
    if net_arch is not None:
        kw["policy_kwargs"] = {"net_arch": [int(n) for n in net_arch]}
    for k, v in kw.items():
        if isinstance(v, str) and v.startswith(SCHEDULE_PREFIX):
            kw[k] = schedule(v)
    return kw


def schedule_strings(hp: Mapping) -> dict:
    """The schedule-capable hyperparameters as written in the config (provenance)."""
    keys = ("learning_rate", "clip_range", "clip_range_vf")
    return {k: (v if isinstance(v, str) else repr(v)) for k, v in hp.items() if k in keys}


def update_to_data(algo: str, hp: Mapping, n_envs: int) -> float | None:
    """DQN gradient steps per collected transition: gradient_steps / (n_envs x
    train_freq) (D71); None for PPO."""
    if algo != "dqn":
        return None
    freq = hp.get("train_freq", 4)
    freq = int(freq[0] if isinstance(freq, list | tuple) else freq)
    return round(int(hp.get("gradient_steps", 1)) / (int(n_envs) * freq), 6)


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


def summary_row(summary: Mapping, extended: bool = False) -> dict:
    """Flat row of an episode summary: cost sum(c), per-class totals, interventions (total
    and per key), danger, curb wait, decisions, advances, ambulance-feature share and the
    preemption outcome (`PREEMPT_KEYS`: requests, served, timeouts; a request is dropped
    after `max_hold_s` without progress, so preemption is bounded, not unconditional).

    `extended` (Part 7 runs) adds `control_s` and the D77 intervention split
    `split_cut`, `split_replaced`, `split_cut_foreseeable` (NaN when the summary has no
    `iv_split`). The Part 6 row is unchanged without it."""
    row = _summary_row(summary)
    if extended:
        split = summary.get("iv_split")
        row["control_s"] = float(summary.get("control_s", math.nan))
        row |= {
            f"split_{k}": (int(split.get(k, 0) or 0) if split is not None else math.nan)
            for k in SPLIT_KEYS
        }
    return row


def _summary_row(summary: Mapping) -> dict:
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
# ("model", model_path, algo, seed). A process caches ONE model, keyed by the path AND
# the file's sha256: a task naming another path, or the same path rewritten with another
# model, evicts it (a run's validation checkpoints follow each other; a path alone is no
# identity: a persistent worker would keep scoring the first model saved under a reused
# name).
_POOL: dict[str, Any] = {}
SCRATCH_MODEL_PREFIX = "_validation_model_step_"  # pooled validation, no keep_dir


def _pool_init(
    backend: EnvBackend,
    config: str,
    mode: str,
    work_root: Path | None = None,
    extended: bool = False,
) -> None:
    set_torch_threads(1)
    _POOL.clear()
    _POOL.update(
        backend=backend,
        config=config,
        mode=mode,
        work_root=work_root,
        extended=bool(extended),
        env=None,
        model=None,
    )


def _pool_env():
    if _POOL["env"] is None:
        _POOL["env"] = _POOL["backend"].env(
            _POOL["config"], mode=_POOL["mode"], seeds=[0], work_root=_POOL["work_root"]
        )
    return _POOL["env"]


def _pool_model(path: str, algo: str):
    key = (path, algo, file_sha256(Path(path)))
    cached = _POOL.get("model")
    if cached is None or cached[0] != key:
        _POOL["model"] = None  # evict before loading: one model in memory
        _POOL["model"] = (key, load_model(Path(path), algo))
    return _POOL["model"][1]


def _run_task(task: tuple) -> dict:
    env, backend = _pool_env(), _POOL["backend"]
    extended = bool(_POOL.get("extended", False))
    if task[0] == "baseline":
        _, controller, seed = task
        return summary_row(backend.run_baseline(env, controller, int(seed)), extended)
    if task[0] == "model":
        _, path, algo, seed = task
        model = _pool_model(str(path), algo)
        return summary_row(play_episode(env, model_policy(model), int(seed)), extended)
    raise ValueError(f"unknown episode task {task[0]!r}")


def _warm() -> None:
    """Pool warm-up task: build the process's env and import the SB3 model classes ahead
    of the first episode."""
    _pool_env()
    model_class("dqn"), model_class("ppo")


def _close_pool_state() -> None:
    env = _POOL.get("env")
    if env is not None:
        env.close()
    _POOL.clear()


class EpisodePool:
    """A persistent spawned process pool playing episode tasks (`_run_task`): created
    once per training run for its validations and baselines (D71), closed by `close()`.
    Each process holds one env and one cached model; the processes start with
    `CHILD_ENV` (one OpenBLAS thread, about 0.3 GB of commit each instead of 1 GB)."""

    def __init__(
        self,
        backend: EnvBackend,
        config: str | Path,
        workers: int,
        mode: str = "train",
        work_root: Path | None = None,
        extended: bool = False,
    ) -> None:
        if int(workers) < 1:
            raise ConfigError(f"workers must be >= 1, not {workers}")
        self.workers = int(workers)
        self._ex: ProcessPoolExecutor | None = ProcessPoolExecutor(
            self.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_pool_init,
            initargs=(backend, str(config), mode, work_root, extended),
        )

    def map(self, tasks: Sequence[tuple]) -> list[dict]:
        """Rows of the tasks, in task order."""
        if self._ex is None:
            raise RuntimeError("the episode pool is closed")
        if not tasks:
            return []
        with child_env():  # the workers are spawned at the first submit
            return list(self._ex.map(_run_task, tasks))

    def start(self) -> None:
        """Spawn the workers now (one warm-up task each, `_warm`) without waiting: a later
        `map` finds them booted."""
        if self._ex is None:
            raise RuntimeError("the episode pool is closed")
        with child_env():
            for _ in range(self.workers):
                self._ex.submit(_warm)

    def close(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True, cancel_futures=True)
            self._ex = None

    def __enter__(self) -> EpisodePool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def run_episodes(
    backend: EnvBackend,
    config: str | Path,
    tasks: Sequence[tuple],
    workers: int = 1,
    mode: str = "train",
    work_root: Path | None = None,
    *,
    extended: bool = False,
    pool: EpisodePool | None = None,
) -> list[dict]:
    """Rows (`summary_row`, `extended` for Part 7) of the episode tasks, in task order.
    With `pool`, its processes play them; with `workers > 1`, a spawned process pool made
    for this call. Every episode is seeded, so the result does not depend on the worker
    that played it."""
    if not tasks:
        return []
    if pool is not None:
        return pool.map(tasks)
    if workers <= 1:
        _pool_init(backend, str(config), mode, work_root, extended)
        try:
            return [_run_task(t) for t in tasks]
        finally:
            _close_pool_state()
    n = min(int(workers), len(tasks))
    with EpisodePool(backend, config, n, mode, work_root, extended) as p:
        return p.map(tasks)


def baseline_frame(
    backend: EnvBackend,
    config: str | Path,
    seeds: Sequence[int],
    workers: int = 1,
    labels: Mapping[str, str] = BASELINES,
    mode: str = "train",
    work_root: Path | None = None,
    *,
    extended: bool = False,
    pool: EpisodePool | None = None,
) -> pd.DataFrame:
    """label, policy, seed and the summary row of every (baseline, seed)."""
    tasks = [("baseline", ctl, s) for ctl in labels.values() for s in seeds]
    rows = run_episodes(
        backend, config, tasks, workers, mode, work_root, extended=extended, pool=pool
    )
    meta = [{"label": lab, "policy": ctl, "seed": s} for lab, ctl in labels.items() for s in seeds]
    return pd.DataFrame([m | r for m, r in zip(meta, rows, strict=True)])


# -- validation (D60) ------------------------------------------------------------------
def _rates_per_hour(ep: pd.DataFrame, mean: pd.Series) -> dict[str, float]:
    """D77 split per window (mean) and per hour of control (sums over the windows / their
    control hours); NaN when the summaries carry no split or no control time."""
    hours = float(ep["control_s"].sum()) / 3600.0 if "control_s" in ep else math.nan

    def rate(col: str) -> float:
        if not hours > 0 or col not in ep or ep[col].isna().any():
            return math.nan
        return float(ep[col].sum()) / hours

    out = {"interventions_per_h": rate("interventions")}
    for k in SPLIT_KEYS:
        col = f"split_{k}"
        out[col] = float(mean[col]) if col in mean else math.nan
        out[f"{col}_per_h"] = rate(col)
    return out


class Validator:
    """Scores the deterministic policy on fixed selection windows: appends a row to
    `validation.csv` (and the per-window rows to `validation_episodes.csv`) and saves the
    model as `best_model.zip` whenever its mean cost is the lowest so far.

    Part 7 options (D71, D72):

    - `workers > 1`: the windows are played by a persistent spawned `EpisodePool` (`pool`,
      or one the validator makes from `backend` / `config` / `work_root` and closes in
      `close()`); each scored model is saved first (under `keep_dir`, else a scratch
      file `_validation_model_step_<n>.zip`, deleted after the validation). Seeded
      episodes: the costs equal the in-process ones (`env`, `workers` 1).
    - `pool_factory` (instead of `pool`; `--eval-pool transient`): a pool is made by
      `prepare()` (the validation callback calls it `PREWARM_STEPS` ahead) or at the
      validation, and closed right after it.
    - `keep_dir`: every validated model is kept as `<keep_dir>/step_<n>.zip`.
    - `pick_from_step`: validations at `step >= pick_from_step` are "late": their costs go
      to `late_costs`, and the lowest one is the pick (`pick_step`, `pick_cost`), saved
      as `pick_model.zip`.
    - `extended`: per-window rows with `control_s` and the D77 split; validation.csv adds
      `split_*_per_h`, `interventions_per_h` (per hour of control) and `late`."""

    def __init__(
        self,
        env,
        seeds: Sequence[int],
        baselines: Mapping[str, float],
        out_dir: Path,
        workers: int = 1,
        keep_dir: Path | None = None,
        pick_from_step: int | None = None,
        *,
        pool: EpisodePool | None = None,
        algo: str = "dqn",
        extended: bool = False,
        backend: EnvBackend | None = None,
        config: str | Path | None = None,
        work_root: Path | None = None,
        pool_factory: Callable[[], EpisodePool] | None = None,
    ) -> None:
        self.env, self.seeds = env, [int(s) for s in seeds]
        self.baselines = dict(baselines)
        self.out_dir = Path(out_dir)
        self.workers = max(int(workers), 1)
        self.keep_dir = Path(keep_dir) if keep_dir is not None else None
        self.pick_from_step = int(pick_from_step) if pick_from_step is not None else None
        self.algo, self.extended = algo, bool(extended)
        self.rows: list[dict] = []
        self.episodes: list[dict] = []
        self.best_cost, self.best_step = math.inf, None
        self.pick_cost, self.pick_step = math.inf, None
        self.late_costs: list[float] = []
        self.late_rows: list[dict] = []
        self.last_step: int | None = None
        self.eval_s = 0.0
        self.pool, self._own_pool = pool, False
        self.pool_factory = pool_factory if pool is None else None
        if self.workers > 1 and pool is None and pool_factory is None:
            if backend is None or config is None:
                raise ValueError("a pooled Validator needs `pool` or `backend` and `config`")
            self.pool = EpisodePool(backend, config, self.workers, "train", work_root, extended)
            self._own_pool = True
        if self.workers <= 1 and env is None and pool_factory is None:
            raise ValueError("an in-process Validator needs `env`")
        if self.keep_dir is not None:
            self.keep_dir.mkdir(parents=True, exist_ok=True)

    @property
    def late_mean(self) -> float:
        """Mean selection cost of the late validations (D72); NaN before the first."""
        return float(np.mean(self.late_costs)) if self.late_costs else math.nan

    def late_totals(self) -> dict[str, float]:
        """Mean per-class totals (`TOTAL_KEYS`) of the late validations (D73 guard)."""
        if not self.late_rows:
            return dict.fromkeys(TOTAL_KEYS, math.nan)
        return {k: float(np.mean([r[k] for r in self.late_rows])) for k in TOTAL_KEYS}

    @property
    def transient(self) -> bool:
        """The pool is made per validation (`pool_factory`)."""
        return self.pool_factory is not None

    def prepare(self) -> None:
        """Transient pool: make and start it now, ahead of the next validation (no-op
        otherwise or when it exists)."""
        if self.transient and self.pool is None:
            self.pool = self.pool_factory()
            self.pool.start()

    def close(self) -> None:
        """Close the validator's own or transient pool (a pool passed in is closed by its
        owner)."""
        if (self._own_pool or self.transient) and self.pool is not None:
            self.pool.close()
            self.pool = None

    def _play(self, model, step: int) -> tuple[list[dict], Path | None]:
        """Per-window rows of `model` and the file it was saved to (None in-process
        without `keep_dir`)."""
        saved = None
        pooled = self.pool is not None or self.transient
        if self.keep_dir is not None:
            saved = self.keep_dir / f"step_{int(step)}.zip"
        elif pooled:  # one scratch file per validation, deleted after it (__call__)
            saved = self.out_dir / f"{SCRATCH_MODEL_PREFIX}{int(step)}.zip"
        if saved is not None:
            model.save(str(saved))
        if pooled:
            self.prepare()
            tasks = [("model", saved.as_posix(), self.algo, s) for s in self.seeds]
            try:
                rows = self.pool.map(tasks)
            finally:
                if self.transient:
                    self.close()
        else:
            act = model_policy(model)
            rows = [summary_row(play_episode(self.env, act, s), self.extended) for s in self.seeds]
        eps = [{"step": int(step), "seed": s} | r for s, r in zip(self.seeds, rows, strict=True)]
        return eps, saved

    def _save_copy(self, model, saved: Path | None, name: str) -> None:
        if saved is not None and saved.parent == self.keep_dir:
            shutil.copyfile(saved, self.out_dir / name)
        else:
            model.save(str(self.out_dir / name))

    def __call__(self, model, step: int) -> dict:
        t0 = time.perf_counter()
        eps, saved = self._play(model, step)
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
        if self.extended:
            row |= _rates_per_hour(ep, mean)
        best = cost < self.best_cost
        if best:
            self.best_cost, self.best_step = cost, int(step)
            self._save_copy(model, saved, "best_model.zip")
        row["best"] = best
        if self.pick_from_step is not None:
            late = int(step) >= self.pick_from_step
            row["late"] = late
            if late:
                self.late_costs.append(cost)
                self.late_rows.append(row)
                if cost < self.pick_cost:
                    self.pick_cost, self.pick_step = cost, int(step)
                    self._save_copy(model, saved, PICK_MODEL_FILE)
        if saved is not None and saved.parent != self.keep_dir:
            saved.unlink(missing_ok=True)
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
    `every` (`eval_every_steps`); a transient pool is started `min(PREWARM_STEPS,
    every // 5)` steps ahead (`Validator.prepare`)."""
    prewarm = min(PREWARM_STEPS, int(every) // 5)
    from stable_baselines3.common.callbacks import BaseCallback

    class ValidationCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.next = int(every)

        def _on_step(self) -> bool:
            if validator.transient and self.num_timesteps >= self.next - prewarm:
                validator.prepare()
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


def is_clean_version(cv: str) -> bool:
    """A code version naming a commit that can reproduce the run (not `-dirty`, not
    `unknown`)."""
    return bool(cv) and cv != "unknown" and not cv.endswith("-dirty")


def check_clean(rl_cfg: Mapping, config: str | Path, allow_dirty: bool) -> str | None:
    """`train.require_clean` (D80): the code version, refused when it is not a clean
    commit unless `allow_dirty`. None when the config does not require a clean tree."""
    if not bool((rl_cfg.get("train") or {}).get("require_clean", False)):
        return None
    cv = code_version()
    if not is_clean_version(cv) and not allow_dirty:
        raise ConfigError(
            f"{Path(config).as_posix()}: train.require_clean and the code version is {cv!r}; "
            "commit first or pass --allow-dirty (recorded in the run, D80)"
        )
    return cv


# -- provenance (D81) ------------------------------------------------------------------
def cpu_name() -> str:
    """The processor's marketing name when the OS gives it, else `platform.processor()`."""
    if sys.platform == "win32":
        try:
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def machine_info() -> dict:
    """CPU, platform, the head of `torch.__config__.parallel_info()` (in this process,
    after `set_torch_threads`), the OMP_* / MKL_* / OPENBLAS_* environment variables of
    this process and the `CHILD_ENV` values of the spawned workers (`child_env`)."""
    import torch

    info = [ln.strip() for ln in torch.__config__.parallel_info().splitlines() if ln.strip()]
    return {
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "torch_parallel_info": info[:PARALLEL_INFO_LINES],
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)},
        "child_env": child_env_values(),
    }


def calibration_files(scenario: Mapping, day_types: Sequence[str]) -> list[str]:
    """The base scenario's calibration file and its sibling per randomized day type
    (`<stem>_<day_type>.yaml`, as `rl.env.day_type_calibration` picks them)."""
    demand = scenario.get("demand") or {}
    base = demand.get("calibration")
    if not base:
        return []
    files = [Path(base)]
    for dt in day_types:
        if dt != demand.get("day_type"):
            files.append(Path(base).with_name(f"{Path(base).stem}_{dt}{Path(base).suffix}"))
    return list(dict.fromkeys(p.as_posix() for p in files))


def input_hashes(rl_cfg: Mapping) -> dict:
    """Base scenario `config_hash` and `demand_hash`, and the sha256 of the calibration
    files, the incident sites and the net file (D81). None where a file is absent."""
    base = rl_cfg.get("base_config")
    out: dict[str, Any] = {
        "base_config": base,
        "base_config_hash": None,
        "base_demand_hash": None,
        "calibration_sha256": {},
        "incident_sites_sha256": None,
        "net_sha256": None,
    }
    rand = rl_cfg.get("randomize") or {}
    sites = (rand.get("incidents") or {}).get("sites")
    if sites:
        out["incident_sites"] = Path(sites).as_posix()
        out["incident_sites_sha256"] = text_sha256(sites)
    if not base:
        return out
    try:
        scenario = load_scenario(base)
    except (ConfigError, OSError) as e:
        log.warning("provenance: base scenario %s not loaded (%s)", base, e)
        return out
    out["base_config_hash"] = config_hash(scenario)
    out["base_demand_hash"] = demand_hash(scenario)
    files = calibration_files(scenario, list(rand.get("day_types") or {}))
    out["calibration_sha256"] = {f: text_sha256(f) for f in files}
    net = (scenario.get("network") or {}).get("net_file")
    out["net_sha256"] = xml_sha256(net)
    return out


def config_files(config: str | Path) -> list[dict]:
    """The files `config` is assembled from (bases first), with their text sha256."""
    return [
        {"path": p.as_posix(), "text_sha256": text_sha256(p)} for p in config_sources(Path(config))
    ]


def provenance_doc(
    rl_cfg: Mapping,
    config: str | Path,
    *,
    run: str,
    algo: str,
    seed: int,
    steps: int,
    steps_done: int,
    n_envs: int,
    threads: int,
    hp: Mapping,
    hp_config: Mapping,
    n_features: int | None,
    selection: Sequence[int],
    validator: Validator,
    pick_from: float | None,
    allow_dirty: bool,
    code_version_: str,
    machine: Mapping,
    pick_model_sha256: str | None,
    pick_policy_sha256: str | None,
    eval_workers: int = 1,
    eval_pool: str = "in-process",
) -> dict:
    """provenance.json (D81): what a run was made from, on which machine, and its pick."""
    obs = rl_cfg.get("observation") or {}
    decision = rl_cfg.get("decision") or {}
    return {
        "run": run,
        "variant": rl_cfg.get("variant"),
        "run_config_sha": run_config_sha(rl_cfg),
        "config": Path(config).as_posix(),
        "sources": config_files(config),
        "algo": algo,
        "seed": int(seed),
        "steps": int(steps),
        "steps_done": int(steps_done),
        "n_envs": int(n_envs),
        "reward": rl_cfg.get("reward"),
        "observation": {
            "version": obs.get("version"),
            "drop": list(obs.get("drop") or []),
            "n_features": n_features,
        },
        "decision_interval_s": decision.get("interval_s"),
        **input_hashes(rl_cfg),
        "machine": dict(machine),
        "learner_threads": int(threads),
        "worker_threads": int(threads),
        "eval_worker_threads": 1,
        "eval_workers": int(eval_workers),
        "eval_pool": eval_pool,
        "schedules": schedule_strings(hp_config),
        "exploration_steps": hp_config.get("exploration_steps"),
        "exploration_fraction": hp.get("exploration_fraction"),
        "update_to_data": update_to_data(algo, hp, n_envs),
        "selection_seeds": [int(s) for s in selection],
        "selection_metric": SELECTION_METRIC,
        "pick_from": pick_from,
        "pick_from_step": validator.pick_from_step,
        "pick_step": validator.pick_step,
        "pick_val_cost": None if validator.pick_step is None else validator.pick_cost,
        "late_mean": None if not validator.late_costs else validator.late_mean,
        "late_n": len(validator.late_costs),
        "late_totals": None if not validator.late_costs else validator.late_totals(),
        "pick_model_sha256": pick_model_sha256,
        "pick_policy_sha256": pick_policy_sha256,
        "allow_dirty": bool(allow_dirty),
        "require_clean": bool((rl_cfg.get("train") or {}).get("require_clean", False)),
        "code_version": code_version_,
        "created_at": datetime.now(UTC).isoformat(),
    }


def json_safe(obj: Any) -> Any:
    """`obj` with NaN / inf floats as None, so `json.dumps(allow_nan=False)` accepts it."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    return obj


def train(
    config: str | Path = RL_CONFIG,
    algo: str | None = None,
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
    eval_workers: int | None = None,
    eval_pool: str | None = None,
    force: bool = False,
    allow_dirty: bool = False,
    argv: Sequence[str] | None = None,
) -> Path:
    """Train one agent and write its run folder (module docstring); returns the folder.
    `algo` defaults to the config's `algo` key (else dqn)."""
    rl_cfg = load_config(config)
    algo = resolve_algo(rl_cfg, algo)
    variant = rl_cfg.get("variant")
    part7 = variant is not None
    tcfg = rl_cfg.get("train") or {}
    if part7 and not steps:
        raise ConfigError(f"{Path(config).as_posix()}: a Part 7 run config needs --steps")
    s6 = (tcfg.get("s6") or {}).get(algo) or {}
    steps = int(steps or s6.get("steps") or 200_000)
    n_envs = int(n_envs or tcfg.get("n_envs", 8))
    threads = int(threads or tcfg.get("torch_threads", 1))
    eval_every = int(eval_every or tcfg.get("eval_every_steps", 10_000))
    checkpoint_every = int(checkpoint_every or tcfg.get("checkpoint_every_steps", 50_000))
    eval_workers = int(eval_workers or tcfg.get("eval_workers", 1))
    eval_pool = eval_pool or "persistent"
    if eval_pool not in EVAL_POOLS:
        raise ConfigError(f"eval_pool must be one of {EVAL_POOLS}, not {eval_pool!r}")
    transient = eval_workers > 1 and eval_pool == "transient"
    pick_from = tcfg.get("pick_from")
    pick_from = float(pick_from) if pick_from is not None else None
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
    check_clean(rl_cfg, config, allow_dirty)
    cfg_sha = text_sha256(config)
    name_sha = run_config_sha(rl_cfg) if part7 else cfg_sha
    out_root = Path(out_root or (rl_cfg.get("outputs") or {}).get("runs") or RL_RUNS_DIR)
    check_out_root(out_root)
    out = out_root / run_name(algo, seed, steps, name_sha, variant)
    prepare_out_dir(out, force)
    work_root = out_root / "_work"
    train_seeds = seed_range(rl_cfg["seeds"]["train"])
    selection = [int(s) for s in (validation_seeds or seed_range(rl_cfg["seeds"]["selection"]))]
    hp_config = hyperparameters(rl_cfg, algo, hp_overrides)  # as written (provenance)
    hp = hyperparameters(rl_cfg, algo, hp_overrides, steps=steps)
    pick_from_step = math.ceil(pick_from * steps) if pick_from is not None else None
    started_at = datetime.now(UTC).isoformat()
    t_start = time.perf_counter()

    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    set_torch_threads(threads)
    set_random_seed(int(seed))

    pool: EpisodePool | None = None
    validator: Validator | None = None
    venv = None
    eval_env = None
    try:
        pool_factory = partial(
            EpisodePool, backend, config, eval_workers, "train", work_root, part7
        )
        if eval_workers > 1 and not transient:  # persistent pool, also plays the baselines (D71)
            pool = pool_factory()

        # 1. reference costs on the selection windows (cached for the whole run)
        t0 = time.perf_counter()
        if baseline_workers is not None:
            workers = baseline_workers
        elif transient:  # a pool for the baselines only
            workers = eval_workers
        else:
            workers = n_envs if backend.name == "sumo" else 1
        base = baseline_frame(
            backend,
            config,
            selection,
            workers,
            work_root=work_root,
            extended=part7,
            pool=pool if baseline_workers is None else None,
        )
        base.to_csv(out / "baselines.csv", index=False)
        baselines = {
            k: float(v) for k, v in base.groupby("label", sort=False)["cost"].mean().items()
        }
        baseline_s = time.perf_counter() - t0
        log.info(
            "baselines on %d selection windows (%.1f s): %s",
            len(selection),
            baseline_s,
            {k: round(v, 1) for k, v in baselines.items()},
        )

        # 2. training envs, validation env (in-process validation only), model
        factories = [
            EnvFactory(backend, config, i, n_envs, train_seeds, work_root, threads)
            for i in range(n_envs)
        ]
        with child_env():  # SubprocVecEnv starts its workers here
            if vec != "subproc":
                venv = DummyVecEnv(factories)
            elif part7:  # env processes without SB3 / torch (memory, gate)
                venv = light_subproc_vec_env(factories, start_method="spawn")
            else:
                venv = SubprocVecEnv(factories, start_method="spawn")
        venv = VecMonitor(venv, str(out / "monitor.csv"))
        if eval_workers <= 1:
            eval_env = backend.env(config, mode="train", seeds=selection, work_root=work_root)
        n_features = int(np.prod(venv.observation_space.shape))
        model = make_model(algo, venv, hp, seed)
        model.set_logger(configure(str(out), ["csv"]))
        validator = Validator(
            eval_env,
            selection,
            baselines,
            out,
            workers=eval_workers,
            keep_dir=out / VAL_DIR if part7 else None,
            pick_from_step=pick_from_step,
            pool=pool,
            algo=algo,
            extended=part7,
            pool_factory=pool_factory if transient else None,
        )
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
        if venv is not None:
            venv.close()
        if eval_env is not None:
            eval_env.close()
        if pool is not None:
            pool.close()
        if validator is not None:
            validator.close()

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
    pick_file = out / PICK_MODEL_FILE
    has_pick = validator.pick_step is not None and pick_file.is_file()
    if part7 and validator.rows:  # mark the pick in validation.csv
        val = pd.DataFrame(validator.rows)
        val["pick"] = val["step"] == validator.pick_step
        val.to_csv(out / "validation.csv", index=False)
    prov = provenance_doc(
        rl_cfg,
        config,
        run=out.name,
        algo=algo,
        seed=seed,
        steps=steps,
        steps_done=int(model.num_timesteps),
        n_envs=n_envs,
        threads=threads,
        hp=hp,
        hp_config=hp_config,
        n_features=n_features,
        selection=selection,
        validator=validator,
        pick_from=pick_from,
        allow_dirty=allow_dirty,
        code_version_=cv,
        machine=machine_info(),
        pick_model_sha256=file_sha256(pick_file) if has_pick else None,
        pick_policy_sha256=policy_sha256(load_model(pick_file, algo)) if has_pick else None,
        eval_workers=eval_workers,
        eval_pool=eval_pool if eval_workers > 1 else "in-process",
    )
    prov = json_safe(prov)
    (out / PROVENANCE_FILE).write_text(json.dumps(prov, indent=2), encoding="utf-8")
    if part7:  # Part 7 manifests carry the provenance keys too (D81)
        manifest = json_safe({**manifest, **{k: v for k, v in prov.items() if k not in manifest}})
    # written last: a manifest marks a finished run
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(
        "%s: %d steps in %.0f s (%.0f steps/s, validation %.0f s); best cost %.1f at %s "
        "(vs S0 %+.1f %%, vs stay %+.1f %%)%s",
        out.name,
        model.num_timesteps,
        manifest["wall_s"],
        manifest["steps_per_s"],
        validator.eval_s,
        validator.best_cost,
        validator.best_step,
        best_vs.get("S0", math.nan),
        best_vs.get("stay", math.nan),
        (
            f"; pick {validator.pick_cost:.1f} at {validator.pick_step}, "
            f"late mean {validator.late_mean:.1f} over {len(validator.late_costs)}"
            if validator.pick_step is not None
            else ""
        ),
    )
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, default=RL_CONFIG)
    p.add_argument(
        "--algo", choices=ALGOS, help="algorithm (default: the config's `algo`, else dqn)"
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, help="decision steps (default: rl.yaml train.s6)")
    p.add_argument("--n-envs", type=int, help="SubprocVecEnv workers (default: rl.yaml)")
    p.add_argument("--out", type=Path, help="runs root (default: rl.yaml outputs.runs)")
    p.add_argument("--eval-every", type=int, help="validation interval (default: rl.yaml)")
    p.add_argument("--baseline-workers", type=int, help="processes for the baselines")
    p.add_argument(
        "--eval-workers", type=int, help="validation pool processes (default: train.eval_workers)"
    )
    p.add_argument(
        "--eval-pool",
        choices=EVAL_POOLS,
        help="validation pool lifetime (default persistent, D71; transient: started ahead of "
        "each validation and closed after it, less memory per job)",
    )
    p.add_argument(
        "--validation-seeds",
        nargs="+",
        metavar="SEED",
        help="validation windows instead of seeds.selection: N or LO-HI (development runs)",
    )
    p.add_argument("--force", action="store_true", help="redo a finished run folder")
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="train from an uncommitted tree although train.require_clean (recorded, D80)",
    )
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
            eval_workers=args.eval_workers,
            eval_pool=args.eval_pool,
            validation_seeds=parse_seeds(args.validation_seeds) if args.validation_seeds else None,
            force=args.force,
            allow_dirty=args.allow_dirty,
            argv=sys.argv if argv is None else ["traffic_twin.rl.train", *argv],
        )
    except (ConfigError, FileExistsError) as e:
        raise SystemExit(f"error: {e}") from None
    print(out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

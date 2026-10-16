"""RL controller `rl` (Part 6, D62; contract 4.7): a trained stable-baselines3 policy
behind the same decision clock (`rl/clock.py`) and observation code (`rl/features.py`)
as `rl.env.TrafficEnv`, so a run of `configs/pitie_rl.yaml` replays exactly what the
policy did in training (`tests/test_rl_parity.py`).

`configs/pitie_rl.yaml` selects it: `controller: rl`, `rl: {model, config}`, where
`model` is an SB3 zip with a `manifest.json` next to it (written by `rl.train` /
`rl.evaluate select`) and `config` is `configs/rl.yaml`. The manifest needs `algo` (dqn
or ppo), `obs_version` and `reward_version`; both versions must equal those of the
config and of this code. Anything missing (model, manifest, stable-baselines3, a version)
raises ConfigError: never a silent fallback to another controller, which would corrupt
A-06.

Part 7 (D86, D73):
- `rl.model_sha256`, when the scenario gives it (`configs/pitie_rl_s3.yaml`), pins the
  zip: another file is a ConfigError, so the scenario's `config_hash` follows the model;
- `observation.drop` of `rl.config` must equal the run's recorded `obs_drop`
  (`manifest.json`, else `provenance.json` next to the model; missing = no drop);
- the zip is loaded with `custom_objects` that replace the pickled training schedules
  (`learning_rate`, `lr_schedule`, `exploration_schedule`, `clip_range`) by a constant:
  no pickled training function is run and the policy weights are unchanged;
- `report()` adds the D77 intervention split `iv_split` (`rl/clock.py`) and `obs_drop`.

Importing this module imports nothing heavy (the registry imports every controller
module on first lookup): stable-baselines3 and torch are imported inside `from_config`.

Every step `decide` asks the clock. At a decision point (D56) it samples the junction
(`JunctionSampler`, bound in `bind`), builds the observation and takes the policy's
deterministic action (0 stay, 1 advance, D55); otherwise it holds the current stage. The
Safety Guard and the preemption stay in `SimSession.step`, with priority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from traffic_twin.config import ConfigError, get, load_config
from traffic_twin.control.base import Controller, Observation
from traffic_twin.control.registry import register
from traffic_twin.rl.clock import IV_SPLIT_KEYS, DecisionClock, InterventionSplit
from traffic_twin.rl.features import (
    NORM_KEYS,
    OBS_VERSION,
    VEH_LENGTH_M,
    check_drop,
    feature_names,
    observation,
    site_spec,
)
from traffic_twin.rl.reward import REWARD_VERSION

MANIFEST_FILE = "manifest.json"  # next to the model zip
PROVENANCE_FILE = "provenance.json"  # Part 7 runs (D81), next to the manifest
ALGOS = ("dqn", "ppo")
ACTION_MODES = ("binary",)  # D55; free order with masking is a Part 7 option


def check_rl_config(rl_cfg: Mapping, where: str = "rl config") -> None:
    """ConfigError unless `configs/rl.yaml` matches this code: observation and reward
    versions, the binary action, a decision interval and every observation norm."""
    obs_v = get(rl_cfg, "observation.version", None)
    if obs_v != OBS_VERSION:
        raise ConfigError(f"{where}: observation.version {obs_v!r}, this code is {OBS_VERSION}")
    rew_v = get(rl_cfg, "reward.version", None)
    if rew_v != REWARD_VERSION:
        raise ConfigError(f"{where}: reward.version {rew_v!r}, this code is {REWARD_VERSION}")
    mode = get(rl_cfg, "action.mode", None)
    if mode not in ACTION_MODES:
        raise ConfigError(f"{where}: action.mode {mode!r} not in {ACTION_MODES}")
    interval = get(rl_cfg, "decision.interval_s", None)
    if isinstance(interval, bool) or not isinstance(interval, int | float) or interval <= 0:
        raise ConfigError(f"{where}: decision.interval_s must be > 0, got {interval!r}")
    missing = [k for k in NORM_KEYS if k not in (rl_cfg.get("observation") or {})]
    if missing:
        raise ConfigError(f"{where}: observation misses {missing}")
    try:
        check_drop(get(rl_cfg, "observation.drop", None))
    except ValueError as e:
        raise ConfigError(f"{where}: {e}") from None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _constant(_progress_remaining: float) -> float:
    """Stands for a training schedule at inference (never used to act)."""
    return 0.0


# pickled training schedules replaced at load (`custom_objects`): the policy weights do
# not depend on them, and no pickled function of the training code is run
CUSTOM_OBJECTS = {
    "learning_rate": _constant,
    "lr_schedule": _constant,
    "exploration_schedule": _constant,
    "clip_range": _constant,
}


def recorded_obs_drop(model_path: Path, manifest: Mapping) -> tuple[str, ...]:
    """The observation groups the model was trained without: `obs_drop` or
    `observation.drop` of the manifest, else of `provenance.json` next to it (D81); ()
    when neither records one (Part 6 runs)."""

    def drop_of(doc: Mapping) -> tuple[bool, object]:
        if "obs_drop" in doc:
            return True, doc["obs_drop"]
        obs = doc.get("observation")
        if isinstance(obs, Mapping) and "drop" in obs:
            return True, obs["drop"]
        return False, None

    found, drop = drop_of(manifest)
    if not found:
        prov_path = model_path.with_name(PROVENANCE_FILE)
        if prov_path.is_file():
            try:
                prov = json.loads(prov_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise ConfigError(f"{prov_path}: unreadable provenance ({e})") from e
            if isinstance(prov, Mapping):
                _found, drop = drop_of(prov)
    try:
        return check_drop(drop)
    except ValueError as e:
        raise ConfigError(f"{model_path}: recorded {e}") from None


def _load_policy(algo: str, path: Path):
    """The SB3 model (lazy import; ConfigError when stable-baselines3 is missing),
    loaded with `CUSTOM_OBJECTS`."""
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ConfigError(
            "controller 'rl' needs stable-baselines3 and torch (pip install -e .[rl])"
        ) from e
    cls = {"dqn": DQN, "ppo": PPO}[algo]
    try:
        return cls.load(path, device="cpu", custom_objects=dict(CUSTOM_OBJECTS))
    except Exception as e:  # a corrupt or foreign zip
        raise ConfigError(f"rl.model: cannot load {path} as {algo}: {e}") from e


@register("rl")
class RLController(Controller):
    def __init__(
        self,
        policy,
        rl_cfg: Mapping,
        n_stages: int,
        min_green_s: float,
        model_version: str,
        model_sha256: str,
    ) -> None:
        self.policy = policy
        self.rl_cfg = rl_cfg
        self.norm = dict(rl_cfg["observation"])
        self.obs_drop = check_drop(self.norm.get("drop"))
        self.n_stages = int(n_stages)
        self.model_version, self.model_sha256 = model_version, model_sha256
        decision = rl_cfg["decision"]
        self.clock = DecisionClock(
            float(decision["interval_s"]),
            float(min_green_s),
            bool(decision.get("skip_during_preemption", True)),
        )
        self.split = InterventionSplit(self.clock.interval_s)
        self._last: tuple[Observation, int] | None = None  # (obs, request) of the last step
        self.sim = None
        self.sampler = None
        self.site = None
        self.decisions = 0
        self.advances = 0

    @classmethod
    def from_config(cls, cfg: dict) -> RLController:
        """cfg["rl"]{model, config, model_sha256 (optional pin)}; lazy SB3 import; the
        model manifest's obs / reward versions must match rl.yaml (and this code), the
        pin the zip and the recorded `obs_drop` the config's `observation.drop`, else
        ConfigError."""
        model = get(cfg, "rl.model", None)
        config = get(cfg, "rl.config", None)
        if not model or not config:
            raise ConfigError("controller 'rl' needs rl.model and rl.config")
        model_path, config_path = Path(model), Path(config)
        if not config_path.is_file():
            raise ConfigError(f"rl.config: file not found: {config_path}")
        rl_cfg = load_config(config_path)
        check_rl_config(rl_cfg, str(config_path))
        if not model_path.is_file():
            raise ConfigError(f"rl.model: file not found: {model_path} (train and select one)")
        manifest_path = model_path.with_name(MANIFEST_FILE)
        if not manifest_path.is_file():
            raise ConfigError(f"rl.model: no {MANIFEST_FILE} next to {model_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ConfigError(f"{manifest_path}: unreadable manifest ({e})") from e
        for key, want in (
            ("obs_version", rl_cfg["observation"]["version"]),
            ("reward_version", rl_cfg["reward"]["version"]),
        ):
            if manifest.get(key) != want:
                raise ConfigError(
                    f"{manifest_path}: {key} {manifest.get(key)!r} differs from {config_path}"
                    f" ({want!r})"
                )
        algo = str(manifest.get("algo", "")).lower()
        if algo not in ALGOS:
            raise ConfigError(f"{manifest_path}: algo must be one of {ALGOS}, got {algo!r}")
        sha = file_sha256(model_path)
        pin = get(cfg, "rl.model_sha256", None)
        if pin is not None and (not isinstance(pin, str) or pin.lower() != sha):
            raise ConfigError(
                f"rl.model_sha256 {str(pin)[:12]}... does not match {model_path} ({sha[:12]}...)"
            )
        recorded = manifest.get("model_sha256")
        if isinstance(recorded, str) and recorded != sha:
            raise ConfigError(f"{manifest_path}: model_sha256 does not match {model_path}")
        want_drop = check_drop(get(rl_cfg, "observation.drop", None))
        trained_drop = recorded_obs_drop(model_path, manifest)
        if want_drop != trained_drop:
            raise ConfigError(
                f"{config_path}: observation.drop {list(want_drop)} differs from the "
                f"obs_drop {list(trained_drop)} recorded for {model_path}"
            )
        policy = _load_policy(algo, model_path)
        n_actions = getattr(policy.action_space, "n", None)
        if n_actions != 2:
            raise ConfigError(f"rl.model: expected a Discrete(2) policy, got {policy.action_space}")
        version = manifest.get("model_version") or f"rl-{algo}-{sha[:8]}"
        return cls(
            policy,
            rl_cfg,
            n_stages=len(cfg["network"]["stages"]),
            min_green_s=float(get(cfg, "safety.min_green_s", 0.0)),
            model_version=str(version),
            model_sha256=sha,
        )

    def bind(self, sim) -> None:
        """Build the sampler and the site spec of the started session; the policy's
        observation size must match the site's."""
        from traffic_twin.sim.sampler import JunctionSampler

        self.sim = sim
        self.sampler = JunctionSampler(sim)
        self.site = site_spec(sim, float(self.norm.get("veh_length_m", VEH_LENGTH_M)))
        n_obs = len(feature_names(self.site, self.obs_drop))
        shape = tuple(getattr(self.policy.observation_space, "shape", ()) or ())
        if shape != (n_obs,):
            raise ConfigError(f"rl.model observes {shape}, the junction gives ({n_obs},)")

    def reset(self, seed: int) -> None:
        self.clock.reset()
        self.decisions = 0
        self.advances = 0
        self.split.reset(getattr(self.sim, "guard", None))
        self._last = None
        self.clear_events()

    def _classify_last_step(self) -> None:
        """D77: classify the guard's interventions of the step after the last `decide`."""
        if self._last is not None and self.sim is not None:
            obs, request = self._last
            self.split.after_step(self.sim.guard, obs, request)
        self._last = None

    def decide(self, obs: Observation) -> int:
        """Clock every step; at decision points: sample -> observation -> predict."""
        self._classify_last_step()
        if not self.clock.is_decision_point(obs):
            request = self.clock.request(obs, None, self.n_stages)
        else:
            if self.sampler is None:
                raise RuntimeError("RLController.decide before bind(sim)")
            x = observation(self.sampler.sample(), self.site, self.norm)
            action, _state = self.policy.predict(x, deterministic=True)
            action = int(np.asarray(action).item())
            self.decisions += 1
            self.advances += action
            request = self.clock.request(obs, action, self.n_stages)
            self.split.decided(obs, action, self.sim.guard.max_green_s)
        self._last = (obs, request)
        return request

    def report(self) -> dict:
        """{model_version, model_sha256, obs_version, reward_version, decisions, advances,
        iv_split (D77: cut, replaced, cut_foreseeable), obs_drop}."""
        self._classify_last_step()
        guard = getattr(self.sim, "guard", None)
        iv_split = (
            self.split.result(guard) if guard is not None else dict.fromkeys(IV_SPLIT_KEYS, 0)
        )
        return {
            "model_version": self.model_version,
            "model_sha256": self.model_sha256,
            "obs_version": OBS_VERSION,
            "reward_version": REWARD_VERSION,
            "decisions": self.decisions,
            "advances": self.advances,
            "iv_split": iv_split,
            "obs_drop": list(self.obs_drop),
        }

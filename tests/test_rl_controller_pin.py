"""RL controller checks of Part 7 (D73, D77, D86): the `rl.model_sha256` pin, the
observation-drop check against the run's recorded `obs_drop`, the `custom_objects` load
(same policy weights as a plain SB3 load, no pickled schedule kept) and the `report()`
keys. Random-init models on a dummy env; no SUMO."""

import copy
import json

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from traffic_twin.config import ConfigError  # noqa: E402
from traffic_twin.control.rl import (  # noqa: E402
    CUSTOM_OBJECTS,
    PROVENANCE_FILE,
    RLController,
    file_sha256,
    recorded_obs_drop,
)
from traffic_twin.rl.features import OBS_VERSION  # noqa: E402
from traffic_twin.rl.reward import REWARD_VERSION  # noqa: E402

RL6 = "configs/rl.yaml"
V1 = "configs/rl7/v1.yaml"  # observation.drop: [pedestrian]
V2 = "configs/rl7/v2.yaml"


def policy_sha256(model) -> str:
    """sha256 of the policy parameters (as `rl.train.policy_sha256`)."""
    import hashlib

    h = hashlib.sha256()
    for name, tensor in model.policy.state_dict().items():
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def lin(a: float, b: float):
    """A pickled-by-value training schedule (like rl.train's "lin:a:b")."""
    return lambda progress: b + (a - b) * progress


def save_model(folder, algo: str, n_obs: int, manifest_extra: dict | None = None):
    """A random-init SB3 model with lin schedules, saved with a manifest."""
    import gymnasium as gym
    from stable_baselines3 import DQN, PPO

    class Spaces(gym.Env):
        observation_space = gym.spaces.Box(0.0, 1.0, (n_obs,), np.float32)
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(n_obs, np.float32), {}

        def step(self, action):
            return np.zeros(n_obs, np.float32), 0.0, False, True, {}

    folder.mkdir(parents=True, exist_ok=True)
    kw = {"seed": 0, "device": "cpu", "policy_kwargs": {"net_arch": [16, 16]}}
    if algo == "dqn":
        model = DQN("MlpPolicy", Spaces(), learning_rate=lin(5e-4, 5e-5), **kw)
    else:
        model = PPO("MlpPolicy", Spaces(), learning_rate=lin(3e-4, 3e-5), n_steps=64,
                    batch_size=64, clip_range=lin(0.2, 0.1), **kw)  # fmt: skip
    path = folder / "model.zip"
    model.save(path)
    manifest = {"algo": algo, "obs_version": OBS_VERSION, "reward_version": REWARD_VERSION,
                "model_version": f"pin-test-{algo}", **(manifest_extra or {})}  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path, model


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("rl_pin")
    return {
        "full": save_model(root / "full", "dqn", 74),
        "v1": save_model(root / "v1", "dqn", 62, {"observation": {"drop": ["pedestrian"]}}),
        "ppo": save_model(root / "ppo", "ppo", 74),
    }


def scenario(rl_scenario: dict, model, config: str, **rl) -> dict:
    cfg = copy.deepcopy(rl_scenario)
    cfg["rl"] = {"model": str(model), "config": config, **rl}
    return cfg


def test_pin_accepts_the_file_and_refuses_another(rl_scenario, models):
    path, _ = models["full"]
    sha = file_sha256(path)
    ctl = RLController.from_config(scenario(rl_scenario, path, RL6, model_sha256=sha))
    assert ctl.model_sha256 == sha
    assert RLController.from_config(scenario(rl_scenario, path, RL6, model_sha256=sha.upper()))
    for bad in ("0" * 64, sha[:12], 123):
        with pytest.raises(ConfigError, match="model_sha256"):
            RLController.from_config(scenario(rl_scenario, path, RL6, model_sha256=bad))
    other, _ = models["ppo"]
    with pytest.raises(ConfigError, match="model_sha256"):  # the pin names another zip
        RLController.from_config(scenario(rl_scenario, other, RL6, model_sha256=sha))
    assert RLController.from_config(scenario(rl_scenario, path, RL6))  # no pin: Part 6


def test_drop_must_match_the_recorded_obs_drop(rl_scenario, models, tmp_path):
    full, _ = models["full"]
    v1, _ = models["v1"]
    assert RLController.from_config(scenario(rl_scenario, v1, V1)).obs_drop == ("pedestrian",)
    assert RLController.from_config(scenario(rl_scenario, full, V2)).obs_drop == ()
    with pytest.raises(ConfigError, match="observation.drop"):
        RLController.from_config(scenario(rl_scenario, full, V1))  # trained with all 74
    with pytest.raises(ConfigError, match="observation.drop"):
        RLController.from_config(scenario(rl_scenario, v1, RL6))  # trained without peds
    # the provenance file next to the model when the manifest records nothing
    path, _ = save_model(tmp_path / "prov", "dqn", 62)
    assert recorded_obs_drop(path, {}) == ()
    (tmp_path / "prov" / PROVENANCE_FILE).write_text(
        json.dumps({"observation": {"version": OBS_VERSION, "drop": ["pedestrian"]}}),
        encoding="utf-8",
    )
    assert recorded_obs_drop(path, {}) == ("pedestrian",)
    assert recorded_obs_drop(path, {"obs_drop": []}) == ()  # the manifest wins
    assert RLController.from_config(scenario(rl_scenario, path, V1)).obs_drop == ("pedestrian",)
    with pytest.raises(ConfigError, match="recorded"):
        recorded_obs_drop(path, {"obs_drop": ["peds"]})


@pytest.mark.parametrize("name", ["full", "ppo"])
def test_custom_objects_load_keeps_the_policy(rl_scenario, models, name):
    path, model = models[name]
    ctl = RLController.from_config(scenario(rl_scenario, path, RL6))
    algo = type(model)
    plain = algo.load(path, device="cpu")
    assert policy_sha256(ctl.policy) == policy_sha256(plain) == policy_sha256(model)
    x = np.random.default_rng(0).random((32, 74), dtype=np.float32)
    np.testing.assert_array_equal(ctl.policy.predict(x, deterministic=True)[0],
                                  plain.predict(x, deterministic=True)[0])  # fmt: skip
    # the training schedules were replaced, not unpickled
    assert ctl.policy.lr_schedule(1.0) == 0.0 and plain.lr_schedule(1.0) > 0
    if name == "ppo":
        assert ctl.policy.clip_range(1.0) == 0.0 and plain.clip_range(1.0) == pytest.approx(0.2)
    assert set(CUSTOM_OBJECTS) == {"learning_rate", "lr_schedule", "exploration_schedule",
                                   "clip_range"}  # fmt: skip


def test_report_keys_before_a_run(rl_scenario, models):
    v1, _ = models["v1"]
    rep = RLController.from_config(scenario(rl_scenario, v1, V1)).report()
    assert rep["iv_split"] == {"cut": 0, "replaced": 0, "cut_foreseeable": 0}
    assert rep["obs_drop"] == ["pedestrian"]
    assert {"model_version", "model_sha256", "obs_version", "reward_version", "decisions",
            "advances"} <= set(rep)  # fmt: skip

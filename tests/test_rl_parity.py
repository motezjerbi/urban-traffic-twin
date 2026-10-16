"""Train/serve parity of the RL policy (Part 6, D56, D57, D62): one random-init SB3 DQN
drives (a) `TrafficEnv` in full mode (no randomization, no warm-up, 900 s, measurement
and junction monitor off) and (b) `run_scenario.run` on `configs/pitie_rl.yaml` with
`controller: rl` (900 s, measurement and monitor on). The phases shown must be
identical: the env and the controller share the clock and the observation code, and the
removed measurement and monitor are passive. Training seed 1001 only. `@sumo`.

The base ambulance departs at 08:10 (2400 s), after the 900 s window, so both arms move it
to 300 s (`AMB_DEPART_S`, the first trip of `configs/pitie/emergency_probe.yaml`): the
comparison then covers the preemption path too (the clock's `skip_during_preemption`, the
ambulance and preemption features, the guard's preemption target), and both arms must
serve the request with no policy decision while `EXTRA_PREEMPT` is set."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.sumo

SEED = 1001
DURATION_S = 900
AMB_DEPART_S = 300  # inside the window (the base trip is at 2400 s)
REPO_ROOT = Path(__file__).resolve().parent.parent


class PhaseCapture:
    """RunObserver keeping the phases shown by the run."""

    phases: list[int] | None = None

    def bind(self, sim) -> None:
        pass

    def after_step(self, sim) -> None:
        pass

    def finish(self, sim, out_dir, run_id) -> None:
        self.phases = list(sim.phases_shown)


@pytest.fixture
def clock_calls(monkeypatch):
    """Every `DecisionClock.is_decision_point` call as (preemption active, result); the
    env and the controller both use that class. Cleared between the two arms."""
    from traffic_twin.control.base import EXTRA_PREEMPT
    from traffic_twin.rl.clock import DecisionClock

    calls: list[tuple[bool, bool]] = []
    original = DecisionClock.is_decision_point

    def recording(self, obs):
        result = original(self, obs)
        calls.append((obs.extras.get(EXTRA_PREEMPT) is not None, bool(result)))
        return result

    monkeypatch.setattr(DecisionClock, "is_decision_point", recording)
    return calls


def assert_preemption_path(calls: list[tuple[bool, bool]], preemption: dict, arm: str) -> None:
    """The ambulance was preempted and served, and the clock never fired meanwhile."""
    assert preemption["enabled"], arm
    assert preemption["requests"] >= 1 and preemption["served"] == preemption["requests"], (
        arm,
        preemption,
    )
    assert preemption["timeouts"] == 0, (arm, preemption)
    assert any(active for active, _ in calls), f"{arm}: no step with EXTRA_PREEMPT set"
    assert not any(active and fired for active, fired in calls), (
        f"{arm}: a decision fell while EXTRA_PREEMPT was set"
    )


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """A random-init DQN (seed 0) saved with an ad hoc manifest."""
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    from stable_baselines3 import DQN

    from traffic_twin.rl.features import OBS_VERSION
    from traffic_twin.rl.reward import REWARD_VERSION

    class Spaces(gym.Env):
        observation_space = gym.spaces.Box(0.0, 1.0, (74,), np.float32)
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(74, np.float32), {}

        def step(self, action):
            return np.zeros(74, np.float32), 0.0, False, True, {}

    folder = tmp_path_factory.mktemp("rl_model")
    model = DQN("MlpPolicy", Spaces(), seed=0, device="cpu", policy_kwargs={"net_arch": [128, 128]})
    path = folder / "model.zip"
    model.save(path)
    manifest = {"algo": "dqn", "obs_version": OBS_VERSION, "reward_version": REWARD_VERSION,
                "model_version": "parity-test"}  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_env_and_run_scenario_show_the_same_phases(
    model_path, rl_train_scenario, rl_scenario, clock_calls, tmp_path
):
    import yaml
    from stable_baselines3 import DQN

    from traffic_twin.rl.env import TrafficEnv
    from traffic_twin.sim.run_scenario import run

    # (a) the environment, full mode, base scenario with the ambulance at 300 s, 900 s
    # from 07:30
    base = tmp_path / "pitie_rl_amb.yaml"
    extends = (REPO_ROOT / rl_train_scenario["base_config"]).as_posix()
    base.write_text(
        yaml.safe_dump({"extends": [extends], "emergency": {"depart_s": AMB_DEPART_S}}),
        encoding="utf-8",
    )
    rl_cfg = copy.deepcopy(rl_train_scenario)
    rl_cfg["base_config"] = base.as_posix()
    rl_cfg["episode"]["full"] = {"warmup_s": 0, "duration_s": DURATION_S}
    config = tmp_path / "rl.yaml"
    config.write_text(yaml.safe_dump(rl_cfg), encoding="utf-8")
    policy = DQN.load(model_path, device="cpu")
    env = TrafficEnv(config, mode="full", randomize=False, work_root=tmp_path / "env")
    assert env.base["emergency"]["depart_s"] == AMB_DEPART_S
    try:
        x, _ = env.reset(options={"episode_seed": SEED})
        truncated, actions = False, []
        while not truncated:
            action, _ = policy.predict(x, deterministic=True)
            actions.append(int(action))
            x, _r, _term, truncated, info = env.step(int(action))
        env_phases = list(env.sim.phases_shown)
        summary = info["episode_summary"]
    finally:
        env.close()
    assert len(env_phases) == DURATION_S
    assert 0 < sum(actions) < len(actions)  # the policy both stays and advances
    assert summary["danger"] == 0
    assert_preemption_path(clock_calls, summary["preemption"], "env")
    clock_calls.clear()

    # (b) run_scenario with controller rl (measurement and junction monitor on)
    cfg = copy.deepcopy(rl_scenario)
    cfg["duration_s"] = DURATION_S
    cfg["emergency"]["depart_s"] = AMB_DEPART_S
    cfg["rl"] = {"model": str(model_path), "config": "configs/rl.yaml"}
    assert cfg.get("measurement")
    capture = PhaseCapture()
    meta = run(cfg, SEED, tmp_path / "runs", demand_dir=tmp_path / "demand", observers=[capture])
    assert capture.phases is not None and len(capture.phases) >= DURATION_S
    assert capture.phases[:DURATION_S] == env_phases
    control = meta["control"]
    assert control["model_version"] == "parity-test"
    assert control["obs_version"] == "rl-obs-1" and control["reward_version"] == "rl-reward-1"
    assert control["decisions"] >= summary["decisions"] > 0
    assert meta["safety"]["danger"] == 0
    (run_json,) = (tmp_path / "runs").rglob("run.json")
    written = json.loads(run_json.read_text(encoding="utf-8"))
    assert written["preemption"] == meta["preemption"]
    assert_preemption_path(clock_calls, written["preemption"], "run_scenario")
    assert written["preemption"] == summary["preemption"]  # same requests, hold, jumps

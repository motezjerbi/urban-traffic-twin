"""RL training smoke tests on the SUMO-free `StubTrafficEnv` (Part 6, WS5; D60, D61):
the stub honours the TrafficEnv contract, DQN and PPO train for 300 steps with the
validation callback (validation.csv, best_model.zip), the manifest is complete, a saved
model loads back to the same policy, and two runs with the same seed give the same
parameters and the same episode returns."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_twin.rl.stub import (
    OBS_DIM,
    STEP_INFO_KEYS,
    SUMMARY_KEYS,
    TOTAL_KEYS,
    StubTrafficEnv,
    run_baseline,
)
from traffic_twin.rl.train import (
    BASELINES,
    PREEMPT_KEYS,
    RUN_FILES,
    Validator,
    check_out_root,
    hyperparameters,
    pct_change,
    play_episode,
    run_name,
    s6_run_pass,
    sb3_kwargs,
    seed_range,
    stub_backend,
    summary_row,
)

WINDOW_S = 300  # short stub episodes: ~40 decisions
SMALL_HP = {
    "dqn": {
        "learning_starts": 50,
        "buffer_size": 1000,
        "batch_size": 32,
        "net_arch": [16],
        "target_update_interval": 50,
    },
    "ppo": {"n_steps": 64, "batch_size": 32, "n_epochs": 2, "net_arch": [16]},
}


def _train(tmp_path, algo, name="runs", seed=7, steps=300):
    from traffic_twin.rl.train import train

    return train(
        "configs/rl.yaml",
        algo,
        seed,
        steps,
        2,
        tmp_path / name,
        backend=stub_backend(window_s=WINDOW_S),
        vec="dummy",
        hp_overrides=SMALL_HP[algo],
        eval_every=100,
        checkpoint_every=200,
        validation_seeds=[3000, 3001],
    )


# -- the stub env (no SB3 needed) ----------------------------------------------------------
def test_stub_env_contract():
    from gymnasium.utils.env_checker import check_env

    env = StubTrafficEnv(window_s=WINDOW_S)
    check_env(env, skip_render_check=True)
    assert env.action_space.n == 2
    assert env.observation_space.shape == (OBS_DIM,) == (74,)
    obs, info = env.reset(options={"episode_seed": 1000})
    assert "params" in info and info["params"]["episode_seed"] == 1000
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(steps % 2)
        steps += 1
        assert env.observation_space.contains(obs)
        assert set(STEP_INFO_KEYS) <= set(info) and "episode" not in info
        assert not terminated
        if truncated:
            break
    summary = info["episode_summary"]
    assert set(SUMMARY_KEYS) <= set(summary)
    assert set(TOTAL_KEYS) <= set(summary["totals"])
    assert summary["decisions"] == steps and summary["danger"] == 0
    assert summary["totals"]["cost"] > 0 and reward <= 0


def test_stub_is_deterministic_and_follows_the_worker_schedule():
    a, b = StubTrafficEnv(window_s=WINDOW_S), StubTrafficEnv(window_s=WINDOW_S)
    for ctl in ("actuated", "random"):
        assert run_baseline(a, ctl, 1003) == run_baseline(b, ctl, 1003)
    seeds = [10, 11, 12, 13, 14, 15]
    w1 = StubTrafficEnv(window_s=WINDOW_S, worker=1, n_workers=3, seeds=seeds)
    played = [w1.reset()[1]["params"]["episode_seed"] for _ in range(3)]
    assert played == [11, 14, 11]  # worker i plays seeds[i + k * n]
    assert w1.reset(seed=5)[1]["params"]["episode_seed"] == 11  # a seeded reset restarts it


def test_stub_baselines_order_and_unknown_name():
    env = StubTrafficEnv(window_s=900)
    cost = {c: run_baseline(env, c, 3000)["totals"]["cost"] for c in ("actuated", "stay")}
    assert cost["actuated"] < cost["stay"]
    with pytest.raises(ValueError, match="unknown baseline"):
        run_baseline(env, "magic", 3000)


# -- helpers ---------------------------------------------------------------------------
def test_helpers():
    assert seed_range([3000, 3002]) == [3000, 3001, 3002]
    assert run_name("dqn", 0, 200_000, "c0b7dd7b6ca4") == "dqn_s0_200k_c0b7dd7b"
    assert pct_change(95.0, 100.0) == pytest.approx(-5.0)
    assert math.isnan(pct_change(1.0, 0.0))
    kw = sb3_kwargs({"learning_rate": 1e-3, "net_arch": [128, 128]})
    assert kw == {"learning_rate": 1e-3, "policy_kwargs": {"net_arch": [128, 128]}}


def test_hyperparameters_come_from_rl_yaml(rl_train_scenario):
    hp = hyperparameters(rl_train_scenario, "dqn")
    assert hp["learning_rate"] == pytest.approx(5e-4) and hp["net_arch"] == [128, 128]
    assert hyperparameters(rl_train_scenario, "ppo", {"n_steps": 64})["n_steps"] == 64


@pytest.mark.parametrize(
    ("best", "s0", "stay", "expected"),
    [
        (95.0, 100.0, 200.0, True),  # exactly 5 % below S0: passes
        (95.01, 100.0, 200.0, False),  # 4.99 %: fails
        (90.0, 100.0, 94.0, False),  # 4.3 % below stay only: fails
        (90.0, 100.0, None, None),  # missing reference: undecided
    ],
)
def test_s6_run_pass_edges(best, s0, stay, expected):
    base = {"S0": s0} | ({"stay": stay} if stay is not None else {})
    assert s6_run_pass(best, base) is expected


def test_summary_row_flattens_the_summary():
    env = StubTrafficEnv(window_s=WINDOW_S)
    row = summary_row(run_baseline(env, "fixed", 3000))
    assert set(TOTAL_KEYS) <= set(row)
    assert {"iv_ped_wait", "iv_max_green", "interventions", "danger", "decisions"} <= set(row)
    assert row["interventions"] == sum(v for k, v in row.items() if k.startswith("iv_"))
    assert {f"preempt_{k}" for k in PREEMPT_KEYS} <= set(row)


def test_summary_row_keeps_the_preemption_outcome():
    """Preemption is bounded (max_hold_s): a timeout must reach validation.csv and
    compare.csv instead of being dropped."""
    env = StubTrafficEnv(window_s=WINDOW_S)
    summary = run_baseline(env, "fixed", 3000)
    summary["preemption"] = {"enabled": True, "requests": 2, "served": 1, "timeouts": 1}
    row = summary_row(summary)
    assert (row["preempt_requests"], row["preempt_served"], row["preempt_timeouts"]) == (2, 1, 1)


def test_validator_sums_the_preemption_outcome(tmp_path):
    class Fixed:
        def predict(self, obs, deterministic=True):
            return 0, None

        def save(self, path):
            open(path, "wb").close()

    v = Validator(StubTrafficEnv(window_s=WINDOW_S), [3000, 3001], {"S0": 1.0}, tmp_path)
    row = v(Fixed(), 100)
    assert {f"preempt_{k}" for k in PREEMPT_KEYS} <= set(row)
    val = pd.read_csv(tmp_path / "validation.csv")
    eps = pd.read_csv(tmp_path / "validation_episodes.csv")
    assert "preempt_timeouts" in val and "preempt_timeouts" in eps


@pytest.mark.parametrize("out", ["data/results", "data/results/rl", "data/results/../results/x"])
def test_train_refuses_data_results(tmp_path, monkeypatch, out):
    """`--out data/results` is refused before any folder, baseline or worker output is
    created (cwd is a scratch copy so a regression cannot touch the real data/results)."""
    from traffic_twin.config import ConfigError
    from traffic_twin.rl.train import train

    config = (Path.cwd() / "configs" / "rl.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "data" / "results"
    results.mkdir(parents=True)
    with pytest.raises(ConfigError, match="data/results"):
        train(
            config,
            "dqn",
            7,
            300,
            2,
            Path(out),
            backend=stub_backend(window_s=WINDOW_S),
            vec="dummy",
        )
    assert list(results.rglob("*")) == []


def test_check_out_root_allows_other_folders(tmp_path):
    check_out_root(tmp_path / "rl_runs")
    check_out_root(Path("data") / "rl_runs")


# -- SB3 trainings ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """algo -> (run a, run b): two 300-step stub trainings with the same seed."""
    pytest.importorskip("stable_baselines3")
    root = tmp_path_factory.mktemp("rl_train")
    return {algo: (_train(root, algo, "a"), _train(root, algo, "b")) for algo in ("dqn", "ppo")}


@pytest.mark.parametrize("algo", ["dqn", "ppo"])
def test_train_writes_a_complete_run_folder(trained, algo):
    out = trained[algo][0]
    assert out.name.startswith(f"{algo}_s7_0k_")
    for name in RUN_FILES:
        assert (out / name).is_file(), name
    assert any((out / "ckpt").glob("*.zip"))
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for key in (
        "algo",
        "seed",
        "steps",
        "n_envs",
        "threads",
        "rl_yaml_sha256",
        "obs_version",
        "reward_version",
        "seeds",
        "code_version",
        "dirty",
        "versions",
        "wall_s",
        "eval_s",
        "steps_per_s",
        "best_val_cost",
        "best_model_sha256",
        "best_policy_sha256",
        "final_policy_sha256",
        "baselines",
    ):
        assert key in m, key
    assert m["backend"] == "stub" and m["algo"] == algo and m["n_envs"] == 2
    assert set(m["baselines"]) == set(BASELINES)
    val = pd.read_csv(out / "validation.csv")
    assert list(val["step"]) == sorted(val["step"]) and len(val) >= 3
    assert {"cost", "cost_vs_S0", "cost_vs_S1", "cost_vs_S2", "cost_vs_stay", "best"} <= set(val)
    assert m["best_val_cost"] == pytest.approx(val["cost"].min())
    assert m["best_step"] == int(val.loc[val["cost"].idxmin(), "step"])
    assert val["step"].iloc[-1] == m["steps_done"]  # the final policy is always scored
    base = pd.read_csv(out / "baselines.csv")
    assert len(base) == len(BASELINES) * 2  # 2 selection windows


@pytest.mark.parametrize("algo", ["dqn", "ppo"])
def test_same_seed_same_policy_and_save_load_round_trip(trained, tmp_path, algo):
    from traffic_twin.rl.evaluate import repro_check
    from traffic_twin.rl.train import load_model, policy_sha256

    a, b = trained[algo]
    ma = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    assert ma["final_policy_sha256"] == mb["final_policy_sha256"]
    assert ma["best_policy_sha256"] == mb["best_policy_sha256"]
    assert repro_check(a, b)["same"]
    model = load_model(a / "final_model.zip", algo)  # round trip
    assert policy_sha256(model) == ma["final_policy_sha256"]
    obs = np.full(OBS_DIM, 0.3, np.float32)
    best = load_model(a / "best_model.zip", algo)
    assert policy_sha256(best) == ma["best_policy_sha256"]
    assert model.predict(obs, deterministic=True)[0] in (0, 1)
    if algo == "dqn":
        c = _train(tmp_path, algo, "c", seed=8)  # another seed, other weights
        mc = json.loads((c / "manifest.json").read_text(encoding="utf-8"))
        assert mc["final_policy_sha256"] != ma["final_policy_sha256"]


def test_finished_run_is_not_overwritten(trained):
    a = trained["ppo"][0]
    with pytest.raises(FileExistsError, match="--force"):
        _train(a.parent.parent, "ppo", a.parent.name)


def test_validator_keeps_the_best_checkpoint(tmp_path):
    class FakeModel:
        def __init__(self, action):
            self.action, self.saved = action, []

        def predict(self, obs, deterministic=True):
            return np.asarray(self.action), None

        def save(self, path):
            self.saved.append(path)

    env = StubTrafficEnv(window_s=WINDOW_S)
    base = {"S0": 1000.0}
    v = Validator(env, [3000, 3001], base, tmp_path)
    stay, advance = FakeModel(0), FakeModel(1)
    r1 = v(stay, 10)
    r2 = v(advance, 20)
    worse_first = r1["cost"] > r2["cost"]
    assert r1["best"] and r2["best"] == worse_first
    assert len(stay.saved) == 1 and len(advance.saved) == int(worse_first)
    expected = play_episode(StubTrafficEnv(window_s=WINDOW_S), lambda o: 0, 3000)["totals"]["cost"]
    ep = pd.read_csv(tmp_path / "validation_episodes.csv")
    assert ep.loc[(ep["step"] == 10) & (ep["seed"] == 3000), "cost"].iloc[0] == pytest.approx(
        expected
    )
    val = pd.read_csv(tmp_path / "validation.csv")
    assert val["cost_vs_S0"].iloc[0] == pytest.approx(pct_change(r1["cost"], 1000.0))

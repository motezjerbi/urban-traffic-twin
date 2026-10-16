"""RL training smoke tests on the SUMO-free `StubTrafficEnv` (Part 6, WS5; D60, D61):
the stub honours the TrafficEnv contract, DQN and PPO train for 300 steps with the
validation callback (validation.csv, best_model.zip), the manifest is complete, a saved
model loads back to the same policy, and two runs with the same seed give the same
parameters and the same episode returns.

Part 7 (WS1; D71, D72, D81): a Part 6 config keeps the Part 6 manifest key set (capture
(d)) and run folder; the pooled validator (2 spawned workers) scores exactly as the
in-process one; the D72 pick at the 50 % edge; `lin:` schedules decay to their end
value; exploration_steps -> exploration_fraction; two Part 7 runs with the same seed
give the same pick and final policy."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_twin.config import ConfigError, load_config
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
    PICK_MODEL_FILE,
    PREEMPT_KEYS,
    PROVENANCE_FILE,
    RUN_FILES,
    VAL_DIR,
    LinearDecay,
    Validator,
    check_out_root,
    hyperparameters,
    pct_change,
    play_episode,
    run_name,
    s6_run_pass,
    sb3_kwargs,
    schedule,
    seed_range,
    stub_backend,
    summary_row,
)

WINDOW_S = 300  # short stub episodes: ~40 decisions
# Capture (d) of Phase 0: the manifest keys of a Part 6 run (`rl.train --stub`, rl.yaml)
PART6_MANIFEST_KEYS = frozenset(
    {
        "algo", "argv", "backend", "baseline_s", "baselines", "best_model_sha256",
        "best_policy_sha256", "best_step", "best_val_cost", "best_vs",
        "checkpoint_every_steps", "code_version", "config", "dirty", "eval_every_steps",
        "eval_s", "final_model_sha256", "final_policy_sha256", "final_val_cost",
        "hyperparameters", "model_bytes", "n_envs", "obs_version", "reward_version",
        "rl_yaml_sha256", "run", "s6_run_pass", "seed", "seeds", "started_at", "steps",
        "steps_done", "steps_per_s", "threads", "train_s", "validations", "vec", "versions",
        "wall_s",
    }
)  # fmt: skip
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
    assert set(m) == PART6_MANIFEST_KEYS  # Part 6 format, key for key (D81)
    hp = hyperparameters(load_config("configs/rl.yaml"), algo, SMALL_HP[algo])
    assert m["hyperparameters"] == hp  # no exploration_steps in rl.yaml: unchanged
    assert not (out / PICK_MODEL_FILE).exists() and not (out / VAL_DIR).exists()
    prov = json.loads((out / PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert prov["variant"] is None and prov["pick_step"] is None
    assert set(m["baselines"]) == set(BASELINES)
    val = pd.read_csv(out / "validation.csv")
    assert list(val["step"]) == sorted(val["step"]) and len(val) >= 3
    assert {"cost", "cost_vs_S0", "cost_vs_S1", "cost_vs_S2", "cost_vs_stay", "best"} <= set(val)
    assert not {"late", "pick", "split_cut", "control_s"} & set(val)  # Part 6 columns
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


# -- Part 7: schedules, exploration, pool validator, pick (D71, D72) ---------------------
def test_lin_schedule_decays_from_a_to_b():
    lr = schedule("lin:5.0e-4:5.0e-5")
    assert lr == LinearDecay(5e-4, 5e-5) and str(lr) == "lin:0.0005:5e-05"
    assert lr(1.0) == pytest.approx(5e-4)  # progress_remaining 1: start of training
    assert lr(0.5) == pytest.approx(2.75e-4)
    assert lr(0.0) == pytest.approx(5e-5)  # end of training: b
    assert lr(-0.001) == pytest.approx(5e-5)  # SB3 overshoots past total_timesteps
    assert schedule(3e-4) == 3e-4 and schedule(0.2) == 0.2  # numbers pass through
    for bad in ("lin:1e-3", "exp:1:2", "lin:a:b", True):
        with pytest.raises(ConfigError):
            schedule(bad)
    kw = sb3_kwargs({"learning_rate": "lin:3.0e-4:3.0e-5", "clip_range": 0.2})
    assert kw["learning_rate"] == LinearDecay(3e-4, 3e-5) and kw["clip_range"] == 0.2


def test_exploration_steps_become_a_fraction_of_the_budget(rl7_cfg):
    hp = hyperparameters(rl7_cfg, "dqn", steps=400_000)
    assert hp["exploration_fraction"] == pytest.approx(50_000 / 400_000)
    assert "exploration_steps" not in hp and hp["learning_rate"] == 5e-4  # t0 = rl7 base
    assert hyperparameters(rl7_cfg, "dqn", steps=1_000_000)["exploration_fraction"] == 0.05
    assert hyperparameters(rl7_cfg, "dqn", steps=20_000)["exploration_fraction"] == 1.0
    raw = hyperparameters(rl7_cfg, "dqn")  # without steps: as written
    assert raw["exploration_steps"] == 50_000
    with pytest.raises(ConfigError, match="exploration_steps"):
        sb3_kwargs(raw)
    assert "exploration_fraction" not in hyperparameters(rl7_cfg, "ppo", steps=400_000)


class _Fixed:
    """A stand-in model: always `action`; `save` writes a placeholder file."""

    def __init__(self, action: int) -> None:
        self.action = action

    def predict(self, obs, deterministic=True):
        return np.asarray(self.action), None

    def save(self, path):
        Path(path).write_bytes(f"model {self.action}".encode())


def test_pick_counts_validations_from_the_50_percent_edge(tmp_path):
    """D72: validations at step >= pick_from_step are late; the pick is the lowest late
    cost (a better earlier checkpoint is the best model, never the pick)."""
    env = StubTrafficEnv(window_s=WINDOW_S)
    v = Validator(env, [3000, 3001], {}, tmp_path, keep_dir=tmp_path / "val", pick_from_step=200)
    stay, advance = _Fixed(0), _Fixed(1)
    cost = {a: v(m, s)["cost"] for a, m, s in ((0, stay, 100), (1, advance, 200), (0, stay, 300))}
    rows = pd.read_csv(tmp_path / "validation.csv")
    assert list(rows["late"]) == [False, True, True]  # 200 = the edge, included
    assert v.late_costs == [cost[1], cost[0]]
    assert v.late_mean == pytest.approx((cost[0] + cost[1]) / 2)
    assert v.pick_step == (200 if cost[1] <= cost[0] else 300)
    assert v.best_step == (100 if cost[0] <= cost[1] else 200)
    pick = tmp_path / PICK_MODEL_FILE
    assert pick.read_bytes() == (tmp_path / "val" / f"step_{v.pick_step}.zip").read_bytes()
    kept = sorted(p.name for p in (tmp_path / "val").iterdir())
    assert kept == ["step_100.zip", "step_200.zip", "step_300.zip"]
    late_ped = rows.loc[rows["late"], "pedestrian_s"].mean()
    assert v.late_totals()["pedestrian_s"] == pytest.approx(late_ped)


def test_pooled_validator_equals_in_process(trained, tmp_path):
    """Two spawned workers score a saved model exactly as the in-process validator scores
    the live model (seeded episodes), extended rows included."""
    from traffic_twin.rl.train import load_model

    model = load_model(trained["dqn"][0] / "final_model.zip", "dqn")
    seeds, base = [3000, 3001, 3002], {"S0": 1000.0}
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    env = StubTrafficEnv(window_s=WINDOW_S)
    inproc = Validator(env, seeds, base, tmp_path / "a", extended=True)
    pooled = Validator(
        None,
        seeds,
        base,
        tmp_path / "b",
        workers=2,
        keep_dir=tmp_path / "b" / "val",
        extended=True,
        backend=stub_backend(window_s=WINDOW_S),
        config="configs/rl.yaml",
    )
    try:
        ra, rb = inproc(model, 100), pooled(model, 100)
        rb2 = pooled(model, 200)  # cached-model eviction: a new checkpoint path
    finally:
        pooled.close()
    assert pooled.pool is None
    drop = {"step", "eval_s"}
    assert {k: v for k, v in ra.items() if k not in drop} == {
        k: v for k, v in rb.items() if k not in drop
    }
    assert rb2["cost"] == rb["cost"]
    ea = pd.read_csv(tmp_path / "a" / "validation_episodes.csv")
    eb = pd.read_csv(tmp_path / "b" / "validation_episodes.csv").query("step == 100")
    pd.testing.assert_frame_equal(ea, eb.reset_index(drop=True))
    assert {"split_cut", "split_replaced", "split_cut_foreseeable", "control_s"} <= set(ea)
    per_h = 3600 * ea["split_cut"].sum() / (len(seeds) * WINDOW_S)
    assert ra["split_cut_per_h"] == pytest.approx(per_h)


def _flipped(model):
    """`model` with its Q-head negated in place: another greedy policy (and cost)."""
    import torch

    with torch.no_grad():
        *_, weight, bias = model.policy.q_net.parameters()
        weight.neg_()
        bias.neg_()
    return model


def test_pool_model_cache_reloads_a_rewritten_file(trained, tmp_path, monkeypatch):
    """The worker's one-model cache is keyed by the file content, not the path alone: a
    path rewritten with another model is reloaded (a persistent worker kept scoring the
    first model saved under a reused name)."""
    from traffic_twin.rl import train as train_mod
    from traffic_twin.rl.train import load_model, policy_sha256

    monkeypatch.setattr(train_mod, "_POOL", {})
    a = load_model(trained["dqn"][0] / "final_model.zip", "dqn")
    b = _flipped(load_model(trained["dqn"][0] / "final_model.zip", "dqn"))
    assert policy_sha256(a) != policy_sha256(b)
    path = tmp_path / "same_name.zip"
    a.save(str(path))
    assert policy_sha256(train_mod._pool_model(str(path), "dqn")) == policy_sha256(a)
    assert train_mod._pool_model(str(path), "dqn") is train_mod._pool_model(str(path), "dqn")
    b.save(str(path))
    assert policy_sha256(train_mod._pool_model(str(path), "dqn")) == policy_sha256(b)


def test_pooled_validator_without_keep_dir_scores_each_model(trained, tmp_path):
    """A persistent pool without `keep_dir` (a Part 6 config with --eval-workers 2) scores
    every validation's model like the in-process validator: two different models, two
    validations, the same costs; the scratch files are removed."""
    from traffic_twin.rl.train import SCRATCH_MODEL_PREFIX, load_model

    final = trained["dqn"][0] / "final_model.zip"
    models = [load_model(final, "dqn"), _flipped(load_model(final, "dqn"))]
    seeds, base = [3000, 3001], {"S0": 1000.0}
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    inproc = Validator(StubTrafficEnv(window_s=WINDOW_S), seeds, base, tmp_path / "a")
    pooled = Validator(
        None,
        seeds,
        base,
        tmp_path / "b",
        workers=2,
        backend=stub_backend(window_s=WINDOW_S),
        config="configs/rl.yaml",
    )
    try:
        for step, model in zip((100, 200), models, strict=True):
            inproc(model, step)
            pooled(model, step)
    finally:
        pooled.close()
    ca, cb = (pd.read_csv(tmp_path / d / "validation.csv")["cost"].tolist() for d in "ab")
    assert ca[0] != ca[1]  # the two models really differ
    assert cb == ca
    assert inproc.best_step == pooled.best_step
    assert all((tmp_path / d / "best_model.zip").is_file() for d in "ab")
    assert not list((tmp_path / "b").glob(f"{SCRATCH_MODEL_PREFIX}*")), "scratch file left"


def test_summary_row_extended_adds_the_d77_split():
    env = StubTrafficEnv(window_s=WINDOW_S)
    summary = run_baseline(env, "stay", 3000)
    assert summary_row(summary) == summary_row(summary, extended=False)
    row = summary_row(summary, extended=True)
    split = summary["iv_split"]
    assert row["split_cut"] + row["split_replaced"] == row["interventions"]
    assert row["split_cut_foreseeable"] <= row["split_cut"] == split["cut"]
    assert row["control_s"] == WINDOW_S
    del summary["iv_split"]
    assert math.isnan(summary_row(summary, extended=True)["split_cut"])


def _train_rl7(cfg_path, out, seed=9, steps=600):
    from traffic_twin.rl.train import train

    return train(
        cfg_path,
        None,
        seed,
        steps,
        None,
        out,
        backend=stub_backend(window_s=WINDOW_S),
        vec="dummy",
        hp_overrides={**SMALL_HP["dqn"], "learning_starts": 100},
        validation_seeds=[5100, 5101],
    )


@pytest.fixture
def rl7_runs(stub_rl7_cfg, tmp_path):
    """Two identical Part 7 t1 stub runs (seed 9, 600 steps) on the conftest stub rl7
    tree (development seeds, 2 envs, validation every 200 steps)."""
    pytest.importorskip("stable_baselines3")
    t1 = stub_rl7_cfg().parent / "rl7" / "t1.yaml"
    return _train_rl7(t1, tmp_path / "a"), _train_rl7(t1, tmp_path / "b")


def test_rl7_run_folder_pick_and_determinism(rl7_runs):
    from traffic_twin.rl.evaluate import repro_check
    from traffic_twin.rl.train import load_model, policy_sha256

    a, b = rl7_runs
    assert a.name.startswith("dqn_t1_s9_0k_")
    for name in (*RUN_FILES, PROVENANCE_FILE, PICK_MODEL_FILE):
        assert (a / name).is_file(), name
    ma = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    pa = json.loads((a / PROVENANCE_FILE).read_text(encoding="utf-8"))
    pb = json.loads((b / PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert PART6_MANIFEST_KEYS < set(ma) and set(pa) <= set(ma)  # carries the provenance
    assert ma["hyperparameters"]["learning_rate"] == "lin:5.0e-4:5.0e-5"
    assert ma["hyperparameters"]["exploration_fraction"] == pytest.approx(300 / 600)
    val = pd.read_csv(a / "validation.csv")
    late = val[val["step"] >= 300]
    assert list(val["late"]) == list(val["step"] >= 300) and pa["pick_from_step"] == 300
    assert pa["pick_val_cost"] == pytest.approx(late["cost"].min())
    assert pa["pick_step"] == int(late.loc[late["cost"].idxmin(), "step"])
    assert pa["late_mean"] == pytest.approx(late["cost"].mean()) and pa["late_n"] == len(late)
    assert val.loc[val["pick"], "step"].tolist() == [pa["pick_step"]]
    kept = sorted(int(p.stem.split("_")[1]) for p in (a / VAL_DIR).glob("step_*.zip"))
    assert kept == list(val["step"])
    pick = load_model(a / PICK_MODEL_FILE, "dqn")
    assert policy_sha256(pick) == pa["pick_policy_sha256"]
    # same seed, same run: pick, final policy, validation and monitor
    assert repro_check(a, b)["same"]
    for key in ("pick_step", "pick_policy_sha256", "late_mean", "run_config_sha"):
        assert pa[key] == pb[key], key

"""Part 7 training identity and provenance (WS1; D69, D71, D80, D81), SUMO-free: the run
identity `run_config_sha` follows the merged config (a parent edit changes it; `changes`,
`preregistered`, `protocol` and `outputs` do not), run names carry the variant,
`provenance.json` is strict JSON with the D81 keys, a dirty tree is refused under
`train.require_clean` unless `--allow-dirty` (recorded), the V1 drop and the P1 PPO
schedules train and load on the stub. Memory (Phase 2 gate: 13 processes per job hit the
commit limit): spawned workers get one OpenBLAS thread, the light env processes import
neither SB3 nor torch and step exactly as SB3's, a transient validation pool gives the
in-process rows, prediction loads carry no replay buffer."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from traffic_twin.config import ConfigError, load_config, text_sha256
from traffic_twin.rl.train import (
    PROVENANCE_FILE,
    build_parser,
    parse_seeds,
    resolve_algo,
    run_config_sha,
    run_name,
    stub_backend,
)

WINDOW_S = 300


class _ProbeFactory:
    """Picklable env thunk: a stub env that reports whether its process loaded torch."""

    def __init__(self, worker: int) -> None:
        self.worker = worker

    def __call__(self):
        from traffic_twin.rl.stub import StubTrafficEnv

        env = StubTrafficEnv(window_s=WINDOW_S, worker=self.worker, n_workers=2, seeds=[5000, 5001])
        env.torch_loaded = lambda: "torch" in sys.modules
        return env


SMALL_DQN = {"learning_starts": 100, "buffer_size": 1000, "batch_size": 32, "net_arch": [16]}
SMALL_PPO = {"n_steps": 64, "batch_size": 32, "n_epochs": 2, "net_arch": [16]}
D81_KEYS = {
    "run_config_sha", "sources", "variant", "reward", "observation", "decision_interval_s",
    "base_config_hash", "base_demand_hash", "calibration_sha256", "incident_sites_sha256",
    "net_sha256", "machine", "learner_threads", "worker_threads", "schedules",
    "exploration_steps", "exploration_fraction", "update_to_data", "selection_seeds",
    "selection_metric", "pick_step", "pick_val_cost", "late_mean", "allow_dirty",
}  # fmt: skip


def _train(config, out, *, algo=None, steps=400, hp=None, **kw):
    from traffic_twin.rl.train import train

    return train(
        config,
        algo,
        3,
        steps,
        None,
        out,
        backend=stub_backend(window_s=WINDOW_S),
        vec="dummy",
        hp_overrides=hp if hp is not None else SMALL_DQN,
        validation_seeds=[5100],
        **kw,
    )


# -- identity (D69) --------------------------------------------------------------------
def test_run_config_sha_follows_parent_files_and_ignores_bookkeeping(stub_rl7_cfg):
    rl7 = stub_rl7_cfg()
    t1 = rl7.parent / "rl7" / "t1.yaml"
    text, sha = text_sha256(t1), run_config_sha(load_config(t1))
    # bookkeeping keys of the parent do not change the identity
    stub_rl7_cfg(
        changes=[{"date": "2026-10-17", "what": "x"}],
        preregistered="2026-10-18",
        protocol={"slots": 6},
        outputs={"runs": "elsewhere"},
    )
    assert run_config_sha(load_config(t1)) == sha
    # a parent edit that changes the merged config changes it; the file text does not
    stub_rl7_cfg(reward={"weights": {"bus": 30.0}})
    assert text_sha256(t1) == text and run_config_sha(load_config(t1)) != sha
    cfg = load_config(t1)
    assert run_config_sha({**cfg, "changes": ["y"]}) == run_config_sha(cfg)
    assert run_config_sha({**cfg, "seeds": {**cfg["seeds"], "train": [5000, 5001]}}) != (
        run_config_sha(cfg)
    )


def test_repo_run_configs_have_distinct_identities():
    shas = {
        p.stem: run_config_sha(load_config(p)) for p in sorted(Path("configs/rl7").glob("*.yaml"))
    }
    assert len(set(shas.values())) == len(shas) == 14


def test_run_names_and_algo_resolution(stub_rl7_cfg):
    assert run_name("dqn", 0, 400_000, "abcdef0123", "t1") == "dqn_t1_s0_400k_abcdef01"
    assert run_name("dqn", 0, 200_000, "c0b7dd7b6ca4") == "dqn_s0_200k_c0b7dd7b"  # Part 6
    rl7 = stub_rl7_cfg()
    p1 = load_config(rl7.parent / "rl7" / "p1.yaml")
    assert resolve_algo(p1) == "ppo" and resolve_algo(p1, "ppo") == "ppo"
    with pytest.raises(ConfigError, match="declares algo"):
        resolve_algo(p1, "dqn")
    assert resolve_algo(load_config(rl7.parent / "rl7" / "t1.yaml")) == "dqn"
    assert resolve_algo(load_config("configs/rl.yaml"), "ppo") == "ppo"


def test_cli_options():
    a = build_parser().parse_args(["--seed", "1"])
    assert a.algo is None and not a.allow_dirty  # --algo: the config's key, else dqn
    a = build_parser().parse_args(
        ["--seed", "1", "--allow-dirty", "--validation-seeds", "5100-5102", "5105"]
    )
    assert a.allow_dirty and parse_seeds(a.validation_seeds) == [5100, 5101, 5102, 5105]
    for bad in (["51o0"], ["5105-5100"]):
        with pytest.raises(ConfigError):
            parse_seeds(bad)


# -- worker environment (gate: commit limit) --------------------------------------------
def test_child_env_sets_absent_variables_for_spawned_processes_only(monkeypatch):
    from traffic_twin.rl.train import CHILD_ENV, EpisodePool, child_env, child_env_values

    assert CHILD_ENV == {"OPENBLAS_NUM_THREADS": "1"}
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with child_env() as seen:
        assert seen == {"OPENBLAS_NUM_THREADS": "1"} == child_env_values()
    assert "OPENBLAS_NUM_THREADS" not in os.environ  # restored
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")  # a user setting is kept
    with child_env() as seen:
        assert seen == {"OPENBLAS_NUM_THREADS": "3"}
    assert os.environ["OPENBLAS_NUM_THREADS"] == "3"
    monkeypatch.delenv("OPENBLAS_NUM_THREADS")
    # the pool's workers are spawned at the first map, inside child_env
    with EpisodePool(stub_backend(window_s=WINDOW_S), "configs/rl.yaml", 1) as pool:
        (row,) = pool.map([("baseline", "stay", 5100)])
        assert row["cost"] > 0
        assert pool._ex.submit(os.getenv, "OPENBLAS_NUM_THREADS").result() == "1"
    assert "OPENBLAS_NUM_THREADS" not in os.environ


def test_transient_eval_pool_gives_the_in_process_results(stub_rl7_cfg, tmp_path):
    # --eval-pool transient: 2 workers started ahead of each validation and closed after
    # it (less memory per concurrent job); seeded episodes -> the in-process rows and pick
    pytest.importorskip("stable_baselines3")
    import pandas as pd

    from traffic_twin.rl.train import PICK_MODEL_FILE

    t1 = stub_rl7_cfg().parent / "rl7" / "t1.yaml"
    a = _train(t1, tmp_path / "inproc")
    b = _train(
        t1, tmp_path / "transient", eval_workers=2, eval_pool="transient", baseline_workers=1
    )
    pa, pb = (json.loads((r / PROVENANCE_FILE).read_text(encoding="utf-8")) for r in (a, b))
    assert (pa["eval_pool"], pb["eval_pool"], pb["eval_workers"]) == ("in-process", "transient", 2)
    for f in ("baselines.csv", "validation.csv", "validation_episodes.csv"):
        da, db = (pd.read_csv(r / f).drop(columns=["eval_s"], errors="ignore") for r in (a, b))
        pd.testing.assert_frame_equal(da, db)
    assert len(pd.read_csv(b / "validation.csv")) == 2  # two pools, one per validation
    for key in ("pick_step", "pick_policy_sha256", "late_mean"):
        assert pa[key] == pb[key], key
    assert (b / PICK_MODEL_FILE).is_file() and not list(b.glob("_validation_model*"))
    with pytest.raises(ConfigError, match="eval_pool"):
        _train(t1, tmp_path / "bad", eval_workers=2, eval_pool="forever")


def test_light_env_processes_step_like_sb3_without_torch():
    pytest.importorskip("stable_baselines3")
    import numpy as np
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from traffic_twin.rl.train import light_subproc_vec_env

    fns = [_ProbeFactory(0), _ProbeFactory(1)]
    light, ref = light_subproc_vec_env(fns), SubprocVecEnv(fns, start_method="spawn")
    try:
        assert light.env_method("torch_loaded") == [False, False]  # ~150 MB less each
        assert ref.env_method("torch_loaded") == [True, True]
        assert light.env_is_wrapped(Monitor) == ref.env_is_wrapped(Monitor) == [False, False]
        assert light.get_attr("window_s") == [WINDOW_S, WINDOW_S]
        for v in (light, ref):
            v.seed(7)
        oa, ob = light.reset(), ref.reset()
        ends = 0
        np.testing.assert_array_equal(oa, ob)
        for t in range(2 * WINDOW_S // 10):  # past an episode end (auto-reset)
            act = np.array([t % 3 == 0, t % 2 == 0], dtype=np.int64)
            (oa, ra, da, ia), (ob, rb, db, ib) = light.step(act), ref.step(act)
            np.testing.assert_array_equal(oa, ob)
            np.testing.assert_array_equal(ra, rb)
            np.testing.assert_array_equal(da, db)
            ends += int(da.sum())
            assert [sorted(i) for i in ia] == [sorted(i) for i in ib]
        assert ends >= 2  # both envs ended an episode and reset
        assert light.env_method("torch_loaded") == [False, False]
    finally:
        light.close()
        ref.close()


def test_prediction_loads_carry_no_replay_buffer(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import DQN

    from traffic_twin.rl.stub import StubTrafficEnv
    from traffic_twin.rl.train import load_model, policy_sha256

    model = DQN("MlpPolicy", StubTrafficEnv(window_s=WINDOW_S), buffer_size=5000, device="cpu")
    model.save(str(tmp_path / "m.zip"))
    loaded = load_model(tmp_path / "m.zip", "dqn")
    assert loaded.buffer_size == 1 and loaded.replay_buffer.buffer_size == 1
    assert policy_sha256(loaded) == policy_sha256(model)


# -- clean tree (D80) ------------------------------------------------------------------
@pytest.mark.parametrize("version", ["685679d-dirty", "unknown"])
def test_dirty_tree_is_refused_unless_allowed(stub_rl7_cfg, tmp_path, monkeypatch, version):
    monkeypatch.setattr("traffic_twin.rl.train.code_version", lambda: version)
    rl7 = stub_rl7_cfg(train={"require_clean": True})
    t1 = rl7.parent / "rl7" / "t1.yaml"
    out = tmp_path / "runs"
    with pytest.raises(ConfigError, match="allow-dirty"):
        _train(t1, out)
    assert not out.exists()  # refused before any folder or baseline
    pytest.importorskip("stable_baselines3")
    run = _train(t1, out, allow_dirty=True)
    prov = json.loads((run / PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert prov["allow_dirty"] is True and prov["require_clean"] is True
    assert prov["code_version"] == version


def test_part7_config_needs_steps(stub_rl7_cfg, tmp_path):
    t1 = stub_rl7_cfg().parent / "rl7" / "t1.yaml"
    with pytest.raises(ConfigError, match="--steps"):
        _train(t1, tmp_path / "runs", steps=None)


# -- provenance (D81) and the variants on the stub -------------------------------------
@pytest.fixture
def v1_run(stub_rl7_cfg, tmp_path):
    pytest.importorskip("stable_baselines3")
    return _train(stub_rl7_cfg().parent / "rl7" / "v1.yaml", tmp_path / "runs")


def test_provenance_is_strict_json_with_the_d81_keys(v1_run):
    text = (v1_run / PROVENANCE_FILE).read_text(encoding="utf-8")
    prov = json.loads(text)
    json.dumps(prov, allow_nan=False)  # no NaN / inf anywhere
    assert "NaN" not in text and "Infinity" not in text
    assert D81_KEYS <= set(prov)
    assert prov["variant"] == "v1" and v1_run.name.startswith("dqn_v1_s3_0k_")
    assert v1_run.name.endswith(prov["run_config_sha"][:8])
    assert prov["observation"] == {"version": "rl-obs-1", "drop": ["pedestrian"], "n_features": 62}
    assert prov["base_demand_hash"].startswith("83a4f220fd20")  # S0 demand (D86)
    assert len(prov["calibration_sha256"]) == 3 and all(prov["calibration_sha256"].values())
    assert prov["schedules"] == {"learning_rate": "lin:5.0e-4:5.0e-5"}
    assert prov["exploration_steps"] == 300 and prov["exploration_fraction"] == 0.75
    assert prov["update_to_data"] == pytest.approx(2 / (2 * 1))  # gradient_steps / (envs x freq)
    assert prov["selection_seeds"] == [5100] and prov["learner_threads"] == 1
    assert set(prov["machine"]["child_env"]) == {"OPENBLAS_NUM_THREADS"}
    assert [Path(s["path"]).name for s in prov["sources"]] == [
        "rl.yaml",
        "rl7.yaml",
        "rl7.yaml",
        "t1.yaml",
        "v1.yaml",
    ]
    m = json.loads((v1_run / "manifest.json").read_text(encoding="utf-8"))
    assert {k: m[k] for k in prov} == prov  # Part 7 manifests carry the provenance keys
    assert yaml.safe_load(Path(prov["config"]).read_text(encoding="utf-8"))["variant"] == "v1"


def test_v2_interval_and_p1_ppo_schedules_train_and_load(stub_rl7_cfg, tmp_path):
    pytest.importorskip("stable_baselines3")
    from traffic_twin.rl.train import LinearDecay, load_model, policy_sha256

    root = stub_rl7_cfg().parent / "rl7"
    v2 = _train(root / "v2.yaml", tmp_path / "v2")
    assert json.loads((v2 / PROVENANCE_FILE).read_text("utf-8"))["decision_interval_s"] == 10
    p1 = _train(root / "p1.yaml", tmp_path / "p1", hp=SMALL_PPO, steps=256)
    m = json.loads((p1 / "manifest.json").read_text(encoding="utf-8"))
    assert m["algo"] == "ppo" and p1.name.startswith("ppo_p1_s3_")
    assert m["hyperparameters"]["learning_rate"] == "lin:3.0e-4:3.0e-5"
    assert m["update_to_data"] is None and m["exploration_fraction"] is None
    from stable_baselines3 import PPO

    full = PPO.load(str(p1 / "final_model.zip"), device="cpu")  # the saved schedule
    assert full.learning_rate == LinearDecay(3e-4, 3e-5)
    fast = load_model(p1 / "pick_model.zip", "ppo")  # schedules replaced by constants
    assert policy_sha256(fast) == m["pick_policy_sha256"]

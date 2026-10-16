"""The committed S3 model (Part 7, D86): configs/pitie_rl_s3.yaml pins
models/rl/<model_version>/model.zip by sha256; the policy parameters equal the
manifest's; the Q-values (PPO: logits) and actions on the 29 golden observations of
training seed 5000 are those recorded by `rl.study golden` (platform independent); and
two 900 s `run_scenario` runs on seed 5000 are identical, with the golden phase hash on
Windows (where it was recorded; SUMO results differ across platforms, as for the grid
regression). Skipped until Part 7 Phase 3 writes the config and the golden."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from traffic_twin.config import load_config, load_scenario
from traffic_twin.paths import MODELS_RL_DIR, PITIE_RL_S3_CONFIG
from traffic_twin.rl import study

pytestmark = pytest.mark.skipif(
    not PITIE_RL_S3_CONFIG.is_file(),
    reason="configs/pitie_rl_s3.yaml is written in Part 7 Phase 3 (D86)",
)


@pytest.fixture(scope="module")
def s3() -> dict:
    pytest.importorskip("stable_baselines3")
    return study.s3_model(PITIE_RL_S3_CONFIG)  # checks the rl.model_sha256 pin


@pytest.fixture(scope="module")
def model(s3):
    from traffic_twin.rl.train import load_model

    return load_model(s3["path"], s3["algo"])


@pytest.fixture(scope="module")
def gold() -> dict:
    if not study.GOLDEN_FILE.is_file():
        pytest.skip(f"{study.GOLDEN_FILE.as_posix()} not written (`rl.study golden`)")
    return json.loads(study.GOLDEN_FILE.read_text(encoding="utf-8"))


def test_zip_sha_equals_the_pin(s3):
    from traffic_twin.rl.train import file_sha256

    cfg = load_scenario(PITIE_RL_S3_CONFIG)
    assert cfg["scenario"] == "pitie_rl_s3" and cfg["controller"] == "rl"
    sha = file_sha256(s3["path"])
    assert cfg["rl"]["model_sha256"] == sha == s3["manifest"]["model_sha256"]
    assert s3["path"].parent.parent.resolve() == MODELS_RL_DIR.resolve()
    assert s3["path"].parent.name == s3["manifest"]["model_version"]
    assert s3["path"].stat().st_size < 1_000_000


def test_policy_sha_equals_the_manifest(s3, model):
    from traffic_twin.rl.train import policy_sha256

    psha = policy_sha256(model)
    assert psha == s3["manifest"]["policy_sha256"]
    assert s3["manifest"]["model_version"].endswith(psha[:8])


def test_manifest_matches_the_run_config(s3):
    rl_cfg = load_config(s3["config"])
    m = s3["manifest"]
    assert m["obs_version"] == rl_cfg["observation"]["version"]
    assert m["reward_version"] == rl_cfg["reward"]["version"]
    assert list(m.get("obs_drop") or []) == list(rl_cfg["observation"].get("drop") or [])


def test_golden_values_and_actions(s3, model, gold):
    assert gold["model_sha256"] == s3["model_sha256"]
    assert gold["seed"] == study.GOLDEN_SEED and len(gold["obs"]) == study.GOLDEN_N_OBS
    obs = np.asarray(gold["obs"], dtype=np.float32)
    values = study.policy_values(model, s3["algo"], obs)
    np.testing.assert_allclose(values, np.asarray(gold["q"]), rtol=0, atol=1e-4)
    actions = [int(np.asarray(model.predict(o, deterministic=True)[0]).item()) for o in obs]
    assert actions == gold["actions"]


@pytest.mark.sumo
def test_short_runs_are_identical(tmp_path, s3, gold):
    phases = []
    for arm in ("a", "b"):
        shown, meta = study.scenario_phases(
            load_scenario(PITIE_RL_S3_CONFIG), gold["seed"], gold["duration_s"], tmp_path / arm
        )
        control = meta["control"]
        assert control["model_sha256"] == s3["model_sha256"]
        assert 0 < control["advances"] < control["decisions"]
        assert meta["safety"]["danger"] == 0
        phases.append(shown)
    assert phases[0] == phases[1] and len(phases[0]) == gold["duration_s"]
    if sys.platform == "win32" and gold["episode"]["platform"] == "win32":
        assert study.phases_sha256(phases[0]) == gold["episode"]["phases_sha256"]

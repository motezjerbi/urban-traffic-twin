import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from traffic_twin.config import ConfigError
from traffic_twin.control.actuated import ActuatedController
from traffic_twin.control.base import Controller, Observation
from traffic_twin.control.fixed import FixedTimeController
from traffic_twin.control.registry import available, make_controller, register
from traffic_twin.control.rules import RuleBasedController


def test_fixed_from_scenario_config(base_scenario):
    ctrl = make_controller(base_scenario)
    assert isinstance(ctrl, FixedTimeController) and ctrl.name == "fixed"
    assert ctrl.green_s == [35, 10, 22, 7]
    assert "fixed" in available()


def test_actuated_is_registered(pitie_scenario):
    assert {"actuated", "fixed"} <= set(available())
    ctrl = make_controller({**pitie_scenario, "controller": "actuated"})
    assert isinstance(ctrl, ActuatedController) and ctrl.name == "actuated"


def test_rules_is_registered(rules_scenario):
    """S2 (Part 4, D19): the spec's annex B name `rules`."""
    assert {"actuated", "fixed", "rules"} <= set(available())
    ctrl = make_controller(rules_scenario)
    assert isinstance(ctrl, RuleBasedController) and ctrl.name == "rules"


def test_unknown_controller_lists_available(base_scenario):
    with pytest.raises(ConfigError, match=r"available: .*actuated.*fixed.*rules"):
        make_controller({**base_scenario, "controller": "nope"})


def test_missing_controller_key():
    with pytest.raises(ConfigError, match="controller"):
        make_controller({})


def test_fixed_needs_a_green_time_per_stage(scenario_cfg):
    del scenario_cfg["fixed"]["green_s"]["NS"]
    with pytest.raises(ConfigError, match="NS"):
        make_controller(scenario_cfg)
    del scenario_cfg["fixed"]
    with pytest.raises(ConfigError, match=r"fixed\.green_s"):
        make_controller(scenario_cfg)


def test_register_new_controller():
    @register("always_first_test")
    class AlwaysFirst(Controller):
        def reset(self, seed: int) -> None:
            pass

        def decide(self, obs: Observation) -> int:
            return 0

    assert isinstance(make_controller({"controller": "always_first_test"}), AlwaysFirst)
    with pytest.raises(ValueError):
        register("always_first_test")(FixedTimeController)


# -- the RL controller (Part 6, D62) --------------------------------------------------
def test_rl_is_registered():
    assert {"actuated", "fixed", "rl", "rules"} <= set(available())


def test_importing_the_controllers_imports_no_heavy_package():
    """The registry imports every controller module on first lookup: `control.rl` must
    not pull torch, stable-baselines3 or gymnasium into every scenario run."""
    code = (
        "import sys\n"
        "from traffic_twin.control.registry import available\n"
        "import traffic_twin.control.rl\n"
        "assert 'rl' in available()\n"
        "heavy = ('torch', 'stable_baselines3', 'gymnasium')\n"
        "print(sorted(m for m in heavy if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert out.stdout.strip().splitlines()[-1] == "[]"


def rl_model_files(tmp_path: Path, **manifest) -> Path:
    """A placeholder model zip with a manifest next to it (never loaded by these tests:
    every check below fails before stable-baselines3 is imported)."""
    model = tmp_path / "selected" / "model.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"not a model")
    base = {"algo": "dqn", "obs_version": "rl-obs-1", "reward_version": "rl-reward-1"}
    (model.parent / "manifest.json").write_text(json.dumps({**base, **manifest}), "utf-8")
    return model


def test_rl_without_model_or_block_is_a_config_error(rl_cfg, tmp_path):
    rl_cfg["rl"]["model"] = str(tmp_path / "missing.zip")
    with pytest.raises(ConfigError, match="not found"):
        make_controller(rl_cfg)
    del rl_cfg["rl"]
    with pytest.raises(ConfigError, match="rl.model"):
        make_controller(rl_cfg)


def test_rl_model_without_manifest_is_a_config_error(rl_cfg, tmp_path):
    model = rl_model_files(tmp_path)
    (model.parent / "manifest.json").unlink()
    rl_cfg["rl"]["model"] = str(model)
    with pytest.raises(ConfigError, match="manifest"):
        make_controller(rl_cfg)


@pytest.mark.parametrize(
    ("manifest", "match"),
    [
        ({"obs_version": "rl-obs-0"}, "obs_version"),
        ({"reward_version": "rl-reward-0"}, "reward_version"),
        ({"algo": "a2c"}, "algo"),
        ({"model_sha256": "0" * 64}, "model_sha256"),
    ],
)
def test_rl_manifest_must_match(rl_cfg, tmp_path, manifest, match):
    rl_cfg["rl"]["model"] = str(rl_model_files(tmp_path, **manifest))
    with pytest.raises(ConfigError, match=match):
        make_controller(rl_cfg)


def test_rl_config_versions_must_match_the_code(rl_cfg, rl_train_scenario, tmp_path):
    bad = copy.deepcopy(rl_train_scenario)
    bad["observation"]["version"] = "rl-obs-0"
    path = tmp_path / "rl.yaml"
    path.write_text(yaml.safe_dump(bad), "utf-8")
    rl_cfg["rl"] = {"model": str(rl_model_files(tmp_path)), "config": str(path)}
    with pytest.raises(ConfigError, match="observation.version"):
        make_controller(rl_cfg)

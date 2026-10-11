import pytest

from traffic_twin.config import ConfigError
from traffic_twin.control.base import Controller, Observation
from traffic_twin.control.fixed import FixedTimeController
from traffic_twin.control.registry import available, make_controller, register


def test_fixed_from_scenario_config(base_scenario):
    ctrl = make_controller(base_scenario)
    assert isinstance(ctrl, FixedTimeController) and ctrl.name == "fixed"
    assert ctrl.green_s == [35, 10, 22, 7]
    assert "fixed" in available()


def test_unknown_controller_lists_available(base_scenario):
    with pytest.raises(ConfigError, match=r"available: .*fixed"):
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

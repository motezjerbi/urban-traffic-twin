import pytest

from traffic_twin.config import (
    DATA_KEYS,
    INTERSECTIONS_KEYS,
    ConfigError,
    get,
    load_config,
    load_scenario,
    stage_names,
)
from traffic_twin.paths import DATA_CONFIG, INTERSECTIONS_CONFIG, SCENARIO_CONFIG


def test_repo_configs_have_required_keys():
    assert stage_names(load_scenario(SCENARIO_CONFIG)) == ["EW", "EW_LEFT", "NS", "NS_LEFT"]
    load_config(DATA_CONFIG, DATA_KEYS)
    load_config(INTERSECTIONS_CONFIG, INTERSECTIONS_KEYS)


def test_missing_keys_are_listed(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("scenario: s\nnetwork:\n  tls_id: C\n")
    with pytest.raises(ConfigError, match=r"network\.net_file") as exc:
        load_scenario(p)
    assert "network.tls_id" not in str(exc.value)


def test_missing_file_and_bad_top_level(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    (tmp_path / "list.yaml").write_text("- a\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(tmp_path / "list.yaml")


def test_get_dotted():
    cfg = {"a": {"b": 1}}
    assert get(cfg, "a.b") == 1
    assert get(cfg, "a.c", None) is None
    with pytest.raises(ConfigError, match=r"'a\.c'"):
        get(cfg, "a.c")


def test_empty_values_count_as_missing(write_file):
    p = write_file("c.yaml", "a:\n  b:\n  c: 0\n  d: ''\n")
    with pytest.raises(ConfigError, match=r"missing keys a\.b$"):
        load_config(p, ["a.b", "a.c", "a.d"])  # 0 and '' are values, only None is missing


def test_all_missing_keys_reported_together(write_file):
    p = write_file("c.yaml", "x: 1\n")
    with pytest.raises(ConfigError) as exc:
        load_config(p, ["a", "b.c", "x"])
    assert "a, b.c" in str(exc.value)
    assert str(p) in str(exc.value)


def test_empty_file_is_rejected(write_file):
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write_file("empty.yaml", ""))


def test_get_through_a_non_mapping():
    cfg = {"a": [1, 2], "b": {"c": None}}
    with pytest.raises(ConfigError, match=r"'a\.x'"):
        get(cfg, "a.x")
    assert get(cfg, "a.x", "dflt") == "dflt"
    assert get(cfg, "b.c", "dflt") is None  # present but None: no default
    assert get(cfg, "b") == {"c": None}


def test_config_error_is_a_value_error():
    assert issubclass(ConfigError, ValueError)


def test_stage_names_need_stages():
    with pytest.raises(ConfigError, match=r"network\.stages"):
        stage_names({"network": {}})


def test_scenario_round_trip_through_yaml(base_scenario, write_file):
    import yaml

    p = write_file("s.yaml", yaml.safe_dump(base_scenario))
    assert load_scenario(p) == base_scenario

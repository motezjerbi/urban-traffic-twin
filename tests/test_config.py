import copy
import re
from pathlib import Path

import pytest
import yaml

from traffic_twin.config import (
    DATA_KEYS,
    INTERSECTIONS_KEYS,
    ConfigError,
    approach_in_edges,
    approach_out_edges,
    config_hash,
    config_sources,
    demand_hash,
    get,
    load_config,
    load_scenario,
    loaded_sources,
    merge,
    stage_names,
)
from traffic_twin.paths import (
    DATA_CONFIG,
    INTERSECTIONS_CONFIG,
    PITIE_ACTUATED_CONFIG,
    PITIE_FIXED_CONFIG,
    SCENARIO_CONFIG,
)

PITIE_STAGES = ["HOP", "S_LEFT", "STM", "BUS_91", "BUS_57"]


def test_repo_configs_have_required_keys():
    assert stage_names(load_scenario(SCENARIO_CONFIG)) == ["EW", "EW_LEFT", "NS", "NS_LEFT"]
    load_config(DATA_CONFIG, DATA_KEYS)
    load_config(INTERSECTIONS_CONFIG, INTERSECTIONS_KEYS)
    for path in (PITIE_FIXED_CONFIG, PITIE_ACTUATED_CONFIG):
        assert stage_names(load_scenario(path)) == PITIE_STAGES


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
    p = write_file("s.yaml", yaml.safe_dump(base_scenario))
    assert load_scenario(p) == base_scenario


# -- extends ---------------------------------------------------------------------------
def _yaml(write_file, name: str, data) -> Path:
    return write_file(name, yaml.safe_dump(data, sort_keys=False))


def test_extends_deep_merges_mappings(write_file):
    _yaml(write_file, "base.yaml", {"a": {"x": 1, "y": {"p": 1, "q": 2}}, "b": 1})
    top = _yaml(write_file, "top.yaml", {"extends": "base.yaml", "a": {"y": {"q": 3}}, "c": 4})
    assert load_config(top) == {"a": {"x": 1, "y": {"p": 1, "q": 3}}, "b": 1, "c": 4}


def test_extends_replaces_lists_and_scalars(write_file):
    _yaml(write_file, "base.yaml", {"l": [1, 2, 3], "s": "base", "m": {"k": 1}})
    top = _yaml(write_file, "top.yaml", {"extends": "base.yaml", "l": [9], "s": "top", "m": 5})
    assert load_config(top) == {"l": [9], "s": "top", "m": 5}


def test_extends_bases_merge_in_order_then_the_file(write_file):
    _yaml(write_file, "one.yaml", {"v": 1, "only_one": True, "d": {"a": 1}})
    _yaml(write_file, "two.yaml", {"v": 2, "d": {"b": 2}})
    top = _yaml(write_file, "top.yaml", {"extends": ["one.yaml", "two.yaml"], "d": {"c": 3}})
    cfg = load_config(top)
    assert cfg == {"v": 2, "only_one": True, "d": {"a": 1, "b": 2, "c": 3}}
    assert "extends" not in cfg


def test_extends_paths_are_relative_to_the_including_file(write_file, tmp_path, monkeypatch):
    _yaml(write_file, "configs/frag/net.yaml", {"net": "n"})
    _yaml(write_file, "configs/frag/more.yaml", {"extends": "../shared.yaml", "more": 1})
    _yaml(write_file, "configs/shared.yaml", {"shared": 2})
    top = _yaml(write_file, "configs/top.yaml", {"extends": ["frag/net.yaml", "frag/more.yaml"]})
    monkeypatch.chdir(tmp_path / "configs" / "frag")  # not the folder of top.yaml
    assert load_config(top) == {"net": "n", "shared": 2, "more": 1}


def test_config_sources_lists_files_in_merge_order(write_file, tmp_path, monkeypatch):
    _yaml(write_file, "c/a.yaml", {"a": 1})
    _yaml(write_file, "c/b.yaml", {"extends": "a.yaml", "b": 1})
    _yaml(write_file, "c/top.yaml", {"extends": ["b.yaml", "a.yaml"], "t": 1})
    monkeypatch.chdir(tmp_path)
    assert config_sources("c/top.yaml") == [Path("c/a.yaml"), Path("c/b.yaml"), Path("c/top.yaml")]
    assert config_sources("c/a.yaml") == [Path("c/a.yaml")]


def test_loaded_sources_follow_the_loaded_object(write_file, tmp_path, monkeypatch):
    _yaml(write_file, "c/a.yaml", {"a": 1})
    _yaml(write_file, "c/top.yaml", {"extends": "a.yaml", "t": 1})
    monkeypatch.chdir(tmp_path)
    cfg = load_config("c/top.yaml")
    cfg["t"] = 2  # modified in place: same object, same files
    assert loaded_sources(cfg) == [Path("c/a.yaml"), Path("c/top.yaml")]
    assert loaded_sources(copy.deepcopy(cfg)) is None
    assert loaded_sources({"a": 1, "t": 2}) is None


@pytest.mark.parametrize("self_cycle", [True, False])
def test_extends_cycle_is_an_error(write_file, self_cycle):
    if self_cycle:
        top = _yaml(write_file, "top.yaml", {"extends": "top.yaml"})
    else:
        _yaml(write_file, "a.yaml", {"extends": "b.yaml"})
        _yaml(write_file, "b.yaml", {"extends": "a.yaml"})
        top = _yaml(write_file, "top.yaml", {"extends": "a.yaml"})
    with pytest.raises(ConfigError, match="cycle"):
        load_config(top)
    with pytest.raises(ConfigError, match="cycle"):
        config_sources(top)


def test_extends_missing_base_names_both_files(write_file):
    top = _yaml(write_file, "top.yaml", {"extends": "nope.yaml"})
    with pytest.raises(ConfigError, match=r"not found: .*nope\.yaml.*extended by .*top\.yaml"):
        load_config(top)


@pytest.mark.parametrize("bad", [3, [1], [""], {"a": "b"}])
def test_extends_must_be_paths(write_file, bad):
    top = _yaml(write_file, "top.yaml", {"extends": bad})
    with pytest.raises(ConfigError, match="extends"):
        load_config(top)


def test_extends_base_must_be_a_mapping(write_file):
    write_file("base.yaml", "- a\n")
    top = _yaml(write_file, "top.yaml", {"extends": "base.yaml"})
    with pytest.raises(ConfigError, match="mapping"):
        load_config(top)


def test_required_keys_are_checked_after_the_merge(write_file):
    _yaml(write_file, "base.yaml", {"a": {"b": 1}})
    top = _yaml(write_file, "top.yaml", {"extends": "base.yaml", "c": 2})
    assert load_config(top, ["a.b", "c"])["a"]["b"] == 1
    with pytest.raises(ConfigError, match=r"missing keys d$"):
        load_config(top, ["a.b", "c", "d"])
    unset = _yaml(write_file, "unset.yaml", {"extends": "base.yaml", "a": {"b": None}})
    with pytest.raises(ConfigError, match=r"missing keys a\.b$"):  # None overrides the base
        load_config(unset, ["a.b"])


def test_merge_does_not_modify_its_inputs():
    base, over = {"a": {"x": [1]}}, {"a": {"y": 2}}
    out = merge(base, over)
    out["a"]["x"].append(2)
    assert base == {"a": {"x": [1]}} and over == {"a": {"y": 2}}


# -- scenario keys ---------------------------------------------------------------------
def test_scenario_needs_demand_approaches_or_calibration(scenario_cfg, write_file):
    del scenario_cfg["demand"]["approaches"]
    p = write_file("s.yaml", yaml.safe_dump(scenario_cfg))
    with pytest.raises(ConfigError, match=r"keys demand\.approaches or demand\.calibration$"):
        load_scenario(p)
    scenario_cfg["demand"]["calibration"] = "configs/calibration/x.yaml"
    p = write_file("s.yaml", yaml.safe_dump(scenario_cfg))
    assert load_scenario(p)["demand"]["calibration"] == "configs/calibration/x.yaml"


def test_alternatives_are_listed_with_other_missing_keys(write_file):
    p = write_file("c.yaml", "x: 1\n")
    with pytest.raises(ConfigError) as exc:
        load_config(p, ["a", ("b", "c"), "x"])
    assert str(exc.value).endswith("missing keys a, b or c")


# -- Pitié configs ---------------------------------------------------------------------
def test_pitie_actuated_extends_the_fixed_scenario():
    fixed, actuated = load_scenario(PITIE_FIXED_CONFIG), load_scenario(PITIE_ACTUATED_CONFIG)
    assert actuated["controller"] == "actuated" and actuated["scenario"] == "pitie_actuated"
    assert fixed["controller"] == "fixed" and fixed["network"]["tls_id"] == "C"

    def rest(cfg: dict) -> dict:
        return {k: v for k, v in cfg.items() if k not in ("scenario", "controller")}

    assert rest(actuated) == rest(fixed)
    assert demand_hash(fixed) == demand_hash(actuated)
    assert config_hash(fixed) != config_hash(actuated)


def test_pitie_config_sources():
    names = [p.as_posix() for p in config_sources(PITIE_ACTUATED_CONFIG)]
    assert names == [
        "configs/pitie/network.yaml", "configs/pitie/demand.yaml", "configs/pitie/control.yaml",
        "configs/pitie/measurement.yaml", "configs/pitie_fixed.yaml", "configs/pitie_actuated.yaml",
    ]  # fmt: skip


def test_pitie_contract_values(pitie_scenario):
    net = pitie_scenario["network"]
    assert set(net["approaches"]) == {"N", "S", "W", "NB", "WB"}
    links = sorted(i for g in net["signal_groups"].values() for i in g)
    assert links == list(range(16))  # every TLS link in exactly one group
    assert {c["link"] for c in net["crossings"].values()} == {12, 13, 14, 15}
    for section in ("fixed.green_s", "actuated.min_green_s", "actuated.max_green_s"):
        assert list(get(pitie_scenario, section)) == PITIE_STAGES, section
    safety = pitie_scenario["safety"]
    assert safety["ped_max_wait_s"] - safety["ped_red_margin_s"] == 88


# -- approaches ------------------------------------------------------------------------
def test_approach_edges_grid_form(base_scenario):
    assert approach_in_edges(base_scenario) == {
        "W": ["w_in"], "E": ["e_in"], "N": ["n_in"], "S": ["s_in"],
    }  # fmt: skip
    assert approach_out_edges(base_scenario) == {
        "W": ["w_out"], "E": ["e_out"], "N": ["n_out"], "S": ["s_out"],
    }  # fmt: skip


def test_approach_edges_list_form(pitie_scenario):
    assert approach_in_edges(pitie_scenario)["N"] == ["n_in"]
    assert approach_out_edges(pitie_scenario)["WB"] == ["wb_out"]
    cfg = {"network": {"approaches": {"X": {"in_edges": ["a_in", "b_in"], "out_edges": ["c"]}}}}
    assert approach_in_edges(cfg) == {"X": ["a_in", "b_in"]}
    assert approach_out_edges(cfg) == {"X": ["c"]}


@pytest.mark.parametrize(
    "approach", [{"in_edges": []}, {"out_edges": ["x_out"]}, {"in_edges": "x_in"}, 3]
)
def test_approach_in_edges_bad_forms(approach):
    with pytest.raises(ConfigError, match=r"approaches\.X"):
        approach_in_edges({"network": {"approaches": {"X": approach}}})


def test_approach_out_edge_of_a_grid_entry_needs_the_naming_rule():
    with pytest.raises(ConfigError, match=r"approaches\.X"):
        approach_out_edges({"network": {"approaches": {"X": "entry"}}})


# -- hashes ----------------------------------------------------------------------------
def test_config_hash_is_canonical(base_scenario):
    shuffled = dict(reversed(list(copy.deepcopy(base_scenario).items())))
    assert config_hash(shuffled) == config_hash(base_scenario)
    assert len(config_hash(base_scenario)) == 64
    moved = copy.deepcopy(base_scenario)
    moved["outputs"]["dir"] = "elsewhere"  # where results go does not change them
    assert config_hash(moved) == config_hash(base_scenario)
    changed = copy.deepcopy(base_scenario)
    changed["fixed"]["green_s"]["EW"] += 1
    assert config_hash(changed) != config_hash(base_scenario)


def test_demand_hash_covers_demand_inputs_only(base_scenario):
    h = demand_hash(base_scenario)
    other = copy.deepcopy(base_scenario)
    other["controller"], other["seed"], other["fixed"]["green_s"]["EW"] = "actuated", 7, 99
    assert demand_hash(other) == h
    edits = (
        lambda c: c["demand"].update(multiplier=1.1),
        lambda c: c["emergency"].update(depart_s=1),
        lambda c: c.update(begin_s=0),
        lambda c: c["network"].update(net_file="other.net.xml"),
    )
    for edit in edits:
        changed = copy.deepcopy(base_scenario)
        edit(changed)
        assert demand_hash(changed) != h


def test_demand_hash_reads_the_net_and_calibration_files(scenario_cfg, write_file):
    scenario_cfg["network"]["net_file"] = str(write_file("n.net.xml", "<net/>"))
    scenario_cfg["demand"]["calibration"] = str(write_file("cal.yaml", "a: 1\n"))
    h = demand_hash(scenario_cfg)
    write_file("cal.yaml", "a: 2\n")
    assert demand_hash(scenario_cfg) != h
    h = demand_hash(scenario_cfg)
    write_file("n.net.xml", "<net></net>")
    assert demand_hash(scenario_cfg) != h


NET_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n\n'
    "<!-- generated on {stamp} by Eclipse SUMO netconvert 1.28.0\n"
    "<netconvertConfiguration>\n"
    '    <input><node-files value="{nodes}"/></input>\n'
    "</netconvertConfiguration>\n"
    "-->\n\n"
)
NET_BODY = '<net version="1.20">\n    <edge id="e"/>\n</net>\n'


def _net(stamp: str, eol: bytes = b"\n", nodes: str = "a.nod.xml", body: str = NET_BODY) -> bytes:
    return (NET_HEADER.format(stamp=stamp, nodes=nodes) + body).encode().replace(b"\n", eol)


def test_demand_hash_ignores_line_ends_and_the_netconvert_header(scenario_cfg, tmp_path):
    """A git autocrlf checkout (CRLF) and a rebuild of the net that only rewrites the
    netconvert header (build time, input paths) leave the demand unchanged, so the hash
    published in docs/validation.md must not move either."""
    net, cal = tmp_path / "n.net.xml", tmp_path / "cal.yaml"
    scenario_cfg["network"]["net_file"] = str(net)
    scenario_cfg["demand"]["calibration"] = str(cal)
    net.write_bytes(_net("2026-10-11T22:06:36.657071+01:00"))
    cal.write_bytes(b"# generated\na: 1\nb: [1, 2]\n")
    h = demand_hash(scenario_cfg)
    cal.write_bytes(b"# generated\r\na: 1\r\nb: [1, 2]\r\n")  # Windows clone, autocrlf
    assert demand_hash(scenario_cfg) == h
    net.write_bytes(_net("2026-10-11T22:06:36.657071+01:00", b"\r\n"))
    assert demand_hash(scenario_cfg) == h
    net.write_bytes(b"\xef\xbb\xbf" + _net("2026-10-11T22:06:36.657071+01:00"))  # BOM
    assert demand_hash(scenario_cfg) == h
    net.write_bytes(_net("2027-01-02T03:04:05+01:00", nodes="/elsewhere/a.nod.xml"))  # rebuild
    assert demand_hash(scenario_cfg) == h
    net.write_bytes(_net("2027-01-02T03:04:05+01:00", body=NET_BODY.replace('"e"', '"f"')))
    assert demand_hash(scenario_cfg) != h  # the net itself changed
    net.write_bytes(_net("2026-10-11T22:06:36.657071+01:00"))
    cal.write_bytes(b"# generated\na: 2\nb: [1, 2]\n")
    assert demand_hash(scenario_cfg) != h  # the calibration changed


def test_demand_hash_of_the_pitie_scenario_does_not_depend_on_the_checkout(
    pitie_cfg, tmp_path, monkeypatch
):
    """The Pitié net and calibration file as a Linux (LF) or Windows (CRLF) clone checks
    them out, and the net after a no-op rebuild (new netconvert time stamp), all give the
    demand_hash of the working tree."""
    files = [Path(pitie_cfg["network"]["net_file"]), Path(pitie_cfg["demand"]["calibration"])]
    texts = [f.read_bytes().replace(b"\r\n", b"\n") for f in files]
    assert texts[0].count(b"<!-- generated on ") == 1  # the header the hash must skip
    h = demand_hash(pitie_cfg)
    restamped = re.sub(rb"generated on \S+", b"generated on 2099-01-01T00:00:00", texts[0])
    for name, eol, contents in (
        ("lf", b"\n", texts),
        ("crlf", b"\r\n", texts),
        ("rebuilt", b"\n", [restamped, texts[1]]),
    ):
        root = tmp_path / name
        for f, text in zip(files, contents, strict=True):
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            (root / f).write_bytes(text.replace(b"\n", eol))
        monkeypatch.chdir(root)
        assert demand_hash(pitie_cfg) == h, name

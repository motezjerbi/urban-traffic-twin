"""Field measurement file: absent and partial files, `source: assumption`, schema errors
listed together, OUT_OF_WINDOW instants, and the versioned template (all NOT_MEASURED)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from traffic_twin.config import ConfigError
from traffic_twin.validation.field import inventory, load_field, main, summary_line
from traffic_twin.validation.queues import queue_table

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "data/field/pitie_measurements.yaml"

CFG = {
    "controller": "fixed",
    "begin_s": 27000,
    "duration_s": 7200,
    "network": {
        "tls_id": "C",
        "stages": [{"name": "HOP"}, {"name": "STM"}],
        "signal_groups": {"N_straight": [2], "W": [10, 11], "ped_S": [13]},
        "approaches": {"N": {}, "S": {}, "W": {}, "NB": {}, "WB": {}},
        "crossings": {"S": {"link": 13}, "W": {"link": 15}},
    },
    "demand": {"bus": {"lines": [{"id": "91_E"}, {"id": "57_N"}]}},
}
SESSION = {"id": "s1", "date": "2026-10-13", "start": "07:30", "end": "09:30"}


def write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "field.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, **doc}, sort_keys=False), "utf-8")
    return path


def test_absent_file_is_empty_and_never_an_error(tmp_path):
    data = load_field(tmp_path / "nope.yaml", CFG)
    assert not data.exists and data.queues == [] and data.cycle_values() == []
    assert data.head_values("W", "green_s") == [] and data.signal_source == "field"
    assert all(v == (0, 0) for v in inventory(data).values())
    assert summary_line(data) == "0 measured; all rows NOT_MEASURED"


def test_partial_file(tmp_path):
    path = write(
        tmp_path,
        {
            "sessions": [SESSION],
            "signal": {
                "source": "field",
                "session": "s1",
                "cycle_s": [117, 118],
                "stage_order": [],
                "heads": {
                    "N_straight": {
                        "green_s": [33, None, 34],
                        "yellow_s": 3,
                        "all_red_after_s": None,
                    }
                },
            },
            "queues": [
                {
                    "session": "s1",
                    "time": "08:15:20",
                    "approach": "N",
                    "kind": "cycle_max",
                    "vehicles": 12,
                    "visible_m": 80,
                },
                {
                    "session": "s1",
                    "time": None,
                    "approach": "S",
                    "kind": "cycle_max",
                    "vehicles": None,
                },
            ],
        },
    )
    data = load_field(path, CFG)
    assert (
        data.exists and data.sessions["s1"].start_s == 27000 and data.sessions["s1"].end_s == 34200
    )
    assert data.cycle_values() == [117.0, 118.0]
    assert data.head_values("N_straight", "green_s") == [33.0, 34.0]
    assert data.head_values("N_straight", "yellow_s") == [3.0]
    assert data.head_values("N_straight", "all_red_after_s") == []
    inv = inventory(data)
    assert inv["signal"] == (3, 2)  # cycle_s, green_s, yellow_s | stage_order, all_red_after_s
    assert inv["queues"] == (1, 1)
    assert summary_line(data) == "4 measured; 3 NOT_MEASURED"


def test_source_assumption(tmp_path):
    path = write(
        tmp_path, {"sessions": [SESSION], "signal": {"source": "assumption", "cycle_s": 100}}
    )
    data = load_field(path, CFG)
    assert data.signal_source == "assumption" and data.cycle_values() == [100.0]


def test_schema_problems_are_listed_in_one_error(tmp_path):
    path = write(
        tmp_path,
        {
            "sessions": [SESSION],
            "signal": {
                "source": "video",
                "heads": {"N_through": {"green_s": [30]}, "W": {"green_s": [-5], "colour": "red"}},
            },
            "queues": [
                {
                    "session": "s1",
                    "time": "10:15",
                    "approach": "E",
                    "kind": "cycle_max",
                    "vehicles": -1,
                },
                {"session": "s2", "time": 29700, "approach": "N", "kind": "longest"},
            ],
            "counts": [
                {
                    "session": "s1",
                    "begin": "08:00",
                    "end": "08:15",
                    "movements": {"N_E": 3},
                    "buses": {"91_X": 1},
                    "pedestrians": {"N": 4},
                }
            ],
            "extra": 1,
        },
    )
    with pytest.raises(ConfigError) as err:
        load_field(path, CFG)
    msg = str(err.value)
    for needle in [
        "signal.source",
        "unknown signal group 'N_through'",
        "signal.heads.W.green_s",
        "unknown key 'colour'",
        "outside session s1",
        "unknown approach 'E'",
        "queues[0].vehicles",
        "unknown session 's2'",
        "bad clock time 29700",
        "queues[1].kind",
        "unknown movement 'N_E'",
        "unknown bus line '91_X'",
        "unknown crossing 'N'",
        "unknown key 'extra'",
    ]:
        assert needle in msg, needle
    assert "14 problem(s)" in msg


def test_out_of_window_instants(tmp_path):
    session = {**SESSION, "start": "07:00", "end": "10:00"}  # longer than the simulation
    queues = [
        {"session": "s1", "time": t, "approach": "N", "kind": "instant", "vehicles": 5}
        for t in ("07:10", "07:34", "09:45")
    ]
    data = load_field(write(tmp_path, {"sessions": [session], "queues": queues}), CFG)
    t = queue_table(_empty_queue(), data, CFG, {"queue_veh": 2, "queue_window_s": 300})
    assert t["status"].tolist() == ["OUT_OF_WINDOW"] * 3


def _empty_queue():
    import pandas as pd

    return pd.DataFrame(columns=["seed", "t", "approach", "jam_veh", "jam_m", "halting_veh"])


def test_repo_template_is_valid_and_all_not_measured(pitie_cfg):
    data = load_field(TEMPLATE, pitie_cfg)
    assert data.exists and data.signal_source == "field"
    assert sum(m for m, _ in inventory(data).values()) == 0
    assert inventory(data)["queues"] == (0, 6)  # 2 instants per general approach
    assert summary_line(data) == "0 measured; all rows NOT_MEASURED"
    assert set(data.signal.get("heads") or {}) == set(pitie_cfg["network"]["signal_groups"])


def test_check_cli(tmp_path, capsys):
    scenario = str(REPO_ROOT / "configs/pitie_fixed.yaml")
    main(["--check", "--file", str(TEMPLATE), "--scenario", scenario])
    assert capsys.readouterr().out.strip().endswith("0 measured; all rows NOT_MEASURED")
    main(["--check", "--file", str(tmp_path / "absent.yaml"), "--scenario", scenario])
    assert "absent: 0 measured; all rows NOT_MEASURED" in capsys.readouterr().out
    bad = write(tmp_path, {"queues": [{"approach": "Z", "kind": "instant"}]})
    with pytest.raises(SystemExit) as e:
        main(["--check", "--file", str(bad), "--scenario", scenario])
    assert e.value.code == 1
    assert "unknown approach 'Z'" in capsys.readouterr().err


def test_head_green_twice_per_cycle_needs_one_list_per_stage(tmp_path, pitie_cfg):
    """ped_NB is green in HOP and again in STM in the model: a plain list (one value per
    cycle, a mean or a total) cannot be matched to a stage and is refused; a per-stage
    mapping is accepted (new stage names too), and a head green once keeps its list."""

    def heads(ped_nb):
        return {
            "signal": {"heads": {"ped_NB": {"green_s": ped_nb}, "N_straight": {"green_s": [33]}}}
        }

    with pytest.raises(ConfigError) as err:
        load_field(write(tmp_path, heads([28, 28])), pitie_cfg)
    msg = str(err.value)
    assert "signal.heads.ped_NB.green_s" in msg and "green 2 times per cycle (HOP, STM)" in msg
    assert "1 problem(s)" in msg  # N_straight's plain list is fine
    data = load_field(write(tmp_path, heads({"HOP": [36, 35], "STM": [20], "PED": []})), pitie_cfg)
    assert data.head_raw("ped_NB", "green_s") == {"HOP": [36, 35], "STM": [20], "PED": []}
    assert data.head_values("ped_NB", "green_s") == [36.0, 35.0, 20.0]  # pooled
    assert inventory(data)["signal"] == (3, 1)
    with pytest.raises(ConfigError, match=r"ped_NB.green_s.HOP: must be positive"):
        load_field(write(tmp_path, heads({"HOP": [0]})), pitie_cfg)
    load_field(write(tmp_path, heads([])), pitie_cfg)  # not measured: never an error

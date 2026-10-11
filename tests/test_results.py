import json
import subprocess

import pandas as pd
import pytest

import traffic_twin.sim.results as results_mod
from traffic_twin.sim.kpi import KPI_COLUMNS
from traffic_twin.sim.results import RunResults, code_version, collect_results, write_results

STATE_ROWS = [
    {"t": 0, "entity": "intersection", "id": "C", "x": None, "y": None, "speed": None,
     "phase": 0, "queue": 3},
    {"t": 0, "entity": "car", "id": "v0", "x": 1.5, "y": 2.5, "speed": 8.0, "phase": None,
     "queue": None},
]  # fmt: skip


@pytest.fixture
def run_results(sumo_outputs) -> RunResults:
    return collect_results("run-1", "s", 7, *sumo_outputs, STATE_ROWS)


def test_collect_results_tags_every_table_with_run_id(run_results):
    for df in (run_results.kpis, run_results.trips, run_results.state):
        assert df.columns[0] == "run_id"
        assert (df["run_id"] == "run-1").all()
    assert list(run_results.kpis.columns) == ["run_id", *KPI_COLUMNS]
    assert sorted(run_results.trips["type"]) == ["bus", "car", "car", "pedestrian"]
    assert len(run_results.state) == 2


def test_collect_results_without_stops_file(sumo_outputs, tmp_path):
    res = collect_results("r", "s", 1, sumo_outputs[0], tmp_path / "none.xml", [])
    assert "headway_std" not in set(res.kpis["kpi"])
    assert res.state.empty


def test_write_results_round_trip(run_results, tmp_path):
    meta = {"run_id": "run-1", "scenario": "s", "seed": 7, "config": {"a": [1, 2]}}
    write_results(tmp_path, run_results, meta)
    assert {p.name for p in tmp_path.iterdir()} >= {
        "kpi_run.csv", "trip.parquet", "state_log.parquet", "run.json",
    }  # fmt: skip
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "kpi_run.csv"), run_results.kpis)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "trip.parquet"), run_results.trips)
    state = pd.read_parquet(tmp_path / "state_log.parquet")
    assert state["id"].tolist() == ["C", "v0"]
    assert state["queue"].iloc[0] == 3 and pd.isna(state["queue"].iloc[1])
    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8")) == meta


def test_code_version_is_short_hash_or_unknown():
    v = code_version()
    assert v == "unknown" or (4 <= len(v) <= 40 and v.isalnum())


@pytest.mark.parametrize("error", [OSError("no git"), subprocess.CalledProcessError(128, ["git"])])
def test_code_version_without_git(monkeypatch, error):
    def fail(*a, **kw):
        raise error

    monkeypatch.setattr(results_mod.subprocess, "run", fail)
    assert code_version() == "unknown"

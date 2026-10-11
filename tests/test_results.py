import copy
import json
import shutil
import subprocess

import pandas as pd
import pytest

import traffic_twin.sim.results as results_mod
from traffic_twin.config import load_scenario
from traffic_twin.sim.detectors import DetectorLayout, Loop
from traffic_twin.sim.kpi import KPI_COLUMNS
from traffic_twin.sim.results import (
    DETECTOR_COLUMNS,
    HEALTH_KEYS,
    QUEUE_COLUMNS,
    SIGNAL_COLUMNS,
    RunResults,
    code_version,
    collect_results,
    config_files,
    detector_table,
    queue_table,
    run_health,
    run_metadata,
    signal_table,
    write_results,
)

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
    head = v.removesuffix("-dirty")
    assert v == "unknown" or (4 <= len(head) <= 40 and head.isalnum())


def _fake_git(porcelain: str):
    def run(cmd, **kwargs):
        out = "abc1234\n" if "rev-parse" in cmd else porcelain
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    return run


@pytest.mark.parametrize(
    "porcelain, expected",
    [
        ("", "abc1234"),
        (" M src/traffic_twin/sim/kpi.py\n", "abc1234-dirty"),  # modified tracked file
        ("?? src/traffic_twin/control/actuated.py\n", "abc1234-dirty"),  # new, not committed
    ],
)
def test_code_version_marks_an_uncommitted_tree(monkeypatch, porcelain, expected):
    monkeypatch.setattr(results_mod.subprocess, "run", _fake_git(porcelain))
    assert code_version() == expected


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_code_version_of_a_real_repo(tmp_path, monkeypatch):
    """HEAD alone names the code only while the tree matches it: an edited tracked file
    or a new untracked file makes it `<hash>-dirty` (runs from an uncommitted tree must
    not be credited to the parent commit)."""
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)

    def git(*args: str) -> str:
        cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
        return subprocess.run([*cmd, *args], capture_output=True, text=True, check=True).stdout

    git("init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    git("add", "a.py")
    git("commit", "-q", "-m", "init")
    head = git("rev-parse", "--short", "HEAD").strip()
    assert code_version() == head
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert code_version() == f"{head}-dirty"
    git("checkout", "--", "a.py")
    assert code_version() == head
    (tmp_path / "b.py").write_text("y = 1\n", encoding="utf-8")
    assert code_version() == f"{head}-dirty"


@pytest.mark.parametrize("error", [OSError("no git"), subprocess.CalledProcessError(128, ["git"])])
def test_code_version_without_git(monkeypatch, error):
    def fail(*a, **kw):
        raise error

    monkeypatch.setattr(results_mod.subprocess, "run", fail)
    assert code_version() == "unknown"


# -- Part 3 additions ---------------------------------------------------------------------
LAYOUT = DetectorLayout(
    loops=(Loop("det_w_in_1", "w_in_1", 200.0, "control"),
           Loop("e1_W_in_w_in_1", "w_in_1", 100.0, "station"),
           Loop("e1_W_in_w_in_2", "w_in_2", 100.0, "station")),
    stage_loops=(("det_w_in_1",),),
    stations={"W_in": ("e1_W_in_w_in_1", "e1_W_in_w_in_2")},
    queues={},
)  # fmt: skip
E1 = pd.DataFrame(
    {"det_id": ["e1_W_in_w_in_1", "e1_W_in_w_in_2", "e1_other"], "begin_s": 27000.0,
     "end_s": 27900.0, "n_veh": [50, 40, 9], "occupancy_pct": [5.0, 4.0, 1.0],
     "mean_speed": [10.0, 11.0, 12.0]}
)  # fmt: skip
SWITCHES = pd.DataFrame(
    {"tls_id": "C", "from_lane": ["w_in_1", ":C_w0_0", "n_in_1"],
     "to_lane": ["e_out_1", ":C_c3_0", "s_out_1"], "begin_s": [27058.0, 27000.0, 27000.0],
     "end_s": [27085.0, 27035.0, 27040.0], "duration_s": [27.0, 35.0, 40.0]}
)  # fmt: skip
LINKS = {("n_in_1", "s_out_1"): 1, ("w_in_1", "e_out_1"): 11, (":C_w0_0", ":C_c3_0"): 17}


def test_detector_table_keeps_station_loops_with_relative_times():
    df = detector_table("r", E1, LAYOUT, 27000.0)
    assert list(df.columns) == DETECTOR_COLUMNS
    assert df["det_id"].tolist() == ["e1_W_in_w_in_1", "e1_W_in_w_in_2"]
    assert set(df["station"]) == {"W_in"} and df["lane"].tolist() == ["w_in_1", "w_in_2"]
    assert df["begin_s"].tolist() == [0.0, 0.0] and df["end_s"].tolist() == [900.0, 900.0]


def test_queue_table_schema():
    rows = [{"t": 0, "approach": "N", "jam_veh": 2, "jam_m": 14.5, "halting_veh": 3}]
    df = queue_table("r", rows)
    assert list(df.columns) == QUEUE_COLUMNS and df["run_id"].tolist() == ["r"]
    assert list(queue_table("r", []).columns) == QUEUE_COLUMNS


def test_signal_table_maps_lane_pairs_to_links():
    df = signal_table("r", SWITCHES, LINKS, 27000.0)
    assert list(df.columns) == SIGNAL_COLUMNS
    assert df[["link", "begin_s", "end_s"]].values.tolist() == [
        [1, 0, 40], [17, 0, 35], [11, 58, 85],
    ]  # fmt: skip
    assert df["link"].dtype == "int64"
    with pytest.raises(KeyError, match="w_in_1"):
        signal_table("r", SWITCHES, {("n_in_1", "s_out_1"): 1}, 27000.0)
    assert signal_table("r", SWITCHES.iloc[:0], LINKS, 0.0).empty


def test_run_health_combines_statistics_trips_and_drain():
    stats = {"teleports": 0, "collisions": 1, "emergency_stops": 0, "emergency_braking": 2,
             "vehicles_waiting_end": 0, "persons_jammed": 0}  # fmt: skip
    trips = pd.DataFrame(
        {"type": ["car", "bus", "pedestrian"], "depart_delay_s": [2.0, 10.0, float("nan")]}
    )
    health = run_health(stats, trips, drain_timeout=True)
    assert list(health)[: len(HEALTH_KEYS)] == list(HEALTH_KEYS)
    assert health["collisions"] == 1 and health["persons_jammed"] == 0
    assert health["depart_delay_mean"] == 6.0 and health["depart_delay_max"] == 10.0
    assert health["drain_timeout"] is True
    empty = run_health({}, pd.DataFrame(columns=["type", "depart_delay_s"]), False)
    assert empty["teleports"] is None and empty["depart_delay_mean"] is None


def test_run_metadata_has_the_part3_keys(scenario_cfg):
    health = dict.fromkeys(HEALTH_KEYS, 0)
    safety = {"interventions": {}, "audit": {}}
    meta = run_metadata(scenario_cfg, "r", 3, "2026-10-11T08:00:00+00:00", 1.23, health,
                        safety, ["configs/scenario_base.yaml"])  # fmt: skip
    assert {"config_hash", "demand_hash", "config_sources", "health", "safety"} <= set(meta)
    assert meta["config_sources"] == ["configs/scenario_base.yaml"]
    assert meta["wall_time_s"] == 1.2 and meta["config"] is scenario_cfg
    assert json.loads(json.dumps(meta)) == meta  # JSON serialisable


def test_config_files_lists_the_extends_chain():
    files = config_files({}, "configs/pitie_actuated.yaml")
    assert files[-2:] == ["configs/pitie_fixed.yaml", "configs/pitie_actuated.yaml"]
    assert "configs/pitie/network.yaml" in files and all("\\" not in f for f in files)
    assert config_files({}) is None  # built in memory


def test_config_files_of_a_loaded_config():
    cfg = load_scenario("configs/pitie_fixed.yaml")
    cfg["duration_s"] = 900  # modified in place: still the loaded object
    assert config_files(cfg)[-2:] == ["configs/pitie/measurement.yaml", "configs/pitie_fixed.yaml"]
    assert config_files(copy.deepcopy(cfg)) is None


def test_optional_tables_are_written_only_when_present(run_results, tmp_path):
    write_results(tmp_path, run_results, {"run_id": "run-1"})
    assert not {"detector.parquet", "queue.parquet", "signal.parquet"} & {
        p.name for p in tmp_path.iterdir()
    }
    run_results.detectors = detector_table("run-1", E1, LAYOUT, 27000.0)
    run_results.signal = signal_table("run-1", SWITCHES, LINKS, 27000.0)
    write_results(tmp_path, run_results, {"run_id": "run-1"})
    names = {p.name for p in tmp_path.iterdir()}
    assert {"detector.parquet", "signal.parquet"} <= names and "queue.parquet" not in names
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "detector.parquet"), run_results.detectors
    )


def test_collect_results_with_single_waits_and_routes(sumo_outputs, write_file):
    routes = write_file("r.rou.xml", '<routes><flow id="bus_B1" line="B1"/></routes>')
    res = collect_results("r", "s", 1, *sumo_outputs, [], pd.Series({"p": 42.0}), routes)
    get = res.kpis.set_index(["user_type", "kpi"])["value"]
    assert get["pedestrian", "ped_wait_single_max"] == 42.0
    assert "headway_std" in set(res.kpis["kpi"]) and "depart_delay_s" in res.trips

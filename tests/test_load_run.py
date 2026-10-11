import json
from unittest import mock

import pandas as pd
import pytest

import traffic_twin.db.load_run as load_run
from traffic_twin.db.load_run import (
    KPI_SQL,
    RUN_SQL,
    TRIP_SQL,
    load_run_dir,
    storage_path,
    trip_rows,
    upload_state_log,
)
from traffic_twin.sim.results import collect_results, write_results

META = {
    "run_id": "11111111-2222-3333-4444-555555555555",
    "scenario": "baseline_fixed",
    "seed": 7,
    "code_version": "abc1234",
    "config": {"scenario": "baseline_fixed", "seed": 7},
    "started_at": "2026-10-11T08:00:00+00:00",
    "duration_s": 600,
    "wall_time_s": 1.2,
}


@pytest.fixture
def run_dir(sumo_outputs, tmp_path):
    """A results directory as written by run_scenario."""
    out = tmp_path / "baseline_fixed_seed7"
    out.mkdir()
    state = [{"t": 0, "entity": "intersection", "id": "C", "x": None, "y": None,
              "speed": None, "phase": 0, "queue": 1}]  # fmt: skip
    results = collect_results(META["run_id"], META["scenario"], 7, *sumo_outputs, state)
    write_results(out, results, META)
    return out


@pytest.fixture
def no_storage_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)


def test_storage_path():
    assert storage_path({"scenario": "s", "run_id": "r"}) == "s/r/state_log.parquet"


def test_trip_rows_fill_missing_columns_with_none():
    walks = pd.DataFrame(
        {"run_id": ["r"], "id": ["p.0"], "type": ["pedestrian"], "depart_s": [1.0],
         "arrival_s": [2.0], "duration_s": [1.0], "time_loss_s": [0.5]}
    )  # fmt: skip
    (row,) = trip_rows(walks)
    assert row[:3] == ("r", "p.0", "pedestrian")
    assert row[7] is None and row[8] is None


def test_load_run_dir_inserts_run_kpis_and_trips(run_dir, no_storage_env):
    conn = mock.MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value

    assert load_run_dir(run_dir, conn) == META["run_id"]

    (sql, params), _ = cur.execute.call_args
    assert sql == RUN_SQL
    assert params[:4] == (META["run_id"], "baseline_fixed", 7, "abc1234")
    assert json.loads(params[4]) == META["config"]
    assert params[-1] == storage_path(META)

    calls = {c.args[0]: c.args[1] for c in cur.executemany.call_args_list}
    kpi_rows, trips = calls[KPI_SQL], calls[TRIP_SQL]
    kpis = pd.read_csv(run_dir / "kpi_run.csv")
    assert len(kpi_rows) == len(kpis)
    assert all(r[0] == META["run_id"] and isinstance(r[2], int) for r in kpi_rows)
    assert len(trips) == 4  # 2 cars, 1 bus, 1 walk
    walk = next(r for r in trips if r[2] == "pedestrian")
    assert walk[7] is None and walk[8] is None  # no waiting_s / stops for walks
    conn.commit.assert_called_once()


def test_state_log_upload_is_skipped_without_credentials(tmp_path, no_storage_env, caplog):
    with mock.patch.object(load_run.requests, "post") as post:
        upload_state_log(tmp_path / "missing.parquet", "s/r/state_log.parquet")
    post.assert_not_called()
    assert "not uploaded" in caplog.text


def test_state_log_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co/")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
    log_file = tmp_path / "state_log.parquet"
    log_file.write_bytes(b"PAR1")
    with mock.patch.object(load_run.requests, "post") as post:
        upload_state_log(log_file, "s/r/state_log.parquet")
    (url,), kw = post.call_args
    assert url == "https://example.supabase.co/storage/v1/object/sim-logs/s/r/state_log.parquet"
    assert kw["data"] == b"PAR1"
    assert kw["headers"]["Authorization"] == "Bearer test-key"
    post.return_value.raise_for_status.assert_called_once()


def test_main_loads_every_run_dir(run_dir, monkeypatch, no_storage_env):
    conn = mock.MagicMock()
    conn.__enter__.return_value = conn
    monkeypatch.setattr(load_run, "connect", lambda: conn)
    load_run.main([str(run_dir), str(run_dir)])
    assert conn.commit.call_count == 2

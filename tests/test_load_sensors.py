import os
import sys
import types
from unittest import mock

import pandas as pd
import pytest
import yaml

import traffic_twin.db.env as env
import traffic_twin.db.load_sensors as load_sensors
from traffic_twin.data.clean import SENSOR_HOURLY_COLUMNS
from traffic_twin.db.load_sensors import UPSERT_SQL, to_rows
from traffic_twin.paths import DATA_CONFIG, SENSOR_HOURLY_FILE


def sensor_frame(n: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iu_ac": [str(i) for i in range(n)],
            "libelle": ["x"] * n,
            "zone": ["z"] * n,
            "ts": [pd.Timestamp("2026-09-01 08:00", tz="Europe/Paris")] * n,
            "flow_veh_h": [float("nan")] + [500.0] * (n - 1),
            "occupancy_pct": [3.0] * n,
            "traffic_state": [None] * n,
            "lat": [48.8] * n,
            "lon": [2.3] * n,
        }
    )


def test_to_rows_turns_nan_into_none():
    (row,) = to_rows(sensor_frame())
    assert row[4] is None and row[6] is None and row[5] == 3.0


def test_to_rows_uses_table_column_order():
    df = sensor_frame(2)
    shuffled = df[list(reversed(df.columns))].assign(extra=1)
    assert to_rows(shuffled) == to_rows(df)
    assert len(to_rows(df)[0]) == len(SENSOR_HOURLY_COLUMNS)


def test_main_upserts_in_batches(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    sensor_frame(5).to_parquet(processed / SENSOR_HOURLY_FILE)
    cfg = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))
    cfg["paths"]["processed"] = str(processed)
    cfg_path = tmp_path / "data.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    conn = mock.MagicMock()
    conn.__enter__.return_value = conn
    cur = conn.cursor.return_value.__enter__.return_value
    monkeypatch.setattr(load_sensors, "connect", lambda: conn)

    load_sensors.main(["--config", str(cfg_path), "--batch", "2"])

    batches = [c.args for c in cur.executemany.call_args_list]
    assert [len(rows) for _, rows in batches] == [2, 2, 1]
    assert all(sql == UPSERT_SQL for sql, _ in batches)
    assert [r[0] for _, rows in batches for r in rows] == ["0", "1", "2", "3", "4"]


def test_read_dotenv(tmp_path, monkeypatch):
    for name in ("TT_TEST_A", "TT_TEST_B"):  # registered so monkeypatch removes them after
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("TT_TEST_KEEP", "from-env")
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comment\nTT_TEST_A = postgres://u:p@h/db?x=1\n\nnot a pair\n"
        "TT_TEST_KEEP=from-file\n#TT_TEST_B=commented\n"
    )
    env.read_dotenv(str(dotenv))
    assert os.environ["TT_TEST_A"] == "postgres://u:p@h/db?x=1"  # split on the first '='
    assert os.environ["TT_TEST_KEEP"] == "from-env"  # environment wins
    assert "TT_TEST_B" not in os.environ
    env.read_dotenv(str(tmp_path / "missing.env"))  # no file: no error


@pytest.fixture
def fake_psycopg(monkeypatch):
    fake = types.SimpleNamespace(connect=mock.MagicMock(name="psycopg.connect"))
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    return fake


def test_connect_needs_db_url(tmp_path, monkeypatch, fake_psycopg):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    with pytest.raises(SystemExit, match="SUPABASE_DB_URL"):
        env.connect()
    fake_psycopg.connect.assert_not_called()


def test_connect_uses_db_url(tmp_path, monkeypatch, fake_psycopg):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://test@localhost/db")
    assert env.connect() is fake_psycopg.connect.return_value
    fake_psycopg.connect.assert_called_once_with("postgresql://test@localhost/db")

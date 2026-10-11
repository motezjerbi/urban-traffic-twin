import pandas as pd

from traffic_twin.db.load_run import storage_path, trip_rows


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

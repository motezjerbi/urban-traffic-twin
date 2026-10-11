import pandas as pd

from traffic_twin.db.load_sensors import to_rows


def test_to_rows_turns_nan_into_none():
    df = pd.DataFrame(
        {
            "iu_ac": ["1"],
            "libelle": ["x"],
            "zone": ["z"],
            "ts": [pd.Timestamp("2026-09-01 08:00", tz="Europe/Paris")],
            "flow_veh_h": [float("nan")],
            "occupancy_pct": [3.0],
            "traffic_state": [None],
            "lat": [48.8],
            "lon": [2.3],
        }
    )
    (row,) = to_rows(df)
    assert row[4] is None and row[6] is None and row[5] == 3.0

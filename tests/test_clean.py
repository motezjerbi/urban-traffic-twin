import pandas as pd

from traffic_twin.data.clean import (
    SENSOR_HOURLY_COLUMNS,
    arc_coverage,
    clean_sensor_frame,
    hourly_profile,
    select_weekdays,
)


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iu_ac": [101, 101, 101, 101, 202, 202],
            "libelle": ["Bd_Hopital"] * 4 + ["Av_Jaures"] * 2,
            "t_1h": [
                "2026-09-01T06:00:00+00:00",  # Tue 08:00 Paris
                "2026-09-01T06:00:00+00:00",  # duplicate
                "2026-09-01T07:00:00+00:00",
                "2026-09-02T06:00:00+00:00",
                "2026-09-01T06:00:00+00:00",
                "2026-09-06T06:00:00+00:00",  # Sunday
            ],
            "q": [900, 900, 99999, 1100, 500, 300],  # 99999 is a fault
            "k": [12.0, 12.0, 15.0, 14.0, 3.0, 2.0],
            "etat_trafic": ["Fluide"] * 6,
            "etat_barre": ["Ouvert", "Ouvert", "Ouvert", "Ouvert", "Invalide", "Ouvert"],
            "geo_point_2d": [{"lat": 48.838, "lon": 2.361}] * 6,
        }
    )


def test_clean_drops_faults_duplicates_and_invalid_arcs():
    df = clean_sensor_frame(raw_frame(), zone="z")
    assert list(df.columns) == SENSOR_HOURLY_COLUMNS
    assert len(df) == 3  # 101 x2 (dup and fault removed), 202 Sunday only
    assert df["flow_veh_h"].max() <= 4000
    assert str(df["ts"].dt.tz) == "Europe/Paris"
    assert df.loc[0, "ts"].hour == 8
    assert df["lat"].iloc[0] == 48.838


def test_geo_as_string():
    raw = raw_frame().assign(geo_point_2d="48.88, 2.39")
    df = clean_sensor_frame(raw, zone="z")
    assert df["lon"].iloc[0] == 2.39


def test_weekdays_and_profile():
    df = select_weekdays(clean_sensor_frame(raw_frame(), zone="z"), [1, 2, 3])
    assert set(df["ts"].dt.dayofweek) <= {1, 2, 3}
    prof = hourly_profile(df)
    row = prof[(prof["iu_ac"] == "101") & (prof["hour"] == 8)].iloc[0]
    assert row["flow_mean"] == 1000
    assert row["n_days"] == 2


def test_coverage_capped_at_one():
    df = clean_sensor_frame(raw_frame(), zone="z")
    cov = arc_coverage(df, df["ts"].min(), df["ts"].min() + pd.Timedelta(hours=1))
    assert cov["coverage"].max() <= 1.0

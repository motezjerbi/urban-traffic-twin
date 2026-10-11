import struct
from datetime import date

import numpy as np
import pandas as pd

from traffic_twin.data.clean import (
    SENSOR_HOURLY_COLUMNS,
    _split_geo,
    arc_coverage,
    clean_sensor_frame,
    hourly_profile,
    select_weekdays,
    valid_flow,
    valid_occupancy,
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
    assert df.loc[0, "ts"].hour == 8  # the published stamp is kept (end of the period)
    assert df["lat"].iloc[0] == 48.838


def test_valid_flow_kept_when_occupancy_is_missing_or_invalid():
    raw = raw_frame()
    raw.loc[0, "k"] = np.nan
    raw.loc[3, "k"] = 250.0  # out of range
    df = clean_sensor_frame(raw, zone="z")
    assert len(df) == 3
    a = df[df["iu_ac"] == "101"].set_index("ts")
    assert a["flow_veh_h"].tolist() == [900, 1100]
    assert a["occupancy_pct"].isna().all()


def test_validity_masks_are_independent():
    bar = pd.Series(["Ouvert", "Ouvert", "Barré", "Ouvert"])
    q = pd.Series([500.0, np.nan, 500.0, 5000.0])
    k = pd.Series([np.nan, 3.0, 3.0, 3.0])
    assert valid_flow(q, bar).tolist() == [True, False, False, False]
    assert valid_occupancy(k, bar).tolist() == [False, True, False, True]
    assert valid_flow(q).tolist() == [True, False, True, False]


def test_geo_as_string():
    raw = raw_frame().assign(geo_point_2d="48.88, 2.39")
    df = clean_sensor_frame(raw, zone="z")
    assert df["lon"].iloc[0] == 2.39


def test_geo_as_wkb_point():
    # GeoParquet exports give geo_point_2d as little-endian WKB (x=lon, y=lat)
    wkb = struct.pack("<BIdd", 1, 1, 2.361, 48.838)
    lat, lon = _split_geo(pd.Series([wkb, None]))
    assert lat.iloc[0] == 48.838
    assert lon.iloc[0] == 2.361


def test_weekdays_and_profile():
    df = select_weekdays(clean_sensor_frame(raw_frame(), zone="z"), [1, 2, 3])
    assert set(df["ts"].dt.dayofweek) <= {1, 2, 3}
    prof = hourly_profile(df)
    assert "hour" not in prof
    # stamp 08:00 is the 07:00-08:00 period
    row = prof[(prof["iu_ac"] == "101") & (prof["hour_start"] == 7)].iloc[0]
    assert row["flow_mean"] == 1000
    assert row["n_days"] == 2
    # no calendar: the label says every date was kept (holidays included)
    assert set(prof["day_set"]) == {"Tue–Wed, all dates"}
    assert set(hourly_profile(df, None, [1, 2, 3])["day_set"]) == {"Tue–Thu, all dates"}


def test_weekday_of_a_period_is_that_of_its_start():
    # stamp Wed 00:00 = Tue 23:00-24:00: a Tuesday period
    raw = raw_frame().iloc[[0]].assign(t_1h="2026-09-01T22:00:00+00:00")
    df = clean_sensor_frame(raw, zone="z")
    assert df["ts"].dt.dayofweek.tolist() == [2]
    assert len(select_weekdays(df, [1])) == 1
    assert select_weekdays(df, [2]).empty


def test_profile_with_calendar_keeps_typical_days_only():
    cal = {
        "weekdays": [1, 2, 3],
        "school_holidays": [{"from": date(2026, 8, 1), "to": date(2026, 8, 31)}],
        "public_holidays": [],
    }
    raw = raw_frame()
    raw.loc[3, "t_1h"] = "2026-08-26T06:00:00+00:00"  # a Wednesday in the holidays
    prof = hourly_profile(clean_sensor_frame(raw, zone="z"), cal)
    assert set(prof["iu_ac"]) == {"101"}  # 202 only has a Sunday row
    row = prof[prof["hour_start"] == 7].iloc[0]
    assert row["flow_mean"] == 900 and row["n_days"] == 1
    assert set(prof["day_set"]) == {"school-term Tue–Thu"}


def test_weekend_profile_with_calendar_keeps_term_weekends_only():
    cal = {
        "weekdays": [1, 2, 3],
        "typical_period": {"from": date(2026, 8, 1), "to": date(2026, 12, 18)},
        "school_holidays": [{"from": date(2026, 8, 1), "to": date(2026, 8, 31)}],
        "public_holidays": [],
    }
    raw = raw_frame().iloc[[5, 5]].reset_index(drop=True)  # arc 202, Sunday 6 Sep
    raw.loc[1, "t_1h"] = "2026-08-30T06:00:00+00:00"  # a Sunday in the summer holidays
    sensors = clean_sensor_frame(raw.assign(etat_barre="Ouvert"), zone="z")
    prof = hourly_profile(sensors, cal, [5, 6])
    assert set(prof["day_set"]) == {"school-term Sat–Sun"}
    assert prof["n_days"].tolist() == [1] and prof["flow_mean"].tolist() == [300]
    every = hourly_profile(sensors, None, [5, 6])  # no calendar: the holiday Sunday too
    assert every["n_days"].tolist() == [2] and set(every["day_set"]) == {"Sat–Sun, all dates"}


def test_coverage_capped_at_one():
    df = clean_sensor_frame(raw_frame(), zone="z")
    cov = arc_coverage(df, df["ts"].min(), df["ts"].min() + pd.Timedelta(hours=1))
    assert cov["coverage"].max() <= 1.0

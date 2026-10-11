"""Real side of the flow validation, on a synthetic reference extract: slot weights,
the end-of-period stamp, typical days only, incomplete days dropped, the MAD filter,
NO_DATA / NO_SENSOR and the sensitivity arc. Uses `data.calendar` (WS2)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from traffic_twin.data import calendar
from traffic_twin.validation.reference import read_hours, real_flows

CAL = {
    "weekdays": [1, 2, 3],
    "school_holidays": [
        {"from": "2026-07-04", "to": "2026-08-31", "name": "summer"},
        {"from": "2026-10-17", "to": "2026-11-01", "name": "toussaint"},
    ],
    "public_holidays": ["2026-07-14", "2026-11-11"],
}
VCFG = {"reference": {"day_set": "term", "min_days": 8, "mad_k": 3.5}}
SLOTS = [
    {"name": "07:30-08:30", "begin": "07:30", "end": "08:30", "role": "A-02"},
    {"name": "08:30-09:30", "begin": "08:30", "end": "09:30", "role": "A-02"},
    {"name": "08:00-09:00", "begin": "08:00", "end": "09:00", "role": "control"},
    {"name": "07:30-09:30", "begin": "07:30", "end": "09:30", "role": "control"},
]
STATIONS = [
    {"station": "N_in", "iu_ac": "508", "alt_iu_ac": "510"},
    {"station": "X_in", "iu_ac": None},
]
BASE = {7: 600.0, 8: 700.0, 9: 800.0}  # flow by hour_start (07:00-08:00 -> 600)
DAYS = pd.date_range("2026-08-25", "2026-10-28", freq="D").date.tolist() + [dt.date(2026, 7, 14)]
TYPICAL = [
    d
    for d in DAYS
    if d.weekday() in (1, 2, 3) and dt.date(2026, 9, 1) <= d <= dt.date(2026, 10, 16)
]


def factor(day: dt.date) -> float:
    """Day-to-day variation (+-4 %), so the MAD is not zero."""
    return 1 + 0.02 * (day.toordinal() % 5 - 2)


def hours(arcs=("508", "510"), drop=(), override=None) -> pd.DataFrame:
    """Reference rows built like the CSV: from the published end-of-period stamps."""
    ts_end = pd.DatetimeIndex(
        [pd.Timestamp(d) + pd.Timedelta(hours=h) for d in DAYS for h in range(6, 13)]
    ).tz_localize("Europe/Paris")
    start = pd.DatetimeIndex(calendar.period_start(pd.Series(ts_end)))
    rows = []
    for iu in arcs:
        for end, s in zip(ts_end, start, strict=True):
            day = s.date()
            if (iu, str(day), s.hour) in drop:
                continue
            q = BASE.get(s.hour, 100.0) * factor(day) * (2.0 if iu == "510" else 1.0)
            if day not in TYPICAL:
                q *= 3  # holidays, weekends and Mondays would bias the mean
            if override:
                q = override(iu, day, s.hour, q)
            rows.append(
                {
                    "iu_ac": iu,
                    "libelle": "x",
                    "date": str(day),
                    "hour_start": s.hour,
                    "weekday": day.weekday(),
                    "flow_veh_h": q,
                    "occupancy_pct": 10.0,
                    "flow_valid": True,
                    "occ_valid": True,
                    "etat_barre": "Ouvert",
                    "ts_end": end.isoformat(),
                }
            )
    df = pd.DataFrame(rows)
    types = calendar.classify_days(sorted(set(df["date"])), CAL)
    types = types.assign(date=types["date"].astype(str).str[:10]).set_index("date")["day_type"]
    return df.assign(day_type=df["date"].map(types))


def mean_typical(weights: dict[int, float]) -> float:
    return float(
        np.mean([sum(w * BASE[h] * factor(d) for h, w in weights.items()) for d in TYPICAL])
    )


def _row(t: pd.DataFrame, slot: str, iu: str = "508") -> pd.Series:
    return t[(t["slot"] == slot) & (t["iu_ac"] == iu)].iloc[0]


def test_slot_weights_and_typical_days():
    t = real_flows(hours(), STATIONS, SLOTS, CAL, VCFG)
    assert calendar.slot_weights("07:30", "08:30") == {7: 0.5, 8: 0.5}
    r = _row(t, "07:30-08:30")
    assert r["n_days"] == len(TYPICAL) == 21 and r["status"] == "ok"
    assert r["real_mean"] == pytest.approx(mean_typical({7: 0.5, 8: 0.5}))
    assert _row(t, "07:30-09:30")["real_mean"] == pytest.approx(
        mean_typical({7: 0.25, 8: 0.5, 9: 0.25})
    )
    assert (r["first_day"], r["last_day"]) == ("2026-09-01", "2026-10-15")
    assert r["real_p10"] <= r["real_mean"] <= r["real_p90"] and r["real_std"] > 0


def test_stamp_0900_is_the_0800_0900_period():
    def spike(iu, day, hour, q):  # the value stamped 09:00 (period start 08:00)
        return 999.0 if hour == 8 else q

    t = real_flows(hours(override=spike), STATIONS, SLOTS, CAL, VCFG)
    assert _row(t, "08:00-09:00")["real_mean"] == pytest.approx(999.0)


def test_day_with_a_missing_hour_is_dropped():
    gap = {("508", "2026-09-09", 9)}  # 09:00-10:00 missing on one day
    t = real_flows(hours(drop=gap), STATIONS, SLOTS, CAL, VCFG)
    assert _row(t, "07:30-08:30")["n_days"] == 21  # does not need hour 9
    assert _row(t, "08:30-09:30")["n_days"] == 20


def test_mad_filter_drops_an_incident_day():
    def incident(iu, day, hour, q):
        return q / 4 if day == dt.date(2026, 9, 23) else q

    t = real_flows(hours(override=incident), STATIONS, SLOTS, CAL, VCFG)
    r = _row(t, "07:30-08:30")
    assert r["n_days"] == 20
    kept = [d for d in TYPICAL if d != dt.date(2026, 9, 23)]
    expected = np.mean([(BASE[7] + BASE[8]) / 2 * factor(d) for d in kept])
    assert r["real_mean"] == pytest.approx(expected)


def test_sensitivity_no_sensor_and_no_data():
    t = real_flows(hours(), STATIONS, SLOTS, CAL, {"reference": {"min_days": 30, "mad_k": 3.5}})
    alt = _row(t, "07:30-08:30", "510")
    assert bool(alt["sensitivity"]) and alt["station"] == "N_in"
    assert alt["real_mean"] == pytest.approx(2 * mean_typical({7: 0.5, 8: 0.5}))
    assert set(t[t["iu_ac"].notna()]["status"]) == {"NO_DATA"}  # 21 days < 30
    x = t[t["station"] == "X_in"]
    assert len(x) == len(SLOTS) and set(x["status"]) == {"NO_SENSOR"}


def test_read_hours_keeps_arc_ids_as_strings(tmp_path):
    path = tmp_path / "h.csv"
    hours(arcs=("508",)).head(3).to_csv(path, index=False)
    h = read_hours(path)
    assert h["iu_ac"].tolist() == ["508"] * 3
    with pytest.raises(FileNotFoundError, match="calib-data"):
        read_hours(tmp_path / "absent.csv")

from datetime import date

import numpy as np
import pandas as pd
import pytest

from traffic_twin.config import ConfigError, load_config
from traffic_twin.data.calendar import (
    WEEKEND,
    classify_days,
    day_label,
    period_start,
    robust_keep,
    slot_flows,
    slot_weights,
    term_days,
    typical_day_values,
    typical_period,
)
from traffic_twin.paths import DATA_CONFIG

CAL = {
    "weekdays": [1, 2, 3],
    "covers": {"from": date(2026, 4, 1), "to": date(2027, 1, 31)},
    "school_holidays": [
        {"from": date(2026, 7, 4), "to": date(2026, 8, 31), "name": "été"},
        {"from": "2026-10-17", "to": "2026-11-01", "name": "Toussaint"},
    ],
    "public_holidays": [date(2026, 7, 14), "2026-11-11"],
}


def stamps(*utc: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(utc), utc=True)).dt.tz_convert("Europe/Paris")


def test_stamp_is_the_end_of_the_period():
    start = period_start(stamps("2026-09-01T07:00Z"))  # 09:00 Paris (CEST)
    assert start.iloc[0] == pd.Timestamp("2026-09-01 08:00", tz="Europe/Paris")
    assert start.iloc[0].hour == 8
    # UTC input (the raw t_1h) gives the same local period start
    utc = pd.Series(pd.to_datetime(["2026-09-01T07:00Z"], utc=True))
    assert period_start(utc).iloc[0] == start.iloc[0]


def test_period_start_across_the_dst_change():
    # 25 Oct 2026: 03:00 CEST -> 02:00 CET. Stamp 09:00 CET is 08:00-09:00.
    s = period_start(stamps("2026-10-25T08:00Z", "2026-10-25T02:00Z", "2026-10-25T01:00Z"))
    assert [t.hour for t in s] == [8, 2, 2]
    assert s.iloc[0] == pd.Timestamp("2026-10-25 08:00", tz="Europe/Paris")
    # the doubled 02:00 hour: CET and CEST periods, one hour of absolute time apart
    assert s.iloc[1] - s.iloc[2] == pd.Timedelta(hours=1)
    assert [t.utcoffset() for t in s.iloc[1:]] == [pd.Timedelta(hours=1), pd.Timedelta(hours=2)]
    # 29 Mar 2026: 02:00 CET -> 03:00 CEST; stamp 03:00 CEST covers 01:00-02:00 CET
    spring = period_start(stamps("2026-03-29T01:00Z")).iloc[0]
    assert spring == pd.Timestamp("2026-03-29 01:00", tz="Europe/Paris")


def test_period_start_needs_aware_timestamps():
    with pytest.raises(ValueError, match="timezone-aware"):
        period_start(pd.Series(pd.to_datetime(["2026-09-01 09:00"])))


def test_classify_days():
    days = classify_days(
        ["2026-09-01", date(2026, 7, 14), date(2026, 8, 12), date(2026, 10, 20),
         date(2026, 9, 7), date(2026, 9, 5), pd.Timestamp("2026-11-11 08:00"),
         date(2026, 3, 3), date(2026, 9, 4)],
        CAL,
    ).set_index("date")  # fmt: skip
    assert list(days.columns) == ["weekday", "day_type", "typical", "hold_out"]
    assert not days["hold_out"].any()  # no typical_period: every term Tue-Thu is typical
    expect = {
        date(2026, 9, 1): ("term", True),  # Tuesday, school term
        date(2026, 7, 14): ("public_holiday", False),  # Fête nationale (a Tuesday)
        date(2026, 8, 12): ("school_holiday", False),  # August (Wednesday)
        date(2026, 10, 20): ("school_holiday", False),  # Toussaint 2026 (Tuesday)
        date(2026, 9, 7): ("term", False),  # Monday
        date(2026, 9, 4): ("term", False),  # Friday
        date(2026, 9, 5): ("weekend", False),
        date(2026, 11, 11): ("public_holiday", False),  # Wednesday
        date(2026, 3, 3): ("unknown", False),  # Tuesday outside calendar.covers
    }
    got = {d: (r.day_type, bool(r.typical)) for d, r in days.iterrows()}
    assert got == expect
    assert days.loc[date(2026, 9, 1), "weekday"] == 1


def test_classify_days_rejects_bad_calendars():
    with pytest.raises(ConfigError, match="weekdays"):
        classify_days([date(2026, 9, 1)], {})
    bad = {**CAL, "school_holidays": [{"from": date(2026, 9, 1)}]}
    with pytest.raises(ConfigError, match="school_holidays"):
        classify_days([date(2026, 9, 1)], bad)
    for period in ({"from": "2026-09-01"}, {"from": "2026-12-01", "to": "2026-09-01"}):
        with pytest.raises(ConfigError, match="typical_period"):
            classify_days([date(2026, 9, 1)], {**CAL, "typical_period": period})


AUTUMN = {**CAL, "typical_period": {"from": date(2026, 9, 1), "to": "2026-12-18"}}


def test_typical_period_turns_other_term_days_into_hold_out_days():
    assert typical_period(CAL) is None
    assert typical_period(AUTUMN) == (date(2026, 9, 1), date(2026, 12, 18))
    days = classify_days(
        [date(2026, 5, 12), date(2026, 9, 1), date(2026, 6, 15), date(2026, 8, 25)], AUTUMN
    ).set_index("date")
    got = {d: (r.day_type, bool(r.typical), bool(r.hold_out)) for d, r in days.iterrows()}
    assert got == {
        date(2026, 5, 12): ("term", False, True),  # spring term Tuesday: hold-out
        date(2026, 6, 15): ("term", False, False),  # Monday: neither
        date(2026, 8, 25): ("school_holiday", False, False),
        date(2026, 9, 1): ("term", True, False),
    }


def test_repo_calendar_gives_the_18_autumn_term_days():
    cal = load_config(DATA_CONFIG)["calendar"]
    days = classify_days(pd.date_range("2026-07-13", "2026-10-10").date, cal)
    typical = [d.isoformat() for d in days.loc[days["typical"], "date"]]
    sep = [f"2026-09-{d:02d}" for d in (1, 2, 3, 8, 9, 10, 15, 16, 17, 22, 23, 24, 29, 30)]
    assert typical == sep + ["2026-10-01", "2026-10-06", "2026-10-07", "2026-10-08"]
    assert not days["hold_out"].any()


def test_repo_calendar_keeps_spring_out_of_a_12_month_download():
    """The documented after-visit re-download pulls 12 months (configs/data.yaml:
    source.months). Its spring-term days must become hold-out days, not typical days
    mixed into the calibration and the A-02 reference (plan Phase 3, step 3)."""
    cal = load_config(DATA_CONFIG)["calendar"]
    days = classify_days(pd.date_range("2025-10-25", "2026-10-20").date, cal)
    typical = days.loc[days["typical"], "date"]
    held = days.loc[days["hold_out"], "date"]
    assert typical.min() >= date(2026, 9, 1) and len(typical) == 21  # to Thu 15 Oct
    assert len(held) == 34 and held.max() < date(2026, 9, 1)  # April to early July
    assert not (days["typical"] & days["hold_out"]).any()


def test_term_days_for_weekdays_and_weekends():
    dates = pd.date_range("2026-07-13", "2026-10-10").date
    cal = load_config(DATA_CONFIG)["calendar"]
    days = classify_days(dates, cal)
    assert term_days(dates, cal) == list(days.loc[days["typical"], "date"])
    weekends = term_days(dates, cal, WEEKEND)
    assert all(d.weekday() in WEEKEND for d in weekends)
    assert min(weekends) == date(2026, 9, 5)  # the summer-holiday weekends are left out
    assert len(weekends) == 11  # 5-6 Sep .. 3-4 Oct, then Sat 10 Oct
    assert date(2026, 7, 14) not in term_days(dates, cal, [1])  # public holiday
    assert term_days([date(2026, 3, 7)], CAL, WEEKEND) == []  # outside calendar.covers


def test_day_label():
    assert day_label([1, 2, 3]) == "Tue–Thu"
    assert day_label(WEEKEND) == "Sat–Sun"
    assert day_label([2, 0]) == "Mon, Wed"
    assert day_label([4]) == "Fri"


def test_slot_weights():
    assert slot_weights("07:30", "08:30") == {7: 0.5, 8: 0.5}
    assert slot_weights("07:30", "09:30") == {7: 0.25, 8: 0.5, 9: 0.25}
    assert slot_weights("08:00", "09:00") == {8: 1.0}
    assert slot_weights("07:30", "07:45") == {7: 1.0}
    assert slot_weights("23:00", "24:00") == {23: 1.0}
    assert sum(slot_weights("07:10", "09:50").values()) == pytest.approx(1.0)
    for bad in (("09:00", "08:00"), ("08:00", "08:00"), ("8h", "9h"), ("07:00", "25:00")):
        with pytest.raises(ValueError):
            slot_weights(*bad)


def hours_frame(values: dict[date, dict[int, float]], arc: str = "1") -> pd.DataFrame:
    rows = [
        {"iu_ac": arc, "date": d, "hour_start": h, "flow_veh_h": q, "flow_valid": not np.isnan(q)}
        for d, by_hour in values.items()
        for h, q in by_hour.items()
    ]
    return pd.DataFrame(rows)


def test_slot_flows_weights_hours_and_drops_incomplete_days():
    tue, wed, thu = date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)
    h = hours_frame(
        {
            tue: {7: 100.0, 8: 200.0, 9: 300.0},
            wed: {7: 100.0, 9: 300.0},  # hour 8 missing
            thu: {7: 100.0, 8: float("nan"), 9: 300.0},  # hour 8 invalid
        }
    )
    other = hours_frame({tue: {7: 999.0, 8: 999.0, 9: 999.0}}, arc="2")
    v = slot_flows(pd.concat([h, other]), "1", "07:30", "09:30")
    assert list(v.index) == [tue, wed, thu]
    assert v[tue] == pytest.approx(0.25 * 100 + 0.5 * 200 + 0.25 * 300)
    assert np.isnan(v[wed]) and np.isnan(v[thu])
    # an hour the window does not use may be missing
    assert slot_flows(h, "1", "07:00", "08:00").tolist() == [100.0, 100.0, 100.0]
    # a row flagged invalid counts as missing even with a number in it
    flagged = h.assign(flow_valid=[True, False, True] + [True] * 5)
    assert np.isnan(slot_flows(flagged, "1", "07:30", "09:30")[tue])


def test_robust_keep_drops_an_incident_day():
    v = pd.Series([400.0, 410, 395, 405, 398, 402, 620, np.nan])
    keep = robust_keep(v)
    assert keep.tolist() == [True] * 6 + [False, False]
    assert robust_keep(v, k=100).iloc[6]  # a loose filter keeps it
    flat = pd.Series([300.0, 300, 300, 450])  # zero MAD: nothing can be judged
    assert robust_keep(flat).all()
    assert robust_keep(pd.Series([], dtype=float)).empty


def test_typical_day_values_filters_days_and_outliers():
    days = {
        date(2026, 9, 1): 400.0,  # Tue term
        date(2026, 9, 2): 410.0,
        date(2026, 9, 3): 395.0,
        date(2026, 9, 8): 405.0,
        date(2026, 9, 9): 640.0,  # incident
        date(2026, 9, 10): 402.0,
        date(2026, 9, 7): 380.0,  # Monday
        date(2026, 8, 25): 250.0,  # Tue, summer holidays
        date(2026, 7, 14): 150.0,  # public holiday
    }
    h = hours_frame({d: {7: q, 8: q, 9: q} for d, q in days.items()})
    h = pd.concat([h, hours_frame({date(2026, 9, 15): {7: 400.0, 9: 400.0}})])  # incomplete
    v = typical_day_values(h, "1", "07:30", "09:30", CAL)
    assert list(v.index) == [date(2026, 9, d) for d in (1, 2, 3, 8, 10)]
    assert v.tolist() == [400.0, 410.0, 395.0, 405.0, 402.0]
    # without the MAD filter the incident day stays
    assert date(2026, 9, 9) in typical_day_values(h, "1", "07:30", "09:30", CAL, k=np.inf).index


def test_typical_and_hold_out_values_never_share_a_day():
    days = {date(2026, 9, 1): 400.0, date(2026, 9, 2): 410.0, date(2026, 5, 12): 380.0,
            date(2026, 5, 13): 385.0, date(2026, 8, 25): 250.0}  # fmt: skip
    h = hours_frame({d: {7: q, 8: q, 9: q} for d, q in days.items()})
    typical = typical_day_values(h, "1", "07:30", "09:30", AUTUMN)
    held = typical_day_values(h, "1", "07:30", "09:30", AUTUMN, hold_out=True)
    assert list(typical.index) == [date(2026, 9, 1), date(2026, 9, 2)]
    assert list(held.index) == [date(2026, 5, 12), date(2026, 5, 13)]
    # without a typical_period every term day is typical and nothing is held out
    assert len(typical_day_values(h, "1", "07:30", "09:30", CAL)) == 4
    assert typical_day_values(h, "1", "07:30", "09:30", CAL, hold_out=True).empty


def test_the_day_set_depends_on_the_window():
    """Calibration (07:30-09:30) and validation (per slot) run the same filter, yet keep
    different days: completeness and the MAD filter are judged on each window."""
    base = {date(2026, 9, d): {7: 400.0, 8: 400.0 + d, 9: 400.0} for d in (1, 2, 3, 8, 9, 10)}
    base[date(2026, 9, 10)][9] = float("nan")  # hour 9 invalid on Thu 10 Sep
    base[date(2026, 9, 9)][7] = 700.0  # an incident in hour 7 only
    h = hours_frame(base)
    window = typical_day_values(h, "1", "07:30", "09:30", CAL)
    first = typical_day_values(h, "1", "07:30", "08:30", CAL)
    assert date(2026, 9, 10) in first.index and date(2026, 9, 10) not in window.index
    late = typical_day_values(h, "1", "08:30", "09:30", CAL)
    assert date(2026, 9, 9) in late.index and date(2026, 9, 9) not in first.index

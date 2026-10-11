"""Typical-day selection for the Paris sensor data, shared by calibration and validation.

- Timestamps: the published hourly `t_1h` (cleaned `ts`) is the END of the period, so
  the period starts one hour earlier (`period_start`; Paris notice, decision D3).
- Typical days: Tue-Thu (`calendar.weekdays`), school term (zone C), no public holiday,
  inside `calendar.typical_period`, then a per-arc robust MAD filter on the window
  values (`robust_keep`, decision D4).
- Hold-out days: days that would be typical but fall outside `calendar.typical_period`
  (e.g. the spring term after a 12-month download). They never enter the calibration
  or the A-02 reference; `typical_day_values(..., hold_out=True)` returns them for a
  separate hold-out row.
- Windows: a time window (e.g. 07:30-09:30) is built from the hourly values with
  uniform weights inside each hour (`slot_weights`, decision D5).

Holiday tables come from `configs/data.yaml: calendar`. Times are Europe/Paris. Dates
outside `calendar.covers` (the range the holiday tables were checked for) are classified
`unknown` (or `weekend`) and are never typical nor hold-out.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import numpy as np
import pandas as pd

from traffic_twin.config import ConfigError

TIMEZONE = "Europe/Paris"
DAY_TYPES = ("term", "school_holiday", "public_holiday", "weekend", "unknown")
WEEKEND = (5, 6)  # Saturday, Sunday
MAD_SCALE = 1.4826  # MAD -> standard deviation for normally distributed values
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def period_start(ts_end: pd.Series) -> pd.Series:
    """Start of each hourly period from its published (end-of-period) timestamp, in
    Europe/Paris time. One hour of absolute time earlier, so the DST change is handled
    (stamp 03:00 CET on 25 Oct 2026 starts at 02:00 CET, stamp 02:00 CET at 02:00 CEST)."""
    ts = pd.Series(ts_end)
    if not isinstance(ts.dtype, pd.DatetimeTZDtype):
        raise ValueError("period_start needs timezone-aware timestamps (published t_1h is UTC)")
    return ts.dt.tz_convert(TIMEZONE) - pd.Timedelta(hours=1)


def to_date(value) -> dt.date:
    """A date from a date, a datetime / Timestamp (its calendar date) or an ISO string."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


def _dates(values: Iterable) -> list[dt.date]:
    return [to_date(v) for v in values]


def _calendar_dates(cal: dict, key: str) -> set[dt.date]:
    try:
        return set(_dates(cal.get(key) or []))
    except (TypeError, ValueError) as e:
        raise ConfigError(f"calendar.{key}: invalid date ({e})") from e


def _holiday_ranges(cal: dict) -> list[tuple[dt.date, dt.date]]:
    ranges = []
    for i, h in enumerate(cal.get("school_holidays") or []):
        if not isinstance(h, dict) or "from" not in h or "to" not in h:
            raise ConfigError(f"calendar.school_holidays[{i}] needs 'from' and 'to'")
        lo, hi = to_date(h["from"]), to_date(h["to"])
        if hi < lo:
            raise ConfigError(f"calendar.school_holidays[{i}]: 'to' is before 'from'")
        ranges.append((lo, hi))
    return ranges


def _date_range(cal: dict, key: str) -> tuple[dt.date, dt.date] | None:
    c = cal.get(key)
    if c is None:
        return None
    if not isinstance(c, dict) or "from" not in c or "to" not in c:
        raise ConfigError(f"calendar.{key} needs 'from' and 'to'")
    try:
        lo, hi = to_date(c["from"]), to_date(c["to"])
    except (TypeError, ValueError) as e:
        raise ConfigError(f"calendar.{key}: invalid date ({e})") from e
    if hi < lo:
        raise ConfigError(f"calendar.{key}: 'to' is before 'from'")
    return lo, hi


def typical_period(cal: dict) -> tuple[dt.date, dt.date] | None:
    """`calendar.typical_period` as (first, last) date, or None (no restriction)."""
    return _date_range(cal, "typical_period")


def _within(d: dt.date, span: tuple[dt.date, dt.date] | None) -> bool:
    return span is None or span[0] <= d <= span[1]


def _day_types(dates, cal: dict) -> list[tuple[dt.date, int, str, bool]]:
    """(date, weekday, day_type, inside typical_period) per distinct date, sorted.

    day_type precedence: public_holiday, weekend, school_holiday, term. A weekday outside
    `calendar.covers` is `unknown`."""
    public = _calendar_dates(cal, "public_holidays")
    holidays = _holiday_ranges(cal)
    covers = _date_range(cal, "covers")
    period = typical_period(cal)
    rows = []
    for d in sorted(set(_dates(dates))):
        wd = d.weekday()
        if d in public:
            kind = "public_holiday"
        elif wd in WEEKEND:
            kind = "weekend"
        elif not _within(d, covers):
            kind = "unknown"
        elif any(lo <= d <= hi for lo, hi in holidays):
            kind = "school_holiday"
        else:
            kind = "term"
        rows.append((d, wd, kind, _within(d, period)))
    return rows


def _weekday_set(cal: dict, weekdays=None) -> set[int]:
    if weekdays is None:
        if "weekdays" not in cal:
            raise ConfigError("calendar.weekdays is missing")
        weekdays = cal["weekdays"]
    return {int(d) for d in weekdays}


def classify_days(dates, cal: dict) -> pd.DataFrame:
    """One row per distinct date (sorted): date (datetime.date), weekday (0 = Monday),
    day_type, typical (bool) and hold_out (bool).

    day_type precedence: public_holiday, weekend, school_holiday, term. A weekday outside
    `calendar.covers` is `unknown`. A term day whose weekday is in `calendar.weekdays` is
    typical inside `calendar.typical_period` (every such day when the key is absent) and
    hold_out outside it. Hold-out days never enter the calibration or the validation
    reference; they are kept apart for a hold-out check.
    """
    weekdays = _weekday_set(cal)
    rows = []
    for d, wd, kind, in_period in _day_types(dates, cal):
        candidate = kind == "term" and wd in weekdays
        rows.append((d, wd, kind, candidate and in_period, candidate and not in_period))
    out = pd.DataFrame(rows, columns=["date", "weekday", "day_type", "typical", "hold_out"])
    return out.astype({"weekday": "int64", "typical": "bool", "hold_out": "bool"})


def term_days(dates, cal: dict, weekdays=None) -> list[dt.date]:
    """The distinct dates (sorted) of the school term whose weekday is in `weekdays`
    (default `calendar.weekdays`; 0 = Monday; weekends allowed): inside `calendar.covers`
    and `calendar.typical_period`, no school or public holiday.

    With the default weekdays these are the typical days of `classify_days`; with
    `WEEKEND` they are the term weekends the profile figures compare them with."""
    wanted = _weekday_set(cal, weekdays)
    holidays = _holiday_ranges(cal)
    covers = _date_range(cal, "covers")
    out = []
    for d, wd, kind, in_period in _day_types(dates, cal):
        if wd not in wanted or not in_period:
            continue
        if kind == "term" or (
            kind == "weekend"  # a weekend is in term when no school holiday spans it
            and _within(d, covers)
            and not any(lo <= d <= hi for lo, hi in holidays)
        ):
            out.append(d)
    return out


def day_label(weekdays) -> str:
    """Short label of a weekday set (0 = Monday): [1, 2, 3] -> "Tue–Thu",
    [5, 6] -> "Sat–Sun", [0, 2] -> "Mon, Wed"."""
    days = sorted({int(d) for d in weekdays})
    if len(days) > 1 and days == list(range(days[0], days[-1] + 1)):
        return f"{DAY_NAMES[days[0]]}–{DAY_NAMES[days[-1]]}"
    return ", ".join(DAY_NAMES[d] for d in days)


def _minutes(hhmm: str) -> int:
    try:
        h, m = (int(x) for x in str(hhmm).split(":"))
    except ValueError as e:
        raise ValueError(f"time '{hhmm}' is not HH:MM") from e
    if not (0 <= h <= 24 and 0 <= m < 60) or h * 60 + m > 24 * 60:
        raise ValueError(f"time '{hhmm}' is out of range")
    return h * 60 + m


def slot_weights(begin: str, end: str) -> dict[int, float]:
    """hour_start -> weight of that hour in the [begin, end) window (uniform flow inside
    each hour), e.g. ('07:30', '08:30') -> {7: 0.5, 8: 0.5}. Weights sum to 1."""
    b, e = _minutes(begin), _minutes(end)
    if e <= b:
        raise ValueError(f"window {begin}-{end}: end must be after begin")
    weights = {}
    for h in range(b // 60, (e - 1) // 60 + 1):
        overlap = min(e, (h + 1) * 60) - max(b, h * 60)
        if overlap > 0:
            weights[h] = overlap / (e - b)
    return weights


def _valid_flows(hours: pd.DataFrame, iu_ac: str) -> pd.DataFrame:
    a = hours[hours["iu_ac"].astype(str) == str(iu_ac)]
    ok = a["flow_veh_h"].notna()
    if "flow_valid" in a:
        ok &= a["flow_valid"].astype(bool)
    return a[ok]


def slot_flows(hours: pd.DataFrame, iu_ac: str, begin: str, end: str) -> pd.Series:
    """Mean flow (veh/h) of arc `iu_ac` over the window, per date; NaN if any needed
    hour is missing.

    `hours` has one row per arc, date (of the period start) and `hour_start`, with
    `flow_veh_h` and optionally `flow_valid` (rows with an invalid flow count as
    missing). The result is indexed by date (datetime.date), every date of the arc.
    """
    w = slot_weights(begin, end)
    arc = hours[hours["iu_ac"].astype(str) == str(iu_ac)]
    dates = sorted(set(_dates(arc["date"])))
    a = _valid_flows(hours, iu_ac)
    a = a[a["hour_start"].astype(int).isin(w)]
    grid = (
        pd.DataFrame(
            {
                "date": _dates(a["date"]),
                "hour": a["hour_start"].astype(int),
                "q": a["flow_veh_h"].astype(float),
            }
        )
        .groupby(["date", "hour"])["q"]
        .mean()  # only the doubled 02:00 hour of the DST change has two rows
        .unstack("hour")
        .reindex(index=dates, columns=sorted(w))
    )
    values = sum(grid[h].to_numpy(dtype=float) * wt for h, wt in w.items())
    values = np.where(grid.isna().any(axis=1).to_numpy(), np.nan, values)
    return pd.Series(values, index=pd.Index(dates, name="date"), name="flow_veh_h", dtype=float)


def robust_keep(values: pd.Series, k: float = 3.5) -> pd.Series:
    """Boolean mask of the values within k robust standard deviations (MAD) of the
    median. NaN values are never kept; with a zero MAD every non-NaN value is kept."""
    v = pd.Series(values, dtype=float)
    ok = v.notna()
    x = v[ok]
    if x.empty:
        return ok
    med = x.median()
    sigma = MAD_SCALE * (x - med).abs().median()
    if sigma == 0:
        return ok
    return ok & ((v - med).abs() <= k * sigma)


def typical_day_values(hours, iu_ac, begin, end, cal, k=3.5, hold_out=False) -> pd.Series:
    """Window flows of arc `iu_ac` on typical days after the MAD filter, per date.

    Typical: `classify_days(...).typical` and every hour of the [begin, end) window
    valid. The MAD filter (`robust_keep`, `k` robust SDs) runs on those window values.
    With `hold_out=True` the same rule runs on the hold-out days instead (typical days
    outside `calendar.typical_period`), for a separate hold-out row.

    Calibration (`data.calibrate`: one call on the 07:30-09:30 window) and validation
    (`validation.reference`: one call per slot) apply this same filter, but the days it
    keeps depend on the window: the completeness check and the MAD filter run on that
    window's values. A one-hour slot can keep a day the two-hour window loses (an hour
    outside the slot is missing) or drop a day the window keeps (an outlier on that slot
    only). The calibration file lists its days (`day_filter.days_used`); every
    validation row gives its own day count.
    """
    v = slot_flows(hours, iu_ac, begin, end)
    if v.empty:
        return v
    days = classify_days(v.index, cal)
    chosen = set(days.loc[days["hold_out" if hold_out else "typical"], "date"])
    v = v[[d in chosen for d in v.index]].dropna()
    return v[robust_keep(v, k)]

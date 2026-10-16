"""Replay clock of the API (Part 5, D53): which instant of a run is "now".

A request names a run (or gets the default run) and a time `t`: seconds since the
scenario start, or a clock time `HH:MM[:SS]` converted with the run's `begin_s`. Without
`t` the configured clock applies: `fixed` (always `clock.default_t`, deterministic) or
`loop` (the wall-clock time of day folded onto the run, for demos).

Replay instants are stamped on `clock.replay_date` (today when null) in Europe/Paris.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from traffic_twin.api.schemas import ApiError

PARIS = ZoneInfo("Europe/Paris")
MAX_PAST_END_S = 86400.0  # how far past the end of a run a replay time may go (stale answer)
CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


def clock_seconds(raw: str) -> int | None:
    """Seconds since midnight of `HH:MM[:SS]`, or None when `raw` is not a clock time.
    Raises ApiError(bad_t) for an out-of-range clock (25:00, 08:61)."""
    m = CLOCK_RE.match(raw.strip())
    if m is None:
        return None
    h, mi, s = int(m[1]), int(m[2]), int(m[3] or 0)
    if h > 23 or mi > 59 or s > 59:
        raise ApiError("bad_t", f"invalid clock time {raw!r}")
    return h * 3600 + mi * 60 + s


def wall_now() -> datetime:
    return datetime.now(PARIS)


def parse_t(
    raw: str | None,
    *,
    begin_s: float,
    duration_s: float,
    default_t: str,
    mode: str = "fixed",
    now: datetime | None = None,
) -> float:
    """Replay time (s since the scenario start) of a request. ApiError(bad_t) for an
    unparsable value, a time before the start of the run or more than MAX_PAST_END_S
    past its end; a time up to that bound past the end is allowed (the answer is then
    the last known value, marked stale)."""
    if raw is None or not raw.strip():
        if mode == "loop" and duration_s > 0:
            now = now or wall_now()
            of_day = now.hour * 3600 + now.minute * 60 + now.second
            return float((of_day - begin_s) % duration_s)
        return float((clock_seconds(default_t) or 0) - begin_s)
    of_day = clock_seconds(raw)
    if of_day is not None:
        t = float(of_day - begin_s)
    else:
        try:
            t = float(raw)
        except ValueError as e:
            raise ApiError("bad_t", f"t must be seconds or HH:MM[:SS], got {raw!r}") from e
        if not math.isfinite(t):
            raise ApiError("bad_t", f"t must be finite, got {raw!r}")
    if t < 0:
        raise ApiError("bad_t", f"t {raw!r} is before the start of the run")
    if t > max(float(duration_s), 0.0) + MAX_PAST_END_S:
        raise ApiError("bad_t", f"t {raw!r} is more than a day past the end of the run")
    return t


def replay_day(replay_date: date | None, now: datetime | None = None) -> date:
    return replay_date or (now or wall_now()).date()


def replay_instant(
    begin_s: float, t: float, replay_date: date | None, now: datetime | None = None
) -> datetime:
    """The Europe/Paris instant of replay time `t` (midnight + begin_s + t).
    ApiError(bad_t) when that instant is not representable (parse_t bounds t first)."""
    day = replay_day(replay_date, now)
    midnight = datetime(day.year, day.month, day.day, tzinfo=PARIS)
    try:
        return midnight + timedelta(seconds=begin_s + t)
    except (OverflowError, ValueError) as e:
        raise ApiError("bad_t", f"t {t!r} is out of the representable range") from e


def parse_deadline(raw: str | None, now: datetime) -> datetime | None:
    """A deadline: `HH:MM[:SS]` on the day of `now`, or an ISO datetime (naive means
    Europe/Paris). ApiError(bad_deadline) otherwise."""
    if raw is None or not raw.strip():
        return None
    try:
        of_day = clock_seconds(raw)
    except ApiError as e:
        raise ApiError("bad_deadline", e.message) from e
    if of_day is not None:
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight + timedelta(seconds=of_day)
    try:
        d = datetime.fromisoformat(raw.strip())
    except ValueError as e:
        raise ApiError("bad_deadline", f"deadline must be HH:MM or an ISO datetime: {raw!r}") from e
    return d if d.tzinfo else d.replace(tzinfo=PARIS)


def age_s(now: float, then: float) -> float:
    """Data age in replay seconds, never negative, rounded to 0.1 s."""
    return round(max(0.0, now - then), 1)

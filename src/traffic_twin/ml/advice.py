"""Departure advice (spec 11.5, D52; contract 4.5) as a pure function of the ETA bounds
and the deadline: no routing, no I/O, no clock (the caller passes `now`).

- high bound + rest of trip <= deadline -> `comfortable` ("Partez maintenant, marge
  confortable");
- else median + rest of trip <= deadline -> `small_margin` ("Marge faible");
- else -> `risk_of_delay` ("Risque de retard, prévenez").

A bound exactly at the deadline counts as on time; the low bound is not used. In degraded
mode (the reference interval, `approximate`) the message gets " (estimation
approximative)". The API repairs quantile crossing and adds the rest of the trip (D40)
before calling it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum


class Advice(StrEnum):  # a str: serialises as its value
    COMFORTABLE = "comfortable"  # high bound + rest of trip <= deadline
    SMALL_MARGIN = "small_margin"  # only the median arrives in time
    RISK = "risk_of_delay"  # even the median is late


MESSAGES_FR = {
    Advice.COMFORTABLE: "Partez maintenant, marge confortable",
    Advice.SMALL_MARGIN: "Marge faible",
    Advice.RISK: "Risque de retard, prévenez",
}
APPROXIMATE_SUFFIX = " (estimation approximative)"


@dataclass(frozen=True)
class DepartureAdvice:
    advice: Advice
    message: str
    arrival_median: datetime
    arrival_high: datetime
    margin_s: float  # deadline - arrival_median (negative: late)
    approximate: bool  # degraded mode: reference interval


def departure_advice(
    now: datetime,
    eta_median_s: float,
    eta_high_s: float,
    deadline: datetime,
    rest_of_trip_s: float = 0.0,
    approximate: bool = False,
) -> DepartureAdvice:
    """Rule 11.5 (D52). `eta_*_s` are the junction ETA bounds for a departure at `now`;
    `rest_of_trip_s` is the fixed time beyond the junction (D40, no uncertainty). Raises
    ValueError on a non-finite or negative input, or a high bound below the median
    (quantile crossing not repaired)."""
    for name, v in (
        ("eta_median_s", eta_median_s),
        ("eta_high_s", eta_high_s),
        ("rest_of_trip_s", rest_of_trip_s),
    ):
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{name} must be finite and >= 0, got {v}")
    if eta_high_s < eta_median_s:
        raise ValueError("eta_high_s < eta_median_s (quantile crossing not repaired)")
    arrival_median = now + timedelta(seconds=eta_median_s + rest_of_trip_s)
    arrival_high = now + timedelta(seconds=eta_high_s + rest_of_trip_s)
    if arrival_high <= deadline:
        advice = Advice.COMFORTABLE
    elif arrival_median <= deadline:
        advice = Advice.SMALL_MARGIN
    else:
        advice = Advice.RISK
    message = MESSAGES_FR[advice] + (APPROXIMATE_SUFFIX if approximate else "")
    margin_s = (deadline - arrival_median).total_seconds()
    return DepartureAdvice(advice, message, arrival_median, arrival_high, margin_s, approximate)

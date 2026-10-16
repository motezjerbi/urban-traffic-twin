"""Departure advice (spec 11.5, D52): the 17 prototype tests, ported unchanged apart from
the module path."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from traffic_twin.ml.advice import Advice, departure_advice

NOW = datetime(2026, 11, 4, 8, 0, 0)


def dl(minutes: float) -> datetime:
    return NOW + timedelta(minutes=minutes)


@pytest.mark.parametrize(
    "med,high,rest,deadline_min,expected",
    [
        (120, 300, 600, 20, Advice.COMFORTABLE),  # high arrives 08:15 < 08:20
        (120, 300, 600, 15, Advice.COMFORTABLE),  # high exactly at the deadline: in time
        (120, 300, 600, 14.9, Advice.SMALL_MARGIN),  # median 08:12 in time, high 08:15 late
        (120, 300, 600, 12, Advice.SMALL_MARGIN),  # median exactly at the deadline
        (120, 300, 600, 11.9, Advice.RISK),  # median late
        (120, 120, 0, 2, Advice.COMFORTABLE),  # zero-width interval
        (0, 0, 0, 0, Advice.COMFORTABLE),  # degenerate but valid
    ],
)
def test_rule(med, high, rest, deadline_min, expected):
    assert departure_advice(NOW, med, high, dl(deadline_min), rest).advice is expected


def test_messages_and_margin():
    r = departure_advice(NOW, 120, 300, dl(11), 600)
    assert r.message == "Risque de retard, prévenez"
    assert r.margin_s == pytest.approx(-60)
    assert r.arrival_median == NOW + timedelta(seconds=720)
    assert r.arrival_high == NOW + timedelta(seconds=900)


def test_approximate_flag_in_message():
    r = departure_advice(NOW, 120, 300, dl(30), 600, approximate=True)
    assert r.advice is Advice.COMFORTABLE and r.approximate
    assert r.message.endswith("(estimation approximative)")


def test_low_bound_is_not_used():
    # only median and high matter (11.5); a very low lower bound changes nothing
    a = departure_advice(NOW, 200, 400, dl(5), 0)
    assert a.advice is Advice.SMALL_MARGIN


def test_deadline_in_the_past():
    assert departure_advice(NOW, 60, 90, dl(-1), 0).advice is Advice.RISK


@pytest.mark.parametrize(
    "med,high,rest",
    [(math.nan, 100, 0), (100, math.inf, 0), (-1, 100, 0), (100, 200, -5), (300, 200, 0)],
)
def test_invalid_inputs_raise(med, high, rest):
    with pytest.raises(ValueError):
        departure_advice(NOW, med, high, dl(30), rest)


def test_monotone_in_deadline():
    order = {Advice.RISK: 0, Advice.SMALL_MARGIN: 1, Advice.COMFORTABLE: 2}
    prev = -1
    for m in range(30):
        a = order[departure_advice(NOW, 400, 700, dl(m), 120).advice]
        assert a >= prev
        prev = a

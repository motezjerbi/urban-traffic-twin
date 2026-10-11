"""Queue validation: the visible-range cap, cycle_max and instant samples pooled over
seeds, and the row statuses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_twin.validation.field import FieldData
from traffic_twin.validation.queues import capped, cycle_of, queue_sample, queue_table

THRESHOLDS = {"queue_veh": 2, "queue_window_s": 300}
CFG = {
    "controller": "fixed",
    "begin_s": 27000,  # 07:30
    "duration_s": 7200,
    "measurement": {"queues": {"N": {"lanes": ["n_in_1", "n_in_2"]}, "W": {"lanes": ["w_in_1"]}}},
}


def test_cap_to_the_visible_distance():
    veh, jam = pd.Series([10.0, 4.0, 0.0]), pd.Series([75.0, 30.0, 0.0])
    v, t = capped(veh, jam, 30.0)
    assert v.tolist() == [4.0, 4.0, 0.0] and t.tolist() == [True, False, False]
    v, t = capped(veh, jam, 30.0, n_lanes=2)  # 30 m of each of two lanes
    assert v.tolist() == [8.0, 4.0, 0.0] and t.tolist() == [True, False, False]
    v, t = capped(veh, jam, None)
    assert v.tolist() == veh.tolist() and not t.any()


def test_cycle_index_from_the_green_onsets():
    assert cycle_of(np.array([0, 99, 100, 250]), np.array([100.0, 200.0])).tolist() == [0, 0, 1, 2]


def sawtooth(seeds=(1, 2, 3), period=100.0, offset=0.0) -> tuple[pd.DataFrame, dict]:
    """Queue growing 0..19 vehicles over each 100 s cycle (samples every 5 s)."""
    t = np.arange(0, 7200, 5.0)
    rows = [
        pd.DataFrame(
            {
                "seed": s,
                "t": t,
                "approach": "N",
                "value": ((t - offset) % period) / 5,
                "truncated": False,
            }
        )
        for s in seeds
    ]
    onsets = {s: np.arange(offset, 7200, period) for s in seeds}
    return pd.concat(rows, ignore_index=True), onsets


def test_cycle_max_takes_one_maximum_per_seed_and_overlapping_cycle():
    q, onsets = sawtooth()
    s = queue_sample(q, 1000.0, "cycle_max", 300.0, onsets)
    # cycles overlapping [700, 1300]: those starting at 700, 800, ..., 1300 -> 7 per seed
    assert len(s) == 3 * 7
    assert s["value"].max() == 19 and s["value"].iloc[-1] <= 19


def test_instant_pools_every_sample_in_the_window():
    q, onsets = sawtooth()
    s = queue_sample(q, 1000.0, "instant", 300.0, onsets)
    assert len(s) == 3 * 121  # 700..1300 s every 5 s, 3 seeds
    assert s["value"].mean() == pytest.approx(9.5, abs=0.2)


def _field(queues: list[dict]) -> FieldData:
    return FieldData(Path("f.yaml"), True, {"queues": queues})


def _queue_parquet() -> pd.DataFrame:
    q, _ = sawtooth()
    return q.assign(jam_veh=q["value"], jam_m=q["value"] * 7.5, halting_veh=q["value"]).drop(
        columns=["value", "truncated"]
    )


def test_statuses():
    _, onsets = sawtooth()
    obs = [
        {"session": "s1", "time": "08:00", "approach": "N", "kind": "cycle_max", "vehicles": 18},
        {"session": "s1", "time": "08:00", "approach": "N", "kind": "cycle_max", "vehicles": 8},
        {"session": "s1", "time": "08:00:30", "approach": "N", "kind": "instant", "vehicles": 9},
        {"session": "s1", "time": "08:00", "approach": "N", "kind": "cycle_max", "vehicles": None},
        {"session": "s1", "time": None, "approach": "W", "kind": "cycle_max", "vehicles": None},
        {"session": "s1", "time": "07:33", "approach": "N", "kind": "cycle_max", "vehicles": 3},
        {"session": "s1", "time": "09:31", "approach": "N", "kind": "cycle_max", "vehicles": 3},
        {"session": "s1", "time": "08:00", "approach": "W", "kind": "cycle_max", "vehicles": 3},
        {
            "session": "s1",
            "time": "08:00",
            "approach": "N",
            "kind": "cycle_max",
            "vehicles": 30,
            "visible_m": 45.0,
        },
    ]
    t = queue_table(_queue_parquet(), _field(obs), CFG, THRESHOLDS, onsets)
    assert t["status"].tolist() == [
        "PASS",
        "FAIL",
        "PASS",
        "NOT_MEASURED",
        "NOT_MEASURED",
        "OUT_OF_WINDOW",
        "OUT_OF_WINDOW",
        "NOT_IN_SIM",
        "FAIL",
    ]
    first = t.iloc[0]
    assert first["sim_p50"] == 19 and first["abs_err"] == 1 and first["sim_n"] == 21
    capped_row = t.iloc[-1]  # 45 m x 2 lanes visible: at most 12 of the 19 queued
    assert capped_row["sim_p90"] == pytest.approx(12.0)
    seen = {**obs[-1], "kind": "instant"}
    inst = queue_table(_queue_parquet(), _field([seen]), CFG, THRESHOLDS, onsets).iloc[0]
    assert inst["truncated_share"] == pytest.approx(7 / 20, abs=0.01)  # samples above 12
    actuated = queue_table(
        _queue_parquet(), _field(obs[:1]), {**CFG, "controller": "actuated"}, THRESHOLDS, onsets
    )
    assert actuated["status"].tolist() == ["INFO"]


def test_no_field_file_gives_an_empty_table():
    t = queue_table(_queue_parquet(), FieldData(Path("absent.yaml")), CFG, THRESHOLDS, {})
    assert t.empty and "status" in t.columns

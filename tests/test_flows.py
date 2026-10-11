"""Flow validation (A-02): simulated loop flows per slot, the table statuses and the GEH
share. Synthetic inputs only (no sensor data, no SUMO)."""

from __future__ import annotations

import pandas as pd
import pytest

from traffic_twin.validation.flows import (
    SlotNotCovered,
    a02_summary,
    flow_table,
    geh_summary,
    sim_flows,
    slot_flow,
    station_counts,
)
from traffic_twin.validation.reference import REAL_COLUMNS

THRESHOLDS = {"flow_rel": 0.15, "geh": 5.0, "geh_share": 0.85}
SLOTS = [
    {"name": "07:30-08:30", "begin": "07:30", "end": "08:30", "role": "A-02"},
    {"name": "08:00-09:00", "begin": "08:00", "end": "09:00", "role": "control"},
]
STATIONS = [
    {"station": "N_in", "arm": "N", "direction": "in", "iu_ac": "508", "alt_iu_ac": "510",
     "group": "hop_sb_508", "alt_group": "copy_of_505", "a02": True},
    {"station": "S_in", "arm": "S", "direction": "in", "iu_ac": "511", "group": "hop_nb",
     "a02": True},
    {"station": "N_out", "arm": "N", "direction": "out", "iu_ac": "509", "group": "hop_nb",
     "a02": False},
    {"station": "W_in", "arm": "W", "direction": "in", "iu_ac": None, "group": None, "a02": True},
]  # fmt: skip
LANES = {"N_in": ["n_in_1", "n_in_2"], "S_in": ["s_in_1"], "N_out": ["n_out_1"], "W_in": ["w_in_1"]}


def _det(rows):
    return pd.DataFrame(rows, columns=["seed", "station", "lane", "begin_s", "end_s", "n_veh"])


def test_two_quarter_hours_of_50_vehicles_give_200_veh_h():
    det = _det([(1, "A", "a_1", 0.0, 900.0, 50), (1, "A", "a_1", 900.0, 1800.0, 50)])
    assert slot_flow(station_counts(det), 0.0, 1800.0)["A"] == pytest.approx(200.0)


def test_lanes_are_summed_into_the_station():
    det = _det([(1, "A", "a_1", 0.0, 900.0, 30), (1, "A", "a_2", 0.0, 900.0, 20)])
    counts = station_counts(det)
    assert len(counts) == 1 and counts["n_veh"].iloc[0] == 50
    assert slot_flow(counts, 0.0, 900.0)["A"] == pytest.approx(200.0)


def test_slot_must_be_a_union_of_detector_intervals():
    det = _det([(1, "A", "a_1", t, t + 900.0, 10) for t in (0.0, 900.0, 1800.0)])
    counts = station_counts(det)
    with pytest.raises(ValueError, match="union"):
        slot_flow(counts, 450.0, 1350.0)
    with pytest.raises(SlotNotCovered):
        slot_flow(counts, 0.0, 3600.0)


def test_sim_flows_per_seed_and_slot_relative_to_the_scenario_start():
    rows = []
    for seed, n in ((1, 50), (2, 60)):
        rows += [(seed, "A", "a_1", float(t), t + 900.0, n) for t in range(0, 7200, 900)]
    sim = sim_flows(_det(rows), SLOTS, begin_s=27000)  # 07:30
    got = sim.set_index(["seed", "slot"])["flow_veh_h"]
    assert got[(1, "07:30-08:30")] == pytest.approx(200.0)
    assert got[(2, "08:00-09:00")] == pytest.approx(240.0)
    short = _det([(1, "A", "a_1", 0.0, 900.0, 50)])  # 15-min run: no slot is covered
    assert sim_flows(short, SLOTS, 27000).empty


def _real(values: dict[tuple[str, str], float], n_days: int = 16) -> pd.DataFrame:
    rows = []
    for (station, iu), v in values.items():
        for slot in SLOTS:
            rows.append(
                {
                    "station": station,
                    "iu_ac": iu,
                    "sensitivity": iu == "510",
                    "slot": slot["name"],
                    "real_mean": v,
                    "real_std": 0.1 * v,
                    "real_p10": 0.9 * v,
                    "real_p90": 1.1 * v,
                    "n_days": n_days,
                    "status": "ok" if n_days >= 8 else "NO_DATA",
                }
            )
    rows += [
        {
            "station": "W_in",
            "iu_ac": None,
            "sensitivity": False,
            "slot": s["name"],
            "n_days": 0,
            "status": "NO_SENSOR",
        }
        for s in SLOTS
    ]
    return pd.DataFrame(rows).reindex(columns=REAL_COLUMNS)


def _sim(values: dict[str, float], seeds=range(1, 11)) -> pd.DataFrame:
    return pd.DataFrame(
        [(s, st, slot["name"], v) for s in seeds for st, v in values.items() for slot in SLOTS],
        columns=["seed", "station", "slot", "flow_veh_h"],
    )


REAL = {
    ("N_in", "508"): 100.0,
    ("N_in", "510"): 1300.0,
    ("S_in", "511"): 400.0,
    ("N_out", "509"): 400.0,
}


@pytest.mark.parametrize(
    "sim_n, status, info",
    [(114.9, "PASS", "ok"), (115.1, "FAIL", "off"), (85.1, "PASS", "ok"), (84.9, "FAIL", "off")],
)
def test_fifteen_percent_boundary(sim_n, status, info):
    t = flow_table(
        _real(REAL),
        _sim({"N_in": sim_n, "S_in": 400.0, "N_out": 400.0}),
        STATIONS,
        SLOTS,
        THRESHOLDS,
        LANES,
    )
    row = t[(t["station"] == "N_in") & (t["iu_ac"] == "508")].set_index("slot")
    assert row.loc["07:30-08:30", "role"] == "A-02"
    assert row.loc["07:30-08:30", "status"] == status
    assert row.loc["08:00-09:00", ("role")] == "control"
    assert row.loc["08:00-09:00", "status"] == info
    assert row.loc["07:30-08:30", "rel_err_pct"] == pytest.approx(100 * (sim_n / 100 - 1))
    assert row.loc["07:30-08:30", "seed_pass_share"] == (1.0 if status == "PASS" else 0.0)


def test_table_columns_roles_and_special_statuses():
    t = flow_table(
        _real(REAL),
        _sim({"N_in": 100.0, "S_in": 400.0, "N_out": 390.0}),
        STATIONS,
        SLOTS,
        THRESHOLDS,
        LANES,
    )
    assert list(t.columns)[:9] == [
        "station",
        "arm",
        "direction",
        "iu_ac",
        "alt_iu_ac",
        "group",
        "edges",
        "slot",
        "role",
    ]
    n_in = t[(t["station"] == "N_in") & (t["slot"] == "07:30-08:30")].set_index("iu_ac")
    assert n_in.loc["508", "edges"] == "n_in" and n_in.loc["508", "alt_iu_ac"] == "510"
    assert n_in.loc["510", "role"] == "sensitivity" and n_in.loc["510", "group"] == "copy_of_505"
    assert n_in.loc["510", "status"] == "off"  # 100 vs 1300
    assert set(t[t["station"] == "W_in"]["status"]) == {"NO_SENSOR"}
    assert set(t[t["station"] == "N_out"]["role"]) == {"info"}
    assert t.loc[t["station"] == "S_in", "n_seeds"].eq(10).all()
    assert t.loc[t["station"] == "S_in", "sim_std"].eq(0).all()
    assert t.loc[t["station"] == "S_in", "in_real_p10_p90"].all()
    few = flow_table(_real(REAL, n_days=5), _sim({"N_in": 100.0}), STATIONS, SLOTS, THRESHOLDS)
    assert set(few[few["station"] == "N_in"]["status"]) == {"NO_DATA"}
    assert set(few[few["station"] == "S_in"]["status"]) == {"NO_DATA"}  # no simulated value


def test_geh_share_counts_a_shared_station_once():
    # S_in (511) and N_out (509) are one Paris station (group hop_nb): one count per slot
    t = flow_table(
        _real(REAL),
        _sim({"N_in": 100.0, "S_in": 400.0, "N_out": 600.0}),
        STATIONS,
        SLOTS,
        THRESHOLDS,
        LANES,
    )
    g = geh_summary(t, SLOTS, THRESHOLDS)
    assert g["n_counts"] == 2  # hop_sb_508 and hop_nb in the one A-02 slot
    assert g["counts"] == ["hop_sb_508 07:30-08:30", "hop_nb 07:30-08:30"]
    assert g["share"] == 1.0 and g["ok"]  # N_out (GEH 8.9) is the second row of its group
    bad = flow_table(
        _real(REAL),
        _sim({"N_in": 100.0, "S_in": 600.0, "N_out": 400.0}),
        STATIONS,
        SLOTS,
        THRESHOLDS,
        LANES,
    )
    g = geh_summary(bad, SLOTS, THRESHOLDS)
    assert g["share"] == 0.5 and not g["ok"]


def test_a02_summary_verdicts():
    full = flow_table(
        _real(REAL), _sim({"N_in": 100.0, "S_in": 400.0}), STATIONS, SLOTS, THRESHOLDS
    )
    assert a02_summary(full)["verdict"] == "INCOMPLETE"  # W_in has no sensor
    stations = STATIONS[:3]
    ok = flow_table(_real(REAL), _sim({"N_in": 100.0, "S_in": 400.0}), stations, SLOTS, THRESHOLDS)
    assert a02_summary(ok) == {"verdict": "PASS", "rows": 2, "pass": 2, "fail": 0}
    bad = flow_table(_real(REAL), _sim({"N_in": 130.0, "S_in": 400.0}), stations, SLOTS, THRESHOLDS)
    assert a02_summary(bad)["verdict"] == "FAIL"
    no_sim = flow_table(_real(REAL), _sim({"N_in": 100.0}), stations, SLOTS, THRESHOLDS)
    assert a02_summary(no_sim)["verdict"] == "INCOMPLETE"  # S_in has no simulated value

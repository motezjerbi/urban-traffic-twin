"""ETA features `eta-features-1` (D41, D42) on synthetic snapshots: time to green on the
Pitié program, rolling windows, the leak-free join and the model / prediction frames."""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from traffic_twin.config import load_scenario
from traffic_twin.control.base import BusCall, EmergencyCall
from traffic_twin.ml.features import (
    CATEGORIES,
    DIAGNOSTIC,
    ETA_FEATURES,
    GAP_CAP_S,
    MODEL_FLOAT,
    ROW_COLUMNS,
    ROW_FEATURES,
    TRAJETS,
    FeatureState,
    build_site,
    feature_dtypes,
    join_samples,
    model_frame,
    prediction_frame,
    rows_frame,
    time_to_green,
)
from traffic_twin.paths import PITIE_ACTUATED_CONFIG, PITIE_FIXED_CONFIG, PITIE_RULES_CONFIG
from traffic_twin.sim.incidents import ActiveIncident
from traffic_twin.sim.sampler import JunctionSnapshot
from traffic_twin.sim.tls_check import load_program

CONFIGS = {
    "fixed": PITIE_FIXED_CONFIG,
    "actuated": PITIE_ACTUATED_CONFIG,
    "rules": PITIE_RULES_CONFIG,
}
HOP, S_LEFT, STM, BUS_91, BUS_57 = range(5)
CLEARANCE = (9.0, 10.0, 7.0, 12.0, 11.0)  # per stage, from pitie.tll.xml


def snapshot(**kw) -> JunctionSnapshot:
    """A quiet Pitié snapshot at 08:00 in HOP; keyword arguments override fields."""
    arms = ("N", "S", "W", "NB", "WB")
    stations = ("N_in", "S_in", "W_in", "N_out", "S_out", "W_out")
    base = {
        "t": 1800.0, "clock_s": 28800.0, "phase": 0, "stage": HOP, "in_transition": False,
        "time_in_stage": 0.0, "clearance_left_s": 0.0, "target_stage": None,
        "red_s": (0.0,) * 5, "gap_s": (math.inf,) * 5, "lanes": (),
        "arm_halting": dict.fromkeys(arms, 0), "arm_vehicles": dict.fromkeys(arms, 0),
        "arm_wait_max_s": dict.fromkeys(arms, 0.0), "exit_halting": dict.fromkeys(arms, 0),
        "pending": dict.fromkeys((*arms, "other"), 0),
        "pending_by_class": {"car": 0, "bus": 0, "ambulance": 0},
        "station_occupancy": dict.fromkeys(stations, 0.0),
        "station_new_vehicles": dict.fromkeys(stations, 0),
        "queue_jam_m": {"N": 0.0, "S": 0.0, "W": 0.0},
        "ped_count": (0, 0, 0, 0), "ped_wait_s": (0.0,) * 4, "ped_slack_s": (math.inf,) * 4,
        "emergency": (), "preempt_target": None, "buses": (), "bus_time_loss_s": {},
        "incidents": (),
    }  # fmt: skip
    return JunctionSnapshot(**(base | kw))


@pytest.fixture(scope="module")
def program():
    cfg = load_scenario(PITIE_FIXED_CONFIG)
    prog = load_program(cfg["network"]["net_file"], cfg["network"]["tls_id"])
    links = [(prog.links[i].from_lane, prog.links[i].to_lane) for i in sorted(prog.links)]
    return links, prog.states, prog.durations


@pytest.fixture(scope="module")
def sites(program):
    return {c: build_site(load_scenario(p), c, *program) for c, p in CONFIGS.items()}


# -- schema ------------------------------------------------------------------------------
def test_feature_lists_are_consistent(eta_cfg):
    assert len(set(ETA_FEATURES)) == len(ETA_FEATURES) == 72
    assert ETA_FEATURES[0] == "trajet" and set(CATEGORIES) <= set(ETA_FEATURES)
    assert not set(DIAGNOSTIC) & set(ETA_FEATURES)
    assert tuple(eta_cfg["model"]["excluded_features"]) == DIAGNOSTIC
    assert ROW_COLUMNS == ("t_feat", *ROW_FEATURES, *DIAGNOSTIC)
    assert "trajet" not in ROW_FEATURES and len(ROW_FEATURES) == 71
    assert tuple(eta_cfg["trajets"]) == TRAJETS == CATEGORIES["trajet"]
    assert set(eta_cfg["controllers"]) == set(CATEGORIES["controller"])
    assert feature_dtypes()["stage"] == "category" and feature_dtypes()["halt_N"] == MODEL_FLOAT


# -- site and time to green (D42) ---------------------------------------------------------
def test_site_from_the_pitie_program(sites):
    site = sites["fixed"]
    assert site.stage_names == ("HOP", "S_LEFT", "STM", "BUS_91", "BUS_57")
    assert site.clearance_s == CLEARANCE
    assert dict(site.trajet_stages) == {
        "N_S": (HOP,), "N_W": (STM, BUS_91, BUS_57), "S_N": (HOP, S_LEFT),
        "S_W": (S_LEFT,), "W_N": (STM,), "W_S": (STM,),
    }  # fmt: skip
    assert all(s.min_green_s == (6.0,) * 5 for s in sites.values())


def test_nominal_greens_per_controller(sites, program):
    assert sites["fixed"].nominal_green_s == (36.0, 6.0, 20.0, 6.0, 6.0)  # program greens
    assert sites["actuated"].nominal_green_s == (38.0, 6.0, 22.0, 6.0, 6.0)  # S1 max greens
    assert sites["rules"].nominal_green_s == (50.0, 6.0, 30.0, 6.0, 6.0)  # S2 max greens
    # an unseen controller (rl) falls back to the actuated max greens
    rl = build_site(load_scenario(PITIE_FIXED_CONFIG), "rl", *program)
    assert rl.nominal_green_s == sites["actuated"].nominal_green_s


def test_time_to_green_from_hop_at_min_greens(sites):
    site = sites["fixed"]
    state = FeatureState(load_scenario(PITIE_FIXED_CONFIG), "fixed", site)
    row = state.update(snapshot(stage=HOP, time_in_stage=0.0))
    assert row["green_N_S"] == 1 and row["ttg_min_N_S"] == 0.0 == row["ttg_nom_N_S"]
    assert row["ttg_min_W_N"] == 6 + 9 + 6 + 10 == 31.0  # HOP min, S_LEFT min, clearances
    assert row["ttg_nom_W_N"] == 36 + 9 + 6 + 10 == 61.0  # fixed: exact
    assert row["ttg_min_S_W"] == 6 + 9 == 15.0
    assert row["rem_min_s"] == 6.0 and row["rem_max_s"] == 36.0
    late = state.update(snapshot(stage=HOP, time_in_stage=40.0))  # beyond the nominal
    assert late["rem_min_s"] == 0.0 == late["rem_max_s"]
    assert late["ttg_nom_W_N"] == 9 + 6 + 10


def test_time_to_green_during_a_transition(sites):
    site = sites["rules"]
    snap = snapshot(
        stage=S_LEFT, in_transition=True, time_in_stage=6.0, clearance_left_s=4.0,
        target_stage=STM,
    )  # fmt: skip
    row = FeatureState(load_scenario(PITIE_RULES_CONFIG), "rules", site).update(snap)
    assert row["green_S_N"] == 0  # S_LEFT is being left
    assert row["ttg_min_W_N"] == 4.0 == row["ttg_nom_W_N"]  # target serves W
    assert row["ttg_min_N_S"] == 4 + (6 + 7) + (6 + 12) + (6 + 11)
    assert row["ttg_nom_N_S"] == 4 + (30 + 7) + (6 + 12) + (6 + 11)
    assert row["rem_min_s"] == 0.0 == row["rem_max_s"] and row["clear_left_s"] == 4.0
    assert row["in_transition"] == 1 and row["stage"] == "S_LEFT"


def test_time_to_green_edge_cases(sites):
    site = sites["fixed"]
    greens = site.min_green_s
    assert math.isnan(time_to_green(site, (), HOP, 0.0, False, 0.0, None, greens))
    # unknown target during a transition: the cycle successor
    assert time_to_green(site, (STM,), S_LEFT, 0.0, True, 3.0, None, greens) == 3.0


# -- other features -----------------------------------------------------------------------
def test_rolling_occupancy_and_flow_windows(sites):
    state = FeatureState(load_scenario(PITIE_FIXED_CONFIG), "fixed", sites["fixed"])
    occ = np.linspace(0.0, 0.9, 400)
    new = [k % 3 for k in range(400)]
    for k in range(400):
        stations = {"N_in": float(occ[k]), "S_in": 0.5, "W_in": 0.0}
        counts = {"N_in": new[k], "S_in": 1, "W_in": 0}
        row = state.update(
            snapshot(t=float(k), station_occupancy=stations, station_new_vehicles=counts)
        )
        if k == 29:  # fewer samples than the windows: the mean of what was seen
            assert row["occ60_N"] == pytest.approx(occ[:30].mean())
            assert row["flow300_S"] == pytest.approx(3600.0)
    assert row["occ60_N"] == pytest.approx(occ[-60:].mean())
    assert row["occ300_N"] == pytest.approx(occ[-300:].mean())
    assert row["flow300_N"] == pytest.approx(sum(new[-300:]) * 3600.0 / 300)
    assert row["flow300_S"] == pytest.approx(3600.0) and row["occ300_S"] == pytest.approx(0.5)
    assert row["occ60_W"] == 0.0 == row["flow300_W"]


def test_no_measurement_gives_nan_detector_features(sites):
    state = FeatureState(load_scenario(PITIE_FIXED_CONFIG), "fixed", sites["fixed"])
    row = state.update(snapshot(station_occupancy={}, station_new_vehicles={}, queue_jam_m={}))
    assert all(math.isnan(row[f"{k}_N"]) for k in ("occ60", "occ300", "flow300", "jam_m"))


def test_row_values(sites):
    cfg = load_scenario(PITIE_RULES_CONFIG)
    state = FeatureState(cfg, "rules", sites["rules"])
    bus = BusCall("bus_57_N.0", "57", 5, (S_LEFT,), "S", 40.0, 5.0, 0.0)
    calls = (
        EmergencyCall("ambulance_1", 2, HOP, math.inf, 100.0, True, "N"),
        EmergencyCall("ambulance_0", 10, STM, 120.0, 90.0, False, "W"),
    )
    snap = snapshot(
        t=2000.0, gap_s=(1.5, 2.0, math.inf, math.inf, math.inf), buses=(bus,),
        emergency=calls, preempt_target=STM,
        incidents=(ActiveIncident("w_in_1", "W", "w_in_1", 1900.0),),
        ped_count=(1, 0, 2, 0), ped_wait_s=(10.0, 0.0, 30.0, 0.0),
        pending={"N": 4, "S": 0, "W": 1, "NB": 0, "WB": 0, "other": 0},
        exit_bus_halting={"S": 1},
    )  # fmt: skip
    row = state.update(snap)
    assert tuple(row) == ROW_COLUMNS
    assert row["t_feat"] == 2000.0 and row["controller"] == "rules"
    assert row["day_type"] == "tue_wed_thu_term"
    assert row["tod_sin"] == pytest.approx(math.sin(2 * math.pi * 28800 / 86400))
    assert row["gap_cur_s"] == 1.5
    assert row["bus_S"] == 1 and row["bus_NB"] == 0 and row["bus_dwell_s_out"] == 1
    # nearest inserted call: 120 m on W -> bin 3; a pending call is present too
    assert row["amb_present"] == 1 and row["amb_pending"] == 1
    assert row["amb_dist_bin"] == 3 and row["amb_arm"] == "W" and row["preempt_active"] == 1
    assert row["inc_W"] == 1 and row["inc_age_s_W"] == 100.0
    assert row["inc_N"] == 0 and row["inc_age_s_N"] == 0.0
    assert row["ped_wait_max"] == 30.0 and row["ped_count"] == 3
    assert (row["pending_N"], row["pending_S"], row["pending_W"]) == (4, 0, 1)
    quiet = state.update(snapshot(t=2001.0))
    assert quiet["gap_cur_s"] == GAP_CAP_S and quiet["amb_arm"] == "none"
    assert quiet["amb_dist_bin"] == 0 and quiet["bus_dwell_s_out"] == 0
    pending_only = state.update(snapshot(t=2002.0, emergency=calls[:1]))
    assert pending_only["amb_dist_bin"] == 1 and pending_only["amb_arm"] == "N"
    near = dataclasses.replace(calls[1], dist_m=49.0)
    assert state.update(snapshot(t=2003.0, emergency=(near,)))["amb_dist_bin"] == 2
    far = dataclasses.replace(calls[1], dist_m=150.0)
    assert state.update(snapshot(t=2004.0, emergency=(far,)))["amb_dist_bin"] == 4


# -- frames ------------------------------------------------------------------------------
@pytest.fixture
def rows(sites) -> pd.DataFrame:
    state = FeatureState(load_scenario(PITIE_FIXED_CONFIG), "fixed", sites["fixed"])
    out = []
    for t in range(60):
        if 20 <= t < 30:  # a hole in the 1 Hz rows
            continue
        out.append(state.update(snapshot(t=float(t), time_in_stage=float(t))))
    return rows_frame(out)


def test_rows_frame_schema(rows):
    assert tuple(rows.columns) == ROW_COLUMNS
    assert rows["stage"].map(type).eq(str).all() and rows["halt_N"].dtype == "float64"


def test_join_never_uses_a_later_row(rows):
    samples = pd.DataFrame(
        {"veh_id": list("abcdef"), "insert_t": [10.0, 10.9, 0.0, 24.0, 33.0, 59.5],
         "controller": "fixed"}
    )  # fmt: skip
    out = join_samples(samples, rows, lag_s=1, tol_s=5.0)
    assert list(out["veh_id"]) == list("abcdef")  # order kept
    assert out.loc[0, "t_feat"] == 9.0 and out.loc[1, "t_feat"] == 9.0
    assert math.isnan(out.loc[2, "t_feat"])  # no row at -1 s
    assert out.loc[3, "t_feat"] == 19.0  # latest row at or before 23 s (hole 20-29)
    assert out.loc[4, "t_feat"] == 32.0 and out.loc[5, "t_feat"] == 58.0
    joined = out.dropna(subset=["t_feat"])
    assert (joined["t_feat"] <= np.floor(joined["insert_t"]) - 1).all()
    assert (joined["time_in_stage"] == joined["t_feat"]).all()  # values of that row
    assert "controller" in out.columns and out["controller"].eq("fixed").all()
    # beyond the tolerance: no features
    late = join_samples(pd.DataFrame({"veh_id": ["x"], "insert_t": [28.0]}), rows, tol_s=5.0)
    assert math.isnan(late.loc[0, "t_feat"]) and math.isnan(late.loc[0, "halt_N"])


def test_model_frame_dtypes_and_categories(rows):
    frame = prediction_frame(rows)
    frame.loc[0, "controller"] = "rl"  # unseen controller -> missing category
    mf = model_frame(frame)
    assert tuple(mf.columns) == ETA_FEATURES
    for name, cats in CATEGORIES.items():
        assert isinstance(mf[name].dtype, pd.CategoricalDtype)
        assert tuple(mf[name].cat.categories) == cats
    assert all(mf[f].dtype == MODEL_FLOAT for f in ETA_FEATURES if f not in CATEGORIES)
    assert pd.isna(mf.loc[0, "controller"]) and mf.loc[1, "controller"] == "fixed"
    with pytest.raises(KeyError, match="halt_N"):
        model_frame(frame.drop(columns=["halt_N"]))


def test_prediction_frame_shapes(rows):
    pred = prediction_frame(rows)
    assert len(pred) == len(rows) * len(TRAJETS)
    assert tuple(pred.columns) == ("t", "t_feat", *ETA_FEATURES, *DIAGNOSTIC)
    assert (pred["t"] == pred["t_feat"] + 1).all()
    assert list(pred["trajet"].unique()) == list(TRAJETS)
    for _, g in pred.groupby("trajet"):
        assert g["t"].is_monotonic_increasing and len(g) == len(rows)
    # from persisted snapshots carrying only `t`
    snaps = rows.drop(columns=["t_feat"]).assign(t=rows["t_feat"] + 1.0)
    again = prediction_frame(snaps)
    pd.testing.assert_series_equal(again["t_feat"], pred["t_feat"])

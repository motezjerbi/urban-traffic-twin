"""Part 4 evaluation (evaluation.evaluate) on synthetic run directories, no SUMO: the
N-scenario pairing refusals, A-04 (danger count, a 91 s pedestrian, one actuated
intervention), A-05 (exact one-sided p), the 13.3 hidden-degradation and rate rules, the
person-weighted delay, the S0-S1 pair being `compare()` itself, and the CLI writing
every output (exit 1 on an A-04 failure), the Part 8 ones included (D98; the Part 8
rules themselves are in test_evaluate_part8)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from traffic_twin.config import ConfigError, load_scenario
from traffic_twin.evaluation.compare import COMPARE_COLUMNS, compare
from traffic_twin.evaluation.evaluate import (
    EQUITY_COLUMNS,
    EvaluationError,
    Family,
    Member,
    a04_table,
    a05_report,
    a05_test,
    ambulance_plan,
    check_family,
    check_kpi_sets,
    equity_table,
    hidden_degradation,
    intervention_keys,
    load_evaluation,
    main,
    occupancy_sets,
    pair_tables,
    person_delay_mean,
    planned_trips,
    prereg_order,
    program_text,
    resolve_keys,
)
from traffic_twin.evaluation.runs import expected_hashes
from traffic_twin.sim.kpi import KPI_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_CONFIG = REPO_ROOT / "configs/evaluation.yaml"
SEEDS = list(range(1, 11))
# The danger audit keys of plan section 4.6 (sim.audit.DANGER_KEYS once WS1 lands)
DANGER = (
    "veh_green_to_red_without_yellow", "ped_green_to_red_outside_clearance", "yellow_to_green",
    "green_shorter_than_min", "clearance_incomplete", "intergreen_short", "foe_vehicle_greens",
    "illegal_transition", "unknown_phase",
)  # fmt: skip
HEADLINE = [
    ("car", "time_loss_mean"), ("car", "wait_p95"), ("bus", "delay_mean"),
    ("bus", "headway_std"), ("pedestrian", "ped_wait_mean"),
    ("pedestrian", "ped_wait_single_max"), ("ambulance", "time_loss_mean"),
]  # fmt: skip
BASE_KPIS = {
    ("car", "count"): 2000.0,
    ("car", "time_loss_mean"): 60.0,
    ("car", "wait_p95"): 140.0,
    ("car", "throughput_vph"): 1180.0,
    ("car", "delay_total_mean"): 80.0,
    ("car", "delay_total_mean__arm_N"): 90.0,
    ("car", "delay_total_mean__arm_W"): 100.0,
    ("bus", "delay_mean"): 60.0,
    ("bus", "headway_std"): 25.0,
    ("bus", "delay_mean__flow_91_E"): 62.0,
    ("bus", "delay_mean__flow_57_N"): 55.0,
    ("pedestrian", "ped_wait_mean"): 20.0,
    ("pedestrian", "ped_wait_single_max"): 80.0,
    ("pedestrian", "ped_wait_single_mean__crossing_S"): 21.0,
    ("pedestrian", "ped_wait_single_max__crossing_S"): 78.0,
    ("pedestrian", "ped_wait_single_max__crossing_W"): 80.0,
    ("pedestrian", "ped_curb_wait_mean"): 26.0,
    ("pedestrian", "ped_curb_wait_max"): 84.0,
    ("pedestrian", "ped_curb_wait_max__crossing_S"): 82.0,
    ("pedestrian", "ped_curb_wait_mean__crossing_S"): 23.0,
    ("pedestrian", "ped_curb_wait_max__crossing_W"): 84.0,
    ("pedestrian", "crossing_complete_rate"): 0.999,
    ("ambulance", "time_loss_mean"): 100.0,
    ("ambulance", "travel_time"): 120.0,
    ("ambulance", "travel_time_total"): 150.0,
    ("ambulance", "stops"): 1.3,
    ("ambulance", "stop_duration_mean"): 40.0,
}


def kpi_table(scale: dict | None = None, add: dict | None = None, seed0: int = 0) -> pd.DataFrame:
    """Ten seeds of KPIs with a little noise; `scale` multiplies, `add` adds to some."""
    rng = np.random.default_rng(seed0)
    rows = []
    for (u, k), v in BASE_KPIS.items():
        noise = rng.uniform(0.97, 1.03, len(SEEDS)) if k != "count" else np.ones(len(SEEDS))
        for i, s in enumerate(SEEDS):
            x = v * noise[i] * (scale or {}).get((u, k), 1.0) + (add or {}).get((u, k), 0.0)
            rows.append({"seed": s, "user_type": u, "kpi": k, "value": float(x)})
    return pd.DataFrame(rows)


def member(label: str, controller: str = "fixed", scenario: str | None = None) -> Member:
    cfg = {"scenario": scenario or f"sc_{label}", "controller": controller,
           "safety": {"ped_max_wait_s": 90}}  # fmt: skip
    return Member(label, Path(f"{label}.yaml"), cfg, Path("."))


def family(kpis: dict[str, pd.DataFrame], controllers: dict[str, str] | None = None) -> Family:
    labels = list(kpis)
    members = {x: member(x, (controllers or {}).get(x, "fixed")) for x in labels}
    pairs = [{"a": labels[0], "b": x, "role": "confirmatory"} for x in labels[1:]]
    return Family("main", labels[0], pairs, members, "d1", kpis)


# -- pairing ---------------------------------------------------------------------------
def _metas(seeds=SEEDS, demand="d1", config="c1", version=2, scenario="a"):
    return {
        s: {"seed": s, "scenario": scenario, "demand_hash": demand, "config_hash": config,
            "kpi_version": version}
        for s in seeds
    }  # fmt: skip


def test_family_pairing_refuses_seed_sets_demand_and_old_kpi_versions():
    exp = {"config_hash": "c1", "demand_hash": "d1"}
    metas = {"S0": _metas(scenario="a"), "S1": _metas(scenario="b"), "S2": _metas(scenario="c")}
    expected = dict.fromkeys(metas, exp)
    assert check_family(metas, SEEDS, expected, 2) == "d1"

    short = {**metas, "S2": _metas(SEEDS[:-1], scenario="c")}
    with pytest.raises(EvaluationError, match=r"S2 has no run for seeds \[10\].*seed sets differ"):
        check_family(short, SEEDS, expected, 2)

    other = {**metas, "S2": _metas(demand="d2", scenario="c")}
    with pytest.raises(EvaluationError, match="runs differ in demand_hash"):
        check_family(other, SEEDS, None, 2)
    with pytest.raises(EvaluationError, match="S2 runs are stale.*demand_hash d2 != current d1"):
        check_family(other, SEEDS, expected, 2)

    old = {**metas, "S1": _metas(version=1, scenario="b")}
    with pytest.raises(EvaluationError, match="S1 runs are stale.*kpi_version 1 < required 2"):
        check_family(old, SEEDS, expected, 2)
    legacy = {**metas, "S1": {s: {k: v for k, v in m.items() if k != "kpi_version"}
                              for s, m in _metas(scenario="b").items()}}  # fmt: skip
    with pytest.raises(EvaluationError, match="kpi_version 1 < required 2"):
        check_family(legacy, SEEDS, expected, 2)  # a Part 3 run.json has no kpi_version

    other_demand = {**expected, "S2": {"config_hash": "c1", "demand_hash": "d2"}}
    with pytest.raises(EvaluationError, match="do not share one demand"):
        check_family(metas, SEEDS, other_demand, 2)
    twice = {**metas, "S2": _metas(scenario="a")}
    with pytest.raises(EvaluationError, match="appear under two labels"):
        check_family(twice, SEEDS, expected, 2)


def test_a03_needs_one_kpi_set():
    k = kpi_table()
    assert check_kpi_sets({"S0": k, "S1": k}) == len(BASE_KPIS)
    fewer = k[k["kpi"] != "throughput_vph"]
    with pytest.raises(EvaluationError, match=r"A-03.*S1 lacks car\.throughput_vph"):
        check_kpi_sets({"S0": k, "S1": fewer})


# -- pairs -----------------------------------------------------------------------------
def test_s0_s1_pair_is_compare_itself():
    s0, s1 = kpi_table(), kpi_table({("car", "time_loss_mean"): 0.9}, seed0=1)
    fam = family({"S0": s0, "S1": s1})
    fam.pairs = [{"a": "S0", "b": "S1", "role": "part3"}]
    t = pair_tables(fam, HEADLINE, 0.05, 5.0)[("S0", "S1")]
    assert list(t.columns) == [*COMPARE_COLUMNS, "family", "role"]
    assert set(t["role"]) == {"part3"} and set(t["family"]) == {"main"}
    ref = compare(s0, s1, HEADLINE, 5.0, 0.05)
    pd.testing.assert_frame_equal(t[COMPARE_COLUMNS], ref)
    assert t["headline"].sum() == 7 and t.loc[t["headline"], "p_holm"].notna().all()


# -- A-05 ------------------------------------------------------------------------------
def test_a05_ten_uniformly_better_seeds_give_p_1_over_1024():
    s0 = kpi_table()
    s2 = kpi_table({("ambulance", "travel_time_total"): 0.6})  # same noise: every seed better
    fam = family({"S0": s0, "S2": s2}, {"S0": "fixed", "S2": "rules"})
    spec = {"a": "S0", "b": "S2", "user": "ambulance", "kpi": "travel_time_total",
            "alternative": "less", "alpha": 0.05}  # fmt: skip
    r = a05_test(fam, spec)
    assert r["n"] == 10 and r["seeds"] == SEEDS
    assert r["p_one_sided"] == pytest.approx(1 / 1024) and r["pass"]
    assert r["min_attainable_p"] == pytest.approx(1 / 1024)
    assert r["hl"] < 0 and r["hl_ci"][1] < 0
    assert r["diff"] == pytest.approx(list(np.subtract(r["b_values"], r["a_values"])))
    worse = a05_test(family({"S0": s2, "S2": s0}), spec)  # the other way round
    assert worse["p_one_sided"] == pytest.approx(1.0) and not worse["pass"]


# -- equity ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ecfg() -> dict:
    return load_evaluation(EVAL_CONFIG)


def _with_global(k: pd.DataFrame, scale: float) -> pd.DataFrame:
    g = k[(k["user_type"] == "car") & (k["kpi"] == "time_loss_mean")].copy()
    g["user_type"], g["kpi"], g["value"] = "all", "person_delay_mean", g["value"] * scale
    return pd.concat([k, g], ignore_index=True)


@pytest.mark.parametrize("worse_pct, hidden", [(6.0, True), (4.0, False)])
def test_hidden_degradation_fires_on_a_6_percent_subgroup_loss(ecfg, worse_pct, hidden):
    s0 = _with_global(kpi_table(), 1.0)
    s2 = _with_global(kpi_table({("car", "delay_total_mean__arm_W"): 1 + worse_pct / 100}), 0.9)
    fam = family({"S0": s0, "S2": s2})
    eq = equity_table(fam, ecfg, 0.05)
    assert list(eq.columns) == EQUITY_COLUMNS
    row = eq.set_index("kpi").loc["delay_total_mean__arm_W"]
    assert row["subgroup"] and not row["primary"] and row["qualifier"] == "arm_W"
    assert row["base_kpi"] == "delay_total_mean"
    assert row["rel_change_pct"] == pytest.approx(worse_pct)
    assert bool(row["worse_flag"]) is hidden
    assert eq.set_index(["user_type", "kpi"]).loc[("car", "time_loss_mean"), "primary"]
    (h,) = hidden_degradation(eq, ecfg, 0.05)
    assert h["global_better"] and h["global_p"] == pytest.approx(2 / 1024)
    assert h["hidden_degradation"] is hidden
    assert h["flagged"] == (["car.delay_total_mean__arm_W"] if hidden else [])


def test_no_hidden_degradation_without_a_significant_global_gain(ecfg):
    s0 = _with_global(kpi_table(), 1.0)
    s2 = _with_global(kpi_table({("car", "time_loss_mean"): 1.2}), 1.0)
    (h,) = hidden_degradation(equity_table(family({"S0": s0, "S2": s2}), ecfg, 0.05), ecfg, 0.05)
    assert h["flagged"] == ["car.time_loss_mean"] and not h["hidden_degradation"]


def test_rates_are_flagged_in_percentage_points(ecfg):
    s0 = kpi_table()
    rate = ("pedestrian", "crossing_complete_rate")
    small = kpi_table(add={rate: -0.004})  # -0.4 pp: not flagged
    large = kpi_table(add={rate: -0.014})  # -1.4 pp: flagged, although only -1.4 % relative
    eq = equity_table(family({"S0": s0, "S1": small, "S2": large}), ecfg, 0.05)
    r = eq[eq["kpi"] == "crossing_complete_rate"].set_index("scenario")
    assert r.loc["S1", "pp_change"] == pytest.approx(-0.4)
    assert r.loc["S2", "pp_change"] == pytest.approx(-1.4)
    assert abs(r.loc["S2", "rel_change_pct"]) < 5.0
    assert not r.loc["S1", "worse_flag"] and r.loc["S2", "worse_flag"]
    better = equity_table(family({"S0": s0, "S1": kpi_table(add={rate: 0.0009})}), ecfg, 0.05)
    assert not better.set_index("kpi").loc["crossing_complete_rate", "worse_flag"]
    assert eq[eq["kpi"] != "crossing_complete_rate"]["pp_change"].isna().all()


# -- person-weighted delay -------------------------------------------------------------
def test_person_delay_weights_vehicles_and_walks_by_occupancy(ecfg):
    trips = pd.DataFrame(
        [
            {"id": "car_N_1.0", "type": "car", "time_loss_s": 10.0, "depart_delay_s": 2.0},
            {"id": "car_W_1.0", "type": "car", "time_loss_s": 20.0, "depart_delay_s": np.nan},
            {"id": "bus_91_E.0", "type": "bus", "time_loss_s": 30.0, "depart_delay_s": 0.0},
            {"id": "ambulance_0", "type": "ambulance", "time_loss_s": 40.0, "depart_delay_s": 5.0},
            {"id": "ped_S_0.0", "type": "pedestrian", "time_loss_s": 6.0, "depart_delay_s": np.nan},
            {"id": "ped_S_0.0", "type": "pedestrian", "time_loss_s": 4.0, "depart_delay_s": np.nan},
            {"id": "ped_W_0.0", "type": "pedestrian", "time_loss_s": 8.0, "depart_delay_s": np.nan},
        ]
    )
    occ = {"car": 1.3, "bus": 25.0, "pedestrian": 1.0, "ambulance": 1.0}
    # cars 12 and 20 (missing departDelay = 0), bus 30, ambulance 45, persons 10 and 8
    num = 1.3 * 12 + 1.3 * 20 + 25 * 30 + 45 + 10 + 8
    den = 1.3 * 2 + 25 + 1 + 2
    assert person_delay_mean(trips, occ) == pytest.approx(num / den)
    sets = occupancy_sets(ecfg)
    assert list(sets) == [
        "person_delay_mean", "person_delay_mean__occ_bus_20", "person_delay_mean__occ_bus_30",
    ]  # fmt: skip
    assert sets["person_delay_mean"] == occ and sets["person_delay_mean__occ_bus_20"]["bus"] == 20
    with pytest.raises(EvaluationError, match="no occupancy"):
        person_delay_mean(trips, {"car": 1.0})


# -- A-04 ------------------------------------------------------------------------------
def _safety_meta(seed: int, danger: dict | None = None, inter: dict | None = None) -> dict:
    return {
        "seed": seed,
        "health": {"collisions": 0, "teleports": 0},
        "safety": {
            "audit": {**dict.fromkeys(DANGER, 0), "out_of_order_green": 0, **(danger or {})},
            "interventions": {**dict.fromkeys(intervention_keys(), 0), **(inter or {})},
        },
    }


class _Runs:
    def __init__(self, meta):
        self.meta = meta


def _a04_family(
    ped_max: float = 85.0, danger=None, inter=None, controller="actuated", seed=3
) -> Family:
    """`ped_max`: the curb wait of `seed`; every single wait is 80 s (reported only)."""
    k = kpi_table()
    k.loc[k["kpi"] == "ped_wait_single_max", "value"] = 80.0
    k.loc[k["kpi"] == "ped_curb_wait_max", "value"] = 82.0
    k.loc[(k["kpi"] == "ped_curb_wait_max") & (k["seed"] == seed), "value"] = ped_max
    fam = family({"S1": k}, {"S1": controller})
    meta = {s: _safety_meta(s) for s in SEEDS}
    meta[seed] = _safety_meta(seed, danger, inter)
    fam.members["S1"].runs = _Runs(meta)
    return fam


def test_a04_passes_a_clean_family(ecfg):
    t, failures = a04_table(_a04_family(), ecfg, DANGER, intervention_keys())
    assert failures == [] and t["pass"].all() and len(t) == 10
    assert list(t.columns[:6]) == ["family", "scenario", "controller", "seed", "collisions",
                                   "teleports"]  # fmt: skip
    assert {"danger_total", "out_of_order_green", "ped_curb_wait_max", "ped_max_wait_s",
            "ped_wait_single_max", "pass"} <= set(t.columns)  # fmt: skip
    assert (t["ped_max_wait_s"] == 90.0).all() and (t["ped_wait_single_max"] == 80.0).all()


def test_a04_fails_on_one_danger_count(ecfg):
    t, failures = a04_table(_a04_family(danger={"intergreen_short": 1}), ecfg, DANGER,
                            intervention_keys())  # fmt: skip
    assert len(failures) == 1 and "seed 3: intergreen_short 1" in failures[0]
    assert (
        t.set_index("seed").loc[3, "danger_total"] == 1 and not t.set_index("seed").loc[3, "pass"]
    )
    info, ok = a04_table(_a04_family(danger={"out_of_order_green": 2}), ecfg, DANGER,
                         intervention_keys())  # fmt: skip
    assert ok == [] and info["out_of_order_green"].sum() == 2  # information key only


def test_a04_fails_on_a_91_s_pedestrian(ecfg):
    """A-04 judges the curb wait (changes 2026-10-12): a 91 s curb wait fails although
    every single wait (getWaitingTime, reset by creeps) is 80 s."""
    _, failures = a04_table(_a04_family(ped_max=91.0), ecfg, DANGER, intervention_keys())
    assert failures and "pedestrian waited 91 s at the curb > 90 s" in failures[0]
    _, ok = a04_table(_a04_family(ped_max=90.0), ecfg, DANGER, intervention_keys())
    assert ok == []


def test_a04_bound_is_the_scenarios_ped_max_wait(ecfg):
    assert ecfg["a04"]["ped_curb_wait_max_s"] == "safety.ped_max_wait_s"
    fam = _a04_family(ped_max=91.0)
    fam.members["S1"].cfg["safety"]["ped_max_wait_s"] = 95
    t, ok = a04_table(fam, ecfg, DANGER, intervention_keys())
    assert ok == [] and (t["ped_max_wait_s"] == 95.0).all()
    del fam.members["S1"].cfg["safety"]
    _, failures = a04_table(fam, ecfg, DANGER, intervention_keys())
    assert "missing config key 'safety.ped_max_wait_s'" in failures[0]
    assert any("seed 3: no pedestrian bound" in f for f in failures)


def test_a04_never_reads_a_missing_curb_wait_as_zero(ecfg):
    fam = _a04_family()
    k = fam.kpis["S1"]
    fam.kpis["S1"] = k[~((k["kpi"] == "ped_curb_wait_max") & (k["seed"] == 4))]
    _, failures = a04_table(fam, ecfg, DANGER, intervention_keys())
    assert len(failures) == 1 and "seed 4: no ped_curb_wait_max" in failures[0]


def test_a04_fails_on_one_actuated_intervention_but_reports_rules(ecfg):
    key = intervention_keys()[0]
    _, failures = a04_table(_a04_family(inter={key: 1}), ecfg, DANGER, intervention_keys())
    assert len(failures) == 1 and "actuated guard interventions" in failures[0]
    t, ok = a04_table(_a04_family(inter={key: 1}, controller="rules"), ecfg, DANGER,
                      intervention_keys())  # fmt: skip
    assert ok == [] and t[key].sum() == 1  # S2 interventions are reported, not judged


def test_a04_never_reads_a_missing_count_as_zero(ecfg):
    fam = _a04_family()
    del fam.members["S1"].runs.meta[3]["safety"]["audit"]["unknown_phase"]
    fam.members["S1"].runs.meta[4]["health"]["collisions"] = None
    _, failures = a04_table(fam, ecfg, DANGER, intervention_keys())
    assert any("audit lacks unknown_phase" in f for f in failures)
    assert any("seed 4: collisions not reported" in f for f in failures)


# -- config ----------------------------------------------------------------------------
def test_the_preregistered_config_loads(ecfg):
    assert str(ecfg["preregistered"]) == "2026-10-12" and ecfg["kpi_version"] == KPI_VERSION == 3
    whats = " ".join(str(c["what"]) for c in ecfg["changes"])
    assert "`ped_curb_wait_max`" in whats and "kpi_version 2 -> 3" in whats  # A-04, D14/D27
    assert "Preemption time-out (D24)" in whats  # 120 s without progress
    assert "Equity (D29): the pedestrian primary KPI is `ped_curb_wait_max`" in whats
    assert ecfg["equity"]["primary"]["pedestrian"] == "ped_curb_wait_max"
    assert ecfg["equity"]["subgroups"]["pedestrian"] == "ped_curb_wait_mean__crossing_"


def test_equity_judges_pedestrians_on_the_curb_wait(ecfg):
    s0 = _with_global(kpi_table(), 1.0)
    s2 = _with_global(kpi_table({("pedestrian", "ped_curb_wait_mean__crossing_S"): 1.08,
                                 ("pedestrian", "ped_wait_single_mean__crossing_S"): 1.08}),
                      0.9)  # fmt: skip
    eq = equity_table(family({"S0": s0, "S2": s2}), ecfg, 0.05).set_index("kpi")
    assert eq.loc["ped_curb_wait_max", "primary"] and not eq.loc["ped_wait_single_max", "primary"]
    assert eq.loc["ped_curb_wait_mean__crossing_S", "subgroup"]
    assert not eq.loc["ped_wait_single_mean__crossing_S", "subgroup"]
    (h,) = hidden_degradation(eq.reset_index(), ecfg, 0.05)
    assert h["flagged"] == ["pedestrian.ped_curb_wait_mean__crossing_S"]
    assert list(ecfg["families"]) == ["main", "ambulance_probe"]
    assert resolve_keys(["a", "b"]) == ("a", "b")


def test_the_danger_keys_resolve_from_the_audit(ecfg):
    """`a04.danger_keys: sim.audit.DANGER_KEYS` (needs the Part 4 audit, WS1)."""
    from traffic_twin.sim import audit

    keys = resolve_keys(ecfg["a04"]["danger_keys"])
    assert keys == tuple(audit.DANGER_KEYS) and "out_of_order_green" not in keys


def test_config_cross_references_are_checked(tmp_path):
    raw = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
    raw["families"]["main"]["pairs"].append({"a": "S0", "b": "S9", "role": "confirmatory"})
    last = len(raw["families"]["main"]["pairs"]) - 1  # the appended pair
    raw["a05"]["primary"]["alternative"] = "smaller"
    raw["changes"] = [{"date": "2026-10-13", "what": "x"}]
    bad = tmp_path / "evaluation.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=rf"pairs\[{last}\].*alternative.*changes\[0\] needs why"):
        load_evaluation(bad)


# -- CLI on synthetic runs -------------------------------------------------------------
def trips_table(seed: int, factor: float) -> pd.DataFrame:
    rows = [
        {"id": f"car_N_{i}.0", "type": "car", "depart_s": 27000.0 + i, "arrival_s": 27100.0,
         "duration_s": 100.0, "time_loss_s": 40.0 * factor + i, "waiting_s": 10.0, "stops": 1,
         "depart_delay_s": 1.0}
        for i in range(4)
    ]  # fmt: skip
    rows += [
        {"id": "bus_91_E.0", "type": "bus", "depart_s": 27000.0, "arrival_s": 27200.0,
         "duration_s": 200.0, "time_loss_s": 50.0 * factor, "waiting_s": 20.0, "stops": 1,
         "depart_delay_s": 0.0},
        {"id": "ambulance_0", "type": "ambulance", "depart_s": 27600.0, "arrival_s": 27700.0,
         "duration_s": 100.0, "time_loss_s": 60.0 * factor + seed, "waiting_s": 20.0,
         "stops": 1, "depart_delay_s": 5.0},
        {"id": "ped_S_0.0", "type": "pedestrian", "depart_s": 27000.0, "arrival_s": 27100.0,
         "duration_s": 100.0, "time_loss_s": 20.0, "waiting_s": np.nan, "stops": np.nan,
         "depart_delay_s": np.nan},
    ]  # fmt: skip
    return pd.DataFrame(rows)


def write_run(results: Path, cfg: dict, seed: int, kpis: pd.DataFrame, factor: float, **meta):
    d = results / f"{cfg['scenario']}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    k = kpis[kpis["seed"] == seed].drop(columns="seed")
    k.assign(run_id="r", scenario=cfg["scenario"], seed=seed).to_csv(d / "kpi_run.csv", index=False)
    trips_table(seed, factor).assign(run_id="r").to_parquet(d / "trip.parquet", index=False)
    if cfg["controller"] == "rules":
        events = pd.DataFrame(
            [{"run_id": "r", "t": 10.0, "source": "controller", "rule": "ped_protect",
              "stage_from": 0, "stage_to": 1, "detail": ""}]
        )  # fmt: skip
        events.to_parquet(d / "events.parquet", index=False)
    safety = _safety_meta(seed)
    run = {
        "run_id": "r",
        "scenario": cfg["scenario"],
        "seed": seed,
        "code_version": "abc1234",
        "started_at": "2026-10-13T08:00:00+00:00",
        "kpi_version": KPI_VERSION,
        **expected_hashes(cfg),
        "health": safety["health"],
        "safety": {**safety["safety"], "danger": 0, "ped_forcing": "detected"},
        "preemption": {"enabled": cfg["controller"] == "rules", "requests": 1, "served": 1,
                       "timeouts": 0, "active_s": 30.0, "hold_s": 10.0, "jumps": 1,
                       "ped_yield": 0},
        "control": {"ped_protect": 1, "box_gapouts": 2,
                    "tsp": {"calls": 3, "extended_greens": 1, "extension_s": 6.0}},
    }  # fmt: skip
    for key, value in meta.items():  # e.g. safety=... to break a run
        run[key] = value
    (d / "run.json").write_text(json.dumps(run), encoding="utf-8")


EFFECTS = {  # label -> (KPI scale, trip delay factor)
    "S0": ({}, 1.0),
    "S1": ({("car", "time_loss_mean"): 0.9}, 0.95),
    "S2": ({("car", "time_loss_mean"): 0.8, ("ambulance", "travel_time_total"): 0.6}, 0.8),
    "S2_noP": ({("car", "time_loss_mean"): 0.78}, 0.85),
    # Part 8 (registered 2026-10-23): synthetic S3/S4 so the copied config's families run
    "S3": ({("car", "time_loss_mean"): 0.75, ("ambulance", "travel_time_total"): 0.65}, 0.78),
    "S4": ({("car", "time_loss_mean"): 0.74}, 0.82),
}


def synthetic_kpis(label: str, cfg: dict) -> pd.DataFrame:
    """The KPIs of a synthetic run of `label`: kpi_table with its EFFECTS scale, plus the
    ambulance `count` its config plans (every ambulance arrived)."""
    k = kpi_table(EFFECTS[label][0], seed0=7)
    planned = planned_trips(cfg, "ambulance")
    if planned is None:
        return k
    count = pd.DataFrame(
        {"seed": SEEDS, "user_type": "ambulance", "kpi": "count", "value": float(planned)}
    )
    return pd.concat([k, count], ignore_index=True)


def synthetic_setup(tmp_path: Path) -> tuple[Path, Path, dict]:
    """A copy of configs/evaluation.yaml with absolute paths and every run of both
    families under tmp_path/results, carrying the current configs' hashes."""
    raw = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
    raw["headline_from"] = (REPO_ROOT / raw["headline_from"]).as_posix()
    for fam in raw["families"].values():
        fam["scenarios"] = {k: (REPO_ROOT / v).as_posix() for k, v in fam["scenarios"].items()}
    raw["a04"]["danger_keys"] = list(DANGER)
    raw["outputs"] = {"dir": (tmp_path / "out").as_posix(),
                      "figures": (tmp_path / "fig").as_posix(),
                      "docs": (tmp_path / "evaluation.md").as_posix()}  # fmt: skip
    config = tmp_path / "evaluation.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    results = tmp_path / "results"
    cfgs = {}
    for name, fam in raw["families"].items():
        for label, path in fam["scenarios"].items():
            cfg = load_scenario(path)
            cfgs[(name, label)] = cfg
            kp = synthetic_kpis(label, cfg)  # one noise per family: the scale is the effect
            factor = EFFECTS[label][1]
            for seed in SEEDS:
                write_run(results, cfg, seed, kp, factor)
    return config, results, cfgs


def test_cli_writes_every_output(tmp_path, capsys):
    config, results, _ = synthetic_setup(tmp_path)
    docs = tmp_path / "evaluation.md"
    main(["--config", str(config), "--results", str(results), "--docs", str(docs)])
    out = results / "evaluation"
    for fam, pairs in (("main", ["S0__vs__S1", "S0__vs__S2", "S1__vs__S2", "S2_noP__vs__S2",
                                 "S0__vs__S3", "S1__vs__S3", "S2__vs__S3", "S4__vs__S3",
                                 "S2_noP__vs__S4"]),
                       ("ambulance_probe", ["S0__vs__S2", "S2_noP__vs__S2", "S0__vs__S3",
                                            "S4__vs__S3", "S2__vs__S3"])):  # fmt: skip
        for p in pairs:
            t = pd.read_csv(out / fam / "pairs" / f"{p}.csv")
            assert t["n_pairs"].eq(10).all() and set(t["family"]) == {fam}
            assert (out / fam / "pairs" / f"{p}.md").is_file()
        for name in ("equity.csv", "a04.csv", "kpis.csv"):
            assert (out / fam / name).is_file(), name
    a04 = pd.read_csv(out / "main" / "a04.csv")
    assert len(a04) == 60 and a04["pass"].all()  # S0, S1, S2, S2_noP, S3, S4 x 10 seeds
    kp = pd.read_csv(out / "main" / "kpis.csv")
    assert {"person_delay_mean", "person_delay_mean__occ_bus_20"} <= set(kp["kpi"])
    s01 = pd.read_csv(out / "main" / "pairs" / "S0__vs__S1.csv")
    assert set(s01["role"]) == {"part3"} and s01["headline"].sum() == 7

    a05 = json.loads((out / "a05.json").read_text(encoding="utf-8"))
    assert a05["verdict"] == "PASS"
    assert a05["primary"]["p_one_sided"] == pytest.approx(1 / 1024)
    assert [s["status"] for s in a05["secondary"]] == ["PASS"] * 5  # 3 Part 8 tests (S3)
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summ["a04"]["pass"] and summ["a03"]["text"] == "4/4"
    assert summ["preregistered_before_runs"] is True
    assert set(summ["person_delay"]["main"]["S0"]) == {
        "person_delay_mean", "person_delay_mean__occ_bus_20", "person_delay_mean__occ_bus_30",
    }  # fmt: skip
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert prov["preregistered"] == "2026-10-12" and len(prov["evaluation_sha256"]) == 64
    assert prov["families"]["main"]["scenarios"]["S2"]["scenario"] == "pitie_rules"
    assert (
        prov["families"]["main"]["demand_hash"]
        != prov["families"]["ambulance_probe"]["demand_hash"]
    )

    for name in ("final_table.csv", "a06.json", "series.csv", "digest.json"):  # D98
        assert (out / name).is_file(), name
    a06 = json.loads((out / "a06.json").read_text(encoding="utf-8"))
    assert a06["status"] == "PASS" and [c["reference"] for c in a06["comparisons"]] == [
        "S1", "S2", "S0",
    ]  # fmt: skip
    assert summ["a06"]["status"] == "PASS" and summ["final_available"] is True
    assert summ["registration"]["order"] == "n/a"  # tmp config: no registration commit
    assert set(prov["model_inputs"]) == set(raw_pins())

    page = docs.read_text(encoding="utf-8")
    for word in ("## 4. Runs", "A-03", "A-04 PASS", "A-05", "A-06 PASS", "ASSUMED",
                 "NOT_MEASURED", "## 15. Limitations", "S2_noP", "ped_protect"):  # fmt: skip
        assert word in page, word
    printed = capsys.readouterr().out
    assert "A-04 PASS" in printed and "A-05 PASS" in printed


def raw_pins() -> dict:
    """The D100 pins of the registered config (a06.model.inputs_text_sha8)."""
    raw = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
    return raw["a06"]["model"]["inputs_text_sha8"]


def test_cli_exits_1_on_an_a04_failure_after_writing(tmp_path, capsys):
    config, results, cfgs = synthetic_setup(tmp_path)
    cfg = cfgs[("main", "S1")]
    kp = synthetic_kpis("S1", cfg)
    broken = _safety_meta(2, inter={intervention_keys()[0]: 1})
    write_run(results, cfg, 2, kp, 0.95, safety=broken["safety"])
    with pytest.raises(SystemExit) as err:
        main(["--config", str(config), "--results", str(results)])
    assert err.value.code == 1
    summ = json.loads((results / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    assert not summ["a04"]["pass"] and "seed 2" in summ["a04"]["failures"][0]
    assert "A-04 FAIL" in capsys.readouterr().out


def test_cli_refuses_stale_or_unpaired_runs(tmp_path):
    config, results, cfgs = synthetic_setup(tmp_path)
    cfg = copy.deepcopy(cfgs[("main", "S2")])
    kp = synthetic_kpis("S2", cfg)
    write_run(results, cfg, 4, kp, 0.8, kpi_version=1)
    with pytest.raises(SystemExit, match=r"S2 runs are stale.*seed 4: kpi_version 1 < required 3"):
        main(["--config", str(config), "--results", str(results), "--family", "main"])
    assert not (results / "evaluation").exists()  # nothing published
    # the probe family alone is still paired; the main-family A-05 is then not evaluated
    main(["--config", str(config), "--results", str(results), "--family", "ambulance_probe"])
    a05 = json.loads((results / "evaluation" / "a05.json").read_text(encoding="utf-8"))
    assert a05["primary"]["status"] == "not evaluated" and math.isfinite(
        a05["secondary"][0]["p_value"]
    )


def test_run_missing_reruns_old_kpi_versions(tmp_path, monkeypatch):
    config, results, cfgs = synthetic_setup(tmp_path)
    cfg = cfgs[("main", "S1")]
    kp = synthetic_kpis("S1", cfg)
    write_run(results, cfg, 5, kp, 0.95, kpi_version=1)

    import traffic_twin.sim.run_scenario as rs

    rerun = []

    def fake_run(c, seed, out_root, gui=False, config_path=None):
        rerun.append((c["scenario"], seed))
        write_run(Path(out_root), c, seed, kp, 0.95)

    monkeypatch.setattr(rs, "run", fake_run)
    # seed 5 is a pre-registered evaluation seed: the D92 run lock refuses it into every
    # results folder until the registration is committed, so the test stands in for the
    # clean registration commit (the runner is faked: nothing is launched)
    import traffic_twin.evaluation.evaluate as evaluate_mod

    monkeypatch.setattr(evaluate_mod, "git_head", lambda: ("abc1234", False))
    monkeypatch.setattr(
        evaluate_mod,
        "registration_commit",
        lambda path=None: {"commit": "abc1234", "committed_at": "2026-10-23T18:00:00+02:00"},
    )
    main(["--config", str(config), "--results", str(results), "--family", "main",
          "--run-missing"])  # fmt: skip
    assert rerun == [("pitie_actuated", 5)]


# -- review fixes: incomplete samples (A-03 per seed, A-05 validity, n_pairs) ------------
def _drop(k: pd.DataFrame, user: str, kpi: str, seed: int) -> pd.DataFrame:
    return k[~((k["user_type"] == user) & (k["kpi"] == kpi) & (k["seed"] == seed))]


AMB = ("ambulance", "travel_time_total")


def test_a03_refuses_a_kpi_missing_in_a_single_seed():
    """The union over seeds hid a run without the KPI (an ambulance that never arrived)."""
    k = kpi_table()
    s2 = _drop(k, *AMB, seed=3)
    with pytest.raises(EvaluationError, match=r"S2 seeds \[3\] lacks ambulance\.travel_time_total"):
        check_kpi_sets({"S0": k, "S2": s2})
    nan = k.copy()
    nan.loc[(nan["kpi"] == AMB[1]) & (nan["seed"] == 4), "value"] = np.nan
    with pytest.raises(EvaluationError, match=r"S2 seeds \[4\] lacks ambulance\.travel_time_total"):
        check_kpi_sets({"S0": k, "S2": nan})  # a non-finite value is no value
    no_seed = k[k["seed"] != 10]
    with pytest.raises(EvaluationError, match=r"S2 seeds \[10\] lacks"):
        check_kpi_sets({"S0": k, "S2": no_seed}, SEEDS)


def test_a05_with_a_seed_missing_is_invalid_not_pass():
    """Reproduction of the review: S2 better on 9 seeds, its ambulance row missing in
    seed 3. Before the fix: n 9, p 1/512, PASS."""
    s0 = kpi_table()
    s2 = _drop(kpi_table({AMB: 0.6}), *AMB, seed=3)
    fam = family({"S0": s0, "S2": s2}, {"S0": "fixed", "S2": "rules"})
    spec = {"family": "main", "a": "S0", "b": "S2", "user": "ambulance",
            "kpi": "travel_time_total", "alternative": "less", "alpha": 0.05}  # fmt: skip
    r = a05_test(fam, spec)
    assert r["n"] == 9 and r["expected_n"] == 10 and 3 not in r["seeds"]
    assert r["p_value"] < 0.05  # the 9 survivors alone would pass
    assert not r["valid"] and not r["pass"] and "seeds [3] lack it" in r["problems"][0]
    ecfg = {"a05": {"primary": spec, "secondary": [], "also_report": []}}
    rep = a05_report({"main": fam}, ecfg)
    assert rep["verdict"] == "INVALID" and rep["primary"]["status"] == "INVALID"
    fam.seeds = SEEDS  # the evaluated seeds, not only those the KPIs report
    assert a05_test(fam, spec)["expected_n"] == 10


def test_a05_with_an_ambulance_that_did_not_arrive_is_invalid():
    """The probe KPI is a mean over the ambulances that finished: a run where one of
    the planned ambulances did not arrive must not count."""
    counts = pd.DataFrame({"seed": SEEDS, "user_type": "ambulance", "kpi": "count", "value": 7.0})
    s0 = pd.concat([kpi_table(), counts], ignore_index=True)
    s2 = pd.concat([kpi_table({AMB: 0.6}), counts], ignore_index=True)
    s2.loc[(s2["kpi"] == "count") & (s2["seed"] == 6), "value"] = 6.0
    fam = family({"S0": s0, "S2": s2})
    trips = [{"depart_s": 300 + 900 * i, "route": ["n_in", "s_out"]} for i in range(7)]
    for m in fam.members.values():
        m.cfg["emergency"] = {"enabled": True, "depart_s": None, "route": None, "trips": trips}
    assert planned_trips(fam.members["S2"].cfg, "ambulance") == 7
    spec = {"a": "S0", "b": "S2", "user": "ambulance", "kpi": "travel_time_total",
            "alternative": "less"}  # fmt: skip
    r = a05_test(fam, spec)
    assert r["n"] == 10 and not r["valid"] and not r["pass"]
    assert r["problems"] == ["S2: ambulance trips finished != 7 planned (seed 6: 6)"]
    s2.loc[(s2["kpi"] == "count") & (s2["seed"] == 6), "value"] = 7.0
    fam.kpis["S2"] = s2
    assert a05_test(fam, spec)["pass"]


def test_pair_and_equity_tables_refuse_fewer_pairs_than_seeds(ecfg):
    s0 = kpi_table()
    s2 = _drop(kpi_table(), *AMB, seed=3)
    fam = family({"S0": s0, "S2": s2})
    fam.seeds = SEEDS
    with pytest.raises(EvaluationError, match=r"fewer than the 10 seeds: ambulance\.travel_time"):
        pair_tables(fam, HEADLINE, 0.05, 5.0)
    with pytest.raises(EvaluationError, match="fewer than the 10 seeds"):
        equity_table(fam, ecfg, 0.05)


def test_cli_refuses_a_run_whose_ambulance_never_arrived(tmp_path):
    config, results, cfgs = synthetic_setup(tmp_path)
    cfg = cfgs[("main", "S2")]
    kp = synthetic_kpis("S2", cfg)
    kp = kp[~((kp["user_type"] == "ambulance") & (kp["seed"] == 3))]  # no row at all
    write_run(results, cfg, 3, kp, 0.8)
    with pytest.raises(SystemExit, match=r"S2 seeds \[3\] lacks ambulance\.count, ambulance\."):
        main(["--config", str(config), "--results", str(results), "--family", "main"])


# -- review fixes: equity with a zero baseline -------------------------------------------
def test_equity_flags_a_count_rising_from_a_zero_baseline(ecfg):
    red = ("pedestrian", "crossing_red_entries")

    def with_red(values):
        rows = pd.DataFrame({"seed": SEEDS, "user_type": red[0], "kpi": red[1], "value": values})
        return pd.concat([kpi_table(), rows], ignore_index=True)

    zero = with_red([0.0] * 10)
    one = with_red([0.0] * 9 + [1.0])  # one red entry in ten runs
    eq = equity_table(family({"S0": zero, "S1": zero, "S2": one}), ecfg, 0.05)
    r = eq[eq["kpi"] == red[1]].set_index("scenario")
    assert math.isnan(r.loc["S2", "rel_change_pct"])
    assert r.loc["S2", "abs_change"] == pytest.approx(0.1)
    assert r.loc["S2", "worse_flag"] and not r.loc["S1", "worse_flag"]
    fewer = equity_table(family({"S0": one, "S2": zero}), ecfg, 0.05)
    assert not fewer.set_index("kpi").loc[red[1], "worse_flag"]  # better, not worse


# -- review fixes: generated page ------------------------------------------------------
def _section(page: str, number: int) -> str:
    start = page.index(f"\n## {number}. ")
    end = page.find(f"\n## {number + 1}. ", start)
    return page[start : end if end > 0 else None]


def test_page_sections_follow_the_preregistered_roles_and_config(tmp_path):
    config, results, cfgs = synthetic_setup(tmp_path)
    docs = tmp_path / "evaluation.md"
    main(["--config", str(config), "--results", str(results), "--docs", str(docs)])
    page = docs.read_text(encoding="utf-8")
    s7, s8 = _section(page, 10), _section(page, 11)
    assert "Other confirmatory pairs" in s7 and "role *secondary*" not in s7
    assert "role *confirmatory*" in s7
    assert "**S2_noP** (`pitie_amb_rules_nopreempt`) vs **S2**" in s8  # probe attribution
    assert "role *secondary*" in s8 and "role *exploratory*" in s8

    prereg = _section(page, 5)
    assert "`served_share` by more than 1 pp (no run produces this KPI: not applied)" in prereg
    assert "`crossing_complete_rate` by more than 1 pp, " in prereg  # produced: not marked

    assumed = _section(page, 6)
    s0 = cfgs[("main", "S0")]
    assert f"`main` 1 ambulance per run ({ambulance_plan(s0)[1]})" in assumed
    assert "`ambulance_probe` 7 ambulances per run" in assumed
    assert "bus detection for transit priority is ideal" in assumed
    items = [x for x in assumed.splitlines() if x.startswith("- ASSUMED")]
    assert "pedestrian detection is ideal" in items[0] and "curb wait" in items[0]
    assert "bus detection for transit priority is ideal" in items[1]  # next to pedestrians
    assert "exit blocking for box clearing is read ideally" in items[2]
    a04 = [x for x in prereg.splitlines() if x.startswith("- **A-04**")][0]
    assert "`ped_curb_wait_max` ≤ `safety.ped_max_wait_s` (90 s)" in a04
    assert "`ped_wait_single_max` ≤ 90 s, now reported alongside" in a04
    defs = _section(page, 7)
    for kpi in ("ped_curb_wait_mean", "ped_curb_wait_max"):
        assert f"| `{kpi}` | pedestrian | TraCI per step: curb clock" in defs
    assert "`rules.tsp.enabled` in main/S2, main/S2_noP" in assumed
    assert "exit blocking for box clearing" in assumed and "no bluelight device" in assumed
    assert f"the signal program ({program_text(s0)})" in assumed
    assert "Accès Pitié" not in page
    limits = _section(page, 15)
    assert f"a single ambulance per run ({ambulance_plan(s0)[1]})" in limits


def test_ambulance_plan_and_program_come_from_the_config(monkeypatch):
    from traffic_twin.paths import PITIE_FIXED_CONFIG
    from traffic_twin.sim.tls_check import load_program

    monkeypatch.chdir(REPO_ROOT)  # the net path of the config is relative to the repo
    cfg = load_scenario(PITIE_FIXED_CONFIG)
    begin = float(cfg["begin_s"])

    def clock(depart: float) -> str:
        hh, mm = divmod(int((begin + depart) // 60), 60)
        return f"{hh:02d}:{mm:02d}"

    assert ambulance_plan(cfg) == (1, f"{clock(cfg['emergency']['depart_s'])} N → S")
    moved = copy.deepcopy(cfg)
    moved["emergency"].update(
        depart_s=None,
        route=None,
        trips=[{"depart_s": 600, "route": ["w_in", "s_out"]},
               {"depart_s": 60, "route": ["n_in", "s_out"]}],
    )  # fmt: skip
    assert ambulance_plan(moved) == (2, f"{clock(60)} N → S, {clock(600)} W → S")

    prog = load_program(cfg["network"]["net_file"], cfg["network"]["tls_id"])
    text = program_text(cfg)
    assert text.startswith(f"cycle {prog.cycle:g} s")
    assert text.endswith(" → ".join(s["name"] for s in cfg["network"]["stages"]))
    longer = copy.deepcopy(cfg)
    longer["fixed"]["green_s"]["HOP"] += 10
    assert program_text(longer).startswith(f"cycle {prog.cycle + 10:g} s")
    assert program_text({"network": {}}) is None


# -- review fixes: pre-registration order ----------------------------------------------
def test_preregistration_on_the_day_of_the_runs_is_not_reported_as_before(tmp_path):
    assert prereg_order("2026-10-12", "2026-10-13") == "before"
    assert prereg_order("2026-10-12", "2026-10-12") == "same day"
    assert prereg_order("2026-10-12", "2026-10-11") == "after"
    assert prereg_order("2026-10-12", None) == "n/a"

    config, results, cfgs = synthetic_setup(tmp_path)
    for cfg in cfgs.values():
        for seed in SEEDS:
            run = results / f"{cfg['scenario']}_seed{seed}" / "run.json"
            meta = json.loads(run.read_text(encoding="utf-8"))
            meta["started_at"] = "2026-10-12T12:00:00+00:00"  # the pre-registration day
            run.write_text(json.dumps(meta), encoding="utf-8")
    docs = tmp_path / "evaluation.md"
    main(["--config", str(config), "--results", str(results), "--docs", str(docs)])
    summ = json.loads((results / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    assert summ["preregistration_order"] == "same day"
    assert summ["preregistered_before_runs"] is None
    assert summ["first_run_started_at"] == "2026-10-12T12:00:00+00:00"
    prereg = _section(docs.read_text(encoding="utf-8"), 5)
    assert "same day as the pre-registration" in prereg and "(before the runs)" not in prereg

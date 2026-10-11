"""data.calibrate on a synthetic raw export, plus checks of the versioned artifacts
(data/reference/*.csv, configs/calibration/pitie_302.yaml) that CI runs without
data/raw or data/processed."""

from __future__ import annotations

import copy
import struct
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from traffic_twin.config import ConfigError, load_config
from traffic_twin.data import calibrate as cb
from traffic_twin.data.calendar import classify_days
from traffic_twin.paths import DATA_CONFIG

REPO_ROOT = Path(__file__).resolve().parent.parent

BASE = {"101": 600.0, "102": 350.0, "103": 350.0, "104": 200.0}  # 102 and 103 share a station
DATES = pd.date_range("2026-08-25", "2026-09-17", freq="D").date  # holidays, then term
INCIDENT = date(2026, 9, 9)  # arc 101 doubles that day


def flow(arc: str, d: date, h: int) -> float:
    """Synthetic hourly flow: arc level + 10 veh/h per hour of day + 1 on even days."""
    q = BASE[arc] + 10 * h + (1 if d.day % 2 == 0 else -1)
    return 2 * q if (arc == "101" and d == INCIDENT) else q


def wkb(lon: float, lat: float) -> bytes:
    return struct.pack("<BIdd", 1, 1, lon, lat)


def raw_frame() -> pd.DataFrame:
    rows = []
    for arc in BASE:
        for d in DATES:
            for h in range(4, 13):
                end = pd.Timestamp(f"{d} {h:02d}:00", tz="Europe/Paris") + pd.Timedelta(hours=1)
                rows.append(
                    {
                        "iu_ac": arc, "libelle": f"Rue_{arc}", "t_1h": end,
                        "q": flow(arc, d, h), "k": 5.0, "etat_trafic": "Fluide",
                        "etat_barre": "Ouvert", "iu_nd_amont": "1", "libelle_nd_amont": "a",
                        "iu_nd_aval": "2", "libelle_nd_aval": "b", "geo_point_2d": wkb(2.36, 48.84),
                    }
                )  # fmt: skip
    raw = pd.DataFrame(rows)
    raw["t_1h"] = pd.to_datetime(raw["t_1h"]).dt.tz_convert("UTC")
    at = (raw["iu_ac"] == "104") & (raw["t_1h"] == pd.Timestamp("2026-09-02 07:00", tz="UTC"))
    raw.loc[at, "k"] = np.nan  # valid flow, missing occupancy (hour_start 8 local)
    at = (raw["iu_ac"] == "104") & (raw["t_1h"] == pd.Timestamp("2026-09-03 07:00", tz="UTC"))
    raw.loc[at, "etat_barre"] = "Invalide"  # whole measurement invalid
    gone = (raw["iu_ac"] == "104") & (raw["t_1h"] == pd.Timestamp("2026-09-08 06:00", tz="UTC"))
    return raw[~gone].reset_index(drop=True)  # hour 7 of Tue 8 Sep absent from the export


def data_cfg(tmp_path: Path | None = None) -> dict:
    cfg = {
        "cleaning": {"timezone": "Europe/Paris", "max_flow_veh_h": 4000, "max_occupancy_pct": 100},
        "paths": {"raw": "raw"},
        "calendar": {
            "weekdays": [1, 2, 3],
            "school_holidays": [{"from": date(2026, 7, 4), "to": date(2026, 8, 31)}],
            "public_holidays": [date(2026, 7, 14)],
        },
        "calibration": {
            "site": "test_site", "raw_zone": "zone",
            "reference_csv": "ref/hours.csv", "arcs_csv": "ref/arcs.csv",
            "artifact": "cal/test.yaml", "reference_arcs": ["104", "101", "102", "103"],
            "hours": [5, 11], "window": {"begin": "07:30", "end": "09:30", "slot_s": 900},
            "arms": {"N": {"arc": "101", "status": "proxy", "sensitivity": ["102"]},
                     "S": {"arc": "103"}, "W": {"arc": "104"}},
            "mad_k": 3.5, "min_days": 5, "min_common_hours": 50,
            "turn_prior": {"N": {"S": 0.6, "W": 0.4}, "S": {"N": 0.9, "W": 0.1},
                           "W": {"N": 0.8, "S": 0.2}},
            "field_counts": "field.yaml",
        },
    }  # fmt: skip
    return cfg


@pytest.fixture(scope="module")
def hours() -> pd.DataFrame:
    return cb.extract_hours(raw_frame(), data_cfg())


# -- extract_hours ---------------------------------------------------------------------
def test_extract_hours_schema_and_grid(hours):
    assert list(hours.columns) == cb.REFERENCE_COLUMNS
    assert len(hours) == 4 * len(DATES) * 7  # every arc x date x hour_start 5..11
    assert hours["iu_ac"].unique().tolist() == ["101", "102", "103", "104"]  # numeric order
    key = hours[["iu_ac", "date", "hour_start"]]
    assert not key.duplicated().any()
    assert key.equals(key.sort_values(["iu_ac", "date", "hour_start"]).reset_index(drop=True))
    assert set(hours["hour_start"]) == set(range(5, 12))


def test_extract_hours_shifts_stamps_and_types_days(hours):
    row = hours[(hours["iu_ac"] == "101") & (hours["date"] == date(2026, 9, 1))]
    row = row.set_index("hour_start")
    assert row.loc[7, "ts_end"] == "2026-09-01T08:00:00+02:00"  # stamp 08:00 = 07:00-08:00
    assert row.loc[7, "flow_veh_h"] == flow("101", date(2026, 9, 1), 7)
    assert row.loc[7, "weekday"] == 1
    types = hours.drop_duplicates("date").set_index("date")["day_type"]
    assert types[date(2026, 8, 25)] == "school_holiday"
    assert types[date(2026, 9, 1)] == "term"
    assert types[date(2026, 9, 5)] == "weekend"


def test_extract_hours_validity_flags(hours):
    a = hours[hours["iu_ac"] == "104"].set_index(["date", "hour_start"])
    nan_occ = a.loc[(date(2026, 9, 2), 8)]
    assert nan_occ["flow_valid"] and not nan_occ["occ_valid"]
    assert nan_occ["flow_veh_h"] == flow("104", date(2026, 9, 2), 8)
    assert np.isnan(nan_occ["occupancy_pct"])
    invalid = a.loc[(date(2026, 9, 3), 8)]
    assert not invalid["flow_valid"] and np.isnan(invalid["flow_veh_h"])
    assert invalid["etat_barre"] == "Invalide"
    absent = a.loc[(date(2026, 9, 8), 7)]
    assert not absent["flow_valid"] and pd.isna(absent["ts_end"]) and pd.isna(absent["etat_barre"])
    assert absent["libelle"] == "Rue_104"


def test_reference_csv_round_trip(hours, tmp_path):
    path = cb.write_hours(hours, tmp_path / "hours.csv")
    assert b"\r\n" not in path.read_bytes()
    back = cb.read_hours(path)
    pd.testing.assert_frame_equal(back, hours, check_dtype=False)
    assert cb.text_sha256(path) == cb.text_sha256(path)
    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert cb.text_sha256(crlf) == cb.text_sha256(path)  # same hash after a CRLF checkout
    assert cb.text_sha256(tmp_path / "absent.csv") is None


# -- shared stations -------------------------------------------------------------------
def test_station_groups_pairs_identical_arcs(hours):
    groups = cb.station_groups(hours, min_common=50)
    assert groups == {"101": "101", "102": "102=103", "103": "102=103", "104": "104"}
    # not enough common hours to tell: every arc on its own
    lone = cb.station_groups(hours, min_common=10_000)
    assert set(lone.values()) == {"101", "102", "103", "104"}


def test_arc_table(tmp_path):
    arcs = cb.arc_table(raw_frame(), data_cfg())
    assert list(arcs.columns) == cb.ARC_COLUMNS
    a = arcs.set_index("iu_ac")
    assert a.loc["102", "station_group"] == "102=103"
    assert not a.loc["102", "own_flow_station"] and a.loc["101", "own_flow_station"]
    assert a.loc["101", "lat"] == 48.84 and a.loc["101", "lon"] == 2.36
    expect = len(DATES) * 9 / (len(DATES) * 24 - 15)  # periods from the first to the last start
    assert a.loc["101", "coverage"] == pytest.approx(expect, abs=1e-4)
    assert a["coverage"].max() <= 1


# -- calibration -----------------------------------------------------------------------
def expected_days(arc: str) -> list[date]:
    days = classify_days(DATES, data_cfg()["calendar"])
    typical = list(days.loc[days["typical"], "date"])
    if arc == "101":
        typical.remove(INCIDENT)  # dropped by the MAD filter
    if arc == "104":
        typical = [d for d in typical if d not in (date(2026, 9, 3), date(2026, 9, 8))]
    return typical


def test_slot_flows_match_hand_computation(hours):
    cal = cb.build_calibration(hours, data_cfg())
    for arm, arc in (("N", "101"), ("S", "103"), ("W", "104")):
        days = expected_days(arc)
        a = cal["arms"][arm]
        assert a["arc"] == arc and a["n_days"] == len(days)
        assert cal["day_filter"]["days_used"][arc] == [d.isoformat() for d in days]
        hourly = {h: np.mean([flow(arc, d, h) for d in days]) for h in (7, 8, 9)}
        expect = [hourly[7]] * 2 + [hourly[8]] * 4 + [hourly[9]] * 2
        assert [s["begin_s"] for s in a["slots"]] == list(range(27000, 34200, 900))
        assert [s["flow_veh_h"] for s in a["slots"]] == pytest.approx(expect, abs=1e-3)
        window = 0.25 * hourly[7] + 0.5 * hourly[8] + 0.25 * hourly[9]
        assert a["mean_veh_h"] == pytest.approx(window, abs=1e-3)
    assert cal["arms"]["N"]["status"] == "proxy" and "status" not in cal["arms"]["S"]
    assert cal["day_filter"]["mad_dropped"] == {"101": [INCIDENT.isoformat()]}
    assert cal["sensitivity"]["N"]["arc"] == "102"
    assert cal["window"] == {"begin_s": 27000, "end_s": 34200, "slot_s": 900}
    assert cal["station_groups"]["103"] == "102=103"
    assert cal["source"]["raw_period"] == ["2026-08-25", "2026-09-17"]
    assert cal["source"]["source_sha256"] is None  # ref/hours.csv does not exist here
    assert cal["day_filter"]["typical_period"] is None
    assert cal["day_filter"]["hold_out_days"] == {}


def test_days_outside_the_typical_period_are_held_out(hours):
    """Typical days outside calendar.typical_period (e.g. spring days after a 12-month
    re-download) are listed as hold-out days and never enter the slot flows."""
    cfg = data_cfg()
    cfg["calendar"]["typical_period"] = {"from": date(2026, 9, 8), "to": date(2026, 12, 18)}
    cal = cb.build_calibration(hours, cfg)
    early = [d for d in expected_days("103") if d < date(2026, 9, 8)]
    late = [d for d in expected_days("103") if d >= date(2026, 9, 8)]
    assert early and late
    f = cal["day_filter"]
    assert f["typical_period"] == ["2026-09-08", "2026-12-18"]
    assert f["days_used"]["103"] == [d.isoformat() for d in late]
    assert f["hold_out_days"]["103"] == [d.isoformat() for d in early]
    hourly = {h: np.mean([flow("103", d, h) for d in late]) for h in (7, 8, 9)}
    window = 0.25 * hourly[7] + 0.5 * hourly[8] + 0.25 * hourly[9]
    assert cal["arms"]["S"]["mean_veh_h"] == pytest.approx(window, abs=1e-3)


def test_turns_are_the_prior_without_counts(hours):
    turns = cb.build_calibration(hours, data_cfg())["turns"]
    assert turns == {"source": "assumption", "N": {"S": 0.6, "W": 0.4},
                     "S": {"N": 0.9, "W": 0.1}, "W": {"N": 0.8, "S": 0.2}}  # fmt: skip
    template = {"counts": [{"movements": {"N_S": None, "N_W": None, "W_N": None}}]}
    assert cb.build_calibration(hours, data_cfg(), template)["turns"] == turns


def test_turns_are_the_counted_shares_with_counts(hours):
    first = {"N_S": 30, "N_W": 10, "S_N": 18, "S_W": 2, "W_N": 7, "W_S": 3}
    second = {"N_S": 30, "N_W": 30, "S_N": 18, "S_W": 2, "W_N": 7, "W_S": None}
    counts = [{"begin": "08:00", "movements": first}, {"begin": "08:15", "movements": second}]
    assert cb.turns_from_counts(counts) == {
        "N": {"S": 0.6, "W": 0.4},
        "S": {"N": 0.9, "W": 0.1},
        "W": {"N": 0.7, "S": 0.3},  # second bin skipped for W (one movement not counted)
    }
    turns = cb.build_calibration(hours, data_cfg(), {"counts": counts})["turns"]
    assert turns["source"] == "field" and turns["W"] == {"N": 0.7, "S": 0.3}
    partial = [{"movements": {"N_S": 9, "N_W": 1, "S_N": None, "S_W": None}}]
    turns = cb.build_calibration(hours, data_cfg(), {"counts": partial})["turns"]
    assert turns["source"] == "mixed" and turns["field_arms"] == ["N"]
    assert turns["N"] == {"S": 0.9, "W": 0.1} and turns["S"] == {"N": 0.9, "W": 0.1}


def test_bad_counts_and_priors_raise(hours):
    with pytest.raises(ConfigError, match="not <from>_<to>"):
        cb.turns_from_counts([{"movements": {"NS": 3}}])
    with pytest.raises(ConfigError, match="negative"):
        cb.turns_from_counts([{"movements": {"N_S": -1}}])
    with pytest.raises(ConfigError, match="unknown approach"):
        cb.build_calibration(hours, data_cfg(), {"counts": [{"movements": {"E_N": 3}}]})
    cfg = data_cfg()
    cfg["calibration"]["turn_prior"]["N"] = {"S": 0.6, "W": 0.6}
    with pytest.raises(ConfigError, match="sum to 1"):
        cb.build_calibration(hours, cfg)
    cfg = data_cfg()
    cfg["calibration"]["turn_prior"]["N"] = {"S": 0.5, "E": 0.5}
    with pytest.raises(ConfigError, match="unknown destination E"):
        cb.build_calibration(hours, cfg)


def test_too_few_days_raises(hours):
    cfg = data_cfg()
    cfg["calibration"]["min_days"] = 30
    with pytest.raises(ValueError, match="min_days"):
        cb.build_calibration(hours, cfg)


def test_calibration_file_round_trips(hours, tmp_path):
    cal = cb.build_calibration(hours, data_cfg())
    path = cb.write_calibration(cal, tmp_path / "cal.yaml")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Demand calibration") and "GENERATED" in text
    assert yaml.safe_load(text) == cal


def test_cli_from_raw_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "raw").mkdir()
    raw_frame().to_parquet(tmp_path / "raw" / "zone.parquet")
    cfg = data_cfg()
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    outputs = [Path("ref/hours.csv"), Path("ref/arcs.csv"), Path("cal/test.yaml")]

    cb.main(["--config", "data.yaml", "--from-raw"])
    first = [p.read_bytes() for p in outputs]
    cb.main(["--config", "data.yaml", "--from-raw"])
    assert [p.read_bytes() for p in outputs] == first
    cal = yaml.safe_load(first[2])
    assert cal["source"]["source_sha256"] == cb.text_sha256("ref/hours.csv")
    assert cal == cb.build_calibration(cb.read_hours("ref/hours.csv"), cfg)

    # without --from-raw only the calibration file is rebuilt, here from field counts
    counts = [{"movements": {"N_S": 1, "N_W": 1, "S_N": 1, "S_W": 1, "W_N": 1, "W_S": 1}}]
    Path("field.yaml").write_text(yaml.safe_dump({"counts": counts}), encoding="utf-8")
    Path("raw/zone.parquet").unlink()
    cb.main(["--config", "data.yaml"])
    assert yaml.safe_load(Path("cal/test.yaml").read_text(encoding="utf-8"))["turns"]["source"] == (
        "field"
    )
    with pytest.raises(SystemExit, match="missing"):
        cb.main(["--config", "data.yaml", "--from-raw"])


# -- versioned artifacts (what CI checks) ----------------------------------------------
@pytest.fixture
def repo_cfg(monkeypatch) -> dict:
    monkeypatch.chdir(REPO_ROOT)
    return load_config(DATA_CONFIG)


def test_committed_calibration_matches_the_reference_csv(repo_cfg):
    """`configs/calibration/pitie_302.yaml` is exactly what `data.calibrate` builds from
    the versioned reference CSV (regenerate it with `tasks.py calib-data`)."""
    c = repo_cfg["calibration"]
    built = cb.build_calibration(
        cb.read_hours(c["reference_csv"]), repo_cfg, cb.load_field(repo_cfg)
    )
    committed = yaml.safe_load(Path(c["artifact"]).read_text(encoding="utf-8"))
    assert built == committed


def test_committed_calibration_levels(repo_cfg):
    cal = yaml.safe_load(Path(repo_cfg["calibration"]["artifact"]).read_text(encoding="utf-8"))
    arms = cal["arms"]
    assert {a: arms[a]["arc"] for a in arms} == {"N": "508", "S": "511", "W": "513"}  # D6
    assert 600 <= arms["N"]["mean_veh_h"] <= 650
    assert 350 <= arms["S"]["mean_veh_h"] <= 380
    assert 195 <= arms["W"]["mean_veh_h"] <= 215
    for a in arms.values():
        assert 16 <= a["n_days"] <= 18
        assert len(a["slots"]) == 8
        mean = np.mean([s["flow_veh_h"] for s in a["slots"]])
        assert mean == pytest.approx(a["mean_veh_h"], abs=1e-2)
    assert cal["sensitivity"]["N"]["arc"] == "510"
    assert cal["sensitivity"]["N"]["mean_veh_h"] == pytest.approx(1310, rel=0.03)
    # calibration days come from the autumn term only (calendar.typical_period)
    lo, hi = cal["day_filter"]["typical_period"]
    for days in cal["day_filter"]["days_used"].values():
        assert all(lo <= d <= hi for d in days)
    assert cal["timestamp_convention"] == "period_end"
    assert cal["window"] == {"begin_s": 27000, "end_s": 34200, "slot_s": 900}
    assert cal["station_groups"]["511"] == cal["station_groups"]["509"] == "509=511=5071"
    assert cal["station_groups"]["510"] == "505=510"
    if cal["turns"]["source"] == "assumption":  # D7 until the video count
        assert cal["turns"]["N"] == {"S": 0.63, "W": 0.37}


def test_committed_reference_csv(repo_cfg):
    c = repo_cfg["calibration"]
    hours = cb.read_hours(c["reference_csv"])
    assert list(hours.columns) == cb.REFERENCE_COLUMNS
    assert sorted(hours["iu_ac"].unique(), key=int) == sorted(c["reference_arcs"], key=int)
    assert set(hours["hour_start"]) == set(range(c["hours"][0], c["hours"][1] + 1))
    n_dates = hours["date"].nunique()
    assert len(hours) == n_dates * len(c["reference_arcs"]) * 7
    assert set(hours["day_type"]) <= {"term", "school_holiday", "public_holiday", "weekend"}
    assert not hours.loc[~hours["flow_valid"], "flow_veh_h"].notna().any()
    days = classify_days(hours["date"].unique(), repo_cfg["calendar"]).set_index("date")
    assert (hours["day_type"] == hours["date"].map(days["day_type"])).all()


def test_committed_arc_table(repo_cfg):
    arcs = pd.read_csv(repo_cfg["calibration"]["arcs_csv"], dtype={"iu_ac": str})
    assert list(arcs.columns) == cb.ARC_COLUMNS
    a = arcs.set_index("iu_ac")
    assert set(repo_cfg["calibration"]["reference_arcs"]) <= set(a.index)
    assert a.loc["508", "own_flow_station"] and not a.loc["510", "own_flow_station"]
    assert a.loc["510", "station_group"] == a.loc["505", "station_group"]
    inbound = a[a["iu_nd_aval"].astype(str) == "302"].index
    assert set(inbound) == {"510", "511", "513"}  # topology of node 302


def test_repo_data_config_has_the_calibration_keys():
    cfg = copy.deepcopy(load_config(REPO_ROOT / DATA_CONFIG))
    cb._calib(cfg)
    del cfg["calibration"]["arms"]
    with pytest.raises(ConfigError, match="calibration.arms"):
        cb._calib(cfg)

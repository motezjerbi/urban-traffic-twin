"""Day-type calibration files (D44): the generated files are up to date, the ratio
method is the documented one and the base type reproduces the base calibration."""

import copy
import csv

import pytest
import yaml

from traffic_twin.config import load_config, load_scenario
from traffic_twin.ml import daytypes
from traffic_twin.ml.daytypes import (
    BASE_DAY_TYPE,
    DAY_TYPES,
    HOURS_CSV,
    day_type_calibration,
    day_type_ratios,
    window_hours,
)
from traffic_twin.paths import ETA_CONFIG, PITIE_FIXED_CONFIG
from traffic_twin.sim.demand import build_routes

BASE = "configs/calibration/pitie_302.yaml"


@pytest.fixture(scope="module")
def base() -> dict:
    return load_config(BASE)


@pytest.fixture(scope="module")
def ratios(base) -> dict:
    return day_type_ratios(HOURS_CSV, base)


def test_eta_config_names_the_three_files():
    files = daytypes.calibration_files(load_config(ETA_CONFIG))
    assert files[BASE_DAY_TYPE].as_posix() == BASE
    assert set(files) == set(DAY_TYPES)


def test_generated_files_are_up_to_date(capsys):
    assert daytypes.main(["--check"]) == 0
    assert capsys.readouterr().out.count("up to date") == 2


def test_ratio_one_reproduces_the_base_slot_flows(base, ratios):
    assert set(ratios) == set(DAY_TYPES)
    assert all(r == 1.0 for arm in ratios[BASE_DAY_TYPE].values() for r in arm.values())
    same = day_type_calibration(base, BASE_DAY_TYPE, ratios)
    assert same["arms"] == base["arms"]
    assert same["window"] == base["window"] and same["turns"] == base["turns"]
    assert base == load_config(BASE)  # the base mapping is not modified


def hand_ratio(day_type: str, arc: str, hour: int, base_days: list[str]) -> float:
    """Mean valid flow of the day type's dates / mean over the base days, read with csv."""
    weekdays = {
        "mon_fri_term": ("term", {0, 4}),
        "school_holiday": ("school_holiday", set(range(5))),
    }
    kind, days = weekdays[day_type]
    num, den = [], []
    with open(HOURS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["iu_ac"] != arc or int(row["hour_start"]) != hour:
                continue
            if row["flow_valid"] != "True" or row["flow_veh_h"] == "":
                continue
            if row["day_type"] == kind and int(row["weekday"]) in days:
                num.append(float(row["flow_veh_h"]))
            if row["date"] in base_days:
                den.append(float(row["flow_veh_h"]))
    return (sum(num) / len(num)) / (sum(den) / len(den))


@pytest.mark.parametrize("day_type", ["mon_fri_term", "school_holiday"])
def test_ratio_of_arc_508_hour_8_matches_a_hand_computation(base, ratios, day_type):
    assert base["arms"]["N"]["arc"] == "508"
    expected = hand_ratio(day_type, "508", 8, base["day_filter"]["days_used"]["508"])
    assert ratios[day_type]["N"][8] == pytest.approx(expected, abs=5e-5)
    assert 0.3 < expected < 1.5  # a plausible ratio, not a unit slip


@pytest.mark.parametrize("day_type", ["mon_fri_term", "school_holiday"])
def test_generated_slots_are_base_slots_times_the_hour_ratio(base, ratios, day_type):
    gen = load_config(f"configs/calibration/pitie_302_{day_type}.yaml")
    assert window_hours(base) == [7, 8, 9]
    assert gen["day_type"]["name"] == day_type and gen["sensitivity"] == {}
    for arm, a in base["arms"].items():
        for slot, new in zip(a["slots"], gen["arms"][arm]["slots"], strict=True):
            hour = slot["begin_s"] // 3600  # 900 s slots inside one sensor hour
            want = round(slot["flow_veh_h"] * ratios[day_type][arm][hour], 3)
            assert new == {"begin_s": slot["begin_s"], "flow_veh_h": want}
    assert gen["turns"] == base["turns"] and gen["window"] == base["window"]


@pytest.mark.parametrize("day_type", ["mon_fri_term", "school_holiday"])
def test_a_scenario_runs_its_demand_from_a_day_type_file(day_type):
    cfg = copy.deepcopy(load_scenario(PITIE_FIXED_CONFIG))
    cfg["demand"]["calibration"] = f"configs/calibration/pitie_302_{day_type}.yaml"
    assert build_routes(cfg) != build_routes(load_scenario(PITIE_FIXED_CONFIG))


def test_check_tolerates_a_last_digit_difference():
    """A last-ulp numpy difference may flip one rounded ratio (and the flows scaled by it);
    the check accepts that, but not a real change or an edited header."""
    files = daytypes.calibration_files(load_config(ETA_CONFIG))
    text = files["mon_fri_term"].read_bytes().decode("utf-8").replace("\r\n", "\n")
    data = yaml.safe_load(text)
    arm = next(iter(data["day_type"]["ratios"]))
    hour = next(iter(data["day_type"]["ratios"][arm]))
    r = data["day_type"]["ratios"][arm][hour]
    slot = data["arms"][arm]["slots"][0]
    header = "".join(ln + "\n" for ln in text.splitlines() if ln.startswith("#"))

    def dump(d):
        return header + yaml.safe_dump(d, sort_keys=False, allow_unicode=True,
                                       default_flow_style=None, width=100)  # fmt: skip

    near = copy.deepcopy(data)
    near["day_type"]["ratios"][arm][hour] = round(r + 1e-4, 4)
    near["arms"][arm]["slots"][0]["flow_veh_h"] = round(slot["flow_veh_h"] * (1 + 1e-4), 3)
    assert daytypes.same_calibration(dump(near), dump(data))
    far = copy.deepcopy(data)
    far["day_type"]["ratios"][arm][hour] = round(r * 1.01, 4)
    assert not daytypes.same_calibration(dump(far), dump(data))
    assert not daytypes.same_calibration(dump(data).replace("GENERATED", "EDITED"), dump(data))

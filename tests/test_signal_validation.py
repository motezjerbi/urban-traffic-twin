"""Signal validation on the provisional PitiÃ© program v7 (cycle 117 s), from a synthetic
switch table: cycle, head greens with their trailing clearance, yellow and all-red from
the program, a head green twice per cycle (ped_NB) validated per stage, and the row
statuses against field values."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traffic_twin.config import ConfigError
from traffic_twin.validation.field import FieldData
from traffic_twin.validation.signal import (
    group_intervals,
    head_stages,
    link_intervals,
    model_durations,
    ped_only_stage,
    program_items,
    read_program,
    signal_table,
    sim_items,
)

# Lead sketch v7: 17 phases, links 0-11 vehicles, 12-15 crossings (NB, S, WB, W)
V7 = [
    (32, "rrGrrrGGrrrrGrGG"), (1, "rrGrrrGGrrrrrrGr"), (3, "rryrrrGGrrrrrrGr"),
    (5, "rrrrrrGGrrrrrrGr"), (6, "rrrrrGGGGrrrrrGr"), (3, "rrrrryyyyrrrrrGr"),
    (7, "rrrrrrrrrrrrrrGr"), (18, "GGrrrrrrrrgGGGGr"), (3, "GGrrrrrrrryyrGrr"),
    (4, "GGrrrrrrrrrrrGrr"), (6, "GGrGrrrrrGrrrGrr"), (2, "GGrGrrrrrGrrrrrr"),
    (3, "GGrGrrrrryrrrrrr"), (7, "GGrGrrrrrrrrrrrr"), (6, "GGrGGrrrrrrrrrrr"),
    (3, "yyryyrrrrrrrrrrr"), (8, "rrrrrrrrrrrrrrrr"),
]  # fmt: skip
DURATIONS = [float(d) for d, _ in V7]
STATES = [s for _, s in V7]
STAGES = [
    {"name": "HOP", "green_phase": 0, "clearance_phases": [1, 2, 3]},
    {"name": "S_LEFT", "green_phase": 4, "clearance_phases": [5, 6]},
    {"name": "STM", "green_phase": 7, "clearance_phases": [8, 9]},
    {"name": "BUS_91", "green_phase": 10, "clearance_phases": [11, 12, 13]},
    {"name": "BUS_57", "green_phase": 14, "clearance_phases": [15, 16]},
]
GROUPS = {
    "N_right": [0, 1],
    "N_straight": [2],
    "bus_91_W": [3],
    "bus_57_S": [4],
    "bus_57_N": [5],
    "S_straight": [6, 7],
    "S_left": [8],
    "bus_91_E": [9],
    "W": [10, 11],
    "ped_NB": [12],
    "ped_S": [13],
    "ped_WB": [14],
    "ped_W": [15],
}
PEDS = {12, 13, 14, 15}
CFG = {
    "controller": "fixed",
    "network": {
        "stages": STAGES,
        "signal_groups": GROUPS,
        "reference_group": "N_straight",
        "crossings": {"NB": {"link": 12}, "S": {"link": 13}, "WB": {"link": 14}, "W": {"link": 15}},
    },
}
THRESHOLDS = {"cycle_s": 2.0, "green_s": 2.0, "yellow_s": 0.5, "all_red_s": 0.5}


def switch_table(seeds=(1, 2), n_cycles=60, durations=DURATIONS) -> pd.DataFrame:
    """What signal.parquet holds for fixed-time runs of v7: green intervals per link."""
    one = link_intervals(STATES, durations, n_cycles=n_cycles)
    return pd.concat([one.assign(seed=s) for s in seeds], ignore_index=True)


TAGS = head_stages(STATES, DURATIONS, GROUPS, STAGES, "N_straight")


def sim(durations=DURATIONS):
    table = switch_table(durations=durations)
    return sim_items(table, GROUPS, "N_straight", duration_s=7200, stage_tags=TAGS)


def model(durations=DURATIONS):
    return program_items(STATES, durations, GROUPS, PEDS, STAGES, "N_straight")


def field(signal: dict | None = None) -> FieldData:
    raw = {"schema_version": 1, "signal": signal} if signal is not None else {}
    return FieldData(Path("f.yaml"), signal is not None, raw)


def test_v7_fixed_plan_has_a_117_s_cycle_with_no_spread():
    s = sim()
    cyc = s[s["item"] == "cycle_s"]["value"]
    assert len(cyc) > 100 and cyc.mean() == pytest.approx(117.0) and cyc.std() == 0


def test_vehicle_green_includes_the_trailing_clearance_phases():
    g = sim().groupby("item")["value"].mean()
    assert g["N_straight.green_s"] == 33  # HOP 32 + 1 s while the NB / W crossings clear
    assert g["S_straight.green_s"] == 47  # green through HOP, its clearance and S_LEFT
    assert g["N_right.green_s"] == 49  # STM + BUS_91 + BUS_57 and their clearances
    assert g["W.green_s"] == 18
    assert g["ped_W.green_s"] == 32  # pedestrian green = the green phase only
    assert g["ped_WB.green_s"] == 75
    # ped_NB is green twice per cycle: each green on its own, never their 25 s mean
    assert "ped_NB.green_s" not in g
    assert (g["ped_NB.green_s@HOP"], g["ped_NB.green_s@STM"]) == (32, 18)


def test_heads_green_twice_per_cycle_are_tagged_by_stage():
    assert TAGS["ped_NB"] == ["HOP", "STM"]
    assert TAGS["N_straight"] == ["HOP"] and TAGS["W"] == ["STM"] and TAGS["ped_WB"] == ["HOP"]
    assert all(len(t) == 1 for g, t in TAGS.items() if g != "ped_NB")
    m = model()
    assert (m["ped_NB.green_s@HOP"], m["ped_NB.green_s@STM"]) == (32, 18)
    assert "ped_NB.green_s" not in m and m["ped_W.green_s"] == 32
    # without stages (older callers): one pooled item, as before
    assert program_items(STATES, DURATIONS, GROUPS, PEDS)["ped_NB.green_s"] == 25
    s = sim()
    n = s[s["item"] == "ped_NB.green_s@STM"].groupby("seed").size()
    assert n.tolist() == [59, 59]  # one per complete cycle of the reference head


def test_ped_nb_36_20_split_is_compared_per_stage():
    """The 123 s program (HOP 36, STM 20): a field sheet timing each green under its stage
    passes; their mean (28 s) or total (56 s) as a plain list is refused (it cannot be
    matched to a stage); swapped greens fail instead of passing on the mean."""
    durations = model_durations(DURATIONS, STAGES, {"HOP": 36, "STM": 20})
    m, s = model(durations), sim(durations)
    assert (m["ped_NB.green_s@HOP"], m["ped_NB.green_s@STM"]) == (36, 20)

    def rows(green):
        heads = {"ped_NB": {"green_s": green}}
        t = signal_table(CFG, field({"source": "field", "heads": heads}), s, m, THRESHOLDS)
        return _rows(t)

    ok = rows({"HOP": [36, 35, 36, 37, 36], "STM": [20, 21, 20, 20, 19]})
    assert ok.loc["ped_NB.green_s@HOP", "status"] == "PASS"
    assert ok.loc["ped_NB.green_s@STM", "status"] == "PASS"
    assert ok.loc["ped_NB.green_s@STM", "sim_mean"] == 20 and "ped_NB.green_s" not in ok.index
    swapped = rows({"HOP": [20] * 5, "STM": [36] * 5})
    assert set(swapped.loc[["ped_NB.green_s@HOP", "ped_NB.green_s@STM"], "status"]) == {"FAIL"}
    for plain in ([28] * 5, [56] * 5, [36, 20] * 5):
        with pytest.raises(ConfigError, match=r"ped_NB green 2 times per cycle \(HOP, STM\)"):
            rows(plain)
    partial = rows({"HOP": [36], "STM": [], "PED": [12]})  # a stage the model lacks
    assert partial.loc["ped_NB.green_s@STM", "status"] == "NOT_MEASURED"
    assert partial.loc["ped_NB.green_s@PED", "status"] == "NOT_IN_SIM"


def test_single_green_head_measured_twice_per_cycle_is_not_in_sim():
    m = model()
    heads = {"W": {"green_s": {"STM": [18, 18]}}, "ped_S": {"green_s": {"STM": [30], "X": [9]}}}
    r = _rows(signal_table(CFG, field({"heads": heads}), sim(), m, THRESHOLDS))
    assert r.loc["W.green_s", "status"] == "PASS"  # one stage: compared as the head's green
    assert r.loc["ped_S.green_s", "status"] == "NOT_MEASURED"
    assert r.loc["ped_S.green_s@STM", "status"] == "NOT_IN_SIM"
    assert r.loc["ped_S.green_s@X", "status"] == "NOT_IN_SIM"


def test_program_items_yellow_and_all_red():
    m = program_items(STATES, DURATIONS, GROUPS, PEDS)
    assert m["cycle_s"] == 117
    assert m["N_straight.green_s"] == 33 and m["N_straight.yellow_s"] == 3
    assert m["N_straight.all_red_after_s"] == 5  # until S_LEFT (links 5, 8) turns green
    assert m["S_straight.all_red_after_s"] == 7
    assert m["W.all_red_after_s"] == 4
    assert m["N_right.all_red_after_s"] == 8
    assert (m["bus_91_E.green_s"], m["bus_91_E.yellow_s"], m["bus_91_E.all_red_after_s"]) == (
        8,
        3,
        7,
    )
    assert "ped_S.yellow_s" not in m
    assert not ped_only_stage(STATES, STAGES, PEDS)
    assert ped_only_stage(["rrrrG", "GGGGr"], [{"green_phase": 0}], {4})


def test_model_durations_use_the_fixed_greens():
    d = model_durations(DURATIONS, STAGES, {"HOP": 40, "STM": 20})
    assert d[0] == 40 and d[7] == 20 and d[4] == 6 and sum(d) == 117 + 8 + 2


def test_group_interval_is_the_union_of_its_links():
    iv = pd.DataFrame({"link": [0, 1, 1], "begin_s": [0.0, 5.0, 30.0], "end_s": [10.0, 12.0, 40.0]})
    g = group_intervals(iv, [0, 1])
    assert g[["begin_s", "end_s"]].values.tolist() == [[0.0, 12.0], [30.0, 40.0]]


def test_read_program_from_a_net(tmp_path):
    net = tmp_path / "x.net.xml"
    phases = "".join(f'<phase duration="{d}" state="{s}"/>' for d, s in V7[:3])
    net.write_text(f'<net><tlLogic id="C" type="static">{phases}</tlLogic></net>', "utf-8")
    states, durations = read_program(net, "C")
    assert states == STATES[:3] and durations == [32.0, 1.0, 3.0]


def _rows(table: pd.DataFrame) -> pd.DataFrame:
    return table.set_index("item")


def test_without_field_data_every_row_is_not_measured():
    t = signal_table(CFG, field(), sim(), model(), THRESHOLDS, ped_only=False)
    assert set(t["status"]) == {"NOT_MEASURED"}
    r = _rows(t)
    assert r.loc["cycle_s", "sim_mean"] == 117 and r.loc["cycle_s", "model_value"] == 117
    assert r.loc["N_straight.yellow_s", "sim_mean"] == 3  # replayed from the program
    assert r.loc["stage_order", "model_value"] == "HOP > S_LEFT > STM > BUS_91 > BUS_57"
    assert {"ped_S.flashing_s", "W.all_red_after_s"} <= set(r.index)
    assert {"ped_NB.green_s@HOP", "ped_NB.green_s@STM"} <= set(r.index)


def test_statuses_against_field_values():
    model = program_items(STATES, DURATIONS, GROUPS, PEDS)
    sig = {
        "source": "field",
        "cycle_s": [117, 118, 116, 117, 119],
        "stage_order": ["HOP", "S_LEFT", "STM", "BUS_91", "BUS_57"],
        "varies_between_cycles": False,
        "heads": {
            "N_straight": {"green_s": [34, 33], "yellow_s": 3, "all_red_after_s": 6},
            "W": {"green_s": [25]},
            "ped_S": {"green_s": [30], "flashing_s": 4},
        },
    }
    r = _rows(signal_table(CFG, field(sig), sim(), model, THRESHOLDS, ped_only=False))
    assert r.loc["cycle_s", "status"] == "PASS" and r.loc["cycle_s", "field_n"] == 5
    assert r.loc["N_straight.green_s", "status"] == "PASS"  # 33 vs 33.5
    assert r.loc["N_straight.yellow_s", "status"] == "PASS"
    assert r.loc["N_straight.all_red_after_s", "status"] == "FAIL"  # 5 vs 6, tolerance 0.5 s
    assert r.loc["W.green_s", "status"] == "FAIL"  # 18 vs 25
    assert r.loc["W.green_s", "abs_err"] == pytest.approx(-7)
    assert r.loc["ped_S.green_s", "status"] == "PASS"  # 31 vs 30
    assert r.loc["ped_S.flashing_s", "status"] == "NOT_IN_SIM"
    assert r.loc["stage_order", "status"] == "PASS"
    assert r.loc["varies_between_cycles", "status"] == "PASS"
    assert r.loc["S_left.green_s", "status"] == "NOT_MEASURED"

    new_stage = {**sig, "stage_order": ["HOP", "PED", "STM"]}
    r = _rows(signal_table(CFG, field(new_stage), sim(), model, THRESHOLDS))
    assert r.loc["stage_order", "status"] == "NOT_IN_SIM"

    assumed = _rows(
        signal_table(CFG, field({**sig, "source": "assumption"}), sim(), model, THRESHOLDS)
    )
    assert assumed.loc["W.green_s", "status"] == "ASSUMED"
    actuated = _rows(
        signal_table({**CFG, "controller": "actuated"}, field(sig), sim(), model, THRESHOLDS)
    )
    assert actuated.loc["W.green_s", "status"] == "INFO"  # S1 is descriptive only
    assert actuated.loc["ped_S.flashing_s", "status"] == "NOT_IN_SIM"

"""RL observation `rl-obs-1` (Part 6, D57): 74 names at Pitié, every value in [0, 1] for
extreme synthetic snapshots, exclusive one-hots; the Part 7 feature groups and
`observation.drop` (D73 V1: 62 names without the pedestrian block). No SUMO."""

import math

import numpy as np
import pytest

from traffic_twin.control.base import BusCall, EmergencyCall
from traffic_twin.rl.features import (
    GROUPS,
    OBS_VERSION,
    SiteSpec,
    check_drop,
    feature_names,
    group_names,
    group_of,
    observation,
    site_spec_from_config,
)
from traffic_twin.sim.incidents import ActiveIncident
from traffic_twin.sim.sampler import JunctionSnapshot, LaneStat

LANES = ("n_in_1", "n_in_2", "nb_in_0", "s_in_1", "s_in_2", "wb_in_1", "w_in_1")
STAGES = ("HOP", "S_LEFT", "STM", "BUS_91", "BUS_57")
CROSSINGS = ("NB", "S", "WB", "W")
SITE = SiteSpec(stage_names=STAGES, lanes=LANES, lane_cap=(10.0,) * len(LANES), crossings=CROSSINGS)


@pytest.fixture(scope="module")
def norm(rl_train_scenario) -> dict:
    return dict(rl_train_scenario["observation"])


def snapshot(**over) -> JunctionSnapshot:
    """An ordinary snapshot at 08:15 on stage S_LEFT; `over` replaces fields."""
    lanes = tuple(
        LaneStat(lane, "N", 75.0, halting=2, vehicles=4, occupancy=0.2, max_wait_s=12.0)
        for lane in LANES
    )
    fields = {
        "t": 2700.0, "clock_s": 27000.0 + 2700.0, "phase": 4, "stage": 1,
        "in_transition": False, "time_in_stage": 8.0, "clearance_left_s": 0.0,
        "target_stage": None, "red_s": (20.0, 0.0, 40.0, 60.0, 80.0),
        "gap_s": (math.inf,) * 5, "lanes": lanes,
        "arm_halting": {}, "arm_vehicles": {}, "arm_wait_max_s": {}, "exit_halting": {},
        "pending": {"N": 3, "S": 0, "W": 1, "NB": 0, "WB": 0, "other": 0},
        "pending_by_class": {"car": 4, "bus": 0, "ambulance": 0},
        "station_occupancy": {}, "station_new_vehicles": {}, "queue_jam_m": {},
        "ped_count": (1, 0, 2, 0), "ped_wait_s": (10.0, 0.0, 30.0, 0.0),
        "ped_slack_s": (40.0, math.inf, 10.0, 55.0),
        "emergency": (), "preempt_target": None, "buses": (), "bus_time_loss_s": {},
        "incidents": (),
    }  # fmt: skip
    fields.update(over)
    return JunctionSnapshot(**fields)


def named(x: np.ndarray) -> dict[str, float]:
    return dict(zip(feature_names(SITE), x.tolist(), strict=True))


def test_74_names_unique_and_versioned():
    names = feature_names(SITE)
    assert len(names) == 74 and len(set(names)) == 74
    assert names[:3] == ("halt_n_in_1", "veh_n_in_1", "wait_n_in_1")
    assert names[-2:] == ("tod_sin", "tod_cos")
    assert OBS_VERSION == "rl-obs-1"


def test_site_spec_from_the_pitie_net(rl_scenario, norm):
    site = site_spec_from_config(rl_scenario, norm["veh_length_m"])
    assert site.lanes == LANES  # TLS order, as the JunctionSampler
    assert site.stage_names == STAGES and site.crossings == CROSSINGS
    assert all(cap >= 1.0 for cap in site.lane_cap)
    assert len(feature_names(site)) == 74


def test_ordinary_values(norm):
    x = observation(snapshot(), SITE, norm)
    assert x.dtype == np.float32 and x.shape == (74,)
    v = named(x)
    assert v["halt_n_in_1"] == pytest.approx(0.2) and v["veh_n_in_1"] == pytest.approx(0.4)
    assert v["wait_n_in_1"] == pytest.approx(12 / 120)
    assert v["pending_N"] == pytest.approx(3 / 20) and v["pending_W"] == pytest.approx(1 / 20)
    assert v["stage_S_LEFT"] == 1.0 and v["time_in_stage"] == pytest.approx(8 / 60)
    assert v["red_BUS_57"] == pytest.approx(80 / 180)
    assert v["ped_count_WB"] == pytest.approx(0.2) and v["ped_wait_WB"] == pytest.approx(30 / 88)
    assert v["ped_slack_S"] == 1.0  # infinite slack
    assert v["ped_slack_WB"] == pytest.approx(10 / 88)
    # 08:15 -> 29700 s of the day
    angle = 2 * math.pi * 29700 / 86400
    assert v["tod_sin"] == pytest.approx(0.5 + 0.5 * math.sin(angle), abs=1e-6)
    assert v["tod_cos"] == pytest.approx(0.5 + 0.5 * math.cos(angle), abs=1e-6)
    # no ambulance, no bus
    assert v["amb_present"] == v["amb_pending"] == v["preempt_active"] == 0.0
    for a in ("S", "NB", "WB"):
        assert v[f"bus_present_{a}"] == 0.0 and v[f"bus_dist_{a}"] == 1.0
        assert sum(v[f"bus_delay_{b}_{a}"] for b in ("low", "mid", "high")) == 0.0


def extreme_snapshots():
    huge = tuple(
        LaneStat(lane, "N", 7.5, halting=500, vehicles=900, occupancy=1.0, max_wait_s=1e6)
        for lane in LANES
    )
    pending_amb = EmergencyCall("amb", 1, 0, math.inf, 100.0, True, "N")
    near_amb = EmergencyCall("amb2", 7, 2, 12.0, 90.0, False, "W")
    bus = BusCall("bus_57_N.0", "57", 5, (1, 4), "S", 3000.0, 0.0, 0.0)
    late = BusCall("bus_91_E.0", "91", 9, (3,), "WB", 10.0, 2.0, 5.0)
    return {
        "empty": snapshot(
            lanes=tuple(LaneStat(n, "N", 70.0, 0, 0, 0.0, 0.0) for n in LANES),
            pending={},
            ped_count=(),
            ped_wait_s=(),
            ped_slack_s=(),
            red_s=(0.0,) * 5,
            time_in_stage=0.0,
            clock_s=0.0,
        ),  # fmt: skip
        "saturated": snapshot(
            lanes=huge,
            pending={"N": 10_000, "S": 500, "W": 99},
            red_s=(1e5,) * 5,
            time_in_stage=1e5,
            ped_count=(10_000,) * 4,
            ped_wait_s=(1e4,) * 4,
            ped_slack_s=(-500.0, -math.inf, math.inf, 1e9),
        ),  # fmt: skip
        "pending_ambulance": snapshot(emergency=(pending_amb,), preempt_target=0),
        "ambulances": snapshot(emergency=(pending_amb, near_amb), preempt_target=2),
        "buses": snapshot(buses=(bus, late), bus_time_loss_s={bus.vid: 45.0, late.vid: 500.0}),
        "transition": snapshot(in_transition=True, target_stage=2, clearance_left_s=9.0),
        "incident": snapshot(incidents=(ActiveIncident("w_in_1", "W", "w_in_1", 10.0),)),
    }


@pytest.mark.parametrize("case", sorted(extreme_snapshots()))
def test_every_value_in_unit_box(case, norm):
    x = observation(extreme_snapshots()[case], SITE, norm)
    assert x.dtype == np.float32 and x.shape == (74,)
    assert np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all(), case


@pytest.mark.parametrize("case", sorted(extreme_snapshots()))
def test_one_hots_are_exclusive(case, norm):
    v = named(observation(extreme_snapshots()[case], SITE, norm))
    assert sum(v[f"stage_{s}"] for s in STAGES) == 1.0
    assert sum(v[f"amb_dist_{b}"] for b in ("near", "mid", "far")) <= 1.0
    for a in ("S", "NB", "WB"):
        delay = sum(v[f"bus_delay_{b}_{a}"] for b in ("low", "mid", "high"))
        assert delay == v[f"bus_present_{a}"]  # one bin exactly when a bus is there
    for name in ("in_transition", "amb_present", "amb_pending", "preempt_active"):
        assert v[name] in (0.0, 1.0)


def test_ambulance_block(norm):
    cases = extreme_snapshots()
    v = named(observation(cases["pending_ambulance"], SITE, norm))
    assert v["amb_present"] == v["amb_pending"] == v["amb_arm_N"] == v["preempt_active"] == 1.0
    assert v["amb_dist_near"] == v["amb_dist_mid"] == v["amb_dist_far"] == 0.0  # not inserted
    v = named(observation(cases["ambulances"], SITE, norm))
    assert v["amb_dist_near"] == 1.0  # nearest inserted call at 12 m (< 50 m)
    assert v["amb_arm_N"] == v["amb_arm_W"] == 1.0 and v["amb_arm_S"] == 0.0


def test_bus_block(norm):
    v = named(observation(extreme_snapshots()["buses"], SITE, norm))
    assert v["bus_present_S"] == 1.0 and v["bus_dist_S"] == 1.0  # 3000 m, clipped
    assert v["bus_delay_mid_S"] == 1.0  # 45 s time loss
    assert v["bus_present_WB"] == 1.0 and v["bus_dist_WB"] == pytest.approx(10 / 300)
    assert v["bus_delay_high_WB"] == 1.0  # 500 s
    assert v["bus_present_NB"] == 0.0


def test_saturated_values_clip_to_one(norm):
    v = named(observation(extreme_snapshots()["saturated"], SITE, norm))
    assert v["halt_n_in_1"] == v["wait_w_in_1"] == v["pending_N"] == 1.0
    assert v["ped_slack_NB"] == 0.0 and v["ped_slack_S"] == 0.0  # negative slack
    assert v["ped_slack_WB"] == 1.0 and v["ped_slack_W"] == 1.0


def test_layout_mismatch_is_refused(norm):
    with pytest.raises(ValueError, match="lanes"):
        observation(snapshot(lanes=snapshot().lanes[:-1]), SITE, norm)
    bad = dict(norm, amb_dist_bins_m=[150, 50])
    with pytest.raises(ValueError, match="amb_dist_bins_m"):
        observation(snapshot(), SITE, bad)


# -- feature groups and drop (Part 7, D73 V1) --------------------------------------------
PED_NAMES = tuple(f"ped_{k}_{c}" for c in CROSSINGS for k in ("count", "wait", "slack"))


def test_groups_partition_the_names():
    names = feature_names(SITE)
    for name in names:
        hits = [g for g, prefixes in GROUPS.items() if name.startswith(prefixes)]
        assert hits == [group_of(name)], name
    assert sum(len(group_names(SITE, g)) for g in GROUPS) == len(names) == 74
    assert group_names(SITE, "pedestrian") == PED_NAMES


def test_drop_pedestrian_leaves_62_names_in_order():
    names = feature_names(SITE, ["pedestrian"])
    assert len(names) == 62 and not set(names) & set(PED_NAMES)
    assert names == tuple(n for n in feature_names(SITE) if n not in PED_NAMES)
    assert feature_names(SITE, ()) == feature_names(SITE)
    two = feature_names(SITE, ["pedestrian", "time"])
    assert len(two) == 60 and "tod_sin" not in two


@pytest.mark.parametrize("case", sorted(extreme_snapshots()))
def test_drop_removes_exactly_the_group_values(case, norm):
    snap = extreme_snapshots()[case]
    full = observation(snap, SITE, norm)
    x = observation(snap, SITE, {**norm, "drop": ["pedestrian"]})
    assert x.dtype == np.float32 and x.shape == (62,)
    assert np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all()
    keep = [i for i, n in enumerate(feature_names(SITE)) if n not in PED_NAMES]
    np.testing.assert_array_equal(x, full[keep])
    np.testing.assert_array_equal(observation(snap, SITE, {**norm, "drop": []}), full)


@pytest.mark.parametrize(
    ("drop", "match"),
    [(["pedestrians"], "unknown groups"), ("pedestrian", "list"),
     (["bus", "bus"], "twice"), ([None], "unknown")],
)  # fmt: skip
def test_bad_drop_is_refused(drop, match, norm):
    with pytest.raises(ValueError, match=match):
        check_drop(drop)
    with pytest.raises(ValueError, match=match):
        observation(snapshot(), SITE, {**norm, "drop": drop})
    with pytest.raises(ValueError):
        group_names(SITE, "nope")


def test_rl7_v1_config_drops_the_pedestrian_block(rl_scenario):
    from traffic_twin.config import ConfigError, load_config
    from traffic_twin.control.rl import check_rl_config

    v1 = load_config("configs/rl7/v1.yaml")
    assert v1["observation"]["drop"] == ["pedestrian"]
    check_rl_config(v1, "v1")
    site = site_spec_from_config(rl_scenario, v1["observation"]["veh_length_m"])
    assert len(feature_names(site, v1["observation"]["drop"])) == 62
    for name in ("t0", "t1", "v2", "p1", "wb12", "lam100", "car_centric"):
        cfg = load_config(f"configs/rl7/{name}.yaml")
        assert check_drop(cfg["observation"].get("drop")) == (), name
    bad = {**v1, "observation": {**v1["observation"], "drop": ["peds"]}}
    with pytest.raises(ConfigError, match="unknown groups"):
        check_rl_config(bad, "bad")

"""The RL environment (Part 6, EF-10; D55-D59). Episode parameters and configs without
SUMO; `check_env`, a training episode under a random policy, determinism and the
baselines on the real Pitié junction (`@sumo`, training seeds 1000-1002 only).

Part 7: forced conditions (`params_override`, D78) and the intervention split (D77)
without SUMO; `check_env` of the rl7 configs t1 / v1 / v2, the override on a real
episode, the split against the guard's counts and the default behaviour against its
pre-Part-7 capture (`@sumo`, Part 7 training seeds 5000-5003 only)."""

import hashlib
import math
import sys

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from traffic_twin.config import ConfigError, demand_hash  # noqa: E402
from traffic_twin.control.base import EXTRA_PED_SLACK, Observation  # noqa: E402
from traffic_twin.control.safety import INTERVENTION_KEYS  # noqa: E402
from traffic_twin.rl.clock import InterventionSplit  # noqa: E402
from traffic_twin.rl.env import (  # noqa: E402
    TrafficEnv,
    check_override,
    episode_config,
    episode_params,
    run_baseline,
)

WINDOW = (27000.0, 7200)
SUMMARY_KEYS = {
    "params", "decisions", "advances", "overrides", "return", "totals", "interventions",
    "iv_split", "danger", "audit", "preemption", "ped_curb_wait_max", "amb_seen_share", "t_end",
    "control_s", "obs_version", "reward_version", "wall",
}  # fmt: skip


# -- episode parameters (no SUMO) ------------------------------------------------------
def test_params_are_a_pure_function_of_the_seed(rl_train_scenario):
    a = episode_params(1000, rl_train_scenario, "train", window=WINDOW)
    assert a == episode_params(1000, rl_train_scenario, "train", window=WINDOW)
    assert a != episode_params(1001, rl_train_scenario, "train", window=WINDOW)
    assert a["episode_seed"] == 1000 and a["mode"] == "train"


def test_train_windows_and_draws(rl_train_scenario):
    draws = [
        episode_params(s, rl_train_scenario, "train", window=WINDOW) for s in range(1000, 1300)
    ]
    day_types, counts, incidents = set(), set(), 0
    for p in draws:
        assert (p["warmup_s"], p["window_s"], p["duration_s"]) == (300, 1800, 2100)
        assert p["offset_s"] % 60 == 0 and 0 <= p["offset_s"] <= 7200 - 2100
        assert p["begin_s"] == 27000 + p["offset_s"]
        assert 0.6 <= p["multiplier"] <= 1.4
        day_types.add(p["day_type"])
        counts.add(len(p["ambulances"]))
        for trip in p["ambulances"]:
            assert 300 <= trip["depart_s"] < 2100 - 120 and trip["trajet"] in ("N_S", "W_S")
        assert len(p["incidents"]) <= 1
        for inc in p["incidents"]:
            assert 300 <= inc["start_s"] < 2100 - 300 and inc["duration_s"] in (300, 600, 900)
            assert inc["speed_mps"] == 2.0
        incidents += len(p["incidents"])
    assert day_types == {"tue_wed_thu_term", "mon_fri_term", "school_holiday"}
    assert counts == {0, 1, 2}
    assert 0.2 < incidents / len(draws) < 0.4  # p 0.3


def test_full_mode_and_no_randomization(rl_train_scenario, rl_scenario):
    p = episode_params(1000, rl_train_scenario, "full", window=WINDOW)
    assert (p["begin_s"], p["warmup_s"], p["duration_s"]) == (27000, 0, 7200)
    fixed = episode_params(1000, rl_train_scenario, "full", randomize=False, window=WINDOW)
    assert fixed["multiplier"] is fixed["day_type"] is fixed["ambulances"] is None
    assert fixed["incidents"] == []
    # the base demand unchanged: same demand_hash as S0 / S1 / S2 (83a4f220fd20)
    cfg = episode_config(rl_scenario, fixed)
    assert "measurement" not in cfg and cfg["seed"] == 1000
    assert demand_hash(cfg) == demand_hash(rl_scenario)
    assert demand_hash(cfg).startswith("83a4f220fd20")


def test_episode_config_applies_the_draws(rl_train_scenario, rl_scenario):
    p = next(
        q
        for s in range(1000, 1100)
        if (q := episode_params(s, rl_train_scenario, "train", window=WINDOW))["ambulances"]
        and q["day_type"] == "school_holiday"
    )
    cfg = episode_config(rl_scenario, p)
    assert cfg["begin_s"] == p["begin_s"] and cfg["duration_s"] == 2100
    assert cfg["demand"]["multiplier"] == p["multiplier"]
    assert cfg["demand"]["calibration"] == "configs/calibration/pitie_302_school_holiday.yaml"
    trips = cfg["emergency"]["trips"]
    assert [t["depart_s"] for t in trips] == [a["depart_s"] for a in p["ambulances"]]
    assert all(t["route"][1] == "s_out" and t["route"][0] in ("n_in", "w_in") for t in trips)
    assert rl_scenario["emergency"].get("trips") is None  # the base is not modified
    none = dict(p, ambulances=[])
    assert episode_config(rl_scenario, none)["emergency"] == {"enabled": False}


def test_bad_mode_and_window(rl_train_scenario):
    with pytest.raises(ValueError, match="mode"):
        episode_params(1000, rl_train_scenario, "eval", window=WINDOW)
    with pytest.raises(Exception, match="fit"):
        episode_params(1000, rl_train_scenario, "train", window=(27000.0, 1800))


# -- the environment on SUMO -----------------------------------------------------------
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    env = TrafficEnv(mode="train", work_root=tmp_path_factory.mktemp("rl_env"))
    yield env
    env.close()


def play(env: TrafficEnv, seed: int, policy_seed: int = 0):
    """One episode with a random policy: (obs hash, return, infos, last obs)."""
    rng = np.random.default_rng(policy_seed)
    x, info = env.reset(options={"episode_seed": seed})
    h = hashlib.sha256(x.tobytes())
    infos, total, truncated = [info], 0.0, False
    while not truncated:
        x, reward, terminated, truncated, info = env.step(int(rng.integers(2)))
        assert terminated is False
        assert x in env.observation_space
        h.update(x.tobytes())
        total += reward
        infos.append(info)
    return h.hexdigest(), total, infos


@pytest.mark.sumo
def test_check_env(env):
    from gymnasium.utils.env_checker import check_env

    check_env(env, skip_render_check=True, skip_close_check=True)


@pytest.mark.sumo
def test_train_episode_random_policy(env):
    digest, total, infos = play(env, 1000)
    reset_info, steps, last = infos[0], infos[1:], infos[-1]
    assert reset_info["params"]["episode_seed"] == 1000
    assert all("episode" not in i for i in infos)  # SB3's Monitor reserves it (D61)
    assert all("episode_summary" not in i for i in steps[:-1])
    assert last["t"] == pytest.approx(2100.0)  # truncated at the window end
    for i in steps:
        assert {"t", "decision_s", "action", "requested", "guard_override", "reward_parts"} <= set(
            i
        )
        assert i["decision_s"] >= 1
    s = last["episode_summary"]
    assert SUMMARY_KEYS <= set(s)
    assert s["decisions"] == len(steps) and 0 < s["advances"] < s["decisions"]
    assert s["return"] == pytest.approx(total)
    assert s["danger"] == 0 and s["interventions"]["order"] == 0
    assert s["interventions"]["min_green"] == 0
    assert s["ped_curb_wait_max"] <= 90.0
    assert s["control_s"] == pytest.approx(1800.0)
    assert 0.0 <= s["amb_seen_share"] <= 1.0
    # the return is -0.001 x the cost accrued after the first decision
    t = s["totals"]
    assert t["cost"] > 0 and -total <= 0.001 * t["cost"] + 1e-9
    assert t["car_pending_s"] <= t["car_s"]


@pytest.mark.sumo
def test_same_seed_same_episode(env):
    a, b = play(env, 1000), play(env, 1000)
    assert a[0] == b[0] and a[1] == b[1]
    assert play(env, 1001)[0] != a[0]


@pytest.mark.sumo
def test_seed_schedule_and_seeded_reset(tmp_path):
    env = TrafficEnv(
        mode="train", worker=1, n_workers=4, work_root=tmp_path, seeds=range(1000, 1010)
    )
    try:
        played = [env.reset()[1]["params"]["episode_seed"] for _ in range(3)]
        assert played == [1001, 1005, 1009]  # seeds[i + k n]
        assert env.reset(seed=7)[1]["params"]["episode_seed"] == 1001  # schedule restarts
        assert env.reset(options={"episode_seed": 1002})[1]["params"]["episode_seed"] == 1002
        assert env.reset()[1]["params"]["episode_seed"] == 1005  # schedule untouched
    finally:
        env.close()


@pytest.mark.sumo
@pytest.mark.parametrize("policy", ["rules", "stay"])
def test_baselines(env, policy):
    s = run_baseline(env, policy, 1002)
    assert s["policy"] == policy and s["danger"] == 0 and s["ped_curb_wait_max"] <= 90.0
    assert s["t_end"] == pytest.approx(2100.0) and s["totals"]["cost"] > 0
    if policy == "rules":
        assert s["decisions"] == 0
    else:
        assert s["advances"] == 0 and s["decisions"] > 0


# -- Part 7: forced conditions (D78), no SUMO --------------------------------------------
DRAWN = ("begin_s", "offset_s", "multiplier", "day_type", "ambulances", "incidents")


def test_override_forces_only_its_keys(rl7_train_scenario):
    for seed in range(5000, 5040):
        own = episode_params(seed, rl7_train_scenario, "train", window=WINDOW)
        assert "override" not in own
        assert (
            episode_params(seed, rl7_train_scenario, "train", window=WINDOW, override=None) == own
        )
        forced = episode_params(
            seed, rl7_train_scenario, "train", window=WINDOW,
            override={"multiplier": 0.6, "day_type": "school_holiday"},
        )  # fmt: skip
        assert forced["multiplier"] == 0.6 and forced["day_type"] == "school_holiday"
        assert forced["override"] == {"multiplier": 0.6, "day_type": "school_holiday"}
        rest = {k: v for k, v in forced.items() if k not in ("multiplier", "day_type", "override")}
        assert rest == {k: v for k, v in own.items() if k not in ("multiplier", "day_type")}


def test_override_ambulances_and_incident(rl7_train_scenario):
    cfg = rl7_train_scenario
    sites = {"n_in_1", "n_in_2", "s_in_1", "s_in_2", "w_in_1"}
    for seed in range(5000, 5040):
        own = episode_params(seed, cfg, "train", window=WINDOW)
        amb = episode_params(seed, cfg, "train", window=WINDOW, override={"ambulances": 2})
        assert len(amb["ambulances"]) == 2
        for trip in amb["ambulances"]:
            assert 300 <= trip["depart_s"] < 2100 - 120 and trip["trajet"] in ("N_S", "W_S")
        assert amb == episode_params(seed, cfg, "train", window=WINDOW, override={"ambulances": 2})
        assert {k: amb[k] for k in DRAWN if k != "ambulances"} == {
            k: own[k] for k in DRAWN if k != "ambulances"
        }
        none = episode_params(seed, cfg, "train", window=WINDOW, override={"ambulances": 0})
        assert none["ambulances"] == []
        site = sorted(sites)[seed % 5]
        inc = episode_params(seed, cfg, "train", window=WINDOW, override={"incident": site})
        (one,) = inc["incidents"]
        assert one["site"] == site and one["lane"] == site and one["speed_mps"] == 2.0
        assert 300 <= one["start_s"] < 2100 - 300 and one["duration_s"] in (300, 600, 900)
        assert {k: inc[k] for k in DRAWN if k != "incidents"} == {
            k: own[k] for k in DRAWN if k != "incidents"
        }
        off = episode_params(seed, cfg, "train", window=WINDOW, override={"incident": None})
        assert off["incidents"] == [] and off["override"] == {"incident": None}


def test_override_mode_and_no_randomization(rl7_train_scenario):
    full = episode_params(
        5000, rl7_train_scenario, "train", window=WINDOW,
        override={"mode": "full", "multiplier": 1.0, "day_type": "tue_wed_thu_term"},
    )  # fmt: skip
    assert (full["mode"], full["begin_s"], full["warmup_s"], full["duration_s"]) == (
        "full", 27000, 0, 7200,
    )  # fmt: skip
    assert full["multiplier"] == 1.0 and full["day_type"] == "tue_wed_thu_term"
    fixed = episode_params(
        5000, rl7_train_scenario, "full", randomize=False, window=WINDOW,
        override={"multiplier": 1.4},
    )  # fmt: skip
    assert fixed["multiplier"] == 1.4 and fixed["day_type"] is fixed["ambulances"] is None


def test_every_stress_cell_is_a_valid_override(rl7_train_scenario):
    cells = rl7_train_scenario["protocol"]["stress"]["cells"]
    assert len(cells) == 13
    for name, cell in cells.items():
        forced = check_override(cell, rl7_train_scenario)
        assert set(forced) == set(cell), name
        p = episode_params(5000, rl7_train_scenario, "train", window=WINDOW, override=cell)
        for key in ("multiplier", "day_type"):
            if key in cell:
                assert p[key] == cell[key], name
        if "incident" in cell:
            assert [i["site"] for i in p["incidents"]] == [cell["incident"]], name
        if "ambulances" in cell:
            assert len(p["ambulances"]) == cell["ambulances"], name
        assert p["mode"] == cell.get("mode", "train"), name


@pytest.mark.parametrize(
    "bad",
    [{"multiplier": 0}, {"multiplier": True}, {"day_type": "sunday"}, {"ambulances": -1},
     {"ambulances": 1.5}, {"incident": "n_in_9"}, {"mode": "eval"}, {"demand": 1.0}, [1]],
)  # fmt: skip
def test_bad_override_is_refused(rl7_train_scenario, bad):
    with pytest.raises(ConfigError, match="params_override"):
        episode_params(5000, rl7_train_scenario, "train", window=WINDOW, override=bad)


# -- Part 7: intervention split (D77), no SUMO -------------------------------------------
class FakeGuard:
    def __init__(self) -> None:
        self.interventions = dict.fromkeys(INTERVENTION_KEYS, 0)


def obs_at(stage: int, tis: float, slack=(30.0,), transition: bool = False) -> Observation:
    return Observation(100.0, stage, tis, transition, {}, {EXTRA_PED_SLACK: tuple(slack)})


def test_split_classifies_cuts_and_replacements():
    guard, split = FakeGuard(), InterventionSplit(5.0)
    guard.interventions["starvation"] = 1  # before the controlled part: replaced
    split.reset(guard)
    max_green = [40.0, 40.0, 40.0]
    # a stay with visible slack 3 s < 5 s, then the shield cuts the held green
    split.decided(obs_at(0, 10.0, slack=(3.0, 50.0)), 0, max_green)
    split.after_step(guard, obs_at(0, 10.0), 0)
    guard.interventions["ped_wait"] += 1
    split.after_step(guard, obs_at(0, 11.0), 0)
    # a stay with room (20 + 5 < 40, slack 30 s): a later max-green cut is not foreseeable
    split.decided(obs_at(1, 20.0), 0, max_green)
    guard.interventions["max_green"] += 1
    split.after_step(guard, obs_at(1, 22.0), 1)
    # max green on a step that requested another stage: replaced
    guard.interventions["max_green"] += 1
    split.after_step(guard, obs_at(1, 23.0), 2)
    # a stay at tis 36 + 5 >= 40: foreseeable max green
    split.decided(obs_at(2, 36.0), 0, max_green)
    guard.interventions["max_green"] += 1
    split.after_step(guard, obs_at(2, 40.0), 2)
    # starvation on a held step: replaced; a cut after a transition: not foreseeable
    guard.interventions["starvation"] += 1
    split.after_step(guard, obs_at(2, 41.0), 2)
    split.after_step(guard, obs_at(2, 0.0, transition=True), 2)
    guard.interventions["ped_wait"] += 1
    split.after_step(guard, obs_at(0, 7.0), 0)
    res = split.result(guard)
    assert res == {"cut": 4, "replaced": 3, "cut_foreseeable": 2}
    assert res["cut"] + res["replaced"] == sum(guard.interventions.values())
    split.decided(obs_at(0, 7.0, slack=(1.0,)), 1, max_green)  # an advance clears the stay
    guard.interventions["ped_wait"] += 1
    split.after_step(guard, obs_at(0, 8.0), 0)
    assert split.result(guard)["cut_foreseeable"] == 2
    assert not split.foreseeable(obs_at(0, 7.0, slack=(math.inf,)), None)
    assert split.foreseeable(obs_at(0, 7.0, slack=(4.999,)), None)


# -- Part 7 on SUMO ------------------------------------------------------------------------
@pytest.mark.sumo
@pytest.mark.parametrize(("name", "n_obs"), [("t1", 74), ("v1", 62), ("v2", 74)])
def test_check_env_rl7(name, n_obs, tmp_path):
    from gymnasium.utils.env_checker import check_env

    env = TrafficEnv(f"configs/rl7/{name}.yaml", mode="train", work_root=tmp_path)
    try:
        assert env.observation_space.shape == (n_obs,)
        assert env.seeds[0] == 5000 and env.clock.interval_s == (10.0 if name == "v2" else 5.0)
        check_env(env, skip_render_check=True, skip_close_check=True)
    finally:
        env.close()


@pytest.mark.sumo
def test_reset_with_params_override(env):
    cell = {"multiplier": 1.6, "day_type": "mon_fri_term", "incident": "w_in_1"}
    own = env.reset(options={"episode_seed": 5003})[1]["params"]
    _x, info = env.reset(options={"episode_seed": 5003, "params_override": cell})
    p = info["params"]
    assert p["override"] == cell and p["multiplier"] == 1.6 and p["day_type"] == "mon_fri_term"
    assert [i["site"] for i in p["incidents"]] == ["w_in_1"]
    assert {k: p[k] for k in ("begin_s", "ambulances")} == {
        k: own[k] for k in ("begin_s", "ambulances")
    }
    assert env.cfg["demand"]["multiplier"] == 1.6
    assert env.cfg["demand"]["calibration"].endswith("pitie_302_mon_fri_term.yaml")
    assert [i.site for i in env.incidents.incidents] == ["w_in_1"]
    s = run_baseline(env, "rules", 5003, params_override=cell)
    assert s["params"]["override"] == cell and s["params"]["incidents"] == p["incidents"]


@pytest.mark.sumo
@pytest.mark.parametrize("policy", ["stay", "random", "rules"])
def test_iv_split_sums_to_the_interventions(env, policy):
    s = run_baseline(env, policy, 5000, params_override={"multiplier": 1.4})
    split, inter = s["iv_split"], s["interventions"]
    assert set(split) == {"cut", "replaced", "cut_foreseeable"}
    assert split["cut"] + split["replaced"] == sum(inter.values())
    # binary action, strict order: only the shield and max green replace a request, and
    # only a held one; the actuated warm-up causes none
    assert split["cut"] == inter["ped_wait"] + inter["max_green"]
    assert 0 <= split["cut_foreseeable"] <= split["cut"]
    if policy == "stay":
        assert split["cut"] > 0 and split["cut_foreseeable"] == split["cut"]
    if policy == "rules":
        assert split["cut_foreseeable"] == 0  # no policy decision


# seed 5000, random policy rng(0), configs/rl.yaml: the observations and the return
# before the Part 7 changes (captured on Windows, SUMO 1.28.0)
CAPTURE_5000 = ("fcd24a932247c4624bccdb5bd9db585719e01e5bf8c82deb936e2f4f619e5ff8", -129.3606)


@pytest.mark.sumo
def test_default_behaviour_unchanged(env, tmp_path):
    digest, total, _infos = play(env, 5000)
    t1 = TrafficEnv("configs/rl7/t1.yaml", mode="train", work_root=tmp_path)
    try:
        assert play(t1, 5000)[:2] == (digest, total)  # same env keys as rl.yaml
    finally:
        t1.close()
    if sys.platform != "win32":
        pytest.skip("the capture is bit-exact on Windows only")
    assert (digest, round(total, 6)) == CAPTURE_5000

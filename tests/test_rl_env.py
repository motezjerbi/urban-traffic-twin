"""The RL environment (Part 6, EF-10; D55-D59). Episode parameters and configs without
SUMO; `check_env`, a training episode under a random policy, determinism and the
baselines on the real Pitié junction (`@sumo`, training seeds 1000-1002 only)."""

import hashlib

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from traffic_twin.config import demand_hash  # noqa: E402
from traffic_twin.rl.env import (  # noqa: E402
    TrafficEnv,
    episode_config,
    episode_params,
    run_baseline,
)

WINDOW = (27000.0, 7200)
SUMMARY_KEYS = {
    "params", "decisions", "advances", "overrides", "return", "totals", "interventions",
    "danger", "audit", "preemption", "ped_curb_wait_max", "amb_seen_share", "t_end",
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

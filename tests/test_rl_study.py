"""Part 7 study (WS3; D72-D75, D78, D79, D86): the stage-T rule at its edges, the D72
late window, the D75 tie rule, the D79 verdict (C1-C4) at its edges, the single
dev-test look, the holdout, selection, publication and stress panel end to end on
synthetic run folders, tiny untrained models and the SUMO-free stub env. Only the
development seeds of the `stub_rl7_cfg` workspace (5000-5219) are played; no SUMO."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_twin.config import ConfigError, load_config
from traffic_twin.evaluation.stats import wilcoxon
from traffic_twin.paths import PITIE_RL_CONFIG
from traffic_twin.rl import study
from traffic_twin.rl.stub import run_baseline as stub_run_baseline
from traffic_twin.rl.train import TOTAL_KEYS, EnvBackend, file_sha256, stub_backend

pytest.importorskip("stable_baselines3")

T_CONFIGS = ("t0", "t1", "v1", "v2", "p1")
WINDOW_S = 60  # stub episodes of one minute: the code paths, never a result


# -- synthetic artifacts ---------------------------------------------------------------
def tiny_model(path: Path, rl7: Path, seed: int) -> Path:
    """An untrained DQN on the stub's spaces (a real SB3 zip, distinct per seed)."""
    from stable_baselines3 import DQN

    from traffic_twin.rl.stub import StubTrafficEnv

    env = StubTrafficEnv(rl7, window_s=WINDOW_S)
    model = DQN(
        "MlpPolicy",
        env,
        seed=int(seed),
        device="cpu",
        buffer_size=100,
        learning_starts=10**6,
        policy_kwargs={"net_arch": [8]},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return path


def make_run(
    root: Path,
    rl7: Path,
    name: str,
    seed: int,
    steps: int,
    costs: list[float],
    ped: list[float] | None = None,
    *,
    model: Path | None = None,
    algo: str = "dqn",
    sha: str | None = None,
    windows: list[int] | None = None,
) -> Path:
    """A finished Part 7 run folder as rl.train writes it: validations at
    1/n ... n/n of the budget with `costs` (and `ped` pedestrian seconds) on the
    validation `windows` (default: the config's seeds.selection; [] records none)."""
    config = study.run_config_path(rl7, name)
    if windows is None:
        windows = study.seed_range(load_config(rl7)["seeds"]["selection"])
    sha = sha or study.config_sha(config)
    d = Path(root) / f"{algo}_{name}_s{seed}_{steps // 1000}k_{sha[:8]}"
    d.mkdir(parents=True)
    n = len(costs)
    step = [int(steps * (i + 1) / n) for i in range(n)]
    val = pd.DataFrame({"step": step, "cost": costs})
    val["pedestrian_s"] = ped if ped is not None else [1000.0] * n
    for k in TOTAL_KEYS:
        if k != "pedestrian_s":
            val[k] = val["cost"] / 10
    val.to_csv(d / "validation.csv", index=False)
    late = val[val["step"] >= 0.5 * steps]
    pick = late.loc[late["cost"].idxmin()]
    manifest = {
        "run": d.name,
        "algo": algo,
        "variant": name,
        "seed": seed,
        "steps": steps,
        "backend": "sumo",
        "config": config.as_posix(),
        "run_config_sha": sha,
        "pick_step": int(pick["step"]),
        "pick_val_cost": float(pick["cost"]),
        "late_mean": float(late["cost"].mean()),
        "obs_version": "rl-obs-1",
        "reward_version": "rl-reward-1",
        "n_envs": 2,
        "code_version": "abc1234",
    }
    if windows:
        manifest["selection_seeds"] = list(windows)
    if model is not None:
        shutil.copyfile(model, d / study.PICK_MODEL_FILE)
        (d / study.VAL_DIR).mkdir()
        for s in step[-2:]:
            shutil.copyfile(model, d / study.VAL_DIR / f"step_{s}.zip")
        manifest["pick_model_sha256"] = file_sha256(d / study.PICK_MODEL_FILE)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return d


def _stub_baseline(env, controller, episode_seed, params_override=None):
    """The stub's run_baseline, with forced conditions through the stub's reset."""
    if not params_override:
        return stub_run_baseline(env, controller, episode_seed)
    orig = env.reset

    def reset(*, seed=None, options=None):
        return orig(seed=seed, options={**(options or {}), "params_override": params_override})

    env.reset = reset
    try:
        return stub_run_baseline(env, controller, episode_seed)
    finally:
        del env.reset


def backend() -> EnvBackend:
    stub = stub_backend(window_s=WINDOW_S)
    return EnvBackend("stub", stub.make_env, _stub_baseline, dict(stub.env_kwargs))


def s3_scenario(tmp_path: Path, rl7: Path, model: Path, variant: str = "t1") -> Path:
    """A pitie_rl_s3-like scenario pinning `model` (a manifest next to it)."""
    import yaml

    (model.parent / "manifest.json").write_text(
        json.dumps(
            {"algo": "dqn", "model_version": "rl7-dqn-t1-s0-test", "obs_version": "rl-obs-1"}
        ),
        encoding="utf-8",
    )
    path = tmp_path / "pitie_rl_s3.yaml"
    cfg = {
        "extends": PITIE_RL_CONFIG.resolve().as_posix(),
        "scenario": "pitie_rl_s3",
        "rl": {
            "model": model.as_posix(),
            "model_sha256": file_sha256(model),
            "config": study.run_config_path(rl7, variant).as_posix(),
        },
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


# -- D72 late window -------------------------------------------------------------------
def test_late_window_starts_at_half_the_budget():
    val = pd.DataFrame({"step": [400, 499, 500, 600, 1000], "cost": [5.0, 4.0, 3.0, 2.0, 2.5]})
    late = study.late_window(val, 1000, 0.5)
    assert late["step"].tolist() == [500, 600, 1000]
    # a `late` column written by rl.train takes precedence
    val["late"] = [False, False, False, True, True]
    assert study.late_window(val, 1000, 0.5)["step"].tolist() == [600, 1000]


def test_late_stats_pick_and_tie(tmp_path, stub_rl7_cfg):
    rl7 = stub_rl7_cfg()
    d = make_run(tmp_path, rl7, "t1", 0, 1000, [1.0, 9.0, 7.0, 7.0], ped=[0, 10, 20, 30])
    stats = study.late_stats(study.read_run(d), 0.5)
    # steps 250 (early), 500, 750, 1000: the pick is the earliest of the tied lowest
    assert stats["pick_step"] == 750 and stats["pick_cost"] == 7.0
    assert stats["late_mean"] == pytest.approx(23 / 3)
    assert stats["late_ped_s"] == pytest.approx(20.0)
    assert stats["n_late"] == 3


# -- D73 rule --------------------------------------------------------------------------
def _scores(**per_config):
    return {k: dict(enumerate(v)) for k, v in per_config.items()}


RULE = {"reference": "t1", "margin_rel": 0.015, "min_seed_wins": 2, "ped_guard_rel": 0.05}


def _decide(scores, ped=None):
    ped = ped or {k: dict.fromkeys(v, 100.0) for k, v in scores.items()}
    return study.decide_rule(scores, ped, order=list(scores), **RULE)


def test_rule_margin_edge():
    base = [100.0, 100.0, 100.0]
    at_edge = _decide(_scores(t1=base, v1=[98.5, 98.5, 98.5]))
    assert at_edge["chosen"] == "v1" and at_edge["outcome"]["v1"]["a"]
    short = _decide(_scores(t1=base, v1=[98.6, 98.6, 98.6]))
    assert short["chosen"] == "t1" and not short["outcome"]["v1"]["a"]


def test_rule_needs_two_of_three_seeds():
    # mean 97 (3 % better) but below T1 on one seed only
    one = _decide(_scores(t1=[100.0, 100.0, 100.0], v2=[91.0, 100.0, 100.0]))
    assert one["outcome"]["v2"]["a"] and one["outcome"]["v2"]["wins"] == 1
    assert one["chosen"] == "t1"
    two = _decide(_scores(t1=[100.0, 100.0, 100.0], v2=[92.0, 99.0, 100.0]))
    assert two["outcome"]["v2"]["wins"] == 2 and two["chosen"] == "v2"


def test_rule_pedestrian_guard_edge():
    s = _scores(t1=[100.0] * 3, v1=[90.0] * 3)
    ok = _decide(s, {"t1": dict.fromkeys(range(3), 100.0), "v1": dict.fromkeys(range(3), 105.0)})
    assert ok["chosen"] == "v1" and ok["outcome"]["v1"]["c"]
    bad = _decide(s, {"t1": dict.fromkeys(range(3), 100.0), "v1": dict.fromkeys(range(3), 105.1)})
    assert bad["chosen"] == "t1" and not bad["outcome"]["v1"]["c"]


def test_rule_lowest_qualifier_wins():
    res = _decide(
        _scores(
            t0=[97.0, 97.0, 97.0],
            t1=[100.0, 100.0, 100.0],
            v1=[95.0, 96.0, 97.0],
            v2=[94.0, 101.0, 101.0],  # lowest single seed but one win only
            p1=[96.0, 96.0, 96.0],
        )
    )
    assert [k for k, v in res["outcome"].items() if v["qualifies"]] == ["t0", "v1", "p1"]
    assert res["chosen"] == "v1"
    assert not res["outcome"]["t1"]["qualifies"] and res["outcome"]["t1"]["reference"]


def _stage_t(root: Path, rl7: Path, steps: int, special: dict | None = None) -> None:
    special = special or {}
    for name in T_CONFIGS:
        for seed in (0, 1, 2):
            late = special.get(name, 100.0) + seed
            make_run(root, rl7, name, seed, steps, [200.0, late, late, late], ped=[50.0] * 4)


def test_decide_t_writes_the_decision(stub_rl7_cfg):
    rl7 = stub_rl7_cfg()
    cfg = load_config(rl7)
    steps = int(cfg["protocol"]["stage_t"]["steps"])
    runs = study.runs_root(cfg) / "T"
    _stage_t(runs, rl7, steps, {"v1": 97.0, "p1": 99.0})
    doc = study.decide_t(rl7)
    assert doc["chosen"] == "v1"
    assert doc["scores"]["v1"] == {"0": 97.0, "1": 98.0, "2": 99.0}
    assert doc["means"]["t1"] == pytest.approx(101.0)
    assert not doc["outcome"]["p1"]["a"]  # 100 / 101: 1 % better, short of the 1.5 % margin
    on_disk = json.loads((study.study_root(cfg) / study.DECISION_T_FILE).read_text("utf-8"))
    assert on_disk["chosen"] == "v1" and on_disk["rule"]["margin_rel"] == 0.015
    assert {"scores", "means", "ped_s", "rule", "outcome", "chosen", "created_at"} <= set(on_disk)
    with pytest.raises(FileExistsError):
        study.decide_t(rl7)
    assert study.decide_t(rl7, force=True)["chosen"] == "v1"


def test_decide_t_needs_every_current_run(stub_rl7_cfg):
    rl7 = stub_rl7_cfg()
    cfg = load_config(rl7)
    steps = int(cfg["protocol"]["stage_t"]["steps"])
    runs = study.runs_root(cfg) / "T"
    _stage_t(runs, rl7, steps)
    shutil.rmtree(next(runs.glob("dqn_v2_s1_*")))
    make_run(runs, rl7, "v2", 1, steps, [1.0, 1.0], sha="0" * 64)  # another config version
    with pytest.raises(ConfigError, match=r"v2 seed 1: .*stale"):
        study.decide_t(rl7)


@pytest.mark.parametrize(
    ("windows", "fragment"),
    [
        (list(range(5090, 5100)), "validated on 5090-5099, not on 5100-5109"),  # a dev run
        ([5112, 5113], "validated on 5112-5113"),  # the stub's holdout.full seeds
        ([], "no .selection_seeds. record"),
    ],
)
def test_decide_t_refuses_runs_validated_on_other_windows(stub_rl7_cfg, windows, fragment):
    """D73 reads late_mean only from runs validated on the registered selection windows:
    a run validated elsewhere (dev windows, a held-out range, unrecorded) is named."""
    rl7 = stub_rl7_cfg()
    cfg = load_config(rl7)
    steps = int(cfg["protocol"]["stage_t"]["steps"])
    runs = study.runs_root(cfg) / "T"
    _stage_t(runs, rl7, steps)
    shutil.rmtree(next(runs.glob("dqn_v1_s2_*")))
    make_run(runs, rl7, "v1", 2, steps, [200.0, 1.0, 1.0, 1.0], windows=windows)
    with pytest.raises(ConfigError, match=r"v1 seed 2: dqn_v1_s2_\S+: " + fragment):
        study.decide_t(rl7)
    assert not (study.study_root(cfg) / study.DECISION_T_FILE).exists()


def test_decide_t_development_windows_only_outside_the_study_root(stub_rl7_cfg, tmp_path):
    """The gate's mini-campaign validates on development (training) windows: decide-t
    accepts them only when named, only training seeds, never into outputs.study."""
    rl7 = stub_rl7_cfg()
    cfg = load_config(rl7)
    runs = tmp_path / "gate" / "runs" / "T"
    dev = list(range(5090, 5100))
    for name in T_CONFIGS:
        for seed in (0, 1, 2):
            make_run(runs, rl7, name, seed, 300, [9.0, 5.0, 5.0, 5.0], windows=dev)
    with pytest.raises(ConfigError, match="5090-5099, not on 5100-5109"):
        study.decide_t(rl7, runs, tmp_path / "gate", 300)  # without the windows: refused
    with pytest.raises(ConfigError, match="registered study root"):
        study.decide_t(rl7, runs, None, 300, validation_seeds=dev)
    with pytest.raises(ConfigError, match=r"training seeds .*5110-5111"):
        study.decide_t(rl7, runs, tmp_path / "gate", 300, validation_seeds=[5110, 5111])
    doc = study.decide_t(rl7, runs, tmp_path / "gate", 300, validation_seeds=dev)
    assert doc["development"] and doc["validation_seeds"] == dev
    assert not (study.study_root(cfg) / study.DECISION_T_FILE).exists()
    gate2 = str(tmp_path / "gate2")
    study.main(["--config", str(rl7), "decide-t", "--runs", str(runs), "--out", gate2,
                "--steps", "300", "--validation-seeds", "5090-5099"])  # fmt: skip
    assert study.read_json(tmp_path / "gate2" / study.DECISION_T_FILE)["development"]


# -- D75 ties --------------------------------------------------------------------------
def test_choose_s3_ties():
    c = [
        {"run": "a", "seed": 0, "stage1": 100.0, "stage2": 1000.0},
        {"run": "b", "seed": 1, "stage1": 90.0, "stage2": 1004.0},  # within 0.5 %
        {"run": "c", "seed": 2, "stage1": 80.0, "stage2": 1006.0},  # outside
    ]
    assert study.choose_s3(c, 0.005)["run"] == "b"
    assert study.choose_s3(c, 0.0)["run"] == "a"
    same = [{**c[0], "seed": 3, "run": "d"}, {**c[0], "seed": 1, "run": "e"}]
    assert study.choose_s3(same, 0.005)["run"] == "e"  # same stage 1: lower seed
    edge = [c[0], {"run": "f", "seed": 4, "stage1": 1.0, "stage2": 1005.0}]
    assert study.choose_s3(edge, 0.005)["run"] == "f"  # exactly 0.5 % is a tie


# -- D79 verdict ------------------------------------------------------------------------
SEEDS = list(range(3020, 3030))
KPI_BASE = {
    ("car", "time_loss_mean"): 60.0,
    ("bus", "delay_mean"): 50.0,
    ("pedestrian", "ped_curb_wait_mean"): 30.0,
    ("pedestrian", "ped_curb_wait_max"): 88.0,
    ("ambulance", "travel_time_total"): 95.0,
    ("all", "person_delay_mean"): 70.0,
    ("car", "depart_delay_mean"): 0.5,
}


def kpi_frame(s3_delta=None, s3_scale=None) -> pd.DataFrame:
    """S0/S1/S2/S3 KPI rows; S3 = S1 x scale + delta (per KPI; delta may be per seed)."""
    s3_delta, s3_scale = s3_delta or {}, s3_scale or {}
    rows = []
    for i, seed in enumerate(SEEDS):
        for (user, kpi), v in KPI_BASE.items():
            s1 = v + 0.3 * i
            vals = {"S0": s1 * 1.2, "S1": s1, "S2": s1 * 0.97}
            d = s3_delta.get(f"{user}/{kpi}", 0.0)
            d = d[i] if isinstance(d, list) else d
            vals["S3"] = s1 * s3_scale.get(f"{user}/{kpi}", 1.0) + d
            if kpi == "ped_curb_wait_max":
                vals = dict.fromkeys(vals, 88.0)
            rows += [
                {"label": k, "seed": seed, "user_type": user, "kpi": kpi, "value": x}
                for k, x in vals.items()
            ]
    return pd.DataFrame(rows)


def safety_frame(**s3) -> pd.DataFrame:
    rows = []
    for label in ("S0", "S1", "S2", "S3"):
        for seed in SEEDS:
            r = {
                "label": label,
                "seed": seed,
                "danger": 0,
                "collisions": 0,
                "teleports": 0,
                "ped_curb_wait_max": 88.0,
                "preempt_requests": 1,
                "preempt_served": 1,
                "preempt_timeouts": 0,
            }
            if label == "S3":
                r |= {k: v for k, v in s3.items() if k != "seed_only"}
            rows.append(r)
    return pd.DataFrame(rows)


HOLD = [{"run": f"r{i}", "holdout_w30": 90.0 + i, "w30_S1": 93.5} for i in range(5)]  # 4 below


def verdict(rl7_cfg, kpis, safety=None, hold=HOLD, repro=None, stress_worse=()):
    dv = rl7_cfg["protocol"]["devtest"]
    paired = study.paired_tables(kpis, dv["pairs"], 5.0, 0.05)
    return study.compute_verdict(
        paired,
        safety if safety is not None else safety_frame(),
        hold,
        repro,
        list(stress_worse),
        rl7_cfg,
        seeds=SEEDS,
        model_version="rl7-test",
    )


def test_verdict_pass_with_zero_variance_and_exact_p(rl7_cfg):
    v = verdict(rl7_cfg, kpi_frame({"all/person_delay_mean": -5.0}), repro={"same": True})
    assert v["C1"]["hl"] == pytest.approx(-5.0)
    assert v["C1"]["p"] == pytest.approx(2 / 1024)  # n = 10, every difference negative
    assert v["C1"]["pass"] and v["C2"]["pass"] and v["C3"]["pass"] and v["C4"]["pass"]
    assert v["verdict"] == "pass" and v["pending"] == []
    assert v["C4"]["runs_below_s1"] == 4
    assert {"C1", "C2", "C3", "C4", "stress_cells_worse", "s2_framing", "verdict", "seeds",
            "model_version"} <= set(v)  # fmt: skip
    json.loads(study.json_text(v))  # strict JSON


def test_verdict_c1_edges(rl7_cfg):
    zero = verdict(rl7_cfg, kpi_frame())
    assert zero["C1"]["p"] == 1.0 and not zero["C1"]["pass"] and zero["verdict"] == "fallback"
    # 7 negative, 3 positive differences: exact p of the rank sums, not significant
    d = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, 8.0, 9.0, 10.0]
    mixed = verdict(rl7_cfg, kpi_frame({"all/person_delay_mean": d}))
    want = wilcoxon(np.asarray(d)).pvalue
    assert mixed["C1"]["p"] == pytest.approx(want) and want > 0.05
    assert not mixed["C1"]["pass"]
    # 9 negative and a small positive: p = 4 / 1024 < 0.05
    d = [-2.0] * 9 + [1.0]
    nine = verdict(rl7_cfg, kpi_frame({"all/person_delay_mean": d}))
    want = wilcoxon(np.asarray(d)).pvalue
    assert nine["C1"]["p"] == pytest.approx(want) and want < 0.05
    assert nine["C1"]["pass"]


def test_verdict_c2_equity_edge(rl7_cfg):
    base = {"all/person_delay_mean": -5.0}
    at = verdict(rl7_cfg, kpi_frame(base, {"pedestrian/ped_curb_wait_mean": 1.05}))
    row = next(r for r in at["C2"]["rows"] if r["kpi"] == "pedestrian/ped_curb_wait_mean")
    assert row["rel_pct"] == pytest.approx(5.0) and not row["worse"] and at["C2"]["pass"]
    over = verdict(rl7_cfg, kpi_frame(base, {"car/time_loss_mean": 1.051}))
    assert not over["C2"]["pass"] and over["verdict"] == "fallback"
    # the pinned curb max and the ambulance are not equity-gated (C3 / reported only)
    amb = verdict(rl7_cfg, kpi_frame(base, {"ambulance/travel_time_total": 1.5}))
    assert amb["C2"]["pass"]


@pytest.mark.parametrize(
    ("s3", "fragment"),
    [
        ({"danger": 1}, "danger 1"),
        ({"collisions": 2}, "collisions 2"),
        ({"teleports": 1}, "teleports 1"),
        ({"ped_curb_wait_max": 90.5}, "curb max 90.5"),
        ({"preempt_served": 0}, "unresolved"),
        ({"preempt_served": 0, "preempt_timeouts": 1}, "timeouts 10 > S2's 0"),
    ],
)
def test_verdict_c3_safety(rl7_cfg, s3, fragment):
    v = verdict(rl7_cfg, kpi_frame({"all/person_delay_mean": -5.0}), safety_frame(**s3))
    assert not v["C3"]["pass"] and v["verdict"] == "fallback"
    assert any(fragment in x for x in v["C3"]["violations"]), v["C3"]["violations"]


def test_verdict_c3_curb_at_the_bound(rl7_cfg):
    v = verdict(
        rl7_cfg, kpi_frame({"all/person_delay_mean": -5.0}), safety_frame(ped_curb_wait_max=90.0)
    )
    assert v["C3"]["pass"]


def test_verdict_c4_runs_and_repro(rl7_cfg):
    kp = kpi_frame({"all/person_delay_mean": -5.0})
    three = [dict(h, holdout_w30=95.0) if i < 2 else h for i, h in enumerate(HOLD)]
    v = verdict(rl7_cfg, kp, hold=three)
    assert v["C4"]["runs_below_s1"] == 2 and not v["C4"]["pass"] and v["verdict"] == "fallback"
    # reproduce7 not run yet (it runs after the commit, D82): pending, never a pass
    pending = verdict(rl7_cfg, kp)
    assert pending["C4"]["pass"] is None and pending["C4"]["runs_pass"]
    assert pending["pending"] == ["C4.repro"] and pending["verdict"] == "pending"
    assert pending["C4"]["repro"]["status"] == "pending"
    differs = verdict(rl7_cfg, kp, repro={"same": False})
    assert differs["C4"]["pass"] is False and differs["verdict"] == "fallback"
    # a failed criterion decides the fallback before reproduce7 runs
    few = verdict(rl7_cfg, kp, hold=three)
    assert few["C4"]["pass"] is False and few["verdict"] == "fallback"
    c1_fails = verdict(rl7_cfg, kpi_frame())
    assert c1_fails["C4"]["pass"] is None and c1_fails["verdict"] == "fallback"


def test_verdict_of_never_passes_a_missing_check():
    assert study.verdict_of([True, True, True, True]) == "pass"
    assert study.verdict_of([True, True, True, None]) == "pending"
    assert study.verdict_of([False, True, True, None]) == "fallback"
    assert study.verdict_of([True, True, True, False]) == "fallback"
    assert study.c4_pass(True, None) is None and study.c4_pass(False, None) is False
    assert study.c4_pass(True, True) is True and study.c4_pass(True, False) is False


def test_final_verdict_completes_the_frozen_look_with_reproduce7(rl7_cfg, tmp_path):
    """The frozen verdict.json (C4 pending) is completed by reproduce7.json without a
    second look: PASS only when it reproduced, FALLBACK when it did not."""
    frozen = json.loads(
        study.json_text(verdict(rl7_cfg, kpi_frame({"all/person_delay_mean": -5.0})))
    )
    assert frozen["verdict"] == "pending"
    still = study.final_verdict(frozen, None)
    assert still["verdict"] == "pending" and still["pending"] == ["C4.repro"]
    ok = study.final_verdict(frozen, {"head": "abc1234", "same": True})
    assert ok["verdict"] == "pass" and ok["C4"]["pass"] is True and ok["pending"] == []
    assert ok["C4"]["repro"] == {"status": "done", "same": True, "head": "abc1234",
                                 "after_look": True}  # fmt: skip
    assert ok["verdict_at_look"] == "pending" and frozen["verdict"] == "pending"  # a copy
    bad = study.final_verdict(frozen, {"head": "abc1234", "same": False})
    assert bad["verdict"] == "fallback" and bad["C4"]["pass"] is False
    # a criterion that failed at the look stays failed whatever reproduce7 says
    failed = json.loads(study.json_text(verdict(rl7_cfg, kpi_frame())))
    assert study.final_verdict(failed, {"same": True})["verdict"] == "fallback"
    # a verdict.json of the earlier format (C4 pass true while reproduce7 pending)
    legacy = {**frozen, "verdict": "pass"}
    legacy["C4"] = {k: v for k, v in frozen["C4"].items() if k != "runs_pass"} | {"pass": True}
    assert study.final_verdict(legacy, None)["verdict"] == "pending"
    # resolve_verdict: verdict.json + the newest _reproduce/<HEAD>/reproduce7.json
    assert study.resolve_verdict(tmp_path) is None
    study.write_json(tmp_path / study.DEVTEST_DIR / study.VERDICT_FILE, frozen)
    assert study.resolve_verdict(tmp_path)["verdict"] == "pending"
    rep = tmp_path / study.REPRODUCE_DIR / "abc1234" / study.REPRODUCE_FILE
    study.write_json(rep, {"head": "abc1234", "same": False, "items": []})
    assert study.resolve_verdict(tmp_path)["verdict"] == "fallback"


def test_verdict_s2_framing_and_stress(rl7_cfg):
    # S2 = 0.97 x S1, S3 = S1 - 5 on a ~71 s mean: about 4 % better than S2
    better = verdict(
        rl7_cfg,
        kpi_frame({"all/person_delay_mean": -5.0}),
        repro={"same": True},
        stress_worse=["x0_6"],
    )
    assert better["s2_framing"]["framing"] == "better"
    assert better["stress_cells_worse"] == ["x0_6"] and better["verdict"] == "pass"
    near = verdict(rl7_cfg, kpi_frame({"all/person_delay_mean": -2.5}))
    assert near["s2_framing"]["framing"] == "approx" and near["s2_framing"]["text"] == "RL ≈ rules"


# -- single dev-test look --------------------------------------------------------------
def _c4_inputs(root: Path, sha: str) -> None:
    """select/decision.json (chosen r0, whose pick has sha256 `sha`) and holdout rows."""
    cands = [{"run": f"r{i}", "dir": f"runs/F/r{i}", "pick_model_sha256": sha if i == 0 else "x"}
             for i in range(5)]  # fmt: skip
    study.write_json(
        root / study.SELECT_DIR / study.SELECT_DECISION_FILE, {"candidates": cands, "chosen": "r0"}
    )
    rows = [{"dir": c["dir"], "run": c["run"], "stage": "F", "holdout_w30": 90.0, "w30_S1": 95.0}
            for c in cands]  # fmt: skip
    (root / study.HOLDOUT_DIR).mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / study.HOLDOUT_DIR / study.HOLDOUT_SUMMARY_FILE, index=False)


def test_devtest_looks_once(tmp_path, stub_rl7_cfg, monkeypatch):
    rl7 = stub_rl7_cfg()
    cfg = load_config(rl7)
    root = study.study_root(cfg)
    model = tiny_model(tmp_path / "s3" / "model.zip", rl7, 0)
    scenario = s3_scenario(tmp_path, rl7, model)
    seeds = study.seed_range(cfg["protocol"]["devtest"]["seeds"])
    calls = []

    def fake_windows(backend, s3, seeds_, workers, work_root):
        calls.append("windows")
        rows = [{"label": lab, "seed": s, "cost": 100.0 + (lab != "S3") * 10 + s % 3}
                for lab in study.WINDOW_POLICIES for s in seeds_]  # fmt: skip
        return pd.DataFrame(rows)

    def fake_kpi_runs(seeds_, out_root, configs, force=False):
        calls.append("kpi_runs")
        assert set(configs) == {"S0", "S1", "S2", "S3"}

    def fake_kpi_frames(out_root, configs, seeds_, occupancy):
        k = kpi_frame({"all/person_delay_mean": -5.0})
        k = k[k["seed"].isin(SEEDS[: len(seeds_)])]
        k["seed"] = k["seed"].map(dict(zip(SEEDS, seeds_, strict=False)))
        s = safety_frame()
        s = s[s["seed"].isin(SEEDS[: len(seeds_)])]
        s["seed"] = s["seed"].map(dict(zip(SEEDS, seeds_, strict=False)))
        return k, s.assign(code_version="abc1234")

    monkeypatch.setattr(study, "window_frame", fake_windows)
    monkeypatch.setattr(study, "kpi_runs", fake_kpi_runs)
    monkeypatch.setattr(study, "kpi_frames", fake_kpi_frames)
    with pytest.raises(ConfigError, match="select"):
        study.devtest(rl7, 1, scenario=scenario, backend=backend())
    assert calls == []  # the preconditions fail before any run
    ddir = root / study.DEVTEST_DIR
    assert not (ddir / study.LOOK_FILE).exists()  # no look recorded either
    _c4_inputs(root, file_sha256(model))
    v = study.devtest(rl7, 1, scenario=scenario, backend=backend())
    assert calls == ["windows", "kpi_runs"]
    for name in ("compare.csv", "kpi.csv", "kpi_safety.csv", "paired.csv", "verdict.json",
                 "look.json", "attempts.jsonl"):  # fmt: skip
        assert (ddir / name).is_file(), name
    assert v["seeds"] == seeds and v["model_version"] == "rl7-dqn-t1-s0-test"
    # reproduce7 has not run (it runs after the commit): C4 pending, never passed; with
    # n = 2 stub seeds C1 cannot reach p < 0.05, so the look already decides the fallback
    assert v["C4"]["pass"] is None and v["pending"] == ["C4.repro"]
    assert not v["C1"]["pass"] and v["verdict"] == "fallback"
    assert v["look"]["model_sha256"] == file_sha256(model) and v["look"]["attempts"] == 1
    assert not v["look"]["resumed"] and v["look"]["crashes"] == []
    assert v["C4"]["runs_below_s1"] == 5 and v["windows"]["S3"]["vs_S1_pct"] < 0
    paired = pd.read_csv(ddir / "paired.csv")
    assert set(paired["ref"]) == {"S0", "S1", "S2"} and set(paired["cand"]) == {"S3"}
    assert paired.groupby("ref")["p_holm"].count().eq(len(study.HEADLINE7)).all()
    with pytest.raises(FileExistsError):
        study.devtest(rl7, 1, scenario=scenario, backend=backend())
    assert calls == ["windows", "kpi_runs"]  # the second look ran nothing


def _fake_look(monkeypatch, calls, crash=False):
    """Fake dev-test windows / KPI runs (as in test_devtest_looks_once); `crash`: the
    KPI step raises right after the window comparison was written (the look happened)."""

    def fake_windows(backend, s3, seeds_, workers, work_root):
        calls.append(("windows", s3["model_sha256"]))
        rows = [{"label": lab, "seed": s, "cost": 100.0 + (lab != "S3") * 10 + s % 3}
                for lab in study.WINDOW_POLICIES for s in seeds_]  # fmt: skip
        return pd.DataFrame(rows)

    def fake_kpi_runs(seeds_, out_root, configs, force=False):
        calls.append(("kpi_runs", None))
        if crash:
            raise ConfigError("stale run (simulated crash)")

    def fake_kpi_frames(out_root, configs, seeds_, occupancy):
        k = kpi_frame({"all/person_delay_mean": -5.0})
        k = k[k["seed"].isin(SEEDS[: len(seeds_)])]
        k["seed"] = k["seed"].map(dict(zip(SEEDS, seeds_, strict=False)))
        s = safety_frame()
        s = s[s["seed"].isin(SEEDS[: len(seeds_)])]
        s["seed"] = s["seed"].map(dict(zip(SEEDS, seeds_, strict=False)))
        return k, s.assign(code_version="abc1234")

    monkeypatch.setattr(study, "window_frame", fake_windows)
    monkeypatch.setattr(study, "kpi_runs", fake_kpi_runs)
    monkeypatch.setattr(study, "kpi_frames", fake_kpi_frames)


def _reselect(root: Path, sha: str) -> None:
    shutil.rmtree(root / study.SELECT_DIR)
    shutil.rmtree(root / study.HOLDOUT_DIR)
    _c4_inputs(root, sha)


def test_devtest_crash_allows_only_a_resume_of_the_same_look(tmp_path, stub_rl7_cfg, monkeypatch):
    """A crash after the look (compare.csv written, no verdict.json) must not allow a
    second look with another S3 model: look.json pins the first one; the same model
    resumes, and verdict.json shows the resume and the crash."""
    rl7 = stub_rl7_cfg()
    root = study.study_root(load_config(rl7))
    ddir = root / study.DEVTEST_DIR
    model = tiny_model(tmp_path / "s3" / "model.zip", rl7, 0)
    scenario = s3_scenario(tmp_path, rl7, model)
    sha = file_sha256(model)
    _c4_inputs(root, sha)
    calls: list = []
    _fake_look(monkeypatch, calls, crash=True)
    with pytest.raises(ConfigError, match="simulated crash"):
        study.devtest(rl7, 1, scenario=scenario, backend=backend())
    assert (ddir / "compare.csv").is_file() and not (ddir / study.VERDICT_FILE).exists()
    look = study.read_json(ddir / study.LOOK_FILE)
    assert look["model_sha256"] == sha and look["selected_run"] == "r0"
    assert [a["status"] for a in study.read_attempts(ddir)] == ["started", "crashed"]

    # another model, re-selected and pinned: the second look is refused before any run
    other_dir = tmp_path / "s3b"
    other_dir.mkdir()
    other = tiny_model(other_dir / "model.zip", rl7, 7)
    other_scenario = s3_scenario(other_dir, rl7, other)
    _reselect(root, file_sha256(other))
    n = len(calls)
    with pytest.raises(ConfigError, match=r"look\.json: the dev-test seeds were already looked"):
        study.devtest(rl7, 1, scenario=other_scenario, backend=backend())
    assert len(calls) == n  # nothing played
    # the original S3 again while the selection names another pick: refused too
    with pytest.raises(ConfigError, match="not the selected pick"):
        study.devtest(rl7, 1, scenario=scenario, backend=backend())

    # the same look resumes (selection restored); the verdict records the resume
    _reselect(root, sha)
    _fake_look(monkeypatch, calls)
    v = study.devtest(rl7, 1, scenario=scenario, backend=backend())
    assert v["look"]["attempts"] == 2 and v["look"]["resumed"]
    assert [c["status"] for c in v["look"]["crashes"]] == ["crashed"]
    assert "simulated crash" in v["look"]["crashes"][0]["error"]
    assert [a["status"] for a in study.read_attempts(ddir)] == [
        "started", "crashed", "refused", "started", "done",
    ]  # fmt: skip
    with pytest.raises(FileExistsError):
        study.devtest(rl7, 1, scenario=scenario, backend=backend())


def test_devtest_refuses_a_development_selection(tmp_path, stub_rl7_cfg):
    rl7 = stub_rl7_cfg()
    root = study.study_root(load_config(rl7))
    model = tiny_model(tmp_path / "s3" / "model.zip", rl7, 0)
    _c4_inputs(root, file_sha256(model))
    path = root / study.SELECT_DIR / study.SELECT_DECISION_FILE
    study.write_json(path, {**study.read_json(path), "development": True})
    with pytest.raises(ConfigError, match="development windows"):
        study.devtest(rl7, 1, scenario=s3_scenario(tmp_path, rl7, model), backend=backend())
    assert not (root / study.DEVTEST_DIR / study.LOOK_FILE).exists()


def test_devtest_refuses_seeds_outside_dev_test(tmp_path, stub_rl7_cfg):
    rl7 = stub_rl7_cfg(protocol={"devtest": {"seeds": [5100, 5101]}})  # selection seeds
    model = tiny_model(tmp_path / "s3" / "model.zip", rl7, 0)
    with pytest.raises(ConfigError, match="dev_test"):
        study.devtest(rl7, 1, scenario=s3_scenario(tmp_path, rl7, model))


# -- holdout, select, publish on the stub ----------------------------------------------
@pytest.fixture
def f_runs(tmp_path, stub_rl7_cfg):
    """Two stage-F runs of C* = t1 (seeds 0, 1) with real tiny models, and decision_T."""
    rl7 = stub_rl7_cfg(protocol={"stage_f": {"seeds": [0, 1]}})
    cfg = load_config(rl7)
    steps = int(cfg["protocol"]["stage_f"]["steps"])
    root = study.runs_root(cfg) / "F"
    dirs = []
    for seed, costs in ((0, [9.0, 5.0, 4.0, 6.0]), (1, [9.0, 4.5, 6.0, 7.0])):
        model = tiny_model(tmp_path / "models" / f"m{seed}.zip", rl7, seed)
        dirs.append(make_run(root, rl7, "t1", seed, steps, costs, model=model))
    study.write_json(study.study_root(cfg) / study.DECISION_T_FILE, {"chosen": "t1"})
    return rl7, cfg, dirs


def test_holdout_scores_picks_and_curves(f_runs, monkeypatch):
    rl7, cfg, dirs = f_runs
    paths = study.holdout(dirs, rl7, curves=True, workers=1, backend=backend())
    windows, full = study.holdout_seeds(cfg)
    for d, p in zip(dirs, paths, strict=True):
        h = pd.read_csv(p)
        assert len(h) == len(windows) + len(full)
        assert set(h["mode"]) == {"train", "full"} and h["run"].eq(d.name).all()
        curve = pd.read_csv(d / study.HOLDOUT_CURVE_FILE)
        assert len(curve) == 2 and curve["n"].eq(len(windows)).all()
        assert {"cost", "cost_vs_S1", "cost_vs_S2", "pick"} <= set(curve.columns)
    hdir = study.study_root(cfg) / study.HOLDOUT_DIR
    summary = pd.read_csv(hdir / study.HOLDOUT_SUMMARY_FILE)
    assert len(summary) == 2 and set(summary["stage"]) == {"F"}
    assert summary["holdout_w30"].notna().all() and summary["w30_S1"].notna().all()
    base = list(hdir.glob("baselines_*.csv"))
    assert len(base) == 1  # one shared environment
    assert len(pd.read_csv(base[0])) == 4 * (len(windows) + len(full))
    played = []
    real = study.run_tasks

    def spy(backend_, episodes, workers=1, work_root=None):
        played.extend(e.policy for e in episodes)
        return real(backend_, episodes, workers, work_root)

    monkeypatch.setattr(study, "run_tasks", spy)
    study.holdout(dirs[:1], rl7, workers=1, backend=backend())
    assert set(played) == {"model"}  # baselines reused from the cache
    assert len(pd.read_csv(hdir / study.HOLDOUT_SUMMARY_FILE)) == 2  # row replaced


def test_holdout_expands_roots_and_globs(f_runs):
    rl7, cfg, dirs = f_runs
    runs = study.runs_root(cfg)
    assert study.expand_runs([runs]) == dirs
    assert study.expand_runs([runs / "F" / "dqn_t1_*"]) == dirs
    with pytest.raises(ConfigError):
        study.expand_runs([runs / "T"])


def test_select_and_publish(f_runs, tmp_path, capsys):
    rl7, cfg, dirs = f_runs
    study.holdout(dirs, rl7, workers=1, backend=backend())
    doc = study.select(rl7, 1, backend=backend())
    sdir = study.study_root(cfg) / study.SELECT_DIR
    assert {p.name for p in sdir.iterdir()} == {"stage1.csv", "stage2.csv", "decision.json"}
    cands = {c["seed"]: c for c in doc["candidates"]}
    assert cands[0]["stage1"] == 4.0 and cands[1]["stage1"] == 4.5  # the D72 picks
    assert cands[0]["pick_step"] == 900 and cands[1]["pick_step"] == 600
    assert all(
        math.isfinite(c["stage2"]) and math.isfinite(c["holdout_w30"]) for c in cands.values()
    )
    want = study.choose_s3(doc["candidates"], 0.005)["run"]
    assert doc["chosen"] == want
    stage2 = pd.read_csv(sdir / "stage2.csv")
    n2 = len(study.seed_range(cfg["seeds"]["select2"]))
    assert len(stage2) == n2 * (2 + len(study.STAGE2_BASELINES)) and set(stage2["mode"]) == {"full"}

    out = tmp_path / "models_rl"
    dest = study.publish(rl7, out)
    chosen = Path(doc["chosen_dir"])
    seed = next(c["seed"] for c in doc["candidates"] if c["run"] == doc["chosen"])
    assert dest.parent == out and dest.name.startswith(f"rl7-dqn-t1-s{seed}-")
    assert (dest / "model.zip").read_bytes() == (chosen / study.PICK_MODEL_FILE).read_bytes()
    for name in ("manifest.json", "validation.csv"):
        assert b"\r\n" not in (dest / name).read_bytes(), name
    m = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert m["model_version"] == dest.name and m["model_sha256"] == file_sha256(dest / "model.zip")
    assert m["obs_drop"] == [] and m["algo"] == "dqn" and m["variant"] == "t1"
    assert dest.name.endswith(m["policy_sha256"][:8])
    text = study.s3_config_text(m, dest / "model.zip")
    assert "scenario: pitie_rl_s3" in text and f"model_sha256: {m['model_sha256']}" in text
    assert study.publish(rl7, out) == dest  # idempotent for the same model

    # a pick replaced after the selection is refused
    other = tiny_model(tmp_path / "other.zip", rl7, 7)
    shutil.copyfile(other, chosen / study.PICK_MODEL_FILE)
    with pytest.raises(ConfigError, match="sha256"):
        study.publish(rl7, tmp_path / "models_again")


def test_publish_refuses_data_models(f_runs):
    rl7, _, _ = f_runs
    with pytest.raises(ConfigError, match="select"):
        study.publish(rl7, Path("data/models/rl"))
    with pytest.raises(ConfigError):
        study.check_out(Path("data/models/rl/x"))
    with pytest.raises(ConfigError):
        study.check_out(Path("data/results/x"))


def test_select_refuses_an_f_run_validated_on_other_windows(f_runs):
    """D75 stage 1 is each F run's pick: a run validated on other windows is named."""
    rl7, cfg, dirs = f_runs
    m = json.loads((dirs[1] / "manifest.json").read_text(encoding="utf-8"))
    m["selection_seeds"] = list(range(5090, 5100))
    (dirs[1] / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"t1 seed 1: .*validated on 5090-5099"):
        study.select(rl7, 1, backend=backend())
    with pytest.raises(ConfigError, match="registered study root"):
        study.select(rl7, 1, backend=backend(), validation_seeds=list(range(5090, 5100)))
    assert not (study.study_root(cfg) / study.SELECT_DIR / study.SELECT_DECISION_FILE).exists()


def test_publish_refuses_a_development_selection_into_the_models_root(f_runs, tmp_path):
    rl7, cfg, dirs = f_runs
    sdir = study.study_root(cfg) / study.SELECT_DIR
    cand = {"run": dirs[0].name, "dir": dirs[0].as_posix()}
    study.write_json(
        sdir / study.SELECT_DECISION_FILE,
        {"candidates": [cand], "chosen": dirs[0].name, "development": True},
    )
    with pytest.raises(ConfigError, match="development windows"):
        study.publish(rl7)
    assert study.publish(rl7, tmp_path / "dev_models").is_dir()


def test_select_needs_every_f_seed(f_runs):
    rl7, _, dirs = f_runs
    shutil.rmtree(dirs[1])
    with pytest.raises(ConfigError, match="t1 seed 1"):
        study.select(rl7, 1, backend=backend())


# -- stress -----------------------------------------------------------------------------
def test_stress_panel_on_the_stub(tmp_path, stub_rl7_cfg):
    rl7 = stub_rl7_cfg()
    cfg = load_config(rl7)
    model = tiny_model(tmp_path / "s3" / "model.zip", rl7, 0)
    path = study.stress(rl7, 1, scenario=s3_scenario(tmp_path, rl7, model), backend=backend())
    summary = pd.read_csv(path)
    cells = cfg["protocol"]["stress"]["cells"]
    assert summary["cell"].tolist() == list(cells)
    eps = pd.read_csv(path.parent / "episodes.csv")
    seeds = study.seed_range(cfg["seeds"]["stress"])
    assert len(eps) == len(cells) * len(study.STRESS_LABELS) * len(seeds)
    x06 = eps[eps["cell"] == "x0_6"]
    assert x06["multiplier"].eq(0.6).all()  # forced for every policy, model included
    full = eps[eps["cell"] == "full_nominal"]
    assert set(full["mode"]) == {"full"}
    assert {"vs_S1_pct", "vs_S2_pct", "p_vs_S1", "worse_than_s1", "pedestrian_s_S3"} <= set(summary)


def test_stress_summary_flags_cells_worse_than_s1():
    rows = []
    for cell, s3 in (("light", 106.0), ("edge", 105.0), ("heavy", 90.0)):
        for seed in range(4):
            for label, cost in (("S3", s3), ("S0", 120.0), ("S1", 100.0), ("S2", 95.0)):
                rows.append({"cell": cell, "label": label, "seed": seed, "mode": "train",
                             "override": "{}", "cost": cost + seed, "pedestrian_s": 10.0,
                             "danger": 0, "ped_curb_wait_max": 88.0})  # fmt: skip
    s = study.stress_summary(pd.DataFrame(rows), 0.05).set_index("cell")
    assert bool(s.loc["light", "worse_than_s1"]) and not bool(s.loc["heavy", "worse_than_s1"])
    assert s.loc["heavy", "wins_vs_S1"] == 4 and s.loc["heavy", "p_vs_S1"] == pytest.approx(0.125)
    # 105 + s vs 100 + s: 4.88 % on the means, not flagged
    assert not bool(s.loc["edge", "worse_than_s1"])


def test_stress_refuses_a_backend_that_cannot_force(tmp_path, stub_rl7_cfg):
    rl7 = stub_rl7_cfg()
    model = tiny_model(tmp_path / "s3" / "model.zip", rl7, 0)
    plain = EnvBackend("plain", stub_backend(window_s=WINDOW_S).make_env,
                       lambda env, ctl, seed: stub_run_baseline(env, ctl, seed),
                       {"window_s": WINDOW_S})  # fmt: skip
    with pytest.raises(ConfigError, match="cannot force"):
        study.stress(rl7, 1, scenario=s3_scenario(tmp_path, rl7, model), backend=plain)


def test_s3_model_needs_the_pin(tmp_path, stub_rl7_cfg):
    import yaml

    rl7 = stub_rl7_cfg()
    model = tiny_model(tmp_path / "s3" / "model.zip", rl7, 0)
    scenario = s3_scenario(tmp_path, rl7, model)
    assert study.s3_model(scenario)["model_sha256"] == file_sha256(model)
    cfg = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    cfg["rl"]["model_sha256"] = "0" * 64
    scenario.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match="model_sha256"):
        study.s3_model(scenario)


def test_golden_indices():
    margins = list(np.linspace(1, 0.01, 100))
    idx = study.golden_indices(margins, 29, 8)
    assert len(idx) == 29 and idx == sorted(set(idx))
    assert set(range(92, 100)) <= set(idx) and 0 in idx  # closest calls + spread
    assert study.golden_indices(margins[:10], 29, 8) == list(range(10))


def test_cli_help_lists_the_subcommands(capsys):
    with pytest.raises(SystemExit):
        study.main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("decide-t", "holdout", "select", "publish", "golden", "stress", "devtest"):
        assert cmd in out

"""Part 8 evaluation (D92-D100) on synthetic data, no SUMO: the seed 1-10 run lock and
the registration record (D92), the D99 page and section list, the frozen S3 inputs
(D100), the three new scenario configs (D93), A-06 (exact p, Holm, outcomes, the
procedural status: INVALID on n = 9, a stale run or a pin mismatch, PASS on an
unfavourable result), the 10.6 replication edges (C1-C4), the final table, the series,
the masked digest and the config checks. No run is ever launched: every test that
reaches the lock replaces the runner with one that fails the test."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import traffic_twin.evaluation.evaluate as ev
from traffic_twin.config import (
    ConfigError,
    config_hash,
    demand_hash,
    load_scenario,
    text_sha256,
)
from traffic_twin.evaluation import final
from traffic_twin.evaluation.evaluate import (
    PART8_TITLE,
    SECTIONS,
    EvaluationError,
    RunLockError,
    load_evaluation,
    registration_commit,
    run_lock,
)
from traffic_twin.paths import (
    EVALUATION_CONFIG,
    PITIE_AMB_RL_S3_CONFIG,
    PITIE_AMB_RL_S4_CONFIG,
    PITIE_RL_S3_CONFIG,
    PITIE_RL_S4_CONFIG,
    RESULTS_DIR,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTERED = {"commit": "abc1234", "committed_at": "2026-10-23T18:00:00+02:00"}


@pytest.fixture(scope="module")
def ecfg() -> dict:
    return load_evaluation(REPO_ROOT / EVALUATION_CONFIG)


@pytest.fixture
def no_runner(monkeypatch):
    """Any attempt to launch or load a run fails the test."""

    def refuse(*args, **kwargs):
        pytest.fail("a run was launched or loaded despite the run lock")

    monkeypatch.setattr(ev, "run_missing", refuse)
    monkeypatch.setattr(ev, "load_runs", refuse)


def registered(monkeypatch, *, dirty: bool, commit: dict | None = REGISTERED) -> None:
    monkeypatch.setattr(ev, "git_head", lambda: ("abc1234", dirty))
    monkeypatch.setattr(ev, "registration_commit", lambda path=EVALUATION_CONFIG: commit)


# -- run_lock --------------------------------------------------------------------------
def test_dirty_tree_and_seed_1_refuse_before_any_run(ecfg, monkeypatch, no_runner):
    """The Phase 0 acceptance case: S3/S4 registered, tree dirty, every run planned
    (nothing found on disk) into data/results -> EvaluationError, nothing launched."""
    registered(monkeypatch, dirty=True)
    monkeypatch.setattr(ev, "is_complete", lambda d: False)
    with pytest.raises(EvaluationError, match="run lock") as exc:
        ev.evaluate(ecfg, EVALUATION_CONFIG, list(ecfg["families"]), [1], RESULTS_DIR, True)
    assert isinstance(exc.value, RunLockError)
    assert "pitie_rl_s3 seed 1" in str(exc.value) or "seed 1" in str(exc.value)
    assert "differs from HEAD" in str(exc.value)


def test_cli_exits_2_with_run_lock(ecfg, monkeypatch, no_runner, capsys):
    registered(monkeypatch, dirty=True)
    monkeypatch.setattr(ev, "is_complete", lambda d: False)
    with pytest.raises(SystemExit) as exc:
        ev.main(["--config", str(REPO_ROOT / EVALUATION_CONFIG), "--run-missing"])
    assert exc.value.code == ev.RUN_LOCK_EXIT == 2
    assert "run lock" in capsys.readouterr().err


def test_lock_needs_a_planned_evaluation_seed(ecfg, monkeypatch):
    registered(monkeypatch, dirty=True)
    run_lock(ecfg, [], RESULTS_DIR)  # nothing planned
    run_lock(ecfg, [("pitie_rl_s3", 3210)], RESULTS_DIR)  # Part 8 dev seed
    run_lock(ecfg, [("pitie_rl_s3", 4000)], RESULTS_DIR)  # demo seed
    with pytest.raises(RunLockError, match="run lock"):
        run_lock(ecfg, [("pitie_rl_s3", 3210), ("pitie_fixed", 10)], RESULTS_DIR)
    with pytest.raises(RunLockError, match="run lock"):  # a folder inside data/results
        run_lock(ecfg, [("pitie_fixed", 1)], RESULTS_DIR / "sub")
    with pytest.raises(RunLockError, match="run lock"):  # any other results folder too
        run_lock(ecfg, [("pitie_rl_s3", 1)], Path("C:/tmp/elsewhere") / "results")


def test_lock_opens_only_on_a_clean_registered_tree(ecfg, monkeypatch):
    planned = [("pitie_rl_s3", 1), ("pitie_amb_rl_s3_nopreempt", 10)]
    registered(monkeypatch, dirty=False)
    run_lock(ecfg, planned, RESULTS_DIR)  # the registered Part 8 step (I2)
    registered(monkeypatch, dirty=False, commit=None)
    with pytest.raises(RunLockError, match="no commit adds a06"):
        run_lock(ecfg, planned, RESULTS_DIR)
    registered(monkeypatch, dirty=False)
    no_a06 = {k: v for k, v in ecfg.items() if k != "a06"}
    with pytest.raises(RunLockError, match="no a06 block"):
        run_lock(no_a06, planned, RESULTS_DIR)


def test_tmp_results_are_locked_on_seeds_1_to_10(ecfg, monkeypatch, tmp_path):
    """Review fix: a tmp results folder is no exemption. S3 on seeds 1-10 into tmp would
    show the A-06 outcomes before the registration commit; only other seeds pass."""
    registered(monkeypatch, dirty=True, commit=None)
    with pytest.raises(RunLockError, match="refusing 10 run"):
        run_lock(ecfg, [("pitie_rl_s3", s) for s in range(1, 11)], tmp_path / "results")
    run_lock(ecfg, [("pitie_rl_s3", s) for s in range(3210, 3230)], tmp_path / "results")
    registered(monkeypatch, dirty=False)  # the clean registration commit opens it anywhere
    run_lock(ecfg, [("pitie_rl_s3", s) for s in range(1, 11)], tmp_path / "results")


def test_a_config_copy_with_other_seeds_does_not_unlock_seeds_1_to_10(ecfg, monkeypatch):
    """Review fix: the locked seeds are the canonical configs/evaluation.yaml seeds,
    read independently of --config, united with those of the evaluated config."""
    registered(monkeypatch, dirty=True)
    copy_cfg = copy.deepcopy(ecfg)
    copy_cfg["seeds"]["evaluation"] = [3210, 3211]
    assert ev.locked_seeds(copy_cfg) == set(range(1, 11)) | {3210, 3211}
    for where in (RESULTS_DIR, Path("C:/tmp/elsewhere")):
        with pytest.raises(RunLockError, match="pitie_rl_s3 seed 1"):
            run_lock(copy_cfg, [("pitie_rl_s3", 1)], where)
        with pytest.raises(RunLockError, match="pitie_rl_s3 seed 3210"):
            run_lock(copy_cfg, [("pitie_rl_s3", 3210)], where)  # its own seeds stay locked
    no_seeds = {k: v for k, v in ecfg.items() if k != "seeds"}
    with pytest.raises(RunLockError):
        run_lock(no_seeds, [("pitie_fixed", 7)], RESULTS_DIR)


def test_locked_seeds_fail_closed_when_the_canonical_config_is_unreadable(ecfg, monkeypatch):
    def broken(path, required=()):
        raise OSError("unreadable")

    monkeypatch.setattr(ev, "load_config", broken)
    assert ev.locked_seeds({"seeds": {"evaluation": [3210]}}) == set(range(1, 11)) | {3210}


def test_cli_into_tmp_with_the_default_seeds_exits_2(monkeypatch, no_runner, tmp_path, capsys):
    """The dev command of plan section 5 with a forgotten --seeds: `--results <tmp>
    --run-missing` plans every family on seeds 1-10 and must launch nothing."""
    registered(monkeypatch, dirty=True, commit=None)
    monkeypatch.setattr(ev, "is_complete", lambda d: False)
    with pytest.raises(SystemExit) as exc:
        ev.main(["--config", str(REPO_ROOT / EVALUATION_CONFIG), "--results",
                 str(tmp_path / "results"), "--run-missing"])  # fmt: skip
    assert exc.value.code == ev.RUN_LOCK_EXIT
    err = capsys.readouterr().err
    assert "run lock" in err and "seed 1" in err


def test_git_head_reports_a_hash_and_a_flag():
    head, dirty = ev.git_head()
    assert isinstance(head, str) and head and isinstance(dirty, bool)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_registration_commit_is_the_first_commit_adding_a06(tmp_path):
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(tmp_path)]
    subprocess.run([*git, "init", "-q"], check=True)
    cfg = tmp_path / "evaluation.yaml"
    cfg.write_text("preregistered: 2026-10-12\n", encoding="utf-8")
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "prereg"], check=True)
    assert registration_commit(cfg) is None
    cfg.write_text("preregistered: 2026-10-12\na06: {candidate: S3}\n", encoding="utf-8")
    subprocess.run([*git, "commit", "-q", "-am", "register"], check=True)
    first = registration_commit(cfg)
    assert first is not None and set(first) == {"commit", "committed_at"}
    cfg.write_text("preregistered: 2026-10-12\na06: {candidate: S3, n: 10}\nx: 1\n", "utf-8")
    subprocess.run([*git, "commit", "-q", "-am", "later edit"], check=True)
    assert registration_commit(cfg) == first  # the first commit, not the last
    assert registration_commit(tmp_path / "missing" / "evaluation.yaml") is None


# -- the registered config -------------------------------------------------------------
def test_part8_keys_are_registered(ecfg):
    main, probe = ecfg["families"]["main"], ecfg["families"]["ambulance_probe"]
    assert set(main["scenarios"]) == {"S0", "S1", "S2", "S2_noP", "S3", "S4"}
    assert set(probe["scenarios"]) == {"S0", "S2", "S2_noP", "S3", "S4"}
    assert {(p["a"], p["b"], p["role"]) for p in main["pairs"]} >= {
        ("S0", "S3", "confirmatory"), ("S1", "S3", "confirmatory"),
        ("S2", "S3", "confirmatory"), ("S4", "S3", "exploratory"),
        ("S2_noP", "S4", "exploratory"),
    }  # fmt: skip
    a06 = ecfg["a06"]
    assert (a06["candidate"], a06["primary"], a06["comparisons"]) == (
        "S3",
        "S1",
        ["S1", "S2", "S0"],
    )
    assert a06["n_required"] == 10 and a06["status"] == "procedural"
    assert ecfg["final_table"]["columns"] == ["S0", "S1", "S2", "S3"]
    assert [c["date"].isoformat() for c in ecfg["changes"][-3:]] == ["2026-10-23"] * 3
    assert len(ecfg["changes"]) == 9


def test_frozen_s3_inputs_match_their_pins(ecfg):
    """D100: config_hash does not cover the text of rl.config, so a06 pins it."""
    model = ecfg["a06"]["model"]
    s3 = load_scenario(REPO_ROOT / model["sha256_from"])
    assert s3["rl"]["config"] == "configs/rl7/v1.yaml"
    for path, sha8 in model["inputs_text_sha8"].items():
        # str(): YAML reads the all-digit pin 96113433 (configs/rl7.yaml) as an int
        assert text_sha256(REPO_ROOT / path)[:8] == str(sha8), path


@pytest.mark.parametrize(
    ("path", "scenario", "chash", "dhash", "preempt"),
    [
        (PITIE_RL_S3_CONFIG, "pitie_rl_s3", "8dac6ad90ea8", "83a4f220fd20", True),
        (PITIE_RL_S4_CONFIG, "pitie_rl_s3_nopreempt", "a4127fffba7c", "83a4f220fd20", False),
        (PITIE_AMB_RL_S3_CONFIG, "pitie_amb_rl_s3", "26e4a3d48e3f", "a54278790e5a", True),
        (PITIE_AMB_RL_S4_CONFIG, "pitie_amb_rl_s3_nopreempt", "46f986aba8fd", "a54278790e5a",
         False),
    ],
)  # fmt: skip
def test_s3_s4_configs_hash_as_registered(path, scenario, chash, dhash, preempt):
    """D93 / lead check 2.1: the S4 and probe configs are S3 plus one change each; same
    demand as their family (S0 83a4f220fd20, probe a54278790e5a)."""
    cfg = load_scenario(REPO_ROOT / path)
    assert cfg["scenario"] == scenario and cfg["controller"] == "rl"
    assert config_hash(cfg)[:12] == chash and demand_hash(cfg)[:12] == dhash
    assert bool(cfg["preemption"]["enabled"]) is preempt
    s3 = load_scenario(REPO_ROOT / PITIE_RL_S3_CONFIG)
    assert cfg["rl"] == s3["rl"]  # the same frozen policy and inputs


def test_sections_are_the_d99_list():
    titles = [s.split(". ", 1)[1] for s in SECTIONS]
    assert [int(s.split(".")[0]) for s in SECTIONS] == list(range(1, 16))
    assert titles[:3] == [
        "Final comparison table (Part 8)", "A-06: S3 against the baselines (Part 8)",
        "Spec 10.6 check on seeds 1-10 (Part 8)",
    ]  # fmt: skip
    assert titles[12] == "Acceptance: A-03 to A-06" and titles[-1] == "Limitations"
    assert PART8_TITLE.startswith("# Evaluation: S0 to S4") and "(Parts 4 and 8)" in PART8_TITLE


# -- A-06 on a synthetic family (real configs, synthetic runs) --------------------------
SEEDS = list(range(1, 11))
BASE = np.array([50.0, 52.0, 49.0, 51.0, 53.0, 48.0, 50.0, 52.0, 51.0, 49.0])
CONFIGS = {"S0": "pitie_fixed", "S1": "pitie_actuated", "S2": "pitie_rules", "S3": "pitie_rl_s3"}
OTHER = {  # (user, kpi) -> value of every label (the 10.6 / equity KPIs)
    ("car", "time_loss_mean"): 60.0, ("bus", "delay_mean"): 50.0,
    ("pedestrian", "ped_curb_wait_mean"): 20.0, ("pedestrian", "ped_curb_wait_max"): 80.0,
    ("ambulance", "travel_time_total"): 100.0,
}  # fmt: skip
STARTED = "2026-10-24T08:00:00+00:00"


@pytest.fixture(scope="module")
def cfgs() -> dict[str, dict]:
    return {x: load_scenario(REPO_ROOT / "configs" / f"{c}.yaml") for x, c in CONFIGS.items()}


def kpi_frame(delay, other: dict | None = None, seeds=SEEDS) -> pd.DataFrame:
    """seed, user_type, kpi, value: the person delay `delay` (one value per seed) and
    the OTHER KPIs (overridden by `other`), with a 1 % spread across seeds."""
    rows = []
    for i, s in enumerate(seeds):
        rows.append({"seed": s, "user_type": "all", "kpi": "person_delay_mean",
                     "value": float(np.asarray(delay, float)[i])})  # fmt: skip
        for (u, k), v in {**OTHER, **(other or {})}.items():
            val = v[i] if isinstance(v, list | np.ndarray) else v * (1 + 0.001 * i)
            rows.append({"seed": s, "user_type": u, "kpi": k, "value": float(val)})
    return pd.DataFrame(rows)


def run_meta(cfg: dict, seed: int, **over) -> dict:
    from traffic_twin.evaluation.runs import expected_hashes
    from traffic_twin.sim.audit import DANGER_KEYS

    meta = {
        "run_id": f"run-{cfg['scenario']}-{seed}",
        "seed": seed,
        "scenario": cfg["scenario"],
        "started_at": STARTED,
        **expected_hashes(cfg),
        "health": {"collisions": 0, "teleports": 0},
        "safety": {"audit": dict.fromkeys(DANGER_KEYS, 0), "interventions": {}},
        "preemption": {"requests": 1, "served": 1, "timeouts": 0},
        "control": (
            {"model_sha256": cfg["rl"]["model_sha256"], "model_version": "rl7-test",
             "decisions": 800, "advances": 270,
             "iv_split": {"cut": 20, "replaced": 1, "cut_foreseeable": 19}}
            if cfg["controller"] == "rl"
            else {}
        ),
    }  # fmt: skip
    return {**meta, **over}


def a06_family(cfgs, delays: dict, other: dict | None = None, seeds=SEEDS, metas=None):
    """Family `main` of S0-S3 (the repo configs) with the given person delays."""
    from traffic_twin.evaluation.runs import Runs

    members = {}
    for label, cfg in cfgs.items():
        m = ev.Member(label, Path(f"configs/{CONFIGS[label]}.yaml"), cfg, Path("."))
        meta = {s: run_meta(cfg, s) for s in seeds}
        meta.update((metas or {}).get(label, {}))
        m.runs = Runs(cfg["scenario"], list(seeds), meta)
        members[label] = m
    kpis = {x: kpi_frame(delays[x], (other or {}).get(x), seeds) for x in cfgs}
    return ev.Family("main", "S0", [], members, "83a4f220fd20", kpis, list(seeds))


A03_OK = {"n": 4, "of": 4, "text": "4/4"}


def report(fam, ecfg, a03=A03_OK, problems=(), tmp=None):
    return final.a06_report(
        {"main": fam}, ecfg, a03, model_problems=list(problems),
        verdict_path=(tmp or Path("no-such-dir")) / "verdict.json",
    )  # fmt: skip


def test_a06_exact_p_at_n_10_holm_and_the_three_outcomes(ecfg, cfgs):
    """S3 below S1 and S0 on every seed (exact two-sided p = 2/1024 = 0.00195), above
    S2 on every seed: PASS, outcomes better / worse / better, Holm 3 x 2/1024."""
    delays = {"S0": BASE + 10, "S1": BASE + 5, "S2": BASE - 2, "S3": BASE}
    a = report(a06_family(cfgs, delays), ecfg)
    assert a["status"] == "PASS" and a["problems"] == []
    by = {c["reference"]: c for c in a["comparisons"]}
    assert [c["reference"] for c in a["comparisons"]] == ["S1", "S2", "S0"]
    for ref in ("S1", "S2", "S0"):
        assert by[ref]["n"] == 10 and by[ref]["p"] == pytest.approx(2 / 1024)
        assert by[ref]["p"] == pytest.approx(0.001953125)
        assert by[ref]["p_holm"] == pytest.approx(3 * 2 / 1024)
    assert (by["S1"]["hl"], by["S2"]["hl"], by["S0"]["hl"]) == pytest.approx((-5, 2, -10))
    assert (by["S1"]["outcome"], by["S2"]["outcome"]) == ("better", "worse")
    assert by["S1"]["outcome_text"] == "S3 better" and by["S2"]["outcome_text"] == "S3 worse"
    assert by["S1"]["mean_ref"] == pytest.approx(BASE.mean() + 5)
    assert by["S1"]["sd_s3"] == pytest.approx(BASE.std(ddof=1))
    assert a["primary_sentence"].startswith("S3 vs S1 (actuated): all/person_delay_mean")
    assert a["primary_sentence"].endswith(": S3 better.")
    assert a["model"]["model_sha256"] == cfgs["S3"]["rl"]["model_sha256"]
    assert a["model"]["inputs_ok"] and set(a["model"]["inputs"]) == set(
        ecfg["a06"]["model"]["inputs_text_sha8"]
    )


def test_a06_unfavourable_result_is_still_pass(ecfg, cfgs):
    """D94: the status is procedural; S3 worse than every reference is presented."""
    delays = {"S0": BASE - 10, "S1": BASE - 5, "S2": BASE - 1, "S3": BASE}
    a = report(a06_family(cfgs, delays), ecfg)
    assert a["status"] == "PASS"
    assert [c["outcome_text"] for c in a["comparisons"]] == ["S3 worse"] * 3
    assert "FAIL" not in json.dumps(a)


def test_a06_holm_follows_the_p_order_and_no_difference_is_none(ecfg, cfgs):
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1.0, 10)
    s1 = BASE + 5
    s0 = BASE + np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, -0.5])  # S3 lower on 9 of 10 seeds
    delays = {"S0": s0, "S1": s1, "S2": BASE + noise, "S3": BASE}
    a = report(a06_family(cfgs, delays), ecfg)
    ps = [c["p"] for c in a["comparisons"]]
    from traffic_twin.evaluation.stats import holm

    assert [c["p_holm"] for c in a["comparisons"]] == pytest.approx(list(holm(ps)))
    by = {c["reference"]: c for c in a["comparisons"]}
    assert by["S1"]["p"] < by["S0"]["p"] < by["S2"]["p"]
    assert by["S2"]["p_holm"] >= 0.05 and by["S2"]["outcome"] == "none"
    assert by["S2"]["outcome_text"] == "no significant difference"
    assert a["status"] == "PASS"


def test_a06_zero_variance_differences(ecfg, cfgs):
    """d identical to 0 on every seed: p 1, HL 0, no difference; d constant -5: exact
    p 2/1024 and a degenerate CI [-5, -5]."""
    delays = {"S0": BASE + 1, "S1": BASE + 5, "S2": BASE.copy(), "S3": BASE}
    a = report(a06_family(cfgs, delays), ecfg)
    by = {c["reference"]: c for c in a["comparisons"]}
    assert by["S2"]["p"] == 1.0 and by["S2"]["hl"] == 0 and by["S2"]["outcome"] == "none"
    assert (by["S1"]["hl"], by["S1"]["ci_low"], by["S1"]["ci_high"]) == pytest.approx((-5,) * 3)
    assert by["S1"]["p"] == pytest.approx(2 / 1024) and a["status"] == "PASS"


def test_a06_with_n_9_is_invalid(ecfg, cfgs):
    delays = {"S0": BASE + 10, "S1": BASE + 5, "S2": BASE + 1, "S3": BASE}
    fam = a06_family(cfgs, delays)
    k = fam.kpis["S3"]
    fam.kpis["S3"] = k[~((k["kpi"] == "person_delay_mean") & (k["seed"] == 10))]
    a = report(fam, ecfg)
    assert a["status"] == "INVALID" and all(c["n"] == 9 for c in a["comparisons"])
    assert "S3 vs S1: 9 paired seeds of 10 evaluated, 10 required" in a["problems"]
    nine = a06_family(cfgs, {x: v[:9] for x, v in delays.items()}, seeds=SEEDS[:9])
    b = report(nine, ecfg)  # the evaluated seeds are 9 (e.g. a mini run on 6 dev seeds)
    assert b["status"] == "INVALID" and b["comparisons"][0]["n"] == 9


def test_a06_invalid_on_pins_stale_runs_a03_and_another_model(ecfg, cfgs, tmp_path):
    delays = {"S0": BASE + 10, "S1": BASE + 5, "S2": BASE + 1, "S3": BASE}
    fam = a06_family(cfgs, delays)
    bad = copy.deepcopy(ecfg)
    bad["a06"]["model"]["inputs_text_sha8"]["configs/rl.yaml"] = "deadbeef"
    a = final.a06_report({"main": fam}, bad, A03_OK, verdict_path=tmp_path / "v.json")
    assert a["status"] == "INVALID" and not a["model"]["inputs_ok"]
    assert any(p.startswith("D100 frozen S3 input: configs/rl.yaml: text sha8 ") and
               p.endswith("!= pinned deadbeef") for p in a["problems"])  # fmt: skip
    assert report(fam, ecfg, a03={"n": 3, "of": 4, "text": "3/4"})["status"] == "INVALID"
    stale = a06_family(cfgs, delays, metas={"S3": {4: run_meta(cfgs["S3"], 4, config_hash="x")}})
    s = report(stale, ecfg)
    assert (
        s["status"] == "INVALID"
        and "S3 runs of seeds [4] are not of the current config" in (s["problems"])
    )
    other = run_meta(cfgs["S3"], 2, control={"model_sha256": "f" * 64})
    o = report(a06_family(cfgs, delays, metas={"S3": {2: other}}), ecfg)
    assert "S3 runs of seeds [2] report another model_sha256" in o["problems"]
    missing = report(a06_family({k: v for k, v in cfgs.items() if k != "S2"},
                                {k: v for k, v in delays.items() if k != "S2"}), ecfg)  # fmt: skip
    assert missing["status"] == "INVALID" and missing["comparisons"][1]["n"] == 0
    gone = final.a06_report({}, ecfg, A03_OK, verdict_path=tmp_path / "v.json")
    assert gone["status"] == "INVALID" and "family main was not evaluated" in gone["problems"]


def test_check_model_inputs(ecfg, tmp_path):
    assert final.check_model_inputs(ecfg) == []  # the registered pins (D100)
    assert final.check_model_inputs({"seeds": {}}) == []  # a Part 4 config: nothing pinned
    bad = copy.deepcopy(ecfg)
    pins = bad["a06"]["model"]["inputs_text_sha8"]
    pins["configs/rl7/v1.yaml"] = "00000000"
    pins["configs/rl7/nope.yaml"] = "12345678"
    del pins["configs/rl7.yaml"]
    msgs = final.check_model_inputs(bad)
    assert any("configs/rl7/v1.yaml: text sha8 10ac6ce8 != pinned 00000000" in m for m in msgs)
    assert "configs/rl7/nope.yaml: missing (pinned text sha8 12345678)" in msgs
    unpinned = copy.deepcopy(ecfg)
    unpinned["a06"]["model"]["inputs_text_sha8"].pop("configs/rl7/v1.yaml")
    assert any("rl.config configs/rl7/v1.yaml is not pinned" in m
               for m in final.check_model_inputs(unpinned))  # fmt: skip
    rec = final.model_inputs(ecfg)["configs/rl7.yaml"]
    assert rec == {"pinned": "96113433", "actual": "96113433", "ok": True}  # int pin read as str


# -- registration order (D92), second resolution ---------------------------------------
@pytest.mark.parametrize(
    ("committed", "started", "order"),
    [
        ("2026-10-23T18:00:00+02:00", "2026-10-23T16:00:01+00:00", "before"),
        ("2026-10-23T18:00:00+02:00", "2026-10-23T16:00:00.900000+00:00", "after"),  # same second
        ("2026-10-23T18:00:00+02:00", "2026-10-23T15:59:59+00:00", "after"),
        (None, "2026-10-23T16:00:01+00:00", "n/a"),
        ("2026-10-23T18:00:00+02:00", None, "n/a"),
    ],
)
def test_registration_order_at_second_resolution(committed, started, order):
    assert final.registration_order(committed, started) == order


def test_registration_record_uses_the_first_rl_run(cfgs):
    delays = {"S0": BASE + 10, "S1": BASE + 5, "S2": BASE + 1, "S3": BASE}
    early = run_meta(cfgs["S0"], 1, started_at="2026-10-01T00:00:00+00:00")  # not an rl run
    first = run_meta(cfgs["S3"], 7, started_at="2026-10-23T17:30:00.5+00:00")
    fam = a06_family(cfgs, delays, metas={"S0": {1: early}, "S3": {7: first}})
    rec = final.registration_record({"main": fam}, REGISTERED)
    assert rec["first_s3_started_at"] == "2026-10-23T17:30:00.5+00:00"
    assert rec["commit"] == "abc1234" and rec["order"] == "before"  # 16:00 UTC < 17:30 UTC
    assert final.registration_record({"main": fam}, None)["order"] == "n/a"


# -- 10.6 replication (D95) ------------------------------------------------------------
def replication(ecfg, fam, tmp_path, verdict: dict | None = None):
    path = tmp_path / "rl7" / "devtest" / "verdict.json"
    if verdict is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(verdict), encoding="utf-8")
    return final.replication_10_6({"main": fam}, ecfg, path)


def test_replication_c2_edge_is_5_percent_and_confirms_fallback(ecfg, cfgs, tmp_path):
    delays = {"S0": BASE + 10, "S1": BASE + 5, "S2": BASE + 1, "S3": BASE}
    ped = ("pedestrian", "ped_curb_wait_mean")
    at = {"S1": {ped: [20.0] * 10}, "S3": {ped: [21.0] * 10}}  # exactly +5 %: not worse
    r = replication(ecfg, a06_family(cfgs, delays, at), tmp_path)
    assert r["c1"]["pass"] and r["c2"]["pass"] and r["c3"]["pass"]
    assert r["c4"]["cited"] == "not available" and not r["confirms_fallback"]
    assert r["text"].startswith("Seeds 1-10 do not confirm the interim FALLBACK")
    assert "The main solution stays S2" in r["text"]
    over = {"S1": {ped: [20.0] * 10}, "S3": {ped: [21.02] * 10}}  # +5.1 %
    r = replication(ecfg, a06_family(cfgs, delays, over), tmp_path)
    row = next(x for x in r["c2"]["rows"] if x["kpi"] == "pedestrian/ped_curb_wait_mean")
    assert row["worse"] and row["rel_pct"] == pytest.approx(5.1) and not r["c2"]["pass"]
    assert r["confirms_fallback"] and r["main_solution"] == "S2"
    assert "confirm the interim FALLBACK (C2 fails on pedestrian/ped_curb_wait_mean)" in r["text"]


def test_replication_c1_needs_hl_below_0_and_p_below_alpha(ecfg, cfgs, tmp_path):
    mixed = BASE + np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 0.5])  # S3 slightly lower, n.s.
    delays = {"S0": BASE + 10, "S1": mixed, "S2": BASE + 1, "S3": BASE}
    r = replication(ecfg, a06_family(cfgs, delays), tmp_path)
    assert r["c1"]["p"] > 0.05 and not r["c1"]["pass"] and r["confirms_fallback"]
    assert "C1 fails" in r["text"]
    worse = replication(ecfg, a06_family(cfgs, {**delays, "S1": BASE - 3}), tmp_path)
    assert worse["c1"]["hl"] > 0 and worse["c1"]["p"] < 0.05 and not worse["c1"]["pass"]


def test_replication_c3_edges(ecfg, cfgs, tmp_path):
    delays = {"S0": BASE + 10, "S1": BASE + 5, "S2": BASE + 1, "S3": BASE}
    curb = ("pedestrian", "ped_curb_wait_max")
    s3 = cfgs["S3"]
    ok = replication(ecfg, a06_family(cfgs, delays, {"S3": {curb: [90.0] * 10}}), tmp_path)
    assert ok["c3"]["violations"] == []  # 90 s is the bound: allowed
    metas = {
        "S3": {
            2: run_meta(s3, 2, safety={"audit": {**run_meta(s3, 2)["safety"]["audit"],
                                                 "intergreen_short": 1}}),
            3: run_meta(s3, 3, health={"collisions": None, "teleports": 0}),
            4: run_meta(s3, 4, preemption={"requests": 2, "served": 1, "timeouts": 0}),
            5: run_meta(s3, 5, preemption={"requests": 1, "served": 0, "timeouts": 1}),
            6: run_meta(s3, 6, preemption={"requests": 1, "served": 0, "timeouts": 1}),
        },
        "S2": {6: run_meta(cfgs["S2"], 6, preemption={"requests": 1, "served": 0,
                                                       "timeouts": 1})},
    }  # fmt: skip
    over = {"S3": {curb: [80.0] * 9 + [90.5]}}
    r = replication(ecfg, a06_family(cfgs, delays, over, metas=metas), tmp_path)
    v = r["c3"]["violations"]
    assert "seed 2: danger 1" in v and "seed 3: collisions not reported" in v
    assert "seed 4: preemption: 1 of 2 requests unresolved" in v
    assert "seed 10: curb max 90.5 s > 90 s" in v
    assert "preemption timeouts 2 > S2's 1 on the same seeds" in v
    assert not any(x.startswith("seed 5:") for x in v)  # a time-out resolves its request
    assert not r["c3"]["pass"] and r["confirms_fallback"]


def test_replication_cites_c4_with_the_reproduce7_result(ecfg, cfgs, tmp_path):
    fam = a06_family(cfgs, {"S0": BASE + 10, "S1": BASE + 5, "S2": BASE + 1, "S3": BASE})
    frozen = {"C1": {"pass": True}, "C2": {"pass": False}, "C3": {"pass": True},
              "C4": {"runs_below_s1": 5, "of": 5, "required": 4, "runs_pass": True,
                     "repro": {"status": "pending", "same": None}, "pass": None},
              "verdict": "fallback", "pending": ["C4.repro"]}  # fmt: skip
    r = replication(ecfg, fam, tmp_path, frozen)
    assert r["c4"]["cited"] == "pending" and r["c4"]["source"].endswith("verdict.json")
    repro = tmp_path / "rl7" / "_reproduce" / "abc1234" / "reproduce7.json"
    repro.parent.mkdir(parents=True)
    repro.write_text(json.dumps({"same": True, "head": "abc1234"}), encoding="utf-8")
    assert replication(ecfg, fam, tmp_path)["c4"]["cited"] == "pass"  # completed, D82
    repro.write_text(json.dumps({"same": False, "head": "abc1234"}), encoding="utf-8")
    assert replication(ecfg, fam, tmp_path)["c4"]["cited"] == "fail"


def test_replication_states_no_verdict_without_data(ecfg, cfgs, tmp_path):
    """Review fix: without family main (or without S3 / S1 in it) the 10.6 record is
    "not evaluated" with every check and confirms_fallback None, never a FALLBACK
    sentence built from absent data."""
    gone = final.replication_10_6({}, ecfg, tmp_path / "verdict.json")
    assert gone["status"] == "not evaluated" and gone["confirms_fallback"] is None
    assert [gone[c]["pass"] for c in ("c1", "c2", "c3")] == [None, None, None]
    assert "violations" not in gone["c3"] and gone["c2"]["rows"] == [] and gone["seeds"] == []
    assert gone["text"] == ("Spec 10.6 check not evaluated (family main was not evaluated): "
                            "no result is stated.")  # fmt: skip
    assert "FALLBACK" not in gone["text"] and "confirm" not in gone["text"]
    assert gone["c4"]["cited"] == "not available" and gone["main_solution"] == "S2"
    no_s1 = {k: v for k, v in cfgs.items() if k != "S1"}
    fam = a06_family(no_s1, {"S0": BASE + 10, "S2": BASE + 1, "S3": BASE})
    r = final.replication_10_6({"main": fam}, ecfg, tmp_path / "verdict.json")
    assert r["status"] == "not evaluated" and r["confirms_fallback"] is None
    assert "family main has no S1" in r["text"]
    summ = ev.replication_summary(r)
    assert summ["confirms_fallback"] is None and summ["c1"] is None
    full = a06_family(cfgs, {"S0": BASE + 10, "S1": BASE + 5, "S2": BASE + 1, "S3": BASE})
    assert final.replication_10_6({"main": full}, ecfg, tmp_path / "v.json")["status"] == (
        "evaluated"
    )


def test_partial_evaluation_keeps_the_d98_files(synthetic, tmp_path):
    """Review fix: `--family ambulance_probe` into the out folder of the full evaluation
    leaves a06.json, final_table.csv, series.csv and digest.json byte-identical, and
    states no 10.6 result; `--family main` rewrites a06.json only."""
    config, results, out, _ = synthetic
    shutil.copytree(out, tmp_path / "out")
    d98 = ("a06.json", "final_table.csv", "series.csv", "digest.json")
    before = {n: (tmp_path / "out" / n).read_bytes() for n in d98}
    docs = tmp_path / "evaluation.md"
    ev.main(["--config", str(config), "--results", str(results), "--out", str(tmp_path / "out"),
             "--family", "ambulance_probe", "--docs", str(docs)])  # fmt: skip
    assert {n: (tmp_path / "out" / n).read_bytes() for n in d98} == before
    summ = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    rep = summ["replication_10_6"]
    assert rep["confirms_fallback"] is None and "not evaluated" in rep["text"]
    page = docs.read_text(encoding="utf-8")
    assert "confirm the interim FALLBACK" not in page and "10.6 check not evaluated" in page
    (tmp_path / "out" / "a06.json").write_text("{}", encoding="utf-8")
    ev.main(["--config", str(config), "--results", str(results), "--out", str(tmp_path / "out"),
             "--family", "main"])  # fmt: skip
    a06 = json.loads((tmp_path / "out" / "a06.json").read_text(encoding="utf-8"))
    assert a06["family"] == "main" and a06["replication_10_6"]["status"] == "evaluated"
    for n in ("final_table.csv", "series.csv", "digest.json"):
        assert (tmp_path / "out" / n).read_bytes() == before[n], n


def test_equity_vs_flags_s3_against_s1_and_s2(ecfg, cfgs):
    delays = {"S0": BASE + 10, "S1": BASE + 5, "S2": BASE + 1, "S3": BASE}
    car = ("car", "time_loss_mean")
    fam = a06_family(cfgs, delays, {"S2": {car: [50.0] * 10}, "S3": {car: [53.0] * 10}})
    eq = final.equity_vs({"main": fam}, ecfg)
    assert list(eq) == ["S1", "S2"]
    keys = [(r["user"], r["kpi"]) for r in eq["S2"]]
    assert keys == [
        ("car", "time_loss_mean"), ("bus", "delay_mean"), ("pedestrian", "ped_curb_wait_max"),
        ("ambulance", "travel_time_total"), ("pedestrian", "ped_curb_wait_mean"),
    ]  # fmt: skip
    row = eq["S2"][0]
    assert row["rel_pct"] == pytest.approx(6.0) and row["worse"] and row["primary"]
    assert not eq["S1"][0]["worse"]  # 53 vs 60: better


# -- the full synthetic evaluation: outputs, page, figures -----------------------------
@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """The synthetic runs of test_evaluate (S0-S4 in both families) evaluated by the CLI:
    (config, results, out, page)."""
    from test_evaluate import synthetic_setup

    tmp = tmp_path_factory.mktemp("part8")
    config, results, _ = synthetic_setup(tmp)
    docs = tmp / "evaluation.md"
    ev.main(["--config", str(config), "--results", str(results), "--docs", str(docs)])
    return config, results, results / "evaluation", docs.read_text(encoding="utf-8")


def test_final_table_rows_columns_and_ablation_flags(synthetic, ecfg):
    _, _, out, _ = synthetic
    t = pd.read_csv(out / "final_table.csv")
    assert tuple(t.columns) == final.FINAL_TABLE_COLUMNS
    ft = ecfg["final_table"]
    main = t[t["section"] == "main"]
    assert list(dict.fromkeys(main["user"] + "/" + main["kpi"])) == ft["rows"]
    assert list(dict.fromkeys(main["label"])) == ["S0", "S1", "S2", "S3", "S2_noP", "S4"]
    assert set(main.loc[main["ablation"], "label"]) == {"S2_noP", "S4"}
    assert len(main) == len(ft["rows"]) * 6
    pd_rows = main[main["kpi"] == "person_delay_mean"].set_index("label")
    assert pd_rows.loc["S0", "rel_vs_s0_pct"] == 0 and pd_rows.loc["S3", "rel_vs_s0_pct"] < 0
    assert (pd_rows["n"] == 10).all() and (pd_rows["unit"] == "s").all()
    assert set(pd_rows["better_if"]) == {"lower"}
    probe = t[t["section"] == "probe"]
    assert list(dict.fromkeys(probe["user"] + "/" + probe["kpi"])) == ft["probe_rows"]
    assert list(dict.fromkeys(probe["label"])) == ["S0", "S2", "S3", "S2_noP", "S4"]
    assert {"safety", "activity", "equity", "occupancy"} <= set(t["section"])
    a04 = t[(t["section"] == "safety") & (t["kpi"] == "a04_pass")]
    assert (
        set(a04["label"]) == {"S0", "S1", "S2", "S3", "S2_noP", "S4"} and (a04["mean"] == 1).all()
    )
    occ = t[t["section"] == "occupancy"]
    assert set(occ["kpi"]) == {"person_delay_mean__occ_bus_20", "person_delay_mean__occ_bus_30"}
    served = t[t["kpi"] == "preemption_served"].set_index("label")
    assert served.loc["S3", "mean"] == 10  # one request served per run, 10 runs
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summ["final_available"] is True


def test_the_page_has_every_d99_section_and_unchanged_part4_tables(synthetic):
    """D99 headings in order, the Part 8 title; the S0-S2 pair blocks, A-04 rows and
    per-user columns byte-identical to the Part 4 render of the same synthetic runs
    (hashes taken before the Part 8 page existed)."""
    _, _, _, page = synthetic
    heads = [ln[3:] for ln in page.splitlines() if ln.startswith("## ")]
    assert heads == list(SECTIONS)
    assert page.startswith(PART8_TITLE + "\n")
    pairs, a04 = part4_blocks(page)
    assert pairs.count(" vs **") == 6 and "**S3**" not in pairs
    assert hashlib.sha256(pairs.encode()).hexdigest() == PART4_PAIRS_SHA256
    assert hashlib.sha256(a04.encode()).hexdigest() == PART4_A04_SHA256
    assert hashlib.sha256(per_user_part4(page).encode()).hexdigest() == PART4_PER_USER_SHA256
    for word in ("**A-06 PASS**", "**Primary result:** S3 vs S1 (actuated)", "| C2 |",
                 "Spec 10.6 check (reported, not gating)", "OUT OF DISTRIBUTION",
                 "**Power note (D96):**", "S3 against the other controllers (D96)",
                 "### A-06", "a06_forest.png",
                 "no commit adds `a06:` to the evaluation config yet"):  # fmt: skip
        assert word in page, word


def test_page_states_a06_invalid_and_d100_warnings(synthetic, tmp_path, monkeypatch):
    config, results, _, _ = synthetic
    raw = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    raw["a06"]["model"]["inputs_text_sha8"]["configs/rl.yaml"] = "deadbeef"
    bad = tmp_path / "evaluation.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    docs = tmp_path / "evaluation.md"
    ev.main(["--config", str(bad), "--results", str(results), "--out", str(tmp_path / "out"),
             "--docs", str(docs)])  # fmt: skip
    page = docs.read_text(encoding="utf-8")
    assert "**A-06 INVALID**" in page and "**Warning (D100):** configs/rl.yaml: text sha8" in page
    a06 = json.loads((tmp_path / "out" / "a06.json").read_text(encoding="utf-8"))
    assert a06["status"] == "INVALID" and not a06["model"]["inputs_ok"]


def test_activity_reports_rl_counters_and_starvation_runs(cfgs):
    from traffic_twin.evaluation.runs import Runs

    s3 = cfgs["S3"]
    meta = {s: run_meta(s3, s) for s in (1, 2, 3)}
    meta[2]["safety"]["interventions"] = {"starvation": 2, "ped_wait": 5}
    m = ev.Member("S3", Path("s3.yaml"), s3, Path("."))
    m.runs = Runs("pitie_rl_s3", [1, 2, 3], meta)
    fam = ev.Family("main", "S3", [], {"S3": m})
    row = ev.activity(fam, ev.intervention_keys()).iloc[0]
    assert row["rl.decisions"] == 2400 and row["rl.advances"] == 810
    assert row["rl.iv_split.cut"] == 60 and row["rl.iv_split.cut_foreseeable"] == 57
    assert row["guard.starvation"] == 2 and row["guard_runs.starvation"] == 1


def test_digest_masks_the_run_id(tmp_path, synthetic):
    _, results, out, _ = synthetic
    dig = json.loads((out / "digest.json").read_text(encoding="utf-8"))
    assert dig["n"] == 110 and len(dig["runs"]) == 110 and dig["mask"] == "run_id -> <run_id>"
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"run_id,seed,kpi,value\nuuid-1,1,x,2.5\n")  # bytes: no CRLF translation
    b.write_bytes(b"run_id,seed,kpi,value\nuuid-2,1,x,2.5\n")
    assert final.masked_sha256(a, "uuid-1") == final.masked_sha256(b, "uuid-2")
    assert final.masked_sha256(a, None) != final.masked_sha256(b, None)
    masked = hashlib.sha256(b"run_id,seed,kpi,value\n<run_id>,1,x,2.5\n").hexdigest()
    assert final.masked_sha256(a, "uuid-1") == masked
    name = "pitie_rl_s3_seed3"
    run_id = json.loads((results / name / "run.json").read_text(encoding="utf-8"))["run_id"]
    assert dig["runs"][name] == final.masked_sha256(results / name / "kpi_run.csv", run_id)
    lines = "".join(f"{k} {v}\n" for k, v in sorted(dig["runs"].items()))
    assert dig["combined"] == hashlib.sha256(lines.encode()).hexdigest()


def test_series_bins_means_and_quantiles(tmp_path, ecfg):
    """60 s bins of halting_veh per approach (mean over the bin's samples), `all` the
    sum over approaches, then mean / p10 / p90 over the seeds."""
    from traffic_twin.evaluation.runs import Runs

    cfg = {"scenario": "sc", "controller": "fixed"}
    m = ev.Member("S0", Path("s0.yaml"), cfg, tmp_path)
    seeds = [1, 2, 3, 4, 5]
    for s in seeds:
        d = tmp_path / f"sc_seed{s}"
        d.mkdir()
        rows = [{"run_id": "r", "t": t, "approach": a, "jam_veh": 0, "jam_m": 0.0,
                 "halting_veh": s * (1 if a == "N" else 2) + (t >= 60)}
                for t in (0.0, 30.0, 60.0, 90.0) for a in ("N", "S")]  # fmt: skip
        pd.DataFrame(rows).to_parquet(d / "queue.parquet", index=False)
    (tmp_path / "sc_seed6").mkdir()  # a run without queue.parquet is left out
    m.runs = Runs("sc", seeds, {})
    fam = ev.Family("main", "S0", [], {"S0": m}, seeds=[*seeds, 6])
    t = final.series(fam, ecfg)
    assert tuple(t.columns) == final.SERIES_COLUMNS
    assert set(t["approach"]) == {"N", "S", "all"} and set(t["t_s"]) == {0, 60}
    n0 = t[(t["approach"] == "N") & (t["t_s"] == 0)].iloc[0]
    assert n0["mean"] == pytest.approx(3.0) and n0["n_seeds"] == 5  # seeds 1..5
    assert n0["p10"] == pytest.approx(np.quantile([1, 2, 3, 4, 5], 0.1))
    assert n0["p90"] == pytest.approx(np.quantile([1, 2, 3, 4, 5], 0.9))
    all60 = t[(t["approach"] == "all") & (t["t_s"] == 60)].iloc[0]
    assert all60["mean"] == pytest.approx(np.mean([3 * s + 2 for s in seeds]))
    assert list(t["approach"].drop_duplicates()) == ["N", "S", "all"]
    assert set(t["family"]) == {"main"} and set(t["label"]) == {"S0"}


# -- config checks ---------------------------------------------------------------------
def _bad_config(tmp_path, edit) -> Path:
    raw = yaml.safe_load((REPO_ROOT / EVALUATION_CONFIG).read_text(encoding="utf-8"))
    for fam in raw["families"].values():
        fam["scenarios"] = {k: (REPO_ROOT / v).as_posix() for k, v in fam["scenarios"].items()}
    edit(raw)
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_load_evaluation_checks_the_part8_blocks(tmp_path):
    def edit(raw):
        raw["a06"]["primary"] = "S9"
        raw["a06"]["comparisons"] = ["S1", "S2", "S9"]
        raw["a06"]["candidate"] = "S2"  # a rules scenario
        raw["a06"]["outcome"].pop("none")
        raw["final_table"]["rows"].append("car/no_such_kpi")
        raw["final_table"]["ablations"].append("S7")
        raw["final_table"]["extra"].append("colour")
        raw["part8"]["series"]["quantiles"] = [0.9, 0.1]

    with pytest.raises(ConfigError) as err:
        load_evaluation(_bad_config(tmp_path, edit))
    msg = str(err.value)
    for part in ("a06.candidate S2", "must be an rl scenario", "'S9' is not a scenario",
                 "a06.outcome needs better, worse, none", "unknown KPI 'no_such_kpi'",
                 "final_table.ablations: 'S7'", "final_table.extra: unknown ['colour']",
                 "part8.series.quantiles"):  # fmt: skip
        assert part in msg, part
    ok = load_evaluation(_bad_config(tmp_path, lambda raw: None))
    assert ok["a06"]["candidate"] == "S3"

    def without_part8(raw):
        for key in ("a06", "final_table", "part8"):
            raw.pop(key)

    assert "a06" not in load_evaluation(_bad_config(tmp_path, without_part8))  # Part 4 config


# -- pre-change hashes of the Part 4 tables (synthetic runs of test_evaluate) -----------
PART4_PAIRS_SHA256 = "cbe797cba1c2ac2b62126507a758237128a7e764a50e4fbf7f4bcb62f2694899"
PART4_A04_SHA256 = "76ba2c4a9bf3c47b7046730ee1120a44442498610bb14abe77350d1182f72223"
PART4_PER_USER_SHA256 = "58a67d0ec544126c7d3895b213a51976ad93ac77a61dd76f15e1d45d62a73aa6"


def part4_blocks(page: str) -> tuple[str, str]:
    """(the S0-S2 pair blocks, the S0-S2 rows of the A-04 table) of an evaluation page,
    in page order."""
    paras = [x.strip("\n") for x in page.split("\n\n")]  # a block may follow 2 blank lines
    out, i = [], 0
    while i < len(paras):
        p = paras[i]
        if p.startswith("**") and ") vs **" in p and ", family `" in p:
            head = p.split(", family")[0]
            block = [p]
            while i + 1 < len(paras) and paras[i + 1].startswith(("|", "Other KPIs", "- ")):
                i += 1
                block.append(paras[i])
            if "**S3**" not in head and "**S4**" not in head:
                out.append("\n\n".join(block))
        i += 1
    rows = [
        ln
        for ln in page.splitlines()
        if ln.startswith(("| main | `pitie_", "| ambulance_probe | `pitie_")) and "rl_s3" not in ln
    ]
    return "\n\n".join(out), "\n".join(rows)


def per_user_part4(page: str) -> str:
    """The S0-S2 columns of the per-user tables (the first 4 / 3 labels)."""
    keep = {"main": 4, "ambulance_probe": 3}
    out, fam = [], None
    for ln in page.splitlines():
        if ln.startswith("Family `") and ": mean ± sd over the seeds" in ln:
            fam = ln.split("`")[1]
            continue
        if fam and ln.startswith("| ") and "±" in ln:
            cells = ln.strip("|").split(" | ")
            out.append(f"{fam}|" + "|".join(c.strip() for c in cells[: 3 + keep[fam]]))
        elif fam and ln and not ln.startswith("|"):
            fam = None
    return "\n".join(out)


def test_every_rl_scenario_uses_the_pinned_inputs(ecfg, cfgs):
    """D100 for each rl scenario of the families (S3, S4, probe versions), not only the
    pinned config: its rl.config chain is pinned and its model is the pinned one."""
    delays = {"S0": BASE + 10, "S1": BASE + 5, "S2": BASE + 1, "S3": BASE}
    fam = a06_family(cfgs, delays)
    assert final.rl_scenario_problems({"main": fam}, ecfg) == []
    other = copy.deepcopy(cfgs["S3"])
    other["rl"] = {**other["rl"], "config": "configs/rl7/v2.yaml", "model_sha256": "0" * 64}
    fam.members["S3"].cfg = other
    msgs = final.rl_scenario_problems({"main": fam}, ecfg)
    assert msgs == [
        "main/S3 (pitie_rl_s3): rl.config configs/rl7/v2.yaml is not pinned",
        "main/S3 (pitie_rl_s3): rl.model_sha256 is not the pinned model",
    ]
    assert final.rl_scenario_problems({"main": fam}, {"seeds": {}}) == []

"""Part 7 training campaign (WS1; D73, D74, D76, D84; contract 4.2), SUMO-free: the
pre-registered plan (T 15, S 33, F 5 jobs, priorities T > F > S, the stage-T base runs
reused by S), a stub mini-campaign of 3 real `rl.train --stub` jobs on 2 slots that
completes and is skipped on rerun, one automatic `--force` rerun per crash (a second
crash fails the campaign, exit 2), and `study.decide_t` -> stage F once T is done.

The crash and decision tests run a fake `rl.train` module (FAKE_TRAIN below) that only
writes the manifest and provenance a finished run leaves."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from traffic_twin.config import ConfigError, load_config, text_sha256
from traffic_twin.rl import campaign
from traffic_twin.rl.campaign import DECISION_T_FILE, PRIORITY, Job, finished, jobs, run
from traffic_twin.rl.train import run_config_sha, run_name

FAKE_TRAIN = '''
"""Fake rl.train: records its argv, crashes while FAKE_CRASH says so, else writes the
manifest and provenance of a finished run. Only traffic_twin.config is imported (fast
start); the run identity repeats rl.train.run_config_sha / run_name (checked by
test_fake_train_names_runs_like_rl_train)."""
import argparse, hashlib, json, os, sys
from pathlib import Path

from traffic_twin.config import load_config

p = argparse.ArgumentParser()
for opt in ("--config", "--algo", "--seed", "--steps", "--out", "--eval-pool"):
    p.add_argument(opt)
p.add_argument("--validation-seeds", nargs="+")
for flag in ("--force", "--allow-dirty", "--stub"):
    p.add_argument(flag, action="store_true")
a = p.parse_args()
cfg = load_config(a.config)
skip = ("changes", "preregistered", "protocol", "outputs")
payload = {k: v for k, v in cfg.items() if k not in skip}
text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
name = f"{a.algo}_{cfg['variant']}_s{a.seed}_{int(a.steps) // 1000}k_{sha[:8]}"
calls = Path(os.environ["FAKE_CALLS"])
with calls.open("a", encoding="utf-8") as f:
    rec = {"run": name, "argv": sys.argv[1:], "openblas": os.environ.get("OPENBLAS_NUM_THREADS")}
    f.write(json.dumps(rec) + "\\n")
n = sum(json.loads(ln)["run"] == name for ln in calls.read_text(encoding="utf-8").splitlines())
crash = json.loads(os.environ.get("FAKE_CRASH", "{}"))
if n <= int(crash.get(name.split("_s")[0], 0)):
    sys.exit(3)
out = Path(a.out) / name
out.mkdir(parents=True, exist_ok=True)
prov = {"run_config_sha": sha, "steps": int(a.steps)}
(out / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
(out / "manifest.json").write_text(
    json.dumps({"steps": int(a.steps), "steps_per_s": 1234.5}), encoding="utf-8"
)
'''


@pytest.fixture
def fake_train(tmp_path, monkeypatch):
    """Run the campaign's jobs with FAKE_TRAIN; returns `calls() -> [{run, argv}]` and
    `crash(**{variant tag: crashes})`."""
    mod = tmp_path / "fakemods"
    mod.mkdir()
    (mod / "fake_rl_train.py").write_text(FAKE_TRAIN, encoding="utf-8")
    calls = tmp_path / "calls.jsonl"
    calls.touch()
    path = os.pathsep.join(filter(None, [str(mod), os.environ.get("PYTHONPATH")]))
    monkeypatch.setenv("PYTHONPATH", path)
    monkeypatch.setenv("FAKE_CALLS", str(calls))
    monkeypatch.setattr(campaign, "TRAIN_MODULE", "fake_rl_train")
    monkeypatch.setattr(campaign, "POLL_S", 0.05)

    class Fake:
        @staticmethod
        def calls() -> list[dict]:
            lines = calls.read_text(encoding="utf-8").splitlines()
            return [json.loads(ln) for ln in lines]

        @staticmethod
        def crash(**counts: int) -> None:
            monkeypatch.setenv("FAKE_CRASH", json.dumps(counts))

    return Fake


def _decision(rl7: Path, chosen: str = "t1", steps_override: int | None = None) -> dict:
    """The provenance keys `study.decide_t` records, for the current protocol: the run
    names are the stage-T jobs' (`jobs`), independent of `campaign.check_decision`."""
    cfg = load_config(rl7)
    st = cfg["protocol"]["stage_t"]
    runs: dict[str, dict[str, str]] = {}
    for j in jobs(rl7, ["T"], steps_override=steps_override):
        runs.setdefault(j.config.stem, {})[str(j.seed)] = j.run
    return {
        "chosen": chosen,
        "runs": runs,
        "steps": int(steps_override or st["steps"]),
        "seeds": [int(s) for s in st["seeds"]],
        "rl7_sha256": text_sha256(rl7),
    }


def _decide(study: Path, rl7: Path, chosen: str = "t1", **changed) -> Path:
    """Write a decision_T.json under `study`: the current protocol's, with `changed` keys
    replaced (None removes a key)."""
    doc = _decision(rl7, chosen) | changed
    doc = {k: v for k, v in doc.items() if v is not None}
    study.mkdir(parents=True, exist_ok=True)
    path = study / DECISION_T_FILE
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# -- the plan --------------------------------------------------------------------------
def test_plan_has_the_preregistered_jobs_in_order(stub_rl7_cfg, tmp_path, capsys):
    rl7 = stub_rl7_cfg()
    cfg = load_config(rl7)
    study = Path(cfg["outputs"]["study"])
    before = jobs(rl7, ["T", "S", "F"])
    assert [j.stage for j in before].count("F") == 0  # F waits for decision_T.json
    _decide(study, rl7, "v1")
    plan = jobs(rl7, ["T", "S", "F"])
    by = {s: [j for j in plan if j.stage == s] for s in "TSF"}
    assert {s: len(v) for s, v in by.items()} == {"T": 15, "S": 33, "F": 5}
    assert [j.stage for j in plan] == ["T"] * 15 + ["F"] * 5 + ["S"] * 33  # T > F > S
    assert [j.priority for j in plan] == sorted(j.priority for j in plan)
    assert [(j.config.stem, j.seed) for j in by["T"]][:4] == [
        ("t0", 0), ("t0", 1), ("t0", 2), ("t1", 0)
    ]  # fmt: skip
    assert [j.config.stem for j in by["T"][::3]] == ["t0", "t1", "v1", "v2", "p1"]
    assert {j.algo for j in by["T"] if j.config.stem == "p1"} == {"ppo"}
    assert [(j.config.stem, j.seed) for j in by["F"]] == [("v1", s) for s in range(5)]
    # S: the base t1 reuses its stage-T seeds 0-2 (only 3, 4 are new); the bus axis has 5
    s_count = {}
    for j in by["S"]:
        s_count[j.config.stem] = s_count.get(j.config.stem, 0) + 1
    assert s_count == {
        "t1": 2, "wb12": 5, "wb50": 5, "wp06": 3, "wp24": 3, "wp48": 3,
        "lam0": 3, "lam100": 3, "transit_ped": 3, "car_centric": 3,
    }  # fmt: skip
    assert [j.seed for j in by["S"] if j.config.stem == "t1"] == [3, 4]
    steps = cfg["protocol"]
    assert {j.steps for j in by["T"]} == {steps["stage_t"]["steps"]}
    assert {j.steps for j in by["F"]} == {steps["stage_f"]["steps"]}
    assert {j.out.name for j in plan} == {"T", "S", "F"}
    for j in plan[:3]:
        c = load_config(j.config)
        assert j.run == run_name(j.algo, j.seed, j.steps, run_config_sha(c), c["variant"])
    assert len({j.run for j in plan}) == len(plan)
    state = run(rl7, ["T", "S", "F"], dry_run=True)
    out = capsys.readouterr().out
    assert "jobs: T 15, S 33, F 5" in out and state["dry_run"]
    assert not (study / campaign.CAMPAIGN_FILE).exists()  # a dry run writes nothing


def test_steps_override_and_out_root(stub_rl7_cfg, tmp_path):
    rl7 = stub_rl7_cfg()
    plan = jobs(rl7, ["T", "S"], steps_override=1000, out=tmp_path / "gate")
    assert {j.steps for j in plan} == {1000} and len(plan) == 15 + 33
    assert {j.out for j in plan} == {tmp_path / "gate" / "runs" / s for s in ("T", "S")}
    assert all("_1k_" in j.run for j in plan)
    with pytest.raises(ConfigError, match=DECISION_T_FILE):
        run(rl7, ["T", "F"], out=tmp_path / "gate", dry_run=True)
    with pytest.raises(ConfigError, match="stage F needs"):
        run(rl7, ["F"], dry_run=True)
    with pytest.raises(ConfigError, match="stages"):
        run(rl7, ["X"], dry_run=True)


def test_registered_roots_refuse_overridden_budgets_and_dev_windows(stub_rl7_cfg):
    # gate or dev runs (and their decision) never go to outputs.runs / outputs.study
    rl7 = stub_rl7_cfg()
    with pytest.raises(ConfigError, match="need --out"):
        run(rl7, ["T"], steps_override=1000, dry_run=True)
    with pytest.raises(ConfigError, match="need --out"):
        run(rl7, ["T"], validation_seeds=[5000, 5001], dry_run=True)
    registered = list(range(5100, 5110))  # the stub's seeds.selection
    assert run(rl7, ["T"], validation_seeds=registered, dry_run=True)["dry_run"]


# A decision_T.json that is not the stage-T decision of the current protocol: each
# provenance key the campaign checks, changed alone (None: the key is missing).
FOREIGN_DECISIONS = {
    "gate budget": ({"steps": 20000}, "steps 20000 != 600"),
    "no steps": ({"steps": None}, "steps None != 600"),
    "other seeds": ({"seeds": [0, 1]}, r"seeds \[0, 1\] != stage_t.seeds"),
    "other runs": ({"runs": {"t1": {"0": "dqn_t1_s0_20k_deadbeef"}}}, "not the current stage-T"),
    "other rl7.yaml": ({"rl7_sha256": "deadbeef"}, "rl7_sha256 'deadbeef'"),
    "not a T config": ({"chosen": "wb12"}, "chosen 'wb12' is not a stage_t config"),
    "no choice": ({"chosen": ""}, "no `chosen`"),
    "dev windows": ({"validation_seeds": [5000, 5001]}, r"validation_seeds \[5000, 5001\]"),
    "development": (
        {"validation_seeds": list(range(5100, 5110)), "development": True},
        "development-window decision",
    ),
}


@pytest.mark.parametrize("case", list(FOREIGN_DECISIONS))
def test_stage_f_refuses_a_decision_of_another_protocol(stub_rl7_cfg, case):
    changed, why = FOREIGN_DECISIONS[case]
    rl7 = stub_rl7_cfg()
    study = Path(load_config(rl7)["outputs"]["study"])
    path = _decide(study, rl7, **({"chosen": "v1"} | changed))
    for stages in (["F"], ["T", "S", "F"]):
        with pytest.raises(ConfigError, match=why):
            run(rl7, stages, dry_run=True)
    with pytest.raises(ConfigError, match="is not the stage-T decision of the current protocol"):
        jobs(rl7, ["T", "F"])
    path.write_text(json.dumps(_decision(rl7, "v1")), encoding="utf-8")
    assert {j.config.stem for j in jobs(rl7, ["F"])} == {"v1"}  # the current one passes
    assert campaign.check_decision(rl7, _decision(rl7, "v1")) == "v1"


def test_a_gate_decision_drives_only_its_own_campaign(stub_rl7_cfg, tmp_path):
    """The finding's scenario: `study decide-t` on the gate runs (override budget, dev
    windows) written into the registered study root must not drive the real stage F."""
    rl7 = stub_rl7_cfg()
    gate = tmp_path / "gate"
    gate_doc = _decision(rl7, "v2", steps_override=500) | {
        "validation_seeds": list(range(5100, 5110)),
        "development": True,
    }
    _decide(gate, rl7, **gate_doc)
    f = jobs(rl7, ["F"], steps_override=500, out=gate)  # the gate's own mini-campaign
    assert [(j.config.stem, j.steps) for j in f] == [("v2", 500)] * 5
    study = Path(load_config(rl7)["outputs"]["study"])
    _decide(study, rl7, **gate_doc)  # copied (or decided) into outputs.study by mistake
    with pytest.raises(ConfigError, match="steps 500 != 600"):
        run(rl7, ["T", "S", "F"], dry_run=True)


def test_a_changed_run_config_voids_the_decision(stub_rl7_cfg):
    rl7 = stub_rl7_cfg()
    study = Path(load_config(rl7)["outputs"]["study"])
    _decide(study, rl7, "v1")
    rl7 = stub_rl7_cfg(dqn={"exploration_steps": 400})  # new run identities (and rl7 text)
    with pytest.raises(ConfigError, match="not the current stage-T runs"):
        jobs(rl7, ["F"])


# -- running ---------------------------------------------------------------------------
def test_stub_mini_campaign_completes_and_skips_on_rerun(stub_rl7_cfg, monkeypatch):
    """3 real `rl.train --stub` jobs on 2 slots, then a rerun that starts nothing."""
    pytest.importorskip("stable_baselines3")
    monkeypatch.setattr(campaign, "POLL_S", 0.1)
    rl7 = stub_rl7_cfg(
        seeds={"selection": [5100, 5101]},
        protocol={"stage_t": {"configs": ["t1", "v2", "p1"], "seeds": [0], "steps": 300}},
        ppo={"n_steps": 64, "batch_size": 32, "n_epochs": 1},
    )
    state = run(rl7, ["T"], slots=2, stub=True)
    assert state["status"] == "done", state
    recs = state["jobs"]
    assert len(recs) == 3 and {r["status"] for r in recs.values()} == {"done"}
    assert all(r["attempts"] == 1 and r["steps_per_s"] > 0 for r in recs.values())
    study = Path(load_config(rl7)["outputs"]["study"])
    saved = json.loads((study / campaign.CAMPAIGN_FILE).read_text(encoding="utf-8"))
    assert saved["jobs"].keys() == recs.keys() and saved["status"] == "done"
    for name, r in recs.items():
        assert Path(r["out"]).name == name and (Path(r["out"]) / "pick_model.zip").is_file()
        assert (study / campaign.LOGS_DIR / f"{name}.log").is_file()
    again = run(rl7, ["T"], slots=2, stub=True)
    assert {r["status"] for r in again["jobs"].values()} == {"skipped"}
    assert all(r["attempts"] == 0 for r in again["jobs"].values())


def test_a_crash_is_rerun_once_with_force(stub_rl7_cfg, fake_train):
    rl7 = stub_rl7_cfg(protocol={"stage_t": {"configs": ["t0", "t1"], "seeds": [0, 1]}})
    fake_train.crash(dqn_t1=1)  # each t1 job crashes on its first attempt
    state = run(rl7, ["T"], slots=2)
    assert state["status"] == "done"
    recs = state["jobs"]
    assert {n: r["attempts"] for n, r in recs.items() if "_t1_" in n} == dict.fromkeys(
        [n for n in recs if "_t1_" in n], 2
    )
    assert all(r["attempts"] == 1 for n, r in recs.items() if "_t0_" in n)
    assert all(r["status"] == "done" and r["steps_per_s"] == 1234.5 for r in recs.values())
    calls = fake_train.calls()
    t1 = [c["argv"] for c in calls if "_t1_" in c["run"]]
    assert sum("--force" in a for a in t1) == 2 and all("--force" not in a for a in t1[:2])
    assert all("--validation-seeds" not in c["argv"] for c in calls)  # headline: selection


def test_a_second_crash_fails_the_campaign(stub_rl7_cfg, fake_train):
    rl7 = stub_rl7_cfg(protocol={"stage_t": {"configs": ["t1", "t0"], "seeds": [0, 1, 2]}})
    fake_train.crash(dqn_t1=99)  # t1 always crashes
    state = run(rl7, ["T"], slots=1)
    assert state["status"] == "failed"
    statuses = [r["status"] for r in state["jobs"].values()]
    assert statuses[0] == "failed" and statuses.count("failed") == 1
    assert state["not_started"] == list(state["jobs"])[1:]  # the queue stops
    assert len(fake_train.calls()) == 2  # the failed job's two attempts, nothing else
    cli = [
        "--config", str(rl7), "--stages", "T", "--slots", "1",
    ]  # fmt: skip
    assert campaign.main(cli) == campaign.FAILED_EXIT


def test_decide_t_then_stage_f_jumps_the_queue(stub_rl7_cfg, fake_train, monkeypatch):
    from traffic_twin.rl import study as study_mod

    rl7 = stub_rl7_cfg(
        protocol={
            "stage_t": {"configs": ["t0", "t1"], "seeds": [0]},
            "stage_f": {"seeds": [0, 1]},
            "sensitivity": {"settings": ["wb12"], "wide": [], "seeds_default": [0, 1]},
        }
    )
    study = Path(load_config(rl7)["outputs"]["study"])
    called = []

    def decide_t(path, **kw):
        called.append((Path(path), kw))
        return {"chosen": "t1", "path": _decide(study, rl7, "t1")}

    monkeypatch.setattr(study_mod, "decide_t", decide_t)
    state = run(rl7, ["T", "S", "F"], slots=1)
    assert state["status"] == "done"
    # the runs and study root the campaign uses, the registered budget (no override)
    assert called == [(Path(rl7), {"runs": study / "runs" / "T", "out": study, "steps": None})]
    order = [c["run"] for c in fake_train.calls()]
    stages = [state["jobs"][r]["stage"] for r in order]
    # T first; once T is done, F jumps ahead of the S jobs still queued
    assert stages == ["T", "T", "F", "F", "S", "S", "S"]
    assert state["decision_T"] == {"chosen": "t1"}
    # S: base t1 seed 0 reused from T; seed 1 and wb12 x 2 are new
    assert sum(s == "S" for s in stages) == 3


def test_a_foreign_decision_after_stage_t_stops_the_campaign(stub_rl7_cfg, fake_train, monkeypatch):
    from traffic_twin.rl import study as study_mod

    rl7 = stub_rl7_cfg(
        protocol={"stage_t": {"configs": ["t0", "t1"], "seeds": [0]}, "stage_f": {"seeds": [0]}}
    )
    study = Path(load_config(rl7)["outputs"]["study"])
    monkeypatch.setattr(
        study_mod, "decide_t", lambda path, **kw: _decide(study, rl7, "t1", steps=20000)
    )
    with pytest.raises(ConfigError, match="steps 20000 != 600"):
        run(rl7, ["T", "F"], slots=1)
    assert all("_t1_s0_1k_" not in c["run"] for c in fake_train.calls())  # no F job ran
    saved = json.loads((study / campaign.CAMPAIGN_FILE).read_text(encoding="utf-8"))
    assert saved["status"] == "interrupted" and saved["decision_T"] is None


def test_gate_runs_validate_on_development_windows(stub_rl7_cfg, fake_train, tmp_path):
    rl7 = stub_rl7_cfg(protocol={"stage_t": {"configs": ["t1"], "seeds": [0]}})
    state = run(rl7, ["T"], slots=1, steps_override=500, out=tmp_path / "gate")
    assert state["status"] == "done"
    (call,) = fake_train.calls()
    i = call["argv"].index("--validation-seeds")
    assert call["argv"][i + 1 : i + 11] == [str(s) for s in range(5100, 5110)]
    assert call["argv"][call["argv"].index("--steps") + 1] == "500"
    assert (tmp_path / "gate" / campaign.CAMPAIGN_FILE).is_file()
    (rec,) = state["jobs"].values()
    assert Path(rec["out"]).parent == tmp_path / "gate" / "runs" / "T"


def test_finished_requires_matching_identity_and_steps(stub_rl7_cfg, tmp_path):
    rl7 = stub_rl7_cfg()
    cfg_path = rl7.parent / "rl7" / "t1.yaml"
    sha = run_config_sha(load_config(cfg_path))
    job = Job("T", cfg_path, "dqn", 0, 600, PRIORITY["T"], tmp_path, run="r")
    assert not finished(job)
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "manifest.json").write_text(json.dumps({"steps": 600}), encoding="utf-8")
    prov = tmp_path / "r" / "provenance.json"
    prov.write_text(json.dumps({"run_config_sha": sha, "steps": 600}), encoding="utf-8")
    assert finished(job)
    prov.write_text(json.dumps({"run_config_sha": "0" * 64, "steps": 600}), encoding="utf-8")
    assert not finished(job)
    assert campaign.train_command(job, force=True, allow_dirty=True)[:3] == [
        sys.executable, "-m", campaign.TRAIN_MODULE,
    ]  # fmt: skip


@pytest.mark.parametrize("preset", [None, "4"])
def test_jobs_run_with_one_openblas_thread_unless_set(
    stub_rl7_cfg, fake_train, monkeypatch, preset
):
    # 13 processes per job: OpenBLAS's per-thread buffers (~0.74 GB each) exhausted the
    # commit limit at 4-6 slots (gate); a value set by the user is kept
    if preset is None:
        monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    else:
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", preset)
    rl7 = stub_rl7_cfg(protocol={"stage_t": {"configs": ["t1"], "seeds": [0]}})
    assert run(rl7, ["T"], slots=1)["status"] == "done"
    (call,) = fake_train.calls()
    assert call["openblas"] == (preset or "1")
    assert os.environ.get("OPENBLAS_NUM_THREADS") == preset  # the campaign's own env is untouched


def test_eval_pool_is_passed_to_every_job(stub_rl7_cfg, fake_train):
    rl7 = stub_rl7_cfg(protocol={"stage_t": {"configs": ["t1"], "seeds": [0, 1]}})
    assert run(rl7, ["T"], slots=2)["eval_pool"] == "persistent"  # rl.train's default
    assert all("--eval-pool" not in c["argv"] for c in fake_train.calls())
    rl7 = stub_rl7_cfg(protocol={"stage_t": {"configs": ["t0"], "seeds": [0]}})
    state = run(rl7, ["T"], slots=1, eval_pool="transient")
    assert state["eval_pool"] == "transient"
    (call,) = [c for c in fake_train.calls() if "_t0_" in c["run"]]
    assert call["argv"][call["argv"].index("--eval-pool") + 1] == "transient"
    with pytest.raises(ConfigError, match="eval_pool"):
        run(rl7, ["T"], slots=1, eval_pool="forever")


def test_fake_train_names_runs_like_rl_train(stub_rl7_cfg, fake_train, tmp_path):
    rl7 = stub_rl7_cfg(protocol={"stage_t": {"configs": ["p1", "v1"], "seeds": [2]}})
    state = run(rl7, ["T"], slots=2)
    assert {r["status"] for r in state["jobs"].values()} == {"done"}
    assert sorted(c["run"] for c in fake_train.calls()) == sorted(state["jobs"])

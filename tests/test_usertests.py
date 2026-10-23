"""User tests (Part 9, spec 12.2, A-10; D111, D112): protocol, privacy guard, scoring
rules, fix loop, synthesis, kit and expected answers. No SUMO; the expected-answer test on
the demo run is skipped when data/demo is absent (CI)."""

from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
from pathlib import Path

import pytest
import yaml

from traffic_twin.config import ConfigError
from traffic_twin.paths import USERTESTS_CONFIG, USERTESTS_RESULTS
from traffic_twin.usertests import __main__ as cli
from traffic_twin.usertests import answers as ans
from traffic_twin.usertests import kit
from traffic_twin.usertests import protocol as pr
from traffic_twin.usertests import synthesis as syn

REPO = Path(__file__).resolve().parent.parent
SYNTHETIC = REPO / "tests" / "data" / "usertests_synthetic.yaml"
DEMO_RUN = REPO / "data" / "demo" / "pitie_rules_seed4000"


@pytest.fixture(scope="module")
def protocol() -> dict:
    return pr.load_protocol(REPO / USERTESTS_CONFIG)


@pytest.fixture
def synthetic(protocol) -> dict:
    raw = yaml.safe_load(SYNTHETIC.read_text(encoding="utf-8"))
    assert pr.check_results(raw, protocol) == []
    return raw


def _errors(raw: dict, protocol: dict) -> str:
    return "\n".join(pr.check_results(raw, protocol))


def _fake_answers(protocol: dict, *, pending: bool = False, problems=()) -> dict:
    """An expected-answer record shaped like `answers.expected_answers` (no API)."""
    tasks = {}
    for t in protocol["tasks"]:
        checks = [{"url": e["endpoint"], "field": e["field"], "value": e.get("equals", 7),
                   "expected": "argmin" if e.get("computed") else e.get("equals"),
                   "tolerance": e.get("tolerance"), "unit": e.get("unit"), "holds": True}
                  for e in t["expect"]]  # fmt: skip
        tasks[t["id"]] = {"checks": checks, "holds": True}
    values = {"S0": 60.0, "S1": 55.0, "S2": 50.0, "S3": None if pending else 45.0}
    return {
        "run": {
            "run_id": "00000000-test-run",
            "scenario": "pitie_rules",
            "label": "S2",
            "seed": 4000,
            "dir": "x",
            "current": True,
            "code_version": "fixture",
            "signal_sha256": "ab" * 32,
        },  # fmt: skip
        "api": {"data_source": "simulation", "calibrated_on": "2026-10-08", "provisional": True},
        "operator": {
            "family": "main",
            "kpi": "all/person_delay_mean",
            "values": values,
            "missing": ["S3"] if pending else [],
            "pending": ["S3"] if pending else [],
            "argmin": None if pending else "S3",
        },  # fmt: skip
        "tasks": tasks,
        "problems": list(problems),
    }


# -- protocol --------------------------------------------------------------------------
def test_protocol_is_registered_with_the_d111_tasks(protocol):
    assert protocol["preregistered"] == "2026-10-23"
    # one dated change, made before any participant: the T-OP question names the main
    # scenarios (ablations excluded), as the key (operator.labels) always did
    assert [c["date"] for c in protocol["changes"]] == ["2026-10-23"]
    assert "T-OP" in protocol["changes"][0]["what"]
    assert protocol["operator"]["labels"] == ["S0", "S1", "S2", "S3"]
    assert protocol["run"] == {"scenario": "pitie_rules", "seed": 4000}
    assert protocol["participants"] == {"min": 3, "max": 5}
    tasks = pr.tasks_by_id(protocol)
    assert list(tasks) == ["T-PED", "T-PMR", "T-STU", "T-BUS", "T-AMB", "T-OP", "T-REPLAY"]
    assert [t for t in tasks if tasks[t].get("optional")] == ["T-REPLAY"]
    instants = {tid: (t.get("start") or {}).get("t") for tid, t in tasks.items()}
    assert instants == {"T-PED": 1805, "T-PMR": 1805, "T-STU": 2700, "T-BUS": 1305,
                        "T-AMB": 2440, "T-OP": None, "T-REPLAY": None}  # fmt: skip
    assert tasks["T-STU"]["check"]["params"]["exam"] == "08:18"
    assert tasks["T-AMB"]["start"] == {"view": "ambulance", "t": 2440, "kiosk": 1}
    assert protocol["order"]["first"] == "T-PED"
    # the 5 spec views each have a required task; cards are French
    required = {tasks[t]["view"] for t in pr.required_tasks(protocol)}
    assert set(pr.SPEC_VIEWS) <= required
    assert all(any(c in t["card_fr"] for c in "éèàêç?") for t in protocol["tasks"])
    # T-OP asks a factual question, never a recommendation
    assert "plus faible retard moyen par personne" in tasks["T-OP"]["card_fr"]


def test_protocol_problems_are_listed_together(protocol, tmp_path):
    raw = yaml.safe_load((REPO / USERTESTS_CONFIG).read_text(encoding="utf-8"))
    raw["tasks"][1]["id"] = "T-PED"  # duplicate
    raw["tasks"][2]["view"] = "cyclist"
    del raw["tasks"][3]["card_fr"]
    raw["run"]["seed"] = 1  # an evaluation seed
    raw["surname"] = "x"
    path = tmp_path / "usertests.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        pr.load_protocol(path)
    msg = str(e.value)
    for part in ("duplicate id", "cyclist", "card_fr", "demo seed", "unknown key 'surname'"):
        assert part in msg
    assert "problem(s)" in msg


def test_task_order_puts_t_ped_first_and_rotates(protocol):
    orders = [pr.task_order(protocol, k) for k in range(1, 6)]
    assert all(o[0] == "T-PED" and o[-1] == "T-REPLAY" for o in orders)
    assert all(sorted(o) == sorted(pr.tasks_by_id(protocol)) for o in orders)
    seconds = [o[1] for o in orders]
    assert len(set(seconds)) == 5  # every core task comes second once
    assert pr.task_order(protocol, 6) == orders[0]
    with pytest.raises(ValueError):
        pr.task_order(protocol, 0)


def test_deep_links_follow_d106(protocol):
    tasks = pr.tasks_by_id(protocol)
    base = "http://127.0.0.1:8000/mock?"
    run = "scenario=pitie_rules&seed=4000"
    assert pr.deep_link(protocol, tasks["T-PED"]) == f"{base}view=home&{run}&t=1805"
    assert pr.deep_link(protocol, tasks["T-AMB"]) == f"{base}view=ambulance&{run}&t=2440&kiosk=1"
    assert pr.deep_link(protocol, tasks["T-OP"]) == f"{base}view=home&{run}"
    check = pr.deep_link(protocol, tasks["T-PMR"], "check")
    assert check == f"{base}view=pedestrian&{run}&t=1805&crossing=S&speed=reduced"
    assert pr.deep_link(protocol, tasks["T-REPLAY"], "check").endswith(
        "view=replay&" + run + "&t=08%3A40"
    )
    assert pr.deep_link(protocol, tasks["T-STU"], "check", "http://h/m").startswith("http://h/m?")
    assert pr.clock(1805) == "08:00:05" and pr.clock(None) == "–"


# -- results: template, privacy, scoring, fix loop -------------------------------------
def test_committed_template_is_valid_and_has_no_session(protocol):
    results = pr.load_results(REPO / USERTESTS_RESULTS, protocol)
    assert results["sessions"] == [] and not results.get("synthetic")
    issues = results["issues"]
    assert issues and all(i["found_by"] == "walkthrough" and i["sessions"] == [] for i in issues)
    assert [i["id"] for i in issues] == [f"U{k:02d}" for k in range(1, len(issues) + 1)]
    synth = syn.synthesis(results, protocol)
    assert synth["status"] == "PENDING" and synth["n"] == 0


def test_empty_results_are_pending_everywhere(protocol):
    synth = syn.synthesis(pr.empty_results(), protocol)
    assert synth["a10"] == {"status": "PENDING", "n_complete": 0, "n_sessions": 0,
                            "range": [3, 5], "text": synth["a10"]["text"]}  # fmt: skip
    page = syn.render(synth)
    assert "**A-10: PENDING.**" in page
    rows = [line for line in page.splitlines() if line.startswith("| T-")]
    assert len(rows) == len(protocol["tasks"])
    assert all(row.count("PENDING") == 8 for row in rows)
    for line in page.splitlines():
        if line.startswith("- ") and ("ENF-05" in line or "clicks on every task" in line):
            assert line.endswith("PENDING (n<3)")
    assert "PENDING: no issue recorded yet." in page
    assert "MET" not in page.replace("PENDING", "")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: s.update(name="Jean Dupont"), "unknown key 'name'"),
        (lambda s: s.update(age=34), "unknown key 'age'"),
        (lambda s: s.update(id="Jean"), "anonymous ids only"),
        (lambda s: s["comments"].append("écrire à jean.dupont@mail.fr"), "contact details"),
        (lambda s: s["tasks"][0].update(notes="rappeler au 06 12 34 56 78"), "contact details"),
        (lambda s: s["comments"].append("tel +33 6 12 34 56 78"), "contact details"),
        (lambda s: s["tasks"][0].update(mobility="cane"), "unknown key 'mobility'"),
        (lambda s: s.update(profile_match="wheelchair user"), "profile_match"),
    ],
)
def test_privacy_guard_refuses_personal_data(synthetic, protocol, edit, message):
    raw = copy.deepcopy(synthetic)
    edit(raw["sessions"][0])
    assert message in _errors(raw, protocol)


def test_privacy_guard_does_not_flag_shas_or_answers(synthetic, protocol):
    raw = copy.deepcopy(synthetic)
    raw["sessions"][0]["app_version"] = "0612345678"  # digits only, looks like a phone
    raw["sessions"][0]["tasks"][3]["answer"] = "+81 s, 08:18"
    assert pr.check_results(raw, protocol) == []
    assert pr.looks_personal("a@b.fr") and not pr.looks_personal("U03 fixed in 3104666")


@pytest.mark.parametrize(
    "node",
    [
        "tests/test_usertests.py::test_x[jean.dupont@gmail.com 06 12 34 56 78]",
        "tests/test_usertests.py::test_x[jean.dupont@gmail.com]",
        "tests/test_usertests.py::test_x[06.12.34.56.78]",
        "tests/test_usertests.py::test_x[0612345678]",
        "tests/test_usertests.py::test_x[+33612345678]",
    ],
)
def test_privacy_guard_covers_the_verified_test_node(synthetic, protocol, node):
    """`verified` skips the free-text scan, so its parametrize id must not carry contact
    details (ENF-07)."""
    raw = copy.deepcopy(synthetic)
    u01 = next(i for i in raw["issues"] if i["id"] == "U01")
    u01["verified"] = node
    errs = _errors(raw, protocol)
    assert "U01" in errs and ("contact details" in errs or "test node" in errs)


def test_verified_accepts_ordinary_parametrize_ids(synthetic, protocol):
    raw = copy.deepcopy(synthetic)
    u01 = next(i for i in raw["issues"] if i["id"] == "U01")
    nodes = ("tests/test_usertests.py::test_x[edit0-message0]",
             "tests/test_usertests.py::test_x[T-PED]", "tests/test_x.py::test_y[0.5]")  # fmt: skip
    for node in nodes:
        u01["verified"] = node
        assert pr.check_results(raw, protocol) == [], node


def test_load_results_raises_one_error_listing_every_problem(synthetic, protocol, tmp_path):
    raw = copy.deepcopy(synthetic)
    raw["sessions"][0]["name"] = "x"
    raw["sessions"][1]["date"] = "2026-10-01"  # before the registration
    raw["issues"][0]["severity"] = 9
    path = tmp_path / "results.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        pr.load_results(path, protocol)
    msg = str(e.value)
    assert "3 problem(s)" in msg and "before the protocol was registered" in msg
    assert pr.load_results(tmp_path / "absent.yaml", protocol) == pr.empty_results()


def test_scoring_rules_are_enforced(synthetic, protocol):
    raw = copy.deepcopy(synthetic)
    recs = raw["sessions"][0]["tasks"]
    recs[0]["time_s"] = 31.0  # T-PED limit 30 s
    recs[1]["clicks"] = 3  # T-PMR success over 2 clicks
    recs[3]["hint"] = True  # T-BUS success after the hint
    recs[4]["time_s"] = None  # a scored task without a time
    recs[5]["order"] = 1  # two tasks share order 1
    errs = _errors(raw, protocol)
    for part in ("fail (timeout)", "over 2 clicks: partial", "after the scripted hint",
                 "needs time_s and clicks", "share an order number"):  # fmt: skip
        assert part in errs


def test_every_required_task_is_recorded(synthetic, protocol):
    raw = copy.deepcopy(synthetic)
    raw["sessions"][2]["tasks"] = [r for r in raw["sessions"][2]["tasks"] if r["task"] != "T-AMB"]
    assert "task T-AMB missing" in _errors(raw, protocol)
    raw = copy.deepcopy(synthetic)
    raw["sessions"][1]["tasks"][0]["order"] = 2
    raw["sessions"][1]["tasks"][1]["order"] = 1
    assert "T-PED comes first" in _errors(raw, protocol)


def test_fixed_issue_needs_a_commit(synthetic, protocol):
    raw = copy.deepcopy(synthetic)
    raw["issues"][0]["commit"] = None
    assert "status fixed needs the sha" in _errors(raw, protocol)
    raw = copy.deepcopy(synthetic)
    raw["issues"][1]["verified"] = "P2"  # an open issue cannot be verified
    assert "only a fixed issue can be verified" in _errors(raw, protocol)
    raw = copy.deepcopy(synthetic)
    raw["issues"][0]["verified"] = "P9"
    assert "unknown session P9" in _errors(raw, protocol)
    raw = copy.deepcopy(synthetic)
    raw["issues"][1]["sessions"] = ["P1"]  # a walkthrough happens before P1
    assert "walkthrough issue" in _errors(raw, protocol)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def test_verify_git_checks_the_fix_loop(synthetic, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_x.py").write_text("def test_fixed():\n    pass\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "before")
    before = _git(repo, "rev-parse", "--short", "HEAD")
    (repo / "a.txt").write_text("fix", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "fix U01")
    fix = _git(repo, "rev-parse", "--short", "HEAD")
    raw = copy.deepcopy(synthetic)
    issue = raw["issues"][0]
    issue.update(commit=fix, verified="tests/test_x.py::test_fixed")
    assert pr.verify_git(raw, repo) == []
    issue["verified"] = "tests/test_x.py::test_missing"
    assert "test tests/test_x.py::test_missing not found" in pr.verify_git(raw, repo)[0]
    issue["verified"] = "P2"
    raw["sessions"][1]["app_version"] = fix
    assert pr.verify_git(raw, repo) == []
    raw["sessions"][1]["app_version"] = before  # the session ran the app before the fix
    assert "does not contain the fix" in pr.verify_git(raw, repo)[0]
    issue["commit"] = "deadbee"
    assert "not found in the repository" in pr.verify_git(raw, repo)[0]


# -- synthesis -------------------------------------------------------------------------
def test_synthetic_rates_and_medians(synthetic, protocol):
    s = syn.synthesis(synthetic, protocol)
    assert s["synthetic"] and s["status"] == "SYNTHETIC" and s["n"] == 3  # never DONE
    rows = {t["id"]: t for t in s["tasks"]}
    ped = rows["T-PED"]
    assert (ped["n"], ped["success"], ped["time_median_s"]) == (3, 3, 4.6)
    assert ped["times_s"] == [3.2, 4.6, 6.1] and ped["errors"] == {"hesitation": 1}
    bus = rows["T-BUS"]
    assert (bus["success"], bus["success_or_partial"], bus["time_median_s"]) == (2, 2, 18.0)
    assert rows["T-STU"]["hints"] == 1 and rows["T-STU"]["errors"] == {"term": 1}
    assert rows["T-OP"]["clicks_max"] == 3 and rows["T-OP"]["within_clicks"] == 2
    replay = rows["T-REPLAY"]
    assert (replay["n"], replay["skipped"], replay["success_or_partial"]) == (1, 2, 1)
    e6 = s["enf06"]
    assert e6["pedestrian_time"] == {"status": "MET", "median_s": 4.6, "under": 2, "of": 3}
    assert e6["clicks"] == {"status": "NOT MET", "within": 17, "of": 19}
    assert e6["glance"] == {"status": "NOT MET", "success": 2, "of": 3}
    assert s["enf05"] == {"status": "NOT MET", "understood": 2, "of": 3, "unanswered": 0}
    assert s["sus"] == {"n": 1, "mean": 85.0, "scores": [85.0]}  # (17 + 17) x 2.5
    assert s["issues"]["fixed"] == 1 and s["issues"]["verified"] == 1
    assert [i["id"] for i in s["issues"]["rows"]] == ["U01", "U02"]  # by severity
    own = {p["session"]: p for p in s["profiles"]}
    assert own["P1"]["own_tasks"] == 2 and own["P1"]["own_success"] == 2  # T-PED, T-PMR
    assert own["P3"]["own_tasks"] == 1 and own["P3"]["own_success"] == 1  # T-BUS
    page = syn.render(s)
    assert "SYNTHETIC TEST DATA" in page and "**A-10: SYNTHETIC.** Synthetic test data" in page
    assert "A-10: DONE" not in page
    assert "| T-PED | pedestrian | 3 | 3/3 | 3/3 | 4.6 (3.2, 4.6, 6.1) |" in page
    assert "median 4.6 s, 2/3 under the target: MET" in page


def test_verdicts_stay_pending_below_three_sessions(synthetic, protocol):
    raw = copy.deepcopy(synthetic)
    raw["sessions"] = raw["sessions"][:2]
    raw["issues"] = [i for i in raw["issues"] if not i["sessions"] or i["sessions"] == ["P1"]]
    assert syn.synthesis(raw, protocol)["status"] == "SYNTHETIC"  # the fixture: never DONE
    raw["synthetic"] = False  # the PENDING rule itself
    s = syn.synthesis(raw, protocol)
    assert s["status"] == "PENDING" and s["n"] == 2
    assert all(v == {"status": "PENDING (n<3)"} for v in s["enf06"].values())
    assert s["enf05"] == {"status": "PENDING (n<3)"}
    page = syn.render(s)
    assert "MET" not in page.replace("PENDING", "")
    assert "| T-PED | pedestrian | 2 | 2/2 |" in page  # descriptive rows stay visible


def test_incomplete_session_is_not_counted(synthetic, protocol):
    raw = copy.deepcopy(synthetic)
    raw["sessions"][2]["tasks"][2]["outcome"] = None  # T-AMB not scored yet
    assert pr.check_results(raw, protocol) == []
    assert syn.synthesis(raw, protocol)["a10"]["n_complete"] == 2


def test_over_the_spec_range_is_flagged(synthetic, protocol):
    raw = copy.deepcopy(synthetic)
    extra = []
    for k in range(4, 7):
        s = copy.deepcopy(raw["sessions"][k % 3])
        s["id"] = f"P{k}"
        extra.append(s)
    raw["sessions"] += extra
    assert pr.check_results(raw, protocol) == []
    raw["synthetic"] = False  # the DONE rule itself; synthetic data never reach DONE
    s = syn.synthesis(raw, protocol)
    assert s["status"] == "DONE" and "over the spec range 3-5" in s["a10"]["text"]


def test_rounds_are_split_when_both_exist(synthetic, protocol):
    raw = copy.deepcopy(synthetic)
    raw["sessions"][2]["round"] = 2
    s = syn.synthesis(raw, protocol)
    assert set(s["rounds"]) == {"1", "2"} and s["rounds"]["2"]["n"] == 1
    assert "### Round 2 (on the fixed app, n = 1)" in syn.render(s)


def test_synthesis_is_deterministic(synthetic, protocol, tmp_path):
    src = syn.sources(REPO / USERTESTS_CONFIG, SYNTHETIC)
    a = syn.synthesis(synthetic, protocol, sources=src)
    b = syn.synthesis(copy.deepcopy(synthetic), protocol, sources=src)
    assert a == b and syn.render(a) == syn.render(b)
    doc, js = syn.write(a, tmp_path / "u.md", tmp_path / "u.json")
    assert json.loads(js.read_text(encoding="utf-8")) == a
    assert b"\r\n" not in doc.read_bytes()
    crlf = tmp_path / "crlf.yaml"
    crlf.write_bytes(SYNTHETIC.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert syn.text_sha256(crlf) == syn.text_sha256(SYNTHETIC)


# -- kit -------------------------------------------------------------------------------
def test_kit_builds_and_records_the_run(protocol, tmp_path):
    answers = _fake_answers(protocol)
    out = kit.build_kit(protocol, tmp_path / "kit.pdf", answers=answers, app_version="0123abc")
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    last = kit.page_numbers(protocol)["issues"]
    assert kit.page_count(out) == int(last) == 14
    ident = kit.kit_identity(out)
    assert ident["run_id"] == "00000000-test-run" and ident["signal_sha256"] == "ab" * 32
    assert ident["draft"] == "no" and ident["seed"] == "4000"
    assert b"/CreationDate" not in data
    again = kit.build_kit(protocol, tmp_path / "again.pdf", answers=answers,
                          app_version="0123abc")  # fmt: skip
    assert again.read_bytes() == data  # same inputs, same bytes
    assert kit.kit_identity(tmp_path / "absent.pdf") is None


def test_kit_refuses_pending_s3_and_failing_answers(protocol, tmp_path):
    pending = _fake_answers(protocol, pending=True)
    with pytest.raises(kit.KitError, match="S3 missing from /kpi"):
        kit.build_kit(protocol, tmp_path / "k.pdf", answers=pending, app_version="x")
    out = kit.build_kit(protocol, tmp_path / "draft.pdf", answers=pending, app_version="x",
                        allow_pending_s3=True)  # fmt: skip
    assert b"DRAFT: S3 pending" in out.read_bytes()
    broken = _fake_answers(protocol, problems=["T-PED: can_cross = False"])
    with pytest.raises(kit.KitError, match="no longer hold"):
        kit.build_kit(protocol, tmp_path / "b.pdf", answers=broken, app_version="x",
                      allow_pending_s3=True)  # fmt: skip
    assert not (tmp_path / "k.pdf").exists() and not (tmp_path / "b.pdf").exists()


# -- expected answers ------------------------------------------------------------------
def test_expected_answers_on_the_fake_fixtures(fake_api_config, protocol):
    from traffic_twin.api.settings import load_settings

    client = ans.api_client(load_settings(fake_api_config))
    got = ans.expected_answers(protocol, client)
    assert got["run"]["scenario"] == "pitie_rules" and got["run"]["seed"] == 4000
    assert got["run"]["signal_sha256"] and len(got["run"]["signal_sha256"]) == 64
    assert set(got["tasks"]) == set(pr.tasks_by_id(protocol))
    ped = got["tasks"]["T-PED"]["checks"][0]
    assert ped["url"].startswith("/crossing/S?speed=normal&run_id=") and ped["url"].endswith(
        "&t=1805"
    )
    assert "/replay/" in got["tasks"]["T-REPLAY"]["checks"][0]["url"]
    assert got["api"]["data_source"] == "simulation"
    op = got["operator"]
    assert set(op["values"]) == {"S0", "S1", "S2", "S3"} and op["kpi"] == "all/person_delay_mean"
    assert (op["argmin"] is None) == bool(op["missing"])
    with pytest.raises(ans.AnswerError, match="no pitie_rules run at seed 4001"):
        ans.resolve_run(client, "pitie_rules", 4001)


@pytest.mark.skipif(not DEMO_RUN.is_dir(), reason="data/demo/pitie_rules_seed4000 absent")
def test_expected_answers_hold_on_the_demo_run(protocol):
    got = ans.expected_answers(protocol)
    assert got["problems"] == []
    assert got["run"]["dir"].replace("\\", "/").endswith("pitie_rules_seed4000")
    assert all(got["operator"]["values"][k] is not None for k in ("S0", "S1", "S2"))
    if got["operator"]["pending"]:  # before the Part 8 run: the kit must refuse
        with pytest.raises(kit.KitError, match="missing from /kpi"):
            kit.build_kit(protocol, Path("unused.pdf"), answers=got, app_version="x")
        assert not Path("unused.pdf").exists()


# -- CLI -------------------------------------------------------------------------------
def test_cli_check_links_and_docs(tmp_path, capsys):
    assert cli.main(["check", "--no-git"]) == 0
    assert "A-10: PENDING" in capsys.readouterr().out
    assert cli.main(["links", "--participant", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("P2: T-PED -> T-STU") and "start http://127.0.0.1:8000/mock?" in out
    doc, js = tmp_path / "usertests.md", tmp_path / "synthesis.json"
    args = ["--doc", str(doc), "--json", str(js)]
    assert cli.main(["docs", "--check", *args]) == 1
    assert cli.main(["docs", *args]) == 0
    assert "A-10: PENDING" in capsys.readouterr().out
    assert cli.main(["docs", "--check", *args]) == 0
    assert json.loads(js.read_text(encoding="utf-8"))["status"] == "PENDING"
    assert "configs/usertests.yaml" in json.loads(js.read_text(encoding="utf-8"))["sources"]
    doc.write_text(doc.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
    assert cli.main(["docs", "--check", *args]) == 1


def test_cli_reports_a_bad_results_file(synthetic, tmp_path, capsys):
    raw = copy.deepcopy(synthetic)
    raw["sessions"][0]["email"] = "x"
    bad = tmp_path / "r.yaml"
    bad.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    assert cli.main(["check", "--no-git", "--results", str(bad)]) == 1
    assert "unknown key 'email'" in capsys.readouterr().err
    with pytest.raises(ConfigError, match="synthetic"):
        cli._refuse_synthetic({"synthetic": True}, USERTESTS_RESULTS)
    cli._refuse_synthetic({"synthetic": True}, bad)  # a fixture elsewhere is fine


def test_synthetic_data_never_reach_the_committed_docs(tmp_path, monkeypatch, capsys):
    """The fixture must never overwrite docs/usertests.md or synthesis.json, whatever the
    spelling of the paths (relative, absolute, from another working directory)."""
    from traffic_twin.paths import USERTESTS_DOC, USERTESTS_SYNTHESIS

    real = [REPO / p for p in (USERTESTS_DOC, USERTESTS_SYNTHESIS)]
    before = [p.read_bytes() if p.is_file() else None for p in real]
    doc, js = tmp_path / "usertests.md", tmp_path / "synthesis.json"
    monkeypatch.chdir(REPO)
    attempts = [
        ["docs", "--results", str(SYNTHETIC)],  # both outputs default to docs/
        ["docs", "--results", str(SYNTHETIC), "--doc", str(doc)],  # --json defaults to docs/
        ["docs", "--results", str(SYNTHETIC), "--json", str(js)],
        [
            "docs",
            "--results",
            str(SYNTHETIC),
            "--doc",
            str(REPO / USERTESTS_DOC),
            "--json",
            str(js),
        ],  # fmt: skip
    ]
    for argv in attempts:
        assert cli.main(argv) == 1, argv
        assert "synthetic test data would overwrite" in capsys.readouterr().err
    monkeypatch.chdir(tmp_path)  # absolute spellings from elsewhere are caught too
    assert cli.main(["docs", "--results", str(SYNTHETIC), "--doc",
                     str(REPO / USERTESTS_DOC), "--json", str(js)]) == 1  # fmt: skip
    with pytest.raises(ConfigError, match="real results file"):
        cli._refuse_synthetic({"synthetic": True}, (REPO / USERTESTS_RESULTS).resolve())
    with pytest.raises(ConfigError, match="real results file"):
        cli._refuse_synthetic({"synthetic": True}, Path(os.path.relpath(REPO / USERTESTS_RESULTS)))
    assert [p.read_bytes() if p.is_file() else None for p in real] == before
    monkeypatch.chdir(REPO)
    capsys.readouterr()
    assert cli.main(["docs", "--results", str(SYNTHETIC), "--doc", str(doc),
                     "--json", str(js)]) == 0  # fmt: skip
    assert "A-10: SYNTHETIC" in capsys.readouterr().out
    assert cli.main(["check", "--no-git", "--results", str(SYNTHETIC)]) == 0
    assert "A-10: SYNTHETIC" in capsys.readouterr().out


def test_synthetic_fixture_never_closes_a10_in_the_report(tmp_path, monkeypatch, capsys):
    """Even a synthesis.json built from the fixture and placed where the report reads it
    gives A-10 PENDING, never DONE (D112: invented sessions are not results)."""
    from traffic_twin.paths import USERTESTS_DOC, USERTESTS_SYNTHESIS
    from traffic_twin.report.facts import collect_facts

    monkeypatch.chdir(REPO)
    doc, js = tmp_path / USERTESTS_DOC, tmp_path / USERTESTS_SYNTHESIS
    assert cli.main(["docs", "--results", str(SYNTHETIC), "--doc", str(doc),
                     "--json", str(js)]) == 0  # fmt: skip
    synth = json.loads(js.read_text(encoding="utf-8"))
    assert synth["synthetic"] and synth["n"] == 3
    assert synth["status"] != "DONE" and synth["a10"]["status"] != "DONE"
    facts = collect_facts(root=tmp_path, code="unknown")
    assert facts["usertests.status"]["value"] != "DONE"
    assert facts["acceptance.A-10"]["value"] == "PENDING"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_serve_refuses_a_changed_campaign(protocol, fake_api_config, tmp_path):
    from traffic_twin.api.settings import load_settings

    settings = load_settings(fake_api_config)
    answers = _fake_answers(protocol)
    kit_pdf = kit.build_kit(protocol, tmp_path / "kit.pdf", answers=answers, app_version="x")
    port = _free_port()
    common = {"settings": settings, "host": "127.0.0.1", "port": port}
    ok = cli.serve_problems(protocol, kit=kit_pdf, allow_dirty=False, version="0123abc",
                            answers=answers, **common)  # fmt: skip
    assert ok == []
    dirty = cli.serve_problems(protocol, kit=kit_pdf, allow_dirty=False,
                               version="0123abc-dirty", answers=answers, **common)  # fmt: skip
    assert len(dirty) == 1 and "tree is modified" in dirty[0]
    assert cli.serve_problems(protocol, kit=kit_pdf, allow_dirty=True, version="0123abc-dirty",
                              answers=answers, **common) == []  # fmt: skip
    moved = copy.deepcopy(answers)
    moved["run"]["signal_sha256"] = "cd" * 32
    msgs = cli.serve_problems(protocol, kit=kit_pdf, allow_dirty=False, version="0123abc",
                              answers=moved, **common)  # fmt: skip
    assert any("signal_sha256 changed" in m for m in msgs)
    none = cli.serve_problems(protocol, kit=tmp_path / "absent.pdf", allow_dirty=False,
                              version="0123abc", answers=answers, **common)  # fmt: skip
    assert any("build the kit first" in m for m in none)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", port))
        busy.listen()
        msgs = cli.serve_problems(protocol, kit=kit_pdf, allow_dirty=False,
                                  version="0123abc", answers=answers, **common)  # fmt: skip
    assert any("already in use" in m for m in msgs)
    pending = _fake_answers(protocol, pending=True)
    draft = kit.build_kit(protocol, tmp_path / "draft.pdf", answers=pending, app_version="x",
                          allow_pending_s3=True)  # fmt: skip
    assert kit.kit_identity(draft)["draft"] == "yes"
    msgs = cli.serve_problems(protocol, kit=draft, allow_dirty=False, version="0123abc",
                              answers=pending, **common)  # fmt: skip
    assert len(msgs) == 1 and "DRAFT kit" in msgs[0]


def test_protocol_page_names_every_task(protocol):
    text = (REPO / "docs" / "usertest_protocol.md").read_text(encoding="utf-8")
    for tid in pr.tasks_by_id(protocol):
        assert f"`{tid}`" in text
    for part in ("usertest-serve", "usertest-check", "usertest-docs", "ENF-07", "proxy", "U01"):
        assert part in text

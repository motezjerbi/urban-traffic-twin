"""The rehearsed demo (Part 10, D116; `traffic_twin.demo`, configs/demo.yaml): the demo
script, prerequisite messages, the impostor check, the fixtures rehearsal, the demo pack
and bootstrap, `demo clean`, the rehearsal evidence (A-08 / A-09) and the Windows
teardown. No SUMO, no real data, nothing written into the repo.
"""

from __future__ import annotations

import copy
import http.server
import json
import subprocess
import sys
import textwrap
import threading
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import uvicorn
import yaml

from traffic_twin.api.main import create_app
from traffic_twin.api.settings import load_settings
from traffic_twin.app import browser
from traffic_twin.config import ConfigError
from traffic_twin.demo import clean as clean_mod
from traffic_twin.demo.__main__ import main as demo_main
from traffic_twin.demo.bootstrap import bootstrap, rebuild_demo_runs
from traffic_twin.demo.common import (
    DemoError,
    ProcessGroup,
    Report,
    free_port,
    port_open,
    probe_api,
)
from traffic_twin.demo.config import (
    load_demo_config,
    resolve_runs,
    step_calls,
    step_url,
    validate,
)
from traffic_twin.demo.pack import MANIFEST, pack, verify
from traffic_twin.demo.prereq import prerequisites
from traffic_twin.demo.rehearsal import evidence, record_rehearsal, rehearsal
from traffic_twin.demo.scan import scan_mock
from traffic_twin.demo.serve import port_checks

REPO_ROOT = Path(__file__).resolve().parent.parent
TITLE = "Urban Traffic Twin API"


@pytest.fixture(scope="module")
def demo_cfg() -> dict:
    return load_demo_config(REPO_ROOT / "configs" / "demo.yaml")


# -- the demo script ---------------------------------------------------------------------
def test_demo_config_is_the_registered_script(demo_cfg):
    """Plan §4.5: ports, runs by (scenario, seed) at seed 4000, steps and degraded cases."""
    assert demo_cfg["api"] == {"host": "127.0.0.1", "port": 8000, "title": TITLE}
    assert demo_cfg["operator"] == {"host": "127.0.0.1", "port": 8501, "script": "app.py"}
    assert demo_cfg["runs"] == {
        "main": {"scenario": "pitie_rules", "seed": 4000},
        "s3": {"scenario": "pitie_rl_s3", "seed": 4000},
    }
    assert [s["id"] for s in demo_cfg["steps"]] == [
        "home",
        "pedestrian",
        "student",
        "bus",
        "ambulance",
        "replay",
        "operator",
        "streamlit",
    ]
    by = {s["id"]: s for s in demo_cfg["steps"]}
    assert (by["pedestrian"]["t"], by["student"]["t"], by["bus"]["t"]) == (1805, 2700, 1305)
    assert (by["ambulance"]["t"], by["ambulance"]["kiosk"], by["replay"]["run"]) == (2440, 1, "s3")
    assert demo_cfg["degraded"] == ["no_eta_bundle", "no_evaluation", "no_runs", "api_down"]
    assert set(demo_cfg["prerequisites"]) == {
        "demo_runs",
        "evaluation_outputs",
        "eta_reference",
        "s3_model_sha",
        "evaluation_page",
    }


def test_demo_config_errors_list_every_problem(demo_cfg):
    raw = yaml.safe_load((REPO_ROOT / "configs" / "demo.yaml").read_text(encoding="utf-8"))
    bad = copy.deepcopy(raw)
    bad["runs"]["main"]["seed"] = 1  # an evaluation seed: never in the demo (D101)
    bad["steps"].append({"id": "x", "view": "cockpit"})
    bad["api"]["host"] = "0.0.0.0"
    with pytest.raises(ConfigError) as e:
        validate(bad)
    msg = str(e.value)
    assert "demo seed 4000-4009" in msg and "cockpit" in msg and "must be local" in msg


def test_deep_links_name_scenario_and_seed_never_the_run_id(demo_cfg):
    by = {s["id"]: s for s in demo_cfg["steps"]}
    url = step_url(demo_cfg, by["pedestrian"], "http://h:1")
    assert url == (
        "http://h:1/mock?view=pedestrian&scenario=pitie_rules&seed=4000&t=1805&crossing=S"
        "&speed=normal"
    )
    assert "run_id" not in url
    assert step_url(demo_cfg, by["replay"], "http://h:1").endswith("&t=2440&play=0")
    assert "kiosk=1" in step_url(demo_cfg, by["ambulance"], "http://h:1")
    op = step_url(demo_cfg, by["operator"], "http://h:1", operator_url="http://o:2")
    assert op.endswith("view=operator&operator=http%3A%2F%2Fo%3A2")
    assert step_url(demo_cfg, by["streamlit"], "http://h:1") == "http://127.0.0.1:8501"


def test_runs_resolve_by_scenario_and_seed(demo_cfg):
    payload = {
        "scenarios": [
            {
                "scenario": "pitie_rules",
                "label": "S2",
                "controller": "rules",
                "runs": [
                    {"run_id": "old", "seed": 4000, "current": False, "dir": "a"},
                    {"run_id": "new", "seed": 4000, "current": True, "dir": "b"},
                    {"run_id": "eval", "seed": 1, "current": True, "dir": "c"},
                ],
            },
        ]
    }
    runs = resolve_runs(demo_cfg, payload)
    assert runs["main"]["run_id"] == "new" and runs["main"]["label"] == "S2"
    assert runs["s3"] is None
    replay = next(s for s in demo_cfg["steps"] if s["id"] == "replay")
    assert "not found by /scenarios" in step_calls(demo_cfg, replay, runs, [])
    ped = next(s for s in demo_cfg["steps"] if s["id"] == "pedestrian")
    (call,) = step_calls(demo_cfg, ped, runs, [])
    assert call.path == "/crossing/S?speed=normal&t=1805&run_id=new"


# -- prerequisites -----------------------------------------------------------------------
@pytest.fixture
def fake_settings(fake_api_config):
    return load_settings(fake_api_config, repo_root=REPO_ROOT, environ={})


def _rows(cfg, settings) -> dict[str, dict]:
    return {r["name"]: r for r in prerequisites(cfg, settings)}


def test_prerequisite_messages_carry_fix_hints(demo_cfg, fake_settings, tmp_path):
    rows = _rows(demo_cfg, fake_settings)
    # the FAKE fixtures have S0-S2 at seed 4000 but no S3 run: S3 is made in integration I3
    assert rows["demo_runs"]["status"] == "pending"
    assert "demo-runs8" in rows["demo_runs"]["hint"]
    assert rows["eta_reference"]["status"] == "ok"
    assert rows["s3_model_sha"]["status"] == "ok"  # the committed S3 model
    cfg = copy.deepcopy(demo_cfg)
    cfg["runs"]["main"]["scenario"] = "pitie_nothing"
    r = _rows(cfg, fake_settings)["demo_runs"]
    assert r["status"] == "missing" and "demo-runs" in r["hint"] and "bootstrap" in r["hint"]
    empty = tmp_path / "empty"
    empty.mkdir()
    no_eta = replace(fake_settings, eta_models_dir=empty)
    r = _rows(demo_cfg, no_eta)["eta_reference"]
    assert r["status"] == "missing" and "part5" in r["hint"]
    no_eval = replace(fake_settings, evaluation_dir=empty)
    r = _rows(demo_cfg, no_eval)["evaluation_outputs"]
    assert r["status"] == "missing" and "kpis.csv" in r["detail"]


def test_prerequisites_s3_model_and_page(demo_cfg, fake_settings, tmp_path):
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "models" / "m.zip").write_bytes(b"not the model")
    (root / "configs" / "pitie_rl_s3.yaml").write_text(
        "rl: {model: models/m.zip, model_sha256: " + "0" * 64 + "}\n", encoding="utf-8"
    )
    (root / "docs").mkdir()
    (root / "docs" / "evaluation.md").write_text("# Evaluation\n\n## 1. KPIs\n", encoding="utf-8")
    rows = _rows(demo_cfg, replace(fake_settings, repo_root=root))
    assert rows["s3_model_sha"]["status"] == "missing"
    assert "sha256" in rows["s3_model_sha"]["detail"]
    assert rows["evaluation_page"]["status"] == "pending"
    (root / "docs" / "evaluation.md").write_text(
        "# Evaluation\n\n## 1. Final comparison table (Part 8)\n", encoding="utf-8"
    )
    assert (
        _rows(demo_cfg, replace(fake_settings, repo_root=root))["evaluation_page"]["status"] == "ok"
    )


# -- impostor check ----------------------------------------------------------------------
class _Stub(http.server.BaseHTTPRequestHandler):
    title: str | None = "Mock API"

    def do_GET(self):  # noqa: N802
        if self.path == "/openapi.json" and self.title is not None:
            body = json.dumps({"info": {"title": self.title}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        else:
            body = b"<html>FAUSSES DONNEES</html>"
            self.send_response(404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _stub(title):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), type("S", (_Stub,), {"title": title}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.mark.parametrize("title", ["Mock API", None])
def test_impostor_check_rejects_a_stub_server(title, demo_cfg):
    srv, base = _stub(title)
    try:
        kind, detail = probe_api(base, TITLE)
        assert kind == "impostor", detail
        port = srv.server_address[1]
        problems = port_checks(
            {**demo_cfg, "api": {**demo_cfg["api"], "port": port}},
            port,
            free_port(),
            operator=False,
        )
        assert len(problems) == 1 and "run_mock.py" in problems[0]
    finally:
        srv.shutdown()
        srv.server_close()
    assert probe_api(base, TITLE)[0] == "free"


def test_impostor_check_accepts_the_real_api(fake_api_config):
    settings = replace(load_settings(fake_api_config, repo_root=REPO_ROOT, environ={}), mock=True)
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="warning")
    )
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        assert probe_api(f"http://127.0.0.1:{port}", TITLE) == (
            "api",
            f"/openapi.json title {TITLE!r}",
        )
    finally:
        server.should_exit = True
        th.join(10)


# -- the fixtures rehearsal --------------------------------------------------------------
def test_rehearsal_fixtures_no_browser_passes(demo_cfg, tmp_path):
    """The CI rehearsal: FAKE data, every API step, the degraded cases, the static scan and
    Streamlit AppTest pass; only Part 8 outputs may be PENDING (D110, D116)."""
    t0 = time.monotonic()
    res = rehearsal(
        demo_cfg, fixtures=True, browser=False, root=REPO_ROOT, out_dir=tmp_path / "r", quiet=True
    )
    elapsed = time.monotonic() - t0
    failed = [r for r in res["rows"] if r["result"] == "FAIL"]
    assert not failed, failed
    assert res["passed"] and res["status"] in ("PASS", "PENDING")
    pending = [r for r in res["rows"] if r["result"] == "PENDING"]
    assert all(r["group"] == "prereq" or "pas encore" in r["detail"] for r in pending), pending
    groups = {r["group"] for r in res["rows"]}
    assert {
        "prereq",
        "boot",
        "home",
        "pedestrian",
        "student",
        "bus",
        "ambulance",
        "replay",
        "operator",
        "no_eta_bundle",
        "no_evaluation",
        "no_runs",
        "api_down",
        "streamlit",
        "teardown",
    } <= groups
    assert res["degraded"] == dict.fromkeys(
        ["no_eta_bundle", "no_evaluation", "no_runs", "api_down"], "PASS"
    )
    assert (tmp_path / "r" / "report.json").is_file() and (tmp_path / "r" / "report.md").is_file()
    assert elapsed < 90, f"fixtures rehearsal took {elapsed:.0f} s"


def test_rehearsal_refuses_to_record_fake_data(demo_cfg, tmp_path):
    with pytest.raises(DemoError, match="no A-08 evidence"):
        rehearsal(
            demo_cfg,
            fixtures=True,
            browser=False,
            record=True,
            root=REPO_ROOT,
            out_dir=tmp_path,
            quiet=True,
        )
    assert demo_main(["rehearsal", "--fixtures", "--record", "--out", str(tmp_path)]) == 1


def test_cli_reports_a_bad_config(tmp_path, capsys):
    bad = tmp_path / "demo.yaml"
    bad.write_text("api: {}\n", encoding="utf-8")
    assert demo_main(["--config", str(bad), "clean", "--dry-run"]) == 2
    assert "missing key" in capsys.readouterr().err


# -- evidence ----------------------------------------------------------------------------
def _row(group: str, check: str, result: str = "PASS") -> dict:
    return {"group": group, "check": check, "result": result, "detail": ""}


def _result(
    status: str,
    *,
    fresh: bool,
    dirty: bool = False,
    root: Path,
    browser: str | None = "chrome.exe",
    dom: str = "PASS",
    clone_rows: bool = True,
) -> dict:
    """A recorded run: `fresh` makes it a `demo readiness` result (kind readiness, clone
    rows); `browser` None / `dom` SKIP model a run without DOM checks."""
    rows = [
        _row("g", f"{root}/x", status),
        _row("dom home", "data-view-state root", dom),
        _row("dom pedestrian", "deep link DOM", dom),
        _row("operator", "Operator (Streamlit) page", "SKIP"),  # allowed: AppTest covers it
    ]
    if fresh and clone_rows:
        from traffic_twin.demo.readiness import CLONE_CHECKS

        rows += [_row("clone", c) for c in CLONE_CHECKS]
    return {
        "kind": "readiness" if fresh else "rehearsal",
        "mode": "real",
        "fresh_clone": fresh,
        "status": status,
        "passed": status != "FAIL",
        "summary": {},
        "started_at": "x",
        "browser": browser,
        "steps": [
            {"id": "home", "front": "mock", "url": "<api>/mock?view=home"},
            {"id": "pedestrian", "front": "mock", "url": "<api>/mock?view=pedestrian"},
            {"id": "operator", "front": "streamlit", "url": "http://127.0.0.1:8501"},
        ],
        "git": {"code_version": "abc1234" + ("-dirty" if dirty else ""), "dirty": dirty},
        "rows": rows,
    }


def test_rehearsal_evidence_a08_a09(tmp_path):
    root = tmp_path
    path = root / "docs" / "demo" / "rehearsal.json"
    doc = record_rehearsal(_result("PASS", fresh=False, root=root), path, root)
    assert doc["a08"]["status"] == "PASS" and doc["a09"]["status"] == "PENDING"
    assert doc["rehearsal"]["coverage"]["dom"] == {"home": "PASS", "pedestrian": "PASS"}
    assert doc["rehearsed_demo"]["status"] == "PENDING"
    assert str(root) not in path.read_text(encoding="utf-8")  # local paths scrubbed
    doc = record_rehearsal(_result("PASS", fresh=True, dirty=True, root=root), path, root)
    assert doc["a09"]["status"] == "PENDING" and "dirty" in doc["a09"]["reason"]
    doc = record_rehearsal(_result("PASS", fresh=True, root=root), path, root)
    assert doc["a09"]["status"] == "PASS" and doc["a08"]["from"] == "readiness"
    assert doc["rehearsal"]["status"] == "PASS"  # the earlier block is kept
    doc = record_rehearsal(_result("FAIL", fresh=True, root=root), path, root)
    assert doc["a09"]["status"] == "FAIL" and doc["a08"]["status"] == "FAIL"
    log = root / "docs" / "demo" / "rehearsal_log.yaml"
    log.write_text(
        "entries:\n- {date: 2026-11-01, commit: abc1234, manual_code_edits: 1}\n", encoding="utf-8"
    )
    assert evidence(doc, root)["rehearsed_demo"]["status"] == "PENDING"
    log.write_text(
        "entries:\n- {date: 2026-11-01, commit: abc1234, manual_code_edits: 0}\n", encoding="utf-8"
    )
    assert evidence(doc, root)["rehearsed_demo"]["status"] == "DONE"


def test_a09_needs_a_real_readiness_run_not_a_fresh_clone_label(tmp_path):
    """Finding: a plain rehearsal labelled `fresh_clone` was filed as the readiness block
    and gave A-09 PASS with no clone, pack or bootstrap."""
    root = tmp_path
    path = root / "docs" / "demo" / "rehearsal.json"
    forged = {
        "kind": "rehearsal",
        "fresh_clone": True,
        "status": "PASS",
        "rows": [],
        "git": {"dirty": False},
    }
    doc = record_rehearsal(forged, path, root)
    assert "readiness" not in doc  # filed by kind, never by the label
    assert doc["a09"]["status"] == "PENDING" and doc["a08"]["status"] == "PENDING"
    # a readiness-kind result without the clone / bootstrap / clean-tree rows: no A-09
    doc = record_rehearsal(_result("PASS", fresh=True, clone_rows=False, root=root), path, root)
    assert doc["a09"]["status"] == "PENDING"
    assert "bootstrap --from-pack MISSING" in doc["a09"]["reason"]
    # a hand-made readiness block with another kind is refused too
    doc["readiness"]["kind"] = "rehearsal"
    assert evidence(doc, root)["a09"]["status"] == "PENDING"


@pytest.mark.parametrize(
    ("browser_name", "dom", "why"),
    [(None, "SKIP", "no headless browser"), ("chrome.exe", "SKIP", "dom home SKIP")],
)
def test_a08_a09_need_the_dom_checks(tmp_path, browser_name, dom, why):
    """Finding: a --no-browser (or browser-less) run recorded A-08 / A-09 PASS although no
    view's data-view-state, hooks or footer was checked (SKIP rows do not fail a run)."""
    root = tmp_path
    path = root / "docs" / "demo" / "rehearsal.json"
    doc = record_rehearsal(
        _result("PASS", fresh=False, browser=browser_name, dom=dom, root=root), path, root
    )
    assert doc["a08"]["status"] == "PENDING" and why in doc["a08"]["reason"]
    doc = record_rehearsal(
        _result("PASS", fresh=True, browser=browser_name, dom=dom, root=root), path, root
    )
    assert doc["a09"]["status"] == "PENDING" and why in doc["a09"]["reason"]
    # a block recorded before coverage existed is no evidence either
    del doc["readiness"]["coverage"]
    assert evidence(doc, root)["a09"]["status"] == "PENDING"


def test_record_refuses_no_browser_and_fresh_clone(demo_cfg, tmp_path):
    for kw, msg in (
        ({"browser": False}, "needs the DOM checks"),
        ({"fresh_clone": True}, "demo-readiness"),
    ):
        with pytest.raises(DemoError, match=msg):
            rehearsal(demo_cfg, record=True, root=REPO_ROOT, out_dir=tmp_path, quiet=True, **kw)
    out = str(tmp_path / "o")
    assert demo_main(["rehearsal", "--record", "--no-browser", "--out", out]) == 1
    assert demo_main(["rehearsal", "--record", "--fresh-clone", "--out", out]) == 1
    from traffic_twin.demo.readiness import readiness

    with pytest.raises(DemoError, match="needs the DOM checks"):
        readiness(demo_cfg, root=REPO_ROOT, browser=False, record=True, quiet=True)
    assert demo_main(["readiness", "--no-browser"]) == 1
    assert not (tmp_path / "o").exists()


def test_recorded_rehearsal_without_a_browser_is_pending(demo_cfg, tmp_path, monkeypatch):
    """No Chrome / Edge while recording: a PENDING row (not SKIP), so neither the run nor
    A-08 can be PASS. The walkthrough itself is stubbed (no API, no real data read)."""
    from traffic_twin.demo import rehearsal as rehearsal_mod

    monkeypatch.setattr(browser, "find_browser", lambda: None)
    monkeypatch.setattr(rehearsal_mod, "_rehearse", lambda *a, **k: None)
    rec = tmp_path / "rehearsal.json"
    res = rehearsal(
        demo_cfg, record=True, root=REPO_ROOT, out_dir=tmp_path / "r", record_path=rec, quiet=True
    )
    row = next(r for r in res["rows"] if r["check"] == "headless browser")
    assert row["result"] == "PENDING" and res["status"] == "PENDING"
    doc = json.loads(rec.read_text(encoding="utf-8"))
    assert doc["a08"]["status"] == "PENDING" and doc["a09"]["status"] == "PENDING"


def test_committed_rehearsal_log_is_empty():
    raw = yaml.safe_load((REPO_ROOT / "docs" / "demo" / "rehearsal_log.yaml").read_text("utf-8"))
    assert raw["entries"] == []  # filled by a human after a real rehearsal only


# -- pack and bootstrap ------------------------------------------------------------------
def _fake_tree(root: Path, scenarios=("pitie_rules",)) -> None:
    for s in scenarios:
        d = root / "data" / "demo" / f"{s}_seed4000"
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps({"scenario": s, "seed": 4000}), encoding="utf-8")
        (d / "kpi_run.csv").write_text("kpi,value\nx,1\n", encoding="utf-8")
        pd.DataFrame({"t": [0.0, 5.0]}).to_parquet(d / "state_log.parquet")
        (d / "tripinfo.xml").write_text("<tripinfos/>", encoding="utf-8")  # never packed
    ev = root / "data" / "results" / "evaluation" / "main"
    (ev / "pairs").mkdir(parents=True)
    (ev / "kpis.csv").write_text("label,value\nS0,1\n", encoding="utf-8")
    (ev / "pairs" / "S0__vs__S2.csv").write_text("kpi\nx\n", encoding="utf-8")
    eta = root / "data" / "models" / "eta"
    (eta / "v1").mkdir(parents=True)
    (eta / "LATEST").write_text("v1\n", encoding="utf-8")
    (eta / "freeze.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({"route": ["N_S"]}).to_parquet(eta / "v1" / "reference.parquet")
    (eta / "v1" / "meta.json").write_text("{}", encoding="utf-8")
    (eta / "v1" / "model.ubj").write_bytes(b"\0" * 64)  # the big model: never packed
    (root / "data" / "results" / "pitie_rules_seed1").mkdir(parents=True)  # an evaluation run


def _pack_cfg(demo_cfg, scenarios=("pitie_rules",)) -> dict:
    return {**demo_cfg, "pack": {**demo_cfg["pack"], "runs": list(scenarios)}}


def test_pack_manifest_verifies_and_bootstrap_restores(demo_cfg, tmp_path):
    src = tmp_path / "src"
    _fake_tree(src)
    cfg = _pack_cfg(demo_cfg)
    out = pack(cfg, src / "docs" / "demo" / "demo_pack.zip", root=src)
    first = out.read_bytes()
    pack(cfg, out, root=src)
    assert out.read_bytes() == first  # deterministic zip
    manifest = verify(out, out.parent / MANIFEST)
    names = [e["path"] for e in manifest["files"]]
    assert "data/demo/pitie_rules_seed4000/run.json" in names
    assert "data/results/evaluation/main/pairs/S0__vs__S2.csv" in names
    assert "data/models/eta/v1/reference.parquet" in names
    assert not [n for n in names if n.endswith((".xml", ".ubj"))]
    assert not [n for n in names if "seed1" in n]  # no evaluation run folder
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    restored = bootstrap(out, fresh)
    assert len(restored) == len(names)
    for n in names:
        assert (fresh / n).read_bytes() == (src / n).read_bytes()
    with pytest.raises(DemoError, match="refuses to overwrite"):
        bootstrap(out, fresh)
    other = tmp_path / "partial"
    (other / "data" / "demo" / "pitie_rules_seed4000").mkdir(parents=True)
    (other / "data" / "demo" / "pitie_rules_seed4000" / "run.json").write_text("{}", "utf-8")
    with pytest.raises(DemoError, match="nothing was written"):
        bootstrap(out, other)
    assert not (other / "data" / "results").exists()  # all or nothing


def test_pack_refuses_missing_inputs_and_tampering(demo_cfg, tmp_path):
    src = tmp_path / "src"
    _fake_tree(src)
    with pytest.raises(DemoError, match="pitie_rl_s3_seed4000 not found"):
        pack(_pack_cfg(demo_cfg, ("pitie_rules", "pitie_rl_s3")), src / "p.zip", root=src)
    out = pack(
        _pack_cfg(demo_cfg, ("pitie_rules", "pitie_rl_s3")),
        src / "p.zip",
        root=src,
        allow_missing=True,
    )
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(out) as zin, zipfile.ZipFile(bad, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith("kpis.csv"):
                data += b"S3,0\n"
            zout.writestr(item, data)
    with pytest.raises(DemoError, match="sha256 or size"):
        verify(bad)
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(out) as zin, zipfile.ZipFile(evil, "w") as zout:
        m = json.loads(zin.read(MANIFEST))
        m["files"].append({"path": "../escape.txt", "sha256": "0", "bytes": 1})
        zout.writestr(MANIFEST, json.dumps(m))
        zout.writestr("../escape.txt", b"x")
        for item in zin.infolist():
            if item.filename != MANIFEST:
                zout.writestr(item, zin.read(item.filename))
    with pytest.raises(DemoError, match="unsafe path"):
        verify(evil)


def test_bootstrap_demo_only_rebuilds_seed_4000_only(demo_cfg, tmp_path):
    cfg = {**demo_cfg, "pack": {**demo_cfg["pack"], "seed": 1}}
    with pytest.raises(DemoError, match="seed 4000 only"):
        rebuild_demo_runs(tmp_path, cfg)


# -- clean -------------------------------------------------------------------------------
def test_clean_never_touches_results_demo_or_models(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    keep = [
        root / "data" / "results" / "pitie_rules_seed1" / "kpi_run.csv",
        root / "data" / "demo" / "pitie_rules_seed4000" / "run.json",
        root / "data" / "models" / "eta" / "LATEST",
        root / "models" / "rl" / "m" / "model.zip",
    ]
    for f in keep:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    check = root / "data" / "demo_check"
    for name in ("20261020-100000-rehearsal", "20261021-100000-check", "20261022-100000-serve"):
        (check / name).mkdir(parents=True)
        (check / name / "report.json").write_text("{}", encoding="utf-8")
    (check / "stray").mkdir()
    (root / "last_dom.html").write_text("<html/>", encoding="utf-8")
    tmp = tmp_path / "systmp"
    (tmp / "tt_demo_rehearsal_old").mkdir(parents=True)
    (tmp / "other_tool").mkdir()
    old = time.time() - 7200
    import os

    os.utime(tmp / "tt_demo_rehearsal_old", (old, old))
    (tmp / "tt_demo_browser_new").mkdir()  # young: may belong to a running rehearsal
    monkeypatch.setattr(clean_mod.tempfile, "gettempdir", lambda: str(tmp))
    removed = clean_mod.clean(root, keep=1)
    assert all(f.is_file() for f in keep)
    assert not (check / "20261020-100000-rehearsal").exists()
    assert not (check / "20261021-100000-check").exists()
    assert (check / "20261022-100000-serve").is_dir()  # the newest is kept
    assert not (root / "last_dom.html").exists() and not (check / "stray").exists()
    assert not (tmp / "tt_demo_rehearsal_old").exists()
    assert (tmp / "tt_demo_browser_new").is_dir() and (tmp / "other_tool").is_dir()
    assert len(removed) == 5
    for f in keep:
        with pytest.raises(DemoError):
            clean_mod._remove(f, root, dry_run=True)


# -- static scan and browser helpers -----------------------------------------------------
def test_static_scan_flags_invented_values(tmp_path):
    page = tmp_path / "src" / "traffic_twin" / "app" / "mock.html"
    page.parent.mkdir(parents=True)
    page.write_text(
        textwrap.dedent("""
        <script>const S3_DATA = {gain: "-25.5 %"};
        const x = b.next_arrival_s || "08:28"; const n = Math.random();
        api(`/kpi?scenario=S0`); api("/secret");</script>"""),
        encoding="utf-8",
    )
    rep = Report(quiet=True)
    scan_mock(rep, tmp_path)
    failed = {r["check"]: r["detail"] for r in rep.rows if r["result"] == "FAIL"}
    assert "S3_DATA" in failed["no banned string (D107)"]
    assert "Math.random" in failed["no banned string (D107)"]
    assert "next_arrival_s" in failed["no b.<field> || <literal> fallback"]
    assert "/secret" in failed["every fetched path is an API route"]


def test_dom_hooks_are_parsed_without_a_browser():
    dom = (
        '<main data-view="pedestrian" data-view-state="degraded"><div data-field="ped-verdict">'
        'Vous <b>pouvez</b> traverser</div><input data-field="exam" value="08:18">'
        '<script>var s = "<div data-field=x>";</script></main>'
        '<footer><span data-field="footer-source">Données simulées</span></footer>'
    )
    assert browser.fields(dom) == {
        "ped-verdict": "Vous pouvez traverser",
        "exam": "08:18",
        "footer-source": "Données simulées",
    }
    (state,) = browser.view_states(dom)
    assert state["state"] == "degraded" and state["attrs"]["data-view"] == "pedestrian"


def test_find_browser_honours_tt_browser(monkeypatch, tmp_path):
    exe = tmp_path / "chrome.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("TT_BROWSER", str(exe))
    assert browser.find_browser() == exe
    monkeypatch.setenv("TT_BROWSER", str(tmp_path / "missing.exe"))
    assert browser.find_browser() is None


# -- teardown ----------------------------------------------------------------------------
LISTENER = """
import socket, subprocess, sys, time
if sys.argv[1] == "parent":
    subprocess.Popen([sys.executable, __file__, "child", sys.argv[2]])
    time.sleep(600)
else:
    s = socket.socket(); s.bind(("127.0.0.1", int(sys.argv[2]))); s.listen()
    time.sleep(600)
"""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process-tree teardown")
def test_windows_teardown_leaves_no_listener(tmp_path):
    """A child that started a listening grandchild: the group stop frees the port and
    removes the pids file (CTRL_BREAK, then taskkill /T /F)."""
    script = tmp_path / "listener.py"
    script.write_text(LISTENER, encoding="utf-8")
    port = free_port()
    pids = tmp_path / "demo_check" / ".pids.json"
    group = ProcessGroup(pids, stop_s=10)
    child = group.start(
        "tree",
        [sys.executable, str(script), "parent", str(port)],
        cwd=tmp_path,
        env=dict(__import__("os").environ),
        port=port,
    )
    deadline = time.monotonic() + 20
    while not port_open("127.0.0.1", port) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert port_open("127.0.0.1", port), "the grandchild never listened"
    assert json.loads(pids.read_text(encoding="utf-8"))["children"][0]["pid"] == child.proc.pid
    freed = group.stop()
    assert freed == {"tree": True}
    assert not port_open("127.0.0.1", port)
    assert not pids.exists()
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {child.proc.pid}"], capture_output=True, text=True
    ).stdout
    assert str(child.proc.pid) not in out

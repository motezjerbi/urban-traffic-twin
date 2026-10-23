"""Full demo rehearsal (D116), headless:

1. prerequisites (configs/demo.yaml), each with its fix hint;
2. the API on a free port (uvicorn child process);
3. the walkthrough of every step (`demo.check`: API calls, and DOM checks with a browser);
4. the static honesty scan of mock.html and app.py (`demo.scan`);
5. the degraded cases, each on its own API: no ETA bundle, no evaluation outputs, no runs;
   and the API down (the page served without an API; Streamlit pointed at a dead port);
6. Streamlit AppTest of the Operator (`app.py`), nominal;
7. teardown: every port freed.

Writes `data/demo_check/<ts>-rehearsal/report.{json,md}` (plus the DOMs and logs).
`--record` writes the A-08 / A-09 evidence `docs/demo/rehearsal.json` (real data only);
`--fixtures` runs on `api.fixtures` FAKE data in a temporary folder (CI: no SUMO, no real
data). Never runs a simulation.

    python -m traffic_twin.demo rehearsal [--fixtures] [--no-browser] [--record]
"""

from __future__ import annotations

import contextlib
import http.server
import io
import json
import platform
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import yaml

from traffic_twin.demo.apptest import check_down, check_nominal, run_apptest
from traffic_twin.demo.check import Call, call_api, check, dump_many, fold, view_root
from traffic_twin.demo.common import (
    DemoError,
    ProcessGroup,
    Report,
    api_cmd,
    check_dir,
    child_env,
    free_port,
    git_info,
    now_iso,
    wait_http,
    write_report,
)
from traffic_twin.demo.config import resolve_runs, step_url
from traffic_twin.demo.prereq import prerequisites
from traffic_twin.paths import API_CONFIG, DEMO_DOCS_DIR

REHEARSAL_JSON = DEMO_DOCS_DIR / "rehearsal.json"
REHEARSAL_LOG = DEMO_DOCS_DIR / "rehearsal_log.yaml"
PREREQ_RESULT = {"ok": "PASS", "pending": "PENDING", "missing": "FAIL"}
DOWN_STATES = frozenset({"error"})
ANY_STATE = frozenset({"ok", "degraded", "unavailable", "error"})
SCHEMA = 1


# -- fixtures ----------------------------------------------------------------------------
def build_fixtures(work: Path) -> Path:
    """FAKE data (`api.fixtures`, 7200 s runs so every scripted instant exists) plus a FAKE
    S3 run without ETA predictions (the replay step); returns its api.yaml."""
    from traffic_twin.api import fixtures

    out = work / "fake"
    with contextlib.redirect_stdout(io.StringIO()):
        fixtures.main(["--out", str(out), "--n-runs", "3", "--duration-s", "7200"])
    s3 = out / "pitie_rl_s3_seed4000"
    if not (s3 / "run.json").is_file():
        fixtures.make_run(s3, scenario="pitie_rl_s3", controller="rl", seed=4000, duration_s=7200.0)
        rules = out / "pitie_rules_seed4000" / "events.parquet"
        if rules.is_file() and not (s3 / "events.parquet").is_file():  # a FAKE event log too
            import pandas as pd

            run_id = json.loads((s3 / "run.json").read_text(encoding="utf-8"))["run_id"]
            pd.read_parquet(rules).assign(run_id=run_id).to_parquet(s3 / "events.parquet")
    return out / "api.yaml"


def api_config_path(root: Path, api_config: Path | None) -> Path:
    import os

    from traffic_twin.api.settings import ENV_CONFIG

    p = Path(api_config or os.environ.get(ENV_CONFIG) or API_CONFIG)
    return p if p.is_absolute() else root / p


# -- servers -----------------------------------------------------------------------------
def start_api(group: ProcessGroup, root: Path, cfg_path: Path, log: Path, name: str = "api"):
    port = free_port()
    env = child_env(TT_API_CONFIG=str(cfg_path), TT_RESULTS_DIRS=None, TT_API_INPROCESS=None)
    child = group.start(name, api_cmd(port), cwd=root, env=env, port=port, log=log)
    return child, f"http://127.0.0.1:{port}"


class _PageOnly(http.server.BaseHTTPRequestHandler):
    """Serves mock.html (and three.min.js) with no API behind: every other path answers
    503 text, as when the API process is down."""

    root: Path

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        app = self.root / "src" / "traffic_twin" / "app"
        path = self.path.split("?", 1)[0]
        files = {
            "/mock": ("mock.html", "text/html; charset=utf-8"),
            "/three.min.js": ("three.min.js", "application/javascript"),
        }
        if path in files:
            body = (app / files[path][0]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", files[path][1])
        else:
            body = b"API down"
            self.send_response(503)
            self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


@contextlib.contextmanager
def page_without_api(root: Path):
    handler = type("PageOnly", (_PageOnly,), {"root": root})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


# -- degraded cases ----------------------------------------------------------------------
def degraded_configs(raw: dict, work: Path, cases: list[str]) -> dict[str, Path]:
    empty = work / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    e = empty.as_posix()
    variants = {
        "no_eta_bundle": {"eta": {**raw["eta"], "models_dir": e}},
        "no_evaluation": {"evaluation": {**raw["evaluation"], "dir": e}},
        "no_runs": {"results_dirs": [e], "default_run": [e]},
    }
    out = {}
    for case in cases:
        if case in variants:
            p = work / f"api_{case}.yaml"
            p.write_text(
                yaml.safe_dump({**raw, **variants[case]}, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            out[case] = p
    return out


def _step(cfg: dict, view: str) -> dict | None:
    return next((s for s in cfg["steps"] if s.get("view") == view), None)


def _dom_generic(
    rep: Report, group: str, view: str, dom: str | None, err: str, allowed: frozenset[str]
) -> None:
    from traffic_twin.app.browser import text, view_states

    if dom is None:
        rep.add(group, f"{view} view DOM", "FAIL", err)
        return
    root = view_root(view_states(dom), view)
    if root is None:
        rep.add(group, f"{view} view state", "FAIL", "no data-view-state in the DOM")
        return
    rep.check(
        group, f"{view} view state in {sorted(allowed)}", root["state"] in allowed, root["state"]
    )
    body = root["text"] or text(dom)
    rep.check(
        group,
        f"{view} view not blank, no loading left",
        bool(body.strip()) and "chargement" not in fold(root["text"]),
        body[:120],
    )


def run_degraded(
    rep: Report,
    cfg: dict,
    root: Path,
    cfg_path: Path,
    work: Path,
    out: Path,
    group: ProcessGroup,
    browser: bool,
    budget: float,
) -> dict[str, str]:
    """The degraded cases into `rep`; {case: PASS | FAIL}."""
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cases = [c for c in cfg["degraded"] if c != "api_down"]
    paths = degraded_configs(raw, work, cases)
    servers = {
        c: start_api(group, root, p, out / f"api_{c}.log", name=f"api_{c}")
        for c, p in paths.items()
    }
    status: dict[str, str] = {}
    dom_jobs = []
    for case, (child, base) in servers.items():
        sub = Report(quiet=rep.quiet)
        try:
            wait_http(base + "/health", timeout_s=cfg["budgets"]["api_start_s"], proc=child.proc)
        except DemoError as e:
            sub.add(case, "API up", "FAIL", str(e))
            rep.extend(sub)
            status[case] = "FAIL"
            continue
        with httpx.Client(base_url=base, timeout=10.0) as c:
            runs = resolve_runs(cfg, c.get("/scenarios").json())
            if case == "no_eta_bundle":
                h = call_api(sub, c, Call(case, "/health", "health"), budget) or {}
                sub.check(
                    case,
                    "/health eta_model_version null",
                    h.get("eta_model_version") is None,
                    str(h.get("eta_model_version")),
                )
                st = _step(cfg, "student") or {"trajet": "N_S", "t": 2700}
                q = f"trajet={st.get('trajet', 'N_S')}&t={st.get('t', 2700)}"
                for name in ("main", "s3"):
                    rid = (runs.get(name) or {}).get("run_id")
                    if rid is None:
                        continue
                    b = (
                        call_api(
                            sub,
                            c,
                            Call(
                                case,
                                f"/eta?{q}&run_id={rid}",
                                "eta",
                                ok=frozenset(ANY_STATE - {"error"}),
                            ),
                            budget,
                        )
                        or {}
                    )
                    if b.get("status") != "ok":
                        sub.check(
                            case,
                            f"/eta ({name} run) {b.get('status')} says why",
                            bool(b.get("notice")),
                            b.get("notice") or "no notice",
                        )
                if (s := _step(cfg, "student")) is not None:
                    dom_jobs.append(
                        (
                            case,
                            "student",
                            step_url(cfg, s, base),
                            frozenset({"ok", "degraded", "unavailable"}),
                        )
                    )
            elif case == "no_evaluation":
                b = call_api(sub, c, Call(case, "/kpi?scenario=S1", "kpi_table"), budget) or {}
                sub.check(
                    case,
                    "/kpi falls back to the runs (degraded, source runs)",
                    b.get("status") == "degraded" and b.get("source") == "runs",
                    f"{b.get('status')} source={b.get('source')}",
                )
                b = (
                    call_api(
                        sub,
                        c,
                        Call(
                            case,
                            "/evaluation?family=main",
                            "evaluation",
                            ok=frozenset({"unavailable", "degraded"}),
                        ),
                        budget,
                    )
                    or {}
                )
                sub.check(
                    case, "/evaluation says why", bool(b.get("notice")), b.get("notice") or ""
                )
                if (s := _step(cfg, "operator")) is not None:
                    dom_jobs.append(
                        (
                            case,
                            "operator",
                            step_url(cfg, s, base),
                            frozenset({"ok", "degraded", "unavailable"}),
                        )
                    )
            elif case == "no_runs":
                call_api(
                    sub, c, Call(case, "/health", "health", ok=frozenset({"degraded"})), budget
                )
                call_api(
                    sub,
                    c,
                    Call(case, "/scenarios", "scenario_list", ok=frozenset({"degraded"})),
                    budget,
                )
                r = c.get("/replay/default?t=08:15")
                code = (
                    (
                        r.json()
                        if r.headers.get("content-type", "").startswith("application/json")
                        else {}
                    ).get("error")
                    or {}
                ).get("code")
                sub.check(
                    case,
                    "/replay/default -> 404 unknown_run (never a 500)",
                    r.status_code == 404 and code == "unknown_run",
                    f"http {r.status_code} {code}",
                )
                if (s := _step(cfg, "pedestrian")) is not None:
                    dom_jobs.append(
                        (
                            case,
                            "pedestrian",
                            step_url(cfg, s, base),
                            frozenset({"degraded", "unavailable", "error"}),
                        )
                    )
        rep.extend(sub)
        status[case] = "FAIL" if sub.count("FAIL") else "PASS"
    if browser and dom_jobs:
        doms = dump_many([u for _c, _v, u, _a in dom_jobs], int(cfg["budgets"]["browser_wait_ms"]))
        for (case, view, _u, allowed), (dom, err) in zip(dom_jobs, doms, strict=True):
            sub = Report(quiet=rep.quiet)
            _dom_generic(sub, case, view, dom, err, allowed)
            rep.extend(sub)
            if sub.count("FAIL"):
                status[case] = "FAIL"
    return status


def run_api_down(rep: Report, cfg: dict, root: Path, browser: bool, res: dict) -> str:
    """The API-down case: `res` is the AppTest answer with TT_API_URL on a dead port."""
    sub = Report(quiet=rep.quiet)
    check_down(sub, res)
    if browser:
        with page_without_api(root) as base:
            views = [s for s in (_step(cfg, "home"), _step(cfg, "pedestrian")) if s]
            doms = dump_many(
                [step_url(cfg, s, base) for s in views], int(cfg["budgets"]["browser_wait_ms"])
            )
        for s, (dom, err) in zip(views, doms, strict=True):
            _dom_generic(sub, "api_down", s["view"], dom, err, DOWN_STATES)
    else:
        sub.add("api_down", "mock views with the API down", "SKIP", "--no-browser")
    rep.extend(sub)
    return "FAIL" if sub.count("FAIL") else "PASS"


# -- the rehearsal -----------------------------------------------------------------------
def rehearsal(
    cfg: dict,
    *,
    fixtures: bool = False,
    browser: bool = True,
    record: bool = False,
    fresh_clone: bool = False,
    root: Path | None = None,
    api_config: Path | None = None,
    out_dir: Path | None = None,
    record_path: Path | None = None,
    quiet: bool = False,
) -> dict:
    """The rehearsal report {status, passed, summary, rows, ...}; written to
    `data/demo_check/<ts>-rehearsal/` (or `out_dir`), and to docs/demo/rehearsal.json with
    `record`. `record` is refused with `fixtures` (FAKE data is no A-08 evidence), without
    `browser` (no DOM check is no A-08 evidence) and with `fresh_clone` (the A-09 block is
    written by `demo readiness` only, after its clone, pack and clean-tree checks)."""
    from traffic_twin.api.settings import find_repo_root
    from traffic_twin.app.browser import find_browser

    if record and fixtures:
        raise DemoError("--record needs the real data: a fixtures rehearsal is no A-08 evidence")
    if record and not browser:
        raise DemoError(
            "--record needs the DOM checks: a --no-browser rehearsal is no A-08 evidence"
        )
    if record and fresh_clone:
        raise DemoError(
            "--record with --fresh-clone is refused: the A-09 evidence is recorded by "
            "`python tasks.py demo-readiness` (clone + pack + clean-tree checks) only"
        )
    root = Path(root or find_repo_root()).resolve()
    out = check_dir(root, "rehearsal", out_dir)
    rep = Report(quiet=quiet)
    t0 = time.monotonic()
    started = now_iso()
    script = cfg["operator"]["script"]
    budget = float(cfg["budgets"]["api_ms"])
    exe = find_browser() if browser else None
    use_browser = exe is not None
    if browser and exe is None:
        rep.add(
            "boot",
            "headless browser",
            "PENDING" if record else "SKIP",
            "no Chrome or Edge found: DOM checks skipped",
        )
    work = Path(tempfile.mkdtemp(prefix="tt_demo_rehearsal_"))
    result: dict = {
        "kind": "rehearsal",
        "mode": "fixtures" if fixtures else "real",
        "fresh_clone": fresh_clone,
        "started_at": started,
        "browser": exe.name if exe else None,
    }
    group = ProcessGroup(stop_s=float(cfg["budgets"]["stop_s"]))
    try:
        _rehearse(
            cfg,
            rep,
            result,
            root,
            out,
            work,
            group,
            fixtures,
            api_config,
            use_browser,
            budget,
            script,
            quiet,
        )
    finally:
        names = [c.name for c in group.children]
        freed = group.stop()
        for name in names:
            rep.check("teardown", f"{name}: port freed", freed.get(name, True))
        shutil.rmtree(work, ignore_errors=True)
    return _done(result, rep, t0, out, root, record, record_path)


def _rehearse(
    cfg: dict,
    rep: Report,
    result: dict,
    root: Path,
    out: Path,
    work: Path,
    group: ProcessGroup,
    fixtures: bool,
    api_config: Path | None,
    use_browser: bool,
    budget: float,
    script: str,
    quiet: bool,
) -> None:
    from traffic_twin.api.settings import load_settings
    from traffic_twin.demo.scan import scan

    cfg_path = build_fixtures(work) if fixtures else api_config_path(root, api_config)
    settings = load_settings(cfg_path, repo_root=root)
    result["api_config"] = "api.fixtures (FAKE)" if fixtures else _rel(cfg_path, root)
    # the two AppTests run meanwhile (nominal: in-process API; down: a dead port)
    pool = ThreadPoolExecutor(max_workers=2)
    nominal = pool.submit(
        run_apptest,
        root,
        script,
        {"TT_API_INPROCESS": "1", "TT_API_CONFIG": str(cfg_path), "TT_API_URL": None},
    )
    down = pool.submit(
        run_apptest,
        root,
        script,
        {
            "TT_API_URL": f"http://127.0.0.1:{free_port()}",
            "TT_API_INPROCESS": None,
            "TT_API_CONFIG": None,
        },
    )
    pool.shutdown(wait=False)
    pre = prerequisites(cfg, settings)
    result["prerequisites"] = pre
    for r in pre:
        detail = r["detail"] + (f" (fix: {r['hint']})" if r["hint"] else "")
        rep.add("prereq", r["name"], PREREQ_RESULT[r["status"]], detail)
    child, base = start_api(group, root, cfg_path, out / "api.log")
    try:
        up = wait_http(base + "/health", timeout_s=cfg["budgets"]["api_start_s"], proc=child.proc)
    except DemoError as e:
        rep.check("boot", "API up on a free port", False, str(e))
        nominal.result(), down.result()
        return
    rep.check("boot", "API up on a free port", True, f"{up:.1f} s")
    calibrated = httpx.get(base + "/health", timeout=10).json().get("calibrated_on")
    walk = check(cfg, base, browser=use_browser, out_dir=out, root=root, quiet=quiet)
    rep.rows.extend(walk["rows"])
    result["steps"] = [
        {"id": s["id"], "front": s["front"], "url": s["url"].replace(base, "<api>")}
        for s in walk["steps"]
    ]
    scan(rep, root, script)
    result["degraded"] = run_degraded(
        rep, cfg, root, cfg_path, work, out, group, use_browser, budget
    )
    if "api_down" in cfg["degraded"]:
        result["degraded"]["api_down"] = run_api_down(rep, cfg, root, use_browser, down.result())
    check_nominal(rep, nominal.result(), calibrated)


def _rel(p: Path, root: Path) -> str:
    try:
        return Path(p).resolve().relative_to(root).as_posix()
    except ValueError:
        return Path(p).name


def _done(
    result: dict,
    rep: Report,
    t0: float,
    out: Path,
    root: Path,
    record: bool,
    record_path: Path | None,
) -> dict:
    info = git_info(root)
    result.update(
        status=rep.status,
        passed=rep.passed,
        summary=rep.summary(),
        duration_s=round(time.monotonic() - t0, 1),
        git=info,
        platform=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        rows=rep.rows,
        out_dir=str(out),
    )
    header = [
        f"- mode: {result['mode']}" + (" (fresh clone)" if result["fresh_clone"] else ""),
        f"- code: {info['code_version']}",
        f"- started: {result['started_at']}",
        f"- duration: {result['duration_s']} s",
        f"- browser: {result['browser']}",
    ]
    write_report(out, result, rep.markdown("Demo rehearsal", header))
    if record:
        path = record_path or (root / REHEARSAL_JSON)
        record_rehearsal(result, path, root)
        result["recorded"] = str(path)
    return result


# -- evidence: docs/demo/rehearsal.json --------------------------------------------------
def human_rehearsals(path: Path) -> dict:
    """{status DONE | PENDING, entries, valid}: DONE once a human rehearsal is logged with
    `manual_code_edits: 0` (D116)."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        raw = {}
    entries = raw.get("entries") or [] if isinstance(raw, dict) else []
    valid = [
        e
        for e in entries
        if isinstance(e, dict)
        and e.get("date")
        and e.get("commit")
        and e.get("manual_code_edits") == 0
    ]
    return {
        "status": "DONE" if valid else "PENDING",
        "entries": len(entries),
        "valid": len(valid),
        "log": REHEARSAL_LOG.as_posix(),
    }


def _scrub(obj, root: Path):
    """Drop local absolute paths (repo root, temp folder) from recorded text."""
    if isinstance(obj, dict):
        return {k: _scrub(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v, root) for v in obj]
    if isinstance(obj, str):
        for p, rep in (
            (str(root), "<repo>"),
            (root.as_posix(), "<repo>"),
            (tempfile.gettempdir(), "<tmp>"),
            (Path(tempfile.gettempdir()).as_posix(), "<tmp>"),
        ):
            obj = obj.replace(p, rep)
        return obj
    return obj


# SKIP rows that do not weaken A-08: the Operator page is covered by the Streamlit AppTest
ALLOWED_SKIPS = frozenset({"Operator (Streamlit) page"})


def _agg(rows: list[dict]) -> str:
    """One status for a group of rows: MISSING (none), FAIL, PENDING, SKIP or PASS."""
    if not rows:
        return "MISSING"
    got = {r["result"] for r in rows}
    return next((s for s in ("FAIL", "PENDING", "SKIP") if s in got), "PASS")


def coverage(result: dict) -> dict:
    """What a recorded run really checked: the browser, the DOM of every mock step, the
    disallowed SKIP rows and, for a readiness run, the clone checks (D115, D116)."""
    from traffic_twin.demo.readiness import CLONE_CHECKS

    rows = result.get("rows") or []
    mock = [s["id"] for s in result.get("steps") or [] if s.get("front", "mock") == "mock"]
    cov: dict = {
        "browser": result.get("browser"),
        "dom": {sid: _agg([r for r in rows if r["group"] == f"dom {sid}"]) for sid in mock},
        "skipped": [
            f"{r['group']}: {r['check']}"
            for r in rows
            if r["result"] == "SKIP" and r["check"] not in ALLOWED_SKIPS
        ],
    }
    if result.get("kind") == "readiness":
        cov["clone"] = {
            c: _agg([r for r in rows if r["group"] == "clone" and r["check"] == c])
            for c in CLONE_CHECKS
        }
    return cov


def _gaps(block: dict, *, readiness: bool) -> list[str]:
    """Why a PASS block is still no A-08 (or, with `readiness`, A-09) evidence."""
    from traffic_twin.demo.readiness import CLONE_CHECKS

    cov = block.get("coverage")
    if not isinstance(cov, dict):
        return ["no check coverage recorded"]
    gaps = []
    if readiness and block.get("kind") != "readiness":
        gaps.append(f"kind {block.get('kind')!r}: not a `demo readiness` run")
    if not cov.get("browser"):
        gaps.append("no headless browser: no DOM check")
    dom = cov.get("dom") or {}
    if not dom:
        gaps.append("no mock step recorded")
    gaps += [f"dom {k} {v}" for k, v in dom.items() if v != "PASS"]
    gaps += [f"{s} SKIP" for s in cov.get("skipped") or []]
    if readiness:
        clone = cov.get("clone") or {}
        gaps += [
            f"clone: {c} {clone.get(c, 'MISSING')}" for c in CLONE_CHECKS if clone.get(c) != "PASS"
        ]
    return gaps


def _block(result: dict) -> dict:
    keep = ("rows",)
    s = {k: v for k, v in result.items() if k not in keep and k != "out_dir"}
    s["failed"] = [r for r in result["rows"] if r["result"] == "FAIL"]
    s["pending"] = [r for r in result["rows"] if r["result"] == "PENDING"]
    s["coverage"] = coverage(result)
    s["recorded_at"] = now_iso()
    return s


def evidence(doc: dict, root: Path) -> dict:
    """The A-08 / A-09 / rehearsed-demo statuses from the recorded blocks."""
    key = doc.get("last") if doc.get(doc.get("last") or "") else None
    if key is None:
        blocks = [k for k in ("rehearsal", "readiness") if doc.get(k)]
        key = max(blocks, key=lambda k: doc[k]["recorded_at"]) if blocks else None
    latest = doc.get(key) if key else None
    a08 = {
        "status": latest["status"] if latest else "PENDING",
        "from": key,
        "commit": (latest or {}).get("git", {}).get("code_version"),
        "recorded_at": (latest or {}).get("recorded_at"),
    }
    if latest and a08["status"] == "PASS" and (gaps := _gaps(latest, readiness=False)):
        a08.update(status="PENDING", reason="; ".join(gaps[:6]))
    rd = doc.get("readiness")
    if not rd:
        a09 = {
            "status": "PENDING",
            "reason": "no fresh-clone readiness rehearsal yet "
            "(python tasks.py demo-readiness on the release commit)",
        }
    elif rd["status"] == "FAIL":
        a09 = {"status": "FAIL", "reason": f"{len(rd.get('failed') or [])} failed checks"}
    elif rd["status"] == "PENDING":
        a09 = {"status": "PENDING", "reason": f"{len(rd.get('pending') or [])} pending checks"}
    elif gaps := _gaps(rd, readiness=True):
        a09 = {"status": "PENDING", "reason": "; ".join(gaps[:6])}
    elif rd.get("git", {}).get("dirty"):
        a09 = {"status": "PENDING", "reason": "the readiness clone was made from a dirty tree"}
    else:
        a09 = {"status": "PASS", "reason": "fresh clone + pack + rehearsal, no manual step"}
    if rd:
        a09.update(commit=rd.get("git", {}).get("code_version"), recorded_at=rd.get("recorded_at"))
    return {"a08": a08, "a09": a09, "rehearsed_demo": human_rehearsals(root / REHEARSAL_LOG)}


def record_rehearsal(result: dict, path: Path, root: Path) -> dict:
    """Update `path` (docs/demo/rehearsal.json): the rehearsal (or, for a `demo readiness`
    result, the readiness) block, then the A-08 / A-09 statuses. The block is chosen by
    `kind`, never by the `fresh_clone` label."""
    doc: dict = {}
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            doc = {}
    doc["schema"] = SCHEMA
    doc["note"] = (
        "A-08 / A-09 evidence written by `python -m traffic_twin.demo rehearsal "
        "--record` and `demo readiness` (D115, D116). Simulated data only."
    )
    key = "readiness" if result.get("kind") == "readiness" else "rehearsal"
    doc[key] = _scrub(_block(result), root)
    doc["last"] = key
    doc.update(evidence(doc, root))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    return doc


def main_status(result: dict) -> int:
    return 0 if result["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use: python -m traffic_twin.demo rehearsal")

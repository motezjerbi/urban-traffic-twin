"""Walkthrough of every configs/demo.yaml step (D116): the API calls each view makes and,
with a headless browser, the DOM of its deep link.

Per API call: HTTP 200, the spec 9.2 envelope (`data_source` "simulation", a
`calibrated_on` date, the message type), an allowed status and the ENF-02 latency budget
(500 ms). An `unavailable` "pas encore évalué" answer on a Part 8 output is PENDING until
the integration made it (`tasks.py part8`), never PASS.

Per DOM (plan §4.4): the view root's `data-view-state` is ok or degraded, the view's
`data-field` hooks and the footer hooks are present and filled, no "Chargement" is left,
and the hook values agree with the API answers (pedestrian verdict, ambulance junction
statuses, student ETA, bus delay, replay time).

    python -m traffic_twin.demo check [--url http://127.0.0.1:8000] [--operator-url ...]
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from traffic_twin.demo.common import Report, probe_api, probe_streamlit
from traffic_twin.demo.config import (
    FOOTER_HOOKS,
    HOOKS,
    Call,
    registered_labels,
    resolve_runs,
    step_calls,
    step_url,
)

ENVELOPE = (
    "data_source",
    "calibrated_on",
    "timestamp",
    "data_age_s",
    "stale",
    "type",
    "status",
    "notice",
    "provisional",
    "run",
    "sim_t_s",
    "error",
)
PART8_NOTICE = "pas encore évalué"  # D103: a label registered but not in the outputs
STATUS_FR = {
    "clear": "dégagé",
    "green_wave": "onde verte",
    "blocked": "bloqué",
    "unknown": "non modélisé",
}
VIEW_OK = frozenset({"ok", "degraded"})
LOADING = "chargement"


def fold(s: str) -> str:
    """Lower case without accents (for French label comparisons)."""
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).casefold()


def envelope_problems(b: dict, want_type: str | None) -> list[str]:
    bad = [f"no {k}" for k in ENVELOPE if k not in b]
    if b.get("data_source") != "simulation":
        bad.append(f"data_source={b.get('data_source')!r}")
    if not b.get("calibrated_on"):
        bad.append("calibrated_on empty")
    if want_type and b.get("type") != want_type:
        bad.append(f"type={b.get('type')!r} (expected {want_type!r})")
    if b.get("status") not in ("ok", "degraded", "unavailable"):
        bad.append(f"status={b.get('status')!r}")
    return bad


def is_part8_pending(b: dict) -> bool:
    return b.get("status") == "unavailable" and fold(PART8_NOTICE) in fold(b.get("notice") or "")


def call_api(rep: Report, client: httpx.Client, call: Call, budget_ms: float) -> dict | None:
    """One API call checked into `rep`; the parsed body (None when not JSON)."""
    t0 = time.perf_counter()
    try:
        r = client.get(call.path)
    except httpx.HTTPError as e:
        rep.add(call.group, call.path, "FAIL", f"{type(e).__name__}: {e}")
        return None
    ms = (time.perf_counter() - t0) * 1000
    try:
        b = r.json()
    except ValueError:
        rep.add(call.group, call.path, "FAIL", f"http {r.status_code}, not JSON", ms)
        return None
    bad = envelope_problems(b, call.type)
    if r.status_code != call.http:
        bad.append(f"http {r.status_code}")
    if ms > budget_ms:
        bad.append(f"{ms:.0f} ms > {budget_ms:.0f} ms (ENF-02)")
    status = b.get("status")
    detail = f"{status} {ms:.0f} ms"
    if call.part8 and is_part8_pending(b) and not [x for x in bad if "ms >" in x]:
        rep.add(call.group, call.path, "PENDING", f"{detail}: {b.get('notice')}", ms)
        return b
    if status not in call.ok:
        bad.append(
            f"status {status!r} not in {sorted(call.ok)}"
            + (f" ({b.get('notice')})" if b.get("notice") else "")
        )
    for key, want in call.expect.items():
        if b.get(key) != want:
            bad.append(f"{key}={b.get(key)!r} (expected {want!r})")
    rep.add(
        call.group,
        call.path,
        "FAIL" if bad else "PASS",
        f"{detail}; {'; '.join(bad)}" if bad else detail,
        ms,
    )
    return b


# -- DOM checks --------------------------------------------------------------------------
def view_root(states: list[dict], view: str) -> dict | None:
    """The `data-view-state` root of `view`: `data-view` = view, else an id naming it,
    else the only (or first) root."""
    for s in states:
        if s["attrs"].get("data-view") == view:
            return s
    for s in states:
        if view in (s["attrs"].get("id") or ""):
            return s
    return states[0] if states else None


def _ints_near(x: float) -> set[int]:
    return {math.floor(x), round(x), math.ceil(x)}


def _has_int(text: str, candidates: set[int]) -> bool:
    found = {int(m) for m in re.findall(r"-?\d+", text.replace("−", "-"))}
    found |= {abs(v) for v in found}
    return bool(found & candidates)


def value_problems(view: str, hooks: dict[str, str], bodies: list[dict | None]) -> list[str]:
    """Hook values that disagree with the API answers of the step."""
    b = next((x for x in bodies if x), None) or {}
    bad: list[str] = []
    if view == "pedestrian" and b.get("can_cross") is not None:
        verdict = fold(hooks.get("ped-verdict", ""))
        if b["can_cross"] and ("traverser" not in verdict or verdict.startswith("attendez")):
            bad.append(f"ped-verdict {hooks.get('ped-verdict')!r} but can_cross true")
        if not b["can_cross"] and "attendez" not in verdict:
            bad.append(f"ped-verdict {hooks.get('ped-verdict')!r} but can_cross false")
    if view == "ambulance":
        for i, j in enumerate((b.get("intersections") or [])[:3], start=1):
            want = STATUS_FR.get(j.get("status"), "")
            if not j.get("modelled", True):
                want = STATUS_FR["unknown"]
            got = hooks.get(f"amb-j{i}-status", "")
            if want and fold(want) not in fold(got):
                bad.append(f"amb-j{i}-status {got!r} but API {j.get('status')!r} ({want})")
    if view == "student" and b.get("eta_median_s") is not None:
        got = hooks.get("stu-eta", "")
        total = b["eta_median_s"] + (b.get("rest_of_trip_s") or 0.0)  # the student's whole trip
        if not _has_int(got, _ints_near(total / 60.0)):
            bad.append(f"stu-eta {got!r} but eta_median_s + rest_of_trip_s = {total:.0f} s")
    if view == "bus" and b.get("delay_s") is not None:
        got = hooks.get("bus-delay", "")
        cands = _ints_near(abs(b["delay_s"])) | _ints_near(abs(b["delay_s"]) / 60.0)
        if not _has_int(got, cands):
            bad.append(f"bus-delay {got!r} but delay_s {b['delay_s']}")
    if view == "replay" and b.get("timestamp"):
        hhmm = str(b["timestamp"])[11:16]
        if hhmm and hhmm not in hooks.get("replay-t", ""):
            bad.append(f"replay-t {hooks.get('replay-t')!r} but the API instant is {hhmm}")
    return bad


def dom_check(
    rep: Report,
    step: dict,
    dom: str,
    bodies: list[dict | None],
    *,
    allowed: frozenset[str] = VIEW_OK,
    pending_states: frozenset[str] = frozenset(),
    values: bool = True,
    group: str | None = None,
) -> None:
    """The DOM rows of one mock step (view state, hooks, footer, loading, values)."""
    from traffic_twin.app.browser import fields, view_states

    view = step["view"]
    g = group or f"dom {step['id']}"
    states = view_states(dom)
    root = view_root(states, view)
    if root is None:
        rep.add(
            g, "data-view-state root", "FAIL", f"no data-view-state in the DOM ({len(dom)} chars)"
        )
        return
    st = root["state"]
    if st in allowed:
        rep.add(g, "data-view-state", "PASS", st)
    elif st in pending_states:
        rep.add(g, "data-view-state", "PENDING", f"{st} (Part 8 output not made yet)")
    else:
        rep.add(
            g, "data-view-state", "FAIL", f"{st!r} not in {sorted(allowed)}: {root['text'][:160]}"
        )
    if LOADING in fold(root["text"]):
        rep.add(g, "no loading indicator left", "FAIL", root["text"][:160])
    else:
        rep.add(g, "no loading indicator left", "PASS")
    hooks = fields(dom)
    want = HOOKS.get(view, ())
    if view == "home":
        cards = [k for k in hooks if k.startswith("home-card-")]
        rep.check(g, "profile cards (home-card-*)", len(cards) >= 5, ", ".join(sorted(cards)))
    else:
        missing = [h for h in want if not hooks.get(h)]
        rep.check(
            g,
            "hooks " + ", ".join(want),
            not missing,
            f"missing or empty: {missing}" if missing else "",
        )
    foot = [h for h in FOOTER_HOOKS if not hooks.get(h)]
    src = fold(hooks.get("footer-source", ""))
    rep.check(
        g,
        "footer (source, calibrated_on)",
        not foot and "simul" in src,
        f"missing: {foot}" if foot else hooks.get("footer-source", "")[:80],
    )
    if values and st in allowed:
        bad = value_problems(view, hooks, bodies)
        rep.check(g, "hook values equal the API answers", not bad, "; ".join(bad))


def _dump(url: str, wait_ms: int) -> tuple[str | None, str]:
    from traffic_twin.app.browser import BrowserError, dump_dom

    try:
        return dump_dom(url, wait_ms=wait_ms), ""
    except BrowserError as e:
        return None, str(e)


def dump_many(urls: list[str], wait_ms: int, workers: int = 4) -> list[tuple[str | None, str]]:
    """DOMs of `urls` (parallel headless browsers, one profile each)."""
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(urls))) as ex:
        return list(ex.map(lambda u: _dump(u, wait_ms), urls))


# -- the walkthrough ---------------------------------------------------------------------
def check(
    cfg: dict,
    base_url: str,
    *,
    browser: bool = True,
    operator_url: str | None = None,
    out_dir: Path | None = None,
    root: Path | None = None,
    quiet: bool = False,
) -> dict:
    """The step report {status, passed, summary, rows, steps, runs, ...} of the API at
    `base_url` (and the Operator at `operator_url` when given)."""
    from traffic_twin.api.settings import find_repo_root
    from traffic_twin.app.browser import find_browser

    root = root or find_repo_root()
    rep = Report(quiet=quiet)
    t_all = time.monotonic()
    budget = float(cfg["budgets"]["api_ms"])
    base_url = base_url.rstrip("/")
    kind, detail = probe_api(base_url, cfg["api"]["title"])
    rep.check("boot", f"real API at {base_url} (impostor check)", kind == "api", detail)
    result = {
        "kind": "check",
        "base_url": base_url,
        "operator_url": operator_url,
        "rows": rep.rows,
        "steps": [],
        "runs": {},
        "browser": None,
    }
    if kind != "api":
        return _finish(result, rep, t_all, out_dir)
    exe = find_browser() if browser else None
    result["browser"] = exe.name if exe else None
    with httpx.Client(base_url=base_url, timeout=10.0) as c:
        sc = call_api(rep, c, Call("boot", "/scenarios", "scenario_list"), budget) or {}
        runs = resolve_runs(cfg, sc)
        result["runs"] = runs
        for name, r in runs.items():
            want = cfg["runs"][name]
            label = f"run {name}: {want['scenario']} seed {want['seed']}"
            if r is not None:
                rep.add("boot", label, "PASS", f"run_id {r['run_id']}")
            elif want["scenario"] in set(cfg.get("part8_scenarios") or []):
                rep.add("boot", label, "PENDING", "not made yet (tasks.py demo-runs8, I3)")
            else:
                rep.add("boot", label, "FAIL", "not found by /scenarios")
        ev = c.get("/evaluation", params={"family": "main"})
        labels = []
        try:
            labels = [x.get("label") for x in ev.json().get("labels") or [] if x.get("label")]
        except (ValueError, AttributeError):
            pass
        labels = labels or registered_labels(root)
        dom_jobs: list[tuple[dict, str, list[dict | None]]] = []
        for step in cfg["steps"]:
            calls = step_calls(cfg, step, runs, labels)
            url = step_url(cfg, step, base_url, operator_url)
            entry = {
                "id": step["id"],
                "view": step.get("view"),
                "front": step.get("front", "mock"),
                "url": url,
                "calls": [],
            }
            result["steps"].append(entry)
            if isinstance(calls, str):
                pend = step.get("run") and cfg["runs"][step["run"]]["scenario"] in set(
                    cfg.get("part8_scenarios") or []
                )
                rep.add(step["id"], "run resolved", "PENDING" if pend else "FAIL", calls)
                continue
            bodies = []
            for call in calls:
                entry["calls"].append(call.path)
                bodies.append(call_api(rep, c, call, budget))
            if entry["front"] == "streamlit":
                if operator_url:
                    k, d = probe_streamlit(operator_url)
                    rep.check(
                        step["id"], f"Operator (Streamlit) at {operator_url}", k == "streamlit", d
                    )
                else:
                    rep.add(
                        step["id"],
                        "Operator (Streamlit) page",
                        "SKIP",
                        "not started here (the rehearsal runs Streamlit AppTest)",
                    )
                continue
            if exe is not None:
                dom_jobs.append((step, url, bodies))
            else:
                rep.add(
                    f"dom {step['id']}",
                    "deep link DOM",
                    "SKIP",
                    "no browser" if browser else "--no-browser",
                )
        # 12.1 error state: an unknown id is a 404 envelope, never a 500
        for path, code in (
            ("/crossing/XX", "unknown_crossing"),
            ("/replay/no-such-run?t=08:15", "unknown_run"),
        ):
            r = c.get(path)
            try:
                b = r.json()
            except ValueError:
                b = {}
            err = (b.get("error") or {}).get("code")
            rep.check(
                "errors",
                f"{path} -> 404 {code}",
                r.status_code == 404 and err == code and b.get("data_source") == "simulation",
                f"http {r.status_code} {err}",
            )
        m = c.get("/mock")
        rep.check(
            "front",
            "/mock served",
            m.status_code == 200 and "<html" in m.text.lower(),
            f"http {m.status_code}",
        )
        j = c.get("/three.min.js")
        rep.check("front", "/three.min.js served", j.status_code == 200, f"{len(j.content)} B")
    if dom_jobs:
        doms = dump_many([u for _s, u, _b in dom_jobs], int(cfg["budgets"]["browser_wait_ms"]))
        for (step, _url, bodies), (dom, err) in zip(dom_jobs, doms, strict=True):
            if dom is None:
                rep.add(f"dom {step['id']}", "deep link DOM", "FAIL", err)
                continue
            if out_dir:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / f"dom_{step['id']}.html").write_text(dom, encoding="utf-8")
            pend = frozenset({"unavailable"}) if step["view"] == "operator" else frozenset()
            if step.get("run") and runs.get(step["run"]) is None:
                pend = frozenset({"unavailable", "error"})
            dom_check(rep, step, dom, bodies, pending_states=pend)
    return _finish(result, rep, t_all, out_dir)


def _finish(result: dict, rep: Report, t0: float, out_dir: Path | None) -> dict:
    result.update(
        status=rep.status,
        passed=rep.passed,
        summary=rep.summary(),
        duration_s=round(time.monotonic() - t0, 1),
        rows=rep.rows,
    )
    return result

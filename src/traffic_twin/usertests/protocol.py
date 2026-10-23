"""The user-test protocol (`configs/usertests.yaml`) and the results file
(`docs/usertests/results.yaml`), loaded and checked (D111).

Like the field file (`validation.field`), every schema problem is collected and ONE
ConfigError lists them all. The results check is also the ENF-07 privacy guard:

- participants are `P<n>` and issues `U<nn>`; any key outside the schema is refused, so
  there is no place for a name, an age, an employer or a health condition (reduced
  mobility is never recorded: only the role played and `profile_match: real | proxy`);
- any text that looks like an e-mail address or a phone number is refused;
- a session dated before the pre-registration is refused.

It also enforces the scoring rules (a success within the time and click limits, without
the hint), and the fix loop: a fixed issue names its commit; `verify_git` then checks
that the commit exists and that a verifying session runs an app version descending from
it, or that a verifying test node exists.

Results live under docs/ so filling them in never makes `code_version` dirty.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import yaml

from traffic_twin.config import ConfigError
from traffic_twin.paths import USERTESTS_CONFIG, USERTESTS_RESULTS

SCHEMA_VERSION = 1
SPEC_VIEWS = ("pedestrian", "student", "bus", "ambulance", "operator")  # spec 12: 5 users
LINK_VIEWS = ("home", "pedestrian", "student", "bus", "ambulance", "replay", "operator")  # D106
DEMO_SEEDS = (4000, 4009)  # D53 / D101: app, user-test and demo work use the demo seeds only

TOP_KEYS = {
    "schema_version", "preregistered", "participants", "profiles", "run", "app_url",
    "session_minutes", "targets", "order", "operator", "tasks", "after_tasks", "questionnaire",
    "error_codes", "outcomes", "issue_kinds", "changes",
}  # fmt: skip
TASK_KEYS = {
    "id", "view", "optional", "start", "check", "card_fr", "card_en", "expect", "answer_fr",
    "glance", "time_limit_s",
}  # fmt: skip
EXPECT_KEYS = {"endpoint", "query", "field", "equals", "not_null", "tolerance", "unit", "computed"}
START_KEYS = {"view", "t", "kiosk"}
TARGET_KEYS = {"pedestrian_time_s", "max_clicks", "ambulance_glance_s"}
OUTCOMES = ("success", "partial", "fail", "skipped")

RESULT_KEYS = {"schema_version", "synthetic", "sessions", "issues"}
SESSION_KEYS = {
    "id", "date", "round", "profile", "profile_match", "app_version", "device", "consent",
    "simulated_understood", "sus", "comments", "tasks",
}  # fmt: skip
RECORD_KEYS = {"task", "order", "time_s", "clicks", "answer", "outcome", "errors", "seq",
               "hint", "notes"}  # fmt: skip
ISSUE_KEYS = {
    "id", "found_by", "tasks", "sessions", "severity", "kind", "description", "fix", "status",
    "commit", "verified",
}  # fmt: skip
PROFILE_MATCH = ("real", "proxy")
DEVICES = ("laptop", "phone", "tablet")
CONSENTS = ("oral", "written")
FOUND_BY = ("test", "walkthrough")
STATUSES = ("open", "fixed", "wontfix")

SESSION_ID = re.compile(r"P\d+")
ISSUE_ID = re.compile(r"U\d{2,}")
SHA = re.compile(r"[0-9a-f]{7,40}")
# A pytest node id. The parametrize id is limited to a safe charset (no "@", "+" or spaces)
# and also scanned by looks_personal(): `verified` skips the free-text scan (ENF-07)
TEST_NODE = re.compile(r"(tests/[\w/.-]+\.py)::(\w+)(\[[\w.-]*\])?")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_PATTERNS = (
    re.compile(r"(?<!\d)0[1-9](?:[\s.-]?\d{2}){4}(?!\d)"),  # French national: 06 12 34 56 78
    re.compile(r"(?:\+|\b00)\d{2,3}[\s.-]?\d(?:[\s.-]?\d{2}){3,5}(?!\d)"),  # +33 6 12 ...
)
# Values validated by their own pattern (a commit sha could look like a phone number)
NOT_FREE_TEXT = {"app_version", "commit", "verified", "id", "date"}


# -- helpers ---------------------------------------------------------------------------
def _read(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _date(value: Any) -> dt.date | None:
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        return None


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _raise(path: Path, errors: list[str]) -> None:
    if errors:
        listing = "\n  - ".join(errors)
        raise ConfigError(f"{path}: {len(errors)} problem(s):\n  - {listing}")


def _unknown(where: str, node: dict, allowed: set[str]) -> list[str]:
    return [f"{where}: unknown key '{k}'" for k in sorted(set(node) - allowed, key=str)]


# -- protocol --------------------------------------------------------------------------
def check_protocol(raw: Any) -> list[str]:
    """Every problem of a parsed `usertests.yaml` (empty list: valid)."""
    if not isinstance(raw, dict):
        return ["expected a mapping at the top level"]
    errs = _unknown("protocol", raw, TOP_KEYS)
    missing = sorted(TOP_KEYS - {"session_minutes"} - set(raw))
    errs += [f"protocol: missing key '{k}'" for k in missing]
    if raw.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version must be {SCHEMA_VERSION}")
    prereg = _date(raw.get("preregistered"))
    if prereg is None:
        errs.append("preregistered must be a date (YYYY-MM-DD)")
    part = raw.get("participants") or {}
    lo, hi = part.get("min"), part.get("max")
    if not (_int(lo) and _int(hi) and 1 <= lo <= hi):
        errs.append("participants needs integers 1 <= min <= max")
    profiles = raw.get("profiles") or []
    bad = [p for p in profiles if p not in SPEC_VIEWS]
    if bad or len(set(profiles)) != len(profiles) or not profiles:
        errs.append(f"profiles must be distinct names among {SPEC_VIEWS}, got {profiles}")
    run = raw.get("run") or {}
    if not isinstance(run.get("scenario"), str) or not run.get("scenario"):
        errs.append("run.scenario must be a scenario name (pitie_rules)")
    seed = run.get("seed")
    if not (_int(seed) and DEMO_SEEDS[0] <= seed <= DEMO_SEEDS[1]):
        errs.append(f"run.seed must be a demo seed {DEMO_SEEDS[0]}-{DEMO_SEEDS[1]} (D101)")
    if not str(raw.get("app_url", "")).startswith(("http://127.0.0.1", "http://localhost")):
        errs.append("app_url must be a local http://127.0.0.1 URL (the app runs offline)")
    targets = raw.get("targets") or {}
    errs += _unknown("targets", targets, TARGET_KEYS)
    for k in sorted(TARGET_KEYS):
        if not (_number(targets.get(k)) and targets[k] > 0):
            errs.append(f"targets.{k} must be a number > 0")
    outcomes = raw.get("outcomes") or []
    if tuple(outcomes) != OUTCOMES:
        errs.append(f"outcomes must be {list(OUTCOMES)} (the scoring rules depend on them)")
    for key in ("error_codes", "issue_kinds"):
        v = raw.get(key)
        if not (isinstance(v, list) and v and all(isinstance(x, str) for x in v)):
            errs.append(f"{key} must be a non-empty list of names")
    errs += _check_tasks(raw)
    errs += _check_operator(raw)
    order = raw.get("order") or {}
    ids = [t.get("id") for t in raw.get("tasks") or [] if isinstance(t, dict)]
    if order.get("first") not in ids:
        errs.append(f"order.first must be a task id, got {order.get('first')!r}")
    if order.get("rest") != "rotate":
        errs.append("order.rest must be 'rotate'")
    after = raw.get("after_tasks") or {}
    if not after.get("transparency_fr") or after.get("expected") != "simulated":
        errs.append("after_tasks needs transparency_fr and expected: simulated (ENF-05)")
    changes = raw.get("changes")
    if not isinstance(changes, list):
        errs.append("changes must be a list (empty until a dated change)")
    else:
        for i, c in enumerate(changes):
            where = f"changes[{i}]"
            if not isinstance(c, dict) or set(c) != {"date", "what", "why"}:
                errs.append(f"{where}: needs exactly date, what and why")
                continue
            d = _date(c["date"])
            if d is None or (prereg is not None and d < prereg):
                errs.append(f"{where}: date must be on or after preregistered")
    return errs


def _check_tasks(raw: dict) -> list[str]:
    errs: list[str] = []
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return ["tasks must be a non-empty list"]
    seen: set[str] = set()
    covered: set[str] = set()
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            errs.append(f"tasks[{i}]: expected a mapping")
            continue
        tid = t.get("id")
        where = f"task {tid or i}"
        errs += _unknown(where, t, TASK_KEYS)
        if not (isinstance(tid, str) and re.fullmatch(r"T-[A-Z]+", tid)):
            errs.append(f"{where}: id must look like T-XXX")
        elif tid in seen:
            errs.append(f"{where}: duplicate id")
        seen.add(str(tid))
        view = t.get("view")
        if view not in LINK_VIEWS or view == "home":
            errs.append(f"{where}: view {view!r} not in {LINK_VIEWS[1:]}")
        elif not t.get("optional"):
            covered.add(view)
        start = t.get("start") or {}
        errs += _unknown(f"{where}.start", start, START_KEYS)
        if start.get("view") not in LINK_VIEWS:
            errs.append(f"{where}: start.view must be one of {LINK_VIEWS}")
        if "t" in start and not (_int(start["t"]) and 0 <= start["t"] <= 7200):
            errs.append(f"{where}: start.t must be seconds since 07:30 (0-7200)")
        check = t.get("check") or {}
        if check.get("view") != view or not isinstance(check.get("params", {}), dict):
            errs.append(f"{where}: check needs view: {view} and a params mapping")
        for key in ("card_fr", "card_en", "answer_fr"):
            if not (isinstance(t.get(key), str) and t[key].strip()):
                errs.append(f"{where}: {key} must be a non-empty text")
        if not (_number(t.get("time_limit_s")) and t["time_limit_s"] > 0):
            errs.append(f"{where}: time_limit_s must be a number > 0")
        expect = t.get("expect")
        if not isinstance(expect, list) or not expect:
            errs.append(f"{where}: expect must be a non-empty list")
            continue
        for k, e in enumerate(expect):
            ew = f"{where}.expect[{k}]"
            if not isinstance(e, dict):
                errs.append(f"{ew}: expected a mapping")
                continue
            errs += _unknown(ew, e, EXPECT_KEYS)
            if not str(e.get("endpoint", "")).startswith("/") or not e.get("field"):
                errs.append(f"{ew}: needs an endpoint (/...) and a field")
            if e.get("computed") not in (None, "operator"):
                errs.append(f"{ew}: computed must be 'operator'")
            conds = [k for k in ("equals", "not_null", "computed") if k in e]
            if not conds:
                errs.append(f"{ew}: needs a condition (equals, not_null or computed)")
            if "tolerance" in e and not (_number(e["tolerance"]) and e["tolerance"] >= 0):
                errs.append(f"{ew}: tolerance must be a number >= 0")
    missing = [v for v in SPEC_VIEWS if v not in covered]
    if missing:
        errs.append(f"tasks: no required task for the view(s) {missing} (spec 12: 5 users)")
    return errs


def _check_operator(raw: dict) -> list[str]:
    op = raw.get("operator") or {}
    errs = _unknown("operator", op, {"family", "user", "kpi", "labels", "required"})
    labels = op.get("labels") or []
    if not labels or not all(isinstance(x, str) for x in labels):
        errs.append("operator.labels must list evaluation labels (S0 ... S3)")
    if not set(op.get("required") or []) <= set(labels):
        errs.append("operator.required must be among operator.labels")
    for k in ("family", "user", "kpi"):
        if not isinstance(op.get(k), str):
            errs.append(f"operator.{k} must be a name")
    return errs


def load_protocol(path: Path = USERTESTS_CONFIG) -> dict:
    """The checked user-test protocol; ONE ConfigError lists every problem. The
    `preregistered` and change dates are returned as ISO strings."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"{path}: user-test protocol not found")
    raw = _read(path)
    _raise(path, check_protocol(raw))
    raw["preregistered"] = _date(raw["preregistered"]).isoformat()
    for c in raw["changes"]:
        c["date"] = _date(c["date"]).isoformat()
    return raw


def tasks_by_id(protocol: dict) -> dict[str, dict]:
    return {t["id"]: t for t in protocol["tasks"]}


def required_tasks(protocol: dict) -> list[str]:
    return [t["id"] for t in protocol["tasks"] if not t.get("optional")]


def task_order(protocol: dict, participant: int) -> list[str]:
    """Task order of participant `P<participant>` (1-based): `order.first` for everyone,
    the other required tasks rotated by the participant index, optional tasks last."""
    if participant < 1:
        raise ValueError("participants are numbered from 1")
    first = protocol["order"]["first"]
    rest = [t for t in required_tasks(protocol) if t != first]
    k = (participant - 1) % len(rest) if rest else 0
    optional = [t["id"] for t in protocol["tasks"] if t.get("optional")]
    return [first, *rest[k:], *rest[:k], *optional]


def clock(t_s: float | None) -> str:
    """Seconds since 07:30 as the simulated clock time HH:MM:SS."""
    if t_s is None:
        return "–"
    s = int(round(7.5 * 3600 + t_s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def deep_link(protocol: dict, task: dict, kind: str = "start", base: str | None = None) -> str:
    """The `/mock` deep link (D106) of a task: `start` (what the facilitator opens: the
    run and the instant, the view only for the ambulance screen) or `check` (the view
    that shows the answer, for the automated tests and the answer key)."""
    if kind not in ("start", "check"):
        raise ValueError("kind must be 'start' or 'check'")
    start = task.get("start") or {}
    spec = start if kind == "start" else task["check"]
    query: dict[str, Any] = {"view": spec["view"], "scenario": protocol["run"]["scenario"],
                             "seed": protocol["run"]["seed"]}  # fmt: skip
    extra = dict(spec.get("params") or {}) if kind == "check" else {}
    params_t = extra.pop("t", None)
    t = start.get("t", params_t)
    if t is not None:
        query["t"] = t
    if "kiosk" in start:
        query["kiosk"] = start["kiosk"]
    query.update(extra)
    return f"{base or protocol['app_url']}?{urlencode(query)}"


# -- results ---------------------------------------------------------------------------
def _texts(node: Any, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            if k not in NOT_FREE_TEXT:
                yield from _texts(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _texts(v, f"{path}[{i}]")


def looks_personal(text: str) -> bool:
    """True when `text` looks like contact details (an e-mail address or a phone number)."""
    return bool(EMAIL.search(text)) or any(p.search(text) for p in PHONE_PATTERNS)


def check_results(raw: Any, protocol: dict) -> list[str]:
    """Every problem of a parsed results file (schema, privacy guard, scoring rules,
    fix loop without git). An empty file (`sessions: []`, `issues: []`) is valid."""
    if not isinstance(raw, dict):
        return ["expected a mapping at the top level"]
    errs = _unknown("results", raw, RESULT_KEYS)
    if raw.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version must be {SCHEMA_VERSION}")
    if "synthetic" in raw and raw["synthetic"] is not True:
        errs.append("synthetic: only `true` is allowed (test fixtures)")
    sessions, issues = raw.get("sessions"), raw.get("issues")
    if not isinstance(sessions, list):
        errs.append("sessions must be a list ([] until the first session)")
        sessions = []
    if not isinstance(issues, list):
        errs.append("issues must be a list ([] until the first issue)")
        issues = []
    ids: list[str] = []
    for i, s in enumerate(sessions):
        if not isinstance(s, dict):
            errs.append(f"sessions[{i}]: expected a mapping")
            continue
        errs += _check_session(s, protocol, ids)
        ids.append(str(s.get("id")))
    for i, issue in enumerate(issues):
        if not isinstance(issue, dict):
            errs.append(f"issues[{i}]: expected a mapping")
            continue
        errs += _check_issue(issue, protocol, ids, issues[:i])
    for where, text in _texts(raw, "results"):
        if looks_personal(text):
            errs.append(f"{where}: looks like contact details (ENF-07): remove them")
    return errs


def _check_session(s: dict, protocol: dict, earlier: list[str]) -> list[str]:
    sid = s.get("id")
    where = f"session {sid}"
    errs = _unknown(where, s, SESSION_KEYS)
    errs += [f"{where}: missing key '{k}'" for k in sorted({"id", "date", "tasks"} - set(s))]
    if not (isinstance(sid, str) and SESSION_ID.fullmatch(sid)):
        errs.append(f"session id {sid!r}: use P1, P2, ... (anonymous ids only, ENF-07)")
    elif sid in earlier:
        errs.append(f"{where}: duplicate session id")
    d = _date(s.get("date"))
    if d is None:
        errs.append(f"{where}: date must be YYYY-MM-DD")
    elif d < _date(protocol["preregistered"]):
        errs.append(f"{where}: dated before the protocol was registered")
    if s.get("round") not in (1, 2):
        errs.append(f"{where}: round must be 1 (before the fixes) or 2 (on the fixed app)")
    if s.get("profile") not in protocol["profiles"]:
        errs.append(f"{where}: profile {s.get('profile')!r} not in {protocol['profiles']}")
    choices = (("profile_match", PROFILE_MATCH), ("device", DEVICES), ("consent", CONSENTS))
    for key, allowed in choices:
        if s.get(key) not in allowed:
            errs.append(f"{where}: {key} must be one of {allowed}")
    if not (isinstance(s.get("app_version"), str) and SHA.fullmatch(s["app_version"])):
        errs.append(f"{where}: app_version must be the clean git sha printed by usertest-serve")
    if s.get("simulated_understood") not in (None, True, False):
        errs.append(f"{where}: simulated_understood must be true, false or empty")
    sus = s.get("sus")
    if sus is not None and not (
        isinstance(sus, list) and len(sus) == 10 and all(_int(v) and 1 <= v <= 5 for v in sus)
    ):
        errs.append(f"{where}: sus must be empty or 10 answers 1-5")
    comments = s.get("comments", [])
    if not (isinstance(comments, list) and all(isinstance(c, str) for c in comments)):
        errs.append(f"{where}: comments must be a list of texts")
    records = s.get("tasks")
    if not isinstance(records, list):
        return [*errs, f"{where}: tasks must be a list"]
    known = tasks_by_id(protocol)
    done: set[str] = set()
    orders: list[int] = []
    for k, r in enumerate(records):
        if not isinstance(r, dict):
            errs.append(f"{where}.tasks[{k}]: expected a mapping")
            continue
        tid = r.get("task")
        rw = f"{where}/{tid}"
        errs += _unknown(rw, r, RECORD_KEYS)
        if tid not in known:
            errs.append(f"{where}: unknown task {tid!r}")
            continue
        if tid in done:
            errs.append(f"{rw}: recorded twice")
        done.add(tid)
        errs += _check_record(rw, r, known[tid], protocol)
        if _int(r.get("order")):
            orders.append(r["order"])
        first = protocol["order"]["first"]
        if tid == first and r.get("outcome") not in (None, "skipped") and r.get("order") != 1:
            errs.append(f"{rw}: {first} comes first for every participant (order 1)")
    if len(orders) != len(set(orders)):
        errs.append(f"{where}: two tasks share an order number")
    missing = [t for t in required_tasks(protocol) if t not in done]
    errs += [f"{where}: task {t} missing (record it with outcome: skipped)" for t in missing]
    return errs


def _check_record(rw: str, r: dict, task: dict, protocol: dict) -> list[str]:
    errs: list[str] = []
    outcome = r.get("outcome")
    if outcome not in (*OUTCOMES, None):
        errs.append(f"{rw}: outcome {outcome!r} not in {list(OUTCOMES)}")
    if not (_int(r.get("order")) and r["order"] >= 1):
        errs.append(f"{rw}: order must be an integer >= 1")
    for key in ("time_s", "clicks"):
        v = r.get(key)
        ok = v is None or (_number(v) if key == "time_s" else _int(v))
        if not ok or (v is not None and v < 0):
            errs.append(f"{rw}: {key} must be empty or a number >= 0")
    if outcome in ("success", "partial", "fail") and (
        r.get("time_s") is None or r.get("clicks") is None
    ):
        errs.append(f"{rw}: a scored task needs time_s and clicks (timeout: the limit)")
    seq = r.get("seq")
    if seq is not None and not (_int(seq) and 1 <= seq <= 7):
        errs.append(f"{rw}: seq must be empty or 1-7")
    codes = r.get("errors", [])
    if not isinstance(codes, list):
        errs.append(f"{rw}: errors must be a list of codes")
        codes = []
    errs += [f"{rw}: error code {c!r} not in error_codes" for c in codes
             if c not in protocol["error_codes"]]  # fmt: skip
    for key in ("answer", "notes"):
        if key in r and r[key] is not None and not isinstance(r[key], str):
            errs.append(f"{rw}: {key} must be a text")
    if "hint" in r and r["hint"] not in (True, False, None):
        errs.append(f"{rw}: hint must be true or false")
    if outcome == "success":  # the scoring rules of docs/usertest_protocol.md
        t, c = r.get("time_s"), r.get("clicks")
        if _number(t) and t > task["time_limit_s"]:
            errs.append(f"{rw}: success over the {task['time_limit_s']} s limit: fail (timeout)")
        if _int(c) and c > protocol["targets"]["max_clicks"]:
            errs.append(f"{rw}: success over {protocol['targets']['max_clicks']} clicks: partial")
        if r.get("hint"):
            errs.append(f"{rw}: success after the scripted hint: partial")
    return errs


def _check_issue(issue: dict, protocol: dict, sessions: list[str], earlier: list) -> list[str]:
    iid = issue.get("id")
    where = f"issue {iid}"
    errs = _unknown(where, issue, ISSUE_KEYS)
    if not (isinstance(iid, str) and ISSUE_ID.fullmatch(iid)):
        errs.append(f"issue id {iid!r}: use U01, U02, ...")
    elif any(isinstance(e, dict) and e.get("id") == iid for e in earlier):
        errs.append(f"{where}: duplicate issue id")
    found_by = issue.get("found_by")
    if found_by not in FOUND_BY:
        errs.append(f"{where}: found_by must be one of {FOUND_BY}")
    tasks = issue.get("tasks") or []
    known = tasks_by_id(protocol)
    errs += [f"{where}: unknown task {t!r}" for t in tasks if t not in known]
    seen = issue.get("sessions") or []
    errs += [f"{where}: unknown session {s!r}" for s in seen if s not in sessions]
    if found_by == "test" and not seen:
        errs.append(f"{where}: an issue found in a test names its session(s)")
    if found_by == "walkthrough" and seen:
        errs.append(f"{where}: a walkthrough issue (before P1) has no session")
    sev = issue.get("severity")
    if not (_int(sev) and 0 <= sev <= 4):
        errs.append(f"{where}: severity must be 0-4 (4 blocks the task)")
    if issue.get("kind") not in protocol["issue_kinds"]:
        errs.append(f"{where}: kind must be one of {protocol['issue_kinds']}")
    if not (isinstance(issue.get("description"), str) and issue["description"].strip()):
        errs.append(f"{where}: description must be a non-empty text")
    status = issue.get("status")
    if status not in STATUSES:
        errs.append(f"{where}: status must be one of {STATUSES}")
    commit = issue.get("commit")
    if status == "fixed" and not (isinstance(commit, str) and SHA.fullmatch(commit)):
        errs.append(f"{where}: status fixed needs the sha of the fixing commit")
    elif commit is not None and not (isinstance(commit, str) and SHA.fullmatch(commit)):
        errs.append(f"{where}: commit must be a git sha")
    verified = issue.get("verified")
    if verified is not None:
        if status != "fixed":
            errs.append(f"{where}: only a fixed issue can be verified")
        is_session = isinstance(verified, str) and SESSION_ID.fullmatch(verified)
        if is_session and verified not in sessions:
            errs.append(f"{where}: verified by unknown session {verified}")
        node = TEST_NODE.fullmatch(verified) if isinstance(verified, str) else None
        if not is_session and not node:
            errs.append(f"{where}: verified must be a later session (P4) or a test node "
                        "(tests/test_x.py::test_name[param-id])")  # fmt: skip
        elif node and node.group(3) and looks_personal(node.group(3)):
            errs.append(f"{where}.verified: looks like contact details (ENF-07): remove them")
    return errs


def empty_results() -> dict:
    return {"schema_version": SCHEMA_VERSION, "sessions": [], "issues": []}


def load_results(path: Path = USERTESTS_RESULTS, protocol: dict | None = None) -> dict:
    """The checked results file (sessions, issues); ONE ConfigError lists every problem.
    An absent file is the empty template."""
    path = Path(path)
    protocol = protocol if protocol is not None else load_protocol()
    if not path.is_file():
        return empty_results()
    raw = _read(path)
    _raise(path, check_results(raw, protocol))
    for s in raw["sessions"]:
        s["date"] = _date(s["date"]).isoformat()
    return raw


def complete_sessions(results: dict, protocol: dict) -> list[dict]:
    """Sessions in which every required task has an outcome (`skipped` included) and at
    least one required task was scored: the sessions A-10 counts (D112)."""
    required = set(required_tasks(protocol))
    out = []
    for s in results.get("sessions") or []:
        rec = {r["task"]: r.get("outcome") for r in s.get("tasks") or []}
        if all(rec.get(t) is not None for t in required) and any(
            rec.get(t) not in (None, "skipped") for t in required
        ):
            out.append(s)
    return out


# -- fix loop against git --------------------------------------------------------------
def _git(args: list[str], cwd: Path) -> bool:
    try:
        r = subprocess.run(["git", "--no-optional-locks", *args], cwd=cwd,
                           capture_output=True, text=True)  # fmt: skip
    except OSError:
        return False
    return r.returncode == 0


def verify_git(results: dict, repo: Path = Path(".")) -> list[str]:
    """Fix-loop problems that need the repository: a fixed issue's commit must exist; a
    verifying session must run an app version that descends from the fix; a verifying
    test node must exist."""
    repo = Path(repo)
    errs: list[str] = []
    versions = {s.get("id"): s.get("app_version") for s in results.get("sessions") or []}
    for issue in results.get("issues") or []:
        if issue.get("status") != "fixed":
            continue
        iid, commit = issue.get("id"), str(issue.get("commit"))
        if not _git(["cat-file", "-e", f"{commit}^{{commit}}"], repo):
            errs.append(f"issue {iid}: fix commit {commit} not found in the repository")
            continue
        verified = issue.get("verified")
        if not isinstance(verified, str):
            continue
        node = TEST_NODE.fullmatch(verified)
        if node:
            file, name = repo / node.group(1), node.group(2)
            text = file.read_text(encoding="utf-8") if file.is_file() else ""
            if not re.search(rf"^\s*def {re.escape(name)}\(", text, re.MULTILINE):
                errs.append(f"issue {iid}: test {verified} not found")
        elif verified in versions:
            version = str(versions[verified])
            if not _git(["merge-base", "--is-ancestor", commit, version], repo):
                errs.append(f"issue {iid}: session {verified} ran {version}, which does not "
                            f"contain the fix {commit}")  # fmt: skip
    return errs

"""`docs/usertests.md` and `docs/usertests/synthesis.json` from the results (D112).

- **A-10** is PENDING while fewer than `participants.min` (3) sessions are complete, DONE
  for 3 to 5, and DONE with an "over the spec range" flag above 5. It is never PASS: the
  criterion asks that the tests are run and synthesised, not that a score is reached.
  Results marked `synthetic: true` (the test fixture) give **SYNTHETIC**, never DONE,
  whatever their session count: invented sessions can never close A-10.
- The **ENF-06 / ENF-05 verdict lines** print "PENDING (n<3)" below 3 complete sessions,
  so one or two sessions never read as a verdict. The per-task rows are descriptive and
  list every raw time.
- Nothing is estimated: an empty results file gives PENDING in every cell.

The output is deterministic (no timestamp; source hashes over LF-normalised text), so a
test can compare the committed page with a rebuild.
"""

from __future__ import annotations

import hashlib
import json
import statistics as st
from pathlib import Path
from typing import Any

from traffic_twin.paths import USERTESTS_CONFIG, USERTESTS_RESULTS
from traffic_twin.usertests.protocol import complete_sessions, required_tasks, tasks_by_id

PENDING = "PENDING"
SYNTHETIC = "SYNTHETIC"  # invented test data: never DONE (A-10 is read as PENDING)
SCORED = ("success", "partial", "fail")


def a10_status(results: dict, n: int, lo: int) -> str:
    """A-10 from the number of complete sessions: PENDING below `lo`, else DONE; always
    SYNTHETIC for `synthetic: true` results, so the fixture can never close A-10."""
    if results.get("synthetic"):
        return SYNTHETIC
    return PENDING if n < lo else "DONE"


def text_sha256(path: Path) -> str | None:
    """sha256 of a text file with LF line ends (the same on Windows and Linux)."""
    path = Path(path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _median(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return None if not xs else float(st.median(xs))


def sus_score(answers: list[int]) -> float:
    """SUS-10 score 0-100 (odd items positive, even items negative)."""
    return 2.5 * sum((v - 1) if i % 2 == 0 else (5 - v) for i, v in enumerate(answers))


def _task_stats(task: dict, records: list[dict], max_clicks: int) -> dict:
    scored = [r for r in records if r.get("outcome") in SCORED]
    times = [r.get("time_s") for r in scored]
    clicks = [r.get("clicks") for r in scored if r.get("clicks") is not None]
    errors: dict[str, int] = {}
    for r in scored:
        for e in r.get("errors") or []:
            errors[e] = errors.get(e, 0) + 1
    return {
        "id": task["id"],
        "view": task["view"],
        "optional": bool(task.get("optional")),
        "n": len(scored),
        "skipped": sum(r.get("outcome") == "skipped" for r in records),
        "success": sum(r["outcome"] == "success" for r in scored),
        "success_or_partial": sum(r["outcome"] in ("success", "partial") for r in scored),
        "time_median_s": _median(times),
        "times_s": times,
        "time_limit_s": task["time_limit_s"],
        "clicks_median": _median(clicks),
        "clicks_max": max(clicks) if clicks else None,
        "within_clicks": sum(c <= max_clicks for c in clicks),
        "hints": sum(bool(r.get("hint")) for r in scored),
        "errors": dict(sorted(errors.items())),
        "seq_median": _median([r.get("seq") for r in scored]),
    }


def _records(sessions: list[dict], tid: str) -> list[dict]:
    return [r for s in sessions for r in s.get("tasks") or [] if r.get("task") == tid]


def synthesis(results: dict, protocol: dict, *, sources: dict | None = None) -> dict:
    """The synthesis.json content (D112)."""
    sessions = results.get("sessions") or []
    complete = complete_sessions(results, protocol)
    n = len(complete)
    lo, hi = protocol["participants"]["min"], protocol["participants"]["max"]
    targets = protocol["targets"]
    status = a10_status(results, n, lo)
    if n < lo:
        text = (f"{n} complete session(s) of the {lo}-{hi} that spec 12.2 asks for; the "
                "protocol and kit are ready, no result is estimated")  # fmt: skip
    else:
        text = f"{n} complete sessions"
        if n > hi:
            text += f" (over the spec range {lo}-{hi})"
    if status == SYNTHETIC:
        text = (f"synthetic test data, not real participants ({text}); A-10 stays open "
                "until real sessions are recorded")  # fmt: skip
    tasks = tasks_by_id(protocol)
    per_task = [_task_stats(t, _records(complete, tid), targets["max_clicks"])
                for tid, t in tasks.items()]  # fmt: skip
    rounds = sorted({s.get("round") for s in complete if s.get("round") is not None})
    by_round = {}
    if len(rounds) > 1:
        for rd in rounds:
            group = [s for s in complete if s.get("round") == rd]
            by_round[str(rd)] = {
                "n": len(group),
                "tasks": [
                    _task_stats(t, _records(group, tid), targets["max_clicks"])
                    for tid, t in tasks.items()
                ],  # fmt: skip
            }
    verdict = n >= lo
    gate = f"{PENDING} (n<{lo})"
    first = protocol["order"]["first"]
    ped = [r.get("time_s") for r in _records(complete, first) if r.get("outcome") in SCORED]
    clicks = [r["clicks"] for s in complete for r in s.get("tasks") or []
              if r.get("outcome") in SCORED and r.get("clicks") is not None]  # fmt: skip
    glance = [r for t in tasks.values() if t.get("glance")
              for r in _records(complete, t["id"]) if r.get("outcome") in SCORED]  # fmt: skip
    if verdict and ped:
        med = _median(ped)
        ped_line = {
            "status": "MET" if med < targets["pedestrian_time_s"] else "NOT MET",
            "median_s": med, "under": sum(x < targets["pedestrian_time_s"] for x in ped),
            "of": len(ped),
        }  # fmt: skip
    else:
        ped_line = {"status": gate}
    if verdict and clicks:
        within = sum(c <= targets["max_clicks"] for c in clicks)
        clicks_line = {"status": "MET" if within == len(clicks) else "NOT MET",
                       "within": within, "of": len(clicks)}  # fmt: skip
    else:
        clicks_line = {"status": gate}
    if verdict and glance:
        ok = sum(r["outcome"] == "success" for r in glance)
        glance_line = {"status": "MET" if ok == len(glance) else "NOT MET", "success": ok,
                       "of": len(glance)}  # fmt: skip
    else:
        glance_line = {"status": gate}
    understood = [s.get("simulated_understood") for s in complete]
    if verdict:
        k = sum(v is True for v in understood)
        enf05 = {"status": "MET" if k == n else "NOT MET", "understood": k, "of": n,
                 "unanswered": sum(v is None for v in understood)}  # fmt: skip
    else:
        enf05 = {"status": gate}
    sus = [sus_score(s["sus"]) for s in complete if s.get("sus")]
    own = []
    for s in complete:
        mine = [r for r in s.get("tasks") or []
                if tasks.get(r.get("task"), {}).get("view") == s.get("profile")
                and r.get("outcome") in SCORED]  # fmt: skip
        own.append({"session": s["id"], "profile": s.get("profile"),
                    "profile_match": s.get("profile_match"), "round": s.get("round"),
                    "own_tasks": len(mine),
                    "own_success": sum(r["outcome"] == "success" for r in mine)})  # fmt: skip
    issues = sorted(results.get("issues") or [],
                    key=lambda i: (-int(i.get("severity") or 0), str(i.get("id"))))  # fmt: skip
    rows = [{k: i.get(k) for k in ("id", "found_by", "tasks", "sessions", "severity", "kind",
                                   "description", "fix", "status", "commit", "verified")}
            for i in issues]  # fmt: skip
    return {
        "schema_version": 1,
        "synthetic": bool(results.get("synthetic")),
        "status": status,
        "n": n,
        "a10": {
            "status": status,
            "n_complete": n,
            "n_sessions": len(sessions),
            "range": [lo, hi],
            "text": text,
        },  # fmt: skip
        "preregistered": protocol["preregistered"],
        "required_tasks": required_tasks(protocol),
        "targets": dict(targets),
        "tasks": per_task,
        "rounds": by_round,
        "enf06": {"pedestrian_time": ped_line, "clicks": clicks_line, "glance": glance_line},
        "enf05": enf05,
        "sus": {"n": len(sus), "mean": st.mean(sus), "scores": sus} if sus else None,
        "profiles": own,
        "issues": {
            "n": len(rows),
            "open": sum(i["status"] == "open" for i in rows),
            "fixed": sum(i["status"] == "fixed" for i in rows),
            "wontfix": sum(i["status"] == "wontfix" for i in rows),
            "verified": sum(bool(i["verified"]) for i in rows),
            "rows": rows,
        },
        "sources": sources or {},
    }


# -- page ------------------------------------------------------------------------------
def _num(x: Any, nd: int = 1) -> str:
    if x is None:
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _task_table(rows: list[dict], max_clicks: int) -> list[str]:
    out = [
        "| Task | View | n | Success | Success + partial | Median time s (all values) "
        f"| Clicks median / max | ≤ {max_clicks} clicks | Hints | Errors | SEQ median |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in rows:
        name = t["id"] + (" (optional)" if t["optional"] else "")
        if t["n"] == 0:
            out.append(f"| {name} | {t['view']} | 0 | " + " | ".join([PENDING] * 8) + " |")
            continue
        k = t["n"]
        times = ", ".join(_num(x) for x in t["times_s"])
        errs = ", ".join(f"{e} ×{c}" for e, c in t["errors"].items()) or "–"
        out.append(
            f"| {name} | {t['view']} | {k} | {t['success']}/{k} | {t['success_or_partial']}/{k} "
            f"| {_num(t['time_median_s'])} ({times}) | {_num(t['clicks_median'])} / "
            f"{_num(t['clicks_max'])} | {t['within_clicks']}/{k} | {t['hints']} | {errs} "
            f"| {_num(t['seq_median'])} |"
        )
    return out


def render(synth: dict) -> str:
    """The docs/usertests.md page."""
    a10, tg = synth["a10"], synth["targets"]
    lo, hi = a10["range"]
    src = synth.get("sources") or {}
    lines = [
        "# User tests (spec 12.2, A-10)",
        "",
        "Generated by `tasks.py usertest-docs` from `docs/usertests/results.yaml` and the "
        "pre-registered protocol `configs/usertests.yaml` "
        f"(registered {synth['preregistered']}). Do not edit by hand. Protocol: "
        "[usertest_protocol.md](usertest_protocol.md); printable kit: "
        "[usertests/usertest_kit.pdf](usertests/usertest_kit.pdf).",
        "",
    ]
    if synth.get("synthetic"):
        lines += ["**SYNTHETIC TEST DATA: these are not results of real participants.**", ""]
    text = a10["text"]
    lines += [f"**A-10: {a10['status']}.** {text[:1].upper()}{text[1:]}.", ""]
    if a10["status"] == PENDING:
        lines += [
            f"A-10 needs {lo} to {hi} complete sessions with real participants. Until then "
            "every result cell below reads PENDING; nothing is estimated, and the verdict "
            f"lines stay PENDING while fewer than {lo} sessions are complete.",
            "",
        ]
    lines += ["## Results per task", "", *_task_table(synth["tasks"], tg["max_clicks"]), ""]
    if synth["rounds"]:
        lines += ["## Before and after the fixes", ""]
        for rd, group in synth["rounds"].items():
            label = "before the fixes" if rd == "1" else "on the fixed app"
            lines += [f"### Round {rd} ({label}, n = {group['n']})", "",
                      *_task_table(group["tasks"], tg["max_clicks"]), ""]  # fmt: skip
    e6, e5 = synth["enf06"], synth["enf05"]
    lines += ["## ENF-06 and ENF-05 checks (descriptive, n ≤ 5)", ""]
    p = e6["pedestrian_time"]
    detail = (f": median {_num(p['median_s'])} s, {p['under']}/{p['of']} under the target"
              if "median_s" in p else "")  # fmt: skip
    lines.append(
        f"- Pedestrian view < {_num(tg['pedestrian_time_s'], 0)} s on "
        f"{synth['required_tasks'][0]} (first exposure){detail}: {p['status']}"
    )
    c = e6["clicks"]
    detail = f": {c['within']}/{c['of']} task attempts" if "of" in c else ""
    lines.append(f"- At most {tg['max_clicks']} clicks on every task{detail}: {c['status']}")
    g = e6["glance"]
    detail = f": {g['success']}/{g['of']} correct" if "of" in g else ""
    lines.append(f"- Ambulance screen read in a {_num(tg['ambulance_glance_s'], 0)} s glance"
                 f"{detail}: {g['status']}")  # fmt: skip
    detail = f": {e5['understood']}/{e5['of']} answered 'simulated'" if "of" in e5 else ""
    lines.append(f"- ENF-05, the data are understood to be simulated{detail}: {e5['status']}")
    if synth["sus"]:
        s = synth["sus"]
        scores = ", ".join(_num(x) for x in s["scores"])
        lines.append(f"- SUS-10 (optional, indicative with n = {s['n']}; our own French "
                     f"wording, not a validated translation): mean {_num(s['mean'])}, "
                     f"scores {scores}")  # fmt: skip
    lines.append("")
    if synth["profiles"]:
        lines += ["## Participants (anonymous ids)", "",
                  "| Id | Round | Profile | Role | Own-profile tasks: success |",
                  "|---|---|---|---|---|"]  # fmt: skip
        for p in synth["profiles"]:
            lines.append(
                f"| {p['session']} | {p['round']} | {p['profile']} | "
                f"{p['profile_match']} | {p['own_success']}/{p['own_tasks']} |"
            )
        lines.append("")
    iss = synth["issues"]
    lines += ["## Issues and fixes", ""]
    if not iss["rows"]:
        lines += [f"{PENDING}: no issue recorded yet.", ""]
    else:
        lines += [
            f"{iss['n']} issues: {iss['open']} open, {iss['fixed']} fixed, {iss['wontfix']} "
            f"won't fix; {iss['verified']} verified after the fix (on a later session or by an "
            "automated test). Issues found by the expert walkthrough before the first "
            "participant are marked `walkthrough`.",
            "",
            "| Id | Found by | Tasks | Sessions | Severity | Problem | Fix | Status | Commit "
            "| Verified |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i in iss["rows"]:
            lines.append(
                f"| {i['id']} | {i['found_by']} | {', '.join(i['tasks'] or []) or '–'} | "
                f"{', '.join(i['sessions'] or []) or '–'} | {i['severity']} | "
                f"{_cell(i['description'])} | {_cell(i['fix'] or '–')} | {i['status']} | "
                f"{i['commit'] or '–'} | {i['verified'] or '–'} |"
            )
        lines.append("")
    lines += [
        "## Limitations",
        "",
        f"- {lo} to {hi} participants give qualitative evidence, not statistics: medians are "
        "fragile and every raw time is listed.",
        "- Participants may play a role they do not hold (`proxy`); the table says which.",
        "- The facilitator is a developer of the app; a second observer is advised.",
        "- Times come from a phone stopwatch (about ± 0.3 s near the 5 s threshold).",
        "- The app shows simulated data (calibrated 2026-10-08) and an ASSUMED signal "
        "program until the field visit.",
        "",
    ]
    if src:
        lines += ["## Sources", "", "| File | sha256 (LF) |", "|---|---|"]
        for name, sha in src.items():
            lines.append(f"| `{name}` | {sha or 'absent'} |")
        lines.append("")
    return "\n".join(lines)


def sources(protocol_path: Path = USERTESTS_CONFIG, results_path: Path = USERTESTS_RESULTS):
    return {Path(protocol_path).as_posix(): text_sha256(protocol_path),
            Path(results_path).as_posix(): text_sha256(results_path)}  # fmt: skip


def write(synth: dict, doc: Path, json_path: Path) -> list[Path]:
    """Write the page and the JSON (UTF-8, LF line ends)."""
    doc, json_path = Path(doc), Path(json_path)
    for p in (doc, json_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render(synth), encoding="utf-8", newline="\n")
    payload = json.dumps(synth, indent=2, ensure_ascii=False) + "\n"
    json_path.write_text(payload, encoding="utf-8", newline="\n")
    return [doc, json_path]

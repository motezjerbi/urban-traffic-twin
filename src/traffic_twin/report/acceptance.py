"""The generated acceptance table A-01..A-10 (spec 17, D115).

Statuses: PASS, FAIL, INVALID, PENDING, DONE. Every status and every number of the
evidence comes from the facts (report.facts); a criterion whose evidence does not exist
yet is PENDING, with what it waits for. Two rules are procedural:

- A-06 is "PASS — presented: <primary sentence>" when the A-06 test of record is valid
  (whatever its direction, spec 17 and D94), INVALID with its problems otherwise.
- A-10 is DONE (never PASS) with 3 to 5 complete user sessions, PENDING below 3 (D112).
"""

from __future__ import annotations

CRITERIA = {
    "A-01": "The simulation runs headless and reproducibly: two runs with the same seed give "
    "the same KPIs.",
    "A-02": "The simulated flow per arm is within the fixed threshold of the real flow.",
    "A-03": "The four controllers (fixed, actuated, rules, RL) run on the same scenario and "
    "produce the KPI file.",
    "A-04": "No dangerous signal sequence in the 10 test seeds, and the longest pedestrian "
    "wait stays under the bound.",
    "A-05": "Preemption reduces the ambulance travel time compared with S0.",
    "A-06": "RL is compared with the baselines on 10 seeds with a statistical test; the "
    "result is presented, favourable or not.",
    "A-07": "The announced 80 % ETA interval covers between 75 % and 85 % of the cases.",
    "A-08": "Every view answers through the API, shows 'données simulées' and handles the "
    "loading, error and degraded states.",
    "A-09": "The full demo runs without manual intervention on the code.",
    "A-10": "A test with 3 to 5 users is carried out and synthesised.",
}
REQUIREMENTS = {
    "A-01": "EF-03, ENF-01",
    "A-02": "EF-02, O1",
    "A-03": "EF-05 to EF-10",
    "A-04": "EF-11, ENF-11, O6",
    "A-05": "EF-07, O4",
    "A-06": "EF-10, O3",
    "A-07": "EF-13, O7",
    "A-08": "EF-15 to EF-20, ENF-03, ENF-05",
    "A-09": "ENF-09",
    "A-10": "O8",
}
STATUSES = ("PASS", "FAIL", "INVALID", "PENDING", "DONE")
SOURCES = {
    "A-01": "data/results/_rebaseline/<HEAD>/verify.json",
    "A-02": "data/results/validation/pitie_fixed/summary.json",
    "A-03": "data/results/evaluation/summary.json",
    "A-04": "data/results/evaluation/summary.json",
    "A-05": "data/results/evaluation/summary.json",
    "A-06": "data/results/evaluation/a06.json",
    "A-07": "data/models/eta/<LATEST>/eval.json",
    "A-08": "docs/demo/rehearsal.json",
    "A-09": "docs/demo/rehearsal.json",
    "A-10": "docs/usertests/synthesis.json",
}


def _t(facts: dict, key: str) -> str | None:
    f = facts.get(key)
    return None if f is None else f.get("text")


def _v(facts: dict, key: str):
    f = facts.get(key)
    return None if f is None else f.get("value")


def _src(facts: dict, key: str, default: str) -> str:
    f = facts.get(key) or {}
    return f.get("source") or default


def part8_available(facts: dict) -> bool:
    """The single Part 8 run exists: its A-06 record has a status."""
    return _v(facts, "a06.status") in ("PASS", "INVALID")


def _verdict(v) -> str:
    v = str(v or "").upper()
    return v if v in STATUSES else ("PASS" if v == "DONE" else "PENDING")


def acceptance(facts: dict) -> list[dict]:
    """One row per criterion: {id, criterion, requirement, status, evidence, source}."""
    rows = []

    def add(cid: str, status: str, evidence: str, key: str | None = None) -> None:
        assert status in STATUSES, status
        rows.append(
            {
                "id": cid,
                "criterion": CRITERIA[cid],
                "requirement": REQUIREMENTS[cid],
                "status": status,
                "evidence": evidence,
                "source": _src(facts, key, SOURCES[cid]) if key else SOURCES[cid],
            }
        )

    p8 = part8_available(facts)

    # A-01: the re-made evaluation runs byte-identical (D97); only what verify.json holds
    ident, of = _v(facts, "a01.identical"), _v(facts, "a01.of")
    if ident is None or of is None:
        add(
            "A-01",
            "PENDING",
            "waits for the Part 8 rebaseline record (verify.json)",
            "a01.identical",
        )
    else:
        ok = int(ident) == int(of)
        add(
            "A-01",
            "PASS" if ok else "FAIL",
            f"{ident} of {of} re-made evaluation runs "
            f"byte-identical (code {_t(facts, 'a01.head')})",
            "a01.identical",
        )

    # A-02
    v = _t(facts, "val.a02.verdict")
    if v is None:
        add("A-02", "PENDING", "validation summary missing", "val.a02.verdict")
    else:
        add(
            "A-02",
            _verdict(v),
            f"{_t(facts, 'val.a02.pass')} of {_t(facts, 'val.a02.rows')} "
            f"inbound arm × slot rows within ±{_t(facts, 'val.flow_threshold_pct')} of the real "
            f"flow; GEH < 5 on {_t(facts, 'val.geh.n_below')} of "
            f"{_t(facts, 'val.geh.n_counts')} counts",
            "val.a02.verdict",
        )

    # A-03: 4 of 4 controllers only once S3 ran on seeds 1-10
    n, of = _v(facts, "eval.a03.n"), _v(facts, "eval.a03.of")
    if n is None or of is None:
        add("A-03", "PENDING", "evaluation summary missing", "eval.a03.n")
    elif int(n) == int(of):
        add(
            "A-03",
            "PASS",
            f"{n} of {of} controllers on one scenario and one KPI set "
            f"({_t(facts, 'eval.a03.n_kpis')} KPIs per family)",
            "eval.a03.n",
        )
    elif p8:
        add(
            "A-03", "FAIL", f"{n} of {of} controllers on one scenario and one KPI set", "eval.a03.n"
        )
    else:
        add(
            "A-03",
            "PENDING",
            f"{n} of {of} so far (fixed, actuated, rules); RL (S3) runs on seeds 1-10 in Part 8",
            "eval.a03.n",
        )

    # A-04: judged over every run; S3/S4 join in Part 8
    v = _t(facts, "eval.a04.verdict")
    runs = _t(facts, "eval.a04.runs")
    worst, bound = _t(facts, "eval.a04.worst_curb_wait"), _t(facts, "eval.a04.bound")
    tail = f"; worst curb wait {worst} (bound {bound})" if worst and bound else ""
    if v is None:
        add("A-04", "PENDING", "evaluation summary missing", "eval.a04.verdict")
    elif p8:
        add(
            "A-04",
            _verdict(v),
            f"{v} over {runs} runs, 0 dangerous sequences required{tail}",
            "eval.a04.verdict",
        )
    else:
        add(
            "A-04",
            "PENDING",
            f"{v} over the {runs} runs of S0-S2{tail}; the S3/S4 runs are judged in Part 8",
            "eval.a04.verdict",
        )

    # A-05: the primary test is a Part 4 result, reproduced unchanged in Part 8
    v = _t(facts, "eval.a05.verdict")
    if v is None:
        add("A-05", "PENDING", "evaluation summary missing", "eval.a05.verdict")
    else:
        add(
            "A-05",
            _verdict(v),
            f"main family S2 vs S0, ambulance travel time: HL "
            f"{_t(facts, 'eval.a05.hl')} {_t(facts, 'eval.a05.ci')}, exact one-sided p "
            f"{_t(facts, 'eval.a05.p')} (n = {_t(facts, 'eval.a05.n')})",
            "eval.a05.verdict",
        )

    # A-06: procedural (D94)
    st = _v(facts, "a06.status")
    if st == "PASS":
        sent = _t(facts, "a06.primary_sentence") or "primary comparison shown"
        add("A-06", "PASS", f"presented: {sent}", "a06.status")
    elif st == "INVALID":
        add("A-06", "INVALID", f"problems: {_t(facts, 'a06.problems')}", "a06.status")
    else:
        add(
            "A-06", "PENDING", "the single Part 8 run on seeds 1-10 has not been made", "a06.status"
        )

    # A-07
    v = _t(facts, "eta.a07.verdict")
    if v is None:
        add("A-07", "PENDING", "ETA evaluation missing", "eta.a07.verdict")
    else:
        add(
            "A-07",
            _verdict(v),
            f"coverage {_t(facts, 'eta.a07.coverage_pct')} "
            f"{_t(facts, 'eta.a07.ci')} on the realistic test worlds (band "
            f"{_t(facts, 'eta.a07.band')}); median MAE {_t(facts, 'eta.a07.mae')} against "
            f"{_t(facts, 'eta.a07.ref_mae')} for the reference",
            "eta.a07.verdict",
        )

    # A-08 / A-09: the rehearsal record (D115, D116)
    for cid, key, what in (
        ("A-08", "demo.a08", "the recorded rehearsal (every view, states, footer)"),
        ("A-09", "demo.a09", "the fresh-clone readiness rehearsal"),
    ):
        v = _t(facts, key)
        if v is None:
            add(cid, "PENDING", f"waits for {what}: docs/demo/rehearsal.json", key)
        else:
            date, commit = _t(facts, f"{key}.date"), _t(facts, f"{key}.commit")
            fresh = _v(facts, "demo.fresh_clone")
            extra = " on a fresh clone" if cid == "A-09" and fresh else ""
            why = _t(facts, f"{key}.reason") if v != "PASS" else None
            ev = f"{what}{extra}" + (f", {date}" if date else "")
            ev += f", code {commit}" if commit else ""
            ev += f" ({why})" if why else ""
            add(cid, _verdict(v), ev, key)

    # A-10: DONE with 3-5 complete sessions, never PASS (D112)
    v, n = _t(facts, "usertests.status"), _v(facts, "usertests.n")
    if v is None or n is None or int(n) < 3:
        got = "no session yet" if not n else f"{n} complete session(s)"
        add(
            "A-10", "PENDING", f"{got}; 3 to 5 real sessions needed (spec 12.2)", "usertests.status"
        )
    else:
        status = "DONE" if v in ("DONE", "PASS") else _verdict(v)
        over = " (over the spec range of 5)" if int(n) > 5 else ""
        add("A-10", status, f"{n} complete sessions synthesised{over}", "usertests.status")
    return rows

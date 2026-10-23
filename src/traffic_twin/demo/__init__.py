"""Rehearsed demo (Part 10, D116): serve both fronts, walk the configs/demo.yaml steps,
rehearse (real and fixtures modes), pack, bootstrap and fresh-clone readiness. No
simulation runs during the demo (spec 7.1).

    python -m traffic_twin.demo serve | check | rehearsal | clean | pack | bootstrap | readiness

Modules: `config` (the demo script, deep links and API calls per step), `common` (ports,
the impostor probe, process groups, reports), `prereq`, `check`, `scan`, `apptest`,
`rehearsal`, `serve`, `clean`, `pack`, `bootstrap`, `readiness`.
"""

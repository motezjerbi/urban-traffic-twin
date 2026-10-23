"""User tests of the app (spec 12.2, A-10; D111, D112): the pre-registered protocol
(`configs/usertests.yaml`), expected answers read from the API on the frozen demo run,
the printable kit, the results check (ENF-07 privacy guard, scoring rules, fix loop) and
the synthesis (`docs/usertests.md`, `docs/usertests/synthesis.json`).

    python -m traffic_twin.usertests check | kit | links | serve | docs
"""

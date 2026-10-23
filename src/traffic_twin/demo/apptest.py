"""Streamlit AppTest of the Operator (`app.py`) for the rehearsal, run in a child process
so the app's environment (`TT_API_INPROCESS`, `TT_API_CONFIG`, `TT_API_URL`) and
Streamlit's state stay out of the caller:

    python -m traffic_twin.demo.apptest --script app.py [--timeout 60]

prints one JSON line {exceptions, texts, n_elements, seconds}. `run_apptest` starts it and
checks the answer: no exception, 'données simulées' and the calibration date shown in the
nominal case; with the API down a simple message and no traceback (spec 12.1, A-08).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from traffic_twin.demo.common import Report, child_env

TEXT_ATTRS = ("value", "label", "body", "message")


def _walk(node):
    yield node
    for child in (getattr(node, "children", None) or {}).values():
        yield from _walk(child)


def collect(script: str, timeout: float) -> dict:
    from streamlit.testing.v1 import AppTest

    t0 = time.perf_counter()
    at = AppTest.from_file(script, default_timeout=timeout)
    at.run()
    texts: list[str] = []
    n = 0
    for node in _walk(at._tree):
        n += 1
        for attr in TEXT_ATTRS:
            v = getattr(node, attr, None)
            if isinstance(v, str) and v.strip():
                texts.append(v)
    exceptions = []
    for e in at.exception:
        exceptions.append(str(getattr(e, "message", None) or getattr(e, "value", None) or e))
    return {
        "exceptions": exceptions,
        "texts": texts,
        "n_elements": n,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m traffic_twin.demo.apptest")
    p.add_argument("--script", default="app.py")
    p.add_argument("--timeout", type=float, default=60.0)
    a = p.parse_args(argv)
    try:
        out = collect(a.script, a.timeout)
    except Exception as e:  # the parent reports it as a failed check
        out = {
            "exceptions": [f"AppTest could not run: {type(e).__name__}: {e}"],
            "texts": [],
            "n_elements": 0,
            "seconds": None,
        }
    sys.stdout.write("APPTEST_JSON " + json.dumps(out, ensure_ascii=False) + "\n")
    return 0


def run_apptest(root: Path, script: str, env: dict[str, str | None], timeout: float = 90.0) -> dict:
    """The AppTest answer of `script` under `env` (added to os.environ; None removes)."""
    cmd = [
        sys.executable,
        "-m",
        "traffic_twin.demo.apptest",
        "--script",
        str(Path(root) / script),
        "--timeout",
        str(min(timeout, 60.0)),
    ]
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(root),
            env=child_env(**env),
            capture_output=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"exceptions": [f"AppTest did not finish in {timeout:.0f} s"], "texts": []}
    line = next((ln for ln in cp.stdout.splitlines() if ln.startswith("APPTEST_JSON ")), None)
    if line is None:
        tail = (cp.stderr or cp.stdout).strip().splitlines()[-3:]
        return {"exceptions": [f"AppTest exited {cp.returncode}: {' | '.join(tail)}"], "texts": []}
    return json.loads(line[len("APPTEST_JSON ") :])


def check_nominal(
    rep: Report, res: dict, calibrated_on: str | None, group: str = "streamlit"
) -> None:
    from traffic_twin.demo.check import fold

    exc = res.get("exceptions") or []
    rep.check(
        group,
        "AppTest renders without exception",
        not exc,
        "; ".join(exc)[:300] or f"{res.get('n_elements')} elements in {res.get('seconds')} s",
    )
    text = fold(" ".join(res.get("texts") or []))
    rep.check(group, "'Données simulées' shown (ENF-05)", "donnees simulees" in text)
    if calibrated_on:
        rep.check(
            group,
            f"calibration date {calibrated_on} shown",
            calibrated_on in text or _fr_date(calibrated_on) in text,
        )
    rep.check(group, "no traceback shown", "traceback" not in text)


def check_down(rep: Report, res: dict, group: str = "api_down") -> None:
    from traffic_twin.demo.check import fold

    res = {
        **res,
        "texts": [t for t in res.get("texts") or [] if not t.lstrip().startswith("<style")],
    }
    exc = res.get("exceptions") or []
    rep.check(group, "Streamlit with the API down: no exception", not exc, "; ".join(exc)[:300])
    text = fold(" ".join(res.get("texts") or []))
    words = ("indisponible", "injoignable", "erreur", "pas de reponse")
    shown = [t for t in res.get("texts") or [] if any(w in fold(t) for w in words)]
    rep.check(
        group,
        "Streamlit with the API down: simple message, no traceback",
        bool(shown) and "traceback" not in text,
        (shown[0] if shown else " ".join(res.get("texts") or []))[:160],
    )


def _fr_date(iso: str) -> str:
    y, m, d = (iso.split("-") + ["", ""])[:3]
    return f"{d}/{m}/{y}"


if __name__ == "__main__":
    raise SystemExit(main())

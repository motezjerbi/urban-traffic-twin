"""Static honesty scan of the two front ends for the rehearsal (D107; the full contract
tests are tests/test_app_static.py and tests/test_app_streamlit.py):

- `mock.html`: none of the banned strings (hard-coded S3 results, fake run ids, invented
  countdowns, "certifié", "en direct", `Math.random`, ...), no `b.<field> || <literal>`
  fallback outside `notice` / `message`, no backslash-r escape, and every fetched path is
  an API route;
- `app.py`: none of its banned strings, no `EXPORT_DIR`, no read of docs/ or data/.
"""

from __future__ import annotations

import re
from pathlib import Path

from traffic_twin.demo.common import Report

MOCK = Path("src/traffic_twin/app/mock.html")
MOCK_BANNED = (
    "S3_DATA",
    "s3_rl_dqn",
    "next_arrival",
    "countdown_text",
    "unit_text",
    "00:28",
    "Garantie",
    "certifié",
    "en direct",
    "V2I",
    "Télématique",
    "Capteurs réels 30",
    "SUMO 60 FPS",
    "Math.random",
)
FALLBACK_ALLOWED = ("notice", "message")
APP_BANNED = ("Seed 5", "0,0020", "0,0010", "parties 6-7", "cycle 123 s", "EXPORT_DIR", "EVAL_MD")
API_PREFIXES = (
    "/health",
    "/scenarios",
    "/kpi",
    "/replay",
    "/crossing",
    "/bus",
    "/ambulance",
    "/eta",
    "/evaluation",
    "/compare",
    "/series",
    "/sensitivity",
    "/three.min.js",
    "/mock",
)
_FALLBACK = re.compile(r"\bb\.(\w+)\s*\|\|\s*(?:[\"'`]|-?\d)")
_FETCH = re.compile(r"\bapi\(\s*[`\"'](/[A-Za-z0-9_./-]*)")
_EVAL_MD = re.compile(r"[\"'][^\"'\n]*evaluation\.md[\"']")  # a path literal to the page
_READS = re.compile(r"(open\(|read_csv\(|read_text\(|read_parquet\().{0,80}(docs|data)[/\\\"']")


def scan_mock(rep: Report, root: Path, group: str = "static mock.html") -> None:
    path = root / MOCK
    html = path.read_text(encoding="utf-8")
    found = [s for s in MOCK_BANNED if s in html]
    rep.check(group, "no banned string (D107)", not found, f"found: {found}" if found else "")
    fb = sorted({m[1] for m in _FALLBACK.finditer(html)} - set(FALLBACK_ALLOWED))
    rep.check(group, "no b.<field> || <literal> fallback", not fb, f"fields: {fb}" if fb else "")
    rep.check(group, "no backslash-r escape", "\\r" not in html)
    paths = sorted({m[1] for m in _FETCH.finditer(html)})
    bad = [p for p in paths if not p.startswith(API_PREFIXES)]
    rep.check(
        group,
        "every fetched path is an API route",
        bool(paths) and not bad,
        f"not routes: {bad}" if bad else ", ".join(paths),
    )


def scan_app(rep: Report, root: Path, script: str = "app.py", group: str = "static app.py") -> None:
    text = (root / script).read_text(encoding="utf-8")
    found = [s for s in APP_BANNED if s in text]
    rep.check(group, "no banned string (D107)", not found, f"found: {found}" if found else "")
    reads = [m.group(0)[:60] for m in _READS.finditer(text)]
    reads += [m.group(0) for m in _EVAL_MD.finditer(text)]
    rep.check(group, "no direct read of docs/ or data/ (D102)", not reads, "; ".join(reads))


def scan(rep: Report, root: Path, script: str = "app.py") -> None:
    scan_mock(rep, root)
    scan_app(rep, root, script)

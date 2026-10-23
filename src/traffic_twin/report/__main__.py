"""CLI of the report and the deck (D113-D115). Run from the repo root.

    python -m traffic_twin.report facts   [--out docs/report/facts.json]
    python -m traffic_twin.report report  [--facts F] [--out-dir docs/report] [--no-figures]
    python -m traffic_twin.report slides  [--facts F] [--out docs/slides/urban_traffic_twin.pptx]
    python -m traffic_twin.report figures [--facts F] [--out docs/figures/report]
    python -m traffic_twin.report shots   [--out docs/figures/app] [--api-url URL]
    python -m traffic_twin.report check   [--facts F] [--qa-dir data/report/_qa]

`facts` needs the local outputs under data/ (not versioned); every other command reads
only the facts file (default: the committed docs/report/facts.json), so the report and
the deck rebuild on a fresh clone. `report` writes report.md, report.pdf and build.json
(the platform fingerprint that decides whether the committed PDF must match a rebuild
byte for byte). A missing fact is a visible PENDING; a broken include, a missing figure
or a table under 6.5 pt fails the command (exit 1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

from traffic_twin.paths import APP_FIGURES, REPORT_DIR, REPORT_FACTS, REPORT_SRC, SLIDES_PPTX

log = logging.getLogger("traffic_twin.report")
COMMANDS = ("facts", "report", "slides", "figures", "shots", "check")
BUILD_JSON = "build.json"
QA_DIR = Path("data/report/_qa")


def fingerprint() -> dict:
    """What the PDF bytes depend on besides the input (D113: byte equality is checked
    only on a matching platform)."""
    from importlib.metadata import version

    def v(pkg: str) -> str:
        try:
            return version(pkg)
        except Exception:  # noqa: BLE001 - a missing package is just "none"
            return "none"

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "reportlab": v("reportlab"),
        "pillow": v("pillow"),
        "markdown": v("markdown"),
        "matplotlib": v("matplotlib"),
        "zlib": zlib.ZLIB_RUNTIME_VERSION,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def footer_text(facts: dict) -> str:
    code = (facts.get("meta.code") or {}).get("text") or "unknown"
    return f"Simulated data · ASSUMED signal program · code {code}"


def build_report(
    facts: dict, out_dir: Path = REPORT_DIR, *, figures: bool = True, root: Path = Path(".")
) -> dict:
    """report.md + report.pdf + build.json into `out_dir`; returns the build record."""
    from traffic_twin.report.assemble import REPORT_MD, assemble
    from traffic_twin.report.figures import REPORT_FIGURES, make_figures
    from traffic_twin.report.pdf import render_pdf

    if figures:
        make_figures(facts, Path(root) / REPORT_FIGURES)
    md, pending = assemble(REPORT_SRC, facts, REPORT_MD, root=root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, pdf_path = out_dir / "report.md", out_dir / "report.pdf"
    md_path.write_text(md, encoding="utf-8", newline="\n")
    title = (facts.get("meta.title") or {}).get("text") or "Urban Traffic Twin"
    authors = (facts.get("meta.authors") or {}).get("text") or "Urban Traffic Twin"
    pages = render_pdf(
        md,
        Path(root) / REPORT_DIR,
        pdf_path,
        footer_text(facts),
        title=f"{title}: final report",
        author=authors,
    )
    record = {
        "platform": fingerprint(),
        "pages": pages,
        "pending": len(pending),
        "report_md_sha256": _sha(md_path),
        "report_pdf_sha256": _sha(pdf_path),
        "facts_code_version": (facts.get("meta.code") or {}).get("value"),
    }
    (out_dir / BUILD_JSON).write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return record


def _load(path: Path) -> dict:
    from traffic_twin.report.facts import load_facts

    return load_facts(path)


def cmd_facts(args) -> int:
    from traffic_twin.report.facts import collect_facts, write_facts

    facts = collect_facts()
    out = write_facts(facts, args.out)
    pending = sorted(k for k, f in facts.items() if f.get("text") is None)
    print(f"{out.as_posix()}: {len(facts)} facts, {len(pending)} PENDING")
    for k in pending:
        log.info("PENDING %s: %s", k, facts[k].get("pending"))
    return 0


def cmd_report(args) -> int:
    rec = build_report(_load(args.facts), args.out_dir, figures=not args.no_figures)
    print(
        f"{Path(args.out_dir).as_posix()}/report.pdf: {rec['pages']} pages, "
        f"{rec['pending']} PENDING item(s)"
    )
    return 0


def cmd_slides(args) -> int:
    from traffic_twin.report.slides import build_deck

    out = build_deck(_load(args.facts), args.out)
    print(f"{out.as_posix()}: 15 slides")
    return 0


def cmd_figures(args) -> int:
    from traffic_twin.report.figures import make_figures

    for p in make_figures(_load(args.facts), args.out):
        print(p.as_posix())
    return 0


def cmd_shots(args) -> int:
    from traffic_twin.report.shots import load_demo_config, take_shots

    files = take_shots(
        load_demo_config(args.demo_config),
        args.out,
        api_url=args.api_url,
        operator_url=args.operator_url,
    )
    for p in files:
        print(p.as_posix())
    return 0


def cmd_check(args) -> int:
    """Optional QA: template lint, keys.txt, stale sources, and (when LibreOffice is
    installed) renders of the PDF and the deck under data/report/_qa."""
    from traffic_twin.report.assemble import PLACEHOLDER, lint, templates
    from traffic_twin.report.facts import stale_sources

    problems = []
    used = set()
    for t in templates(REPORT_SRC):
        text = t.read_text(encoding="utf-8")
        used |= set(PLACEHOLDER.findall(text))
        problems += [f"{t.name} {p}" for p in lint(text)]
    keys_file = REPORT_DIR / "keys.txt"
    listed = {
        ln.strip()
        for ln in keys_file.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    }
    problems += [f"keys.txt lacks {k}" for k in sorted(used - listed)]
    problems += [f"keys.txt lists unused {k}" for k in sorted(listed - used)]
    facts = _load(args.facts)
    stale = stale_sources(facts)
    for line in problems:
        print(f"PROBLEM {line}")
    for line in stale:
        print(f"STALE {line} (rerun: python -m traffic_twin.report facts)")
    soffice = shutil.which("soffice") or next(
        (p for p in (r"C:\Program Files\LibreOffice\program\soffice.exe",) if Path(p).is_file()),
        None,
    )
    if soffice is None:
        print("LibreOffice not found: renders skipped")
    else:
        qa = Path(args.qa_dir)
        qa.mkdir(parents=True, exist_ok=True)
        for src in (REPORT_DIR / "report.pdf", SLIDES_PPTX):
            if not src.is_file():
                print(f"{src.as_posix()} not built: render skipped")
                continue
            fmt = "pdf" if src.suffix == ".pptx" else "png"
            subprocess.run(
                [soffice, "--headless", "--convert-to", fmt, "--outdir", str(qa), str(src)],
                check=False,
                capture_output=True,
                timeout=300,
            )
            print(f"rendered {src.as_posix()} -> {qa.as_posix()} ({fmt})")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m traffic_twin.report", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("facts", help="collect docs/report/facts.json from the local outputs")
    p.add_argument("--out", type=Path, default=REPORT_FACTS)
    p = sub.add_parser("report", help="report.md + report.pdf + build.json from the facts")
    p.add_argument("--facts", type=Path, default=REPORT_FACTS)
    p.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    p.add_argument(
        "--no-figures",
        action="store_true",
        help="do not redraw docs/figures/report (they render PENDING if absent)",
    )
    p = sub.add_parser("slides", help="the 15-slide deck from the facts")
    p.add_argument("--facts", type=Path, default=REPORT_FACTS)
    p.add_argument("--out", type=Path, default=SLIDES_PPTX)
    p = sub.add_parser("figures", help="docs/figures/report/*.png from the facts")
    p.add_argument("--facts", type=Path, default=REPORT_FACTS)
    p.add_argument("--out", type=Path, default=Path("docs/figures/report"))
    p = sub.add_parser("shots", help="app screenshots of every configs/demo.yaml step")
    p.add_argument("--out", type=Path, default=APP_FIGURES)
    p.add_argument("--api-url", default=None, help="a running API (default: start one)")
    p.add_argument("--operator-url", default=None, help="a running Streamlit Operator")
    p.add_argument("--demo-config", type=Path, default=Path("configs/demo.yaml"))
    p = sub.add_parser("check", help="lint, keys, stale sources, optional LibreOffice renders")
    p.add_argument("--facts", type=Path, default=REPORT_FACTS)
    p.add_argument("--qa-dir", type=Path, default=QA_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    from traffic_twin.report.assemble import ReportError
    from traffic_twin.report.facts import FactsError
    from traffic_twin.report.shots import ShotsError
    from traffic_twin.report.slides import DeckError

    try:
        from traffic_twin.app.browser import BrowserError
    except ImportError:  # pragma: no cover - the helper ships with the package
        BrowserError = ShotsError  # noqa: N806
    args = build_parser().parse_args(argv)
    handler = {
        "facts": cmd_facts,
        "report": cmd_report,
        "slides": cmd_slides,
        "figures": cmd_figures,
        "shots": cmd_shots,
        "check": cmd_check,
    }[args.command]
    try:
        return handler(args)
    except (ReportError, FactsError, DeckError, ShotsError, BrowserError) as exc:
        print(f"traffic_twin.report {args.command}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

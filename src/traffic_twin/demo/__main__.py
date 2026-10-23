"""CLI: python -m traffic_twin.demo <command> (D116; the `tasks.py demo*` tasks call it).

    serve      start the API (8000) and the Streamlit Operator (8501); Ctrl+C stops both
    check      walk every configs/demo.yaml step against a running API (--url)
    rehearsal  headless rehearsal (real data, or --fixtures); --record writes
               docs/demo/rehearsal.json (A-08 / A-09 evidence)
    clean      remove demo leftovers (never data/results, data/demo or models)
    pack       write docs/demo/demo_pack.zip + MANIFEST.json
    bootstrap  restore a pack into a fresh clone (--from-pack), or rebuild the demo runs at
               seed 4000 (--demo-only); never the evaluation
    readiness  fresh `git clone --local` + bootstrap + rehearsal (A-09)

Exit status: 0 = no failed check (PENDING items are listed), 1 = a check failed or a
command refused, 2 = bad arguments or configuration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COMMANDS = ("serve", "check", "rehearsal", "clean", "pack", "bootstrap", "readiness")


def build_parser() -> argparse.ArgumentParser:
    from traffic_twin.paths import DEMO_CONFIG, DEMO_PACK

    parser = argparse.ArgumentParser(
        prog="python -m traffic_twin.demo", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEMO_CONFIG,
        help=f"demo script (default {DEMO_CONFIG.as_posix()})",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    def api_cfg(p):
        p.add_argument(
            "--api-config",
            type=Path,
            default=None,
            help="API settings (default: TT_API_CONFIG, else configs/api.yaml)",
        )

    p = sub.add_parser("serve", help="start the API and the Operator; Ctrl+C stops both")
    api_cfg(p)
    p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    p.add_argument("--no-operator", action="store_true", help="API and mock only")
    p.add_argument("--port", type=int, default=None, help="API port (default 8000)")
    p.add_argument("--operator-port", type=int, default=None, help="Streamlit port (8501)")
    p.add_argument(
        "--allow-missing", action="store_true", help="start despite a missing prerequisite"
    )
    p.add_argument("--fixtures", action="store_true", help="serve FAKE api.fixtures data")
    p.add_argument(
        "--exit-after",
        type=float,
        default=None,
        metavar="S",
        help="stop after S seconds (smoke tests)",
    )

    p = sub.add_parser("check", help="walk every step against a running API")
    p.add_argument("--url", default=None, help="API base URL (default http://127.0.0.1:8000)")
    p.add_argument("--operator-url", default=None, help="Streamlit Operator URL to check")
    p.add_argument("--no-browser", action="store_true", help="API calls only, no DOM checks")
    p.add_argument("--strict", action="store_true", help="PENDING counts as a failure")
    p.add_argument("--out", type=Path, default=None, help="report folder")

    p = sub.add_parser("rehearsal", help="headless rehearsal; --record writes rehearsal.json")
    api_cfg(p)
    p.add_argument("--fixtures", action="store_true", help="FAKE api.fixtures data (CI)")
    p.add_argument("--no-browser", action="store_true", help="skip the DOM checks")
    p.add_argument(
        "--record",
        action="store_true",
        help="write docs/demo/rehearsal.json (A-08; needs the browser, refused with --fixtures)",
    )
    p.add_argument(
        "--fresh-clone",
        action="store_true",
        help="internal: set by `demo readiness` in its clone (refused with --record; A-09 is "
        "recorded by `demo readiness` only)",
    )
    p.add_argument("--strict", action="store_true", help="PENDING counts as a failure")
    p.add_argument("--out", type=Path, default=None, help="report folder")

    p = sub.add_parser("clean", help="remove demo leftovers")
    p.add_argument("--keep", type=int, default=3, help="newest report folders kept (3)")
    p.add_argument("--dry-run", action="store_true", help="list, remove nothing")

    p = sub.add_parser("pack", help="write the demo pack")
    p.add_argument("--out", type=Path, default=DEMO_PACK, help=f"default {DEMO_PACK.as_posix()}")
    p.add_argument("--allow-missing", action="store_true", help="skip missing demo runs")

    p = sub.add_parser("bootstrap", help="restore the demo pack (or rebuild the demo runs)")
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--from-pack",
        type=Path,
        default=DEMO_PACK,
        metavar="ZIP",
        help=f"pack to restore (default {DEMO_PACK.as_posix()})",
    )
    g.add_argument(
        "--demo-only",
        action="store_true",
        help="rebuild the missing demo runs at seed 4000 by simulation",
    )

    p = sub.add_parser("readiness", help="fresh clone + bootstrap + rehearsal (A-09)")
    p.add_argument("--no-browser", action="store_true", help="skip the DOM checks")
    p.add_argument(
        "--no-record",
        action="store_true",
        help="do not write rehearsal.json (required with --no-browser)",
    )
    p.add_argument("--keep", action="store_true", help="keep the temporary clone")
    p.add_argument("--pack", type=Path, default=None, help="pack when none is committed")
    return parser


def _summary(res: dict, strict: bool) -> int:
    s = res["summary"]
    print(
        f"\n{res['kind']}: {res['status']}: {s['pass']} PASS, {s['fail']} FAIL, "
        f"{s['pending']} PENDING, {s['skip']} SKIP in {res['duration_s']} s"
    )
    for r in res["rows"]:
        if r["result"] in ("FAIL", "PENDING"):
            print(f"  [{r['result']}] {r['group']}: {r['check']} {r['detail']}")
    if res.get("out_dir"):
        print(f"report: {res['out_dir']}")
    if res.get("recorded"):
        print(f"recorded: {res['recorded']}")
    if not res["passed"]:
        return 1
    return 1 if strict and res["status"] == "PENDING" else 0


def main(argv: list[str] | None = None) -> int:
    from traffic_twin.api.settings import find_repo_root
    from traffic_twin.config import ConfigError
    from traffic_twin.demo.common import DemoError
    from traffic_twin.demo.config import api_base, load_demo_config

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    root = find_repo_root()
    try:
        cfg = load_demo_config(args.config, root=root)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    try:
        if args.command == "serve":
            from traffic_twin.demo.serve import serve

            return serve(
                cfg,
                open_browser=not args.no_browser,
                root=root,
                api_config=args.api_config,
                api_port=args.port,
                operator_port=args.operator_port,
                operator=not args.no_operator,
                allow_missing=args.allow_missing,
                fixtures=args.fixtures,
                exit_after=args.exit_after,
            )
        if args.command == "check":
            from traffic_twin.demo.check import check
            from traffic_twin.demo.common import check_dir, write_report

            out = check_dir(root, "check", args.out)
            res = check(
                cfg,
                args.url or api_base(cfg),
                browser=not args.no_browser,
                operator_url=args.operator_url,
                out_dir=out,
                root=root,
            )
            from traffic_twin.demo.common import Report

            rep = Report(quiet=True)
            rep.rows = res["rows"]
            write_report(out, res, rep.markdown("Demo check", [f"- API: {res['base_url']}"]))
            res["out_dir"] = str(out)
            return _summary(res, args.strict)
        if args.command == "rehearsal":
            from traffic_twin.demo.rehearsal import rehearsal

            res = rehearsal(
                cfg,
                fixtures=args.fixtures,
                browser=not args.no_browser,
                record=args.record,
                fresh_clone=args.fresh_clone,
                root=root,
                api_config=args.api_config,
                out_dir=args.out,
            )
            return _summary(res, args.strict)
        if args.command == "clean":
            from traffic_twin.demo.clean import clean

            removed = clean(root, keep=args.keep, dry_run=args.dry_run)
            verb = "would remove" if args.dry_run else "removed"
            print(f"{verb} {len(removed)} path(s)")
            for p in removed:
                print(f"  {p}")
            return 0
        if args.command == "pack":
            from traffic_twin.demo.pack import pack, read_manifest

            out = pack(cfg, args.out, root=root, allow_missing=args.allow_missing)
            m = read_manifest(out)
            print(
                f"{out}: {len(m['files'])} files, {m['total_bytes'] / 1e6:.1f} MB raw, "
                f"{out.stat().st_size / 1e6:.1f} MB zip, code {m['code_version']}"
            )
            return 0
        if args.command == "bootstrap":
            from traffic_twin.demo.bootstrap import bootstrap

            done = bootstrap(
                None if args.demo_only else args.from_pack, root, demo_only=args.demo_only
            )
            print(
                f"{'made' if args.demo_only else 'restored'} {len(done)} "
                f"{'run folder(s)' if args.demo_only else 'file(s)'}"
            )
            return 0
        if args.command == "readiness":
            from traffic_twin.demo.readiness import readiness

            res = readiness(
                cfg,
                root=root,
                browser=not args.no_browser,
                record=not args.no_record,
                keep=args.keep,
                pack=args.pack,
            )
            return _summary(res, strict=False)
    except DemoError as e:
        print(f"demo {args.command}: {e}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

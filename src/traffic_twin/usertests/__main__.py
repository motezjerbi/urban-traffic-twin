"""User-test CLI (spec 12.2, A-10; D111, D112).

    python -m traffic_twin.usertests check   [--no-git]                 # usertest-check
    python -m traffic_twin.usertests kit     [--out PDF] [--allow-pending-s3]   # usertest-kit
    python -m traffic_twin.usertests links   [--participant N] [--base URL]
    python -m traffic_twin.usertests serve   [--port 8000] [--allow-dirty]      # usertest-serve
    python -m traffic_twin.usertests docs    [--check] [--doc MD] [--json JSON] # usertest-docs

`check` validates the protocol and the results (privacy guard, scoring rules, fix loop
against git). `kit` writes the printable kit; it refuses when the frozen run is missing,
when an expected answer no longer holds, or when S3 is missing from /kpi. `serve` starts
the API with the mock app on 127.0.0.1 for a session, after checking that the tree is
clean and that the run is still the one the kit recorded. `docs` writes
docs/usertests.md and docs/usertests/synthesis.json (`--check`: exit 1 when stale).
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

from traffic_twin.config import ConfigError
from traffic_twin.paths import (
    USERTESTS_CONFIG,
    USERTESTS_DOC,
    USERTESTS_KIT,
    USERTESTS_RESULTS,
    USERTESTS_SYNTHESIS,
)

COMMANDS = ("check", "kit", "links", "serve", "docs")
HOST, PORT = "127.0.0.1", 8000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m traffic_twin.usertests",
                                     description=__doc__.split("\n\n")[0])  # fmt: skip
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", type=Path, default=USERTESTS_CONFIG)
    parser.add_argument("--results", type=Path, default=USERTESTS_RESULTS)
    parser.add_argument("--no-git", action="store_true", help="check: skip the git checks")
    parser.add_argument("--out", type=Path, default=USERTESTS_KIT, help="kit: the PDF")
    parser.add_argument(
        "--allow-pending-s3",
        action="store_true",
        help="kit: build a DRAFT kit while S3 is missing from /kpi (gate)",
    )
    parser.add_argument("--participant", type=int, default=None, help="links: in P<n> order")
    parser.add_argument("--base", default=None, help="links: app URL (default app_url)")
    parser.add_argument("--host", default=HOST, help="serve: bind address")
    parser.add_argument("--port", type=int, default=PORT, help="serve: port")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="serve: allow a modified tree (rehearsals only, never a session)",
    )
    parser.add_argument(
        "--kit",
        type=Path,
        default=USERTESTS_KIT,
        help="serve: the kit whose run identity must match",
    )
    parser.add_argument("--doc", type=Path, default=USERTESTS_DOC, help="docs: the page")
    parser.add_argument("--json", type=Path, default=USERTESTS_SYNTHESIS, help="docs: the JSON")
    parser.add_argument(
        "--check",
        dest="stale_check",
        action="store_true",
        help="docs: exit 1 when the page or the JSON is stale",
    )
    return parser


REPO_ROOT = Path(__file__).resolve().parents[3]  # src/traffic_twin/usertests/__main__.py


def _is_real(path: Path, real: Path) -> bool:
    """`path` is the committed `real` file (paths resolved, from the working directory
    and from the repo root, so any relative or absolute spelling of it matches)."""
    p = Path(path).resolve()
    return any(p == base.joinpath(real).resolve() for base in (Path.cwd(), REPO_ROOT))


def _refuse_synthetic(results: dict, path: Path, outputs: tuple[Path, ...] = ()) -> None:
    """Synthetic test data never goes into the real results file nor into the committed
    docs/usertests.md and synthesis.json (the report reads them for A-10)."""
    if not results.get("synthetic"):
        return
    if _is_real(path, USERTESTS_RESULTS):
        raise ConfigError(f"{path}: synthetic test data in the real results file")
    for out, real in zip(outputs, (USERTESTS_DOC, USERTESTS_SYNTHESIS), strict=False):
        if _is_real(out, real):
            raise ConfigError(f"{path}: synthetic test data would overwrite {real.as_posix()} "
                              "(pass --doc and --json outside docs/)")  # fmt: skip


def cmd_check(args, protocol: dict) -> int:
    from traffic_twin.usertests.protocol import complete_sessions, load_results, verify_git

    results = load_results(args.results, protocol)
    _refuse_synthetic(results, args.results)
    problems = [] if args.no_git else verify_git(results)
    if problems:
        print("user-test results: fix-loop problem(s):", *problems, sep="\n  - ",
              file=sys.stderr)  # fmt: skip
        return 1
    n = len(complete_sessions(results, protocol))
    issues = results.get("issues") or []
    print(f"protocol {args.config.as_posix()} (registered {protocol['preregistered']}, "
          f"{len(protocol['tasks'])} tasks): ok")  # fmt: skip
    print(f"results {args.results.as_posix()}: schema and privacy ok; "
          f"{len(results.get('sessions') or [])} session(s), {n} complete, "
          f"{len(issues)} issue(s)")  # fmt: skip
    from traffic_twin.usertests.synthesis import a10_status

    lo = protocol["participants"]["min"]
    print(f"A-10: {a10_status(results, n, lo)} ({n} of {lo}-"
          f"{protocol['participants']['max']} complete sessions)")  # fmt: skip
    return 0


def cmd_kit(args, protocol: dict) -> int:
    from traffic_twin.usertests.answers import AnswerError
    from traffic_twin.usertests.kit import KitError, build_kit, page_count

    try:
        out = build_kit(protocol, args.out, allow_pending_s3=args.allow_pending_s3)
    except (KitError, AnswerError) as e:
        print(f"user-test kit: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out.as_posix()} ({page_count(out)} pages)")
    return 0


def cmd_links(args, protocol: dict) -> int:
    from traffic_twin.usertests.protocol import clock, deep_link, task_order, tasks_by_id

    tasks = tasks_by_id(protocol)
    order = task_order(protocol, args.participant) if args.participant else list(tasks)
    if args.participant:
        print(f"P{args.participant}: {' -> '.join(order)}")
    for tid in order:
        t = tasks[tid]
        when = (t.get("start") or {}).get("t")
        print(f"{tid:<9} {t['view']:<10} {clock(when) if when is not None else '':<8}")
        print(f"  start {deep_link(protocol, t, 'start', args.base)}")
        print(f"  check {deep_link(protocol, t, 'check', args.base)}")
    return 0


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def serve_problems(protocol: dict, *, settings, kit: Path, allow_dirty: bool, version: str,
                   answers: dict | None, host: str, port: int) -> list[str]:  # fmt: skip
    """Why a session must not start (empty: ready). D111: clean tree, fixed clock, the
    mock app on, the run the kit recorded, a free local port."""
    from traffic_twin.usertests.kit import kit_identity

    problems = []
    if version.endswith("-dirty") and not allow_dirty:
        problems.append(f"the tree is modified ({version}): commit first, so every session "
                        "runs one app version (--allow-dirty for a rehearsal)")  # fmt: skip
    if version == "unknown":
        problems.append("no git repository: the app version cannot be recorded")
    if settings.clock_mode != "fixed":
        problems.append(f"api settings: clock.mode is {settings.clock_mode!r}, the sessions "
                        "need the fixed clock")  # fmt: skip
    if not settings.mock:
        problems.append("api settings: dev.mock is off, the app is not served")
    if host not in ("127.0.0.1", "localhost"):
        problems.append(f"bind address {host}: the sessions run on 127.0.0.1 only")
    identity = kit_identity(kit)
    if identity is None:
        problems.append(f"{Path(kit).as_posix()} absent or without a run identity: build the "
                        "kit first (`tasks.py usertest-kit`)")  # fmt: skip
    elif identity.get("draft") != "no" and not allow_dirty:
        problems.append(
            f"{Path(kit).as_posix()} is a DRAFT kit (S3 pending): rebuild it with "
            "`tasks.py usertest-kit` after Part 8 (--allow-dirty for a rehearsal)"
        )
    if identity is not None and answers is not None:
        run = answers["run"]
        for key in ("run_id", "signal_sha256"):
            if identity.get(key) != str(run.get(key)):
                problems.append(f"{key} changed since the kit was built "
                                f"({identity.get(key)} -> {run.get(key)}): the run is frozen "
                                "for the campaign")  # fmt: skip
    if answers is not None and answers["problems"]:
        problems += [f"expected answer: {p}" for p in answers["problems"]]
    if port_in_use(host, port):
        problems.append(f"{host}:{port} is already in use (stop the other server)")
    return problems


def cmd_serve(args, protocol: dict) -> int:
    import uvicorn

    from traffic_twin.api.main import create_app
    from traffic_twin.api.settings import load_settings
    from traffic_twin.sim.results import code_version
    from traffic_twin.usertests.answers import AnswerError, api_client, expected_answers
    from traffic_twin.usertests.protocol import deep_link

    settings = load_settings()
    version = code_version()
    try:
        answers = expected_answers(protocol, api_client(settings))
    except AnswerError as e:
        print(f"user-test serve: {e}", file=sys.stderr)
        return 1
    problems = serve_problems(protocol, settings=settings, kit=args.kit,
                              allow_dirty=args.allow_dirty, version=version, answers=answers,
                              host=args.host, port=args.port)  # fmt: skip
    if problems:
        print("user-test serve refused:", *problems, sep="\n  - ", file=sys.stderr)
        return 1
    base = f"http://{args.host}:{args.port}/mock"
    print(f"app version {version} · run {answers['run']['scenario']} seed "
          f"{answers['run']['seed']} ({answers['run']['run_id']})")  # fmt: skip
    for t in protocol["tasks"]:
        print(f"{t['id']:<9} {deep_link(protocol, t, 'start', base)}")
    print("Ctrl+C to stop.")
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_docs(args, protocol: dict) -> int:
    from traffic_twin.usertests.protocol import load_results
    from traffic_twin.usertests.synthesis import render, sources, synthesis, write

    results = load_results(args.results, protocol)
    _refuse_synthetic(results, args.results, (args.doc, args.json))
    synth = synthesis(results, protocol, sources=sources(args.config, args.results))
    if args.stale_check:
        import json

        want = {args.doc: render(synth),
                args.json: json.dumps(synth, indent=2, ensure_ascii=False) + "\n"}  # fmt: skip
        stale = [p.as_posix() for p, text in want.items()
                 if not p.is_file() or p.read_text(encoding="utf-8") != text]  # fmt: skip
        if stale:
            print(f"stale: {', '.join(stale)} (run `tasks.py usertest-docs`)", file=sys.stderr)
            return 1
        print(f"{args.doc.as_posix()} and {args.json.as_posix()} are up to date")
        return 0
    for p in write(synth, args.doc, args.json):
        print(f"wrote {p.as_posix()}")
    print(f"A-10: {synth['status']} ({synth['a10']['text']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from traffic_twin.usertests.protocol import load_protocol

    try:
        protocol = load_protocol(args.config)
        handler = {"check": cmd_check, "kit": cmd_kit, "links": cmd_links,
                   "serve": cmd_serve, "docs": cmd_docs}[args.command]  # fmt: skip
        return handler(args, protocol)
    except ConfigError as e:
        print(f"user tests: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

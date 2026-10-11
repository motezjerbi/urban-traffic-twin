#!/usr/bin/env python
"""Cross-platform task runner: the Makefile targets without `make` (stdlib only).

Run it with the project's interpreter, which every task reuses (sys.executable):

    .venv\\Scripts\\python.exe tasks.py sim --seeds 1 2 3      (Windows)
    .venv/bin/python tasks.py baseline                       (Linux / macOS)
    python tasks.py check                                    (lint + tests)

Several tasks run in order (`tasks.py net sim plots`); the run stops at the first
failing command and exits with its code. `--dry-run` only prints the commands.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

Command = list[str]


def _m(module: str, *args: str) -> Command:
    return [PY, "-m", module, *args]


def _run_dirs(results: Path) -> list[str]:
    """Finished runs (directories holding a run.json), as `make db-runs` globs them."""
    root = ROOT / results
    dirs = sorted(p for p in root.glob("*/") if (p / "run.json").is_file())
    if not dirs:
        raise SystemExit(f"no finished runs under {results}/ (run `tasks.py sim` first)")
    return [p.relative_to(ROOT).as_posix() for p in dirs]


# Each task returns the commands to run, given the parsed options
TASKS: dict[str, tuple[str, Callable[[argparse.Namespace], list[Command]]]] = {
    "install": (
        "pip install -e .[dev,db,sim] into the current interpreter",
        lambda o: [[PY, "-m", "pip", "install", "-e", ".[dev,db,sim]"]],
    ),
    "data": (
        "Part 1: download Paris sensors near the candidate zones and clean them",
        lambda o: [
            _m("traffic_twin.data.download", "--config", o.config),
            _m("traffic_twin.data.clean", "--config", o.config),
        ],
    ),
    "explore": (
        "plot hourly sensor profiles",
        lambda o: [_m("traffic_twin.data.explore", "--config", o.config)],
    ),
    "score": (
        "rank candidate intersections (spec 2.2)",
        lambda o: [_m("traffic_twin.data.intersection_score", "configs/intersections.yaml")],
    ),
    "db-load": (
        "push cleaned sensor rows to Supabase (needs SUPABASE_DB_URL)",
        lambda o: [_m("traffic_twin.db.load_sensors", "--config", o.config)],
    ),
    "net": (
        "build the SUMO network (rewrites sumo/network/intersection.net.xml)",
        lambda o: [_m("traffic_twin.sim.network")],
    ),
    "demand": (
        "write the SUMO demand files for --scenario",
        lambda o: [_m("traffic_twin.sim.demand", "--config", o.scenario)],
    ),
    "sim": (
        "run --scenario headless for --seeds",
        lambda o: [_m("traffic_twin.sim.run_scenario", "--config", o.scenario, "--seed", *o.seeds)],
    ),
    "sim-gui": (
        "run --scenario in sumo-gui (first seed)",
        lambda o: [
            _m(
                "traffic_twin.sim.run_scenario",
                "--config",
                o.scenario,
                "--seed",
                o.seeds[0],
                "--gui",
            )
        ],  # fmt: skip
    ),
    "db-runs": (
        "load finished runs from data/results into Supabase",
        lambda o: [_m("traffic_twin.db.load_run", *_run_dirs(Path("data/results")))],
    ),
    "plots": (
        "write KPI figures from data/results into docs/figures",
        lambda o: [_m("traffic_twin.viz.plots")],
    ),
    "lint": (
        "ruff check and format check",
        lambda o: [_m("ruff", "check", "."), _m("ruff", "format", "--check", ".")],
    ),
    "format": (
        "apply ruff formatting and safe lint fixes",
        lambda o: [_m("ruff", "format", "."), _m("ruff", "check", "--fix", ".")],
    ),
    "test": ("run the test suite", lambda o: [_m("pytest", "-q")]),
    "test-fast": (
        "run the tests that do not need SUMO",
        lambda o: [_m("pytest", "-q", "-m", "not sumo")],
    ),
}
# Composite tasks
TASKS["baseline"] = (
    "net + sim (all --seeds) + plots: rebuild the fixed-time baseline",
    lambda o: [c for t in ("net", "sim", "plots") for c in TASKS[t][1](o)],
)
TASKS["check"] = (
    "lint + test (what CI runs)",
    lambda o: [c for t in ("lint", "test") for c in TASKS[t][1](o)],
)


def format_command(cmd: Command) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def run(commands: list[Command], dry_run: bool = False) -> int:
    """Run commands in order from the repo root; return the first non-zero exit code."""
    for cmd in commands:
        print(f"+ {format_command(cmd)}", flush=True)
        if dry_run:
            continue
        try:
            code = subprocess.call(cmd, cwd=ROOT)
        except KeyboardInterrupt:
            return 130
        if code != 0:
            print(f"tasks.py: command failed with exit code {code}", file=sys.stderr)
            return code
    return 0


def parser() -> argparse.ArgumentParser:
    width = max(map(len, TASKS))
    task_help = "\n".join(f"  {name:<{width}}  {desc}" for name, (desc, _) in TASKS.items())
    p = argparse.ArgumentParser(
        prog="tasks.py",
        description=__doc__.split("\n\n")[0],
        epilog="tasks:\n" + task_help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("tasks", nargs="+", choices=list(TASKS), metavar="TASK", help="see below")
    p.add_argument(
        "--scenario", default="configs/scenario_base.yaml", help="scenario config (SCENARIO)"
    )
    p.add_argument("--seeds", nargs="+", default=["1", "2", "3"], help="seeds (SEEDS)")
    p.add_argument("--config", default="configs/data.yaml", help="data config (CONFIG)")
    p.add_argument("-n", "--dry-run", action="store_true", help="print commands, run nothing")
    return p


def build(args: argparse.Namespace) -> list[Command]:
    for s in args.seeds:
        if not s.isdigit():
            raise SystemExit(f"tasks.py: seed must be a non-negative integer, got {s!r}")
    return [cmd for t in args.tasks for cmd in TASKS[t][1](args)]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return run(build(args), dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

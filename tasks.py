#!/usr/bin/env python
"""Cross-platform task runner: the Makefile targets without `make` (stdlib only).

Run it with the project's interpreter, which every task reuses (sys.executable):

    .venv\\Scripts\\python.exe tasks.py sim --seeds 1 2 3      (Windows)
    .venv/bin/python tasks.py baseline                       (Linux / macOS)
    python tasks.py check                                    (lint + tests)
    python tasks.py part3                                    (Pitié: checks, S0 vs S1, validation)

Several tasks run in order (`tasks.py net sim plots`); the run stops at the first
failing command and exits with its code. `--dry-run` only prints the commands.

`--scenario` defaults to the grid scenario for the Part 2 tasks and to the Pitié S0
config for the Part 3 ones (S1 for `compare`). Inside `part3`, S0 is `--baseline` and
S1 is `--scenario`.
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

GRID_SCENARIO = "configs/scenario_base.yaml"
PITIE_S0 = "configs/pitie_fixed.yaml"  # S0: fixed time on the Pitié junction
PITIE_S1 = "configs/pitie_actuated.yaml"  # S1: actuated, same demand
DEFAULT_SEEDS = ["1", "2", "3"]
VALIDATION_DOC = "docs/validation.md"
PITIE_FIGURES = "docs/figures/pitie"

Command = list[str]


def _m(module: str, *args: str) -> Command:
    return [PY, "-m", module, *args]


def _grid(o: argparse.Namespace) -> str:
    """Scenario of the Part 2 tasks (the grid unless --scenario is given)."""
    return o.scenario or GRID_SCENARIO


def _s0(o: argparse.Namespace) -> str:
    """Scenario of the Part 3 tasks run on one config (Pitié S0 unless --scenario)."""
    return o.scenario or PITIE_S0


def _seeds(o: argparse.Namespace) -> list[str]:
    return o.seeds or DEFAULT_SEEDS


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
        "build the SUMO network of --scenario (the grid build rewrites intersection.net.xml)",
        lambda o: [_m("traffic_twin.sim.network", "--config", _grid(o))],
    ),
    "demand": (
        "write the SUMO demand files for --scenario",
        lambda o: [_m("traffic_twin.sim.demand", "--config", _grid(o))],
    ),
    "sim": (
        "run --scenario headless for --seeds (default 1 2 3)",
        lambda o: [_m("traffic_twin.sim.run_scenario", "--config", _grid(o), "--seed", *_seeds(o))],
    ),
    "sim-gui": (
        "run --scenario in sumo-gui (first seed)",
        lambda o: [
            _m(
                "traffic_twin.sim.run_scenario",
                "--config",
                _grid(o),
                "--seed",
                _seeds(o)[0],
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
    # -- Part 3: Pitié-Salpêtrière junction --------------------------------------------
    "osm": (
        "Pitie: OSM extract -> curated plain XML under sumo/network/pitie",
        lambda o: [_m("traffic_twin.sim.osm_import", "--config", _s0(o))],
    ),
    "tls-check": (
        "Pitie: check the signal program (foe greens, intergreens, crossing red)",
        lambda o: [_m("traffic_twin.sim.tls_check", "--config", _s0(o))],
    ),
    "calib-data": (
        "Pitie: reference CSV + demand calibration file from data/raw (--config)",
        lambda o: [_m("traffic_twin.data.calibrate", "--config", o.config, "--from-raw")],
    ),
    "field-check": (
        "Pitie: check the field measurement file",
        lambda o: [_m("traffic_twin.validation.field", "--check", "--scenario", _s0(o))],
    ),
    "calibrate": (
        "Pitie: simulation-in-the-loop demand scaling (--seeds, else calibration seeds)",
        lambda o: [
            _m(
                "traffic_twin.validation.calibration_loop",
                "--scenario",
                _s0(o),
                *(["--seeds", *o.seeds] if o.seeds else []),
            )
        ],
    ),
    "compare": (
        "Pitie: paired --baseline (S0) vs --scenario (S1) on the evaluation seeds",
        lambda o: [
            _m(
                "traffic_twin.evaluation.compare",
                "--baseline-config",
                o.baseline or PITIE_S0,
                "--scenario-config",
                o.scenario or PITIE_S1,
                "--run-missing",
            )
        ],
    ),
    "validate": (
        f"Pitie: sim vs reality tables and {VALIDATION_DOC} (runs missing seeds)",
        lambda o: [
            _m(
                "traffic_twin.validation.report",
                "--scenario",
                _s0(o),
                "--run-missing",
                "--docs",
                VALIDATION_DOC,
            )
        ],
    ),
    "plots-pitie": (
        f"Pitie: KPI figures of pitie_fixed vs pitie_actuated into {PITIE_FIGURES}",
        lambda o: [
            _m(
                "traffic_twin.viz.plots",
                "--scenario",
                "pitie_fixed",
                "pitie_actuated",
                "--out",
                PITIE_FIGURES,
            )
        ],
    ),
    # -- quality -------------------------------------------------------------------------
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
PART3_STEPS = ("tls-check", "field-check", "calibrate", "compare", "validate", "plots-pitie")


def _part3(o: argparse.Namespace) -> list[Command]:
    """The Part 3 chain: every step on S0 (--baseline) except `compare`, which pits S1
    (--scenario) against it."""
    s0 = o.baseline or PITIE_S0
    s1 = o.scenario or PITIE_S1
    on_s0 = argparse.Namespace(**{**vars(o), "scenario": s0})
    on_s1 = argparse.Namespace(**{**vars(o), "scenario": s1, "baseline": s0})
    return [c for t in PART3_STEPS for c in TASKS[t][1](on_s1 if t == "compare" else on_s0)]


TASKS["baseline"] = (
    "net + sim (all --seeds) + plots: rebuild the fixed-time baseline",
    lambda o: [c for t in ("net", "sim", "plots") for c in TASKS[t][1](o)],
)
TASKS["part3"] = (
    f"Pitie chain, S0 = --baseline, S1 = --scenario: {' '.join(PART3_STEPS)}",
    _part3,
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
        "--scenario",
        help=f"scenario config (SCENARIO); default {GRID_SCENARIO}, or {PITIE_S0} for the "
        f"Pitie tasks; for compare and part3 the S1 config, default {PITIE_S1}",
    )
    p.add_argument(
        "--baseline", help=f"S0 config of compare and part3 (BASELINE); default {PITIE_S0}"
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        help="seeds (SEEDS); default 1 2 3, or for calibrate (CALIB_SEEDS) the calibration "
        "seeds of configs/validation.yaml",
    )
    p.add_argument("--config", default="configs/data.yaml", help="data config (CONFIG)")
    p.add_argument("-n", "--dry-run", action="store_true", help="print commands, run nothing")
    return p


def build(args: argparse.Namespace) -> list[Command]:
    for s in args.seeds or ():
        if not s.isdigit():
            raise SystemExit(f"tasks.py: seed must be a non-negative integer, got {s!r}")
    return [cmd for t in args.tasks for cmd in TASKS[t][1](args)]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return run(build(args), dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

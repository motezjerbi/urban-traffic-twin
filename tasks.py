#!/usr/bin/env python
"""Cross-platform task runner: the Makefile targets without `make` (stdlib only).

Run it with the project's interpreter, which every task reuses (sys.executable):

    .venv\\Scripts\\python.exe tasks.py sim --seeds 1 2 3      (Windows)
    .venv/bin/python tasks.py baseline                       (Linux / macOS)
    python tasks.py check                                    (lint + tests)
    python tasks.py part3                                    (Pitié: checks, S0 vs S1, validation)
    python tasks.py part4                                    (Pitié: S0 vs S1 vs S2 evaluation)
    python tasks.py provenance                               (are the published runs committed?)

Several tasks run in order (`tasks.py net sim plots`); the run stops at the first
failing command and exits with its code. `--dry-run` only prints the commands.

`--scenario` defaults to the grid scenario for the Part 2 tasks and to the Pitié S0
config for the Part 3 ones (S1 for `compare`). Inside `part3`, S0 is `--baseline` and
S1 is `--scenario`. Inside `part4`, `--scenario` is the S2 config checked by `tls-check`;
`evaluate` and `plots-eval` read their scenarios from configs/evaluation.yaml
(pre-registered, so it has no command-line override).

`provenance` and `rebaseline` tie the published Pitié results to a commit. A run records
the git version it was made from (`<sha>-dirty` from an uncommitted tree, which no commit
reproduces). After a commit, `rebaseline` moves the evaluation runs aside, re-makes them
from the committed tree (`compare validate part4`, same configs and seeds: a
reproducibility check, not a retune) and fails unless every run's KPI rows and run
summary are identical to the moved ones; the regenerated pages then name the commit.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

GRID_SCENARIO = "configs/scenario_base.yaml"
PITIE_S0 = "configs/pitie_fixed.yaml"  # S0: fixed time on the Pitié junction
PITIE_S1 = "configs/pitie_actuated.yaml"  # S1: actuated, same demand
PITIE_S2 = "configs/pitie_rules.yaml"  # S2: rule based (Part 4), same demand
EVALUATION_CONFIG = "configs/evaluation.yaml"  # pre-registered S0/S1/S2 evaluation (D33)
DEFAULT_SEEDS = ["1", "2", "3"]
VALIDATION_DOC = "docs/validation.md"
PITIE_FIGURES = "docs/figures/pitie"
EVALUATION_DOC = "docs/evaluation.md"
EVAL_FIGURES = "docs/figures/pitie/eval"
RESULTS = Path("data/results")
EVAL_PROVENANCE = RESULTS / "evaluation" / "provenance.json"  # written by `evaluate`
REBASELINE_DIR = RESULTS / "_rebaseline"  # runs moved aside by `rebaseline`, per commit
PUBLISHED_DOCS = (VALIDATION_DOC, EVALUATION_DOC)  # pages that print the runs' code version
# run.json keys that legitimately differ between two runs of one config and seed
VOLATILE_RUN_KEYS = frozenset({"run_id", "code_version", "started_at", "wall_time_s"})
COMPARED_RUN_FILES = ("kpi_run.csv",)  # must be byte-identical after a rebaseline


class Step:
    """An in-process step of a task (stdlib only), run and printed like a command. A
    plain class: tests load tasks.py without registering it in sys.modules, which a
    dataclass needs."""

    __slots__ = ("name", "fn")

    def __init__(self, name: str, fn: Callable[[], int]) -> None:
        self.name = name
        self.fn = fn

    def __repr__(self) -> str:
        return f"Step({self.name!r})"


Command = list[str] | Step


def _m(module: str, *args: str) -> list[str]:
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


# -- provenance of the published Pitié runs ------------------------------------------------
def git_head() -> tuple[str, bool]:
    """(short HEAD hash, tree has uncommitted changes), computed as
    `sim.results.code_version` does (ignored files such as data/results and files under
    docs/ do not count)."""
    git = ["git", "--no-optional-locks"]
    try:
        head = subprocess.run(
            [*git, "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        status = subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=normal", "--", ".",
             ":(exclude)docs"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"tasks.py: git is not available: {exc}") from exc
    return head, bool(status.strip())


def evaluation_run_dirs() -> list[Path]:
    """Run directories (relative to ROOT) behind the published Part 3 and Part 4 pages:
    every scenario and seed of the last evaluation (its provenance.json). The Part 3 S0
    and S1 runs are the main family's S0 and S1."""
    path = ROOT / EVAL_PROVENANCE
    if not path.is_file():
        raise SystemExit(f"tasks.py: no {EVAL_PROVENANCE.as_posix()} (run `tasks.py evaluate`)")
    prov = json.loads(path.read_text(encoding="utf-8"))
    names = sorted(
        {s["scenario"] for fam in prov["families"].values() for s in fam["scenarios"].values()}
    )
    return [RESULTS / f"{name}_seed{seed}" for name in names for seed in prov["seeds"]]


def _code_version(run_dir: Path) -> str:
    meta = ROOT / run_dir / "run.json"
    if not meta.is_file():
        return "missing"
    return str(json.loads(meta.read_text(encoding="utf-8")).get("code_version", "unknown"))


def check_provenance() -> int:
    """0 when the tree is committed, every evaluation run was made from HEAD and no
    published page names an uncommitted tree; 1 otherwise, with the reasons."""
    head, dirty = git_head()
    runs = evaluation_run_dirs()
    versions = Counter(_code_version(d) for d in runs)
    print(
        f"HEAD {head}{' with uncommitted changes' if dirty else ''}; {len(runs)} runs:", flush=True
    )
    for version, n in sorted(versions.items()):
        print(f"  code version {version}: {n} runs", flush=True)
    problems = []
    if dirty:
        problems.append("the tree has uncommitted changes: commit, then `tasks.py rebaseline`")
    if set(versions) != {head}:
        problems.append(f"not every run was made from {head}: run `tasks.py rebaseline`")
    for doc in PUBLISHED_DOCS:
        page = ROOT / doc
        if page.is_file() and "-dirty" in page.read_text(encoding="utf-8"):
            problems.append(f"{doc} names an uncommitted tree (`-dirty`)")
    for problem in problems:
        print(f"provenance: {problem}", file=sys.stderr)
    return 1 if problems else 0


def rebaseline_backup() -> int:
    """Move the evaluation runs not made from the clean HEAD to REBASELINE_DIR/<HEAD>/, so
    that the `--run-missing` steps that follow re-make them from the committed tree."""
    head, dirty = git_head()
    if dirty:
        print("rebaseline: the tree has uncommitted changes; commit first", file=sys.stderr)
        return 1
    backup = ROOT / REBASELINE_DIR / head
    if backup.is_dir() and any(backup.iterdir()):
        print(
            f"rebaseline: {(REBASELINE_DIR / head).as_posix()} already holds runs; "
            "inspect and remove it first",
            file=sys.stderr,
        )
        return 1
    moved = 0
    for run_dir in evaluation_run_dirs():
        if _code_version(run_dir) in (head, "missing"):
            continue
        backup.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ROOT / run_dir), str(backup / run_dir.name))
        moved += 1
    print(f"rebaseline: moved {moved} runs to {(REBASELINE_DIR / head).as_posix()}")
    return 0


def _run_summary(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return {k: v for k, v in meta.items() if k not in VOLATILE_RUN_KEYS}


def rebaseline_verify() -> int:
    """Every moved run, re-made from HEAD, must have byte-identical KPI rows
    (kpi_run.csv) and the same run summary (run.json without run id, code version and
    timings); then the provenance check runs. The moved runs are kept for inspection."""
    head, _ = git_head()
    backup = ROOT / REBASELINE_DIR / head
    old_runs = sorted(p for p in backup.glob("*/") if (p / "run.json").is_file())
    differing: dict[str, list[str]] = {}
    for old in old_runs:
        new = ROOT / RESULTS / old.name
        if not (new / "run.json").is_file():
            differing.setdefault(old.name, []).append("not re-made")
            continue
        for name in COMPARED_RUN_FILES:
            if (old / name).read_bytes() != (new / name).read_bytes():
                differing.setdefault(old.name, []).append(f"{name} differs")
        if _run_summary(old) != _run_summary(new):
            differing.setdefault(old.name, []).append("run.json differs")
    for name, what in differing.items():
        print(f"rebaseline: {name}: {', '.join(what)}", file=sys.stderr)
    same = len(old_runs) - len(differing)
    print(f"rebaseline: {same} of {len(old_runs)} re-made runs identical to the moved ones")
    return 1 if differing else check_provenance()


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
    # -- Part 4: rule-based control (S2), Safety Guard, per-user evaluation ---------------
    "evaluate": (
        f"Pitie: pre-registered S0/S1/S2 evaluation and {EVALUATION_DOC} (runs missing seeds)",
        lambda o: [
            _m(
                "traffic_twin.evaluation.evaluate",
                "--config",
                EVALUATION_CONFIG,
                "--run-missing",
                "--docs",
                EVALUATION_DOC,
            )
        ],
    ),
    "plots-eval": (
        f"Pitie: evaluation figures (S0/S1/S2, equity, A-04, A-05) into {EVAL_FIGURES}",
        lambda o: [
            _m("traffic_twin.viz.evaluation", "--config", EVALUATION_CONFIG, "--out", EVAL_FIGURES)
        ],
    ),
    "provenance": (
        "Pitie: exit 1 unless the evaluation runs and published pages come from a clean HEAD",
        lambda o: [Step("provenance", check_provenance)],
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


PART4_STEPS = ("tls-check", "evaluate", "plots-eval")


def _part4(o: argparse.Namespace) -> list[Command]:
    """The Part 4 chain: `tls-check` on S2 (--scenario, default the rules config), then the
    evaluation and its figures, whose scenarios come from configs/evaluation.yaml."""
    on_s2 = argparse.Namespace(**{**vars(o), "scenario": o.scenario or PITIE_S2})
    return [c for t in PART4_STEPS for c in TASKS[t][1](on_s2)]


REBASELINE_STEPS = ("compare", "validate", "part4")


def _rebaseline(o: argparse.Namespace) -> list[Command]:
    """Move the runs aside, re-make the published pages with every default (`compare
    validate part4`, which run the missing seeds), then compare the runs."""
    defaults = argparse.Namespace(**{**vars(o), "scenario": None, "baseline": None})
    return [
        Step("rebaseline-backup", rebaseline_backup),
        *(c for t in REBASELINE_STEPS for c in TASKS[t][1](defaults)),
        Step("rebaseline-verify", rebaseline_verify),
    ]


TASKS["baseline"] = (
    "net + sim (all --seeds) + plots: rebuild the fixed-time baseline",
    lambda o: [c for t in ("net", "sim", "plots") for c in TASKS[t][1](o)],
)
TASKS["part3"] = (
    f"Pitie chain, S0 = --baseline, S1 = --scenario: {' '.join(PART3_STEPS)}",
    _part3,
)
TASKS["part4"] = (
    f"Pitie chain, S2 = --scenario: {' '.join(PART4_STEPS)}",
    _part4,
)
TASKS["rebaseline"] = (
    "Pitie, after a commit: re-make the evaluation runs from HEAD (compare validate "
    "part4), require identical KPI rows",
    _rebaseline,
)
TASKS["check"] = (
    "lint + test (what CI runs)",
    lambda o: [c for t in ("lint", "test") for c in TASKS[t][1](o)],
)


def format_command(cmd: Command) -> str:
    if isinstance(cmd, Step):
        return f"tasks.py:{cmd.name} (in-process)"
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
            code = cmd.fn() if isinstance(cmd, Step) else subprocess.call(cmd, cwd=ROOT)
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
        f"Pitie tasks; for compare and part3 the S1 config, default {PITIE_S1}; for part4 "
        f"the S2 config, default {PITIE_S2}",
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

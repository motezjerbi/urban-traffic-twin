#!/usr/bin/env python
"""Cross-platform task runner: the Makefile targets without `make` (stdlib only).

Run it with the project's interpreter, which every task reuses (sys.executable):

    .venv\\Scripts\\python.exe tasks.py sim --seeds 1 2 3      (Windows)
    .venv/bin/python tasks.py baseline                       (Linux / macOS)
    python tasks.py check                                    (lint + tests)
    python tasks.py part3                                    (Pitié: checks, S0 vs S1, validation)
    python tasks.py part4                                    (Pitié: S0 vs S1 vs S2 evaluation)
    python tasks.py provenance                               (are the published runs committed?)
    python tasks.py part5                                    (ETA dataset, model, A-07, API)
    python tasks.py part6                                    (RL timing, trainings, evaluation)
    python tasks.py part8                                    (once: S0-S4 on seeds 1-10)
    python tasks.py demo                                     (API + Operator for the demo)

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

The Part 5 and Part 6 tasks take their seeds, splits and budgets from the pre-registered
configs (configs/eta.yaml, configs/rl.yaml, D38): `--scenario`, `--baseline` and
`--seeds` do not reach them. `--allow-dirty` lets `eta-data` and `eta-data-test` build a
development dataset from an uncommitted tree. `rl-train` runs the S6 trainings listed in
configs/rl.yaml (`train.s6`, read with PyYAML, a core dependency of the package);
`--algo`, `--seed` and `--steps` narrow or override them.

`reproduce56` is the D65 check of Parts 5 and 6 after a commit. Plain `part5` / `part6`
on top of their own outputs stop by design: the ETA freeze refuses a second model
(single-shot test, D50) and `rl.train` keeps a finished run. So the task first moves the
Phase 3 outputs aside to `<dir>/_reproduce/<HEAD>/` (dir = data/eta, data/models,
data/demo, data/rl_runs). It then reruns `part5 part6` from the clean tree with every
default and compares old with new. What must match: the model and reference files
(sha256), the dataset counts, every eval.json metric, the RL policy parameters (sha256),
episode returns and validation costs, the selection, the dev-test tables and the demo and
dev-test KPI rows. Fields that must differ are not compared: dataset id and model version
(a clean tree drops the `-dev` suffix), bundle sha256, code versions, timestamps, wall
times, steps/s and the SB3 zip sha256 (archive timestamps).

The Part 7 tasks (`rl7-*`, `part7`) read everything from the pre-registered
configs/rl7.yaml and configs/rl7/*.yaml (D68-D90) and write under data/rl7/; only
`rl7-campaign` takes options (`--slots`, `--allow-dirty`). `part7` stops after
`rl7-publish` until configs/pitie_rl_s3.yaml exists (written by hand from the lines
`rl7-publish` prints, D86). `reproduce7` is the D82 check after a commit: it reruns
stage-T run t1 seed 0 into `data/rl7/_reproduce/<HEAD>/`, launched as the campaign
launched it (the OMP_/MKL_/OPENBLAS_ variables and validation pool its provenance
records), and requires the same policies,
validation rows and episode returns, the same S3 golden and the same docs/rl_training.md
(apart from its code-version lines); the rest of the campaign is not rerun.

Parts 8-10 (D91-D120). `part8` is the single seed 1-10 run of S3 and S4, on the clean
registration commit (the first commit adding `a06:` to configs/evaluation.yaml, D92):
`part8-guard` refuses a dirty tree, a HEAD without the `a06` block, a HEAD part8 already
ran from (`data/results/_rebaseline/<HEAD>/`) and existing runs of the new scenarios on
the evaluation seeds; then `rebaseline` re-makes the 70 S0-S2 runs (identical, recorded in
`_rebaseline/<HEAD>/verify.json`) while `evaluate --run-missing` makes the 40 S3 / S4 runs,
then `plots-eval` and `provenance`. On a STOP (A-04 FAIL) it is resumed with
`tasks.py plots-eval rebaseline-verify`. `sim` and `sim-gui` check the same run lock
in-process before they launch a registered evaluation scenario on an evaluation seed;
`compare` and `validate` (so `part3` and `rebaseline`) check it for every run their
`--run-missing` would launch (compare.py and validation/report.py do not). Those frozen
modules and sim/run_scenario.py, called directly, launch runs WITHOUT the lock: never
call them on seeds 1-10 (one early S3 look is disclosed in configs/evaluation.yaml). The
Part 9 and Part 10 tasks call `python -m traffic_twin.usertests`, `traffic_twin.report`
and `traffic_twin.demo`; `--fixtures`, `--no-browser`, `--record` and
`--allow-pending-s3` reach only them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
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
VALIDATION_CONFIG = "configs/validation.yaml"  # its evaluation seeds: compare and validate
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
# Must be byte-identical after a rebaseline once each run's own run_id (a uuid4 in every
# KPI row) is masked: two runs of one config and seed never share a run_id
COMPARED_RUN_FILES = ("kpi_run.csv",)
RUN_ID_MASK = b"<run_id>"

# -- Parts 5 and 6 (pre-registered D37-D67) ----------------------------------------------
ETA_CONFIG = "configs/eta.yaml"  # ETA dataset, model and evaluation (D38-D52)
RL_CONFIG = "configs/rl.yaml"  # RL environment and training (D55-D62)
ETA_DOC = "docs/eta.md"
ETA_FIGURES = "docs/figures/eta"
API_DOC = "docs/api.md"
RL_DOC = "docs/rl.md"
RL_FIGURES = "docs/figures/rl"
ETA_FIT_SPLITS = ("train", "val", "calib")  # generated before the freeze
ETA_TEST_SPLIT = "test"  # worlds 2900-2999: generated and evaluated once, after the freeze
DEMO_SEEDS = ("4000",)  # Part 5 demo / replay run (api.yaml seeds.demo 4000-4009)
API_RESULTS = ("data/results", "data/demo")  # read only by the API and its benchmark
API_HOST, API_PORT = "127.0.0.1", "8000"
RL_ALGOS = ("dqn", "ppo")
RL_DEVTEST_SEEDS = tuple(str(s) for s in range(3020, 3030))  # Part 6 share of 3020-3049
# `install`: every extra the tasks and the test suite use. PyTorch comes first from the CPU
# index (a plain install pulls the CUDA toolkit on Linux), at the `rl` extra's pin.
INSTALL_EXTRAS = ("dev", "db", "sim", "ml", "api", "rl", "app", "report")  # D118
TORCH_PIN = "torch==2.14.1"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# -- D65 reproduction of Parts 5 and 6 (`reproduce56`) -------------------------------------
# Output roots of `part5 part6` (configs/eta.yaml and configs/rl.yaml `outputs`)
ETA_DATA_DIR = Path("data/eta")
ETA_MODELS_DIR = Path("data/models/eta")
RL_SELECTED_DIR = Path("data/models/rl/selected")
DEMO_DIR = Path("data/demo")
RL_RUNS_DIR = Path("data/rl_runs")
REPRODUCE_PARENTS = (ETA_DATA_DIR, Path("data/models"), DEMO_DIR, RL_RUNS_DIR)
REPRODUCE_DIR = "_reproduce"  # <parent>/_reproduce/<HEAD>/: the moved-aside originals
# Kept in place: `_`-prefixed entries (work dirs, the step-11 determinism runs, backups),
# dot files and this determinism check, which `part6` does not redo
REPRODUCE_KEEP = frozenset({(RL_RUNS_DIR / "repro.json").as_posix()})
REPRODUCE_STEPS = ("part5", "part6")
# Keys (at any depth) that must differ between an original and its reproduction
ETA_META_VOLATILE = frozenset(
    {"model_version", "dataset_id", "code_version", "dataset_code_version", "created_at", "fit_s"}
)
ETA_EVAL_VOLATILE = frozenset(
    {"dataset_id", "model_version", "bundle_sha256", "code_version", "freeze"}
)
ETA_DATASET_VOLATILE = frozenset(
    {"dataset_id", "code_version", "argv", "created_at", "updated_at", "builds"}
)
RL_MANIFEST_VOLATILE = frozenset(
    {"code_version", "dirty", "started_at", "wall_s", "baseline_s", "eval_s", "train_s",
     "steps_per_s", "argv", "best_model_sha256", "final_model_sha256", "model_bytes"}
)  # fmt: skip
RL_SELECTED_VOLATILE = frozenset({"code_version", "selected_at", "model_sha256"})
RL_SUMMARY_VOLATILE = frozenset({"created_at", "code_version", "dirty", "model_sha256"})
RUN_JSON_VOLATILE = VOLATILE_RUN_KEYS | {"model_sha256"}  # pitie_rl: the zip's sha256
ETA_BUNDLE_FILES = ("model.ubj", "reference.parquet")
DEVTEST_TABLES = ("compare.csv", "kpi.csv", "kpi_safety.csv")
DEVTEST_SUMMARIES = ("compare_summary.json", "kpi_summary.json")
REPRODUCED_DOCS = (ETA_DOC, API_DOC, RL_DOC)  # rewritten by the reproduction
DIRTY_VERSION = re.compile(r"\b[0-9a-f]{7,40}-dirty\b")

# -- Part 7 (pre-registered D68-D90, configs/rl7.yaml) -------------------------------------
RL7_CONFIG = "configs/rl7.yaml"  # extends rl.yaml; rl.yaml stays the Part 6 record (D69)
RL7_CONFIG_DIR = "configs/rl7"  # one run config per stage-T config and sensitivity setting
PITIE_RL_S3 = "configs/pitie_rl_s3.yaml"  # S3, written from the `rl7-publish` output (D86)
RL7_DOC = "docs/rl_training.md"  # generated by `rl.report7` (D83)
RL7_FIGURES = "docs/figures/rl7"
RL7_DIR = Path("data/rl7")  # rl7.yaml outputs.study (gitignored, D85)
RL7_RUNS_DIR = RL7_DIR / "runs"  # rl7.yaml outputs.runs: <stage>/<run>/ (campaign Job.out)
RL7_STAGES = ("T", "S", "F")  # campaign stages, run folders under RL7_RUNS_DIR/<stage>/
RL7_CURVE_STAGES = ("F",)  # `rl7-holdout` also scores the kept checkpoints of these runs
RL7_DEVTEST_RANGE = (3020, 3029)  # rl7.yaml protocol.devtest.seeds: looked at once (D79)
RL7_GOLDEN = Path("tests/data/rl_s3_golden.json")  # `rl.study golden`, committed (D86)
# D82: the stage-T run `reproduce7` reruns (run name <algo>_<variant>_s<seed>_<k>k_<sha8>)
REPRODUCE7_RUN = {"config": f"{RL7_CONFIG_DIR}/t1.yaml", "algo": "dqn", "variant": "t1",
                  "seed": 0, "steps": 400000, "stage": "T"}  # fmt: skip
# Keys (at any depth) of a Part 7 manifest / provenance that differ between an original
# run and its reproduction: code version, clean-tree flag, timings, the zip sha256
RL7_RUN_VOLATILE = RL_MANIFEST_VOLATILE | {
    "allow_dirty", "created_at", "finished_at", "config", "pick_model_sha256", "model_sha256",
    "provenance_s", "val_s",
}  # fmt: skip
# The launch of a training, mirrored from rl.train (tasks.py is stdlib only; test_tasks
# checks the copies): the variable prefixes provenance.json records under machine.env
# (ENV_PREFIXES) and what rl.campaign.train_env() adds where unset (CHILD_ENV)
RL7_ENV_PREFIXES = ("OMP_", "MKL_", "OPENBLAS_")
RL7_CHILD_ENV = {"OPENBLAS_NUM_THREADS": "1"}
RL7_EVAL_POOLS = ("persistent", "transient")  # rl.train --eval-pool (EVAL_POOLS)
RL7_VALIDATION_VOLATILE = frozenset({"eval_s", "wall_s"})  # timing columns of validation.csv
RL7_GOLDEN_VOLATILE = frozenset({"created_at", "code_version", "dirty"})
CODE_VERSION_LINE = re.compile(r"code[ -]version", re.IGNORECASE)  # masked in the page

# -- Parts 8-10 (pre-registered D91-D120 in configs/evaluation.yaml) ------------------------
PITIE_S2_NOP = "configs/pitie_rules_nopreempt.yaml"  # S2_noP, the preemption ablation
PITIE_S4 = "configs/pitie_rl_s3_nopreempt.yaml"  # S4: the S3 policy, preemption off (D93)
REGISTRATION_KEY = "a06"  # top-level key whose first commit is the registration (D92)
VERIFY_FILE = "verify.json"  # REBASELINE_DIR/<HEAD>/verify.json: A-01 evidence (D97)
PART8_STEPS = ("part8-guard", "rebaseline", "plots-eval", "provenance")  # D97
# D109: the demo runs Part 9 adds at seed 4000 (S2_noP, S3, S4); `demo-runs` (S0-S2) and
# `reproduce56` are unchanged
DEMO_RUNS8_CONFIGS = (PITIE_S2_NOP, PITIE_RL_S3, PITIE_S4)
OPERATOR_SCRIPT = "app.py"  # the Streamlit Operator (Motez's UI, D102)
OPERATOR_HOST, OPERATOR_PORT = "127.0.0.1", "8501"
DEMO = "traffic_twin.demo"  # D116: serve, check, rehearsal, clean, pack, bootstrap, readiness
USERTESTS = "traffic_twin.usertests"  # D111: check, kit, links, serve, docs
REPORT = "traffic_twin.report"  # D113-D115: facts, report, slides, shots, check
PART9_STEPS = ("demo-runs8", "api-bench", "usertest-check", "usertest-docs")
PART10_STEPS = ("report-facts", "app-shots", "report", "slides", "demo-pack", "demo-rehearsal")


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


def _no_browser(o: argparse.Namespace) -> list[str]:
    return ["--no-browser"] if o.no_browser else []


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


def _run_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _run_summary(run_dir: Path) -> dict:
    return {k: v for k, v in _run_meta(run_dir).items() if k not in VOLATILE_RUN_KEYS}


def _masked(run_dir: Path, name: str) -> bytes:
    """The bytes of a run file with the run's own run_id (from run.json) masked: the KPI
    rows carry it in every line, and a re-made run gets a new one."""
    data = (run_dir / name).read_bytes()
    run_id = str(_run_meta(run_dir).get("run_id") or "")
    return data.replace(run_id.encode("utf-8"), RUN_ID_MASK) if run_id else data


def rebaseline_verify() -> int:
    """Every moved run, re-made from HEAD, must have byte-identical KPI rows
    (kpi_run.csv, each run's own run_id masked) and the same run summary (run.json
    without run id, code version and timings); then the provenance check runs. The moved
    runs are kept for inspection."""
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
            if _masked(old, name) != _masked(new, name):
                differing.setdefault(old.name, []).append(f"{name} differs")
        if _run_summary(old) != _run_summary(new):
            differing.setdefault(old.name, []).append("run.json differs")
    for name, what in differing.items():
        print(f"rebaseline: {name}: {', '.join(what)}", file=sys.stderr)
    same = len(old_runs) - len(differing)
    print(f"rebaseline: {same} of {len(old_runs)} re-made runs identical to the moved ones")
    if backup.is_dir():  # D97: the A-01 evidence, next to the moved runs
        report = {
            "head": head,
            "identical": same,
            "of": len(old_runs),
            "differing": [{"run": n, "what": w} for n, w in sorted(differing.items())],
        }
        (backup / VERIFY_FILE).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"rebaseline: report {(REBASELINE_DIR / head / VERIFY_FILE).as_posix()}")
    else:
        print(f"rebaseline: nothing was moved aside for {head}: no {VERIFY_FILE} written")
    return 1 if differing else check_provenance()


# -- Part 8: the single seed 1-10 run of S3 and S4 (D92, D97) -----------------------------
def _git(*args: str) -> str | None:
    """stdout of `git <args>` in ROOT, or None when git fails (no repo, unknown revision)."""
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None


def _scenario_name(config: str) -> str:
    """The `scenario` key a run folder is named after (`<scenario>_seed<n>`): the config's
    own key, else the file stem (every Pitie config sets it)."""
    import yaml  # core dependency of the package

    path = ROOT / config
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    name = doc.get("scenario") if isinstance(doc, dict) else None
    return str(name) if name else Path(config).stem


def part8_new_runs(ecfg: dict) -> list[str]:
    """Run folders (relative to ROOT) that already exist on the evaluation seeds for a
    scenario the evaluation `ecfg` registers but the last evaluation (its provenance.json)
    did not run: the S3 and S4 runs Part 8 makes once (D97)."""
    prov = _read_json(ROOT / EVAL_PROVENANCE)
    done = (
        {s["scenario"] for fam in prov["families"].values() for s in fam["scenarios"].values()}
        if isinstance(prov, dict)
        else set()
    )
    registered = {
        _scenario_name(path)
        for fam in (ecfg.get("families") or {}).values()
        for path in (fam.get("scenarios") or {}).values()
    }
    seeds = (ecfg.get("seeds") or {}).get("evaluation") or []
    return [
        (RESULTS / f"{name}_seed{seed}").as_posix()
        for name in sorted(registered - done)
        for seed in seeds
        if (ROOT / RESULTS / f"{name}_seed{seed}").exists()
    ]


def part8_guard() -> int:
    """`part8` starts only on the registration (D92, D97): the tree is clean, HEAD's
    configs/evaluation.yaml has the `a06` block, `data/results/_rebaseline/<HEAD>/` does
    not exist (part8 runs once per commit) and no run of a scenario added since the last
    evaluation exists on the evaluation seeds. 0 or 1, with every reason."""
    import yaml  # core dependency of the package

    head, dirty = git_head()
    problems = []
    if dirty:
        problems.append("the tree has uncommitted changes: commit the registration first")
    text = _git("show", f"HEAD:{EVALUATION_CONFIG}")
    ecfg = yaml.safe_load(text) if text else None
    if not isinstance(ecfg, dict) or REGISTRATION_KEY not in ecfg:
        problems.append(
            f"{EVALUATION_CONFIG} at HEAD {head} has no `{REGISTRATION_KEY}` block: the Part 8 "
            "registration is not committed"
        )
        ecfg = {}
    backup = REBASELINE_DIR / head
    if (ROOT / backup).exists():
        problems.append(
            f"{backup.as_posix()} exists: part8 (its rebaseline) already ran from {head}; "
            "inspect it, and resume with `tasks.py plots-eval rebaseline-verify` (D97)"
        )
    if not (ROOT / EVAL_PROVENANCE).is_file():
        problems.append(f"no {EVAL_PROVENANCE.as_posix()}: the Part 4 evaluation is missing")
    early = part8_new_runs(ecfg)
    if early:
        problems.append(
            f"{len(early)} run(s) of the scenarios added since the last evaluation already "
            f"exist on the evaluation seeds ({', '.join(early[:4])}"
            f"{', ...' if len(early) > 4 else ''}): only part8 makes them (D97); STOP and "
            "investigate"
        )
    for problem in problems:
        print(f"part8: {problem}", file=sys.stderr)
    if problems:
        return 1
    first = (_git("log", "--reverse", "--format=%h %cI", "-S", f"{REGISTRATION_KEY}:", "--",
                  EVALUATION_CONFIG) or "").splitlines()  # fmt: skip
    print(f"part8: registration committed ({first[0] if first else head}); HEAD {head}, clean")
    return 0


# -- the D92 run lock for `sim`, `compare` and `validate` ----------------------------------
def sim_run_lock(scenario: str, seeds: list[str]) -> int:
    """The D92 run lock for `sim` and `sim-gui` (sim/run_scenario.py is frozen and does
    not check it). A run of a scenario registered in configs/evaluation.yaml on an
    evaluation seed writes the very folder the evaluation reads (`<scenario>_seed<n>`
    under data/results), so it is refused, with nothing launched, until the registration
    is committed on a clean tree (`evaluation.evaluate.run_lock`). Other scenarios (the
    grid) and other seeds pass. Imports the package, only when the task runs."""
    from traffic_twin.config import ConfigError, load_scenario
    from traffic_twin.evaluation import evaluate

    try:
        cfg = load_scenario(ROOT / scenario)
    except (ConfigError, OSError):
        return 0  # not loadable: run_scenario reports the config error itself
    try:
        ecfg = evaluate.load_evaluation(ROOT / EVALUATION_CONFIG)
        registered = {
            load_scenario(ROOT / path)["scenario"]
            for fam in ecfg["families"].values()
            for path in fam["scenarios"].values()
        }
    except (ConfigError, evaluate.EvaluationError, OSError, KeyError) as exc:
        print(f"run lock: cannot read {EVALUATION_CONFIG}: {exc}; nothing was launched",
              file=sys.stderr)  # fmt: skip
        return 1
    name = cfg["scenario"]
    if name not in registered:
        return 0
    try:
        evaluate.run_lock(ecfg, [(name, int(s)) for s in seeds], ROOT / cfg["outputs"]["dir"])
    except evaluate.RunLockError as exc:
        print(str(exc), file=sys.stderr)
        return evaluate.RUN_LOCK_EXIT
    return 0


def _sim_lock(o: argparse.Namespace, seeds: list[str]) -> Step:
    scenario = _grid(o)
    return Step("run-lock", lambda: sim_run_lock(scenario, seeds))


def _would_run(runs_mod, cfg: dict, seeds: list[int], results: Path) -> list[tuple[str, int]]:
    """(scenario, seed) of every run `runs.run_missing(cfg, seeds, results)` would launch:
    absent seeds and stale runs (its own test, nothing is run). A run whose freshness
    cannot be read (hashes or run.json unreadable) counts as launched."""
    try:
        expected = runs_mod.expected_hashes(cfg)
    except (OSError, ValueError, KeyError):
        expected = None
    out = []
    for seed in seeds:
        d = runs_mod.run_path(results, cfg["scenario"], seed)
        try:
            fresh = (
                expected is not None
                and runs_mod.is_complete(d)
                and runs_mod.stale_reason(runs_mod.read_meta(d), expected) is None
            )
        except (OSError, ValueError, KeyError):
            fresh = False
        if not fresh:
            out.append((cfg["scenario"], int(seed)))
    return out


def missing_run_lock(configs: list[str], results_of: str, every: bool = False) -> int:
    """The D92 run lock for `compare` and `validate` (and so `part3` and `rebaseline`):
    evaluation/compare.py and validation/report.py are frozen, and their `--run-missing`
    calls `runs.run_missing` without the lock. The runs it would launch (the absent or
    stale seeds of each of `configs` on configs/validation.yaml's evaluation seeds, into
    the outputs.dir of `results_of`, the folder both modules write) go through
    `evaluation.evaluate.run_lock`: on an evaluation seed (of configs/evaluation.yaml or
    validation.yaml) into data/results they are refused, exit 2 and nothing launched,
    until the registration is committed on a clean tree. Unlike `sim`, every controller
    is locked, registered or not (D92). Fresh runs launch nothing, so re-making a page
    from them still works on a dirty tree. `every` counts every evaluation seed of each
    config as launched (`rebaseline` moves all runs aside before it re-makes them).
    Imports the package only when the task runs."""
    from traffic_twin.config import ConfigError, load_scenario
    from traffic_twin.evaluation import evaluate, runs
    from traffic_twin.validation import load_validation

    try:
        cfgs = [load_scenario(ROOT / path) for path in configs]
        results = ROOT / load_scenario(ROOT / results_of)["outputs"]["dir"]
    except (ConfigError, OSError, KeyError):
        return 0  # not loadable: the module reports the config error before any run
    try:
        ecfg = evaluate.load_evaluation(ROOT / EVALUATION_CONFIG)
        seeds = [int(s) for s in load_validation(ROOT / VALIDATION_CONFIG)["seeds"]["evaluation"]]
    except (ConfigError, evaluate.EvaluationError, OSError, KeyError, TypeError) as exc:
        print(f"run lock: cannot read the evaluation seeds ({EVALUATION_CONFIG}, "
              f"{VALIDATION_CONFIG}): {exc}; nothing was launched", file=sys.stderr)  # fmt: skip
        return 1
    registered = [int(s) for s in (ecfg.get("seeds") or {}).get("evaluation") or []]
    lock_cfg = {**ecfg, "seeds": {**(ecfg.get("seeds") or {}),
                                  "evaluation": sorted({*registered, *seeds})}}  # fmt: skip
    if every:
        planned = [(cfg["scenario"], seed) for cfg in cfgs for seed in seeds]
    else:
        planned = [run for cfg in cfgs for run in _would_run(runs, cfg, seeds, results)]
    try:
        evaluate.run_lock(lock_cfg, planned, results)
    except evaluate.RunLockError as exc:
        print(str(exc), file=sys.stderr)
        return evaluate.RUN_LOCK_EXIT
    return 0


def _missing_lock(configs: list[str], results_of: str, every: bool = False) -> Step:
    return Step("run-lock", lambda: missing_run_lock(configs, results_of, every))


# -- D65: reproduction of Parts 5 and 6 after a commit ------------------------------------
def json_diff(
    a: object, b: object, ignore: frozenset[str] = frozenset(), path: str = ""
) -> list[str]:
    """Paths where two JSON documents differ, skipping the keys in `ignore` at any depth.
    Values must be equal and of the same type (bitwise reproduction); NaN equals NaN."""
    here = path or "<root>"
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b), key=str):
            if key in ignore:
                continue
            sub = f"{path}.{key}" if path else str(key)
            if key not in a or key not in b:
                out.append(sub)
            else:
                out += json_diff(a[key], b[key], ignore, sub)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [here]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in json_diff(x, y, ignore, f"{path}[{i}]")
        ]
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if type(a) is type(b) and a == b else [here]


def _read_json(path: Path) -> object | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def call(cmd: list[str], env: dict[str, str] | None = None) -> int:
    """Run one command from the repo root, in `env` (None: this process's environment);
    a Step's subprocess, tests replace it."""
    return subprocess.call(cmd, cwd=ROOT, env=env)


def reproduce_backup_dir(parent: Path, head: str) -> Path:
    return ROOT / parent / REPRODUCE_DIR / head


def reproduce_entries() -> list[tuple[Path, Path]]:
    """(parent, entry) of every output of `part5 part6` to move aside: each entry of
    REPRODUCE_PARENTS but the `_`-prefixed ones, dot files and REPRODUCE_KEEP."""
    entries = []
    for parent in REPRODUCE_PARENTS:
        root = ROOT / parent
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if p.name.startswith(("_", ".")) or (parent / p.name).as_posix() in REPRODUCE_KEEP:
                continue
            entries.append((parent, p))
    return entries


def _bundle_dir(models: Path) -> Path | None:
    """The ETA bundle named by `<models>/LATEST`, or None."""
    latest = models / "LATEST"
    if not latest.is_file():
        return None
    bundle = models / latest.read_text(encoding="utf-8").strip()
    return bundle if bundle.is_dir() else None


def _rl_run_dirs(root: Path) -> list[Path]:
    """Training runs (holding manifest.json) directly under `root`, as rl.evaluate lists
    them (`_`-prefixed folders skipped)."""
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "manifest.json").is_file()
    )


def reproduce_backup() -> int:
    """Move the Phase 3 outputs of `part5 part6` to `<parent>/_reproduce/<HEAD>/`, so that
    the chains that follow rebuild them from the committed tree: a fresh freeze, fresh run
    folders. Refuses a dirty tree, a backup already there and a tree with nothing to
    reproduce."""
    head, dirty = git_head()
    if dirty:
        print("reproduce56: the tree has uncommitted changes; commit first", file=sys.stderr)
        return 1
    taken = [
        (parent / REPRODUCE_DIR / head).as_posix()
        for parent in REPRODUCE_PARENTS
        if (b := reproduce_backup_dir(parent, head)).is_dir() and any(b.iterdir())
    ]
    if taken:
        print(
            f"reproduce56: {', '.join(taken)} already holds a backup; inspect and remove it first",
            file=sys.stderr,
        )
        return 1
    if _bundle_dir(ROOT / ETA_MODELS_DIR) is None or not _rl_run_dirs(ROOT / RL_RUNS_DIR):
        print(
            f"reproduce56: nothing to reproduce: needs the ETA bundle "
            f"({ETA_MODELS_DIR.as_posix()}/LATEST) and the RL runs under {RL_RUNS_DIR.as_posix()}/",
            file=sys.stderr,
        )
        return 1
    entries = reproduce_entries()
    for parent, entry in entries:
        backup = reproduce_backup_dir(parent, head)
        backup.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entry), str(backup / entry.name))
    print(f"reproduce56: moved {len(entries)} entries to <dir>/{REPRODUCE_DIR}/{head}/")
    return 0


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _json_file_diff(old: Path, new: Path, ignore: frozenset[str]) -> list[str]:
    a, b = _read_json(old), _read_json(new)
    if a is None or b is None:
        return [f"{new.name} missing" if b is None else f"{old.name} missing in the backup"]
    return [f"{new.name}: {p}" for p in json_diff(a, b, ignore)]


def _run_diff(old: Path, new: Path) -> list[str]:
    """A run_scenario run re-made: KPI rows byte-identical (run_id masked), run.json equal
    apart from RUN_JSON_VOLATILE."""
    if not (new / "run.json").is_file():
        return ["not re-made"]
    out = [f"{n} differs" for n in COMPARED_RUN_FILES if _masked(old, n) != _masked(new, n)]
    return out + [
        f"run.json: {p}" for p in json_diff(_run_meta(old), _run_meta(new), RUN_JSON_VOLATILE)
    ]


def _eta_diffs(head: str) -> dict[str, list[str]]:
    """ETA bundle (model and reference files, meta.json, eval.json) and dataset manifest."""
    old_b = _bundle_dir(reproduce_backup_dir(Path("data/models"), head) / ETA_MODELS_DIR.name)
    new_b = _bundle_dir(ROOT / ETA_MODELS_DIR)
    if old_b is None or new_b is None:
        return {
            "eta bundle": [
                "no bundle (LATEST) in the " + ("backup" if old_b is None else "rebuild")
            ]
        }
    diffs: dict[str, list[str]] = {}
    item = f"eta bundle {old_b.name} -> {new_b.name}"
    out = []
    for name in ETA_BUNDLE_FILES:
        if not (old_b / name).is_file() or not (new_b / name).is_file():
            out.append(f"{name} missing")
        elif _sha256(old_b / name) != _sha256(new_b / name):
            out.append(f"{name} sha256 differs")
    out += _json_file_diff(old_b / "meta.json", new_b / "meta.json", ETA_META_VOLATILE)
    out += _json_file_diff(old_b / "eval.json", new_b / "eval.json", ETA_EVAL_VOLATILE)
    diffs[item] = out
    ids = [(_read_json(b / "meta.json") or {}).get("dataset_id") for b in (old_b, new_b)]
    old_d = reproduce_backup_dir(ETA_DATA_DIR, head) / str(ids[0])
    new_d = ROOT / ETA_DATA_DIR / str(ids[1])
    diffs[f"eta dataset {ids[0]} -> {ids[1]}"] = _json_file_diff(
        old_d / "manifest.json", new_d / "manifest.json", ETA_DATASET_VOLATILE
    )
    return diffs


def _rl_diffs(head: str) -> dict[str, list[str]]:
    """Every training run (manifest, plus `rl.evaluate repro`: policy parameters, episode
    returns and lengths, validation costs), the selection and the dev-test outputs."""
    backup = reproduce_backup_dir(RL_RUNS_DIR, head)
    diffs: dict[str, list[str]] = {}
    for old in _rl_run_dirs(backup):
        new = ROOT / RL_RUNS_DIR / old.name
        if not (new / "manifest.json").is_file():
            diffs[f"rl run {old.name}"] = ["not re-made"]
            continue
        out = _json_file_diff(old / "manifest.json", new / "manifest.json", RL_MANIFEST_VOLATILE)
        report = backup / "repro" / f"{old.name}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        cmd = _m("traffic_twin.rl.evaluate", "repro", _rel(old), _rel(new), "--out", _rel(report))
        print(f"+ {format_command(cmd)}", flush=True)
        if call(cmd) != 0:
            out.append(f"rl.evaluate repro: differences, see {_rel(report)}")
        diffs[f"rl run {old.name}"] = out
    old_sel = reproduce_backup_dir(Path("data/models"), head) / RL_SELECTED_DIR.relative_to(
        "data/models"
    )
    diffs["rl selection"] = _json_file_diff(
        old_sel / "manifest.json", ROOT / RL_SELECTED_DIR / "manifest.json", RL_SELECTED_VOLATILE
    )
    old_dev, new_dev = backup / "devtest", ROOT / RL_RUNS_DIR / "devtest"
    if old_dev.is_dir():
        out = []
        for name in DEVTEST_TABLES:
            if (old_dev / name).is_file() and (
                not (new_dev / name).is_file()
                or (old_dev / name).read_bytes() != (new_dev / name).read_bytes()
            ):
                out.append(f"{name} differs")
        for name in DEVTEST_SUMMARIES:
            if (old_dev / name).is_file():
                out += _json_file_diff(old_dev / name, new_dev / name, RL_SUMMARY_VOLATILE)
        diffs["rl devtest tables"] = out
        for run in sorted(p.parent for p in old_dev.glob("*/run.json")):
            diffs[f"rl devtest run {run.name}"] = _run_diff(run, new_dev / run.name)
    return diffs


def reproduce_verify() -> int:
    """Compare every moved original with its reproduction from HEAD (see the module
    docstring for what must match and what may differ), and require the rewritten pages
    to name no uncommitted tree. Writes `data/models/_reproduce/<HEAD>/reproduce56.json`;
    the originals are kept for inspection."""
    head, _ = git_head()
    diffs = _eta_diffs(head)
    for run in sorted(p.parent for p in reproduce_backup_dir(DEMO_DIR, head).glob("*/run.json")):
        diffs[f"demo run {run.name}"] = _run_diff(run, ROOT / DEMO_DIR / run.name)
    diffs.update(_rl_diffs(head))
    for doc in REPRODUCED_DOCS:
        page = ROOT / doc
        found = (
            sorted(set(DIRTY_VERSION.findall(page.read_text(encoding="utf-8"))))
            if page.is_file()
            else []
        )
        diffs[f"page {doc}"] = [f"names {v}" for v in found] if page.is_file() else ["missing"]
    for item, what in diffs.items():
        status = "same" if not what else "DIFFERENT: " + "; ".join(what[:8])
        print(f"reproduce56: {item}: {status}", file=sys.stderr if what else sys.stdout)
    report = reproduce_backup_dir(Path("data/models"), head) / "reproduce56.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    same = sum(not what for what in diffs.values())
    report.write_text(
        json.dumps({"head": head, "same": same == len(diffs), "items": diffs}, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"reproduce56: {same} of {len(diffs)} items reproduced; report {_rel(report)}")
    return 0 if same == len(diffs) else 1


# -- Parts 5 and 6 ------------------------------------------------------------------------
def _eta_data(o: argparse.Namespace, splits: tuple[str, ...]) -> list[Command]:
    """ETA dataset generation of `splits` (D46: resumable, one world per process task)."""
    dirty = ["--allow-dirty"] if o.allow_dirty else []
    return [_m("traffic_twin.ml.eta_dataset", "--config", ETA_CONFIG, "--splits", *splits, *dirty)]


def rl_s6_runs() -> dict[str, dict]:
    """`train.s6` of configs/rl.yaml: {algo: {"seeds": [...], "steps": N}} (D61). PyYAML is
    imported here only, so the other tasks (and `install`) need nothing beyond stdlib."""
    import yaml  # core dependency of the package

    cfg = yaml.safe_load((ROOT / RL_CONFIG).read_text(encoding="utf-8"))
    try:
        s6 = cfg["train"]["s6"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"tasks.py: {RL_CONFIG} has no train.s6 block") from exc
    return {str(algo): dict(plan) for algo, plan in s6.items()}


def _rl_train(o: argparse.Namespace) -> list[Command]:
    """One `rl.train` run per (algo, seed) of `train.s6`, in config order, one after the
    other (each run uses `train.n_envs` worker processes). `--algo` keeps one algorithm,
    `--seed` replaces its seeds, `--steps` its budget."""
    s6 = rl_s6_runs()
    algos = [o.algo] if o.algo else list(s6)
    cmds: list[Command] = []
    for algo in algos:
        plan = s6.get(algo, {})
        seeds = o.seed or [str(s) for s in plan.get("seeds", ())]
        steps = o.steps or (str(plan["steps"]) if "steps" in plan else None)
        if not seeds or steps is None:
            raise SystemExit(
                f"tasks.py: {RL_CONFIG} train.s6 has no {algo} run; give --seed and --steps"
            )
        n_envs = ["--n-envs", o.n_envs] if o.n_envs else []
        cmds += [
            _m("traffic_twin.rl.train", "--config", RL_CONFIG, "--algo", algo,
               "--seed", seed, "--steps", steps, *n_envs)
            for seed in seeds
        ]  # fmt: skip
    return cmds


# -- Part 7 ---------------------------------------------------------------------------------
RL7_STUDY = "traffic_twin.rl.study"
RL7_SENSITIVITY = "traffic_twin.rl.sensitivity"


def _rl7_campaign(o: argparse.Namespace) -> list[Command]:
    """Every pre-registered Part 7 training (stages T, S and F of configs/rl7.yaml) through
    `rl.campaign`, which picks C* for stage F itself (`study decide-t`) once T is done."""
    slots = ["--slots", o.slots] if o.slots else []
    dirty = ["--allow-dirty"] if o.allow_dirty else []
    return [
        _m("traffic_twin.rl.campaign", "--config", RL7_CONFIG, "--stages", *RL7_STAGES,
           *slots, *dirty)
    ]  # fmt: skip


def rl7_run_dirs(stage: str) -> list[Path]:
    """Finished Part 7 runs (folders holding manifest.json) of one campaign stage, relative
    to ROOT; `_`-prefixed folders are skipped."""
    return [p.relative_to(ROOT) for p in _rl_run_dirs(ROOT / RL7_RUNS_DIR / stage)]


def rl7_holdout_commands() -> list[list[str]]:
    """`rl.study holdout` of every finished run, listed when the step runs (the campaign
    makes them): the T and S runs on their pick, the F runs with `--curves`."""
    plain = [d.as_posix() for s in RL7_STAGES if s not in RL7_CURVE_STAGES for d in rl7_run_dirs(s)]
    curves = [d.as_posix() for s in RL7_CURVE_STAGES for d in rl7_run_dirs(s)]
    cmds = [_m(RL7_STUDY, "holdout", *plain)] if plain else []
    return cmds + ([_m(RL7_STUDY, "holdout", *curves, "--curves")] if curves else [])


def _call_all(cmds: list[list[str]]) -> int:
    for cmd in cmds:
        print(f"+ {format_command(cmd)}", flush=True)
        code = call(cmd)
        if code != 0:
            return code
    return 0


def rl7_holdout() -> int:
    cmds = rl7_holdout_commands()
    if not cmds:
        print(
            f"rl7-holdout: no finished run under {RL7_RUNS_DIR.as_posix()}/<stage>/ "
            "(run `tasks.py rl7-campaign` first)",
            file=sys.stderr,
        )
        return 1
    return _call_all(cmds)


def part7_s3_check() -> int:
    """`part7` goes on past `rl7-publish` only once configs/pitie_rl_s3.yaml exists: it is
    written by hand from the lines `rl7-publish` prints (D86; no tool edits a config)."""
    if (ROOT / PITIE_RL_S3).is_file():
        return 0
    print(
        f"part7: stopped after rl7-publish: {PITIE_RL_S3} is missing. Write it from the "
        f"lines `rl7-publish` printed (D86), run `python -m {RL7_STUDY} golden --config "
        f"{PITIE_RL_S3}`, then `tasks.py rl7-stress rl7-sens rl7-devtest rl7-docs`",
        file=sys.stderr,
    )
    return 1


def reproduce7_backup_dir(head: str) -> Path:
    return ROOT / RL7_DIR / REPRODUCE_DIR / head


def reproduce7_name_glob() -> str:
    """Folder name pattern of the D82 run (contract: <algo>_<variant>_s<seed>_<k>k_<sha8>)."""
    r = REPRODUCE7_RUN
    return f"{r['algo']}_{r['variant']}_s{r['seed']}_{int(r['steps']) // 1000}k_*"


def _run_record(run: Path) -> dict:
    """What a Part 7 run recorded: its provenance.json over its manifest.json."""
    out: dict = {}
    for name in ("manifest.json", "provenance.json"):
        doc = _read_json(run / name)
        if isinstance(doc, dict):
            out.update(doc)
    return out


def reproduce7_train_env(original: Path | None) -> dict[str, str]:
    """Environment of the reproduce7 training: the original learner's, so that the rerun
    records the same provenance `machine.env` and `child_env` and its learner and workers
    run with the same thread variables. This environment without its OMP_* / MKL_* /
    OPENBLAS_* variables plus those the original recorded (machine.env); for a run without
    that record, what `rl.campaign.train_env()` gives (RL7_CHILD_ENV where unset)."""
    machine = _run_record(original).get("machine") if original is not None else None
    recorded = machine.get("env") if isinstance(machine, dict) else None
    if not isinstance(recorded, dict):
        env = dict(os.environ)
        for k, v in RL7_CHILD_ENV.items():
            env.setdefault(k, v)
        return env
    env = {k: v for k, v in os.environ.items() if not k.startswith(RL7_ENV_PREFIXES)}
    env.update({str(k): str(v) for k, v in recorded.items()})
    return env


def reproduce7_train_options(original: Path | None) -> list[str]:
    """The `rl.train` options of the original launch reproduce7 repeats: its validation
    pool (`--eval-pool`, from provenance eval_pool; the campaign's `--eval-pool`)."""
    pool = _run_record(original).get("eval_pool") if original is not None else None
    return ["--eval-pool", pool] if pool in RL7_EVAL_POOLS else []


def reproduce7_commands(backup: Path, original: Path | None = None) -> dict[str, list[str]]:
    """The three reruns of `reproduce7`, all into `backup`: the t1 seed 0 training (with
    the `original` run's validation pool, reproduce7_train_options), the S3 golden and the
    page with its figures (same layout as the repo, so the page's relative figure links
    are the same)."""
    r, rel = REPRODUCE7_RUN, _rel(backup)
    return {
        "train": _m("traffic_twin.rl.train", "--config", str(r["config"]), "--algo",
                    str(r["algo"]), "--seed", str(r["seed"]), "--steps", str(r["steps"]),
                    "--out", f"{rel}/runs", *reproduce7_train_options(original)),
        "golden": _m(RL7_STUDY, "golden", "--config", PITIE_RL_S3, "--out",
                     f"{rel}/{RL7_GOLDEN.name}"),
        "docs": _m("traffic_twin.rl.report7", "--docs", f"{rel}/{RL7_DOC}", "--figures",
                   f"{rel}/{RL7_FIGURES}"),
    }  # fmt: skip


def reproduce7_prepare(state: dict) -> int:
    """Refuse a dirty tree, a missing original (the one finished t1 seed 0 stage-T run,
    the S3 config, golden and page) and a backup folder already in use; record them."""
    head, dirty = git_head()
    if dirty:
        print("reproduce7: the tree has uncommitted changes; commit first", file=sys.stderr)
        return 1
    stage_dir = RL7_RUNS_DIR / str(REPRODUCE7_RUN["stage"])
    found = [
        p for p in sorted((ROOT / stage_dir).glob(reproduce7_name_glob()))
        if (p / "manifest.json").is_file()
    ]  # fmt: skip
    if len(found) != 1:
        print(
            f"reproduce7: needs exactly one finished run {stage_dir.as_posix()}/"
            f"{reproduce7_name_glob()}, found {len(found)} (run `tasks.py rl7-campaign`)",
            file=sys.stderr,
        )
        return 1
    missing = [p for p in (PITIE_RL_S3, RL7_GOLDEN.as_posix(), RL7_DOC) if not (ROOT / p).is_file()]
    if missing:
        print(f"reproduce7: missing {', '.join(missing)} (Part 7 Phase 3)", file=sys.stderr)
        return 1
    backup = reproduce7_backup_dir(head)
    if backup.is_dir() and any(backup.iterdir()):
        print(
            f"reproduce7: {_rel(backup)} already holds a reproduction; inspect and remove it first",
            file=sys.stderr,
        )
        return 1
    backup.mkdir(parents=True, exist_ok=True)
    state.update(head=head, original=found[0], backup=backup)
    print(f"reproduce7: original {_rel(found[0])}; reproduction into {_rel(backup)}")
    return 0


def reproduce7_run(state: dict, what: str) -> int:
    """One rerun of `reproduce7`; the training is launched as the campaign launched the
    original (its environment and validation pool), so a faithful rerun records the same
    provenance."""
    original = state.get("original")
    cmd = reproduce7_commands(state["backup"], original)[what]
    if what != "train":
        return _call_all([cmd])
    print(f"+ {format_command(cmd)}", flush=True)
    return call(cmd, env=reproduce7_train_env(original))


def _csv_rows(
    path: Path, drop: frozenset[str] = frozenset(), keep: tuple[str, ...] = (), skip: int = 0
) -> list[dict[str, str]] | None:
    """Rows of a CSV file as text (bitwise comparison), without the `drop` columns, or
    only the `keep` ones; `skip` header lines first (VecMonitor's JSON line). None if absent."""
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        for _ in range(skip):
            f.readline()
        rows = list(csv.DictReader(f))
    return [
        {k: v for k, v in row.items() if k not in drop and (not keep or k in keep)} for row in rows
    ]


def _rl7_json_diff(old: Path, new: Path, ignore: frozenset[str]) -> list[str]:
    a, b = _read_json(old), _read_json(new)
    if a is None or b is None:
        return [f"{old.name} missing in the {'original' if a is None else 'reproduction'}"]
    return [f"{old.name}: {p}" for p in json_diff(a, b, ignore)]


def rl7_run_diff(old: Path, new: Path) -> list[str]:
    """A Part 7 run reproduced (D82): manifest.json and provenance.json equal apart from
    RL7_RUN_VOLATILE (so the best, final and pick policy sha256 and steps), the same
    validation rows (timing columns aside) and the same episode returns and lengths."""
    if not (new / "manifest.json").is_file():
        return ["not re-made"]
    out = [
        d
        for name in ("manifest.json", "provenance.json")
        for d in _rl7_json_diff(old / name, new / name, RL7_RUN_VOLATILE)
    ]
    recorded = {
        **(_read_json(old / "provenance.json") or {}),
        **(_read_json(old / "manifest.json") or {}),
    }
    out += [
        f"no {key} recorded" for key in ("final_policy_sha256", "pick_step") if key not in recorded
    ]
    tables = (
        ("validation.csv", {"drop": RL7_VALIDATION_VOLATILE}),
        ("monitor.csv", {"keep": ("r", "l"), "skip": 1}),
    )
    for name, how in tables:
        a, b = _csv_rows(old / name, **how), _csv_rows(new / name, **how)
        if a is None or b is None:
            out.append(f"{name} missing in the {'original' if a is None else 'reproduction'}")
        elif a != b:
            out.append(f"{name} differs")
    return out


def _page_lines(path: Path, figures: str | None = None) -> list[tuple[int, str]] | None:
    """(line number, line) of the page without its code-version lines, with `figures`
    written as the repo's figure folder; None if absent."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if figures:
        text = text.replace(figures, RL7_FIGURES)
    return [
        (n, line)
        for n, line in enumerate(text.splitlines(), start=1)
        if not CODE_VERSION_LINE.search(line)
    ]


def reproduce7_verify(state: dict) -> int:
    """Compare the reproduction with the originals (D82) and write
    `data/rl7/_reproduce/<HEAD>/reproduce7.json`; the originals are never touched."""
    backup, original = state["backup"], state["original"]
    diffs: dict[str, list[str]] = {}
    new = backup / "runs" / original.name
    if (new / "manifest.json").is_file():
        diffs[f"rl7 run {original.name}"] = rl7_run_diff(original, new)
    else:
        made = [p.name for p in _rl_run_dirs(backup / "runs")]
        diffs[f"rl7 run {original.name}"] = [
            "not re-made" + (f" (the rerun is {', '.join(made)})" if made else "")
        ]
    diffs[f"golden {RL7_GOLDEN.as_posix()}"] = _rl7_json_diff(
        ROOT / RL7_GOLDEN, backup / RL7_GOLDEN.name, RL7_GOLDEN_VOLATILE
    )
    old_page = _page_lines(ROOT / RL7_DOC)
    new_page = _page_lines(backup / RL7_DOC, f"{_rel(backup)}/{RL7_FIGURES}")
    if old_page is None or new_page is None:
        page = [f"missing in the {'original' if old_page is None else 'reproduction'}"]
    elif [t for _, t in old_page] != [t for _, t in new_page]:
        pairs = zip(old_page, new_page, strict=False)
        line = next((a[0] for a, b in pairs if a[1] != b[1]), None)
        if line is None:  # one page is the other plus lines at its end
            shorter = min(old_page, new_page, key=len)
            line = shorter[-1][0] + 1 if shorter else 1
        page = [f"differs from line {line} of the original (code-version lines aside)"]
    else:
        page = []
    diffs[f"page {RL7_DOC}"] = page
    for item, what in diffs.items():
        status = "same" if not what else "DIFFERENT: " + "; ".join(what[:8])
        print(f"reproduce7: {item}: {status}", file=sys.stderr if what else sys.stdout)
    same = sum(not what for what in diffs.values())
    report = backup / "reproduce7.json"
    report.write_text(
        json.dumps({"head": state["head"], "same": same == len(diffs), "items": diffs}, indent=1)
        + "\n",
        encoding="utf-8",
    )
    print(f"reproduce7: {same} of {len(diffs)} items reproduced; report {_rel(report)}")
    return 0 if same == len(diffs) else 1


# Each task returns the commands to run, given the parsed options
TASKS: dict[str, tuple[str, Callable[[argparse.Namespace], list[Command]]]] = {
    "install": (
        f"CPU {TORCH_PIN}, then pip install -e .[{','.join(INSTALL_EXTRAS)}] into the current "
        "interpreter",
        lambda o: [
            _m("pip", "install", TORCH_PIN, "--index-url", TORCH_CPU_INDEX),
            _m("pip", "install", "-e", f".[{','.join(INSTALL_EXTRAS)}]"),
        ],
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
        "run --scenario headless for --seeds (default 1 2 3); an evaluation scenario on "
        "seeds 1-10 only after the registration commit (run lock, D92)",
        lambda o: [
            _sim_lock(o, _seeds(o)),
            _m("traffic_twin.sim.run_scenario", "--config", _grid(o), "--seed", *_seeds(o)),
        ],
    ),
    "sim-gui": (
        "run --scenario in sumo-gui (first seed; run lock as for sim)",
        lambda o: [
            _sim_lock(o, _seeds(o)[:1]),
            _m(
                "traffic_twin.sim.run_scenario",
                "--config",
                _grid(o),
                "--seed",
                _seeds(o)[0],
                "--gui",
            ),
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
    "field-sheets": (
        "Pitie: printable field-visit sheets (docs/field/pitie_field_sheets.pdf)",
        lambda o: [_m("traffic_twin.validation.field_sheets", "--scenario", _s0(o))],
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
        "Pitie: paired --baseline (S0) vs --scenario (S1) on the evaluation seeds (run "
        "lock on the missing runs, D92)",
        lambda o: [
            _missing_lock([o.baseline or PITIE_S0, o.scenario or PITIE_S1], o.baseline or PITIE_S0),
            _m(
                "traffic_twin.evaluation.compare",
                "--baseline-config",
                o.baseline or PITIE_S0,
                "--scenario-config",
                o.scenario or PITIE_S1,
                "--run-missing",
            ),
        ],
    ),
    "validate": (
        f"Pitie: sim vs reality tables and {VALIDATION_DOC} (runs missing seeds; run lock, D92)",
        lambda o: [
            _missing_lock([_s0(o)], _s0(o)),
            _m(
                "traffic_twin.validation.report",
                "--scenario",
                _s0(o),
                "--run-missing",
                "--docs",
                VALIDATION_DOC,
            ),
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
        f"Pitie: pre-registered S0-S4 evaluation and {EVALUATION_DOC} (runs missing seeds; "
        "run lock on seeds 1-10, D92)",
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
        f"Pitie: evaluation figures (S0-S4, equity, A-04 to A-06) into {EVAL_FIGURES}",
        lambda o: [
            _m("traffic_twin.viz.evaluation", "--config", EVALUATION_CONFIG, "--out", EVAL_FIGURES)
        ],
    ),
    "provenance": (
        "Pitie: exit 1 unless the evaluation runs and published pages come from a clean HEAD",
        lambda o: [Step("provenance", check_provenance)],
    ),
    # -- Part 5: ETA dataset, quantile model, A-07, API v1 and mock screens -----------------
    "eta-daytypes": (
        "ETA: check the generated day-type calibration files (D44)",
        lambda o: [_m("traffic_twin.ml.daytypes", "--check")],
    ),
    "eta-data": (
        "ETA: dataset of the train, val and calib worlds (EF-12; --allow-dirty for a dev id)",
        lambda o: _eta_data(o, ETA_FIT_SPLITS),
    ),
    "eta-train": (
        "ETA: quantile model, conformal step, bundle (EF-13), then freeze it before test",
        lambda o: [
            _m("traffic_twin.ml.train_eta", "--config", ETA_CONFIG),
            _m("traffic_twin.ml.train_eta", "--config", ETA_CONFIG, "--freeze"),
        ],
    ),
    "eta-data-test": (
        "ETA: dataset of the test worlds 2900-2999 (once, after the freeze)",
        lambda o: _eta_data(o, (ETA_TEST_SPLIT,)),
    ),
    "eta-eval": (
        f"ETA: A-07 / EF-14 evaluation on the test worlds and {ETA_DOC}",
        lambda o: [
            _m(
                "traffic_twin.ml.evaluate_eta",
                "--config",
                ETA_CONFIG,
                "--split",
                ETA_TEST_SPLIT,
                "--docs",
                ETA_DOC,
                "--figures",
                ETA_FIGURES,
            )
        ],  # fmt: skip
    ),
    "demo-runs": (
        f"ETA: demo runs S0/S1/S2 seed {' '.join(DEMO_SEEDS)} into data/demo with predictions",
        lambda o: [_m("traffic_twin.ml.predict", "demo", "--seeds", *DEMO_SEEDS)],
    ),
    "api": (
        f"API v1 and /mock on http://{API_HOST}:{API_PORT} (uvicorn, no access log)",
        lambda o: [
            _m(
                "uvicorn",
                "traffic_twin.api.main:app",
                "--no-access-log",
                "--host",
                API_HOST,
                "--port",
                API_PORT,
            )
        ],  # fmt: skip
    ),
    "api-bench": (
        f"API latency (ENF-02) on {' and '.join(API_RESULTS)}, written to {API_DOC}",
        lambda o: [_m("traffic_twin.api.bench", "--results", *API_RESULTS, "--docs", API_DOC)],
    ),
    # -- Part 6: Gymnasium environment, first RL trainings ---------------------------------
    "rl-timing": (
        "RL: episode timing and vectorised scaling (spec 10.4, S6 measurement)",
        lambda o: [_m("traffic_twin.rl.evaluate", "timing")],
    ),
    "rl-train": (
        f"RL: the S6 trainings of {RL_CONFIG} train.s6, one after the other "
        "(--algo, --seed, --steps override)",
        _rl_train,
    ),
    "rl-select": (
        "RL: best run by validation cost -> data/models/rl/selected",
        lambda o: [_m("traffic_twin.rl.evaluate", "select")],
    ),
    "rl-eval": (
        f"RL: dev-test {RL_DEVTEST_SEEDS[0]}-{RL_DEVTEST_SEEDS[-1]} window comparison, "
        f"full-run KPIs, {RL_DOC}",
        lambda o: [
            _m("traffic_twin.rl.evaluate", "compare", "--seeds", *RL_DEVTEST_SEEDS),
            _m("traffic_twin.rl.evaluate", "kpi", "--seeds", *RL_DEVTEST_SEEDS),
            _m("traffic_twin.rl.evaluate", "docs", "--docs", RL_DOC, "--figures", RL_FIGURES),
        ],
    ),
    # -- Part 7: full RL training, tuning, weight sensitivity, S3 (configs/rl7.yaml) -------
    "rl7-campaign": (
        f"RL7: every pre-registered training of {RL7_CONFIG} (stages T S F, concurrent; "
        "--slots, --allow-dirty)",
        _rl7_campaign,
    ),
    "rl7-holdout": (
        "RL7: holdout scores of every finished run's pick, checkpoint curves for stage F "
        "(reporting only)",
        lambda o: [Step("rl7-holdout", rl7_holdout)],
    ),
    "rl7-select": (
        "RL7: two-stage S3 selection among the stage-F picks (D75)",
        lambda o: [_m(RL7_STUDY, "select")],
    ),
    "rl7-publish": (
        f"RL7: the S3 model into models/rl/<model_version>/, prints the {PITIE_RL_S3} lines",
        lambda o: [_m(RL7_STUDY, "publish")],
    ),
    "rl7-stress": (
        "RL7: stress panel of S3, S0, S1 and S2 on the stress seeds (D78, not gating)",
        lambda o: [_m(RL7_STUDY, "stress")],
    ),
    "rl7-sens": (
        "RL7: weight-sensitivity KPI runs on the sensitivity seeds, then its tables (D76)",
        lambda o: [_m(RL7_SENSITIVITY, "run"), _m(RL7_SENSITIVITY, "tables")],
    ),
    "rl7-devtest": (
        f"RL7: the single dev-test look {RL7_DEVTEST_RANGE[0]}-{RL7_DEVTEST_RANGE[1]} and the "
        "interim 10.6 verdict (D79; refuses a second look)",
        lambda o: [_m(RL7_STUDY, "devtest")],
    ),
    "rl7-docs": (
        f"RL7: {RL7_DOC} and its figures in {RL7_FIGURES}",
        lambda o: [_m("traffic_twin.rl.report7", "--docs", RL7_DOC, "--figures", RL7_FIGURES)],
    ),
    # -- Part 8: final evaluation of S0 to S4 on seeds 1-10 (configs/evaluation.yaml) -----
    "part8-guard": (
        "Part 8: exit 1 unless HEAD is the clean registration commit (a06) and part8 has not "
        "run from it",
        lambda o: [Step("part8-guard", part8_guard)],
    ),
    "rebaseline-verify": (
        "Pitie: compare the re-made evaluation runs with the moved ones, write "
        f"{VERIFY_FILE}, then provenance (resumes part8 after a STOP)",
        lambda o: [Step("rebaseline-verify", rebaseline_verify)],
    ),
    # -- Part 9: app on the final results, replay, user-test package --------------------
    "demo-runs8": (
        f"demo runs S2_noP/S3/S4 seed {' '.join(DEMO_SEEDS)} into data/demo (D109; S0-S2: "
        "demo-runs)",
        lambda o: [
            _m(
                "traffic_twin.ml.predict",
                "demo",
                "--seeds",
                *DEMO_SEEDS,
                "--configs",
                *DEMO_RUNS8_CONFIGS,
            )
        ],  # fmt: skip
    ),
    "operator": (
        f"the Streamlit Operator ({OPERATOR_SCRIPT}) on http://{OPERATOR_HOST}:{OPERATOR_PORT}, "
        "through the API",
        lambda o: [
            _m(
                "streamlit",
                "run",
                OPERATOR_SCRIPT,
                "--server.address",
                OPERATOR_HOST,
                "--server.port",
                OPERATOR_PORT,
            )
        ],  # fmt: skip
    ),
    "usertest-kit": (
        "user tests: the printable kit docs/usertests/usertest_kit.pdf (--allow-pending-s3)",
        lambda o: [_m(USERTESTS, "kit", *(["--allow-pending-s3"] if o.allow_pending_s3 else []))],
    ),
    "usertest-serve": (
        "user tests: the frozen app for a session (--allow-dirty: from an uncommitted tree)",
        lambda o: [_m(USERTESTS, "serve", *(["--allow-dirty"] if o.allow_dirty else []))],
    ),
    "usertest-check": (
        "user tests: validate docs/usertests/results.yaml (schema, privacy, fix loop)",
        lambda o: [_m(USERTESTS, "check")],
    ),
    "usertest-docs": (
        "user tests: docs/usertests.md and synthesis.json (A-10 PENDING below 3 sessions)",
        lambda o: [_m(USERTESTS, "docs")],
    ),
    "test-browser": (
        "the headless-browser tests only (pytest -m browser)",
        lambda o: [_m("pytest", "-q", "-m", "browser")],
    ),
    # -- Part 10: report, slides, rehearsed demo -----------------------------------------
    "report-facts": (
        "report: docs/report/facts.json from the local outputs (data/ and docs/)",
        lambda o: [_m(REPORT, "facts")],
    ),
    "report": (
        "report: docs/report/report.md and report.pdf from facts.json and the templates",
        lambda o: [_m(REPORT, "report")],
    ),
    "slides": (
        "report: the deck docs/slides/urban_traffic_twin.pptx from facts.json",
        lambda o: [_m(REPORT, "slides")],
    ),
    "app-shots": (
        "report: headless-browser screenshots of the demo steps into docs/figures/app",
        lambda o: [_m(REPORT, "shots")],
    ),
    "report-check": (
        "report: optional render of the PDF and the deck to data/report/_qa (LibreOffice)",
        lambda o: [_m(REPORT, "check")],
    ),
    "demo": (
        f"demo: the API ({API_HOST}:{API_PORT}) and the Operator ({OPERATOR_PORT}), checked, "
        "until Ctrl+C (--no-browser)",
        lambda o: [_m(DEMO, "serve", *_no_browser(o))],
    ),
    "demo-check": (
        "demo: walk every configs/demo.yaml step against the API and the views (--no-browser)",
        lambda o: [_m(DEMO, "check", *_no_browser(o))],
    ),
    "demo-rehearsal": (
        "demo: headless rehearsal to data/demo_check (--fixtures, --no-browser; --record "
        "writes docs/demo/rehearsal.json)",
        lambda o: [
            _m(
                DEMO,
                "rehearsal",
                *(["--fixtures"] if o.fixtures else []),
                *_no_browser(o),
                *(["--record"] if o.record else []),
            )
        ],  # fmt: skip
    ),
    "demo-clean": (
        "demo: stop leftover demo servers, remove old data/demo_check reports",
        lambda o: [_m(DEMO, "clean")],
    ),
    "demo-pack": (
        "demo: docs/demo/demo_pack.zip + MANIFEST.json (demo runs, evaluation outputs, ETA)",
        lambda o: [_m(DEMO, "pack")],
    ),
    "bootstrap": (
        "demo: restore the demo pack into data/ on a fresh clone (verifies hashes, never "
        "overwrites; never rebuilds the evaluation)",
        lambda o: [_m(DEMO, "bootstrap")],  # restores docs/demo/demo_pack.zip (--from-pack default)
    ),
    "demo-readiness": (
        "demo: fresh local clone of HEAD + demo pack + recorded rehearsal (A-09)",
        lambda o: [_m(DEMO, "readiness")],
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
    steps = [c for t in PART3_STEPS for c in TASKS[t][1](on_s1 if t == "compare" else on_s0)]
    # fail fast: the run lock of compare and validate, before the calibration loop
    return [_missing_lock([s0, s1], s0), *steps]


PART4_STEPS = ("tls-check", "evaluate", "plots-eval")


def _part4(o: argparse.Namespace) -> list[Command]:
    """The Part 4 chain: `tls-check` on S2 (--scenario, default the rules config), then the
    evaluation and its figures, whose scenarios come from configs/evaluation.yaml."""
    on_s2 = argparse.Namespace(**{**vars(o), "scenario": o.scenario or PITIE_S2})
    return [c for t in PART4_STEPS for c in TASKS[t][1](on_s2)]


REBASELINE_STEPS = ("compare", "validate", "part4")


def _rebaseline(o: argparse.Namespace) -> list[Command]:
    """Move the runs aside, re-make the published pages with every default (`compare
    validate part4`, which run the missing seeds), then compare the runs. The run lock
    goes first: once the runs are moved aside, every seed 1-10 run is to be re-made."""
    defaults = argparse.Namespace(**{**vars(o), "scenario": None, "baseline": None})
    return [
        _missing_lock([PITIE_S0], PITIE_S0, every=True),
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
PART5_STEPS = (
    "eta-daytypes", "eta-data", "eta-train", "eta-data-test", "eta-eval", "demo-runs",
    "api-bench",
)  # fmt: skip
PART6_STEPS = ("rl-timing", "rl-train", "rl-select", "rl-eval")
TASKS["part5"] = (
    f"ETA and API chain (configs/eta.yaml): {' '.join(PART5_STEPS)}",
    lambda o: [c for t in PART5_STEPS for c in TASKS[t][1](o)],
)
TASKS["part6"] = (
    f"RL chain (configs/rl.yaml): {' '.join(PART6_STEPS)}",
    lambda o: [c for t in PART6_STEPS for c in TASKS[t][1](o)],
)
TASKS["rebaseline"] = (
    "Pitie, after a commit: re-make the evaluation runs from HEAD (compare validate "
    "part4), require identical KPI rows",
    _rebaseline,
)


def _reproduce56(o: argparse.Namespace) -> list[Command]:
    """Move the Parts 5-6 outputs aside, rerun `part5 part6` with every default (no
    --allow-dirty, the train.s6 plan unchanged), then compare old with new."""
    defaults = argparse.Namespace(
        **{**vars(o), "allow_dirty": False, "algo": None, "seed": None, "steps": None,
           "n_envs": None}
    )  # fmt: skip
    return [
        Step("reproduce56-backup", reproduce_backup),
        *(c for t in REPRODUCE_STEPS for c in TASKS[t][1](defaults)),
        Step("reproduce56-verify", reproduce_verify),
    ]


TASKS["reproduce56"] = (
    "Parts 5-6, after a commit (D65): move their outputs aside, rerun part5 part6 from HEAD, "
    "require identical metrics and model/policy sha256",
    _reproduce56,
)
PART7_STEPS = (
    "rl7-campaign", "rl7-holdout", "rl7-select", "rl7-publish", "rl7-stress", "rl7-sens",
    "rl7-devtest", "rl7-docs",
)  # fmt: skip
PART7_S3_AFTER = "rl7-publish"  # S3 exists from here on: part7 needs configs/pitie_rl_s3.yaml


def _part7(o: argparse.Namespace) -> list[Command]:
    """The Part 7 chain; it stops after `rl7-publish` until configs/pitie_rl_s3.yaml exists."""
    cmds: list[Command] = []
    for t in PART7_STEPS:
        cmds += TASKS[t][1](o)
        if t == PART7_S3_AFTER:
            cmds.append(Step("part7-s3-config", part7_s3_check))
    return cmds


def _reproduce7(o: argparse.Namespace) -> list[Command]:
    """D82: rerun t1 seed 0 (400k), the S3 golden and the page into
    data/rl7/_reproduce/<HEAD>/, then compare. Paths depend on HEAD, so every step runs
    in-process and builds its command when it runs."""
    state: dict = {}
    return [
        Step("reproduce7-prepare", lambda: reproduce7_prepare(state)),
        *(Step(f"reproduce7-{what}", lambda what=what: reproduce7_run(state, what))
          for what in ("train", "golden", "docs")),
        Step("reproduce7-verify", lambda: reproduce7_verify(state)),
    ]  # fmt: skip


TASKS["part7"] = (
    f"RL Part 7 chain ({RL7_CONFIG}): {' '.join(PART7_STEPS)} (stops after "
    f"{PART7_S3_AFTER} until {PITIE_RL_S3} exists)",
    _part7,
)
TASKS["reproduce7"] = (
    "Part 7, after a commit (D82): rerun t1 seed 0 (400k), the S3 golden and "
    f"{RL7_DOC} into data/rl7/_reproduce/<HEAD>/, require identical policies and results",
    _reproduce7,
)


def _part8(o: argparse.Namespace) -> list[Command]:
    """D97: the single Part 8 run, on the clean registration commit. The guard, then
    `rebaseline` with every default (the 70 S0-S2 runs moved aside and re-made, which
    must be identical; `evaluate --run-missing` makes the 40 S3 / S4 runs, the final
    table and the page), the figures and the provenance check."""
    defaults = argparse.Namespace(**{**vars(o), "scenario": None, "baseline": None})
    return [c for t in PART8_STEPS for c in TASKS[t][1](defaults)]


TASKS["part8"] = (
    f"Part 8, once, on the registration commit (D97): {' '.join(PART8_STEPS)}",
    _part8,
)
TASKS["part9"] = (
    f"Part 9 chain: {' '.join(PART9_STEPS)}",
    lambda o: [c for t in PART9_STEPS for c in TASKS[t][1](o)],
)
TASKS["part10"] = (
    f"Part 10 chain: {' '.join(PART10_STEPS)}",
    lambda o: [c for t in PART10_STEPS for c in TASKS[t][1](o)],
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
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="eta-data, eta-data-test: build from an uncommitted tree (dataset id `-dev`); "
        "rl7-campaign, part7: train from one, recorded in every run (D80); usertest-serve: "
        "serve one (ALLOW_DIRTY=1)",
    )
    p.add_argument(
        "--fixtures", action="store_true", help="demo-rehearsal, part10: on the synthetic "
        "api.fixtures in a temporary folder, no real data (FIXTURES=1)",
    )  # fmt: skip
    p.add_argument(
        "--no-browser", action="store_true", help="demo, demo-check, demo-rehearsal, part10: "
        "open no browser, skip the headless-browser checks (NO_BROWSER=1)",
    )  # fmt: skip
    p.add_argument(
        "--record", action="store_true", help="demo-rehearsal, part10: write the A-08/A-09 "
        "evidence docs/demo/rehearsal.json (RECORD=1)",
    )  # fmt: skip
    p.add_argument(
        "--allow-pending-s3", action="store_true", help="usertest-kit: build the kit while "
        "S3 is not yet served by /kpi (Phase 2 gate only) (ALLOW_PENDING_S3=1)",
    )  # fmt: skip
    p.add_argument(
        "--algo", choices=RL_ALGOS, help=f"rl-train: only this algorithm (ALGO); default all "
        f"of {RL_CONFIG} train.s6",
    )  # fmt: skip
    p.add_argument(
        "--seed", nargs="+", help="rl-train: training seeds instead of train.s6's (SEED)"
    )
    p.add_argument("--steps", help="rl-train: decision steps instead of train.s6's (STEPS)")
    p.add_argument("--n-envs", help="rl-train: worker processes instead of train.n_envs (N_ENVS)")
    p.add_argument(
        "--slots", help="rl7-campaign, part7: concurrent trainings instead of rl7.yaml "
        "protocol.slots (SLOTS)",
    )  # fmt: skip
    p.add_argument("-n", "--dry-run", action="store_true", help="print commands, run nothing")
    return p


def _check_int(option: str, value: str, minimum: int = 0) -> None:
    if not value.isdigit() or int(value) < minimum:
        kind = "a positive" if minimum > 0 else "a non-negative"
        raise SystemExit(f"tasks.py: {option} must be {kind} integer, got {value!r}")


def build(args: argparse.Namespace) -> list[Command]:
    for s in args.seeds or ():
        if not s.isdigit():
            raise SystemExit(f"tasks.py: seed must be a non-negative integer, got {s!r}")
    for s in args.seed or ():
        _check_int("--seed", s)
    if args.steps is not None:
        _check_int("--steps", args.steps, minimum=1)
    if args.n_envs is not None:
        _check_int("--n-envs", args.n_envs, minimum=1)
    if args.slots is not None:
        _check_int("--slots", args.slots, minimum=1)
    return [cmd for t in args.tasks for cmd in TASKS[t][1](args)]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return run(build(args), dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

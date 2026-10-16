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
"""

from __future__ import annotations

import argparse
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
INSTALL_EXTRAS = ("dev", "db", "sim", "ml", "api", "rl")
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
    return 1 if differing else check_provenance()


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


def call(cmd: list[str]) -> int:
    """Run one command from the repo root (a Step's subprocess; tests replace it)."""
    return subprocess.call(cmd, cwd=ROOT)


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
        help="eta-data, eta-data-test: build from an uncommitted tree (dataset id `-dev`) "
        "(ALLOW_DIRTY=1)",
    )
    p.add_argument(
        "--algo", choices=RL_ALGOS, help=f"rl-train: only this algorithm (ALGO); default all "
        f"of {RL_CONFIG} train.s6",
    )  # fmt: skip
    p.add_argument(
        "--seed", nargs="+", help="rl-train: training seeds instead of train.s6's (SEED)"
    )
    p.add_argument("--steps", help="rl-train: decision steps instead of train.s6's (STEPS)")
    p.add_argument("--n-envs", help="rl-train: worker processes instead of train.n_envs (N_ENVS)")
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
    return [cmd for t in args.tasks for cmd in TASKS[t][1](args)]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return run(build(args), dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

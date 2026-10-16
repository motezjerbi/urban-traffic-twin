"""Part 7 training campaign (spec S7, 10.4; D69, D71, D73, D74, D76, D84; contract 4.2).

    python -m traffic_twin.rl.campaign --config configs/rl7.yaml --stages T S F
        [--slots N] [--dry-run] [--allow-dirty] [--steps-override N] [--out DIR]
        [--validation-seeds 5100-5109] [--stub]

Runs the pre-registered Part 7 trainings of configs/rl7.yaml as concurrent subprocesses
(`python -m traffic_twin.rl.train --config configs/rl7/<variant>.yaml ...`):

- stage T (D73): `protocol.stage_t.configs` x seeds x `stage_t.steps`;
- stage S (D76): the sensitivity base and settings x their seeds (`seeds_wide` for the
  `wide` settings, `seeds_default` otherwise) x `sensitivity.steps`; a (config, seed,
  steps) that stage T already trains (the base's seeds 0-2) is reused, not rerun;
- stage F (D74): the chosen config C* (`decision_T.json` `chosen`) x `stage_f.seeds` x
  `stage_f.steps`.

At most `--slots` jobs at once (default `protocol.slots`), priorities T > F > S, then
config order of the protocol, then seed. The runs of a stage go under
`<outputs.runs>/<stage>/<run name>` (`rl.train.run_name`). A run whose manifest and
provenance already match (`run_config_sha`, steps) is skipped; a crashed job is rerun
once with `--force`, a second crash marks it failed: no new job starts, the running ones
finish and the campaign exits 2. When every T job is done and `decision_T.json` is
missing, the campaign calls `study.decide_t()` and enqueues F (ahead of S). An
existing `decision_T.json` drives stage F only when it is the stage-T decision of the
current protocol (`check_decision`: its `steps`, `seeds`, run names and `rl7_sha256`
match, its `chosen` is a stage-T config, it is no development-window decision);
otherwise the campaign refuses (ConfigError) instead of training F on it. State:
`<outputs.study>/campaign.json` (per job: status, wall, steps/s, attempts); logs:
`<outputs.study>/logs/<run>.log`. `--dry-run` lists the jobs and runs nothing. Each job
runs with `rl.train.CHILD_ENV` (`OPENBLAS_NUM_THREADS=1` unless set): about 0.3 GB of
commit per process instead of 1 GB, 13 processes per job (learner, 6 envs, 6 pool workers).
`--eval-pool transient` passes `rl.train --eval-pool transient` to every job: the 6 pool
workers then live only around each validation (same results, less memory per job).

`--steps-override N` and `--out DIR` serve the Phase 2 gate's mini-campaign: every job
trains N steps, runs and state go under DIR (DIR/runs/<stage>/, DIR/campaign.json,
DIR/logs/). Such a run is never a result, so unless `--validation-seeds` is given it
validates on the development windows `DEV_VALIDATION_SEEDS` (5100-5109), never on the
pre-registered selection windows. `--steps-override`, and validation windows other than
`seeds.selection`, need `--out`: such runs and decisions never go to the registered
roots `outputs.runs` / `outputs.study`. With `--out`, stage F needs an existing
`DIR/decision_T.json` made for these runs (`study decide-t --runs DIR/runs/T --out DIR
--steps N --validation-seeds ...`).
`--stub` trains on the SUMO-free stub env (tests).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from traffic_twin.config import ConfigError, load_config, text_sha256
from traffic_twin.paths import RL7_CONFIG
from traffic_twin.rl.train import (
    CHILD_ENV,
    EVAL_POOLS,
    PROVENANCE_FILE,
    check_clean,
    parse_seeds,
    resolve_algo,
    run_config_sha,
    run_name,
    seed_range,
)

log = logging.getLogger(__name__)

STAGES = ("T", "S", "F")
PRIORITY = {"T": 0, "F": 1, "S": 2}  # lower runs first (D84)
CAMPAIGN_FILE = "campaign.json"  # under rl7.yaml outputs.study
LOGS_DIR = "logs"  # under rl7.yaml outputs.study
DECISION_T_FILE = "decision_T.json"  # = rl.study.DECISION_T_FILE, under outputs.study
RUN_CONFIG_DIR = "rl7"  # run configs next to rl7.yaml: <rl7 folder>/rl7/<name>.yaml
FAILED_EXIT = 2  # a job crashed twice
MAX_ATTEMPTS = 2  # one automatic rerun per crash
TRAIN_MODULE = "traffic_twin.rl.train"
POLL_S = 2.0  # seconds between two looks at the running jobs
DEV_VALIDATION_SEEDS = tuple(range(5100, 5110))  # --steps-override runs (gate, dev)


@dataclass(frozen=True)
class Job:
    """One training run of the campaign."""

    stage: str  # "T", "S" or "F"
    config: Path  # configs/rl7/<variant>.yaml
    algo: str  # "dqn" or "ppo" (the config's `algo`, default dqn)
    seed: int
    steps: int
    priority: int  # PRIORITY[stage]
    out: Path  # runs root of the stage: <outputs.runs>/<stage>
    run: str = ""  # run folder name (rl.train.run_name), <out>/<run>

    @property
    def run_dir(self) -> Path:
        return self.out / self.run


# -- the plan --------------------------------------------------------------------------
def _roots(cfg: dict, out: Path | None) -> tuple[Path, Path]:
    """(study root, runs root): rl7.yaml `outputs.study` / `outputs.runs`, or `out` and
    `out/runs`."""
    if out is not None:
        return Path(out), Path(out) / "runs"
    outputs = cfg.get("outputs") or {}
    try:
        return Path(outputs["study"]), Path(outputs["runs"])
    except KeyError as e:
        raise ConfigError(f"rl7 config: missing outputs.{e.args[0]}") from None


def run_config_path(rl7: Path, name: str) -> Path:
    """`<folder of rl7.yaml>/rl7/<name>.yaml`."""
    path = Path(rl7).parent / RUN_CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"run config {name!r} not found: {path.as_posix()}")
    return path


def read_decision(study: Path) -> dict | None:
    """`decision_T.json` under the study root, or None (raw: `check_decision` says
    whether it belongs to the current protocol)."""
    path = Path(study) / DECISION_T_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validation_windows(
    cfg: dict, steps_override: int | None = None, validation_seeds: Sequence[int] | None = None
) -> list[int]:
    """The validation windows the campaign's jobs use: `validation_seeds`, else the
    development windows with `steps_override`, else the registered `seeds.selection`."""
    if validation_seeds:
        return [int(s) for s in validation_seeds]
    if steps_override is not None:
        return list(DEV_VALIDATION_SEEDS)
    try:
        return seed_range(cfg["seeds"]["selection"])
    except (KeyError, TypeError):
        raise ConfigError("rl7 config: missing seeds.selection") from None


def stage_t_runs(
    rl7: Path, cfg: dict | None = None, steps_override: int | None = None
) -> dict[str, dict[str, str]]:
    """{config: {str(seed): run name}} of the current stage-T jobs (what `study.decide_t`
    records as `runs`: the run folder names carry `run_config_sha` and the steps)."""
    rl7 = Path(rl7)
    cfg = load_config(rl7) if cfg is None else cfg
    ident = _Identity()
    out: dict[str, dict[str, str]] = {}
    for name, seeds, steps in _stage_plan(cfg, "T", rl7):
        algo, variant, sha = ident(run_config_path(rl7, name))
        steps = int(steps_override or steps)
        out[name] = {str(s): run_name(algo, s, steps, sha, variant) for s in seeds}
    return out


def check_decision(
    rl7: Path,
    decision: dict,
    *,
    steps_override: int | None = None,
    validation_seeds: Sequence[int] | None = None,
    cfg: dict | None = None,
    where: Path | None = None,
) -> str:
    """C* (`decision["chosen"]`) when `decision` is the stage-T decision of the CURRENT
    protocol for this campaign: `steps` = `stage_t.steps` (or `steps_override`), `seeds`
    = `stage_t.seeds`, `runs` = the current stage-T run names (`stage_t_runs`: same run
    configs, same steps), `rl7_sha256` = the current rl7.yaml, `chosen` a stage-T config,
    and, when recorded, the same validation windows and no development decision for a
    registered campaign (no override, `seeds.selection` windows). ConfigError otherwise:
    a gate, dev or stale decision must never drive stage F (nor `select`). `where`: the
    file, for the message."""
    rl7 = Path(rl7)
    cfg = load_config(rl7) if cfg is None else cfg
    st = (cfg.get("protocol") or {}).get("stage_t") or {}
    steps = int(steps_override or st["steps"])
    problems: list[str] = []
    chosen = decision.get("chosen")
    configs = [str(c) for c in st.get("configs") or ()]
    if not chosen:
        problems.append("no `chosen` config")
    elif str(chosen) not in configs:
        problems.append(f"chosen {chosen!r} is not a stage_t config {configs}")
    if decision.get("steps") != steps:
        problems.append(f"steps {decision.get('steps')!r} != {steps}")
    seeds = [int(s) for s in st.get("seeds") or ()]
    got_seeds = decision.get("seeds")
    if not isinstance(got_seeds, list) or [int(s) for s in got_seeds] != seeds:
        problems.append(f"seeds {got_seeds!r} != stage_t.seeds {seeds}")
    expected_runs = stage_t_runs(rl7, cfg, steps_override)
    got_runs = decision.get("runs")
    if got_runs != expected_runs:
        problems.append(
            f"runs {got_runs!r} are not the current stage-T runs {expected_runs!r} "
            "(another run config version or budget)"
        )
    sha = text_sha256(rl7)
    if decision.get("rl7_sha256") != sha:
        problems.append(f"rl7_sha256 {decision.get('rl7_sha256')!r} != {rl7.as_posix()} {sha!r}")
    windows = validation_windows(cfg, steps_override, validation_seeds)
    if "validation_seeds" in decision:
        got = decision.get("validation_seeds") or []
        if sorted(int(s) for s in got) != sorted(windows):
            problems.append(f"validation_seeds {got!r} != this campaign's windows {windows}")
    registered = steps_override is None and sorted(windows) == validation_windows(cfg)
    if decision.get("development") and registered:
        problems.append("a development-window decision (gate or dev), never a result")
    if problems:
        path = (Path(where) if where else Path("decision_T.json")).as_posix()
        raise ConfigError(
            f"{path} is not the stage-T decision of the current protocol: "
            + "; ".join(problems)
            + ". Remove it, or remake it from the current stage-T runs "
            "(`rl.study decide-t --force`)."
        )
    return str(chosen)


class _Identity:
    """(algo, variant, run_config_sha) per run config, loaded once."""

    def __init__(self) -> None:
        self._cache: dict[Path, tuple[str, str | None, str]] = {}

    def __call__(self, config: Path) -> tuple[str, str | None, str]:
        key = Path(config)
        if key not in self._cache:
            cfg = load_config(key)
            variant = cfg.get("variant")
            if variant is None:
                raise ConfigError(f"{key.as_posix()}: a campaign run config needs `variant`")
            self._cache[key] = (resolve_algo(cfg), str(variant), run_config_sha(cfg))
        return self._cache[key]


def _stage_plan(
    cfg: dict,
    stage: str,
    rl7: Path,
    study: Path | None = None,
    steps_override: int | None = None,
    validation_seeds: Sequence[int] | None = None,
) -> list[tuple[str, list[int], int]] | None:
    """[(config name, seeds, steps)] of a stage in protocol order; None for F before
    `decision_T.json` exists under `study` (an existing one must pass `check_decision`)."""
    proto = cfg.get("protocol") or {}
    if stage == "T":
        t = proto["stage_t"]
        return [(str(c), [int(s) for s in t["seeds"]], int(t["steps"])) for c in t["configs"]]
    if stage == "S":
        s = proto["sensitivity"]
        wide = {str(c) for c in s.get("wide") or ()}
        names = [str(s["base"]), *[str(c) for c in s["settings"]]]
        return [
            (n, [int(x) for x in (s["seeds_wide"] if n in wide else s["seeds_default"])],
             int(s["steps"]))
            for n in names
        ]  # fmt: skip
    if stage == "F":
        if study is None:
            raise ConfigError("stage F: no study root")
        decision = read_decision(study)
        if decision is None:
            return None
        chosen = check_decision(
            rl7,
            decision,
            steps_override=steps_override,
            validation_seeds=validation_seeds,
            cfg=cfg,
            where=Path(study) / DECISION_T_FILE,
        )
        f = proto["stage_f"]
        return [(str(chosen), [int(x) for x in f["seeds"]], int(f["steps"]))]
    raise ConfigError(f"unknown stage {stage!r} (expected one of {STAGES})")


def jobs(
    rl7: Path,
    stages: Sequence[str],
    *,
    steps_override: int | None = None,
    out: Path | None = None,
    validation_seeds: Sequence[int] | None = None,
) -> list[Job]:
    """The campaign's jobs for `stages`, in run order (priority, then config, then seed).
    Stage F needs `decision_T.json` (its jobs are listed once C* is decided; a decision
    that is not the current protocol's raises ConfigError, `check_decision`). A stage-S
    (config, seed, steps) that stage T trains is left out (reused, D76)."""
    rl7 = Path(rl7)
    cfg = load_config(rl7)
    study, runs = _roots(cfg, out)
    ident = _Identity()
    t_plan = _stage_plan(cfg, "T", rl7) or []
    t_keys = {
        (name, seed, int(steps_override or steps))
        for name, seeds, steps in t_plan
        for seed in seeds
    }
    out_jobs: list[Job] = []
    for stage in [s for s in STAGES if s in set(stages)]:
        plan = _stage_plan(cfg, stage, rl7, study, steps_override, validation_seeds)
        for name, seeds, steps in plan or []:
            steps = int(steps_override or steps)
            path = run_config_path(rl7, name)
            algo, variant, sha = ident(path)
            for seed in seeds:
                if stage == "S" and (name, seed, steps) in t_keys:
                    continue  # the stage-T run is the sensitivity base run
                out_jobs.append(
                    Job(
                        stage=stage,
                        config=path,
                        algo=algo,
                        seed=int(seed),
                        steps=steps,
                        priority=PRIORITY[stage],
                        out=runs / stage,
                        run=run_name(algo, seed, steps, sha, variant),
                    )
                )
    return sorted(out_jobs, key=lambda j: j.priority)  # stable: protocol order, then seed


def finished(job: Job) -> bool:
    """The run folder holds a finished run of this job: manifest.json and provenance.json
    with the job's `run_config_sha` and steps."""
    manifest, prov = job.run_dir / "manifest.json", job.run_dir / PROVENANCE_FILE
    if not (manifest.is_file() and prov.is_file()):
        return False
    try:
        m = json.loads(manifest.read_text(encoding="utf-8"))
        p = json.loads(prov.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    _, _, sha = _Identity()(job.config)
    return (
        p.get("run_config_sha") == sha
        and p.get("steps") == job.steps
        and m.get("steps") == job.steps
    )


# -- running ---------------------------------------------------------------------------
def train_command(
    job: Job,
    *,
    force: bool = False,
    allow_dirty: bool = False,
    validation_seeds: Sequence[int] | None = None,
    stub: bool = False,
    eval_pool: str | None = None,
) -> list[str]:
    """The `rl.train` command of a job."""
    cmd = [
        sys.executable, "-m", TRAIN_MODULE,
        "--config", job.config.as_posix(), "--algo", job.algo, "--seed", str(job.seed),
        "--steps", str(job.steps), "--out", job.out.as_posix(),
    ]  # fmt: skip
    if validation_seeds:
        cmd += ["--validation-seeds", *[str(s) for s in validation_seeds]]
    if allow_dirty:
        cmd.append("--allow-dirty")
    if force:
        cmd.append("--force")
    if stub:
        cmd.append("--stub")
    if eval_pool:
        cmd += ["--eval-pool", eval_pool]
    return cmd


def train_env() -> dict[str, str]:
    """Environment of a training subprocess: this one's plus `rl.train.CHILD_ENV` where
    unset (one OpenBLAS thread: the learner then commits about 0.3 GB instead of 1 GB;
    its env and pool workers inherit it)."""
    env = dict(os.environ)
    for k, v in CHILD_ENV.items():
        env.setdefault(k, str(v))
    return env


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _job_record(job: Job) -> dict[str, Any]:
    return {
        "run": job.run,
        "stage": job.stage,
        "config": job.config.as_posix(),
        "algo": job.algo,
        "seed": job.seed,
        "steps": job.steps,
        "out": job.run_dir.as_posix(),
        "status": "pending",
        "attempts": 0,
        "wall_s": None,
        "steps_per_s": None,
        "returncode": None,
        "log": None,
        "started_at": None,
        "finished_at": None,
    }


def _write_state(path: Path, state: dict) -> None:
    state["updated_at"] = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _manifest_rate(job: Job) -> float | None:
    try:
        m = json.loads((job.run_dir / "manifest.json").read_text(encoding="utf-8"))
        return float(m["steps_per_s"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _print_plan(planned: Sequence[Job], f_pending: bool) -> None:
    for j in planned:
        status = "done" if finished(j) else "todo"
        print(f"{j.stage}  {status:4}  {j.run}  ({j.config.as_posix()}, {j.steps} steps)")
    counts = {s: sum(j.stage == s for j in planned) for s in STAGES}
    print("jobs: " + ", ".join(f"{s} {n}" for s, n in counts.items()))
    if f_pending:
        print("F: listed once decision_T.json exists (stage T, then study decide-t)")


def run(
    rl7: Path,
    stages: Sequence[str],
    slots: int | None = None,
    dry_run: bool = False,
    allow_dirty: bool = False,
    steps_override: int | None = None,
    out: Path | None = None,
    *,
    validation_seeds: Sequence[int] | None = None,
    stub: bool = False,
    eval_pool: str | None = None,
) -> dict:
    """Run the jobs of `stages` with at most `slots` at once; returns the campaign state
    (also written to `campaign.json`). `steps_override` and `out` serve the Phase 2 gate's
    mini-campaign (`--steps-override 20000 --out data/rl7/_gate`). `eval_pool` is passed
    to every job (`rl.train --eval-pool`; None: rl.train's default, persistent)."""
    rl7 = Path(rl7)
    if eval_pool is not None and eval_pool not in EVAL_POOLS:
        raise ConfigError(f"eval_pool must be one of {EVAL_POOLS}, not {eval_pool!r}")
    unknown = sorted(set(stages) - set(STAGES))
    if unknown or not stages:
        raise ConfigError(f"stages must be among {STAGES}, not {unknown or list(stages)}")
    stages = [s for s in STAGES if s in set(stages)]
    cfg = load_config(rl7)
    study, runs_root = _roots(cfg, out)
    slots = int(slots or (cfg.get("protocol") or {}).get("slots") or 1)
    if slots < 1:
        raise ConfigError(f"slots must be >= 1, not {slots}")
    registered = validation_windows(cfg)
    if steps_override is not None and validation_seeds is None:
        validation_seeds = list(DEV_VALIDATION_SEEDS)  # never a result: dev windows only
    if out is None and (
        steps_override is not None
        or (validation_seeds and sorted(int(s) for s in validation_seeds) != registered)
    ):
        raise ConfigError(
            "--steps-override and development validation windows need --out: such runs "
            f"never go to the registered roots ({study.as_posix()})"
        )
    if out is not None and "F" in stages and read_decision(study) is None:
        raise ConfigError(
            f"--out {Path(out).as_posix()} with stage F needs {DECISION_T_FILE} there, made "
            f"for these runs (`rl.study decide-t --runs {Path(out).as_posix()}/runs/T --out "
            f"{Path(out).as_posix()} --steps N --validation-seeds ...`)"
        )
    if "F" in stages and "T" not in stages and read_decision(study) is None:
        raise ConfigError(f"stage F needs {(study / DECISION_T_FILE).as_posix()}: run stage T")

    planned = jobs(
        rl7, stages, steps_override=steps_override, out=out, validation_seeds=validation_seeds
    )
    f_pending = "F" in stages and read_decision(study) is None
    if dry_run:
        _print_plan(planned, f_pending)
        return {"dry_run": True, "jobs": [_job_record(j) for j in planned], "f_pending": f_pending}
    check_clean(cfg, rl7, allow_dirty)

    state: dict[str, Any] = {
        "config": rl7.as_posix(),
        "stages": stages,
        "slots": slots,
        "steps_override": steps_override,
        "validation_seeds": list(validation_seeds) if validation_seeds else None,
        "allow_dirty": bool(allow_dirty),
        "eval_pool": eval_pool or "persistent",
        "started_at": _now(),
        "status": "running",
        "decision_T": None,
        "jobs": {},
    }
    state_file, logs = study / CAMPAIGN_FILE, study / LOGS_DIR
    logs.mkdir(parents=True, exist_ok=True)
    queue: list[Job] = []
    records: dict[str, dict] = state["jobs"]

    def enqueue(new: Sequence[Job]) -> None:
        for j in new:
            if j.run in records:
                continue
            records[j.run] = _job_record(j)
            if finished(j):
                records[j.run]["status"] = "skipped"
            else:
                queue.append(j)
        queue.sort(key=lambda j: j.priority)  # stable

    enqueue(planned)
    running: dict[str, tuple[Job, subprocess.Popen, float, Any]] = {}
    retry: set[str] = set()
    stop = False
    f_done = not f_pending
    _write_state(state_file, state)

    def launch(job: Job) -> None:
        rec = records[job.run]
        rec["attempts"] += 1
        cmd = train_command(
            job,
            force=job.run in retry,
            allow_dirty=allow_dirty,
            validation_seeds=validation_seeds,
            stub=stub,
            eval_pool=eval_pool,
        )
        log_path = logs / f"{job.run}.log"
        fh = log_path.open("a", encoding="utf-8")
        fh.write(f"== attempt {rec['attempts']} at {_now()}: {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=train_env())
        rec.update(status="running", log=log_path.as_posix(), started_at=_now())
        running[job.run] = (job, proc, time.perf_counter(), fh)
        log.info("started %s (attempt %d)", job.run, rec["attempts"])

    try:
        while True:
            changed = False
            while not stop and queue and len(running) < slots:
                launch(queue.pop(0))
                changed = True
            if changed:
                _write_state(state_file, state)
                changed = False
            for name, (job, proc, t0, fh) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                fh.close()
                del running[name]
                changed = True
                rec = records[name]
                rec.update(
                    returncode=rc, wall_s=round(time.perf_counter() - t0, 1), finished_at=_now()
                )
                if rc == 0 and finished(job):
                    rec.update(status="done", steps_per_s=_manifest_rate(job))
                    log.info("done %s (%.0f s)", name, rec["wall_s"])
                elif rec["attempts"] < MAX_ATTEMPTS:
                    rec["status"] = "retry"
                    retry.add(name)
                    queue.insert(0, job)
                    queue.sort(key=lambda j: j.priority)
                    log.warning("%s crashed (exit %s): one rerun with --force", name, rc)
                else:
                    rec["status"] = "failed"
                    stop = True
                    log.error("%s failed twice (exit %s): no new job starts", name, rc)
            t_open = any(
                r["stage"] == "T" and r["status"] not in ("done", "skipped")
                for r in records.values()
            )
            if not f_done and not stop and not t_open:
                if read_decision(study) is None:
                    from traffic_twin.rl import study as study_mod

                    log.info("stage T complete: study.decide_t")
                    kw = {"validation_seeds": list(validation_seeds)} if validation_seeds else {}
                    study_mod.decide_t(
                        rl7, runs=runs_root / "T", out=study, steps=steps_override, **kw
                    )
                decision = read_decision(study)
                if decision is None:
                    raise ConfigError(f"study.decide_t wrote no {DECISION_T_FILE} in {study}")
                f_jobs = jobs(
                    rl7,
                    ["F"],
                    steps_override=steps_override,
                    out=out,
                    validation_seeds=validation_seeds,
                )  # check_decision: the decision just made must be the current protocol's
                state["decision_T"] = {"chosen": decision.get("chosen")}
                enqueue(f_jobs)
                f_done, changed = True, True
            if changed:
                _write_state(state_file, state)
            if not running and (stop or not queue):
                break
            if running:
                time.sleep(POLL_S)
    except BaseException:
        for _, proc, _, _ in running.values():
            proc.terminate()
        for _, proc, _, fh in running.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            fh.close()
        for name in running:
            records[name]["status"] = "interrupted"
        state["status"] = "interrupted"
        _write_state(state_file, state)
        raise
    failed = [r["run"] for r in records.values() if r["status"] == "failed"]
    state["status"] = "failed" if failed else "done"
    if stop:
        state["not_started"] = [r["run"] for r in records.values() if r["status"] == "pending"]
    if "F" in stages and not f_done:
        state["f_pending"] = True
    _write_state(state_file, state)
    counts: dict[str, int] = {}
    for r in records.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    log.info("campaign %s: %s", state["status"], counts)
    return state


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, default=RL7_CONFIG)
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--slots", type=int, help="concurrent jobs (default: protocol.slots)")
    p.add_argument("--dry-run", action="store_true", help="list the jobs, run nothing")
    p.add_argument("--allow-dirty", action="store_true", help="train from a dirty tree (D80)")
    p.add_argument("--steps-override", type=int, help="decision steps of every job (gate only)")
    p.add_argument("--out", type=Path, help="study root (default: rl7.yaml outputs.study)")
    p.add_argument(
        "--validation-seeds",
        nargs="+",
        metavar="SEED",
        help="validation windows of every job: N or LO-HI (default: seeds.selection; "
        "5100-5109 with --steps-override)",
    )
    p.add_argument(
        "--eval-pool",
        choices=EVAL_POOLS,
        help="validation pool lifetime of every job (rl.train --eval-pool; default persistent)",
    )
    p.add_argument("--stub", action="store_true", help="SUMO-free stub env (tests only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        state = run(
            args.config,
            args.stages,
            args.slots,
            dry_run=args.dry_run,
            allow_dirty=args.allow_dirty,
            steps_override=args.steps_override,
            out=args.out,
            validation_seeds=parse_seeds(args.validation_seeds) if args.validation_seeds else None,
            stub=args.stub,
            eval_pool=args.eval_pool,
        )
    except ConfigError as e:
        raise SystemExit(f"error: {e}") from None
    return FAILED_EXIT if state.get("status") == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())

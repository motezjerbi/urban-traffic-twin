"""Part 7 study: stage-T decision, holdout, S3 selection and publication, golden, stress
panel and the single dev-test look (spec S7, 10.4, 10.6; D72-D75, D78, D79, D86;
contract 4.2).

    python -m traffic_twin.rl.study [--config configs/rl7.yaml] decide-t [--force]
    python -m traffic_twin.rl.study holdout RUN... [--curves] [--workers N]
    python -m traffic_twin.rl.study select [--workers N]
    python -m traffic_twin.rl.study publish [--out models/rl]
    python -m traffic_twin.rl.study golden --config configs/pitie_rl_s3.yaml [--out FILE]
    python -m traffic_twin.rl.study stress [--workers N]
    python -m traffic_twin.rl.study devtest [--workers N]

Outputs (all under rl7.yaml `outputs.study`, i.e. data/rl7/, except `publish` and
`golden`):

- `decide-t` (D73) -> `decision_T.json`: {scores{config:{seed:late_mean}}, means, ped_s,
  rule, outcome, chosen, created_at}. Each stage-T run (configs/rl7/<config>.yaml x
  seeds, `stage_t.steps`, the current `run_config_sha`) gives its `late_mean` and late
  mean `pedestrian_s` (validations at >= `train.pick_from` x steps, D72). C* = the
  reference (T1) unless a candidate X has (a) score <= (1 - margin_rel) x score(T1), (b)
  a lower late_mean than T1 on >= min_seed_wins of the matching seeds and (c) a late mean
  pedestrian_s <= (1 + ped_guard_rel) x T1's; the qualifier with the lowest score wins.
  An existing decision is never overwritten without `--force`;
- `holdout` -> `<run>/holdout.csv` (the run's pick on the holdout windows, train mode,
  and on the `holdout.full` seeds in full mode; reporting only, never to select) and,
  with `--curves`, `<run>/holdout_curve.csv` (every kept checkpoint `val/step_<n>.zip` on
  the holdout windows). S0, S1, S2 and always-stay are played once per environment
  (reward, decision, episode and randomization blocks) and shared:
  `holdout/baselines_<env_key>.csv`; `holdout/summary.csv` keeps one row per scored run;
- `select` (D75) -> `select/{stage1.csv, stage2.csv, decision.json}`: stage 1 = each
  stage-F run's pick (C* of decision_T.json, `stage_f` seeds and steps); stage 2 = each
  pick on the `select2` seeds in `stage2_mode` (full 2 h env episodes; S0 / S1 / S2 on
  the same episodes for context). S3 = lowest stage-2 mean cost; candidates within
  `tie_rel` of it: lower stage-1 cost, then lower seed. decision.json = {candidates[{run,
  pick_step, stage1, stage2, holdout_w30, holdout_full, late_mean, ...}], chosen, rule};
- `publish` (D86) -> `models/rl/<model_version>/{model.zip, manifest.json,
  validation.csv}` with model_version `rl7-<algo>-<variant>-s<seed>-<policy_sha8>`;
  refuses a pick whose file or policy sha256 differs from its run's records or from the
  selection; text files are written with LF line ends (`.gitattributes`: `models/**
  -text`); prints the configs/pitie_rl_s3.yaml lines, never edits a config;
- `golden` -> tests/data/rl_s3_golden.json: the S3 policy's Q-values (PPO: action
  logits) and actions on 29 observations of a full-mode, non-randomized 900 s episode of
  training seed 5000, and the phase hash of a 900 s `run_scenario` of the S3 scenario on
  seed 5000 (`tests/test_rl_s3_model.py`);
- `stress` (D78) -> `stress/{episodes.csv, summary.csv}`: S3, S0, S1 and S2 on the
  stress seeds under every `protocol.stress.cells` condition (`params_override`, every
  other draw the seed's own, so the episodes are paired by seed); per cell the cost of S3
  vs S1 and S2 (ratio of means, exact Wilcoxon), pedestrian waiting and penalty seconds,
  interventions and their D77 split; a cell where S3 is more than `worse_than_s1_rel`
  worse than S1 is flagged. Reported, never gating;
- `devtest` (D79, once) -> `devtest/{compare.csv, kpi.csv, kpi_safety.csv, paired.csv,
  verdict.json}`: the 30-min window comparison (S3, S0, S1, S2, random, stay), full 2 h
  KPI v3 runs of pitie_{fixed, actuated, rules} and the S3 scenario on the dev-test seeds,
  the pairs of `protocol.devtest.pairs` through `evaluation.compare` (exact Wilcoxon, HL,
  Holm over the 7 headline KPIs within each pair) and the interim 10.6 verdict (C1-C4,
  `compute_verdict`). Refuses to run when `verdict.json` exists; `devtest/look.json`
  records the look (model, scenario, seeds, rl7 sha) before anything is played, so a
  crash allows only a resume of that same look (`devtest/attempts.jsonl`). verdict.json =
  {C1{hl, p, pass}, C2{rows, pass}, C3{violations, pass}, C4{runs_below_s1, runs_pass,
  repro, pass}, stress_cells_worse, s2_framing, verdict: "pass" | "fallback" | "pending",
  pending, seeds, model_version, look}. reproduce7 runs after the commit (D82), so at the
  look C4 is usually pending (`pass` None, verdict "pending" when C1-C3 and the C4 run
  count hold); `final_verdict` / `resolve_verdict` complete it from reproduce7.json
  without a second look, and a page states that value.

Validation windows: decide-t and select accept only runs whose provenance
`selection_seeds` are the registered `seeds.selection` (D70); development windows
(`--validation-seeds`, training seeds) only into a study root other than
`outputs.study`.

Seeds: every subcommand takes only its own rl7.yaml range (holdout, select2, stress,
dev_test; golden a training seed) and never a Part 3-4 seed of configs/evaluation.yaml.
Read-only imports: rl.evaluate.{compare_frame, kpi_frames, kpi_runs, ...},
evaluation.compare. Nothing is written under data/results, data/rl_runs or data/models.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import inspect
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_twin.config import ConfigError, get, load_config, load_scenario, text_sha256
from traffic_twin.evaluation.compare import better_if, compare, worse_than
from traffic_twin.evaluation.stats import wilcoxon
from traffic_twin.paths import (
    EVALUATION_CONFIG,
    PITIE_ACTUATED_CONFIG,
    PITIE_FIXED_CONFIG,
    PITIE_RL_S3_CONFIG,
    PITIE_RULES_CONFIG,
    RL7_CONFIG,
    RL7_DIR,
)
from traffic_twin.rl.evaluate import check_seeds, compare_frame, kpi_configs, kpi_frames, kpi_runs
from traffic_twin.rl.train import (
    TOTAL_KEYS,
    EnvBackend,
    check_out_root,
    file_sha256,
    library_versions,
    load_model,
    model_policy,
    parse_seeds,
    pct_change,
    policy_sha256,
    real_backend,
    seed_range,
    set_torch_threads,
    summary_row,
)
from traffic_twin.sim.results import code_version

log = logging.getLogger(__name__)

# file names under rl7.yaml outputs.study (data/rl7/)
DECISION_T_FILE = "decision_T.json"
HOLDOUT_DIR = "holdout"  # shared holdout baselines and summary.csv
HOLDOUT_FILE = "holdout.csv"  # inside each run folder
HOLDOUT_CURVE_FILE = "holdout_curve.csv"  # inside each run folder (--curves)
HOLDOUT_SUMMARY_FILE = "summary.csv"  # under HOLDOUT_DIR
SELECT_DIR = "select"  # stage1.csv, stage2.csv, decision.json
SELECT_DECISION_FILE = "decision.json"
STRESS_DIR = "stress"  # episodes.csv, summary.csv
DEVTEST_DIR = "devtest"  # compare.csv, kpi.csv, kpi_safety.csv, paired.csv, verdict.json
VERDICT_FILE = "verdict.json"
LOOK_FILE = "look.json"  # under DEVTEST_DIR: what the single dev-test look is of (D79)
ATTEMPTS_FILE = "attempts.jsonl"  # under DEVTEST_DIR: every start, crash and finish
LOOK_KEYS = (  # a resume after a crash must match all of them
    "model_sha256", "model_version", "selected_run", "scenario", "scenario_sha256",
    "rl_config", "rl_config_sha256", "seeds", "rl7_sha256",
)  # fmt: skip
REPRODUCE_DIR = "_reproduce"  # tasks.py reproduce7: _reproduce/<HEAD>/reproduce7.json
REPRODUCE_FILE = "reproduce7.json"
WORK_DIR = "_work"  # env work folders (route files, SUMO outputs)
GOLDEN_FILE = Path("tests") / "data" / "rl_s3_golden.json"
GOLDEN_SEED = 5000  # a Part 7 training seed
GOLDEN_N_OBS = 29
GOLDEN_N_CLOSE = 8  # of which the closest calls (smallest |Q1 - Q0| or logit gap)
GOLDEN_DURATION_S = 900

# run folders and their files (written by rl.train, D71, D72, D81)
MANIFEST_FILE = "manifest.json"
PROVENANCE_FILE = "provenance.json"
PICK_MODEL_FILE = "pick_model.zip"
VAL_DIR = "val"  # val/step_<n>.zip
STAGES = ("T", "S", "F")

# policies
HOLDOUT_BASELINES = {"S0": "fixed", "S1": "actuated", "S2": "rules", "stay": "stay"}
STAGE2_BASELINES = {"S0": "fixed", "S1": "actuated", "S2": "rules"}
STRESS_LABELS = {"S3": "model", "S0": "fixed", "S1": "actuated", "S2": "rules"}
WINDOW_POLICIES = {
    "S3": "rl",
    "S0": "fixed",
    "S1": "actuated",
    "S2": "rules",
    "random": "random",
    "stay": "stay",
}
KPI_BASELINES = {"S0": PITIE_FIXED_CONFIG, "S1": PITIE_ACTUATED_CONFIG, "S2": PITIE_RULES_CONFIG}
CANDIDATE = "S3"
# the Holm family of each dev-test pair (D79: "Holm over the 7 KPIs within each pair")
HEADLINE7 = (
    ("car", "time_loss_mean"),
    ("bus", "delay_mean"),
    ("pedestrian", "ped_curb_wait_mean"),
    ("pedestrian", "ped_curb_wait_max"),
    ("ambulance", "travel_time_total"),
    ("all", "person_delay_mean"),
    ("car", "depart_delay_mean"),
)
SPLIT_KEYS = ("cut", "replaced", "cut_foreseeable")  # D77 (episode_summary["iv_split"])
REL_TOL = 1e-9  # float slack on the pre-registered `<=` edges (a tie at the edge passes)
FORBIDDEN_OUT = (Path("data") / "rl_runs", Path("data") / "models")  # Part 6 record


# -- small helpers ---------------------------------------------------------------------
def _now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Path):
        return x.as_posix()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON serializable: {type(x).__name__}")


def _clean(obj: Any) -> Any:
    """NaN / inf -> None (strict JSON), recursively."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    return obj


def json_text(obj: Any) -> str:
    """Indented strict JSON (NaN as null), ASCII, LF line ends and a final newline."""
    return json.dumps(_clean(obj), indent=2, default=_jsonable, ensure_ascii=True) + "\n"


def write_json(path: Path, obj: Any, exclusive: bool = False) -> Path:
    """Write `json_text(obj)`; `exclusive`: fail (FileExistsError) when `path` exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as f:
        f.write(json_text(obj))
    return path


def read_json(path: Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def check_out(path: Path) -> None:
    """Never under data/results (`train.check_out_root`), nor under the Part 6 record
    data/rl_runs and data/models."""
    check_out_root(path)
    p = Path(path).resolve()
    for forbidden in FORBIDDEN_OUT:
        f = forbidden.resolve()
        if p == f or f in p.parents:
            raise ConfigError(f"{path}: Part 7 never writes under {forbidden.as_posix()}")


def default_workers() -> int:
    return max(1, min(16, (os.cpu_count() or 2) - 2))


def _float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def _mean(values: Sequence[float]) -> float:
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(v)) if v else math.nan


# -- config ------------------------------------------------------------------------------
def protocol(rl_cfg: Mapping) -> dict:
    p = rl_cfg.get("protocol")
    if not isinstance(p, Mapping):
        raise ConfigError("the Part 7 config has no `protocol` block (configs/rl7.yaml)")
    return dict(p)


def study_root(rl_cfg: Mapping, out: Path | None = None) -> Path:
    return Path(out or (rl_cfg.get("outputs") or {}).get("study") or RL7_DIR)


def runs_root(rl_cfg: Mapping, runs: Path | None = None) -> Path:
    return Path(runs or (rl_cfg.get("outputs") or {}).get("runs") or RL7_DIR / "runs")


def run_config_path(rl7: Path, name: str) -> Path:
    """configs/rl7/<name>.yaml next to the Part 7 config `rl7`."""
    return Path(rl7).parent / "rl7" / f"{name}.yaml"


def config_sha(config: Path) -> str:
    """`rl.train.run_config_sha` of a run config (D69): the identity its runs carry."""
    from traffic_twin.rl import train

    return train.run_config_sha(load_config(config))


def env_key(rl_cfg: Mapping) -> str:
    """12 hex digits naming what a baseline episode's cost depends on: base scenario,
    action, decision, reward, episode and randomization blocks (not the observation)."""
    keys = ("base_config", "action", "decision", "reward", "episode", "randomize")
    payload = json.dumps({k: rl_cfg.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _seeds_of(rl_cfg: Mapping, role: str) -> list[int]:
    try:
        return seed_range(rl_cfg["seeds"][role])
    except KeyError:
        raise ConfigError(f"rl7.yaml seeds has no {role!r} range") from None


# -- training runs -----------------------------------------------------------------------
def read_run(run_dir: Path) -> dict:
    """manifest.json of a run folder with the provenance.json keys it lacks, plus `dir`
    and `stage` (the parent folder name when it is T, S or F)."""
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / MANIFEST_FILE)
    if manifest is None:
        raise ConfigError(f"{run_dir}: no {MANIFEST_FILE} (not a finished training run)")
    prov = read_json(run_dir / PROVENANCE_FILE) or {}
    m = {**prov, **manifest}
    m["dir"] = run_dir
    stage = run_dir.parent.name
    m["stage"] = stage if stage in STAGES else m.get("stage")
    return m


def find_runs(root: Path) -> list[Path]:
    """Run folders (holding manifest.json) directly under `root`, `_`-prefixed skipped."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / MANIFEST_FILE).is_file()
    )


def expand_runs(paths: Sequence[Path | str]) -> list[Path]:
    """Run folders from paths that are run folders, glob patterns (Windows shells do not
    expand them) or roots holding run folders (one or two levels, e.g. data/rl7/runs)."""
    out: list[Path] = []
    for raw in paths:
        text = str(raw)
        candidates = (
            [Path(p) for p in sorted(glob.glob(text))]
            if any(c in text for c in "*?[")
            else [Path(text)]
        )
        for p in candidates:
            if (p / MANIFEST_FILE).is_file():
                out.append(p)
            elif p.is_dir():
                found = find_runs(p)
                for sub in sorted(q for q in p.iterdir() if q.is_dir()):
                    found += find_runs(sub)
                out += found
    unique = list(dict.fromkeys(Path(p) for p in out))
    if not unique:
        raise ConfigError(f"no finished training run in {[str(p) for p in paths]}")
    return unique


def validation_frame(run: Mapping) -> pd.DataFrame:
    path = Path(run["dir"]) / "validation.csv"
    if not path.is_file():
        raise ConfigError(f"{run['dir']}: no validation.csv")
    return pd.read_csv(path)


def late_window(val: pd.DataFrame, steps: int, pick_from: float) -> pd.DataFrame:
    """D72: the validations at >= `pick_from` x `steps` (the final policy included). A
    `late` column written by rl.train takes precedence."""
    if "late" in val.columns:
        late = val["late"].astype(str).str.lower().isin(("true", "1", "1.0"))
        return val[late]
    return val[val["step"] >= float(pick_from) * int(steps) - REL_TOL]


def late_stats(run: Mapping, pick_from: float) -> dict:
    """The run's D72 numbers from its validation.csv: late_mean (mean selection cost of
    the late validations), late_ped_s (their mean pedestrian_s), the pick (lowest late
    cost, the earliest step on a tie) and the number of late validations."""
    val = validation_frame(run).sort_values("step", kind="mergesort")
    late = late_window(val, int(run["steps"]), pick_from)
    if late.empty:
        raise ConfigError(f"{run['dir']}: no validation at >= {pick_from:g} of the budget")
    i = late["cost"].astype(float).idxmin()
    out = {
        "late_mean": float(late["cost"].mean()),
        "late_ped_s": float(late["pedestrian_s"].mean()) if "pedestrian_s" in late else math.nan,
        "n_late": len(late),
        "pick_step": int(late.loc[i, "step"]),
        "pick_cost": float(late.loc[i, "cost"]),
    }
    recorded = run.get("late_mean")
    if recorded is not None and not math.isclose(float(recorded), out["late_mean"], rel_tol=1e-9):
        log.warning(
            "%s: manifest late_mean %.3f differs from validation.csv %.3f (validation.csv used)",
            Path(run["dir"]).name,
            float(recorded),
            out["late_mean"],
        )
    return out


def seeds_text(seeds: Sequence[int]) -> str:
    """`lo-hi` for a contiguous seed list, else the sorted list."""
    v = sorted(int(x) for x in seeds)
    if len(v) > 1 and v == list(range(v[0], v[-1] + 1)):
        return f"{v[0]}-{v[-1]}"
    return str(v[0]) if len(v) == 1 else str(v)


def window_problems(runs: Sequence[Mapping], windows: Sequence[int]) -> list[str]:
    """The runs whose recorded validation windows (provenance `selection_seeds`: the
    windows behind `pick_step`, `pick_val_cost` and `late_mean`) are not `windows`, or
    that record none. Their selection numbers come from other windows (a development
    run, or one validated on a held-out range) and never enter a D73 / D75 decision."""
    want = sorted(int(x) for x in windows)
    out = []
    for m in runs:
        name = str(m.get("run") or Path(m["dir"]).name)
        got = m.get("selection_seeds")
        if not got:
            out.append(f"{name}: no `selection_seeds` record (provenance.json)")
        elif sorted(int(x) for x in got) != want:
            out.append(f"{name}: validated on {seeds_text(got)}, not on {seeds_text(want)}")
    return out


def check_windows(runs: Sequence[Mapping], windows: Sequence[int], what: str) -> None:
    """ConfigError naming every run of `runs` not validated on `windows`."""
    problems = window_problems(runs, windows)
    if problems:
        raise ConfigError(f"{what}: " + "; ".join(problems))


def validation_windows(
    rl_cfg: Mapping, validation_seeds: Sequence[int] | None, root: Path, what: str
) -> tuple[list[int], bool]:
    """(windows, development) a decision accepts: the registered `seeds.selection`
    (D70) unless `validation_seeds` names others. Those must be training seeds
    (`seeds.train`: the development windows of the gate's mini-campaign, never a
    selection, holdout or test range), and such a decision never goes to the registered
    study root `outputs.study`."""
    registered = _seeds_of(rl_cfg, "selection")
    if not validation_seeds:
        return registered, False
    windows = sorted(int(s) for s in validation_seeds)
    if windows == registered:
        return registered, False
    train = set(_seeds_of(rl_cfg, "train"))
    outside = [s for s in windows if s not in train]
    if outside:
        raise ConfigError(
            f"{what}: development validation windows must be training seeds "
            f"(seeds.train), not {seeds_text(outside)}"
        )
    if Path(root).resolve() == study_root(rl_cfg).resolve():
        raise ConfigError(
            f"{what}: a decision on the development windows {seeds_text(windows)} never "
            f"goes to the registered study root {study_root(rl_cfg).as_posix()} (pass --out)"
        )
    return windows, True


def stage_runs(
    root: Path,
    rl7: Path,
    names: Sequence[str],
    seeds: Sequence[int],
    steps: int,
    backends: Sequence[str] = ("sumo",),
    windows: Sequence[int] | None = None,
) -> dict[str, dict[int, dict]]:
    """{config name: {seed: run}} of the finished runs under `root` made with the CURRENT
    run config (`run_config_sha`) and `steps` and validated on `windows` (default: the
    registered `seeds.selection`). ConfigError when one is missing, two qualify (the
    error names stale runs of another config version) or a run was validated on other
    windows (`window_problems`: it is named, never silently skipped)."""
    if windows is None:
        windows = _seeds_of(load_config(rl7), "selection")
    found = [read_run(d) for d in find_runs(root)]
    out: dict[str, dict[int, dict]] = {}
    problems: list[str] = []
    for name in names:
        sha = config_sha(run_config_path(rl7, name))
        out[name] = {}
        for seed in seeds:
            same = [
                m
                for m in found
                if m.get("variant") == name
                and int(m.get("seed", -1)) == int(seed)
                and int(m.get("steps", -1)) == int(steps)
                and m.get("backend", "sumo") in backends
            ]
            ok = [m for m in same if m.get("run_config_sha") == sha]
            wrong = window_problems(ok, windows)
            if wrong:
                problems += [f"{name} seed {seed}: {w}" for w in wrong]
            elif len(ok) == 1:
                out[name][int(seed)] = ok[0]
            elif len(ok) > 1:
                problems.append(f"{name} seed {seed}: {len(ok)} runs ({[m['run'] for m in ok]})")
            else:
                stale = [Path(m["dir"]).name for m in same]
                problems.append(
                    f"{name} seed {seed}: no run of {steps} steps with the current config"
                    + (f" (stale: {stale})" if stale else "")
                )
    if problems:
        raise ConfigError(f"{root}: " + "; ".join(problems))
    return out


# -- D73: stage-T decision ---------------------------------------------------------------
def decide_rule(
    scores: Mapping[str, Mapping[int, float]],
    ped: Mapping[str, Mapping[int, float]],
    reference: str,
    margin_rel: float,
    min_seed_wins: int,
    ped_guard_rel: float,
    order: Sequence[str] | None = None,
) -> dict:
    """D73 on per-run late means (`scores[config][seed]`) and late pedestrian seconds
    (`ped[config][seed]`): {outcome{config: {score, ratio, wins, seeds, a, b, c,
    qualifies}}, chosen}. Ties in score keep `order` (the pre-registered list)."""
    if reference not in scores:
        raise ConfigError(f"stage T: no runs of the reference {reference!r}")
    order = list(order or scores)
    ref = scores[reference]
    score_ref = _mean(list(ref.values()))
    ped_ref = _mean(list(ped[reference].values()))
    outcome: dict[str, dict] = {}
    for name in order:
        if name not in scores:
            continue
        s = scores[name]
        score = _mean(list(s.values()))
        ped_x = _mean(list(ped[name].values()))
        row: dict[str, Any] = {
            "score": score,
            "ratio": score / score_ref if score_ref else math.nan,
            "ped_s": ped_x,
            "ped_ratio": ped_x / ped_ref if ped_ref else math.nan,
        }
        if name == reference:
            row |= {"reference": True, "qualifies": False}
        else:
            matching = sorted(set(s) & set(ref))
            wins = sum(float(s[k]) < float(ref[k]) for k in matching)
            a = score <= (1.0 - float(margin_rel)) * score_ref + REL_TOL * abs(score_ref)
            b = wins >= int(min_seed_wins)
            c = ped_x <= (1.0 + float(ped_guard_rel)) * ped_ref + REL_TOL * abs(ped_ref)
            row |= {
                "reference": False,
                "wins": int(wins),
                "seeds": len(matching),
                "a": bool(a),
                "b": bool(b),
                "c": bool(c),
                "qualifies": bool(a and b and c),
            }
        outcome[name] = row
    qualifiers = [n for n in order if outcome.get(n, {}).get("qualifies")]
    chosen = (
        min(qualifiers, key=lambda n: (outcome[n]["score"], order.index(n)))
        if qualifiers
        else reference
    )
    return {"outcome": outcome, "chosen": chosen, "score_ref": score_ref, "ped_ref": ped_ref}


def decide_t(
    rl7: Path = RL7_CONFIG,
    runs: Path | None = None,
    out: Path | None = None,
    steps: int | None = None,
    force: bool = False,
    validation_seeds: Sequence[int] | None = None,
) -> dict:
    """D73: the stage-T scores (mean over seeds of each run's `late_mean`) and the rule
    outcome; writes and returns `decision_T.json`. `runs`: the stage-T runs folder
    (default `<outputs.runs>/T`); `out`: the study root (default `outputs.study`);
    `steps`: the runs' budget (default `stage_t.steps`; the gate's mini-campaign passes
    its override). An existing decision is kept unless `force`. Every run must have been
    validated on the registered selection windows; `validation_seeds` (development
    windows, training seeds only) serves the gate's mini-campaign and then needs an `out`
    other than `outputs.study` (`validation_windows`)."""
    rl7 = Path(rl7)
    cfg = load_config(rl7)
    st = protocol(cfg)["stage_t"]
    root = study_root(cfg, out)
    path = root / DECISION_T_FILE
    check_out(root)
    if path.is_file() and not force:
        raise FileExistsError(f"{path} exists: the stage-T decision is made (use --force)")
    steps = int(steps or st["steps"])
    names = [str(n) for n in st["configs"]]
    reference = str(st["reference"])
    if reference not in names:
        raise ConfigError(f"stage_t.reference {reference!r} is not in stage_t.configs")
    seeds = [int(s) for s in st["seeds"]]
    windows, dev = validation_windows(cfg, validation_seeds, root, "decide-t")
    found = stage_runs(
        Path(runs or runs_root(cfg) / "T"), rl7, names, seeds, steps, windows=windows
    )
    pick_from = float((cfg.get("train") or {}).get("pick_from", 0.5))
    scores: dict[str, dict[int, float]] = {}
    ped: dict[str, dict[int, float]] = {}
    run_names: dict[str, dict[int, str]] = {}
    for name in names:
        scores[name], ped[name], run_names[name] = {}, {}, {}
        for seed, run in found[name].items():
            stats = late_stats(run, pick_from)
            scores[name][seed] = stats["late_mean"]
            ped[name][seed] = stats["late_ped_s"]
            run_names[name][seed] = str(run.get("run") or Path(run["dir"]).name)
    rule = {
        "text": (
            f"C* = {reference} unless a candidate X has (a) score(X) <= "
            f"{1 - float(st['margin_rel']):g} x score({reference}), (b) a late_mean below "
            f"{reference}'s on >= {st['min_seed_wins']} of the matching seeds and (c) a late "
            f"mean pedestrian_s <= {1 + float(st['ped_guard_rel']):g} x {reference}'s; the "
            "qualifier with the lowest score wins (D73). score = mean over seeds of "
            f"late_mean, the mean selection cost of the validations at >= {pick_from:g} of "
            "the budget (D72)."
        ),
        "reference": reference,
        "margin_rel": float(st["margin_rel"]),
        "min_seed_wins": int(st["min_seed_wins"]),
        "ped_guard_rel": float(st["ped_guard_rel"]),
        "pick_from": pick_from,
    }
    res = decide_rule(
        scores,
        ped,
        reference,
        rule["margin_rel"],
        rule["min_seed_wins"],
        rule["ped_guard_rel"],
        order=names,
    )
    doc = {
        "scores": {n: {str(s): v for s, v in d.items()} for n, d in scores.items()},
        "means": {n: res["outcome"][n]["score"] for n in names},
        "ped_s": {n: res["outcome"][n]["ped_s"] for n in names},
        "ped_s_runs": {n: {str(s): v for s, v in d.items()} for n, d in ped.items()},
        "rule": rule,
        "outcome": res["outcome"],
        "chosen": res["chosen"],
        "chosen_config": run_config_path(rl7, res["chosen"]).as_posix(),
        "runs": {n: {str(s): r for s, r in d.items()} for n, d in run_names.items()},
        "steps": steps,
        "seeds": seeds,
        "validation_seeds": windows,
        "development": dev,
        "rl7_sha256": text_sha256(rl7),
        "code_version": code_version(),
        "created_at": _now(),
    }
    write_json(path, doc)
    log.info("stage T: C* = %s (%s)", res["chosen"], path.as_posix())
    return doc


# -- episodes (in-process or in a spawned pool) -----------------------------------------
@dataclass
class Episode:
    """One episode task: `policy` "model" (`model`, `algo`) or a `run_baseline`
    controller, on env config `config` in `mode`, with forced conditions `override`
    (D78, `reset(options={"params_override": ...})`)."""

    config: str
    mode: str
    seed: int
    policy: str
    model: str | None = None
    algo: str | None = None
    override: dict = field(default_factory=dict)


_W: dict[str, Any] = {}


def _worker_init(backend: EnvBackend, work_root: Path | None) -> None:
    set_torch_threads(1)
    _W.clear()
    _W.update(backend=backend, work_root=work_root, env=None, env_key=None, model=None)


def _worker_close() -> None:
    env = _W.get("env")
    if env is not None:
        env.close()
    _W.clear()


def _worker_env(config: str, mode: str):
    """The process's one env (libsumo: one simulation per process), rebuilt when the
    config or mode changes."""
    key = (config, mode)
    if _W.get("env_key") != key:
        if _W.get("env") is not None:
            _W["env"].close()
        _W["env"], _W["env_key"] = None, None
        _W["env"] = _W["backend"].env(config, mode=mode, seeds=[0], work_root=_W["work_root"])
        _W["env_key"] = key
    return _W["env"]


def _worker_model(path: str, algo: str):
    """One cached model per process (evicted by a task naming another file)."""
    cached = _W.get("model")
    if cached is None or cached[0] != (path, algo):
        _W["model"] = None
        _W["model"] = ((path, algo), load_model(Path(path), algo))
    return _W["model"][1]


def play(env, act: Callable[[np.ndarray], int], seed: int, override: Mapping | None) -> dict:
    """One episode of `act` on `env` (forced conditions `override`); its summary."""
    options: dict[str, Any] = {"episode_seed": int(seed)}
    if override:
        options["params_override"] = dict(override)
    obs, _ = env.reset(options=options)
    while True:
        obs, _, terminated, truncated, info = env.step(act(obs))
        if terminated or truncated:
            summary = info.get("episode_summary")
            if summary is None:
                raise RuntimeError("episode ended without info['episode_summary']")
            return summary


def baseline(backend: EnvBackend, env, controller: str, seed: int, override: Mapping | None):
    """`backend.run_baseline`, with `params_override` when conditions are forced (a
    backend whose run_baseline cannot force them is refused, never silently ignored)."""
    if not override:
        return backend.run_baseline(env, controller, int(seed))
    if "params_override" not in inspect.signature(backend.run_baseline).parameters:
        raise ConfigError(f"backend {backend.name!r}: run_baseline cannot force conditions")
    return backend.run_baseline(env, controller, int(seed), params_override=dict(override))


def episode_row(summary: Mapping) -> dict:
    """`train.summary_row` (extended: control_s and the D77 split) plus the forced
    conditions and the drawn ambulances / incident of the episode."""
    row = summary_row(summary, True)
    params = summary.get("params") or {}
    if "split_cut" not in row:  # an older summary_row: the split straight from the summary
        split = summary.get("iv_split")
        row |= {
            f"split_{k}": (int(split.get(k, 0) or 0) if split is not None else math.nan)
            for k in SPLIT_KEYS
        }
    amb = params.get("ambulances")
    row["n_ambulances"] = len(amb) if isinstance(amb, list) else math.nan
    incidents = params.get("incidents") or []
    row["incident"] = ",".join(str(i.get("site", i.get("id", "?"))) for i in incidents) or None
    return row


def _run_episode(ep: Episode) -> dict:
    env = _worker_env(ep.config, ep.mode)
    if ep.policy == "model":
        model = _worker_model(str(ep.model), str(ep.algo))
        summary = play(env, model_policy(model), ep.seed, ep.override)
    else:
        summary = baseline(_W["backend"], env, ep.policy, ep.seed, ep.override)
    return episode_row(summary)


def run_tasks(
    backend: EnvBackend,
    episodes: Sequence[Episode],
    workers: int = 1,
    work_root: Path | None = None,
) -> list[dict]:
    """Rows (`episode_row`) of the episodes, in order; `workers > 1`: a spawned pool (the
    episodes are seeded, so the worker that plays one does not matter). They are played
    grouped by (config, mode, model), so a process rarely rebuilds its env or reloads a
    model."""
    if not episodes:
        return []
    order = sorted(
        range(len(episodes)),
        key=lambda i: (episodes[i].config, episodes[i].mode, episodes[i].model or "", i),
    )
    todo = [episodes[i] for i in order]
    if workers <= 1:
        _worker_init(backend, work_root)
        try:
            rows = [_run_episode(e) for e in todo]
        finally:
            _worker_close()
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        n = min(int(workers), len(todo))
        chunk = max(1, len(todo) // (n * 4))
        with ProcessPoolExecutor(
            n,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init,
            initargs=(backend, work_root),
        ) as ex:
            rows = list(ex.map(_run_episode, todo, chunksize=chunk))
    out: list[dict] = [{}] * len(episodes)
    for i, row in zip(order, rows, strict=True):
        out[i] = row
    return out


def _frame(meta: Sequence[Mapping], rows: Sequence[Mapping]) -> pd.DataFrame:
    return pd.DataFrame([dict(m) | dict(r) for m, r in zip(meta, rows, strict=True)])


def _run_model(run: Mapping) -> tuple[Path, str]:
    path = Path(run["dir"]) / PICK_MODEL_FILE
    if not path.is_file():
        raise ConfigError(f"{run['dir']}: no {PICK_MODEL_FILE} (a Part 7 run, D72)")
    algo = str(run.get("algo") or "")
    if not algo:
        raise ConfigError(f"{run['dir']}: the manifest names no algo")
    return path, algo


def _run_config(run: Mapping) -> str:
    config = run.get("config")
    if not config or not Path(config).is_file():
        raise ConfigError(f"{run['dir']}: its run config {config!r} is not a file")
    return str(config)


# -- holdout (reporting only) ------------------------------------------------------------
def holdout_seeds(rl_cfg: Mapping) -> tuple[list[int], list[int]]:
    """(holdout windows, full-mode seeds) of `protocol.holdout`, checked against the
    rl7.yaml holdout range."""
    h = protocol(rl_cfg).get("holdout") or {}
    windows = _seeds_of(rl_cfg, str(h.get("windows", "holdout")))
    full = seed_range(h["full"]) if h.get("full") else []
    check_seeds(windows + full, rl_cfg, ("holdout",), "holdout")
    return windows, full


def baseline_table(
    backend: EnvBackend,
    config: str,
    hdir: Path,
    plan: Sequence[tuple[str, Sequence[int]]],
    workers: int,
    work_root: Path,
    labels: Mapping[str, str] = HOLDOUT_BASELINES,
) -> pd.DataFrame:
    """The baselines of `config`'s environment on every (mode, seeds) of `plan`, cached in
    `<hdir>/baselines_<env_key>.csv` and shared by every run of that environment: only
    missing (label, mode, seed) rows are played."""
    key = env_key(load_config(config))
    path = Path(hdir) / f"baselines_{key}.csv"
    have = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    done = (
        set(zip(have["label"], have["mode"], have["seed"].astype(int), strict=True))
        if len(have)
        else set()
    )
    meta, eps = [], []
    for mode, seeds in plan:
        for label, ctl in labels.items():
            for s in seeds:
                if (label, mode, int(s)) not in done:
                    meta.append(
                        {
                            "env_key": key,
                            "label": label,
                            "policy": ctl,
                            "mode": mode,
                            "seed": int(s),
                        }
                    )
                    eps.append(Episode(config, mode, int(s), ctl))
    if eps:
        log.info("holdout baselines %s: %d episodes", key, len(eps))
        new = _frame(meta, run_tasks(backend, eps, workers, work_root))
        have = pd.concat([have, new], ignore_index=True) if len(have) else new
        path.parent.mkdir(parents=True, exist_ok=True)
        have.to_csv(path, index=False)
    return have


def _means(df: pd.DataFrame, mode: str, value: str = "cost") -> dict[str, float]:
    g = df[df["mode"] == mode]
    return {str(k): float(v) for k, v in g.groupby("label")[value].mean().items()}


def holdout_summary_row(
    run: Mapping, pick: pd.DataFrame, base: pd.DataFrame, pick_from: float
) -> dict:
    """One holdout/summary.csv row: the pick's mean cost on the windows (`holdout_w30`) and
    the full episodes (`holdout_full`), the baselines' and the % changes, wins and exact
    Wilcoxon p against S1 on the windows."""
    try:
        stats = late_stats(run, pick_from)
    except ConfigError:
        stats = {}
    row: dict[str, Any] = {
        "run": str(run.get("run") or Path(run["dir"]).name),
        "dir": Path(run["dir"]).as_posix(),
        "stage": run.get("stage"),
        "variant": run.get("variant"),
        "algo": run.get("algo"),
        "seed": run.get("seed"),
        "steps": run.get("steps"),
        "pick_step": run.get("pick_step", stats.get("pick_step")),
        "pick_val_cost": run.get("pick_val_cost", stats.get("pick_cost")),
        "late_mean": stats.get("late_mean", run.get("late_mean")),
    }
    for mode, tag in (("train", "w30"), ("full", "full")):
        p = pick[pick["mode"] == mode]
        m = float(p["cost"].mean()) if len(p) else math.nan
        row[f"holdout_{tag}"] = m
        row[f"n_{tag}"] = len(p)
        means = _means(base, mode)
        for label in HOLDOUT_BASELINES:
            row[f"{tag}_{label}"] = means.get(label, math.nan)
            row[f"{tag}_vs_{label}"] = pct_change(m, means.get(label, math.nan))
    p = pick[pick["mode"] == "train"].set_index("seed")["cost"]
    b = base[(base["mode"] == "train") & (base["label"] == "S1")].set_index("seed")["cost"]
    seeds = sorted(set(p.index) & set(b.index))
    if seeds:
        x, y = p.loc[seeds].to_numpy(float), b.loc[seeds].to_numpy(float)
        row["w30_wins_vs_S1"] = int((x < y).sum())
        row["w30_p_vs_S1"] = float(wilcoxon(x, y).pvalue)
    return row


def curve_rows(ckpt: pd.DataFrame, base: pd.DataFrame, pick_step: Any) -> pd.DataFrame:
    """holdout_curve.csv: per kept checkpoint, the mean window cost, its % change vs each
    baseline, the class totals and interventions."""
    means = _means(base, "train")
    rows = []
    for step, g in ckpt.groupby("step", sort=True):
        m = g.mean(numeric_only=True)
        row: dict[str, Any] = {"step": int(step), "n": len(g), "cost": float(m["cost"])}
        row |= {f"cost_vs_{k}": pct_change(float(m["cost"]), v) for k, v in means.items()}
        row |= {k: float(m[k]) for k in TOTAL_KEYS if k in m}
        row |= {k: float(m[k]) for k in ("interventions", "split_cut", "split_replaced") if k in m}
        row["pick"] = pick_step is not None and int(step) == int(pick_step)
        rows.append(row)
    return pd.DataFrame(rows)


def kept_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in sorted((Path(run_dir) / VAL_DIR).glob("step_*.zip")):
        try:
            out.append((int(p.stem.split("_", 1)[1]), p))
        except ValueError:
            continue
    return sorted(out)


def holdout(
    runs: Sequence[Path],
    rl7: Path = RL7_CONFIG,
    curves: bool = False,
    workers: int | None = None,
    *,
    out: Path | None = None,
    backend: EnvBackend | None = None,
) -> list[Path]:
    """Score each run's pick (and, with `curves`, its kept checkpoints) on the holdout
    seeds; returns the holdout.csv paths. Reporting only: nothing here selects."""
    cfg = load_config(rl7)
    windows, full = holdout_seeds(cfg)
    root = study_root(cfg, out)
    hdir, work_root = root / HOLDOUT_DIR, root / WORK_DIR
    for p in (hdir, work_root):
        check_out(p)
    backend = backend or real_backend()
    workers = int(workers or default_workers())
    pick_from = float((cfg.get("train") or {}).get("pick_from", 0.5))
    infos = [read_run(d) for d in expand_runs(runs)]
    plan = [("train", windows)] + ([("full", full)] if full else [])
    bases: dict[str, pd.DataFrame] = {}
    for run in infos:
        config = _run_config(run)
        key = env_key(load_config(config))
        if key not in bases:
            bases[key] = baseline_table(backend, config, hdir, plan, workers, work_root)
        run["_env_key"] = key
    meta, eps = [], []
    for i, run in enumerate(infos):
        config = _run_config(run)
        model, algo = _run_model(run)
        for mode, seeds in plan:
            for s in seeds:
                meta.append(
                    {"_i": i, "kind": "pick", "step": run.get("pick_step"), "mode": mode, "seed": s}
                )
                eps.append(Episode(config, mode, int(s), "model", model.as_posix(), algo))
        if curves:
            for step, ckpt in kept_checkpoints(run["dir"]):
                for s in windows:
                    meta.append({"_i": i, "kind": "ckpt", "step": step, "mode": "train", "seed": s})
                    eps.append(Episode(config, "train", int(s), "model", ckpt.as_posix(), algo))
    log.info("holdout: %d runs, %d episodes", len(infos), len(eps))
    df = _frame(meta, run_tasks(backend, eps, workers, work_root))
    written, summary = [], []
    for i, run in enumerate(infos):
        d = Path(run["dir"])
        base = bases[run["_env_key"]]
        mine = df[df["_i"] == i].drop(columns="_i")
        pick = mine[mine["kind"] == "pick"].drop(columns="kind")
        pick.insert(0, "run", d.name)
        pick.to_csv(d / HOLDOUT_FILE, index=False)
        written.append(d / HOLDOUT_FILE)
        if curves:
            ckpt = mine[mine["kind"] == "ckpt"]
            if len(ckpt):
                curve_rows(ckpt, base, run.get("pick_step")).assign(run=d.name).to_csv(
                    d / HOLDOUT_CURVE_FILE, index=False
                )
            else:
                log.warning("%s: no kept checkpoint under %s/ (no curve)", d.name, VAL_DIR)
        summary.append(
            holdout_summary_row(run, pick, base, pick_from) | {"env_key": run["_env_key"]}
        )
    update_summary(hdir / HOLDOUT_SUMMARY_FILE, summary)
    return written


def update_summary(path: Path, rows: Sequence[Mapping]) -> pd.DataFrame:
    """Merge `rows` into holdout/summary.csv (a run's newer row replaces its older one)."""
    new = pd.DataFrame(rows)
    if Path(path).is_file():
        old = pd.read_csv(path)
        old = old[~old["dir"].isin(set(new["dir"]))]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["stage", "run"], na_position="last", kind="mergesort")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def read_holdout(run_dir: Path) -> dict[str, float]:
    """{holdout_w30, holdout_full} of a run's holdout.csv (NaN when not scored)."""
    path = Path(run_dir) / HOLDOUT_FILE
    if not path.is_file():
        return {"holdout_w30": math.nan, "holdout_full": math.nan}
    h = pd.read_csv(path)
    return {
        "holdout_w30": float(h.loc[h["mode"] == "train", "cost"].mean()),
        "holdout_full": float(h.loc[h["mode"] == "full", "cost"].mean()),
    }


# -- D75: S3 selection --------------------------------------------------------------------
def choose_s3(candidates: Sequence[Mapping], tie_rel: float) -> dict:
    """D75: the lowest stage-2 mean cost; the candidates within `tie_rel` of it are
    ordered by stage-1 cost, then seed."""
    if not candidates:
        raise ConfigError("no S3 candidate")
    best = min(float(c["stage2"]) for c in candidates)
    limit = best * (1.0 + float(tie_rel)) + REL_TOL * abs(best)
    tied = [c for c in candidates if float(c["stage2"]) <= limit]
    return dict(min(tied, key=lambda c: (float(c["stage1"]), int(c["seed"]))))


def read_decision_t(root: Path) -> dict:
    doc = read_json(Path(root) / DECISION_T_FILE)
    if doc is None:
        raise ConfigError(f"{Path(root) / DECISION_T_FILE}: missing (run `rl.study decide-t`)")
    return doc


def select(
    rl7: Path = RL7_CONFIG,
    workers: int | None = None,
    *,
    runs: Path | None = None,
    out: Path | None = None,
    backend: EnvBackend | None = None,
    validation_seeds: Sequence[int] | None = None,
) -> dict:
    """D75: stage 1 (each F run's pick) and stage 2 (select2 episodes in `stage2_mode`);
    writes and returns `select/decision.json`. Every F run must have been validated on
    the registered selection windows (stage 1 is its pick); `validation_seeds` as in
    `decide_t` (development windows, never into `outputs.study`)."""
    rl7 = Path(rl7)
    cfg = load_config(rl7)
    prot = protocol(cfg)
    sp, sf = prot["select"], prot["stage_f"]
    root = study_root(cfg, out)
    sdir, work_root = root / SELECT_DIR, root / WORK_DIR
    for p in (sdir, work_root):
        check_out(p)
    chosen_t = str(read_decision_t(root)["chosen"])
    seeds2 = _seeds_of(cfg, str(sp["stage2"]))
    check_seeds(seeds2, cfg, (str(sp["stage2"]),), "select stage 2")
    mode2 = str(sp.get("stage2_mode", "full"))
    windows, dev = validation_windows(cfg, validation_seeds, root, "select")
    found = stage_runs(
        Path(runs or runs_root(cfg) / "F"),
        rl7,
        [chosen_t],
        [int(s) for s in sf["seeds"]],
        int(sf["steps"]),
        windows=windows,
    )[chosen_t]
    pick_from = float((cfg.get("train") or {}).get("pick_from", 0.5))
    backend = backend or real_backend()
    workers = int(workers or default_workers())
    cands, meta, eps = [], [], []
    for seed, run in sorted(found.items()):
        model, algo = _run_model(run)
        stats = late_stats(run, pick_from)
        stage1 = _float(run.get("pick_val_cost"))
        if not math.isfinite(stage1):
            stage1 = stats["pick_cost"]
        cands.append(
            {
                "run": str(run.get("run") or Path(run["dir"]).name),
                "dir": Path(run["dir"]).as_posix(),
                "config": _run_config(run),
                "algo": algo,
                "variant": run.get("variant"),
                "seed": int(seed),
                "pick_step": run.get("pick_step", stats["pick_step"]),
                "stage1": stage1,
                "late_mean": stats["late_mean"],
                "pick_model_sha256": file_sha256(model),
            }
        )
        for s in seeds2:
            meta.append({"label": f"s{seed}", "run": cands[-1]["run"], "mode": mode2, "seed": s})
            eps.append(Episode(cands[-1]["config"], mode2, int(s), "model", model.as_posix(), algo))
    config = cands[0]["config"]
    for label, ctl in STAGE2_BASELINES.items():
        for s in seeds2:
            meta.append({"label": label, "run": None, "mode": mode2, "seed": s})
            eps.append(Episode(config, mode2, int(s), ctl))
    log.info("select: %d candidates, %d stage-2 episodes", len(cands), len(eps))
    df = _frame(meta, run_tasks(backend, eps, workers, work_root))
    base = {lab: float(df.loc[df["label"] == lab, "cost"].mean()) for lab in STAGE2_BASELINES}
    for c in cands:
        g = df[df["run"] == c["run"]]
        c["stage2"] = float(g["cost"].mean())
        c["stage2_vs"] = {k: pct_change(c["stage2"], v) for k, v in base.items()}
        c |= read_holdout(Path(c["dir"]))
    tie_rel = float(sp.get("tie_rel", 0.005))
    chosen = choose_s3(cands, tie_rel)
    sdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in c.items() if k != "stage2_vs"} for c in cands]).to_csv(
        sdir / "stage1.csv", index=False
    )
    df.to_csv(sdir / "stage2.csv", index=False)
    doc = {
        "candidates": cands,
        "chosen": chosen["run"],
        "chosen_dir": chosen["dir"],
        "config": chosen_t,
        "rule": {
            "text": (
                f"S3 = the stage-F pick with the lowest mean cost on {len(seeds2)} "
                f"{mode2}-mode episodes of the {sp['stage2']} seeds; candidates within "
                f"{100 * tie_rel:g} % of it: lower stage-1 (selection) cost, then lower seed "
                "(D75). The holdout is reported, never used."
            ),
            "stage2": str(sp["stage2"]),
            "stage2_mode": mode2,
            "stage2_seeds": seeds2,
            "tie_rel": tie_rel,
        },
        "stage2_baselines": base,
        "validation_seeds": windows,
        "development": dev,
        "rl7_sha256": text_sha256(rl7),
        "code_version": code_version(),
        "created_at": _now(),
    }
    write_json(sdir / SELECT_DECISION_FILE, doc)
    log.info("select: S3 = %s (stage 2 %.1f)", chosen["run"], chosen["stage2"])
    return doc


# -- D86: publication --------------------------------------------------------------------
def model_version(algo: str, variant: str, seed: int, policy_sha: str) -> str:
    return f"rl7-{algo}-{variant}-s{int(seed)}-{policy_sha[:8]}"


def _lf_bytes(path: Path) -> bytes:
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def _rel_posix(path: Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def s3_config_text(manifest: Mapping, model_path: Path) -> str:
    """The configs/pitie_rl_s3.yaml lines for a published model (contract 4.1)."""
    return (
        "# S3 (Part 7, D86): the selected Part 7 policy. Demand/emergency/network as S0 "
        "(demand_hash 83a4f220fd20).\n"
        "extends: pitie_rl.yaml\n"
        "scenario: pitie_rl_s3\n"
        f"rl: {{model: {_rel_posix(model_path)}, model_sha256: {manifest['model_sha256']}, "
        f"config: {_rel_posix(Path(manifest['config']))}}}\n"
    )


def publish(rl7: Path = RL7_CONFIG, out: Path | None = None, *, root: Path | None = None) -> Path:
    """D86: copy the chosen pick to `<out or outputs.models>/<model_version>/`; returns
    that folder. Refuses a sha mismatch (file or policy parameters against the run's
    records and the selection) and a different model already published under the name."""
    rl7 = Path(rl7)
    cfg = load_config(rl7)
    study = study_root(cfg, root)
    dec = read_json(study / SELECT_DIR / SELECT_DECISION_FILE)
    if dec is None:
        raise ConfigError(f"{study / SELECT_DIR / SELECT_DECISION_FILE}: missing (run `select`)")
    cand = next((c for c in dec["candidates"] if c["run"] == dec["chosen"]), None)
    if cand is None:
        raise ConfigError(f"select decision: chosen {dec['chosen']!r} is not a candidate")
    registered = Path((cfg.get("outputs") or {}).get("models") or "models/rl")
    if dec.get("development") and Path(out or registered).resolve() == registered.resolve():
        raise ConfigError(
            "select decision on development windows: never published to the registered "
            f"models root {registered.as_posix()} (pass --out)"
        )
    run = read_run(Path(cand["dir"]))
    src, algo = _run_model(run)
    sha = file_sha256(src)
    for what, want in (
        ("the selection", cand.get("pick_model_sha256")),
        ("the run manifest", run.get("pick_model_sha256")),
    ):
        if want and want != sha:
            raise ConfigError(f"{src}: sha256 {sha[:12]} differs from {what} ({str(want)[:12]})")
    model = load_model(src, algo)
    psha = policy_sha256(model)
    want = run.get("pick_policy_sha256")
    if want and want != psha:
        raise ConfigError(f"{src}: policy sha256 {psha[:12]} differs from the run ({want[:12]})")
    variant = str(run.get("variant") or cand.get("variant"))
    version = model_version(algo, variant, int(run["seed"]), psha)
    dest = Path(out or registered) / version
    check_out(dest)
    if (dest / "model.zip").is_file() and file_sha256(dest / "model.zip") != sha:
        raise ConfigError(f"{dest}: holds another model.zip under the same version")
    config = _run_config(run)
    rl_cfg = load_config(config)
    run_obs = run.get("observation") if isinstance(run.get("observation"), Mapping) else {}
    drop = run.get("obs_drop", run_obs.get("drop"))
    if drop is None:
        drop = (rl_cfg.get("observation") or {}).get("drop") or []
    want_drop = list((rl_cfg.get("observation") or {}).get("drop") or [])
    if list(drop) != want_drop:
        raise ConfigError(f"{run['dir']}: obs drop {list(drop)} differs from {config} {want_drop}")
    manifest = {
        "model_version": version,
        "algo": algo,
        "variant": variant,
        "model": "model.zip",
        "model_sha256": sha,
        "policy_sha256": psha,
        "obs_version": run.get("obs_version") or (rl_cfg.get("observation") or {}).get("version"),
        "reward_version": run.get("reward_version") or (rl_cfg.get("reward") or {}).get("version"),
        "obs_drop": list(drop),
        "n_features": run.get("n_features", run_obs.get("n_features")),
        "decision_interval_s": (rl_cfg.get("decision") or {}).get("interval_s"),
        "reward": rl_cfg.get("reward"),
        "config": _rel_posix(Path(config)),
        "run_config_sha": run.get("run_config_sha"),
        "rl7_sha256": dec.get("rl7_sha256"),
        "source_run": run.get("run"),
        "source_stage": run.get("stage"),
        "seed": int(run["seed"]),
        "steps": run.get("steps"),
        "n_envs": run.get("n_envs"),
        "hyperparameters": run.get("hyperparameters"),
        "pick_step": cand.get("pick_step"),
        "pick_val_cost": cand.get("stage1"),
        "late_mean": cand.get("late_mean"),
        "selection": {
            "rule": (dec.get("rule") or {}).get("text"),
            "stage1": cand.get("stage1"),
            "stage2": cand.get("stage2"),
            "stage2_baselines": dec.get("stage2_baselines"),
            "holdout_w30": cand.get("holdout_w30"),
            "holdout_full": cand.get("holdout_full"),
            "candidates": [c.get("run") for c in dec["candidates"]],
        },
        "code_version": run.get("code_version"),
        "dirty": run.get("dirty"),
        "allow_dirty": run.get("allow_dirty"),
        "versions": run.get("versions"),
        "published_by": code_version(),
        "published_at": _now(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "model.zip.tmp"
    shutil.copyfile(src, tmp)
    if file_sha256(tmp) != sha:
        tmp.unlink()
        raise ConfigError(f"{dest}: the copied model.zip differs from {src}")
    os.replace(tmp, dest / "model.zip")
    write_json(dest / "manifest.json", manifest)
    val = Path(run["dir"]) / "validation.csv"
    if val.is_file():
        (dest / "validation.csv").write_bytes(_lf_bytes(val))
    log.info("published %s -> %s", version, dest.as_posix())
    return dest


# -- golden ------------------------------------------------------------------------------
def s3_model(scenario: Path) -> dict:
    """{path, algo, config, model_sha256, model_version, manifest} of the S3 scenario's
    pinned model (rl.model_sha256 must equal the file)."""
    cfg = load_scenario(scenario)
    if cfg.get("controller") != "rl":
        raise ConfigError(f"{scenario}: not an `rl` scenario")
    path, config = Path(get(cfg, "rl.model", "") or ""), Path(get(cfg, "rl.config", "") or "")
    if not path.is_file():
        raise ConfigError(f"{scenario}: rl.model {path} is not a file")
    if not config.is_file():
        raise ConfigError(f"{scenario}: rl.config {config} is not a file")
    sha = file_sha256(path)
    pin = get(cfg, "rl.model_sha256", None)
    if pin is None or pin != sha:
        raise ConfigError(f"{scenario}: rl.model_sha256 {pin!r} does not pin {path} ({sha[:12]})")
    manifest = read_json(path.with_name(MANIFEST_FILE)) or {}
    algo = manifest.get("algo")
    if algo not in ("dqn", "ppo"):
        raise ConfigError(f"{path}: no manifest.json naming the algo next to it")
    return {
        "path": path,
        "algo": algo,
        "config": config,
        "scenario": Path(scenario),
        "model_sha256": sha,
        "model_version": manifest.get("model_version"),
        "manifest": manifest,
    }


def policy_values(model, algo: str, obs: np.ndarray) -> np.ndarray:
    """Q-values (DQN) or action logits (PPO) of a batch of observations."""
    import torch

    x = torch.as_tensor(np.asarray(obs, dtype=np.float32))
    with torch.no_grad():
        if algo == "dqn":
            return model.policy.q_net(x).cpu().numpy()
        return model.policy.get_distribution(x).distribution.logits.cpu().numpy()


def golden_indices(
    margins: Sequence[float], n: int = GOLDEN_N_OBS, n_close: int = GOLDEN_N_CLOSE
) -> list[int]:
    """`n` decision indices: the `n_close` closest calls (smallest value gap) and the rest
    spread evenly over the episode."""
    total = len(margins)
    if total <= n:
        return list(range(total))
    chosen = {int(i) for i in np.argsort(np.asarray(margins), kind="mergesort")[:n_close]}
    k = n - len(chosen)
    while True:
        spread = [int(i) for i in np.round(np.linspace(0, total - 1, k)).astype(int)]
        extra = [i for i in dict.fromkeys(spread) if i not in chosen]
        if len(chosen) + len(extra) >= n or k >= total:
            break
        k += 1
    return sorted(chosen | set(extra[: n - len(chosen)]))


class _PhaseCapture:
    """Passive RunObserver keeping the phases shown."""

    def __init__(self) -> None:
        self.shown: list[int] = []

    def bind(self, sim) -> None:
        pass

    def after_step(self, sim) -> None:
        pass

    def finish(self, sim, out_dir, run_id) -> None:
        self.shown = list(sim.phases_shown)


def phases_sha256(phases: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(p) for p in phases]).encode()).hexdigest()


def scenario_phases(
    cfg: dict, seed: int, duration_s: int, out_root: Path
) -> tuple[list[int], dict]:
    """The phases of the first `duration_s` seconds of a `run_scenario` run of `cfg`
    shortened to `duration_s` (route files in a scratch folder), and its run.json."""
    from traffic_twin.sim.run_scenario import run

    cfg = dict(cfg)
    cfg["duration_s"] = int(duration_s)
    cap = _PhaseCapture()
    meta = run(cfg, int(seed), out_root, demand_dir=out_root / "demand", observers=[cap])
    return cap.shown[: int(duration_s)], meta


def golden(
    config: Path = PITIE_RL_S3_CONFIG,
    out: Path = GOLDEN_FILE,
    *,
    seed: int = GOLDEN_SEED,
    duration_s: int = GOLDEN_DURATION_S,
    n_obs: int = GOLDEN_N_OBS,
) -> Path:
    """Write the S3 Q-value and phase-hash golden for tests/test_rl_s3_model.py: the
    policy's values and actions on `n_obs` decisions of a full-mode, non-randomized
    `duration_s` env episode of training seed `seed`, and the phase hash of a
    `duration_s` run_scenario of the S3 scenario on `seed` (Windows reference)."""
    import yaml

    from traffic_twin.rl.env import TrafficEnv

    s3 = s3_model(config)
    rl_cfg = load_config(s3["config"])
    check_seeds([int(seed)], rl_cfg, ("train",), "golden")
    model = load_model(s3["path"], s3["algo"])
    psha = policy_sha256(model)
    if s3["manifest"].get("policy_sha256") not in (None, psha):
        raise ConfigError(f"{s3['path']}: policy sha256 differs from its manifest")
    obs, actions = [], []
    with tempfile.TemporaryDirectory(prefix="rl7_golden_") as tmp:
        tmp = Path(tmp)
        env_cfg = tmp / "golden_env.yaml"
        env_cfg.write_text(
            yaml.safe_dump(
                {
                    "extends": s3["config"].resolve().as_posix(),
                    "episode": {"full": {"warmup_s": 0, "duration_s": int(duration_s)}},
                }
            ),
            encoding="utf-8",
        )
        env = TrafficEnv(env_cfg, mode="full", randomize=False, work_root=tmp / "env")
        try:
            x, _ = env.reset(options={"episode_seed": int(seed)})
            while True:
                a, _ = model.predict(x, deterministic=True)
                obs.append(np.asarray(x, dtype=np.float32))
                actions.append(int(np.asarray(a).item()))
                x, _, term, trunc, info = env.step(actions[-1])
                if term or trunc:
                    break
            env_phases = list(env.sim.phases_shown)[: int(duration_s)]
            summary = info["episode_summary"]
        finally:
            env.close()
        phases, meta = scenario_phases(load_scenario(config), int(seed), duration_s, tmp / "run")
    values = policy_values(model, s3["algo"], np.stack(obs))
    gaps = np.abs(values[:, 1] - values[:, 0])
    idx = golden_indices(gaps.tolist(), n_obs)
    run_hash, env_hash = phases_sha256(phases), phases_sha256(env_phases)
    if run_hash != env_hash:
        log.warning("golden: run_scenario and env phases differ (see tests/test_rl_parity.py)")
    cv = code_version()
    doc = {
        "model_version": s3["model_version"],
        "model_sha256": s3["model_sha256"],
        "policy_sha256": psha,
        "algo": s3["algo"],
        "values": "q" if s3["algo"] == "dqn" else "logits",
        "scenario": _rel_posix(s3["scenario"]),
        "rl_config": _rel_posix(s3["config"]),
        "seed": int(seed),
        "duration_s": int(duration_s),
        "source": "rl.env.TrafficEnv full mode, randomize False, warm-up 0",
        "index": idx,
        "obs": [obs[i].tolist() for i in idx],
        "actions": [actions[i] for i in idx],
        "q": [values[i].tolist() for i in idx],
        "min_gap": float(gaps[idx].min()) if idx else math.nan,
        "episode": {
            "decisions": len(actions),
            "advances": int(sum(actions)),
            "danger": int(summary.get("danger", 0) or 0),
            "env_phases_sha256": env_hash,
            "phases_sha256": run_hash,
            "run_decisions": (meta.get("control") or {}).get("decisions"),
            "run_advances": (meta.get("control") or {}).get("advances"),
            "run_danger": (meta.get("safety") or {}).get("danger"),
            "platform": sys.platform,
        },
        "versions": library_versions(),
        "code_version": cv,
        "dirty": cv.endswith("-dirty"),
        "created_at": _now(),
    }
    return write_json(out, doc)


# -- D78: stress panel -------------------------------------------------------------------
def stress_cells(rl_cfg: Mapping) -> dict[str, dict]:
    cells = (protocol(rl_cfg).get("stress") or {}).get("cells") or {}
    if not cells:
        raise ConfigError("protocol.stress.cells is empty")
    return {str(k): dict(v) for k, v in cells.items()}


def stress_summary(
    df: pd.DataFrame, worse_rel: float, order: Sequence[str] | None = None
) -> pd.DataFrame:
    """One row per cell: mean cost per label, S3 vs S1 and S2 (% of the means, wins,
    exact Wilcoxon p), pedestrian and penalty seconds, interventions and their split,
    danger and curb maximum; `worse_than_s1` when S3 is more than `worse_rel` worse."""
    rows = []
    cells = list(order or dict.fromkeys(df["cell"]))
    for cell in cells:
        g = df[df["cell"] == cell]
        if g.empty:
            continue
        row: dict[str, Any] = {
            "cell": cell,
            "mode": g["mode"].iloc[0],
            "n": int(g["seed"].nunique()),
        }
        row["override"] = g["override"].iloc[0]
        for label in STRESS_LABELS:
            v = g.loc[g["label"] == label, "cost"]
            row[f"cost_{label}"] = float(v.mean()) if len(v) else math.nan
        s3 = g[g["label"] == CANDIDATE].set_index("seed")
        for ref in ("S1", "S2"):
            r = g[g["label"] == ref].set_index("seed")
            seeds = sorted(set(s3.index) & set(r.index))
            x = s3.loc[seeds, "cost"].to_numpy(float)
            y = r.loc[seeds, "cost"].to_numpy(float)
            row[f"vs_{ref}_pct"] = pct_change(x.mean(), y.mean()) if seeds else math.nan
            row[f"wins_vs_{ref}"] = int((x < y).sum()) if seeds else 0
            row[f"p_vs_{ref}"] = float(wilcoxon(x, y).pvalue) if seeds else math.nan
            if "pedestrian_s" in g:
                row[f"ped_s_vs_{ref}_pct"] = (
                    pct_change(
                        float(s3.loc[seeds, "pedestrian_s"].mean()),
                        float(r.loc[seeds, "pedestrian_s"].mean()),
                    )
                    if seeds
                    else math.nan
                )
        for col in (
            "pedestrian_s",
            "penalty_s",
            "interventions",
            "split_cut",
            "split_replaced",
            "split_cut_foreseeable",
        ):
            if col in g:
                for label in ("S3", "S1", "S2"):
                    row[f"{col}_{label}"] = float(g.loc[g["label"] == label, col].mean())
        row["danger_max"] = int(g["danger"].max()) if "danger" in g else 0
        row["curb_max"] = (
            float(g["ped_curb_wait_max"].max()) if "ped_curb_wait_max" in g else math.nan
        )
        if "preempt_timeouts" in g:
            row["preempt_timeouts_S3"] = int(
                g.loc[g["label"] == CANDIDATE, "preempt_timeouts"].sum()
            )
        row["worse_than_s1"] = bool(
            math.isfinite(row["vs_S1_pct"])
            and row["vs_S1_pct"] > 100.0 * float(worse_rel) + REL_TOL
        )
        rows.append(row)
    return pd.DataFrame(rows)


def stress(
    rl7: Path = RL7_CONFIG,
    workers: int | None = None,
    *,
    scenario: Path = PITIE_RL_S3_CONFIG,
    out: Path | None = None,
    backend: EnvBackend | None = None,
) -> Path:
    """D78: S3, S0, S1, S2 on the stress cells; returns `stress/summary.csv`."""
    cfg = load_config(rl7)
    seeds = _seeds_of(cfg, "stress")
    check_seeds(seeds, cfg, ("stress",), "stress")
    cells = stress_cells(cfg)
    worse_rel = float((protocol(cfg).get("stress") or {}).get("worse_than_s1_rel", 0.05))
    root = study_root(cfg, out)
    sdir, work_root = root / STRESS_DIR, root / WORK_DIR
    for p in (sdir, work_root):
        check_out(p)
    s3 = s3_model(scenario)
    backend = backend or real_backend()
    workers = int(workers or default_workers())
    config = str(s3["config"])
    meta, eps = [], []
    for cell, override in cells.items():
        mode = str(override.get("mode", "train"))
        text = json.dumps(override, sort_keys=True)
        for label, policy in STRESS_LABELS.items():
            for s in seeds:
                meta.append(
                    {
                        "cell": cell,
                        "label": label,
                        "policy": policy,
                        "mode": mode,
                        "seed": s,
                        "override": text,
                    }
                )
                if policy == "model":
                    eps.append(
                        Episode(
                            config,
                            mode,
                            int(s),
                            "model",
                            s3["path"].as_posix(),
                            s3["algo"],
                            override,
                        )
                    )
                else:
                    eps.append(Episode(config, mode, int(s), policy, override=override))
    log.info("stress: %d cells, %d episodes", len(cells), len(eps))
    df = _frame(meta, run_tasks(backend, eps, workers, work_root))
    sdir.mkdir(parents=True, exist_ok=True)
    df.assign(model_version=s3["model_version"]).to_csv(sdir / "episodes.csv", index=False)
    summary = stress_summary(df, worse_rel, list(cells))
    summary.to_csv(sdir / "summary.csv", index=False)
    worse = summary.loc[summary["worse_than_s1"], "cell"].tolist()
    if worse:
        log.warning("stress: S3 more than %g %% worse than S1 in %s", 100 * worse_rel, worse)
    return sdir / "summary.csv"


# -- D79: dev-test look and the interim 10.6 verdict --------------------------------------
def paired_tables(
    kpis: pd.DataFrame, pairs: Sequence[Sequence[str]], equity_pct: float, alpha: float
) -> pd.DataFrame:
    """`evaluation.compare` of every (reference, candidate) pair on the 7 headline KPIs
    (Holm within each pair); s0 = the reference, s1 = the candidate."""
    keep = {f"{u}/{k}" for u, k in HEADLINE7}
    k = kpis[(kpis["user_type"] + "/" + kpis["kpi"]).isin(keep)]
    tabs = []
    for ref, cand in pairs:
        a = k[k["label"] == ref][["seed", "user_type", "kpi", "value"]]
        b = k[k["label"] == cand][["seed", "user_type", "kpi", "value"]]
        t = compare(a, b, [list(h) for h in HEADLINE7], equity_pct=equity_pct, alpha=alpha)
        tabs.append(t.assign(ref=ref, cand=cand))
    return pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()


def _row(paired: pd.DataFrame, ref: str, key: str) -> pd.Series | None:
    user, kpi = key.split("/", 1)
    r = paired[(paired["ref"] == ref) & (paired["user_type"] == user) & (paired["kpi"] == kpi)]
    return None if r.empty else r.iloc[0]


def _sense(kpi: str) -> int:
    return {"lower": -1, "higher": 1}.get(better_if(kpi), 0)


def c3_violations(
    safety: pd.DataFrame, curb_max_s: float, cand: str = CANDIDATE, ref: str = "S2"
) -> list[str]:
    """D79 C3 on the full runs: in every candidate run danger 0, collisions 0, teleports
    0, curb maximum <= `curb_max_s`, every preemption request served (or dropped as a
    timeout) and, over the seeds, no more preemption timeouts than `ref`."""
    out = []
    s = safety[safety["label"] == cand]
    if s.empty:
        return [f"no {cand} run"]
    for r in s.itertuples(index=False):
        why = []
        for col in ("danger", "collisions", "teleports"):
            v = int(getattr(r, col, 0) or 0)
            if v:
                why.append(f"{col} {v}")
        curb = _float(getattr(r, "ped_curb_wait_max", math.nan))
        if not math.isfinite(curb) or curb > float(curb_max_s):
            why.append(f"curb max {curb:g} s > {float(curb_max_s):g} s")
        req = int(getattr(r, "preempt_requests", 0) or 0)
        served = int(getattr(r, "preempt_served", 0) or 0)
        timeouts = int(getattr(r, "preempt_timeouts", 0) or 0)
        if served + timeouts < req:
            why.append(f"preemption: {req - served - timeouts} of {req} requests unresolved")
        if why:
            out.append(f"seed {r.seed}: " + ", ".join(why))
    if "preempt_timeouts" in safety:
        seeds = set(s["seed"])
        other = safety[(safety["label"] == ref) & safety["seed"].isin(seeds)]
        t_c, t_r = int(s["preempt_timeouts"].sum()), int(other["preempt_timeouts"].sum())
        if t_c > t_r:
            out.append(f"preemption timeouts {t_c} > {ref}'s {t_r} on the same seeds")
    return out


def c4_pass(runs_ok: bool, same: bool | None) -> bool | None:
    """D79 C4: False when the run count fails or reproduce7 differs, None (pending)
    while reproduce7 has not run, True only when both hold."""
    if not runs_ok or same is False:
        return False
    return None if same is None else True


def verdict_of(passes: Sequence[bool | None]) -> str:
    """ "fallback" when a criterion failed, "pending" when one is not evaluated yet, else
    "pass" (D79: PASS iff all of C1-C4; a missing check is never a pass)."""
    if any(p is not None and not p for p in passes):
        return "fallback"
    return "pending" if any(p is None for p in passes) else "pass"


def compute_verdict(
    paired: pd.DataFrame,
    safety: pd.DataFrame,
    holdout_rows: Sequence[Mapping],
    repro: Mapping | None,
    stress_worse: Sequence[str] | None,
    rl_cfg: Mapping,
    *,
    seeds: Sequence[int] = (),
    model_version: str | None = None,
) -> dict:
    """The interim 10.6 verdict (D79), computed exactly from the pre-registered
    `protocol.devtest` and `protocol.gate_10_6` blocks:

    - C1: primary KPI, S3 vs S1: HL < 0 and exact two-sided Wilcoxon p < alpha;
    - C2: no equity KPI more than `equity_rel` worse than S1 (`compare.worse_than` on the
      means over seeds);
    - C3: `c3_violations` empty;
    - C4: >= `repro_min_runs` stage-F picks with a mean holdout window cost below S1's on
      the same windows, and the reproduce7 check the same. Without a reproduce7 result
      C4 `pass` is None (pending), never true: a check that has not run is not passed;

    "pass" iff all four pass, "fallback" as soon as one fails, else "pending" (C1-C3 and
    the C4 run count hold, reproduce7 has not run: it runs only after the commit, D82).
    The frozen verdict.json is completed by `final_verdict` with the reproduce7 result,
    without a second look. Reported, not gating: the stress cells
    worse than S1, and the S2 framing ("RL ≈ rules" unless S3 beats S2 with C1-type
    evidence by more than `s2_parity_rel`)."""
    prot = protocol(rl_cfg)
    dv, gate = prot["devtest"], prot["gate_10_6"]
    alpha, primary = float(dv.get("alpha", 0.05)), str(dv["primary"])
    pending: list[str] = []

    r = _row(paired, "S1", primary)
    if r is None:
        c1 = {"kpi": primary, "hl": None, "p": None, "n": 0, "pass": False, "why": "missing"}
    else:
        hl, p = _float(r["hl_diff"]), _float(r["p_value"])
        c1 = {
            "kpi": primary,
            "hl": hl,
            "hl_ci": [_float(r["hl_ci_low"]), _float(r["hl_ci_high"])],
            "p": p,
            "p_holm": _float(r["p_holm"]),
            "n": int(r["n_pairs"]),
            "s1_mean": _float(r["s0_mean"]),
            "s3_mean": _float(r["s1_mean"]),
            "pass": bool(math.isfinite(hl) and math.isfinite(p) and hl < 0 and p < alpha),
        }

    pct = 100.0 * float(gate["equity_rel"])
    rows = []
    for key in gate["equity_kpis"]:
        e = _row(paired, "S1", str(key))
        if e is None:
            rows.append(
                {
                    "kpi": key,
                    "s1_mean": None,
                    "s3_mean": None,
                    "rel_pct": None,
                    "worse": True,
                    "why": "missing",
                }
            )
            continue
        m0, m1 = _float(e["s0_mean"]), _float(e["s1_mean"])
        rel = 100.0 * (m1 / m0 - 1.0) if m0 else math.nan
        worse = worse_than(m0, m1, rel, _sense(str(e["kpi"])), pct + REL_TOL)
        rows.append(
            {"kpi": key, "s1_mean": m0, "s3_mean": m1, "rel_pct": rel, "worse": bool(worse)}
        )
    c2 = {"limit_pct": pct, "rows": rows, "pass": not any(x["worse"] for x in rows)}

    violations = c3_violations(safety, float(gate["curb_max_s"]))
    c3 = {"violations": violations, "pass": not violations}

    below = [
        h["run"]
        for h in holdout_rows
        if math.isfinite(_float(h.get("holdout_w30")))
        and _float(h["holdout_w30"]) < _float(h.get("w30_S1"))
    ]
    need = int(gate["repro_min_runs"])
    if repro is None:
        repro_doc: dict[str, Any] = {"status": "pending", "same": None}
        pending.append("C4.repro")
    else:
        repro_doc = {"status": "done", "same": bool(repro.get("same")), "head": repro.get("head")}
    runs_ok = len(below) >= need
    c4 = {
        "runs_below_s1": len(below),
        "of": len(holdout_rows),
        "required": need,
        "runs": below,
        "runs_pass": bool(runs_ok),
        "repro": repro_doc,
        "pass": c4_pass(runs_ok, repro_doc["same"]),
    }

    s2 = _row(paired, "S2", primary)
    if s2 is None:
        framing: dict[str, Any] = {"framing": "unknown", "text": "no S2 pair"}
    else:
        hl, p, m = _float(s2["hl_diff"]), _float(s2["p_value"]), _float(s2["s0_mean"])
        rel = hl / m if m else math.nan
        within = math.isfinite(rel) and abs(rel) <= float(gate["s2_parity_rel"]) + REL_TOL
        if hl < 0 and p < alpha and not within:
            label, text = "better", "S3 beats rules (S2) on the primary KPI"
        elif hl > 0 and p < alpha and not within:
            label, text = "worse", "S3 is worse than rules (S2) on the primary KPI"
        else:
            label, text = "approx", "RL ≈ rules"
        framing = {
            "framing": label,
            "text": text,
            "hl": hl,
            "p": p,
            "hl_rel": rel,
            "within_parity": within,
        }

    return {
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "C4": c4,
        "stress_cells_worse": None if stress_worse is None else list(stress_worse),
        "s2_framing": framing,
        "verdict": verdict_of([c1["pass"], c2["pass"], c3["pass"], c4["pass"]]),
        "pending": pending,
        "seeds": [int(s) for s in seeds],
        "model_version": model_version,
        "alpha": alpha,
        "rule": (
            "RL beats actuated iff C1 (primary S3 vs S1: HL < 0 and exact two-sided "
            f"Wilcoxon p < {alpha:g}), C2 (no equity KPI > {pct:g} % worse than S1), C3 "
            f"(safety in every S3 run) and C4 (>= {need} stage-F picks below S1 on the "
            "holdout windows; reproduce7 the same). Otherwise 10.6 applies (D79)."
        ),
    }


def latest_reproduce(root: Path) -> dict | None:
    """The newest `_reproduce/<HEAD>/reproduce7.json` (tasks.py reproduce7), or None."""
    found = sorted(
        Path(root).glob(f"{REPRODUCE_DIR}/*/{REPRODUCE_FILE}"), key=lambda p: p.stat().st_mtime
    )
    return read_json(found[-1]) if found else None


def final_verdict(verdict: Mapping, repro: Mapping | None) -> dict:
    """The interim 10.6 verdict of a frozen `devtest/verdict.json` completed with the
    reproduce7 result (D79 C4, D82) without a second look: C1-C3 and the C4 run count
    as computed at the look, reproduce7 from `repro` when the look had none. "pass" only
    once reproduce7 reproduced (`same` true), "fallback" when it did not or any other
    criterion failed, "pending" while it has not run. `verdict_at_look` keeps the frozen
    value. This is what a page states (report7), never verdict.json's value alone."""
    v = copy.deepcopy(dict(verdict))
    c4 = dict(v.get("C4") or {})
    rec = dict(c4["repro"]) if isinstance(c4.get("repro"), Mapping) else {}
    same = rec.get("same")
    if same is None and repro is not None:
        rec = {
            "status": "done",
            "same": bool(repro.get("same")),
            "head": repro.get("head"),
            "after_look": True,
        }
        same = rec["same"]
    elif same is None:
        rec = {"status": "pending", "same": None}
    runs_ok = c4.get("runs_pass")
    if runs_ok is None:  # a verdict.json written before `runs_pass` was recorded
        below, need = c4.get("runs_below_s1"), c4.get("required")
        n = len(below) if isinstance(below, list | tuple) else int(below or 0)
        runs_ok = n >= int(need) if need is not None else bool(c4.get("pass"))
    c4 |= {"runs_pass": bool(runs_ok), "repro": rec, "pass": c4_pass(bool(runs_ok), same)}
    v["C4"] = c4
    v["verdict_at_look"] = verdict.get("verdict")
    v["verdict"] = verdict_of(
        [(v.get(k) or {}).get("pass") for k in ("C1", "C2", "C3")] + [c4["pass"]]
    )
    pending = [p for p in v.get("pending") or [] if p != "C4.repro"]
    v["pending"] = pending + (["C4.repro"] if same is None else [])
    return v


def resolve_verdict(root: Path) -> dict | None:
    """`final_verdict` of `<root>/devtest/verdict.json` and the newest reproduce7 result
    (`latest_reproduce`), or None before the dev-test look."""
    v = read_json(Path(root) / DEVTEST_DIR / VERDICT_FILE)
    return None if v is None else final_verdict(v, latest_reproduce(root))


def c4_holdout_rows(root: Path) -> list[dict]:
    """The stage-F candidates of select/decision.json with their holdout summary rows
    (`holdout_w30`, `w30_S1`). ConfigError when the selection or a holdout is missing:
    the single dev-test look waits for them."""
    dec = read_json(Path(root) / SELECT_DIR / SELECT_DECISION_FILE)
    if dec is None:
        raise ConfigError("devtest: run `rl.study select` first (C4 needs the stage-F picks)")
    path = Path(root) / HOLDOUT_DIR / HOLDOUT_SUMMARY_FILE
    summ = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=["dir"])
    rows, missing = [], []
    for c in dec["candidates"]:
        hit = summ[summ["dir"] == c["dir"]]
        if hit.empty or not math.isfinite(_float(hit.iloc[0]["holdout_w30"])):
            missing.append(c["run"])
            continue
        h = hit.iloc[0]
        rows.append(
            {"run": c["run"], "holdout_w30": float(h["holdout_w30"]), "w30_S1": float(h["w30_S1"])}
        )
    if missing:
        raise ConfigError(
            f"devtest: no holdout for {missing} (run `rl.study holdout` on the F runs)"
        )
    return rows


def window_frame(
    backend: EnvBackend, s3: Mapping, seeds: Sequence[int], workers: int, work_root: Path
) -> pd.DataFrame:
    """The dev-test 30-min windows of S3 and the reference policies (`rl.evaluate.compare_frame`
    on S3's own run config)."""
    model = {"path": Path(s3["path"]), "algo": s3["algo"]}
    return compare_frame(
        backend, s3["config"], seeds, model, workers, WINDOW_POLICIES, work_root=work_root
    )


def window_block(df: pd.DataFrame) -> dict:
    out = {}
    for label, g in df.groupby("label", sort=False):
        d: dict[str, Any] = {"cost_mean": float(g["cost"].mean()), "n": len(g)}
        if label != "S1":
            a = g.set_index("seed")["cost"]
            b = df[df["label"] == "S1"].set_index("seed")["cost"]
            seeds = sorted(set(a.index) & set(b.index))
            if seeds:
                x, y = a.loc[seeds].to_numpy(float), b.loc[seeds].to_numpy(float)
                d |= {
                    "vs_S1_pct": pct_change(x.mean(), y.mean()),
                    "wins_vs_S1": int((x < y).sum()),
                    "p_vs_S1": float(wilcoxon(x, y).pvalue),
                }
        out[str(label)] = d
    return out


def devtest(
    rl7: Path = RL7_CONFIG,
    workers: int | None = None,
    *,
    scenario: Path = PITIE_RL_S3_CONFIG,
    out: Path | None = None,
    backend: EnvBackend | None = None,
) -> dict:
    """D79: the single dev-test look and the interim 10.6 verdict; writes and returns
    `devtest/verdict.json`. Refuses when it exists. Before anything is played, the look
    is recorded in `devtest/look.json` (written once, exclusively: model, scenario,
    seeds, rl7 sha, selected run); a later call with another identity is refused, so a
    crash after the look allows only a resume of that same look (finished KPI runs are
    reused). Every attempt and crash goes to `devtest/attempts.jsonl` and into
    verdict.json `look` (`resumed`, `crashes`). The S3 model must be the selected pick
    of select/decision.json (registered windows, not a development selection)."""
    rl7 = Path(rl7)
    cfg = load_config(rl7)
    dv = protocol(cfg)["devtest"]
    root = study_root(cfg, out)
    ddir, work_root = root / DEVTEST_DIR, root / WORK_DIR
    verdict_path = ddir / VERDICT_FILE
    if verdict_path.exists():
        raise FileExistsError(f"{verdict_path} exists: the dev-test seeds are looked at once (D79)")
    for p in (ddir, work_root):
        check_out(p)
    seeds = seed_range(dv["seeds"])
    check_seeds(seeds, cfg, ("dev_test",), "devtest")
    s3 = s3_model(scenario)
    holdout_rows = c4_holdout_rows(root)  # before the look: fail early, nothing run
    chosen = check_selected(root, s3)
    backend = backend or real_backend()
    workers = int(workers or default_workers())
    identity = look_identity(s3, Path(scenario), seeds, rl7, chosen)
    look, attempt = open_look(ddir, identity)  # the look starts here
    try:
        verdict = _look(
            cfg, rl7, root, ddir, work_root, s3, Path(scenario), seeds, holdout_rows, backend,
            workers,
        )  # fmt: skip
        attempts = read_attempts(ddir)
        verdict["look"] = {
            **look,
            "attempts": attempt,
            "resumed": attempt > 1,
            "crashes": [a for a in attempts if a.get("status") == "crashed"],
        }
        write_json(verdict_path, verdict, exclusive=True)
    except BaseException as e:
        log_attempt(ddir, attempt, "crashed", error=f"{type(e).__name__}: {e}"[:500])
        raise
    log_attempt(ddir, attempt, "done")
    log.info("devtest: interim 10.6 verdict %s (%s)", verdict["verdict"], verdict_path.as_posix())
    return verdict


def check_selected(root: Path, s3: Mapping) -> str:
    """The run of select/decision.json that the dev-test looks at: the selection must be
    on the registered windows and the S3 scenario's model its chosen pick (same sha256,
    `publish` copies the bytes). Returns the chosen run name."""
    dec = read_json(Path(root) / SELECT_DIR / SELECT_DECISION_FILE)
    if dec is None:
        raise ConfigError("devtest: run `rl.study select` first (S3 is its chosen pick)")
    if dec.get("development"):
        raise ConfigError("devtest: select/decision.json was made on development windows")
    chosen = dec.get("chosen")
    cand = next((c for c in dec.get("candidates") or [] if c.get("run") == chosen), None)
    if cand is None:
        raise ConfigError(f"devtest: the selection's chosen run {chosen!r} is not a candidate")
    want = cand.get("pick_model_sha256")
    if want != s3["model_sha256"]:
        raise ConfigError(
            f"devtest: the S3 scenario's model ({s3['model_sha256'][:12]}) is not the selected "
            f"pick {chosen} ({str(want)[:12]}): publish the selection and pin it"
        )
    return str(chosen)


def look_identity(
    s3: Mapping, scenario: Path, seeds: Sequence[int], rl7: Path, chosen: str
) -> dict:
    """What the dev-test look is of (`LOOK_KEYS`), as JSON values."""
    ident = {
        "model_sha256": s3["model_sha256"],
        "model_version": s3.get("model_version"),
        "selected_run": chosen,
        "scenario": _rel_posix(scenario),
        "scenario_sha256": text_sha256(scenario),
        "rl_config": _rel_posix(Path(s3["config"])),
        "rl_config_sha256": text_sha256(Path(s3["config"])),
        "seeds": [int(s) for s in seeds],
        "rl7_sha256": text_sha256(rl7),
    }
    return json.loads(json_text(ident))


def open_look(ddir: Path, identity: Mapping) -> tuple[dict, int]:
    """Record the dev-test look before anything is played (D79, looked at once):
    `look.json` is created exclusively with `identity`, the code version and the start
    time; when it exists (a crash after the look), the call must be the same look
    (every `LOOK_KEYS` value equal) or it is refused. Returns (look, attempt number) and
    logs the attempt in `attempts.jsonl`."""
    path = Path(ddir) / LOOK_FILE
    first = {**identity, "code_version": code_version(), "started_at": _now()}
    try:
        write_json(path, first, exclusive=True)
    except FileExistsError:
        pass
    look = read_json(path) or {}
    diff = [k for k in LOOK_KEYS if look.get(k) != identity.get(k)]
    if diff:
        was = ", ".join(f"{k} {look.get(k)!r} (now {identity.get(k)!r})" for k in diff)
        log_attempt(ddir, 0, "refused", differs={k: identity.get(k) for k in diff})
        raise ConfigError(
            f"{path}: the dev-test seeds were already looked at with another {was}; a second "
            "look is refused (D79: looked at once). Only a resume of the same look runs; a "
            "rerun needs a dated `changes` entry and both results are shown."
        )
    attempt = 1 + sum(a.get("status") == "started" for a in read_attempts(ddir))
    log_attempt(ddir, attempt, "started")
    if attempt > 1:
        log.warning(
            "devtest: resuming the look of %s (attempt %d)", look["model_sha256"][:12], attempt
        )
    return look, attempt


def read_attempts(ddir: Path) -> list[dict]:
    path = Path(ddir) / ATTEMPTS_FILE
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def log_attempt(ddir: Path, attempt: int, status: str, **extra: Any) -> None:
    rec = {"attempt": int(attempt), "status": status, "code_version": code_version(), "at": _now()}
    with (Path(ddir) / ATTEMPTS_FILE).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(_clean({**rec, **extra}), ensure_ascii=True) + "\n")


def _look(
    cfg: Mapping,
    rl7: Path,
    root: Path,
    ddir: Path,
    work_root: Path,
    s3: Mapping,
    scenario: Path,
    seeds: Sequence[int],
    holdout_rows: Sequence[Mapping],
    backend: EnvBackend,
    workers: int,
) -> dict:
    """The dev-test windows, KPI runs, pairs and the verdict (`devtest` after the look
    is recorded)."""
    prot = protocol(cfg)
    dv, gate = prot["devtest"], prot["gate_10_6"]
    windows = window_frame(backend, s3, seeds, workers, work_root)
    windows.to_csv(ddir / "compare.csv", index=False)
    configs = kpi_configs(None, {**KPI_BASELINES, CANDIDATE: Path(scenario)})
    kpi_runs(seeds, ddir, configs)
    occupancy = load_config(EVALUATION_CONFIG)["occupancy"]
    kpis, safety = kpi_frames(ddir, configs, seeds, occupancy)
    kpis.to_csv(ddir / "kpi.csv", index=False)
    safety.to_csv(ddir / "kpi_safety.csv", index=False)
    alpha = float(dv.get("alpha", 0.05))
    paired = paired_tables(kpis, dv["pairs"], 100.0 * float(gate["equity_rel"]), alpha)
    paired.to_csv(ddir / "paired.csv", index=False)
    stress_path = root / STRESS_DIR / "summary.csv"
    worse = None
    if stress_path.is_file():
        st = pd.read_csv(stress_path)
        worse = st.loc[st["worse_than_s1"].astype(bool), "cell"].tolist()
    verdict = compute_verdict(
        paired,
        safety,
        holdout_rows,
        latest_reproduce(root),
        worse,
        cfg,
        seeds=seeds,
        model_version=s3["model_version"],
    )
    return verdict | {
        "model_sha256": s3["model_sha256"],
        "scenario": _rel_posix(Path(scenario)),
        "windows": window_block(windows),
        "rl7_sha256": text_sha256(rl7),
        "code_version": code_version(),
        "runs_code_version": sorted({str(v) for v in safety.get("code_version", [])}),
        "created_at": _now(),
    }


# -- CLI -------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, default=RL7_CONFIG, help="Part 7 config (rl7.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)
    dt = sub.add_parser("decide-t", help="stage-T decision (D73) -> decision_T.json")
    dt.add_argument("--force", action="store_true", help="redo an existing decision")
    dt.add_argument("--runs", type=Path, help="stage-T runs folder (default <outputs.runs>/T)")
    dt.add_argument("--out", type=Path, help="study root (default: outputs.study)")
    dt.add_argument("--steps", type=int, help="the runs' budget (default: stage_t.steps)")
    dt.add_argument(
        "--validation-seeds",
        nargs="+",
        metavar="SEED",
        help="development windows the runs validated on (N or LO-HI; training seeds only, "
        "needs --out; default: seeds.selection)",
    )
    h = sub.add_parser("holdout", help="holdout scores of run picks (reporting only)")
    h.add_argument("runs", nargs="+", type=Path)
    h.add_argument("--curves", action="store_true", help="also score the kept checkpoints")
    h.add_argument("--workers", type=int)
    s = sub.add_parser("select", help="two-stage S3 selection (D75)")
    s.add_argument("--workers", type=int)
    pb = sub.add_parser("publish", help="write models/rl/<model_version>/ (D86)")
    pb.add_argument("--out", type=Path, help="models root (default: rl7.yaml outputs.models)")
    g = sub.add_parser("golden", help="write tests/data/rl_s3_golden.json")
    g.add_argument(  # `golden --config configs/pitie_rl_s3.yaml` (contract 4.2)
        "--config", dest="scenario", type=Path, default=PITIE_RL_S3_CONFIG, help="S3 scenario"
    )
    g.add_argument("--out", type=Path, default=GOLDEN_FILE)
    st = sub.add_parser("stress", help="stress panel (D78)")
    st.add_argument("--workers", type=int)
    st.add_argument("--scenario", type=Path, default=PITIE_RL_S3_CONFIG, help="S3 scenario")
    d = sub.add_parser("devtest", help="single dev-test look and 10.6 verdict (D79)")
    d.add_argument("--workers", type=int)
    d.add_argument("--scenario", type=Path, default=PITIE_RL_S3_CONFIG, help="S3 scenario")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.cmd == "decide-t":
            windows = parse_seeds(args.validation_seeds) if args.validation_seeds else None
            doc = decide_t(
                args.config, args.runs, args.out, args.steps, args.force, validation_seeds=windows
            )
            print(f"C* = {doc['chosen']}")
        elif args.cmd == "holdout":
            for path in holdout(args.runs, args.config, args.curves, args.workers):
                print(path.as_posix())
        elif args.cmd == "select":
            print(f"S3 = {select(args.config, args.workers)['chosen']}")
        elif args.cmd == "publish":
            dest = publish(args.config, args.out)
            manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
            print(dest.as_posix())
            print("configs/pitie_rl_s3.yaml:")
            print(s3_config_text(manifest, dest / "model.zip"), end="")
        elif args.cmd == "golden":
            print(golden(args.scenario, args.out).as_posix())
        elif args.cmd == "stress":
            print(stress(args.config, args.workers, scenario=args.scenario).as_posix())
        elif args.cmd == "devtest":
            v = devtest(args.config, args.workers, scenario=args.scenario)
            pend = f" (pending: {', '.join(v['pending'])})" if v.get("pending") else ""
            print(f"interim 10.6 verdict: {v['verdict']}{pend}")
    except (ConfigError, FileExistsError) as e:
        raise SystemExit(f"error: {e}") from None
    return 0


if __name__ == "__main__":
    sys.exit(main())

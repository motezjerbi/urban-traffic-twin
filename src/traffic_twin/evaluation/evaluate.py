"""Pre-registered evaluation of the Pitié controllers (Parts 4 and 8; spec 13.2, 13.3,
A-03 to A-06), driven by configs/evaluation.yaml.

    python -m traffic_twin.evaluation.evaluate --config configs/evaluation.yaml
        [--family main|ambulance_probe|all] [--run-missing] [--results DIR]
        [--docs docs/evaluation.md]

For each family (main: S0, S1, S2, S2_noP, S3, S4; ambulance_probe: S0, S2, S2_noP, S3,
S4):

- pairing (`check_family`): every scenario has a run for every seed, made with the
  CURRENT config (run.json `config_hash` / `demand_hash` equal those of the config) and
  KPI version >= `kpi_version`; one config_hash per scenario; one demand_hash for the
  whole family; and every scenario reports the same (user, kpi) set in every seed
  (A-03; a KPI missing from one run, e.g. an ambulance that never arrived, is refused).
  `--run-missing` runs absent seeds and reruns stale ones first;
- the person-weighted delay (user `all`, `person_delay_mean`, D29) per run from
  trip.parquet, for the pre-registered occupancies and the bus sensitivity values, added
  to the KPI table before pairing;
- one `compare()` per pre-registered pair (exact paired Wilcoxon, Holm over the 7 Part 3
  headline KPIs within the pair only, D28);
- the equity table (D29): relative change vs the baseline of every KPI (percentage
  points for rates), the 5 % flag and the hidden-degradation flag (13.3);
- A-04 (D27) over every run of the `a04.families`: 0 danger audit counts, 0 collisions,
  longest pedestrian curb wait (`ped_curb_wait_max`, creeps at the curb included) <=
  the scenario's `safety.ped_max_wait_s` (90 s; amended 2026-10-12, the longest single
  wait `ped_wait_single_max` is reported alongside), 0 guard interventions for `fixed`
  and `actuated`;
- A-05 (D26): the pre-registered one-sided exact Wilcoxon on the ambulance travel time
  from its planned departure, plus the secondary tests. A test whose sample is
  incomplete (fewer pairs than seeds, or a run whose ambulances did not all arrive) is
  INVALID, never PASS;
- Part 8 (`evaluation.final`, D94-D100): A-06 (S3 against S1, S2 and S0 on the person
  delay, procedural PASS / INVALID, never FAIL on direction), the spec 10.6 check on the
  evaluation seeds (reported, not gating), the final comparison table, the queue series,
  the masked KPI digest and the check of the frozen S3 inputs (D100).

Writes `<out>/<family>/{pairs/<a>__vs__<b>.{csv,md}, equity.csv, a04.csv, kpis.csv}` and
`<out>/{a05.json, summary.json, provenance.json, final_table.csv, a06.json, series.csv,
digest.json}`; `<out>` is `outputs.dir` of evaluation.yaml, or `<results>/evaluation`
with `--results`. A partial evaluation (`--family`) does not rewrite a D98 file that
reports on a family it did not evaluate (`part8_skipped`). `--docs` renders the page
(docs/evaluation.md, the D99 sections; never edited by hand). Exit status 1 on a config,
pairing or staleness error, or when A-04 FAILS (the outputs are written first); an A-05
FAIL or INVALID and an A-06 INVALID are reported and exit 0.

Run lock (Part 8, D92): before launching any missing or stale run, `--run-missing`
exits 2 ("run lock: ...") and launches nothing when a planned run is on a
pre-registered evaluation seed (seeds 1-10 of the canonical configs/evaluation.yaml,
whatever `--config` says, plus the config's own `seeds.evaluation`), into ANY results
folder, while the tree is dirty, the config has no `a06` block, or no commit adds
`a06:` yet. Seeds 1-10 are run only by the registered `tasks.py part8` step on the
clean registration commit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_twin.config import ConfigError, get, load_config, load_scenario
from traffic_twin.evaluation.compare import (
    COMPARE_COLUMNS,
    QUALIFIER_SEP,
    _sense,
    base_kpi,
    better_if,
    compare,
    diff_text,
    incomplete_rows,
    paired,
    unit,
    worse_than,
)
from traffic_twin.evaluation.final import (
    FINAL_EXTRAS,
    OUTCOME_KEYS,
    a06_report,
    check_model_inputs,
    digest,
    final_table,
    model_inputs,
    quantile_names,
    registration_record,
    rl_scenario_problems,
    series,
    split_key,
)
from traffic_twin.evaluation.runs import (
    Runs,
    expected_hashes,
    is_complete,
    kpi_version,
    load_runs,
    read_meta,
    run_missing,
    run_path,
    short,
    stale_reason,
)
from traffic_twin.evaluation.stats import hodges_lehmann, wilcoxon
from traffic_twin.paths import (
    EVAL_A06,
    EVAL_DIGEST,
    EVAL_FINAL_TABLE,
    EVAL_SERIES,
    EVALUATION_CONFIG,
)

log = logging.getLogger(__name__)

FAMILIES = ("main", "ambulance_probe", "all")
EVALUATION_KEYS = (
    "preregistered", "seeds.evaluation", "kpi_version", "headline_from", "families",
    "a05.primary", "a04.families", "a04.danger_keys", "a04.collisions_max",
    "a04.ped_curb_wait_max_s", "equity.pct", "equity.primary", "equity.global",
    "occupancy", "outputs.dir", "outputs.figures",
)  # fmt: skip
ROLES = ("part3", "confirmatory", "exploratory", "secondary")
ALTERNATIVES = ("less", "greater", "two-sided")
CHANGE_KEYS = ("date", "what", "why")
SPEC_CONTROLLERS = ("fixed", "actuated", "rules", "rl")  # spec annex B
LATER_PARTS = {"rl": "RL in Parts 6-7"}
RUN_TABLES = ("kpis", "trips", "events")
KPI_KEYS = ["seed", "user_type", "kpi"]
PAIR_COLUMNS = [*COMPARE_COLUMNS, "family", "role"]
EQUITY_COLUMNS = [
    "family", "scenario", "user_type", "kpi", "base_kpi", "qualifier", "baseline_mean", "mean",
    "abs_change", "rel_change_pct", "pp_change", "p_value", "worse_flag", "significant_flag",
    "primary", "subgroup",
]  # fmt: skip
PERSON_DELAY = "person_delay_mean"
SUMMARY_A05_KEYS = (
    "family", "a", "b", "kpi", "n", "expected_n", "hl", "hl_ci", "p_one_sided", "valid",
    "problems", "status",
)  # fmt: skip
SENSITIVITY_PREFIX = "occ_"  # person_delay_mean__occ_bus_20
PREEMPTION_COUNTS = ("requests", "served", "timeouts", "active_s", "hold_s", "jumps", "ped_yield")
# The Part 8 page (D99, exact text; reports include a section by the text after the number)
SECTIONS = (
    "1. Final comparison table (Part 8)",
    "2. A-06: S3 against the baselines (Part 8)",
    "3. Spec 10.6 check on seeds 1-10 (Part 8)",
    "4. Runs, code versions, seeds and demand",
    "5. Pre-registration",
    "6. ASSUMED inputs added in Parts 4 and 8",
    "7. KPI definitions",
    "8. KPIs per user and scenario",
    "9. Pairs vs the baseline (Holm)",
    "10. Other confirmatory pairs",
    "11. Secondary and exploratory pairs (ablation, probe attribution, S4)",
    "12. Equity and hidden degradation (13.3)",
    "13. Acceptance: A-03 to A-06",
    "14. Guard and controller activity",
    "15. Limitations",
)
PART8_TITLE = "# Evaluation: S0 to S4 on the Pitié-Salpêtrière junction (Parts 4 and 8)"
RUN_LOCK_EXIT = 2  # exit status of `--run-missing` when the run lock refuses (D92)
# Base KPI -> (SUMO source, formula). Rendered in section 4 for the KPIs the runs report.
TRACI = "TraCI per step"
CURB = (
    "TraCI per step: curb clock from `person.getWaitingTime`, kept running through creeps "
    "at a red curb"
)
KPI_DEFINITIONS = {
    "count": ("tripinfo / personinfo", "finished trips (pedestrians: persons)"),
    "travel_time": ("tripinfo `duration`", "mean in-network travel time (walks: duration)"),
    "time_loss_mean": ("tripinfo `timeLoss`", "mean time lost (insertion delay excluded)"),
    "wait_mean": ("tripinfo `waitingTime`", "mean time standing (speed < 0.1 m/s)"),
    "wait_p95": ("tripinfo `waitingTime`", "95th percentile of the time standing"),
    "stops": ("tripinfo `waitingCount`", "mean number of stops"),
    "depart_delay_mean": ("tripinfo `departDelay`", "mean insertion delay"),
    "depart_delay_max": ("tripinfo `departDelay`", "largest insertion delay"),
    "delay_mean": ("bus tripinfo `timeLoss`", "mean bus delay (scheduled dwell excluded)"),
    "headway_std": ("stopinfo `started`", "headway std per (stop, line), mean over groups"),
    "dwell_mean": ("stopinfo", "mean `ended - started` (set by the dwell config)"),
    "ped_wait_mean": ("personinfo walk `timeLoss`", "mean time lost per walk"),
    "ped_wait_max": ("personinfo walk `timeLoss`", "largest time lost over a walk"),
    "ped_wait_p95": ("personinfo walk `timeLoss`", "95th percentile of the time lost"),
    "ped_wait_single_mean": ("TraCI `person.getWaitingTime`", "longest single wait, mean"),
    "ped_wait_single_p95": ("TraCI `person.getWaitingTime`", "longest single wait, p95"),
    "ped_wait_single_max": ("TraCI `person.getWaitingTime`", "longest single wait"),
    "ped_curb_wait_mean": (CURB, "longest curb wait per person, mean"),
    "ped_curb_wait_p95": (CURB, "longest curb wait per person, p95"),
    "ped_curb_wait_max": (CURB, "longest curb wait (90 s target, A-04)"),
    "throughput_vph": ("tripinfo `arrival`", "car arrivals inside the window ÷ hours"),
    "delay_total_mean": ("tripinfo `timeLoss` + `departDelay`", "mean delay incl. insertion"),
    "travel_time_total": ("tripinfo `duration` + `departDelay`", "arrival − planned departure"),
    "stop_duration_mean": (
        "tripinfo `waitingTime`, `waitingCount`",
        "Σ time standing ÷ Σ stops (0 without a stop)",
    ),
    "crossing_complete_rate": (
        TRACI,
        "1 − exposed ÷ crossing traversals; exposed: a foe vehicle link showed G while "
        "the person was on the crossing",
    ),
    "crossing_red_entries": (TRACI, "entries onto a crossing while its own link showed r"),
    "box_halt_red_s": (TRACI, "vehicle-seconds halted inside the junction on a link not G/g"),
    "box_foe_onsets": (TRACI, "green onsets of a foe of a halted box vehicle's link"),
    "exit_blocked_s": (
        TRACI,
        "seconds with a vehicle halted (< 0.5 m/s) within 15 m of an exit lane's start",
    ),
    "exit_blocked_green_s": (TRACI, "the same while a link feeding that exit is green"),
    PERSON_DELAY: (
        "trip.parquet (evaluation)",
        "occupancy-weighted mean of vehicle `timeLoss` + `departDelay` and walk `timeLoss`",
    ),
}
QUALIFIER_WORDS = {
    "arm": "per arm",
    "flow": "per bus flow",
    "line": "per bus line",
    "crossing": "per crossing",
    "occ": "occupancy sensitivity",
}


class EvaluationError(ValueError):
    """The evaluation cannot run: the runs of a family cannot be paired (seeds, hashes,
    KPI version, demand or KPI set differ), or the evaluation config is invalid."""


class RunLockError(EvaluationError):
    """`run_lock` refused to launch runs on the pre-registered evaluation seeds (D92)."""


# -- run lock (Part 8, D92) ------------------------------------------------------------
def git_head() -> tuple[str, bool]:
    """(short HEAD hash, tree differs from HEAD), with `sim.results.code_version`
    semantics: ignored files (data/results, ...) and files under docs/ do not count. A
    missing git or repository counts as dirty ("unknown")."""
    git = ["git", "--no-optional-locks"]
    try:
        head = subprocess.run(
            [*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=normal", "--", ".",
             ":(exclude)docs"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return head, bool(status.strip())


def registration_commit(path: str | Path = EVALUATION_CONFIG) -> dict | None:
    """{commit, committed_at} of the FIRST commit that added `a06:` to the evaluation
    config (D92: `git log --reverse --format="%h %cI" -S "a06:" -- <path>`, first line);
    None when no commit adds it yet, or git or the repository is not available."""
    path = Path(path)
    try:
        out = subprocess.run(
            ["git", "--no-optional-locks", "log", "--reverse", "--format=%h %cI", "-S",
             "a06:", "--", path.name],
            cwd=path.resolve().parent, capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    lines = [line.split() for line in out.splitlines() if line.strip()]
    if not lines or len(lines[0]) != 2:
        return None
    commit, committed_at = lines[0]
    return {"commit": commit, "committed_at": committed_at}


REPO_ROOT = Path(__file__).resolve().parents[3]
# D33: the evaluation seeds registered on 2026-10-12; the lock falls back to them when the
# canonical configs/evaluation.yaml cannot be read (fail closed)
PREREGISTERED_SEEDS = tuple(range(1, 11))


def locked_seeds(ecfg: dict) -> set[int]:
    """The seeds `run_lock` protects: `seeds.evaluation` of the canonical
    configs/evaluation.yaml of this checkout, read independently of `--config`
    (PREREGISTERED_SEEDS when it cannot be read), united with those of `ecfg`. A config
    copy that registers other seeds therefore never unlocks seeds 1-10."""
    try:
        canonical = load_config(REPO_ROOT / EVALUATION_CONFIG)
        seeds = {int(s) for s in (canonical.get("seeds") or {}).get("evaluation") or []}
    except (ConfigError, OSError, AttributeError, TypeError, ValueError):
        seeds = set()
    seeds = seeds or set(PREREGISTERED_SEEDS)
    return seeds | {int(s) for s in (ecfg.get("seeds") or {}).get("evaluation") or []}


def run_lock(ecfg: dict, planned: list[tuple[str, int]], results: Path) -> None:
    """Refuse (RunLockError, "run lock: ...") to launch any of `planned` (scenario, seed)
    runs when one of them is on a pre-registered evaluation seed (`locked_seeds`: the
    canonical seeds 1-10 and the `seeds.evaluation` of `ecfg`) and the registration is
    not committed: the tree is dirty (`git_head`), or the config has no `a06` block, or
    no commit adds `a06:` to configs/evaluation.yaml yet (`registration_commit`). Seeds
    1-10 are then run only by the registered Part 8 step (`tasks.py part8`, D97) on the
    clean registration commit. Applies to every controller and to EVERY results folder
    (`results` only names it in the message): runs into a tmp folder would still show
    A-06 outcomes before the registration. Tests that need such runs monkeypatch
    `git_head` and `registration_commit`; dev and demo runs use other seeds (3210-3229,
    4000)."""
    eval_seeds = locked_seeds(ecfg)
    hits = sorted({(str(sc), int(sd)) for sc, sd in planned if int(sd) in eval_seeds})
    if not hits:
        return
    reasons = []
    head, dirty = git_head()
    if dirty:
        reasons.append(f"the tree differs from HEAD {head} (uncommitted or untracked files)")
    if "a06" not in ecfg:
        reasons.append("the evaluation config has no a06 block")
    if registration_commit(REPO_ROOT / EVALUATION_CONFIG) is None:
        reasons.append("no commit adds a06: to configs/evaluation.yaml yet")
    if not reasons:
        return
    shown = ", ".join(f"{sc} seed {sd}" for sc, sd in hits[:4])
    more = f" and {len(hits) - 4} more" if len(hits) > 4 else ""
    raise RunLockError(
        f"run lock: refusing {len(hits)} run(s) on the pre-registered evaluation seeds into "
        f"{Path(results).as_posix()} ({shown}{more}): {'; '.join(reasons)}. Seeds 1-10 are "
        "run only by the registered Part 8 step (`tasks.py part8`, D92/D97) on the clean "
        "registration commit; nothing was launched"
    )


def planned_runs(fam: Family, seeds: list[int], min_kpi_version: int | None) -> list:
    """(results dir, scenario, seed) of every run `run_missing` would launch for `fam`:
    absent seeds and stale runs (same test as `runs.run_missing`, nothing is run)."""
    out = []
    for m in fam.members.values():
        expected = expected_hashes(m.cfg)
        for seed in seeds:
            d = run_path(m.results, m.scenario, seed)
            if is_complete(d) and stale_reason(read_meta(d), expected, min_kpi_version) is None:
                continue
            out.append((Path(m.results), m.scenario, int(seed)))
    return out


# -- config ----------------------------------------------------------------------------
def load_evaluation(path: str | Path = EVALUATION_CONFIG) -> dict:
    """configs/evaluation.yaml with its keys and cross references checked: every pair,
    baseline and A-05 test names scenarios of an existing family, roles and alternatives
    are known, and each dated `changes` entry has a date, what and why."""
    ecfg = load_config(path, EVALUATION_KEYS)
    bad: list[str] = []
    families = ecfg["families"]
    if not isinstance(families, dict) or not families:
        raise ConfigError(f"{path}: families must be a non-empty mapping")
    for name, fam in families.items():
        scen = (fam or {}).get("scenarios") or {}
        if not scen:
            bad.append(f"family {name} has no scenarios")
            continue
        if fam.get("baseline") not in scen:
            bad.append(f"family {name}: baseline {fam.get('baseline')!r} is not a scenario")
        for i, p in enumerate(fam.get("pairs") or []):
            if p.get("a") not in scen or p.get("b") not in scen or p.get("a") == p.get("b"):
                bad.append(f"family {name}: pairs[{i}] must name two scenarios of the family")
            if p.get("role") not in ROLES:
                bad.append(f"family {name}: pairs[{i}] role must be one of {ROLES}")
    a05 = ecfg["a05"]
    for where, spec in [("a05.primary", a05["primary"])] + [
        (f"a05.secondary[{i}]", s) for i, s in enumerate(a05.get("secondary") or [])
    ]:
        scen = (families.get(spec.get("family")) or {}).get("scenarios") or {}
        if spec.get("a") not in scen or spec.get("b") not in scen:
            bad.append(f"{where} must name two scenarios of an existing family")
        if spec.get("alternative", "two-sided") not in ALTERNATIVES:
            bad.append(f"{where}: alternative must be one of {ALTERNATIVES}")
        if not spec.get("user") or not spec.get("kpi"):
            bad.append(f"{where} needs user and kpi")
    for name in ecfg["a04"]["families"]:
        if name not in families:
            bad.append(f"a04.families: unknown family {name!r}")
    bad += part8_problems(ecfg)
    changes = ecfg.get("changes") or []
    if not isinstance(changes, list):
        bad.append("changes must be a list of {date, what, why}")
        changes = []
    for i, c in enumerate(changes):
        missing = [k for k in CHANGE_KEYS if not (isinstance(c, dict) and c.get(k))]
        if missing:
            bad.append(f"changes[{i}] needs {', '.join(missing)}")
    if bad:
        raise ConfigError(f"{path}: {'; '.join(bad)}")
    ecfg["changes"] = changes
    return ecfg


A06_KEYS = (
    "family", "candidate", "user", "kpi", "alternative", "alpha", "comparisons", "primary",
    "n_required",
)  # fmt: skip
A06_HOLM = ("comparisons",)  # the only Holm family A-06 knows (D94)


def _user_kpi(key, where: str, bad: list[str]) -> tuple[str, str] | None:
    """A 'user/kpi' key whose base KPI is a known one (KPI_DEFINITIONS), else a problem."""
    try:
        user, kpi = split_key(key)
    except ConfigError as e:
        bad.append(f"{where}: {e}")
        return None
    if base_kpi(kpi) not in KPI_DEFINITIONS:
        bad.append(f"{where}: unknown KPI {kpi!r}")
    return user, kpi


def _candidate_controller(path) -> str | None:
    """The controller of a scenario config (None when it cannot be loaded)."""
    from traffic_twin.evaluation.final import repo_file

    try:
        return str(load_scenario(repo_file(path)).get("controller"))
    except (ConfigError, FileNotFoundError, OSError):
        return None


def part8_problems(ecfg: dict) -> list[str]:
    """The Part 8 blocks (D94, D98, D101) cross-checked against the families: `a06`
    (comparisons and the replication / equity references are labels of its family,
    primary in comparisons, the candidate an `rl` scenario, known alternative, Holm
    family and outcome words, the model pins), `final_table` (columns and ablations are
    labels, rows known KPIs, known extras) and `part8` (seed ranges, series). A block
    that is absent is not checked (a Part 4 config)."""
    bad: list[str] = []
    families = ecfg.get("families") or {}
    a06 = ecfg.get("a06")
    if a06 is not None:
        if not isinstance(a06, dict):
            return ["a06 must be a mapping"]
        missing = [k for k in A06_KEYS if a06.get(k) in (None, "", [])]
        if missing:
            bad.append(f"a06 needs {', '.join(missing)}")
        fam = families.get(a06.get("family"))
        scen = (fam or {}).get("scenarios") or {}
        if fam is None:
            bad.append(f"a06.family {a06.get('family')!r} is not a family")
        cand = a06.get("candidate")
        comps = list(a06.get("comparisons") or [])
        if scen and cand not in scen:
            bad.append(f"a06.candidate {cand!r} is not a scenario of family {a06.get('family')}")
        elif scen and _candidate_controller(scen[cand]) != "rl":
            bad.append(f"a06.candidate {cand} ({scen[cand]}) must be an rl scenario")
        for x in comps:
            if scen and x not in scen:
                bad.append(f"a06.comparisons: {x!r} is not a scenario of family {a06['family']}")
        if cand in comps or len(set(comps)) != len(comps):
            bad.append("a06.comparisons must be distinct references other than the candidate")
        if a06.get("primary") not in comps:
            bad.append(f"a06.primary {a06.get('primary')!r} must be one of a06.comparisons")
        if a06.get("alternative", "two-sided") not in ALTERNATIVES:
            bad.append(f"a06.alternative must be one of {ALTERNATIVES}")
        if a06.get("holm", "comparisons") not in A06_HOLM:
            bad.append(f"a06.holm must be one of {A06_HOLM}")
        if a06.get("status", "procedural") != "procedural":
            bad.append("a06.status must be procedural (D94: never FAIL on direction)")
        try:
            if int(a06.get("n_required", 0)) < 1:
                bad.append("a06.n_required must be >= 1")
        except (TypeError, ValueError):
            bad.append("a06.n_required must be an integer")
        words = a06.get("outcome") or {}
        if not isinstance(words, dict) or any(not words.get(k) for k in OUTCOME_KEYS):
            bad.append(f"a06.outcome needs {', '.join(OUTCOME_KEYS)}")
        if a06.get("user") and a06.get("kpi"):
            _user_kpi(f"{a06['user']}/{a06['kpi']}", "a06", bad)
        model = a06.get("model") or {}
        if not model.get("sha256_from") or not isinstance(model.get("inputs_text_sha8"), dict):
            bad.append("a06.model needs sha256_from and an inputs_text_sha8 mapping")
        rep = a06.get("replication_10_6")
        if rep is not None:
            refs = [("reference", rep.get("reference"))]
            refs.append(("c3.timeouts_le", (rep.get("c3") or {}).get("timeouts_le")))
            for where, x in refs:
                if x is not None and scen and x not in scen:
                    bad.append(f"a06.replication_10_6.{where} {x!r} is not a scenario")
            _user_kpi((rep.get("c1") or {}).get("kpi"), "a06.replication_10_6.c1", bad)
            for k in (rep.get("c2") or {}).get("kpis") or []:
                _user_kpi(k, "a06.replication_10_6.c2", bad)
        for x in a06.get("equity_vs") or []:
            if scen and x not in scen:
                bad.append(f"a06.equity_vs: {x!r} is not a scenario of family {a06['family']}")
    ft = ecfg.get("final_table")
    if ft is not None:
        fam = families.get((ft or {}).get("family"))
        scen = (fam or {}).get("scenarios") or {}
        if fam is None:
            bad.append(f"final_table.family {(ft or {}).get('family')!r} is not a family")
        for key in ("columns", "ablations"):
            for x in ft.get(key) or []:
                if scen and x not in scen:
                    bad.append(f"final_table.{key}: {x!r} is not a scenario of the family")
        if not ft.get("columns"):
            bad.append("final_table needs columns")
        for key in ("rows", "probe_rows"):
            for x in ft.get(key) or []:
                _user_kpi(x, f"final_table.{key}", bad)
        unknown = sorted(set(ft.get("extra") or []) - set(FINAL_EXTRAS))
        if unknown:
            bad.append(f"final_table.extra: unknown {unknown} (known: {FINAL_EXTRAS})")
    p8 = ecfg.get("part8")
    if p8 is not None:
        for name, rng in ((p8 or {}).get("seeds") or {}).items():
            ok = isinstance(rng, list) and len(rng) == 2 and all(isinstance(x, int) for x in rng)
            if not ok or rng[0] > rng[1]:
                bad.append(f"part8.seeds.{name} must be [first, last]")
        sc = (p8 or {}).get("series")
        if sc is not None:
            if not float(sc.get("step_s") or 0) > 0:
                bad.append("part8.series.step_s must be > 0")
            qs = sc.get("quantiles") or []
            if len(qs) != 2 or not all(0 < float(q) < 1 for q in qs) or qs[0] >= qs[1]:
                bad.append("part8.series.quantiles must be two values in (0, 1), low first")
            elif quantile_names(qs) != ["p10", "p90"]:
                bad.append("part8.series.quantiles must be [0.1, 0.9] (series.csv p10, p90)")
            if not sc.get("metric") or not sc.get("approaches"):
                bad.append("part8.series needs metric and approaches")
    return bad


def headline_settings(ecfg: dict) -> tuple[list[tuple[str, str]], float, float]:
    """(headline KPIs, alpha, equity_pct) of `headline_from` (configs/validation.yaml):
    the Part 3 Holm family, so the S0-S1 pair reproduces Part 3 exactly."""
    from traffic_twin.validation import load_validation

    vcfg = load_validation(ecfg["headline_from"])
    th = vcfg["thresholds"]
    headline = [tuple(h) for h in vcfg["compare"]["headline"]]
    return headline, float(th.get("alpha", 0.05)), float(th.get("equity_pct", 5.0))


def resolve_keys(spec) -> tuple[str, ...]:
    """A key list given in the config: a list, or the dotted name of a tuple inside
    `traffic_twin` (e.g. `sim.audit.DANGER_KEYS`)."""
    if isinstance(spec, list | tuple):
        return tuple(str(k) for k in spec)
    module, _, name = str(spec).rpartition(".")
    try:
        keys = getattr(importlib.import_module(f"traffic_twin.{module}"), name)
    except (ImportError, AttributeError, ValueError):
        raise ConfigError(f"cannot resolve the key list {spec!r} (traffic_twin.{spec})") from None
    return tuple(str(k) for k in keys)


def intervention_keys() -> tuple[str, ...]:
    from traffic_twin.control.safety import INTERVENTION_KEYS

    return tuple(INTERVENTION_KEYS)


def occupancy_sets(ecfg: dict) -> dict[str, dict[str, float]]:
    """KPI name -> occupancy per user type: the pre-registered set (`person_delay_mean`)
    and one set per sensitivity value (`person_delay_mean__occ_bus_20`, ...)."""
    base = {u: float(v) for u, v in ecfg["occupancy"].items()}
    out = {PERSON_DELAY: base}
    for user, values in (ecfg.get("occupancy_sensitivity") or {}).items():
        for v in values:
            name = f"{PERSON_DELAY}{QUALIFIER_SEP}{SENSITIVITY_PREFIX}{user}_{float(v):g}"
            out[name] = {**base, user: float(v)}
    return out


def file_sha256(path: Path) -> str:
    """sha256 of a text file with its line endings normalised (same on every checkout)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# -- families --------------------------------------------------------------------------
@dataclass
class Member:
    """One scenario of a family: its label (S0, S1, ...), config and runs."""

    label: str
    path: Path
    cfg: dict
    results: Path
    runs: Runs | None = None

    @property
    def scenario(self) -> str:
        return str(self.cfg["scenario"])

    @property
    def controller(self) -> str:
        return str(self.cfg["controller"])

    @property
    def expected(self) -> dict[str, str]:
        return expected_hashes(self.cfg)


@dataclass
class Family:
    name: str
    baseline: str
    pairs: list[dict]
    members: dict[str, Member]
    demand_hash: str | None = None
    kpis: dict[str, pd.DataFrame] = field(default_factory=dict)  # label -> seed, user, kpi, value
    seeds: list[int] = field(default_factory=list)  # the evaluation seeds every test needs

    @property
    def labels(self) -> list[str]:
        return list(self.members)

    def expected_seeds(self, *labels: str) -> list[int]:
        """The seeds a pair must cover: the evaluated seeds, else every seed any of
        `labels` (default all) reports."""
        if self.seeds:
            return sorted(int(s) for s in self.seeds)
        frames = [self.kpis[x] for x in (labels or self.labels) if x in self.kpis]
        return sorted({int(s) for k in frames for s in k["seed"]})


def load_family(ecfg: dict, name: str, results: Path | None = None) -> Family:
    """The configs of a family (`results`: where the runs are; default each config's
    outputs.dir)."""
    fam = ecfg["families"][name]
    members = {}
    for label, path in fam["scenarios"].items():
        path = Path(path)
        try:
            cfg = load_scenario(path)
        except (ConfigError, FileNotFoundError) as e:
            raise EvaluationError(f"family {name}, scenario {label}: {e}") from None
        where = Path(results) if results else Path(cfg["outputs"]["dir"])
        members[label] = Member(label, path, cfg, where)
    pairs = [dict(p) for p in fam.get("pairs") or []]
    return Family(name, str(fam["baseline"]), pairs, members)


def check_family(
    metas: dict[str, dict[int, dict]],
    seeds: list[int],
    expected: dict[str, dict[str, str]] | None = None,
    min_kpi_version: int | None = None,
) -> str:
    """N-scenario pairing check: the shared demand_hash. Raises EvaluationError when a
    scenario has no run for a seed, its runs differ in config_hash, a run is stale (its
    hashes are not `expected[label]` or its KPI version is below `min_kpi_version`), the
    seed sets differ, the configs or runs do not share one demand_hash, or two labels are
    one scenario."""
    problems = []
    for label, m in metas.items():
        missing = sorted(set(seeds) - set(m))
        if missing:
            problems.append(f"{label} has no run for seeds {missing}")
        configs = {x.get("config_hash") for x in m.values()}
        if len(configs) > 1:
            problems.append(f"{label} runs differ in config_hash ({len(configs)} values)")
        if expected:
            reasons = {
                s: stale_reason(x, expected[label], min_kpi_version) for s, x in sorted(m.items())
            }
            stale = "; ".join(f"seed {s}: {r}" for s, r in reasons.items() if r)
            if stale:
                problems.append(f"{label} runs are stale, not of the current config ({stale})")
        elif min_kpi_version is not None:
            old = sorted(s for s, x in m.items() if kpi_version(x) < min_kpi_version)
            if old:
                problems.append(f"{label} seeds {old} have kpi_version < {min_kpi_version}")
    seed_sets = {label: sorted(m) for label, m in metas.items()}
    if len({tuple(v) for v in seed_sets.values()}) > 1:
        listing = ", ".join(f"{k} {v}" for k, v in seed_sets.items())
        problems.append(f"seed sets differ: {listing}")
    if expected and len({e["demand_hash"] for e in expected.values()}) > 1:
        problems.append("the configs do not share one demand (demand_hash differs)")
    hashes = {x.get("demand_hash") for m in metas.values() for x in m.values()}
    if None in hashes:
        problems.append("a run.json has no demand_hash (rerun it with the current code)")
    elif len(hashes) > 1:
        problems.append(f"runs differ in demand_hash ({len(hashes)} values): not the same demand")
    scenarios = Counter(next(iter(m.values())).get("scenario") for m in metas.values() if m)
    dup = sorted(str(s) for s, n in scenarios.items() if n > 1 and s is not None)
    if dup:
        problems.append(f"scenarios {dup} appear under two labels")
    if problems:
        raise EvaluationError("cannot pair the runs: " + "; ".join(problems))
    return hashes.pop() if hashes else ""


def check_kpi_sets(kpis: dict[str, pd.DataFrame], seeds: list[int] | None = None) -> int:
    """A-03: every scenario reports the same (user, kpi) set, in EVERY seed (a run that
    lacks a KPI, e.g. an ambulance that never arrived and so has no travel time, must not
    silently leave the paired sample). A non-finite value counts as absent. `seeds`
    (default: every seed of any scenario) are the seeds each scenario must cover. The
    size of the set; EvaluationError naming the scenario, seeds and KPIs otherwise."""
    per_seed: dict[str, dict[int, set]] = {}
    for label, k in kpis.items():
        ok = k[np.isfinite(pd.to_numeric(k["value"], errors="coerce"))] if len(k) else k
        per_seed[label] = {
            int(s): set(zip(g["user_type"], g["kpi"], strict=True))
            for s, g in ok.groupby("seed", sort=True)
        }
    all_seeds = sorted(
        {int(s) for s in seeds} if seeds else {s for m in per_seed.values() for s in m}
    )
    union = set().union(*(x for m in per_seed.values() for x in m.values()))
    gaps = []
    for label, m in per_seed.items():
        lacking: dict[tuple[str, str], list[int]] = {}
        for s in all_seeds:
            for key in sorted(union - m.get(s, set())):
                lacking.setdefault(key, []).append(s)
        by_seeds: dict[tuple[int, ...], list[str]] = {}
        for (u, k), ss in lacking.items():
            by_seeds.setdefault(tuple(ss), []).append(f"{u}.{k}")
        for ss, names in by_seeds.items():
            where = label if list(ss) == all_seeds else f"{label} seeds {list(ss)}"
            gaps.append(f"{where} lacks {', '.join(names)}")
    if gaps:
        raise EvaluationError(
            "A-03: the scenarios do not report one KPI set in every seed: " + "; ".join(gaps)
        )
    return len(union)


# -- person-weighted delay -------------------------------------------------------------
def trip_delays(trips: pd.DataFrame) -> pd.DataFrame:
    """type, delay per trip: vehicles timeLoss + departDelay (missing departDelay = 0),
    persons the sum of their walks' timeLoss."""
    walk = trips["type"] == "pedestrian"
    veh = trips[~walk]
    dep = veh["depart_delay_s"] if "depart_delay_s" in veh else pd.Series(0.0, index=veh.index)
    v = pd.DataFrame({"type": veh["type"], "delay": veh["time_loss_s"] + dep.fillna(0.0)})
    w = trips[walk].groupby("id", sort=True)["time_loss_s"].sum()
    p = pd.DataFrame({"type": "pedestrian", "delay": w.to_numpy(float)})
    return pd.concat([v, p], ignore_index=True)


def person_delay_mean(trips: pd.DataFrame, occupancy: dict[str, float]) -> float:
    """Occupancy-weighted mean delay per person (D29): Σ occ·delay ÷ Σ occ over trips."""
    d = trip_delays(trips)
    if d.empty:
        return math.nan
    w = d["type"].map(occupancy)
    if w.isna().any():
        missing = sorted(set(d.loc[w.isna(), "type"]))
        raise EvaluationError(f"no occupancy for user types {missing} (evaluation.yaml)")
    total = float(w.sum())
    return float((w * d["delay"]).sum() / total) if total > 0 else math.nan


def person_delay_rows(runs: Runs, sets: dict[str, dict[str, float]]) -> pd.DataFrame:
    """seed, user_type `all`, kpi, value: one row per occupancy set and run."""
    rows = []
    trips = runs.trips
    for seed in runs.seeds:
        t = trips[trips["seed"] == seed] if len(trips) else trips
        if t.empty:
            raise EvaluationError(
                f"{runs.scenario} seed {seed} has no trip.parquet: the person-weighted delay "
                "needs it (rerun the run)"
            )
        for name, occ in sets.items():
            rows.append(
                {"seed": seed, "user_type": "all", "kpi": name, "value": person_delay_mean(t, occ)}
            )
    return pd.DataFrame(rows, columns=["seed", "user_type", "kpi", "value"])


def family_kpis(runs: Runs, sets: dict[str, dict[str, float]]) -> pd.DataFrame:
    """The run KPIs (seed, user_type, kpi, value) plus the person-weighted delays."""
    k = runs.kpis[["seed", "user_type", "kpi", "value"]] if len(runs.kpis) else runs.kpis
    return pd.concat([k, person_delay_rows(runs, sets)], ignore_index=True)


# -- pairs and equity ------------------------------------------------------------------
def pair_tables(
    family: Family, headline: list[tuple[str, str]], alpha: float, equity_pct: float
) -> dict[tuple[str, str], pd.DataFrame]:
    """(a, b) -> `compare(a, b)` with `family` and `role`: Holm within each pair (D28).
    EvaluationError when a KPI is paired on fewer than the family's seeds."""
    out = {}
    for p in family.pairs:
        a, b = p["a"], p["b"]
        t = compare(family.kpis[a], family.kpis[b], headline, equity_pct, alpha)
        require_all_seeds(t, family, a, b)
        out[(a, b)] = t.assign(family=family.name, role=p["role"]).reindex(columns=PAIR_COLUMNS)
    return out


def require_all_seeds(table: pd.DataFrame, family: Family, a: str, b: str) -> None:
    """Every row of `compare(a, b)` is paired on every seed of the family (`n_pairs`);
    EvaluationError naming the short KPIs otherwise (check_kpi_sets catches this
    earlier; this guards every caller of the pair and equity tables)."""
    n = len(family.expected_seeds(a, b))
    short = incomplete_rows(table, n)
    if len(short):
        listing = ", ".join(
            f"{r.user_type}.{r.kpi} ({r.n_pairs})" for r in short.itertuples(index=False)
        )
        where = f"family {family.name}, {b} vs {a}"
        raise EvaluationError(f"{where}: KPIs paired on fewer than the {n} seeds: {listing}")


def split_kpi(kpi: str) -> tuple[str, str]:
    base = base_kpi(kpi)
    return base, kpi[len(base) + len(QUALIFIER_SEP) :] if QUALIFIER_SEP in kpi else ""


def equity_table(family: Family, ecfg: dict, alpha: float) -> pd.DataFrame:
    """D29: every KPI of every scenario against the family baseline. Relative change in
    %; rates (`equity.rates_pp`) in percentage points. `worse_flag`: worse by more than
    `equity.pct` % (rates: by more than their pp limit) in the KPI's bad direction; a
    KPI whose baseline mean is 0 (no relative change) is flagged on any move in its bad
    direction (configs/evaluation.yaml `changes`, 2026-10-12)."""
    eq = ecfg["equity"]
    pct = float(eq["pct"])
    rates = {k: float(v) for k, v in (eq.get("rates_pp") or {}).items()}
    primary = dict(eq.get("primary") or {})
    subgroups = dict(eq.get("subgroups") or {})
    base = family.kpis[family.baseline]
    rows = []
    for label in family.labels:
        if label == family.baseline:
            continue
        t = compare(base, family.kpis[label], [], pct, alpha)
        require_all_seeds(t, family, family.baseline, label)
        for r in t.itertuples(index=False):
            b, qual = split_kpi(r.kpi)
            sense = _sense(r.kpi)
            m0, m1 = float(r.s0_mean), float(r.s1_mean)
            rel = float(r.rel_change_pct)
            pp = 100.0 * (m1 - m0) if b in rates else math.nan
            if not sense:
                worse = False
            elif b in rates:
                worse = bool(pp * sense < -rates[b])
            else:
                worse = worse_than(m0, m1, rel, sense, pct)
            prefix = subgroups.get(r.user_type)
            rows.append(
                {
                    "family": family.name,
                    "scenario": label,
                    "user_type": r.user_type,
                    "kpi": r.kpi,
                    "base_kpi": b,
                    "qualifier": qual,
                    "baseline_mean": m0,
                    "mean": m1,
                    "abs_change": m1 - m0,
                    "rel_change_pct": rel,
                    "pp_change": pp,
                    "p_value": float(r.p_value),
                    "worse_flag": worse,
                    "significant_flag": bool(r.p_value < alpha),
                    "primary": primary.get(r.user_type) == r.kpi,
                    "subgroup": bool(prefix) and str(r.kpi).startswith(str(prefix)),
                }
            )
    return pd.DataFrame(rows, columns=EQUITY_COLUMNS)


def hidden_degradation(equity: pd.DataFrame, ecfg: dict, alpha: float) -> list[dict]:
    """13.3, per scenario: the person-weighted global delay is significantly better
    (p < alpha, two-sided, better direction) while a primary or subgroup KPI is flagged."""
    g = ecfg["equity"]["global"]
    out = []
    for (fam, label), t in equity.groupby(["family", "scenario"], sort=False):
        glob = t[(t["user_type"] == g["user"]) & (t["kpi"] == g["kpi"])]
        better = False
        rel = p = math.nan
        if len(glob):
            r = glob.iloc[0]
            rel, p = float(r["rel_change_pct"]), float(r["p_value"])
            better = bool(_sense(g["kpi"]) * (r["mean"] - r["baseline_mean"]) > 0 and p < alpha)
        flagged = t[t["worse_flag"] & (t["primary"] | t["subgroup"])]
        out.append(
            {
                "family": fam,
                "scenario": label,
                "global_better": better,
                "global_rel_change_pct": rel,
                "global_p": p,
                "flagged": [f"{r.user_type}.{r.kpi}" for r in flagged.itertuples()],
                "hidden_degradation": bool(better and len(flagged)),
            }
        )
    return out


# -- A-04 ------------------------------------------------------------------------------
PED_A04 = "ped_curb_wait_max"  # judged by A-04 (changes 2026-10-12; D27 named the single wait)


def a04_table(
    family: Family, ecfg: dict, danger_keys: tuple[str, ...], inter_keys: tuple[str, ...]
) -> tuple[pd.DataFrame, list[str]]:
    """D27, one row per run: collisions, teleports, every danger audit count, the
    information key `out_of_order_green`, every guard intervention, the longest pedestrian
    curb wait (judged against `a04.ped_curb_wait_max_s`, see `ped_bound`) and the longest
    single wait (reported only); `pass` per run. Also the failure reasons. A count or a
    curb wait a run does not report is a failure (never read as 0)."""
    a04 = ecfg["a04"]
    coll_max = float(a04["collisions_max"])
    zero = set(a04.get("zero_interventions") or [])
    rows, failures = [], []
    for label, m in family.members.items():
        k = family.kpis[label]
        ped = k[k["user_type"] == "pedestrian"]
        single, curb = (
            dict(zip(g["seed"], g["value"], strict=True))
            for g in (ped[ped["kpi"] == name] for name in ("ped_wait_single_max", PED_A04))
        )
        try:
            ped_max = ped_bound(a04, m.cfg)
        except ConfigError as e:
            ped_max = None
            failures.append(f"{family.name} {label} ({m.scenario}): {e}")
        for seed, meta in sorted(m.runs.meta.items()):
            health = meta.get("health") or {}
            safety = meta.get("safety") or {}
            audit = safety.get("audit") or {}
            inter = safety.get("interventions") or {}
            why = []
            row = {
                "family": family.name,
                "scenario": m.scenario,
                "controller": m.controller,
                "seed": seed,
                "collisions": health.get("collisions"),
                "teleports": health.get("teleports"),
            }
            for key in danger_keys:
                row[key] = audit.get(key)
            known = [row[key] for key in danger_keys if row[key] is not None]
            row["danger_total"] = int(sum(known))
            row["out_of_order_green"] = audit.get("out_of_order_green")
            for key in inter_keys:
                row[key] = inter.get(key)
            row[PED_A04] = curb.get(seed)
            row["ped_max_wait_s"] = ped_max
            row["ped_wait_single_max"] = single.get(seed)
            if row["collisions"] is None:
                why.append("collisions not reported")
            elif row["collisions"] > coll_max:
                why.append(f"{row['collisions']} collision(s)")
            lacking = [key for key in danger_keys if row[key] is None]
            if lacking:
                why.append(f"audit lacks {', '.join(lacking)}")
            for key in danger_keys:
                if row[key]:
                    why.append(f"{key} {row[key]}")
            wait = row[PED_A04]
            if wait is None or not math.isfinite(wait):
                why.append(f"no {PED_A04}")
            elif ped_max is None:
                why.append("no pedestrian bound")
            elif wait > ped_max:
                why.append(f"pedestrian waited {wait:g} s at the curb > {ped_max:g} s")
            if m.controller in zero:
                lacking = [key for key in inter_keys if row[key] is None]
                if lacking:
                    why.append(f"interventions lack {', '.join(lacking)}")
                busy = {key: row[key] for key in inter_keys if row[key]}
                if busy:
                    listing = ", ".join(f"{a} {b}" for a, b in busy.items())
                    why.append(f"{m.controller} guard interventions ({listing}), D13")
            row["pass"] = not why
            rows.append(row)
            failures += [f"{family.name} {label} ({m.scenario}) seed {seed}: {w}" for w in why]
    columns = [
        "family", "scenario", "controller", "seed", "collisions", "teleports", *danger_keys,
        "danger_total", "out_of_order_green", *inter_keys, PED_A04, "ped_max_wait_s",
        "ped_wait_single_max", "pass",
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=columns), failures


def ped_bound(a04: dict, cfg: dict) -> float:
    """The A-04 pedestrian bound of a scenario: `a04.ped_curb_wait_max_s` is a number, or
    a dotted key of the scenario config (`safety.ped_max_wait_s`). ConfigError if the key
    is absent or not a positive number."""
    spec = a04["ped_curb_wait_max_s"]
    value = get(cfg, spec) if isinstance(spec, str) else spec
    try:
        bound = float(value)
    except (TypeError, ValueError):
        bound = math.nan
    if not bound > 0:
        raise ConfigError(f"a04.ped_curb_wait_max_s ({spec}) must be > 0, got {value!r}")
    return bound


# -- A-05 ------------------------------------------------------------------------------
def _f(x: float) -> float | None:
    return None if x is None or not math.isfinite(float(x)) else float(x)


def planned_trips(cfg: dict, user: str) -> int | None:
    """How many trips of `user` a run of `cfg` plans: the ambulances of its `emergency`
    block (sim.demand.emergency_trips); None for other users or without the block."""
    if user != "ambulance" or "emergency" not in cfg:
        return None
    from traffic_twin.sim.demand import emergency_trips

    try:
        return len(emergency_trips(cfg))
    except ConfigError:
        return None


def arrivals(kpis: pd.DataFrame, user: str) -> dict[int, float]:
    """seed -> finished trips of `user` (its `count` KPI; a seed without one finished
    none)."""
    c = kpis[(kpis["user_type"] == user) & (kpis["kpi"] == "count")]
    return dict(zip(c["seed"].astype(int), c["value"].astype(float), strict=True))


def a05_validity(family: Family, a: str, b: str, user: str, seeds_with: list[int]) -> list[str]:
    """Why a test of (user, kpi) is not valid: a seed of the family is missing from the
    pairs (`seeds_with` are the seeds both sides report), or a run finished fewer trips
    of `user` than its config plans (an ambulance that never arrived leaves the mean over
    the others, or no value at all). Empty when valid."""
    expected = family.expected_seeds(a, b)
    problems = []
    missing = sorted(set(expected) - set(seeds_with))
    if missing:
        problems.append(
            f"{len(seeds_with)} of {len(expected)} seeds paired, seeds {missing} lack it"
        )
    for label in (a, b):
        plan = planned_trips(family.members[label].cfg, user)
        if plan is None:
            continue
        got = arrivals(family.kpis[label], user)
        short = {s: got.get(s, 0.0) for s in expected if got.get(s, 0.0) != plan}
        if short:
            listing = ", ".join(f"seed {s}: {v:g}" for s, v in sorted(short.items()))
            problems.append(f"{label}: {user} trips finished != {plan} planned ({listing})")
    return problems


def a05_test(family: Family, spec: dict, alpha: float = 0.05) -> dict:
    """One pre-registered test: exact Wilcoxon on d = b - a of (user, kpi) per seed with
    the given alternative (`less`: b lower), Hodges-Lehmann estimate and 95 % CI. PASS
    iff the test is valid (every seed paired, every planned trip finished: D26 n = the
    evaluation seeds; `problems` otherwise) and p < alpha."""
    a, b = spec["a"], spec["b"]
    user, kpi = spec["user"], spec["kpi"]
    alt = spec.get("alternative", "two-sided")
    alpha = float(spec.get("alpha", alpha))
    pr = paired(family.kpis[a], family.kpis[b])
    g = pr[(pr["user_type"] == user) & (pr["kpi"] == kpi)].sort_values("seed")
    g = g[np.isfinite(g["diff"].to_numpy(float))]
    d = g["diff"].to_numpy(float)
    w = wilcoxon(d, alternative=alt) if len(d) else None
    hl, lo, hi = hodges_lehmann(d)
    p = w.pvalue if w else math.nan
    problems = a05_validity(family, a, b, user, [int(s) for s in g["seed"]])
    return {
        "family": family.name,
        "a": a,
        "b": b,
        "scenario_a": family.members[a].scenario,
        "scenario_b": family.members[b].scenario,
        "user": user,
        "kpi": kpi,
        "alternative": alt,
        "alpha": alpha,
        "n": int(len(d)),
        "seeds": [int(s) for s in g["seed"]],
        "a_values": [float(v) for v in g["s0"]],
        "b_values": [float(v) for v in g["s1"]],
        "diff": [float(v) for v in d],
        "a_mean": _f(g["s0"].mean()) if len(d) else None,
        "b_mean": _f(g["s1"].mean()) if len(d) else None,
        "diff_mean": _f(d.mean()) if len(d) else None,
        "hl": _f(hl),
        "hl_ci": [_f(lo), _f(hi)],
        "p_value": _f(p),
        "p_one_sided": _f(p) if alt != "two-sided" else None,
        "min_attainable_p": _f(w.min_attainable_p) if w else None,
        "expected_n": len(family.expected_seeds(a, b)),
        "valid": not problems,
        "problems": problems,
        "pass": bool(not problems and w is not None and p < alpha),
    }


def also_report(family: Family, a: str, b: str, user: str, kpis: list[str]) -> list[dict]:
    """The accompanying KPIs of A-05 (two-sided, descriptive)."""
    out = []
    for kpi in kpis:
        r = a05_test(family, {"a": a, "b": b, "user": user, "kpi": kpi})
        out.append(
            {k: r[k] for k in ("kpi", "n", "a_mean", "b_mean", "diff_mean", "hl", "p_value")}
        )
    return out


def a05_report(families: dict[str, Family], ecfg: dict) -> dict:
    """D26: the primary test (acceptance) and the secondary ones (reported); a test whose
    family was not evaluated is `not evaluated`, one whose sample is incomplete (a seed
    not paired, a planned ambulance that did not arrive) is `INVALID`, never PASS."""
    cfg = ecfg["a05"]
    primary_alpha = float(cfg["primary"].get("alpha", 0.05))

    def run(spec: dict) -> dict:
        fam = families.get(spec["family"])
        if fam is None:
            return {
                **{k: spec[k] for k in ("family", "a", "b", "user", "kpi")},
                "status": "not evaluated",
            }
        r = a05_test(fam, spec, primary_alpha)
        extra = list(cfg.get("also_report") or [])
        r["also_report"] = also_report(fam, spec["a"], spec["b"], spec["user"], extra)
        r["status"] = "INVALID" if not r["valid"] else "PASS" if r["pass"] else "FAIL"
        return r

    primary = run(cfg["primary"])
    return {
        "primary": primary,
        "secondary": [run(s) for s in cfg.get("secondary") or []],
        "verdict": primary["status"],
    }


# -- A-03 and activity -----------------------------------------------------------------
def a03_report(families: dict[str, Family], n_kpis: dict[str, int]) -> dict:
    ran = {m.controller for f in families.values() for m in f.members.values()}
    present = [c for c in SPEC_CONTROLLERS if c in ran]
    later = [LATER_PARTS.get(c, c) for c in SPEC_CONTROLLERS if c not in ran]
    text = f"{len(present)}/{len(SPEC_CONTROLLERS)}" + (f", {', '.join(later)}" if later else "")
    return {
        "controllers": present,
        "n": len(present),
        "of": len(SPEC_CONTROLLERS),
        "text": text,
        "same_kpi_set": dict.fromkeys(n_kpis, True),
        "n_kpis": n_kpis,
    }


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


RL_COUNTS = ("decisions", "advances")  # control/rl.py report()
STARVATION = "starvation"  # guard rule counted per run as well (D96: per controller)
RL_SPLIT = ("cut", "replaced", "cut_foreseeable")  # D77 iv_split of the RL report


def activity(family: Family, inter_keys: tuple[str, ...]) -> pd.DataFrame:
    """Per scenario, summed over its runs: guard interventions (and
    `guard_runs.starvation`, the runs in which the starvation rule fired), preemption
    counts, the rule controller's report (pedestrian protection, box gap-outs, TSP), the
    RL controller's report (`rl.decisions`, `rl.advances`, `rl.iv_split.*`, D77) and the
    events by source and rule."""
    rows = []
    for label, m in family.members.items():
        tot: Counter = Counter()
        for meta in m.runs.meta.values():
            inter = (meta.get("safety") or {}).get("interventions") or {}
            for k in inter_keys:
                tot[f"guard.{k}"] += _num(inter.get(k))
            if STARVATION in inter_keys:
                tot[f"guard_runs.{STARVATION}"] += int(_num(inter.get(STARVATION)) > 0)
            pre = meta.get("preemption") or {}
            for k in PREEMPTION_COUNTS:
                tot[f"preemption.{k}"] += _num(pre.get(k))
            ctl = meta.get("control") or {}
            for k in ("ped_protect", "box_gapouts"):
                tot[f"control.{k}"] += _num(ctl.get(k))
            tsp = ctl.get("tsp") or {}
            for k in (
                "calls",
                "extended_greens",
                "extension_s",
                "truncated_greens",
                "truncation_s",
            ):
                tot[f"tsp.{k}"] += _num(tsp.get(k))
            if any(k in ctl for k in RL_COUNTS):
                for k in RL_COUNTS:
                    tot[f"rl.{k}"] += _num(ctl.get(k))
                split = ctl.get("iv_split") or {}
                for k in RL_SPLIT:
                    tot[f"rl.iv_split.{k}"] += _num(split.get(k))
        ev = m.runs.events
        if ev is not None and len(ev):
            for (src, rule), n in ev.groupby(["source", "rule"]).size().items():
                tot[f"events.{src}.{rule}"] += int(n)
        rows.append(
            {"family": family.name, "scenario": label, "runs": len(m.runs.meta), **dict(tot)}
        )
    return pd.DataFrame(rows).fillna(0)


# -- provenance ------------------------------------------------------------------------
def run_date(meta: dict) -> str | None:
    """The local date (YYYY-MM-DD) a run started: run.json `started_at` is UTC, the
    pre-registration date is a local one."""
    started = meta.get("started_at")
    if not started:
        return None
    try:
        return datetime.fromisoformat(str(started)).astimezone().date().isoformat()
    except ValueError:
        return str(started)[:10]


def _dates(runs: Runs) -> tuple[str, str]:
    d = sorted(x for x in map(run_date, runs.meta.values()) if x)
    return (d[0], d[-1]) if d else ("n/a", "n/a")


def provenance(
    families: dict[str, Family], ecfg: dict, config_path: Path, seeds: list[int]
) -> dict:
    out = {
        "evaluation_config": Path(config_path).as_posix(),
        "evaluation_sha256": file_sha256(config_path),
        "preregistered": str(ecfg["preregistered"]),
        "kpi_version": int(ecfg["kpi_version"]),
        "headline_from": str(ecfg["headline_from"]),
        "seeds": [int(s) for s in seeds],
        "changes": [{k: str(v) for k, v in c.items()} for c in ecfg["changes"]],
        # D100: config_hash does not cover the text of rl.config; the pinned files' sha8
        "model_inputs": model_inputs(ecfg),
        "families": {},
    }
    for name, fam in families.items():
        members = {}
        for label, m in fam.members.items():
            first, last = _dates(m.runs)
            members[label] = {
                "scenario": m.scenario,
                "controller": m.controller,
                "config": m.path.as_posix(),
                **m.expected,
                "code_versions": sorted(
                    {str(x.get("code_version", "unknown")) for x in m.runs.meta.values()}
                ),
                "kpi_versions": sorted({kpi_version(x) for x in m.runs.meta.values()}),
                "first_run": first,
                "last_run": last,
            }
        out["families"][name] = {
            "baseline": fam.baseline,
            "demand_hash": fam.demand_hash,
            "pairs": fam.pairs,
            "scenarios": members,
        }
    return out


# -- markdown --------------------------------------------------------------------------
def _fmt(x, nd: int = 1) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool | np.bool_):
        return "yes" if x else "no"
    if isinstance(x, int | np.integer):
        return str(int(x))
    if isinstance(x, float | np.floating):
        return "–" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def _p(x) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:.4f}" if x >= 0.0001 else f"{x:.1e}"


def _nd(*values) -> int:
    finite = [abs(v) for v in values if v is not None and np.isfinite(v)]
    top = max(finite) if finite else 10.0
    return 3 if top < 1 else 2 if top < 10 else 1


def _table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(c if isinstance(c, str) else _fmt(c) for c in r) + " |" for r in rows]
    return out


def _rel(path: Path, docs: Path) -> str:
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(docs).resolve().parent)).as_posix()
    except ValueError:  # another drive on Windows
        return Path(path).resolve().as_posix()


def _seeds_text(seeds: list[int]) -> str:
    if not seeds:
        return "none"
    if seeds == list(range(seeds[0], seeds[-1] + 1)) and len(seeds) > 2:
        return f"{seeds[0]}-{seeds[-1]} ({len(seeds)} runs per scenario)"
    return f"{', '.join(map(str, seeds))} ({len(seeds)} runs per scenario)"


def pair_markdown(table: pd.DataFrame, family: Family, pair: dict, alpha: float, pct: float) -> str:
    """Headline table of one pair (b - a, Holm within the pair) and the other KPIs more
    than `pct` % worse."""
    a, b = pair["a"], pair["b"]
    sa, sb = family.members[a].scenario, family.members[b].scenario
    n = int(table["n_pairs"].max()) if len(table) else 0
    head = ["User", "KPI", f"{a} mean ± sd", f"{b} mean ± sd", f"{b} − {a} [95 % CI]",
            "Change", "p", "p (Holm)", "Effect", "Equity flag"]  # fmt: skip
    lines = [
        f"**{a}** (`{sa}`) vs **{b}** (`{sb}`), family `{family.name}`, role "
        f"*{pair['role']}*: {n} seeds, one demand. Exact two-sided Wilcoxon, Holm over the "
        f"{int(table['headline'].sum())} headline KPIs of this pair only; α = {alpha}.",
        "",
    ]
    rows = []
    for _, r in table[table["headline"]].iterrows():
        nd = _nd(r["s0_mean"], r["s1_mean"])
        rows.append(
            [
                r["user_type"],
                f"`{r['kpi']}`",
                f"{_fmt(r['s0_mean'], nd)} ± {_fmt(r['s0_std'], nd)}",
                f"{_fmt(r['s1_mean'], nd)} ± {_fmt(r['s1_std'], nd)}",
                diff_text(r),
                f"{_fmt(r['rel_change_pct'])} %",
                _p(r["p_value"]),
                _p(r["p_holm"]),
                r["effect"],
                "**yes**" if r["equity_flag"] else "no",
            ]
        )
    lines += _table(head, rows)
    others = table[~table["headline"] & table["equity_flag"]]
    if len(others):
        lines += ["", f"Other KPIs more than {pct:g} % worse (secondary, unadjusted p):", ""]
        for _, r in others.iterrows():
            nd = _nd(r["s0_mean"], r["s1_mean"])
            u = f" {unit(r['kpi'])}" if unit(r["kpi"]) else ""
            lines.append(
                f"- {r['user_type']} `{r['kpi']}`: {_fmt(r['s0_mean'], nd)} → "
                f"{_fmt(r['s1_mean'], nd)}{u} ({r['rel_change_pct']:+.1f} %, p {_p(r['p_value'])})"
            )
    return "\n".join(lines) + "\n"


@dataclass
class Evaluation:
    """Everything the page and the summary are made of."""

    ecfg: dict
    config_path: Path
    seeds: list[int]
    families: dict[str, Family]
    pairs: dict[str, dict[tuple[str, str], pd.DataFrame]]
    equity: pd.DataFrame
    hidden: list[dict]
    a04: pd.DataFrame
    a04_failures: list[str]
    a05: dict
    a03: dict
    activity: pd.DataFrame
    alpha: float
    equity_pct: float
    danger_keys: tuple[str, ...]
    inter_keys: tuple[str, ...]
    person_delay: dict = field(default_factory=dict)
    # Part 8 (evaluation.final): None / empty when the config has no a06 / final_table
    a06: dict | None = None
    final: pd.DataFrame = field(default_factory=pd.DataFrame)
    series: pd.DataFrame = field(default_factory=pd.DataFrame)
    digest: dict = field(default_factory=dict)
    model_problems: list[str] = field(default_factory=list)
    registration: dict = field(default_factory=dict)

    @property
    def a04_pass(self) -> bool:
        return not self.a04_failures and len(self.a04) > 0

    @property
    def replication(self) -> dict | None:
        return (self.a06 or {}).get("replication_10_6")

    @property
    def final_available(self) -> bool:
        """Every column and ablation label of `final_table` has rows."""
        ft = self.ecfg.get("final_table") or {}
        wanted = {*(ft.get("columns") or []), *(ft.get("ablations") or [])}
        if self.final.empty or not wanted:
            return False
        return wanted <= set(self.final.loc[self.final["section"] == "main", "label"])


def person_delay_summary(families: dict[str, Family], sets: dict) -> dict:
    """family -> label -> occupancy set -> mean over seeds."""
    out: dict = {}
    for name, fam in families.items():
        out[name] = {}
        for label, k in fam.kpis.items():
            v = k[k["user_type"] == "all"].groupby("kpi")["value"].mean()
            out[name][label] = {s: _f(v.get(s, math.nan)) for s in sets}
    return out


def prereg_order(prereg: str, first: str | None) -> str:
    """`before`, `same day`, `after` or `n/a`: the pre-registration date (a day) against
    the local date of the first run. A same-day pre-registration cannot be shown to come
    first at day resolution, so it is never reported as `before`."""
    if first is None:
        return "n/a"
    return "before" if prereg < first else "same day" if prereg == first else "after"


def first_started(ev: Evaluation) -> str | None:
    """The earliest run.json `started_at` (UTC, as written) of every evaluated run."""
    stamps = []
    for f in ev.families.values():
        for mem in f.members.values():
            for m in mem.runs.meta.values():
                try:
                    stamps.append(datetime.fromisoformat(str(m["started_at"])))
                except (KeyError, ValueError, TypeError):
                    continue
    return min(stamps).isoformat() if stamps else None


def summary(ev: Evaluation) -> dict:
    dates = [
        run_date(m)
        for f in ev.families.values()
        for mem in f.members.values()
        for m in mem.runs.meta.values()
    ]
    first = min((d for d in dates if d), default=None)
    prereg = str(ev.ecfg["preregistered"])
    order = prereg_order(prereg, first)
    return {
        "preregistered": prereg,
        "first_run_date": first,
        "first_run_started_at": first_started(ev),
        "preregistration_order": order,
        # True only when the pre-registration date is strictly earlier than the first run:
        # on the same day the order cannot be told at day resolution (None)
        "preregistered_before_runs": {"before": True, "after": False}.get(order),
        "seeds": ev.seeds,
        "families": {n: f.labels for n, f in ev.families.items()},
        "a03": ev.a03,
        "a04": {
            "verdict": "PASS" if ev.a04_pass else "FAIL",
            "pass": ev.a04_pass,
            "runs": int(len(ev.a04)),
            "failures": ev.a04_failures,
        },
        "a05": {
            "verdict": ev.a05["verdict"],
            "primary": {k: ev.a05["primary"].get(k) for k in SUMMARY_A05_KEYS},
        },
        "hidden_degradation": ev.hidden,
        "person_delay": ev.person_delay,
        "registration": ev.registration or None,
        "a06": a06_summary(ev.a06),
        "replication_10_6": replication_summary(ev.replication),
        "final_available": ev.final_available,
    }


A06_SUMMARY_KEYS = (
    "reference", "n", "mean_ref", "mean_s3", "hl", "ci_low", "ci_high", "p", "p_holm",
    "outcome", "sentence",
)  # fmt: skip


def a06_summary(a06: dict | None) -> dict | None:
    """summary.json['a06']: the status, its problems and the comparisons (a06.json has
    the full record)."""
    if not a06:
        return None
    return {
        "status": a06["status"],
        "problems": a06["problems"],
        "primary": a06["primary"],
        "primary_sentence": a06["primary_sentence"],
        "inputs_ok": a06["model"]["inputs_ok"],
        "comparisons": [{k: c.get(k) for k in A06_SUMMARY_KEYS} for c in a06["comparisons"]],
    }


def replication_summary(rep: dict | None) -> dict | None:
    """summary.json['replication_10_6']: pass per check, the citation, the sentence."""
    if not rep:
        return None
    return {
        "reference": rep["reference"],
        "c1": rep["c1"]["pass"],
        "c2": rep["c2"]["pass"],
        "c3": rep["c3"]["pass"],
        "c4": rep["c4"]["cited"],
        "confirms_fallback": rep["confirms_fallback"],
        "main_solution": rep["main_solution"],
        "text": rep["text"],
    }


def _labels_line(fam: Family) -> str:
    return ", ".join(
        f"{label} = `{m.scenario}` ({m.controller})" for label, m in fam.members.items()
    )


def _runs_section(ev: Evaluation) -> list[str]:
    lines = [
        f"Evaluation seeds {_seeds_text(ev.seeds)}; KPI version required "
        f"{int(ev.ecfg['kpi_version'])}. Every run was checked against its CURRENT config "
        "(`config_hash`, `demand_hash`) before use; a family shares one `demand_hash` "
        "(same demand, paired per seed).",
        "",
    ]
    rows = []
    for name, fam in ev.families.items():
        for label, m in fam.members.items():
            first, last = _dates(m.runs)
            versions = sorted({str(x.get("code_version", "unknown")) for x in m.runs.meta.values()})
            rows.append(
                [
                    name,
                    label,
                    f"`{m.scenario}`",
                    m.controller,
                    f"`{m.path.as_posix()}`",
                    f"`{short(m.expected['config_hash'])}`",
                    f"`{short(fam.demand_hash)}`",
                    ", ".join(versions),
                    first if first == last else f"{first} to {last}",
                    ", ".join(
                        str(v) for v in sorted({kpi_version(x) for x in m.runs.meta.values()})
                    ),
                ]
            )
    head = ["Family", "Label", "Scenario", "Controller", "Config", "config_hash",
            "demand_hash", "Code version", "Run dates", "kpi_version"]  # fmt: skip
    return lines + _table(head, rows)


ORDER_WORDS = {
    "before": "pre-registered before the runs",
    "same day": "same day as the pre-registration: at day resolution the order is not "
    "verifiable from these files",
    "after": "NOT before the runs",
    "n/a": "n/a",
}


def produced_base_kpis(ev: Evaluation) -> set[str]:
    """Base names of every KPI some evaluated run reports (a pre-registered rule on any
    other KPI cannot fire and is marked so on the page)."""
    return {
        base_kpi(str(k)) for f in ev.families.values() for t in f.kpis.values() for k in t["kpi"]
    }


def _utc(stamp: str) -> str:
    """'2026-10-12 00:49 UTC' of an ISO timestamp (as written when it has no zone)."""
    try:
        t = datetime.fromisoformat(stamp)
    except ValueError:
        return stamp
    if t.tzinfo is None:
        return t.strftime("%Y-%m-%d %H:%M")
    return t.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _prereg_section(ev: Evaluation, summ: dict) -> list[str]:
    e = ev.ecfg
    order = summ["preregistration_order"]
    started = summ.get("first_run_started_at")
    a05 = e["a05"]["primary"]
    a04 = e["a04"]
    eq = e["equity"]
    produced = produced_base_kpis(ev)
    lines = [
        f"`{ev.config_path.as_posix()}` (sha256 `{short(file_sha256(ev.config_path))}`), "
        f"pre-registered **{e['preregistered']}**; first run {summ['first_run_date'] or 'n/a'}"
        + (f", started {_utc(started)}" if started else "")
        + f" ({ORDER_WORDS[order]}). It changes only through dated entries in `changes`.",
        "",
        "Families and pairs (each pair is its own Holm family over the 7 Part 3 headline "
        f"KPIs of `{e['headline_from']}`; new KPIs are secondary, unadjusted):",
        "",
    ]
    for name, fam in ev.families.items():
        pairs = ", ".join(f"{p['a']} vs {p['b']} ({p['role']})" for p in fam.pairs)
        lines.append(f"- `{name}`: {_labels_line(fam)}; baseline {fam.baseline}; pairs {pairs}.")
    occ = ", ".join(f"{u} {v:g}" for u, v in e["occupancy"].items())
    sens = "; ".join(
        f"{u} {', '.join(f'{x:g}' for x in vs)}"
        for u, vs in (e.get("occupancy_sensitivity") or {}).items()
    )
    lines += [
        "",
        f"- **A-05** (D26): {a05['family']} family, {a05['b']} vs {a05['a']}, "
        f"{a05['user']} `{a05['kpi']}`, exact one-sided Wilcoxon (alternative "
        f"`{a05.get('alternative')}`: {a05['b']} lower), α {a05.get('alpha', 0.05)}; PASS iff "
        "p < α. The secondary tests (ambulance-probe family; S3 in both families) are "
        "reported, not judged.",
        f"- **A-04** (D27): every run of {', '.join(a04['families'])}: each danger audit "
        f"count 0, collisions ≤ {a04['collisions_max']}, `{PED_A04}` ≤ "
        f"`{a04['ped_curb_wait_max_s']}`{_bounds_text(ev)} (pre-registered: "
        f"`ped_wait_single_max` ≤ {a04.get('ped_wait_single_max_s', 'n/a')} s, now reported "
        "alongside; see the 2026-10-12 changes), 0 guard interventions for "
        f"{', '.join(a04.get('zero_interventions') or [])} (D13).",
        f"- **Equity** (D29): a KPI more than {eq['pct']:g} % worse than the baseline in its "
        "bad direction is flagged (a KPI whose baseline mean is 0: any move in its bad "
        "direction; rates: "
        + ", ".join(
            f"`{k}` by more than {v:g} pp"
            + ("" if k in produced else " (no run produces this KPI: not applied)")
            for k, v in (eq.get("rates_pp") or {}).items()
        )
        + "). Hidden degradation: the person-weighted delay "
        f"(`{eq['global']['user']}.{eq['global']['kpi']}`) significantly better while a "
        "primary ("
        + ", ".join(f"{u} `{k}`" for u, k in eq["primary"].items())
        + ") or subgroup KPI ("
        + ", ".join(f"{u} `{k}*`" for u, k in (eq.get("subgroups") or {}).items())
        + ") is flagged.",
        f"- Occupancies (ASSUMED, spec 10.3): {occ}; sensitivity: {sens or 'none'}.",
        "",
    ]
    if ev.ecfg["changes"]:
        lines += _table(
            ["Date", "Change", "Why"],
            [[str(c["date"]), str(c["what"]), str(c["why"])] for c in ev.ecfg["changes"]],
        )
    else:
        lines.append("Dated changes since pre-registration: none.")
    return lines


def _bounds_text(ev: Evaluation) -> str:
    """' (90 s)': the A-04 pedestrian bound(s) of the evaluated scenarios."""
    bounds = set()
    for fam in ev.families.values():
        for m in fam.members.values():
            try:
                bounds.add(ped_bound(ev.ecfg["a04"], m.cfg))
            except ConfigError:
                continue
    return f" ({', '.join(f'{b:g}' for b in sorted(bounds))} s)" if bounds else ""


def program_text(cfg: dict) -> str | None:
    """The fixed program of `cfg` in words, from config and net (D36: never hand-copied):
    the cycle (greens `fixed.green_s`, else the program's; clearances from the program of
    `network.net_file`) and the stage order of `network.stages`. None when unreadable."""
    from traffic_twin.sim.tls_check import load_program, scenario_stages

    try:
        stages = scenario_stages(cfg)
        prog = load_program(cfg["network"]["net_file"], cfg["network"]["tls_id"])
        fixed = (cfg.get("fixed") or {}).get("green_s") or {}
        green = sum(float(fixed.get(st.name, prog.durations[st.green_phase])) for st in stages)
        clear = sum(float(prog.durations[c]) for st in stages for c in st.clearance_phases)
    except (KeyError, IndexError, OSError, TypeError, ValueError, SyntaxError, ConfigError):
        return None  # SyntaxError: xml ParseError
    order = " → ".join(st.name for st in stages)
    return f"cycle {green + clear:g} s = {green:g} s green + {clear:g} s clearance, stages {order}"


def ambulance_plan(cfg: dict) -> tuple[int, str]:
    """(ambulances per run, 'HH:MM N → S, ...') of the `emergency` block of `cfg`: clock
    time of the planned departure and the arms of the first and last route edges."""
    from traffic_twin.config import approach_in_edges, approach_out_edges
    from traffic_twin.sim.demand import emergency_trips

    try:
        trips = emergency_trips(cfg)
    except ConfigError:
        return 0, ""
    try:
        arm_in = {e: a for a, es in approach_in_edges(cfg).items() for e in es}
        arm_out = {e: a for a, es in approach_out_edges(cfg).items() for e in es}
    except (ConfigError, KeyError, TypeError):
        arm_in, arm_out = {}, {}
    begin = float(cfg.get("begin_s") or 0.0)
    parts = []
    for depart, _vid, route in trips:
        hh, mm = divmod(int((begin + float(depart)) // 60), 60)
        frm, to = arm_in.get(route[0], route[0]), arm_out.get(route[-1], route[-1])
        parts.append(f"{hh % 24:02d}:{mm:02d} {frm} → {to}")
    return len(trips), ", ".join(parts)


def _ambulances(n: int) -> str:
    return f"{n} ambulance{'s' if n != 1 else ''}"


def ambulance_has_bluelight() -> bool:
    """Whether the ambulance vType the demand writes (sim.demand.VTYPES) has a bluelight
    device (SUMO's rescue lane and red-light passing)."""
    from traffic_twin.sim.demand import VTYPES

    return "bluelight" in VTYPES


def enabled_in(ev: Evaluation, key: str) -> list[str]:
    """Labels (`family/label`) whose config sets `<key>.enabled` true, e.g. `rules.tsp`.
    A `rules.*` feature counts only for the rule controller: the RL configs inherit the
    S2 fragment but leave its `rules:` block unused (configs/pitie_rl.yaml)."""
    out = []
    for name, fam in ev.families.items():
        for label, m in fam.members.items():
            if key.split(".")[0] == "rules" and m.controller != "rules":
                continue
            node: object = m.cfg
            for part in key.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            if isinstance(node, dict) and node.get("enabled"):
                out.append(f"{name}/{label}")
    return out


def _junction_default(name: str) -> float | None:
    """A default of the junction monitor the runs use (e.g. `exit_zone_m`)."""
    import inspect

    from traffic_twin.sim.junction import JunctionMonitor

    p = inspect.signature(JunctionMonitor.__init__).parameters.get(name)
    return None if p is None or p.default is inspect.Parameter.empty else float(p.default)


def _assumed_section(ev: Evaluation) -> list[str]:
    """Every ASSUMED input of Part 4, worded from the evaluated configs (D36): the item of
    a feature appears only when some evaluated config enables it."""
    main = next(iter(ev.families.values()))
    cfgs = [m.cfg for f in ev.families.values() for m in f.members.values()]
    forcing = sorted({str((c.get("safety") or {}).get("ped_forcing", "off")) for c in cfgs})
    pending = any(((c.get("preemption") or {}).get("detection") or {}).get("pending") for c in cfgs)
    occ = ", ".join(f"{u} {v:g}" for u, v in ev.ecfg["occupancy"].items())
    items = [
        f"pedestrian detection is ideal (camera / push button): the guard's pedestrian shield "
        f"reads the curb wait of every waiting pedestrian (`safety.ped_forcing`: "
        f"{', '.join(forcing)})",
    ]
    # the other ideal detectors, next to the pedestrian one
    tsp = enabled_in(ev, "rules.tsp")
    if tsp:
        lines = sorted(
            {
                str(x)
                for c in cfgs
                for x in (((c.get("rules") or {}).get("tsp") or {}).get("lines") or [])
            }
        )
        items.append(
            "bus detection for transit priority is ideal (TraCI positions of every bus on an "
            f"approach, standing for AVL or a check-in loop; lines {', '.join(lines)}; "
            f"`rules.tsp.enabled` in {', '.join(tsp)})"
        )
    box = enabled_in(ev, "rules.box")
    if box:
        zone, halt = _junction_default("exit_zone_m"), _junction_default("exit_halt_mps")
        where = ""
        if zone is not None and halt is not None:
            where = f"a vehicle below {halt:g} m/s within {zone:g} m of an exit lane's start, "
        items.append(
            f"exit blocking for box clearing is read ideally (TraCI: {where}standing for a "
            f"queue detector at each exit; `rules.box.enabled` in {', '.join(box)})"
        )
    plans = {name: ambulance_plan(f.members[f.baseline].cfg) for name, f in ev.families.items()}
    if any(n for n, _ in plans.values()):
        if ambulance_has_bluelight():
            items.append("the ambulance carries a bluelight device (`sim.demand` vType)")
        else:
            items.append(
                "the ambulance obeys the signals: its vType (`sim.demand`) has no bluelight "
                "device, so it never passes a red and no rescue lane forms"
            )
        items.append(
            "ambulance departures and routes (`emergency`): "
            + "; ".join(
                f"`{name}` {_ambulances(n)} per run ({text})" for name, (n, text) in plans.items()
            )
        )
    if enabled_in(ev, "preemption"):
        items.append(
            "the ambulance is detected before it enters the network (upstream / V2I "
            f"detection, `preemption.detection.pending`: {'true' if pending else 'false'})"
        )
    items.append(f"occupancies of the person-weighted delay: {occ}")
    prog = program_text(main.members[main.baseline].cfg)
    items.append(
        f"the signal program ({prog or 'stage order and clearances'}) and every Pitié number "
        "derived from it (maxima, hold caps, crossing reds) until the field visit"
    )
    lines = [f"- ASSUMED: {t}." for t in items]
    lines.append(
        f"- `{main.name}` demand frozen as in Part 3 ({_ambulances(plans[main.name][0])} per "
        f"run; demand_hash `{short(main.demand_hash)}`); bus headway updates and per-seed "
        "jitter are left for a later sensitivity family."
    )
    return lines


def _definitions_section(ev: Evaluation) -> list[str]:
    seen: dict[str, set[str]] = {}
    users: dict[str, set[str]] = {}
    for fam in ev.families.values():
        for k in fam.kpis.values():
            for user, kpi in zip(k["user_type"], k["kpi"], strict=True):
                b, q = split_kpi(kpi)
                seen.setdefault(b, set())
                users.setdefault(b, set()).add(user)
                if q:
                    seen[b].add(q.split("_")[0])
    rows = []
    for b in sorted(
        seen, key=lambda x: (list(KPI_DEFINITIONS).index(x) if x in KPI_DEFINITIONS else 99, x)
    ):
        src, formula = KPI_DEFINITIONS.get(b, ("run KPI", "see sim.kpi"))
        quals = ", ".join(QUALIFIER_WORDS.get(q, q) for q in sorted(seen[b]))
        u = unit(b) or "–"
        rows.append(
            [
                f"`{b}`",
                ", ".join(sorted(users[b])),
                src,
                formula + (f"; also {quals}" if quals else ""),
                u,
                better_if(b),
            ]
        )
    return [
        "Qualified KPIs are named `<base>__<qualifier>` (e.g. `time_loss_mean__arm_N`) and "
        "take the direction and unit of their base.",
        "",
        *_table(["KPI", "Users", "SUMO source", "Formula", "Unit", "Better if"], rows),
    ]


USER_ORDER = ("car", "bus", "pedestrian", "ambulance", "junction", "all")


def _per_user_section(ev: Evaluation) -> list[str]:
    rates = set((ev.ecfg["equity"].get("rates_pp") or {}).keys())
    lines = []
    for name, fam in ev.families.items():
        base = fam.kpis[fam.baseline]
        order = list(dict.fromkeys(zip(base["user_type"], base["kpi"], strict=True)))
        order = [(u, k) for u, k in order if QUALIFIER_SEP not in k]
        order.sort(key=lambda r: USER_ORDER.index(r[0]) if r[0] in USER_ORDER else 99)
        rows = []
        for u, k in order:
            cells = []
            for label in fam.labels:
                t = fam.kpis[label]
                v = t[(t["user_type"] == u) & (t["kpi"] == k)]["value"]
                nd = 4 if k in rates else _nd(v.mean()) if len(v) else 1
                cells.append(
                    f"{_fmt(v.mean(), nd)} ± {_fmt(v.std(ddof=1), nd)}"
                    if len(v) > 1
                    else _fmt(v.mean() if len(v) else None, nd)
                )
            rows.append([u, f"`{k}`", unit(k) or "–", *cells])
        lines += [f"Family `{name}`: mean ± sd over the seeds (qualified KPIs: `kpis.csv`).", ""]
        lines += _table(["User", "KPI", "Unit", *fam.labels], rows) + [""]
    return lines


def _pairs_section(ev: Evaluation, select, docs: Path, figure: str | None = None) -> list[str]:
    lines = []
    for name, fam in ev.families.items():
        for p in fam.pairs:
            if not select(fam, p):
                continue
            t = ev.pairs[name][(p["a"], p["b"])]
            lines += [pair_markdown(t, fam, p, ev.alpha, ev.equity_pct), ""]
    if figure:
        lines.append(f"![Paired differences vs the baseline]({_rel(figure, docs)})")
    return lines or ["No pair of this kind is pre-registered for the evaluated families."]


def _change_text(pp: float, rel: float) -> str:
    return f"{pp:+.2f} pp" if np.isfinite(pp) else f"{rel:+.1f} %"


def _equity_section(ev: Evaluation, docs: Path, figure: Path) -> list[str]:
    lines = []
    rows = [
        [
            h["family"],
            h["scenario"],
            f"{_fmt(h['global_rel_change_pct'])} % (p {_p(h['global_p'])})",
            "yes" if h["global_better"] else "no",
            ", ".join(f"`{x}`" for x in h["flagged"]) or "none",
            "**HIDDEN DEGRADATION**" if h["hidden_degradation"] else "no",
        ]
        for h in ev.hidden
    ]
    lines += _table(
        ["Family", "Scenario vs baseline", "Person delay change", "Significantly better",
         "Flagged primary / subgroup KPIs", "13.3 verdict"], rows,
    ) + [""]  # fmt: skip
    eq = ev.equity
    flagged = eq[eq["worse_flag"]]
    lines += [f"Every KPI flagged worse than the baseline ({len(flagged)}):", ""]
    if len(flagged):
        lines += _table(
            ["Family", "Scenario", "User", "KPI", "Baseline", "Scenario mean", "Change", "p"],
            [
                [
                    r.family, r.scenario, r.user_type, f"`{r.kpi}`",
                    _fmt(r.baseline_mean, _nd(r.baseline_mean)), _fmt(r.mean, _nd(r.mean)),
                    _change_text(r.pp_change, r.rel_change_pct),
                    _p(r.p_value),
                ]
                for r in flagged.itertuples()
            ],
        )  # fmt: skip
    else:
        lines.append("none")
    lines += [
        "",
        f"![Equity heatmap]({_rel(figure, docs)})",
        "",
        "Full table: `equity.csv` of each family (relative change, pp for rates, p, flags).",
    ]
    return lines


def _acceptance_section(ev: Evaluation, docs: Path, figures: Path) -> list[str]:
    a03 = ev.a03
    lines = [
        "### A-03",
        "",
        f"Controllers evaluated on one scenario and one KPI set: {a03['text']} "
        f"({', '.join(a03['controllers'])}); KPI set size per family: "
        + ", ".join(f"{k} {v}" for k, v in a03["n_kpis"].items())
        + ".",
        "",
        "### A-04",
        "",
        f"**A-04 {'PASS' if ev.a04_pass else 'FAIL'}** over {len(ev.a04)} runs.",
        "",
    ]
    rows = []
    for (fam, scen), g in ev.a04.groupby(["family", "scenario"], sort=False):
        inter = int(g[list(ev.inter_keys)].fillna(0).to_numpy().sum()) if ev.inter_keys else 0
        rows.append(
            [
                fam, f"`{scen}`", str(g["controller"].iloc[0]), len(g),
                int(g["collisions"].fillna(0).sum()), int(g["teleports"].fillna(0).sum()),
                int(g["danger_total"].sum()), int(g["out_of_order_green"].fillna(0).sum()), inter,
                _fmt(g[PED_A04].max()), _fmt(g["ped_max_wait_s"].max()),
                _fmt(g["ped_wait_single_max"].max()), "PASS" if g["pass"].all() else "FAIL",
            ]
        )  # fmt: skip
    lines += _table(
        ["Family", "Scenario", "Controller", "Runs", "Collisions", "Teleports", "Danger audit",
         "Out-of-order greens (info)", "Guard interventions", "Worst curb ped wait (s)",
         "Bound (s)", "Worst single ped wait (s, reported)", "A-04"], rows,
    )  # fmt: skip
    if ev.a04_failures:
        lines += ["", "Failures:", "", *[f"- {f}" for f in ev.a04_failures]]
    lines += [
        "",
        f"![Pedestrian waits per crossing]({_rel(figures / 'ped_wait_by_crossing.png', docs)})",
        "",
    ]
    lines += ["### A-05", ""]
    lines += _a05_lines(ev.a05["primary"], primary=True)
    for s in ev.a05["secondary"]:
        lines += _a05_lines(s, primary=False)
    lines += [f"![Ambulance travel time]({_rel(figures / 'a05_ambulance.png', docs)})"]
    return lines


SIDES = {"less": "one-sided", "greater": "one-sided", "two-sided": "two-sided"}


def _a05_lines(r: dict, primary: bool) -> list[str]:
    title = "Primary (acceptance)" if primary else "Secondary (reported)"
    head = f"{title}: family `{r['family']}`, {r['b']} vs {r['a']}, {r['user']} `{r['kpi']}`"
    if r.get("status") == "not evaluated":
        return [f"- {head}: not evaluated (family not run).", ""]
    ci = r["hl_ci"]
    out = [
        f"- {head}: **{r['status']}**. n = {r['n']}, mean {_fmt(r['a_mean'])} → "
        f"{_fmt(r['b_mean'])} s, HL {_fmt(r['hl'])} s [{_fmt(ci[0])}, {_fmt(ci[1])}], "
        f"exact {SIDES[r['alternative']]} p {_p(r['p_value'])} (smallest possible "
        f"{_p(r['min_attainable_p'])}), α {r['alpha']}.",
    ]
    if r.get("problems"):
        out[0] = out[0].replace(f"n = {r['n']}", f"n = {r['n']} of {r.get('expected_n')}")
        out += [f"  INVALID sample (never PASS): {'; '.join(r['problems'])}."]
    if primary:
        values = zip(r["seeds"], r["a_values"], r["b_values"], r["diff"], strict=True)
        rows = [[s, _fmt(x), _fmt(y), _fmt(d)] for s, x, y, d in values]
        out += ["", *_table(["Seed", r["a"], r["b"], f"{r['b']} − {r['a']}"], rows)]
    if r.get("also_report"):
        keys = ("a_mean", "b_mean", "diff_mean", "hl")
        rows = [
            [f"`{x['kpi']}`", *[_fmt(x[k], 2) for k in keys], _p(x["p_value"])]
            for x in r["also_report"]
        ]
        head = ["Also reported", r["a"], r["b"], "Mean diff", "HL", "p (two-sided)"]
        out += ["", *_table(head, rows)]
    return [*out, ""]


def _activity_section(ev: Evaluation) -> list[str]:
    """Per family: one row per counter (guard interventions always; preemption, the rule
    controller's report and the events when any scenario has one), one column per
    scenario, totals over its runs."""
    act = ev.activity
    if act.empty:
        return ["No activity recorded."]
    lines = [
        "Totals over the runs of each scenario: guard interventions (`guard.*`, D20), "
        "preemption (`preemption.*`), the rule controller's report (`control.*`, `tsp.*`) "
        "and the events of `events.parquet` (`events.<source>.<rule>`). Counters that are 0 "
        "for every scenario are left out, except the guard's.",
        "",
    ]
    for name, t in act.groupby("family", sort=False):
        t = t.set_index("scenario")
        keys = [c for c in t.columns if c not in ("family", "runs")]
        keys = [k for k in keys if k.startswith("guard.") or (t[k].fillna(0) != 0).any()]
        rows = [["runs", *[_fmt(int(t.loc[x, "runs"])) for x in t.index]]]
        rows += [[f"`{k}`", *[_fmt(float(t.loc[x, k]), 0) for x in t.index]] for k in keys]
        lines += [f"Family `{name}`:", "", *_table(["Counter", *t.index], rows), ""]
    return lines


def _limitations_section(ev: Evaluation) -> list[str]:
    a05 = ev.ecfg["a05"]
    lines = [
        "- Provisional: the signal program is ASSUMED until the field visit; every Pitié "
        "number (maxima, hold caps, crossing reds, cycle) is derived from it. After the "
        "visit, rerun `tasks.py net tls-check part3 part4`.",
        "- The 13.4 signal-timing and queue checks stay NOT_MEASURED until the field visit "
        "(docs/validation.md).",
    ]
    primary = ev.families.get(a05["primary"]["family"])
    if primary is not None:
        n, text = ambulance_plan(primary.members[primary.baseline].cfg)
        secondary = {s.get("family") for s in a05.get("secondary") or []}
        probes = [
            f"`{f.name}` family ({_ambulances(ambulance_plan(f.members[f.baseline].cfg)[0])} "
            "per run)"
            for f in ev.families.values()
            if f.name != primary.name and f.name in secondary
        ]
        count = "a single ambulance" if n == 1 else _ambulances(n)
        line = f"- The `{primary.name}` family has {count} per run" + (f" ({text})" if text else "")
        if probes:
            line += (
                "; the attribution of the ambulance gain to preemption rests on the "
                + ", ".join(probes)
            )
        lines.append(line + ".")
    ideal = ["pedestrian"]
    if enabled_in(ev, "rules.tsp"):
        ideal.append("bus (TSP)")
    if enabled_in(ev, "rules.box"):
        ideal.append("exit-blocking (box clearing)")
    lines += [
        f"- {', '.join(ideal)[:1].upper()}{', '.join(ideal)[1:]} detection is ideal "
        "(ASSUMED): without pedestrian "
        "detection the shield falls back to the static bound and S2's relaxed maxima would be "
        "cut back.",
        "- Bus demand is frozen (Part 3 headways, no per-seed jitter, no timetable): bus delay "
        "is SUMO `timeLoss`, not lateness.",
        "- Occupancies are ASSUMED; the person-weighted delay is also reported for the bus "
        "sensitivity values (`person_delay_mean__occ_*`).",
    ]
    return lines


def _vs_baseline(fam: Family, p: dict) -> bool:
    return p["a"] == fam.baseline


def _other_pair(fam: Family, p: dict) -> bool:
    """Section 7: the pre-registered confirmatory pairs not against the baseline."""
    return p["a"] != fam.baseline and p["role"] == "confirmatory"


def _exploratory(fam: Family, p: dict) -> bool:
    """Section 8: every other pair not against the baseline (secondary, exploratory)."""
    return p["a"] != fam.baseline and p["role"] != "confirmatory"


# -- Part 8 sections (D98, D99) --------------------------------------------------------
FINAL_SECTIONS = {
    "safety": "Safety and preemption (A-04 per scenario; preemption totals over the runs)",
    "activity": "Guard interventions per run",
    "equity": "Equity vs S0 (13.3): primary / subgroup KPIs flagged, every KPI flagged, "
    "hidden degradation (1 = yes)",
    "occupancy": "Person delay at the bus occupancy sensitivity values (ASSUMED occupancies)",
}


def _final_cell(r, base: str, rate: bool) -> str:
    """'mean ± sd (+x.x %)' of one final_table row (no change for the baseline)."""
    nd = 4 if rate else _nd(r["mean"])
    cell = f"{_fmt(r['mean'], nd)}"
    if r["n"] > 1 and r["sd"] is not None and np.isfinite(r["sd"]):
        cell += f" ± {_fmt(r['sd'], nd)}"
    if r["label"] != base and r["rel_vs_s0_pct"] is not None and np.isfinite(r["rel_vs_s0_pct"]):
        cell += f" ({r['rel_vs_s0_pct']:+.1f} %)"
    return cell


def _final_rows_table(t: pd.DataFrame, labels: list[str], base: str, abl: set, rates) -> list:
    head = ["User", "KPI", "Unit", "Better if",
            *[f"{x} (ablation)" if x in abl else x for x in labels]]  # fmt: skip
    rows = []
    for (user, kpi), g in t.groupby(["user", "kpi"], sort=False):
        g = g.set_index("label")
        cells = [
            _final_cell(g.loc[x].to_dict() | {"label": x}, base, base_kpi(kpi) in rates)
            if x in g.index
            else "not evaluated"
            for x in labels
        ]
        first = g.iloc[0]
        rows.append([user, f"`{kpi}`", first["unit"] or "–", first["better_if"], *cells])
    return _table(head, rows)


def _extra_table(t: pd.DataFrame, labels: list[str]) -> list:
    rows = []
    for (user, kpi), g in t.groupby(["user", "kpi"], sort=False):
        g = g.set_index("label")
        cells = []
        for x in labels:
            if x not in g.index:
                cells.append("–")
                continue
            v = g.loc[x, "mean"]
            nd = 0 if v is not None and np.isfinite(v) and float(v).is_integer() else 2
            cells.append(_fmt(v, nd) if v is not None else "–")
        rows.append([user, f"`{kpi}`", *cells])
    return _table(["User", "Counter", *labels], rows)


def _final_section(ev: Evaluation, docs: Path) -> list[str]:
    """Section 1: the pre-registered final comparison table (`final_table`, D98)."""
    ft = ev.ecfg.get("final_table")
    if not ft:
        return ["No final table is pre-registered in the evaluation config."]
    fam = ev.families.get(ft["family"])
    if fam is None or ev.final.empty:
        return [f"Family `{ft['family']}` was not evaluated: no final table."]
    abl = set(ft.get("ablations") or [])
    wanted = [*ft["columns"], *(ft.get("ablations") or [])]
    rates = set((ev.ecfg["equity"].get("rates_pp") or {}).keys())
    t = ev.final
    missing = [x for x in wanted if x not in set(t["label"])]
    lines = [
        f"Family `{fam.name}`, pre-registered layout (`final_table`, changes 2026-10-23): "
        f"mean ± sd over the {len(fam.seeds)} evaluation seeds, change vs "
        f"{fam.baseline} in brackets; columns {', '.join(ft['columns'])}, ablations "
        f"{', '.join(sorted(abl)) or 'none'} (greyed in the app). The person delay "
        "(`all/person_delay_mean`) is the A-06 KPI; the tests are in the next section and in "
        "the pair tables. Machine-readable: `final_table.csv` next to `summary.json` (the API, "
        "the app and the report read it).",
        "",
    ]
    if missing:
        lines += [f"Not evaluated: {', '.join(missing)}.", ""]
    main = t[t["section"] == "main"]
    lines += _final_rows_table(main, [x for x in wanted if x in set(main["label"])],
                               fam.baseline, abl, rates)  # fmt: skip
    probe = t[t["section"] == "probe"]
    if len(probe):
        others = [f for n, f in ev.families.items() if n != fam.name]
        pf = others[0] if others else fam
        labels = [x for x in wanted if x in set(probe["label"])]
        lines += [
            "",
            f"Ambulance-probe family `{pf.name}` "
            f"({_ambulances(ambulance_plan(pf.members[pf.baseline].cfg)[0])} per run, the "
            "robust ambulance evidence):",
            "",
            *_final_rows_table(probe, labels, pf.baseline, abl, rates),
        ]
    labels = [x for x in wanted if x in set(main["label"])]
    for section, title in FINAL_SECTIONS.items():
        g = t[t["section"] == section]
        if g.empty:
            continue
        if section == "occupancy":
            lines += ["", f"{title}:", "", *_final_rows_table(g, labels, fam.baseline, abl, rates)]
        else:
            lines += ["", f"{title}:", "", *_extra_table(g, labels)]
    return lines


def _a06_section(ev: Evaluation, docs: Path, figures: Path) -> list[str]:
    """Section 2: A-06 (D94), procedural status, the three comparisons, the model."""
    cfg = ev.ecfg.get("a06")
    if not cfg or ev.a06 is None:
        return ["A-06 is not pre-registered in the evaluation config."]
    a = ev.a06
    cand = a["candidate"]
    fam = ev.families.get(a["family"])
    m = fam.members.get(cand) if fam is not None else None
    words = cfg.get("outcome") or {}
    lines = [
        f"**A-06 {a['status']}** (procedural, D94: PASS when every comparison has its "
        f"{a['n_required']} paired seeds, the S3 runs are current and the frozen inputs match; "
        "INVALID otherwise; never FAIL because of the direction of a result).",
        "",
    ]
    if a["problems"]:
        lines += ["INVALID because:", "", *[f"- {p}" for p in a["problems"]], ""]
    lines += [
        f"Pre-registered (`a06`, changes 2026-10-23): family `{a['family']}`, candidate "
        f"{cand}" + (f" (`{m.scenario}`)" if m else "") + f", `{a['kpi']}`, d = {cand} − "
        f"reference per seed, exact {a['alternative']} paired Wilcoxon, Hodges-Lehmann "
        f"estimate and 95 % CI, Holm across the {len(a['comparisons'])} comparisons, "
        f'α {a["alpha"]:g}. Outcome: "{words.get("better")}" if Holm p < α and HL < 0, '
        f'"{words.get("worse")}" if Holm p < α and HL > 0, else "{words.get("none")}".',
        "",
    ]
    rows = []
    for c in a["comparisons"]:
        nd = _nd(c["mean_ref"], c["mean_s3"])
        role = "primary" if c["reference"] == a["primary"] else "Holm family"
        ctl = f" ({c['controller']})" if c.get("controller") else ""
        rows.append(
            [
                f"{cand} vs {c['reference']}{ctl}",
                role,
                c["n"],
                f"{_fmt(c['mean_ref'], nd)} ± {_fmt(c['sd_ref'], nd)}",
                f"{_fmt(c['mean_s3'], nd)} ± {_fmt(c['sd_s3'], nd)}",
                f"{_signed(c['hl'], nd)} [{_signed(c['ci_low'], nd)}, "
                f"{_signed(c['ci_high'], nd)}] s",
                _p(c["p"]),
                _p(c["p_holm"]),
                f"**{c['outcome_text']}**",
            ]
        )
    lines += _table(
        ["Comparison", "Role", "n", "Reference mean ± sd (s)", f"{cand} mean ± sd (s)",
         f"{cand} − reference: HL [95 % CI]", "p", "p (Holm)", "Outcome"], rows,
    )  # fmt: skip
    lines += ["", f"**Primary result:** {a['primary_sentence'] or 'not tested'}", ""]
    mod = a["model"]
    if mod["inputs_ok"]:
        inputs = f"all {len(mod['inputs'])} pinned rl.config files match (D100)"
    else:
        inputs = "**the frozen inputs do not match (D100, see the warnings)**"
    lines.append(
        f"- Model: `{mod.get('model_version') or 'n/a'}`, sha256 "
        f"`{short(mod.get('model_sha256'))}` (pinned by `{cfg['model']['sha256_from']}`: "
        f"`{short(mod.get('pinned_sha256'))}`), config_hash `{short(mod.get('config_hash'))}`; "
        f"{inputs}."
    )
    for prob in ev.model_problems:
        lines.append(f"- **Warning (D100):** {prob}.")
    lines.append(f"- Registration (D92): {_registration_text(a['registration'])}.")
    rep = (cfg.get("replication_10_6") or {}).get("main_solution")
    if rep:
        lines.append(
            f"- The main solution stays {rep} (Part 7 interim FALLBACK, spec 15 week-7 rule): "
            "A-06 is the confirmatory test of record for RL against the baselines, presented "
            "whatever its direction; it does not reopen that choice (changes 2026-10-23)."
        )
    lines += ["", f"![A-06: S3 against the baselines]({_rel(figures / 'a06_forest.png', docs)})"]
    return lines


REG_WORDS = {
    "before": "committed before the first S3 run (second resolution)",
    "after": "NOT before the first S3 run (second resolution)",
    "n/a": "order n/a",
}


def _registration_text(r: dict | None) -> str:
    r = r or {}
    if not r.get("commit"):
        commit = "no commit adds `a06:` to the evaluation config yet"
    else:
        commit = f"commit `{r['commit']}` at {_utc(str(r['committed_at']))}"
    first = r.get("first_s3_started_at")
    started = f"first S3 run started {_utc(str(first))}" if first else "no S3 run"
    return f"{commit}; {started}; {REG_WORDS.get(r.get('order'), r.get('order'))}"


def _yes(x) -> str:
    return "–" if x is None else "yes" if x else "**no**"


def _rep106_section(ev: Evaluation) -> list[str]:
    """Section 3: Part 7's D79 C1-C3 on the evaluation seeds, C4 cited (D95)."""
    rep = ev.replication
    cfg = (ev.ecfg.get("a06") or {}).get("replication_10_6")
    if not cfg or rep is None:
        return ["The spec 10.6 check is not pre-registered in the evaluation config."]
    if rep.get("status") == "not evaluated":
        return [f"**{rep['text']}**"]
    c1, c2, c3, c4 = rep["c1"], rep["c2"], rep["c3"], rep["c4"]
    ref = rep["reference"]
    nd = _nd(c1.get("mean_ref"), c1.get("mean_s3"))
    ci = c1.get("hl_ci") or [None, None]
    c2_text = "; ".join(
        f"`{r['kpi']}` {_signed(r['rel_pct'], 1)} %" + (" (worse)" if r["worse"] else "")
        for r in c2["rows"]
    )
    rows = [
        ["C1", f"`{c1['kpi']}`: HL < 0 and exact two-sided p < {c1.get('alpha', 0.05):g} vs {ref}",
         f"HL {_signed(c1.get('hl'), nd)} s [{_signed(ci[0], nd)}, {_signed(ci[1], nd)}], "
         f"p {_p(c1.get('p'))}, n {c1['n']}", _yes(c1["pass"])],
        ["C2", f"no listed KPI more than {c2['limit_pct']:g} % worse than {ref} (means)",
         c2_text or "–", _yes(c2["pass"])],
        ["C3", f"every S3 run: danger, collisions, teleports 0; curb wait ≤ "
         f"{c3['curb_max_s']:g} s; every preemption served or timed out; time-outs ≤ "
         f"{c3.get('timeouts_le') or 'n/a'}",
         f"{len(c3['violations'])} violation(s)", _yes(c3["pass"])],
        ["C4", "training property, cited from Part 7", f"`{c4['source']}`: {c4['cited']}",
         "cited"],
    ]  # fmt: skip
    lines = [
        "Reported, not gating (D95): Part 7's D79 checks C1-C3 repeated on the evaluation "
        f"seeds against {ref}; C4 is a training property and is cited from Part 7.",
        "",
        *_table(["Check", "Rule", "Result", "Pass"], rows),
        "",
        f"**{rep['text']}**",
    ]
    if c2["rows"]:
        lines += [
            "",
            *_table(
                ["C2 KPI", f"{ref} mean", "S3 mean", "Change", "Worse"],
                [[f"`{r['kpi']}`", _fmt(r["mean_s1"], _nd(r["mean_s1"])),
                  _fmt(r["mean_s3"], _nd(r["mean_s3"])), f"{_signed(r['rel_pct'], 1)} %",
                  _yes(r["worse"]) if r["worse"] else "no"] for r in c2["rows"]],
            ),
        ]  # fmt: skip
    if c3["violations"]:
        lines += ["", "C3 violations:", "", *[f"- {v}" for v in c3["violations"]]]
    return lines


def _signed(x, nd: int = 1) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def _equity_vs_lines(ev: Evaluation) -> list[str]:
    """S3 against S1 and S2 on the per-user primaries and the C2 KPIs (D96)."""
    tables = (ev.a06 or {}).get("equity_vs") or {}
    if not any(tables.values()):
        return []
    cand = (ev.a06 or {}).get("candidate", "S3")
    pct = ev.ecfg["equity"]["pct"]
    rows = []
    for ref, rs in tables.items():
        for r in rs:
            nd = _nd(r["mean_ref"], r["mean_s3"])
            change = (
                f"{r['pp']:+.2f} pp" if r.get("pp") is not None else f"{_signed(r['rel_pct'])} %"
            )
            rows.append(
                [f"{cand} vs {ref}", r["user"], f"`{r['kpi']}`", _fmt(r["mean_ref"], nd),
                 _fmt(r["mean_s3"], nd), change, _p(r["p"]),
                 "**yes**" if r["worse"] else "no"]
            )  # fmt: skip
    return [
        "",
        f"{cand} against the other controllers (D96): the per-user primary KPIs and the 10.6 C2 "
        f"KPIs, flagged when more than {pct:g} % worse (two-sided p, unadjusted):",
        "",
        *_table(["Pair", "User", "KPI", "Reference mean", f"{cand} mean", "Change", "p",
                 "Flag"], rows),
    ]  # fmt: skip


def _rl_members(ev: Evaluation) -> list[tuple[str, Member]]:
    return [
        (f"{n}/{lb}", m)
        for n, f in ev.families.items()
        for lb, m in f.members.items()
        if m.controller == "rl"
    ]


def _no_preemption(m: Member) -> bool:
    return not bool((m.cfg.get("preemption") or {}).get("enabled"))


def _part8_assumed(ev: Evaluation) -> list[str]:
    """The ASSUMED items Part 8 adds: the RL policy's training program and S4."""
    rl = _rl_members(ev)
    if not rl:
        return []
    items = []
    versions = sorted({str((m.cfg.get("rl") or {}).get("model", "")) for _, m in rl})
    items.append(
        "the RL policy (`" + "`, `".join(v for v in versions if v) + "`) was trained and "
        "selected in Part 7 on the ASSUMED signal program and demand; it is evaluated "
        "unchanged (frozen inputs, D100)"
    )
    off = [(where, m) for where, m in rl if _no_preemption(m)]
    if off:
        where = ", ".join(f"{w} (`{m.scenario}`)" for w, m in off)
        items.append(
            f"{where} run the S3 policy with emergency preemption off: OUT OF DISTRIBUTION "
            "(the policy never decided with an ambulance present without preemption), so "
            "their results describe how the agent degrades, not a design; exploratory, never "
            "part of A-06"
        )
    return [f"- ASSUMED: {t}." for t in items]


def _part8_prereg(ev: Evaluation) -> list[str]:
    """The Part 8 pre-registration bullets: the A-06 rule, the registration record (D92)
    and the D100 warnings."""
    a = ev.ecfg.get("a06")
    if not a:
        return []
    rep = a.get("replication_10_6") or {}
    out = [
        f"- **A-06** (D94, changes 2026-10-23): family `{a['family']}`, {a['candidate']} vs "
        f"{', '.join(a['comparisons'])} (primary {a['primary']}) on "
        f"`{a['user']}.{a['kpi']}`, exact {a.get('alternative', 'two-sided')} paired "
        f"Wilcoxon, Holm across the comparisons, n = {a['n_required']} required; procedural "
        "status (PASS / INVALID, never FAIL on direction)."
        + (
            f" The spec 10.6 check is repeated against {rep['reference']} (reported, not "
            f"gating); the main solution stays {rep.get('main_solution', 'S2')}."
            if rep
            else ""
        ),
        f"- **Registration** (D92): {_registration_text(ev.registration)}.",
    ]
    out += [f"- **Warning (D100):** {prob}." for prob in ev.model_problems]
    return out


def _a05_power_note(ev: Evaluation) -> list[str]:
    a05 = ev.ecfg["a05"]["primary"]
    fam = ev.families.get(a05["family"])
    if fam is None:
        return []
    n, _ = ambulance_plan(fam.members[fam.baseline].cfg)
    probes = [
        f"`{f.name}` ({_ambulances(ambulance_plan(f.members[f.baseline].cfg)[0])} per run)"
        for f in ev.families.values()
        if f.name != fam.name
    ]
    count = "one ambulance" if n == 1 else _ambulances(n)
    text = (
        f"- **Power note (D96):** the A-05 primary rests on {count} per run in `{fam.name}` "
        "(one travel time per seed, so one long red can decide a seed): its power is limited"
    )
    if probes:
        text += f"; the probe family {', '.join(probes)} is the robust evidence"
    return [text + ". Nothing is retuned on this account."]


def _a06_acceptance(ev: Evaluation) -> list[str]:
    if ev.a06 is None:
        return []
    a = ev.a06
    lines = [
        "",
        "### A-06",
        "",
        f"**A-06 {a['status']}** (procedural; section 2 has the tests, the model identity "
        "and the registration record). Primary: " + (a["primary_sentence"] or "not tested"),
    ]
    if a["problems"]:
        lines += ["", *[f"- INVALID: {p}" for p in a["problems"]]]
    return lines


def _part8_limitations(ev: Evaluation) -> list[str]:
    out = []
    rl = _rl_members(ev)
    if rl:
        out.append(
            "- The RL policy was trained on the ASSUMED signal program: after the field visit "
            'its results are rerun with the unchanged model and labelled "trained on the '
            'ASSUMED program" (D119); retraining is optional.'
        )
    if any(_no_preemption(m) for _, m in rl):
        out.append(
            "- S4 (the S3 policy without preemption) is out of distribution: it shows what "
            "preemption adds under RL, not a design to deploy."
        )
    if ev.a06 is not None:
        main_solution = ev.replication["main_solution"] if ev.replication else "S2"
        out.append(
            f"- A-06 and the 10.6 check are presented whatever their direction; the main "
            f"solution stays {main_solution} (decided at week 7)."
        )
    return out


def render(ev: Evaluation, summ: dict, docs: Path, figures: Path) -> str:
    """docs/evaluation.md: the title, the verdicts and one section per entry of SECTIONS
    (D99; the Part 4 bodies keep their tables)."""
    figures = Path(figures)
    equity = _equity_section(ev, docs, figures / "equity_heatmap.png")
    equity[-4:-4] = _equity_vs_lines(ev)  # before the heatmap and the full-table note
    assumed = _assumed_section(ev)  # its last line is the frozen-demand note
    prereg = _prereg_section(ev, summ)
    occ_at = next((i for i, x in enumerate(prereg) if x.startswith("- Occupancies")), None)
    at = len(prereg) if occ_at is None else occ_at + 1
    prereg[at:at] = _part8_prereg(ev)
    acceptance = _acceptance_section(ev, docs, figures)
    limits = _limitations_section(ev)
    a05_at = next((i for i, x in enumerate(limits) if x.startswith("- The `")), len(limits) - 1)
    limits[a05_at + 1 : a05_at + 1] = _a05_power_note(ev)
    bodies = [
        _final_section(ev, docs),
        _a06_section(ev, docs, figures),
        _rep106_section(ev),
        _runs_section(ev),
        prereg,
        assumed[:-1] + _part8_assumed(ev) + assumed[-1:],
        _definitions_section(ev),
        _per_user_section(ev),
        _pairs_section(ev, _vs_baseline, docs, figures / "paired_vs_s0.png"),
        _pairs_section(ev, _other_pair, docs),
        _pairs_section(ev, _exploratory, docs),
        equity,
        acceptance + _a06_acceptance(ev),
        _activity_section(ev),
        limits + _part8_limitations(ev),
    ]
    a05 = ev.a05["verdict"]
    hidden = [f"{h['family']} {h['scenario']}" for h in ev.hidden if h["hidden_degradation"]]
    verdicts = (
        f"**Verdicts:** A-03 {ev.a03['text']}; A-04 {'PASS' if ev.a04_pass else 'FAIL'}; "
        f"A-05 {a05}; "
        + (f"A-06 {ev.a06['status']}; " if ev.a06 is not None else "")
        + f"hidden degradation: {', '.join(hidden) or 'none'}."
    )
    lines = [
        PART8_TITLE,
        "",
        f"Generated by `python -m traffic_twin.evaluation.evaluate --config "
        f"{ev.config_path.as_posix()}` (spec 13.2, 13.3, A-03 to A-06). Do not edit by "
        "hand: rerun `tasks.py evaluate` (figures: `tasks.py plots-eval`). Results are "
        "provisional until the field visit (ASSUMED signal program).",
        "",
        verdicts,
    ]
    if ev.a06 is not None and ev.a06["primary_sentence"]:
        lines += ["", f"**A-06 primary:** {ev.a06['primary_sentence']}"]
    if ev.replication:
        lines += ["", f"**Spec 10.6 check (reported, not gating):** {ev.replication['text']}"]
    lines += ["", f"![KPIs per user]({_rel(figures / 'kpi_by_user.png', docs)})"]
    for title, body in zip(SECTIONS, bodies, strict=True):
        lines += ["", f"## {title}", "", *body]
    return "\n".join(lines).rstrip() + "\n"


# -- orchestration ---------------------------------------------------------------------
def evaluate(
    ecfg: dict,
    config_path: Path,
    names: list[str],
    seeds: list[int],
    results: Path | None = None,
    run_absent: bool = False,
) -> Evaluation:
    """Load (and with `run_absent` first run) every family, check the pairing, and
    compute the pairs, equity, A-03, A-04 and A-05. Raises EvaluationError."""
    headline, alpha, equity_pct = headline_settings(ecfg)
    danger_keys = resolve_keys(ecfg["a04"]["danger_keys"])
    inter_keys = intervention_keys()
    min_k = int(ecfg["kpi_version"])
    sets = occupancy_sets(ecfg)
    families: dict[str, Family] = {}
    problems = []
    loaded = {name: load_family(ecfg, name, results) for name in names}
    if run_absent:  # D92: refuse before the first missing run is launched
        planned: dict[Path, list[tuple[str, int]]] = {}
        for fam in loaded.values():
            for where, scenario, seed in planned_runs(fam, seeds, min_k):
                planned.setdefault(where, []).append((scenario, seed))
        for where, runs in planned.items():
            run_lock(ecfg, runs, where)
    for name in names:
        fam = loaded[name]
        if run_absent:
            for m in fam.members.values():
                run_missing(m.cfg, seeds, m.results, m.path, min_kpi_version=min_k)
        for m in fam.members.values():
            m.runs = load_runs(m.results, m.scenario, seeds, tables=RUN_TABLES)
        try:
            fam.demand_hash = check_family(
                {lb: m.runs.meta for lb, m in fam.members.items()},
                seeds,
                {lb: m.expected for lb, m in fam.members.items()},
                min_k,
            )
        except EvaluationError as e:
            problems.append(f"family {name}: {e}")
            continue
        families[name] = fam
    if problems:
        raise EvaluationError(
            "; ".join(problems) + " (--run-missing runs absent seeds and reruns stale ones)"
        )
    n_kpis = {}
    for name, fam in families.items():
        fam.seeds = [int(s) for s in seeds]
        fam.kpis = {lb: family_kpis(m.runs, sets) for lb, m in fam.members.items()}
        try:
            n_kpis[name] = check_kpi_sets(fam.kpis, fam.seeds)
        except EvaluationError as e:
            raise EvaluationError(f"family {name}: {e}") from None
    pairs = {n: pair_tables(f, headline, alpha, equity_pct) for n, f in families.items()}
    frames = [equity_table(f, ecfg, alpha) for f in families.values()]
    equity = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EQUITY_COLUMNS)
    )
    a04_rows, failures = [], []
    for name in ecfg["a04"]["families"]:
        if name in families:
            t, f = a04_table(families[name], ecfg, danger_keys, inter_keys)
            a04_rows.append(t)
            failures += f
    a04 = pd.concat(a04_rows, ignore_index=True) if a04_rows else pd.DataFrame()
    act = [activity(f, inter_keys) for f in families.values()]
    ev = Evaluation(
        ecfg=ecfg,
        config_path=Path(config_path),
        seeds=[int(s) for s in seeds],
        families=families,
        pairs=pairs,
        equity=equity,
        hidden=hidden_degradation(equity, ecfg, alpha),
        a04=a04,
        a04_failures=failures,
        a05=a05_report(families, ecfg),
        a03=a03_report(families, n_kpis),
        activity=pd.concat(act, ignore_index=True).fillna(0) if act else pd.DataFrame(),
        alpha=alpha,
        equity_pct=equity_pct,
        danger_keys=danger_keys,
        inter_keys=inter_keys,
        person_delay=person_delay_summary(families, sets),
    )
    part8_outputs(ev)
    return ev


def part8_outputs(ev: Evaluation) -> None:
    """The Part 8 results of `ev` (evaluation.final): the registration record (D92), the
    D100 input check, A-06 with the 10.6 replication, the final table, the series and the
    digest. Blocks absent from the config are skipped (a Part 4 config)."""
    ecfg, families = ev.ecfg, ev.families
    ev.registration = registration_record(families, registration_commit(ev.config_path))
    ev.model_problems = check_model_inputs(ecfg) + rl_scenario_problems(families, ecfg)
    if "a06" in ecfg:
        ev.a06 = a06_report(
            families, ecfg, ev.a03, registration=ev.registration, model_problems=ev.model_problems
        )
    if "final_table" in ecfg:
        ev.final = final_table(
            families, ecfg, ev.a04, ev.activity, equity=ev.equity, hidden=ev.hidden
        )
    if "part8" in ecfg:
        frames = [f for f in (series(fam, ecfg) for fam in families.values()) if len(f)]
        cols = series(Family("", "", [], {}), ecfg).columns
        ev.series = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    ev.digest = digest(families)


def write_outputs(ev: Evaluation, out: Path) -> dict:
    """Every output file under `out`; returns the summary."""
    out = Path(out)
    for name, fam in ev.families.items():
        d = out / name
        (d / "pairs").mkdir(parents=True, exist_ok=True)
        for p in fam.pairs:
            t = ev.pairs[name][(p["a"], p["b"])]
            stem = d / "pairs" / f"{p['a']}__vs__{p['b']}"
            t.to_csv(stem.with_suffix(".csv"), index=False)
            md = pair_markdown(t, fam, p, ev.alpha, ev.equity_pct)
            stem.with_suffix(".md").write_text(md, encoding="utf-8")
        ev.equity[ev.equity["family"] == name].to_csv(d / "equity.csv", index=False)
        if len(ev.a04):
            ev.a04[ev.a04["family"] == name].to_csv(d / "a04.csv", index=False)
        kp = pd.concat(
            [
                k.assign(family=name, label=lb, scenario=fam.members[lb].scenario,
                         controller=fam.members[lb].controller)
                for lb, k in fam.kpis.items()
            ],
            ignore_index=True,
        )  # fmt: skip
        cols = ["family", "label", "scenario", "controller", *KPI_KEYS, "value"]
        kp[cols].to_csv(d / "kpis.csv", index=False)
    summ = summary(ev)
    (out / "a05.json").write_text(
        json.dumps(ev.a05, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(summ, indent=2, default=str) + "\n", encoding="utf-8"
    )
    prov = provenance(ev.families, ev.ecfg, ev.config_path, ev.seeds)
    prov["registration"] = ev.registration or None
    (out / "provenance.json").write_text(
        json.dumps(prov, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_part8_outputs(ev, out)
    return summ


def part8_skipped(ev: Evaluation) -> dict[str, str]:
    """D98 file name -> why it is not written: a06.json needs the a06 family evaluated;
    final_table.csv, series.csv and digest.json cover every family of the config. A
    partial evaluation (`--family ambulance_probe`) therefore leaves the files of the
    full evaluation, which the API, app and report read, untouched."""
    every = list(ev.ecfg.get("families") or [])
    missing = [n for n in every if n not in ev.families]
    partial = f"families {', '.join(missing)} were not evaluated" if missing else None
    a06_fam = (ev.ecfg.get("a06") or {}).get("family")
    skipped = {}
    if ev.a06 is not None and a06_fam not in ev.families:
        skipped[EVAL_A06.name] = f"family {a06_fam} was not evaluated"
    blocks = {
        EVAL_FINAL_TABLE.name: "final_table" in ev.ecfg,
        EVAL_SERIES.name: "part8" in ev.ecfg,
        EVAL_DIGEST.name: True,
    }
    if partial:
        skipped.update((name, partial) for name, on in blocks.items() if on)
    return skipped


def write_part8_outputs(ev: Evaluation, out: Path) -> list[Path]:
    """D98: final_table.csv, a06.json, series.csv and digest.json under `out` (names of
    paths.EVAL_*), each when its block is registered (the digest always) and the families
    it reports on were evaluated (`part8_skipped`: a skipped file is left as it is)."""
    skipped = part8_skipped(ev)
    for name, why in skipped.items():
        log.warning("%s not written (%s): the existing file is kept", name, why)
    written = []
    if "final_table" in ev.ecfg and EVAL_FINAL_TABLE.name not in skipped:
        written.append(out / EVAL_FINAL_TABLE.name)
        ev.final.to_csv(written[-1], index=False)
    if ev.a06 is not None and EVAL_A06.name not in skipped:
        written.append(out / EVAL_A06.name)
        written[-1].write_text(json.dumps(ev.a06, indent=2, default=str) + "\n", "utf-8")
    if "part8" in ev.ecfg and EVAL_SERIES.name not in skipped:
        written.append(out / EVAL_SERIES.name)
        ev.series.to_csv(written[-1], index=False)
    if EVAL_DIGEST.name not in skipped:
        written.append(out / EVAL_DIGEST.name)
        written[-1].write_text(json.dumps(ev.digest, indent=2) + "\n", encoding="utf-8")
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=EVALUATION_CONFIG)
    parser.add_argument("--family", choices=FAMILIES, default="all")
    parser.add_argument("--run-missing", action="store_true", help="run absent or stale runs")
    parser.add_argument("--results", type=Path, help="default: the scenario configs' outputs.dir")
    parser.add_argument("--docs", type=Path, help="write the evaluation page (docs/evaluation.md)")
    parser.add_argument(
        "--out", type=Path, help="default: outputs.dir, or <results>/evaluation with --results"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="default: seeds.evaluation (calibration checks only)"
    )
    return parser


CONSOLE_ASCII = {"→": "->", "−": "-", "α": "alpha", "≤": "<=", "±": "+/-"}


def _echo(text: str) -> None:
    """print() that never fails on a narrow console encoding (cp1252 on Windows): the
    generated sentences carry → and α."""
    for a, b in CONSOLE_ASCII.items():
        text = text.replace(a, b)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(enc, "replace").decode(enc, "replace"))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        ecfg = load_evaluation(args.config)
    except (ConfigError, FileNotFoundError) as e:
        raise SystemExit(f"evaluation config error: {e}") from None
    names = list(ecfg["families"]) if args.family == "all" else [args.family]
    seeds = args.seeds or [int(s) for s in ecfg["seeds"]["evaluation"]]
    if args.seeds and sorted(args.seeds) != sorted(int(s) for s in ecfg["seeds"]["evaluation"]):
        log.warning("seeds %s are not the pre-registered evaluation seeds", args.seeds)
    try:
        ev = evaluate(ecfg, args.config, names, seeds, args.results, args.run_missing)
    except RunLockError as e:
        print(f"evaluation error: {e}", file=sys.stderr)
        raise SystemExit(RUN_LOCK_EXIT) from None
    except (EvaluationError, ConfigError) as e:
        raise SystemExit(f"evaluation error: {e}") from None
    out = args.out or (
        Path(args.results) / "evaluation" if args.results else Path(ecfg["outputs"]["dir"])
    )
    summ = write_outputs(ev, out)
    if summ["preregistered_before_runs"] is False:
        log.warning(
            "pre-registration %s is later than the first run (%s)",
            summ["preregistered"],
            summ["first_run_date"],
        )
    if args.docs:
        figures = Path(ecfg["outputs"]["figures"])
        args.docs.parent.mkdir(parents=True, exist_ok=True)
        args.docs.write_text(render(ev, summ, args.docs, figures), encoding="utf-8")
        log.info("evaluation page: %s", args.docs.as_posix())
    p = ev.a05["primary"]
    print(f"A-03 {ev.a03['text']}")
    print(f"A-04 {summ['a04']['verdict']}: {len(ev.a04)} runs, {len(ev.a04_failures)} failure(s)")
    for f in ev.a04_failures[:20]:
        print(f"  {f}")
    if p.get("status") == "not evaluated":
        print("A-05 not evaluated (family not run)")
    else:
        print(
            f"A-05 {p['status']}: {p['b']} vs {p['a']} {p['user']} {p['kpi']}, n {p['n']}, "
            f"HL {_fmt(p['hl'])} s, one-sided p {_p(p['p_value'])}"
        )
        for problem in p.get("problems") or []:
            print(f"  invalid sample: {problem}")
    hidden = [f"{h['family']} {h['scenario']}" for h in ev.hidden if h["hidden_degradation"]]
    print(f"hidden degradation: {', '.join(hidden) or 'none'}")
    if ev.a06 is not None:
        _echo(f"A-06 {ev.a06['status']} (procedural): {ev.a06['primary_sentence'] or 'no test'}")
        for problem in ev.a06["problems"]:
            _echo(f"  invalid: {problem}")
        if ev.replication:
            _echo(f"10.6 check (reported, not gating): {ev.replication['text']}")
    for problem in ev.model_problems:
        log.warning("D100 frozen S3 input: %s", problem)
    log.info("evaluation outputs: %s", Path(out).as_posix())
    if not ev.a04_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

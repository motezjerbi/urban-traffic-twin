"""Part 7 page `docs/rl_training.md` (spec S7, 10.3-10.6; D83; contract 4.2).

    python -m traffic_twin.rl.report7 --docs docs/rl_training.md --figures docs/figures/rl7

Builds the page and its figures (`viz.rl7`) from the Part 7 artifacts under rl7.yaml
`outputs.study` (data/rl7/): campaign, decision_T.json, run folders, holdout, select,
stress, sens and devtest. The interim 10.6 text is computed from `devtest/verdict.json`
combined with the reproduce7 results made on this page's code (C4, D79/D82: a stored
PASS is provisional until reproduce7 reproduces and FALLBACK if it does not; results for
other code are ignored, so a later reproduce7 rebuilds the committed page unchanged);
every absent input gives a "Missing: ... Run `tasks.py ...`" placeholder. Sections:

 1. Summary and interim 10.6 verdict
 2. Pre-registration (rl7 sha, seed map, rules, `changes`)
 3. Protocol and compute used
 4. Stage T: table, decision record, learning curves with seed bands
 5. KPI curves per training step (10.4); holdout curves
 6. Action/state variants and PPO; tried-and-dropped prototypes (exploratory)
 7. Stage F and the S3 selection (all 5 seeds)
 8. Weight sensitivity (10.3): table, Pareto, equity vs S0, convergence, reweighting
 9. Stress panel and the light-demand failure mode
10. First S3 results on dev-test (windows, full 2 h, Holm table, forest)
11. Safety, interventions split, ambulance share (10.5)
12. Reproducibility
13. Limitations and the Part 8 handoff (A-06, S4)

The page is deterministic: rebuilt from the same artifacts it is byte-identical except
the one `Code version of this page` line (D82). docs/rl.md stays the Part 6 record
(`rl.evaluate docs`).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from traffic_twin.config import ConfigError, load_config, text_sha256
from traffic_twin.paths import PITIE_RL_S3_CONFIG, RL7_CONFIG, RL7_DIR, RL_CONFIG

log = logging.getLogger(__name__)

SECTIONS = (
    "Summary and interim 10.6 verdict",
    "Pre-registration",
    "Protocol and compute used",
    "Stage T",
    "Learning curves",
    "Action/state variants and PPO",
    "Stage F and the S3 selection",
    "Weight sensitivity (10.3)",
    "Stress panel",
    "First S3 results on dev-test",
    "Safety, interventions and ambulances (10.5)",
    "Reproducibility",
    "Limitations and the Part 8 handoff",
)
TITLE = "# Full RL training, tuning and weight sensitivity (Part 7)"
CODE_LINE = "Code version of this page"  # the only line allowed to differ on a rebuild (D82)
STAGES = ("T", "S", "F")
BASELINE_LABELS = ("S0", "S1", "S2", "stay", "random")
S3_LABELS = ("S3", "RL", "model")  # the S3 policy's label in the study outputs
FULL_RUN_S = 7200.0  # 07:30-09:30, the control window of a full 2 h run
SEED_USES = {
    "train": "training windows: a 1M-step run plays each about once (D70)",
    "selection": "validation of every kept checkpoint; the per-run pick (D72) and the "
    "stage-T score (D73)",
    "select2": "stage 2 of the S3 pick: full 2 h env episodes (D75)",
    "holdout": "reporting only, never to select: windows plus full episodes per pick",
    "sensitivity": "full 2 h KPI v3 runs of the weight settings and S0/S1/S2 (D76)",
    "stress": "forced-condition windows of the stress panel (D78)",
    "reserve": "unused",
    "burned": "seen by a Part 6 prototype, never used",
    "dev_test": "the single dev-test look (D79) uses the first 10; the rest stay unused",
}
TASKS = {
    "campaign": "tasks.py rl7-campaign",
    "holdout": "tasks.py rl7-holdout",
    "select": "tasks.py rl7-select",
    "publish": "tasks.py rl7-publish",
    "stress": "tasks.py rl7-stress",
    "sens": "tasks.py rl7-sens",
    "devtest": "tasks.py rl7-devtest",
    "reproduce": "tasks.py reproduce7",
}
# tried and dropped in the Part 7 prototypes (exploratory: 200k steps, 1-2 seeds, scored
# on the spent windows 1950-1999; analyst action_state; D68, D77)
PROTOTYPES = (
    (
        "Free-order Discrete(5), MaskablePPO, look-ahead masks",
        "1",
        "+4.6 % (p 0.005)",
        "a real third option at about 2 % of decisions; no skip chosen in 10 windows",
    ),
    ("Free-order Discrete(5), MaskablePPO, adaptive hold", "1", "+5.8 % (p < 1e-4)", "as above"),
    (
        "DQN binary, look-ahead masks + auto-forced actions",
        "2",
        "+5.5 % (p 0.007)",
        "0 interventions, but greens left early (car cost up)",
    ),
    (
        "DQN binary, adaptive hold",
        "2",
        "+1.3 % (p 0.73)",
        "0 interventions by relabelling: the same cuts become forced advances",
    ),
    (
        "Decision interval 3 s",
        "1",
        "+4.6 % (equal steps), +9.2 % (equal episodes)",
        "dropped; 10 s kept as V2",
    ),
    ("Decision interval 10 s", "2", "−0.9 % (p 0.20)", "pre-registered as V2"),
    ("No pedestrian feature block", "2", "−3.2 % (p 0.003)", "pre-registered as V1"),
    ("No bus feature block", "1", "+3.4 % (p 0.003)", "dropped"),
    ("VecNormalize (observations) in training", "1", "−0.4 % (p 0.31)", "dropped (no gain)"),
)
# tried and dropped in the robustness prototypes (exploratory: DQN, 1 seed, 200k steps, 6
# envs; analyst robustness; D77): prototype, seeds, interventions, cost
ROBUSTNESS_PROTOTYPES = (
    (
        "Inference-time mask on the Part 6 selected model (stay -> advance when the slack is "
        "below 5 s or max green is due before the next decision)",
        "– (no training)",
        "all removed",
        "+2.4 % mean vs the unmasked model (−2.7 % to +7.5 % by stress condition)",
    ),
    (
        "Mask during training",
        "1",
        "about 0",
        "late validation mean +7.8 % vs a matched 200k control (85.0k vs 78.9k); ×0.6 +20 % vs S1",
    ),
    (
        "Intervention penalty κ = 1000 weighted person-seconds per intervention",
        "1",
        "−69 % (775 vs 2,528 over 230 episodes)",
        "late mean +3.8 % vs the control (best +0.6 %); ×0.6 +9.4 % vs S1, holiday +5.0 %",
    ),
)
# tried in the tuning prototypes (exploratory: DQN, 2 seeds, 200k steps, 3 envs, best
# checkpoint on the held-out windows 1950-1999; analyst tuning): prototype, seeds, cost
# vs S2 (mean ± over the 2 seeds), note
TUNING_PROTOTYPES = (
    ("Part 6 hyperparameters", "2", "+5.0 ± 0.3 %", "registered as T0"),
    ("Linear learning-rate decay", "2", "+4.3 ± 3.5 %", "best-final gap 0.3 pp; registered as T1"),
    (
        "γ 0.99",
        "2",
        "+5.2 ± 1.3 %",
        "best on validation by 3.5 pp, a lead that vanished on the held-out windows; dropped",
    ),
    (
        "Update-to-data 0.25 (train_freq 4, gradient_steps 3)",
        "2",
        "+5.4 ± 1.2 %",
        "no detectable effect; dropped",
    ),
    (
        "Full 2 h training episodes",
        "2",
        "+9.7 ± 3.1 % (+8.0 % on full 2 h episodes)",
        "about 2× the guard interventions; dropped",
    ),
)
# what a `-dirty` code version recorded by a Part 7 step stood for (analyst record, from
# the tree status logged at the start of each step); a dirty version not listed here is
# stated without an explanation
DIRTY_NOTES = {
    "5bec559-dirty": "the tree was `5bec559` plus untracked planned outputs only "
    "(`configs/pitie_rl_s3.yaml`, the S3 model under `models/rl/`, "
    "`tests/data/rl_s3_golden.json` and `docs/figures/rl7/sensitivity.csv`); no tracked "
    "file differed (tree status logged at the start of the stress and dev-test steps)",
}
PED_GROUP = "pedestrian"  # the feature group V1 drops (features.GROUPS)


# -- formatting ------------------------------------------------------------------------
def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def _f(x: Any, nd: int = 1, unit: str = "") -> str:
    v = _num(x)
    return "–" if not math.isfinite(v) else f"{v:,.{nd}f}{unit}"


def _pct(x: Any, nd: int = 1) -> str:
    v = _num(x)
    return "–" if not math.isfinite(v) else f"{v:+.{nd}f} %"


def _p(x: Any) -> str:
    v = _num(x)
    return "–" if not math.isfinite(v) else (f"{v:.3f}" if v >= 0.001 else "< 0.001")


def _yes(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return "yes" if bool(x) else "no"


def _cell(x: Any) -> str:
    """A generic table cell: numbers rounded, containers compact, None as a dash."""
    if x is None:
        return "–"
    if isinstance(x, bool):
        return _yes(x)
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        if not math.isfinite(x):
            return "–"
        return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"
    if isinstance(x, Mapping):
        return ", ".join(f"{k} {_cell(v)}" for k, v in x.items())
    if isinstance(x, list | tuple):
        return ", ".join(_cell(v) for v in x)
    return str(x).replace("|", "/")


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _dict_table(rows: Sequence[Mapping]) -> list[str]:
    """A table of dict rows with the union of their keys as columns (generic artifacts)."""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    return _table(keys, [[_cell(r.get(k)) for k in keys] for r in rows])


def _missing(what: str, task: str) -> list[str]:
    return [f"_Missing: {what}. Run `{task}`._", ""]


def _rel(path: Path, docs: Path) -> str:
    import os

    return Path(os.path.relpath(Path(path).resolve(), Path(docs).resolve().parent)).as_posix()


def _fig(figs: Mapping[str, Path], name: str, alt: str, docs: Path) -> list[str]:
    return [f"![{alt}]({_rel(figs[name], docs)})", ""] if name in figs else []


def _range(pair: Any) -> str:
    try:
        lo, hi = pair
    except (TypeError, ValueError):
        return "–"
    return f"{lo}-{hi}"


def _steps(n: Any) -> str:
    v = _num(n)
    if not math.isfinite(v):
        return "–"
    return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"


def _sf(x: Any, nd: int = 1) -> str:
    """A signed number (no unit), a dash when absent."""
    v = _num(x)
    return "–" if not math.isfinite(v) else f"{v:+.{nd}f}"


def _limit_text(value: Any, limit: float, what: str) -> str:
    """`what 0.98478 against the limit 0.98500 (below it by 0.00022)`: a rule ratio printed
    precisely enough to see how far it is from its threshold."""
    v = _num(value)
    if not math.isfinite(v):
        return f"{what} –"
    if not math.isfinite(limit):
        return f"{what} {v:.5f}"
    side = "below" if v <= limit else "above"
    return f"{what} {v:.5f} against the limit {limit:.5f} ({side} it by {abs(limit - v):.5f})"


# -- observation of a policy -----------------------------------------------------------
def obs_of(m: Any) -> tuple[tuple[str, ...], int | None] | None:
    """(dropped feature groups, number of features) of a run manifest (`observation`) or a
    published model manifest (`obs_drop`, `n_features`); None when it records neither."""
    if not isinstance(m, Mapping):
        return None
    o = m.get("observation")
    if isinstance(o, Mapping) and ("drop" in o or "n_features" in o):
        drop, n = o.get("drop"), o.get("n_features")
    elif "obs_drop" in m or "n_features" in m:
        drop, n = m.get("obs_drop"), m.get("n_features")
    else:
        return None
    k = _num(n)
    return tuple(sorted(str(x) for x in drop or ())), int(k) if math.isfinite(k) else None


def s3_obs(data: Mapping) -> tuple[tuple[str, ...], int | None] | None:
    """S3's observation: the published manifest, else the selected stage-F run's."""
    obs = obs_of(data.get("published"))
    if obs is None:
        chosen = (data.get("select") or {}).get("chosen")
        for m in data.get("runs") or []:
            if chosen and m.get("run") == chosen:
                obs = obs_of(m)
    return obs


def sens_obs(data: Mapping) -> set[tuple[tuple[str, ...], int | None]]:
    """The observations of the weight-sensitivity policies (their run manifests)."""
    by_run = {m.get("run"): m for m in data.get("runs") or []}
    out = set()
    for p in (data.get("sens_plan") or {}).get("policies") or []:
        obs = obs_of(by_run.get(p.get("run")))
        if obs is not None:
            out.add(obs)
    return out


def _obs_text(obs: tuple[tuple[str, ...], int | None]) -> str:
    drop, n = obs
    feats = f"{n} features" if n is not None else "its features"
    if not drop:
        return f"{feats} (the full observation, no group dropped)"
    return f"{feats} (group{'s' if len(drop) > 1 else ''} {', '.join(drop)} dropped)"


def s3_variant(data: Mapping) -> str:
    v = (data.get("published") or {}).get("variant") or (data.get("decision_t") or {}).get("chosen")
    return str(v).upper() if v else "S3's config"


def s3_without_pedestrians(data: Mapping) -> bool:
    obs = s3_obs(data)
    return obs is not None and PED_GROUP in obs[0]


def sens_other_agent(data: Mapping) -> str | None:
    """When the sensitivity policies observe differently from S3: the sentence saying so
    (plan section 11, V1 adoption), else None."""
    s3, sens = s3_obs(data), sens_obs(data)
    if s3 is None or not sens or sens == {s3}:
        return None
    seen = "; ".join(_obs_text(o) for o in sorted(sens, key=str))
    return (
        f"The sensitivity policies observe {seen}; S3 ({s3_variant(data)}) observes "
        f"{_obs_text(s3)}. Their results, in particular those of the pedestrian-weight "
        "settings, therefore describe a different agent from S3."
    )


# -- inputs ----------------------------------------------------------------------------
def _json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        log.warning("%s: unreadable (%s)", path, e)
        return None


def _csv(path: Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError) as e:
        log.warning("%s: unreadable (%s)", path, e)
        return None


def holdout_cost(df: pd.DataFrame | None) -> float:
    """Mean `cost` of the policy's rows of a holdout table (baseline labels and full-mode
    rows left out when the table marks them)."""
    if df is None or "cost" not in df or df.empty:
        return math.nan
    d = df
    if "label" in d:
        d = d[~d["label"].astype(str).isin(BASELINE_LABELS)]
    if "mode" in d:
        d = d[d["mode"].astype(str) != "full"]
    return float(pd.to_numeric(d["cost"], errors="coerce").mean()) if len(d) else math.nan


def baseline_totals(df: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    """label -> mean per window of the numeric columns of a run's baselines.csv."""
    if df is None or "label" not in df:
        return {}
    return {
        str(k): {c: float(v) for c, v in g.mean(numeric_only=True).items()}
        for k, g in df.groupby("label", sort=False)
    }


def collect_runs(root: Path) -> list[dict]:
    """Every Part 7 training run under `<root>/{T,S,F}/`: its manifest with `stage`,
    `variant`, provenance, validation, monitor, holdout tables and baseline totals."""
    from traffic_twin.rl.evaluate import find_runs, read_manifest, read_monitor

    out = []
    for stage in STAGES:
        for d in find_runs(Path(root) / stage):
            m = read_manifest(d)
            prov = _json(d / "provenance.json") or {}
            m["stage"] = stage
            m["provenance"] = prov
            m["variant"] = m.get("variant") or prov.get("variant") or "?"
            m["validation"] = _csv(d / "validation.csv")
            try:
                m["monitor"] = (
                    read_monitor(d / "monitor.csv") if (d / "monitor.csv").is_file() else None
                )
            except (OSError, ValueError, pd.errors.ParserError):
                m["monitor"] = None
            m["holdout"] = _csv(d / "holdout.csv")
            m["holdout_curve"] = _csv(d / "holdout_curve.csv")
            base = _csv(d / "baselines.csv")
            m["baseline_totals"] = baseline_totals(base)
            out.append(m)
    return out


def s3_label(labels: Sequence[str]) -> str | None:
    """The S3 policy's label among a table's labels."""
    labels = [str(x) for x in labels]
    for lab in S3_LABELS:
        if lab in labels:
            return lab
    rest = [x for x in labels if x not in BASELINE_LABELS]
    return rest[0] if rest else None


def _col(df: pd.DataFrame, *tokens: str) -> str | None:
    """The first column whose name contains every token (case-insensitive)."""
    for c in df.columns:
        name = str(c).lower()
        if all(t.lower() in name for t in tokens):
            return c
    return None


def stress_frame(summary: pd.DataFrame | None, episodes: pd.DataFrame | None) -> pd.DataFrame:
    """cell, ref (S1 / S2), pct: S3's env cost change per stress cell (ratio of means),
    from `stress/episodes.csv` (cell, label, seed, cost) or, failing that, from
    `stress/summary.csv` (cell, label, cost mean; or cell and `vs_S1` / `vs_S2` % columns)."""
    rows = []
    src = None
    for df in (episodes, summary):
        if df is not None and {"cell", "label"} <= set(df.columns):
            cost = "cost" if "cost" in df else _col(df, "cost", "mean")
            if cost:
                src = df.assign(_cost=pd.to_numeric(df[cost], errors="coerce"))
                break
    if src is not None:
        means = src.groupby(["cell", "label"], sort=False)["_cost"].mean()
        lab = s3_label(list(dict.fromkeys(src["label"])))
        for cell in dict.fromkeys(src["cell"]):
            for ref in ("S1", "S2"):
                a, b = means.get((cell, lab), math.nan), means.get((cell, ref), math.nan)
                if math.isfinite(a) and math.isfinite(b) and b > 0:
                    rows.append({"cell": str(cell), "ref": ref, "pct": 100.0 * (a / b - 1.0)})
        return pd.DataFrame(rows, columns=["cell", "ref", "pct"])
    if summary is not None and "cell" in summary:
        for ref in ("S1", "S2"):
            c = _col(summary, f"vs_{ref}") or _col(summary, ref, "pct")
            if c is None:
                continue
            for cell, v in zip(summary["cell"], summary[c], strict=True):
                if math.isfinite(_num(v)):
                    rows.append({"cell": str(cell), "ref": ref, "pct": float(v)})
    return pd.DataFrame(rows, columns=["cell", "ref", "pct"])


def devtest_runs(devtest: Path) -> pd.DataFrame:
    """One row per full 2 h dev-test run folder (`<scenario>_seed<n>/run.json`): danger,
    collisions, teleports, curb max, interventions and their D77 split, preemption."""
    rows = []
    if not Path(devtest).is_dir():
        return pd.DataFrame()
    for d in sorted(Path(devtest).glob("*_seed*")):
        meta = _json(d / "run.json")
        if not meta:
            continue
        sb, health = meta.get("safety") or {}, meta.get("health") or {}
        ctl, pre = meta.get("control") or {}, meta.get("preemption") or {}
        curb = math.nan
        k = _csv(d / "kpi_run.csv")
        if k is not None and {"user_type", "kpi", "value"} <= set(k.columns):
            v = k[(k["user_type"] == "pedestrian") & (k["kpi"] == "ped_curb_wait_max")]["value"]
            curb = float(v.iloc[0]) if len(v) else math.nan
        split = ctl.get("iv_split") or {}
        rows.append(
            {
                "scenario": meta.get("scenario"),
                "seed": meta.get("seed"),
                "danger": int(sb.get("danger", 0) or 0),
                "collisions": int(health.get("collisions", 0) or 0),
                "teleports": int(health.get("teleports", 0) or 0),
                "curb_max": curb,
                "interventions": sum(int(v) for v in (sb.get("interventions") or {}).values()),
                **{f"split_{x}": split.get(x) for x in ("cut", "replaced", "cut_foreseeable")},
                "duration_s": _num(meta.get("duration_s")),
                "preempt_requests": pre.get("requests"),
                "preempt_served": pre.get("served"),
                "preempt_timeouts": pre.get("timeouts"),
                "model_version": ctl.get("model_version"),
                "code_version": meta.get("code_version"),
            }
        )
    return pd.DataFrame(rows)


def repro_applies(head: str, code: str) -> bool:
    """Does a reproduce7 result made at commit `head` speak for the code this page is
    generated from (`code` = `sim.results.code_version()`)? Only for a clean tree whose
    files outside docs/ equal `head`'s (code_version ignores docs/ too): the same commit,
    or a later one that changed only docs/ (the regenerated page). A dirty or unknown
    tree, other code or an unknown commit: no."""
    if not head or not code or code == "unknown" or code.endswith("-dirty"):
        return False
    if head == code:
        return True
    import subprocess

    try:
        r = subprocess.run(
            ["git", "--no-optional-locks", "diff", "--quiet", head, "--", ".", ":(exclude)docs"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return r.returncode == 0


def find_repro(study: Path, code: str | None = None) -> list[dict]:
    """The reproduce7 results (`<study>/_reproduce/<HEAD>/*.json` with a `same` key, as
    `tasks.py reproduce7` writes them) that apply to this code (`repro_applies`), oldest
    first. A result for other code is left out, so the page does not change when a
    reproduce7 runs on other code (the page rebuilt by a later reproduce7 is then the
    committed one, D82), and never cites a check of code it was not built from."""
    root = Path(study) / "_reproduce"
    if not root.is_dir():
        return []
    if code is None:
        from traffic_twin.sim.results import code_version

        code = code_version()
    found = []
    for p in sorted(root.glob("*/*.json")):
        doc = _json(p)
        if not isinstance(doc, Mapping) or "same" not in doc:
            continue
        head = str(doc.get("head") or p.parent.name)
        if repro_applies(head, code):
            found.append((p.stat().st_mtime, p.as_posix(), {**doc, "head": head}))
    return [doc for *_, doc in sorted(found, key=lambda t: (t[0], t[1]))]


def repro_status(results: Sequence[Mapping]) -> bool | None:
    """True when every reproduce7 result reproduced, False when one did not, None when
    there is none (pending)."""
    if not results:
        return None
    return all(bool(r.get("same")) for r in results)


def _pass_of(x: Any) -> bool | None:
    return None if x is None else bool(x)


def effective_verdict(v: Mapping, results: Sequence[Mapping]) -> dict:
    """The interim 10.6 verdict as the page states it (D79): `devtest/verdict.json`
    (written once, at the dev-test look, before the commit and so before reproduce7)
    combined with the reproduce7 results that apply to this code. C4 needs reproduce7's
    determinism check: while it has not run, C4 is pending and a PASS is provisional; a
    result that did not reproduce fails C4 and turns the verdict into FALLBACK.
    verdict.json itself is never changed, and a stored FALLBACK is never lifted.

    Returns verdict (pass / pending / fallback), provisional (pass except a pending
    reproduce7), downgraded (a look-time pass or pending failed by reproduce7), c4
    (yes / no / pending), repro (cell text) and pending (verdict.json's list, C4.repro
    dropped once a result applies)."""
    c4 = v.get("C4") or {}
    stored = c4.get("repro")
    stored_same = stored.get("same") if isinstance(stored, Mapping) else None
    checks = [*results]
    if isinstance(stored, Mapping) and stored_same is not None:
        if not any(str(r.get("head")) == str(stored.get("head")) for r in checks):
            checks.insert(0, {"head": stored.get("head"), "same": bool(stored_same)})
    same = repro_status(checks)
    runs_ok = _pass_of(c4.get("runs_pass"))
    if runs_ok is None:  # a verdict.json without `runs_pass`: count, else its C4 pass
        below, need = c4.get("runs_below_s1"), c4.get("required")
        if isinstance(below, list | tuple):
            below = len(below)
        if math.isfinite(_num(below)) and math.isfinite(_num(need)):
            runs_ok = _num(below) >= _num(need)
        elif stored_same is not False:
            runs_ok = _pass_of(c4.get("pass"))
    if runs_ok is False or same is False:
        c4_pass: bool | None = False
    else:
        c4_pass = None if (runs_ok is None or same is None) else True
    passes = [_pass_of((v.get(k) or {}).get("pass")) for k in ("C1", "C2", "C3")] + [c4_pass]
    at_look = str(v.get("verdict"))
    if at_look == "fallback" or any(p is False for p in passes):
        verdict = "fallback"
    else:
        verdict = "pending" if any(p is None for p in passes) else "pass"
    if checks:
        repro = "; ".join(
            f"{'identical' if r.get('same') else 'NOT identical'} at `{r.get('head')}`"
            for r in checks
        )
    else:
        repro = "pending (runs after the commit)"
    resolved = {"C4.repro"} if same is not None else set()
    pending = [x for x in _names(v.get("pending") or []) if x not in resolved]
    return {
        "verdict": verdict,
        "provisional": verdict == "pending" and all(passes[:3]) and runs_ok is True,
        "downgraded": at_look != "fallback" and verdict == "fallback" and same is False,
        "c4": {True: "yes", False: "no", None: "pending"}[c4_pass],
        "repro": repro,
        "pending": pending,
    }


def collect(config: Path = RL7_CONFIG, study: Path | None = None) -> dict:
    """Every artifact the page reads (None or empty when absent)."""
    rl7_cfg = load_config(config)
    outs = rl7_cfg.get("outputs") or {}
    root = Path(study or outs.get("study") or RL7_DIR)
    runs_root = root / "runs" if study else Path(outs.get("runs") or root / "runs")
    from traffic_twin.rl import sensitivity as sens_mod

    sens = root / sens_mod.SENS_DIR
    dev = root / "devtest"
    sel = root / "select"
    verdict = _json(dev / "verdict.json")
    select = _json(sel / "decision.json")
    models_root = Path(outs.get("models") or "models/rl")
    version = (verdict or {}).get("model_version") or (select or {}).get("model_version")
    published = _json(models_root / str(version) / "manifest.json") if version else None
    if published is None and select and models_root.is_dir():
        for p in sorted(models_root.glob("*/manifest.json")):
            doc = _json(p)
            if isinstance(doc, Mapping) and doc.get("source_run") == select.get("chosen"):
                published, version = doc, doc.get("model_version") or p.parent.name
    return {
        "config": Path(config),
        "rl7": rl7_cfg,
        "study": root,
        "runs": collect_runs(runs_root),
        "campaign": _json(root / "campaign.json"),
        "decision_t": _json(root / "decision_T.json"),
        "select": select,
        "stage1": _csv(sel / "stage1.csv"),
        "stage2": _csv(sel / "stage2.csv"),
        "holdout_summary": _csv(root / "holdout" / "summary.csv"),
        "stress_summary": _csv(root / "stress" / "summary.csv"),
        "stress_episodes": _csv(root / "stress" / "episodes.csv"),
        "sens_plan": _json(sens / sens_mod.PLAN_FILE),
        "sens": {
            name: _csv(sens / f"{name}.csv") for name in (*sens_mod.TABLES, "kpi_runs", "safety")
        },
        "verdict": verdict,
        "paired": _csv(dev / "paired.csv"),
        "devtest_kpi": _csv(dev / "kpi.csv"),
        "devtest_compare": _csv(dev / "compare.csv"),
        "devtest_runs": devtest_runs(dev),
        "model_version": version,
        "published": published,
        "models_root": models_root,
        "repro": find_repro(root),
    }


# -- figure inputs ---------------------------------------------------------------------
def pareto_points(kpi_runs: pd.DataFrame | None, plan: Mapping | None) -> pd.DataFrame | None:
    """Per-policy means of the Pareto metrics (viz.rl7.plot_pareto) from sens/kpi_runs.csv."""
    from traffic_twin.rl.sensitivity import BASELINES, PARETO_METRICS, direction

    if kpi_runs is None or plan is None or kpi_runs.empty:
        return None
    k = kpi_runs[kpi_runs["metric"].isin(PARETO_METRICS)]
    w = k.pivot_table(index=["label", "setting"], columns="metric", values="value", aggfunc="mean")
    w = w.reset_index()
    rewards = plan.get("reward") or {}
    base = rewards.get(plan.get("base"))
    w["kind"] = w["setting"].map(lambda s: "baseline" if s in BASELINES else "policy")
    w["direction"] = w["setting"].map(
        lambda s: direction(rewards[s], base) if s in rewards and base else ""
    )
    return w


def interventions_rows(data: Mapping) -> pd.DataFrame:
    """Guard interventions per hour split cut / replaced / cut_foreseeable (D77): the S3
    dev-test 2 h runs, each F pick on the selection windows and each sensitivity setting
    (2 h runs)."""
    rows = []
    dr = data.get("devtest_runs")
    if dr is not None and len(dr) and "split_cut" in dr:
        g = dr[dr["split_cut"].notna()]
        if len(g):
            hours = float(g["duration_s"].fillna(FULL_RUN_S).clip(upper=FULL_RUN_S).sum()) / 3600
            rows.append(
                {
                    "label": "S3, dev-test 2 h runs",
                    **{
                        f"{k}_per_h": float(g[f"split_{k}"].sum()) / hours
                        for k in ("cut", "replaced", "cut_foreseeable")
                    },
                }
            )
    for m in sorted(
        (m for m in data.get("runs") or [] if m["stage"] == "F"), key=lambda m: int(m["seed"])
    ):
        v = m.get("validation")
        if v is None or "split_cut_per_h" not in v or m.get("pick_step") is None:
            continue
        r = v[v["step"] == m["pick_step"]]
        if len(r) and math.isfinite(_num(r["split_cut_per_h"].iloc[0])):
            rows.append(
                {
                    "label": f"F {m['variant'].upper()} s{m['seed']} pick, selection windows",
                    **{
                        f"{k}_per_h": float(r[f"split_{k}_per_h"].iloc[0])
                        for k in ("cut", "replaced", "cut_foreseeable")
                    },
                }
            )
    saf = (data.get("sens") or {}).get("safety")
    plan = data.get("sens_plan")
    if saf is not None and plan and "cut" in saf:
        setting = {p["label"]: p["setting"] for p in plan.get("policies", [])}
        s = saf.assign(setting=saf["label"].map(setting)).dropna(subset=["setting", "cut"])
        for name, g in s.groupby("setting", sort=False):
            hours = len(g) * FULL_RUN_S / 3600
            rows.append(
                {
                    "label": f"{name}, sensitivity 2 h runs",
                    **{
                        f"{k}_per_h": float(g[k].sum()) / hours
                        for k in ("cut", "replaced", "cut_foreseeable")
                    },
                }
            )
    return pd.DataFrame(rows)


def figure_data(data: Mapping) -> dict:
    curves = []
    for m in data.get("runs") or []:
        hc = m.get("holdout_curve")
        if m["stage"] != "F" or hc is None or "step" not in hc or "cost" not in hc:
            continue
        d = hc
        if "label" in d:
            d = d[~d["label"].astype(str).isin(BASELINE_LABELS)]
        if "mode" in d:
            d = d[d["mode"].astype(str) != "full"]
        c = d.groupby("step", sort=True)["cost"].mean().reset_index()
        c["run"], c["seed"] = m.get("run"), m.get("seed")
        c["pick"] = c["step"] == m.get("pick_step")
        curves.append(c)
    hbase = None
    hs = data.get("holdout_summary")
    f_names = {m.get("run") for m in data.get("runs") or [] if m["stage"] == "F"}
    if hs is not None and "run" in hs:
        rows = hs[hs["run"].isin(f_names)]
        if len(rows):
            r = rows.iloc[0]
            hbase = {k: float(r[f"w30_{k}"]) for k in ("S0", "S1", "S2") if f"w30_{k}" in rows}
    gate = ((data["rl7"].get("protocol") or {}).get("stress") or {}).get("worse_than_s1_rel", 0.05)
    return {
        "runs": data.get("runs"),
        "reference": ((data["rl7"].get("protocol") or {}).get("stage_t") or {}).get("reference"),
        "holdout_curves": pd.concat(curves, ignore_index=True) if curves else None,
        "holdout_baselines": hbase,
        "decision_t": data.get("decision_t"),
        "pareto": pareto_points((data.get("sens") or {}).get("kpi_runs"), data.get("sens_plan")),
        "stress": stress_frame(data.get("stress_summary"), data.get("stress_episodes")),
        "stress_limit_pct": 100.0 * float(gate),
        "paired": data.get("paired"),
        "interventions": interventions_rows(data),
    }


# -- sections --------------------------------------------------------------------------
def _pick_row(m: Mapping) -> dict:
    """The validation row of a run's pick (empty when absent)."""
    v = m.get("validation")
    if v is None or m.get("pick_step") is None or "step" not in v:
        return {}
    r = v[v["step"] == m["pick_step"]]
    return r.iloc[0].to_dict() if len(r) else {}


def holdout_of(data: Mapping, m: Mapping) -> dict[str, float]:
    """A run's holdout window cost and its % vs S1 and S2: `holdout/summary.csv`, else
    the run's own holdout.csv (cost only). Reporting only (D72, D75)."""
    hs = data.get("holdout_summary")
    if hs is not None and "run" in hs:
        r = hs[hs["run"] == m.get("run")]
        if len(r):
            r = r.iloc[-1]
            return {
                k: _num(r.get(c))
                for k, c in (
                    ("cost", "holdout_w30"),
                    ("vs_S1", "w30_vs_S1"),
                    ("vs_S2", "w30_vs_S2"),
                )
            }
    return {"cost": holdout_cost(m.get("holdout")), "vs_S1": math.nan, "vs_S2": math.nan}


def _vs(m: Mapping, value: Any, ref: str) -> float:
    b = _num((m.get("baselines") or {}).get(ref))
    v = _num(value)
    return 100.0 * (v / b - 1.0) if math.isfinite(v) and math.isfinite(b) and b > 0 else math.nan


def summary_section(data: Mapping) -> list[str]:
    lines = ["## 1. Summary and interim 10.6 verdict", ""]
    v = data.get("verdict")
    proto = data["rl7"].get("protocol") or {}
    gate, dev = proto.get("gate_10_6") or {}, proto.get("devtest") or {}
    head = []
    dt = data.get("decision_t") or {}
    if dt.get("chosen"):
        head.append(f"stage T chose **{str(dt['chosen']).upper()}** (D73)")
    f_runs = [m for m in data.get("runs") or [] if m["stage"] == "F"]
    if f_runs:
        head.append(f"{len(f_runs)} final trainings (stage F)")
    if data.get("model_version"):
        head.append(f"S3 = `{data['model_version']}`")
    if head:
        lines += ["**So far:** " + "; ".join(head) + ".", ""]
    if not v:
        lines += _missing("the dev-test verdict (`devtest/verdict.json`)", TASKS["devtest"])
        lines += [
            "Until then no 10.6 statement is made. The rule is pre-registered (D79): RL beats "
            "the actuated baseline S1 when C1 (person delay, HL < 0 and exact Wilcoxon p < "
            f"{dev.get('alpha', 0.05)}), C2 (no class more than "
            f"{100 * float(gate.get('equity_rel', 0.05)):g} % worse than S1), C3 (safety) and "
            "C4 (reproducibility) all hold.",
            "",
        ]
        return lines
    eff = effective_verdict(v, data.get("repro") or [])
    passed = eff["verdict"] == "pass"
    word = {"pass": "PASS", "pending": "PENDING", "fallback": "FALLBACK"}[eff["verdict"]]
    if eff["provisional"]:
        word = "PASS, provisional"
    lines += [
        f"**Interim 10.6 verdict: {word}** "
        f"(dev-test seeds {_seeds_text(v.get('seeds'))}, looked at once; exploratory, n = "
        f"{len(v.get('seeds') or []) or '–'}).",
        "",
    ]
    if eff["provisional"]:
        lines += [
            "Provisional: C4 also needs `reproduce7`'s determinism check (D82), which runs "
            f"after the commit (`{TASKS['reproduce']}`). Until it reproduces, the verdict is "
            "not final; if it does not, C4 fails and the verdict becomes FALLBACK (D79).",
            "",
        ]
    elif eff["downgraded"]:
        lines += [
            "`devtest/verdict.json` (computed at the dev-test look, before reproduce7) records "
            "no failed criterion, with reproduce7 pending; reproduce7 then gave: "
            f"{eff['repro']}. C4 therefore fails and the verdict is FALLBACK (D79). "
            "verdict.json is left as written.",
            "",
        ]
    rows = []
    c1 = v.get("C1") or {}
    rows.append(
        [
            "C1 person delay vs S1",
            f"HL {_f(c1.get('hl'), 1, ' s')}, p {_p(c1.get('p'))}",
            _yes(c1.get("pass")),
        ]
    )
    c2 = v.get("C2") or {}
    c2_parts = [
        f"`{r.get('kpi')}` {_pct(r.get('rel_pct'))}" + (" (worse)" if r.get("worse") else "")
        for r in c2.get("rows") or []
    ]
    c2_txt = "; ".join(c2_parts) if c2_parts else "no KPI rows"
    if c2.get("limit_pct") is not None:
        c2_txt += f"; limit +{_f(c2['limit_pct'], 0, ' %')}"
    rows.append(["C2 equity vs S1", c2_txt, _yes(c2.get("pass"))])
    c3 = v.get("C3") or {}
    viol = c3.get("violations") or []
    rows.append(
        [
            "C3 safety",
            "no violation" if not viol else f"{len(viol)} violations",
            _yes(c3.get("pass")),
        ]
    )
    c4 = v.get("C4") or {}
    below = c4.get("runs_below_s1")
    below_txt = f"{len(below)}" if isinstance(below, list | tuple) else _cell(below)
    if c4.get("of") is not None:
        below_txt += f" of {c4['of']} (need {_cell(c4.get('required'))})"
    rows.append(
        [
            "C4 reproducibility",
            f"F picks below S1 on the holdout windows: {below_txt}; reproduce7 {eff['repro']}",
            eff["c4"],
        ]
    )
    lines += _table(["Criterion (D79)", "Result", "Pass"], rows)
    lines.append("")
    if s3_without_pedestrians(data):
        n = (s3_obs(data) or ((), None))[1]
        feats = f"its {n} features" if n is not None else "its features"
        note = (
            f"S3 does not observe pedestrians: {s3_variant(data)} drops the pedestrian feature "
            f"group, so none of {feats} is a pedestrian count, curb wait or shield slack, and "
            "the pedestrian shield alone keeps the curb-wait bound."
        )
        ped_worse = [
            r
            for r in c2.get("rows") or []
            if str(r.get("kpi", "")).startswith(f"{PED_GROUP}/") and r.get("worse")
        ]
        if ped_worse:
            note += (
                " C2's pedestrian result ("
                + ", ".join(f"`{r.get('kpi')}` {_pct(r.get('rel_pct'))}" for r in ped_worse)
                + ") is that of a policy that does not see pedestrians; whether a policy that "
                "observes them would pass C2 was not tested, and no config is added after "
                "the look (D79)."
            )
        lines += [note, ""]
    if eff["provisional"]:
        lines.append(
            "S3 beats the actuated baseline on C1-C3 and on the holdout part of C4. If "
            "reproduce7 reproduces, it is carried into Part 8 as the main candidate (spec "
            "table). This is interim: the binding comparison is A-06 on seeds 1-10 in Part 8."
        )
    elif eff["verdict"] == "pending":
        lines.append(
            "Not every criterion is evaluated yet (see Pending below): no 10.6 statement is "
            "made until they are (D79)."
        )
    elif passed:
        lines.append(
            "S3 beats the actuated baseline on the pre-registered criteria, so it is carried "
            "into Part 8 as the main candidate (spec table). This is interim: the binding "
            "comparison is A-06 on seeds 1-10 in Part 8."
        )
    else:
        lines.append(
            "Spec 10.6 applies: RL does not beat the actuated baseline on the pre-registered "
            "criteria, so the rules (S2) become the main solution and RL is reported as a "
            "study (what was tried and why it does not improve). Part 8 still runs S3 as "
            "registered."
        )
    framing = v.get("s2_framing")
    if isinstance(framing, Mapping):
        extra = ""
        if math.isfinite(_num(framing.get("hl"))):
            extra = f" (person delay HL {_f(framing['hl'], 1, ' s')}, p {_p(framing.get('p'))})"
        lines += ["", f"Against S2 (rules): {framing.get('text', framing.get('framing'))}{extra}."]
    elif framing:
        lines += ["", f"Against S2 (rules): {_cell(framing)}."]
    if eff["pending"]:
        lines += ["", f"Pending: {_cell(eff['pending'])}."]
    worse = v.get("stress_cells_worse") or []
    if worse:
        lines += [
            "",
            "Stress cells where S3 is more than 5 % worse than S1 (reported, not "
            "gating): " + ", ".join(f"`{c}`" for c in _names(worse)) + ".",
        ]
    return [*lines, ""]


def _names(x: Any) -> list[str]:
    if isinstance(x, Mapping):
        return [str(k) for k in x]
    if isinstance(x, list | tuple):
        return [str(e.get("cell", e)) if isinstance(e, Mapping) else str(e) for e in x]
    return [str(x)]


def _seeds_text(seeds: Any) -> str:
    """A list of seeds as '3020-3029' when consecutive, else joined (no separators)."""
    if not isinstance(seeds, list | tuple) or not seeds:
        return "–"
    xs = [int(x) for x in seeds]
    if len(xs) > 2 and xs == list(range(xs[0], xs[0] + len(xs))):
        return f"{xs[0]}-{xs[-1]}"
    return ", ".join(str(x) for x in xs)


def _range_n(pair: Any) -> int | None:
    """Number of seeds of an inclusive [lo, hi] config range."""
    try:
        lo, hi = (int(x) for x in pair)
    except (TypeError, ValueError):
        return None
    return hi - lo + 1 if hi >= lo else None


def run_configs(config: Path) -> list[list[str]]:
    """File, role comment and changed keys of every `configs/rl7/*.yaml`."""
    rows = []
    for p in sorted((Path(config).parent / "rl7").glob("*.yaml")):
        text = p.read_text(encoding="utf-8")
        head = []
        for ln in text.splitlines():
            if not ln.startswith("#"):
                break
            head.append(ln.lstrip("# ").strip())
        role = " ".join(head)
        doc = yaml.safe_load(text) or {}
        keys = {k: v for k, v in doc.items() if k not in ("extends", "variant")}
        change = json.dumps(keys, separators=(", ", ": "))[1:-1].replace('"', "") or "–"
        rows.append([f"`{p.name}`", f"`{doc.get('extends', '')}`", change.replace("|", "/"), role])
    return rows


def prereg_section(data: Mapping) -> list[str]:
    cfg, config = data["rl7"], data["config"]
    sha = text_sha256(config) or ""
    changes = cfg.get("changes") or []
    p = cfg.get("protocol") or {}
    st, sf, ss = p.get("stage_t") or {}, p.get("stage_f") or {}, p.get("sensitivity") or {}
    gate = p.get("gate_10_6") or {}
    dev = p.get("devtest") or {}
    lines = [
        "## 2. Pre-registration",
        "",
        f"`{Path(config).as_posix()}` (text sha256 `{sha[:12]}`, extends `configs/rl.yaml` "
        f"`{(text_sha256(RL_CONFIG) or '')[:8]}`, the frozen Part 6 record), pre-registered "
        f"**{cfg.get('preregistered')}** before any Part 7 training (D68-D90). It changes only "
        f"through dated `changes` entries: {len(changes)} so far.",
        "",
    ]
    for c in changes:
        if isinstance(c, Mapping):
            lines.append(f"- {c.get('date')}: {c.get('what')} ({c.get('why')})")
    if changes:
        lines.append("")
    seeds = cfg.get("seeds") or {}
    lines += _table(
        ["Seed set", "Range", "Use"],
        [
            [k, _range(seeds[k]), SEED_USES.get(k, "")]
            for k in sorted(
                seeds, key=lambda k: (list(SEED_USES).index(k) if k in SEED_USES else 99, k)
            )
        ],
    )
    lines += [
        "",
        "Never run in Part 7: 1-10, 101-103, 201-207 (Parts 3-4, `configs/evaluation.yaml`), "
        "2000-2999 and 4000+ (Part 5); 1000-1999 and 3000-3009 are spent by Part 6 and its "
        "prototypes.",
        "",
        "**Rules fixed before training:**",
        "",
        f"- **Pick per run (D72):** the validation checkpoint (final model included) with the "
        f"lowest mean cost on the selection windows among the validations at ≥ "
        f"{100 * float((cfg.get('train') or {}).get('pick_from', 0.5)):g} % of the budget; "
        "`late_mean` = mean cost of those validations.",
        f"- **Stage T (D73):** configs {', '.join(str(c).upper() for c in st.get('configs', []))} "
        f"× seeds {st.get('seeds')} × {_steps(st.get('steps'))} steps; score = mean over seeds "
        f"of `late_mean`. C* = {str(st.get('reference', 't1')).upper()} unless a candidate has "
        f"(a) a score ≤ {1 - float(st.get('margin_rel', 0.015)):g} × the reference's, (b) a "
        f"lower `late_mean` for ≥ {st.get('min_seed_wins')} of the matching seeds and (c) a "
        f"late pedestrian waiting ≤ {1 + float(st.get('ped_guard_rel', 0.05)):g} × the "
        "reference's; the lowest qualifying score wins.",
        f"- **Stage F (D74):** C* × seeds {sf.get('seeds')} × {_steps(sf.get('steps'))} steps.",
        f"- **S3 (D75):** stage 1 = each F run's pick; stage 2 = mean cost on the select2 full "
        f"episodes; ties within {100 * float((p.get('select') or {}).get('tie_rel', 0.005)):g} %"
        " go to the lower stage-1 cost, then the lower seed.",
        f"- **Sensitivity (D76):** base {str(ss.get('base', 't1')).upper()} and settings "
        f"{', '.join(ss.get('settings', []))}; seeds {ss.get('seeds_wide')} on "
        f"{', '.join(ss.get('wide', []))}, {ss.get('seeds_default')} elsewhere; "
        f"{_steps(ss.get('steps'))} steps; descriptive only.",
        f"- **Interim 10.6 (D79):** dev-test seeds {_range(dev.get('seeds'))}, once. C1 "
        f"`{dev.get('primary')}` S3 vs S1: HL < 0 and exact Wilcoxon p < {dev.get('alpha')}; C2 "
        f"none of {', '.join(f'`{k}`' for k in gate.get('equity_kpis', []))} more than "
        f"{100 * float(gate.get('equity_rel', 0.05)):g} % worse than S1; C3 danger 0, "
        f"collisions 0, teleports 0, curb max ≤ {gate.get('curb_max_s')} s, every preemption "
        f"served, timeouts ≤ S2's; C4 ≥ {gate.get('repro_min_runs')} of the F picks below S1 "
        "on the holdout and `reproduce7` identical.",
        "",
        "Run configs (`configs/rl7/`, each extends `../rl7.yaml` or `t1.yaml` and holds only "
        "what it changes; run identity = `run_config_sha` of the merged config, D69):",
        "",
    ]
    lines += _table(["File", "Extends", "Changes", "Role"], run_configs(config))
    return [*lines, ""]


def protocol_section(data: Mapping) -> list[str]:
    cfg = data["rl7"]
    tr = cfg.get("train") or {}
    dqn = cfg.get("dqn") or {}
    p = cfg.get("protocol") or {}
    st, sf, ss = p.get("stage_t") or {}, p.get("stage_f") or {}, p.get("sensitivity") or {}
    n_envs = int(tr.get("n_envs", 6))
    utd = _num(dqn.get("gradient_steps")) / (n_envs * _num(dqn.get("train_freq", 1)))
    t_steps = len(st.get("configs", [])) * len(st.get("seeds", [])) * int(st.get("steps", 0))
    n_s = s_runs(cfg)
    s_steps = n_s * int(ss.get("steps", 0))
    f_steps = len(sf.get("seeds", [])) * int(sf.get("steps", 0))
    lines = [
        "## 3. Protocol and compute used",
        "",
        f"Every run (D71): SB3 DQN (P1: PPO), {n_envs} `SubprocVecEnv` workers (spawn), torch "
        f"threads {tr.get('torch_threads')} in the learner and the workers, CPU; 30-min "
        f"train-mode windows (D59); exploration over {int(dqn.get('exploration_steps', 0)):,} "
        f"absolute steps; update-to-data {utd:.2f}; validation every "
        f"{int(tr.get('eval_every_steps', 0)):,} steps on the selection windows with "
        f"{tr.get('eval_workers')} persistent workers, every validation checkpoint kept; SB3 "
        f"checkpoints every {int(tr.get('checkpoint_every_steps', 0)):,} steps; no resume (a "
        "crash reruns from scratch).",
        "",
        f"Budget in decision steps: stage T {_steps(t_steps)}, sensitivity {_steps(s_steps)} "
        f"({n_s} new runs), stage F {_steps(f_steps)}; total "
        f"{_steps(t_steps + s_steps + f_steps)}.",
        "",
    ]
    camp = data.get("campaign")
    if camp:
        jobs = camp.get("jobs") if isinstance(camp, Mapping) else camp
        jobs = list(jobs.values()) if isinstance(jobs, Mapping) else list(jobs or [])
        if jobs:
            df = pd.DataFrame(jobs)
            if "stage" not in df:
                df["stage"] = "?"

            def num(g: pd.DataFrame, *cols: str) -> pd.Series | None:
                col = next((c for c in cols if c in g), None)
                return None if col is None else pd.to_numeric(g[col], errors="coerce")

            rows = []
            for stage, g in df.groupby("stage", sort=False):
                status = g["status"].value_counts().sort_index().to_dict() if "status" in g else {}
                wall, sps, att = num(g, "wall_s", "wall"), num(g, "steps_per_s"), num(g, "attempts")
                rows.append(
                    [
                        stage,
                        len(g),
                        _cell(status),
                        _f(wall.sum() / 3600, 1) if wall is not None else "–",
                        _f(sps.mean(), 0) if sps is not None else "–",
                        int((att > 1).sum()) if att is not None else "–",
                    ]
                )
            lines += _table(
                ["Stage", "Jobs", "Status", "Job-hours", "Steps/s (mean)", "Retried"], rows
            )
            slots = camp.get("slots") if isinstance(camp, Mapping) else None
            if slots:
                lines += ["", f"Concurrent jobs: {slots} (D84)."]
            lines.append("")
    else:
        lines += _missing("the campaign record (`campaign.json`)", TASKS["campaign"])
    runs = data.get("runs") or []
    if runs:
        rows = []
        for m in runs:
            rows.append(
                [
                    f"`{m.get('run')}`",
                    m["stage"],
                    str(m.get("variant")).upper(),
                    str(m.get("algo", "")).upper(),
                    m.get("seed"),
                    f"{int(_num(m.get('steps_done', m.get('steps')))):,}"
                    if math.isfinite(_num(m.get("steps_done", m.get("steps"))))
                    else "–",
                    _f(_num(m.get("wall_s")) / 60, 0),
                    _f(m.get("steps_per_s"), 0),
                    f"`{m.get('code_version')}`",
                    _yes(m.get("allow_dirty", (m.get("provenance") or {}).get("allow_dirty"))),
                ]
            )
        lines += _table(
            [
                "Run",
                "Stage",
                "Config",
                "Algo",
                "Seed",
                "Steps",
                "Wall min",
                "Train steps/s",
                "Code",
                "--allow-dirty",
            ],
            rows,
        )
        lines.append("")
    else:
        lines += _missing("training runs", TASKS["campaign"])
    return lines


def s_runs(cfg: Mapping) -> int:
    """New sensitivity runs: every setting's seeds, plus the base seeds stage T lacks."""
    p = cfg.get("protocol") or {}
    ss, st = p.get("sensitivity") or {}, p.get("stage_t") or {}
    wide = set(ss.get("wide", []))

    def seeds(s: str) -> list:
        return list(ss.get("seeds_wide" if s in wide else "seeds_default", []))

    n = sum(len(seeds(s)) for s in ss.get("settings", []))
    base = ss.get("base")
    reuse = set(st.get("seeds", [])) if base in st.get("configs", []) else set()
    return n + len([s for s in seeds(base) if s not in reuse])


OUTCOME_KEYS = (
    "score",
    "ratio",
    "ped_s",
    "ped_ratio",
    "reference",
    "wins",
    "seeds",
    "a",
    "b",
    "c",
    "qualifies",
)


def outcome_text(o: Any, rule: Mapping, ref: str) -> str:
    """One config's D73 rule outcome (`decision_T.json` outcome): the score ratio and the
    pedestrian ratio to 5 decimals against their limits, so a narrow pass is visible."""
    if not isinstance(o, Mapping):
        return _cell(o)
    if o.get("reference"):
        return f"the reference (score {_f(o.get('score'), 0)})"
    margin, guard = _num(rule.get("margin_rel")), _num(rule.get("ped_guard_rel"))
    parts = []
    if "score" in o:
        parts.append(f"score {_f(o['score'], 0)}")
    if "ratio" in o:
        parts.append(_limit_text(o["ratio"], 1.0 - margin, f"ratio to {ref.upper()}"))
    if "a" in o:
        parts.append(f"(a) {_yes(o['a'])}")
    if "wins" in o:
        need = rule.get("min_seed_wins")
        need_txt = f" (need {need})" if need is not None else ""
        parts.append(
            f"late mean below {ref.upper()} on {_cell(o['wins'])} of "
            f"{_cell(o.get('seeds'))} seeds{need_txt}"
        )
    if "b" in o:
        parts.append(f"(b) {_yes(o['b'])}")
    if "ped_ratio" in o:
        parts.append(_limit_text(o["ped_ratio"], 1.0 + guard, "pedestrian waiting ratio"))
    if "c" in o:
        parts.append(f"(c) {_yes(o['c'])}")
    parts += [f"{k} {_cell(v)}" for k, v in o.items() if k not in OUTCOME_KEYS]
    if "qualifies" in o:
        parts.append(f"qualifies: {_yes(o['qualifies'])}")
    return "; ".join(parts)


def stage_t_section(data: Mapping, figs: Mapping[str, Path], docs: Path) -> list[str]:
    lines = ["## 4. Stage T", ""]
    runs = [m for m in data.get("runs") or [] if m["stage"] == "T"]
    dt = data.get("decision_t")
    if not runs and not dt:
        return lines + _missing("stage-T runs", TASKS["campaign"])
    if runs:
        lines += [
            "Each run's pick (D72) on the selection windows: cost, change against S1 and S2 on "
            "the same windows, `late_mean` (the D73 score input), the final model's cost and "
            "the holdout cost (reporting only, never to select).",
            "",
        ]
        rows = []
        for m in sorted(runs, key=lambda m: (str(m["variant"]), int(m.get("seed", 0)))):
            pick = m.get("pick_val_cost")
            rows.append(
                [
                    str(m["variant"]).upper(),
                    m.get("seed"),
                    m.get("pick_step"),
                    _f(pick, 0),
                    _pct(_vs(m, pick, "S1")),
                    _pct(_vs(m, pick, "S2")),
                    _f(m.get("late_mean"), 0),
                    _pct(_vs(m, m.get("late_mean"), "S2")),
                    _f(m.get("final_val_cost"), 0),
                    _f(holdout_of(data, m)["cost"], 0),
                    _pct(holdout_of(data, m)["vs_S2"]),
                ]
            )
        lines += _table(
            [
                "Config",
                "Seed",
                "Pick step",
                "Pick cost",
                "vs S1",
                "vs S2",
                "Late mean",
                "Late vs S2",
                "Final cost",
                "Holdout cost",
                "Holdout vs S2",
            ],
            rows,
        )
        lines.append("")
    if dt:
        means = dt.get("means") or {}
        scores = dt.get("scores") or {}
        ped = dt.get("ped_s") or {}
        ref = str((dt.get("rule") or {}).get("reference") or "t1")
        rows = []
        for c, s in scores.items():
            ref_m = _num(means.get(ref))
            m = _num(means.get(c))
            rows.append(
                [
                    str(c).upper(),
                    ", ".join(f"s{k} {_f(v, 0)}" for k, v in (s or {}).items()),
                    _f(m, 0),
                    _pct(100 * (m / ref_m - 1)) if math.isfinite(m) and ref_m > 0 else "–",
                    _f(ped.get(c), 0),
                    "**chosen**" if c == dt.get("chosen") else "",
                ]
            )
        lines += ["**Decision record** (`decision_T.json`, D73):", ""]
        lines += _table(
            [
                "Config",
                "Late mean per seed",
                "Score",
                f"vs {ref.upper()}",
                "Late pedestrian waiting",
                "",
            ],
            rows,
        )
        outcome = dt.get("outcome")
        if outcome:
            lines += ["", "Rule outcome:", ""]
            if isinstance(outcome, Mapping):
                rule = dt.get("rule") or {}
                lines += [
                    f"- {str(k).upper()}: {outcome_text(v, rule, ref)}" for k, v in outcome.items()
                ]
            else:
                lines.append(f"- {_cell(outcome)}")
        lines += [
            "",
            f"**C\\* = {str(dt.get('chosen')).upper()}.** Stage F trains it (D74); no "
            "config is added after this decision (D79 trial budget).",
            "",
        ]
    else:
        lines += _missing("the stage-T decision (`decision_T.json`)", TASKS["campaign"])
    lines += _fig(figs, "stage_t", "Stage T decision", docs)
    lines += _fig(figs, "learning_bands", "Learning curves with seed bands", docs)
    return lines


def curves_section(data: Mapping, figs: Mapping[str, Path], docs: Path) -> list[str]:
    lines = ["## 5. Learning curves and KPI curves per training step (10.4)", ""]
    if "kpi_curves" not in figs and "holdout_curves" not in figs:
        return [
            *lines,
            *_missing("validation curves", TASKS["campaign"]),
            *_missing("the holdout curves", TASKS["holdout"]),
        ]
    if "kpi_curves" in figs:
        lines += [
            "Spec 10.4 asks for KPI curves per training step: the per-class totals the reward "
            "weights (cars, buses, pedestrians waiting; seconds with a curb wait above θ) and "
            "the guard interventions, at every validation on the selection windows, mean and "
            "seed band per config, with the baselines on the same windows.",
            "",
        ]
        lines += _fig(figs, "kpi_curves", "KPI curves per training step", docs)
    if "holdout_curves" in figs:
        lines += [
            "Holdout curves: every kept checkpoint of the final trainings on the holdout "
            "windows (scored after training, never used to select). They show how far the "
            "selection-window pick is from the holdout optimum.",
            "",
        ]
        lines += _fig(figs, "holdout_curves", "Holdout curves", docs)
    else:
        lines += _missing("the holdout curves", TASKS["holdout"])
    return lines


def variants_section(data: Mapping) -> list[str]:
    lines = ["## 6. Action/state variants and PPO", ""]
    dt = data.get("decision_t") or {}
    means = dt.get("means") or {}
    ref = str((dt.get("rule") or {}).get("reference") or "t1")
    names = {
        "v1": "V1, no pedestrian feature block (62 features)",
        "v2": "V2, 10 s decision interval",
        "p1": "P1, PPO (the 10.4 alternative, D89)",
        "t0": "T0, constant learning rate",
    }
    if means and ref in means:
        rows = []
        for c, label in names.items():
            if c in means:
                m, r = _num(means[c]), _num(means[ref])
                rows.append([label, _f(m, 0), _pct(100 * (m / r - 1)) if r > 0 else "–"])
        lines += [f"Pre-registered variants against {ref.upper()} (stage-T score, D73):", ""]
        lines += _table(["Variant", "Score", f"vs {ref.upper()}"], rows)
        lines.append("")
    else:
        lines += _missing("the stage-T scores", TASKS["campaign"])
    lines += [
        "**Tried and dropped (exploratory prototypes, not part of the formal comparison).** "
        "Action and state prototypes: 200k steps, 1-2 seeds, 6 envs, scored on the spent "
        "windows 1950-1999 against A0, the D55 binary DQN with the rl.yaml hyperparameters "
        "retrained the same way (not the Part 6 model). The masks prototypes either relabel "
        "interventions or pay for removing them (D77), so no mask variant was registered; "
        "sb3-contrib stays out of the pipeline (D89).",
        "",
    ]
    lines += _table(
        ["Prototype", "Seeds", "Cost vs A0 (binary DQN, 6 envs, 200k)", "Note"], PROTOTYPES
    )
    lines += [
        "",
        "Robustness prototypes (DQN, 200k steps, 6 envs; the mask during training and the "
        "penalty against a matched 200k control run, the inference-time mask against the same "
        "model without it): removing or penalising interventions costs performance, most at "
        "light demand, so none was registered (D77).",
        "",
    ]
    lines += _table(["Prototype", "Seeds", "Interventions", "Cost"], ROBUSTNESS_PROTOTYPES)
    lines += [
        "",
        "Tuning prototypes (DQN, 200k steps, 3 envs; best checkpoint on the held-out windows "
        "1950-1999, 30 min, against S2): no knob other than the window length had an effect "
        "that could be detected.",
        "",
    ]
    lines += _table(
        ["Prototype", "Seeds", "Held-out cost vs S2 (mean ± over the seeds)", "Note"],
        TUNING_PROTOTYPES,
    )
    lines += [
        "",
        "Never run: a per-second discount, a held-out day type (D87) and any change of the "
        "deployed reward weights (D76). VecNormalize, the masks, adaptive hold and the "
        "intervention penalty were only prototyped (above), never registered.",
        "",
    ]
    return lines


def stage_f_section(data: Mapping) -> list[str]:
    lines = ["## 7. Stage F and the S3 selection", ""]
    runs = [m for m in data.get("runs") or [] if m["stage"] == "F"]
    if not runs:
        return lines + _missing("the stage-F runs", TASKS["campaign"])
    rows = []
    for m in sorted(runs, key=lambda m: int(m.get("seed", 0))):
        pick = m.get("pick_val_cost")
        rows.append(
            [
                f"`{m.get('run')}`",
                m.get("seed"),
                m.get("pick_step"),
                _f(pick, 0),
                _pct(_vs(m, pick, "S1")),
                _pct(_vs(m, pick, "S2")),
                _f(m.get("late_mean"), 0),
                _f(m.get("final_val_cost"), 0),
                _pct(holdout_of(data, m)["vs_S1"]),
                _pct(holdout_of(data, m)["vs_S2"]),
                _f(m.get("steps_per_s"), 0),
            ]
        )
    lines += _table(
        [
            "Run",
            "Seed",
            "Pick step",
            "Pick cost",
            "vs S1",
            "vs S2",
            "Late mean",
            "Final cost",
            "Holdout vs S1",
            "Holdout vs S2",
            "Steps/s",
        ],
        rows,
    )
    lines.append("")
    sel = data.get("select")
    if not sel:
        return lines + _missing("the S3 selection (`select/decision.json`)", TASKS["select"])
    cands = sel.get("candidates") or []
    lines += [
        "**S3 selection (D75).** Stage 1: each run's pick; stage 2: mean cost on the select2 "
        "full 2 h episodes. All five candidates are shown with their holdout scores so the "
        "selection bias is visible.",
        "",
    ]
    rows = []
    for c in cands:
        rows.append(
            [
                f"`{c.get('run')}`",
                c.get("pick_step"),
                _f(c.get("stage1"), 0),
                _f(c.get("stage2"), 0),
                _f(c.get("holdout_w30"), 0),
                _f(c.get("holdout_full"), 0),
                _f(c.get("late_mean"), 0),
                "**S3**" if c.get("run") == sel.get("chosen") else "",
            ]
        )
    lines += _table(
        [
            "Candidate",
            "Pick step",
            "Stage 1",
            "Stage 2",
            "Holdout windows",
            "Holdout full",
            "Late mean",
            "",
        ],
        rows,
    )
    lines.append("")
    if sel.get("rule"):
        rule = sel["rule"]
        text = rule.get("text") if isinstance(rule, Mapping) else rule
        lines += [f"Rule: {_cell(text)}", ""]
    pub = data.get("published")
    if pub:
        path = Path(data["models_root"]) / str(data.get("model_version")) / "model.zip"
        lines += [
            f"**S3 = `{pub.get('model_version', data.get('model_version'))}`**, committed at "
            f"`{path.as_posix()}` (model sha256 `{str(pub.get('model_sha256'))[:12]}`, policy "
            f"sha256 `{str(pub.get('policy_sha256'))[:12]}`), run through "
            f"`{PITIE_RL_S3_CONFIG.as_posix()}` with the sha256 pin (D86).",
            "",
        ]
    else:
        lines += _missing("the published S3 model", TASKS["publish"])
    return lines


SENS_TABLE = (
    ("car/time_loss_mean", "Car time loss s"),
    ("bus/delay_mean", "Bus delay s"),
    ("pedestrian/ped_curb_wait_mean", "Ped curb mean s"),
    ("pedestrian/ped_curb_wait_p95", "Ped curb p95 s"),
    ("all/person_delay_mean", "Person delay s"),
    ("reward/penalty_s", "Penalty s/run"),
    ("safety/interventions", "Interventions/run"),
)


def resolution_text(boot: pd.DataFrame | None, metric: str, name: str) -> str:
    """What the bootstrap resolved for `metric`, measured (`sens/bootstrap.csv`): the
    smallest change whose CI excludes 0 and the largest whose CI does not; '' if absent."""
    if boot is None or boot.empty or not {"metric", "diff", "beyond_noise"} <= set(boot.columns):
        return ""
    b = boot[boot["metric"] == metric].assign(_abs=lambda d: pd.to_numeric(d["diff"]).abs())
    if b.empty:
        return ""

    def one(r: Any) -> str:
        n = _num(getattr(r, "n_policies", math.nan))
        pol = f", {int(n)} policies" if math.isfinite(n) else ""
        return (
            f"`{r.setting}` {_sf(r.diff)} s [{_sf(r.ci_low)}, {_sf(r.ci_high)}]{pol}"
            if {"ci_low", "ci_high"} <= set(b.columns)
            else f"`{r.setting}` {_sf(r.diff)} s{pol}"
        )

    yes = b[b["beyond_noise"].astype(bool)].sort_values("_abs")
    no = b[~b["beyond_noise"].astype(bool)].sort_values("_abs", ascending=False)
    parts = []
    if len(yes):
        parts.append(f"the smallest change whose CI excludes 0 is {one(next(yes.itertuples()))}")
    if len(no):
        parts.append(f"the largest whose CI includes 0 is {one(next(no.itertuples()))}")
    return f"{name} (measured): " + "; ".join(parts) + ". " if parts else ""


def sens_safety(data: Mapping) -> list[str]:
    """Safety of the sensitivity 2 h runs (`sens/safety.csv`) per setting and baseline:
    danger, collisions, teleports, curb maximum, preemption; the runs with a collision
    named. Reported, not gating."""
    saf = (data.get("sens") or {}).get("safety")
    plan = data.get("sens_plan") or {}
    if saf is None or saf.empty or "label" not in saf:
        return []
    setting = {p["label"]: p["setting"] for p in plan.get("policies", [])}
    s = saf.assign(setting=saf["label"].map(lambda x: setting.get(x, x)))

    def total(g: pd.DataFrame, col: str) -> float:
        return (
            float(pd.to_numeric(g[col], errors="coerce").fillna(0).sum()) if col in g else math.nan
        )

    def peak(g: pd.DataFrame, col: str) -> float:
        return float(pd.to_numeric(g[col], errors="coerce").max()) if col in g else math.nan

    rows = []
    for name, g in s.groupby("setting", sort=False):
        rows.append(
            [
                name,
                len(g),
                _f(peak(g, "danger"), 0),
                _f(total(g, "collisions"), 0),
                _f(total(g, "teleports"), 0),
                _f(peak(g, "ped_curb_wait_max"), 0),
                f"{_f(total(g, 'preempt_served'), 0)}/{_f(total(g, 'preempt_requests'), 0)} "
                f"({_f(total(g, 'preempt_timeouts'), 0)})",
            ]
        )
    lines = [
        "**Safety of the sensitivity runs** (`sens/safety.csv`; full 2 h runs, reported, not "
        "gating):",
        "",
    ]
    lines += _table(
        [
            "Setting",
            "Runs",
            "Danger max",
            "Collisions",
            "Teleports",
            "Curb max s",
            "Preemptions served/requests (timeouts)",
        ],
        rows,
    )
    lines.append("")
    if "collisions" in s:
        hit = s[pd.to_numeric(s["collisions"], errors="coerce").fillna(0) > 0]
        if len(hit):
            from traffic_twin.rl.sensitivity import BASELINES

            named = ", ".join(
                f"`{r.label}` seed {r.seed} ({int(r.collisions)})" for r in hit.itertuples()
            )
            text = f"Runs with a SUMO collision (count in brackets): {named}."
            base_seeds = {r.seed for r in hit.itertuples() if r.label in BASELINES}
            pol_seeds = {r.seed for r in hit.itertuples() if r.label not in BASELINES}
            shared = sorted(base_seeds & pol_seeds)
            if shared:
                text += (
                    " A baseline without RL also collides on seed "
                    + ", ".join(str(x) for x in shared)
                    + ", which points to that seed's traffic rather than to the RL policies; "
                    "it was not investigated further."
                )
            lines += [text, ""]
        else:
            lines += ["No sensitivity run has a SUMO collision.", ""]
    return lines


def sensitivity_section(data: Mapping, figs: Mapping[str, Path], docs: Path) -> list[str]:
    from traffic_twin.rl.sensitivity import BASELINES, direction, weights_row

    lines = ["## 8. Weight sensitivity (10.3)", ""]
    s = data.get("sens") or {}
    plan = data.get("sens_plan")
    kpis = s.get("kpis")
    if plan is None or kpis is None or kpis.empty:
        return lines + _missing("the weight-sensitivity tables", TASKS["sens"])
    base = plan["base"]
    rewards = plan.get("reward") or {}
    seeds = plan.get("seeds") or []
    lines += [
        f"Each setting changes one weight around the base ({str(base).upper()}: car "
        f"{weights_row(rewards[base])['w_car']:g}, bus {weights_row(rewards[base])['w_bus']:g}, "
        f"pedestrian {weights_row(rewards[base])['w_pedestrian']:g}, λ "
        f"{weights_row(rewards[base])['lambda']:g}) or moves two corners; each is trained on "
        "its own weights and picked by D72 under them. Every pick and S0/S1/S2 ran full 2 h "
        f"KPI v3 runs with drain on the {len(seeds)} sensitivity seeds "
        f"({min(seeds) if seeds else '–'}-{max(seeds) if seeds else '–'}). Means over the "
        "policies of a setting and the seeds. Descriptive: S3 keeps the base weights (D76, "
        "Q27).",
        "",
    ]
    other = sens_other_agent(data)
    if other:
        lines += [f"**Not S3's agent.** {other}", ""]
    k = kpis.set_index(["setting", "metric"])
    boot = s.get("bootstrap")
    beyond = set()
    if boot is not None and len(boot):
        beyond = {(r.setting, r.metric) for r in boot.itertuples() if bool(r.beyond_noise)}
    conv = s.get("convergence")
    rows = []
    for setting in [*dict.fromkeys([base, *(p["setting"] for p in plan["policies"])]), *BASELINES]:
        if (setting, "car/time_loss_mean") not in k.index:
            continue
        reward = rewards.get(setting) if setting not in BASELINES else None
        w = weights_row(reward) if reward else None
        cells = []
        for metric, _ in SENS_TABLE:
            if (setting, metric) not in k.index:
                cells.append("–")
                continue
            r = k.loc[(setting, metric)]
            txt = _f(r["mean"], 1)
            if reward and setting != base and math.isfinite(_num(r["vs_base_pct"])):
                mark = "*" if (setting, metric) in beyond else ""
                txt += f" ({_pct(r['vs_base_pct'])}{mark})"
            cells.append(txt)
        n = int(_num(k.loc[(setting, "car/time_loss_mean")]["n_policies"])) if reward else 0
        conv_txt = "–"
        if reward and conv is not None and len(conv):
            c = conv[conv["setting"] == setting]
            conv_txt = f"{int(c['beats_S2'].astype(bool).sum())}/{len(c)}"
        rows.append(
            [
                f"**{setting}**" if setting == base else setting,
                f"{w['w_bus']:g} / {w['w_pedestrian']:g} / {w['lambda']:g}" if w else "–",
                direction(reward, rewards[base]).replace("_", "/") if reward else "baseline",
                n if reward else "–",
                *cells,
                conv_txt,
            ]
        )
    lines += _table(
        [
            "Setting",
            "w_bus / w_ped / λ",
            "Direction",
            "Policies",
            *[name for _, name in SENS_TABLE],
            "Picks beating S2 (own weights)",
        ],
        rows,
    )
    lines += [
        "",
        "In brackets: change vs the base setting; * = the hierarchical bootstrap 95 % CI of the "
        "change excludes 0 (training seeds, then evaluation seeds; "
        f"{int(_num(boot['resamples'].iloc[0])) if boot is not None and len(boot) else 2000:,} "
        "resamples, `sens/bootstrap.csv`). "
        + resolution_text(boot, "bus/delay_mean", "Bus delay")
        + "The last column is the convergence check: a pick that does not beat S2 on its own "
        "objective is read with caution (D76).",
        "",
    ]
    lines += sens_safety(data)
    lines += _fig(figs, "pareto", "Weight sensitivity Pareto", docs)
    eq = s.get("equity")
    if eq is not None and len(eq):
        lines += [
            "**Equity vs S0** (Part 4 rule: more than 5 % worse on a primary or subgroup "
            "KPI; `ped_curb_wait_max` is capped at 88 s by the shield, so it cannot react):",
            "",
        ]
        rows = []
        for setting, g in eq.groupby("setting", sort=False):
            flagged = g[g["worse_flag"].astype(bool)]
            glob = g[g["role"] == "global"]
            gtxt = "–"
            if len(glob):
                gr = glob.iloc[0]
                gtxt = f"{_pct(gr['rel_change_pct'])} (p {_p(gr['p_value'])})"
            rows.append(
                [
                    setting,
                    len(flagged),
                    ", ".join(
                        f"`{r.user_type}.{r.kpi}` {_pct(r.rel_change_pct)}"
                        for r in flagged.itertuples()
                    )
                    or "none",
                    gtxt,
                ]
            )
        lines += _table(["Setting", "Flags", "Flagged KPIs", "Person delay vs S0"], rows)
        lines.append("")
    rew = s.get("reweight")
    if rew is not None and len(rew):
        lines += [
            "**Reweighting without retraining:** every controller's recorded totals "
            "under every setting's weights, % vs S2 under the same weights (rank 1 = "
            "lowest cost):",
            "",
        ]
        ws = list(dict.fromkeys(rew["weights"]))
        ctls = list(dict.fromkeys(rew["controller"]))
        piv = rew.set_index(["controller", "weights"])
        rows = []
        for c in ctls:
            cells = []
            for w in ws:
                if (c, w) in piv.index:
                    r = piv.loc[(c, w)]
                    t = _pct(r["vs_S2_pct"])
                    cells.append(f"**{t}**" if int(r["rank"]) == 1 else t)
                else:
                    cells.append("–")
            rows.append([c, *cells])
        lines += _table(["Controller \\ weights", *ws], rows)
        lines.append("")
    if conv is not None and len(conv):
        lines += [
            "**Convergence check** per pick (own weights; selection windows, then the "
            "sensitivity seeds):",
            "",
        ]
        rows = [
            [
                r.setting,
                r.train_seed,
                r.pick_step,
                _pct(r.pick_vs_S2_pct),
                _pct(r.late_vs_S2_pct),
                _pct(r.sens_vs_S2_pct),
            ]
            for r in conv.itertuples()
        ]
        lines += _table(
            [
                "Setting",
                "Seed",
                "Pick step",
                "Pick vs S2",
                "Late mean vs S2",
                "2 h vs S2 (sensitivity seeds)",
            ],
            rows,
        )
        lines.append("")
    lines += [
        "Ambulance travel time is reported in `sens/kpis.csv` but not interpreted (one "
        "ambulance per run, per-run SD 30-56 s). The operator table (spec 12) is "
        "`docs/figures/rl7/sensitivity.csv`.",
        "",
    ]
    return lines


def _first(row: Mapping, *names: str) -> float:
    """The first of `names` present in a stress summary row, as a number (nan if none)."""
    for n in names:
        if n in row:
            return _num(row[n])
    return math.nan


def stress_details(summary: pd.DataFrame | None, cells: Sequence[str]) -> list[str]:
    """The rest of the D78 stress panel from `stress/summary.csv` (`study.stress_summary`):
    paired tests (wins, exact Wilcoxon p), pedestrian waiting-seconds, penalty-seconds,
    interventions with their D77 split, danger and curb maximum, per cell."""
    if summary is None or summary.empty or "cell" not in summary:
        what = (
            "the stress panel's paired tests and pedestrian, penalty and intervention "
            "results (`stress/summary.csv`)"
        )
        return _missing(what, TASKS["stress"])
    by_cell = {str(r["cell"]): r for r in summary.to_dict("records")}
    order = [c for c in cells if c in by_cell] + [c for c in by_cell if c not in cells]
    tests, costs = [], []
    for c in order:
        r = by_cell[c]
        n = _num(r.get("n"))
        k = f"/{int(n)}" if math.isfinite(n) else ""
        tests.append(
            [
                f"`{c}`",
                _cell(r.get("mode")),
                f"{_f(r.get('wins_vs_S1'), 0)}{k}",
                _p(r.get("p_vs_S1")),
                f"{_f(r.get('wins_vs_S2'), 0)}{k}",
                _p(r.get("p_vs_S2")),
                _pct(r.get("ped_s_vs_S1_pct")),
                _pct(r.get("ped_s_vs_S2_pct")),
            ]
        )
        costs.append(
            [
                f"`{c}`",
                " / ".join(_f(_first(r, f"penalty_s_{lab}"), 0) for lab in ("S3", "S1", "S2")),
                " / ".join(_f(_first(r, f"interventions_{lab}"), 1) for lab in ("S3", "S1", "S2")),
                " / ".join(
                    _f(_first(r, f"split_{x}_S3"), 1)
                    for x in ("cut", "replaced", "cut_foreseeable")
                ),
                _f(r.get("danger_max"), 0),
                _f(r.get("curb_max"), 0),
                _f(r.get("preempt_timeouts_S3"), 0),
            ]
        )
    lines = [
        "Paired tests per cell (S3 against S1 and S2 on the same seeds): wins = seeds where S3's "
        "cost is lower; p = exact two-sided Wilcoxon, unadjusted. Pedestrian waiting-seconds of "
        "S3 against S1 and S2 (ratio of means).",
        "",
    ]
    lines += _table(
        [
            "Cell",
            "Mode",
            "Wins vs S1",
            "p vs S1",
            "Wins vs S2",
            "p vs S2",
            "Ped s vs S1",
            "Ped s vs S2",
        ],
        tests,
    )
    lines += [
        "",
        "Penalty-seconds and guard interventions per episode (means; S3 / S1 / S2), S3's "
        "interventions split cut / replaced / foreseeable cut (D77), and the cell's danger and "
        "pedestrian curb maximum over all four controllers, with S3's preemption timeouts.",
        "",
    ]
    lines += _table(
        [
            "Cell",
            "Penalty s (S3 / S1 / S2)",
            "Interventions (S3 / S1 / S2)",
            "S3 cut / replaced / foreseeable",
            "Danger max",
            "Curb max s",
            "S3 preemption timeouts",
        ],
        costs,
    )
    return [*lines, ""]


def stress_section(data: Mapping, figs: Mapping[str, Path], docs: Path) -> list[str]:
    lines = ["## 9. Stress panel", ""]
    fr = stress_frame(data.get("stress_summary"), data.get("stress_episodes"))
    if fr.empty:
        return lines + _missing("the stress panel", TASKS["stress"])
    gate = ((data["rl7"].get("protocol") or {}).get("stress") or {}).get("worse_than_s1_rel", 0.05)
    piv = fr.pivot_table(index="cell", columns="ref", values="pct", aggfunc="first")
    cells = list(dict.fromkeys(fr["cell"]))
    rows = []
    worse = []
    for c in cells:
        s1 = _num(piv.loc[c].get("S1")) if "S1" in piv else math.nan
        s2 = _num(piv.loc[c].get("S2")) if "S2" in piv else math.nan
        bad = math.isfinite(s1) and s1 > 100 * float(gate)
        if bad:
            worse.append(c)
        rows.append([f"`{c}`", _pct(s1), _pct(s2), "**yes**" if bad else "no"])
    lines += [
        "S3, S0, S1 and S2 on the stress seeds, each cell forcing one condition through "
        '`reset(options={"params_override": ...})` while every other draw stays the seed\'s own, '
        "so the seeds are paired (D78). Env cost change of S3 (ratio of means). Reported, not "
        "gating.",
        "",
    ]
    lines += _table(
        ["Cell", "S3 vs S1", "S3 vs S2", f"> {100 * float(gate):g} % worse than S1"], rows
    )
    lines.append("")
    lines += stress_details(data.get("stress_summary"), cells)
    if s3_without_pedestrians(data) and data.get("stress_summary") is not None:
        lines += [
            "S3's foreseeable cuts are judged from a pedestrian slack S3 does not observe "
            "(section 11).",
            "",
        ]
    lines += _fig(figs, "stress", "Stress panel", docs)
    lines += [
        (
            "Cells where S3 is more than the pre-registered margin worse than S1: "
            + ", ".join(f"`{c}`" for c in worse)
            + "."
        )
        if worse
        else "No cell is more than the pre-registered margin worse than S1.",
        "",
        "Light demand is the known failure mode: in the lead checks the final models lost to "
        "S2 by +5 to +35 % on light windows (multiplier < 0.85) against +1 to +8 % elsewhere; "
        "the ×0.6 cell measures it on fresh seeds.",
        "",
    ]
    light = [
        f"`{c}` {_pct(piv.loc[c].get('S2'))}"
        for c in ("x0_6", "holiday")
        if c in cells and "S2" in piv
    ]
    if light:
        lines += [
            "Measured on the stress seeds, S3 vs S2 (ratio of means): " + ", ".join(light) + ".",
            "",
        ]
    return lines


def version_text(cv: Any) -> str:
    """A recorded code version, with what a `-dirty` one stood for (DIRTY_NOTES)."""
    cv = str(cv)
    if not cv.endswith("-dirty"):
        return f"`{cv}`."
    note = DIRTY_NOTES.get(cv)
    return f"`{cv}`: " + (
        f"{note}." if note else "the tree differed from that commit; the record does not say how."
    )


def devtest_section(data: Mapping, figs: Mapping[str, Path], docs: Path) -> list[str]:
    from traffic_twin.rl.evaluate import paired_vs

    lines = ["## 10. First S3 results on dev-test", ""]
    cmp = data.get("devtest_compare")
    paired = data.get("paired")
    if (cmp is None or cmp.empty) and (paired is None or paired.empty):
        return lines + _missing("the dev-test look", TASKS["devtest"])
    cv = (data.get("verdict") or {}).get("code_version")
    if cv:
        lines += [f"Code version recorded by the look (`verdict.json`): {version_text(cv)}", ""]
    if cmp is not None and {"label", "seed", "cost"} <= set(cmp.columns):
        labels = list(dict.fromkeys(cmp["label"].astype(str)))
        lines += [
            f"Train-mode windows on the dev-test seeds ({cmp['seed'].nunique()} windows), "
            "looked at once (D79). Cost = episode Σc; per-window changes; p = exact two-sided "
            "Wilcoxon, unadjusted.",
            "",
        ]
        rows = []
        df = cmp.assign(label=cmp["label"].astype(str))
        for lab in labels:
            g = df[df["label"] == lab]
            v1 = paired_vs(df, lab, "S1") if lab != "S1" and "S1" in labels else {}
            v2 = paired_vs(df, lab, "S2") if lab != "S2" and "S2" in labels else {}
            rows.append(
                [
                    lab,
                    _f(g["cost"].mean(), 0),
                    _pct(v1.get("pct_mean")),
                    f"{v1.get('wins', '–')}/{v1.get('n', '–')}" if v1 else "–",
                    _p(v1.get("p")),
                    _pct(v2.get("pct_mean")),
                    _p(v2.get("p")),
                ]
            )
        lines += _table(
            ["Policy", "Cost (mean)", "vs S1", "Wins vs S1", "p vs S1", "vs S2", "p vs S2"], rows
        )
        lines.append("")
    if paired is not None and not paired.empty:
        from traffic_twin.viz.rl7 import _ref_of

        p = paired.assign(ref=_ref_of(paired))
        lines += [
            "Full 2 h KPI v3 runs (`run_scenario`, with drain) of `pitie_{fixed,actuated,rules}` "
            "and `pitie_rl_s3` on the same seeds. S3 − reference: Hodges-Lehmann paired "
            "difference with its 95 % CI, exact Wilcoxon p, Holm over the KPIs within each "
            "pair (`evaluation.compare`, as in docs/evaluation.md).",
            "",
        ]
        for ref, g in p.groupby("ref", sort=False):
            rows = []
            for r in g.itertuples(index=False):
                d = r._asdict()
                rows.append(
                    [
                        f"`{d.get('user_type')}/{d.get('kpi')}`",
                        _f(d.get("s0_mean"), 1),
                        _f(d.get("s1_mean"), 1),
                        f"{_f(d.get('hl_diff'), 1)} [{_f(d.get('hl_ci_low'), 1)}, "
                        f"{_f(d.get('hl_ci_high'), 1)}]",
                        _p(d.get("p_value")),
                        _p(d.get("p_holm")),
                        d.get("effect", "–"),
                    ]
                )
            lines += [
                f"**S3 vs {ref}** (n = "
                f"{int(_num(g['n_pairs'].max())) if 'n_pairs' in g else '–'} seeds)",
                "",
            ]
            lines += _table(
                [
                    "KPI",
                    f"{ref} mean",
                    "S3 mean",
                    "S3 − ref (HL, 95 % CI)",
                    "p",
                    "p (Holm)",
                    "Effect",
                ],
                rows,
            )
            lines.append("")
        lines += _fig(figs, "s3_forest", "S3 forest plot", docs)
    else:
        lines += _missing(
            "the full 2 h dev-test comparison (`devtest/paired.csv`)", TASKS["devtest"]
        )
    return lines


def safety_section(data: Mapping, figs: Mapping[str, Path], docs: Path) -> list[str]:
    lines = ["## 11. Safety, interventions and ambulances (10.5)", ""]
    v = data.get("verdict") or {}
    dr = data.get("devtest_runs")
    has = dr is not None and len(dr) > 0
    if not has and not v and "interventions" not in figs:
        return lines + _missing("the dev-test safety results", TASKS["devtest"])
    if has:
        rows = []
        for sc, g in dr.groupby("scenario", sort=False):
            req = pd.to_numeric(g["preempt_requests"], errors="coerce").sum()
            served = pd.to_numeric(g["preempt_served"], errors="coerce").sum()
            tos = pd.to_numeric(g["preempt_timeouts"], errors="coerce").sum()
            rows.append(
                [
                    f"`{sc}`",
                    len(g),
                    int(g["danger"].max()),
                    int(g["collisions"].max()),
                    int(g["teleports"].sum()),
                    _f(g["curb_max"].max(), 0),
                    _f(g["interventions"].mean(), 1),
                    f"{int(served)}/{int(req)} ({int(tos)})",
                ]
            )
        lines += [
            "Full 2 h dev-test runs (danger = dangerous audit findings; curb max bound "
            "90 s, A-04):",
            "",
        ]
        lines += _table(
            [
                "Scenario",
                "Runs",
                "Danger",
                "Collisions",
                "Teleports",
                "Curb max s",
                "Interventions/run",
                "Preemptions served/requests (timeouts)",
            ],
            rows,
        )
        lines.append("")
    c3 = v.get("C3") or {}
    if c3:
        viol = c3.get("violations") or []
        lines += [
            "**C3 (safety, hard):** "
            + (
                "PASS, no violation."
                if c3.get("pass") and not viol
                else "FAIL: " + "; ".join(_cell(x) for x in viol) + "."
            ),
            "",
        ]
    lines += [
        "Interventions are split by their cause (D77): **cut** = the pedestrian shield or max "
        "green ended a green the policy held (the guard's designed job); **foreseeable** cuts "
        "are those where the policy chose to stay with less pedestrian slack than one decision "
        "interval, or with max green due within it; **replaced** = every other intervention. "
        "Masks are not claimed to remove interventions (section 6).",
        "",
    ]
    if s3_without_pedestrians(data):
        lines += [
            "The pedestrian slack behind **foreseeable** is the simulator's (D77). S3 "
            f"({s3_variant(data)}) does not observe it: it sees the stage, the time in stage "
            "and each stage's red time, from which the slack can at most be inferred. For S3 a "
            "foreseeable cut therefore measures how often the shield ends a held green, not "
            "that the policy saw the slack run out.",
            "",
        ]
    lines += _fig(figs, "interventions", "Interventions split", docs)
    shares = []
    cmp = data.get("devtest_compare")
    if cmp is not None and "amb_seen_share" in cmp and "label" in cmp:
        lab = s3_label(list(dict.fromkeys(cmp["label"].astype(str))))
        g = cmp[cmp["label"].astype(str) == lab]
        if len(g):
            shares.append(
                f"S3 on the dev-test windows: mean {_f(100 * g['amb_seen_share'].mean(), 2)} %, "
                f"max {_f(100 * g['amb_seen_share'].max(), 2)} %"
            )
    for m in sorted(
        (m for m in data.get("runs") or [] if m["stage"] == "F"), key=lambda m: int(m["seed"])
    ):
        r = _pick_row(m)
        if r and "amb_seen_share" in r:
            shares.append(
                f"F s{m['seed']} pick on the selection windows "
                f"{_f(100 * _num(r['amb_seen_share']), 2)} %"
            )
    lines += [
        "Preemption stays deterministic and prioritary; decisions are skipped during it, and the "
        "ambulance features stay in the observation (D88). Share of decisions with an ambulance "
        "present: " + ("; ".join(shares) if shares else "not available yet") + " (Part 6: 0.00 %).",
        "",
    ]
    return lines


def repro_section(data: Mapping) -> list[str]:
    lines = [
        "## 12. Reproducibility",
        "",
        "Bitwise reproducibility holds only on the same machine and libraries, at fixed "
        "`n_envs`, torch threads, total steps and schedules (D82): a shorter run's prefix is a "
        "different run. Every run records its code version, `run_config_sha` with the source "
        "files' sha256, the calibration, incident-site and net file sha256, CPU and thread "
        "settings (`provenance.json`, D81). After the commit, `tasks.py reproduce7` reruns "
        "`t1` seed 0 at 400k and requires the same pick and final policy sha256, validation.csv "
        "and monitor, the same S3 Q-value golden and this page regenerated byte-identical "
        "apart from its code-version line. The full campaign (about 6 h) is not rerun.",
        "",
    ]
    runs = data.get("runs") or []
    if runs:
        versions = runs[0].get("versions") or {}
        if versions:
            lines.append("Versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()) + ".")
        cvs = sorted({str(m["code_version"]) for m in runs if m.get("code_version")})
        lines.append(
            "Code versions of the training runs: " + ", ".join(f"`{c}`" for c in cvs) + "."
        )
        dirty = [
            m
            for m in runs
            if (m.get("provenance") or {}).get("allow_dirty") or m.get("allow_dirty")
        ]
        if dirty:
            lines.append(f"{len(dirty)} runs were made with `--allow-dirty` (D80).")
        lines.append("")
    lines += evaluation_versions(data)
    lines += [
        "Only a reproduce7 result made on the code this page is built from counts here and in "
        "C4 (a clean tree whose files outside docs/ equal that commit's); a result for other "
        "code is not shown. After a reproduce7, rerun `tasks.py rl7-docs` and commit the page "
        "so that it states the result.",
        "",
    ]
    results = data.get("repro") or []
    for rep in results:
        lines += [
            f"`reproduce7` at `{rep.get('head')}`: "
            f"**{'identical' if rep.get('same') else 'NOT identical'}**"
            + (f" ({detail})." if (detail := _repro_detail(rep)) else "."),
            "",
        ]
    if not results:
        lines += _missing("the reproduce7 check", TASKS["reproduce"])
    return lines


def evaluation_versions(data: Mapping) -> list[str]:
    """The code version each evaluation step recorded (a `-dirty` one explained when
    DIRTY_NOTES knows it), and the outputs that record none."""
    lines = []
    for what, doc in (
        ("stage-T decision (`decision_T.json`)", data.get("decision_t")),
        ("S3 selection (`select/decision.json`)", data.get("select")),
        ("sensitivity runs (`sens/plan.json`)", data.get("sens_plan")),
        ("dev-test look (`devtest/verdict.json`)", data.get("verdict")),
    ):
        if isinstance(doc, Mapping) and doc.get("code_version"):
            lines.append(f"- {what}: {version_text(doc['code_version'])}")
    silent = []
    for what, df in (
        ("holdout (`holdout/`)", data.get("holdout_summary")),
        ("stress panel (`stress/`)", data.get("stress_summary")),
    ):
        if df is None:
            continue
        if "code_version" in df:
            vs = sorted({str(x) for x in df["code_version"].dropna()})
            lines.append(f"- {what}: " + ", ".join(f"`{x}`" for x in vs) + ".")
        else:
            silent.append(what)
    if silent:
        lines.append(f"- {' and '.join(silent)}: the outputs record no code version.")
    return ["Code versions recorded by the evaluation steps:", "", *lines, ""] if lines else []


def _repro_detail(rep: Mapping) -> str:
    """`tasks.py reproduce7` items ({item: [differences]}, empty = same), or the older
    {check: bool} form."""
    items = rep.get("items")
    if isinstance(items, Mapping):
        text = "; ".join(
            f"{k}: same" if not d else f"{k}: DIFFERENT ({_cell(list(d)[:3])})"
            for k, d in items.items()
        )
        # a reproduce7 page comparison masks the code-version lines: never write one here
        return re.sub(r"code[ -]version", "code line", text, flags=re.IGNORECASE)
    checks = rep.get("checks")
    if isinstance(checks, Mapping):
        return ", ".join(f"{k} {'same' if ok else 'DIFFERENT'}" for k, ok in checks.items())
    return ""


def obs_limit(data: Mapping) -> str:
    """The limitation bullet on what S3 observes (its manifest's dropped groups)."""
    if s3_without_pedestrians(data):
        n = (s3_obs(data) or ((), None))[1]
        feats = f", {n} features" if n is not None else ""
        return (
            f"- S3 ({s3_variant(data)}{feats}) reads every signal-controlled lane from the "
            "simulator (ASSUMED ideal detection) and the ambulance by V2I, and no pedestrian "
            "count, curb wait or shield slack: the pedestrian shield alone keeps the curb-wait "
            "bound. Part of a gain over S1 may be an information advantage."
        )
    return (
        "- The observation reads every signal-controlled lane and the pedestrian curb waits from "
        "the simulator (ASSUMED ideal detection) and the ambulance by V2I; part of a gain over S1 "
        "may be an information advantage."
    )


def limits_section(data: Mapping) -> list[str]:
    seeds = ((data["rl7"].get("protocol") or {}).get("devtest") or {}).get("seeds")
    k = _range_n(seeds)
    n = str(k) if k else "–"
    min_p = 2 / 2**k if k else math.nan
    other = sens_other_agent(data)
    return [
        "## 13. Limitations and the Part 8 handoff",
        "",
        "- Provisional (D90): every number rests on the ASSUMED 123 s signal program, the "
        "calibration and the day-type ratio profiles. It is simulated, not measured. Retraining "
        "after the field visit (3-5 Nov 2026) reruns `part7` (about 7 h).",
        f"- The interim 10.6 verdict uses n = {n} dev-test seeds (smallest exact two-sided "
        f"Wilcoxon p {min_p:.3f}): it is exploratory. The binding comparison is A-06 on "
        "seeds 1-10 in Part 8, with a rule Part 8 registers in `configs/evaluation.yaml` "
        "before any seed 1-10 RL run (D68).",
        "- Training days differ from test days as distinct seeds (each seed a different simulated "
        "day); every day type appears in training and test, a documented deviation from 10.4 "
        "(D87). No held-out day type was run.",
        obs_limit(data),
        "- The weight sensitivity is descriptive; optimiser noise between training seeds is "
        "mixed with the weight effects (the convergence check shows where). S3 keeps the base "
        "weights. λ 100 is a candidate for a later retrain, decided by the user through a dated "
        "change before Part 8, never after looking at dev-test or seeds 1-10 (Q27)."
        + (f" {other}" if other else ""),
        "- The env episodes (validations, holdout, stage-2 selection, stress panel) record the "
        "danger audit and the curb maximum but not SUMO collisions or teleports, so the "
        "collision clause of the campaign STOP rule (plan section 9, step 2) could not be "
        "checked on them. Collisions and teleports are measured only in the full "
        "`run_scenario` runs: the dev-test runs (section 11, C3) and the sensitivity runs "
        "(section 8).",
        "- Part 8: A-06 on seeds 1-10 (S3 against S0/S1/S2), S4 = S3's policy with preemption "
        "off (out of distribution: the policy never decided with an ambulance present), and "
        "every `configs/evaluation.yaml` change.",
        "",
    ]


def build_docs(
    config: Path = RL7_CONFIG,
    docs: Path | None = None,
    figures: Path | None = None,
    study: Path | None = None,
) -> Path:
    """Write `docs` (default rl7.yaml outputs.docs) and its figures (default
    outputs.figures) from the artifacts under `study` (default outputs.study); returns
    the page path."""
    from traffic_twin.sim.results import code_version
    from traffic_twin.viz import rl7 as viz

    data = collect(config, study)
    outs = data["rl7"].get("outputs") or {}
    docs = Path(docs or outs.get("docs") or "docs/rl_training.md")
    figures = Path(figures or outs.get("figures") or "docs/figures/rl7")
    figs = viz.make_figures(figure_data(data), figures)
    lines = [
        TITLE,
        "",
        "Generated by `python -m traffic_twin.rl.report7` (spec S7, 10.3-10.6; decisions "
        "D68-D90). Do not edit by hand: rerun `tasks.py rl7-docs`. Every result is simulated "
        "and provisional until the field visit (ASSUMED signal program). `docs/rl.md` stays "
        "the Part 6 record.",
        "",
        f"{CODE_LINE}: `{code_version()}`.",
        "",
    ]
    lines += summary_section(data)
    lines += prereg_section(data)
    lines += protocol_section(data)
    lines += stage_t_section(data, figs, docs)
    lines += curves_section(data, figs, docs)
    lines += variants_section(data)
    lines += stage_f_section(data)
    lines += sensitivity_section(data, figs, docs)
    lines += stress_section(data, figs, docs)
    lines += devtest_section(data, figs, docs)
    lines += safety_section(data, figs, docs)
    lines += repro_section(data)
    lines += limits_section(data)
    text = "\n".join(lines).rstrip() + "\n"
    text = re.sub(r"\n{3,}", "\n\n", text)
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(text, encoding="utf-8", newline="\n")
    return docs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, default=RL7_CONFIG, help="Part 7 config (rl7.yaml)")
    p.add_argument("--docs", type=Path, help="default: rl7.yaml outputs.docs")
    p.add_argument("--figures", type=Path, help="default: rl7.yaml outputs.figures")
    p.add_argument("--study", type=Path, help="artifacts root (default: outputs.study)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        print(build_docs(args.config, args.docs, args.figures, args.study).as_posix())
    except ConfigError as e:
        raise SystemExit(f"error: {e}") from None
    return 0


if __name__ == "__main__":
    sys.exit(main())

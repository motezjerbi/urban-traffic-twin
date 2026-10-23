"""Part 8 final outputs (D93-D100), called from `evaluation.evaluate`: the A-06 report
(S3 against S1, S2 and S0 on `all/person_delay_mean`, D94), the spec 10.6 replication on
seeds 1-10 (D95, reported, not gating), the final comparison table (D98), the queue
series behind `/series`, the masked digest of the evaluated KPI files (ENF-01) and the
check of the frozen S3 inputs (D100).

Writes, through `evaluate.write_outputs`, `<out>/{final_table.csv, a06.json, series.csv,
digest.json}` (paths.EVAL_FINAL_TABLE, EVAL_A06, EVAL_SERIES, EVAL_DIGEST).

- **A-06** (`a06_report`): per comparison d = S3 − reference per seed, exact paired
  Wilcoxon (`a06.alternative`), Hodges-Lehmann estimate and 95 % CI, Holm across the
  comparisons. The status is procedural: PASS when every comparison has `n_required`
  paired seeds, every S3 run is current (its config_hash, with the pinned model
  sha256), the D100 input pins match, the family shares one demand and A-03 passes;
  INVALID otherwise, with the reasons. It is never FAIL because of a direction. The
  outcome per comparison is fixed in advance: `better` (Holm p < alpha, HL < 0),
  `worse` (Holm p < alpha, HL > 0), else `none`.
- **10.6** (`replication_10_6`): Part 7's D79 C1-C3 repeated against S1 (C1 HL < 0 and
  unadjusted p < alpha; C2 no listed KPI more than `equity_rel` worse, `compare.
  worse_than` on the means; C3 safety in every S3 run and no more preemption time-outs
  than S2); C4 is cited from Part 7's verdict.json. Seeds 1-10 confirm the interim
  FALLBACK unless C1-C3 all pass; the main solution stays `main_solution` either way.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from traffic_twin.config import ConfigError, load_scenario, text_sha256
from traffic_twin.evaluation.compare import _sense, better_if, paired, unit, worse_than
from traffic_twin.evaluation.runs import run_path
from traffic_twin.evaluation.stats import hodges_lehmann, holm, wilcoxon

if TYPE_CHECKING:
    from traffic_twin.evaluation.evaluate import Family

REPO_ROOT = Path(__file__).resolve().parents[3]
# final_table.csv columns (section 4.2); extra rows use section safety|activity|equity|occupancy
FINAL_TABLE_COLUMNS = (
    "section", "user", "kpi", "label", "scenario", "ablation", "n", "mean", "sd",
    "rel_vs_s0_pct", "unit", "better_if",
)  # fmt: skip
A06_STATUSES = ("PASS", "INVALID")  # procedural (D94): never FAIL on direction
OUTCOME_KEYS = ("better", "worse", "none")
SERIES_COLUMNS = (
    "family", "scenario", "label", "metric", "approach", "t_s", "mean", "p10", "p90",
    "n_seeds",
)  # fmt: skip
FINAL_EXTRAS = (
    "a04", "interventions_per_run", "preemption_served", "equity_flags_vs_S0",
    "person_delay_occupancy_sensitivity",
)  # fmt: skip
REL_TOL = 1e-9  # float slack on the pre-registered `<=` edges, as Part 7 (rl.study)
RUN_ID_MASK = b"<run_id>"  # digest: the run's own run_id (a fresh uuid per run) masked
REGISTRATION_ORDERS = ("before", "after", "n/a")
TEST_KEYS = (
    "mean_ref", "sd_ref", "mean_s3", "sd_s3", "hl", "ci_low", "ci_high", "p",
    "min_attainable_p",
)  # fmt: skip  # `paired_test` values (None for a comparison that cannot be tested)


# -- helpers ---------------------------------------------------------------------------
def repo_file(path: str | Path) -> Path:
    """A repo-relative path of a config: as given when it exists from the working
    directory (the repo root in every task), else under this checkout."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def split_key(key: str) -> tuple[str, str]:
    """'all/person_delay_mean' -> ('all', 'person_delay_mean')."""
    user, sep, kpi = str(key).partition("/")
    if not sep or not user or not kpi:
        raise ConfigError(f"{key!r} is not a 'user/kpi' key")
    return user, kpi


def _f(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _pf(p: float | None) -> str:
    if p is None or not math.isfinite(p):
        return "–"
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.1e}"


def _sf(x: float | None, nd: int = 1, signed: bool = False) -> str:
    if x is None or not math.isfinite(x):
        return "–"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def values(fam: Family, label: str, user: str, kpi: str) -> pd.Series:
    """seed -> finite value of (user, kpi) of one scenario of `fam`."""
    k = fam.kpis.get(label)
    if k is None or k.empty:
        return pd.Series(dtype=float)
    g = k[(k["user_type"] == user) & (k["kpi"] == kpi)]
    v = pd.to_numeric(g["value"], errors="coerce").to_numpy(float)
    s = pd.Series(v, index=g["seed"].astype(int).to_numpy()).sort_index()
    return s[np.isfinite(s.to_numpy(float))]


def paired_diff(fam: Family, ref: str, cand: str, user: str, kpi: str) -> pd.DataFrame:
    """seed, ref, cand, d (= cand − ref): the finite pairs of (user, kpi)."""
    pr = paired(fam.kpis[ref], fam.kpis[cand])
    g = pr[(pr["user_type"] == user) & (pr["kpi"] == kpi)].sort_values("seed")
    g = g[np.isfinite(g["diff"].to_numpy(float))]
    return pd.DataFrame(
        {"seed": g["seed"].astype(int), "ref": g["s0"], "cand": g["s1"], "d": g["diff"]}
    ).reset_index(drop=True)


def _sd(x: pd.Series) -> float | None:
    return _f(x.std(ddof=1)) if len(x) > 1 else None


def paired_test(
    fam: Family, ref: str, cand: str, user: str, kpi: str, alternative: str = "two-sided"
) -> dict:
    """The exact paired Wilcoxon of cand − ref on (user, kpi) with means, SDs and the HL
    estimate and 95 % CI (None where undefined)."""
    g = paired_diff(fam, ref, cand, user, kpi)
    d = g["d"].to_numpy(float)
    w = wilcoxon(d, alternative=alternative) if len(d) else None
    hl, lo, hi = hodges_lehmann(d)
    return {
        "n": int(len(d)),
        "seeds": [int(s) for s in g["seed"]],
        "mean_ref": _f(g["ref"].mean()) if len(d) else None,
        "sd_ref": _sd(g["ref"]),
        "mean_s3": _f(g["cand"].mean()) if len(d) else None,
        "sd_s3": _sd(g["cand"]),
        "hl": _f(hl),
        "ci_low": _f(lo),
        "ci_high": _f(hi),
        "p": _f(w.pvalue) if w else None,
        "min_attainable_p": _f(w.min_attainable_p) if w else None,
    }


# -- D100: the frozen S3 inputs --------------------------------------------------------
def model_inputs(ecfg: dict) -> dict[str, dict]:
    """path -> {pinned, actual, ok}: the text sha8 of every file pinned in
    `a06.model.inputs_text_sha8` (provenance.json)."""
    pins = (((ecfg.get("a06") or {}).get("model") or {}).get("inputs_text_sha8")) or {}
    out = {}
    for path, sha8 in pins.items():
        actual = text_sha256(repo_file(path))
        pinned = str(sha8)  # YAML reads an all-digit pin (configs/rl7.yaml) as an int
        out[str(path)] = {
            "pinned": pinned,
            "actual": actual[:8] if actual else None,
            "ok": bool(actual) and actual[:8] == pinned,
        }
    return out


def check_model_inputs(ecfg: dict) -> list[str]:
    """D100: the text sha8 of every file pinned in `a06.model.inputs_text_sha8` against
    the files on disk; one message per mismatch or missing file ([] when all match).
    The `rl.config` of the pinned S3 config (`a06.model.sha256_from`) must be pinned too.
    No `a06` block: nothing is pinned, [] (a Part 4 config)."""
    a06 = ecfg.get("a06")
    if not a06:
        return []
    model = a06.get("model") or {}
    pins = model.get("inputs_text_sha8") or {}
    out = []
    if not pins:
        out.append("a06.model.inputs_text_sha8 pins no file")
    for path, rec in model_inputs(ecfg).items():
        if rec["actual"] is None:
            out.append(f"{path}: missing (pinned text sha8 {rec['pinned']})")
        elif not rec["ok"]:
            out.append(f"{path}: text sha8 {rec['actual']} != pinned {rec['pinned']}")
    src = model.get("sha256_from")
    if src:
        try:
            rl = load_scenario(repo_file(src)).get("rl") or {}
        except (ConfigError, FileNotFoundError) as e:
            return [*out, f"{src}: cannot load ({e})"]
        chain = rl.get("config")
        pinned = {Path(p).as_posix() for p in pins}
        if chain and Path(chain).as_posix() not in pinned:
            out.append(f"{src}: rl.config {chain} is not pinned in a06.model.inputs_text_sha8")
        if not rl.get("model_sha256"):
            out.append(f"{src}: no rl.model_sha256 to pin the S3 model")
    return out


def rl_scenario_problems(families: dict[str, Family], ecfg: dict) -> list[str]:
    """D100 for every evaluated `rl` scenario (S3, S4 and their probe versions): its
    `rl.config` is one of the pinned files and its `rl.model_sha256` is the pinned
    model's (`a06.model.sha256_from`). [] without an `a06` block."""
    model = (ecfg.get("a06") or {}).get("model") or {}
    if not model:
        return []
    pinned = {Path(p).as_posix() for p in model.get("inputs_text_sha8") or {}}
    sha = pinned_model_sha256(ecfg)
    out = []
    for name, fam in families.items():
        for label, m in fam.members.items():
            if m.controller != "rl":
                continue
            rl = m.cfg.get("rl") or {}
            chain = rl.get("config")
            if not chain or Path(chain).as_posix() not in pinned:
                out.append(f"{name}/{label} ({m.scenario}): rl.config {chain} is not pinned")
            if sha and rl.get("model_sha256") != sha:
                out.append(
                    f"{name}/{label} ({m.scenario}): rl.model_sha256 is not the pinned model"
                )
    return out


def pinned_model_sha256(ecfg: dict) -> str | None:
    """`rl.model_sha256` of `a06.model.sha256_from` (configs/pitie_rl_s3.yaml)."""
    src = ((ecfg.get("a06") or {}).get("model") or {}).get("sha256_from")
    if not src:
        return None
    try:
        return (load_scenario(repo_file(src)).get("rl") or {}).get("model_sha256")
    except (ConfigError, FileNotFoundError):
        return None


# -- registration (D92) ----------------------------------------------------------------
def _stamp(text) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None
    return t.replace(microsecond=0) if t.tzinfo is not None else None


def first_rl_started(families: dict[str, Family]) -> str | None:
    """The earliest run.json `started_at` of an `rl` run of the evaluated families (the
    evaluation seeds), as written."""
    stamps = []
    for fam in families.values():
        for m in fam.members.values():
            if m.controller != "rl" or m.runs is None:
                continue
            for meta in m.runs.meta.values():
                t = _stamp(meta.get("started_at"))
                if t is not None:
                    stamps.append((t, str(meta["started_at"])))
    return min(stamps)[1] if stamps else None


def registration_order(committed_at: str | None, first_started: str | None) -> str:
    """`before` when the registration commit is strictly earlier than the first S3 run
    at second resolution, `after` otherwise, `n/a` when either stamp is missing (an
    equal second cannot show the order: `after`)."""
    c, s = _stamp(committed_at), _stamp(first_started)
    if c is None or s is None:
        return "n/a"
    return "before" if c < s else "after"


def registration_record(families: dict[str, Family], commit: dict | None) -> dict:
    """{commit, committed_at, first_s3_started_at, order} (D92)."""
    first = first_rl_started(families)
    commit = commit or {}
    return {
        "commit": commit.get("commit"),
        "committed_at": commit.get("committed_at"),
        "first_s3_started_at": first,
        "order": registration_order(commit.get("committed_at"), first),
    }


# -- A-06 ------------------------------------------------------------------------------
def outcome_of(p_holm: float | None, hl: float | None, alpha: float) -> str:
    """`better` / `worse` / `none` (D94): Holm p < alpha and the sign of HL."""
    if p_holm is None or hl is None or not p_holm < alpha or hl == 0:
        return "none"
    return "better" if hl < 0 else "worse"


def comparison_sentence(c: dict, cand: str, kpi_key: str, alpha: float, words: dict) -> str:
    """One generated sentence per A-06 comparison (the report quotes it, D113)."""
    unit_ = unit(kpi_key.split("/", 1)[-1])
    u = f" {unit_}" if unit_ else ""
    if not c["n"]:
        return f"{cand} vs {c['reference']}: no paired seeds, not tested."
    return (
        f"{cand} vs {c['reference']} ({c['controller']}): {kpi_key} "
        f"{_sf(c['mean_ref'])} → {_sf(c['mean_s3'])}{u} (n = {c['n']}), HL "
        f"{_sf(c['hl'], signed=True)}{u} [95 % CI {_sf(c['ci_low'], signed=True)}, "
        f"{_sf(c['ci_high'], signed=True)}], exact {c['alternative']} p {_pf(c['p'])}, Holm "
        f"p {_pf(c['p_holm'])} (α {alpha:g}): {words[c['outcome']]}."
    )


def a06_validity(fam: Family, cfg: dict, ecfg: dict, a03: dict) -> list[str]:
    """Why A-06 is INVALID beyond the sample sizes: the candidate is not a current `rl`
    run of the pinned model, the family does not share one demand, A-03 does not pass."""
    cand = cfg["candidate"]
    problems = []
    m = fam.members.get(cand)
    if m is None:
        return [f"{cand} is not a scenario of family {fam.name}"]
    if m.controller != "rl":
        problems.append(f"{cand} ({m.scenario}) has controller {m.controller}, not rl")
    sha = (m.cfg.get("rl") or {}).get("model_sha256")
    pinned = pinned_model_sha256(ecfg)
    if not sha:
        problems.append(f"{cand} ({m.scenario}) pins no rl.model_sha256")
    elif pinned and sha != pinned:
        problems.append(f"{cand} model_sha256 {sha[:12]} != pinned {str(pinned)[:12]}")
    expected = m.expected["config_hash"]
    metas = m.runs.meta if m.runs is not None else {}
    stale = sorted(s for s, x in metas.items() if x.get("config_hash") != expected)
    if stale:
        problems.append(f"{cand} runs of seeds {stale} are not of the current config")
    ran = sorted(
        s
        for s, x in metas.items()
        if (x.get("control") or {}).get("model_sha256") not in (None, sha)
    )
    if ran:
        problems.append(f"{cand} runs of seeds {ran} report another model_sha256")
    demands = {x.expected["demand_hash"] for x in fam.members.values()}
    if len(demands) > 1 or not fam.demand_hash:
        problems.append(f"family {fam.name} does not share one demand_hash")
    if not a03 or a03.get("n") != a03.get("of"):
        problems.append(f"A-03 is {(a03 or {}).get('text', 'not evaluated')}, not complete")
    return problems


def a06_report(
    families: dict[str, Family],
    ecfg: dict,
    a03: dict,
    *,
    registration: dict | None = None,
    model_problems: list[str] | None = None,
    verdict_path: Path | None = None,
) -> dict:
    """a06.json (section 4.2): status PASS|INVALID with its problems, one comparison per
    `a06.comparisons` (n, means, SDs, HL and 95 % CI, raw and Holm p, outcome,
    sentence), the model identity and the registration record, plus the 10.6
    replication and the S3-vs-reference equity tables. `registration`: the record of
    `registration_record` (default: no commit); `model_problems`: `check_model_inputs`
    (default: computed); `verdict_path`: Part 7's verdict.json (default the
    `replication_10_6.c4` path)."""
    cfg = ecfg["a06"]
    alpha = float(cfg.get("alpha", 0.05))
    alt = str(cfg.get("alternative", "two-sided"))
    n_req = int(cfg.get("n_required", 10))
    user, kpi = str(cfg["user"]), str(cfg["kpi"])
    key = f"{user}/{kpi}"
    cand = str(cfg["candidate"])
    words = {k: str((cfg.get("outcome") or {}).get(k, k)) for k in OUTCOME_KEYS}
    fam = families.get(cfg["family"])
    problems: list[str] = []
    comparisons: list[dict] = []
    if fam is None:
        problems.append(f"family {cfg['family']} was not evaluated")
    for ref in cfg["comparisons"]:
        c = {"reference": ref, "controller": None, "alternative": alt}
        if fam is None or ref not in fam.members or cand not in fam.members:
            if fam is not None:
                problems.append(f"{cand} vs {ref}: not both scenarios of family {fam.name}")
            c.update(dict.fromkeys(TEST_KEYS), n=0, seeds=[])
        else:
            c["controller"] = fam.members[ref].controller
            c.update(paired_test(fam, ref, cand, user, kpi, alt))
            expected = fam.expected_seeds(ref, cand)
            if c["n"] != n_req or len(expected) != n_req:
                problems.append(
                    f"{cand} vs {ref}: {c['n']} paired seeds of {len(expected)} evaluated, "
                    f"{n_req} required"
                )
        comparisons.append(c)
    ps = [c["p"] for c in comparisons]
    finite = [i for i, p in enumerate(ps) if p is not None]
    adj = holm([ps[i] for i in finite]) if finite else []
    for c in comparisons:
        c["p_holm"] = None
    for i, a in zip(finite, adj, strict=True):
        comparisons[i]["p_holm"] = _f(a)
    for c in comparisons:
        c["outcome"] = outcome_of(c["p_holm"], c["hl"], alpha)
        c["outcome_text"] = words[c["outcome"]]
        c["sentence"] = comparison_sentence(c, cand, key, alpha, words)
    if fam is not None:
        problems += a06_validity(fam, cfg, ecfg, a03)
    if model_problems is None:
        model_problems = check_model_inputs(ecfg)
    problems += [f"D100 frozen S3 input: {p}" for p in model_problems]
    m = fam.members.get(cand) if fam is not None else None
    first_meta = next(iter(m.runs.meta.values()), {}) if m is not None and m.runs else {}
    model = {
        "model_sha256": ((m.cfg.get("rl") or {}).get("model_sha256")) if m else None,
        "pinned_sha256": pinned_model_sha256(ecfg),
        "model_version": (first_meta.get("control") or {}).get("model_version"),
        "config_hash": m.expected["config_hash"] if m else None,
        "inputs_ok": not model_problems,
        "inputs": model_inputs(ecfg),
    }
    primary = next((c for c in comparisons if c["reference"] == cfg.get("primary")), None)
    rep = cfg.get("replication_10_6")
    replication = None
    if rep:
        path = verdict_path if verdict_path is not None else repo_file(rep.get("c4") or "")
        replication = replication_10_6(families, ecfg, Path(path))
    return {
        "status": "PASS" if not problems else "INVALID",
        "problems": problems,
        "family": cfg["family"],
        "candidate": cand,
        "kpi": key,
        "test": cfg.get("test"),
        "alternative": alt,
        "alpha": alpha,
        "n_required": n_req,
        "primary": cfg.get("primary"),
        "holm": cfg.get("holm"),
        "comparisons": comparisons,
        "primary_sentence": primary["sentence"] if primary else None,
        "model": model,
        "registration": registration or registration_record(families, None),
        "replication_10_6": replication,
        "equity_vs": equity_vs(families, ecfg),
    }


# -- 10.6 replication (D95) ------------------------------------------------------------
def _danger_keys(ecfg: dict) -> tuple[str, ...]:
    from traffic_twin.evaluation.evaluate import resolve_keys

    return resolve_keys(ecfg["a04"]["danger_keys"])


def c3_violations(fam: Family, cand: str, ecfg: dict, c3: dict) -> list[str]:
    """D79 C3 on the candidate's runs: danger audit 0, collisions 0, teleports 0, curb
    maximum <= `curb_max_s`, every preemption request served or dropped as a time-out,
    and, over the seeds, no more preemption time-outs than `timeouts_le`. A count a run
    does not report is a violation (never read as 0)."""
    m = fam.members.get(cand)
    if m is None or m.runs is None or not m.runs.meta:
        return [f"no {cand} run"]
    danger = _danger_keys(ecfg)
    curb_max = float(c3.get("curb_max_s", 90.0))
    curbs = values(fam, cand, "pedestrian", "ped_curb_wait_max")
    out = []
    for seed, meta in sorted(m.runs.meta.items()):
        why = []
        audit = (meta.get("safety") or {}).get("audit") or {}
        lacking = [k for k in danger if audit.get(k) is None]
        if lacking:
            why.append(f"danger audit lacks {', '.join(lacking)}")
        total = int(sum(_num(audit.get(k)) for k in danger))
        if total:
            why.append(f"danger {total}")
        health = meta.get("health") or {}
        for col in ("collisions", "teleports"):
            v = health.get(col)
            if v is None:
                why.append(f"{col} not reported")
            elif _num(v):
                why.append(f"{col} {int(_num(v))}")
        curb = _f(curbs.get(seed))
        if curb is None:
            why.append("no ped_curb_wait_max")
        elif curb > curb_max + REL_TOL:
            why.append(f"curb max {curb:g} s > {curb_max:g} s")
        if c3.get("preemption_all_served", True):
            pre = meta.get("preemption") or {}
            req, served = int(_num(pre.get("requests"))), int(_num(pre.get("served")))
            timeouts = int(_num(pre.get("timeouts")))
            if served + timeouts < req:
                why.append(f"preemption: {req - served - timeouts} of {req} requests unresolved")
        if why:
            out.append(f"seed {seed}: " + ", ".join(why))
    ref = c3.get("timeouts_le")
    if ref:
        other = fam.members.get(ref)
        if other is None or other.runs is None:
            out.append(f"no {ref} runs to compare the preemption time-outs with")
        else:
            seeds = set(m.runs.meta)

            def total(mem) -> int:
                return int(
                    sum(
                        _num((x.get("preemption") or {}).get("timeouts"))
                        for s, x in mem.runs.meta.items()
                        if s in seeds
                    )
                )

            t_c, t_r = total(m), total(other)
            if t_c > t_r:
                out.append(f"preemption timeouts {t_c} > {ref}'s {t_r} on the same seeds")
    return out


def cite_c4(path: Path) -> dict:
    """C4 cited from Part 7: `pass` / `fail` / `pending`, or `not available` when the
    file is absent (data/ is not versioned). As Part 7's pages do, the frozen
    `devtest/verdict.json` is completed with the newest reproduce7 result
    (`rl.study.final_verdict`, `<rl7>/_reproduce/<HEAD>/reproduce7.json`), never read
    alone."""
    p = Path(path)
    try:
        v = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"cited": "not available", "source": p.as_posix(), "verdict": None}
    from traffic_twin.rl.study import final_verdict, latest_reproduce

    v = final_verdict(v, latest_reproduce(p.parent.parent))
    c4 = v.get("C4") or {}
    state = {True: "pass", False: "fail"}.get(c4.get("pass"), "pending")
    return {
        "cited": state,
        "source": p.as_posix(),
        "verdict": v.get("verdict"),
        "runs_below_s1": c4.get("runs_below_s1"),
        "of": c4.get("of"),
        "repro": (c4.get("repro") or {}).get("status"),
    }


def replication_10_6(families: dict[str, Family], ecfg: dict, verdict_path: Path) -> dict:
    """The Part 7 D79 C1-C3 checks repeated on seeds 1-10 against S1 (C4 cited from
    `verdict_path`, data/rl7/devtest/verdict.json): reported, never gating (D95)."""
    a06 = ecfg["a06"]
    rc = a06["replication_10_6"]
    cand, ref = str(a06["candidate"]), str(rc["reference"])
    main_solution = str(rc.get("main_solution", "S2"))
    fam = families.get(a06["family"])
    have = fam is not None and cand in fam.members and ref in fam.members
    if not have:  # no verdict from absent data (e.g. `evaluate --family ambulance_probe`)
        return replication_not_evaluated(rc, a06, families, verdict_path)
    # C1
    c1cfg = rc.get("c1") or {}
    alpha = float(c1cfg.get("alpha", 0.05))
    user, kpi = split_key(c1cfg.get("kpi", f"{a06['user']}/{a06['kpi']}"))
    t = paired_test(fam, ref, cand, user, kpi, str(c1cfg.get("alternative", "two-sided")))
    ok = t["hl"] is not None and t["p"] is not None and t["hl"] < 0 and t["p"] < alpha
    c1 = {"kpi": f"{user}/{kpi}", **{k: t[k] for k in ("n", "mean_ref", "mean_s3", "hl")},
          "hl_ci": [t["ci_low"], t["ci_high"]], "p": t["p"], "alpha": alpha,
          "pass": bool(ok)}  # fmt: skip
    # C2
    c2cfg = rc.get("c2") or {}
    pct = 100.0 * float(c2cfg.get("equity_rel", 0.05))
    rows = []
    for k in c2cfg.get("kpis") or []:
        u, name = split_key(k)
        a, b = values(fam, ref, u, name), values(fam, cand, u, name)
        if a.empty or b.empty:
            rows.append({"kpi": k, "mean_s1": None, "mean_s3": None, "rel_pct": None,
                         "worse": True, "why": "missing"})  # fmt: skip
            continue
        m0, m1 = float(a.mean()), float(b.mean())
        rel = 100.0 * (m1 / m0 - 1.0) if m0 else math.nan
        worse = worse_than(m0, m1, rel, _sense(name), pct + REL_TOL)
        rows.append({"kpi": k, "mean_s1": m0, "mean_s3": m1, "rel_pct": _f(rel),
                     "worse": bool(worse)})  # fmt: skip
    c2 = {"limit_pct": pct, "reference": ref, "rows": rows,
          "pass": not any(r["worse"] for r in rows)}  # fmt: skip
    # C3
    c3cfg = rc.get("c3") or {}
    violations = c3_violations(fam, cand, ecfg, c3cfg)
    c3 = {"violations": violations, "pass": not violations,
          "curb_max_s": float(c3cfg.get("curb_max_s", 90.0)),
          "timeouts_le": c3cfg.get("timeouts_le")}  # fmt: skip
    c4 = cite_c4(verdict_path)
    all_pass = bool(c1["pass"] and c2["pass"] and c3["pass"])
    failed = [n for n, c in (("C1", c1), ("C2", c2), ("C3", c3)) if not c["pass"]]
    seeds = fam.expected_seeds(ref, cand)
    where = _seeds_words(seeds)
    if all_pass:
        text = (
            f"{where} do not confirm the interim FALLBACK: C1-C3 pass against {ref} (C4 cited "
            f"from Part 7: {c4['cited']}). The main solution stays {main_solution} (fixed at "
            "week 7, spec 15); RL is a strong candidate for future work, not adopted."
        )
    else:
        reasons = []
        if "C2" in failed:
            worse = [r["kpi"] for r in rows if r["worse"]]
            reasons.append(f"C2 fails on {', '.join(worse)}")
        if "C1" in failed:
            reasons.append("C1 fails")
        if "C3" in failed:
            reasons.append(f"C3 fails ({len(violations)} violation(s))")
        text = (
            f"{where} confirm the interim FALLBACK ({'; '.join(reasons)}). The main solution "
            f"stays {main_solution}."
        )
    return {
        "status": "evaluated",
        "reference": ref,
        "seeds": [int(s) for s in seeds],
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "confirms_fallback": not all_pass,
        "main_solution": main_solution,
        "text": text,
    }


NOT_EVALUATED = "not evaluated"


def replication_not_evaluated(
    rc: dict, a06: dict, families: dict[str, Family], verdict_path: Path
) -> dict:
    """The 10.6 record when the a06 family or its candidate / reference label was not
    evaluated: status "not evaluated", every check and `confirms_fallback` None, and a
    sentence that states no result (C4 is still cited: it does not depend on seeds)."""
    cand, ref = str(a06["candidate"]), str(rc["reference"])
    fam = families.get(a06["family"])
    if fam is None:
        why = f"family {a06['family']} was not evaluated"
    else:
        lacking = [x for x in (cand, ref) if x not in fam.members]
        why = f"family {fam.name} has no {' or '.join(lacking)}"
    c1cfg, c3cfg = rc.get("c1") or {}, rc.get("c3") or {}
    return {
        "status": NOT_EVALUATED,
        "why": why,
        "reference": ref,
        "seeds": [],
        "c1": {"kpi": c1cfg.get("kpi", f"{a06['user']}/{a06['kpi']}"), "n": 0, "hl": None,
               "hl_ci": [None, None], "p": None, "alpha": float(c1cfg.get("alpha", 0.05)),
               "pass": None},
        "c2": {"limit_pct": 100.0 * float((rc.get("c2") or {}).get("equity_rel", 0.05)),
               "reference": ref, "rows": [], "pass": None},
        "c3": {"pass": None, "curb_max_s": float(c3cfg.get("curb_max_s", 90.0)),
               "timeouts_le": c3cfg.get("timeouts_le")},
        "c4": cite_c4(verdict_path),
        "confirms_fallback": None,
        "main_solution": str(rc.get("main_solution", "S2")),
        "text": f"Spec 10.6 check not evaluated ({why}): no result is stated.",
    }  # fmt: skip


def _seeds_words(seeds: list[int]) -> str:
    if not seeds:
        return "The evaluation seeds"
    if seeds == list(range(seeds[0], seeds[-1] + 1)) and len(seeds) > 2:
        return f"Seeds {seeds[0]}-{seeds[-1]}"
    return f"Seeds {', '.join(map(str, seeds))}"


# -- equity of S3 against S1 and S2 (D96) ----------------------------------------------
def equity_vs(families: dict[str, Family], ecfg: dict) -> dict[str, list[dict]]:
    """reference -> rows {user, kpi, mean_ref, mean_s3, rel_pct, p, worse, primary}: the 4
    per-user primary KPIs of `equity.primary` plus the C2 KPIs, S3 against each
    reference of `a06.equity_vs`, flagged by `equity.pct` (rates in pp), p two-sided
    unadjusted."""
    a06 = ecfg["a06"]
    fam = families.get(a06["family"])
    cand = str(a06["candidate"])
    refs = [str(r) for r in a06.get("equity_vs") or []]
    if fam is None or cand not in fam.members:
        return {r: [] for r in refs}
    eq = ecfg["equity"]
    pct = float(eq["pct"])
    rates = {k: float(v) for k, v in (eq.get("rates_pp") or {}).items()}
    keys = [(u, k) for u, k in (eq.get("primary") or {}).items()]
    c2 = (a06.get("replication_10_6") or {}).get("c2") or {}
    keys += [split_key(k) for k in c2.get("kpis") or []]
    keys = list(dict.fromkeys(keys))
    primary = set((eq.get("primary") or {}).items())
    out = {}
    for ref in refs:
        rows = []
        if ref not in fam.members:
            out[ref] = rows
            continue
        for u, k in keys:
            t = paired_test(fam, ref, cand, u, k)
            if not t["n"]:
                rows.append({"user": u, "kpi": k, "n": 0, "mean_ref": None, "mean_s3": None,
                             "rel_pct": None, "pp": None, "p": None, "worse": None,
                             "primary": (u, k) in primary})  # fmt: skip
                continue
            m0, m1 = t["mean_ref"], t["mean_s3"]
            rel = 100.0 * (m1 / m0 - 1.0) if m0 else math.nan
            sense = _sense(k)
            pp = 100.0 * (m1 - m0) if k in rates else None
            if pp is not None:
                worse = bool(sense and pp * sense < -rates[k])
            else:
                worse = worse_than(m0, m1, rel, sense, pct)
            rows.append({"user": u, "kpi": k, "n": t["n"], "mean_ref": m0, "mean_s3": m1,
                         "rel_pct": _f(rel), "pp": pp, "p": t["p"], "worse": bool(worse),
                         "primary": (u, k) in primary})  # fmt: skip
        out[ref] = rows
    return out


# -- final comparison table (D98) ------------------------------------------------------
def _row(section, user, kpi, label, scenario, ablation, n, mean, sd, rel, unit_, better):
    return dict(zip(FINAL_TABLE_COLUMNS,
                    (section, user, kpi, label, scenario, bool(ablation), int(n), _f(mean),
                     _f(sd), _f(rel), unit_, better), strict=True))  # fmt: skip


def _kpi_rows(fam: Family, section: str, keys: list[str], labels: list[str], abl: set) -> list:
    base = fam.baseline
    rows = []
    for key in keys:
        user, kpi = split_key(key)
        b = values(fam, base, user, kpi)
        m0 = float(b.mean()) if len(b) else math.nan
        for label in labels:
            v = values(fam, label, user, kpi)
            mean = float(v.mean()) if len(v) else math.nan
            rel = 100.0 * (mean / m0 - 1.0) if math.isfinite(m0) and m0 != 0 else math.nan
            rows.append(_row(section, user, kpi, label, fam.members[label].scenario,
                             label in abl, len(v), mean, _sd(v), rel, unit(kpi),
                             better_if(kpi)))  # fmt: skip
    return rows


def final_table(
    families: dict[str, Family],
    ecfg: dict,
    a04: pd.DataFrame,
    activity: pd.DataFrame,
    *,
    equity: pd.DataFrame | None = None,
    hidden: list[dict] | None = None,
) -> pd.DataFrame:
    """final_table.csv: one row per (section, user, kpi, label) of `final_table.rows`
    (section `main`) and `probe_rows` (section `probe`, the ambulance-probe family) for
    the columns and ablations of evaluation.yaml, plus the `extra` rows
    (FINAL_TABLE_COLUMNS): `safety` (A-04 per scenario, preemption counts), `activity`
    (guard interventions per run), `equity` (flags vs S0 and hidden degradation, from
    `equity` / `hidden`), `occupancy` (person delay at the bus sensitivity values). A
    registered label without runs has no row (the API says "not evaluated")."""
    ft = ecfg.get("final_table") or {}
    fam = families.get(ft.get("family"))
    if fam is None:
        return pd.DataFrame(columns=list(FINAL_TABLE_COLUMNS))
    abl = {str(x) for x in ft.get("ablations") or []}
    wanted = [str(x) for x in [*(ft.get("columns") or []), *(ft.get("ablations") or [])]]
    labels = [x for x in wanted if x in fam.members]
    rows = _kpi_rows(fam, "main", list(ft.get("rows") or []), labels, abl)
    probes = [f for n, f in families.items() if n != fam.name]
    for pf in probes:
        plabels = [x for x in wanted if x in pf.members]
        rows += _kpi_rows(pf, "probe", list(ft.get("probe_rows") or []), plabels, abl)
    extra = set(ft.get("extra") or [])
    for f in [fam, *probes]:
        flabels = [x for x in wanted if x in f.members]
        suffix = "" if f is fam else "__probe"
        rows += _extra_rows(f, flabels, abl, extra, suffix, a04, activity, equity, hidden)
    return pd.DataFrame(rows, columns=list(FINAL_TABLE_COLUMNS))


def _extra_rows(fam, labels, abl, extra, suffix, a04, activity, equity, hidden) -> list:
    rows = []

    def add(section, user, kpi, label, n, mean, unit_="", better="n/a", sd=None, rel=None):
        rows.append(_row(section, user, kpi + suffix, label, fam.members[label].scenario,
                         label in abl, n, mean, sd, rel, unit_, better))  # fmt: skip

    act = activity[activity["family"] == fam.name] if len(activity) else activity
    act = act.set_index("scenario") if len(act) else act
    for label in labels:
        scen = fam.members[label].scenario
        if "a04" in extra and len(a04) and "family" in a04:
            g = a04[(a04["family"] == fam.name) & (a04["scenario"] == scen)]
            if len(g):
                add("safety", "all", "a04_pass", label, len(g), float(g["pass"].all()), "",
                    "higher")  # fmt: skip
                add("safety", "all", "danger_total", label, len(g),
                    float(g["danger_total"].fillna(0).sum()), "", "lower")  # fmt: skip
                add("safety", "all", "collisions", label, len(g),
                    float(g["collisions"].fillna(0).sum()), "", "lower")  # fmt: skip
                add("safety", "pedestrian", "ped_curb_wait_max_worst", label, len(g),
                    float(g["ped_curb_wait_max"].max()), "s", "lower")  # fmt: skip
        has_act = len(act) and label in act.index
        if not has_act:
            continue
        a = act.loc[label]
        runs = int(_num(a.get("runs")))
        if "interventions_per_run" in extra and runs:
            keys = [c for c in act.columns if c.startswith("guard.")]
            total = sum(_num(a.get(k)) for k in keys)
            add("activity", "all", "interventions_per_run", label, runs, total / runs, "",
                "lower")  # fmt: skip
            for k in keys:
                add("activity", "all", f"interventions_per_run__{k.split('.', 1)[1]}", label,
                    runs, _num(a.get(k)) / runs, "", "lower")  # fmt: skip
        if "preemption_served" in extra:
            for k in ("requests", "served", "timeouts"):
                add("safety", "ambulance", f"preemption_{k}", label, runs,
                    _num(a.get(f"preemption.{k}")), "",
                    {"served": "higher", "timeouts": "lower"}.get(k, "n/a"))  # fmt: skip
    if "equity_flags_vs_S0" in extra and equity is not None and len(equity):
        e = equity[equity["family"] == fam.name]
        flags = {h["scenario"]: h for h in hidden or [] if h["family"] == fam.name}
        for label in labels:
            g = e[e["scenario"] == label]
            if g.empty:
                continue
            key = g["worse_flag"].astype(bool) & (g["primary"] | g["subgroup"]).astype(bool)
            add("equity", "all", "equity_flags_vs_S0", label, len(g), float(key.sum()), "",
                "lower")  # fmt: skip
            add("equity", "all", "worse_kpis_vs_S0", label, len(g),
                float(g["worse_flag"].astype(bool).sum()), "", "lower")  # fmt: skip
            h = flags.get(label)
            if h is not None:
                add("equity", "all", "hidden_degradation", label, 1,
                    float(bool(h["hidden_degradation"])), "", "lower")  # fmt: skip
    if "person_delay_occupancy_sensitivity" in extra and not suffix:
        base = fam.kpis.get(fam.baseline)
        names = sorted(
            {k for k in base["kpi"] if str(k).startswith("person_delay_mean__occ_")}
            if base is not None
            else set()
        )
        for name in names:
            sect = _kpi_rows(fam, "occupancy", [f"all/{name}"], labels, abl)
            rows += sect
    return rows


# -- series (D98) ----------------------------------------------------------------------
def quantile_names(quantiles) -> list[str]:
    """[0.1, 0.9] -> ['p10', 'p90']."""
    return [f"p{100 * float(q):g}" for q in quantiles]


def series(family: Family, ecfg: dict) -> pd.DataFrame:
    """series.csv rows of one family (SERIES_COLUMNS): `part8.series.metric` per
    approach in `step_s` bins from each run's queue.parquet, mean and quantiles over the
    seeds. Per run and bin, an approach's value is its mean over the samples of the bin;
    `all` is the sum over the approaches of the run. A run without queue.parquet is left
    out (`n_seeds` says how many seeds a point has)."""
    sc = (ecfg.get("part8") or {}).get("series") or {}
    qs = [float(q) for q in sc.get("quantiles") or [0.1, 0.9]]
    qnames = quantile_names(qs)
    columns = ["family", "scenario", "label", "metric", "approach", "t_s", "mean", *qnames,
               "n_seeds"]  # fmt: skip
    if not sc:
        return pd.DataFrame(columns=columns)
    step = float(sc.get("step_s", 60))
    metric = str(sc.get("metric", "halting_veh"))
    approaches = [str(a) for a in sc.get("approaches") or ["all"]]
    frames = []
    for label, m in family.members.items():
        seeds = family.seeds or (m.runs.seeds if m.runs is not None else [])
        per = []
        for seed in seeds:
            p = run_path(m.results, m.scenario, seed) / "queue.parquet"
            if not p.is_file():
                continue
            q = pd.read_parquet(p, columns=["t", "approach", metric])
            q = q.assign(t_s=(q["t"] // step) * step, approach=q["approach"].astype(str))
            g = q.groupby(["t_s", "approach"], sort=True)[metric].mean().reset_index()
            tot = g.groupby("t_s", sort=True)[metric].sum().reset_index().assign(approach="all")
            per.append(pd.concat([g, tot], ignore_index=True).assign(seed=int(seed)))
        if not per:
            continue
        df = pd.concat(per, ignore_index=True)
        df = df[df["approach"].isin(approaches)]
        grouped = df.groupby(["approach", "t_s"], sort=True)[metric]
        agg = grouped.mean().rename("mean").to_frame()
        for q, name in zip(qs, qnames, strict=True):
            agg[name] = grouped.quantile(q)
        agg["n_seeds"] = df.groupby(["approach", "t_s"], sort=True)["seed"].nunique()
        agg = agg.reset_index()
        agg["order"] = agg["approach"].map({a: i for i, a in enumerate(approaches)})
        agg = agg.sort_values(["order", "t_s"]).drop(columns="order")
        if float(step).is_integer():
            agg["t_s"] = agg["t_s"].astype(int)
        frames.append(
            agg.assign(family=family.name, scenario=m.scenario, label=label, metric=metric)
        )
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


# -- digest (ENF-01) -------------------------------------------------------------------
def masked_sha256(kpi_csv: Path, run_id: str | None) -> str:
    """sha256 of a kpi_run.csv with the run's own run_id replaced by `<run_id>` (the
    masking of the Part 8 capture and of `rebaseline` comparisons)."""
    data = Path(kpi_csv).read_bytes()
    if run_id:
        data = data.replace(str(run_id).encode(), RUN_ID_MASK)
    return hashlib.sha256(data).hexdigest()


def digest(families: dict[str, Family]) -> dict:
    """digest.json: sha256 of every evaluated kpi_run.csv with the run's own run_id
    masked (ENF-01 record): {algorithm, file, mask, n, combined, runs: {<scenario>_seed<n>:
    sha256}}; `combined` hashes the sorted `name sha` lines."""
    runs = {}
    for fam in families.values():
        for m in fam.members.values():
            if m.runs is None:
                continue
            for seed, meta in sorted(m.runs.meta.items()):
                d = run_path(m.results, m.scenario, seed)
                runs[d.name] = masked_sha256(d / "kpi_run.csv", meta.get("run_id"))
    runs = dict(sorted(runs.items()))
    combined = hashlib.sha256(
        "".join(f"{k} {v}\n" for k, v in runs.items()).encode("utf-8")
    ).hexdigest()
    return {
        "algorithm": "sha256",
        "file": "kpi_run.csv",
        "mask": "run_id -> <run_id>",
        "n": len(runs),
        "combined": combined,
        "runs": runs,
    }

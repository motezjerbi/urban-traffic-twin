"""Validation report (spec 13.4, A-02): flow, signal and queue tables for one scenario
over the evaluation seeds, plus safety, capacity and pedestrian summaries, rendered to
docs/validation.md.

    python -m traffic_twin.validation.report --scenario configs/pitie_fixed.yaml
        [--run-missing] [--docs docs/validation.md] [--compare-csv P]

Inputs: the runs `<outputs.dir>/<scenario>_seed<n>/` (kpi_run.csv, run.json, trip.parquet
and the measurement outputs detector / queue / signal .parquet), the versioned sensor
extract (configs/validation.yaml: reference.file), the calendar (configs/data.yaml), the
field file and the calibration log. Writes `data/results/validation/<scenario>/{
flow_validation.csv, signal_validation.csv, queue_validation.csv, summary.json}` and the
figures `validation_flows.png` (+ `validation_queues.png` once queues are measured).

Every run used must have been made with the CURRENT config (run.json `config_hash` and
`demand_hash` equal those of the scenario config): `--run-missing` runs absent seeds and
reruns stale ones, and stale runs are refused. The S0 vs S1 section and the totals of the
other scenario come from the comparison CSV (default: `<results>/comparison/
<scenario>__vs__*.csv`) only when its provenance file shows it was computed from the
current runs of both configs; otherwise the page says it is out of date.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from traffic_twin.config import ConfigError, config_hash, config_sources, load_config, load_scenario
from traffic_twin.evaluation.compare import provenance_path
from traffic_twin.evaluation.runs import (
    HASH_KEYS,
    Runs,
    StaleRunsError,
    check_current,
    expected_hashes,
    load_runs,
    run_missing,
    short,
    stale_seeds,
)
from traffic_twin.paths import DATA_CONFIG, PITIE_FIXED_CONFIG, VALIDATION_CONFIG
from traffic_twin.validation import clock_str, load_validation, slot_bounds
from traffic_twin.validation.calibration_loop import LOG_FILE as CALIBRATION_LOG
from traffic_twin.validation.field import FieldData, load_field, summary_line
from traffic_twin.validation.flows import a02_summary, flow_table, geh_summary, sim_flows
from traffic_twin.validation.queues import queue_table
from traffic_twin.validation.reference import read_hours, real_flows
from traffic_twin.validation.signal import (
    cycle_onsets,
    head_stages,
    model_durations,
    ped_links_of,
    ped_only_stage,
    program_items,
    read_program,
    signal_table,
    sim_items,
)

log = logging.getLogger(__name__)

DOCS_FILE = Path("docs/validation.md")
FIGURES_DIR = Path("docs/figures/pitie")
RUN_TABLES = ("kpis", "detectors", "queues", "signal", "trips")
SECTIONS = (
    "1. Runs, code version, seeds and sensor days",
    "2. Pre-registered thresholds",
    "3. Data and assumptions",
    "4. Flows (A-02)",
    "5. Signal timing",
    "6. Queues",
    "7. Safety and capacity",
    "8. Pedestrians",
    "9. S0 vs S1",
    "10. Limitations and update after the field visit",
)
HEALTH_KEYS = (
    "collisions", "teleports", "emergency_stops", "emergency_braking", "vehicles_waiting_end",
    "drain_timeout",
)  # fmt: skip
CAR_ARM = r"^car_([A-Za-z]+)_"  # car trip ids: car_<arm>_<destination>_<slot>.<n>
CAPACITY_COLUMNS = [
    "scenario", "vehicles", "trips", "mean_s", "worst_seed_mean_s", "worst_seed",
    "worst_15min_s", "max_s", "seeds_over", "status",
]  # fmt: skip
CAPACITY_DEFAULT_S = 30.0


# -- tables ----------------------------------------------------------------------------
@dataclass
class Validation:
    flows: pd.DataFrame
    signal: pd.DataFrame
    queues: pd.DataFrame
    real: pd.DataFrame
    a02: dict
    geh: dict


def measurement_lanes(cfg: dict) -> dict[str, list[str]]:
    return {s["id"]: list(s["lanes"]) for s in cfg.get("measurement", {}).get("stations", [])}


def build(
    cfg: dict, vcfg: dict, cal: dict, field_data: FieldData, runs: Runs, hours: pd.DataFrame
) -> Validation:
    """All validation tables of one scenario."""
    th = vcfg["thresholds"]
    lanes = measurement_lanes(cfg)
    unknown = [s["station"] for s in vcfg["stations"] if s["station"] not in lanes]
    if unknown:
        raise ConfigError(f"validation stations {unknown} are not in measurement.stations")
    real = real_flows(hours, vcfg["stations"], vcfg["slots"], cal, vcfg)
    det = runs.detectors
    sim = (
        sim_flows(det, vcfg["slots"], float(cfg["begin_s"]))
        if len(det)
        else pd.DataFrame(columns=["seed", "station", "slot", "flow_veh_h"])
    )
    flows = flow_table(real, sim, vcfg["stations"], vcfg["slots"], th, lanes)

    net = cfg["network"]
    groups = {k: list(v) for k, v in net.get("signal_groups", {}).items()}
    peds = ped_links_of(cfg)
    ref_group = net.get("reference_group")
    model, ped_only, tags = {}, None, {}
    try:
        states, durations = read_program(net["net_file"], net["tls_id"])
        durations = model_durations(durations, net["stages"], cfg.get("fixed", {}).get("green_s"))
        model = program_items(states, durations, groups, peds, net["stages"], ref_group)
        tags = head_stages(states, durations, groups, net["stages"], ref_group)
        ped_only = ped_only_stage(states, net["stages"], peds)
    except (FileNotFoundError, OSError) as e:
        log.warning("no model program (%s): model values left empty", e)
    sig_runs = runs.signal
    sims = (
        sim_items(sig_runs, groups, ref_group, float(cfg["duration_s"]), tags)
        if len(sig_runs)
        else pd.DataFrame(columns=["seed", "item", "value"])
    )
    signal = signal_table(cfg, field_data, sims, model, th, ped_only)
    ref_links = groups.get(ref_group, [])
    onsets = cycle_onsets(sig_runs, ref_links) if len(sig_runs) and ref_links else {}
    queues = queue_table(runs.queues, field_data, cfg, th, onsets)
    return Validation(
        flows, signal, queues, real, a02_summary(flows), geh_summary(flows, vcfg["slots"], th)
    )


# -- comparison ------------------------------------------------------------------------
@dataclass
class Comparison:
    """The S0 vs S1 table of a comparison CSV, shown only when it is current."""

    csv: Path
    other: str
    table: pd.DataFrame | None = None
    runs: Runs | None = None
    problem: str | None = None  # why it is out of date (then table and runs are None)


def load_comparison(
    csv: Path, cfg: dict, hashes: dict[str, str], results: Path, seeds: list[int]
) -> Comparison:
    """The comparison in `csv` if it was computed from the current runs: its provenance
    file (written by evaluation.compare) must name the current S0 hashes and seeds, the
    other scenario's config must still have the hashes it was computed with, and that
    scenario's runs on disk must still be those runs."""
    other = Path(csv).stem.split("__vs__")[-1]
    comp = Comparison(Path(csv), other)
    prov_file = provenance_path(csv)
    if not prov_file.is_file():
        comp.problem = f"no provenance file `{prov_file.name}` (made by an older version)"
        return comp
    prov = json.loads(prov_file.read_text(encoding="utf-8"))
    base, side = prov.get("baseline") or {}, prov.get("scenario") or {}
    problems = []
    if base.get("scenario") != cfg["scenario"]:
        problems.append(f"its S0 is {base.get('scenario')!r}, not {cfg['scenario']!r}")
    for k in HASH_KEYS:
        if base.get(k) != hashes[k]:
            problems.append(f"S0 {k} {short(base.get(k))} != current {short(hashes[k])}")
    if side.get("demand_hash") != hashes["demand_hash"]:
        problems.append(f"{other} demand_hash {short(side.get('demand_hash'))} is not S0's")
    if [int(s) for s in prov.get("seeds") or []] != list(seeds):
        problems.append(f"seeds {prov.get('seeds')} are not the evaluation seeds {list(seeds)}")
    try:
        now = expected_hashes(load_scenario(side.get("config") or ""))
        changed = [k for k in HASH_KEYS if side.get(k) != now[k]]
        if changed:
            problems.append(f"the {other} config changed since ({', '.join(changed)})")
    except (ConfigError, OSError) as e:
        problems.append(f"the {other} config {side.get('config')!r} cannot be read ({e})")
    runs = load_runs(results, other, list(seeds), tables=("kpis", "trips"))
    stale = stale_seeds(runs, {k: side.get(k) for k in HASH_KEYS})
    if stale:
        problems.append(f"{other} runs changed since (seeds {sorted(stale)})")
    missing = sorted(set(seeds) - set(runs.seeds))
    if missing:
        problems.append(f"{other} has no run for seeds {missing}")
    if problems:
        comp.problem = "; ".join(problems)
        return comp
    comp.table, comp.runs = pd.read_csv(csv), runs
    return comp


# -- summaries -------------------------------------------------------------------------
def safety_totals(runs: Runs) -> dict:
    """Health, guard interventions and signal-audit counts summed over the runs (n/a when
    no run reports a key)."""
    out: dict = {"runs": len(runs.meta)}
    health: Counter = Counter()
    seen: set[str] = set()
    inter: Counter = Counter()
    audit: Counter = Counter()
    for m in runs.meta.values():
        h = m.get("health") or {}
        for k in HEALTH_KEYS:
            if k in h:
                seen.add(k)
                health[k] += int(h[k])
        s = m.get("safety") or {}
        inter.update({k: int(v) for k, v in (s.get("interventions") or {}).items()})
        audit.update({k: int(v) for k, v in (s.get("audit") or {}).items()})
    out["health"] = {k: health[k] if k in seen else None for k in HEALTH_KEYS}
    delays = [
        m["health"]["depart_delay_max"]
        for m in runs.meta.values()
        if (m.get("health") or {}).get("depart_delay_max") is not None
    ]
    out["depart_delay_max"] = max(delays) if delays else None
    out["interventions"] = dict(inter)
    out["audit"] = dict(audit)
    return out


def insertion_delays(runs: Runs, begin_s: float, threshold: float) -> pd.DataFrame:
    """Depart (insertion) delay per car arm (from the trip id), for buses and for the
    ambulance, over the evaluation runs: the mean over all trips, the worst seed mean,
    the worst 15-min mean (trips grouped by planned departure, depart - departDelay: a
    backlog that grows within the window), the max, and for car arms the seeds whose
    mean exceeds `threshold` (capacity-limited on that seed)."""
    t = runs.trips
    if t.empty or "depart_delay_s" not in t:
        return pd.DataFrame(columns=CAPACITY_COLUMNS)
    v = t[t["type"] != "pedestrian"].dropna(subset=["depart_delay_s"]).copy()
    arm = v["id"].astype(str).str.extract(CAR_ARM, expand=False).fillna("?")
    v["vehicles"] = np.where(v["type"] == "car", "car " + arm, v["type"])
    v["bin"] = ((v["depart_s"] - v["depart_delay_s"] - begin_s) // 900).astype(int)
    rows = []
    for who, g in v.groupby("vehicles"):
        per_seed = g.groupby("seed")["depart_delay_s"].mean()
        car = str(who).startswith("car ")
        over = [int(s) for s, m in per_seed.items() if m > threshold] if car else []
        rows.append(
            {
                "scenario": runs.scenario,
                "vehicles": who,
                "trips": len(g),
                "mean_s": float(g["depart_delay_s"].mean()),
                "worst_seed_mean_s": float(per_seed.max()),
                "worst_seed": int(per_seed.idxmax()),
                "worst_15min_s": float(g.groupby(["seed", "bin"])["depart_delay_s"].mean().max()),
                "max_s": float(g["depart_delay_s"].max()),
                "seeds_over": over,
                "status": ("capacity-limited" if over else "ok") if car else "info",
            }
        )
    order = {"car": 0, "bus": 1, "ambulance": 2}
    rows.sort(key=lambda r: (order.get(str(r["vehicles"]).split()[0], 3), str(r["vehicles"])))
    return pd.DataFrame(rows, columns=CAPACITY_COLUMNS)


def ped_summary(runs: Runs, target: float) -> dict:
    k = runs.kpis
    out = {"target_s": target}
    for kpi in ("ped_wait_single_max", "ped_wait_max"):
        v = k[(k["user_type"] == "pedestrian") & (k["kpi"] == kpi)]["value"] if len(k) else []
        out[kpi] = (
            {"mean": float(np.mean(v)), "max": float(np.max(v)), "n": len(v)} if len(v) else None
        )
    single = out["ped_wait_single_max"]
    out["within_target"] = None if single is None else bool(single["max"] <= target)
    return out


def calibration_status(path: Path, cfg: dict) -> dict | None:
    """The calibration log of this scenario: its status, the recommended arm_scale,
    whether that is the configured one, and whether its last iteration ran the current
    config (config_hash). None when there is no log for this scenario."""
    path = Path(path)
    if not path.is_file():
        return None
    doc = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("calibration") or {}
    if doc.get("scenario_name") != cfg["scenario"]:
        return None
    configured = {a: float(v) for a, v in (cfg.get("demand", {}).get("arm_scale") or {}).items()}
    rec = {a: float(v) for a, v in (doc.get("recommended_arm_scale") or {}).items()}
    rows = doc.get("rows") or []
    applied = all(abs(v - configured.get(a, 1.0)) <= 5e-4 for a, v in rec.items())
    return {
        "file": path.as_posix(),
        "status": doc.get("status"),
        "seeds": doc.get("seeds"),
        "frozen": doc.get("frozen") or {},
        "recommended": rec,
        "applied": applied,
        "current": bool(rows) and rows[-1].get("config_hash") == config_hash(cfg),
    }


def _git_versions(runs: Runs) -> str:
    return ", ".join(sorted({str(m.get("code_version", "unknown")) for m in runs.meta.values()}))


def _run_dates(runs: Runs) -> tuple[str, str]:
    """First and last date (YYYY-MM-DD) the runs were started."""
    dates = sorted(str(m["started_at"])[:10] for m in runs.meta.values() if m.get("started_at"))
    return (dates[0], dates[-1]) if dates else ("n/a", "n/a")


def summary(
    cfg: dict,
    vcfg: dict,
    val: Validation,
    runs: Runs,
    hashes: dict[str, str],
    comparison: Comparison | None,
    fd: FieldData,
    capacity: pd.DataFrame,
    calibration: dict | None,
) -> dict:
    target = float(cfg.get("safety", {}).get("ped_max_wait_s", 90))
    first, last = _run_dates(runs)
    prereg = str(vcfg["preregistered"])
    others = [comparison.runs] if comparison and comparison.runs else []
    return {
        "scenario": cfg["scenario"],
        "controller": cfg["controller"],
        "seeds": runs.seeds,
        "code_version": _git_versions(runs),
        "date": last,
        "first_run_date": first,
        "config_hash": hashes["config_hash"],
        "demand_hash": hashes["demand_hash"],
        "preregistered": prereg,
        "preregistered_before_runs": None if first == "n/a" else prereg <= first,
        "changes": [{k: str(v) for k, v in c.items()} for c in vcfg.get("changes") or []],
        "a02": val.a02,
        "geh": val.geh,
        "signal_status": dict(Counter(val.signal["status"])),
        "queue_status": dict(Counter(val.queues["status"])),
        "field": summary_line(fd),
        "field_file": fd.path.as_posix(),
        "calibration": calibration,
        "comparison": None
        if comparison is None
        else {
            "csv": comparison.csv.as_posix(),
            "other": comparison.other,
            "current": comparison.problem is None,
            "problem": comparison.problem,
        },
        "safety": {r.scenario: safety_totals(r) for r in [runs, *others]},
        "capacity": capacity.to_dict(orient="records"),
        "pedestrians": {r.scenario: ped_summary(r, target) for r in [runs, *others]},
    }


# -- markdown --------------------------------------------------------------------------
def _f(x, nd: int = 0) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if isinstance(x, bool | np.bool_):
        return "yes" if x else "no"
    if isinstance(x, int | np.integer):
        return str(int(x))
    if isinstance(x, float | np.floating):
        return f"{x:.{nd}f}"
    return str(x)


def _table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += [
        "| " + " | ".join(_f(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows
    ]
    return out


def _seeds_text(seeds: list[int]) -> str:
    if not seeds:
        return "none"
    if seeds == list(range(seeds[0], seeds[-1] + 1)) and len(seeds) > 2:
        return f"{seeds[0]}-{seeds[-1]} ({len(seeds)} runs)"
    return f"{', '.join(map(str, seeds))} ({len(seeds)} runs)"


def _pm(mean, sd, nd: int = 0) -> str:
    return f"{_f(mean, nd)} ± {_f(sd, nd)}" if mean is not None and pd.notna(mean) else "–"


STATUS_WORDS = (
    "PASS / FAIL (judged), ok / off (information rows), NOT_MEASURED (no field value yet), "
    "ASSUMED (model input not measured on site), NO_DATA, NO_SENSOR, NOT_IN_SIM, "
    "OUT_OF_WINDOW, INFO (descriptive, not judged)"
)
FLOW_HEAD = [
    "Station",
    "Arc",
    "Slot",
    "Role",
    "Real veh/h",
    "Sim veh/h",
    "Error %",
    "GEH",
    "Status",
]
SAFETY_HEAD = [
    "Scenario", "Runs", "Collisions", "Teleports", "Emergency stops", "Emergency braking",
    "Vehicles left", "Drain timeouts", "Max depart delay (s)", "Guard interventions",
    "Audit violations",
]  # fmt: skip
PED_HEAD = [
    "Scenario", "Single wait max, mean of seeds (s)", "Worst seed (s)", "Target (s)",
    "Within target", "Walk time loss max (s)",
]  # fmt: skip


def _runs_section(cfg: dict, vcfg: dict, real: pd.DataFrame, summ: dict, path: Path) -> list:
    begin, duration = float(cfg["begin_s"]), float(cfg["duration_s"])
    window = [s["name"] for s in vcfg["slots"] if slot_bounds(s, begin) == (0.0, duration)]
    at = real[real["slot"].isin(window)] if window else real
    arc_days = [
        f"{r['iu_ac']}: {_f(r['n_days'])} days ({r['first_day']}..{r['last_day']})"
        if r["first_day"]
        else f"{r['iu_ac']}: no day"
        for _, r in at[at["status"] != "NO_SENSOR"].drop_duplicates("iu_ac").iterrows()
    ]
    sources = ", ".join(p.as_posix() for p in config_sources(path))
    ref = vcfg["reference"]
    hashes = f"`{short(summ['config_hash'])}` / `{short(summ['demand_hash'])}` (every run)"
    scenario = f"`{cfg['scenario']}` (controller `{cfg['controller']}`), from {sources}"
    first, last = summ["first_run_date"], summ["date"]
    return _table(
        ["Item", "Value"],
        [
            ["Scenario", scenario],
            ["Code version (git)", summ["code_version"] or "n/a"],
            ["Date of the runs", first if first == last else f"{first} to {last}"],
            ["Evaluation seeds", _seeds_text(summ["seeds"])],
            ["Window", f"{clock_str(begin)}-{clock_str(begin + duration)}"],
            ["config_hash / demand_hash", hashes],
            ["Sensor reference", f"`{ref['file']}`, school-term Tue-Thu, MAD k {ref['mad_k']}"],
            ["Sensor days per arc", "; ".join(arc_days) or "n/a"],
            ["Field file", f"`{summ['field_file']}`: {summ['field']}"],
        ],
    )


def _thresholds_section(vcfg: dict, summ: dict) -> list:
    th, seeds = vcfg["thresholds"], vcfg["seeds"]
    a02 = ", ".join(s["name"] for s in vcfg["slots"] if s.get("role") == "A-02")
    ctl = ", ".join(s["name"] for s in vcfg["slots"] if s.get("role") != "A-02")
    rel, share = f"{th['flow_rel']:.0%}", f"{th['geh_share']:.0%}"
    flows = (
        f"every inbound arm within ±{rel} of the real typical-day mean in each slot {a02} "
        "(sim = mean of the evaluation seeds, E1 loops)"
    )
    geh = f"GEH < {th['geh']:g} on at least {share} of distinct counts (groups counted once)"
    signal = (
        f"cycle ±{th['cycle_s']:g} s, green ±{th['green_s']:g} s, yellow ±{th['yellow_s']:g} s, "
        f"all-red ±{th['all_red_s']:g} s; S1 descriptive"
    )
    queues = (
        f"observed inside the simulated p10-p90, or within {th['queue_veh']:g} vehicles of the "
        f"median; sample ±{th['queue_window_s']:g} s around the instant, all seeds"
    )
    days = f"at least {vcfg['reference']['min_days']} typical days per arc, else NO_DATA"
    evaluation = _seeds_text([int(s) for s in seeds["evaluation"]])
    calibration = ", ".join(map(str, seeds["calibration"]))
    stats = (
        f"exact Wilcoxon signed-rank, Holm over {len(vcfg['compare']['headline'])} headline "
        f"KPIs, α = {th['alpha']:g}; equity flag above {th['equity_pct']:g} % worse"
    )
    rows = [
        ["A-02 flows", flows],
        ["GEH (reported)", geh],
        ["Control slots", f"{ctl} (information)"],
        ["Signal (S0 only)", signal],
        ["Queues", queues],
        ["Reference days", days],
        ["Seeds", f"evaluation {evaluation}; calibration {calibration} (never overlap)"],
        ["S0 vs S1", stats],
    ]
    if "capacity_depart_delay_s" in th:
        rows.append(
            [
                "Capacity (flag, no verdict)",
                f"a car arm with a mean depart delay above {th['capacity_depart_delay_s']:g} s "
                "over a run is capacity-limited on that seed (section 7)",
            ]
        )
    prereg, first = summ["preregistered"], summ["first_run_date"]
    lines = [
        f"Pre-registered on {prereg} in `configs/validation.yaml`, before any calibrated "
        "result (spec 17). A threshold changes only with a dated entry in `changes` of that "
        "file, listed below (this page is regenerated, so a note written here would be lost).",
    ]
    if summ["preregistered_before_runs"] is False:
        lines += [
            "",
            f"**Warning: the pre-registration date ({prereg}) is later than the first "
            f"evaluation run ({first}).** The thresholds were not fixed before these results.",
        ]
    lines += ["", *_table(["Check", "Rule"], rows), ""]
    changes = vcfg.get("changes") or []
    if changes:
        lines += ["Changes since pre-registration:", ""]
        lines += [f"- {c['date']}: {c['what']}. Why: {c['why']}." for c in changes]
    else:
        lines.append("No change since pre-registration.")
    return [s.rstrip() for s in lines]


def _calibration_text(cfg: dict, calib: dict | None) -> str:
    scale = cfg.get("demand", {}).get("arm_scale") or {}
    text = ", ".join(f"{a} {v}" for a, v in scale.items()) or "none (1.0)"
    head = f"Demand scaling (`demand.arm_scale`): {text}."
    if calib is None:
        return f"{head} No calibration log of this scenario was found."
    seeds = ", ".join(map(str, calib.get("seeds") or []))
    log_text = f"Calibration log `{calib['file']}`: {calib['status']} on seeds {seeds}"
    if calib["status"] == "converged" and calib["applied"]:
        same = (
            "its last iteration ran this config (config_hash)"
            if calib["current"]
            else (
                "**but its last iteration ran another config (config_hash): rerun "
                "`tasks.py calibrate`**"
            )
        )
        return f"{head} {log_text}; recommended scale = configured scale; {same}."
    rec = ", ".join(f"{a} {v}" for a, v in calib["recommended"].items())
    frozen = "; ".join(f"{a}: {w}" for a, w in calib["frozen"].items())
    warn = f"**Warning: the demand is not calibrated.** {log_text}; recommended scale {rec}"
    return f"{head} {warn}" + (f"; frozen arms: {frozen}." if frozen else ".")


def _assumptions_section(cfg: dict, vcfg: dict, calib: dict | None) -> list:
    lines = ["Reference arcs (Paris sensors, D6):", ""]
    for st in vcfg["stations"]:
        alt = f", sensitivity arc {st['alt_iu_ac']}" if st.get("alt_iu_ac") else ""
        use = "A-02" if st.get("a02") else "information only"
        arm = f"{st['direction']}bound arm {st['arm']}"
        lines.append(
            f"- `{st['station']}` ({arm}): arc {st['iu_ac']}{alt}; {use}. {st.get('note', '')}"
        )
    lines += ["", "Model inputs that are ASSUMED (not measured on site):", ""]
    assumptions = [
        *cfg["network"].get("assumptions", []),
        *cfg.get("demand", {}).get("assumptions", []),
    ]
    lines += [f"- ASSUMED: {a}" for a in assumptions] or ["- none listed"]
    lines += ["", _calibration_text(cfg, calib)]
    return [s.rstrip() for s in lines]


def _flow_rows(df: pd.DataFrame) -> list[list]:
    return [
        [
            r["station"],
            r["iu_ac"] if isinstance(r["iu_ac"], str) else "–",
            r["slot"],
            r["role"],
            f"{_pm(r['real_mean'], r['real_std'])} ({_f(r['n_days'])} d)",
            f"{_pm(r['sim_mean'], r['sim_std'])} ({_f(r['n_seeds'])} s)",
            _f(r["rel_err_pct"], 1),
            _f(r["geh"], 2),
            r["status"],
        ]
        for _, r in df.iterrows()
    ]


def _flows_section(val: Validation, vcfg: dict, figures: dict, docs: Path) -> list:
    th, a02, geh, flows = vcfg["thresholds"], val.a02, val.geh, val.flows
    share = _f(100 * geh["share"] if geh["n_counts"] else None)
    lines = [
        f"**A-02 verdict: {a02['verdict']}** ({a02['pass']} of {a02['rows']} inbound arm x "
        f"slot rows within ±{th['flow_rel']:.0%}). GEH < {th['geh']:g} on {geh['n_below']} of "
        f"{geh['n_counts']} distinct counts ({share} %, target {th['geh_share']:.0%}: "
        f"{'met' if geh['ok'] else 'not met'}).",
        "",
    ]
    if "flows" in figures:
        lines += [f"![Simulated vs real flows]({_rel(figures['flows'], docs)})", ""]
    lines += ["A-02 rows (inbound arms):", ""]
    lines += _table(FLOW_HEAD, _flow_rows(flows[flows["role"] == "A-02"]))
    lines += ["", "Control slots, outbound stations (information) and the sensitivity arc:", ""]
    lines += _table(FLOW_HEAD, _flow_rows(flows[flows["role"] != "A-02"]))
    lines += [
        "",
        "**Circularity.** The car demand is built from the same sensors (arcs 508, 511, 513, "
        "typical days, same slots), so inbound A-02 mainly checks that the network carries "
        "and inserts the demand in the right time slots: it is not independent evidence of "
        "realism. The independent checks are the signal timing and queues from the field "
        "visit, the outbound flows (information only: they depend on the ASSUMED turn "
        "shares), and the 510 sensitivity row (arc 510 copies station 505 near Austerlitz). "
        "The loops count vehicles once they are inserted: an insertion backlog (section 7) "
        "shows up as missing flow late in the window.",
    ]
    return lines


def _signal_section(val: Validation) -> list:
    sig = val.signal
    rows = [
        [
            r["item"],
            _f(r["model_value"], 1),
            _sim_cell(r),
            f"{_f(r['field_value'], 1)} ({_f(r['field_n'])})",
            _f(r["tol"], 1),
            r["status"],
        ]
        for _, r in sig.iterrows()
    ]
    return [
        "Model = provisional program (`fixed.green_s` in the net's phase structure; ASSUMED "
        "until the field timing). Sim = green intervals of every evaluation run "
        "(SaveTLSSwitchTimes); a vehicle head's green includes the trailing clearance "
        "phases where it stays green; yellow and all-red are replayed from the program. A "
        "head green more than once per cycle is checked per stage (`<head>.green_s@<stage>`, "
        "e.g. ped_NB in HOP and in STM), never as the mean of its greens.",
        "",
        f"Status counts: {_counts(sig['status'])}.",
        "",
        *_table(["Item", "Model", "Sim mean ± sd (n)", "Field (n)", "Tol", "Status"], rows),
    ]


def _queues_section(val: Validation, figures: dict, docs: Path) -> list:
    q = val.queues
    if q.empty:
        return ["No queue observation in the field file."]
    lines = [f"Status counts: {_counts(q['status'])}.", ""]
    if "queues" in figures:
        lines += [f"![Queues]({_rel(figures['queues'], docs)})", ""]
    head = ["Approach", "Time", "Kind", "Observed", "Sim p10 / p50 / p90 (n)", "Truncated"]
    rows = []
    for _, r in q.iterrows():
        sim = " / ".join(_f(r[k], 1) for k in ("sim_p10", "sim_p50", "sim_p90"))
        time = r["time"] or "–"
        cells = [r["approach"], time, r["kind"], _f(r["obs_veh"]), f"{sim} ({_f(r['sim_n'])})"]
        rows.append([*cells, _f(r["truncated_share"], 2), r["status"]])
    return lines + _table([*head, "Status"], rows)


def _capacity_rows(capacity: pd.DataFrame) -> list[list]:
    rows = []
    for _, r in capacity.iterrows():
        over = ", ".join(map(str, r["seeds_over"])) if r["seeds_over"] else "none"
        rows.append(
            [
                f"`{r['scenario']}`",
                r["vehicles"],
                _f(int(r["trips"])),
                _f(r["mean_s"], 1),
                f"{_f(r['worst_seed_mean_s'], 1)} (seed {r['worst_seed']})",
                _f(r["worst_15min_s"], 0),
                _f(r["max_s"], 0),
                over if str(r["vehicles"]).startswith("car ") else "–",
                r["status"],
            ]
        )
    return rows


def _safety_section(summ: dict, capacity: pd.DataFrame, threshold: float) -> list:
    rows = []
    for name, s in summ["safety"].items():
        inter = sum(s["interventions"].values()) if s["interventions"] else None
        audit = sum(s["audit"].values()) if s["audit"] else None
        health = [_f(s["health"][k]) for k in HEALTH_KEYS]
        rows.append([f"`{name}`", s["runs"], *health, _f(s["depart_delay_max"]), inter, audit])
    lines = [
        *_table(SAFETY_HEAD, rows),
        "",
        "Sums over all runs (run.json `health` and `safety`). A-04 requires 0 collisions and "
        "0 unsafe signal sequences (audit) on every test seed; the guard must not intervene "
        "for the fixed and actuated controllers. `–` = not reported by these runs.",
        "",
        "**Insertion delay and capacity.** Depart delay = time a vehicle waits to enter the "
        "network (SUMO `departDelay`). SUMO's `timeLoss`, behind every delay KPI of this page "
        "and of the S0 vs S1 table, does not include it: a vehicle that cannot be inserted "
        "waits upstream, outside these KPIs. A car arm whose mean depart delay over a run "
        f"exceeds {threshold:g} s (`thresholds.capacity_depart_delay_s`, the criterion of "
        "the network capacity test) is capacity-limited on that seed: its demand is not "
        "served in time and its delay KPIs are understated. The worst 15-min mean groups "
        "vehicles by planned departure (depart - departDelay); far above the run mean, it "
        "shows a backlog growing within the window.",
        "",
    ]
    if capacity.empty:
        return [*lines, "No trip table in these runs: insertion delay not reported."]
    head = [
        "Scenario", "Vehicles", "Trips", "Mean (s)", "Worst seed mean (s)",
        "Worst 15-min mean (s)", "Max (s)", f"Seeds with mean > {threshold:g} s", "Status",
    ]  # fmt: skip
    return [*lines, *_table(head, _capacity_rows(capacity))]


def _pedestrian_section(summ: dict) -> list:
    rows = []
    for name, p in summ["pedestrians"].items():
        single, walk = p["ped_wait_single_max"] or {}, p["ped_wait_max"] or {}
        waits = [_f(single.get("mean"), 1), _f(single.get("max"), 1), _f(p["target_s"], 0)]
        rows.append([f"`{name}`", *waits, _f(p["within_target"]), _f(walk.get("max"), 1)])
    return [
        *_table(PED_HEAD, rows),
        "",
        "The 90 s pedestrian target (A-04, D14) is judged on the longest single wait of a "
        "pedestrian (TraCI waiting time), not on the time lost over a whole walk.",
    ]


def _comparison_section(comparison: Comparison | None, compare_md: str | None) -> list:
    if comparison is None:
        return [
            "No comparison yet: run `tasks.py compare` (S0 vs S1 on the evaluation seeds), "
            "then rerun this report."
        ]
    if comparison.problem:
        return [
            f"**Not shown: the comparison `{comparison.csv.as_posix()}` is out of date** "
            f"({comparison.problem}). Rerun `tasks.py compare`, then this report."
        ]
    return [(compare_md or "").rstrip()]


# -- limitations -----------------------------------------------------------------------
def _s1_text(cfg: dict, comparison: Comparison | None) -> str:
    """How far S1's max greens go beyond S0's fixed greens (from the configs)."""
    fixed = (cfg.get("fixed") or {}).get("green_s") or {}
    maxg = (cfg.get("actuated") or {}).get("max_green_s") or {}
    if not maxg:
        return ""
    safety = cfg.get("safety") or {}
    bound = float(safety.get("ped_max_wait_s", 90)) - float(safety.get("ped_red_margin_s", 0))
    longer = {st: float(maxg[st]) - float(fixed[st]) for st in maxg if st in fixed}
    longer = {st: d for st, d in longer.items() if d > 0}
    caps = ", ".join(f"{st} {_f(float(maxg[st]))} s" for st in maxg if st in longer)
    s0 = ", ".join(f"{st} {_f(float(fixed[st]))} s" for st in maxg if st in longer)
    if not longer:
        return (
            f"- S1 max greens keep the worst-case crossing red at or under {bound:g} s and never "
            "exceed S0's fixed greens: S1 can only end a green earlier than S0 (open question "
            "Q3)."
        )
    ext = ", ".join(f"{st} by up to {_f(d)} s" for st, d in longer.items())
    text = (
        f"- S1 max greens ({caps}) keep the worst-case crossing red at or under {bound:g} s "
        f"(open question Q3), but they are above S0's fixed greens ({s0}): S1 mostly ends greens "
        f"early (gap-out) yet can extend {ext}, which lengthens the longest single "
        "pedestrian wait"
    )
    t = comparison.table if comparison and comparison.table is not None else None
    if t is not None:
        row = t[(t["user_type"] == "pedestrian") & (t["kpi"] == "ped_wait_single_max")]
        if len(row):
            r = row.iloc[0]
            p = f", Holm p {r['p_holm']:.4f}" if pd.notna(r["p_holm"]) else ""
            text += (
                f" (`ped_wait_single_max` {_f(r['s0_mean'], 1)} → {_f(r['s1_mean'], 1)} s"
                f"{p}, section 9)"
            )
    return text + "."


def _capacity_text(capacity: pd.DataFrame, threshold: float) -> str:
    if capacity.empty:
        return ""
    limited = capacity[capacity["status"] == "capacity-limited"]
    if limited.empty:
        return (
            f"- Capacity: no car arm has a mean depart delay above {threshold:g} s on any "
            "evaluation seed (section 7)."
        )
    parts = [
        f"{r['vehicles']} on seed{'s' if len(r['seeds_over']) > 1 else ''} "
        f"{', '.join(map(str, r['seeds_over']))} of `{r['scenario']}` (worst seed mean "
        f"{_f(r['worst_seed_mean_s'], 0)} s, worst 15-min mean {_f(r['worst_15min_s'], 0)} s)"
        for _, r in limited.iterrows()
    ]
    text = (
        f"- Capacity: {'; '.join(parts)} exceed the {threshold:g} s mean depart delay: the "
        "arm is capacity-limited on these evaluation seeds (the calibration loop judges "
        "capacity on the calibration seeds only). That insertion delay is outside every "
        "delay KPI"
    )
    cars = capacity[capacity["vehicles"].astype(str).str.startswith("car ")]
    if cars["scenario"].nunique() > 1:
        per = cars.assign(w=cars["mean_s"] * cars["trips"]).groupby("scenario", sort=False)
        mean = per["w"].sum() / per["trips"].sum()
        text += (
            ": cars wait "
            + " vs ".join(f"{_f(v, 1)} s (`{s}`)" for s, v in mean.items())
            + " on average (pooled over all car trips) before entering, so the delay KPIs "
            "understate the gap between the "
            "scenarios"
        )
    amb = capacity[capacity["vehicles"] == "ambulance"]
    if len(amb) and amb["max_s"].max() > 0:
        r = amb.loc[amb["worst_seed_mean_s"].idxmax()]
        text += (
            f". The ambulance waited up to {_f(r['max_s'], 0)} s to enter (`{r['scenario']}` "
            f"seed {r['worst_seed']}), also outside its time loss"
        )
    return text + "."


def limitations(
    cfg: dict,
    capacity: pd.DataFrame,
    threshold: float,
    comparison: Comparison | None,
    scenario_path: Path,
) -> list[str]:
    """Section 10, from the configs and the results (nothing hard-coded that a config
    change could make false)."""
    bullets = [
        "- Inbound flows are close to circular (see section 4); few typical days (about 16) "
        "and shared Paris stations (509 = 511 = 5071, 510 = 505, 513 copied along ~700 m) "
        "leave about three independent inbound counts.",
        "- Sensors are hourly: half-hour slots assume a uniform flow within each sensor hour "
        "(D5), on both the demand and the validation side.",
        "- The signal program, lane use, turn shares, bus headways and pedestrian flows are "
        "ASSUMED until the field visit (section 3); every signal and queue row stays "
        "NOT_MEASURED until then.",
        _s1_text(cfg, comparison),
        _capacity_text(capacity, threshold),
    ]
    scenario = scenario_path.as_posix()
    return [
        *[b for b in bullets if b],
        "",
        "After the field visit (docs/field_protocol.md):",
        "",
        "1. Fill `data/field/pitie_measurements.yaml` and run `tasks.py field-check`.",
        "2. If the stage structure differs, update `pitie.tll.xml`, `network.stages`, "
        f"`signal_groups` and `crossings`, then `tasks.py net tls-check --scenario {scenario}` "
        "(without `--scenario`, `net` rebuilds the grid network).",
        "3. Set `fixed.green_s` so that each head's green matches the measured mean.",
        "4. Put the turning counts into the calibration (`tasks.py calib-data`).",
        "5. Rerun `tasks.py part3`: signal and queue rows become PASS / FAIL; any failure is "
        "corrected through the demand or the driving parameters and documented (13.4). The "
        "calibration step stops the chain if an arm is capacity-limited or the recommended "
        "`arm_scale` is not the configured one; `compare` and `validate` rerun every run "
        "made with the old program, demand or net (stale `config_hash` / `demand_hash`) and "
        "refuse stale runs.",
    ]


def render(
    cfg: dict,
    vcfg: dict,
    val: Validation,
    summ: dict,
    scenario_path: Path,
    comparison: Comparison | None,
    compare_md: str | None,
    figures: dict[str, Path],
    docs: Path,
    capacity: pd.DataFrame,
    calibration: dict | None = None,
) -> str:
    """docs/validation.md: one section per entry of SECTIONS."""
    threshold = float(vcfg["thresholds"].get("capacity_depart_delay_s", CAPACITY_DEFAULT_S))
    bodies = [
        _runs_section(cfg, vcfg, val.real, summ, scenario_path),
        _thresholds_section(vcfg, summ),
        _assumptions_section(cfg, vcfg, calibration),
        _flows_section(val, vcfg, figures, docs),
        _signal_section(val),
        _queues_section(val, figures, docs),
        _safety_section(summ, capacity, threshold),
        _pedestrian_section(summ),
        _comparison_section(comparison, compare_md),
        limitations(cfg, capacity, threshold, comparison, scenario_path),
    ]
    lines = [
        "# Validation: Pitié-Salpêtrière digital twin vs reality",
        "",
        f"Generated by `python -m traffic_twin.validation.report --scenario "
        f"{scenario_path.as_posix()}` (spec 13.4, acceptance criterion A-02). Do not edit by "
        f"hand: rerun `tasks.py validate`. Status words: {STATUS_WORDS}.",
    ]
    for title, body in zip(SECTIONS, bodies, strict=True):
        lines += ["", f"## {title}", "", *body]
    return "\n".join(lines) + "\n"


def _sim_cell(r: pd.Series) -> str:
    if r["kind"] == "structure":
        return "–"
    n = "program" if r["sim_n"] == 0 and pd.notna(r["sim_mean"]) else _f(r["sim_n"])
    return f"{_pm(r['sim_mean'], r['sim_std'], 1)} ({n})"


def _counts(s: pd.Series) -> str:
    c = Counter(s.dropna())
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items())) or "none"


def _rel(path: Path, docs: Path) -> str:
    """Link from the docs page to a file."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(docs).resolve().parent)).as_posix()
    except ValueError:  # another drive on Windows
        return Path(path).resolve().as_posix()


def find_comparison(results: Path, scenario: str) -> Path | None:
    hits = sorted((Path(results) / "comparison").glob(f"{scenario}__vs__*.csv"))
    return hits[0] if hits else None


# -- CLI -------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scenario", type=Path, default=PITIE_FIXED_CONFIG)
    parser.add_argument(
        "--run-missing", action="store_true", help="first run absent seeds and rerun stale ones"
    )
    parser.add_argument("--docs", type=Path, help=f"write the markdown page (e.g. {DOCS_FILE})")
    parser.add_argument("--compare-csv", type=Path, help="S0 vs S1 table (default: found)")
    parser.add_argument("--results", type=Path, help="default: the config's outputs.dir")
    parser.add_argument("--out", type=Path, help="default: <results>/validation/<scenario>")
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR)
    parser.add_argument("--validation", type=Path, default=VALIDATION_CONFIG)
    parser.add_argument("--data-config", type=Path, default=DATA_CONFIG)
    parser.add_argument("--field", type=Path, help="default: configs/validation.yaml field")
    parser.add_argument("--calibration-log", type=Path, default=CALIBRATION_LOG)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        cfg = load_scenario(args.scenario)
        vcfg = load_validation(args.validation)
        cal = load_config(args.data_config).get("calendar") or {}
        fd = load_field(
            args.field or Path(vcfg.get("field", "data/field/pitie_measurements.yaml")), cfg
        )
        hours = read_hours(vcfg["reference"]["file"])
    except (ConfigError, FileNotFoundError) as e:
        raise SystemExit(f"validation error: {e}") from None
    results = args.results or Path(cfg["outputs"]["dir"])
    seeds = [int(s) for s in vcfg["seeds"]["evaluation"]]
    if args.run_missing:
        run_missing(cfg, seeds, results, args.scenario)
    runs = load_runs(results, cfg["scenario"], seeds, tables=RUN_TABLES)
    if not runs.seeds:
        raise SystemExit(f"no runs of {cfg['scenario']} under {results} (use --run-missing)")
    try:
        hashes = check_current(runs, cfg)
    except StaleRunsError as e:
        raise SystemExit(f"validation error: {e}") from None
    missing = sorted(set(seeds) - set(runs.seeds))
    if missing:
        log.warning("seeds %s have no run: the tables use %d seeds", missing, len(runs.seeds))
    try:
        val = build(cfg, vcfg, cal, fd, runs, hours)
    except ConfigError as e:
        raise SystemExit(f"validation error: {e}") from None

    from traffic_twin.evaluation.compare import markdown as compare_markdown
    from traffic_twin.viz.validation import plot_flows, plot_queues

    th = vcfg["thresholds"]
    threshold = float(th.get("capacity_depart_delay_s", CAPACITY_DEFAULT_S))
    compare_csv = args.compare_csv or find_comparison(results, cfg["scenario"])
    comparison, compare_md = None, None
    if compare_csv and Path(compare_csv).is_file():
        comparison = load_comparison(Path(compare_csv), cfg, hashes, results, seeds)
        if comparison.problem:
            log.warning("comparison %s is out of date: %s", compare_csv, comparison.problem)
        else:
            other = comparison.other
            compare_md = compare_markdown(
                comparison.table, cfg["scenario"], other, th["alpha"], th["equity_pct"]
            )
            figure = Path(args.figures) / f"comparison_{cfg['scenario']}_{other}.png"
            if figure.is_file() and args.docs:
                compare_md += f"\n![Paired differences]({_rel(figure, args.docs)})\n"

    begin = float(cfg["begin_s"])
    capacity = pd.concat(
        [
            insertion_delays(r, begin, threshold)
            for r in [runs, *([comparison.runs] if comparison and comparison.runs else [])]
        ],
        ignore_index=True,
    )
    calibration = calibration_status(args.calibration_log, cfg)
    out = args.out or Path(results) / "validation" / cfg["scenario"]
    out.mkdir(parents=True, exist_ok=True)
    val.flows.to_csv(out / "flow_validation.csv", index=False)
    val.signal.to_csv(out / "signal_validation.csv", index=False)
    val.queues.to_csv(out / "queue_validation.csv", index=False)
    summ = summary(cfg, vcfg, val, runs, hashes, comparison, fd, capacity, calibration)
    if summ["preregistered_before_runs"] is False:
        log.warning(
            "pre-registration date %s is later than the first run (%s)",
            summ["preregistered"],
            summ["first_run_date"],
        )
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")

    figures = {}
    if val.flows["status"].isin(["PASS", "FAIL", "ok", "off"]).any():
        figures["flows"] = plot_flows(
            val.flows,
            Path(args.figures) / "validation_flows.png",
            float(vcfg["thresholds"]["flow_rel"]),
            str(cfg["controller"]),
        )
    if (val.queues["sim_n"].fillna(0) > 0).any():
        figures["queues"] = plot_queues(
            val.queues, Path(args.figures) / "validation_queues.png", str(cfg["controller"])
        )
    if args.docs:
        text = render(
            cfg, vcfg, val, summ, args.scenario, comparison, compare_md, figures, args.docs,
            capacity, calibration,
        )  # fmt: skip
        args.docs.parent.mkdir(parents=True, exist_ok=True)
        args.docs.write_text(text, encoding="utf-8")
        log.info("validation page: %s", args.docs.as_posix())
    a02 = val.a02
    print(
        f"A-02 {a02['verdict']}: {a02['pass']}/{a02['rows']} rows within the limit; "
        f"GEH share {val.geh['share']:.2f} over {val.geh['n_counts']} counts"
    )
    print(f"signal: {_counts(val.signal['status'])}; queues: {_counts(val.queues['status'])}")
    limited = capacity[capacity["status"] == "capacity-limited"]
    if len(limited):
        print(
            "capacity-limited: "
            + "; ".join(
                f"{r['scenario']} {r['vehicles']} seeds {r['seeds_over']}"
                for _, r in limited.iterrows()
            )
        )
    log.info("tables: %s", out.as_posix())


if __name__ == "__main__":
    main()

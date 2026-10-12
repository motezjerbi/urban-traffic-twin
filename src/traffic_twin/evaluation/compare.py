"""Paired comparison of two scenarios over the evaluation seeds (spec 13.3), e.g. S0
fixed time vs S1 actuated on the Pitié junction.

    python -m traffic_twin.evaluation.compare --baseline-config configs/pitie_fixed.yaml
        --scenario-config configs/pitie_actuated.yaml [--run-missing]

Reads `<results>/<scenario>_seed<n>/{kpi_run.csv, run.json}` for both scenarios and
refuses to compare unless both have the same seeds, every run of a scenario was made with
the CURRENT config (its run.json `config_hash` and `demand_hash` equal those of the
config; `--run-missing` runs absent seeds and reruns stale ones) and the two scenarios
share one `demand_hash` (same demand, same seeds: the runs are paired per seed). For
every KPI: means and std, the paired difference (S1 - S0) with a Student-t CI, the
Hodges-Lehmann estimate and CI, the exact Wilcoxon signed-rank p-value (two-sided), and
Holm-adjusted p-values over the pre-registered headline KPIs only (`compare.headline` in
configs/validation.yaml). `count` KPIs are not tested. Equity flag: a KPI more than
`equity_pct` % worse in its bad direction (spec 13.3: a global gain must not hide a loss
for one user type). A qualified KPI `<base>__<qualifier>` (per arm, flow, line or
crossing, Part 4) takes the direction and unit of its base name. The N-scenario Part 4
evaluation (evaluation.evaluate) calls `compare()` once per pre-registered pair.

Writes `<results>/comparison/<a>__vs__<b>.csv`, `.md` (headline table) and `.json`
(provenance: configs, hashes and seeds, checked by the validation report) and
`docs/figures/pitie/comparison_<a>_<b>.png` (per-seed paired differences).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_twin.config import ConfigError, load_scenario
from traffic_twin.evaluation.runs import expected_hashes, load_runs, run_missing, stale_reason
from traffic_twin.evaluation.stats import hodges_lehmann, holm, t_ci, wilcoxon
from traffic_twin.paths import PITIE_ACTUATED_CONFIG, PITIE_FIXED_CONFIG, VALIDATION_CONFIG

log = logging.getLogger(__name__)

FIGURES_DIR = Path("docs/figures/pitie")
# Direction, test and unit sets hold BASE KPI names: a qualified KPI `<base>__<qualifier>`
# (e.g. `time_loss_mean__arm_N`, Part 4) is looked up by `base_kpi(kpi)`.
# `served_share` is reserved for a future preemption / completion share KPI: configs/
# evaluation.yaml pre-registers a pp rule for it, but no run produces it yet (the
# evaluation page says so).
HIGHER_IS_BETTER = {"count", "throughput_vph", "crossing_complete_rate", "served_share"}
NOT_TESTED = {"count"}  # identical demand and seeds: counts carry no paired signal
NEUTRAL = {"dwell_mean"}  # set by the stop dwell config, no better direction
UNITLESS = {  # counts and rates; every other KPI is a time in seconds unless in UNITS
    "count", "stops", "crossing_red_entries", "box_foe_onsets", "crossing_complete_rate",
    "served_share",
}  # fmt: skip
UNITS = {"throughput_vph": "veh/h"}
QUALIFIER_SEP = "__"
COMPARE_COLUMNS = [
    "user_type", "kpi", "headline", "better_if", "n_pairs", "s0_mean", "s0_std", "s1_mean",
    "s1_std", "diff_mean", "diff_ci_low", "diff_ci_high", "hl_diff", "hl_ci_low", "hl_ci_high",
    "rel_change_pct", "wilcoxon_stat", "n_nonzero", "p_value", "p_holm", "significant",
    "significant_holm", "effect", "equity_flag", "equity_flag_significant", "underpowered",
]  # fmt: skip


MD_HEAD = ["User", "KPI", "S0 mean ± sd", "S1 mean ± sd", "S1 - S0 [95 % CI]", "Change", "p",
           "p (Holm)", "Effect", "Equity flag"]  # fmt: skip


class ComparisonError(ValueError):
    """The two scenarios cannot be compared pairwise (seeds or demand differ, or a run was
    not made with the current config)."""


def base_kpi(kpi: str) -> str:
    """The KPI name without its qualifier: `delay_mean__flow_91_E` -> `delay_mean`."""
    return kpi.split(QUALIFIER_SEP)[0]


def better_if(kpi: str) -> str:
    base = base_kpi(kpi)
    if base in NEUTRAL:
        return "n/a"
    return "higher" if base in HIGHER_IS_BETTER else "lower"  # times, waits, delays, stops


def _sense(kpi: str) -> int:
    return {"lower": -1, "higher": 1}.get(better_if(kpi), 0)


def unit(kpi: str) -> str:
    base = base_kpi(kpi)
    if base in UNITS:
        return UNITS[base]
    return "" if base in UNITLESS else "s"


def worse_than(m0: float, m1: float, rel: float, sense: int, pct: float) -> bool:
    """Equity flag: more than `pct` % worse than the baseline mean `m0` in the KPI's bad
    direction. A zero baseline has no relative change (`rel` NaN): any move in the bad
    direction then counts as worse (e.g. pedestrian red entries 0 -> 1), none otherwise."""
    if not sense:
        return False
    if m0 == 0:
        return bool((m1 - m0) * sense < 0)
    return bool(math.isfinite(rel) and rel * sense < -pct)


def incomplete_rows(table: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    """The rows of a comparison table paired on fewer than `n_seeds` seeds (a run lacks
    the KPI, e.g. an ambulance that never arrived): never silently fewer pairs."""
    if table.empty:
        return table
    return table[table["n_pairs"] < n_seeds]


# -- pairing ---------------------------------------------------------------------------
def check_pairing(
    meta_a: dict[int, dict],
    meta_b: dict[int, dict],
    seeds: list[int],
    expected_a: dict[str, str] | None = None,
    expected_b: dict[str, str] | None = None,
) -> str:
    """The shared demand_hash. Raises ComparisonError when a seed is missing on either
    side, a scenario's runs do not share one config_hash, a run is stale (its hashes are
    not `expected_*`, those of the current config), or the runs do not share one
    demand_hash."""
    problems = []
    for name, metas, expected in (
        ("baseline", meta_a, expected_a),
        ("scenario", meta_b, expected_b),
    ):
        missing = sorted(set(seeds) - set(metas))
        if missing:
            problems.append(f"{name} has no run for seeds {missing}")
        configs = {m.get("config_hash") for m in metas.values()}
        if len(configs) > 1:
            problems.append(f"{name} runs differ in config_hash ({len(configs)} values)")
        if expected:
            reasons = {s: stale_reason(m, expected) for s, m in sorted(metas.items())}
            stale = "; ".join(f"seed {s}: {r}" for s, r in reasons.items() if r)
            if stale:
                problems.append(f"{name} runs are stale, not of the current config ({stale})")
    if set(meta_a) != set(meta_b):
        problems.append(f"seed sets differ: {sorted(meta_a)} vs {sorted(meta_b)}")
    if expected_a and expected_b and expected_a["demand_hash"] != expected_b["demand_hash"]:
        problems.append("the two configs do not share one demand (demand_hash differs)")
    hashes = {m.get("demand_hash") for m in [*meta_a.values(), *meta_b.values()]}
    if None in hashes:
        problems.append("a run.json has no demand_hash (rerun it with the current code)")
    elif len(hashes) > 1:
        problems.append(f"runs differ in demand_hash ({len(hashes)} values): not the same demand")
    if problems:
        raise ComparisonError("cannot pair the runs: " + "; ".join(problems))
    return hashes.pop()


# -- statistics ------------------------------------------------------------------------
def paired(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """user_type, kpi, seed, s0, s1, diff (= s1 - s0) for the KPIs both runs report."""
    keys = ["user_type", "kpi", "seed"]
    m = base[[*keys, "value"]].merge(other[[*keys, "value"]], on=keys, suffixes=("_0", "_1"))
    m = m.rename(columns={"value_0": "s0", "value_1": "s1"})
    m["diff"] = m["s1"] - m["s0"]
    return m.sort_values(keys, ignore_index=True)


def compare(
    base: pd.DataFrame,
    other: pd.DataFrame,
    headline: list[tuple[str, str]] | list[list[str]],
    equity_pct: float = 5.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One row per KPI present in both: the comparison table (columns COMPARE_COLUMNS),
    headline KPIs first in their pre-registered order."""
    head = [tuple(h) for h in headline]
    pairs = paired(base, other)
    order = list(dict.fromkeys(zip(base["user_type"], base["kpi"], strict=True)))
    order = [h for h in head if h in order] + [k for k in order if k not in head]
    rows = []
    for user, kpi in order:
        g = pairs[(pairs["user_type"] == user) & (pairs["kpi"] == kpi)]
        if g.empty:
            continue
        d = g["diff"].to_numpy(float)
        tested = base_kpi(kpi) not in NOT_TESTED
        w = wilcoxon(d) if tested else None
        ci = t_ci(d)
        hl, hl_lo, hl_hi = hodges_lehmann(d)
        m0, m1 = float(g["s0"].mean()), float(g["s1"].mean())
        rel = 100.0 * (m1 / m0 - 1) if m0 != 0 else math.nan
        sense = _sense(kpi)
        p = w.pvalue if w else math.nan
        worse = worse_than(m0, m1, rel, sense, equity_pct)
        if not sense:
            effect = "n/a"
        elif m1 == m0:
            effect = "same"
        else:
            effect = "better" if (m1 - m0) * sense > 0 else "worse"
        rows.append(
            {
                "user_type": user,
                "kpi": kpi,
                "headline": (user, kpi) in head,
                "better_if": better_if(kpi),
                "n_pairs": len(d),
                "s0_mean": m0,
                "s0_std": float(g["s0"].std(ddof=1)) if len(d) > 1 else math.nan,
                "s1_mean": m1,
                "s1_std": float(g["s1"].std(ddof=1)) if len(d) > 1 else math.nan,
                "diff_mean": ci.mean,
                "diff_ci_low": ci.low,
                "diff_ci_high": ci.high,
                "hl_diff": hl,
                "hl_ci_low": hl_lo,
                "hl_ci_high": hl_hi,
                "rel_change_pct": rel,
                "wilcoxon_stat": w.statistic if w else math.nan,
                "n_nonzero": w.n if w else math.nan,
                "p_value": p,
                "significant": bool(p < alpha),
                "effect": effect,
                "equity_flag": worse,
                "equity_flag_significant": bool(worse and p < alpha),
                "underpowered": bool(w and w.min_attainable_p >= alpha),
            }
        )
    t = pd.DataFrame(rows)
    if t.empty:
        return t.reindex(columns=COMPARE_COLUMNS)
    family = t["headline"] & t["p_value"].notna()
    t["p_holm"] = np.nan
    if family.any():
        t.loc[family, "p_holm"] = holm(t.loc[family, "p_value"].to_numpy())
    t["significant_holm"] = t["p_holm"] < alpha
    return t.reindex(columns=COMPARE_COLUMNS)


# -- outputs ---------------------------------------------------------------------------
def _num(x: float, nd: int = 1) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _p(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:.4f}" if x >= 0.0001 else f"{x:.1e}"


def _nd(*values: float) -> int:
    """Decimals for one KPI row: 3 below 1 (insertion delays of a few tenths of a second),
    2 below 10, else 1."""
    finite = [abs(v) for v in values if v is not None and np.isfinite(v)]
    top = max(finite) if finite else 10.0
    return 3 if top < 1 else 2 if top < 10 else 1


def _signed(x: float, nd: int) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def diff_text(r: pd.Series) -> str:
    """'S1 - S0 [95 % CI] unit' of one KPI row: the absolute paired difference."""
    nd = _nd(r["s0_mean"], r["s1_mean"])
    u = f" {unit(r['kpi'])}" if unit(r["kpi"]) else ""
    ci = f"[{_num(r['diff_ci_low'], nd)}, {_num(r['diff_ci_high'], nd)}]"
    return f"{_signed(r['diff_mean'], nd)} {ci}{u}"


def markdown(
    table: pd.DataFrame, a: str, b: str, alpha: float = 0.05, equity_pct: float = 5.0
) -> str:
    """Headline table (plus any other flagged KPI) as a markdown snippet. Every change is
    given as the absolute paired difference with its CI next to the relative change, so a
    large percentage on a near-zero baseline reads at its true size."""
    n = int(table["n_pairs"].max()) if len(table) else 0
    lines = [
        f"Paired comparison **{a}** (S0) vs **{b}**: {n} seeds, same demand "
        f"(`demand_hash` checked). Exact two-sided Wilcoxon signed-rank test on the per-seed "
        f"differences, Holm over the headline KPIs; α = {alpha}. With {n} seeds the smallest "
        f"possible p is {_p(2 * 0.5**n) if n else '–'}. S1 - S0: mean paired difference "
        "with its 95 % Student-t CI (seconds unless stated).",
        "",
        "| " + " | ".join(MD_HEAD) + " |",
        "|" + "---|" * len(MD_HEAD),
    ]
    for _, r in table[table["headline"]].iterrows():
        nd = _nd(r["s0_mean"], r["s1_mean"])
        lines.append(
            f"| {r['user_type']} | `{r['kpi']}` | {_num(r['s0_mean'], nd)} ± "
            f"{_num(r['s0_std'], nd)} | {_num(r['s1_mean'], nd)} ± {_num(r['s1_std'], nd)} "
            f"| {diff_text(r)} | {_num(r['rel_change_pct'])} % "
            f"| {_p(r['p_value'])} | {_p(r['p_holm'])} | {r['effect']} "
            f"| {'**yes**' if r['equity_flag'] else 'no'} |"
        )
    others = table[~table["headline"] & table["equity_flag"]]
    if len(others):
        lines += ["", f"Other KPIs more than {equity_pct:g} % worse (not in the Holm family):", ""]
        for _, r in others.iterrows():
            nd = _nd(r["s0_mean"], r["s1_mean"])
            u = f" {unit(r['kpi'])}" if unit(r["kpi"]) else ""
            lines.append(
                f"- {r['user_type']} `{r['kpi']}`: {_num(r['s0_mean'], nd)} → "
                f"{_num(r['s1_mean'], nd)}{u}, S1 - S0 {diff_text(r)} "
                f"({_signed(r['rel_change_pct'], 1)} %, p {_p(r['p_value'])})"
            )
    return "\n".join(lines) + "\n"


def output_paths(results: Path, figures: Path, a: str, b: str) -> tuple[Path, Path, Path]:
    stem = f"{a}__vs__{b}"
    out = Path(results) / "comparison"
    return out / f"{stem}.csv", out / f"{stem}.md", Path(figures) / f"comparison_{a}_{b}.png"


def provenance_path(csv: Path) -> Path:
    """The provenance file next to a comparison CSV."""
    return Path(csv).with_suffix(".json")


def provenance(sides: list[tuple[dict, Path, dict[str, str]]], seeds: list[int]) -> dict:
    """What a comparison was computed from: per side the scenario, its config file and the
    hashes every run of it shares (the validation report checks them)."""
    keys = ("baseline", "scenario")
    out: dict = {"seeds": [int(s) for s in seeds]}
    for key, (cfg, path, hashes) in zip(keys, sides, strict=True):
        out[key] = {"scenario": cfg["scenario"], "config": Path(path).as_posix(), **hashes}
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline-config", type=Path, default=PITIE_FIXED_CONFIG)
    parser.add_argument("--scenario-config", type=Path, default=PITIE_ACTUATED_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", help="default: evaluation seeds")
    parser.add_argument("--results", type=Path, help="default: the configs' outputs.dir")
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR)
    parser.add_argument("--validation", type=Path, default=VALIDATION_CONFIG)
    parser.add_argument(
        "--run-missing", action="store_true", help="first run absent seeds and rerun stale ones"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from traffic_twin.validation import load_validation

    try:
        cfg_a, cfg_b = load_scenario(args.baseline_config), load_scenario(args.scenario_config)
        vcfg = load_validation(args.validation)
    except ConfigError as e:
        raise SystemExit(f"config error: {e}") from None
    a, b = cfg_a["scenario"], cfg_b["scenario"]
    if a == b:
        raise SystemExit(f"both configs are scenario {a!r}: nothing to compare")
    seeds = args.seeds or [int(s) for s in vcfg["seeds"]["evaluation"]]
    results = args.results or Path(cfg_a["outputs"]["dir"])
    exp_a, exp_b = expected_hashes(cfg_a), expected_hashes(cfg_b)
    if args.run_missing:
        for cfg, path in ((cfg_a, args.baseline_config), (cfg_b, args.scenario_config)):
            run_missing(cfg, seeds, results, path)
    runs_a, runs_b = load_runs(results, a, seeds), load_runs(results, b, seeds)
    try:
        check_pairing(runs_a.meta, runs_b.meta, seeds, exp_a, exp_b)
    except ComparisonError as e:
        raise SystemExit(f"{e} (--run-missing runs absent seeds and reruns stale ones)") from None
    kpi_a, kpi_b = runs_a.kpis, runs_b.kpis

    th = vcfg["thresholds"]
    headline = vcfg["compare"]["headline"]
    alpha = float(th.get("alpha", 0.05))
    table = compare(kpi_a, kpi_b, headline, float(th.get("equity_pct", 5.0)), alpha)
    short = incomplete_rows(table, len(seeds))
    if len(short):
        listing = ", ".join(
            f"{r.user_type}.{r.kpi} ({r.n_pairs})" for r in short.itertuples(index=False)
        )
        log.warning("KPIs paired on fewer than %d seeds: %s", len(seeds), listing)
    csv, md, png = output_paths(results, args.figures, a, b)
    csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv, index=False)
    md.write_text(markdown(table, a, b, alpha, float(th.get("equity_pct", 5.0))), encoding="utf-8")
    sides = [(cfg_a, args.baseline_config, exp_a), (cfg_b, args.scenario_config, exp_b)]
    provenance_path(csv).write_text(
        json.dumps(provenance(sides, seeds), indent=2) + "\n", encoding="utf-8"
    )

    from traffic_twin.viz.validation import plot_paired_differences

    plot_paired_differences(paired(kpi_a, kpi_b), table, a, b, png)
    show = [
        "user_type",
        "kpi",
        "s0_mean",
        "s1_mean",
        "rel_change_pct",
        "p_value",
        "p_holm",
        "effect",
        "equity_flag",
    ]
    print(table[table["headline"]][show].round(4).to_string(index=False))
    log.info("comparison: %s, %s, %s", csv.as_posix(), md.as_posix(), png.as_posix())


if __name__ == "__main__":
    main()

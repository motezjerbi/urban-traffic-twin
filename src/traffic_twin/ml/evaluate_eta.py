"""ETA evaluation (Part 5, EF-14, A-07; D50; contract 4.5).

    python -m traffic_twin.ml.evaluate_eta --config configs/eta.yaml --split test
        --docs docs/eta.md --figures docs/figures/eta

val / calib are allowed for development; test needs a freeze.json matching the bundle
(D51), checked before any test row is read. Writes `data/models/eta/<version>/eval.json`
(`eval_<split>.json` for a development split), the figures (`viz.eta`) and the page.

**A-07 / EF-14 rule (D50, pre-registered in configs/eta.yaml `a07`).** Population: the
realistic worlds of the split, all three controllers. PASS iff the pooled coverage of the
served interval (inclusive integer bounds, D48) lies in [0.75, 0.85] (inclusive) **and**
the 95 % bootstrap CI of MAE(model median) - MAE(reference mean) lies entirely below 0.
Bootstrap unit = world (its three controller runs move together), 2,000 resamples, rng
seed 0, percentile CI. Reported besides: coverage CI, mean width, RMSE, the reference's
q10-q90 coverage and width, breakdowns (trajet, controller, day type, active incident,
ambulance present, world kind), the per-run coverage distribution and the stress worlds
(reported, never judged). A subgroup outside the band is a flag, not a failure.

**Post-hoc disclosures** (`POST_HOC_KEYS`, added after the single-shot test verdict by the
Parts 5-6 review; they never change the verdict and use no model output):
- `unfinished`: trajet cars still in the network when a run's drain hit its time limit
  (`runs.parquet` `n_unfinished_cars`, `drain_timeout`). They have no tripinfo record, so
  they never became samples: censored trips, absent from the dataset, not a D39 drop.
- `reference_sensitivity`: the reference refitted on the realistic train worlds only
  (the bundle's table, `reference.population: all`, also holds the stress worlds, all
  labelled `tue_wed_thu_term`), its MAE / q10-q90 coverage / width on the evaluated
  population, and how far the served degraded-mode table moves.

`--render-only` re-renders the page and the figures from the stored eval json (no model,
no re-evaluation, the freeze still checked for test) and adds the missing post-hoc keys.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_twin.config import load_config, text_sha256
from traffic_twin.ml.train_eta import (
    BEGIN_S,
    ETA_MODELS_DIR,
    EVAL_FILE,
    FALLBACK_HOUR,
    META_FILE,
    REFERENCE_FILE,
    TARGET,
    apply_reference,
    bundle_sha256,
    check_freeze,
    conformal_by_group,
    fit_reference,
    interval_coverage,
    load_split,
    read_manifest,
    reference_population,
    reference_rows,
    resolve_dataset,
    served_bounds,
)
from traffic_twin.paths import ETA_CONFIG

log = logging.getLogger(__name__)

SPLITS = ("val", "calib", "test")
GROUPS = ("trajet", "controller", "day_type", "incident_active", "ambulance_present", "kind")
CURVE_OFFSETS = tuple(np.round(np.arange(-5.0, 10.01, 0.5), 2))  # s around q-hat
DIGITS = 6
POST_HOC_KEYS = ("post_hoc", "unfinished", "reference_sensitivity")
POST_HOC_NOTE = (
    "Disclosures added after the single-shot test verdict (review of Parts 5-6): computed "
    "from run metadata, targets and the reference only (no model output); they never "
    "change the verdict."
)
RUNS_FILE = "runs.parquet"
_SENSITIVITY_COLUMNS = ("world_seed", "trajet", "insert_t", "day_type", "kind", TARGET)


# -- statistics ------------------------------------------------------------------------
def _r(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, DIGITS)


def _nanmean(x) -> float:
    """Mean over the finite values; NaN (no warning) when there is none."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def world_bootstrap(
    numer: np.ndarray,
    denom: np.ndarray,
    worlds: np.ndarray,
    n: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap estimates of sum(numer) / sum(denom) with the world as the resampling
    unit: per-world sums, `n` resamples of the worlds with replacement
    (`default_rng(seed).integers`)."""
    df = pd.DataFrame({"w": np.asarray(worlds), "a": numer, "b": denom})
    g = df.groupby("w", sort=True)[["a", "b"]].sum()
    a, b = g["a"].to_numpy(float), g["b"].to_numpy(float)
    k = len(g)
    if k == 0:
        return np.full(n, np.nan)
    idx = np.random.default_rng(seed).integers(0, k, (n, k))
    with np.errstate(invalid="ignore", divide="ignore"):
        return a[idx].sum(axis=1) / b[idx].sum(axis=1)


def percentile_ci(est: np.ndarray, ci: float = 0.95) -> list[float | None]:
    if not np.isfinite(est).any():
        return [None, None]
    tail = (1.0 - ci) / 2.0 * 100.0
    lo, hi = np.nanpercentile(est, [tail, 100.0 - tail])
    return [_r(lo), _r(hi)]


def a07_pass(
    coverage: float,
    mae_diff_ci: Sequence[float | None],
    band: Sequence[float] = (0.75, 0.85),
    inclusive: bool = True,
) -> bool:
    """D50: coverage inside the band (inclusive by default) and the MAE-difference CI
    entirely below 0."""
    lo, hi = float(band[0]), float(band[1])
    c = round(float(coverage), DIGITS)
    in_band = lo <= c <= hi if inclusive else lo < c < hi
    upper = mae_diff_ci[1]
    return bool(in_band and upper is not None and float(upper) < 0.0)


def _flag(coverage: float | None, band: Sequence[float]) -> bool:
    return coverage is not None and not float(band[0]) <= coverage <= float(band[1])


# -- evaluation frame ------------------------------------------------------------------
def eval_frame(
    df: pd.DataFrame,
    pred: np.ndarray,
    qhat: float,
    reference: pd.DataFrame,
    begin_s: int = BEGIN_S,
) -> pd.DataFrame:
    """One row per trip: identity, groups, target, served bounds, raw quantiles and
    reference values."""
    pred = np.sort(np.asarray(pred, dtype=float).reshape(-1, 3), axis=1)
    low, med, high = served_bounds(pred, qhat)
    ref = apply_reference(reference, df, begin_s)

    def col(name: str, default=0.0) -> np.ndarray:
        return df[name].to_numpy() if name in df else np.full(len(df), default)

    inc = sum(np.nan_to_num(col(f"inc_{a}").astype(float)) for a in ("N", "S", "W"))
    amb = np.nan_to_num(col("amb_present").astype(float)) + np.nan_to_num(
        col("amb_pending").astype(float)
    )
    return pd.DataFrame(
        {
            "world": df["world_seed"].astype(int).to_numpy(),
            "run": df["run_id"].astype(str).to_numpy(),
            "trajet": df["trajet"].astype(str).to_numpy(),
            "controller": df["controller"].astype(str).to_numpy(),
            "day_type": df["day_type"].astype(str).to_numpy(),
            "kind": df["kind"].astype(str).to_numpy(),
            "incident_active": np.where(inc > 0, "yes", "no"),
            "ambulance_present": np.where(amb > 0, "yes", "no"),
            "y": df[TARGET].astype(float).to_numpy(),
            "low": low.astype(float),
            "med": med.astype(float),
            "high": high.astype(float),
            "q10": pred[:, 0],
            "q50": pred[:, 1],
            "q90": pred[:, 2],
            "ref_mean": ref["mean_s"].to_numpy(float),
            "ref_p10": ref["p10_s"].to_numpy(float),
            "ref_p90": ref["p90_s"].to_numpy(float),
        }
    )


def _inside(e: pd.DataFrame) -> np.ndarray:
    return ((e["y"] >= e["low"]) & (e["y"] <= e["high"])).to_numpy(float)


def headline(e: pd.DataFrame, boot: Mapping, band: Sequence[float]) -> dict:
    """Pooled metrics of one population with world-bootstrap CIs."""
    if e.empty:
        return {"n": 0}
    n, seed, ci = int(boot.get("n", 2000)), int(boot.get("seed", 0)), float(boot.get("ci", 0.95))
    inside = _inside(e)
    ae = np.abs(e["med"] - e["y"]).to_numpy()
    ae_ref = np.abs(e["ref_mean"] - e["y"]).to_numpy()
    ones = np.ones(len(e))
    w = e["world"].to_numpy()
    cov_est = world_bootstrap(inside, ones, w, n, seed)
    has_ref = np.isfinite(ae_ref)  # always true with the fallback rows (D49)
    diff_est = world_bootstrap(
        np.where(has_ref, ae - ae_ref, 0.0), has_ref.astype(float), w, n, seed
    )
    ref_in = ((e["y"] >= e["ref_p10"]) & (e["y"] <= e["ref_p90"])).to_numpy()
    coverage = float(inside.mean())
    return {
        "n": len(e),
        "n_worlds": int(e["world"].nunique()),
        "n_runs": int(e["run"].nunique()),
        "coverage": _r(coverage),
        "ci": percentile_ci(cov_est, ci),
        "mae": _r(ae.mean()),
        "ref_mae": _r(_nanmean(ae_ref)),
        "mae_diff": _r(_nanmean(ae - ae_ref)),
        "mae_diff_ci": percentile_ci(diff_est, ci),
        "rmse": _r(np.sqrt(np.mean((e["med"] - e["y"]) ** 2))),
        "bias": _r(np.mean(e["med"] - e["y"])),
        "width": _r((e["high"] - e["low"]).mean()),
        "below": _r(np.mean(e["y"] < e["low"])),
        "above": _r(np.mean(e["y"] > e["high"])),
        "raw_coverage": _r(interval_coverage(e["y"].to_numpy(), e["q10"], e["q90"])),
        "ref_cov": _r(ref_in.mean()),
        "ref_width": _r(_nanmean(e["ref_p90"] - e["ref_p10"])),
        "flag": _flag(_r(coverage), band),
    }


def group_table(e: pd.DataFrame, by: str, band: Sequence[float]) -> list[dict]:
    """Coverage, MAE, reference MAE and width per value of `by` (flag: coverage outside
    the band)."""
    rows = []
    for key, g in e.groupby(by, sort=True):
        inside = _inside(g)
        cov = _r(inside.mean())
        rows.append(
            {
                "group": str(key),
                "n": len(g),
                "n_worlds": int(g["world"].nunique()),
                "coverage": cov,
                "mae": _r(np.abs(g["med"] - g["y"]).mean()),
                "ref_mae": _r(_nanmean(np.abs(g["ref_mean"] - g["y"]))),
                "width": _r((g["high"] - g["low"]).mean()),
                "flag": _flag(cov, band),
            }
        )
    return rows


def per_run_coverage(e: pd.DataFrame) -> dict:
    g = e.assign(inside=_inside(e)).groupby("run", sort=True)
    runs = pd.DataFrame(
        {
            "world": g["world"].first(),
            "controller": g["controller"].first(),
            "kind": g["kind"].first(),
            "n": g.size(),
            "coverage": g["inside"].mean(),
        }
    ).reset_index()
    runs = runs.sort_values(["world", "controller"], kind="mergesort")
    cov = runs["coverage"].to_numpy(float)
    summary = {
        "n_runs": len(runs),
        **{
            k: _r(np.percentile(cov, p)) if len(cov) else None
            for k, p in (("min", 0), ("p10", 10), ("median", 50), ("p90", 90), ("max", 100))
        },
    }
    return {
        "summary": summary,
        "runs": [
            {
                "run_id": r.run,
                "world": int(r.world),
                "controller": r.controller,
                "kind": r.kind,
                "n": int(r.n),
                "coverage": _r(r.coverage),
            }  # fmt: skip
            for r in runs.itertuples()
        ],
    }


def coverage_curve(e: pd.DataFrame, qhat: float) -> list[dict]:
    """Coverage of the served bounds when q-hat moves by each offset (calibration curve)."""
    pred = e[["q10", "q50", "q90"]].to_numpy()
    y = e["y"].to_numpy()
    out = []
    for off in CURVE_OFFSETS:
        low, _, high = served_bounds(pred, qhat + float(off))
        out.append(
            {
                "offset_s": float(off),
                "qhat": _r(qhat + float(off)),
                "coverage": _r(interval_coverage(y, low, high)),
                "width": _r(np.mean(high - low)),
            }
        )
    return out


def quantile_reliability(e: pd.DataFrame, quantiles: Sequence[float]) -> list[dict]:
    """Observed share of targets at or below each raw predicted quantile."""
    cols = ("q10", "q50", "q90")
    return [
        {"quantile": float(q), "observed": _r(np.mean(e["y"] <= e[c]))}
        for q, c in zip(quantiles, cols, strict=True)
    ]


def evaluate(
    df: pd.DataFrame,
    pred: np.ndarray,
    qhat: float,
    reference: pd.DataFrame,
    eta_cfg: Mapping,
    begin_s: int = BEGIN_S,
    split: str = "test",
) -> dict:
    """The metrics of D50 for one split (pure: no I/O). `pred` are the raw quantile
    predictions of the rows of `df` (sorted per row here)."""
    a07 = dict(eta_cfg.get("a07") or {})
    band = tuple(a07.get("coverage", (0.75, 0.85)))
    inclusive = bool(a07.get("inclusive", True))
    boot = dict(a07.get("bootstrap") or {})
    population = a07.get("population", "realistic")
    quantiles = (eta_cfg.get("model") or {}).get("quantiles", (0.1, 0.5, 0.9))
    e = eval_frame(df, pred, qhat, reference, begin_s)
    pop = e if population == "all" else e[e["kind"] == population]
    head = headline(pop, boot, band)
    verdict = bool(head.get("n")) and a07_pass(
        head["coverage"], head["mae_diff_ci"], band, inclusive
    )
    out: dict = {
        "split": split,
        "a07": {
            "population": population,
            "band": list(band),
            "inclusive": inclusive,
            "bootstrap": boot,
            "coverage": head.get("coverage"),
            "ci": head.get("ci"),
            "mae": head.get("mae"),
            "ref_mae": head.get("ref_mae"),
            "mae_diff": head.get("mae_diff"),
            "mae_diff_ci": head.get("mae_diff_ci"),
            "n": head.get("n", 0),
            "n_worlds": head.get("n_worlds", 0),
            "n_runs": head.get("n_runs", 0),
            "pass": verdict,
        },
        "rmse": head.get("rmse"),
        "bias": head.get("bias"),
        "width": head.get("width"),
        "raw_coverage": head.get("raw_coverage"),
        "ref_cov": head.get("ref_cov"),
        "ref_width": head.get("ref_width"),
        "qhat": _r(qhat),
    }
    if not pop.empty:
        for by in GROUPS:
            src = e if by == "kind" else pop
            out[f"by_{by}"] = group_table(src, by, band)
        out["per_run_coverage"] = per_run_coverage(pop)
        out["coverage_curve"] = coverage_curve(pop, qhat)
        out["quantile_reliability"] = quantile_reliability(pop, quantiles)
    stress = e[e["kind"] == "stress"]
    out["stress"] = headline(stress, boot, band) if not stress.empty else {"n": 0}
    out["all_worlds"] = headline(e, boot, band) if not e.empty else {"n": 0}
    out["flags"] = [
        f"{by}={row['group']} coverage {row['coverage']:.3f}"
        for by in GROUPS
        if by != "kind"
        for row in out.get(f"by_{by}", [])
        if row["flag"]
    ]
    return out


def trajet_variant(
    e_split: pd.DataFrame,
    calib_pred: np.ndarray,
    calib: pd.DataFrame,
    eta_cfg: Mapping,
) -> dict:
    """Per-trajet q-hat fitted on the calib population (D48 variant: reported, never
    served) and the coverage it would give on the evaluated population."""
    ccfg = eta_cfg.get("conformal") or {}
    level = float(ccfg.get("level", 0.80))
    grid = tuple(ccfg.get("grid_s", (-20.0, 60.0, 0.1)))
    q = conformal_by_group(calib_pred, calib[TARGET].to_numpy(float), calib["trajet"], level, grid)
    out = {}
    inside_all = []
    for trajet, g in e_split.groupby("trajet", sort=True):
        if trajet not in q:
            continue
        low, _, high = served_bounds(g[["q10", "q50", "q90"]].to_numpy(), q[trajet])
        inside = (g["y"].to_numpy() >= low) & (g["y"].to_numpy() <= high)
        inside_all.append(inside)
        out[trajet] = {"qhat": _r(q[trajet]), "coverage": _r(inside.mean()), "n": len(g)}
    pooled = _r(np.concatenate(inside_all).mean()) if inside_all else None
    return {"by_trajet": out, "pooled_coverage": pooled}


# -- post-hoc disclosures --------------------------------------------------------------
def unfinished_cars(dataset_dir: Path) -> dict | None:
    """Censoring diagnostic per split and world kind from `runs.parquet`: trajet cars
    still in the network when a run's drain hit its time limit (`n_unfinished_cars`, any
    insertion time, so an upper bound of the censored in-window trips), the runs whose
    drain timed out, and the recorded samples without an exit (`n_missing_exit`, D39).
    None when the dataset has no runs table."""
    path = Path(dataset_dir) / RUNS_FILE
    if not path.is_file():
        return None
    cols = ["split", "kind", "controller", "drain_timeout", "n_unfinished_cars", "n_missing_exit"]
    r = pd.read_parquet(path, columns=cols)
    by_split: dict = {}
    for (split, kind), g in r.groupby(["split", "kind"], sort=True):
        per_controller = g.groupby("controller", sort=True)["n_unfinished_cars"].sum()
        by_split.setdefault(str(split), {})[str(kind)] = {
            "runs": len(g),
            "drain_timeout_runs": int(g["drain_timeout"].astype(bool).sum()),
            "cars": int(g["n_unfinished_cars"].sum()),
            "missing_exit": int(g["n_missing_exit"].sum()),
            "cars_by_controller": {str(c): int(v) for c, v in per_controller.items()},
        }
    return {
        "source": RUNS_FILE,
        "definition": "trajet cars without a tripinfo record (still in the network when the "
        "drain timed out, any insertion time): no sample, absent from fit and evaluation",
        "cars": int(r["n_unfinished_cars"].sum()),
        "drain_timeout_runs": int(r["drain_timeout"].astype(bool).sum()),
        "by_split": by_split,
    }


def _reference_metrics(reference: pd.DataFrame, rows: pd.DataFrame, begin_s: int) -> dict:
    ref = apply_reference(reference, rows, begin_s)
    y = rows[TARGET].to_numpy(float)
    mean, p10, p90 = (ref[c].to_numpy(float) for c in ("mean_s", "p10_s", "p90_s"))
    return {
        "ref_mae": _r(_nanmean(np.abs(mean - y))),
        "ref_cov": _r(np.mean((y >= p10) & (y <= p90))) if len(y) else None,
        "ref_width": _r(_nanmean(p90 - p10)),
    }


def reference_sensitivity(
    train: pd.DataFrame,
    rows: pd.DataFrame,
    shipped: pd.DataFrame,
    mae: float | None,
    begin_s: int = BEGIN_S,
    used: str = "all",
) -> dict:
    """The shipped reference (fitted on `used` train worlds) against one refitted on the
    realistic train worlds only: the stress share of the train rows per day type, the
    cells of the served table that move, and the reference MAE, q10-q90 coverage and
    width of both on `rows` (the evaluated population). `mae` is the model's pooled MAE
    on `rows`, so `mae_diff` is its pooled point difference (no bootstrap: no model
    output is used here)."""
    kinds = train["kind"].astype(str).to_numpy()
    days = train["day_type"].astype(str).to_numpy()
    share = {str(d): _r(float(np.mean(kinds[days == d] == "stress"))) for d in sorted(set(days))}
    out: dict = {
        "shipped_population": used,
        "alternative_population": "realistic",
        "train_stress_share_by_day_type": share,
        "n_rows": len(rows),
    }
    if used == "realistic":
        out["note"] = "the shipped reference is already fitted on realistic worlds only"
        return out
    alt = fit_reference(reference_rows(train, "realistic"), begin_s)
    keys = ["route", "hour", "day_type"]
    m = shipped[[*keys, "mean_s", "p90_s"]].merge(
        alt[[*keys, "mean_s", "p90_s"]], on=keys, how="outer", suffixes=("_shipped", "_alt")
    )
    m["d_mean_s"] = m["mean_s_shipped"] - m["mean_s_alt"]
    m["d_p90_s"] = m["p90_s_shipped"] - m["p90_s_alt"]
    moved = m[(m["d_mean_s"].abs() > 1e-9) | (m["d_p90_s"].abs() > 1e-9) | m.isna().any(axis=1)]
    moved = moved.sort_values(keys, kind="mergesort")
    fields = ("mean_s_shipped", "mean_s_alt", "d_mean_s", "p90_s_shipped", "p90_s_alt", "d_p90_s")
    out["table_shift"] = [
        {
            "route": str(r.route),
            "hour": int(r.hour),
            "day_type": str(r.day_type),
            **{k: _r(getattr(r, k)) for k in fields},
        }
        for r in moved.itertuples()
    ]
    exact = moved[moved["hour"] != FALLBACK_HOUR]
    out["max_shift"] = {
        "d_mean_s": _r(exact["d_mean_s"].max()) if len(exact) else None,
        "d_p90_s": _r(exact["d_p90_s"].max()) if len(exact) else None,
    }
    if len(rows):
        alt_metrics = _reference_metrics(alt, rows, begin_s)
        alt_mae = alt_metrics["ref_mae"]
        out["shipped"] = _reference_metrics(shipped, rows, begin_s)
        out["alternative"] = {
            **alt_metrics,
            "mae_diff": None if mae is None or alt_mae is None else _r(float(mae) - alt_mae),
        }
        out["by_day_type"] = {
            str(d): {
                "n": len(g),
                "ref_mae_shipped": _reference_metrics(shipped, g, begin_s)["ref_mae"],
                "ref_mae_alt": _reference_metrics(alt, g, begin_s)["ref_mae"],
            }
            for d, g in rows.groupby(rows["day_type"].astype(str), sort=True)
        }
    return out


def post_hoc_disclosures(
    eta_cfg: Mapping,
    split: str,
    dataset_dir: Path,
    bundle_dir: Path,
    ev: Mapping,
    worlds: Sequence[int] | None = None,
) -> dict:
    """The `POST_HOC_KEYS` of eval.json, in that order. Reads `runs.parquet`, the target
    columns of the train worlds and of the evaluated population, and the bundle's
    reference table; never the model, so `--render-only` adds them to a frozen result
    without re-evaluating it."""
    meta = json.loads((Path(bundle_dir) / META_FILE).read_text(encoding="utf-8"))
    begin_s = int(meta.get("begin_s", BEGIN_S))
    out: dict = {"post_hoc": {"keys": list(POST_HOC_KEYS[1:]), "note": POST_HOC_NOTE}}
    out["unfinished"] = unfinished_cars(dataset_dir)
    try:
        cols = list(_SENSITIVITY_COLUMNS)
        train_worlds = (meta.get("worlds_override") or {}).get("train")
        train = (
            load_split(dataset_dir, "*", worlds=train_worlds, columns=cols)
            if train_worlds
            else load_split(dataset_dir, "train", columns=cols)
        )
        rows = (
            load_split(dataset_dir, "*", worlds=worlds, columns=cols)
            if worlds
            else load_split(dataset_dir, split, columns=cols)
        )
        population = (eta_cfg.get("a07") or {}).get("population", "realistic")
        if population != "all":
            rows = rows[rows["kind"].astype(str) == population]
        shipped = pd.read_parquet(Path(bundle_dir) / REFERENCE_FILE)
        used = meta.get("reference_population") or reference_population(eta_cfg)
        out["reference_sensitivity"] = reference_sensitivity(
            train, rows, shipped, (ev.get("a07") or {}).get("mae"), begin_s, used
        )
    except (FileNotFoundError, ValueError, KeyError) as err:
        log.warning("reference sensitivity skipped: %s", err)
        out["reference_sensitivity"] = None
    return out


# -- the page --------------------------------------------------------------------------
def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:.{nd}f}"


def _ci(ci, nd: int = 3) -> str:
    if not ci or ci[0] is None:
        return "n/a"
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def _table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


GROUP_TITLES = {
    "trajet": "Trajet",
    "controller": "Controller",
    "day_type": "Day type",
    "incident_active": "Active incident at entry",
    "ambulance_present": "Ambulance present or pending at entry",
    "kind": "World kind (all worlds)",
}


def _censoring_lines(ev: Mapping, split: str) -> list[str]:
    """Section 1: the trips censored at the drain cut (post-hoc key `unfinished`)."""
    unf = ev.get("unfinished")
    if not unf:
        return ["- Trips censored at the drain cut: not recorded (no `runs.parquet`)."]
    per = (unf.get("by_split") or {}).get(split) or {}
    parts = [
        f"{kind} {_fmt(int(v['cars']))} cars ({_fmt(int(v['drain_timeout_runs']))} of "
        f"{_fmt(int(v['runs']))} runs with a drain timeout)"
        for kind, v in sorted(per.items())
    ]
    a = ev.get("a07") or {}
    pop_cars = int((per.get(a.get("population")) or {}).get("cars", 0))
    bound = ""
    if pop_cars and a.get("coverage") is not None and a.get("n"):
        low = a["coverage"] * a["n"] / (a["n"] + pop_cars)
        bound = (
            f" If every censored {a['population']} trip fell outside the interval, the A-07 "
            f"coverage would be {low:.4f} instead of {a['coverage']:.4f}."
        )
    return [
        "- **Censored, not in the dataset (no D39 drop):** trajet cars still in the network "
        "when a run's drain hit its time limit have no tripinfo record, hence no sample "
        "(`runs.parquet` `n_unfinished_cars`, any insertion time, so an upper bound of the "
        f"censored in-window trips). {split}: " + ("; ".join(parts) or "none") + ". All "
        f"splits: {_fmt(int(unf.get('cars', 0)))} cars, "
        f"{_fmt(int(unf.get('drain_timeout_runs', 0)))} runs with a drain timeout. They are "
        "among the longest trips, and every metric on this page is computed without them." + bound,
    ]


def _stress_censoring_lines(ev: Mapping, split: str, s: Mapping) -> list[str]:
    """Section 7: how far the censored stress trips could move the stress coverage."""
    unf = ev.get("unfinished") or {}
    cars = int((((unf.get("by_split") or {}).get(split) or {}).get("stress") or {}).get("cars", 0))
    if not cars or s.get("coverage") is None or not s.get("n"):
        return []
    low = s["coverage"] * s["n"] / (s["n"] + cars)
    return [
        f"This coverage leaves out up to {_fmt(cars)} stress trips of this split that were "
        "still in the network at the drain cut (censored, section 1), the longest ones: if "
        f"every one fell outside the interval it would be {low:.4f} instead of "
        f"{s['coverage']:.4f}.",
        "",
    ]


def _alt(sens: Mapping | None) -> dict:
    return dict((sens or {}).get("alternative") or {})


def _reference_lines(meta: Mapping, ev: Mapping) -> list[str]:
    """Section 4: what the reference is fitted on, and the realistic-only sensitivity."""
    sens = ev.get("reference_sensitivity") or {}
    pop = meta.get("reference_population") or sens.get("shipped_population") or "all"
    if pop == "realistic":
        return [
            "- **Reference (D49, spec 11.4):** mean, p10 and p90 of `y_zone_s` per (trajet, "
            "hour, day type) on the **realistic** train worlds (`reference.population: "
            "realistic`), falling back to the trajet; it is also the degraded mode."
        ]
    share = {d: v for d, v in (sens.get("train_stress_share_by_day_type") or {}).items() if v}
    mix = (
        " Every stress world is labelled `tue_wed_thu_term` (D43), so "
        + ", ".join(f"{v:.1%} of the `{d}` train rows" for d, v in sorted(share.items()))
        + " come from stress worlds (multiplier U(0.6, 1.4)): for that day type the "
        "reference is not a historical mean of normal days."
        if share
        else ""
    )
    lines = [
        "- **Reference (D49, spec 11.4):** mean, p10 and p90 of `y_zone_s` per (trajet, hour, "
        "day type) on **all** train worlds, realistic and stress (`reference.population: "
        "all`), falling back to the trajet; it is also the degraded mode." + mix
    ]
    alt, shipped = _alt(sens), dict(sens.get("shipped") or {})
    if alt.get("ref_mae") is not None:
        a = ev.get("a07") or {}
        lines.append(
            "- **Sensitivity (post hoc, reported, never changes the verdict):** a reference "
            f"fitted on the realistic train worlds only has MAE {_fmt(alt['ref_mae'], 2)} s on "
            f"the A-07 population instead of {_fmt(shipped.get('ref_mae'), 2)} s, so the "
            f"model's pooled advantage is {_fmt(alt.get('mae_diff'), 2)} s instead of "
            f"{_fmt(a.get('mae_diff'), 2)} s; its q10-q90 coverage / width are "
            f"{_fmt(alt.get('ref_cov'))} / {_fmt(alt.get('ref_width'), 1)} s instead of "
            f"{_fmt(shipped.get('ref_cov'))} / {_fmt(shipped.get('ref_width'), 1)} s. The "
            "frozen comparison above is somewhat flattered by the mix."
        )
    return lines


def _sensitivity_rows(sens: Mapping | None) -> list[list]:
    alt = _alt(sens)
    if alt.get("ref_mae") is None:
        return []
    return [
        [
            "Reference MAE, realistic-only fit / model − it (s), post hoc",
            f"{_fmt(alt['ref_mae'], 2)} / {_fmt(alt.get('mae_diff'), 2)}",
            "",
            "not judged",
        ],
        [
            "Reference q10-q90 coverage / width, realistic-only fit (s), post hoc",
            f"{_fmt(alt.get('ref_cov'))} / {_fmt(alt.get('ref_width'), 1)}",
            "",
            "not judged",
        ],
    ]


def _degraded_lines(meta: Mapping, sens: Mapping | None) -> list[str]:
    """Section 8: how far the stress worlds move the served degraded-mode table."""
    sens = sens or {}
    pop = meta.get("reference_population") or sens.get("shipped_population") or "all"
    rows = [r for r in sens.get("table_shift") or [] if r["hour"] != FALLBACK_HOUR]
    if pop != "all" or not rows:
        return []
    top = sorted(rows, key=lambda r: -(r["d_mean_s"] or 0.0))[:3]
    cells = ", ".join(
        f"{r['route']} {r['hour']:02d}h mean {r['d_mean_s']:+.1f} s / p90 {r['d_p90_s']:+.1f} s"
        for r in top
    )
    mx = sens.get("max_shift") or {}
    days = sorted({r["day_type"] for r in rows})
    return [
        "- **The degraded table holds the stress worlds.** It is the bundle's reference, "
        f"fitted on realistic and stress train worlds: on {', '.join(f'`{d}`' for d in days)} "
        f"its means are up to {_fmt(mx.get('d_mean_s'), 1)} s and its p90 up to "
        f"{_fmt(mx.get('d_p90_s'), 1)} s above a realistic-only table ({cells}), so the "
        "approximate estimate and its bounds are too high and too wide on a normal day "
        "there. `reference.population: realistic` in eta.yaml (with a dated `changes` entry) "
        "fits the next bundle on realistic worlds only (post-field-visit retrain, D67); the "
        "frozen bundle keeps its table, so its evaluation stays the frozen one."
    ]


def render(
    ev: Mapping,
    meta: Mapping,
    manifest: Mapping,
    eta_cfg: Mapping,
    figures_rel: str | None = None,
) -> str:
    """docs/eta.md from eval.json, the bundle meta, the dataset manifest and eta.yaml."""
    a = ev["a07"]
    split = ev["split"]
    test = split == "test"
    seeds = eta_cfg.get("seeds") or {}
    band = a["band"]
    verdict = "PASS" if a["pass"] else "FAIL"
    L: list[str] = []
    title = "ETA model and its evaluation (Part 5, EF-13, EF-14, A-07)"
    L += [f"# {title}", ""]
    L += [
        "Generated by `python -m traffic_twin.ml.evaluate_eta --config configs/eta.yaml "
        f"--split {split}`. Do not edit by hand: rerun `tasks.py eta-eval`. Every number is "
        "**simulated** and **provisional** until the field visit (3-5 Nov 2026): the model "
        "learns the simulator (spec 11.2) running the ASSUMED 123 s signal program.",
        "",
    ]
    if not test:
        L += [
            f"> **Development split `{split}`, not the A-07 verdict.** The pre-registered "
            "verdict is computed once, on the test worlds, after the freeze (D50, D51)."
            + (
                f" Worlds override (development): {ev['worlds_override']}."
                if ev.get("worlds_override")
                else ""
            ),
            "",
        ]
    L += [
        f"**A-07 / EF-14 ({split} split, {a['population']} worlds): {verdict}.** Coverage "
        f"of the served 80 % interval {_fmt(a['coverage'])} (95 % CI {_ci(a['ci'])}, band "
        f"[{band[0]}, {band[1]}] inclusive); median MAE {_fmt(a['mae'], 1)} s vs reference "
        f"{_fmt(a['ref_mae'], 1)} s, difference CI {_ci(a['mae_diff_ci'], 1)} s; mean width "
        f"{_fmt(ev.get('width'), 1)} s.",
        "",
    ]
    if figures_rel:
        L += [f"![Coverage by group]({figures_rel}/coverage_by_group.png)", ""]

    # 1. data
    L += ["## 1. Data and provenance", ""]
    counts = manifest.get("counts") or {}
    L += [
        f"- Dataset `{meta.get('dataset_id')}` (`data/eta/<dataset_id>/`, gitignored), "
        f"sampler `{manifest.get('sampler_version', eta_cfg.get('sampler_version'))}`, feature "
        f"schema `{meta.get('feature_schema_version')}`, dataset code version "
        f"`{manifest.get('code_version', meta.get('dataset_code_version'))}`.",
        "- Each world is run under the three controllers `fixed` (S0), `actuated` (S1) and "
        "`rules` (S2) on the same demand (paired, D43); a world draws its kind (realistic "
        "0.7 / stress 0.3), day type (ASSUMED ratio profiles, D44), demand multiplier and arm "
        "scales, 0-2 ambulances and 0-2 incidents (lane slowdown to 2 m/s, ASSUMED, D45).",
    ]
    worlds = manifest.get("splits") or {}
    if counts:
        L += ["- Sample rows in the dataset (manifest; before the D39 drops):", ""]
        rows = []
        for sp in sorted(counts):
            per = counts[sp]
            rows.append(
                [sp, _fmt(len(worlds.get(sp) or [])),
                 *(_fmt(sum((per.get(c) or {}).values())) for c in ("fixed", "actuated", "rules"))]
            )  # fmt: skip
        L += _table(["Split", "Worlds", "fixed", "actuated", "rules"], rows) + [""]
    n = meta.get("n") or {}
    L += [
        f"- Model fit rows: train {_fmt(n.get('train'))}, val {_fmt(n.get('val'))}, calib "
        f"{_fmt(n.get('calib'))}; evaluated {split}: {_fmt(ev.get('n_rows'))} rows "
        f"({_fmt(a['n'])} in the A-07 population, {_fmt(a['n_worlds'])} worlds, "
        f"{_fmt(a['n_runs'])} runs). Dropped from the recorded samples (teleported / "
        f"recorded without an exit / no feature row): `{json.dumps(ev.get('dropped'))}`.",
        *_censoring_lines(ev, split),
        f"- Bundle `{ev.get('model_version')}` (sha256 `{str(ev.get('bundle_sha256'))[:12]}`), "
        f"model code version `{meta.get('code_version')}`, evaluation code version "
        f"`{ev.get('code_version')}`, xgboost {(meta.get('versions') or {}).get('xgboost')}.",
        "",
    ]
    # 2. target and features
    rest = ((eta_cfg.get("student") or {}).get("routes")) or {}
    L += ["## 2. Target and features", ""]
    L += [
        "- **Target `y_zone_s`** (D39, spec 11.1) = first second on the out-edge minus the "
        "physical insertion on the approach in-edge, whole seconds, for cars of the six "
        "trajets N_S, N_W, S_N, S_W, W_N, W_S inserted in [0, 7200) s. Teleported cars and "
        "recorded cars without an exit are dropped (counted). Cars still in the network when "
        "a run's drain hit its time limit have no record at all: they are censored, absent "
        "from the dataset, and counted separately (section 1). The insertion backlog beyond "
        "the zone is not in the target; the API flags it (`queue_beyond_zone`).",
        "- **Student ETA** (D40) = `y_zone_s` + a fixed rest of trip (ASSUMED, no "
        "uncertainty): "
        + "; ".join(f"{k} {v.get('label')} +{v.get('rest_s')} s" for k, v in rest.items())
        + ". The deadline is a user input; the advice follows spec 11.5 (D52).",
        f"- **Features** `{meta.get('feature_schema_version')}` (D41, "
        f"{len(meta.get('features') or [])} inputs), joined at `t_feat = floor(insert_t) - 1` "
        "(no look-ahead): time of day, day type, controller, trajet; per arm halting and "
        "vehicle counts, longest wait (ASSUMED camera), E2 jam length, exit halting, 1- and "
        "5-min E1 occupancy, 5-min flow; signal stage, transition, time in stage, clearance "
        "left, time to change and to green (D42); bus calls; ambulance presence and distance "
        "bins (ASSUMED V2I); incidents; pedestrians. Excluded: "
        f"{', '.join(meta.get('excluded_features') or []) or 'none'} (simulation-only "
        "insertion backlog, D41) and every world parameter.",
        "",
    ]
    # 3. splits
    L += ["## 3. Splits and pre-registration", ""]
    L += [
        f"`configs/eta.yaml` (sha256 `{str(ev.get('eta_yaml_sha256'))[:12]}`), pre-registered "
        f"**{eta_cfg.get('preregistered')}**, changes only through dated `changes` entries "
        f"({len(eta_cfg.get('changes') or [])} so far). Split by world, never by row (D38):",
        "",
    ]
    L += _table(
        ["Split", "World seeds", "Use"],
        [
            [k, f"{v[0]}-{v[1]}", use]
            for k, use in (
                ("train", "fit, reference table"),
                ("val", "early stopping, max_depth grid"),
                ("calib", "conformal q-hat (realistic worlds)"),
                ("reserve", "unused"),
                ("test", "A-07, once, after the freeze"),
            )
            if (v := seeds.get(k))
        ],
    )
    freeze = ev.get("freeze") or {}
    if freeze:
        L += [
            "",
            f"Freeze: `{freeze.get('model_version')}` frozen at {freeze.get('frozen_at')} "
            f"(bundle sha256 `{str(freeze.get('bundle_sha256'))[:12]}`), before any test world "
            "was generated; the test evaluation checks it (D51).",
            "",
        ]
    else:
        L += ["", "Freeze: not checked (development split).", ""]
    # 4. model
    params = meta.get("params") or {}
    cal = meta.get("calibration") or {}
    L += ["## 4. Model and calibration", ""]
    L += [
        f"- One XGBoost model, `reg:quantileerror`, quantiles {meta.get('quantiles')} (D47); "
        f"max_depth grid {[g.get('max_depth') for g in meta.get('grid') or []]} chosen on the "
        f"val pinball loss: depth {params.get('max_depth')}, best iteration "
        f"{meta.get('best_iteration')}, learning rate {params.get('learning_rate')}, "
        f"min_child_weight {params.get('min_child_weight')}, subsample "
        f"{params.get('subsample')}, colsample {params.get('colsample_bytree')}. Outputs "
        "sorted per row (crossing repair).",
        f"- **Interval (D48):** served `low = floor(q10 - q̂)`, `median = round(q50)`, "
        f"`high = ceil(q90 + q̂)`, clipped to `0 <= low <= median <= high`; q̂ = "
        f"**{_fmt(meta.get('qhat'), 1)} s**, the smallest value of the grid "
        f"{(meta.get('conformal') or {}).get('grid_s')} reaching pooled coverage 0.80 on the "
        f"calib realistic worlds (calib coverage {_fmt(cal.get('calib_coverage'))}, raw "
        f"quantiles {_fmt(cal.get('calib_raw_coverage'))}). The same bounds are stored, "
        "served and evaluated.",
        *_reference_lines(meta, ev),
        "",
    ]
    if figures_rel:
        L += [f"![Calibration curve]({figures_rel}/calibration_curve.png)", ""]
    # 5. verdict
    L += ["## 5. A-07 / EF-14 verdict", ""]
    L += [
        f"Population: {a['population']} worlds of the {split} split, all controllers "
        f"({_fmt(a['n'])} trips, {_fmt(a['n_worlds'])} worlds). Bootstrap: unit world, "
        f"{a['bootstrap'].get('n')} resamples, seed {a['bootstrap'].get('seed')}.",
        "",
    ]
    L += _table(
        ["Metric", "Value", "Rule", "Result"],
        [
            [
                "Coverage of the served interval",
                f"{_fmt(a['coverage'])} {_ci(a['ci'])}",
                f"in [{band[0]}, {band[1]}]",
                "ok"
                if a["coverage"] is not None and band[0] <= a["coverage"] <= band[1]
                else "out",
            ],
            [
                "MAE model − reference (s)",
                f"{_fmt(a['mae_diff'], 2)} {_ci(a['mae_diff_ci'], 2)}",
                "CI < 0",
                "ok" if (a["mae_diff_ci"][1] or 0) < 0 else "not below 0",
            ],
            ["MAE model / reference (s)", f"{_fmt(a['mae'], 2)} / {_fmt(a['ref_mae'], 2)}", "", ""],
            ["RMSE model (s)", _fmt(ev.get("rmse"), 2), "", ""],
            ["Mean width (s)", _fmt(ev.get("width"), 1), "", ""],
            ["Raw q10-q90 coverage", _fmt(ev.get("raw_coverage")), "", ""],
            [
                "Reference q10-q90 coverage / width (s)",
                f"{_fmt(ev.get('ref_cov'))} / {_fmt(ev.get('ref_width'), 1)}",
                "",
                "",
            ],
            *_sensitivity_rows(ev.get("reference_sensitivity")),
            ["**Verdict**", f"**{verdict}**", "both", ""],
        ],  # fmt: skip
    )
    L += [""]
    if figures_rel:
        L += [f"![MAE vs reference]({figures_rel}/mae_vs_reference.png)", ""]
    # 6. breakdowns
    L += ["## 6. Breakdowns and flags", ""]
    L += [
        f"A subgroup whose coverage falls outside [{band[0]}, {band[1]}] is a **flag**, not a "
        "failure (D50).",
        "",
    ]
    for by in GROUPS:
        rows = ev.get(f"by_{by}") or []
        if not rows:
            continue
        L += [f"**{GROUP_TITLES[by]}**", ""]
        L += _table(
            ["Group", "Trips", "Worlds", "Coverage", "MAE (s)", "Ref MAE (s)", "Width (s)", "Flag"],
            [
                [
                    r["group"],
                    _fmt(r["n"]),
                    _fmt(r["n_worlds"]),
                    _fmt(r["coverage"]),
                    _fmt(r["mae"], 1),
                    _fmt(r["ref_mae"], 1),
                    _fmt(r["width"], 1),
                    "**flag**" if r["flag"] else "",
                ]  # fmt: skip
                for r in rows
            ],
        )
        L += [""]
    flags = ev.get("flags") or []
    L += [f"Flags: {'; '.join(flags) if flags else 'none'}.", ""]
    var = ev.get("trajet_variant") or {}
    if var.get("by_trajet"):
        L += [
            "Per-trajet q̂ variant (fitted on calib, reported, never served): pooled coverage "
            f"{_fmt(var.get('pooled_coverage'))}; "
            + "; ".join(
                f"{k} q̂ {_fmt(v['qhat'], 1)} s → {_fmt(v['coverage'])}"
                for k, v in var["by_trajet"].items()
            )
            + ".",
            "",
        ]
    pr = (ev.get("per_run_coverage") or {}).get("summary") or {}
    if pr:
        L += [
            f"Coverage per run ({pr.get('n_runs')} runs): min {_fmt(pr.get('min'))}, p10 "
            f"{_fmt(pr.get('p10'))}, median {_fmt(pr.get('median'))}, p90 "
            f"{_fmt(pr.get('p90'))}, max {_fmt(pr.get('max'))}.",
            "",
        ]
        if figures_rel:
            L += [f"![Per-run coverage]({figures_rel}/per_run_coverage.png)", ""]
    # 7. stress
    s = ev.get("stress") or {}
    L += ["## 7. Stress worlds (reported, never judged)", ""]
    if s.get("n"):
        L += [
            f"{_fmt(s['n'])} trips in {_fmt(s['n_worlds'])} worlds (multiplier U(0.6, 1.4)): "
            f"coverage {_fmt(s['coverage'])} {_ci(s['ci'])}, MAE {_fmt(s['mae'], 1)} s vs "
            f"reference {_fmt(s['ref_mae'], 1)} s, width {_fmt(s['width'], 1)} s.",
            "",
            *_stress_censoring_lines(ev, split, s),
        ]
    else:
        L += ["No stress world in this split.", ""]
    # 8. degraded mode
    L += ["## 8. Degraded mode (ENF-03)", ""]
    L += [
        "- Predictions are precomputed per run (`<run_dir>/eta_prediction.parquet`, the "
        "Supabase `eta_prediction` columns) from the passive recorder's 5 s snapshots; the "
        "API is a lookup and never imports xgboost.",
        "- A run without snapshots, an unseen controller (e.g. `rl`) or an unseen day type "
        "gets no predictions: `/eta` serves the reference table (hour of t, run day type, "
        'falling back to the trajet) with `approximate: true` and the notice "estimation '
        'approximative"; without a bundle it answers `unavailable`.',
        *_degraded_lines(meta, ev.get("reference_sensitivity")),
        "- A bundle is checked on load (sha256 of model and reference, version string, "
        "feature list and categories); a failed check refuses the bundle.",
        "",
    ]
    # 9. limitations
    L += ["## 9. Limitations", ""]
    L += [
        "- **The model learns the simulator** (spec 11.2): SUMO's car-following, insertion "
        "and signal logic on the ASSUMED 123 s program and a single calibrated day "
        "(Tue-Thu, school term). Regenerate, retrain and re-evaluate after the field visit "
        "(D67); the API keeps `provisional: true` until then.",
        "- **Day types** Mon/Fri term and school holiday are ASSUMED ratio profiles of the "
        "sensor extract (D44; the holiday profile is summer only); Saturday is not modelled.",
        "- **Detection** is ASSUMED ideal: longest wait per arm (camera), ambulance distance "
        "(V2I), bus calls; the insertion backlog is excluded as simulation-only.",
        "- **Incidents** are a lane slowdown to 2 m/s (ASSUMED, D45), not a closure.",
        "- The rest of the student trip (120 s) is ASSUMED and carries no uncertainty (D40).",
        "",
    ]
    return "\n".join(L).rstrip() + "\n"


# -- CLI -------------------------------------------------------------------------------
def run_evaluation(
    eta_cfg: dict,
    split: str,
    dataset_dir: Path,
    bundle_dir: Path,
    models_dir: Path,
    loader=None,
    worlds: Sequence[int] | None = None,
) -> tuple[dict, dict, dict]:
    """(eval dict, bundle meta, manifest) for one split; test needs the freeze. `loader`
    goes to `load_bundle` (default: xgboost). `worlds` evaluates those world seeds from
    any split instead (development override of the gate; never for test)."""
    if worlds is not None and split == "test":
        raise ValueError("the test split is evaluated whole, without a worlds override")
    from traffic_twin.ml.predict import load_bundle, predict_quantiles
    from traffic_twin.sim.results import code_version

    freeze = check_freeze(bundle_dir, models_dir) if split == "test" else None
    bundle = load_bundle(bundle_dir, loader)
    manifest = read_manifest(dataset_dir)
    begin_s = int(bundle.meta.get("begin_s", BEGIN_S))
    df = load_split(dataset_dir, "*", worlds=worlds) if worlds else load_split(dataset_dir, split)
    pred = predict_quantiles(bundle, df)
    ev = evaluate(df, pred, bundle.qhat, bundle.reference, eta_cfg, begin_s, split)
    try:
        calib_worlds = (bundle.meta.get("worlds_override") or {}).get("calib")
        calib = (
            load_split(dataset_dir, "*", worlds=calib_worlds)
            if calib_worlds
            else load_split(dataset_dir, "calib")
        )
        pop = (eta_cfg.get("conformal") or {}).get("population", "realistic")
        if pop != "all":
            calib = calib[calib["kind"].astype(str) == pop]
        e = eval_frame(df, pred, bundle.qhat, bundle.reference, begin_s)
        a07_pop = (eta_cfg.get("a07") or {}).get("population", "realistic")
        e = e if a07_pop == "all" else e[e["kind"] == a07_pop]
        ev["trajet_variant"] = trajet_variant(e, predict_quantiles(bundle, calib), calib, eta_cfg)
    except (FileNotFoundError, ValueError) as err:
        log.warning("per-trajet q-hat variant skipped: %s", err)
    ev.update(
        {
            "n_rows": len(df),
            "n": {split: len(df), **(bundle.meta.get("n") or {})},
            "dropped": df.attrs.get("dropped"),
            "dataset_id": bundle.meta.get("dataset_id"),
            "model_version": bundle.version,
            "bundle_sha256": bundle_sha256(bundle_dir),
            "code_version": code_version(),
            "eta_yaml_sha256": text_sha256(ETA_CONFIG) if ETA_CONFIG.is_file() else None,
            "freeze": freeze,
            "worlds_override": [int(w) for w in worlds] if worlds else None,
        }
    )
    ev.update(post_hoc_disclosures(eta_cfg, split, dataset_dir, bundle_dir, ev, worlds))
    return ev, bundle.meta, manifest


def add_post_hoc(ev: dict, eta_cfg: Mapping, dataset_dir: Path, bundle_dir: Path) -> list[str]:
    """Add the post-hoc disclosure keys an eval dict lacks (written before they existed),
    in the order `run_evaluation` writes them; no other key is touched and no model is
    loaded. Returns the keys added (empty when none was missing)."""
    if all(k in ev for k in POST_HOC_KEYS):
        return []
    if ev.get("model_version") and ev.get("model_version") != Path(bundle_dir).name:
        raise ValueError(f"eval of {ev.get('model_version')} is not from bundle {bundle_dir}")
    worlds = ev.get("worlds_override")
    extra = post_hoc_disclosures(eta_cfg, ev["split"], dataset_dir, bundle_dir, ev, worlds)
    added = [k for k in POST_HOC_KEYS if k not in ev]
    for k in POST_HOC_KEYS:
        ev.pop(k, None)
        ev[k] = extra[k]
    return added


def write_outputs(
    ev: Mapping,
    meta: Mapping,
    manifest: Mapping,
    eta_cfg: Mapping,
    docs: Path | None,
    figures: Path | None,
) -> None:
    """The figures (`viz.eta`) and the page from one eval dict."""
    figures_rel = None
    if figures is not None:
        from traffic_twin.viz.eta import make_figures

        for p in make_figures(ev, figures):
            print(f"wrote {p.as_posix()}")
        if docs is not None:
            figures_rel = Path(os.path.relpath(figures, docs.parent)).as_posix()
    if docs is not None:
        docs.parent.mkdir(parents=True, exist_ok=True)
        text = render(ev, meta, manifest, eta_cfg, figures_rel)
        docs.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {docs.as_posix()}")


def render_only(
    eta_cfg: dict,
    split: str,
    dataset_dir: Path,
    bundle_dir: Path,
    models_dir: Path,
    eval_path: Path,
) -> tuple[dict, dict, dict, list[str]]:
    """(stored eval dict, meta, manifest, post-hoc keys added) without re-evaluating:
    the stored result is read back (the freeze still checked for test) and only the
    missing post-hoc keys are computed and written into it."""
    if not Path(eval_path).is_file():
        raise FileNotFoundError(f"no {eval_path}: evaluate the split first")
    if split == "test":
        check_freeze(bundle_dir, models_dir)
    ev = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    if ev.get("split") != split:
        raise ValueError(f"{eval_path} holds split {ev.get('split')!r}, not {split!r}")
    if ev.get("bundle_sha256") and ev["bundle_sha256"] != bundle_sha256(bundle_dir):
        raise ValueError(f"{eval_path} was computed on another state of {bundle_dir}")
    added = add_post_hoc(ev, eta_cfg, dataset_dir, bundle_dir)
    if added:
        Path(eval_path).write_text(json.dumps(ev, indent=1) + "\n", encoding="utf-8")
        print(f"added post-hoc keys {added} to {Path(eval_path).as_posix()}")
    meta = json.loads((Path(bundle_dir) / META_FILE).read_text(encoding="utf-8"))
    return ev, meta, read_manifest(dataset_dir), added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ETA_CONFIG)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--dataset", type=Path, default=None, help="data/eta/<dataset_id>")
    parser.add_argument("--models", type=Path, default=None, help="default outputs.models")
    parser.add_argument("--bundle", type=Path, default=None, help="default: LATEST")
    parser.add_argument("--out", type=Path, default=None, help="eval json (default: bundle)")
    parser.add_argument("--docs", type=Path, default=None, help="write the page (docs/eta.md)")
    parser.add_argument("--figures", type=Path, default=None, help="docs/figures/eta")
    parser.add_argument(
        "--worlds", type=int, nargs="+", default=None,
        help="development override (not test): evaluate these world seeds from any split",
    )  # fmt: skip
    parser.add_argument(
        "--render-only", action="store_true",
        help="no re-evaluation: re-render --docs / --figures from the stored eval json and "
        "add its missing post-hoc disclosures (test: the freeze is still checked)",
    )  # fmt: skip
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from traffic_twin.ml.predict import BundleError, latest_bundle

    eta_cfg = load_config(args.config)
    models_dir = args.models or Path((eta_cfg.get("outputs") or {}).get("models", ETA_MODELS_DIR))
    bundle_dir = args.bundle or latest_bundle(models_dir)
    if bundle_dir is None:
        print(f"no ETA bundle under {models_dir} (LATEST): run eta-train", file=sys.stderr)
        return 1
    if args.split == "test":  # before touching any test row
        try:
            check_freeze(bundle_dir, models_dir)
        except ValueError as e:
            print(f"test evaluation refused: {e}", file=sys.stderr)
            return 1
    out = args.out or Path(bundle_dir) / (
        EVAL_FILE if args.split == "test" else f"eval_{args.split}.json"
    )
    if args.render_only:
        if args.worlds is not None:
            parser.error("--render-only reads the stored result: no --worlds")
        try:
            dataset = resolve_dataset(eta_cfg, args.dataset)
            ev, meta, manifest, _ = render_only(
                eta_cfg, args.split, dataset, bundle_dir, models_dir, out
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"render failed: {e}", file=sys.stderr)
            return 1
        write_outputs(ev, meta, manifest, eta_cfg, args.docs, args.figures)
        return 0
    try:
        dataset = resolve_dataset(eta_cfg, args.dataset)
        ev, meta, manifest = run_evaluation(
            eta_cfg, args.split, dataset, bundle_dir, models_dir, worlds=args.worlds
        )
    except (FileNotFoundError, ValueError, BundleError) as e:
        print(f"evaluation failed: {e}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out.as_posix()}")
    write_outputs(ev, meta, manifest, eta_cfg, args.docs, args.figures)
    a = ev["a07"]
    print(
        f"A-07 ({args.split}, {a['population']}): coverage {a['coverage']} {a['ci']}, "
        f"MAE {a['mae']} vs ref {a['ref_mae']} (diff CI {a['mae_diff_ci']}): "
        f"{'PASS' if a['pass'] else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

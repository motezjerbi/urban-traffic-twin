"""Figures of the ETA evaluation (Part 5, EF-14, A-07), in the project style
(`viz.style`), drawn from `eval.json` alone (`ml.evaluate_eta`):

- `coverage_by_group.png`: coverage of the served 80 % interval per subgroup of the A-07
  population (trajet, controller, day type, active incident at entry, ambulance present
  or pending) against the A-07 band, flagged groups in red;
- `calibration_curve.png`: coverage of the served bounds when q-hat moves (the chosen
  q-hat marked), and the observed share below each raw quantile;
- `mae_vs_reference.png`: median MAE of the model and mean MAE of the reference per
  trajet and pooled;
- `per_run_coverage.png`: histogram of the coverage of each run against the band.

    python -m traffic_twin.viz.eta --eval data/models/eta/<version>/eval.json
        --out docs/figures/eta
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch

from traffic_twin.viz import style

FIGURES = (
    "coverage_by_group.png", "calibration_curve.png", "mae_vs_reference.png",
    "per_run_coverage.png",
)  # fmt: skip
PANELS = (
    ("trajet", "Trajet"),
    ("controller", "Controller"),
    ("day_type", "Day type"),
    ("incident_active", "Active incident"),
    ("ambulance_present", "Ambulance"),
)  # every D50 breakdown of the A-07 population (world kind is not judged: not drawn)
MODEL_COLOR = style.USER_COLORS["car"]  # the ETA serves the car / student user
REFERENCE_COLOR = style.MUTED
BAND_FILL = "#ecebe6"
NOMINAL = 0.80


def _band(ev: Mapping) -> tuple[float, float]:
    b = (ev.get("a07") or {}).get("band") or (0.75, 0.85)
    return float(b[0]), float(b[1])


def _subtitle(ev: Mapping) -> str:
    a = ev.get("a07") or {}
    return (
        f"{ev.get('split', '?')} split, {a.get('population', '?')} worlds · "
        f"{a.get('n', 0):,} trips, {a.get('n_worlds', 0)} worlds · model "
        f"{ev.get('model_version', '?')} · simulated, provisional"
    )


def _shade_band(ax, band: tuple[float, float], axis: str = "y") -> None:
    span = ax.axhspan if axis == "y" else ax.axvspan
    span(band[0], band[1], color=BAND_FILL, zorder=0, lw=0)


def _group_label(key: str, group: str) -> str:
    if key == "controller":
        return style.controller_label(group)
    if key == "trajet":
        return group.replace("_", "→")
    if key == "day_type":
        return {"tue_wed_thu_term": "Tue-Thu", "mon_fri_term": "Mon/Fri",
                "school_holiday": "holiday"}.get(group, group.replace("_", " "))  # fmt: skip
    return group.replace("_", " ")


def plot_coverage_by_group(ev: Mapping, out: Path) -> Path:
    """One panel per grouping of the A-07 population: a dot per group (its coverage),
    the A-07 band shaded, a flagged group (outside the band) in red."""
    band = _band(ev)

    def size(key: str) -> float:  # panel width: room for its tick labels
        labels = [_group_label(key, r["group"]) for r in ev.get(f"by_{key}") or []]
        return max(len(labels), 1) * max(1.0, max(map(len, labels), default=0) / 7.0) + 1.0

    sizes = [size(key) for key, _ in PANELS]
    fig, head, axes = style.figure(
        height=3.2, ncols=len(PANELS), squeeze=False, sharey=True,
        gridspec_kw={"width_ratios": sizes},
    )  # fmt: skip
    axes = axes[0]
    every = [r["coverage"] for key, _ in PANELS for r in ev.get(f"by_{key}") or []]
    finite = [float(c) for c in every if c is not None and np.isfinite(c)]
    lo, hi = min([band[0] - 0.05, *finite]), max([band[1] + 0.05, *finite])
    flagged = False
    for ax, (key, title) in zip(axes, PANELS, strict=True):
        rows = ev.get(f"by_{key}") or []
        flagged = flagged or any(r["flag"] for r in rows)
        _shade_band(ax, band)
        x = np.arange(len(rows))
        cov = [r["coverage"] if r["coverage"] is not None else np.nan for r in rows]
        colors = [style.CRITICAL if r["flag"] else MODEL_COLOR for r in rows]
        ax.scatter(x, cov, s=46, color=colors, zorder=3)
        for xi, r in zip(x, rows, strict=True):
            if r["coverage"] is not None:
                ax.annotate(
                    f"{r['coverage']:.2f}", (xi, r["coverage"]), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom", color=style.INK,
                    fontsize=style.FONT_SIZE - 1,
                )  # fmt: skip
        ax.set_xticks(x, [_group_label(key, r["group"]) for r in rows])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(max(lo - 0.02, 0.0), min(hi + 0.05, 1.0))
        ax.set_xlim(-0.6, max(len(rows), 1) - 0.4)
        ax.set_title(title)
        ax.set_ylabel("Coverage of the 80 % interval" if key == "trajet" else "")
    handles = [
        Patch(color=BAND_FILL, label=f"A-07 band [{band[0]:.2f}, {band[1]:.2f}]"),
        Patch(color=MODEL_COLOR, label="in band"),
    ]
    if flagged:  # no legend entry for a marker no point uses
        handles.append(Patch(color=style.CRITICAL, label="flag (outside the band)"))
    style.header(head, "Interval coverage by subgroup", _subtitle(ev), handles)
    return style.savefig(fig, out)


def plot_calibration_curve(ev: Mapping, out: Path) -> Path:
    """Left: coverage vs q-hat (served bounds); right: observed share below each raw
    quantile vs its nominal level."""
    band = _band(ev)
    fig, head, axes = style.figure(height=3.3, ncols=2, squeeze=False)
    ax, ax2 = axes[0]
    curve = ev.get("coverage_curve") or []
    _shade_band(ax, band)
    if curve:
        q = [c["qhat"] for c in curve]
        cov = [c["coverage"] for c in curve]
        ax.plot(q, cov, color=MODEL_COLOR, zorder=3)
    qhat = ev.get("qhat")
    if qhat is not None:
        ax.axvline(qhat, color=style.INK_2, lw=1.0, ls="--", zorder=2)
        ax.annotate(
            f"q̂ = {qhat:.1f} s", (qhat, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -4),
            textcoords="offset points", ha="left", va="top", color=style.INK_2,
            fontsize=style.FONT_SIZE - 1,
        )  # fmt: skip
    cov_head = (ev.get("a07") or {}).get("coverage")
    if qhat is not None and cov_head is not None:
        ax.scatter([qhat], [cov_head], s=46, color=MODEL_COLOR, edgecolor=style.INK, zorder=4)
    ax.set_title("Coverage of the served bounds vs q̂")
    ax.set_xlabel("q̂ (s): widening of both bounds")
    ax.set_ylabel("Coverage")
    rel = ev.get("quantile_reliability") or []
    ax2.plot([0, 1], [0, 1], color=style.AXIS, lw=1.0, zorder=1)
    if rel:
        ax2.plot(
            [r["quantile"] for r in rel], [r["observed"] for r in rel], marker="o",
            color=MODEL_COLOR, zorder=3,
        )  # fmt: skip
        for r in rel:
            if r["observed"] is not None:
                ax2.annotate(
                    f"{r['observed']:.2f}", (r["quantile"], r["observed"]), xytext=(6, -2),
                    textcoords="offset points", ha="left", va="top", color=style.INK,
                    fontsize=style.FONT_SIZE - 1,
                )  # fmt: skip
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("Raw quantiles: observed share below")
    ax2.set_xlabel("Nominal quantile")
    ax2.set_ylabel("Observed share of trips at or below")
    style.header(
        head, "Calibration of the ETA interval", _subtitle(ev),
        [Patch(color=BAND_FILL, label="A-07 band"), Patch(color=MODEL_COLOR, label="model")],
    )  # fmt: skip
    return style.savefig(fig, out)


def plot_mae_vs_reference(ev: Mapping, out: Path) -> Path:
    """Paired bars per trajet and pooled: model median MAE vs reference mean MAE."""
    a = ev.get("a07") or {}
    rows = [*(ev.get("by_trajet") or []),
            {"group": "all", "mae": a.get("mae"), "ref_mae": a.get("ref_mae")}]  # fmt: skip
    fig, head, ax = style.figure(height=3.3)
    x = np.arange(len(rows))
    w = 0.36
    top = 0.0
    for off, key, color in ((-w / 2, "ref_mae", REFERENCE_COLOR), (w / 2, "mae", MODEL_COLOR)):
        vals = [r.get(key) if r.get(key) is not None else np.nan for r in rows]
        ax.bar(x + off, vals, width=w, color=color, zorder=2)
        for xi, v in zip(x, vals, strict=True):
            if np.isfinite(v):
                ax.annotate(
                    f"{v:.0f}", (xi + off, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", color=style.INK, fontsize=style.FONT_SIZE - 1,
                )  # fmt: skip
                top = max(top, v)
    ax.set_xticks(x, [r["group"] for r in rows])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, top * 1.18 if top > 0 else 1)
    ax.set_ylabel("MAE (s)")
    ci = a.get("mae_diff_ci") or [None, None]
    note = (
        f"pooled difference {a.get('mae_diff', float('nan')):.1f} s, 95 % CI "
        f"[{ci[0]:.1f}, {ci[1]:.1f}] (bootstrap over worlds)"
        if ci[0] is not None and a.get("mae_diff") is not None
        else "pooled difference n/a"
    )
    style.header(
        head, "Median MAE: model vs reference (mean per trajet x hour x day type)",
        f"{_subtitle(ev)} · {note}",
        [Patch(color=REFERENCE_COLOR, label="reference"), Patch(color=MODEL_COLOR, label="model")],
    )  # fmt: skip
    return style.savefig(fig, out)


def plot_per_run_coverage(ev: Mapping, out: Path) -> Path:
    """Histogram of the coverage of each run (A-07 population) with the band."""
    band = _band(ev)
    runs = (ev.get("per_run_coverage") or {}).get("runs") or []
    cov = np.array([r["coverage"] for r in runs if r.get("coverage") is not None], dtype=float)
    fig, head, ax = style.figure(height=3.2)
    _shade_band(ax, band, axis="x")
    if len(cov):
        lo, hi = min(cov.min(), band[0]) - 0.02, max(cov.max(), band[1]) + 0.02
        bins = np.arange(np.floor(lo * 50) / 50, hi + 0.02, 0.02)
        counts, _, _ = ax.hist(cov, bins=bins, color=MODEL_COLOR, edgecolor=style.SURFACE,
                               zorder=2)  # fmt: skip
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, max(counts.max(), 1) * 1.18)  # room for the pooled label
    pooled = (ev.get("a07") or {}).get("coverage")
    if pooled is not None:
        ax.axvline(pooled, color=style.INK, lw=1.2, zorder=3)
        ax.annotate(
            f"pooled {pooled:.3f}", (pooled, 1.0), xycoords=("data", "axes fraction"),
            xytext=(4, -4), textcoords="offset points", ha="left", va="top", color=style.INK,
            fontsize=style.FONT_SIZE - 1,
        )  # fmt: skip
    ax.set_xlabel("Coverage of the served interval in one run")
    ax.set_ylabel("Runs")
    style.header(
        head, f"Coverage per run ({len(cov)} runs)", _subtitle(ev),
        [Patch(color=BAND_FILL, label="A-07 band (pooled rule)"),
         Patch(color=MODEL_COLOR, label="runs")],
    )  # fmt: skip
    return style.savefig(fig, out)


def make_figures(ev: Mapping, out_dir: Path) -> list[Path]:
    """Every figure of FIGURES into `out_dir`, in that order."""
    out_dir = Path(out_dir)
    fns = (plot_coverage_by_group, plot_calibration_curve, plot_mae_vs_reference,
           plot_per_run_coverage)  # fmt: skip
    return [fn(ev, out_dir / name) for fn, name in zip(fns, FIGURES, strict=True)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval", type=Path, required=True, help="eval.json of a bundle")
    parser.add_argument("--out", type=Path, default=Path("docs/figures/eta"))
    args = parser.parse_args(argv)
    if not args.eval.is_file():
        raise SystemExit(f"no {args.eval}: run ml.evaluate_eta first")
    ev = json.loads(args.eval.read_text(encoding="utf-8"))
    for p in make_figures(ev, args.out):
        print(f"wrote {p.as_posix()}")


if __name__ == "__main__":
    main()

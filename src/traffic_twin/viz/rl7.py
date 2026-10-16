"""Figures of the Part 7 page (`docs/rl_training.md`, D83), in the project style
(`viz.style`), drawn from the Part 7 artifacts under data/rl7/:

- `learning_bands.png`: validation cost on the selection windows per decision step,
  mean and seed band per config (stage T, stage F), with the baselines;
- `kpi_curves.png`: per-class validation totals per decision step (10.4);
- `holdout_curves.png`: holdout cost of the kept checkpoints of the F runs;
- `stage_t.png`: stage-T scores (late mean per seed) and the D73 rule threshold;
- `pareto.png`: weight sensitivity, car time loss vs bus delay and car time loss vs
  pedestrian curb mean (2 panels, points coloured by direction, S0/S1/S2 squares, the
  front joined);
- `stress.png`: stress cells, S3 cost in % vs S1 and S2;
- `s3_forest.png`: dev-test paired differences (HL and CI) per KPI and pair;
- `interventions.png`: guard interventions per hour, split cut / replaced /
  cut_foreseeable (D77).

`make_figures(data, figures)` draws the ones whose inputs are in `data` (the mapping
`rl.report7.collect` builds from the artifacts; keys in `make_figures`); `rl.report7`
calls it. Every plot function takes plain frames / dicts, so the tests feed synthetic
inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from traffic_twin.viz import style
from traffic_twin.viz.rl import policy_color, policy_name

FIGURES = (
    "learning_bands",
    "kpi_curves",
    "holdout_curves",
    "stage_t",
    "pareto",
    "stress",
    "s3_forest",
    "interventions",
)
RL = style.CONTROLLER_COLORS["rl"]
# Part 7 run configs (D73): greens for the DQN family, a brown for PPO, grey for T0
CONFIG_STYLE = {
    "t0": {"color": style.INK_2, "ls": "-"},
    "t1": {"color": RL, "ls": "-"},
    "v1": {"color": "#0a5a0a", "ls": "--"},
    "v2": {"color": "#7cc07c", "ls": "-."},
    "p1": {"color": "#8a5a00", "ls": "-"},
}
SEED_SHADES = ("#008300", "#3f9f3f", "#7cc07c", "#0a5a0a", "#a8d8a8")
# weight-change direction of a sensitivity setting (rl.sensitivity.direction)
DIRECTION_STYLE = {
    "base": {"color": RL, "marker": "o", "label": "base weights (T1)"},
    "transit_ped": {"color": "#0a5a0a", "marker": "^", "label": "more bus / pedestrian weight"},
    "car": {"color": "#7cc07c", "marker": "v", "label": "more car weight"},
    "mixed": {"color": style.INK_2, "marker": "D", "label": "mixed change"},
}
KPI_CURVE_KEYS = (
    ("car_s", "Cars waiting", "k veh·s / window", 1000.0),
    ("bus_s", "Buses waiting", "k veh·s / window", 1000.0),
    ("pedestrian_s", "Pedestrians waiting", "k person·s / window", 1000.0),
    ("penalty_s", "Curb wait > θ", "s / window", 1.0),
    ("interventions", "Guard interventions", "per window", 1.0),
)
BASELINE_LINES = ("S0", "S1", "S2")
PARETO_X = "car/time_loss_mean"
PARETO_Y = ("bus/delay_mean", "pedestrian/ped_curb_wait_mean")
AXIS_LABELS = {
    "car/time_loss_mean": "car time loss, mean (s)",
    "bus/delay_mean": "bus delay, mean (s)",
    "pedestrian/ped_curb_wait_mean": "pedestrian curb wait, mean (s)",
}
SPLIT_STYLE = (
    ("cut_only", "cut (not foreseeable)", RL),
    ("cut_foreseeable", "cut, foreseeable", "#7cc07c"),
    ("replaced", "replaced", style.INK_2),
)


def config_style(variant: str, i: int = 0) -> dict:
    return CONFIG_STYLE.get(variant, {"color": SEED_SHADES[i % len(SEED_SHADES)], "ls": ":"})


def _baselines(runs: Sequence[Mapping]) -> dict[str, float]:
    for m in runs:
        if m.get("baselines"):
            return {k: float(v) for k, v in m["baselines"].items() if k in BASELINE_LINES}
    return {}


def _baseline_lines(ax, base: Mapping[str, float], scale: float = 1000.0) -> None:
    """Dotted baseline levels with their names right of the axis, pushed apart."""
    items = sorted(
        ((k, v / scale) for k, v in base.items() if math.isfinite(v)), key=lambda kv: kv[1]
    )
    for label, y in items:
        ax.axhline(y, color=policy_color(label), lw=1.0, ls=":", zorder=1)
    lo, hi = ax.get_ylim()
    gap, last = 0.05 * (hi - lo), -np.inf
    for label, y in items:
        y = max(y, last + gap)
        last = y
        ax.annotate(
            policy_name(label),
            (1.0, y),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=style.FONT_SIZE - 2,
            color=style.INK_2,
            annotation_clip=False,
        )


def band(runs: Sequence[Mapping], key: str = "cost") -> pd.DataFrame:
    """step, mean, lo, hi, n of a validation column across runs (seeds)."""
    cols = [
        m["validation"].set_index("step")[key]
        for m in runs
        if m.get("validation") is not None and key in m["validation"]
    ]
    if not cols:
        return pd.DataFrame(columns=["step", "mean", "lo", "hi", "n"])
    wide = pd.concat(cols, axis=1).sort_index()
    return (
        pd.DataFrame(
            {
                "mean": wide.mean(axis=1),
                "lo": wide.min(axis=1),
                "hi": wide.max(axis=1),
                "n": wide.count(axis=1),
            }
        )
        .rename_axis("step")
        .reset_index()
    )


def groups(runs: Sequence[Mapping]) -> dict[tuple[str, str], list[Mapping]]:
    """(stage, variant) -> runs, in first-seen order."""
    out: dict[tuple[str, str], list[Mapping]] = {}
    for m in runs:
        out.setdefault((str(m.get("stage", "?")), str(m.get("variant", "?"))), []).append(m)
    return out


def plot_learning_bands(runs: Sequence[Mapping], out: Path) -> Path:
    """Validation cost per step: one panel per stage (T, F), mean line and min-max seed
    band per config, open circles at each run's pick (D72), dotted baselines."""
    runs = [m for m in runs if m.get("validation") is not None and not m["validation"].empty]
    stages = [s for s in ("T", "F") if any(m.get("stage") == s for m in runs)] or ["?"]
    fig, head, axes = style.figure(height=3.8, ncols=len(stages), squeeze=False)
    handles, seen = [], set()
    for ax, stage in zip(axes[0], stages, strict=True):
        sel = [m for m in runs if m.get("stage", "?") == stage or stage == "?"]
        for i, ((_, variant), ms) in enumerate(groups(sel).items()):
            st = config_style(variant, i)
            b = band(ms)
            if b.empty:
                continue
            ax.fill_between(
                b["step"], b["lo"] / 1000, b["hi"] / 1000, color=st["color"], alpha=0.15, lw=0
            )
            ax.plot(b["step"], b["mean"] / 1000, lw=1.8, **st)
            for m in ms:
                v = m["validation"]
                p = v[v["step"] == m.get("pick_step")]
                if len(p):
                    ax.scatter(
                        p["step"],
                        p["cost"] / 1000,
                        s=34,
                        facecolor="none",
                        edgecolor=st["color"],
                        lw=1.2,
                        zorder=5,
                    )
            algo = str(ms[0].get("algo", "")).upper()
            name = f"{'F ' if stage == 'F' else ''}{variant.upper()} {algo} (n={len(ms)})"
            if name not in seen:
                seen.add(name)
                handles.append(Line2D([], [], lw=2, label=name, **st))
        _baseline_lines(ax, _baselines(sel))
        ax.set(xlabel="decision steps", ylabel="mean episode cost (k weighted person·s)")
        ax.ticklabel_format(axis="x", style="plain")
        title = {"T": "Stage T (tuning)", "F": "Stage F (final trainings)"}.get(stage, "Runs")
        ax.set_title(f"{title}: selection windows", fontsize=style.FONT_SIZE)
    style.header(
        head,
        "Learning curves with seed bands",
        "Line = mean over training seeds, band = min-max; open circle = each run's pick "
        "(D72); dotted = baselines on the same windows; lower is better",
        handles=handles[:8],
        ncols=min(len(handles), 4) or None,
    )
    return style.savefig(fig, out)


def plot_kpi_curves(runs: Sequence[Mapping], out: Path) -> Path:
    """Per-class validation totals per decision step (spec 10.4): mean and seed band per
    config, baselines dotted (the run's baselines.csv, mean per window)."""
    runs = [m for m in runs if m.get("validation") is not None and not m["validation"].empty]
    keys = [k for k in KPI_CURVE_KEYS if any(k[0] in m["validation"] for m in runs)]
    fig, head, axes = style.figure(
        height=2.9, ncols=len(keys), squeeze=False, width=max(style.WIDTH_IN, 2.4 * len(keys))
    )
    handles = []
    for i, ((stage, variant), ms) in enumerate(groups(runs).items()):
        st = config_style(variant, i)
        for ax, (key, _, _, scale) in zip(axes[0], keys, strict=True):
            b = band(ms, key)
            if b.empty:
                continue
            ax.fill_between(
                b["step"], b["lo"] / scale, b["hi"] / scale, color=st["color"], alpha=0.15, lw=0
            )
            ax.plot(b["step"], b["mean"] / scale, lw=1.6, **st)
        handles.append(Line2D([], [], lw=2, label=f"{stage} {variant.upper()} (n={len(ms)})", **st))
    base = next((m.get("baseline_totals") for m in runs if m.get("baseline_totals")), None) or {}
    for ax, (key, title, unit, scale) in zip(axes[0], keys, strict=True):
        for label in BASELINE_LINES:
            v = (base.get(label) or {}).get(key)
            if v is not None and math.isfinite(float(v)):
                ax.axhline(float(v) / scale, color=policy_color(label), lw=1.0, ls=":", zorder=1)
        ax.set_title(title, fontsize=style.FONT_SIZE - 1)
        ax.set(xlabel="decision steps", ylabel=unit)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.xaxis.set_major_formatter(_k_formatter())
    handles += [
        Line2D([], [], color=policy_color(lab), lw=1.2, ls=":", label=policy_name(lab))
        for lab in BASELINE_LINES
        if lab in base
    ]
    style.header(
        head,
        "KPI curves per training step (spec 10.4)",
        "Selection windows, mean per window of the per-class totals the reward weights; "
        "band = min-max over seeds; dotted = baselines",
        handles=handles[:9],
    )
    return style.savefig(fig, out)


def _k_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda x, _: f"{x / 1000:.0f}k" if abs(x) >= 1000 else f"{x:.0f}")


def plot_holdout_curves(
    curves: pd.DataFrame, out: Path, baselines: Mapping[str, float] | None = None
) -> Path:
    """Holdout cost of the kept validation checkpoints of each F run (columns run, seed,
    step, cost; optional pick bool), with the baselines on the holdout windows."""
    fig, head, ax = style.figure(height=3.6, header_in=0.8)
    handles = []
    for i, (_, g) in enumerate(curves.groupby("run", sort=False)):
        g = g.sort_values("step")
        color = SEED_SHADES[i % len(SEED_SHADES)]
        ax.plot(g["step"], g["cost"] / 1000, color=color, lw=1.5, marker="o", ms=2.5)
        if "pick" in g and g["pick"].astype(bool).any():
            p = g[g["pick"].astype(bool)]
            ax.scatter(
                p["step"],
                p["cost"] / 1000,
                s=46,
                facecolor="none",
                edgecolor=color,
                lw=1.4,
                zorder=5,
            )
        seed = g["seed"].iloc[0] if "seed" in g else ""
        handles.append(Line2D([], [], color=color, lw=2, label=f"seed {seed}"))
    if baselines:
        _baseline_lines(ax, {k: v for k, v in baselines.items() if k in BASELINE_LINES})
    ax.set(xlabel="decision steps", ylabel="mean episode cost (k weighted person·s)")
    ax.ticklabel_format(axis="x", style="plain")
    style.header(
        head,
        "Holdout curves of the final trainings (reporting only)",
        "Every kept checkpoint scored on the holdout windows, never used to select; open "
        "circle = the run's pick",
        handles=handles[:6],
    )
    return style.savefig(fig, out)


def plot_stage_t(decision: Mapping, out: Path) -> Path:
    """Stage-T score per config (bar = mean over seeds of each run's late mean, dots =
    seeds), the D73 threshold (1 - margin) x score(reference); the chosen config is
    outlined. A second panel shows the pedestrian guard when `ped_s` is given."""
    scores = decision.get("scores") or {}
    configs = list(scores)
    means = decision.get("means") or {
        c: float(np.mean([float(v) for v in s.values()])) if s else math.nan
        for c, s in scores.items()
    }
    rule = decision.get("rule") or {}
    ref = str(rule.get("reference") or decision.get("reference") or "t1")
    margin = float(rule.get("margin_rel", 0.015))
    ped = decision.get("ped_s") or {}
    ped = {c: v for c, v in ped.items() if v is not None}
    fig, head, axes = style.figure(height=3.4, ncols=2 if ped else 1, squeeze=False)
    ax = axes[0][0]
    chosen = decision.get("chosen")
    for i, c in enumerate(configs):
        st = config_style(c, i)
        m = float(means.get(c, math.nan))
        ax.bar(
            i,
            m / 1000,
            width=0.6,
            color=st["color"],
            alpha=0.85,
            edgecolor=style.INK if c == chosen else "none",
            lw=1.6,
        )
        vals = [float(v) / 1000 for v in (scores.get(c) or {}).values() if v is not None]
        if vals:
            jit = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter(
                np.asarray(jit) + i,
                vals,
                s=16,
                color=style.SURFACE,
                edgecolor=style.INK,
                lw=0.8,
                zorder=3,
            )
    if ref in means and math.isfinite(float(means[ref])):
        thr = (1 - margin) * float(means[ref]) / 1000
        lo = min(
            [float(v) / 1000 for v in means.values() if v is not None and math.isfinite(float(v))]
        )
        ax.set_ylim(bottom=0.9 * min(lo, thr))
        style.target_line(ax, thr, f"{100 * margin:g} % below {ref.upper()}")
    ax.set_xticks(range(len(configs)), [c.upper() for c in configs])
    ax.set_ylabel("late mean cost (k weighted person·s)")
    ax.set_title("Score = mean over seeds of the late mean (D73)", fontsize=style.FONT_SIZE)
    if ped:
        ax2 = axes[0][1]
        pc = list(ped)
        for i, c in enumerate(pc):
            ax2.bar(
                i, float(ped[c]) / 1000, width=0.6, color=config_style(c, i)["color"], alpha=0.85
            )
        if ref in ped:
            guard = float(rule.get("ped_guard_rel", 0.05))
            style.target_line(
                ax2, (1 + guard) * float(ped[ref]) / 1000, f"+{100 * guard:g} % guard"
            )
        ax2.set_xticks(range(len(pc)), [c.upper() for c in pc])
        ax2.set_ylabel("pedestrian waiting (k person·s / window)")
        ax2.set_title("Pedestrian guard (c)", fontsize=style.FONT_SIZE)
    sub = f"Chosen: {str(chosen).upper()}" if chosen else "Decision pending"
    style.header(
        head, "Stage T: tuning decision", sub + "; dots = training seeds; outlined = chosen"
    )
    return style.savefig(fig, out)


def dominated(points: pd.DataFrame, x: str, y: str) -> pd.Series:
    """True for a point another point beats on both axes (lower is better)."""
    xs, ys = points[x].to_numpy(float), points[y].to_numpy(float)
    out = []
    for i in range(len(points)):
        better = (xs <= xs[i]) & (ys <= ys[i]) & ((xs < xs[i]) | (ys < ys[i]))
        out.append(bool(better.any()))
    return pd.Series(out, index=points.index)


def plot_pareto(kpis: pd.DataFrame, out: Path) -> Path:
    """Weight sensitivity trade-off (D76). `kpis`: one row per policy (columns label,
    setting, kind "policy" | "baseline", direction, and the PARETO metrics as means over
    the evaluation seeds). Settings are drawn as mean ± SD of their policies (policies as
    small dots), coloured by the direction of the weight change; S0/S1/S2 as squares; the
    non-dominated settings and baselines joined in each panel."""
    pol = kpis[kpis["kind"] != "baseline"]
    base = kpis[kpis["kind"] == "baseline"]
    ys = [y for y in PARETO_Y if y in kpis]
    fig, head, axes = style.figure(height=4.0, ncols=len(ys), squeeze=False)
    agg = pol.groupby("setting", sort=False).agg(
        direction=("direction", "first"),
        **{f"{c}|m": (c, "mean") for c in (PARETO_X, *ys)},
        **{f"{c}|s": (c, "std") for c in (PARETO_X, *ys)},
    )
    for ax, y in zip(axes[0], ys, strict=True):
        for setting, r in agg.iterrows():
            st = DIRECTION_STYLE.get(str(r["direction"]), DIRECTION_STYLE["mixed"])
            dots = pol[pol["setting"] == setting]
            ax.scatter(dots[PARETO_X], dots[y], s=10, color=st["color"], alpha=0.45, lw=0, zorder=2)
            xerr = r[f"{PARETO_X}|s"] if math.isfinite(r[f"{PARETO_X}|s"]) else None
            yerr = r[f"{y}|s"] if math.isfinite(r[f"{y}|s"]) else None
            ax.errorbar(
                r[f"{PARETO_X}|m"],
                r[f"{y}|m"],
                xerr=xerr,
                yerr=yerr,
                fmt=st["marker"],
                color=st["color"],
                ms=7,
                capsize=2,
                lw=1,
                zorder=4,
                markeredgecolor=style.SURFACE,
            )
            ax.annotate(
                str(setting),
                (r[f"{PARETO_X}|m"], r[f"{y}|m"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=style.FONT_SIZE - 2.5,
                color=style.INK_2,
            )
        for _, r in base.iterrows():
            bx, by, label = float(r[PARETO_X]), float(r[y]), str(r["label"])
            ax.scatter(
                [bx],
                [by],
                marker="s",
                s=52,
                color=policy_color(label),
                zorder=4,
                edgecolor=style.SURFACE,
            )
            ax.annotate(
                policy_name(label),
                (bx, by),
                xytext=(5, -9),
                textcoords="offset points",
                fontsize=style.FONT_SIZE - 2.5,
                color=style.INK_2,
            )
        pts = pd.concat(
            [
                pd.DataFrame({"x": agg[f"{PARETO_X}|m"], "y": agg[f"{y}|m"]}),
                pd.DataFrame({"x": base[PARETO_X].to_numpy(float), "y": base[y].to_numpy(float)}),
            ],
            ignore_index=True,
        ).dropna()
        if len(pts) > 1:
            front = pts[~dominated(pts, "x", "y")].sort_values("x")
            ax.plot(front["x"], front["y"], color=style.MUTED, lw=1.0, ls="--", zorder=1)
        ax.set(xlabel=AXIS_LABELS[PARETO_X], ylabel=AXIS_LABELS.get(y, y))
    present = [d for d in DIRECTION_STYLE if d in set(agg["direction"])]
    handles = [
        Line2D(
            [],
            [],
            color=DIRECTION_STYLE[d]["color"],
            marker=DIRECTION_STYLE[d]["marker"],
            lw=0,
            label=DIRECTION_STYLE[d]["label"],
        )
        for d in present
    ]
    handles.append(Line2D([], [], color=style.MUTED, ls="--", lw=1, label="non-dominated"))
    style.header(
        head,
        "Weight sensitivity: trade-off between users (10.3)",
        "Full 2 h KPI v3 runs on the sensitivity seeds; marker = setting mean ± SD over its "
        "training seeds, dots = single policies; squares = baselines; lower is better",
        handles=handles,
    )
    return style.savefig(fig, out)


def plot_stress(summary: pd.DataFrame, out: Path, limit_pct: float = 5.0) -> Path:
    """Stress cells (D78): S3's env cost in % vs S1 and vs S2 per cell (columns cell, ref,
    pct); the pre-registered 'more than 5 % worse than S1' line."""
    cells = list(dict.fromkeys(summary["cell"]))
    fig, head, ax = style.figure(height=0.32 * len(cells) + 1.2, header_in=0.8)
    ax.axvline(0, color=style.AXIS, lw=1.0, zorder=1)
    handles = []
    for k, ref in enumerate(r for r in ("S1", "S2") if r in set(summary["ref"])):
        g = summary[summary["ref"] == ref].set_index("cell").reindex(cells)
        off = -0.12 if k == 0 else 0.12
        ax.scatter(g["pct"], np.arange(len(cells)) + off, s=30, color=policy_color(ref), zorder=3)
        handles.append(
            Line2D(
                [], [], color=policy_color(ref), marker="o", lw=0, label=f"vs {policy_name(ref)}"
            )
        )
    style.target_line(ax, limit_pct, f"+{limit_pct:g} % vs S1", axis="x")
    ax.set_yticks(range(len(cells)), cells)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("S3 env cost change (%, ratio of means; negative = S3 better)")
    style.header(
        head,
        "Stress panel (reported, not gating)",
        "Paired seeds; each cell forces one condition (D78)",
        handles=handles,
    )
    return style.savefig(fig, out)


def _ref_of(paired: pd.DataFrame) -> pd.Series:
    """The reference controller of each paired row: `ref`, else `a`, else the first part
    of `pair` ("S1-S3")."""
    for col in ("ref", "a"):
        if col in paired:
            return paired[col].astype(str)
    if "pair" in paired:
        return paired["pair"].astype(str).str.split(r"[-_/ ]", regex=True).str[0]
    raise KeyError("paired rows need a ref, a or pair column")


def plot_s3_forest(paired: pd.DataFrame, out: Path) -> Path:
    """Dev-test full 2 h runs (D79): Hodges-Lehmann paired difference S3 - ref with its
    95 % CI per KPI, one panel per reference; green = S3 better, grey = worse; filled =
    Holm-significant within the pair, hollow otherwise."""
    p = paired.assign(ref=_ref_of(paired), key=paired["user_type"] + "/" + paired["kpi"])
    refs = [r for r in ("S0", "S1", "S2") if r in set(p["ref"])] or list(dict.fromkeys(p["ref"]))
    keys = list(dict.fromkeys(p["key"]))
    fig, head, axes = style.figure(
        height=0.36 * len(keys) + 1.1, ncols=len(refs), squeeze=False, sharey=True
    )
    for ax, ref in zip(axes[0], refs, strict=True):
        g = p[p["ref"] == ref].set_index("key")
        for i, key in enumerate(keys):
            if key not in g.index:
                continue
            r = g.loc[key]
            better = str(r.get("effect", "")) == "better"
            col = RL if better else style.INK_2
            lo, hi = float(r.get("hl_ci_low", math.nan)), float(r.get("hl_ci_high", math.nan))
            if math.isfinite(lo) and math.isfinite(hi):
                ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="butt")
            sig = bool(r.get("significant_holm", False))
            ax.scatter(
                [float(r.get("hl_diff", math.nan))],
                [i],
                s=34,
                zorder=3,
                color=col if sig else style.SURFACE,
                edgecolor=col,
                lw=1.4,
            )
        ax.axvline(0, color=style.AXIS, lw=1)
        ax.set_yticks(range(len(keys)), keys)
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"S3 − {ref}", fontsize=style.FONT_SIZE)
        ax.set_xlabel("HL paired difference, 95 % CI (s)")
    handles = [
        Line2D([], [], color=RL, marker="o", lw=0, label="S3 better"),
        Line2D([], [], color=style.INK_2, marker="o", lw=0, label="S3 worse"),
        Line2D(
            [],
            [],
            color=style.INK_2,
            marker="o",
            lw=0,
            markerfacecolor=style.SURFACE,
            label="not Holm-significant",
        ),
    ]
    style.header(
        head,
        "First S3 results: dev-test full 2 h runs (exploratory)",
        "Exact Wilcoxon, Holm over the KPIs within each pair (D79)",
        handles=handles,
    )
    return style.savefig(fig, out)


def plot_interventions(rows: pd.DataFrame, out: Path) -> Path:
    """Guard interventions per hour of control (D77), one bar per row (columns label,
    cut_per_h, replaced_per_h, cut_foreseeable_per_h): cut split into its foreseeable
    part, then replaced."""
    r = rows.reset_index(drop=True)
    cut = r["cut_per_h"].astype(float).fillna(0.0)
    fore = r["cut_foreseeable_per_h"].astype(float).fillna(0.0).clip(upper=cut)
    parts = {
        "cut_only": cut - fore,
        "cut_foreseeable": fore,
        "replaced": r["replaced_per_h"].astype(float).fillna(0.0),
    }
    fig, head, ax = style.figure(height=0.34 * len(r) + 1.1, header_in=0.8)
    left = np.zeros(len(r))
    handles = []
    for key, label, color in SPLIT_STYLE:
        ax.barh(range(len(r)), parts[key], left=left, color=color, height=0.62)
        left = left + parts[key].to_numpy(float)
        handles.append(Line2D([], [], color=color, lw=6, label=label))
    for i, total in enumerate(left):
        ax.annotate(
            f"{total:.1f}",
            (total, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=style.FONT_SIZE - 2,
            color=style.INK_2,
        )
    ax.set_yticks(range(len(r)), r["label"])
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Safety Guard interventions per hour of control")
    style.header(
        head,
        "Interventions: cut vs replaced (D77)",
        "cut = the shield or max green ended a green the policy held; foreseeable = the "
        "policy chose to stay with too little slack; replaced = any other request",
        handles=handles,
    )
    return style.savefig(fig, out)


def make_figures(data: Mapping[str, Any], figures: Path) -> dict[str, Path]:
    """Draw every figure of FIGURES whose inputs are present; returns name -> path.

    `data` keys (all optional): `runs` (run dicts: stage, variant, algo, seed, validation
    frame, pick_step, baselines, baseline_totals), `holdout_curves` (frame: run, seed,
    step, cost[, pick]) and `holdout_baselines` (label -> cost), `decision_t`
    (decision_T.json), `pareto` (per-policy frame, `plot_pareto`), `stress` (frame: cell,
    ref, pct) and `stress_limit_pct`, `paired` (dev-test paired rows), `interventions`
    (frame: label, cut_per_h, replaced_per_h, cut_foreseeable_per_h)."""
    figures = Path(figures)
    figs: dict[str, Path] = {}

    def ok(frame) -> bool:
        return frame is not None and len(frame) > 0

    runs = [
        m
        for m in data.get("runs") or []
        if m.get("validation") is not None and not m["validation"].empty
    ]
    curve_runs = [m for m in runs if m.get("stage") in ("T", "F")]
    if curve_runs:
        figs["learning_bands"] = plot_learning_bands(curve_runs, figures / "learning_bands.png")
        kpi_runs = [m for m in curve_runs if m.get("stage") == "F"] or [
            m for m in curve_runs if m.get("variant") == (data.get("reference") or "t1")
        ]
        if kpi_runs:
            figs["kpi_curves"] = plot_kpi_curves(kpi_runs, figures / "kpi_curves.png")
    if ok(data.get("holdout_curves")):
        figs["holdout_curves"] = plot_holdout_curves(
            data["holdout_curves"], figures / "holdout_curves.png", data.get("holdout_baselines")
        )
    if (data.get("decision_t") or {}).get("scores"):
        figs["stage_t"] = plot_stage_t(data["decision_t"], figures / "stage_t.png")
    pareto = data.get("pareto")
    if ok(pareto) and (pareto["kind"] != "baseline").any():
        figs["pareto"] = plot_pareto(pareto, figures / "pareto.png")
    if ok(data.get("stress")):
        figs["stress"] = plot_stress(
            data["stress"], figures / "stress.png", float(data.get("stress_limit_pct", 5.0))
        )
    if ok(data.get("paired")):
        figs["s3_forest"] = plot_s3_forest(data["paired"], figures / "s3_forest.png")
    if ok(data.get("interventions")):
        figs["interventions"] = plot_interventions(
            data["interventions"], figures / "interventions.png"
        )
    return {name: figs[name] for name in FIGURES if name in figs}

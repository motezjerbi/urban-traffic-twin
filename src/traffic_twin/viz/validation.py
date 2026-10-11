"""Validation and comparison figures, in the project style (`viz.style`):

- `plot_flows`: simulated vs real flow per station (scatter with the +-15 % A-02 limit,
  the GEH = 5 contour and y = x; filled = within 15 %, hollow = outside) and the A-02
  rows as bars (real mean +- sd over typical days vs simulated mean +- sd over seeds);
- `plot_queues`: each measured queue against the simulated p10-p90 range and median;
- `plot_paired_differences`: per-seed paired differences S1 - S0 per headline KPI.

Red (`style.CRITICAL`) marks only the 15 % limit.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_twin.viz import style

COUNTED = ("PASS", "FAIL", "ok", "off")


def geh_contour(c: np.ndarray, limit: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Model flows M on the GEH = limit contour for counts c: 2 (M - C)^2 = limit^2 (M + C)."""
    k = limit**2 / 4
    root = np.sqrt(k**2 + limit**2 * c)
    return c + k + root, c + k - root


def plot_flows(
    table: pd.DataFrame,
    out: Path,
    flow_rel: float = 0.15,
    controller: str = "fixed",
    note: str = "",
) -> Path:
    a02_slots = table.loc[table["role"] == "A-02", "slot"].unique()
    rows = table[
        table["status"].isin(COUNTED)
        & table["slot"].isin(a02_slots)
        & (table["role"] != "sensitivity")
    ]
    bars = table[table["role"] == "A-02"].copy()
    fig, head, (ax1, ax2) = style.figure(
        height=4.0, ncols=2, gridspec_kw={"width_ratios": [1, 1.5]}
    )
    vals = rows[["sim_mean", "real_mean"]].to_numpy(float)
    top = float(np.nanmax(vals)) * 1.18 if vals.size and np.isfinite(vals).any() else 100.0
    x = np.linspace(0, top, 100)
    ax1.fill_between(x, x * (1 - flow_rel), x * (1 + flow_rel), color=style.GRID, lw=0, zorder=0)
    for sgn in (1, -1):
        ax1.plot(x, x * (1 + sgn * flow_rel), color=style.CRITICAL, lw=1.0, zorder=1)
    hi, lo = geh_contour(x)
    ax1.plot(x, hi, color=style.MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax1.plot(x, np.where(lo > 0, lo, np.nan), color=style.MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax1.plot(x, x, color=style.AXIS, lw=1.0, zorder=1)
    color = style.CONTROLLER_COLORS.get(controller, style.INK)
    for _, r in rows.iterrows():
        ok = r["status"] in ("PASS", "ok")
        ax1.errorbar(
            r["real_mean"],
            r["sim_mean"],
            xerr=_nz(r["real_std"]),
            yerr=_nz(r["sim_std"]),
            fmt="o" if r["direction"] == "in" else "s",
            ms=6.5,
            mfc=color if ok else style.SURFACE,
            mec=color,
            mew=1.4,
            ecolor=style.INK_2,
            elinewidth=0.9,
            capsize=2,
            zorder=3,
        )
        if not ok:  # inbound arms are named in the bar panel; label the misses only
            ax1.annotate(
                f"{r['station']} {r['slot'][:5]}",
                (r["real_mean"], r["sim_mean"]),
                xytext=(-7, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=style.FONT_SIZE - 2.5,
                color=style.INK_2,
            )
    ax1.set_xlim(0, top)
    ax1.set_ylim(0, top)
    ax1.set_aspect("equal")
    ax1.grid(axis="x", visible=True)
    ax1.set_xlabel("Real flow, typical days (veh/h)")
    ax1.set_ylabel("Simulated flow, mean of seeds (veh/h)")
    ax1.set_title("Every station, A-02 slots")
    for y, text, ink in (
        (0.10, f"red lines: ±{flow_rel:.0%} limit", style.CRITICAL),
        (0.04, "dashed: GEH = 5", style.MUTED),
    ):  # the empty corner below the band
        ax1.text(0.97, y, text, transform=ax1.transAxes, ha="right", va="bottom", color=ink,
                 fontsize=style.FONT_SIZE - 2)  # fmt: skip

    xs = np.arange(len(bars))
    ax2.bar(xs - 0.19, bars["real_mean"], width=0.36, color=style.AXIS, label="real (typical days)")
    ax2.errorbar(
        xs - 0.19,
        bars["real_mean"],
        yerr=_nz(bars["real_std"]),
        fmt="none",
        ecolor=style.INK_2,
        capsize=2,
        lw=1,
    )
    ax2.bar(xs + 0.19, bars["sim_mean"], width=0.36, color=color, label="simulated (seeds)")
    ax2.errorbar(
        xs + 0.19,
        bars["sim_mean"],
        yerr=_nz(bars["sim_std"]),
        fmt="none",
        ecolor=style.INK_2,
        capsize=2,
        lw=1,
    )
    ymax = 1.0
    for xi, (_, r) in zip(xs, bars.iterrows(), strict=True):
        top_r = np.nanmax(
            [r["real_mean"] + _nz(r["real_std"]), r["sim_mean"] + _nz(r["sim_std"]), 0.0]
        )
        ymax = max(ymax, top_r)
        txt = (
            f"{r['rel_err_pct']:+.0f} %\nGEH {r['geh']:.1f}"
            if pd.notna(r["rel_err_pct"])
            else r["status"]
        )
        ax2.annotate(
            txt,
            (xi, top_r),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE - 2,
            color=style.INK,
            fontweight="semibold" if r["status"] == "FAIL" else "normal",
        )
    ax2.set_ylim(0, ymax * 1.28)
    ax2.set_xticks(xs, [f"{s}\n{t}" for s, t in zip(bars["station"], bars["slot"], strict=True)])
    ax2.tick_params(axis="x", length=0, labelsize=style.FONT_SIZE - 2)
    ax2.set_ylabel("Flow (veh/h)")
    ax2.set_title("Inbound arms per A-02 slot")
    ax2.legend(loc="upper left", ncols=2)
    style.header(
        head,
        "Simulated vs real flow per arm (A-02)",
        "filled = within the limit, hollow = outside · circles inbound, squares outbound · "
        "whiskers ±1 sd (days / seeds)",
    )
    if note:
        style.note(fig, note)
    return style.savefig(fig, out)


def _nz(v):
    """Error bar sizes with NaN as 0."""
    return np.nan_to_num(np.asarray(v, dtype=float), nan=0.0)


def plot_queues(table: pd.DataFrame, out: Path, controller: str = "fixed") -> Path:
    rows = table[table["sim_n"].fillna(0) > 0].reset_index(drop=True)
    fig, head, ax = style.figure(height=0.6 + 0.42 * max(len(rows), 1))
    color = style.CONTROLLER_COLORS.get(controller, style.INK)
    for i, r in rows.iterrows():
        y = len(rows) - 1 - i
        ax.plot(
            [r["sim_p10"], r["sim_p90"]],
            [y, y],
            color=color,
            lw=7,
            alpha=0.35,
            solid_capstyle="butt",
        )
        ax.plot([r["sim_p50"]] * 2, [y - 0.22, y + 0.22], color=color, lw=2.2)
        ok = r["status"] in ("PASS", "INFO")
        ax.scatter(
            r["obs_veh"],
            y,
            s=46,
            marker="D",
            zorder=4,
            linewidth=1.3,
            facecolor=style.INK if ok else style.SURFACE,
            edgecolor=style.INK,
        )
    ax.set_yticks(
        range(len(rows))[::-1],
        [f"{r['approach']} {r['time']} {r['kind']} ({r['status']})" for _, r in rows.iterrows()],
    )
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(left=0)
    ax.set_xlabel("Queued vehicles (capped to the camera's visible distance)")
    style.header(
        head,
        "Observed queues vs simulation",
        "bar = simulated p10-p90 within ±5 min, all seeds · tick = median · diamond = video "
        "(filled = PASS)",
    )
    return style.savefig(fig, out)


def plot_paired_differences(
    pairs: pd.DataFrame,
    table: pd.DataFrame,
    a: str,
    b: str,
    out: Path,
    controller: str = "actuated",
) -> Path:
    """One panel per headline KPI: per-seed difference b - a (dots), its mean with the
    paired t CI, and the zero line."""
    head_rows = table[table["headline"]].reset_index(drop=True)
    n = max(len(head_rows), 1)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, header, axes = style.figure(height=3.0 * nrows, ncols=ncols, nrows=nrows, squeeze=False)
    color = style.CONTROLLER_COLORS.get(controller, style.SERIES[1])
    flat = axes.ravel()
    for k, (ax, (_, r)) in enumerate(zip(flat, head_rows.iterrows(), strict=False)):
        d = pairs[(pairs["user_type"] == r["user_type"]) & (pairs["kpi"] == r["kpi"])]["diff"]
        jitter = np.linspace(-0.18, 0.18, len(d)) if len(d) > 1 else np.zeros(len(d))
        ax.axhline(0, color=style.AXIS, lw=1.0, zorder=1)
        ax.scatter(jitter, d, s=20, color=color, alpha=0.8, lw=0, zorder=3)
        if pd.notna(r["diff_ci_low"]):
            ax.errorbar(
                [0.42],
                [r["diff_mean"]],
                yerr=[[r["diff_mean"] - r["diff_ci_low"]], [r["diff_ci_high"] - r["diff_mean"]]],
                fmt="o",
                color=style.INK,
                ms=5,
                capsize=3,
                lw=1.3,
                zorder=4,
            )
        ax.set_xlim(-0.5, 0.7)
        ax.set_xticks([])
        p = r["p_holm"] if pd.notna(r["p_holm"]) else r["p_value"]
        flag = " · equity flag" if r["equity_flag"] else ""
        ax.set_title(
            f"{style.user_label(r['user_type'])}: {r['kpi']}", fontsize=style.FONT_SIZE, pad=16
        )
        ax.annotate(
            f"{r['rel_change_pct']:+.1f} % · p(Holm) {p:.3g}{flag}",
            (0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0, 3),
            textcoords="offset points",
            va="bottom",
            fontsize=style.FONT_SIZE - 2,
            color=style.INK_2,
        )
        if k % ncols == 0:
            ax.set_ylabel("Difference per seed (s)")
    for ax in flat[len(head_rows) :]:
        ax.set_visible(False)
    seeds = int(table["n_pairs"].max()) if len(table) else 0
    style.header(
        header,
        f"Paired per-seed differences, {b} − {a}",
        f"{seeds} seeds, same demand · dots = seeds, black = mean ± 95 % t CI · below zero = "
        f"lower in {b}",
    )
    return style.savefig(fig, out)

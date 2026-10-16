"""Figures of the Part 6 RL page (`docs/rl.md`), in the project style (`viz.style`),
drawn from the artifacts of `rl.train` and `rl.evaluate`:

- `learning_curves.png`: training episode return (rolling mean of the VecMonitor
  returns) and validation cost on the selection windows per decision step, every run,
  with the baselines of the same windows;
- `validation_totals.png`: per-class totals (waiting seconds per class, penalty seconds)
  of each run's best checkpoint and of the baselines, mean per selection window;
- `episode_timing.png`: wall time of a full 2 h episode per policy (seeds as dots) and
  SubprocVecEnv throughput against the number of workers;
- `devtest_compare.png`: per-window cost change against S1 on the dev-test windows,
  per policy (exploratory).

`make_figures` draws the ones whose inputs exist. `docs/rl.md` is written by
`python -m traffic_twin.rl.evaluate docs`, which calls it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from traffic_twin.viz import style

FIGURES = ("learning_curves", "validation_totals", "episode_timing", "devtest_compare")
# policy label -> (colour, display name): controllers keep their project colours, the
# synthetic policies are neutral greys
POLICY_STYLE = {
    "S0": (style.CONTROLLER_COLORS["fixed"], "S0 fixed"),
    "S1": (style.CONTROLLER_COLORS["actuated"], "S1 actuated"),
    "S2": (style.CONTROLLER_COLORS["rules"], "S2 rules"),
    "RL": (style.CONTROLLER_COLORS["rl"], "RL"),
    "random": (style.MUTED, "random"),
    "stay": (style.INK_2, "always-stay"),
}
TIMING_LABELS = {
    "fixed": "S0",
    "actuated": "S1",
    "rules": "S2",
    "model": "RL",
    "random": "random",
    "stay": "stay",
}
RUN_SHADES = ("#008300", "#3f9f3f", "#7cc07c", "#0a5a0a", "#a8d8a8")  # RL runs, greens
CLASS_TOTALS = (
    ("car_s", "Cars waiting", "veh·s"),
    ("car_pending_s", "Cars pending insertion", "veh·s"),
    ("bus_s", "Buses waiting", "veh·s"),
    ("pedestrian_s", "Pedestrians waiting", "person·s"),
    ("ambulance_s", "Ambulances waiting", "veh·s"),
    ("penalty_s", "Curb wait > θ", "s"),
)
ROLLING = 50  # episodes in the training-return rolling mean


def policy_color(label: str) -> str:
    return POLICY_STYLE.get(label, (style.MUTED, label))[0]


def policy_name(label: str) -> str:
    return POLICY_STYLE.get(label, (style.MUTED, label))[1]


def run_label(m: Mapping) -> str:
    return f"{str(m.get('algo', '?')).upper()} s{m.get('seed', '?')}"


def run_style(i: int, m: Mapping) -> dict:
    """Greens for RL runs; PPO dashed."""
    return {"color": RUN_SHADES[i % len(RUN_SHADES)], "ls": "--" if m.get("algo") == "ppo" else "-"}


def _baselines(runs: Sequence[Mapping]) -> dict[str, float]:
    for m in runs:
        if m.get("baselines"):
            return {k: float(v) for k, v in m["baselines"].items()}
    return {}


def plot_learning_curves(runs: Sequence[Mapping], out: Path) -> Path:
    """Training return (left) and validation cost (right) per decision step."""
    fig, head, (ax_r, ax_v) = style.figure(height=3.8, ncols=2)
    handles = []
    for i, m in enumerate(runs):
        st = run_style(i, m)
        mon = m.get("monitor")
        if mon is not None and not mon.empty:
            r = mon["r"].rolling(ROLLING, min_periods=max(1, min(10, len(mon)))).mean()
            ax_r.plot(mon["steps"], r, lw=1.6, **st)
        val = m.get("validation")
        if val is not None and not val.empty:
            ax_v.plot(val["step"], val["cost"] / 1000, marker="o", ms=3, lw=1.6, **st)
            if m.get("best_step") is not None:
                b = val[val["step"] == m["best_step"]]
                ax_v.scatter(
                    b["step"],
                    b["cost"] / 1000,
                    s=46,
                    facecolor="none",
                    edgecolor=st["color"],
                    lw=1.4,
                    zorder=5,
                )
        handles.append(Line2D([], [], lw=2, label=run_label(m), **st))
    base = sorted(_baselines(runs).items(), key=lambda kv: kv[1])
    for label, cost in base:
        ax_v.axhline(cost / 1000, color=policy_color(label), lw=1.0, ls=":", zorder=1)
    lo, hi = ax_v.get_ylim()
    min_gap = 0.045 * (hi - lo)  # labels closer than this are pushed apart
    last = -np.inf
    for label, cost in base:
        y = max(cost / 1000, last + min_gap)
        last = y
        ax_v.annotate(
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
    ax_r.set(xlabel="decision steps", ylabel=f"episode return (rolling {ROLLING})")
    ax_r.set_title("Training episodes", fontsize=style.FONT_SIZE)
    ax_v.set(xlabel="decision steps", ylabel="mean episode cost (k weighted person·s)")
    ax_v.set_title(
        "Validation on the selection windows (lower is better)", fontsize=style.FONT_SIZE
    )
    for ax in (ax_r, ax_v):
        ax.ticklabel_format(axis="x", style="plain")
    style.header(
        head,
        "Learning curves",
        "Open circle = best checkpoint (best_model.zip); "
        "dotted lines = baselines on the same windows",
        handles=handles,
    )
    return style.savefig(fig, out)


def best_totals(runs: Sequence[Mapping]) -> pd.DataFrame:
    """label + per-class totals: each run's best validation row, then the baselines
    (mean per selection window from the run's baselines.csv)."""
    rows = []
    for m in runs:
        val = m.get("validation")
        if val is None or val.empty:
            continue
        b = val[val["step"] == m.get("best_step")]
        if not b.empty:
            rows.append({"label": run_label(m), **b.iloc[0].to_dict()})
    for m in runs:
        p = Path(m.get("dir", "")) / "baselines.csv"
        if p.is_file():
            base = pd.read_csv(p)
            for label, g in base.groupby("label", sort=False):
                rows.append({"label": str(label), **g.mean(numeric_only=True).to_dict()})
            break
    return pd.DataFrame(rows)


def plot_validation_totals(totals: pd.DataFrame, out: Path) -> Path:
    """One small panel per class, one bar per policy."""
    keys = [k for k in CLASS_TOTALS if k[0] in totals.columns]
    fig, head, axes = style.figure(
        height=3.0, ncols=len(keys), squeeze=False, width=max(style.WIDTH_IN, 2.0 * len(keys))
    )
    labels = list(totals["label"])
    colors = []
    shade = 0
    for lab in labels:
        if lab in POLICY_STYLE:
            colors.append(policy_color(lab))
        else:
            colors.append(RUN_SHADES[shade % len(RUN_SHADES)])
            shade += 1
    for ax, (key, title, unit) in zip(axes[0], keys, strict=True):
        v = totals[key].to_numpy(float) / 1000
        ax.barh(range(len(labels)), v, color=colors, height=0.7)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0][0] else [""] * len(labels))
        ax.invert_yaxis()
        ax.set_title(title, fontsize=style.FONT_SIZE - 1)
        ax.set_xlabel(f"k {unit} / window")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    style.header(
        head,
        "Per-class totals on the selection windows",
        "Best checkpoint of each run vs the baselines; raw seconds (the cost weights "
        "them: car 1.3, bus 25, pedestrian 1.2, ambulance 20, penalty λ)",
    )
    return style.savefig(fig, out)


def plot_episode_timing(timing: Mapping, out: Path) -> Path:
    """Full-episode wall time per policy (left) and steps/s per n_envs (right)."""
    eps = pd.DataFrame(timing.get("episodes") or [])
    scaling = pd.DataFrame(timing.get("scaling") or [])
    ncols = 2 if not scaling.empty else 1
    fig, head, axes = style.figure(height=3.4, ncols=ncols, squeeze=False)
    ax = axes[0][0]
    policies = list(dict.fromkeys(eps["policy"])) if not eps.empty else []
    for i, pol in enumerate(policies):
        g = eps[eps["policy"] == pol]
        lab = TIMING_LABELS.get(pol, pol)
        ax.bar(i, g["wall_s"].mean(), color=policy_color(lab), width=0.62, alpha=0.9)
        ax.scatter(
            [i] * len(g),
            g["wall_s"],
            s=14,
            color=style.SURFACE,
            edgecolor=style.INK,
            lw=0.8,
            zorder=3,
        )
        dec = g["decisions"].mean()
        ax.annotate(
            f"{dec:.0f} dec." if dec > 0 else "1 s ctl",
            (i, g["wall_s"].max()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE - 2,
            color=style.INK_2,
        )
    ax.set_xticks(range(len(policies)), [policy_name(TIMING_LABELS.get(p, p)) for p in policies])
    ax.set_ylabel("wall s per 2 h episode")
    ax.set_title("Full 07:30-09:30 episode", fontsize=style.FONT_SIZE)
    if not scaling.empty:
        ax2 = axes[0][1]
        ax2.plot(
            scaling["n_envs"],
            scaling["steps_per_s"],
            marker="o",
            color=style.CONTROLLER_COLORS["rl"],
        )
        for r in scaling.itertuples(index=False):
            ax2.annotate(
                f"{r.steps_per_s:.0f}",
                (r.n_envs, r.steps_per_s),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=style.FONT_SIZE - 2,
                color=style.INK_2,
            )
        ax2.set_xticks(scaling["n_envs"])
        ax2.set(xlabel="SubprocVecEnv workers", ylabel="decision steps / s")
        ax2.set_title("Throughput (random policy, no learning)", fontsize=style.FONT_SIZE)
        ax2.set_ylim(bottom=0)
    sub = f"{timing.get('cpu_count')} logical CPUs; code {timing.get('code_version')}"
    style.header(head, "Episode timing (S6 measurement)", sub)
    return style.savefig(fig, out)


def plot_devtest_compare(rows: pd.DataFrame, out: Path, ref: str = "S1") -> Path:
    """Per-window % change of the episode cost against `ref`, one column per policy."""
    base = rows[rows["label"] == ref].set_index("seed")["cost"]
    labels = [lab for lab in dict.fromkeys(rows["label"]) if lab != ref]
    fig, head, ax = style.figure(height=3.4, header_in=0.7)
    ax.axhline(0, color=style.AXIS, lw=1.0, zorder=1)
    for i, lab in enumerate(labels):
        g = rows[rows["label"] == lab].set_index("seed")["cost"]
        seeds = sorted(set(g.index) & set(base.index))
        pct = 100 * (g.loc[seeds].to_numpy(float) / base.loc[seeds].to_numpy(float) - 1)
        jitter = np.linspace(-0.12, 0.12, len(pct)) if len(pct) > 1 else np.zeros(len(pct))
        ax.scatter(i + jitter, pct, s=18, color=policy_color(lab), alpha=0.85, zorder=3)
        if len(pct):
            ax.plot([i - 0.25, i + 0.25], [pct.mean()] * 2, color=style.INK, lw=2, zorder=4)
            ax.annotate(
                f"{pct.mean():+.1f} %",
                (i + 0.27, pct.mean()),
                va="center",
                fontsize=style.FONT_SIZE - 2,
                color=style.INK_2,
            )
    ax.set_xticks(range(len(labels)), [policy_name(lab) for lab in labels])
    ax.set_ylabel(f"cost change vs {policy_name(ref)} (%)")
    style.header(
        head,
        "Dev-test windows (exploratory)",
        f"One dot per window; bar = mean of the per-window changes; reference {ref}",
    )
    return style.savefig(fig, out)


def make_figures(
    runs: Sequence[Mapping],
    timing: Mapping | None,
    compare_rows: pd.DataFrame | None,
    out_dir: Path,
) -> dict[str, Path]:
    """Draw every figure whose inputs exist; name (FIGURES) -> path."""
    out_dir = Path(out_dir)
    figs: dict[str, Path] = {}
    curves = [m for m in runs if m.get("monitor") is not None or m.get("validation") is not None]
    if curves:
        figs["learning_curves"] = plot_learning_curves(curves, out_dir / "learning_curves.png")
        totals = best_totals(curves)
        if not totals.empty and any(k in totals for k, _, _ in CLASS_TOTALS):
            figs["validation_totals"] = plot_validation_totals(
                totals, out_dir / "validation_totals.png"
            )
    if timing and timing.get("episodes"):
        figs["episode_timing"] = plot_episode_timing(timing, out_dir / "episode_timing.png")
    if compare_rows is not None and not compare_rows.empty and "S1" in set(compare_rows["label"]):
        figs["devtest_compare"] = plot_devtest_compare(
            compare_rows, out_dir / "devtest_compare.png"
        )
    return figs

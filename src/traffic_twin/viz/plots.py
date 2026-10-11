"""KPI figures from simulation results (`data/results/<scenario>_seed<n>/`).

    python -m traffic_twin.viz.plots [--results data/results] [--out docs/figures]
                                     [--scenario baseline_fixed [other ...]]

Writes `kpi_by_user.png`, `wait_distribution.png` and `queue_timeline.png`. Every plot
function takes a list of scenarios, so later parts compare controllers side by side by
passing several `--scenario` names (one colour per controller, see `viz.style`).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_twin.viz import style

log = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")
FIGURES_DIR = Path("docs/figures")
PED_MAX_WAIT_S = 90.0  # spec target, used when a run's config does not give one


@dataclass
class Scenario:
    """All seeds of one scenario, concatenated (each table keeps a `seed` column)."""

    name: str
    controller: str
    kpis: pd.DataFrame  # long format: seed, user_type, kpi, value
    trips: pd.DataFrame  # seed, type, waiting_s, time_loss_s, ...
    state: pd.DataFrame  # intersection rows only: seed, t, phase, queue
    config: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{style.controller_label(self.controller)} ({self.name})"

    @property
    def seeds(self) -> list[int]:
        return sorted(self.kpis["seed"].unique().tolist()) if not self.kpis.empty else []


# -- loading ---------------------------------------------------------------------------
def run_dirs(results: Path, scenario: str) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(scenario)}_seed(\d+)$")
    dirs = [d for d in results.iterdir() if d.is_dir() and pattern.match(d.name)]
    return sorted(dirs, key=lambda d: int(pattern.match(d.name).group(1)))


def load_scenario(results: Path, scenario: str) -> Scenario:
    """Read every `<scenario>_seed*` run under `results`."""
    dirs = run_dirs(Path(results), scenario)
    if not dirs:
        raise FileNotFoundError(f"no runs {scenario}_seed* in {results}")
    kpis, trips, states, config = [], [], [], {}
    for d in dirs:
        seed = int(d.name.rsplit("_seed", 1)[1])
        k = pd.read_csv(d / "kpi_run.csv")
        kpis.append(k.assign(seed=seed))
        if (d / "trip.parquet").exists():
            trips.append(pd.read_parquet(d / "trip.parquet").assign(seed=seed))
        if (d / "state_log.parquet").exists():
            s = pd.read_parquet(
                d / "state_log.parquet",
                columns=["t", "entity", "phase", "queue"],
                filters=[("entity", "==", "intersection")],
            )
            states.append(s.assign(seed=seed))
        if not config and (d / "run.json").exists():
            config = json.loads((d / "run.json").read_text(encoding="utf-8")).get("config", {})
    return Scenario(
        name=scenario,
        controller=str(config.get("controller", scenario.rsplit("_", 1)[-1])),
        kpis=pd.concat(kpis, ignore_index=True),
        trips=pd.concat(trips, ignore_index=True) if trips else pd.DataFrame(),
        state=pd.concat(states, ignore_index=True) if states else pd.DataFrame(),
        config=config,
    )


def colors_of(scenarios: list[Scenario]) -> list[str]:
    return style.series_colors(s.controller for s in scenarios)


def ped_target(scenarios: list[Scenario]) -> float:
    for s in scenarios:
        v = s.config.get("safety", {}).get("ped_max_wait_s")
        if v is not None:
            return float(v)
    return PED_MAX_WAIT_S


def _seeds_note(scenarios: list[Scenario]) -> str:
    seeds = sorted({x for s in scenarios for x in s.seeds})
    return f"{len(seeds)} seed{'s' if len(seeds) != 1 else ''} ({', '.join(map(str, seeds))})"


# -- 1. headline KPI per user type -----------------------------------------------------
# (user type, kpi, panel title, axis label)
HEADLINE_KPIS = [
    ("car", "time_loss_mean", "Car / student", "Mean time loss (s)"),
    ("bus", "delay_mean", "Bus", "Mean delay (s)"),
    ("bus", "headway_std", "Bus regularity", "Headway std (s)"),
    ("ambulance", "time_loss_mean", "Ambulance", "Time loss (s)"),
    ("pedestrian", "ped_wait_max", "Pedestrian", "Max wait (s)"),
]


def _bar_width(ax, n: int, max_px: float = 26.0) -> float:
    """Bar width in data units, capped at `max_px` on screen (thin marks)."""
    ax.figure.canvas.draw()  # resolve the layout so the axes size is final
    px_per_unit = ax.get_window_extent().width / max(n, 1.6)
    return min(0.62, max_px / px_per_unit)


def plot_kpi_by_user(scenarios: list[Scenario], out: Path) -> Path:
    """One panel per headline KPI: mean across seeds (bar) ± std (whisker), each seed as
    a dot. One bar per scenario, so several controllers sit side by side."""
    multi = len(scenarios) > 1
    colors = colors_of(scenarios)
    target = ped_target(scenarios)
    panels = [
        p for p in HEADLINE_KPIS
        if any(((s.kpis["user_type"] == p[0]) & (s.kpis["kpi"] == p[1])).any() for s in scenarios)
    ]  # fmt: skip
    fig, head, axes = style.figure(
        height=3.3, header_in=_header_in(multi), ncols=max(len(panels), 1), squeeze=False
    )
    axes = axes[0]
    n = len(scenarios)
    x = np.arange(n)
    for ax, (user, kpi, panel_title, ylabel) in zip(axes, panels, strict=False):
        width = _bar_width(ax, n)
        top = 0.0
        for i, s in enumerate(scenarios):
            v = s.kpis[(s.kpis["user_type"] == user) & (s.kpis["kpi"] == kpi)]["value"]
            if v.empty:
                continue
            mean, std = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
            color = colors[i] if multi else style.user_color(user)
            ax.bar(x[i], mean, width=width, color=color, zorder=2)
            if std > 0:
                ax.errorbar(x[i], mean, yerr=std, color=style.INK_2, lw=1.2, capsize=4, zorder=3)
            jitter = np.linspace(-width / 4, width / 4, len(v)) if len(v) > 1 else [0.0]
            ax.scatter(
                x[i] + np.asarray(jitter), v, s=18, color=style.SURFACE, edgecolor=style.INK,
                linewidth=1.0, zorder=4,
            )  # fmt: skip
            ax.annotate(
                f"{mean:.0f} s", (x[i], max(mean + std, v.max())), xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom", color=style.INK,
                fontsize=style.FONT_SIZE, fontweight="semibold",
            )  # fmt: skip
            top = max(top, mean + std, v.max())
        if user == "pedestrian" and kpi == "ped_wait_max":
            style.target_line(ax, target, f"target\n{target:.0f} s")
            top = max(top, target)
        ax.set_ylim(0, top * 1.22 if top > 0 else 1)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_xticks(x, [s.name for s in scenarios] if multi else [""] * n)
        ax.tick_params(axis="x", length=0, rotation=20 if multi else 0)
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)

    handles = None
    if multi:
        from matplotlib.patches import Patch

        handles = [Patch(color=c, label=s.label) for s, c in zip(scenarios, colors, strict=True)]
    names = ", ".join(s.name for s in scenarios)
    style.header(
        head,
        "Headline KPI per user type",
        f"{names} · mean of {_seeds_note(scenarios)}, whisker ±1 std, dots = seeds",
        handles,
    )
    return style.savefig(fig, out)


# -- 2. waiting time distribution per user type ----------------------------------------
def wait_values(trips: pd.DataFrame) -> pd.DataFrame:
    """Waiting time per trip: SUMO `waitingTime` for vehicles; for pedestrians the time
    lost on the walk (mostly waiting at crossings), as in the KPI `ped_wait_*`."""
    if trips.empty:
        return pd.DataFrame(columns=["type", "wait_s"])
    wait = trips["waiting_s"] if "waiting_s" in trips else pd.Series(np.nan, index=trips.index)
    ped = trips["type"] == "pedestrian"
    wait = wait.where(~ped, trips.get("time_loss_s"))
    return pd.DataFrame({"type": trips["type"], "wait_s": wait}).dropna()


def plot_wait_distribution(scenarios: list[Scenario], out: Path, max_points: int = 500) -> Path:
    """Per user type: box from p25 to p75, median tick, whiskers p5–p95, and the trips
    themselves as a jittered strip (sampled when there are many)."""
    multi = len(scenarios) > 1
    colors = colors_of(scenarios)
    target = ped_target(scenarios)
    waits = [wait_values(s.trips) for s in scenarios]
    present = {u for w in waits for u in w["type"].unique()}
    users = [u for u in style.USER_TYPES if u in present]
    users += sorted(present - set(users))

    fig, head, ax = style.figure(
        height=0.75 + 0.62 * len(users) * max(1, 0.7 * len(scenarios)), header_in=_header_in(multi)
    )
    rng = np.random.default_rng(0)
    band = 0.7 / len(scenarios)
    xmax = 0.0
    yticks, ylabels = [], []
    for r, user in enumerate(users):
        y0 = len(users) - 1 - r  # first user at the top
        yticks.append(y0)
        for i, w in enumerate(waits):
            v = w.loc[w["type"] == user, "wait_s"].to_numpy()
            if v.size == 0:
                continue
            y = y0 + (len(scenarios) - 1) / 2 * band - i * band
            color = colors[i] if multi else style.user_color(user)
            p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
            pts = v if v.size <= max_points else rng.choice(v, max_points, replace=False)
            jit = rng.uniform(-band * 0.32, band * 0.32, pts.size)
            ax.scatter(pts, y + jit, s=7, color=color, alpha=0.28, lw=0, zorder=2)
            ax.plot([p5, p95], [y, y], color=style.INK_2, lw=1.2, zorder=3, solid_capstyle="butt")
            ax.add_patch(_box(p25, p75, y, band * 0.42, color))
            ax.plot(
                [p50, p50], [y - band * 0.24, y + band * 0.24], color=style.INK, lw=2.2, zorder=5
            )
            ax.annotate(
                f"median {p50:.0f} s · p95 {p95:.0f} s · n = {v.size:,}",
                (1.0, y), xycoords=("axes fraction", "data"), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center", color=style.INK_2,
                fontsize=style.FONT_SIZE - 1.5,
            )  # fmt: skip
            xmax = max(xmax, np.percentile(v, 99.5))
        ylabels.append(style.user_label(user) + (" *" if user == "pedestrian" else ""))
    if "pedestrian" in users:
        style.target_line(ax, target, f"pedestrian target {target:.0f} s", axis="x")
        xmax = max(xmax, target)
    ax.set_yticks(yticks, ylabels)
    ax.tick_params(axis="y", length=0, labelsize=style.FONT_SIZE, labelcolor=style.INK)
    ax.set_ylim(-0.6, len(users) - 0.4)
    ax.set_xlim(0, xmax * 1.08 if xmax > 0 else 1)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Waiting time per trip (s)")

    handles = None
    if multi:
        from matplotlib.patches import Patch

        handles = [Patch(color=c, label=s.label) for s, c in zip(scenarios, colors, strict=True)]
    names = ", ".join(s.name for s in scenarios)
    style.header(
        head,
        "Waiting time per trip, by user type",
        f"{names} · {_seeds_note(scenarios)} pooled · box p25–p75, bar = median, "
        "whiskers p5–p95, dots = trips",
        handles,
    )
    if "pedestrian" in users:
        style.note(fig, "* pedestrians: time lost on the walk, mostly waiting at crossings")
    return style.savefig(fig, out)


def _box(x0: float, x1: float, y: float, half_h: float, color: str):
    from matplotlib.patches import FancyBboxPatch

    return FancyBboxPatch(
        (x0, y - half_h), max(x1 - x0, 1e-9), 2 * half_h,
        boxstyle="round,pad=0,rounding_size=0.6", mutation_aspect=0.04,
        facecolor=color, edgecolor="none", alpha=0.85, zorder=4,
    )  # fmt: skip


# -- 3. queue over time with signal stages ---------------------------------------------
def phase_stage_names(config: dict) -> dict[int, str]:
    """SUMO phase index -> stage name for green phases, "clearance" for the others."""
    out: dict[int, str] = {}
    for st in config.get("network", {}).get("stages", []):
        out[int(st["green_phase"])] = st["name"]
        for p in st.get("clearance_phases", []):
            out[int(p)] = "clearance"
    return out


def stage_spans(state: pd.DataFrame, names: dict[int, str]) -> list[tuple[float, float, str]]:
    """Contiguous (start, end, stage) runs from the phase column of the state log."""
    s = state.sort_values("t")
    if s.empty or "phase" not in s:
        return []
    t = s["t"].to_numpy(float)
    stage = [names.get(int(p), f"phase {int(p)}") if pd.notna(p) else "?" for p in s["phase"]]
    step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    spans, start = [], 0
    for i in range(1, len(t) + 1):
        if i == len(t) or stage[i] != stage[start]:
            spans.append((t[start], t[i - 1] + step, stage[start]))
            start = i
    return spans


def _clock(config: dict, t: float) -> str:
    sec = int(config.get("begin_s", 0) + t)
    return f"{sec // 3600 % 24:02d}:{sec // 60 % 60:02d}"


def plot_queue_timeline(
    scenarios: list[Scenario],
    out: Path,
    zoom_start_s: float = 2400.0,
    zoom_len_s: float = 600.0,
    smooth_s: float = 300.0,
) -> Path:
    """Top: vehicles halting on all approaches over the whole run (rolling mean, mean
    of seeds, band = seed range). Below: a zoom on `zoom_len_s` of the first seed of
    each scenario, with the signal stage of each moment as a strip above it."""
    colors = colors_of(scenarios)
    multi = len(scenarios) > 1
    n = len(scenarios)
    ratios = [2.9] + [0.28, 1.72] * n
    fig, head, axes = style.figure(
        height=sum(ratios) + 0.3 * n, header_in=_header_in(multi), nrows=len(ratios),
        gridspec_kw={"height_ratios": ratios},
    )  # fmt: skip
    tmax = max((s.state["t"].max() for s in scenarios if not s.state.empty), default=7200)
    zoom_len_s = min(zoom_len_s, tmax)
    zoom_start_s = min(max(zoom_start_s, 0.0), tmax - zoom_len_s)  # keep the zoom in the run
    ax_all = axes[0]
    cfg = next((s.config for s in scenarios if s.config), {})
    handles = []
    for s, color in zip(scenarios, colors, strict=True):
        if s.state.empty:
            continue
        q = s.state.pivot_table(index="t", columns="seed", values="queue", aggfunc="mean")
        step = float(np.median(np.diff(q.index))) if len(q) > 1 else 1.0
        roll = q.rolling(max(int(smooth_s / step), 1), min_periods=1, center=True).mean()
        t_min = roll.index / 60.0
        if roll.shape[1] > 1:
            ax_all.fill_between(
                t_min, roll.min(axis=1), roll.max(axis=1), color=color, alpha=0.12, lw=0
            )
        (h,) = ax_all.plot(t_min, roll.mean(axis=1), color=color, label=s.label)
        handles.append(h)
    ax_all.axvspan(
        zoom_start_s / 60, (zoom_start_s + zoom_len_s) / 60, color=style.GRID, alpha=0.6, lw=0
    )
    ax_all.annotate(
        "zoom below", ((zoom_start_s + zoom_len_s / 2) / 60, 1.0),
        xycoords=("data", "axes fraction"), xytext=(0, -2), textcoords="offset points",
        ha="center", va="top", color=style.MUTED, fontsize=style.FONT_SIZE - 2,
    )  # fmt: skip
    ax_all.set_title(f"Whole run, {smooth_s / 60:.0f}-min rolling mean, band = seed range")
    ax_all.set_ylabel("Halting vehicles")
    ax_all.set_ylim(bottom=0)
    ax_all.set_xlim(0, tmax / 60)
    ticks = np.arange(0, tmax / 60 + 1, 15)
    ax_all.set_xticks(ticks, [_clock(cfg, m * 60) for m in ticks])

    stage_order: list[str] = []
    zoom_axes = []
    for i, (s, color) in enumerate(zip(scenarios, colors, strict=True)):
        ax_strip, ax_z = axes[1 + 2 * i], axes[2 + 2 * i]
        names = phase_stage_names(s.config)
        for st in s.config.get("network", {}).get("stages", []):
            if st["name"] not in stage_order:
                stage_order.append(st["name"])
        first = s.state[s.state["seed"] == min(s.state["seed"])] if not s.state.empty else s.state
        win = first[(first["t"] >= zoom_start_s) & (first["t"] <= zoom_start_s + zoom_len_s)]
        win = win.sort_values("t")
        _stage_strip(ax_strip, stage_spans(win, names), stage_order)
        ax_strip.set_xlim(zoom_start_s, zoom_start_s + zoom_len_s)
        seed = int(first["seed"].iloc[0]) if not first.empty else 0
        ax_strip.set_title(
            f"{s.label if multi else 'Zoom'} · seed {seed} · signal stage and queue", pad=4
        )
        ax_z.step(win["t"], win["queue"], where="post", color=color if multi else style.INK, lw=1.6)
        ax_z.set_xlim(zoom_start_s, zoom_start_s + zoom_len_s)
        zoom_axes.append(ax_z)
        ax_z.set_ylabel("Halting vehicles")
        zt = np.arange(zoom_start_s, zoom_start_s + zoom_len_s + 1, 60)
        ax_z.set_xticks(zt, [_clock(s.config or cfg, t) for t in zt])

    ymax = max((ax.get_ylim()[1] for ax in zoom_axes), default=1.0)
    for ax in zoom_axes:  # one scale, so scenarios compare at a glance
        ax.set_ylim(0, ymax)

    style.header(
        head,
        "Intersection queue over time",
        f"{', '.join(s.name for s in scenarios)} · vehicles halting on the 4 approaches, "
        f"logged every 5 s · {_seeds_note(scenarios)}",
        handles if multi else None,
    )
    return style.savefig(fig, out)


def _stage_color(name: str, stage_order: list[str]) -> str:
    if name == "clearance":
        return style.CLEARANCE_FILL
    if name in style.STAGE_FILL:
        return style.STAGE_FILL[name]
    idx = stage_order.index(name) if name in stage_order else 0
    return style.STAGE_FILL_OTHER[idx % len(style.STAGE_FILL_OTHER)]


def _stage_name(name: str) -> str:
    return name.replace("_LEFT", " left").replace("_", " ")


def _stage_strip(ax, spans, stage_order: list[str]) -> None:
    from matplotlib.patches import Patch

    for t0, t1, name in spans:
        ax.axvspan(t0, t1, color=_stage_color(name, stage_order), lw=0)
        if name != "clearance" and t1 - t0 >= 18:
            ax.text(
                (t0 + t1) / 2, 0.5, _stage_name(name),
                ha="center", va="center", fontsize=style.FONT_SIZE - 2.5, color=style.INK,
                transform=ax.get_xaxis_transform(), clip_on=True,
            )  # fmt: skip
    names = stage_order + ["clearance"]
    ax.legend(
        handles=[Patch(color=_stage_color(n, stage_order), label=_stage_name(n)) for n in names],
        loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=len(names), borderaxespad=0.2,
        handlelength=1.0, handleheight=0.8, columnspacing=1.0, fontsize=style.FONT_SIZE - 2,
    )  # fmt: skip
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def _header_in(multi: bool) -> float:
    """Header height: room for a legend row only when several scenarios are drawn."""
    return 0.9 if multi else 0.62


# -- CLI -------------------------------------------------------------------------------
def make_figures(scenarios: list[Scenario], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    return [
        plot_kpi_by_user(scenarios, out_dir / "kpi_by_user.png"),
        plot_wait_distribution(scenarios, out_dir / "wait_distribution.png"),
        plot_queue_timeline(scenarios, out_dir / "queue_timeline.png"),
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--scenario", nargs="+", default=["baseline_fixed"],
        help="one or more scenario names; several are drawn side by side",
    )  # fmt: skip
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scenarios = [load_scenario(args.results, name) for name in args.scenario]
    for path in make_figures(scenarios, args.out):
        log.info("figure: %s", path)


if __name__ == "__main__":
    main()

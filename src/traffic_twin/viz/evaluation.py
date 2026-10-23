"""Figures of the Parts 4 and 8 evaluation (configs/evaluation.yaml), in the project
style (`viz.style`), drawn from the outputs of `evaluation.evaluate`
(`<results>/<family>/{kpis.csv, equity.csv, a04.csv, pairs/*.csv}`,
`<results>/{a05.json, a06.json}`):

- `kpi_by_user.png`: S0 / S1 / S2 / S3 per user type (`viz.plots.plot_kpi_by_user`);
- `paired_vs_s0.png`: per-seed paired differences against the baseline, one column per
  scenario in its controller colour, mean ± 95 % t CI and p (Holm within each pair);
- `equity_heatmap.png`: change vs the baseline of the primary, subgroup, rate and global
  KPIs (blue = better, red = worse; `*` p < α, `!` flagged, 13.3);
- `a05_ambulance.png`: per-seed ambulance travel time S0 -> S2 (A-05 primary), the
  secondary tests of the same family (S0 -> S3), and the ambulance-probe family;
- `ped_wait_by_crossing.png`: longest pedestrian curb wait (`ped_curb_wait_max`, creeps
  at the curb included) per crossing and scenario against the A-04 limit (the scenarios'
  `safety.ped_max_wait_s`, 90 s at Pitié);
- `a06_forest.png` (Part 8, when a06.json exists): A-06, S3 − reference on the person
  delay per comparison, HL estimate and 95 % CI, raw and Holm p, the outcome.

    python -m traffic_twin.viz.evaluation --config configs/evaluation.yaml
        --out docs/figures/pitie/eval [--results data/results/evaluation]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from traffic_twin.config import ConfigError
from traffic_twin.evaluation.compare import QUALIFIER_SEP, _sense, unit
from traffic_twin.paths import EVALUATION_CONFIG
from traffic_twin.viz import style
from traffic_twin.viz.plots import Scenario, plot_kpi_by_user

log = logging.getLogger(__name__)

FIGURES = (
    "kpi_by_user.png", "paired_vs_s0.png", "equity_heatmap.png", "a05_ambulance.png",
    "ped_wait_by_crossing.png", "a06_forest.png",
)  # fmt: skip
A06_FIGURE = FIGURES[5]  # drawn only when the evaluation wrote a06.json (Part 8)
OVERVIEW_LABELS = ("S0", "S1", "S2", "S3")  # the controllers of the deliverable (no ablation)
EXTRA_PAIRED = [("all", "person_delay_mean")]  # shown after the headline KPIs
# Diverging pair (13.3 heatmap): cool = better, warm = worse, grey midpoint; light poles
# so the ink of the cell text stays readable on every cell.
DIVERGING = ("#e07b6d", "#f0efec", "#6da7ec")
HEAT_LIMIT = 25.0  # % (pp for rates) at full colour
CROSSING = "__crossing_"
PED_KPI = "ped_curb_wait_max"  # judged by A-04 (configs/evaluation.yaml changes 2026-10-12)
DEFAULT_PED_LIMIT = 90.0  # spec 90 s target, used only when no a04.csv gives the bound


# -- inputs ----------------------------------------------------------------------------
def read_family(results: Path, family: str) -> dict:
    """The outputs of one family (`kpis`, `equity`, `a04`, `pairs`); empty frames for
    absent files."""
    d = Path(results) / family

    def csv(name: str) -> pd.DataFrame:
        p = d / name
        return pd.read_csv(p) if p.is_file() else pd.DataFrame()

    pairs = {}
    for p in sorted((d / "pairs").glob("*__vs__*.csv")):
        a, b = p.stem.split("__vs__")
        pairs[(a, b)] = pd.read_csv(p)
    return {"kpis": csv("kpis.csv"), "equity": csv("equity.csv"), "a04": csv("a04.csv"),
            "pairs": pairs}  # fmt: skip


def _config_controllers(fam_cfg: dict) -> dict[str, str]:
    """label -> controller of each scenario config ({} when one cannot be loaded)."""
    from traffic_twin.config import load_scenario

    try:
        return {x: str(load_scenario(p)["controller"]) for x, p in fam_cfg["scenarios"].items()}
    except (ConfigError, FileNotFoundError, OSError, KeyError):
        return {}


def label_order(fam_cfg: dict, controllers: dict[str, str] | None = None) -> list[str]:
    """The family's labels in the order an ablation reads in: the baseline first, then
    per controller (in the evaluation.yaml order) its variants before its main scenario,
    the first one of the controller in the config (S0, S1, S2_noP, S2, S4, S3).
    `controllers` (label -> controller; default from the scenario configs). Without
    controllers: the baseline first and the scenario every pair leads to last."""
    labels = list(fam_cfg["scenarios"])
    base = fam_cfg["baseline"]
    ctl = dict(controllers or {})
    if not all(x in ctl for x in labels):
        ctl = {**_config_controllers(fam_cfg), **ctl}
    if all(x in ctl for x in labels):
        groups: dict[str, list[str]] = {}
        for x in labels:
            groups.setdefault(ctl[x], []).append(x)
        order = [base]
        for members in groups.values():
            rest = [x for x in members if x != base]
            if members[0] == base:
                order += rest
            elif rest:
                order += [*rest[1:], rest[0]]
        return order
    pairs = fam_cfg.get("pairs") or []
    ends = {p["b"] for p in pairs} - {p["a"] for p in pairs}
    middle = [x for x in labels if x != base and x not in ends]
    return [base, *middle, *[x for x in labels if x in ends and x != base]]


def controllers(kpis: pd.DataFrame) -> dict[str, str]:
    if kpis.empty:
        return {}
    first = kpis.drop_duplicates("label")
    return dict(zip(first["label"], first["controller"], strict=True))


def colors_for(kpis: pd.DataFrame) -> dict[str, str]:
    """Label -> colour, the same in every figure: the first scenario of a controller (in
    the evaluation.yaml order, which kpis.csv keeps) takes the controller's colour, a
    later variant of it (the S2_noP ablation) the muted grey."""
    out: dict[str, str] = {}
    used: set[str] = set()
    for label, ctl in controllers(kpis).items():
        color = style.CONTROLLER_COLORS.get(ctl)
        out[label] = style.MUTED if color is None or ctl in used else color
        used.add(ctl)
    return out


def values(kpis: pd.DataFrame, label: str, user: str, kpi: str) -> pd.Series:
    """seed -> value of one KPI of one scenario."""
    k = kpis[(kpis["label"] == label) & (kpis["user_type"] == user) & (kpis["kpi"] == kpi)]
    return k.set_index("seed")["value"].sort_index()


# -- 1. KPI per user -------------------------------------------------------------------
def plot_overview(kpis: pd.DataFrame, labels: list[str], out: Path, ped_target: float) -> Path:
    """`plot_kpi_by_user` over the deliverable's scenarios (S0, S1, S2, S3)."""
    ctl = controllers(kpis)
    shown = [x for x in labels if x in OVERVIEW_LABELS] or labels
    scenarios = [
        Scenario(
            name=x,
            controller=ctl.get(x, x),
            kpis=kpis[kpis["label"] == x][["seed", "user_type", "kpi", "value"]],
            trips=pd.DataFrame(),
            state=pd.DataFrame(),
            # Pitié runs are judged on the single pedestrian wait (`measurement` set)
            config={"measurement": True, "safety": {"ped_max_wait_s": ped_target}},
        )
        for x in shown
    ]
    return plot_kpi_by_user(scenarios, out)


# -- 2. paired differences vs the baseline ---------------------------------------------
def plot_paired_vs_baseline(
    kpis: pd.DataFrame, pairs: dict, baseline: str, labels: list[str], out: Path
) -> Path:
    """One panel per headline KPI (+ the person-weighted delay): per-seed difference
    scenario − baseline (dots), its mean ± 95 % t CI (black) and p (Holm in the pair)."""
    others = [b for b in labels if (baseline, b) in pairs]
    tables = {b: pairs[(baseline, b)].set_index(["user_type", "kpi"]) for b in others}
    first = pairs[(baseline, others[0])] if others else pd.DataFrame(columns=["headline"])
    panels = [tuple(x) for x in first[first["headline"]][["user_type", "kpi"]].to_numpy()]
    panels += [p for p in EXTRA_PAIRED if others and p in tables[others[0]].index]
    ncols = min(max(len(panels), 1), 4)
    nrows = max(math.ceil(len(panels) / ncols), 1)
    fig, head, axes = style.figure(height=2.9 * nrows, ncols=ncols, nrows=nrows, squeeze=False)
    colors = colors_for(kpis)
    flat = axes.ravel()
    for k, (ax, (user, kpi)) in enumerate(zip(flat, panels, strict=False)):
        ax.axhline(0, color=style.AXIS, lw=1.0, zorder=1)
        base = values(kpis, baseline, user, kpi)
        notes = []
        for i, b in enumerate(others):
            d = (values(kpis, b, user, kpi) - base).dropna()
            jitter = np.linspace(-0.16, 0.16, len(d)) if len(d) > 1 else np.zeros(len(d))
            ax.scatter(i - 0.12 + jitter, d, s=18, color=colors[b], alpha=0.85, lw=0, zorder=3)
            r = tables[b].loc[(user, kpi)] if (user, kpi) in tables[b].index else None
            if r is not None and pd.notna(r["diff_ci_low"]):
                m = r["diff_mean"]
                err = [[m - r["diff_ci_low"]], [r["diff_ci_high"] - m]]
                ax.errorbar(
                    [i + 0.22], [m], yerr=err, fmt="o", color=style.INK, ms=4.5, capsize=3,
                    lw=1.2, zorder=4,
                )  # fmt: skip
            if r is not None:
                p = r["p_holm"] if pd.notna(r["p_holm"]) else r["p_value"]
                notes.append(f"{b} p {p:.2g}" if pd.notna(p) else f"{b} p –")
        ax.set_xlim(-0.6, len(others) - 0.4)
        ax.set_xticks(range(len(others)), others)
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"{style.user_label(user)}: {kpi}", fontsize=style.FONT_SIZE, pad=16)
        ax.annotate(
            " · ".join(notes), (0.0, 1.0), xycoords="axes fraction", xytext=(0, 3),
            textcoords="offset points", va="bottom", fontsize=style.FONT_SIZE - 2,
            color=style.INK_2,
        )  # fmt: skip
        if k % ncols == 0:
            ax.set_ylabel(f"Difference vs {baseline} ({unit(kpi) or 'count'})")
    for ax in flat[len(panels) :]:
        ax.set_visible(False)
    n = kpis["seed"].nunique() if len(kpis) else 0
    handles = [Patch(color=colors[b], label=b) for b in others]
    style.header(
        head,
        f"Paired per-seed differences vs {baseline}",
        f"{n} seeds, same demand · dots = seeds, black = mean ± 95 % t CI · below zero = "
        "lower than the baseline · p: Holm within each pair (headline), else unadjusted",
        handles,
    )
    return style.savefig(fig, out)


# -- 3. equity heatmap -----------------------------------------------------------------
def heat_rows(equity: pd.DataFrame, global_kpi: tuple[str, str]) -> list[tuple[str, str]]:
    """The equity rows shown: primary, rate, subgroup and global KPIs, by user."""
    e = equity
    keep = e["primary"].astype(bool) | e["subgroup"].astype(bool) | e["pp_change"].notna()
    keep |= (e["user_type"] == global_kpi[0]) & (e["kpi"] == global_kpi[1])
    rows = list(dict.fromkeys(zip(e.loc[keep, "user_type"], e.loc[keep, "kpi"], strict=True)))
    users = [*style.USER_TYPES, global_kpi[0]]
    return sorted(
        rows, key=lambda r: (users.index(r[0]) if r[0] in users else 9, QUALIFIER_SEP in r[1])
    )


def plot_equity_heatmap(
    equity: pd.DataFrame, out: Path, alpha: float = 0.05, global_kpi=("all", "person_delay_mean")
) -> Path:
    """Change vs the baseline per scenario column; colour = signed improvement (blue
    better, red worse, clipped at ±HEAT_LIMIT), text = change, `*` p < alpha, `!` flag."""
    rows = heat_rows(equity, tuple(global_kpi))
    cols = list(dict.fromkeys(zip(equity["family"], equity["scenario"], strict=True)))
    grid = np.full((len(rows), len(cols)), np.nan)
    text = [[""] * len(cols) for _ in rows]
    for r in equity.itertuples(index=False):
        key = (r.user_type, r.kpi)
        if key not in rows:
            continue
        i, j = rows.index(key), cols.index((r.family, r.scenario))
        rate = pd.notna(r.pp_change)
        change = r.pp_change if rate else r.rel_change_pct
        if pd.isna(change):
            continue
        grid[i, j] = _sense(r.kpi) * change
        star = "*" if pd.notna(r.p_value) and r.p_value < alpha else ""
        flag = " !" if bool(r.worse_flag) else ""
        text[i][j] = f"{change:+.2f} pp{star}{flag}" if rate else f"{change:+.1f} %{star}{flag}"
    height = 0.32 * max(len(rows), 1) + 0.6
    fig, head, ax = style.figure(width=3.4 + 1.6 * max(len(cols), 1), height=height, header_in=0.6)
    cmap = LinearSegmentedColormap.from_list("equity", DIVERGING)
    ax.imshow(
        np.clip(grid, -HEAT_LIMIT, HEAT_LIMIT), cmap=cmap, vmin=-HEAT_LIMIT, vmax=HEAT_LIMIT,
        aspect="auto",
    )  # fmt: skip
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, text[i][j] or "–", ha="center", va="center", color=style.INK,
                    fontsize=style.FONT_SIZE - 2)  # fmt: skip
    ax.set_yticks(range(len(rows)), [f"{style.user_label(u)} · {k}" for u, k in rows])
    fams = {f for f, _ in cols}
    xt = [f"{s}" + (f"\n({f})" if len(fams) > 1 else "") for f, s in cols]
    ax.set_xticks(range(len(cols)), xt)
    ax.tick_params(axis="both", length=0)
    ax.xaxis.tick_top()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    style.header(
        head,
        "Equity: change vs the baseline per user (13.3)",
        f"blue = better, red = worse (colour clipped at ±{HEAT_LIMIT:g} %) · rates in "
        f"percentage points · * p < {alpha:g} (two-sided, unadjusted) · ! flagged worse",
    )
    return style.savefig(fig, out)


# -- 4. A-05 ---------------------------------------------------------------------------
def _slope_panel(ax, kpis: pd.DataFrame, labels: list[str], user: str, kpi: str, colors) -> None:
    series = {x: values(kpis, x, user, kpi) for x in labels}
    seeds = sorted(set().union(*[set(v.index) for v in series.values()]))
    for s in seeds:
        ys = [series[x].get(s, np.nan) for x in labels]
        ax.plot(range(len(labels)), ys, color=style.GRID, lw=1.2, zorder=1)
    for i, x in enumerate(labels):
        v = series[x]
        ax.scatter(
            [i] * len(v), v, s=26, color=colors[x], edgecolor=style.SURFACE, lw=1.0, zorder=3
        )
        if len(v):  # the mean beside the dots, clear of the panel title
            ax.annotate(
                f"{v.mean():.0f} s", (i, v.mean()), xytext=(9, 0), textcoords="offset points",
                ha="left", va="center", color=style.INK, fontsize=style.FONT_SIZE,
                fontweight="semibold",
            )  # fmt: skip
    top = max((float(v.max()) for v in series.values() if len(v)), default=1.0)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.3, len(labels) - 0.45)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, top * 1.1)


def _test_note(r: dict | None) -> str:
    if not r or r.get("status") == "not evaluated" or r.get("p_value") is None:
        return "not evaluated"
    ci = r.get("hl_ci") or [None, None]
    lo, hi = (f"{x:.0f}" if x is not None else "–" for x in ci)
    hl = f"{r['hl']:.0f}" if r.get("hl") is not None else "–"
    return f"{r['b']} − {r['a']}: HL {hl} s [{lo}, {hi}], one-sided p {r['p_value']:.2g}"


def plot_a05(
    main: pd.DataFrame, probe: pd.DataFrame, a05: dict, probe_labels: list[str], out: Path
) -> Path:
    """Left: the primary test's per-seed values (a -> b); then each secondary test of
    the same family (Part 8: S0 -> S3); right: the ambulance-probe family (mean over its
    ambulances per run). Grey lines join one seed."""
    p = a05.get("primary") or {}
    user, kpi = p.get("user", "ambulance"), p.get("kpi", "travel_time_total")
    panels = [(main, [p.get("a", "S0"), p.get("b", "S2")], p)]
    for sec in a05.get("secondary") or []:
        same = sec.get("family") == p.get("family") and sec.get("status") != "not evaluated"
        if same and not main.empty and {sec["a"], sec["b"]} <= set(main["label"]):
            panels.append((main, [sec["a"], sec["b"]], sec))
    if not probe.empty:
        sec = next(
            (s for s in a05.get("secondary") or [] if s.get("family") == "ambulance_probe"), None
        )
        panels.append((probe, [x for x in probe_labels if (probe["label"] == x).any()], sec))
    fig, head, axes = style.figure(height=3.4, header_in=0.6, ncols=len(panels), squeeze=False)
    for ax, (kp, labels, test) in zip(axes[0], panels, strict=True):
        colors = colors_for(kp)
        _slope_panel(ax, kp, labels, user, kpi, colors)
        fam = kp["family"].iloc[0] if len(kp) else ""
        role = "" if test is p else " · secondary"
        ax.set_title(f"{fam}: {kpi}{role}", fontsize=style.FONT_SIZE, pad=16)
        ax.annotate(
            _test_note(test), (0.0, 1.0), xycoords="axes fraction", xytext=(0, 3),
            textcoords="offset points", va="bottom", fontsize=style.FONT_SIZE - 2,
            color=style.INK_2,
        )  # fmt: skip
        ax.set_ylabel("Arrival − planned departure (s)")
    verdict = p.get("status", "not evaluated")
    style.header(
        head,
        f"Ambulance travel time from its planned departure (A-05 {verdict})",
        "dots = seeds (probe family: mean of the run's ambulances), grey = one seed · "
        "label = mean over seeds",
    )
    return style.savefig(fig, out)


# -- 6. A-06 forest (Part 8) -----------------------------------------------------------
def plot_a06_forest(a06: dict, out: Path, colors: dict[str, str] | None = None) -> Path:
    """One row per A-06 comparison: the HL estimate of S3 − reference on the person
    delay with its 95 % CI, zero line, and the raw / Holm p and the pre-registered
    outcome beside it. The status (PASS / INVALID) is in the title."""
    comps = a06.get("comparisons") or []
    cand = a06.get("candidate", "S3")
    colors = colors or {}
    fig, head, ax = style.figure(height=0.62 * max(len(comps), 1) + 0.9, header_in=0.75)
    ax.axvline(0, color=style.AXIS, lw=1.2, zorder=1)
    span = [0.0]
    for i, c in enumerate(comps):
        y = len(comps) - 1 - i
        hl, lo, hi = c.get("hl"), c.get("ci_low"), c.get("ci_high")
        color = colors.get(cand, style.CONTROLLER_COLORS["rl"])
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [y, y], color=color, lw=2.4, solid_capstyle="round", zorder=2)
            span += [lo, hi]
        if hl is not None:
            ax.scatter([hl], [y], s=46, color=color, edgecolor=style.SURFACE, lw=1.2, zorder=3)
            span.append(hl)
        p, ph = c.get("p"), c.get("p_holm")
        text = (
            f"p {p:.2g} · Holm {ph:.2g} · {c.get('outcome_text', c.get('outcome', ''))}"
            if p is not None and ph is not None
            else "not tested"
        )
        ax.annotate(
            text, (1.0, y), xycoords=("axes fraction", "data"), xytext=(8, 0),
            textcoords="offset points", ha="left", va="center", color=style.INK,
            fontsize=style.FONT_SIZE - 1,
        )  # fmt: skip
    lo, hi = min(span), max(span)
    pad = 0.15 * (hi - lo) if hi > lo else 1.0
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.6, len(comps) - 0.4)
    labels = [
        f"{cand} vs {c['reference']}"
        + (" (primary)" if c["reference"] == a06.get("primary") else "")
        for c in comps
    ]
    ax.set_yticks(range(len(comps)), labels[::-1])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    unit_ = unit(str(a06.get("kpi", "/person_delay_mean")).split("/", 1)[-1]) or "s"
    ax.set_xlabel(f"{cand} − reference, {a06.get('kpi', '')} ({unit_}); below 0 = {cand} lower")
    n = max((c.get("n") or 0 for c in comps), default=0)
    style.header(
        head,
        f"A-06 {a06.get('status', '')}: {cand} against the baselines",
        f"Hodges-Lehmann estimate (dot) and 95 % CI (bar) of the paired per-seed difference, "
        f"n = {n} · exact {a06.get('alternative', 'two-sided')} Wilcoxon, Holm across "
        f"{len(comps)} · procedural status, never FAIL on direction",
    )
    return style.savefig(fig, out)


# -- 5. pedestrian waits per crossing --------------------------------------------------
def crossing_names(kpis: pd.DataFrame) -> list[str]:
    k = kpis["kpi"].astype(str)
    names = k[k.str.startswith(PED_KPI + CROSSING)].str.split(CROSSING).str[1]
    return sorted(set(names))


def plot_ped_waits(
    families: dict[str, tuple[pd.DataFrame, list[str]]], out: Path, limit: float = 90.0
) -> Path:
    """Per family, per crossing (and over all crossings): the longest pedestrian curb
    wait (`ped_curb_wait_max`) of each scenario, worst seed (bar) and every seed (dots),
    against the limit."""
    fig, head, axes = style.figure(height=3.4, ncols=max(len(families), 1), squeeze=False)
    handles: dict[str, Patch] = {}
    for ax, (fam, (kp, labels)) in zip(axes[0], families.items(), strict=False):
        groups = [*crossing_names(kp), "all"]
        colors = colors_for(kp)
        width = 0.8 / max(len(labels), 1)
        top = limit
        for i, x in enumerate(labels):
            for g, name in enumerate(groups):
                kpi = PED_KPI + ("" if name == "all" else CROSSING + name)
                v = values(kp, x, "pedestrian", kpi)
                if v.empty:
                    continue
                pos = g - 0.4 + width * (i + 0.5)
                ax.bar(pos, v.max(), width=width * 0.9, color=colors[x], zorder=2)
                ax.scatter([pos] * len(v), v, s=10, color=style.SURFACE, edgecolor=style.INK,
                           lw=0.7, zorder=3)  # fmt: skip
                top = max(top, float(v.max()))
            handles.setdefault(x, Patch(color=colors[x], label=x))
        style.target_line(ax, limit, f"limit\n{limit:g} s")
        ax.set_ylim(0, top * 1.15)
        ax.set_xticks(range(len(groups)), [g if g != "all" else "all crossings" for g in groups])
        ax.tick_params(axis="x", length=0)
        ax.set_title(fam, fontsize=style.FONT_SIZE)
        ax.set_ylabel("Longest curb wait (s)")
    style.header(
        head,
        "Longest pedestrian curb wait per crossing (A-04)",
        "bar = worst seed, dots = seeds · the guard's pedestrian shield keeps every curb "
        "wait ≤ the limit minus its margin",
        list(handles.values()),
    )
    return style.savefig(fig, out)


# -- all -------------------------------------------------------------------------------
def ped_limit(data: dict[str, dict]) -> float:
    """The A-04 pedestrian bound: the largest `ped_max_wait_s` of the families' a04.csv
    (each scenario's `safety.ped_max_wait_s`), else 90 s."""
    bounds = [
        float(x)
        for d in data.values()
        if "ped_max_wait_s" in d["a04"]
        for x in d["a04"]["ped_max_wait_s"].dropna()
    ]
    return max(bounds) if bounds else DEFAULT_PED_LIMIT


def make_figures(ecfg: dict, results: Path, out_dir: Path) -> list[Path]:
    """Every figure whose inputs exist (the main family is required; `a06_forest.png`
    needs a06.json)."""
    out_dir = Path(out_dir)
    fams = ecfg["families"]
    data = {name: read_family(results, name) for name in fams}
    main_name = next(iter(fams))
    main = data[main_name]
    if main["kpis"].empty:
        raise FileNotFoundError(
            f"no {main_name}/kpis.csv under {results}: run `python -m "
            "traffic_twin.evaluation.evaluate` first"
        )
    limit = ped_limit(data)
    main_labels = label_order(fams[main_name], controllers(main["kpis"]) or None)
    written = [plot_overview(main["kpis"], main_labels, out_dir / FIGURES[0], limit)]
    written.append(
        plot_paired_vs_baseline(
            main["kpis"], main["pairs"], fams[main_name]["baseline"], main_labels,
            out_dir / FIGURES[1],
        )
    )  # fmt: skip
    equity = pd.concat([d["equity"] for d in data.values() if not d["equity"].empty])
    g = ecfg["equity"]["global"]
    written.append(
        plot_equity_heatmap(equity, out_dir / FIGURES[2], global_kpi=(g["user"], g["kpi"]))
    )
    a05_path = Path(results) / "a05.json"
    a05 = json.loads(a05_path.read_text(encoding="utf-8")) if a05_path.is_file() else {}
    probe_name = next((n for n in fams if n != main_name), None)
    probe = data[probe_name]["kpis"] if probe_name else pd.DataFrame()
    probe_ctl = controllers(probe) or None
    probe_labels = label_order(fams[probe_name], probe_ctl) if probe_name else []
    written.append(plot_a05(main["kpis"], probe, a05, probe_labels, out_dir / FIGURES[3]))
    peds = {main_name: (main["kpis"], main_labels)}
    if probe_name and not probe.empty:
        peds[probe_name] = (probe, probe_labels)
    written.append(plot_ped_waits(peds, out_dir / FIGURES[4], limit))
    a06_path = Path(results) / "a06.json"
    if a06_path.is_file():
        a06 = json.loads(a06_path.read_text(encoding="utf-8"))
        written.append(plot_a06_forest(a06, out_dir / A06_FIGURE, colors_for(main["kpis"])))
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=EVALUATION_CONFIG)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/pitie/eval"))
    parser.add_argument("--results", type=Path, help="default: evaluation.yaml outputs.dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from traffic_twin.evaluation.evaluate import load_evaluation

    try:
        ecfg = load_evaluation(args.config)
    except (ConfigError, FileNotFoundError) as e:
        raise SystemExit(f"evaluation config error: {e}") from None
    results = args.results or Path(ecfg["outputs"]["dir"])
    try:
        paths = make_figures(ecfg, results, args.out)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
    for p in paths:
        log.info("figure: %s", p.as_posix())


if __name__ == "__main__":
    main()

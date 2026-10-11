"""First look at the cleaned sensor data: coverage table and hourly profiles per zone
(typical days, i.e. school-term Tue-Thu, plus the term weekends of the busiest arc when
the hourly sensor file is there).

Each hourly mean is drawn as a step over the hour it covers. The published stamp marks
the END of the period (see `data.calendar`), so the value stamped 09:00 is the step
from 08:00 to 09:00. The shaded band is the simulated 07:30-09:30 window in clock time:
it covers half of the 07:00 and 09:00 steps and all of the 08:00 one, as the demand
does (decision D5). The day set of each curve comes from the profile itself (`day_set`,
written by `data.clean`), never from the config. A profile written before the
period-start fix (an end-of-period `hour` column) is refused.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_twin.config import DATA_KEYS, load_config
from traffic_twin.data.calendar import WEEKEND
from traffic_twin.data.clean import hourly_profile
from traffic_twin.paths import (
    ARC_COVERAGE_FILE,
    DATA_CONFIG,
    HOURLY_PROFILE_FILE,
    SENSOR_HOURLY_FILE,
)
from traffic_twin.viz import style

log = logging.getLogger(__name__)

N_ARCS = 4  # one colour per arc, from the validated 4-colour palette
SIM_WINDOW_H = (7.5, 9.5)  # 07:30-09:30 clock time, the simulated morning peak
HOUR = "hour_start"  # profile column: hour the period starts (local time)
DAY_SET = "day_set"  # profile column: label of the days averaged
ARC_COLORS = tuple(style.USER_COLORS.values())
RECLEAN = "rerun `python -m traffic_twin.data.clean --config configs/data.yaml` (no download)"


def check_profile(profile: pd.DataFrame) -> None:
    """Refuse a profile that cannot be labelled honestly: one written before the
    period-start fix (its `hour` was the END of the period and no calendar filtered its
    days) or without its `day_set` label. Raises ValueError."""
    if HOUR not in profile:
        raise ValueError(
            f"profile has no '{HOUR}' column: it predates the period-start fix (its 'hour' "
            f"is the end of the period, holidays included); {RECLEAN}"
        )
    if DAY_SET not in profile:
        raise ValueError(f"profile has no '{DAY_SET}' column (which days it averages); {RECLEAN}")
    labels = profile[DAY_SET].dropna().unique()
    if len(labels) > 1:
        raise ValueError(f"profile mixes day sets {sorted(labels)}")


def day_set(profile: pd.DataFrame) -> str:
    """The label of the days a profile averages, e.g. "school-term Tue–Thu"."""
    check_profile(profile)
    labels = profile[DAY_SET].dropna().unique()
    return str(labels[0]) if len(labels) else ""


def top_arcs(profile: pd.DataFrame, zone: str, n: int = N_ARCS) -> list[str]:
    """Busiest arcs of a zone by mean daily flow, skipping arcs whose hourly profile
    duplicates one already picked (paired sensors reporting the same counts)."""
    check_profile(profile)
    z = profile[profile["zone"] == zone]
    ranked = z.groupby("iu_ac")["flow_mean"].sum().sort_values(ascending=False).index
    picked: list[str] = []
    seen: list[np.ndarray] = []
    for arc in ranked:
        v = z[z["iu_ac"] == arc].sort_values(HOUR)["flow_mean"].to_numpy()
        if any(len(s) == len(v) and np.allclose(s, v, rtol=0.01) for s in seen):
            continue
        picked.append(arc)
        seen.append(v)
        if len(picked) == n:
            break
    return picked


def arc_label(arc: str, libelle: str) -> str:
    name = str(libelle).replace("_", " ").replace("Bd ", "Bd ").replace("Av ", "Av. ")
    return f"{name} ({arc})"


def steps(rows: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Post-step line of one arc's hourly values: x = 0..24, y = the value of each
    `hour_start` (NaN for a missing hour) with the last one repeated, so each mean spans
    [hour_start, hour_start + 1) on a clock-time axis."""
    y = rows.groupby(HOUR)[col].mean().reindex(range(24)).to_numpy(dtype=float)
    return np.arange(25, dtype=float), np.append(y, y[-1])


def _hour_axis(ax) -> None:
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3), [f"{h:02d}h" for h in range(0, 25, 3)])
    ax.set_xlabel("Local time (each step: one sensor hour)")
    ax.axvspan(*SIM_WINDOW_H, color=style.GRID, alpha=0.6, lw=0, zorder=0)


def _n_days(rows: pd.DataFrame) -> int:
    return int(rows["n_days"].median()) if len(rows) and "n_days" in rows else 0


def _band(ax, rows: pd.DataFrame, name: str, color, band_color, ls) -> None:
    """Mean step line and p10-p90 band of one arc, labelled with its day set."""
    x, lo = steps(rows, "flow_p10")
    hi, mean = steps(rows, "flow_p90")[1], steps(rows, "flow_mean")[1]
    ax.fill_between(x, lo, hi, step="post", color=band_color, alpha=0.12, lw=0)
    label = f"{name} (~{_n_days(rows)} days)"
    ax.plot(x, mean, color=color, ls=ls, drawstyle="steps-post", label=label)


def plot_zone(
    profile: pd.DataFrame,
    zone: str,
    label: str,
    out_dir: Path,
    weekend: pd.DataFrame | None = None,
) -> Path:
    """Hourly profile figure of one zone: flow and occupancy of its busiest arcs over the
    profile's days (`day_set`), and the busiest arc against `weekend` (an `hourly_profile`
    of weekend days) when given."""
    wd_name = day_set(profile)
    arcs = top_arcs(profile, zone)
    z = profile[profile["zone"] == zone]
    has_weekend = weekend is not None and not weekend[weekend["zone"] == zone].empty

    ncols = 3 if has_weekend else 2
    fig, head, axes = style.figure(height=3.6, ncols=ncols)
    ax_q, ax_k = axes[0], axes[1]
    handles = []
    for arc, color in zip(arcs, ARC_COLORS, strict=False):
        a = z[z["iu_ac"] == arc]
        (h,) = ax_q.plot(
            *steps(a, "flow_mean"), color=color, drawstyle="steps-post",
            label=arc_label(arc, a["libelle"].iloc[0]),
        )  # fmt: skip
        ax_k.plot(*steps(a, "occupancy_mean"), color=color, drawstyle="steps-post")
        handles.append(h)

    ax_q.set_title(f"Flow, {wd_name} mean")
    ax_q.set_ylabel("Flow (veh/h)")
    ax_k.set_title(f"Occupancy, {wd_name} mean")
    ax_k.set_ylabel("Occupancy (% of time)")
    if has_weekend:
        ax_w = axes[2]
        busiest = arcs[0]
        we = weekend[(weekend["zone"] == zone) & (weekend["iu_ac"] == busiest)]
        _band(ax_w, z[z["iu_ac"] == busiest], wd_name, ARC_COLORS[0], ARC_COLORS[0], "-")
        _band(ax_w, we, day_set(weekend), style.INK_2, style.MUTED, (0, (4, 2)))
        ax_w.set_title(f"Busiest arc ({busiest}), weekday vs weekend")
        ax_w.set_ylabel("Flow (veh/h)")
        ax_w.legend(
            loc="lower right", title="mean, band p10–p90", alignment="left",
            title_fontsize=style.FONT_SIZE - 2,
        )  # fmt: skip

    for ax in axes:
        _hour_axis(ax)
        ax.set_ylim(bottom=0)
    style.header(
        head,
        f"Hourly traffic profile · {label}",
        f"{len(arcs)} busiest sensor arcs · {wd_name} mean over ~{_n_days(z)} days · "
        "shaded: 07:30–09:30, the simulated morning peak",
        handles,
    )
    style.note(
        fig,
        "Source: Ville de Paris, comptages routiers permanents (opendata.paris.fr). "
        "A value stamped hh:00 covers the hour before it.",
    )
    return style.savefig(fig, out_dir / f"profile_{zone}.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DATA_CONFIG)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config, DATA_KEYS)
    processed, figures = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["figures"])
    path = processed / HOURLY_PROFILE_FILE
    profile = pd.read_parquet(path)
    try:
        check_profile(profile)
    except ValueError as e:
        raise SystemExit(f"{path}: {e}") from None
    coverage = pd.read_csv(processed / ARC_COVERAGE_FILE)
    weekend = None
    if (processed / SENSOR_HOURLY_FILE).exists():
        sensors = pd.read_parquet(processed / SENSOR_HOURLY_FILE)
        # term weekends with a calendar (same period and holidays as the weekday curve)
        weekend = hourly_profile(sensors, cfg.get("calendar"), list(WEEKEND))

    for zone, z in cfg["zones"].items():
        zc = coverage[coverage["zone"] == zone]
        good = zc[zc["coverage"] >= 0.8]
        print(f"\n== {z['label']}: {len(zc)} arcs, {len(good)} with >=80% coverage")
        print(zc.head(15).to_string(index=False))
        if zone in set(profile["zone"]):
            out = plot_zone(profile, zone, z["label"], figures, weekend)
            log.info("figure: %s", out)


if __name__ == "__main__":
    main()

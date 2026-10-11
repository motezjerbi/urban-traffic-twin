"""First look at the cleaned sensor data: coverage table and hourly profiles per zone
(typical weekdays, plus weekend for the busiest arc when the hourly sensor file is there)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_twin.config import DATA_KEYS, load_config
from traffic_twin.data.clean import hourly_profile, select_weekdays
from traffic_twin.paths import (
    ARC_COVERAGE_FILE,
    DATA_CONFIG,
    HOURLY_PROFILE_FILE,
    SENSOR_HOURLY_FILE,
)
from traffic_twin.viz import style

log = logging.getLogger(__name__)

WEEKEND = [5, 6]  # Saturday, Sunday
N_ARCS = 4  # one colour per arc, from the validated 4-colour palette
SIM_WINDOW_H = (7.5, 9.5)  # 07:30-09:30, the simulated morning peak
ARC_COLORS = tuple(style.USER_COLORS.values())
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def top_arcs(profile: pd.DataFrame, zone: str, n: int = N_ARCS) -> list[str]:
    """Busiest arcs of a zone by mean daily flow, skipping arcs whose hourly profile
    duplicates one already picked (paired sensors reporting the same counts)."""
    z = profile[profile["zone"] == zone]
    ranked = z.groupby("iu_ac")["flow_mean"].sum().sort_values(ascending=False).index
    picked: list[str] = []
    seen: list[np.ndarray] = []
    for arc in ranked:
        v = z[z["iu_ac"] == arc].sort_values("hour")["flow_mean"].to_numpy()
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


def day_label(weekdays: list[int]) -> str:
    days = sorted(weekdays)
    if len(days) > 1 and days == list(range(days[0], days[-1] + 1)):
        return f"{DAY_NAMES[days[0]]}–{DAY_NAMES[days[-1]]}"
    return ", ".join(DAY_NAMES[d] for d in days)


def _hour_axis(ax) -> None:
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}h" for h in range(0, 24, 3)])
    ax.set_xlabel("Hour of day")
    ax.axvspan(*SIM_WINDOW_H, color=style.GRID, alpha=0.6, lw=0, zorder=0)


def plot_zone(
    profile: pd.DataFrame,
    zone: str,
    label: str,
    out_dir: Path,
    weekend: pd.DataFrame | None = None,
    weekdays: list[int] | None = None,
) -> Path:
    """Hourly profile figure of one zone: flow and occupancy of its busiest arcs on
    typical weekdays, and weekday vs weekend flow for the busiest arc."""
    weekdays = weekdays or [1, 2, 3]
    wd_name = day_label(weekdays)
    arcs = top_arcs(profile, zone)
    z = profile[profile["zone"] == zone]
    has_weekend = weekend is not None and not weekend[weekend["zone"] == zone].empty

    ncols = 3 if has_weekend else 2
    fig, head, axes = style.figure(height=3.6, ncols=ncols)
    ax_q, ax_k = axes[0], axes[1]
    handles = []
    for arc, color in zip(arcs, ARC_COLORS, strict=False):
        a = z[z["iu_ac"] == arc].sort_values("hour")
        (h,) = ax_q.plot(
            a["hour"], a["flow_mean"], color=color, label=arc_label(arc, a["libelle"].iloc[0])
        )
        ax_k.plot(a["hour"], a["occupancy_mean"], color=color)
        handles.append(h)

    ax_q.set_title(f"Flow, {wd_name} mean")
    ax_q.set_ylabel("Flow (veh/h)")
    ax_k.set_title(f"Occupancy, {wd_name} mean")
    ax_k.set_ylabel("Occupancy (% of time)")
    if has_weekend:
        ax_w = axes[2]
        busiest, color = arcs[0], ARC_COLORS[0]
        a = z[z["iu_ac"] == busiest].sort_values("hour")
        w = weekend[(weekend["zone"] == zone) & (weekend["iu_ac"] == busiest)].sort_values("hour")
        ax_w.fill_between(a["hour"], a["flow_p10"], a["flow_p90"], color=color, alpha=0.12, lw=0)
        ax_w.plot(a["hour"], a["flow_mean"], color=color, label=wd_name)
        ax_w.fill_between(
            w["hour"], w["flow_p10"], w["flow_p90"], color=style.MUTED, alpha=0.12, lw=0
        )
        ax_w.plot(w["hour"], w["flow_mean"], color=style.INK_2, ls=(0, (4, 2)), label="Sat–Sun")
        ax_w.set_title(f"Busiest arc ({busiest}), weekday vs weekend")
        ax_w.set_ylabel("Flow (veh/h)")
        ax_w.legend(
            loc="lower right", title="mean, band p10–p90", alignment="left",
            title_fontsize=style.FONT_SIZE - 2,
        )  # fmt: skip

    for ax in axes:
        _hour_axis(ax)
        ax.set_ylim(bottom=0)
    n_days = int(z["n_days"].median()) if "n_days" in z else 0
    style.header(
        head,
        f"Hourly traffic profile · {label}",
        f"{len(arcs)} busiest sensor arcs · {wd_name} mean over ~{n_days} days · "
        "shaded: 07:30–09:30, the simulated morning peak",
        handles,
    )
    style.note(fig, "Source: Ville de Paris, comptages routiers permanents (opendata.paris.fr)")
    return style.savefig(fig, out_dir / f"profile_{zone}.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DATA_CONFIG)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config, DATA_KEYS)
    processed, figures = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["figures"])
    profile = pd.read_parquet(processed / HOURLY_PROFILE_FILE)
    coverage = pd.read_csv(processed / ARC_COVERAGE_FILE)
    weekdays = list(cfg["cleaning"]["weekdays"])
    weekend = None
    if (processed / SENSOR_HOURLY_FILE).exists():
        sensors = pd.read_parquet(processed / SENSOR_HOURLY_FILE)
        weekend = hourly_profile(select_weekdays(sensors, WEEKEND))

    for zone, z in cfg["zones"].items():
        zc = coverage[coverage["zone"] == zone]
        good = zc[zc["coverage"] >= 0.8]
        print(f"\n== {z['label']}: {len(zc)} arcs, {len(good)} with >=80% coverage")
        print(zc.head(15).to_string(index=False))
        if zone in set(profile["zone"]):
            path = plot_zone(profile, zone, z["label"], figures, weekend, weekdays)
            log.info("figure: %s", path)


if __name__ == "__main__":
    main()

"""First look at the cleaned sensor data: coverage table and weekday hourly profiles per zone."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from traffic_twin.config import load_config  # noqa: E402

log = logging.getLogger(__name__)


def top_arcs(profile: pd.DataFrame, zone: str, n: int = 6) -> list[str]:
    """Busiest arcs of a zone by mean daily flow."""
    z = profile[profile["zone"] == zone]
    return z.groupby("iu_ac")["flow_mean"].sum().nlargest(n).index.tolist()


def plot_zone(profile: pd.DataFrame, zone: str, label: str, out_dir: Path) -> Path:
    arcs = top_arcs(profile, zone)
    fig, (ax_q, ax_k) = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for arc in arcs:
        a = profile[(profile["zone"] == zone) & (profile["iu_ac"] == arc)].sort_values("hour")
        name = f"{arc} {a['libelle'].iloc[0]}"[:40]
        ax_q.plot(a["hour"], a["flow_mean"], label=name)
        ax_q.fill_between(a["hour"], a["flow_p10"], a["flow_p90"], alpha=0.12)
        ax_k.plot(a["hour"], a["occupancy_mean"])
    ax_q.set(title="Flow (veh/h), Tue-Thu mean, band p10-p90", xlabel="Hour", ylabel="veh/h")
    ax_k.set(title="Occupancy (%)", xlabel="Hour", ylabel="%")
    for ax in (ax_q, ax_k):
        ax.axvspan(7.5, 9.5, color="grey", alpha=0.1)
        ax.grid(alpha=0.3)
    ax_q.legend(fontsize=7, loc="upper left")
    fig.suptitle(label)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"profile_{zone}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    processed, figures = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["figures"])
    profile = pd.read_parquet(processed / "hourly_profile.parquet")
    coverage = pd.read_csv(processed / "arc_coverage.csv")

    for zone, z in cfg["zones"].items():
        zc = coverage[coverage["zone"] == zone]
        good = zc[zc["coverage"] >= 0.8]
        print(f"\n== {z['label']}: {len(zc)} arcs, {len(good)} with >=80% coverage")
        print(zc.head(15).to_string(index=False))
        if zone in set(profile["zone"]):
            log.info("figure: %s", plot_zone(profile, zone, z["label"], figures))


if __name__ == "__main__":
    main()

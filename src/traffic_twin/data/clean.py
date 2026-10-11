"""Clean raw sensor exports into the `sensor_hourly` table shape (spec 9.1).

Raw columns (Paris dataset): iu_ac, libelle, t_1h, q (flow, veh/h), k (occupancy, %),
etat_trafic, etat_barre, iu_nd_amont, libelle_nd_amont, iu_nd_aval, libelle_nd_aval,
geo_point_2d.
"""

from __future__ import annotations

import argparse
import logging
import struct
from pathlib import Path

import pandas as pd

from traffic_twin.config import load_config

log = logging.getLogger(__name__)

SENSOR_HOURLY_COLUMNS = [
    "iu_ac",
    "libelle",
    "zone",
    "ts",
    "flow_veh_h",
    "occupancy_pct",
    "traffic_state",
    "lat",
    "lon",
]

# Arc states that mean the measurement cannot be trusted
INVALID_BAR_STATES = {"Barré", "Invalide"}


def _split_geo(geo: pd.Series) -> tuple[pd.Series, pd.Series]:
    """geo_point_2d is WKB bytes (GeoParquet), a {lat, lon} dict (json) or a "lat, lon" string."""
    first = geo.dropna().iloc[0] if geo.notna().any() else None
    if isinstance(first, bytes | bytearray):
        # GeoParquet export: WKB point (byte order, uint32 type, x=lon, y=lat)
        def _xy(b):
            if not b:
                return (None, None)
            order = "<" if b[0] == 1 else ">"
            return struct.unpack(order + "dd", b[5:21])

        xy = geo.map(_xy)
        return xy.map(lambda p: p[1]), xy.map(lambda p: p[0])
    if isinstance(first, dict):
        return geo.map(lambda g: g and g.get("lat")), geo.map(lambda g: g and g.get("lon"))
    parts = geo.astype("string").str.split(",", n=1, expand=True)
    return pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")


def clean_sensor_frame(
    raw: pd.DataFrame,
    zone: str,
    timezone: str = "Europe/Paris",
    max_flow_veh_h: float = 4000,
    max_occupancy_pct: float = 100,
) -> pd.DataFrame:
    df = raw.copy()
    n0 = len(df)

    df["ts"] = pd.to_datetime(df["t_1h"], utc=True).dt.tz_convert(timezone)
    df["flow_veh_h"] = pd.to_numeric(df["q"], errors="coerce")
    df["occupancy_pct"] = pd.to_numeric(df["k"], errors="coerce")
    df["traffic_state"] = df.get("etat_trafic", pd.Series(index=df.index, dtype="string"))

    if "etat_barre" in df:
        df = df[~df["etat_barre"].isin(INVALID_BAR_STATES)]
    df = df[df["flow_veh_h"].between(0, max_flow_veh_h)]
    df = df[df["occupancy_pct"].between(0, max_occupancy_pct)]
    df = df.drop_duplicates(subset=["iu_ac", "ts"])

    if "geo_point_2d" in df:
        df["lat"], df["lon"] = _split_geo(df["geo_point_2d"])
    else:
        df["lat"] = df["lon"] = pd.NA
    df["zone"] = zone
    df["iu_ac"] = df["iu_ac"].astype(str)

    log.info("%s: kept %d / %d rows", zone, len(df), n0)
    return df[SENSOR_HOURLY_COLUMNS].sort_values(["iu_ac", "ts"]).reset_index(drop=True)


def select_weekdays(df: pd.DataFrame, weekdays: list[int]) -> pd.DataFrame:
    """Keep typical days (0=Monday). Spec 8.1 uses Tuesday to Thursday."""
    return df[df["ts"].dt.dayofweek.isin(weekdays)]


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of flow and occupancy per arc and hour of day."""
    g = df.assign(hour=df["ts"].dt.hour).groupby(["zone", "iu_ac", "libelle", "hour"])
    out = g.agg(
        flow_mean=("flow_veh_h", "mean"),
        flow_p10=("flow_veh_h", lambda s: s.quantile(0.1)),
        flow_p90=("flow_veh_h", lambda s: s.quantile(0.9)),
        occupancy_mean=("occupancy_pct", "mean"),
        n_days=("flow_veh_h", "size"),
    )
    return out.reset_index()


def arc_coverage(df: pd.DataFrame, since: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    """Share of expected hourly rows present per arc: a sensor below ~80% is unreliable."""
    expected = max(int((until - since) / pd.Timedelta(hours=1)), 1)
    cov = df.groupby(["zone", "iu_ac", "libelle"]).agg(
        rows=("ts", "size"), lat=("lat", "first"), lon=("lon", "first")
    )
    cov["coverage"] = (cov["rows"] / expected).clip(upper=1.0)
    return cov.reset_index().sort_values(["zone", "coverage"], ascending=[True, False])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    c = cfg["cleaning"]
    raw_dir, out_dir = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for zone in cfg["zones"]:
        path = raw_dir / f"{zone}.parquet"
        if not path.exists():
            log.warning("missing %s, run `make data` first", path)
            continue
        frames.append(
            clean_sensor_frame(
                pd.read_parquet(path),
                zone,
                c["timezone"],
                c["max_flow_veh_h"],
                c["max_occupancy_pct"],
            )
        )
    if not frames:
        raise SystemExit("no raw files found")

    sensors = pd.concat(frames, ignore_index=True)
    sensors.to_parquet(out_dir / "sensor_hourly.parquet", index=False)
    hourly_profile(select_weekdays(sensors, c["weekdays"])).to_parquet(
        out_dir / "hourly_profile.parquet", index=False
    )
    arc_coverage(sensors, sensors["ts"].min(), sensors["ts"].max()).to_csv(
        out_dir / "arc_coverage.csv", index=False
    )
    log.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()

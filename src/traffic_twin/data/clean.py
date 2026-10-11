"""Clean raw sensor exports into the `sensor_hourly` table shape (spec 9.1).

Raw columns (Paris dataset): iu_ac, libelle, t_1h, q (flow, veh/h), k (occupancy, %),
etat_trafic, etat_barre, iu_nd_amont, libelle_nd_amont, iu_nd_aval, libelle_nd_aval,
geo_point_2d.

`ts` keeps the published stamp, which is the END of the hourly period (Paris notice):
profiles are built on the period start (`hour_start`, see `data.calendar`). Flow and
occupancy are checked separately: a valid flow is kept when its occupancy is missing or
invalid (the occupancy is then NaN).
"""

from __future__ import annotations

import argparse
import logging
import struct
from pathlib import Path

import pandas as pd

from traffic_twin.config import DATA_KEYS, load_config
from traffic_twin.data.calendar import day_label, period_start, term_days
from traffic_twin.paths import (
    ARC_COVERAGE_FILE,
    DATA_CONFIG,
    HOURLY_PROFILE_FILE,
    SENSOR_HOURLY_FILE,
)

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


def _bar_ok(etat_barre: pd.Series | None, index: pd.Index) -> pd.Series:
    if etat_barre is None:
        return pd.Series(True, index=index)
    return ~etat_barre.isin(INVALID_BAR_STATES)


def valid_flow(
    flow: pd.Series, etat_barre: pd.Series | None = None, max_flow_veh_h: float = 4000
) -> pd.Series:
    """Flows that can be used: numeric, within [0, max] and the arc not barred/invalid."""
    q = pd.to_numeric(flow, errors="coerce")
    return q.between(0, max_flow_veh_h) & _bar_ok(etat_barre, q.index)


def valid_occupancy(
    occupancy: pd.Series, etat_barre: pd.Series | None = None, max_occupancy_pct: float = 100
) -> pd.Series:
    """Occupancies that can be used, checked independently of the flow."""
    k = pd.to_numeric(occupancy, errors="coerce")
    return k.between(0, max_occupancy_pct) & _bar_ok(etat_barre, k.index)


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

    bar = df["etat_barre"] if "etat_barre" in df else None
    occ_ok = valid_occupancy(df["occupancy_pct"], bar, max_occupancy_pct)
    df["occupancy_pct"] = df["occupancy_pct"].where(occ_ok)
    df = df[valid_flow(df["flow_veh_h"], bar, max_flow_veh_h)]
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
    """Keep the rows whose period starts on one of `weekdays` (0=Monday)."""
    return df[period_start(df["ts"]).dt.dayofweek.isin(weekdays).to_numpy()]


def hourly_profile(
    df: pd.DataFrame, cal: dict | None = None, weekdays: list[int] | None = None
) -> pd.DataFrame:
    """Mean and spread of flow and occupancy per arc and `hour_start` (the hour the
    period starts, local time), plus `day_set`, the label of the days used.

    With a calendar (`configs/data.yaml: calendar`) only school-term days are used
    (`data.calendar.term_days`: inside `calendar.typical_period`, no school or public
    holiday) whose weekday is in `weekdays` (default `calendar.weekdays`, i.e. the
    typical days); label e.g. "school-term Tue–Thu". Without a calendar every date of
    the rows is used, holidays included (only `weekdays` when given); label e.g.
    "Tue–Thu, all dates".
    """
    start = period_start(df["ts"])
    df = df.assign(
        hour_start=start.dt.hour.to_numpy(),
        date=start.dt.date.to_numpy(),
        weekday=start.dt.dayofweek.to_numpy(),
    )
    if cal is not None:
        df = df[df["date"].isin(set(term_days(df["date"].unique(), cal, weekdays)))]
        label = f"school-term {day_label(cal['weekdays'] if weekdays is None else weekdays)}"
    else:
        if weekdays is not None:
            df = df[df["weekday"].isin(list(weekdays))]
        days = sorted(set(df["weekday"])) if weekdays is None else weekdays
        label = f"{day_label(days)}, all dates" if len(days) else "all dates"
    g = df.groupby(["zone", "iu_ac", "libelle", "hour_start"])
    out = g.agg(
        flow_mean=("flow_veh_h", "mean"),
        flow_p10=("flow_veh_h", lambda s: s.quantile(0.1)),
        flow_p90=("flow_veh_h", lambda s: s.quantile(0.9)),
        occupancy_mean=("occupancy_pct", "mean"),
        n_days=("flow_veh_h", "count"),
    )
    return out.reset_index().assign(day_set=label)


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
    parser.add_argument("--config", type=Path, default=DATA_CONFIG)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config, DATA_KEYS)
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
    sensors.to_parquet(out_dir / SENSOR_HOURLY_FILE, index=False)
    cal = cfg.get("calendar")
    profile = hourly_profile(sensors, cal, None if cal else c["weekdays"])
    profile.to_parquet(out_dir / HOURLY_PROFILE_FILE, index=False)
    arc_coverage(sensors, sensors["ts"].min(), sensors["ts"].max()).to_csv(
        out_dir / ARC_COVERAGE_FILE, index=False
    )
    log.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()

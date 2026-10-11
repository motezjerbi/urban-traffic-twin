"""Demand calibration from the Paris sensors (EF-02): versioned reference extract and
calibration file of the Pitié-Salpêtrière junction (node 302).

    python -m traffic_twin.data.calibrate --config configs/data.yaml [--from-raw]

`--from-raw` reads `data/raw/<calibration.raw_zone>.parquet` and rewrites the versioned
extracts: `calibration.reference_csv` (hourly rows of the reference arcs, one row per
arc, date and hour_start, invalid values kept as NaN) and `calibration.arcs_csv` (arc
topology and shared flow stations). The calibration file (`calibration.artifact`) is
always rebuilt from the reference CSV, so CI needs neither data/raw nor data/processed.

Calibration (decisions D3-D8): per arm, the typical days are those of
`data.calendar.typical_day_values` over the window (the same rule validation uses, on
its own slots); each 900 s slot gets the mean over those days of its sensor hour (flat
within the hour). Only days inside `calendar.typical_period` are used: typical days
outside it (e.g. the spring term after a 12-month re-download) are hold-out days, listed
in `day_filter.hold_out_days` and never mixed into the demand. Turn shares come from the
prior, or from the field counts when there are some.
Outputs are deterministic: no wall-clock time, sorted rows, LF line ends.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from traffic_twin.config import ConfigError, load_config
from traffic_twin.data.calendar import (
    TIMEZONE,
    classify_days,
    period_start,
    slot_weights,
    to_date,
    typical_day_values,
    typical_period,
)
from traffic_twin.data.clean import _split_geo, valid_flow, valid_occupancy
from traffic_twin.paths import DATA_CONFIG

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
REFERENCE_COLUMNS = [
    "iu_ac", "libelle", "date", "hour_start", "weekday", "day_type", "flow_veh_h",
    "occupancy_pct", "flow_valid", "occ_valid", "etat_barre", "ts_end",
]  # fmt: skip
ARC_COLUMNS = [
    "iu_ac", "libelle", "iu_nd_amont", "libelle_nd_amont", "iu_nd_aval", "libelle_nd_aval",
    "lat", "lon", "coverage", "station_group", "own_flow_station",
]  # fmt: skip
TOPOLOGY_COLUMNS = ARC_COLUMNS[1:6]
CALIBRATION_KEYS = (
    "calendar.weekdays", "cleaning.max_flow_veh_h", "cleaning.max_occupancy_pct", "paths.raw",
    "calibration.site", "calibration.raw_zone", "calibration.reference_csv",
    "calibration.artifact", "calibration.reference_arcs", "calibration.hours",
    "calibration.window", "calibration.arms", "calibration.turn_prior",
)  # fmt: skip
SHARE_TOL = 1e-3  # turn shares of an arm must sum to 1 within this
DEFAULT_MIN_COMMON_HOURS = 200
HEADER = """\
# Demand calibration of the Pitié-Salpêtrière junction (node 302). GENERATED, do not edit:
#   python -m traffic_twin.data.calibrate --config configs/data.yaml [--from-raw]
# Source: {csv} (source_sha256: its text with LF line ends).
# Slot flows in veh/h over typical days (school-term Tue-Thu, configs/data.yaml calendar);
# a sensor value covers the hour before its published stamp (decision D3).
"""


# -- helpers ---------------------------------------------------------------------------
def _arc_key(arc: str) -> tuple[int, int, str]:
    """Numeric order for arc ids ("505" < "5065"), other ids last."""
    return (0, int(arc), "") if str(arc).isdigit() else (1, 0, str(arc))


def _sorted_arcs(arcs) -> list[str]:
    return sorted({str(a) for a in arcs}, key=_arc_key)


def _hhmm(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}"


def _seconds(hhmm: str) -> int:
    h, m = (int(x) for x in str(hhmm).split(":"))
    return h * 3600 + m * 60


def text_sha256(path: str | Path) -> str | None:
    """sha256 of a text file with LF line ends (stable across git autocrlf checkouts),
    or None when the file is absent."""
    path = Path(path)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _calib(cfg: dict) -> dict:
    missing = []
    for key in CALIBRATION_KEYS:
        node = cfg
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            missing.append(key)
    if missing:
        raise ConfigError(f"data config: missing keys {', '.join(missing)}")
    return cfg["calibration"]


# -- raw export -> hourly reference rows -----------------------------------------------
def _raw_rows(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per raw record (duplicates dropped): iu_ac, libelle, date, hour_start,
    flow_veh_h / occupancy_pct (NaN when invalid), validity flags, etat_barre, ts_end."""
    c = cfg["cleaning"]
    df = raw.copy()
    df["iu_ac"] = df["iu_ac"].astype(str)
    ts_end = pd.to_datetime(df["t_1h"], utc=True).dt.tz_convert(TIMEZONE)
    start = period_start(ts_end)
    bar = df["etat_barre"] if "etat_barre" in df else None
    flow_ok = valid_flow(df["q"], bar, c["max_flow_veh_h"])
    occ_ok = valid_occupancy(df["k"], bar, c["max_occupancy_pct"])
    out = pd.DataFrame(
        {
            "iu_ac": df["iu_ac"].to_numpy(),
            "libelle": df["libelle"].astype(object).to_numpy() if "libelle" in df else None,
            "start": start.to_numpy(),
            "date": start.dt.date.to_numpy(),
            "hour_start": start.dt.hour.to_numpy(),
            "flow_veh_h": pd.to_numeric(df["q"], errors="coerce").where(flow_ok).to_numpy(),
            "occupancy_pct": pd.to_numeric(df["k"], errors="coerce").where(occ_ok).to_numpy(),
            "flow_valid": flow_ok.to_numpy(dtype=bool),
            "occ_valid": occ_ok.to_numpy(dtype=bool),
            "etat_barre": bar.astype(object).to_numpy() if bar is not None else None,
            "ts_end": [t.isoformat() for t in ts_end],
        }
    )
    out = out.sort_values(["iu_ac", "ts_end"], kind="stable")
    return out.drop_duplicates(["iu_ac", "ts_end"]).reset_index(drop=True)


def extract_hours(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Reference rows (section 4.3 of the Part 3 plan): every date of the raw period x
    every arc of `calibration.reference_arcs` x every hour_start in `calibration.hours`,
    sorted by arc (numeric), date and hour_start.

    `date` is the local date of the period start. An invalid flow gives NaN and
    `flow_valid` False; occupancy is checked on its own (`occ_valid`). Rows absent from
    the export have empty `etat_barre` and `ts_end` (the published stamp).
    """
    c = _calib(cfg)
    arcs = _sorted_arcs(c["reference_arcs"])
    h0, h1 = (int(h) for h in c["hours"])
    rows = _raw_rows(raw, cfg)
    if rows.empty:
        raise ValueError("raw export is empty")
    dates = pd.date_range(min(rows["date"]), max(rows["date"]), freq="D").date
    libelle = rows.dropna(subset=["libelle"]).groupby("iu_ac")["libelle"].first()
    missing = [a for a in arcs if a not in set(rows["iu_ac"])]
    if missing:
        log.warning("reference arcs absent from the raw export: %s", ", ".join(missing))

    grid = pd.DataFrame(
        list(itertools.product(arcs, dates, range(h0, h1 + 1))),
        columns=["iu_ac", "date", "hour_start"],
    )
    keep = rows[rows["iu_ac"].isin(arcs) & rows["hour_start"].between(h0, h1)]
    keep = keep.drop_duplicates(["iu_ac", "date", "hour_start"])  # DST doubled hour only
    out = grid.merge(keep.drop(columns=["libelle", "start"]), how="left")
    days = classify_days(dates, cfg["calendar"]).set_index("date")
    out["libelle"] = out["iu_ac"].map(libelle)
    out["weekday"] = out["date"].map(days["weekday"]).astype("int64")
    out["day_type"] = out["date"].map(days["day_type"])
    for flag in ("flow_valid", "occ_valid"):
        out[flag] = out[flag].astype("boolean").fillna(False).astype(bool)
    unknown = sorted(days.index[days["day_type"] == "unknown"])
    if unknown:
        log.warning(
            "%d days outside calendar.covers (%s .. %s) are never typical nor hold-out "
            "(holiday tables not checked for them)",
            len(unknown),
            unknown[0],
            unknown[-1],
        )
    held = sorted(days.index[days["hold_out"]])
    if held:
        log.warning(
            "%d typical days outside calendar.typical_period (%s .. %s) are hold-out days: "
            "listed in the calibration file, never used for the demand",
            len(held),
            held[0],
            held[-1],
        )
    return out[REFERENCE_COLUMNS].reset_index(drop=True)


def write_hours(hours: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hours[REFERENCE_COLUMNS].to_csv(path, index=False, lineterminator="\n")
    return path


def read_hours(path: str | Path) -> pd.DataFrame:
    """The reference CSV as written by `write_hours` (dates as datetime.date)."""
    text = dict.fromkeys(("iu_ac", "libelle", "day_type", "etat_barre", "ts_end"), str)
    df = pd.read_csv(path, dtype=text)
    df["date"] = [dt.date.fromisoformat(d) for d in df["date"]]
    for flag in ("flow_valid", "occ_valid"):
        df[flag] = df[flag].astype(bool)
    return df[REFERENCE_COLUMNS]


# -- shared flow stations --------------------------------------------------------------
def station_groups(hours: pd.DataFrame, min_common: int = DEFAULT_MIN_COMMON_HOURS) -> dict:
    """Arc -> flow-station group. Arcs whose valid flows are identical over at least
    `min_common` common hours share a group (the Paris dataset copies one station's
    flow onto several arcs). The group is named by its arcs, e.g. "509=511=5071"; an
    arc with its own series is its own group."""
    arcs = _sorted_arcs(hours["iu_ac"])
    ok = hours["flow_veh_h"].notna()
    if "flow_valid" in hours:
        ok &= hours["flow_valid"].astype(bool)
    v = hours[ok].assign(iu_ac=lambda d: d["iu_ac"].astype(str))
    q = v.pivot_table(
        index=["date", "hour_start"], columns="iu_ac", values="flow_veh_h", aggfunc="first"
    )
    parent = {a: a for a in arcs}

    def root(a: str) -> str:
        while parent[a] != a:
            a = parent[a]
        return a

    for a, b in itertools.combinations([a for a in arcs if a in q], 2):
        both = q[a].notna() & q[b].notna()
        if both.sum() >= min_common and np.array_equal(q.loc[both, a], q.loc[both, b]):
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[max(ra, rb, key=_arc_key)] = min(ra, rb, key=_arc_key)
    members: dict[str, list[str]] = {}
    for a in arcs:
        members.setdefault(root(a), []).append(a)
    return {a: "=".join(members[root(a)]) for a in arcs}


def arc_table(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Every arc of the raw export: topology, position, flow coverage over the raw
    period (share of hourly periods with a valid flow), `station_group` (whole series,
    see `station_groups`) and `own_flow_station` (its flow series is not shared)."""
    c = cfg["calibration"]
    rows = _raw_rows(raw, cfg)
    first = raw.assign(iu_ac=raw["iu_ac"].astype(str)).sort_values(["iu_ac", "t_1h"])
    topo = first.groupby("iu_ac")[TOPOLOGY_COLUMNS].first()
    geo = first.dropna(subset=["geo_point_2d"]).groupby("iu_ac")["geo_point_2d"].first()
    lat, lon = _split_geo(geo)
    starts = pd.to_datetime(rows["start"], utc=True)
    n_periods = int((starts.max() - starts.min()) / pd.Timedelta(hours=1)) + 1
    groups = station_groups(rows, int(c.get("min_common_hours", DEFAULT_MIN_COMMON_HOURS)))
    arcs = _sorted_arcs(rows["iu_ac"])
    out = pd.DataFrame({"iu_ac": arcs}).join(topo, on="iu_ac")
    out["lat"] = out["iu_ac"].map(pd.to_numeric(lat, errors="coerce").round(7))
    out["lon"] = out["iu_ac"].map(pd.to_numeric(lon, errors="coerce").round(7))
    valid = rows[rows["flow_valid"]].groupby("iu_ac").size()
    out["coverage"] = (out["iu_ac"].map(valid).fillna(0) / n_periods).clip(upper=1).round(4)
    out["station_group"] = out["iu_ac"].map(groups)
    out["own_flow_station"] = ~out["station_group"].str.contains("=", regex=False)
    return out[ARC_COLUMNS]


# -- turn shares -----------------------------------------------------------------------
def turns_from_counts(field_counts) -> dict[str, dict[str, float]]:
    """Turn shares per approach from counted movements.

    `field_counts` is the `counts` list of the field file (bins with
    `movements: {N_S: n, N_W: n, ...}`), or the whole field mapping. In each bin an
    approach is used only when all its movements are counted (no null); an approach
    without any count is left out. Shares are rounded to 4 decimals.
    """
    bins = field_counts.get("counts") if isinstance(field_counts, dict) else field_counts
    totals: dict[str, dict[str, float]] = {}
    for b in bins or []:
        by_src: dict[str, dict[str, float | None]] = {}
        for key, n in ((b or {}).get("movements") or {}).items():
            src, sep, dst = str(key).partition("_")
            if not (sep and src and dst) or src == dst:
                raise ConfigError(f"field counts: movement '{key}' is not <from>_<to>")
            if n is not None and float(n) < 0:
                raise ConfigError(f"field counts: movement '{key}' has a negative count")
            by_src.setdefault(src, {})[dst] = None if n is None else float(n)
        for src, moves in by_src.items():
            if any(n is None for n in moves.values()):
                continue
            acc = totals.setdefault(src, {})
            for dst, n in moves.items():
                acc[dst] = acc.get(dst, 0.0) + (n or 0.0)
    shares = {}
    for src, moves in totals.items():
        total = sum(moves.values())
        if total > 0:
            shares[src] = {dst: round(n / total, 4) for dst, n in moves.items()}
    return shares


def _check_shares(arm: str, shares: dict, arms) -> dict[str, float]:
    if not isinstance(shares, dict) or not shares:
        raise ConfigError(f"turn shares of {arm}: expected a mapping destination -> share")
    unknown = sorted(set(shares) - set(arms) | ({arm} & set(shares)))
    if unknown:
        raise ConfigError(f"turn shares of {arm}: unknown destination {', '.join(unknown)}")
    out = {str(d): float(s) for d, s in shares.items()}
    if any(s < 0 for s in out.values()) or abs(sum(out.values()) - 1) > SHARE_TOL:
        raise ConfigError(f"turn shares of {arm} must be >= 0 and sum to 1, got {out}")
    return out


def _turns(c: dict, field: dict | None) -> dict:
    arms = list(c["arms"])
    prior = c["turn_prior"]
    counted = turns_from_counts(field) if field else {}
    extra = sorted(set(counted) - set(arms))
    if extra:
        raise ConfigError(f"field counts: unknown approach {', '.join(extra)}")
    used = [a for a in arms if a in counted]
    source = "field" if len(used) == len(arms) else "mixed" if used else "assumption"
    out: dict = {"source": source}
    if source == "mixed":
        out["field_arms"] = used
    for arm in arms:
        if arm not in counted and arm not in prior:
            raise ConfigError(f"calibration.turn_prior has no shares for {arm}")
        out[arm] = _check_shares(arm, counted.get(arm) or prior[arm], arms)
    return out


# -- calibration file ------------------------------------------------------------------
def _arc_profile(hours, arc, window, cal, k) -> dict:
    """Typical days, hold-out days, window mean and slot flows of one arc."""
    begin, end = window["begin"], window["end"]
    slot_s = int(window.get("slot_s", 900))
    begin_s, end_s = _seconds(begin), _seconds(end)
    if slot_s <= 0 or (end_s - begin_s) % slot_s:
        raise ConfigError(f"calibration.window {begin}-{end} is not a whole number of slots")
    kept = typical_day_values(hours, arc, begin, end, cal, k)
    complete = typical_day_values(hours, arc, begin, end, cal, math.inf)  # no MAD filter
    held = typical_day_values(hours, arc, begin, end, cal, k, hold_out=True)
    days = list(kept.index)
    used = set(days)
    prof: dict = {"n_days": len(days), "days": days, "hold_out": list(held.index)}
    prof["dropped"] = [d for d in complete.index if d not in used]
    if not days:
        prof.update(mean_veh_h=None, slots=None)
        return prof
    ok = (hours["iu_ac"].astype(str) == arc) & hours["flow_veh_h"].notna()
    if "flow_valid" in hours:
        ok &= hours["flow_valid"].astype(bool)
    a = hours[ok]
    a = a[[to_date(d) in used for d in a["date"]]]
    hourly = a.groupby(a["hour_start"].astype(int))["flow_veh_h"].mean()
    slots = []
    for s in range(begin_s, end_s, slot_s):
        w = slot_weights(_hhmm(s), _hhmm(s + slot_s))
        q = sum(float(hourly[h]) * x for h, x in w.items())
        slots.append({"begin_s": s, "flow_veh_h": round(q, 3)})
    prof.update(mean_veh_h=round(float(kept.mean()), 3), slots=slots)
    return prof


def build_calibration(hours: pd.DataFrame, cfg: dict, field: dict | None = None) -> dict:
    """The calibration mapping written to `calibration.artifact` (schema_version 1).

    Per arm: arc, n_days, window mean and 900 s slot flows (veh/h) over the typical days
    (see the module docstring); the sensitivity arcs likewise; the flow-station groups
    of the reference arcs; turn shares from the field counts or the prior. `field` is
    the field-measurement mapping (None: prior only). `day_filter` records the
    `calendar.typical_period` and, per arc, the hold-out days outside it (same window,
    completeness and MAD rule), which are left out of the slot flows.
    """
    c = _calib(cfg)
    cal = cfg["calendar"]
    window = c["window"]
    k = float(c.get("mad_k", 3.5))
    min_days = int(c.get("min_days", 1))
    begin_s, end_s = _seconds(window["begin"]), _seconds(window["end"])

    arms, sensitivity, days_used, n_days, dropped, held = {}, {}, {}, {}, {}, {}

    def profile(arc: str) -> dict:
        p = _arc_profile(hours, arc, window, cal, k)
        days_used[arc] = [d.isoformat() for d in p["days"]]
        n_days[arc] = p["n_days"]
        if p["dropped"]:
            dropped[arc] = [d.isoformat() for d in p["dropped"]]
        if p["hold_out"]:
            held[arc] = [d.isoformat() for d in p["hold_out"]]
        return p

    for arm, a in c["arms"].items():
        arc = str(a["arc"])
        p = profile(arc)
        if p["n_days"] < min_days:
            raise ValueError(
                f"arm {arm} (arc {arc}): {p['n_days']} typical days, fewer than "
                f"calibration.min_days = {min_days}"
            )
        arms[arm] = {"arc": arc, **({"status": a["status"]} if a.get("status") else {})}
        arms[arm].update(n_days=p["n_days"], mean_veh_h=p["mean_veh_h"], slots=p["slots"])
        extra = [str(x) for x in a.get("sensitivity") or []]
        if len(extra) > 1:
            raise ConfigError(f"calibration.arms.{arm}.sensitivity: one arc at most")
        for s_arc in extra:
            s = profile(s_arc)
            sensitivity[arm] = {
                "arc": s_arc, "n_days": s["n_days"], "mean_veh_h": s["mean_veh_h"],
                "slots": s["slots"],
            }  # fmt: skip

    ref = c["reference_csv"]
    groups = station_groups(hours, int(c.get("min_common_hours", DEFAULT_MIN_COMMON_HOURS)))
    dates = sorted(to_date(d) for d in set(hours["date"]))
    period = typical_period(cal)
    return {
        "schema_version": SCHEMA_VERSION,
        "site": c["site"],
        "source": {
            "reference_csv": str(ref),
            "source_sha256": text_sha256(ref),
            "raw_period": [dates[0].isoformat(), dates[-1].isoformat()],
        },
        "timestamp_convention": "period_end",
        "day_filter": {
            "weekdays": [int(d) for d in cal["weekdays"]],
            "day_type": "term",
            "typical_period": [d.isoformat() for d in period] if period else None,
            "mad_k": k,
            "min_days": min_days,
            "days_used": days_used,
            "n_days": n_days,
            "mad_dropped": dropped,
            "hold_out_days": held,
        },
        "window": {"begin_s": begin_s, "end_s": end_s, "slot_s": int(window.get("slot_s", 900))},
        "arms": arms,
        "sensitivity": sensitivity,
        "station_groups": groups,
        "turns": _turns(c, field),
    }


def write_calibration(calibration: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        calibration, sort_keys=False, allow_unicode=True, default_flow_style=None, width=100
    )
    csv = calibration.get("source", {}).get("reference_csv", "?")
    path.write_text(HEADER.format(csv=csv) + body, encoding="utf-8", newline="\n")
    return path


def load_field(cfg: dict) -> dict | None:
    """The field-measurement file named by `calibration.field_counts`, or None when
    absent (counts not measured yet: the turn prior is used)."""
    path = cfg.get("calibration", {}).get("field_counts")
    if not path or not Path(path).is_file():
        return None
    with open(path, encoding="utf-8") as f:
        field = yaml.safe_load(f)
    return field if isinstance(field, dict) else None


def calibrate(cfg: dict, from_raw: bool = False) -> dict:
    """Rebuild the versioned extracts (with `from_raw`) and the calibration file."""
    c = _calib(cfg)
    if from_raw:
        raw_path = Path(cfg["paths"]["raw"]) / f"{c['raw_zone']}.parquet"
        if not raw_path.exists():
            raise SystemExit(f"missing {raw_path}: run `make data` first")
        raw = pd.read_parquet(raw_path)
        log.info("wrote %s", write_hours(extract_hours(raw, cfg), c["reference_csv"]))
        if c.get("arcs_csv"):
            arcs = arc_table(raw, cfg)
            Path(c["arcs_csv"]).parent.mkdir(parents=True, exist_ok=True)
            arcs.to_csv(c["arcs_csv"], index=False, lineterminator="\n")
            log.info("wrote %s", c["arcs_csv"])
    calibration = build_calibration(read_hours(c["reference_csv"]), cfg, load_field(cfg))
    log.info("wrote %s", write_calibration(calibration, c["artifact"]))
    return calibration


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DATA_CONFIG)
    parser.add_argument(
        "--from-raw", action="store_true", help="rewrite the reference CSVs from data/raw first"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cal = calibrate(load_config(args.config), from_raw=args.from_raw)
    for kind, label in (("arms", "arm"), ("sensitivity", "sensitivity")):
        for arm, a in cal[kind].items():
            mean = "n/a" if a["mean_veh_h"] is None else f"{a['mean_veh_h']:.1f} veh/h"
            print(f"{label} {arm}: arc {a['arc']}, {a['n_days']} typical days, mean {mean}")
    f = cal["day_filter"]
    held = ", ".join(f"{arc}: {len(d)}" for arc, d in f["hold_out_days"].items()) or "none"
    print(f"typical period {f['typical_period'] or 'not restricted'}; hold-out days {held}")
    print(f"turn shares: {cal['turns']['source']}")


if __name__ == "__main__":
    main()

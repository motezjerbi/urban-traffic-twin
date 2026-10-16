"""Day-type demand calibrations (Part 5, D44; ASSUMED ratio method).

The calibration `configs/calibration/pitie_302.yaml` describes a typical school-term
Tuesday-Thursday (`tue_wed_thu_term`). The ETA worlds and the RL episodes also draw a
Monday/Friday term day (`mon_fri_term`) and a school-holiday weekday (`school_holiday`,
summer only in the extract). Their calibration files are generated from it:

    slot flow(type) = slot flow(base) x sum_h w_h * ratio(type, arm, h)

- `w_h`: the base file's own slot-to-hour convention (`data.calendar.slot_weights` of
  the slot, a sensor value covering the hour that starts at `hour_start`);
- `ratio(type, arm, h)` = mean valid flow of the arm's arc at sensor hour h over the
  dates of the day type / mean valid flow over the base file's `days_used` of that arc,
  both from `data/reference/pitie_302_hours.csv` (versioned);
- `mon_fri_term`: `day_type` term and weekday 0 or 4; `school_holiday`: `day_type`
  school_holiday and weekday 0-4. Saturday is not a v1 day type. The base type has ratio
  1 by definition.

Ratios are rounded to 4 decimals and slot flows to 3 (as `data.calibrate`). The files keep
the calibration schema (`window`, `arms`, `turns`, ...) plus a `day_type` block with the
rule, the dates, the ratios and the sha256 of both sources; the `sensitivity` arcs are
left out (validation reads only the base file). The base file is never modified, and the
paths are stable, so `config_hash` and `demand_hash` of a world are reproducible.

    python -m traffic_twin.ml.daytypes            # (re)write the generated files
    python -m traffic_twin.ml.daytypes --check    # exit 1 unless they are up to date
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import yaml

from traffic_twin.config import ConfigError, load_config, text_sha256
from traffic_twin.data.calendar import slot_weights
from traffic_twin.paths import ETA_CONFIG, REFERENCE_DIR

DAY_TYPES = ("tue_wed_thu_term", "mon_fri_term", "school_holiday")
BASE_DAY_TYPE = DAY_TYPES[0]
HOURS_CSV = REFERENCE_DIR / "pitie_302_hours.csv"
# Dates of each generated day type: (day_type column of the reference CSV, weekdays)
DAY_RULES = {
    "mon_fri_term": ("term", (0, 4)),
    "school_holiday": ("school_holiday", (0, 1, 2, 3, 4)),
}
RATIO_DECIMALS = 4
FLOW_DECIMALS = 3

HEADER = """\
# Demand calibration of the Pitie-Salpetriere junction (node 302), day type {name}.
# GENERATED, do not edit:  python -m traffic_twin.ml.daytypes   (check: --check)
# ASSUMED ratio method (D44): base slot flow x mean flow of the day type / mean flow over
# the base days_used, per arm (its arc) and sensor hour.
# Base: {base} (sha256 {base_sha}, LF line ends).
# Ratios: {csv} (sha256 {csv_sha}, LF line ends).
"""


def _hhmm(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}"


def _slot_hours(calibration: dict, begin_s: int) -> dict[int, float]:
    """Sensor hour -> weight of one slot of the calibration window."""
    slot_s = int(calibration["window"]["slot_s"])
    return slot_weights(_hhmm(begin_s), _hhmm(begin_s + slot_s))


def window_hours(calibration: dict) -> list[int]:
    """Sensor hours the calibration window's slots draw on (e.g. [7, 8, 9])."""
    w = calibration["window"]
    hours = set()
    for s in range(int(w["begin_s"]), int(w["end_s"]), int(w["slot_s"])):
        hours |= set(_slot_hours(calibration, s))
    return sorted(hours)


def _valid(hours: pd.DataFrame) -> pd.DataFrame:
    ok = hours["flow_veh_h"].notna()
    if "flow_valid" in hours:
        ok &= hours["flow_valid"].astype(str).str.lower().isin(("true", "1"))
    return hours[ok]


def day_type_dates(hours: pd.DataFrame, day_type: str) -> list[str]:
    """ISO dates of a generated day type in the reference CSV (D44 rule)."""
    if day_type not in DAY_RULES:
        raise ValueError(f"no date rule for day type {day_type!r} (rules: {list(DAY_RULES)})")
    kind, weekdays = DAY_RULES[day_type]
    sel = (hours["day_type"].astype(str) == kind) & hours["weekday"].astype(int).isin(weekdays)
    return sorted({str(d) for d in hours.loc[sel, "date"]})


def _read_hours(hours_csv: Path) -> pd.DataFrame:
    return pd.read_csv(hours_csv, dtype={"iu_ac": str, "date": str, "day_type": str})


def day_type_ratios(hours_csv: Path, calibration: dict) -> dict[str, dict[str, dict[int, float]]]:
    """Day type -> arm -> sensor hour -> ratio (rounded to 4 decimals), for every type of
    DAY_TYPES; the base type is 1.0 everywhere. Raises ValueError when a mean is missing
    (no valid flow for an arc and hour)."""
    hours = _valid(_read_hours(Path(hours_csv)))
    used = calibration["day_filter"]["days_used"]
    needed = window_hours(calibration)
    out: dict[str, dict[str, dict[int, float]]] = {
        BASE_DAY_TYPE: {arm: dict.fromkeys(needed, 1.0) for arm in calibration["arms"]}
    }
    for day_type in DAY_TYPES[1:]:
        dates = set(day_type_dates(hours, day_type))
        out[day_type] = {}
        for arm, a in calibration["arms"].items():
            arc = str(a["arc"])
            rows = hours[hours["iu_ac"] == arc]
            base_days = {str(d) for d in used[arc]}
            out[day_type][arm] = {}
            for h in needed:
                at = rows[rows["hour_start"].astype(int) == h]
                num = at.loc[at["date"].isin(dates), "flow_veh_h"]
                den = at.loc[at["date"].isin(base_days), "flow_veh_h"]
                if num.empty or den.empty:
                    raise ValueError(f"{day_type}: no valid flow for arc {arc} at hour {h}")
                ratio = float(num.mean()) / float(den.mean())
                out[day_type][arm][h] = round(ratio, RATIO_DECIMALS)
    return out


def day_type_calibration(calibration: dict, day_type: str, ratios: dict) -> dict:
    """The calibration of `day_type`: `calibration` (the base, not modified) with every
    arm's slot flows scaled by its ratios and its window mean scaled alike, the
    `sensitivity` arcs left out and a `day_type` block (the ratios used; the GENERATED
    provenance fields are added by `generate`)."""
    out = copy.deepcopy(calibration)
    by_arm = ratios[day_type]
    for arm, a in out["arms"].items():
        before = sum(float(s["flow_veh_h"]) for s in a["slots"])
        for slot in a["slots"]:
            w = _slot_hours(calibration, int(slot["begin_s"]))
            factor = sum(x * by_arm[arm][h] for h, x in w.items())
            slot["flow_veh_h"] = round(float(slot["flow_veh_h"]) * factor, FLOW_DECIMALS)
        after = sum(float(s["flow_veh_h"]) for s in a["slots"])
        if a.get("mean_veh_h") is not None and before > 0:  # window mean, scaled alike
            a["mean_veh_h"] = round(float(a["mean_veh_h"]) * after / before, FLOW_DECIMALS)
    out["sensitivity"] = {}
    block = {"name": day_type, "method": "ratio (ASSUMED, D44)"}
    block["ratios"] = {arm: {int(h): float(r) for h, r in hs.items()} for arm, hs in by_arm.items()}
    # the day_type block goes right after `site`
    ordered = {}
    for k, v in out.items():
        ordered[k] = v
        if k == "site":
            ordered["day_type"] = block
    if "day_type" not in ordered:
        ordered["day_type"] = block
    return ordered


def generate(
    base_path: Path, day_type: str, hours_csv: Path = HOURS_CSV, ratios: dict | None = None
) -> str:
    """Text of the generated calibration file of `day_type` (LF line ends)."""
    base = load_config(base_path)
    ratios = ratios or day_type_ratios(hours_csv, base)
    cal = day_type_calibration(base, day_type, ratios)
    hours = _valid(_read_hours(Path(hours_csv)))
    used = {str(a["arc"]) for a in base["arms"].values()}
    dates = day_type_dates(hours, day_type)
    cal["day_type"] = {
        **cal["day_type"],
        "rule": {"day_type": DAY_RULES[day_type][0], "weekdays": list(DAY_RULES[day_type][1])},
        "base_calibration": Path(base_path).as_posix(),
        "base_sha256": text_sha256(base_path),
        "hours_csv": Path(hours_csv).as_posix(),
        "hours_sha256": text_sha256(hours_csv),
        "dates": {
            arc: sorted(set(dates) & {str(d) for d in hours.loc[hours["iu_ac"] == arc, "date"]})
            for arc in sorted(used, key=int)
        },
    }
    body = yaml.safe_dump(
        cal, sort_keys=False, allow_unicode=True, default_flow_style=None, width=100
    )
    header = HEADER.format(
        name=day_type,
        base=Path(base_path).as_posix(),
        base_sha=text_sha256(base_path),
        csv=Path(hours_csv).as_posix(),
        csv_sha=text_sha256(hours_csv),
    )
    return header + body


def calibration_files(eta_cfg: dict) -> dict[str, Path]:
    """Day type -> calibration file, from `eta.yaml: calibration_files` (checked)."""
    files = {k: Path(v) for k, v in (eta_cfg.get("calibration_files") or {}).items()}
    if set(files) != set(DAY_TYPES):
        raise ConfigError(f"calibration_files must list {DAY_TYPES}, got {sorted(files)}")
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ETA_CONFIG)
    parser.add_argument("--hours", type=Path, default=HOURS_CSV)
    parser.add_argument("--check", action="store_true", help="verify, write nothing")
    args = parser.parse_args(argv)
    files = calibration_files(load_config(args.config))
    base_path = files[BASE_DAY_TYPE]
    ratios = day_type_ratios(args.hours, load_config(base_path))
    stale = []
    for day_type in DAY_TYPES[1:]:
        path = files[day_type]
        text = generate(base_path, day_type, args.hours, ratios)
        current = path.read_bytes().replace(b"\r\n", b"\n") if path.is_file() else None
        if args.check:
            ok = current == text.encode("utf-8")
            print(f"{path.as_posix()}: {'up to date' if ok else 'STALE or missing'}")
            if not ok:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {path.as_posix()}")
    if stale:
        print("run: python -m traffic_twin.ml.daytypes", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

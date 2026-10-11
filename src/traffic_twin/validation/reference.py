"""Real side of the flow validation: typical-day sensor flows per station and clock slot.

Built on `data.calendar.typical_day_values`, the same day filter the demand calibration
uses (Tue-Thu school term, no public holiday, robust MAD filter; stamps are the END of
the hour). Input is the versioned extract `data/reference/pitie_302_hours.csv`, so this
runs without `data/processed`. A station's `alt_iu_ac` (510 for N_in) gives an extra
`sensitivity` row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from traffic_twin.config import ConfigError
from traffic_twin.data import calendar

REAL_COLUMNS = [
    "station", "iu_ac", "sensitivity", "slot", "real_mean", "real_std", "real_p10",
    "real_p90", "n_days", "first_day", "last_day", "status",
]  # fmt: skip


def read_hours(path: str | Path) -> pd.DataFrame:
    """The reference CSV (one row per arc, date and starting hour), arc ids as strings."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"reference file {path} not found (build it with `tasks.py calib-data`)"
        )
    return pd.read_csv(path, dtype={"iu_ac": str, "date": str})


def _arcs(stations: list[dict]) -> list[tuple[str, str | None, bool]]:
    """(station, arc, sensitivity) for each station arc, then its sensitivity arc."""
    out = []
    for st in stations:
        iu = st.get("iu_ac")
        out.append((st["station"], str(iu) if iu not in (None, "") else None, False))
        if st.get("alt_iu_ac") not in (None, ""):
            out.append((st["station"], str(st["alt_iu_ac"]), True))
    return out


def real_flows(
    hours: pd.DataFrame, stations: list[dict], slots: list[dict], cal: dict, cfg: dict
) -> pd.DataFrame:
    """Typical-day flow (veh/h) of every station arc in every slot: mean, sample std,
    p10, p90 and the number of days kept. `status` is "ok", NO_DATA (fewer than
    `reference.min_days` days) or NO_SENSOR (station without an arc). `cfg` is the
    validation config (`reference.min_days`, `reference.mad_k`)."""
    ref = cfg.get("reference", {})
    if ref.get("day_set", "term") != "term":
        raise ConfigError("reference.day_set: only 'term' (typical school-term days) exists")
    min_days, k = int(ref.get("min_days", 8)), float(ref.get("mad_k", 3.5))
    rows = []
    for station, iu, sensitivity in _arcs(stations):
        for slot in slots:
            row = {"station": station, "iu_ac": iu, "sensitivity": sensitivity}
            row["slot"] = slot["name"]
            if iu is None:
                rows.append({**row, "n_days": 0, "status": "NO_SENSOR"})
                continue
            v = calendar.typical_day_values(hours, iu, slot["begin"], slot["end"], cal, k=k)
            v = pd.Series(v, dtype=float).dropna()
            n = len(v)
            days = sorted(str(d)[:10] for d in v.index)
            rows.append(
                {
                    **row,
                    "real_mean": v.mean() if n else np.nan,
                    "real_std": v.std(ddof=1) if n > 1 else np.nan,
                    "real_p10": v.quantile(0.10) if n else np.nan,
                    "real_p90": v.quantile(0.90) if n else np.nan,
                    "n_days": n,
                    "first_day": days[0] if days else None,
                    "last_day": days[-1] if days else None,
                    "status": "ok" if n >= min_days else "NO_DATA",
                }
            )
    return pd.DataFrame(rows).reindex(columns=REAL_COLUMNS)

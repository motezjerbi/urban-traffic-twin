"""KPIs per user type from SUMO outputs, in the common long format of spec 9.4:
scenario, seed, user_type, kpi, value."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

KPI_COLUMNS = ["scenario", "seed", "user_type", "kpi", "value"]


def user_type_of(vtype: str) -> str:
    return {"bus": "bus", "ambulance": "ambulance"}.get(vtype, "car")


def read_tripinfo(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vehicle trips and pedestrian walks from a SUMO tripinfo file."""
    trips, walks = [], []
    for el in ET.parse(path).getroot():
        if el.tag == "tripinfo":
            trips.append(
                {
                    "id": el.get("id"),
                    "type": user_type_of(el.get("vType", "")),
                    "depart_s": float(el.get("depart")),
                    "arrival_s": float(el.get("arrival")),
                    "duration_s": float(el.get("duration")),
                    "time_loss_s": float(el.get("timeLoss")),
                    "waiting_s": float(el.get("waitingTime")),
                    "stops": int(el.get("waitingCount")),
                }
            )
        elif el.tag == "personinfo":
            for w in el.findall("walk"):
                walks.append(
                    {
                        "id": el.get("id"),
                        "type": "pedestrian",
                        "depart_s": float(w.get("depart")),
                        "arrival_s": float(w.get("arrival")),
                        "duration_s": float(w.get("duration")),
                        "time_loss_s": float(w.get("timeLoss")),
                    }
                )
    return pd.DataFrame(trips), pd.DataFrame(walks)


def read_bus_stop_arrivals(path: Path) -> pd.Series:
    """Arrival times of buses at their stop, from a SUMO stop-output file."""
    if not path.exists():
        return pd.Series(dtype=float)
    times = [float(el.get("started")) for el in ET.parse(path).getroot() if el.tag == "stopinfo"]
    return pd.Series(sorted(times), dtype=float)


def compute_kpis(
    trips: pd.DataFrame,
    walks: pd.DataFrame,
    bus_arrivals: pd.Series,
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    rows: list[tuple[str, str, float]] = []

    def add(user: str, kpi: str, value: float) -> None:
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            rows.append((user, kpi, float(value)))

    if not trips.empty:
        for user, g in trips.groupby("type"):
            add(user, "count", len(g))
            add(user, "travel_time", g["duration_s"].mean())
            add(user, "time_loss_mean", g["time_loss_s"].mean())
            add(user, "wait_mean", g["waiting_s"].mean())
            add(user, "wait_p95", g["waiting_s"].quantile(0.95))
            add(user, "stops", g["stops"].mean())
        bus = trips[trips["type"] == "bus"]
        if not bus.empty:
            add("bus", "delay_mean", bus["time_loss_s"].mean())
    if len(bus_arrivals) >= 3:
        add("bus", "headway_std", bus_arrivals.diff().dropna().std())
    if not walks.empty:
        # Time lost by pedestrians, mostly waiting at crossings
        add("pedestrian", "count", walks["id"].nunique())
        add("pedestrian", "ped_wait_mean", walks["time_loss_s"].mean())
        add("pedestrian", "ped_wait_max", walks["time_loss_s"].max())
        add("pedestrian", "travel_time", walks["duration_s"].mean())

    df = pd.DataFrame(rows, columns=["user_type", "kpi", "value"])
    df.insert(0, "seed", seed)
    df.insert(0, "scenario", scenario)
    return df[KPI_COLUMNS]

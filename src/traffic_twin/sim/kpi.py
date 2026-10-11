"""KPIs per user type from SUMO outputs, in the common long format of spec 9.4:
scenario, seed, user_type, kpi, value."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

KPI_COLUMNS = ["scenario", "seed", "user_type", "kpi", "value"]


def user_type_of(vtype: str) -> str:
    return {"bus": "bus", "ambulance": "ambulance"}.get(vtype, "car")


TRIP_COLUMNS = [
    "id", "type", "depart_s", "arrival_s", "duration_s", "time_loss_s", "waiting_s", "stops",
    "depart_delay_s",
]  # fmt: skip
WALK_COLUMNS = ["id", "type", "depart_s", "arrival_s", "duration_s", "time_loss_s"]
BUS_STOP_COLUMNS = ["busStop", "vehicle", "line", "started", "ended"]


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
                    "depart_delay_s": float(el.get("departDelay", "nan")),
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
    # Fixed columns so a run where nothing finished still has a usable schema
    return pd.DataFrame(trips, columns=TRIP_COLUMNS), pd.DataFrame(walks, columns=WALK_COLUMNS)


def read_bus_stop_arrivals(path: Path) -> pd.Series:
    """Arrival times of buses at their stop, from a SUMO stop-output file."""
    if not path.exists():
        return pd.Series(dtype=float)
    times = [float(el.get("started")) for el in ET.parse(path).getroot() if el.tag == "stopinfo"]
    return pd.Series(sorted(times), dtype=float)


def flow_of(vehicle: str) -> str:
    """The flow a vehicle comes from (SUMO names flow vehicles `<flow id>.<n>`)."""
    return vehicle.rsplit(".", 1)[0]


def bus_lines(routes: Path | None) -> dict[str, str]:
    """Flow (or vehicle) id -> `line` attribute, from a route file."""
    if routes is None or not Path(routes).is_file():
        return {}
    root = ET.parse(routes).getroot()
    return {
        el.get("id"): el.get("line")
        for el in root
        if el.tag in ("flow", "vehicle", "trip") and el.get("line") is not None
    }


def read_bus_stops(path: Path, lines: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Bus stop events from a SUMO stop-output file. SUMO does not write the line, so it
    comes from `lines` (`bus_lines(routes)`), looked up by vehicle then by flow id; the
    flow id stands in for it otherwise."""
    if not Path(path).exists():
        return pd.DataFrame(columns=BUS_STOP_COLUMNS)
    lines = lines or {}
    rows = []
    for el in ET.parse(path).getroot():
        if el.tag != "stopinfo":
            continue
        vid = el.get("id")
        flow = flow_of(vid)
        rows.append(
            {"busStop": el.get("busStop", el.get("lane", "")), "vehicle": vid,
             "line": lines.get(vid, lines.get(flow, flow)),
             "started": float(el.get("started")), "ended": float(el.get("ended", "nan"))}
        )  # fmt: skip
    return pd.DataFrame(rows, columns=BUS_STOP_COLUMNS)


def _headway_std(arrivals: pd.Series) -> float:
    """Standard deviation of the gaps between successive arrivals."""
    return arrivals.sort_values().diff().dropna().std()


def compute_kpis(
    trips: pd.DataFrame,
    walks: pd.DataFrame,
    bus_stops: pd.DataFrame | pd.Series,
    scenario: str,
    seed: int,
    ped_single_waits: pd.Series | None = None,
) -> pd.DataFrame:
    """KPI rows per user type. `bus_stops` is `read_bus_stops` output (headway_std = mean
    over (busStop, line) groups with at least 3 arrivals) or, as before Part 3, a Series
    of arrival times taken as one group. `ped_single_waits` holds, per pedestrian, the
    longest single wait (max over time of TraCI `person.getWaitingTime`)."""
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
            if "depart_delay_s" in g:  # insertion backlog (absent in pre-Part 3 tables)
                add(user, "depart_delay_mean", g["depart_delay_s"].mean())
                add(user, "depart_delay_max", g["depart_delay_s"].max())
        bus = trips[trips["type"] == "bus"]
        if not bus.empty:
            add("bus", "delay_mean", bus["time_loss_s"].mean())

    if isinstance(bus_stops, pd.Series):
        groups = [bus_stops] if len(bus_stops) >= 3 else []
        dwell = pd.Series(dtype=float)
    else:
        groups = [g["started"] for _, g in bus_stops.groupby(["busStop", "line"]) if len(g) >= 3]
        dwell = bus_stops["ended"] - bus_stops["started"]
    if groups:
        add("bus", "headway_std", float(np.mean([_headway_std(g) for g in groups])))
    if not dwell.empty:
        add("bus", "dwell_mean", dwell.mean())

    if not walks.empty:
        # Time lost by pedestrians, mostly waiting at crossings
        add("pedestrian", "count", walks["id"].nunique())
        add("pedestrian", "ped_wait_mean", walks["time_loss_s"].mean())
        add("pedestrian", "ped_wait_max", walks["time_loss_s"].max())
        add("pedestrian", "travel_time", walks["duration_s"].mean())
        add("pedestrian", "ped_wait_p95", walks["time_loss_s"].quantile(0.95))
    if ped_single_waits is not None and not ped_single_waits.empty:
        # Longest single wait of each pedestrian (spec: 90 s target, decision D14)
        add("pedestrian", "ped_wait_single_mean", ped_single_waits.mean())
        add("pedestrian", "ped_wait_single_p95", ped_single_waits.quantile(0.95))
        add("pedestrian", "ped_wait_single_max", ped_single_waits.max())

    df = pd.DataFrame(rows, columns=["user_type", "kpi", "value"])
    df.insert(0, "seed", seed)
    df.insert(0, "scenario", scenario)
    return df[KPI_COLUMNS]


# SUMO statistic-output (passive) -> run health
STATISTICS = {
    "teleports": ("teleports", "total"),
    "collisions": ("safety", "collisions"),
    "emergency_stops": ("safety", "emergencyStops"),
    "emergency_braking": ("safety", "emergencyBraking"),
    "vehicles_waiting_end": ("vehicles", "waiting"),  # still waiting for insertion
    "persons_jammed": ("persons", "jammed"),
    "person_teleports": ("personTeleports", "total"),
}


def read_statistics(path: Path) -> dict[str, int | None]:
    """Counts from a SUMO `--statistic-output` file (None when absent)."""
    found: dict[str, int | None] = dict.fromkeys(STATISTICS)
    if not Path(path).is_file():
        return found
    root = ET.parse(path).getroot()
    for key, (tag, attr) in STATISTICS.items():
        el = root.find(tag)
        if el is not None and el.get(attr) is not None:
            found[key] = int(float(el.get(attr)))
    return found

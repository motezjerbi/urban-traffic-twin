"""KPIs per user type from SUMO outputs, in the common long format of spec 9.4:
scenario, seed, user_type, kpi, value.

`compute_kpis` gives the Part 2-3 rows (unchanged since); `extra_kpis` appends the
Part 4 rows (KPI version 2): per-arm, per-flow, per-line and per-crossing values, the
ambulance travel time from its planned departure and the junction box / exit blocking;
KPI version 3 adds the pedestrian curb waits (`ped_curb_wait_*`, creeps included).
Qualified KPIs are named `<base>__<qualifier>` (`time_loss_mean__arm_N`)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_twin.config import approach_in_edges

KPI_COLUMNS = ["scenario", "seed", "user_type", "kpi", "value"]
KPI_VERSION = 3  # 1: Parts 2-3 (compute_kpis); 2: + extra_kpis rows (Part 4); 3: + curb waits
QUALIFIER_SEP = "__"  # <base kpi>__<qualifier>


def user_type_of(vtype: str) -> str:
    return {"bus": "bus", "ambulance": "ambulance"}.get(vtype, "car")


TRIP_COLUMNS = [
    "id", "type", "depart_s", "arrival_s", "duration_s", "time_loss_s", "waiting_s", "stops",
    "depart_delay_s", "depart_edge", "arrival_edge", "flow",
]  # fmt: skip
WALK_COLUMNS = ["id", "type", "depart_s", "arrival_s", "duration_s", "time_loss_s"]
BUS_STOP_COLUMNS = ["busStop", "vehicle", "line", "started", "ended"]


def _lane_edge(lane: str | None) -> str | None:
    """Edge of a SUMO lane id (`n_in_1` -> `n_in`); None when absent or malformed."""
    if not lane:
        return None
    edge, sep, index = lane.rpartition("_")
    return edge if sep and edge and index.isdigit() else None


def read_tripinfo(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vehicle trips and pedestrian walks from a SUMO tripinfo file. Trips also carry
    `depart_edge` / `arrival_edge` (from departLane / arrivalLane; None when absent)
    and `flow` (`flow_of(id)`)."""
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
                    "depart_edge": _lane_edge(el.get("departLane")),
                    "arrival_edge": _lane_edge(el.get("arrivalLane")),
                    "flow": flow_of(el.get("id")),
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


# JunctionMonitor.summary() key -> user type of its KPI row
JUNCTION_KPIS = {
    "crossing_complete_rate": "pedestrian",
    "crossing_red_entries": "pedestrian",
    "box_halt_red_s": "junction",
    "box_foe_onsets": "junction",
    "exit_blocked_s": "junction",
    "exit_blocked_green_s": "junction",
}


def crossing_of(person: str, crossings: Mapping) -> str | None:
    """Crossing name of a pedestrian of the `ped_{crossing}_{k}` person flows
    (`demand.crossing_walks`; persons are `ped_{crossing}_{k}.{n}`), else None."""
    name = flow_of(person).removeprefix("ped_").rpartition("_")[0]
    return name if person.startswith("ped_") and name in crossings else None


def extra_kpis(
    trips: pd.DataFrame,
    walks: pd.DataFrame,
    bus_stops: pd.DataFrame,
    cfg: dict,
    junction: Mapping | None,
    ped_single_waits: pd.Series | None = None,
    ped_curb_waits: pd.Series | None = None,
) -> pd.DataFrame:
    """Part 4 KPI rows (KPI version 2, plan section 4.6), appended after `compute_kpis`.
    Columns user_type, kpi, value; a value that cannot be computed gives no row.

    - car: `throughput_vph` (arrivals in [begin, begin + duration) per hour),
      `delay_total_mean` (timeLoss + departDelay, i.e. with the insertion delay), and
      `time_loss_mean__arm_X`, `delay_total_mean__arm_X` per arm X, the
      `network.approaches` key whose in-edges hold the trip's depart edge;
    - bus: `delay_total_mean`, `delay_mean__flow_F` (timeLoss per flow, F the flow id
      without `bus_`) and `headway_std__line_L` (mean over the line's stops with at
      least 3 arrivals; `bus_stops` is `read_bus_stops` output);
    - ambulance: `travel_time_total` (duration + departDelay = arrival - planned
      departure) and `stop_duration_mean` (sum of waitingTime / sum of waitingCount, 0
      without a stop);
    - pedestrian: `ped_wait_single_mean__crossing_X`, `ped_wait_single_max__crossing_X`
      from `ped_single_waits` (longest single wait per person, crossing X from the person
      id, see `crossing_of`), plus the crossing values of `junction`;
    - pedestrian, KPI version 3: `ped_curb_wait_mean`, `ped_curb_wait_p95`,
      `ped_curb_wait_max` and per crossing `ped_curb_wait_mean__crossing_X`,
      `ped_curb_wait_max__crossing_X` from `ped_curb_waits` (longest curb wait per
      person, `SimSession.ped_curb_wait_max`: like the single wait, but a creep at the
      red crossing does not reset it; the A-04 / 90 s measure);
    - junction: box and exit blocking from `junction` (`JunctionMonitor.summary()`,
      None without a monitor; see JUNCTION_KPIS)."""
    rows: list[tuple[str, str, float]] = []

    def add(user: str, kpi: str, value: float) -> None:
        if value is not None and np.isfinite(value):
            rows.append((user, kpi, float(value)))

    def q(base: str, qualifier: str) -> str:
        return f"{base}{QUALIFIER_SEP}{qualifier}"

    if not trips.empty:
        trips = trips.assign(delay_total_s=trips["time_loss_s"] + trips["depart_delay_s"])
        car = trips[trips["type"] == "car"]
        if not car.empty:
            begin, duration = float(cfg["begin_s"]), float(cfg["duration_s"])
            arrived = car["arrival_s"].between(begin, begin + duration, inclusive="left")
            add("car", "throughput_vph", arrived.sum() / (duration / 3600.0))
            add("car", "delay_total_mean", car["delay_total_s"].mean())
            edge_arm = {e: arm for arm, edges in approach_in_edges(cfg).items() for e in edges}
            arm = car["depart_edge"].map(edge_arm)
            arms = [a for a in dict.fromkeys(edge_arm.values()) if (arm == a).any()]
            for base, col in (
                ("time_loss_mean", "time_loss_s"),
                ("delay_total_mean", "delay_total_s"),
            ):
                for a in arms:
                    add("car", q(base, f"arm_{a}"), car.loc[arm == a, col].mean())
        bus = trips[trips["type"] == "bus"]
        if not bus.empty:
            add("bus", "delay_total_mean", bus["delay_total_s"].mean())
            for flow, g in bus.groupby("flow", sort=True):
                add(
                    "bus",
                    q("delay_mean", f"flow_{flow.removeprefix('bus_')}"),
                    g["time_loss_s"].mean(),
                )
        amb = trips[trips["type"] == "ambulance"]
        if not amb.empty:
            add(
                "ambulance", "travel_time_total", (amb["duration_s"] + amb["depart_delay_s"]).mean()
            )
            n_stops = amb["stops"].sum()
            add(
                "ambulance",
                "stop_duration_mean",
                amb["waiting_s"].sum() / n_stops if n_stops else 0.0,
            )

    if isinstance(bus_stops, pd.DataFrame) and not bus_stops.empty:
        for line, g in bus_stops.groupby("line", sort=True):
            groups = [s["started"] for _, s in g.groupby("busStop") if len(s) >= 3]
            if groups:
                add(
                    "bus",
                    q("headway_std", f"line_{line}"),
                    np.mean([_headway_std(s) for s in groups]),
                )

    crossings = (cfg.get("network") or {}).get("crossings") or {}
    if crossings and ped_single_waits is not None and not ped_single_waits.empty:
        names = pd.Series([crossing_of(str(p), crossings) for p in ped_single_waits.index])
        waits = pd.Series(ped_single_waits.to_numpy(dtype=float))
        for base, agg in (("ped_wait_single_mean", "mean"), ("ped_wait_single_max", "max")):
            for name in crossings:
                sel = waits[names == name]
                if not sel.empty:
                    add("pedestrian", q(base, f"crossing_{name}"), sel.agg(agg))
    if crossings and ped_curb_waits is not None and not ped_curb_waits.empty:
        curb = pd.Series(ped_curb_waits.to_numpy(dtype=float))
        add("pedestrian", "ped_curb_wait_mean", curb.mean())
        add("pedestrian", "ped_curb_wait_p95", curb.quantile(0.95))
        add("pedestrian", "ped_curb_wait_max", curb.max())
        names = pd.Series([crossing_of(str(p), crossings) for p in ped_curb_waits.index])
        for base, agg in (("ped_curb_wait_mean", "mean"), ("ped_curb_wait_max", "max")):
            for name in crossings:
                sel = curb[names == name]
                if not sel.empty:
                    add("pedestrian", q(base, f"crossing_{name}"), sel.agg(agg))

    for key, user in JUNCTION_KPIS.items():
        if junction and key in junction:
            add(user, key, junction[key])
    return pd.DataFrame(rows, columns=["user_type", "kpi", "value"])


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

"""Generate the SUMO demand files from the scenario config: `<scenario>.rou.xml` (routes)
and `<scenario>.stops.add.xml` (bus stops). Two schemas:

- Legacy (`demand.approaches`, the grid): car flows per approach and turn over the whole
  run, one bus line with a stop, pedestrian flows across each arm, one ambulance. Its
  output is frozen (Part 3, decision D2).
- Calibrated (`demand.calibration`, the Pitié junction): car flows per slot and movement
  from the calibration file (`data.calibrate`), scaled by `multiplier` and `arm_scale`
  (with `demand.bus.counted_by_sensors` the arm's scheduled buses are taken out first:
  the sensors, like the simulated E1 stations, count every vehicle);
  the bus lines of `demand.bus.lines` with their `network.bus_stops`; pedestrian flows
  along every `network.crossings` walk, pinned near the junction (`departPos` on the
  personFlow, `arrivalPos` on the walk); one ambulance. Elements are sorted by
  departure, as SUMO ignores out-of-order ones.

Both schemas take the ambulances from the `emergency` block: one `ambulance_0`
(`depart_s`, `route`; output unchanged since Part 3), or a list of `trips` (Part 4, the
ambulance-probe family, D25) written as `ambulance_0..k` in departure order
(`emergency_trips`).

Flows use `probability` (cars, pedestrians) or a `period` (buses), so departures change
with the seed while the expected demand stays identical across scenarios (spec 13.3).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import quoteattr

import yaml

from traffic_twin.config import (
    ConfigError,
    approach_in_edges,
    approach_out_edges,
    load_scenario,
)
from traffic_twin.paths import DEMAND_DIR, SCENARIO_CONFIG

# Destination edge for each approach and turn (right-hand traffic)
TURNS = {
    "W": {"left": "n_out", "straight": "e_out", "right": "s_out"},
    "E": {"left": "s_out", "straight": "w_out", "right": "n_out"},
    "N": {"left": "e_out", "straight": "s_out", "right": "w_out"},
    "S": {"left": "w_out", "straight": "n_out", "right": "e_out"},
}


def _attrs(**kw) -> str:
    return " ".join(f"{k}={quoteattr(str(v))}" for k, v in kw.items())


def _elem(tag: str, children: str = "", **kw) -> str:
    if children:
        return f"    <{tag} {_attrs(**kw)}>{children}</{tag}>"
    return f"    <{tag} {_attrs(**kw)}/>"


def _route(edges: list[str]) -> str:
    return f"<route edges={quoteattr(' '.join(edges))}/>"


# -- legacy schema (grid; frozen) ------------------------------------------------------
def car_flows(cfg: dict, begin: int, end: int) -> list[str]:
    mult = cfg["demand"].get("multiplier", 1.0)
    lines = []
    for name, a in cfg["demand"]["approaches"].items():
        total = sum(a["turns"].values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"turn ratios of {name} sum to {total}, expected 1")
        for turn, share in a["turns"].items():
            vph = a["flow_veh_h"] * share * mult
            if vph <= 0:
                continue
            route = _route([a["edge"], TURNS[name][turn]])
            lines.append(
                _elem("flow", route, id=f"car_{name}_{turn}", type="car", begin=begin, end=end,
                      probability=round(vph / 3600, 6), departLane="best", departSpeed="max")
            )  # fmt: skip
    return lines


def bus_flow(cfg: dict, begin: int, end: int) -> list[str]:
    b = cfg["demand"]["bus"]
    line = b["line"]
    stop = f"<stop {_attrs(busStop='stop_' + line, duration=b['dwell_s'])}/>"
    return [
        _elem("flow", _route(b["route"]) + stop, id=f"bus_{line}", type="bus", line=line,
              begin=begin, end=end, period=b["headway_s"], departLane="best")
    ]  # fmt: skip


# Walks along each axis (one or two crossings) and two corner-to-corner walks
PED_WALKS = [
    ("w_in", "e_out"), ("e_in", "w_out"), ("n_in", "s_out"), ("s_in", "n_out"),
    ("w_in", "n_out"), ("s_in", "e_out"),
]  # fmt: skip


def ped_flows(cfg: dict, begin: int, end: int) -> list[str]:
    pph = cfg["demand"]["pedestrians"]["flow_ped_h"] * cfg["demand"].get("multiplier", 1.0)
    walk = "<walk from={} to={}/>"
    return [
        _elem("personFlow", walk.format(quoteattr(src), quoteattr(dst)), id=f"ped_{src}_{dst}",
              begin=begin, end=end, probability=round(pph / 3600, 6))
        for src, dst in PED_WALKS
    ]  # fmt: skip


def ambulance(cfg: dict, begin: int) -> list[str]:
    """The ambulance vehicles' XML, by departure (`emergency_trips`)."""
    return [xml for *_, xml in ambulance_elements(cfg, begin)]


VTYPES = """    <vType id="car" vClass="passenger" speedDev="0.1"/>
    <vType id="bus" vClass="bus" length="12" maxSpeed="13.89" color="0,0.4,1"/>
    <vType id="ambulance" vClass="emergency" guiShape="emergency" speedFactor="1.3"
           color="1,0,0"/>"""


# -- calibrated schema (Pitié) ---------------------------------------------------------
KIND_ORDER = {"car": 0, "bus": 1, "person": 2, "vehicle": 3}  # same departure: this order
SHARE_TOL = 1e-3  # turn shares of an arm must sum to 1 within this

Element = tuple[int, int, str, str]  # (departure, kind order, id, xml)


# -- emergency vehicles (both schemas) -------------------------------------------------
def _known_edges(cfg: dict) -> set[str]:
    """In- and out-edges of `network.approaches` (empty when the config has none)."""
    if not (cfg.get("network") or {}).get("approaches"):
        return set()
    ins, outs = approach_in_edges(cfg), approach_out_edges(cfg)
    return {e for edges in (*ins.values(), *outs.values()) for e in edges}


def _trip_route(route: object, where: str, known: set[str]) -> list[str]:
    if not isinstance(route, list | tuple) or not route:
        raise ConfigError(f"{where}.route must be a non-empty list of edges, got {route!r}")
    if not all(isinstance(e, str) and e for e in route):
        raise ConfigError(f"{where}.route must list edge ids, got {route!r}")
    unknown = [e for e in route if known and e not in known]
    if unknown:
        raise ConfigError(f"{where}.route: unknown edge {', '.join(unknown)}")
    return list(route)


def emergency_trips(cfg: dict) -> list[tuple[float, str, list[str]]]:
    """(departure after begin_s, vehicle id, route) of every ambulance, by departure.

    Two forms of the `emergency` block (when `enabled`):
    - single: `depart_s` and `route`, one vehicle `ambulance_0` (Parts 1-3; its XML is
      frozen);
    - `trips: [{depart_s, route}, ...]` (Part 4, D25) with `depart_s` and `route` null:
      `ambulance_{k}`, k the index in the list sorted by `depart_s` (stable). Each trip
      departs within the demand window, and its route is a non-empty list of
      `network.approaches` edges (checked against the net by `write_demand`)."""
    e = cfg.get("emergency") or {}
    if not e.get("enabled"):
        return []
    trips = e.get("trips")
    if trips is None:
        if e.get("depart_s") is None or not e.get("route"):
            raise ConfigError("emergency needs depart_s and route, or trips")
        return [(e["depart_s"], "ambulance_0", list(e["route"]))]
    if e.get("depart_s") is not None or e.get("route") is not None:
        raise ConfigError(
            "emergency: give either trips or depart_s and route, not both "
            "(with trips, depart_s and route must be null)"
        )
    if not isinstance(trips, list) or not trips:
        raise ConfigError(f"emergency.trips must be a non-empty list, got {trips!r}")
    duration = cfg.get("duration_s")
    known = _known_edges(cfg)
    parsed = []
    for i, trip in enumerate(trips):
        where = f"emergency.trips[{i}]"
        if not isinstance(trip, dict) or set(trip) != {"depart_s", "route"}:
            raise ConfigError(f"{where} must be a mapping of depart_s and route, got {trip!r}")
        depart = trip["depart_s"]
        if isinstance(depart, bool) or not isinstance(depart, int | float) or depart < 0:
            raise ConfigError(f"{where}.depart_s must be a number >= 0, got {depart!r}")
        if duration is not None and depart >= float(duration):
            raise ConfigError(f"{where}.depart_s {depart} is not in the demand window")
        parsed.append((depart, i, _trip_route(trip["route"], where, known)))
    parsed.sort(key=lambda p: (p[0], p[1]))
    return [(depart, f"ambulance_{k}", route) for k, (depart, _, route) in enumerate(parsed)]


def ambulance_elements(cfg: dict, begin: int) -> list[Element]:
    """One `<vehicle type="ambulance">` per emergency trip, as sortable elements."""
    out: list[Element] = []
    for depart, vid, route in emergency_trips(cfg):
        xml = _elem("vehicle", _route(route), id=vid, type="ambulance", depart=begin + depart,
                    departLane="best")  # fmt: skip
        out.append((begin + depart, KIND_ORDER["vehicle"], vid, xml))
    return out


def is_calibrated(cfg: dict) -> bool:
    """True when car demand comes from a calibration file (`demand.calibration`)."""
    return bool((cfg.get("demand") or {}).get("calibration"))


def load_calibration(cfg: dict) -> dict:
    """The calibration file named by `demand.calibration` (relative to the working
    directory, the repo root)."""
    path = Path(cfg["demand"]["calibration"])
    if not path.is_file():
        raise ConfigError(f"demand.calibration: file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cal = yaml.safe_load(f)
    if not isinstance(cal, dict) or not {"window", "arms", "turns"} <= set(cal):
        raise ConfigError(f"{path}: not a calibration file (needs window, arms and turns)")
    return cal


def _general_edges(cfg: dict) -> tuple[dict[str, str], dict[str, str]]:
    """First in- and out-edge of each approach open to general traffic."""
    approaches = cfg["network"]["approaches"]
    ins, outs = approach_in_edges(cfg), approach_out_edges(cfg)
    general = [k for k, a in approaches.items() if not (isinstance(a, dict) and a.get("bus_only"))]
    return {k: ins[k][0] for k in general}, {k: outs[k][0] for k in general}


def entering_bus_lines(cfg: dict, arm: str) -> list[dict]:
    """The `demand.bus.lines` whose route starts on an in-edge of approach `arm`."""
    ins = set(approach_in_edges(cfg).get(arm, []))
    lines = (cfg["demand"].get("bus") or {}).get("lines") or []
    return [ln for ln in lines if list(ln.get("route") or [ln["from"]])[0] in ins]


def scheduled_bus_vph(lines: list[dict], b: int, e: int, begin: int, end: int) -> float:
    """Mean scheduled flow (veh/h) of bus `lines` over [b, e): 3600 / headway_s while a
    line runs (from begin + offset_s to end, as `bus_line_flows` writes it)."""
    if e <= b:
        return 0.0
    total = 0.0
    for ln in lines:
        overlap = min(e, end) - max(b, begin + int(ln.get("offset_s", 0)))
        if overlap > 0:
            total += 3600.0 / float(ln["headway_s"]) * overlap / (e - b)
    return total


def slot_car_flows(cfg: dict, cal: dict, begin: int, end: int) -> list[Element]:
    """One flow per calibration slot and movement, clipped to [begin, end):
    `car_{arm}_{dest}_{k:02d}`, probability = cars * share / 3600 with
    cars = q * multiplier * arm_scale, the calibrated flow q of the arm's sensor.

    With `demand.bus.counted_by_sensors` the sensor flow counts every vehicle (as the
    simulated E1 stations do), so the buses scheduled to enter by that arm are taken
    out: cars = max(q * multiplier * arm_scale - bus flow, 0). Bus timetables are never
    scaled."""
    d = cfg["demand"]
    mult = float(d.get("multiplier", 1.0))
    scale = d.get("arm_scale") or {}
    net_of_buses = bool((d.get("bus") or {}).get("counted_by_sensors"))
    arms, turns, window = cal["arms"], cal["turns"], cal["window"]
    slot_s = int(window["slot_s"])
    if begin < int(window["begin_s"]) or end > int(window["end_s"]):
        raise ConfigError(
            f"scenario window {begin}-{end} s is not covered by the calibration window "
            f"{window['begin_s']}-{window['end_s']} s"
        )
    unknown = sorted(set(scale) - set(arms))
    if unknown:
        raise ConfigError(f"demand.arm_scale: unknown arm {', '.join(unknown)}")
    in_edge, out_edge = _general_edges(cfg)
    out: list[Element] = []
    for arm, a in arms.items():
        if arm not in in_edge:
            raise ConfigError(f"calibration arm {arm} is not a general network.approaches leg")
        shares = turns.get(arm)
        if not isinstance(shares, dict) or abs(sum(shares.values()) - 1) > SHARE_TOL:
            raise ConfigError(f"calibration turns of {arm} must sum to 1, got {shares}")
        buses = entering_bus_lines(cfg, arm) if net_of_buses else []
        factor = mult * float(scale.get(arm, 1.0))
        for dest, share in shares.items():
            if dest not in out_edge or dest == arm:
                raise ConfigError(f"turn {arm} -> {dest}: unknown destination leg")
            if share <= 0:
                continue
            route = _route([in_edge[arm], out_edge[dest]])
            for k, slot in enumerate(a["slots"]):
                b0 = int(slot["begin_s"])
                b, e = max(b0, begin), min(b0 + slot_s, end)
                bus = scheduled_bus_vph(buses, b, e, begin, end)
                cars = float(slot["flow_veh_h"]) * factor
                vph = (max(cars - bus, 0.0) if bus else cars) * share
                prob = round(vph / 3600, 6)
                if e <= b or prob <= 0:
                    continue
                if prob > 1:
                    raise ConfigError(f"car flow {arm} -> {dest}: {vph:.0f} veh/h is above 3600")
                fid = f"car_{arm}_{dest}_{k:02d}"
                xml = _elem("flow", route, id=fid, type="car", begin=b, end=e, probability=prob,
                            departLane="best", departSpeed="max")  # fmt: skip
                out.append((b, KIND_ORDER["car"], fid, xml))
    return out


def bus_line_flows(cfg: dict, begin: int, end: int) -> list[Element]:
    """`bus_{id}` flows of `demand.bus.lines` (route `[from, to]`, or `route`), one bus
    every `headway_s` from begin + `offset_s`, with their stops."""
    stops = cfg["network"].get("bus_stops") or {}
    out: list[Element] = []
    seen: set[str] = set()
    for ln in cfg["demand"]["bus"]["lines"]:
        fid = f"bus_{ln['id']}"
        if fid in seen:
            raise ConfigError(f"demand.bus.lines: duplicate id {ln['id']}")
        seen.add(fid)
        b = begin + int(ln.get("offset_s", 0))
        if b >= end:
            continue
        children = _route(list(ln.get("route") or [ln["from"], ln["to"]]))
        for st in ln.get("stops") or []:
            if st["stop"] not in stops:
                raise ConfigError(f"bus line {ln['id']}: unknown stop {st['stop']}")
            children += f"<stop {_attrs(busStop=st['stop'], duration=st['dwell_s'])}/>"
        xml = _elem("flow", children, id=fid, type="bus", line=ln["line"], begin=b, end=end,
                    period=ln["headway_s"], departLane="best")  # fmt: skip
        out.append((b, KIND_ORDER["bus"], fid, xml))
    return out


def crossing_walks(cfg: dict, begin: int, end: int) -> list[Element]:
    """`ped_{crossing}_{k}` person flows along the walks of `network.crossings`.

    `demand.pedestrians.flow_ped_h` is the flow of each walk: one number, or one per
    crossing (`{S: 120, W: 80}`, missing crossings get none), times the multiplier."""
    crossings = cfg["network"].get("crossings") or {}
    pph = cfg["demand"]["pedestrians"]["flow_ped_h"]
    mult = float(cfg["demand"].get("multiplier", 1.0))
    if isinstance(pph, dict):
        unknown = sorted(set(pph) - set(crossings))
        if unknown:
            raise ConfigError(f"demand.pedestrians.flow_ped_h: unknown crossing {unknown}")
    elif pph and not crossings:
        raise ConfigError("pedestrian demand needs network.crossings (the walks to pin)")
    out: list[Element] = []
    for name, c in crossings.items():
        flow = pph.get(name, 0) if isinstance(pph, dict) else pph
        prob = round(float(flow) * mult / 3600, 6)
        if prob <= 0:
            continue
        for k, w in enumerate(c.get("walks") or []):
            fid = f"ped_{name}_{k}"
            ends = {"from": w["from"], "to": w["to"], "arrivalPos": w["arrival_pos"]}
            xml = _elem("personFlow", f"<walk {_attrs(**ends)}/>", id=fid, begin=begin, end=end,
                        probability=prob, departPos=w["depart_pos"])  # fmt: skip
            out.append((begin, KIND_ORDER["person"], fid, xml))
    return out


def calibrated_routes(cfg: dict, cal: dict) -> str:
    """Route file text of the calibrated schema, every element sorted by (departure,
    car < bus < person < vehicle, id)."""
    begin = int(cfg["begin_s"])
    end = begin + int(cfg["duration_s"])
    items = slot_car_flows(cfg, cal, begin, end) + bus_line_flows(cfg, begin, end)
    items += crossing_walks(cfg, begin, end)
    items += ambulance_elements(cfg, begin)
    body = [xml for *_, xml in sorted(items, key=lambda i: i[:3])]
    return "<routes>\n" + VTYPES + "\n" + "\n".join(body) + "\n</routes>\n"


def build_routes(cfg: dict, calibration: dict | None = None) -> str:
    """Route file text. The calibrated schema is used when `demand.calibration` is set,
    or when a `calibration` mapping is given (then used instead of reading the file)."""
    if calibration is not None or is_calibrated(cfg):
        return calibrated_routes(cfg, calibration or load_calibration(cfg))
    begin = int(cfg["begin_s"])
    end = begin + int(cfg["duration_s"])
    body = car_flows(cfg, begin, end) + bus_flow(cfg, begin, end) + ped_flows(cfg, begin, end)
    body += ambulance(cfg, begin)
    return "<routes>\n" + VTYPES + "\n" + "\n".join(body) + "\n</routes>\n"


# -- bus stops -------------------------------------------------------------------------
def stop_lanes(cfg: dict) -> list[str]:
    """Lanes of the bus stops written to the additional file."""
    bus = cfg["demand"]["bus"]
    if "lines" not in bus:
        return [f"{bus['stop_edge']}_1"]
    stops = cfg["network"].get("bus_stops") or {}
    used = {st["stop"] for ln in bus["lines"] for st in ln.get("stops") or []}
    return [stops[s]["lane"] for s in stops if s in used]


def _line_stops(cfg: dict, lane_lengths: dict[str, float] | None) -> list[str]:
    stops = cfg["network"].get("bus_stops") or {}
    lines: dict[str, set[str]] = {}
    for ln in cfg["demand"]["bus"]["lines"]:
        for st in ln.get("stops") or []:
            if st["stop"] not in stops:
                raise ConfigError(f"bus line {ln['id']}: unknown stop {st['stop']}")
            lines.setdefault(st["stop"], set()).add(str(ln["line"]))
    out = []
    for sid, st in stops.items():
        if sid not in lines:
            continue
        start, end = float(st["start"]), float(st["end"])
        length = None if lane_lengths is None else lane_lengths.get(st["lane"])
        if lane_lengths is not None and length is None:
            raise ConfigError(f"bus stop {sid}: lane {st['lane']} is not in the network")
        if not 0 <= start < end or (length is not None and end > length + 1e-6):
            raise ConfigError(f"bus stop {sid}: [{start}, {end}] m does not fit lane {st['lane']}")
        out.append(_elem("busStop", id=sid, lane=st["lane"], startPos=st["start"],
                         endPos=st["end"], lines=" ".join(sorted(lines[sid]))))  # fmt: skip
    return out


def build_additional(cfg: dict, lane_lengths: dict[str, float] | None = None) -> str:
    """Bus-stop additional file text. Legacy schema: the stop of `bus.line` on lane
    `<stop_edge>_1`, moved back to fit the lane. `demand.bus.lines`: one busStop per
    `network.bus_stops` id in use, `lines` = the lines stopping there. With
    `lane_lengths` (lane id -> length in m) the stops are checked against their lane."""
    b = cfg["demand"]["bus"]
    if "lines" in b:
        stops = "".join(f"{x}\n" for x in _line_stops(cfg, lane_lengths))
        return f"<additional>\n{stops}</additional>\n"
    lane = f"{b['stop_edge']}_1"
    start = b["stop_pos_m"]
    if lane_lengths is not None:
        if lane not in lane_lengths:
            raise ConfigError(f"bus stop lane {lane} is not in the network")
        start = min(start, lane_lengths[lane] - 20)
    stop = _elem("busStop", id=f"stop_{b['line']}", lane=lane,
                 startPos=start, endPos=start + 20, lines=b["line"])  # fmt: skip
    return f"<additional>\n{stop}\n</additional>\n"


def write_demand(cfg: dict, out_dir: Path) -> tuple[Path, Path]:
    """Write `<scenario>.rou.xml` and `<scenario>.stops.add.xml` into `out_dir` (one
    stop file per scenario, so parallel runs never share it)."""
    import sumolib

    net_file = cfg["network"]["net_file"]
    net = sumolib.net.readNet(net_file)
    lane_lengths = {}
    for lane in stop_lanes(cfg):
        try:
            lane_lengths[lane] = net.getLane(lane).getLength()
        except (KeyError, IndexError, ValueError) as e:
            raise ConfigError(f"bus stop lane {lane} is not in {net_file}") from e
    for _depart, vid, route in emergency_trips(cfg):
        missing = [e for e in route if not net.hasEdge(e)]
        if missing:
            raise ConfigError(f"{vid}: route edge {', '.join(missing)} is not in {net_file}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rou = out_dir / f"{cfg['scenario']}.rou.xml"
    add = out_dir / f"{cfg['scenario']}.stops.add.xml"
    rou.write_text(build_routes(cfg), encoding="utf-8")
    add.write_text(build_additional(cfg, lane_lengths), encoding="utf-8")
    return rou, add


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=SCENARIO_CONFIG)
    parser.add_argument("--out", type=Path, default=DEMAND_DIR)
    args = parser.parse_args(argv)
    for p in write_demand(load_scenario(args.config), args.out):
        print(p)


if __name__ == "__main__":
    main()

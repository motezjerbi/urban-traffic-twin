"""Generate the SUMO demand file (.rou.xml) from the scenario config: car flows per
approach and turn, a bus line with a stop, pedestrian flows across each arm, and one
ambulance. Flows use `probability`, so departures change with the seed while the
expected demand stays identical across scenarios (spec 13.3)."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import quoteattr

from traffic_twin.config import load_scenario
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
    e = cfg.get("emergency", {})
    if not e.get("enabled"):
        return []
    return [
        _elem("vehicle", _route(e["route"]), id="ambulance_0", type="ambulance",
              depart=begin + e["depart_s"], departLane="best")
    ]  # fmt: skip


VTYPES = """    <vType id="car" vClass="passenger" speedDev="0.1"/>
    <vType id="bus" vClass="bus" length="12" maxSpeed="13.89" color="0,0.4,1"/>
    <vType id="ambulance" vClass="emergency" guiShape="emergency" speedFactor="1.3"
           color="1,0,0"/>"""


def build_routes(cfg: dict) -> str:
    begin = int(cfg["begin_s"])
    end = begin + int(cfg["duration_s"])
    body = car_flows(cfg, begin, end) + bus_flow(cfg, begin, end) + ped_flows(cfg, begin, end)
    body += ambulance(cfg, begin)
    return "<routes>\n" + VTYPES + "\n" + "\n".join(body) + "\n</routes>\n"


def build_additional(cfg: dict, lane_length: float) -> str:
    b = cfg["demand"]["bus"]
    start = min(b["stop_pos_m"], lane_length - 20)
    stop = _elem("busStop", id=f"stop_{b['line']}", lane=f"{b['stop_edge']}_1",
                 startPos=start, endPos=start + 20, lines=b["line"])  # fmt: skip
    return f"<additional>\n{stop}\n</additional>\n"


def write_demand(cfg: dict, out_dir: Path) -> tuple[Path, Path]:
    import sumolib

    net = sumolib.net.readNet(cfg["network"]["net_file"])
    stop_lane = net.getLane(f"{cfg['demand']['bus']['stop_edge']}_1")
    out_dir.mkdir(parents=True, exist_ok=True)
    rou = out_dir / f"{cfg['scenario']}.rou.xml"
    add = out_dir / "bus_stops.add.xml"
    rou.write_text(build_routes(cfg), encoding="utf-8")
    add.write_text(build_additional(cfg, stop_lane.getLength()), encoding="utf-8")
    return rou, add


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCENARIO_CONFIG)
    parser.add_argument("--out", type=Path, default=DEMAND_DIR)
    args = parser.parse_args(argv)
    for p in write_demand(load_scenario(args.config), args.out):
        print(p)


if __name__ == "__main__":
    main()

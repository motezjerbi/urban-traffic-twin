"""Static description of the junction for the API (Part 5, D53), read from versioned files
only (CI has them): the site config (`configs/pitie_fixed.yaml`), its net file, the
calibration file, the field inventory and `configs/eta.yaml`. Nothing here talks to SUMO.

- Stages and the SUMO phase -> (stage, in clearance) map from `network.stages`.
- Phase states and durations from the net's `<tlLogic>` (the program the replayed
  signal log is made of), crossing lengths and the junction centre from the net.
- Stop lines of the approach in-edges (end of their lane shapes, with the direction of
  travel there) and the TLS link indices of each (in-edge, out-edge) movement, so the
  ambulance answer can tell an ambulance before the stop line from one that crossed.
- Bus lines (flow id, line, scheduled headway) from `demand.bus.lines`.
- `calibrated_on` = the last sensor day the calibration used (`day_filter.days_used`);
  the calibration file has no generation date, and adding one would change
  `demand_hash` and make every run stale.
- `provisional` = signal timing or queues not yet measured in the field
  (`data/field/pitie_measurements.yaml` inventory).
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from traffic_twin.config import ConfigError, approach_in_edges, get, load_config, load_scenario

log = logging.getLogger(__name__)

GREEN = "Gg"  # SUMO link states that let traffic go


@dataclass(frozen=True)
class Crossing:
    id: str
    link: int
    edge: str
    length_m: float | None


@dataclass(frozen=True)
class BusLine:
    flow: str  # demand.bus.lines id, e.g. 91_E (trip flow `bus_91_E`)
    line: str
    headway_s: float
    from_edge: str
    to_edge: str


@dataclass(frozen=True)
class StopLine:
    """Stop line of an in-edge: a point on it and the unit direction of travel there."""

    x: float
    y: float
    ux: float
    uy: float

    def along(self, x: float, y: float) -> float:
        """Signed distance (m) along the direction of travel from the stop line to (x, y):
        negative before the line, positive once past it."""
        return (x - self.x) * self.ux + (y - self.y) * self.uy


@dataclass(frozen=True)
class Phase:
    state: str
    duration_s: float
    green: frozenset[int]  # links with G or g


def phase_map(stages: list[dict]) -> dict[int, tuple[int, bool]]:
    """SUMO phase -> (stage index, in clearance) of `network.stages`."""
    out: dict[int, tuple[int, bool]] = {}
    for k, st in enumerate(stages):
        out[int(st["green_phase"])] = (k, False)
        for p in st.get("clearance_phases", []):
            out[int(p)] = (k, True)
    return out


@dataclass(frozen=True)
class Site:
    tls_id: str
    net_file: str  # as written in the site config (runs of another net are not this site)
    stages: tuple[str, ...]
    stage_green_phase: tuple[int, ...]
    phase_stage: Mapping[int, tuple[int, bool]]  # phase -> (stage index, in clearance)
    phases: tuple[Phase, ...]  # program of the net (empty when the net is unreadable)
    crossings: Mapping[str, Crossing]
    bus_lines: Mapping[str, BusLine]
    approaches: Mapping[str, tuple[str, ...]]  # approach -> in-edges
    junction_xy: tuple[float, float] | None
    calibrated_on: date | None
    provisional: bool
    stop_lines: Mapping[str, StopLine] = field(default_factory=dict)  # in-edge -> stop line
    movement_links: Mapping[tuple[str, str], tuple[int, ...]] = field(default_factory=dict)

    def stage_of(self, phase: int | None) -> tuple[int | None, str | None, bool | None]:
        """(stage index, stage name, in clearance) of a SUMO phase index."""
        if phase is None or phase not in self.phase_stage:
            return None, None, None
        k, clearing = self.phase_stage[phase]
        return k, self.stages[k], clearing

    def phase_of_green(self, green: frozenset[int]) -> tuple[int | None, bool | None]:
        """(stage, in clearance) of the program phase(s) whose green links are exactly
        `green`; (None, None) when no phase or phases of different stages match."""
        found = {
            self.phase_stage[i] for i, p in enumerate(self.phases)
            if p.green == green and i in self.phase_stage
        }  # fmt: skip
        if not found:
            return None, None
        stages = {k for k, _ in found}
        if len(stages) != 1:
            return None, None
        k = stages.pop()
        return k, all(c for _, c in found)

    def link_green_in_stage(self, link: int, stage: int) -> bool:
        if not self.phases:
            return False
        return link in self.phases[self.stage_green_phase[stage]].green

    def lines_of(self, bus_id: str) -> list[BusLine]:
        """The bus flows a client id names: a flow id (`91_E`) or a line (`91`)."""
        key = bus_id.strip().upper()
        return [b for b in self.bus_lines.values() if key in (b.flow.upper(), b.line.upper())]

    def approach_of_edge(self, edge: str) -> str | None:
        return next((a for a, es in self.approaches.items() if edge in es), None)

    def movement_of_route(self, route: list[str] | None) -> tuple[str, str | None] | None:
        """(in-edge, next edge) of a route at this junction: its first approach in-edge
        and the edge after it (None when the route ends there); None without one."""
        for k, e in enumerate(route or []):
            if self.approach_of_edge(e) is not None:
                return e, (route[k + 1] if k + 1 < len(route) else None)
        return None


def calibrated_on(calibration: Path) -> date | None:
    """The last sensor day of `day_filter.days_used` of a calibration file."""
    try:
        cal = yaml.safe_load(calibration.read_text(encoding="utf-8"))
        days = [str(d) for ds in cal["day_filter"]["days_used"].values() for d in ds]
        return max(date.fromisoformat(d) for d in days)
    except (OSError, KeyError, TypeError, ValueError, AttributeError, yaml.YAMLError):
        log.warning("calibrated_on: no days_used in %s", calibration)
        return None


def is_provisional(field_file: Path, cfg: dict) -> bool:
    """True while the field file has no measured signal timing or no measured queue."""
    from traffic_twin.validation.field import inventory, load_field

    try:
        inv = inventory(load_field(field_file, cfg))
    except (ConfigError, OSError, yaml.YAMLError) as e:
        log.warning("field inventory unreadable (%s): provisional", e)
        return True
    return inv.get("signal", (0, 0))[0] == 0 or inv.get("queues", (0, 0))[0] == 0


def _stop_line(edge: ET.Element) -> StopLine | None:
    """Mean end point and mean last-segment direction of an edge's lane shapes."""
    ends, dirs = [], []
    for lane in edge.findall("lane"):
        pts = [tuple(float(v) for v in p.split(",")[:2]) for p in lane.get("shape", "").split()]
        if len(pts) < 2:
            continue
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        n = math.hypot(x1 - x0, y1 - y0)
        if n > 0:
            ends.append((x1, y1))
            dirs.append(((x1 - x0) / n, (y1 - y0) / n))
    if not ends:
        return None
    ux, uy = sum(d[0] for d in dirs), sum(d[1] for d in dirs)
    n = math.hypot(ux, uy)
    if n == 0:
        return None
    k = len(ends)
    return StopLine(sum(e[0] for e in ends) / k, sum(e[1] for e in ends) / k, ux / n, uy / n)


def _read_net(net: Path, tls_id: str, crossing_edges: set[str], in_edges: set[str] = frozenset()):
    lengths: dict[str, float] = {}
    xy = None
    phases: list[Phase] = []
    stops: dict[str, StopLine] = {}
    links: dict[tuple[str, str], set[int]] = {}
    try:
        for _, el in ET.iterparse(net, events=("end",)):
            if el.tag == "edge" and el.get("function") == "crossing":
                lane = el.find("lane")
                if el.get("id") in crossing_edges and lane is not None:
                    lengths[el.get("id")] = float(lane.get("length"))
            elif el.tag == "edge" and el.get("id") in in_edges:
                if (sl := _stop_line(el)) is not None:
                    stops[el.get("id")] = sl
            elif (
                el.tag == "connection"
                and el.get("tl") == tls_id
                and el.get("linkIndex") is not None
            ):
                key = (el.get("from"), el.get("to"))
                links.setdefault(key, set()).add(int(el.get("linkIndex")))
            elif el.tag == "tlLogic" and el.get("id") == tls_id:
                phases = [
                    Phase(
                        p.get("state"),
                        float(p.get("duration")),
                        frozenset(i for i, c in enumerate(p.get("state")) if c in GREEN),
                    )
                    for p in el.iter("phase")
                ]
            elif el.tag == "junction" and el.get("id") == tls_id:
                xy = (float(el.get("x")), float(el.get("y")))
    except (OSError, ET.ParseError, TypeError, ValueError) as e:
        log.warning("net file %s unreadable: %s", net, e)
    movements = {k: tuple(sorted(v)) for k, v in links.items()}
    return lengths, xy, tuple(phases), stops, movements


def load_site(site_config: Path, repo_root: Path, field_file: Path) -> Site:
    """The junction description from a scenario config (paths relative to repo_root)."""
    root = Path(repo_root)

    def at(p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else root / p

    cfg = load_scenario(at(site_config))
    net = get(cfg, "network")
    stages = net["stages"]
    raw_cross = net.get("crossings") or {}
    approaches = {a: tuple(es) for a, es in approach_in_edges(cfg).items()}
    lengths, xy, phases, stops, movements = _read_net(
        at(net["net_file"]),
        net["tls_id"],
        {c["edge"] for c in raw_cross.values()},
        {e for es in approaches.values() for e in es},
    )
    crossings = {
        cid: Crossing(cid, int(c["link"]), c["edge"], lengths.get(c["edge"]))
        for cid, c in raw_cross.items()
    }
    lines = {
        b["id"]: BusLine(b["id"], str(b["line"]), float(b["headway_s"]), b["from"], b["to"])
        for b in get(cfg, "demand.bus", {}).get("lines", [])
    }
    cal = get(cfg, "demand.calibration", None)
    return Site(
        tls_id=net["tls_id"],
        net_file=str(net["net_file"]),
        stages=tuple(s["name"] for s in stages),
        stage_green_phase=tuple(int(s["green_phase"]) for s in stages),
        phase_stage=phase_map(stages),
        phases=phases,
        crossings=crossings,
        bus_lines=lines,
        approaches=approaches,
        junction_xy=xy,
        calibrated_on=calibrated_on(at(cal)) if cal else None,
        provisional=is_provisional(at(field_file), cfg),
        stop_lines=stops,
        movement_links=movements,
    )


@dataclass(frozen=True)
class EtaSpec:
    """What `configs/eta.yaml` tells the API: trajets, the controllers and day types the
    model was trained on, and the student routes' rest-of-trip times (ASSUMED)."""

    trajets: tuple[str, ...]
    controllers: tuple[str, ...]
    day_types: tuple[str, ...]
    rest_s: Mapping[str, float] = field(default_factory=dict)  # student routes
    labels: Mapping[str, str] = field(default_factory=dict)


def load_eta_spec(eta_config: Path) -> EtaSpec:
    cfg = load_config(eta_config, ("trajets", "controllers", "calibration_files"))
    routes = (cfg.get("student") or {}).get("routes") or {}
    return EtaSpec(
        trajets=tuple(cfg["trajets"]),
        controllers=tuple(cfg["controllers"]),
        day_types=tuple(cfg["calibration_files"]),
        rest_s={r: float(v["rest_s"]) for r, v in routes.items()},
        labels={r: str(v.get("label", r)) for r, v in routes.items()},
    )

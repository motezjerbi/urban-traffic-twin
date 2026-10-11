"""Detectors added to a run (Part 3), all passive: they never change what is simulated.

- Control loops (`detectors.control`): one E1 per incoming vehicle lane with a TLS link,
  `detector_distance` before the stop line (SUMO's actuated placement rule; on a shorter
  lane the loop sits at its start with a warning, where SUMO would move it upstream),
  output discarded (`file="NUL"`, which SUMO maps to the null device on every OS). They
  feed `Observation.extras` for actuated control.
- Station loops (`measurement.stations`): E1 stand-ins for the Paris sensors, counted per
  `measurement.period_s` into `e1.xml`.
- Queue detectors (`measurement.queues`): E2 lane-area detectors ending at the stop line,
  read through TraCI by the session.
- Signal switches (`measurement.tls_switch`): SaveTLSSwitchTimes into `tls_switch.xml`
  (one green interval per link).

`write_detectors` puts the file in the run directory, so the relative output names
resolve there and parallel runs never share a file.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

import pandas as pd

from traffic_twin.config import ConfigError, get

log = logging.getLogger(__name__)

DETECTOR_FILE = "detectors.add.xml"
E1_FILE = "e1.xml"  # station loop counts
TLS_SWITCH_FILE = "tls_switch.xml"  # green intervals per link
CONTROL_PERIOD_S = 3600  # control loop output is discarded anyway
VEHICLE_LENGTH_WITH_GAP = 7.5  # SUMO's DEFAULT_LENGTH_WITH_GAP (m)
# E2 jam definition (SUMO defaults, written out so the definition is visible)
JAM = {"timeThreshold": 1, "speedThreshold": 1.39, "jamThreshold": 10}

E1_COLUMNS = ["det_id", "begin_s", "end_s", "n_veh", "occupancy_pct", "mean_speed"]
SWITCH_COLUMNS = ["tls_id", "from_lane", "to_lane", "begin_s", "end_s", "duration_s"]


@dataclass(frozen=True)
class Loop:
    """An E1 induction loop."""

    id: str
    lane: str
    pos: float  # m from the lane start
    role: str  # control | station


@dataclass(frozen=True)
class Area:
    """An E2 lane-area (queue) detector ending at the stop line."""

    id: str
    lane: str
    pos: float  # m from the lane start
    length: float


@dataclass(frozen=True)
class DetectorLayout:
    loops: tuple[Loop, ...]
    stage_loops: tuple[tuple[str, ...], ...]  # control loop ids per stage (network.stages)
    stations: dict[str, tuple[str, ...]]  # measurement station -> loop ids
    queues: dict[str, tuple[str, ...]]  # approach -> E2 ids
    areas: tuple[Area, ...] = ()
    # (from_lane, to_lane) -> TLS link index; set when the switch times are logged
    switch_links: dict[tuple[str, str], int] = field(default_factory=dict)

    @property
    def control_loops(self) -> tuple[Loop, ...]:
        return tuple(lp for lp in self.loops if lp.role == "control")

    @property
    def empty(self) -> bool:
        return not (self.loops or self.areas or self.switch_links)


# -- network -------------------------------------------------------------------------
@dataclass(frozen=True)
class TlsLink:
    index: int
    from_lane: str
    to_lane: str


@dataclass(frozen=True)
class NetInfo:
    """What the detectors need from a .net.xml: lane lengths and speeds, the links of
    each TLS and the phase states of its first program."""

    lanes: dict[str, tuple[float, float]]  # lane id -> (length, speed)
    tls_links: dict[str, tuple[TlsLink, ...]]
    tls_states: dict[str, tuple[str, ...]]


def read_net(path: str | Path) -> NetInfo:
    lanes: dict[str, tuple[float, float]] = {}
    links: dict[str, list[TlsLink]] = {}
    states: dict[str, tuple[str, ...]] = {}
    for el in ET.parse(path).getroot().iter():
        if el.tag == "lane":
            lanes[el.get("id")] = (float(el.get("length")), float(el.get("speed")))
        elif el.tag == "connection" and el.get("tl") is not None:
            link = TlsLink(
                int(el.get("linkIndex")),
                f"{el.get('from')}_{el.get('fromLane')}",
                f"{el.get('to')}_{el.get('toLane')}",
            )
            links.setdefault(el.get("tl"), []).append(link)
        elif el.tag == "tlLogic" and el.get("id") not in states:  # first program
            states[el.get("id")] = tuple(p.get("state") for p in el.iter("phase"))
    tls_links = {t: tuple(sorted(ls, key=lambda k: k.index)) for t, ls in links.items()}
    return NetInfo(lanes, tls_links, states)


def _lane(net: NetInfo, lane: str, where: str) -> tuple[float, float]:
    if lane not in net.lanes:
        raise ConfigError(f"{where}: lane '{lane}' is not in the network")
    return net.lanes[lane]


# -- control loops --------------------------------------------------------------------
def detector_distance(
    speed: float, min_green_s: float, detector_gap_s: float = 2.0, passing_time_s: float = 1.9
) -> float:
    """SUMO's default distance of an actuated loop from the stop line (m)."""
    return min(
        detector_gap_s * speed, (min_green_s / passing_time_s + 0.5) * VEHICLE_LENGTH_WITH_GAP
    )


def _stage_min_greens(cfg: dict) -> list[float]:
    """Per stage: `actuated.min_green_s[name]`, else `safety.min_green_s` (else 0)."""
    default = float(get(cfg, "safety.min_green_s", 0.0) or 0.0)
    per_stage = get(cfg, "actuated.min_green_s", None) or {}
    return [float(per_stage.get(s["name"], default)) for s in get(cfg, "network.stages")]


def _control_loops(
    cfg: dict, net: NetInfo, control: dict
) -> tuple[list[Loop], tuple[tuple[str, ...], ...]]:
    tls = get(cfg, "network.tls_id")
    if tls not in net.tls_links or tls not in net.tls_states:
        raise ConfigError(f"network.tls_id: no traffic light '{tls}' in the network")
    states = net.tls_states[tls]
    stages = get(cfg, "network.stages")
    lane_links: dict[str, list[int]] = {}
    for link in net.tls_links[tls]:
        if not link.from_lane.startswith(":"):  # crossings start on walking areas
            lane_links.setdefault(link.from_lane, []).append(link.index)

    green = [states[int(s["green_phase"])] for s in stages]
    min_greens = _stage_min_greens(cfg)
    loops, stage_loops = [], [[] for _ in stages]
    for lane, idx in lane_links.items():  # in link order
        det = f"det_{lane}"
        for k, state in enumerate(green):  # all links green, at least one major (G)
            chars = [state[i] for i in idx]
            if all(c in "Gg" for c in chars) and "G" in chars:
                stage_loops[k].append(det)
        # SUMO uses the smallest min green of the phases giving the lane any green
        lane_min = [
            m for m, st in zip(min_greens, green, strict=True) if any(st[i] in "Gg" for i in idx)
        ]
        length, speed = _lane(net, lane, "detectors.control")
        distance = control.get("distance_m")
        if distance is None:
            distance = detector_distance(
                speed,
                min(lane_min or min_greens),
                float(control.get("detector_gap_s", 2.0)),
                float(control.get("passing_time_s", 1.9)),
            )
        pos = length - float(distance)
        if pos < 0:
            log.warning("control loop %s: lane shorter than %.1f m, loop at 0", det, distance)
            pos = 0.0
        loops.append(Loop(det, lane, pos, "control"))

    missing = [s["name"] for s, ids in zip(stages, stage_loops, strict=True) if not ids]
    if missing:
        raise ConfigError(
            f"detectors.control: stages {missing} have no controlling loop (a lane whose "
            "links are all green in the stage, at least one of them major)"
        )
    return loops, tuple(tuple(ids) for ids in stage_loops)


# -- measurement ----------------------------------------------------------------------
def _station_pos(pos: float, length: float) -> float:
    """SUMO's friendlyPos rule: a negative position counts from the lane end, and the
    loop is kept on the lane (at most 0.1 m before its end)."""
    if pos < 0:
        pos += length
    return min(max(pos, 0.0), length - 0.1)


def _measurement(cfg: dict, net: NetInfo, meas: dict) -> tuple[
    list[Loop], dict[str, tuple[str, ...]], list[Area], dict[str, tuple[str, ...]],
    dict[tuple[str, str], int],
]:  # fmt: skip
    loops, stations = [], {}
    for st in meas.get("stations") or []:
        ids = []
        for lane in st["lanes"]:
            length, _ = _lane(net, lane, f"measurement.stations.{st['id']}")
            loop = Loop(
                f"e1_{st['id']}_{lane}", lane, _station_pos(float(st["pos_m"]), length), "station"
            )
            loops.append(loop)
            ids.append(loop.id)
        stations[st["id"]] = tuple(ids)

    areas, queues = [], {}
    for approach, q in (meas.get("queues") or {}).items():
        ids = []
        for lane in q["lanes"]:
            length, _ = _lane(net, lane, f"measurement.queues.{approach}")
            span = min(float(q["length_m"]), length - 1.0)
            if span <= 0:
                raise ConfigError(f"measurement.queues.{approach}: lane '{lane}' is too short")
            areas.append(Area(f"q_{approach}_{lane}", lane, length - span, span))
            ids.append(areas[-1].id)
        queues[approach] = tuple(ids)

    switch_links = {}
    if meas.get("tls_switch"):
        tls = get(cfg, "network.tls_id")
        switch_links = {(lk.from_lane, lk.to_lane): lk.index for lk in net.tls_links.get(tls, ())}
    return loops, stations, areas, queues, switch_links


def plan_detectors(cfg: dict, net: NetInfo | str | Path | None = None) -> DetectorLayout:
    """Detector layout of a scenario. Stage rule for control loops: a lane's loop
    controls a stage when all its TLS links are green (G or g) in the stage's green phase
    and at least one is G. The net is read only when detectors are configured."""
    n_stages = len(get(cfg, "network.stages"))
    control = get(cfg, "detectors.control", None)
    meas = cfg.get("measurement") or {}
    if control is None and not meas:
        return DetectorLayout((), ((),) * n_stages, {}, {})
    if not isinstance(net, NetInfo):
        net = read_net(net or get(cfg, "network.net_file"))

    loops, stage_loops = [], ((),) * n_stages
    if control is not None:
        loops, stage_loops = _control_loops(cfg, net, control)
    station_loops, stations, areas, queues, switch_links = _measurement(cfg, net, meas)
    return DetectorLayout(
        tuple(loops + station_loops), stage_loops, stations, queues, tuple(areas), switch_links
    )


# -- additional file ------------------------------------------------------------------
def _elem(tag: str, **kw) -> str:
    attrs = " ".join(
        f"{k}={quoteattr(v if isinstance(v, str) else repr(v))}" for k, v in kw.items()
    )
    return f"    <{tag} {attrs}/>"


def build_additional(layout: DetectorLayout, cfg: dict) -> str:
    """The detector additional file. Positions are written as exact floats (repr), so
    SUMO places the loops exactly where the layout says."""
    period = int(get(cfg, "measurement.period_s", 900))
    lines = []
    for lp in layout.loops:
        if lp.role == "control":
            lines.append(
                _elem("inductionLoop", id=lp.id, lane=lp.lane, pos=lp.pos,
                      period=CONTROL_PERIOD_S, file="NUL")
            )  # fmt: skip
        else:
            lines.append(
                _elem("inductionLoop", id=lp.id, lane=lp.lane, pos=lp.pos, period=period,
                      file=E1_FILE, friendlyPos="true")
            )  # fmt: skip
    for a in layout.areas:
        lines.append(
            _elem("laneAreaDetector", id=a.id, lane=a.lane, pos=a.pos, length=a.length,
                  period=period, file="NUL", **JAM)
        )  # fmt: skip
    if layout.switch_links:
        tls = get(cfg, "network.tls_id")
        lines.append(
            _elem("timedEvent", type="SaveTLSSwitchTimes", source=tls, dest=TLS_SWITCH_FILE)
        )
    return "<additional>\n" + "\n".join(lines) + "\n</additional>\n"


def write_detectors(cfg: dict, out_dir: Path) -> tuple[Path | None, DetectorLayout]:
    """Write `out_dir/detectors.add.xml`; no file (None) when the scenario configures
    neither `detectors.control` nor `measurement` (the grid)."""
    layout = plan_detectors(cfg)
    if layout.empty:
        return None, layout
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / DETECTOR_FILE
    path.write_text(build_additional(layout, cfg), encoding="utf-8")
    return path, layout


# -- outputs --------------------------------------------------------------------------
def _records(path: Path, tag: str):
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == tag:
            yield dict(el.attrib)
            el.clear()


def read_e1(path: str | Path) -> pd.DataFrame:
    """Station loop intervals from an E1 output file (absolute SUMO times). `mean_speed`
    is NaN for an interval with no vehicle."""
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=E1_COLUMNS)
    rows = []
    for a in _records(path, "interval"):
        speed = float(a.get("speed", "nan"))
        rows.append(
            {"det_id": a["id"], "begin_s": float(a["begin"]), "end_s": float(a["end"]),
             "n_veh": int(float(a["nVehContrib"])), "occupancy_pct": float(a["occupancy"]),
             "mean_speed": speed if speed >= 0 else math.nan}
        )  # fmt: skip
    return pd.DataFrame(rows, columns=E1_COLUMNS)


def read_tls_switch(path: str | Path) -> pd.DataFrame:
    """Green intervals per link from a SaveTLSSwitchTimes file (absolute SUMO times)."""
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=SWITCH_COLUMNS)
    rows = [
        {"tls_id": a["id"], "from_lane": a["fromLane"], "to_lane": a["toLane"],
         "begin_s": float(a["begin"]), "end_s": float(a["end"]),
         "duration_s": float(a["duration"])}
        for a in _records(path, "tlsSwitch")
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=SWITCH_COLUMNS)

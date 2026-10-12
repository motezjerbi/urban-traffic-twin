"""Static checks of a SUMO signal program against its network: foe greens, intergreen
times from the junction geometry, and the worst-case red of each pedestrian crossing.

    python -m traffic_twin.sim.tls_check --config configs/pitie_fixed.yaml   (exit 1 on errors)

Rules (spec 8.3 / A-04, lessons of the Pitié design runs):
- two vehicle links that are foes are never green together, not even with a permissive
  `g`: in a large joined junction a vehicle waiting inside on `g` is still there when the
  next stage starts. A vehicle may only be permissive (`g`) against a pedestrian crossing;
- intergreen (end of green, i.e. start of yellow, to the foe's next green) covers the
  path to the conflict point plus the vehicle length at a clearing speed, slower for buses
  leaving a near-side stop; a crossing's foes wait until the crossing is cleared at walking
  speed. It is checked with the fixed greens and with the minimum greens (the shortest
  greens any controller can show through the safety guard);
- the worst-case red of every crossing stays within `safety.ped_max_wait_s -
  safety.ped_red_margin_s` with the fixed greens, the actuated max greens and the
  minimum greens. The rules controller (S2, Part 4) relaxes its max greens beyond that
  static bound: its worst-case red is reported, and when it exceeds the bound the
  scenario must run the pedestrian shield (`safety.ped_forcing` detected or red), which
  bounds every wait dynamically (D21); that is a warning, not an error.

`cached_requirements` gives the audit (`sim.audit`) the intergreens and foe relations of
a net, parsed once per net file.
"""

from __future__ import annotations

import argparse
import functools
import math
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

from traffic_twin.config import ConfigError, get, load_scenario
from traffic_twin.control.pedestrian import forcing_mode
from traffic_twin.control.safety import rules_greens
from traffic_twin.sim.signal import Stage

GREEN = "Gg"
CAR_LENGTH_M = 5.0
BUS_LENGTH_M = 12.0
STOP_NEAR_LINE_M = 30.0  # a bus stop ending this close to the stop line slows the clearing bus


@dataclass(frozen=True)
class Link:
    index: int  # TLS link index
    from_lane: str
    to_lane: str
    via: str | None  # internal lane (or crossing edge) the link drives over
    is_crossing: bool


@dataclass(frozen=True)
class Program:
    tls_id: str
    durations: tuple[float, ...]
    states: tuple[str, ...]
    links: dict[int, Link]
    foes: dict[int, frozenset[int]]  # link -> conflicting links
    crossing_length: dict[int, float]  # crossing link -> length (m)

    @property
    def cycle(self) -> float:
        return float(sum(self.durations))


# -- loading ----------------------------------------------------------------------------
def _edge_functions(root: ET.Element) -> dict[str, str]:
    return {e.get("id"): e.get("function", "") for e in root.findall("edge")}


def load_program(net_file: str | Path, tls_id: str) -> Program:
    """The static program of `tls_id` in a .net.xml with its links and foe relations.

    The first program of the TLS is used. Foes come from the request table of junction
    `tls_id`, so the TLS must control exactly that one junction."""
    root = ET.parse(net_file).getroot()
    tl = next((t for t in root.findall("tlLogic") if t.get("id") == tls_id), None)
    if tl is None:
        raise ValueError(f"{net_file}: no tlLogic '{tls_id}'")
    phases = tl.findall("phase")
    durations = tuple(float(p.get("duration")) for p in phases)
    states = tuple(p.get("state") for p in phases)

    functions = _edge_functions(root)
    links: dict[int, Link] = {}
    for c in root.findall("connection"):
        if c.get("tl") != tls_id:
            continue
        i = int(c.get("linkIndex"))
        frm, to = c.get("from"), c.get("to")
        crossing = functions.get(to) == "crossing"
        links[i] = Link(
            index=i,
            from_lane=f"{frm}_{c.get('fromLane')}",
            to_lane=f"{to}_{c.get('toLane')}",
            via=to if crossing else c.get("via"),
            is_crossing=crossing,
        )
    n_links = len(states[0]) if states else 0
    if sorted(links) != list(range(n_links)) or any(len(s) != n_links for s in states):
        raise ValueError(f"{net_file}: TLS '{tls_id}' links do not match its {n_links} states")

    junction = next((j for j in root.findall("junction") if j.get("id") == tls_id), None)
    requests = [] if junction is None else junction.findall("request")
    if len(requests) != n_links:
        raise ValueError(f"{net_file}: junction '{tls_id}' must hold the {n_links} TLS links")
    foes = {
        int(r.get("index")): frozenset(k for k, ch in enumerate(r.get("foes")[::-1]) if ch == "1")
        for r in requests
    }

    lengths = {ln.get("id"): float(ln.get("length")) for ln in root.iter("lane")}
    crossing_length = {i: lengths[lk.to_lane] for i, lk in links.items() if lk.is_crossing}
    return Program(tls_id, durations, states, links, foes, crossing_length)


def link_table(prog: Program) -> list[str]:
    """One line per link: index, from lane, to lane, kind."""
    lines = []
    for i, lk in sorted(prog.links.items()):
        if lk.is_crossing:
            lines.append(f"{i:2d}  crossing {lk.via}")
        else:
            lines.append(f"{i:2d}  {lk.from_lane} -> {lk.to_lane}")
    return lines


def _label(prog: Program, i: int) -> str:
    lk = prog.links.get(i)
    if lk is None:
        return str(i)
    return f"{i}(crossing {lk.via})" if lk.is_crossing else f"{i}({lk.from_lane}->{lk.to_lane})"


# -- foe greens -------------------------------------------------------------------------
def _is_foe(prog: Program, a: int, b: int) -> bool:
    return b in prog.foes.get(a, ()) or a in prog.foes.get(b, ())


def _green_foe_pairs(prog: Program) -> Iterator[tuple[int, int, int]]:
    """(phase, a, b) with a < b: foe links green in the same phase."""
    for k, state in enumerate(prog.states):
        green = [i for i, ch in enumerate(state) if ch in GREEN]
        for x, a in enumerate(green):
            for b in green[x + 1 :]:
                if _is_foe(prog, a, b):
                    yield k, a, b


def _classify(prog: Program, k: int, a: int, b: int) -> str:
    """'G/G' or 'permissive' (vehicle-vehicle), 'ped_major' (vehicle G against a green
    crossing) or 'ped_ok' (vehicle g yields to the crossing)."""
    state = prog.states[k]
    ped_a, ped_b = prog.links[a].is_crossing, prog.links[b].is_crossing
    if not ped_a and not ped_b:
        return "G/G" if state[a] == state[b] == "G" else "permissive"
    vehicle = b if ped_a else a
    return "ped_major" if state[vehicle] == "G" else "ped_ok"


def foe_green_conflicts(prog: Program) -> list[str]:
    """Errors for vehicle-vehicle foes that are both green (G/G, or with g) in a phase,
    and for a vehicle with priority (G) across a green foe crossing."""
    errors = []
    for k, a, b in _green_foe_pairs(prog):
        kind = _classify(prog, k, a, b)
        pair = f"{_label(prog, a)} and {_label(prog, b)} ({prog.states[k][a]}{prog.states[k][b]})"
        if kind == "G/G":
            errors.append(f"phase {k}: foe vehicle links {pair} both green")
        elif kind == "permissive":
            errors.append(f"phase {k}: permissive foe vehicle links {pair}")
        elif kind == "ped_major":
            errors.append(f"phase {k}: vehicle link with priority across a green crossing {pair}")
    return errors


def permissive_crossing_pairs(prog: Program) -> list[tuple[int, int]]:
    """(vehicle, crossing) links green together where the vehicle yields (`g`)."""
    pairs = set()
    for k, a, b in _green_foe_pairs(prog):
        if _classify(prog, k, a, b) == "ped_ok":
            pairs.add((b, a) if prog.links[a].is_crossing else (a, b))
    return sorted(pairs)


# -- intergreens ------------------------------------------------------------------------
@dataclass
class _Geometry:
    shape: dict[str, list[tuple[float, float]]]
    allow: dict[str, list[str]]
    length: dict[str, float]
    next_internal: dict[str, str]  # internal lane -> internal lane that follows it


def _geometry(net_file: str | Path) -> _Geometry:
    root = ET.parse(net_file).getroot()
    shape, allow, length = {}, {}, {}
    for ln in root.iter("lane"):
        lid = ln.get("id")
        shape[lid] = [tuple(map(float, p.split(","))) for p in ln.get("shape").split()]
        allow[lid] = (ln.get("allow") or "").split()
        length[lid] = float(ln.get("length"))
    next_internal = {
        f"{c.get('from')}_{c.get('fromLane')}": c.get("via")
        for c in root.findall("connection")
        if c.get("from", "").startswith(":") and c.get("via")
    }
    return _Geometry(shape, allow, length, next_internal)


def _path(geo: _Geometry, lk: Link) -> list[tuple[float, float]]:
    """Polyline the link's traffic follows through the junction."""
    if lk.is_crossing:
        return geo.shape[lk.to_lane]
    pts: list[tuple[float, float]] = []
    lane = lk.via
    while lane:
        sh = geo.shape[lane]
        pts += sh if not pts else sh[1:]
        lane = geo.next_internal.get(lane)
    return pts


def _segment_hit(p1, p2, q1, q2) -> float | None:
    """Parameter t along p1-p2 of the intersection with q1-q2, or None."""
    d = (p2[0] - p1[0]) * (q2[1] - q1[1]) - (p2[1] - p1[1]) * (q2[0] - q1[0])
    if abs(d) < 1e-9:
        return None
    t = ((q1[0] - p1[0]) * (q2[1] - q1[1]) - (q1[1] - p1[1]) * (q2[0] - q1[0])) / d
    u = ((q1[0] - p1[0]) * (p2[1] - p1[1]) - (q1[1] - p1[1]) * (p2[0] - p1[0])) / d
    return t if 0 <= t <= 1 and 0 <= u <= 1 else None


def _conflict_distance(pa: list, pb: list) -> float:
    """Distance along path a to the first point where it meets path b (its end when the
    paths only merge)."""
    run = 0.0
    for p1, p2 in zip(pa, pa[1:], strict=False):
        seg = math.dist(p1, p2)
        hits = [_segment_hit(p1, p2, q1, q2) for q1, q2 in zip(pb, pb[1:], strict=False)]
        hits = [t for t in hits if t is not None]
        if hits:
            return run + min(hits) * seg
        run += seg
    return run


def _bus_only(geo: _Geometry, lane: str) -> bool:
    allow = geo.allow.get(lane, [])
    return "bus" in allow and "passenger" not in allow


def intergreen_requirements(
    net_file: str | Path,
    prog: Program,
    *,
    v_clear: float = 6.0,
    v_clear_from_stop: float = 4.0,
    stop_lanes: Sequence[str] = (),
    ped_speed: float = 1.2,
    min_vv: float = 5.0,
) -> dict[tuple[int, int], float]:
    """Required intergreen (s) for each (ending link, starting foe link) pair.

    A crossing must be cleared at `ped_speed`: ceil(length / ped_speed). A vehicle link
    needs max(min_vv, ceil((distance to the conflict point + vehicle length) / speed)),
    with a bus length when its from or to lane is bus-only and `v_clear_from_stop` when
    its from lane is in `stop_lanes` (a bus pulling out of a near-side stop)."""
    geo = _geometry(net_file)
    stops = set(stop_lanes)
    need: dict[tuple[int, int], float] = {}
    for a, lk in sorted(prog.links.items()):
        path_a = _path(geo, lk)
        bus = _bus_only(geo, lk.from_lane) or _bus_only(geo, lk.to_lane)
        length = BUS_LENGTH_M if bus else CAR_LENGTH_M
        speed = v_clear_from_stop if lk.from_lane in stops else v_clear
        for b in sorted(prog.foes.get(a, ())):
            if b not in prog.links:
                continue
            if lk.is_crossing:
                need[(a, b)] = float(math.ceil(prog.crossing_length[a] / ped_speed))
            else:
                dist = _conflict_distance(path_a, _path(geo, prog.links[b]))
                need[(a, b)] = float(max(min_vv, math.ceil((dist + length) / speed)))
    return need


class Requirements(NamedTuple):
    """Inputs of the audit's `intergreen_short` and `foe_vehicle_greens` (read-only)."""

    intergreen: Mapping[tuple[int, int], float]  # (ending link, starting foe) -> s
    foes: Mapping[int, frozenset[int]]  # link -> conflicting links


@functools.lru_cache(maxsize=8)
def _lane_lengths(net_file: str, mtime_ns: int) -> Mapping[str, float]:
    return MappingProxyType(dict(_geometry(net_file).length))


@functools.lru_cache(maxsize=8)
def _requirements(
    net_file: str, mtime_ns: int, tls_id: str, stops: tuple[str, ...]
) -> Requirements:
    prog = load_program(net_file, tls_id)
    req = intergreen_requirements(net_file, prog, stop_lanes=stops)
    return Requirements(MappingProxyType(req), MappingProxyType(dict(prog.foes)))


def cached_requirements(net_file: str | Path, cfg: dict) -> Requirements:
    """Intergreen requirements (with the scenario's near-side bus stops) and foe
    relations of `network.tls_id` in `net_file`, computed once per net file and
    modification time."""
    path = Path(net_file).resolve()
    mtime = path.stat().st_mtime_ns
    stops = stop_lanes(cfg, _lane_lengths(str(path), mtime))
    return _requirements(str(path), mtime, get(cfg, "network.tls_id"), tuple(stops))


def _green_edges(prog: Program, i: int) -> tuple[list[float], list[float]]:
    """Times within one cycle where link i turns green (starts) and stops being green."""
    starts, ends, t = [], [], 0.0
    n = len(prog.states)
    for k in range(n):
        now, before = prog.states[k][i] in GREEN, prog.states[k - 1][i] in GREEN
        if now and not before:
            starts.append(t)
        elif before and not now:
            ends.append(t)
        t += prog.durations[k]
    return starts, ends


def intergreen_violations(prog: Program, req: dict[tuple[int, int], float]) -> list[str]:
    """Errors where the program's intergreen is shorter than required."""
    cycle = prog.cycle
    errors = []
    for (a, b), need in sorted(req.items()):
        _, ends_a = _green_edges(prog, a)
        starts_b, _ = _green_edges(prog, b)
        if not ends_a or not starts_b:
            continue
        gap = min((s - e) % cycle for e in ends_a for s in starts_b)
        if gap < need - 1e-9:
            errors.append(
                f"intergreen {_label(prog, a)} -> {_label(prog, b)}: {gap:g} s < {need:g} s"
            )
    return errors


# -- pedestrian red ---------------------------------------------------------------------
def with_greens(prog: Program, stages: Sequence[Stage], greens: Sequence[float]) -> Program:
    """The program with each stage's green phase lasting `greens[i]`."""
    if len(greens) != len(stages):
        raise ValueError(f"{len(greens)} greens for {len(stages)} stages")
    durations = list(prog.durations)
    for stage, g in zip(stages, greens, strict=True):
        durations[stage.green_phase] = float(g)
    return replace(prog, durations=tuple(durations))


def worst_case_red(
    prog: Program, stages: Sequence[Stage], greens: Sequence[float]
) -> dict[int, float]:
    """Longest red (s) per link over a cycle, with each stage's green phase set to
    `greens[i]`. Red is any state other than G/g; a link never green gets inf."""
    p = with_greens(prog, stages, greens)
    out = {}
    for i in sorted(p.links):
        if not any(s[i] in GREEN for s in p.states):
            out[i] = math.inf
            continue
        best = run = 0.0
        for d, s in list(zip(p.durations, p.states, strict=True)) * 2:  # wrap around
            run = 0.0 if s[i] in GREEN else run + d
            best = max(best, run)
        out[i] = best
    return out


# -- scenario check ---------------------------------------------------------------------
@dataclass
class Report:
    errors: list[str]
    warnings: list[str]
    worst_crossing_red: dict[str, float]  # crossing name -> worst-case red (s), fixed greens
    ok: bool
    summary: dict[str, int] = field(default_factory=dict)  # counts printed by the CLI
    worst_crossing_red_max: dict[str, float] = field(default_factory=dict)  # actuated max greens
    crossing_bound_s: float | None = None
    worst_crossing_red_min: dict[str, float] = field(default_factory=dict)  # min greens
    worst_crossing_red_rules: dict[str, float] = field(default_factory=dict)  # rules max + TSP
    ped_forcing: str = "off"  # safety.ped_forcing
    dynamic_crossings: list[str] = field(default_factory=list)  # bounded by the shield only


def scenario_stages(cfg: dict) -> list[Stage]:
    return [
        Stage(s["name"], int(s["green_phase"]), tuple(int(c) for c in s["clearance_phases"]))
        for s in get(cfg, "network.stages")
    ]


def _greens(cfg: dict, key: str, stages: Sequence[Stage]) -> list[float] | None:
    """Per-stage greens from a {stage name: s} mapping at `key`, or None if absent."""
    by_name = get(cfg, key, None)
    if not isinstance(by_name, dict):
        return None
    missing = [s.name for s in stages if s.name not in by_name]
    if missing:
        raise ConfigError(f"{key}: no green for stage(s) {', '.join(missing)}")
    return [float(by_name[s.name]) for s in stages]


def _min_greens(cfg: dict, stages: Sequence[Stage]) -> list[float] | None:
    """Shortest greens the scenario's controller can show: its per-stage min greens
    (`rules.min_green_s` for the rules controller, else `actuated.min_green_s`), never
    below `safety.min_green_s`."""
    key = "rules.min_green_s" if cfg.get("controller") == "rules" else "actuated.min_green_s"
    per_stage = _greens(cfg, key, stages)
    floor = get(cfg, "safety.min_green_s", None)
    if per_stage is None and floor is None:
        return None
    floor = float(floor or 0.0)
    return [max(floor, g) for g in per_stage] if per_stage else [floor] * len(stages)


def crossing_names(cfg: dict, prog: Program) -> dict[int, str]:
    """Crossing link -> name: from `network.crossings` when configured, else the crossing
    edge id."""
    configured = get(cfg, "network.crossings", None) or {}
    names = {int(c["link"]): name for name, c in configured.items() if "link" in c}
    return {
        i: names.get(i, lk.via or str(i)) for i, lk in sorted(prog.links.items()) if lk.is_crossing
    }


def stop_lanes(cfg: dict, lane_length: dict[str, float]) -> list[str]:
    """Lanes with a configured bus stop ending within STOP_NEAR_LINE_M of the lane end."""
    out = []
    for stop in (get(cfg, "network.bus_stops", None) or {}).values():
        lane = stop["lane"]
        if lane in lane_length and lane_length[lane] - float(stop["end"]) <= STOP_NEAR_LINE_M:
            out.append(lane)
    return sorted(set(out))


def _stage_errors(prog: Program, stages: Sequence[Stage]) -> list[str]:
    n = len(prog.states)
    errors = []
    for s in stages:
        bad = [p for p in (s.green_phase, *s.clearance_phases) if not 0 <= p < n]
        if bad:
            errors.append(f"stage {s.name}: phase(s) {bad} not in the {n}-phase program")
    return errors


def _config_errors(cfg: dict, prog: Program) -> list[str]:
    """Crossing and signal-group settings that do not fit the program."""
    errors = []
    for name, c in (get(cfg, "network.crossings", None) or {}).items():
        lk = prog.links.get(int(c.get("link", -1)))
        if lk is None or not lk.is_crossing:
            errors.append(f"crossing {name}: link {c.get('link')} is not a crossing link")
        elif c.get("edge") and lk.via != c["edge"]:
            errors.append(f"crossing {name}: link {lk.index} crosses {lk.via}, not {c['edge']}")
    groups = get(cfg, "network.signal_groups", None)
    if groups:
        used = sorted(int(i) for links in groups.values() for i in links)
        if used != sorted(prog.links):
            errors.append(f"signal_groups must cover links 0..{len(prog.links) - 1} exactly once")
    return errors


def check_scenario(cfg: dict) -> Report:
    """All checks for a scenario, plus crossing red <= ped_max_wait_s - ped_red_margin_s
    with the fixed greens, the min greens, and the actuated max greens for an actuated
    scenario. For a rules scenario the worst-case red of its plan (max greens + TSP
    extension) is reported; above the bound it needs `safety.ped_forcing` on (error
    otherwise) and is reported as bounded dynamically by the pedestrian shield."""
    net_file = get(cfg, "network.net_file")
    prog = load_program(net_file, get(cfg, "network.tls_id"))
    stages = scenario_stages(cfg)
    errors = _stage_errors(prog, stages)
    if errors:  # the greens cannot be placed: nothing else can be checked
        return Report(errors, [], {}, False)
    errors = _config_errors(cfg, prog)
    warnings: list[str] = []

    kinds = [_classify(prog, *pair) for pair in _green_foe_pairs(prog)]
    errors += foe_green_conflicts(prog)

    geo = _geometry(net_file)
    req = intergreen_requirements(net_file, prog, stop_lanes=stop_lanes(cfg, geo.length))
    fixed = _greens(cfg, "fixed.green_s", stages)
    variants = {"as built": prog}
    if fixed is not None:
        variants["fixed greens"] = with_greens(prog, stages, fixed)
        built = [prog.durations[s.green_phase] for s in stages]
        if built != fixed:
            warnings.append(f"net program greens {built} differ from fixed.green_s {fixed}")
    minimum = _min_greens(cfg, stages)
    if minimum is not None:
        variants["min greens"] = with_greens(prog, stages, minimum)
    ig_errors: dict[str, list[str]] = {}
    for label, p in variants.items():
        for msg in intergreen_violations(p, req):
            ig_errors.setdefault(msg, []).append(label)
    errors += [f"{msg} ({', '.join(labels)})" for msg, labels in ig_errors.items()]

    names = crossing_names(cfg, prog)
    max_wait = get(cfg, "safety.ped_max_wait_s", None)
    bound = None
    if max_wait is not None:
        bound = float(max_wait) - float(get(cfg, "safety.ped_red_margin_s", 0.0) or 0.0)
    else:
        warnings.append("no safety.ped_max_wait_s: crossing red not bounded")

    def crossing_red(greens: Sequence[float], label: str) -> dict[str, float]:
        red = worst_case_red(prog, stages, greens)
        out = {names[i]: red[i] for i in names}
        if bound is not None:
            errors.extend(
                f"crossing {name}: worst-case red {r:g} s > {bound:g} s ({label})"
                for name, r in out.items()
                if r > bound + 1e-9
            )
        return out

    worst = crossing_red(fixed or [prog.durations[s.green_phase] for s in stages], "fixed greens")
    worst_max = {}
    maximum = _greens(cfg, "actuated.max_green_s", stages)
    if maximum is not None and fixed is not None:
        over = [s.name for s, f, m in zip(stages, fixed, maximum, strict=True) if f > m]
        if over:
            warnings.append(f"fixed greens above actuated.max_green_s for {', '.join(over)}")
    if cfg.get("controller") == "actuated" and maximum is not None:
        worst_max = crossing_red(maximum, "actuated max greens")
    worst_min = crossing_red(minimum, "min greens") if minimum is not None else {}

    mode = forcing_mode(cfg)
    worst_rules: dict[str, float] = {}
    dynamic: list[str] = []
    rules = rules_greens(cfg, [s.name for s in stages])
    if cfg.get("controller") == "rules" and rules is not None:
        red = worst_case_red(prog, stages, rules)
        worst_rules = {names[i]: red[i] for i in names}
        for name, r in worst_rules.items():
            if bound is None or r <= bound + 1e-9:
                continue
            if mode == "off":
                errors.append(
                    f"crossing {name}: worst-case red {r:g} s > {bound:g} s (rules max greens "
                    "+ TSP extension) with safety.ped_forcing off: the rules plan needs the "
                    "pedestrian shield"
                )
            else:
                dynamic.append(name)
                warnings.append(
                    f"crossing {name}: worst-case red {r:g} s > {bound:g} s (rules max greens "
                    f"+ TSP extension): dynamic (pedestrian shield, ped_forcing {mode})"
                )

    summary = {
        "foe_greens": kinds.count("G/G") + kinds.count("ped_major"),
        "permissive_vehicle_vehicle": kinds.count("permissive"),
        "intergreen_violations": len(ig_errors),
    }
    return Report(
        errors, warnings, worst, not errors, summary, worst_max, bound,
        worst_min, worst_rules, mode, dynamic,
    )  # fmt: skip


# -- CLI --------------------------------------------------------------------------------
def _print_report(cfg: dict, report: Report, show_links: bool) -> None:
    net_file = get(cfg, "network.net_file")
    prog = load_program(net_file, get(cfg, "network.tls_id"))
    n_cross = sum(lk.is_crossing for lk in prog.links.values())
    print(
        f"TLS {prog.tls_id} ({net_file}): {len(prog.states)} phases, cycle {prog.cycle:g} s, "
        f"{len(prog.links)} links ({n_cross} crossings)"
    )
    if show_links:
        print("\n".join(link_table(prog)))
    s = report.summary
    if s:
        print(
            f"foe greens: {s['foe_greens']} / permissive vehicle-vehicle: "
            f"{s['permissive_vehicle_vehicle']} / intergreen violations: "
            f"{s['intergreen_violations']}"
        )
    bound = "-" if report.crossing_bound_s is None else f"{report.crossing_bound_s:g}"
    columns = [("min greens (s)", report.worst_crossing_red_min),
               ("actuated max (s)", report.worst_crossing_red_max),
               ("rules max (s)", report.worst_crossing_red_rules)]  # fmt: skip
    columns = [(label, values) for label, values in columns if values]
    header = f"{'crossing':10} {'worst red fixed (s)':>20}"
    header += "".join(f" {label:>17}" for label, _ in columns)
    print(f"{header} {'bound (s)':>10}")
    for name, red in report.worst_crossing_red.items():
        row = f"{name:10} {red:20g}"
        row += "".join(f" {values[name]:17g}" for _, values in columns)
        print(f"{row} {bound:>10}")
    if report.dynamic_crossings:
        print(
            f"rules plan: crossings {', '.join(report.dynamic_crossings)} above the static "
            f"bound, dynamic (pedestrian shield, safety.ped_forcing {report.ped_forcing}): "
            "every wait stays <= the bound at run time (D21)"
        )
    for w in report.warnings:
        print("warning:", w)
    for e in report.errors:
        print("error:", e)
    print("OK" if report.ok else f"FAILED: {len(report.errors)} error(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="scenario config")
    parser.add_argument("--links", action="store_true", help="print the TLS link table")
    args = parser.parse_args(argv)
    cfg = load_scenario(args.config)
    report = check_scenario(cfg)
    _print_report(cfg, report, args.links)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

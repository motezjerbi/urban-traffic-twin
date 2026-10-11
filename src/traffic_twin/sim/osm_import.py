"""Import the junction from OpenStreetMap into curated plain XML (run once; the output
is versioned in sumo/network/<site>/ and built into the .net.xml by sim.network).

    python -m traffic_twin.sim.osm_import --config configs/pitie_fixed.yaml [--out DIR]

Stage 1 (`osm_to_raw`): netconvert crops the OSM extract around `network.osm.centre`,
keeps car and bus roads, drops side streets and service roads, and joins the signal
nodes within `join_dist_m` into one TLS node. Its plain XML still uses OSM ids.

Stage 2 (`curate`): driven by the topology, not by OSM ids. Every edge chain from the
TLS node to a dead end is one leg direction, named by its compass direction (n/e/s/w)
plus "b" when it is bus-only; each chain becomes one `<leg>_in` / `<leg>_out` edge with
the lanes it has at the junction, and the TLS node becomes `C`. Lane-to-lane connections
are explicit (from `network.osm.connections`, never U-turns), crossings are explicit
(`network.osm.crossings`), emergency vehicles may use every bus-only lane, and the
`lane_use` variant may override lane permissions. OSM provenance is kept as params.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import quoteattr, unescape

from traffic_twin.config import ConfigError, get, load_config
from traffic_twin.sim.network import netconvert_env
from traffic_twin.sim.sumo import find_binary

CENTRE = "C"  # id of the junction and of its TLS
EARTH_M_PER_DEG = 111_320.0

# Stage 1: keep car and bus roads; drop residential side streets (Rue Poliveau, Rue des
# Wallons), the hospital's psv service roads, isolated bits and signals that are not
# part of the joined junction.
RAW_OPTIONS = [
    "--keep-edges.by-vclass", "passenger,bus",
    "--remove-edges.by-type", "highway.residential,highway.service|psv",
    "--remove-edges.isolated", "--keep-edges.components", "1",
    "--tls.discard-simple", "--junctions.join",
    "--geometry.remove", "--ramps.guess", "false", "--no-turnarounds",
    "--osm.lane-access", "--osm.turn-lanes",
    "--output.street-names", "--output.original-names",
    "--osm.stop-output.length.bus", "20",
]  # fmt: skip
RAW = "raw"  # file prefix of the stage 1 output
PTSTOPS = "osm_ptstops.add.xml"


# -- stage 1 ----------------------------------------------------------------------------
def geo_box(centre: list[float], extent_m: dict[str, float]) -> str:
    """`west,south,east,north` (lon/lat) around `centre` = [lat, lon] for netconvert."""
    lat, lon = (float(v) for v in centre)
    k = EARTH_M_PER_DEG * math.cos(math.radians(lat))
    return (
        f"{lon - extent_m['w'] / k:.6f},{lat - extent_m['s'] / EARTH_M_PER_DEG:.6f},"
        f"{lon + extent_m['e'] / k:.6f},{lat + extent_m['n'] / EARTH_M_PER_DEG:.6f}"
    )


def raw_command(osm_cfg: dict, raw_dir: Path) -> list[str]:
    return [
        find_binary("netconvert"),
        "--osm-files", Path(osm_cfg["file"]).as_posix(),
        "--keep-edges.in-geo-boundary", geo_box(osm_cfg["centre"], osm_cfg["extent_m"]),
        "--junctions.join-dist", str(osm_cfg["join_dist_m"]),
        *RAW_OPTIONS,
        "--ptstop-output", (raw_dir / PTSTOPS).as_posix(),
        "--plain-output-prefix", (raw_dir / RAW).as_posix(),
        "-o", (raw_dir / f"{RAW}.net.xml").as_posix(),
    ]  # fmt: skip


def osm_to_raw(cfg: dict, raw_dir: Path) -> Path:
    """Run stage 1 into `raw_dir`; returns the plain-XML prefix (raw_dir / "raw")."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = raw_command(get(cfg, "network.osm"), raw_dir)
    run = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=netconvert_env(),
    )
    if run.returncode:
        raise RuntimeError(f"netconvert failed ({run.returncode}):\n{run.stdout}{run.stderr}")
    (raw_dir / "stage1.log").write_text(run.stderr, encoding="utf-8")
    return raw_dir / RAW


# -- stage 2 ----------------------------------------------------------------------------
@dataclass(frozen=True)
class CuratedEdge:
    id: str  # <leg>_in or <leg>_out
    leg: str
    raw_edges: tuple[str, ...]  # OSM-id edges merged into it, in driving order
    bearing: float  # compass bearing (deg) from C to the leg's far end
    length: float  # plain shape length (m)
    num_lanes: int


def comment(text: str) -> str:
    """An XML comment; "--" may not appear inside one (netconvert stops on it)."""
    while "--" in text:
        text = text.replace("--", "- -")
    return f"<!-- {text.rstrip('-')} -->"


def _shape(edge: ET.Element, nodes: dict[str, ET.Element]) -> list[tuple[float, float]]:
    if edge.get("shape"):
        return [tuple(map(float, p.split(","))) for p in edge.get("shape").split()]
    a, b = nodes[edge.get("from")], nodes[edge.get("to")]
    return [(float(a.get("x")), float(a.get("y"))), (float(b.get("x")), float(b.get("y")))]


def _bus_only(allow: str | None) -> bool:
    words = (allow or "").split()
    return "bus" in words and "passenger" not in words


def _with_emergency(allow: str) -> str:
    return allow if "emergency" in allow.split() else f"{allow} emergency"


def _chain(edge: ET.Element, outward: bool, nodes: dict, out_of: dict, into: dict) -> list:
    """Edges from the centre to a dead end (outward) or from a dead end to the centre,
    in driving order, following the street (never turning back)."""
    seq = [edge]
    while True:
        node = seq[-1].get("to") if outward else seq[0].get("from")
        back = seq[-1].get("from") if outward else seq[0].get("to")
        cands = [
            c
            for c in (out_of if outward else into).get(node, [])
            if (c.get("to") if outward else c.get("from")) != back
        ]
        if nodes[node].get("type") == "dead_end":
            return seq
        if len(cands) != 1:
            raise ConfigError(
                f"OSM import: the street through node {node} branches; crop or filter the "
                "extract so that every leg is a single chain to a dead end"
            )
        seq = [*seq, cands[0]] if outward else [cands[0], *seq]


def _connections(osm_cfg: dict) -> tuple[list[tuple[str, int, str, int]], dict]:
    """Connections and lane permission overrides of the selected `lane_use` variant."""
    conns, allow = osm_cfg.get("connections") or {}, osm_cfg.get("lane_allow") or {}
    variants = (set(conns) | set(allow)) - {"all"}
    lane_use = osm_cfg.get("lane_use")
    if variants and lane_use not in variants:
        raise ConfigError(f"network.osm.lane_use '{lane_use}' not in {sorted(variants)}")
    chosen = [*conns.get("all", []), *conns.get(lane_use, [])]
    if not chosen:
        raise ConfigError("network.osm.connections: no lane-to-lane connections given")
    out = []
    for c in chosen:
        if not (isinstance(c, list) and len(c) == 4):
            raise ConfigError(f"network.osm.connections: {c} is not [from, lane, to, lane]")
        out.append((str(c[0]), int(c[1]), str(c[2]), int(c[3])))
    return out, allow.get(lane_use, {}) or {}


def curate(raw_prefix: str | Path, out_prefix: str | Path, osm_cfg: dict) -> list[CuratedEdge]:
    """Turn the stage 1 plain XML (OSM ids) into semantic plain XML: one node C (the
    TLS), one dead-end node per leg, one in-edge and one out-edge per leg. Writes
    `<out_prefix>.{nod,edg,con}.xml` and returns the curated edges."""
    raw_prefix, out_prefix = str(raw_prefix), Path(out_prefix)
    nod_root = ET.parse(f"{raw_prefix}.nod.xml").getroot()
    edg_root = ET.parse(f"{raw_prefix}.edg.xml").getroot()
    nodes = {n.get("id"): n for n in nod_root.findall("node")}
    edges = edg_root.findall("edge")
    tls = [n for n in nodes.values() if n.get("type") == "traffic_light"]
    if len(tls) != 1:
        raise ConfigError(f"OSM import: expected one signalised junction, found {len(tls)}")
    centre = tls[0]
    cid, cx, cy = centre.get("id"), float(centre.get("x")), float(centre.get("y"))
    out_of: dict[str, list] = {}
    into: dict[str, list] = {}
    for e in edges:
        out_of.setdefault(e.get("from"), []).append(e)
        into.setdefault(e.get("to"), []).append(e)

    legs: dict[str, dict[str, tuple[list, float, str]]] = {}
    for kind, starts in (("out", out_of.get(cid, [])), ("in", into.get(cid, []))):
        for e in starts:
            seq = _chain(e, kind == "out", nodes, out_of, into)
            end = nodes[seq[-1].get("to")] if kind == "out" else nodes[seq[0].get("from")]
            brg = math.degrees(math.atan2(float(end.get("x")) - cx, float(end.get("y")) - cy))
            brg %= 360
            name = "nesw"[round(brg / 90) % 4] + ("b" if _bus_only(e.get("allow")) else "")
            if kind in legs.get(name, {}):
                raise ConfigError(f"OSM import: two '{name}' legs ({kind}); cannot name them")
            legs.setdefault(name, {})[kind] = (seq, brg, end.get("id"))

    connections, lane_allow = _connections(osm_cfg)
    curated: list[CuratedEdge] = []
    leg_nodes: dict[str, ET.Element] = {}
    edge_xml = []
    for leg, d in sorted(legs.items()):
        ends = {v[2] for v in d.values()}
        if len(ends) != 1:
            raise ConfigError(f"OSM import: leg '{leg}' in and out end at different nodes")
        leg_nodes[leg.upper()] = nodes[ends.pop()]
        for kind in ("in", "out"):
            if kind not in d:
                continue
            seq, brg, _ = d[kind]
            eid = f"{leg}_{kind}"
            shape: list[tuple[float, float]] = []
            for e in seq:
                s = _shape(e, nodes)
                shape += s if not shape else s[1:]
            src = seq[-1] if kind == "in" else seq[0]  # the edge at the junction
            length = sum(math.dist(a, b) for a, b in zip(shape, shape[1:], strict=False))
            raw_ids = tuple(e.get("id") for e in seq)
            curated.append(
                CuratedEdge(eid, leg, raw_ids, round(brg, 1), length, int(src.get("numLanes")))
            )
            edge_xml.append(
                _edge_xml(eid, leg.upper(), kind, src, shape, raw_ids, lane_allow.get(eid, {}))
            )

    edge_ids = {e.id: e for e in curated}
    _check_connections(connections, edge_ids)
    for eid, by_lane in lane_allow.items():
        if eid not in edge_ids or any(not 0 <= int(i) < edge_ids[eid].num_lanes for i in by_lane):
            raise ConfigError(f"network.osm.lane_allow: no lane(s) {list(by_lane)} on '{eid}'")
    crossings = []
    for leg in osm_cfg.get("crossings") or []:
        pair = [f"{leg}_out", f"{leg}_in"]
        if not any(p in edge_ids for p in pair):
            raise ConfigError(f"network.osm.crossings: no leg '{leg}' in the OSM import")
        crossings.append(" ".join(p for p in pair if p in edge_ids))

    src_note = comment(
        f"Generated from {Path(osm_cfg['file']).as_posix()} by traffic_twin.sim.osm_import."
    )
    loc = ET.tostring(nod_root.find("location"), encoding="unicode").strip()
    nod = [f"    {loc}"]
    nod.append(
        f'    <node id="{CENTRE}" x="{cx:.2f}" y="{cy:.2f}" type="traffic_light" '
        f'tl="{CENTRE}"><param key="osm_node" value={quoteattr(cid)}/></node>'
    )
    for nid, n in leg_nodes.items():
        nod.append(
            f'    <node id="{nid}" x="{n.get("x")}" y="{n.get("y")}" type="dead_end">'
            f'<param key="osm_node" value={quoteattr(n.get("id"))}/></node>'
        )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write(f"{out_prefix}.nod.xml", [src_note, "<nodes>", *nod, "</nodes>"])
    _write(f"{out_prefix}.edg.xml", [src_note, "<edges>", f"    {loc}", *edge_xml, "</edges>"])

    rank = {e.id: k for k, e in enumerate(curated)}
    con = [
        f'    <connection from="{f}" to="{t}" fromLane="{fl}" toLane="{tl}"/>'
        for f, fl, t, tl in sorted(connections, key=lambda c: (rank[c[0]], c[1], rank[c[2]], c[3]))
    ]
    con += [f'    <crossing node="{CENTRE}" edges="{c}" priority="true"/>' for c in crossings]
    note = comment(
        f"Lane-to-lane connections at {CENTRE} (lane use '{osm_cfg.get('lane_use')}', "
        "network.osm.connections; lane indices before sidewalks are added). No U-turns. "
        "Explicit crossings (network.osm.crossings)."
    )
    _write(f"{out_prefix}.con.xml", [src_note, note, "<connections>", *con, "</connections>"])
    return curated


def _edge_xml(eid, node_id, kind, src, shape, raw_ids, lane_allow) -> str:
    attrs = {
        "id": eid,
        "from": node_id if kind == "in" else CENTRE,
        "to": CENTRE if kind == "in" else node_id,
        "name": unescape(src.get("name") or "", {"&apos;": "'", "&quot;": '"'}),
        "priority": src.get("priority"),
        "numLanes": src.get("numLanes"),
        "speed": src.get("speed"),
        "shape": " ".join(f"{x:.2f},{y:.2f}" for x, y in shape),
    }
    for k in ("allow", "disallow"):
        if src.get(k):
            attrs[k] = src.get(k)
    if _bus_only(attrs.get("allow")):
        attrs["allow"] = _with_emergency(attrs["allow"])
    lanes = []
    for ln in src.findall("lane"):
        la = {k: v for k, v in ln.attrib.items() if k in ("index", "allow", "disallow")}
        override = lane_allow.get(int(la["index"]), lane_allow.get(str(la["index"])))
        if override:
            la = {"index": la["index"], "allow": str(override)}
        elif _bus_only(la.get("allow")):
            la["allow"] = _with_emergency(la["allow"])
        if len(la) > 1:
            lanes.append(
                "        <lane " + " ".join(f"{k}={quoteattr(v)}" for k, v in la.items()) + "/>"
            )
    head = "    <edge " + " ".join(f"{k}={quoteattr(v)}" for k, v in attrs.items() if v)
    param = f'        <param key="osm_edges" value={quoteattr(" ".join(raw_ids))}/>'
    return head + ">\n" + "\n".join([param, *lanes]) + "\n    </edge>"


def _check_connections(connections: list, edges: dict[str, CuratedEdge]) -> None:
    problems = []
    for f, fl, t, tl in connections:
        if f not in edges or not f.endswith("_in") or t not in edges or not t.endswith("_out"):
            problems.append(f"{f}:{fl} -> {t}:{tl}: unknown edge (expects <leg>_in -> <leg>_out)")
            continue
        if edges[f].leg == edges[t].leg:
            problems.append(f"{f}:{fl} -> {t}:{tl}: U-turn")
        if not 0 <= fl < edges[f].num_lanes or not 0 <= tl < edges[t].num_lanes:
            problems.append(f"{f}:{fl} -> {t}:{tl}: lane index out of range")
    served = {f for f, *_ in connections}
    problems += [f"{e}: no connection" for e in edges if e.endswith("_in") and e not in served]
    if problems:
        raise ConfigError("network.osm.connections: " + "; ".join(problems))


def _write(path: str, lines: list[str]) -> None:
    text = "<?xml version='1.0' encoding='UTF-8'?>\n" + "\n".join(lines) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


# -- OSM bus stops on the built net -----------------------------------------------------
def project_ptstops(raw_dir: Path, net_file: str | Path) -> list[dict]:
    """OSM bus stops (with lines) of a stage 1 run, projected onto the bus lanes of a
    curated net built in the same coordinates: lane, start and end (m), lines."""
    import sumolib

    raw = sumolib.net.readNet(str(raw_dir / f"{RAW}.net.xml"))
    net = sumolib.net.readNet(str(net_file))
    lanes = [
        ln
        for e in net.getEdges()
        if e.getFunction() == "" and e.getID().endswith(("_in", "_out"))
        for ln in e.getLanes()
        if ln.allows("bus") and not ln.allows("pedestrian")
    ]
    out = []
    for stop in ET.parse(raw_dir / PTSTOPS).getroot().findall("busStop"):
        if not stop.get("lines"):
            continue  # netconvert's copy for the opposite direction
        rl = raw.getLane(stop.get("lane"))
        scale = sumolib.geomhelper.polyLength(rl.getShape()) / rl.getLength()
        pts = [
            sumolib.geomhelper.positionAtShapeOffset(rl.getShape(), float(stop.get(k)) * scale)
            for k in ("startPos", "endPos")
        ]
        mid = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
        lane = min(lanes, key=lambda ln: ln.getClosestLanePosAndDist(mid)[1])
        start, end = (lane.getClosestLanePosAndDist(p)[0] for p in pts)
        out.append(
            {
                "osm_id": stop.get("id"),
                "name": stop.get("name"),
                "lines": stop.get("lines"),
                "lane": lane.getID(),
                "start": round(start, 1),
                "end": round(end, 1),
                "lane_length": round(lane.getLength(), 1),
            }
        )
    return sorted(out, key=lambda s: (s["lane"], s["start"]))


# -- CLI --------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="scenario config")
    parser.add_argument(
        "--out", type=Path, help="output folder (default: folder of network.build.plain_prefix)"
    )
    parser.add_argument("--raw-dir", type=Path, help="keep the stage 1 output in this folder")
    parser.add_argument(
        "--stops", action="store_true", help="print the OSM bus stops on network.net_file"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config, ("network.osm.file", "network.build.plain_prefix"))
    plain_prefix = Path(get(cfg, "network.build.plain_prefix"))
    out_prefix = (args.out / plain_prefix.name) if args.out else plain_prefix

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = args.raw_dir or Path(tmp)
        raw_prefix = osm_to_raw(cfg, raw_dir)
        curated = curate(raw_prefix, out_prefix, get(cfg, "network.osm"))
        print(f"{'edge':8} {'brg':>5} {'len (m)':>8} {'lanes':>5}  OSM edges")
        for e in curated:
            osm = " ".join(e.raw_edges)
            print(f"{e.id:8} {e.bearing:5.0f} {e.length:8.1f} {e.num_lanes:5d}  {osm}")
        print(f"wrote {out_prefix}.nod.xml, .edg.xml, .con.xml")
        if args.stops:
            print("OSM bus stops on", get(cfg, "network.net_file"))
            for s in project_ptstops(raw_dir, get(cfg, "network.net_file")):
                print(
                    f"  {s['lane']:10} {s['start']:7.1f} {s['end']:7.1f} (lane {s['lane_length']}) "
                    f"lines {s['lines']:12} OSM {s['osm_id']}"
                )


if __name__ == "__main__":
    main()

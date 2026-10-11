"""OSM import (sim.osm_import): stage 2 curation on a tiny synthetic stage 1 output,
and the full import from sumo/network/osm/pitie.osm against the committed plain XML."""

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from traffic_twin.config import ConfigError
from traffic_twin.sim import osm_import
from traffic_twin.sim.osm_import import comment, curate, geo_box

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCATION = (
    '<location netOffset="0.00,0.00" convBoundary="0.00,0.00,300.00,300.00" '
    'origBoundary="0.00,0.00,1.00,1.00" projParameter="!"/>'
)
# Stage 1 output with OSM-like ids: a two-way general street N-S through the signal
# cluster (the S side split at a priority node, a bus-only lane at the junction) and a
# two-way bus-only road to the east.
RAW_NODES = f"""<nodes>
    {LOCATION}
    <node id="10" x="100.00" y="300.00" type="dead_end"/>
    <node id="20" x="95.00" y="40.00" type="priority"/>
    <node id="21" x="90.00" y="-100.00" type="dead_end"/>
    <node id="30" x="300.00" y="110.00" type="dead_end"/>
    <node id="cluster_1_2" x="100.00" y="100.00" type="traffic_light" tl="cluster_1_2"/>
</nodes>"""
NAME = 'name="Rue de l&amp;apos;Exemple"'  # double-escaped, as netconvert writes it
GENERAL = 'priority="12" type="highway.primary" speed="8.33" disallow="tram"'
BUS = 'priority="1" type="highway.service" speed="8.33" allow="bus bicycle"'
RAW_EDGES = f"""<edges>
    {LOCATION}
    <edge id="-100#1" from="10" to="cluster_1_2" {NAME} numLanes="2" {GENERAL}/>
    <edge id="100#0" from="cluster_1_2" to="10" {NAME} numLanes="2" {GENERAL}/>
    <edge id="200#0" from="21" to="20" numLanes="1" {GENERAL}/>
    <edge id="200#1" from="20" to="cluster_1_2" numLanes="2" priority="12" speed="8.33">
        <lane index="0" allow="bus bicycle"/>
        <lane index="1" disallow="pedestrian tram"/>
    </edge>
    <edge id="-200#1" from="cluster_1_2" to="20" numLanes="1" {GENERAL}/>
    <edge id="-200#0" from="20" to="21" numLanes="1" {GENERAL}/>
    <edge id="300" from="cluster_1_2" to="30" numLanes="1" {BUS}/>
    <edge id="-300" from="30" to="cluster_1_2" numLanes="1" {BUS}/>
</edges>"""
OSM_CFG = {
    "file": "sumo/network/osm/x.osm",
    "lane_use": "provisional",
    "crossings": ["n", "s"],
    "connections": {
        "all": [
            ["n_in", 0, "s_out", 0],
            ["n_in", 1, "s_out", 0],
            ["s_in", 1, "n_out", 1],
            ["s_in", 0, "eb_out", 0],
            ["eb_in", 0, "s_out", 0],
        ],
        "provisional": [["s_in", 0, "n_out", 0]],
        "osm_strict": [["s_in", 1, "n_out", 0]],
    },  # fmt: skip
    "lane_allow": {"provisional": {"s_in": {0: "bus bicycle passenger emergency"}}},
}


@pytest.fixture
def raw_prefix(tmp_path) -> Path:
    (tmp_path / "raw.nod.xml").write_text(RAW_NODES, encoding="utf-8")
    (tmp_path / "raw.edg.xml").write_text(RAW_EDGES, encoding="utf-8")
    return tmp_path / "raw"


def _curate(raw_prefix, osm_cfg=OSM_CFG):
    out = raw_prefix.parent / "out" / "site"
    edges = curate(raw_prefix, out, osm_cfg)
    roots = {k: ET.parse(f"{out}.{k}.xml").getroot() for k in ("nod", "edg", "con")}
    return edges, roots


def test_legs_get_semantic_ids_from_the_topology(raw_prefix):
    edges, roots = _curate(raw_prefix)
    assert [(e.id, e.raw_edges) for e in edges] == [
        ("eb_in", ("-300",)), ("eb_out", ("300",)),
        ("n_in", ("-100#1",)), ("n_out", ("100#0",)),
        ("s_in", ("200#0", "200#1")), ("s_out", ("-200#1", "-200#0")),
    ]  # fmt: skip
    nodes = {n.get("id"): n for n in roots["nod"].findall("node")}
    assert set(nodes) == {"C", "EB", "N", "S"}
    assert nodes["C"].get("tl") == "C" and nodes["C"].get("type") == "traffic_light"
    assert nodes["C"].find("param").get("value") == "cluster_1_2"  # OSM provenance
    assert roots["nod"].find("location") is not None


def test_chains_are_merged_with_the_lanes_at_the_junction(raw_prefix):
    _, roots = _curate(raw_prefix)
    edges = {e.get("id"): e for e in roots["edg"].findall("edge")}
    s_in = edges["s_in"]
    assert (s_in.get("from"), s_in.get("to"), s_in.get("numLanes")) == ("S", "C", "2")
    assert s_in.get("shape") == "90.00,-100.00 95.00,40.00 100.00,100.00"
    assert s_in.find("param").get("value") == "200#0 200#1"
    lanes = {ln.get("index"): ln.attrib for ln in s_in.findall("lane")}
    assert lanes["0"] == {"index": "0", "allow": "bus bicycle passenger emergency"}  # provisional
    assert edges["n_in"].get("name") == "Rue de l'Exemple"  # netconvert's double escape undone
    assert edges["s_out"].get("numLanes") == "1"
    assert all(edges[e].get("allow") == "bus bicycle emergency" for e in ("eb_in", "eb_out"))


def test_connections_are_explicit_without_u_turns_and_crossings_explicit(raw_prefix):
    _, roots = _curate(raw_prefix)
    conns = [
        (c.get("from"), c.get("fromLane"), c.get("to"), c.get("toLane"))
        for c in roots["con"].findall("connection")
    ]
    assert len(conns) == 6 and ("s_in", "0", "n_out", "0") in conns
    assert all(f.rsplit("_", 1)[0] != t.rsplit("_", 1)[0] for f, _, t, _ in conns)
    crossings = [(c.get("node"), c.get("edges")) for c in roots["con"].findall("crossing")]
    assert crossings == [("C", "n_out n_in"), ("C", "s_out s_in")]


def test_osm_strict_keeps_the_bus_lane(raw_prefix):
    cfg = {**OSM_CFG, "lane_use": "osm_strict"}
    _, roots = _curate(raw_prefix, cfg)
    s_in = next(e for e in roots["edg"].findall("edge") if e.get("id") == "s_in")
    assert s_in.find("lane[@index='0']").get("allow") == "bus bicycle emergency"
    conns = {(c.get("from"), c.get("fromLane"), c.get("to")) for c in roots["con"]}
    assert ("s_in", "1", "n_out") in conns and ("s_in", "0", "n_out") not in conns


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: c["connections"]["all"].append(["n_in", 0, "n_out", 0]), "U-turn"),
        (lambda c: c["connections"]["all"].append(["n_in", 2, "s_out", 0]), "out of range"),
        (lambda c: c["connections"]["all"].append(["x_in", 0, "s_out", 0]), "unknown edge"),
        (
            lambda c: c["connections"].update(all=c["connections"]["all"][:-1]),
            "eb_in: no connection",
        ),
        (lambda c: c.update(lane_use="other"), "lane_use"),
        (lambda c: c.update(crossings=["w"]), "no leg 'w'"),
        (lambda c: c["lane_allow"]["provisional"].update(n_in={2: "bus"}), "lane_allow"),
    ],
)
def test_bad_osm_settings_are_rejected(raw_prefix, change, message):
    cfg = copy.deepcopy(OSM_CFG)
    change(cfg)
    with pytest.raises(ConfigError, match=message):
        curate(raw_prefix, raw_prefix.parent / "out" / "site", cfg)


def test_a_branching_leg_is_rejected(raw_prefix):
    extra = '<edge id="400" from="20" to="10" numLanes="1" priority="1" speed="8.33"/>\n</edges>'
    raw_prefix.with_suffix(".edg.xml").write_text(RAW_EDGES.replace("</edges>", extra), "utf-8")
    with pytest.raises(ConfigError, match="branches"):
        curate(raw_prefix, raw_prefix.parent / "out" / "site", OSM_CFG)


def test_comments_never_contain_a_double_hyphen():
    text = comment("built with --sidewalks.guess ---")
    assert text.startswith("<!-- ") and text.endswith(" -->")
    assert "--" not in text[4:-3]
    ET.fromstring(f"<a>{text}</a>")  # well-formed


def test_crop_box_around_the_centre():
    box = geo_box([48.83992, 2.36172], {"n": 180, "s": 100, "w": 180, "e": 130})
    assert box == "2.359263,48.839022,2.363494,48.841537"
    w, s, e, n = map(float, box.split(","))
    assert math.isclose((n - 48.83992) * 111_320, 180, rel_tol=1e-3)


def test_stage1_command(monkeypatch, pitie_scenario, tmp_path):
    monkeypatch.setattr(osm_import, "find_binary", lambda name: name)
    cmd = osm_import.raw_command(pitie_scenario["network"]["osm"], tmp_path)
    flags = dict(zip(cmd[1::2], cmd[2::2], strict=False))
    assert flags["--osm-files"] == "sumo/network/osm/pitie.osm"
    assert flags["--keep-edges.in-geo-boundary"] == "2.359263,48.839022,2.363494,48.841537"
    assert flags["--junctions.join-dist"] == "20"
    assert "--junctions.join" in cmd and "--tls.discard-simple" in cmd
    assert cmd[-1] == (tmp_path / "raw.net.xml").as_posix()


# -- full import (netconvert) --------------------------------------------------------
def _plain(prefix: Path) -> dict:
    """Structure of a plain XML set, shapes as point lists."""
    nod = ET.parse(f"{prefix}.nod.xml").getroot()
    edg = ET.parse(f"{prefix}.edg.xml").getroot()
    con = ET.parse(f"{prefix}.con.xml").getroot()

    def points(shape):
        return [tuple(map(float, p.split(","))) for p in shape.split()]

    return {
        "nodes": {
            n.get("id"): (n.get("type"), n.find("param").get("value")) for n in nod.findall("node")
        },
        "xy": {n.get("id"): (float(n.get("x")), float(n.get("y"))) for n in nod.findall("node")},
        "edges": {
            e.get("id"): {
                **{k: e.get(k) for k in ("from", "to", "numLanes", "allow", "disallow", "name")},
                "lanes": sorted(tuple(sorted(ln.attrib.items())) for ln in e.findall("lane")),
                "osm": e.find("param").get("value"),
                "shape": points(e.get("shape")),
            }
            for e in edg.findall("edge")
        },
        "connections": sorted(tuple(sorted(c.attrib.items())) for c in con),
    }


@pytest.mark.sumo
def test_full_import_reproduces_the_committed_plain_xml(pitie_scenario, tmp_path):
    sumolib = pytest.importorskip("sumolib")
    if not sumolib.version.gitDescribe().startswith("v1.28."):
        pytest.skip("netconvert output is pinned to SUMO 1.28")
    cfg = copy.deepcopy(pitie_scenario)
    osm = cfg["network"]["osm"]
    osm["file"] = str(REPO_ROOT / osm["file"])
    raw = osm_import.osm_to_raw(cfg, tmp_path / "raw")
    curate(raw, tmp_path / "out" / "pitie", osm)

    got = _plain(tmp_path / "out" / "pitie")
    want = _plain(REPO_ROOT / pitie_scenario["network"]["build"]["plain_prefix"])
    assert got["nodes"] == want["nodes"]  # ids, types, OSM provenance
    assert all(math.dist(got["xy"][n], xy) <= 0.05 for n, xy in want["xy"].items())
    assert got["connections"] == want["connections"]
    assert set(got["edges"]) == set(want["edges"])
    for eid, w in want["edges"].items():
        g = got["edges"][eid]
        assert {k: v for k, v in g.items() if k != "shape"} == {
            k: v for k, v in w.items() if k != "shape"
        }, eid
        assert len(g["shape"]) == len(w["shape"]), eid
        assert all(math.dist(a, b) <= 0.05 for a, b in zip(g["shape"], w["shape"], strict=True))

"""Site-visit measurements (`data/field/pitie_measurements.yaml`, protocol in
docs/field_protocol.md): loading, schema checks and an inventory of what is measured.

    python -m traffic_twin.validation.field --check [--file F] --scenario configs/pitie_fixed.yaml

Rules (Part 3, D18): a missing file or a null value is NOT_MEASURED and never an error.
Schema problems (unknown signal group, approach, crossing or bus line names, bad clock
times, negative values, a time outside its session, unknown keys) raise ONE ConfigError
listing all of them. Stage names in `signal.stage_order` may be new (the site may run a
different stage structure): they are reported, not rejected. A head's `green_s` is a list
(one value per green), or a mapping stage -> list for a head green more than once per
cycle: a head the model shows green in several stages (ped_NB: HOP and STM) must use the
mapping, so its greens are never averaged together.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

import yaml

from traffic_twin.config import ConfigError, load_scenario
from traffic_twin.paths import FIELD_FILE, PITIE_FIXED_CONFIG
from traffic_twin.validation import clock_s, clock_str

SCHEMA_VERSION = 1
SIGNAL_SOURCES = ("field", "assumption")
QUEUE_KINDS = ("cycle_max", "instant")
HEAD_KEYS = ("green_s", "yellow_s", "all_red_after_s", "flashing_s")
SECTIONS = (
    "signal", "geometry", "queues", "counts", "bus_stops", "saturation", "loops", "ambulance",
)  # fmt: skip
TOP_KEYS = {"schema_version", "site", "sessions", *SECTIONS}
SIGNAL_KEYS = {
    "source", "session", "cycles_timed", "cycle_s", "stage_order", "varies_between_cycles",
    "bus_stages_on_demand", "ped_only_stage", "heads",
}  # fmt: skip
# Keys that say where / when / how something was measured, not what was measured
META_KEYS = {
    "session", "source", "kind", "lanes", "notes", "approach", "begin", "end", "line",
    "direction", "lane", "time", "iu_ac", "visible_m", "id",
}  # fmt: skip


@dataclass(frozen=True)
class Session:
    id: str
    date: str | None
    start_s: float  # seconds since midnight
    end_s: float


@dataclass
class FieldData:
    """A loaded field file. Every accessor returns empty values when the file is absent."""

    path: Path
    exists: bool = False
    raw: dict = field(default_factory=dict)
    sessions: dict[str, Session] = field(default_factory=dict)

    @property
    def signal(self) -> dict:
        return self.raw.get("signal") or {}

    @property
    def signal_source(self) -> str:
        return self.signal.get("source") or "field"

    @property
    def queues(self) -> list[dict]:
        return [q for q in self.raw.get("queues") or [] if isinstance(q, dict)]

    @property
    def counts(self) -> list[dict]:
        return [c for c in self.raw.get("counts") or [] if isinstance(c, dict)]

    def head_raw(self, group: str, key: str) -> Any:
        """The value of one signal head item as written in the file (None if absent)."""
        heads = self.signal.get("heads") or {}
        return (heads.get(group) or {}).get(key)

    def head_values(self, group: str, key: str) -> list[float]:
        """Measured values of one signal head item (e.g. N_straight, green_s); [] if not
        measured. A per-stage mapping is pooled (signal validation reads it per stage)."""
        raw = self.head_raw(group, key)
        if isinstance(raw, dict):
            return [x for v in raw.values() for x in numbers(v)]
        return numbers(raw)

    def cycle_values(self) -> list[float]:
        return numbers(self.signal.get("cycle_s"))


def numbers(value: Any) -> list[float]:
    """A null, a number or a list of numbers (nulls dropped) as a list of floats."""
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [float(v) for v in values if v is not None]


def numbers_or_none(value: Any) -> list[Any]:
    """The non-null entries of a value or list, unconverted (for schema checks)."""
    if value is None:
        return []
    return [v for v in (value if isinstance(value, list) else [value]) if v is not None]


def multi_green_message(group: str, stages: list[str]) -> str:
    example = ", ".join(f"{s}: [...]" for s in stages)
    return (
        f"the model shows {group} green {len(stages)} times per cycle ({', '.join(stages)}): "
        f"record one list per stage, e.g. {{{example}}}"
    )


def multi_green_heads(cfg: dict) -> dict[str, list[str]]:
    """Heads the model's program shows green more than once per cycle, with their stages
    (from the net's tlLogic); {} when the net or its program cannot be read."""
    net = cfg.get("network", {}) or {}
    path = net.get("net_file")
    if not path or not net.get("stages") or not Path(path).is_file():
        return {}
    from traffic_twin.validation.signal import head_stages, read_program

    try:
        states, durations = read_program(path, net.get("tls_id"))
    except (ConfigError, OSError, ET.ParseError):
        return {}
    groups = {k: list(v) for k, v in (net.get("signal_groups") or {}).items()}
    tags = head_stages(states, durations, groups, net["stages"], net.get("reference_group"))
    return {k: v for k, v in tags.items() if len(v) > 1}


# -- schema checks ---------------------------------------------------------------------
class _Checker:
    """Collects every schema problem, so one ConfigError can list them all."""

    def __init__(self, cfg: dict):
        net = cfg.get("network", {}) or {}
        self.tls_id = net.get("tls_id")
        self.groups = set(net.get("signal_groups", {}) or {})
        self.approaches = set(net.get("approaches", {}) or {})
        self.crossings = set(net.get("crossings", {}) or {})
        self.stages = [s["name"] for s in net.get("stages", []) or []]
        bus = (cfg.get("demand", {}) or {}).get("bus", {}) or {}
        self.bus_lines = {str(b["id"]) for b in bus.get("lines", []) or [] if "id" in b}
        self.multi = multi_green_heads(cfg)
        self.errors: list[str] = []
        self.sessions: dict[str, Session] = {}

    def error(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}")

    def number(self, where: str, v: Any, positive: bool = False) -> None:
        if v is None:
            return
        for x in v if isinstance(v, list) else [v]:
            if x is None:
                continue
            if isinstance(x, bool) or not isinstance(x, Real):
                self.error(where, f"expected a number, got {x!r}")
            elif x < 0 or (positive and x == 0):
                self.error(where, f"must be {'positive' if positive else '>= 0'}, got {x}")

    def flag(self, where: str, v: Any) -> None:
        if v is not None and not isinstance(v, bool):
            self.error(where, f"expected true / false / null, got {v!r}")

    def name(self, where: str, v: Any, allowed: set[str], what: str) -> None:
        if v is not None and str(v) not in allowed:
            self.error(where, f"unknown {what} {v!r} (known: {', '.join(sorted(allowed))})")

    def keys(self, where: str, d: dict, allowed: set[str]) -> None:
        for k in d:
            if k not in allowed:
                self.error(where, f"unknown key {k!r}")

    def clock(self, where: str, v: Any, session: Session | None) -> float | None:
        """Seconds since midnight of a clock value; checks it lies in its session."""
        if v is None:
            return None
        try:
            t = clock_s(v)
        except ConfigError as e:
            self.error(where, str(e))
            return None
        if session is not None and not session.start_s <= t <= session.end_s:
            self.error(where, f"time {v} is outside session {session.id} ({_span(session)})")
        return t

    def session_of(self, where: str, item: dict) -> Session | None:
        sid = item.get("session")
        if sid is None:
            if self.sessions:
                self.error(where, "no session given")
            return None
        if str(sid) not in self.sessions:
            self.error(where, f"unknown session {sid!r}")
            return None
        return self.sessions[str(sid)]

    def items(self, raw: dict, key: str) -> Iterator[tuple[str, dict]]:
        value = raw.get(key)
        if value is None:
            return
        if not isinstance(value, list):
            self.error(key, "expected a list")
            return
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                self.error(f"{key}[{i}]", "expected a mapping")
            else:
                yield f"{key}[{i}]", item

    # -- sections ----------------------------------------------------------------------
    def check(self, raw: dict) -> None:
        self.keys("top level", raw, TOP_KEYS)
        if raw.get("schema_version") != SCHEMA_VERSION:
            self.error("schema_version", f"expected {SCHEMA_VERSION}")
        site = raw.get("site") or {}
        if site.get("tls_id") is not None and self.tls_id and site["tls_id"] != self.tls_id:
            self.error("site.tls_id", f"{site['tls_id']!r} is not the scenario TLS {self.tls_id!r}")
        self.check_sessions(raw)
        self.check_signal(raw.get("signal"))
        self.check_geometry(raw.get("geometry"))
        self.check_queues(raw)
        self.check_counts(raw)
        self.check_lists(raw)

    def check_sessions(self, raw: dict) -> None:
        for where, s in self.items(raw, "sessions"):
            sid = s.get("id")
            if not sid:
                self.error(where, "missing id")
                continue
            if str(sid) in self.sessions:
                self.error(where, f"duplicate session id {sid!r}")
            start = self.clock(f"{where}.start", s.get("start"), None)
            end = self.clock(f"{where}.end", s.get("end"), None)
            if start is None or end is None:
                if s.get("start") is None or s.get("end") is None:
                    self.error(where, "start and end are required")
                continue
            if end <= start:
                self.error(where, "end must be after start")
            date = s.get("date")
            if date is not None and not isinstance(date, dt.date):
                try:
                    dt.date.fromisoformat(str(date))
                except ValueError:
                    self.error(f"{where}.date", f"bad date {date!r} (expected YYYY-MM-DD)")
            self.sessions[str(sid)] = Session(
                str(sid), None if date is None else str(date), start, end
            )

    def check_signal(self, sig: Any) -> None:
        if sig is None:
            return
        if not isinstance(sig, dict):
            self.error("signal", "expected a mapping")
            return
        self.keys("signal", sig, SIGNAL_KEYS)
        if sig.get("source") is not None and sig["source"] not in SIGNAL_SOURCES:
            self.error("signal.source", f"expected one of {', '.join(SIGNAL_SOURCES)}")
        if sig.get("session") is not None:
            self.session_of("signal", sig)
        self.number("signal.cycles_timed", sig.get("cycles_timed"))
        self.number("signal.cycle_s", sig.get("cycle_s"), positive=True)
        order = sig.get("stage_order")
        if order is not None and not (
            isinstance(order, list) and all(isinstance(s, str) for s in order)
        ):
            self.error("signal.stage_order", "expected a list of stage names")
        for k in ("varies_between_cycles", "bus_stages_on_demand", "ped_only_stage"):
            self.flag(f"signal.{k}", sig.get(k))
        heads = sig.get("heads") or {}
        if not isinstance(heads, dict):
            self.error("signal.heads", "expected a mapping of signal groups")
            return
        for group, head in heads.items():
            where = f"signal.heads.{group}"
            self.name(where, group, self.groups, "signal group")
            if head is None:
                continue
            if not isinstance(head, dict):
                self.error(where, "expected a mapping")
                continue
            self.keys(where, head, set(HEAD_KEYS))
            for k in HEAD_KEYS:
                if k == "green_s" and isinstance(head.get(k), dict):
                    for stage, v in head[k].items():
                        self.number(f"{where}.green_s.{stage}", v, positive=True)
                else:
                    self.number(f"{where}.{k}", head.get(k), positive=k == "green_s")
            green = head.get("green_s")
            if group in self.multi and not isinstance(green, dict) and numbers_or_none(green):
                self.error(f"{where}.green_s", multi_green_message(group, self.multi[group]))

    def check_geometry(self, geo: Any) -> None:
        if geo is None:
            return
        if not isinstance(geo, dict):
            self.error("geometry", "expected a mapping")
            return
        for a in geo.get("lanes_at_stop_line") or {}:
            self.name("geometry.lanes_at_stop_line", a, self.approaches, "approach")
        for c, d in (geo.get("crossings") or {}).items():
            where = f"geometry.crossings.{c}"
            self.name(where, c, self.crossings, "crossing")
            if isinstance(d, dict):
                self.keys(where, d, {"signalised", "distance_m", "length_m", "refuge"})
                self.flag(f"{where}.signalised", d.get("signalised"))
                self.number(f"{where}.distance_m", d.get("distance_m"))
                self.number(f"{where}.length_m", d.get("length_m"), positive=True)
        self.number("geometry.s_bus_lane_length_m", geo.get("s_bus_lane_length_m"))

    def check_queues(self, raw: dict) -> None:
        for where, q in self.items(raw, "queues"):
            session = self.session_of(where, q)
            self.clock(f"{where}.time", q.get("time"), session)
            self.name(where, q.get("approach"), self.approaches, "approach")
            if q.get("approach") is None:
                self.error(where, "approach is required")
            if q.get("kind") not in QUEUE_KINDS:
                self.error(f"{where}.kind", f"expected one of {', '.join(QUEUE_KINDS)}")
            self.number(f"{where}.vehicles", q.get("vehicles"))
            self.number(f"{where}.visible_m", q.get("visible_m"), positive=True)

    def check_counts(self, raw: dict) -> None:
        movements = {f"{a}_{b}" for a in self.approaches for b in self.approaches if a != b}
        for where, c in self.items(raw, "counts"):
            session = self.session_of(where, c)
            b = self.clock(f"{where}.begin", c.get("begin"), session)
            e = self.clock(f"{where}.end", c.get("end"), session)
            if b is not None and e is not None and e <= b:
                self.error(where, "end must be after begin")
            for key, allowed, what in (
                ("movements", movements, "movement"),
                ("buses", self.bus_lines, "bus line"),
                ("pedestrians", self.crossings, "crossing"),
            ):
                for k, v in (c.get(key) or {}).items():
                    self.name(f"{where}.{key}", k, allowed, what)
                    self.number(f"{where}.{key}.{k}", v)
            for k in ("two_wheelers", "poliveau_in", "wallons_right"):
                self.number(f"{where}.{k}", c.get(k))

    def check_lists(self, raw: dict) -> None:
        for where, b in self.items(raw, "bus_stops"):
            session = self.session_of(where, b)
            arrive = self.clock(f"{where}.arrive", b.get("arrive"), session)
            depart = self.clock(f"{where}.depart", b.get("depart"), session)
            if arrive is not None and depart is not None and depart < arrive:
                self.error(where, "depart must not be before arrive")
        for where, s in self.items(raw, "saturation"):
            self.session_of(where, s)
            self.name(where, s.get("approach"), self.approaches, "approach")
            self.number(f"{where}.lane", s.get("lane"))
            self.number(f"{where}.headways_s", s.get("headways_s"), positive=True)
        for where, lp in self.items(raw, "loops"):
            self.name(where, lp.get("approach"), self.approaches, "approach")
            self.number(f"{where}.distance_to_stop_line_m", lp.get("distance_to_stop_line_m"))


def _span(s: Session) -> str:
    return f"{clock_str(s.start_s)}-{clock_str(s.end_s)}"


def load_field(path: str | Path, cfg: dict) -> FieldData:
    """Load and check the field file against a scenario config. An absent file gives an
    empty FieldData; schema problems raise one ConfigError listing all of them."""
    path = Path(path)
    if not path.is_file():
        return FieldData(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    checker = _Checker(cfg)
    checker.check(raw)
    if checker.errors:
        listing = "\n  - ".join(checker.errors)
        raise ConfigError(f"{path}: {len(checker.errors)} problem(s):\n  - {listing}")
    return FieldData(path, True, raw, checker.sessions)


def new_stage_names(data: FieldData, cfg: dict) -> list[str]:
    """Observed stage names that the scenario does not model."""
    known = {s["name"] for s in cfg.get("network", {}).get("stages", [])}
    return [s for s in data.signal.get("stage_order") or [] if s not in known]


# -- inventory -------------------------------------------------------------------------
def _leaves(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    """(dotted path, value) of every measurement leaf; lists of numbers are one leaf."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k not in META_KEYS:
                yield from _leaves(v, f"{path}.{k}")
    elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        for i, v in enumerate(value):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, value


def is_measured(value: Any) -> bool:
    return value is not None and value != [] and value != ""


def inventory(data: FieldData) -> dict[str, tuple[int, int]]:
    """Per section: (measured, missing) measurement items; an absent section has none.
    Missing = NOT_MEASURED."""
    out = {}
    for section in SECTIONS:
        value = data.raw.get(section)
        leaves = list(_leaves(value, section)) if value is not None else []
        measured = sum(is_measured(v) for _, v in leaves)
        out[section] = (measured, len(leaves) - measured)
    return out


def summary_line(data: FieldData) -> str:
    inv = inventory(data)
    measured = sum(m for m, _ in inv.values())
    missing = sum(n for _, n in inv.values())
    if measured == 0:
        return "0 measured; all rows NOT_MEASURED"
    return f"{measured} measured; {missing} NOT_MEASURED"


# -- CLI -------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="check the file (default)")
    parser.add_argument("--file", type=Path, default=None, help=f"default {FIELD_FILE}")
    parser.add_argument("--scenario", type=Path, default=PITIE_FIXED_CONFIG)
    args = parser.parse_args(argv)
    try:
        cfg = load_scenario(args.scenario)
        path = args.file or FIELD_FILE
        data = load_field(path, cfg)
    except ConfigError as e:
        print(f"field file error: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    if not data.exists:
        print(f"field file {path.as_posix()} absent: {summary_line(data)}")
        return
    print(f"field file {path.as_posix()}: schema ok, source {data.signal_source}")
    for section, (measured, missing) in inventory(data).items():
        print(f"  {section:<11} {measured:>3} measured, {missing:>3} missing")
    new = new_stage_names(data, cfg)
    if new:
        print(f"  stages not in the model: {', '.join(new)} (network.stages needs an update)")
    print(summary_line(data))


if __name__ == "__main__":
    main()

"""docs/report/facts.json: every result the report and the deck quote (plan 4.6, D113).

Each fact is `{value, unit, text, source, source_sha256, code_version}`: `value` is the
machine value, `text` the formatted string a `{{key}}` placeholder renders ('.' as the
decimal point, a non-breaking space before the unit), `source` the repo-relative file it
was read from with that file's sha256, and `code_version` the code version the source
records (None when it records none). A fact whose source is missing, or that the source
does not hold yet, keeps its key with `value` and `text` None and a `pending` reason: the
templates then render a visible **PENDING**, never a placeholder number.

Every key is declared up front (`declared_keys`), so `facts.json` always lists the full
set, and a template key that no reader can ever produce is a build error (keys.txt test).

Sources (all optional; local outputs under data/ are not versioned, so the committed
facts.json is the reference a fresh clone rebuilds the report from):

- `meta.*`   docs/report/meta.yaml (title, authors, version) and the code version;
- `val.*`    data/results/validation/pitie_fixed/summary.json, configs/validation.yaml;
- `net.*`    the static signal-program check (sim.tls_check) of configs/pitie_fixed.yaml
  on its committed network file;
- `eval.*`   data/results/evaluation/{summary,a05}.json, main/{kpis,a04}.csv,
  configs/evaluation.yaml;
- `a06.*`, `rep106.*`  data/results/evaluation/a06.json (Part 8, D94, D95);
- `final.*`  data/results/evaluation/final_table.csv (Part 8, D98);
- `rl7.*`    data/rl7/devtest/verdict.json, models/rl/<S3>/manifest.json,
  docs/figures/rl7/sensitivity.csv;
- `eta.*`    data/models/eta/LATEST -> <bundle>/eval.json;
- `api.*`    the latency block of docs/api.md;
- `usertests.*`  docs/usertests/synthesis.json (A-10, D112);
- `demo.*`   docs/demo/rehearsal.json and docs/demo/rehearsal_log.yaml (A-08, A-09, D116);
- `a01.*`    data/results/_rebaseline/<HEAD>/verify.json (A-01, D97);
- `acceptance.A-01..A-10`  computed from the facts above (report.acceptance, D115).

    python -m traffic_twin.report facts [--out docs/report/facts.json]
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import yaml

from traffic_twin.paths import (
    ETA_CONFIG,
    EVAL_A06,
    EVAL_DIR,
    EVAL_FINAL_TABLE,
    EVALUATION_CONFIG,
    REPORT_DIR,
    REPORT_FACTS,
    VALIDATION_CONFIG,
)

SCHEMA = "report-facts-1"
NBSP = " "  # between a number and its unit: never broken across lines
META_FILE = REPORT_DIR / "meta.yaml"
VALIDATION_SUMMARY = Path("data/results/validation/pitie_fixed/summary.json")
EVAL_SUMMARY = EVAL_DIR / "summary.json"
EVAL_A05 = EVAL_DIR / "a05.json"
EVAL_PROVENANCE = EVAL_DIR / "provenance.json"
RL7_VERDICT = Path("data/rl7/devtest/verdict.json")
RL_MODELS = Path("models/rl")
S3_CONFIG = Path("configs/pitie_rl_s3.yaml")
SENSITIVITY = Path("docs/figures/rl7/sensitivity.csv")
ETA_LATEST = Path("data/models/eta/LATEST")
API_DOC = Path("docs/api.md")
USERTESTS_SYNTHESIS = Path("docs/usertests/synthesis.json")
REHEARSAL = Path("docs/demo/rehearsal.json")
REHEARSAL_LOG = Path("docs/demo/rehearsal_log.yaml")
REBASELINE_DIR = Path("data/results/_rebaseline")
S0_CONFIG = Path("configs/pitie_fixed.yaml")

PART8_WHY = "the single Part 8 run on seeds 1-10 (tasks.py part8) has not been made"
# Labels of the person-delay facts; S3/S4 exist only after the Part 8 run.
PERSON_DELAY_LABELS = ("S0", "S1", "S2", "S2_noP", "S3", "S4")
# Final-table sections that are not KPI rows (plan 4.2).
EXTRA_SECTIONS = ("safety", "activity", "equity", "occupancy")
UNITS = {"%": "%", "s": "s", "veh/h": "veh/h", "veh": "veh", "–": None, "-": None, "": None}


class FactsError(RuntimeError):
    """A source exists but cannot be read as the contract says (a loud build error)."""


# -- formatting ------------------------------------------------------------------------
def fmt(x, nd: int = 1, unit: str | None = None, *, signed: bool = False) -> str:
    """`x` with `nd` decimals, '.' as the decimal point (locale-proof), an optional sign
    and a non-breaking space before the unit."""
    s = f"{float(x):+.{nd}f}" if signed else f"{float(x):.{nd}f}"
    if float(s) == 0:  # never "-0.0" or "+0.0"
        s = s.lstrip("+-")
    return f"{s}{NBSP}{unit}" if unit else s


def pct(x, nd: int = 1, *, signed: bool = False) -> str:
    return fmt(x, nd, "%", signed=signed)


def pval(p: float) -> str:
    """An exact Wilcoxon p (always >= 0.001 at n = 10) with 4 decimals."""
    return f"{float(p):.4f}"


def seeds_text(seeds: list[int]) -> str:
    seeds = sorted(int(s) for s in seeds)
    if seeds and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]}" if len(seeds) > 1 else str(seeds[0])
    return ", ".join(str(s) for s in seeds)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(label: str) -> str:
    """Fact-key form of a label or reference: S1 -> s1, S2_noP -> s2_nop."""
    return label.lower()


# -- the collector ---------------------------------------------------------------------
class Collector:
    """Reads the sources under `root` (the repo root) into facts. Every reader declares
    its keys first, so a missing source leaves them PENDING with a reason."""

    def __init__(self, root: Path = Path(".")):
        self.root = Path(root)
        self.facts: dict[str, dict] = {}
        self._sha: dict[str, str] = {}

    # sources
    def path(self, rel: Path | str) -> Path:
        return self.root / rel

    def exists(self, rel: Path | str) -> bool:
        return self.path(rel).is_file()

    def sha(self, rel: Path | str) -> str:
        rel = Path(rel).as_posix()
        if rel not in self._sha:
            self._sha[rel] = sha256(self.path(rel))
        return self._sha[rel]

    def json(self, rel: Path | str) -> dict | list:
        try:
            return json.loads(self.path(rel).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FactsError(f"{Path(rel).as_posix()}: cannot read JSON ({exc})") from exc

    def yaml(self, rel: Path | str):
        try:
            return yaml.safe_load(self.path(rel).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise FactsError(f"{Path(rel).as_posix()}: cannot read YAML ({exc})") from exc

    # facts
    def declare(self, keys, why: str, source: Path | str | None = None) -> None:
        """Mark `keys` PENDING (`why`) unless a later `put` fills them."""
        src = Path(source).as_posix() if source is not None else None
        for k in keys:
            if k not in self.facts:
                self.facts[k] = _pending(k, why, src)

    def put(
        self,
        key: str,
        value,
        text: str | None = None,
        *,
        unit: str | None = None,
        source: Path | str | None = None,
        code_version: str | None = None,
    ) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            self.facts[key] = _pending(key, "not in the source", _posix(source))
            return
        if text is None:
            if isinstance(value, bool):
                text = "yes" if value else "no"
            elif isinstance(value, float):
                text = fmt(value, 1, unit)
            else:
                text = f"{value}{NBSP}{unit}" if unit else str(value)
        src = _posix(source)
        self.facts[key] = {
            "value": value,
            "unit": unit,
            "text": text,
            "source": src,
            "source_sha256": self.sha(src) if src and self.exists(src) else None,
            "code_version": code_version,
        }

    def text(self, key: str) -> str | None:
        f = self.facts.get(key)
        return None if f is None else f.get("text")

    def value(self, key: str):
        f = self.facts.get(key)
        return None if f is None else f.get("value")


def _posix(p: Path | str | None) -> str | None:
    return None if p is None else Path(p).as_posix()


def _pending(key: str, why: str, source: str | None) -> dict:
    return {
        "value": None,
        "unit": None,
        "text": None,
        "source": source,
        "source_sha256": None,
        "code_version": None,
        "pending": why,
    }


def is_pending(fact: dict | None) -> bool:
    return fact is None or fact.get("text") is None


# -- readers ---------------------------------------------------------------------------
def read_meta(c: Collector, code: str) -> None:
    keys = [
        "meta.title",
        "meta.subtitle",
        "meta.version",
        "meta.date",
        "meta.authors",
        "meta.institution",
        "meta.supervisor",
        "meta.field_visit",
        "meta.site",
        "meta.code",
    ]
    c.declare(keys, f"{META_FILE.as_posix()} not found", META_FILE)
    c.put("meta.code", code, source=None, code_version=code)
    if not c.exists(META_FILE):
        return
    m = c.yaml(META_FILE) or {}
    for k in ("title", "subtitle", "version", "date", "field_visit", "site"):
        if m.get(k) not in (None, ""):
            c.put(f"meta.{k}", str(m[k]), source=META_FILE)
    authors = [str(a) for a in m.get("authors") or [] if str(a).strip()]
    if authors:
        c.put("meta.authors", authors, ", ".join(authors), source=META_FILE)
    for k in ("institution", "supervisor"):  # blank until provided (plan Q43)
        v = str(m.get(k) or "").strip()
        c.put(f"meta.{k}", v, v or "to be confirmed", source=META_FILE)


def read_validation(c: Collector) -> None:
    keys = [
        "val.a02.verdict",
        "val.a02.pass",
        "val.a02.rows",
        "val.geh.n_below",
        "val.geh.n_counts",
        "val.geh.share_pct",
        "val.seeds",
        "val.preregistered",
        "val.signal_not_measured",
        "val.queue_not_measured",
        "val.calibration",
        "val.calibration.rescaled",
        "val.flow_threshold_pct",
        "val.geh_target_share_pct",
    ]
    c.declare(keys, f"{VALIDATION_SUMMARY.as_posix()} not found (tasks.py validate)")
    if c.exists(VALIDATION_CONFIG):
        th = (c.yaml(VALIDATION_CONFIG) or {}).get("thresholds") or {}
        if "flow_rel" in th:
            c.put(
                "val.flow_threshold_pct",
                100 * th["flow_rel"],
                pct(100 * th["flow_rel"], 0),
                unit="%",
                source=VALIDATION_CONFIG,
            )
        if "geh_share" in th:
            c.put(
                "val.geh_target_share_pct",
                100 * th["geh_share"],
                pct(100 * th["geh_share"], 0),
                unit="%",
                source=VALIDATION_CONFIG,
            )
    if not c.exists(VALIDATION_SUMMARY):
        return
    d = c.json(VALIDATION_SUMMARY)
    src, cv = VALIDATION_SUMMARY, d.get("code_version")
    a02, geh = d.get("a02") or {}, d.get("geh") or {}

    def put(k, v, text=None, unit=None):
        c.put(k, v, text, unit=unit, source=src, code_version=cv)

    put("val.a02.verdict", a02.get("verdict"))
    put("val.a02.pass", a02.get("pass"))
    put("val.a02.rows", a02.get("rows"))
    put("val.geh.n_below", geh.get("n_below"))
    put("val.geh.n_counts", geh.get("n_counts"))
    if geh.get("share") is not None:
        put("val.geh.share_pct", 100 * geh["share"], pct(100 * geh["share"], 0), "%")
    if d.get("seeds"):
        put("val.seeds", d["seeds"], seeds_text(d["seeds"]))
    put("val.preregistered", d.get("preregistered"))
    put("val.signal_not_measured", (d.get("signal_status") or {}).get("NOT_MEASURED"))
    put("val.queue_not_measured", (d.get("queue_status") or {}).get("NOT_MEASURED"))
    cal = d.get("calibration") or {}
    put("val.calibration", cal.get("status"))
    put("val.calibration.rescaled", *_rescaled(cal.get("recommended")))


def _rescaled(scales) -> tuple:
    """(value, text) of the calibration's recommended arm scales: which arms the loop
    rescaled, from the validation summary (never typed in a template, D113)."""
    if not isinstance(scales, dict) or not scales:
        return None, None
    moved = {a: float(v) for a, v in sorted(scales.items()) if abs(float(v) - 1.0) > 1e-9}
    if not moved:
        return scales, "no arm needed rescaling (every recommended arm scale is 1)"
    parts = ", ".join(f"{a} {fmt(v, 2)}" for a, v in moved.items())
    rest = " (the other arms at 1)" if len(moved) < len(scales) else ""
    return scales, f"arm scales applied: {parts}{rest}"


NET_KEYS = [
    "net.tls_check",
    "net.phases",
    "net.cycle",
    "net.green_total",
    "net.clearance",
    "net.foe_greens",
    "net.intergreen_violations",
    "net.worst_red",
    "net.red_bound",
]


def read_network(c: Collector) -> None:
    """`tasks.py tls-check` on the S0 program, recomputed from the committed config and
    network file (static, no simulation): phases, cycle, conflicts and worst-case reds.
    After a rebuild of the network (D119) these facts follow the new program."""
    c.declare(NET_KEYS, f"{S0_CONFIG.as_posix()} or its network file not found", S0_CONFIG)
    if not c.exists(S0_CONFIG):
        return
    import copy

    from traffic_twin.config import get, load_scenario
    from traffic_twin.sim import tls_check as tc

    cfg = copy.deepcopy(load_scenario(c.path(S0_CONFIG)))
    net_rel = Path(get(cfg, "network.net_file"))
    net = net_rel if net_rel.is_absolute() else c.path(net_rel)
    if not net.is_file():
        return
    cfg["network"]["net_file"] = str(net)
    report = tc.check_scenario(cfg)
    prog = tc.load_program(net, get(cfg, "network.tls_id"))
    src = net_rel.as_posix() if not net_rel.is_absolute() else None

    def put(k, v, text=None, unit=None):
        c.put(k, v, text, unit=unit, source=src)

    put("net.tls_check", report.ok, "OK" if report.ok else "FAILED")
    put("net.phases", len(prog.states))
    put("net.cycle", float(prog.cycle), fmt(prog.cycle, 0, "s"), "s")
    greens = tc._greens(cfg, "fixed.green_s", tc.scenario_stages(cfg))
    if greens:
        put("net.green_total", sum(greens), fmt(sum(greens), 0, "s"), "s")
        put("net.clearance", prog.cycle - sum(greens), fmt(prog.cycle - sum(greens), 0, "s"), "s")
    summ = report.summary or {}
    put("net.foe_greens", summ.get("foe_greens"))
    put("net.intergreen_violations", summ.get("intergreen_violations"))
    if report.worst_crossing_red:
        by_red: dict[float, list[str]] = defaultdict(list)
        for name, red in report.worst_crossing_red.items():
            by_red[float(red)].append(str(name))
        text = ", ".join(
            f"{fmt(red, 0, 's')} on crossing{'s' if len(names) > 1 else ''} "
            + (" and ".join(names) if len(names) < 3 else ", ".join(names))
            for red, names in sorted(by_red.items())
        )
        put("net.worst_red", dict(report.worst_crossing_red), text)
    if report.crossing_bound_s is not None:
        b = float(report.crossing_bound_s)
        put("net.red_bound", b, fmt(b, 0, "s"), "s")


def read_evaluation(c: Collector, ecfg: dict | None) -> None:
    """Part 4 (and, after the Part 8 run, Part 8) evaluation summary."""
    keys = [
        "eval.preregistered",
        "eval.seeds",
        "eval.n_seeds",
        "eval.a03.text",
        "eval.a03.n",
        "eval.a03.of",
        "eval.a03.n_kpis",
        "eval.a04.verdict",
        "eval.a04.runs",
        "eval.a04.worst_curb_wait",
        "eval.a04.bound",
        "eval.a05.verdict",
        "eval.a05.hl",
        "eval.a05.ci",
        "eval.a05.p",
        "eval.a05.n",
        "eval.a05.mean_s0",
        "eval.a05.mean_s2",
        "eval.a05.probe.hl",
        "eval.a05.probe.p",
        "eval.a05.probe_nop.hl",
        "eval.a05.probe_nop.p",
        "eval.hidden_degradation",
        "eval.equity.sentence",
        "eval.a05.sentence",
        "eval.person_delay.s2_verb",
        "eval.changes_n",
        "eval.occupancy.car",
        "eval.occupancy.bus",
        "eval.equity_pct",
        "eval.person_delay.rel_s2_vs_s0",
        "eval.person_delay.rel_s1_vs_s0",
    ] + [f"eval.person_delay.{_key(lb)}" for lb in PERSON_DELAY_LABELS]
    for user_kpi in (ecfg or {}).get("final_table", {}).get("rows", []):
        for lb in ("S0", "S1", "S2", "S2_noP"):
            keys.append(f"eval.kpi.{user_kpi.replace('/', '.')}.{_key(lb)}")
    for lb in PERSON_DELAY_LABELS[4:]:
        c.declare([f"eval.person_delay.{_key(lb)}"], PART8_WHY, EVAL_SUMMARY)
    c.declare(keys, f"{EVAL_SUMMARY.as_posix()} not found (tasks.py evaluate)")
    if ecfg:
        c.put("eval.changes_n", len(ecfg.get("changes") or []), source=EVALUATION_CONFIG)
        occ = ecfg.get("occupancy") or {}
        for u in ("car", "bus"):
            if u in occ:
                c.put(
                    f"eval.occupancy.{u}",
                    float(occ[u]),
                    _num_text(occ[u]),
                    source=EVALUATION_CONFIG,
                )
        eq = (ecfg.get("equity") or {}).get("rel_pct", (ecfg.get("equity") or {}).get("pct"))
        if eq is not None:
            c.put("eval.equity_pct", float(eq), pct(eq, 0), unit="%", source=EVALUATION_CONFIG)
    if not c.exists(EVAL_SUMMARY):
        return
    d = c.json(EVAL_SUMMARY)
    src = EVAL_SUMMARY

    def put(k, v, text=None, unit=None, source=src):
        c.put(k, v, text, unit=unit, source=source)

    put("eval.preregistered", d.get("preregistered"))
    seeds = d.get("seeds") or []
    if seeds:
        put("eval.seeds", seeds, seeds_text(seeds))
        put("eval.n_seeds", len(seeds))
    a03 = d.get("a03") or {}
    put("eval.a03.text", a03.get("text"))
    put("eval.a03.n", a03.get("n"))
    put("eval.a03.of", a03.get("of"))
    n_kpis = a03.get("n_kpis") or {}
    if n_kpis:
        put("eval.a03.n_kpis", n_kpis, ", ".join(f"{f} {n}" for f, n in sorted(n_kpis.items())))
    a04 = d.get("a04") or {}
    put("eval.a04.verdict", a04.get("verdict"))
    put("eval.a04.runs", a04.get("runs"))
    a05 = d.get("a05") or {}
    put("eval.a05.verdict", a05.get("verdict"))
    prim = a05.get("primary") or {}
    if prim.get("hl") is not None:
        put("eval.a05.hl", prim["hl"], fmt(prim["hl"], 1, "s"), "s")
    if prim.get("hl_ci"):
        lo, hi = prim["hl_ci"]
        put("eval.a05.ci", [lo, hi], f"[{fmt(lo)}, {fmt(hi)}]{NBSP}s")
    if prim.get("p_one_sided") is not None:
        put("eval.a05.p", prim["p_one_sided"], pval(prim["p_one_sided"]))
    put("eval.a05.n", prim.get("n"))
    if prim.get("hl") is not None:
        put("eval.a05.sentence", prim["hl"], a05_sentence(prim["hl"]))
    rows = d.get("hidden_degradation")
    hidden = [h for h in rows or [] if h.get("hidden_degradation")]
    put(
        "eval.hidden_degradation",
        [f"{h['family']} {h['scenario']}" for h in hidden],
        ", ".join(f"{h['family']} {h['scenario']}" for h in hidden) or "none",
    )
    if rows is not None:
        put("eval.equity.sentence", *equity_sentence(rows, c.text("eval.equity_pct")))
    pdm = (d.get("person_delay") or {}).get("main") or {}
    for lb in PERSON_DELAY_LABELS:
        v = (pdm.get(lb) or {}).get("person_delay_mean")
        if v is not None:
            put(f"eval.person_delay.{_key(lb)}", v, fmt(v, 1, "s"), "s")
    for lb in ("S1", "S2"):
        a, b = (
            (pdm.get("S0") or {}).get("person_delay_mean"),
            (pdm.get(lb) or {}).get("person_delay_mean"),
        )
        if a and b is not None:
            rel = 100 * (b - a) / a
            put(f"eval.person_delay.rel_{_key(lb)}_vs_s0", rel, pct(rel, 1, signed=True), "%")
            if lb == "S2":
                put("eval.person_delay.s2_verb", rel, change_verb(rel))
    if c.exists(EVAL_A05):
        a5 = c.json(EVAL_A05)
        p = a5.get("primary") or {}
        for k, field in (("mean_s0", "a_mean"), ("mean_s2", "b_mean")):
            if p.get(field) is not None:
                put(f"eval.a05.{k}", p[field], fmt(p[field], 1, "s"), "s", EVAL_A05)
        for name, a in (("probe", "S0"), ("probe_nop", "S2_noP")):
            for s in a5.get("secondary") or []:
                if (s.get("family"), s.get("a"), s.get("b")) == ("ambulance_probe", a, "S2"):
                    put(f"eval.a05.{name}.hl", s["hl"], fmt(s["hl"], 1, "s"), "s", EVAL_A05)
                    put(
                        f"eval.a05.{name}.p",
                        s["p_one_sided"],
                        pval(s["p_one_sided"]),
                        source=EVAL_A05,
                    )
    _read_kpis(c, ecfg)
    _read_a04(c, ecfg)


def change_verb(rel: float) -> str:
    """'lowers' / 'raises' / 'leaves unchanged': the direction of a change, from its sign."""
    return "lowers" if rel < 0 else ("raises" if rel > 0 else "leaves unchanged")


def a05_sentence(hl: float) -> str:
    """The A-05 primary shift (S2 - S0, ambulance travel time) as a sentence whose verb
    follows the sign of the Hodges-Lehmann estimate."""
    if hl < 0:
        return f"shortens the ambulance travel time by a Hodges-Lehmann shift of {fmt(-hl, 1, 's')}"
    if hl > 0:
        return f"lengthens the ambulance travel time by a Hodges-Lehmann shift of {fmt(hl, 1, 's')}"
    return f"leaves the ambulance travel time unchanged (Hodges-Lehmann shift {fmt(0, 1, 's')})"


USER_GROUPS = {"car": "car", "bus": "bus", "pedestrian": "pedestrian", "ambulance": "ambulance"}


def equity_sentence(rows: list[dict], eq_text: str | None) -> tuple:
    """(value, text) of the spec 13.3 equity result, from summary.json's
    `hidden_degradation` rows: which controllers lower the person-weighted delay
    significantly, and which of them are flagged, for which user groups (D113)."""
    better = [r for r in rows if r.get("global_better")]
    hidden = [r for r in rows if r.get("hidden_degradation")]
    value = {
        f"{r['family']} {r['scenario']}": sorted(
            {str(k).split(".", 1)[0] for k in r.get("flagged") or []}
        )
        for r in hidden
    }
    if not better and not hidden:
        return value, (
            "No controller lowers the person-weighted delay significantly against S0, so "
            "no hidden degradation is flagged"
        )
    limit = f"more than {eq_text} " if eq_text else ""
    named = "; ".join(
        f"{name} ({', '.join(USER_GROUPS.get(g, g) for g in groups) or 'no group listed'})"
        for name, groups in value.items()
    )
    n_better = len(better) or len(hidden)
    if not hidden:
        return value, (
            f"Of the {n_better} controller runs that lower the person-weighted delay "
            f"significantly against S0, none makes a car arm, bus flow or crossing {limit}"
            "worse than S0"
        )
    text = (
        f"{len(hidden)} of the {n_better} controller runs that lower the person-weighted "
        f"delay significantly against S0 also make a KPI {limit}worse than S0 (user groups "
        f"in brackets): {named}"
    )
    if len(hidden) == n_better:
        text += "; each of them pays for its average gain somewhere"
    return value, text


def _num_text(x) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:g}"


def _read_kpis(c: Collector, ecfg: dict | None) -> None:
    """Means over the seeds of the final-table KPIs for S0-S2_noP from the Part 4 KPI
    file (main family): the objective table cites them before the Part 8 run."""
    rel = EVAL_DIR / "main" / "kpis.csv"
    if not ecfg or not c.exists(rel):
        return
    wanted = {tuple(r.split("/", 1)) for r in ecfg.get("final_table", {}).get("rows", [])}
    acc: dict[tuple, list[float]] = defaultdict(list)
    with c.path(rel).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            uk = (row["user_type"], row["kpi"])
            if uk in wanted and row["label"] in ("S0", "S1", "S2", "S2_noP"):
                try:
                    acc[(uk, row["label"])].append(float(row["value"]))
                except ValueError:
                    continue
    for ((user, kpi), lb), vals in acc.items():
        m = sum(vals) / len(vals)
        unit = "veh/h" if kpi.endswith("_vph") else (None if "rate" in kpi else "s")
        if kpi == "stops":
            unit = None
        nd = 2 if unit is None else 1
        c.put(f"eval.kpi.{user}.{kpi}.{_key(lb)}", m, fmt(m, nd, unit), unit=unit, source=rel)


def _read_a04(c: Collector, ecfg: dict | None) -> None:
    """Worst curb wait over every evaluated run (both families) and its bound."""
    files = [EVAL_DIR / fam / "a04.csv" for fam in ("main", "ambulance_probe")]
    files = [f for f in files if c.exists(f)]
    if not files:
        return
    worst, bound = None, None
    for rel in files:
        with c.path(rel).open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    w, b = float(row["ped_curb_wait_max"]), float(row["ped_max_wait_s"])
                except (KeyError, ValueError):
                    continue
                worst = w if worst is None else max(worst, w)
                bound = b if bound is None else min(bound, b)
    if worst is not None:
        c.put("eval.a04.worst_curb_wait", worst, fmt(worst, 0, "s"), unit="s", source=files[0])
        c.put("eval.a04.bound", bound, fmt(bound, 0, "s"), unit="s", source=files[0])


def a06_keys(ecfg: dict | None) -> list[str]:
    refs = [str(r) for r in ((ecfg or {}).get("a06") or {}).get("comparisons", ["S1", "S2", "S0"])]
    keys = [
        "a06.status",
        "a06.problems",
        "a06.n_required",
        "a06.primary",
        "a06.primary_sentence",
        "a06.model_sha8",
        "a06.inputs_ok",
        "a06.registration.commit",
        "a06.registration.committed_at",
        "a06.registration.order",
        "a06.registration.first_s3_started_at",
    ]
    for r in refs:
        for f in (
            "n",
            "mean_ref",
            "sd_ref",
            "mean_s3",
            "sd_s3",
            "hl",
            "ci",
            "p",
            "p_holm",
            "outcome",
            "sentence",
        ):
            keys.append(f"a06.{_key(r)}.{f}")
    keys += [
        "rep106.c1.hl",
        "rep106.c1.p",
        "rep106.c1.pass",
        "rep106.c2.pass",
        "rep106.c2.worst",
        "rep106.c3.pass",
        "rep106.c3.violations",
        "rep106.c4.cited",
        "rep106.confirms",
        "rep106.main_solution",
        "rep106.text",
    ]
    return keys


def read_a06(c: Collector, ecfg: dict | None) -> None:
    """a06.json (plan 4.2): the A-06 test of record and the spec 10.6 check on seeds
    1-10. Never a pre-Part-8 value: absent -> every key PENDING."""
    c.declare(a06_keys(ecfg), PART8_WHY, EVAL_A06)
    if not c.exists(EVAL_A06):
        return
    d = c.json(EVAL_A06)
    src = EVAL_A06
    if d.get("status") not in ("PASS", "INVALID"):
        raise FactsError(f"{src.as_posix()}: status {d.get('status')!r} is not PASS or INVALID")

    def put(k, v, text=None, unit=None):
        c.put(k, v, text, unit=unit, source=src)

    put("a06.status", d["status"])
    probs = [str(p) for p in d.get("problems") or []]
    put("a06.problems", probs, "; ".join(probs) if probs else "none")
    put("a06.n_required", d.get("n_required"))
    put("a06.primary", d.get("primary"))
    for comp in d.get("comparisons") or []:
        r = _key(str(comp.get("reference")))
        put(f"a06.{r}.n", comp.get("n"))
        for f in ("mean_ref", "sd_ref", "mean_s3", "sd_s3"):
            if comp.get(f) is not None:
                put(f"a06.{r}.{f}", comp[f], fmt(comp[f], 1, "s"), "s")
        if comp.get("hl") is not None:
            put(f"a06.{r}.hl", comp["hl"], fmt(comp["hl"], 1, "s", signed=True), "s")
        if comp.get("ci_low") is not None and comp.get("ci_high") is not None:
            lo, hi = comp["ci_low"], comp["ci_high"]
            put(f"a06.{r}.ci", [lo, hi], f"[{fmt(lo)}, {fmt(hi)}]{NBSP}s")
        for f in ("p", "p_holm"):
            if comp.get(f) is not None:
                put(f"a06.{r}.{f}", comp[f], pval(comp[f]))
        put(f"a06.{r}.outcome", comp.get("outcome"))
        sentence = comp.get("sentence")
        put(f"a06.{r}.sentence", None if sentence is None else str(sentence).strip().rstrip("."))
    prim = _key(str(d.get("primary") or "S1"))
    if c.text(f"a06.{prim}.sentence"):
        put("a06.primary_sentence", c.value(f"a06.{prim}.sentence"))
    model = d.get("model") or {}
    if model.get("model_sha256"):
        put("a06.model_sha8", model["model_sha256"], str(model["model_sha256"])[:8])
    put("a06.inputs_ok", model.get("inputs_ok"))
    reg = d.get("registration") or {}
    for f in ("commit", "committed_at", "order", "first_s3_started_at"):
        put(f"a06.registration.{f}", reg.get(f))
    rep = d.get("replication_10_6") or {}
    c1, c2, c3, c4 = (rep.get(k) or {} for k in ("c1", "c2", "c3", "c4"))
    if c1.get("hl") is not None:
        put("rep106.c1.hl", c1["hl"], fmt(c1["hl"], 1, "s", signed=True), "s")
    if c1.get("p") is not None:
        put("rep106.c1.p", c1["p"], pval(c1["p"]))
    for k, blk in (("c1", c1), ("c2", c2), ("c3", c3)):
        if blk.get("pass") is not None:
            put(f"rep106.{k}.pass", bool(blk["pass"]), "pass" if blk["pass"] else "fail")
    rows = [r for r in c2.get("rows") or [] if r.get("rel_pct") is not None]
    if rows:
        w = max(rows, key=lambda r: r["rel_pct"])
        put("rep106.c2.worst", w, f"{w['kpi']} {pct(w['rel_pct'], 1, signed=True)}")
    if "violations" in c3:
        v = c3.get("violations") or []
        put("rep106.c3.violations", len(v))
    put("rep106.c4.cited", c4.get("cited"))
    if rep.get("confirms_fallback") is not None:
        cf = bool(rep["confirms_fallback"])
        put("rep106.confirms", cf, "confirm" if cf else "do not confirm")
    put("rep106.main_solution", rep.get("main_solution"))
    put("rep106.text", rep.get("text"))


def final_keys(ecfg: dict | None) -> list[str]:
    ft = (ecfg or {}).get("final_table") or {}
    labels = list(ft.get("columns") or []) + list(ft.get("ablations") or [])
    keys = ["final.available", "final.layout"]
    for row in ft.get("rows") or []:
        for lb in labels:
            base = f"final.{row.replace('/', '.')}.{_key(lb)}"
            keys += [base, f"{base}.rel"]
    for row in ft.get("probe_rows") or []:
        for lb in labels:
            keys.append(f"final.probe.{row.replace('/', '.')}.{_key(lb)}")
    return keys


def read_final(c: Collector, ecfg: dict | None) -> None:
    """final_table.csv (plan 4.2 columns) -> final.<user>.<kpi>.<label> (mean) and
    `.rel` (vs S0); probe rows (a section containing 'probe') -> final.probe.*."""
    keys = final_keys(ecfg)
    c.declare(keys, PART8_WHY, EVAL_FINAL_TABLE)
    ft = (ecfg or {}).get("final_table") or {}
    if ft.get("rows"):  # the pre-registered layout (D98), so the table keeps its row order
        layout = {
            "rows": list(ft["rows"]),
            "columns": list(ft.get("columns") or []),
            "ablations": list(ft.get("ablations") or []),
        }
        c.put(
            "final.layout",
            layout,
            f"{len(layout['rows'])} rows, labels "
            + ", ".join(layout["columns"] + layout["ablations"]),
            source=EVALUATION_CONFIG,
        )
    if not c.exists(EVAL_FINAL_TABLE):
        return
    declared = set(keys)
    with c.path(EVAL_FINAL_TABLE).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    need = {"section", "user", "kpi", "label", "mean", "rel_vs_s0_pct", "unit"}
    if rows and not need <= set(rows[0]):
        raise FactsError(
            f"{EVAL_FINAL_TABLE.as_posix()}: missing columns {sorted(need - set(rows[0]))}"
        )
    n = 0
    for r in rows:
        sec = (r.get("section") or "").strip()
        if sec in EXTRA_SECTIONS:
            continue
        pre = "final.probe." if "probe" in sec else "final."
        base = f"{pre}{r['user']}.{r['kpi']}.{_key(r['label'])}"
        if base not in declared:
            continue
        unit = UNITS.get((r.get("unit") or "").strip(), (r.get("unit") or "").strip() or None)
        mean = _float(r.get("mean"))
        if mean is None:
            continue
        nd = 2 if r["kpi"] in ("stops", "crossing_complete_rate") else 1
        c.put(base, mean, fmt(mean, nd, unit), unit=unit, source=EVAL_FINAL_TABLE)
        rel = _float(r.get("rel_vs_s0_pct"))
        if rel is not None and f"{base}.rel" in declared:
            c.put(f"{base}.rel", rel, pct(rel, 1, signed=True), unit="%", source=EVAL_FINAL_TABLE)
        n += 1
    c.put("final.available", n > 0, "yes" if n else "no", source=EVAL_FINAL_TABLE)


def _float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def read_rl7(c: Collector) -> None:
    keys = [
        "rl7.verdict",
        "rl7.verdict_sentence",
        "rl7.criteria_sentence",
        "rl7.failed",
        "rl7.c2.sentence",
        "rl7.c1.hl",
        "rl7.c1.ci",
        "rl7.c1.p",
        "rl7.c1.p_holm",
        "rl7.c1.pass",
        "rl7.c1.s1_mean",
        "rl7.c1.s3_mean",
        "rl7.c2.ped_rel",
        "rl7.c2.car_rel",
        "rl7.c2.bus_rel",
        "rl7.c2.limit",
        "rl7.c2.pass",
        "rl7.c3.violations",
        "rl7.c3.pass",
        "rl7.c4.text",
        "rl7.c4.pass",
        "rl7.s2_framing",
        "rl7.s2.hl",
        "rl7.s2.p",
        "rl7.seeds",
        "rl7.model_version",
        "rl7.model_sha8",
        "rl7.obs_drop",
        "rl7.steps",
        "rl7.n_features",
        "rl7.sensitivity.n_settings",
        "rl7.sensitivity.flagged",
    ]
    c.declare(keys, f"{RL7_VERDICT.as_posix()} not found (tasks.py rl7-devtest)")
    if c.exists(RL7_VERDICT):
        v = c.json(RL7_VERDICT)
        src, cv = RL7_VERDICT, v.get("code_version")

        def put(k, val, text=None, unit=None):
            c.put(k, val, text, unit=unit, source=src, code_version=cv)

        put("rl7.verdict", str(v.get("verdict", "")).upper() or None)
        c1 = v.get("C1") or {}
        if c1.get("hl") is not None:
            put("rl7.c1.hl", c1["hl"], fmt(c1["hl"], 1, "s", signed=True), "s")
        if c1.get("hl_ci"):
            lo, hi = c1["hl_ci"]
            put("rl7.c1.ci", [lo, hi], f"[{fmt(lo)}, {fmt(hi)}]{NBSP}s")
        for f in ("p", "p_holm"):
            if c1.get(f) is not None:
                put(f"rl7.c1.{f}", c1[f], pval(c1[f]))
        for f in ("s1_mean", "s3_mean"):
            if c1.get(f) is not None:
                put(f"rl7.c1.{f}", c1[f], fmt(c1[f], 1, "s"), "s")
        c2 = v.get("C2") or {}
        for r in c2.get("rows") or []:
            name = {"pedestrian": "ped", "car": "car", "bus": "bus"}.get(r["kpi"].split("/")[0])
            if name:
                put(f"rl7.c2.{name}_rel", r["rel_pct"], pct(r["rel_pct"], 1, signed=True), "%")
        if c2.get("limit_pct") is not None:
            put("rl7.c2.limit", c2["limit_pct"], pct(c2["limit_pct"], 0), "%")
        c3 = v.get("C3") or {}
        put("rl7.c3.violations", len(c3.get("violations") or []))
        c4 = v.get("C4") or {}
        if c4.get("of"):
            put(
                "rl7.c4.text",
                f"{c4['runs_below_s1']}/{c4['of']}",
                f"{c4['runs_below_s1']} of {c4['of']}",
            )
        for k, blk, field in (
            ("c1", c1, "pass"),
            ("c2", c2, "pass"),
            ("c3", c3, "pass"),
            ("c4", c4, "runs_pass"),
        ):
            if blk.get(field) is not None:
                put(f"rl7.{k}.pass", bool(blk[field]), "pass" if blk[field] else "fail")
        fr = v.get("s2_framing") or {}
        put("rl7.s2_framing", fr.get("text"))
        if fr.get("hl") is not None:
            put("rl7.s2.hl", fr["hl"], fmt(fr["hl"], 1, "s", signed=True), "s")
        if fr.get("p") is not None:
            put("rl7.s2.p", fr["p"], pval(fr["p"]))
        if v.get("seeds"):
            put("rl7.seeds", v["seeds"], seeds_text(v["seeds"]))
        _rl7_sentences(c, v, put)
    _read_s3_model(c)
    _read_sensitivity(c)


RL7_KPI_NAMES = {
    "car/time_loss_mean": "the car time loss",
    "bus/delay_mean": "the bus delay",
    "pedestrian/ped_curb_wait_mean": "the pedestrian mean curb wait",
}


def _pass_word(x) -> str:
    return "pending" if x is None else ("a pass" if x else "a fail")


def _rl7_sentences(c: Collector, v: dict, put) -> None:
    """The Part 7 result sentences, generated from verdict.json (D113): the four
    criteria, the C2 rows against S1, the failed criteria and what the verdict means."""
    blocks = {k: v.get(k) or {} for k in ("C1", "C2", "C3", "C4")}
    passes = {
        "C1": blocks["C1"].get("pass"),
        "C2": blocks["C2"].get("pass"),
        "C3": blocks["C3"].get("pass"),
        "C4": blocks["C4"].get("runs_pass"),
    }
    failed = [k for k, x in passes.items() if x is False]
    put("rl7.failed", failed, ", ".join(failed) if failed else "none")
    c1, c2 = blocks["C1"], blocks["C2"]
    parts = []
    if c1.get("hl") is not None:
        holm = f", Holm p {pval(c1['p_holm'])}" if c1.get("p_holm") is not None else ""
        parts.append(
            "C1, the person-weighted delay against the actuated baseline S1, is "
            f"{_pass_word(passes['C1'])} (HL {fmt(c1['hl'], 1, 's', signed=True)}{holm})"
        )
    worse = [RL7_KPI_NAMES.get(r["kpi"], r["kpi"]) for r in c2.get("rows") or [] if r.get("worse")]
    if passes["C2"] is not None:
        limit = f" within {pct(c2['limit_pct'], 0)}" if c2.get("limit_pct") is not None else ""
        over = f": over the limit for {', '.join(worse)}" if worse else ""
        parts.append(f"C2, equity against S1{limit}, is {_pass_word(passes['C2'])}{over}")
    if passes["C3"] is not None:
        n = len(blocks["C3"].get("violations") or [])
        parts.append(f"C3, safety, is {_pass_word(passes['C3'])} with {n} violations")
    repro = " (its repeat check is pending)" if "C4.repro" in (v.get("pending") or []) else ""
    parts.append(f"C4, reproducibility, is {_pass_word(passes['C4'])}{repro}")
    put("rl7.criteria_sentence", passes, "; ".join(parts))
    rows = [r for r in c2.get("rows") or [] if r.get("rel_pct") is not None]
    if rows:
        said = []
        for r in rows:
            rel = float(r["rel_pct"])
            how = "lower, better" if rel < 0 else ("higher, worse" if rel > 0 else "unchanged")
            if r.get("worse"):
                how += ", over the limit"
            said.append(
                f"{RL7_KPI_NAMES.get(r['kpi'], r['kpi'])} by {pct(rel, 1, signed=True)} ({how})"
            )
        text = "Against S1, S3 changes " + (
            ", ".join(said[:-1]) + " and " + said[-1] if len(said) > 1 else said[0]
        )
        put("rl7.c2.sentence", [r["rel_pct"] for r in rows], text)
    verdict = str(v.get("verdict") or "").lower()
    meaning = {
        "fallback": "Spec 10.6 applies: RL does not beat the actuated baseline on the "
        "pre-registered criteria, so the rules controller S2 is the main solution and RL is "
        "reported as a study",
        "pass": "RL beats the actuated baseline on the four pre-registered criteria",
        "pending": "At least one pre-registered criterion is still pending",
    }.get(verdict)
    if meaning:
        put("rl7.verdict_sentence", verdict, meaning)


def _read_s3_model(c: Collector) -> None:
    """The committed S3 manifest named by configs/pitie_rl_s3.yaml's model path."""
    if not c.exists(S3_CONFIG):
        return
    text = c.path(S3_CONFIG).read_text(encoding="utf-8")
    m = re.search(r"models/rl/([\w.-]+)", text)
    if not m:
        return
    rel = RL_MODELS / m.group(1) / "manifest.json"
    if not c.exists(rel):
        return
    man = c.json(rel)
    c.put("rl7.model_version", man.get("model_version"), source=rel)
    if man.get("model_sha256"):
        c.put("rl7.model_sha8", man["model_sha256"], str(man["model_sha256"])[:8], source=rel)
    drop = man.get("obs_drop") or []
    c.put("rl7.obs_drop", drop, ", ".join(drop) if drop else "none", source=rel)
    if man.get("steps") is not None:
        c.put("rl7.steps", int(man["steps"]), f"{int(man['steps']):,}", source=rel)
    c.put("rl7.n_features", man.get("n_features"), source=rel)


def _read_sensitivity(c: Collector) -> None:
    if not c.exists(SENSITIVITY):
        return
    with c.path(SENSITIVITY).open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("kind", "setting") == "setting"]
    flagged = sum(1 for r in rows if (_float(r.get("equity_flags_vs_S0")) or 0) > 0)
    c.put("rl7.sensitivity.n_settings", len(rows), source=SENSITIVITY)
    c.put("rl7.sensitivity.flagged", flagged, source=SENSITIVITY)


def read_eta(c: Collector) -> None:
    keys = [
        "eta.model_version",
        "eta.a07.verdict",
        "eta.a07.coverage_pct",
        "eta.a07.ci",
        "eta.a07.band",
        "eta.a07.mae",
        "eta.a07.ref_mae",
        "eta.a07.n_worlds",
        "eta.a07.n",
        "eta.interval_pct",
    ]
    c.declare(keys, f"{ETA_LATEST.as_posix()} not found (tasks.py part5)")
    if c.exists(ETA_CONFIG):
        q = ((c.yaml(ETA_CONFIG) or {}).get("model") or {}).get("quantiles") or []
        if len(q) >= 2:
            width = 100 * (max(q) - min(q))
            c.put("eta.interval_pct", width, pct(width, 0), unit="%", source=ETA_CONFIG)
    if not c.exists(ETA_LATEST):
        return
    bundle = c.path(ETA_LATEST).read_text(encoding="utf-8").strip()
    rel = ETA_LATEST.parent / bundle / "eval.json"
    if not c.exists(rel):
        c.declare(keys, f"{rel.as_posix()} not found")
        return
    d = c.json(rel)
    a7, cv = d.get("a07") or {}, d.get("code_version")

    def put(k, v, text=None, unit=None):
        c.put(k, v, text, unit=unit, source=rel, code_version=cv)

    put("eta.model_version", d.get("model_version") or bundle)
    if a7.get("pass") is not None:
        put("eta.a07.verdict", "PASS" if a7["pass"] else "FAIL")
    if a7.get("coverage") is not None:
        put("eta.a07.coverage_pct", 100 * a7["coverage"], pct(100 * a7["coverage"], 1), "%")
    if a7.get("ci"):
        lo, hi = a7["ci"]
        put("eta.a07.ci", [lo, hi], f"[{pct(100 * lo)}, {pct(100 * hi)}]")
    if a7.get("band"):
        lo, hi = a7["band"]
        put("eta.a07.band", [lo, hi], f"{_num_text(100 * lo)}-{pct(100 * hi, 0)}")
    for f in ("mae", "ref_mae"):
        if a7.get(f) is not None:
            put(f"eta.a07.{f}", a7[f], fmt(a7[f], 1, "s"), "s")
    put("eta.a07.n_worlds", a7.get("n_worlds"))
    if a7.get("n") is not None:
        put("eta.a07.n", int(a7["n"]), f"{int(a7['n']):,}")


LATENCY = re.compile(
    r"Measured (?P<date>\d{4}-\d{2}-\d{2}).*?code `(?P<code>[^`]+)`.*?"
    r"\*\*ENF-02 \([^)]*\): (?P<verdict>[A-Z]+)\*\*, worst p95 (?P<p95>[\d.]+) ms",
    re.S,
)


def read_api(c: Collector) -> None:
    keys = ["api.bench.verdict", "api.bench.worst_p95", "api.bench.date"]
    c.declare(keys, f"{API_DOC.as_posix()} has no latency block (tasks.py api-bench)")
    if not c.exists(API_DOC):
        return
    text = c.path(API_DOC).read_text(encoding="utf-8")
    block = text.split("<!-- latency:begin -->", 1)[-1].split("<!-- latency:end -->", 1)[0]
    m = LATENCY.search(block)
    if not m:
        return
    cv = m.group("code")
    c.put("api.bench.verdict", m.group("verdict"), source=API_DOC, code_version=cv)
    p95 = float(m.group("p95"))
    c.put("api.bench.worst_p95", p95, fmt(p95, 1, "ms"), unit="ms", source=API_DOC, code_version=cv)
    c.put("api.bench.date", m.group("date"), source=API_DOC, code_version=cv)


def read_usertests(c: Collector) -> None:
    """synthesis.json (WS5, D112). Read tolerantly: `a10.status` (or `status`) and the
    number of complete sessions (`a10.n` / `n_complete` / `n_sessions` / `n`)."""
    keys = ["usertests.status", "usertests.n"]
    c.declare(keys, f"{USERTESTS_SYNTHESIS.as_posix()} not found (tasks.py usertest-docs)")
    if not c.exists(USERTESTS_SYNTHESIS):
        return
    d = c.json(USERTESTS_SYNTHESIS)
    a10 = d.get("a10") if isinstance(d.get("a10"), dict) else {}
    status = a10.get("status") or d.get("status")
    n = next(
        (
            v
            for v in (
                a10.get("n"),
                a10.get("n_complete"),
                d.get("n_complete"),
                d.get("n_sessions"),
                d.get("n"),
            )
            if v is not None
        ),
        None,
    )
    cv = d.get("code_version")
    c.put(
        "usertests.status",
        str(status).upper() if status else None,
        source=USERTESTS_SYNTHESIS,
        code_version=cv,
    )
    c.put("usertests.n", None if n is None else int(n), source=USERTESTS_SYNTHESIS, code_version=cv)


def _status_of(x) -> str | None:
    if isinstance(x, dict):
        x = x.get("status", x.get("verdict"))
    if isinstance(x, bool):
        return "PASS" if x else "FAIL"
    return str(x).upper() if x else None


def read_demo(c: Collector) -> None:
    """docs/demo/rehearsal.json as `demo.rehearsal.record_rehearsal` writes it (WS7,
    D116): `a08` / `a09` blocks {status, commit, recorded_at[, reason]} and the
    `readiness` block of the fresh-clone rehearsal. The human rehearsals
    of docs/demo/rehearsal_log.yaml are counted with the rule of
    `demo.rehearsal.human_rehearsals` (date, commit and `manual_code_edits: 0`)."""
    keys = [
        "demo.a08",
        "demo.a09",
        "demo.a08.date",
        "demo.a09.date",
        "demo.a08.commit",
        "demo.a09.commit",
        "demo.a08.reason",
        "demo.a09.reason",
        "demo.date",
        "demo.fresh_clone",
        "demo.commit",
    ]
    c.declare(keys, f"{REHEARSAL.as_posix()} not found (tasks.py demo-rehearsal --record)")
    if c.exists(REHEARSAL):
        d = c.json(REHEARSAL)
        for k in ("a08", "a09"):
            blk = d.get(k) if isinstance(d.get(k), dict) else {}
            cv = blk.get("commit")
            c.put(f"demo.{k}", _status_of(d.get(k)), source=REHEARSAL, code_version=cv)
            date = blk.get("recorded_at")
            c.put(
                f"demo.{k}.date",
                None if date is None else str(date)[:10],
                source=REHEARSAL,
                code_version=cv,
            )
            c.put(f"demo.{k}.commit", cv, source=REHEARSAL, code_version=cv)
            c.put(f"demo.{k}.reason", blk.get("reason"), source=REHEARSAL, code_version=cv)
        a09 = d.get("a09") if isinstance(d.get("a09"), dict) else {}
        # the latest recorded rehearsal (A-08) and the release commit (A-09)
        c.put("demo.date", c.value("demo.a08.date"), source=REHEARSAL)
        c.put("demo.commit", c.value("demo.a09.commit"), source=REHEARSAL)
        # a `readiness` block is written only by `demo readiness` (git clone --local)
        rd = d.get("readiness")
        fresh = isinstance(rd, dict) and rd.get("kind", "readiness") == "readiness"
        c.put("demo.fresh_clone", fresh, source=REHEARSAL, code_version=a09.get("commit"))
    _read_human_rehearsals(c)


def valid_human_rehearsals(log) -> int:
    """The human rehearsals of a parsed rehearsal_log.yaml that count (D116): entries
    under `entries` with a date, a commit and `manual_code_edits` exactly 0 (the rule of
    `demo.rehearsal.human_rehearsals`; tests/test_report.py checks both agree)."""
    entries = (log.get("entries") or []) if isinstance(log, dict) else []
    return sum(
        1
        for e in entries
        if isinstance(e, dict)
        and e.get("date")
        and e.get("commit")
        and e.get("manual_code_edits") == 0
    )


def _read_human_rehearsals(c: Collector) -> None:
    why = (
        f"no human rehearsal in {REHEARSAL_LOG.as_posix()} with a date, a commit and "
        "manual_code_edits: 0 (D116)"
    )
    c.declare(["demo.human_rehearsals", "demo.rehearsed"], why, REHEARSAL_LOG)
    if not c.exists(REHEARSAL_LOG):
        c.facts["demo.human_rehearsals"]["pending"] = f"{REHEARSAL_LOG.as_posix()} not found"
        return
    n = valid_human_rehearsals(c.yaml(REHEARSAL_LOG) or {})
    c.put("demo.human_rehearsals", n, source=REHEARSAL_LOG)
    if n:
        c.put("demo.rehearsed", "DONE", source=REHEARSAL_LOG)


def read_a01(c: Collector) -> None:
    """A-01 evidence: the rebaseline verify.json of the code version the evaluation
    runs record (D97: {head, identical, of, differing[]})."""
    keys = ["a01.identical", "a01.of", "a01.head"]
    c.declare(keys, f"no {REBASELINE_DIR.as_posix()}/<HEAD>/verify.json yet (tasks.py part8)")
    found = (
        sorted(c.path(REBASELINE_DIR).glob("*/verify.json"))
        if c.path(REBASELINE_DIR).is_dir()
        else []
    )
    if not found:
        return
    versions = set()
    if c.exists(EVAL_PROVENANCE):
        versions = set(_strings(c.json(EVAL_PROVENANCE), "code_version"))
    pick = None
    for f in found:
        d = json.loads(f.read_text(encoding="utf-8"))
        if not versions or str(d.get("head")) in versions:
            pick = (f, d)
    if pick is None:
        return
    f, d = pick
    rel = f.relative_to(c.root)
    c.put("a01.identical", d.get("identical"), source=rel, code_version=d.get("head"))
    c.put("a01.of", d.get("of"), source=rel, code_version=d.get("head"))
    c.put("a01.head", d.get("head"), source=rel, code_version=d.get("head"))


def _strings(obj, field: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == field and isinstance(v, str):
                yield v
            else:
                yield from _strings(v, field)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v, field)


# -- entry points ----------------------------------------------------------------------
def collect_facts(root: Path = Path("."), code: str | None = None) -> dict:
    """The facts by key (meta.*, val.*, eval.*, a06.*, final.*, ...), each declared key
    present (PENDING ones with a reason), plus the acceptance rows."""
    from traffic_twin.report.acceptance import acceptance

    c = Collector(root)
    if code is None:
        code = _code_version(root)
    ecfg = c.yaml(EVALUATION_CONFIG) if c.exists(EVALUATION_CONFIG) else None
    read_meta(c, code)
    read_validation(c)
    read_network(c)
    read_evaluation(c, ecfg)
    read_a06(c, ecfg)
    read_final(c, ecfg)
    read_rl7(c)
    read_eta(c)
    read_api(c)
    read_usertests(c)
    read_demo(c)
    read_a01(c)
    for row in acceptance(c.facts):
        c.facts[f"acceptance.{row['id']}"] = {
            "value": row["status"],
            "unit": None,
            "text": row["status"],
            "evidence": row["evidence"],
            "criterion": row["criterion"],
            "source": row["source"],
            "source_sha256": None,
            "code_version": None,
        }
    return dict(sorted(c.facts.items()))


def declared_keys(root: Path = Path(".")) -> set[str]:
    """Every key `collect_facts` can produce (independent of which sources exist)."""
    return set(collect_facts(Path(root), code="unknown"))


def _code_version(root: Path) -> str:
    from traffic_twin.sim.results import code_version

    if Path(root).resolve() != Path.cwd().resolve():
        return "unknown"
    return code_version()


def dump(facts: dict, code: str) -> str:
    doc = {"schema": SCHEMA, "code_version": code, "facts": facts}
    return json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def write_facts(facts: dict, out: Path = REPORT_FACTS) -> Path:
    code = (facts.get("meta.code") or {}).get("value") or "unknown"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump(facts, code), encoding="utf-8", newline="\n")
    return out


def load_facts(path: Path = REPORT_FACTS) -> dict:
    """The facts of a facts.json (the committed reference on a fresh clone)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FactsError(f"{Path(path).as_posix()}: cannot read ({exc})") from exc
    if doc.get("schema") != SCHEMA:
        raise FactsError(f"{Path(path).as_posix()}: schema {doc.get('schema')!r} != {SCHEMA}")
    return doc["facts"]


def stale_sources(facts: dict, root: Path = Path(".")) -> list[str]:
    """Sources whose sha256 changed since the facts were collected (report check)."""
    out = []
    for key, f in sorted(facts.items()):
        src, sha = f.get("source"), f.get("source_sha256")
        if not src or not sha:
            continue
        p = Path(root) / src
        if not p.is_file():
            out.append(f"{key}: {src} is gone")
        elif sha256(p) != sha:
            out.append(f"{key}: {src} changed")
    return sorted(set(out))

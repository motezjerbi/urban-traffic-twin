"""docs/report/report.md from the templates docs/report/src/NN_*.md (D113).

Template language (everything else is plain Markdown):

- `{{key}}` renders the fact's `text`. A fact that is missing or PENDING renders a visible
  **PENDING** and is listed in the PENDING appendix: never a placeholder number.
- `<!-- include: <page>#<heading> | table: <n> | rows: <c1> <c2>, ... | cols: <h1>,<h2>,
  ... | requires: <heading> -->` copies a section of a generated page. `<heading>` is the
  heading text without its number ("KPIs per user and scenario" matches
  "## 5. KPIs per user and scenario"); `table: n` keeps only the n-th table; `rows:`
  keeps the first table of the section and the rows whose first two cells match
  (backticks ignored); `cols:` keeps the named
  columns of every table. A missing page, heading, row or column is a build error
  (ReportError), so a renamed heading in a generated page breaks the build loudly.
  `requires:` makes the include wait for a later part instead: while the page lacks that
  heading the include renders PENDING (listed); once the heading exists the include must
  resolve.
- `<!-- generate: <name> -->` writes a table computed from the facts: `acceptance`,
  `objectives`, `final_table`, `a06`, `rep106`, `provenance`, `pending`.
- `<!-- toc -->` and `<!-- pagebreak -->` are kept for the PDF renderer.

Links and images in an included page are rebased onto the report's folder. The report
always lives at docs/report/report.md (`REPORT_MD`): building into another folder writes
the same text. A missing image is a build error, except the generated figures (app
screenshots, report figures), which render PENDING until they are made.

Template lint (`lint`): outside placeholders, directives and code, no decimal number,
no percentage and no p comparison: every result is a fact (D113). Spec and section
references ("spec 13.3", "§ 4.2") and versions are allowed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from traffic_twin.paths import APP_FIGURES, REPORT_DIR, REPORT_SRC
from traffic_twin.report.acceptance import acceptance as acceptance_rows

REPORT_MD = REPORT_DIR / "report.md"
# Figures made by a later step (app screenshots, report figures, the Part 8 forest plot):
# PENDING until they exist, while any other missing image fails the build.
GENERATED_FIGURES = (
    APP_FIGURES,
    Path("docs/figures/report"),
    Path("docs/figures/pitie/eval/a06_forest.png"),
)
PENDING = "**PENDING**"

PLACEHOLDER = re.compile(r"\{\{\s*([\w.-]+)\s*\}\}")
DIRECTIVE = re.compile(r"<!--\s*(include|generate):\s*(.*?)\s*-->", re.S)
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
NUMBER_PREFIX = re.compile(r"^\d+(\.\d+)*\.?\s+")
LINK = re.compile(r"(!?\[[^\]]*\])\(([^)\s]+)(\s+\"[^\"]*\")?\)")
GENERATORS = ("acceptance", "objectives", "final_table", "a06", "rep106", "provenance", "pending")


class ReportError(RuntimeError):
    """A template, include or figure cannot be resolved: the build fails loudly."""


# -- includes --------------------------------------------------------------------------
def heading_text(text: str) -> str:
    """A heading without its section number: '5. KPIs per user' -> 'KPIs per user'."""
    return NUMBER_PREFIX.sub("", text.strip()).strip()


def headings(md: str) -> list[str]:
    out, fence = [], False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        m = None if fence else HEADING.match(line)
        if m:
            out.append(heading_text(m.group(2)))
    return out


def section(md: str, heading: str, where: str = "page") -> str:
    """The body of the section titled `heading` (number ignored, any level) up to the
    next heading of the same or a higher level. ReportError if absent or ambiguous."""
    lines = md.splitlines()
    want = heading_text(heading)
    hits = []
    fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fence = not fence
        m = None if fence else HEADING.match(line)
        if m and heading_text(m.group(2)) == want:
            hits.append((i, len(m.group(1))))
    if not hits:
        raise ReportError(f"{where}: no heading {want!r}")
    if len(hits) > 1:
        raise ReportError(f"{where}: heading {want!r} appears {len(hits)} times")
    start, level = hits[0]
    body = []
    fence = False
    for line in lines[start + 1 :]:
        if line.lstrip().startswith("```"):
            fence = not fence
        m = None if fence else HEADING.match(line)
        if m and len(m.group(1)) <= level:
            break
        body.append(line)
    return "\n".join(body).strip()


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def filter_rows(body: str, wanted: list[tuple[str, str]], where: str = "include") -> str:
    """The first table of `body`: header, separator and the rows whose first two cells
    (backticks stripped) are in `wanted`, in the table's order. Missing row -> error."""
    out, found, started = [], set(), False
    n = 0
    for line in body.splitlines():
        if not line.lstrip().startswith("|"):
            if started:
                break
            continue
        started = True
        n += 1
        key = tuple(c.strip("`") for c in _cells(line)[:2])
        if n <= 2 or key in wanted:
            out.append(line)
            found.add(key)
    if not started:
        raise ReportError(f"{where}: the section has no table")
    lost = [w for w in wanted if w not in found]
    if lost:
        raise ReportError(f"{where}: rows not found: {', '.join(' '.join(w) for w in lost)}")
    return "\n".join(out)


def nth_table(body: str, n: str, where: str = "include") -> str:
    """Only the `n`-th table (1-based) of `body`, without the text around it."""
    try:
        k = int(n)
    except ValueError:
        raise ReportError(f"{where}: table: {n!r} is not a number") from None
    tables, cur = [], []
    for line in body.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            cur.append(line)
        elif cur:
            tables.append("\n".join(cur))
            cur = []
    if not 1 <= k <= len(tables):
        raise ReportError(f"{where}: table {k} not found (the section has {len(tables)})")
    return tables[k - 1]


def select_cols(body: str, cols: list[str], where: str = "include") -> str:
    """Keep only the named columns (header text) of every table in `body`."""
    out, idx, seen = [], None, False
    for line in body.splitlines():
        if not line.lstrip().startswith("|"):
            idx = None
            out.append(line)
            continue
        cells = _cells(line)
        if idx is None:
            seen = True
            missing = [c for c in cols if c not in cells]
            if missing:
                raise ReportError(f"{where}: columns not found: {', '.join(missing)}")
            idx = [cells.index(c) for c in cols]
        out.append("| " + " | ".join(cells[i] if i < len(cells) else "" for i in idx) + " |")
    if not seen:
        raise ReportError(f"{where}: the section has no table")
    return "\n".join(out)


def rebase_links(body: str, src_dir: Path, out_dir: Path, root: Path) -> str:
    """Rewrite relative link and image targets written for `src_dir` for `out_dir`
    (both repo-relative)."""

    def fix(m: re.Match) -> str:
        target = m.group(2)
        if re.match(r"^([a-z][a-z0-9+.-]*:|#|/)", target):
            return m.group(0)
        path, _, frag = target.partition("#")
        absolute = (root / src_dir / path).resolve()
        rel = Path(os.path.relpath(absolute, (root / out_dir).resolve())).as_posix()
        return f"{m.group(1)}({rel}{'#' + frag if frag else ''}{m.group(3) or ''})"

    return LINK.sub(fix, body)


def _options(spec: str) -> tuple[str, dict[str, str]]:
    parts = [p.strip() for p in spec.split("|")]
    opts = {}
    for p in parts[1:]:
        name, sep, value = p.partition(":")
        if not sep or name.strip() not in ("rows", "cols", "table", "requires"):
            raise ReportError(f"include option {p!r}: use rows:, cols:, table: or requires:")
        opts[name.strip()] = value.strip()
    return parts[0], opts


# -- the assembler ---------------------------------------------------------------------
class Assembler:
    def __init__(self, facts: dict, out: Path, root: Path):
        self.facts, self.root = facts, Path(root)
        self.out_dir = Path(out).parent
        self.pending: dict[str, str] = {}

    def fact(self, key: str) -> str:
        f = self.facts.get(key)
        if f is None or f.get("text") is None:
            self.pending.setdefault(key, (f or {}).get("pending") or "no such fact")
            return PENDING
        return str(f["text"])

    def include(self, spec: str, where: str) -> str:
        target, opts = _options(spec)
        page, sep, heading = target.partition("#")
        if not sep or not heading.strip():
            raise ReportError(f"{where}: include {target!r} needs <page>#<heading>")
        path = self.root / page
        req = opts.get("requires")
        label = f"{page}#{heading_text(heading)}"
        if req is not None:
            if not path.is_file() or heading_text(req) not in headings(
                path.read_text(encoding="utf-8")
            ):
                self.pending.setdefault(
                    f"include {label}", f"{page} has no section {heading_text(req)!r} yet"
                )
                return f"> {PENDING}: `{label}` is generated by a later step."
        if not path.is_file():
            raise ReportError(f"{where}: included page {page} not found")
        body = section(path.read_text(encoding="utf-8"), heading, f"{where}: {page}")
        if opts.get("table"):
            body = nth_table(body, opts["table"], f"{where}: {label}")
        if opts.get("rows"):
            wanted = []
            for r in opts["rows"].split(","):
                two = r.split()
                if len(two) != 2:
                    raise ReportError(f"{where}: row {r.strip()!r} needs '<cell1> <cell2>'")
                wanted.append((two[0], two[1]))
            body = filter_rows(body, wanted, f"{where}: {label}")
        if opts.get("cols"):
            body = select_cols(
                body, [c.strip() for c in opts["cols"].split(",")], f"{where}: {label}"
            )
        body = rebase_links(body, Path(page).parent, self.out_dir, self.root)
        return (
            f"{body}\n\n*Source: [{page}]({_rel(Path(page), self.out_dir)}), section "
            f'"{heading_text(heading)}".*'
        )

    def generate(self, name: str, where: str) -> str:
        name = name.strip()
        if name not in GENERATORS:
            raise ReportError(f"{where}: unknown generator {name!r} ({', '.join(GENERATORS)})")
        if name == "pending":
            return "<!-- generate: pending -->"  # filled after every other directive
        return getattr(self, f"gen_{name}")()

    # generated tables
    def gen_acceptance(self) -> str:
        rows = [["ID", "Criterion", "Status", "Evidence"]]
        for r in acceptance_rows(self.facts):
            rows.append([r["id"], r["criterion"], f"**{r['status']}**", r["evidence"]])
        return _table(rows)

    def gen_objectives(self) -> str:
        f = self.fact
        rows = [
            ["Objective", "Measure of success (spec 3.2)", "Result"],
            [
                "O1 realistic SUMO model",
                "simulated vs real flow within the threshold",
                f"A-02 {f('val.a02.verdict')}: {f('val.a02.pass')} of {f('val.a02.rows')} rows "
                f"within ±{f('val.flow_threshold_pct')}",
            ],
            [
                "O2 baselines",
                "fixed and actuated evaluated on 10 seeds",
                f"S0 and S1 on seeds {f('eval.seeds')}: person delay {f('eval.person_delay.s0')}"
                f" and {f('eval.person_delay.s1')}",
            ],
            [
                "O3 lower the total time lost",
                "RL time loss against fixed time",
                f"car time loss S3 vs S0 {f('final.car.time_loss_mean.s3.rel')}; A-06 "
                f"{f('a06.status')}",
            ],
            [
                "O4 faster ambulance",
                "emergency travel time below the baseline",
                f"A-05 {f('eval.a05.verdict')}: HL {f('eval.a05.hl')}",
            ],
            [
                "O5 regular buses",
                "bus headway std reduced",
                f"S2 {f('eval.kpi.bus.headway_std.s2')} vs S0 {f('eval.kpi.bus.headway_std.s0')};"
                f" S3 {f('final.bus.headway_std.s3')}",
            ],
            [
                "O6 protect pedestrians",
                "longest wait bounded (90 s) in every scenario",
                f"worst curb wait {f('eval.a04.worst_curb_wait')} (bound {f('eval.a04.bound')}); "
                f"A-04 {f('acceptance.A-04')}",
            ],
            [
                "O7 honest travel time",
                "80 % interval covers 75-85 % of cases",
                f"A-07 {f('eta.a07.verdict')}: coverage {f('eta.a07.coverage_pct')}",
            ],
            [
                "O8 usable information",
                "app tested with 3 to 5 people",
                f"A-10 {f('acceptance.A-10')}",
            ],
        ]
        return _table(rows)

    def gen_final_table(self) -> str:
        if (self.facts.get("final.available") or {}).get("value") is not True:
            self.fact("final.available")
            return (
                f"> {PENDING}: the final comparison table comes from the single Part 8 "
                "run on seeds 1-10 (`data/results/evaluation/final_table.csv`)."
            )
        layout = (self.facts.get("final.layout") or {}).get("value")
        if layout:
            rows_cfg = [tuple(r.split("/", 1)) for r in layout["rows"]]
            labels = list(layout["columns"]) + list(layout["ablations"])
        else:
            rows_cfg, labels = _final_layout(final_keys_rows(self.facts))
        rows = [["User", "KPI", *labels]]
        for user, kpi in rows_cfg:
            cells = []
            for lb in labels:
                base = f"final.{user}.{kpi}.{lb.lower()}"
                v = self.fact(base)
                rel = self.facts.get(f"{base}.rel", {}).get("text")
                cells.append(v if lb == "S0" or not rel or v == PENDING else f"{v} ({rel})")
            rows.append([user, f"`{kpi}`", *cells])
        return (
            _table(rows) + "\n\nMean over seeds 1-10; in brackets the change against S0. "
            "S2_noP and S4 are ablations (S4 is out of distribution)."
        )

    def gen_a06(self) -> str:
        refs = [k.split(".")[1] for k in self.facts if re.fullmatch(r"a06\.[^.]+\.outcome", k)]
        order = {"s1": 0, "s2": 1, "s0": 2}  # the primary first (D94)
        refs.sort(key=lambda r: (order.get(r, 9), r))
        rows = [
            [
                "Reference",
                "n",
                "Mean reference",
                "Mean S3",
                "HL (S3 - ref)",
                "95 % CI",
                "p",
                "p (Holm)",
                "Outcome",
            ]
        ]
        for r in refs:
            f = self.fact
            rows.append(
                [
                    r.upper(),
                    f(f"a06.{r}.n"),
                    f(f"a06.{r}.mean_ref"),
                    f(f"a06.{r}.mean_s3"),
                    f(f"a06.{r}.hl"),
                    f(f"a06.{r}.ci"),
                    f(f"a06.{r}.p"),
                    f(f"a06.{r}.p_holm"),
                    f(f"a06.{r}.outcome"),
                ]
            )
        return _table(rows)

    def gen_rep106(self) -> str:
        f = self.fact
        rows = [
            ["Criterion (D79, repeated on seeds 1-10)", "Result", "Pass"],
            [
                "C1 person-weighted delay, S3 vs S1",
                f"HL {f('rep106.c1.hl')}, p {f('rep106.c1.p')}",
                f("rep106.c1.pass"),
            ],
            ["C2 equity vs S1 (5 % limit)", f"worst: {f('rep106.c2.worst')}", f("rep106.c2.pass")],
            ["C3 safety", f"{f('rep106.c3.violations')} violations", f("rep106.c3.pass")],
            ["C4 reproducibility (cited from Part 7)", f("rep106.c4.cited"), f("rep106.c4.cited")],
        ]
        return _table(rows)

    def gen_provenance(self) -> str:
        by_src: dict[str, list[str]] = {}
        for k, v in sorted(self.facts.items()):
            if v.get("text") is not None and v.get("source") and v.get("source_sha256"):
                by_src.setdefault(v["source"], []).append(k)
        rows = [["Source", "sha256", "Code version", "Facts"]]
        for src, keys in sorted(by_src.items()):
            first = self.facts[keys[0]]
            ns = sorted({k.rsplit(".", 1)[0] if k.count(".") > 1 else k for k in keys})
            rows.append(
                [
                    f"`{src}`",
                    f"`{first['source_sha256'][:12]}`",
                    first.get("code_version") or "–",
                    f"{len(keys)}: "
                    + ", ".join(f"`{n}`" for n in ns[:6])
                    + (", …" if len(ns) > 6 else ""),
                ]
            )
        return _table(rows)

    def pending_table(self) -> str:
        """The PENDING items grouped by what they wait for; facts of one namespace
        that wait for the same thing are shown as `<namespace>.*` with their count."""
        if not self.pending:
            return "Nothing is pending."
        by_why: dict[str, list[str]] = {}
        for k, why in sorted(self.pending.items()):
            by_why.setdefault(why, []).append(k)
        rows = [["Facts, sections or figures", "Waits for"]]
        for why, keys in sorted(by_why.items(), key=lambda kv: kv[1][0]):
            groups: dict[str, list[str]] = {}
            for k in keys:
                ns = k.split(".", 1)[0] if " " not in k else k
                groups.setdefault(ns, []).append(k)
            shown = []
            for ns, ks in groups.items():
                if len(ks) > 3:
                    shown.append(f"`{ns}.*` ({len(ks)})")
                else:
                    shown += [f"`{k}`" for k in ks]
            rows.append([", ".join(shown), why])
        return _table(rows)

    # whole report
    def run(self, templates: list[Path]) -> str:
        parts = []
        for t in templates:
            where = t.name
            text = t.read_text(encoding="utf-8")
            # template links are written for the template's folder (docs/report/src)
            text = rebase_links(text, REPORT_SRC, self.out_dir, self.root)

            def directive(m: re.Match, where=where) -> str:
                kind, spec = m.group(1), m.group(2)
                if kind == "include":
                    return self.include(spec, where)
                return self.generate(spec, where)

            text = PLACEHOLDER.sub(lambda m: self.fact(m.group(1)), text)
            text = DIRECTIVE.sub(directive, text)
            parts.append(text.strip())
        md = "\n\n".join(parts) + "\n"
        md = self.check_images(md)
        return md.replace("<!-- generate: pending -->", self.pending_table())

    def check_images(self, md: str) -> str:
        def img(m: re.Match) -> str:
            if not m.group(1).startswith("!"):
                return m.group(0)
            target = m.group(2)
            if re.match(r"^[a-z][a-z0-9+.-]*:", target):
                raise ReportError(f"image {target}: only local figures are allowed")
            path = (self.root / self.out_dir / target).resolve()
            if path.is_file():
                return m.group(0)
            rel = Path(os.path.relpath(path, self.root.resolve())).as_posix()
            if any(
                rel == d.as_posix() or rel.startswith(d.as_posix() + "/") for d in GENERATED_FIGURES
            ):
                self.pending.setdefault(
                    f"figure {rel}", "generated by tasks.py app-shots / report (not made yet)"
                )
                return f"> {PENDING}: figure `{rel}` is generated by a later step."
            raise ReportError(f"image {rel} not found")

        return LINK.sub(img, md)


def final_keys_rows(facts: dict) -> list[str]:
    return sorted(
        k
        for k in facts
        if k.startswith("final.")
        and not k.startswith("final.probe.")
        and k not in ("final.available", "final.layout")
    )


def _final_layout(keys: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    rows, labels = [], []
    for k in keys:
        if k.endswith(".rel"):
            continue
        _, user, kpi, lb = k.split(".", 3)
        if (user, kpi) not in rows:
            rows.append((user, kpi))
        label = {
            "s0": "S0",
            "s1": "S1",
            "s2": "S2",
            "s3": "S3",
            "s2_nop": "S2_noP",
            "s4": "S4",
        }.get(lb, lb)
        if label not in labels:
            labels.append(label)
    order = ["S0", "S1", "S2", "S3", "S2_noP", "S4"]
    labels.sort(key=lambda x: order.index(x) if x in order else 99)
    return rows, labels


def _rel(path: Path, out_dir: Path) -> str:
    return Path(os.path.relpath(path, out_dir)).as_posix()


def _esc(cell) -> str:
    return str(cell).replace("|", "\\|").replace("\n", " ")


def _table(rows: list[list]) -> str:
    head, *body = rows
    out = ["| " + " | ".join(_esc(c) for c in head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(_esc(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def templates(src: Path = REPORT_SRC) -> list[Path]:
    found = sorted(Path(src).glob("[0-9][0-9]_*.md"))
    if not found:
        raise ReportError(f"no templates NN_*.md in {Path(src).as_posix()}")
    return found


def assemble(
    src: Path, facts: dict, out: Path = REPORT_MD, *, root: Path = Path(".")
) -> tuple[str, list[str]]:
    """(report markdown, PENDING keys): the templates of `src` in name order with every
    directive and placeholder resolved, for a report that lives at `out` (repo-relative;
    docs/report/report.md)."""
    out = Path(out)
    if out.is_absolute():
        out = Path(os.path.relpath(out, Path(root).resolve()))
    a = Assembler(facts, out, root)
    md = a.run([Path(root) / t if not t.is_absolute() else t for t in templates(Path(root) / src)])
    return md, sorted(a.pending)


# -- lint ------------------------------------------------------------------------------
LINT_RULES = (
    (re.compile(r"\d+\.\d+"), "decimal number"),
    (re.compile(r"\d+ ?%"), "percentage"),
    (re.compile(r"\bp ?[<=>≤≥]"), "p comparison"),
)
# A typed result claim: a controller (or RL, preemption, a policy) and a comparative
# verb in one clause. Result sentences are generated facts (D113); quoted labels such
# as "S3 better" (an outcome name) are not claims.
LINT_CLAIM = re.compile(
    r"\b(?:S[0-4](?:_noP)?|RL|preemption|controllers?|polic(?:y|ies))\b[^.;:!?]*?"
    r"\b(?:beats?|better|worse|lowers?|raises?|cuts?|reduces?|shortens?|lengthens?|"
    r"improves?|outperforms?)\b",
    re.I,
)
LINT_ALLOWED = re.compile(
    r"(?:\bspecs?|§|\bsections?|\bchapters?|\bSUMO|\bPython|\bversion|\bv)\s?\d+(?:\.\d+)+"
    r"(?:\s?(?:to|and|-|,)\s?\d+(?:\.\d+)+)*",
    re.I,
)


def lint(text: str) -> list[str]:
    """Numbers and result claims typed into a template outside placeholders, directives
    and code: each result must be a fact (D113). Integers (seed counts, design
    constants) are fine."""
    problems = []
    fence = False
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        clean = DIRECTIVE.sub(" ", line)
        clean = re.sub(r"<!--.*?-->", " ", clean)
        clean = PLACEHOLDER.sub(" ", clean)
        clean = re.sub(r"`[^`]*`", " ", clean)
        clean = re.sub(r"\]\([^)]*\)", "]", clean)  # link targets
        clean = LINT_ALLOWED.sub(" ", clean)
        for rx, what in LINT_RULES:
            for m in rx.finditer(clean):
                problems.append(f"line {n}: {what} {m.group(0)!r}")
        unquoted = re.sub(r'"[^"]*"|“[^”]*”', " ", clean)
        for m in LINT_CLAIM.finditer(unquoted):
            problems.append(f"line {n}: typed result claim {m.group(0)!r}")
    return problems

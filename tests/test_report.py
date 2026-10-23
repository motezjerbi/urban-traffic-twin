"""The English report (D113, D115): facts, assembly, template lint, PDF, acceptance.

Every source below is SYNTHETIC (written into tmp_path); no test reads data/ or runs a
simulation. The tests against the committed report (docs/report/report.md, report.pdf)
skip until integration has generated them.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("reportlab")
pytest.importorskip("markdown")

from traffic_twin.report import __main__ as cli  # noqa: E402
from traffic_twin.report.acceptance import acceptance  # noqa: E402
from traffic_twin.report.assemble import (  # noqa: E402
    GENERATED_FIGURES,
    LINK,
    PENDING,
    PLACEHOLDER,
    REPORT_MD,
    ReportError,
    assemble,
    filter_rows,
    lint,
    nth_table,
    section,
    select_cols,
    templates,
)
from traffic_twin.report.facts import (  # noqa: E402
    NBSP,
    FactsError,
    collect_facts,
    declared_keys,
    dump,
    fmt,
    load_facts,
    pct,
    write_facts,
)
from traffic_twin.report.pdf import PdfError, render_pdf, table_font_sizes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "report" / "src"
KEYS = ROOT / "docs" / "report" / "keys.txt"
COMMITTED_MD = ROOT / "docs" / "report" / "report.md"
COMMITTED_FACTS = ROOT / "docs" / "report" / "facts.json"
COMMITTED_PDF = ROOT / "docs" / "report" / "report.pdf"
COMMITTED_BUILD = ROOT / "docs" / "report" / "build.json"


# -- synthetic sources -----------------------------------------------------------------
def _w(root: Path, rel: str, obj) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, str):
        p.write_text(obj, encoding="utf-8")
    else:
        p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def make_root(tmp: Path, *, part8: bool = False, sessions: int | None = None) -> Path:
    """A SYNTHETIC repo root: the real configs and meta.yaml plus made-up outputs.
    With `part8`, also the Part 8 outputs (a06.json, final_table.csv, S3 in summary)."""
    root = tmp / "root"
    for rel in (
        "configs/evaluation.yaml",
        "configs/validation.yaml",
        "configs/eta.yaml",
        "configs/pitie_rl_s3.yaml",
        "docs/report/meta.yaml",
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, root / rel)
    seeds = list(range(1, 11))
    pdm = {
        lb: {"person_delay_mean": v}
        for lb, v in {"S0": 50.0, "S1": 45.0, "S2": 40.0, "S2_noP": 40.5}.items()
    }
    if part8:
        pdm["S3"] = {"person_delay_mean": 38.0}
        pdm["S4"] = {"person_delay_mean": 39.0}
    _w(
        root,
        "data/results/evaluation/summary.json",
        {
            "preregistered": "2026-10-12",
            "seeds": seeds,
            "a03": {
                "n": 4 if part8 else 3,
                "of": 4,
                "text": "SYNTHETIC",
                "n_kpis": {"main": 83, "ambulance_probe": 83},
            },
            "a04": {"verdict": "PASS", "pass": True, "runs": 110 if part8 else 70},
            "a05": {
                "verdict": "PASS",
                "primary": {
                    "hl": -12.5,
                    "hl_ci": [-20.0, -5.0],
                    "p_one_sided": 0.001953125,
                    "n": 10,
                },
            },
            "hidden_degradation": [
                {"family": "main", "scenario": "S2", "hidden_degradation": True}
            ],
            "person_delay": {"main": pdm},
        },
    )
    _w(
        root,
        "data/results/evaluation/main/a04.csv",
        "family,scenario,ped_curb_wait_max,ped_max_wait_s\nmain,x,87.0,90.0\nmain,y,88.0,90.0\n",
    )
    _w(
        root,
        "data/results/evaluation/main/kpis.csv",
        "family,label,scenario,controller,seed,user_type,kpi,value\n"
        "main,S0,a,fixed,1,bus,headway_std,20.0\nmain,S0,a,fixed,2,bus,headway_std,22.0\n"
        "main,S2,b,rules,1,bus,headway_std,19.0\n",
    )
    _w(
        root,
        "data/results/validation/pitie_fixed/summary.json",
        {
            "seeds": seeds,
            "code_version": "synthetic",
            "preregistered": "2026-10-11",
            "a02": {"verdict": "PASS", "rows": 6, "pass": 6},
            "geh": {"n_counts": 10, "n_below": 9, "share": 0.9},
            "signal_status": {"NOT_MEASURED": 41},
            "queue_status": {"NOT_MEASURED": 6},
            "calibration": {"status": "converged"},
        },
    )
    _w(
        root,
        "data/rl7/devtest/verdict.json",
        {
            "verdict": "fallback",
            "seeds": list(range(3020, 3030)),
            "code_version": "syn",
            "C1": {
                "hl": -9.0,
                "hl_ci": [-15.0, -6.0],
                "p": 0.002,
                "p_holm": 0.0137,
                "s1_mean": 57.0,
                "s3_mean": 47.0,
                "pass": True,
            },
            "C2": {
                "limit_pct": 5.0,
                "pass": False,
                "rows": [
                    {"kpi": "car/time_loss_mean", "rel_pct": -10.0},
                    {"kpi": "bus/delay_mean", "rel_pct": -20.0},
                    {"kpi": "pedestrian/ped_curb_wait_mean", "rel_pct": 6.0},
                ],
            },
            "C3": {"violations": [], "pass": True},
            "C4": {"runs_below_s1": 5, "of": 5, "runs_pass": True},
            "s2_framing": {"text": "SYNTHETIC framing", "hl": -1.5, "p": 0.05},
        },
    )
    _w(root, "data/models/eta/LATEST", "eta-syn\n")
    _w(
        root,
        "data/models/eta/eta-syn/eval.json",
        {
            "model_version": "eta-syn",
            "code_version": "syn",
            "a07": {
                "pass": True,
                "coverage": 0.8,
                "ci": [0.79, 0.81],
                "band": [0.75, 0.85],
                "mae": 10.0,
                "ref_mae": 40.0,
                "n_worlds": 72,
                "n": 1000,
            },
        },
    )
    if part8:
        _w(
            root,
            "data/results/evaluation/a06.json",
            {
                "status": "PASS",
                "problems": [],
                "family": "main",
                "candidate": "S3",
                "kpi": "all/person_delay_mean",
                "n_required": 10,
                "primary": "S1",
                "holm": "comparisons",
                "comparisons": [
                    {
                        "reference": r,
                        "n": 10,
                        "mean_ref": m,
                        "sd_ref": 2.0,
                        "mean_s3": 38.0,
                        "sd_s3": 2.0,
                        "hl": 38.0 - m,
                        "ci_low": 38.0 - m - 1,
                        "ci_high": 38.0 - m + 1,
                        "p": 0.002,
                        "p_holm": 0.006,
                        "outcome": "S3 better",
                        "sentence": f"SYNTHETIC: S3 better than {r}.",
                    }
                    for r, m in (("S1", 45.0), ("S2", 40.0), ("S0", 50.0))
                ],
                "model": {"model_sha256": "ab" * 32, "inputs_ok": True, "config_hash": "x"},
                "registration": {
                    "commit": "abc1234",
                    "committed_at": "2026-10-24T10:00:00+02:00",
                    "first_s3_started_at": "2026-10-24T09:00:00+00:00",
                    "order": "before",
                },
                "replication_10_6": {
                    "c1": {"hl": -7.0, "p": 0.002, "pass": True},
                    "c2": {
                        "rows": [
                            {"kpi": "pedestrian/ped_curb_wait_mean", "rel_pct": 6.5, "worse": True}
                        ],
                        "pass": False,
                    },
                    "c3": {"violations": [], "pass": True},
                    "c4": {"cited": "pass", "source": "data/rl7/devtest/verdict.json"},
                    "confirms_fallback": True,
                    "main_solution": "S2",
                    "text": "SYNTHETIC text.",
                },
                "equity_vs": {"S1": [], "S2": []},
            },
        )
        rows = ["section,user,kpi,label,scenario,ablation,n,mean,sd,rel_vs_s0_pct,unit,better_if"]
        for lb, v in (
            ("S0", 50.0),
            ("S1", 45.0),
            ("S2", 40.0),
            ("S3", 38.0),
            ("S2_noP", 40.5),
            ("S4", 39.0),
        ):
            rel = 100 * (v - 50.0) / 50.0
            rows.append(
                f"main,all,person_delay_mean,{lb},x,{lb in ('S2_noP', 'S4')},10,{v},"
                f"1.0,{rel},s,lower"
            )
            rows.append(f"main,car,time_loss_mean,{lb},x,False,10,{v + 10},1.0,{rel},s,lower")
        rows.append("safety,all,a04,S3,x,False,10,0,0,,,")
        _w(root, "data/results/evaluation/final_table.csv", "\n".join(rows) + "\n")
    if sessions is not None:
        _w(
            root,
            "docs/usertests/synthesis.json",
            {"a10": {"status": "DONE" if sessions >= 3 else "PENDING", "n": sessions}},
        )
    return root


@pytest.fixture
def pending_facts(tmp_path) -> dict:
    return collect_facts(make_root(tmp_path))


@pytest.fixture
def part8_facts(tmp_path) -> dict:
    return collect_facts(make_root(tmp_path, part8=True, sessions=3))


# -- facts -----------------------------------------------------------------------------
def test_formatting_uses_a_point_and_a_non_breaking_space():
    assert fmt(63.5581, 1, "s") == f"63.6{NBSP}s"
    assert fmt(-0.04, 1, "s", signed=True) == f"0.0{NBSP}s"
    assert pct(6.196, 1, signed=True) == f"+6.2{NBSP}%"
    assert "," not in fmt(1234.5, 1)


def test_missing_sources_are_pending_with_a_reason(tmp_path):
    root = tmp_path / "empty"
    (root / "configs").mkdir(parents=True)
    shutil.copy(ROOT / "configs" / "evaluation.yaml", root / "configs" / "evaluation.yaml")
    facts = collect_facts(root)
    for key in (
        "val.a02.verdict",
        "eval.a05.hl",
        "a06.status",
        "eta.a07.coverage_pct",
        "final.all.person_delay_mean.s3",
        "usertests.status",
        "demo.a08",
    ):
        assert facts[key]["text"] is None, key
        assert facts[key]["pending"], key
    assert facts["acceptance.A-06"]["value"] == "PENDING"


def test_facts_carry_value_text_source_and_sha(pending_facts):
    f = pending_facts["eval.a05.hl"]
    assert f["value"] == -12.5 and f["text"] == f"-12.5{NBSP}s" and f["unit"] == "s"
    assert f["source"] == "data/results/evaluation/summary.json"
    assert re.fullmatch(r"[0-9a-f]{64}", f["source_sha256"])
    assert pending_facts["val.a02.verdict"]["code_version"] == "synthetic"
    assert pending_facts["eval.person_delay.rel_s2_vs_s0"]["text"] == f"-20.0{NBSP}%"
    assert pending_facts["eval.a04.worst_curb_wait"]["text"] == f"88{NBSP}s"


def test_no_part8_value_before_the_part8_run(pending_facts):
    for key, f in pending_facts.items():
        if key == "final.layout":  # the registered layout (config), not a result
            continue
        if key.startswith(("a06.", "rep106.", "final.")) or key.endswith(".s3"):
            assert f["text"] is None, key


def test_part8_facts_from_a06_and_the_final_table(part8_facts):
    assert part8_facts["a06.status"]["text"] == "PASS"
    assert part8_facts["a06.primary_sentence"]["text"] == "SYNTHETIC: S3 better than S1"
    assert part8_facts["a06.s1.hl"]["text"] == f"-7.0{NBSP}s"
    assert part8_facts["a06.s1.p_holm"]["text"] == "0.0060"
    assert part8_facts["rep106.confirms"]["text"] == "confirm"
    assert part8_facts["final.all.person_delay_mean.s3"]["text"] == f"38.0{NBSP}s"
    assert part8_facts["final.all.person_delay_mean.s3.rel"]["text"] == f"-24.0{NBSP}%"
    assert part8_facts["final.available"]["value"] is True


def test_a06_with_an_unknown_status_is_an_error(tmp_path):
    root = make_root(tmp_path, part8=True)
    p = root / "data/results/evaluation/a06.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["status"] = "FAIL"  # A-06 is never FAIL on direction (D94)
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(FactsError, match="PASS or INVALID"):
        collect_facts(root)


def test_facts_json_round_trip(tmp_path, pending_facts):
    out = write_facts(pending_facts, tmp_path / "facts.json")
    assert load_facts(out) == json.loads(dump(pending_facts, "unknown"))["facts"]
    (tmp_path / "bad.json").write_text('{"schema": "other"}', encoding="utf-8")
    with pytest.raises(FactsError):
        load_facts(tmp_path / "bad.json")


# -- acceptance (D115) -----------------------------------------------------------------
def _status(facts: dict) -> dict:
    return {r["id"]: r for r in acceptance(facts)}


def test_acceptance_pending_before_part8(pending_facts):
    rows = _status(pending_facts)
    assert list(rows) == [f"A-{i:02d}" for i in range(1, 11)]
    assert rows["A-02"]["status"] == "PASS"
    assert rows["A-05"]["status"] == "PASS"
    assert rows["A-07"]["status"] == "PASS"
    for cid in ("A-01", "A-03", "A-04", "A-06", "A-08", "A-09", "A-10"):
        assert rows[cid]["status"] == "PENDING", cid
    assert "3 of 4" in rows["A-03"]["evidence"]


def test_acceptance_after_part8(part8_facts):
    rows = _status(part8_facts)
    assert rows["A-03"]["status"] == "PASS"
    assert rows["A-04"]["status"] == "PASS"
    assert rows["A-06"]["status"] == "PASS"
    assert rows["A-06"]["evidence"] == "presented: SYNTHETIC: S3 better than S1"
    assert rows["A-10"]["status"] == "DONE"


@pytest.mark.parametrize(
    ("n", "status"), [(0, "PENDING"), (2, "PENDING"), (3, "DONE"), (5, "DONE"), (6, "DONE")]
)
def test_a10_is_done_from_three_sessions_and_never_pass(tmp_path, n, status):
    row = _status(collect_facts(make_root(tmp_path, sessions=n)))["A-10"]
    assert row["status"] == status
    assert ("over the spec range" in row["evidence"]) == (n > 5)


def test_a06_invalid_is_shown_with_its_problems(tmp_path):
    root = make_root(tmp_path, part8=True)
    p = root / "data/results/evaluation/a06.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d.update(status="INVALID", problems=["S1: n = 9 < 10"])
    p.write_text(json.dumps(d), encoding="utf-8")
    row = _status(collect_facts(root))["A-06"]
    assert row["status"] == "INVALID" and "n = 9" in row["evidence"]


# -- includes and selection (loud failures) --------------------------------------------
PAGE = """# Page

## 1. Intro

text

## 2. KPIs per user and scenario

| User | KPI | S0 | S1 |
|---|---|---|---|
| car | `time_loss_mean` | 1 | 2 |
| bus | `delay_mean` | 3 | 4 |

| User | KPI | S0 | S1 |
|---|---|---|---|
| car | `time_loss_mean` | 9 | 9 |

### A-04

| Family | Runs |
|---|---|
| main | 10 |

## 3. Next
"""


def test_section_matches_the_heading_without_its_number():
    body = section(PAGE, "KPIs per user and scenario")
    assert "time_loss_mean" in body and "A-04" in body and "Next" not in body
    assert section(PAGE, "A-04").strip().startswith("| Family")
    with pytest.raises(ReportError, match="no heading"):
        section(PAGE, "Renamed section")


def test_rows_keep_the_first_table_and_a_missing_row_fails():
    body = section(PAGE, "KPIs per user and scenario")
    out = filter_rows(body, [("bus", "delay_mean")])
    assert out.count("\n") == 2 and "| 9 |" not in out
    with pytest.raises(ReportError, match="rows not found: car wait_p95"):
        filter_rows(body, [("car", "wait_p95")])


def test_cols_and_table_selection_fail_loudly():
    body = section(PAGE, "KPIs per user and scenario")
    assert "| User | S1 |" in select_cols(nth_table(body, "2"), ["User", "S1"])
    with pytest.raises(ReportError, match="columns not found: S3"):
        select_cols(body, ["User", "S3"])
    with pytest.raises(ReportError, match="table 4 not found"):
        nth_table(body, "4")


def _template_root(tmp_path: Path, text: str, page: str = PAGE) -> Path:
    root = tmp_path / "troot"
    _w(root, "docs/page.md", page)
    _w(root, "docs/report/src/00_t.md", text)
    return root


def test_include_requires_renders_pending_until_the_heading_exists(tmp_path):
    text = "<!-- include: docs/page.md#Final table (Part 8) | requires: Final table (Part 8) -->"
    root = _template_root(tmp_path, text)
    md, pending = assemble(Path("docs/report/src"), {}, REPORT_MD, root=root)
    assert PENDING in md and any("Final table (Part 8)" in p for p in pending)
    # once the marker heading exists, the include must resolve
    root2 = _template_root(
        tmp_path / "b",
        text.replace("#Final table", "#Missing table"),
        PAGE + "\n## 4. Final table (Part 8)\n\nx\n",
    )
    with pytest.raises(ReportError, match="no heading 'Missing table"):
        assemble(Path("docs/report/src"), {}, REPORT_MD, root=root2)


def test_include_errors_are_loud(tmp_path):
    for text, match in (
        ("<!-- include: docs/nope.md#Intro -->", "not found"),
        ("<!-- include: docs/page.md#Gone -->", "no heading"),
        ("<!-- include: docs/page.md#Intro | colz: a -->", "use rows:"),
        ("<!-- generate: nonsense -->", "unknown generator"),
    ):
        root = _template_root(tmp_path / match.replace(" ", "_").replace(":", ""), text)
        with pytest.raises(ReportError, match=match):
            assemble(Path("docs/report/src"), {}, REPORT_MD, root=root)


def test_included_links_are_rebased_onto_the_report_folder(tmp_path):
    page = "## 1. Fig\n\n![x](figures/a.png)\n"
    root = _template_root(tmp_path, "<!-- include: docs/page.md#Fig -->", page)
    _w(root, "docs/figures/a.png", "png")
    md, _ = assemble(Path("docs/report/src"), {}, REPORT_MD, root=root)
    assert "![x](../figures/a.png)" in md


def test_unresolved_placeholder_is_pending_and_listed(tmp_path):
    root = _template_root(tmp_path, "Value {{a.b}} and {{c.d}}.\n\n<!-- generate: pending -->")
    facts = {"a.b": {"text": "42"}, "c.d": {"text": None, "pending": "SYNTHETIC wait"}}
    md, pending = assemble(Path("docs/report/src"), facts, REPORT_MD, root=root)
    assert "Value 42 and **PENDING**." in md
    assert pending == ["c.d"] and "SYNTHETIC wait" in md


def test_missing_image_fails_unless_generated_later(tmp_path):
    root = _template_root(tmp_path, "![a](../../figures/none.png)")
    with pytest.raises(ReportError, match="image docs/figures/none.png not found"):
        assemble(Path("docs/report/src"), {}, REPORT_MD, root=root)
    root = _template_root(tmp_path / "g", "![a](../../figures/app/home.png)")
    md, pending = assemble(Path("docs/report/src"), {}, REPORT_MD, root=root)
    assert PENDING in md and pending == ["figure docs/figures/app/home.png"]


# -- the real templates ----------------------------------------------------------------
def test_templates_pass_the_number_lint():
    problems = {t.name: lint(t.read_text(encoding="utf-8")) for t in templates(SRC)}
    assert {k: v for k, v in problems.items() if v} == {}


def test_lint_flags_typed_results():
    assert lint("delay fell by 23 % (p < 0.05), to 48.9 s") != []
    assert lint("see spec 13.3 and § 4.2; seeds 1-10; {{x.y}}; `k = 3.5`") == []


def test_keys_txt_lists_exactly_the_template_placeholders():
    used = set()
    for t in templates(SRC):
        used |= set(PLACEHOLDER.findall(t.read_text(encoding="utf-8")))
    listed = {
        ln.strip()
        for ln in KEYS.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    }
    assert listed == used
    assert listed <= declared_keys(ROOT)  # every key is one report.facts can produce


def test_template_images_resolve_or_are_generated_later():
    for t in templates(SRC):
        for m in LINK.finditer(t.read_text(encoding="utf-8")):
            if not m.group(1).startswith("!"):
                continue
            rel = (SRC / m.group(2)).resolve().relative_to(ROOT).as_posix()
            later = any(
                rel == d.as_posix() or rel.startswith(d.as_posix() + "/") for d in GENERATED_FIGURES
            )
            assert later or (ROOT / rel).is_file(), f"{t.name}: {rel}"


def test_report_builds_with_every_part8_fact_pending(tmp_path, pending_facts):
    md, pending = assemble(Path("docs/report/src"), pending_facts, REPORT_MD, root=ROOT)
    assert "a06.status" in pending and "final.available" in pending
    assert "Appendix E. PENDING items" in md and PENDING in md
    assert "{{" not in md and "<!-- include" not in md and "<!-- generate" not in md
    pages = render_pdf(md, ROOT / "docs" / "report", tmp_path / "r.pdf", "Simulated data")
    assert pages > 10
    assert (tmp_path / "r.pdf").read_bytes()[:5] == b"%PDF-"
    assert min(table_font_sizes(md, ROOT / "docs" / "report")) >= 6.5


def test_report_with_part8_facts_has_the_final_table(part8_facts):
    md, pending = assemble(Path("docs/report/src"), part8_facts, REPORT_MD, root=ROOT)
    assert "a06.status" not in pending and "final.available" not in pending
    assert f"38.0{NBSP}s (-24.0{NBSP}%)" in md  # S3 person delay and its change vs S0
    assert "SYNTHETIC: S3 better than S1" in md


# -- PDF ------------------------------------------------------------------------------
def test_too_wide_table_is_a_build_error(tmp_path):
    cols = [f"`column_with_a_long_identifier_{i}`" for i in range(14)]
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * 14 + "\n| " + " | ".join(cols) + " |\n"
    with pytest.raises(PdfError, match="6.5 pt"):
        render_pdf(md, tmp_path, tmp_path / "w.pdf", "x")


def test_pdf_is_invariant(tmp_path):
    md = "# T\n\n<!-- toc -->\n\n## 1. A\n\nText é ± ≤ → ×.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    render_pdf(md, tmp_path, tmp_path / "a.pdf", "footer")
    render_pdf(md, tmp_path, tmp_path / "b.pdf", "footer")
    assert (tmp_path / "a.pdf").read_bytes() == (tmp_path / "b.pdf").read_bytes()


def test_cli_builds_from_a_facts_file(tmp_path, pending_facts):
    facts = write_facts(pending_facts, tmp_path / "facts.json")
    assert (
        cli.main(
            ["report", "--facts", str(facts), "--out-dir", str(tmp_path / "out"), "--no-figures"]
        )
        == 0
    )
    build = json.loads((tmp_path / "out" / "build.json").read_text(encoding="utf-8"))
    assert build["platform"] == cli.fingerprint() and build["pages"] > 10
    assert (tmp_path / "out" / "report.md").is_file()


def test_report_figures_are_reproducible(tmp_path, part8_facts):
    from traffic_twin.report.figures import make_figures

    a = make_figures(part8_facts, tmp_path / "a")
    b = make_figures(part8_facts, tmp_path / "b")
    assert [p.name for p in a] == ["pipeline.png", "person_delay.png"]
    assert all(x.read_bytes() == y.read_bytes() for x, y in zip(a, b, strict=True))


# -- the committed report (generated in integration) -----------------------------------
needs_committed = pytest.mark.skipif(
    not (COMMITTED_MD.is_file() and COMMITTED_FACTS.is_file()),
    reason="docs/report/report.md is generated in integration (tasks.py report)",
)


@needs_committed
def test_committed_report_md_matches_the_committed_facts():
    md, _ = assemble(Path("docs/report/src"), load_facts(COMMITTED_FACTS), REPORT_MD, root=ROOT)
    assert md == COMMITTED_MD.read_text(encoding="utf-8")


@needs_committed
def test_committed_pdf_matches_a_rebuild_on_the_same_platform(tmp_path):
    if not COMMITTED_BUILD.is_file():
        pytest.skip("no docs/report/build.json")
    build = json.loads(COMMITTED_BUILD.read_text(encoding="utf-8"))
    if build.get("platform") != cli.fingerprint():
        pytest.skip("PDF bytes are compared only on the platform that built them (D113)")
    facts = load_facts(COMMITTED_FACTS)
    md = COMMITTED_MD.read_text(encoding="utf-8")
    render_pdf(
        md,
        ROOT / "docs" / "report",
        tmp_path / "r.pdf",
        cli.footer_text(facts),
        title=f"{facts['meta.title']['text']}: final report",
        author=facts["meta.authors"]["text"],
    )
    assert (tmp_path / "r.pdf").read_bytes() == COMMITTED_PDF.read_bytes()


# -- app screenshots (no browser needed) -----------------------------------------------
def test_screenshot_plan_follows_the_demo_steps():
    from traffic_twin.report.shots import ShotsError, _slug, load_demo_config, plan, step_url

    cfg = load_demo_config(ROOT / "configs" / "demo.yaml")
    shots = plan(cfg, "http://127.0.0.1:9", None)
    names = [n for n, _, _ in shots]
    mock_steps = [s["id"] for s in cfg["steps"] if s.get("front") != "streamlit"]
    assert names == [*mock_steps, "error"]
    url = {n: u for n, u, _ in shots}["pedestrian"]
    assert url.startswith("http://127.0.0.1:9/mock?view=pedestrian&scenario=pitie_rules&seed=4000")
    assert "t=1805" in url and "crossing=S" in url and "speed=normal" in url
    assert "kiosk=1" in {n: u for n, u, _ in shots}["ambulance"]
    with pytest.raises(ShotsError, match="unknown run"):
        step_url({"id": "x", "view": "bus", "run": "nope"}, cfg, "http://h")
    assert _slug("Sécurité et ambulance") == "securite_et_ambulance"


# -- result sentences are facts, never hand prose (D113) -------------------------------
def _patch_json(root: Path, rel: str, **changes) -> None:
    p = root / rel
    d = json.loads(p.read_text(encoding="utf-8"))
    d.update(changes)
    p.write_text(json.dumps(d), encoding="utf-8")


def test_lint_flags_typed_result_claims():
    for claim in (
        "S3 is better than S1 for cars, buses and the ambulance.",
        "The selected RL policy S3 beat the actuated baseline on delay.",
        "Every controller that lowers the person-weighted delay also pays.",
        "Emergency preemption cuts the ambulance travel time.",
    ):
        assert lint(claim) != [], claim
    # an outcome label in quotes and a placeholder are not claims
    assert lint('The outcome is "S3 better" or "S3 worse" when Holm is small.') == []
    assert lint("S2 {{eval.person_delay.s2_verb}} the delay; {{eval.equity.sentence}}.") == []


def test_equity_sentence_names_the_flagged_controllers_and_groups(tmp_path):
    root = make_root(tmp_path)
    rows = [
        {
            "family": "main",
            "scenario": "S1",
            "global_better": True,
            "flagged": ["bus.delay_mean__flow_24_W", "pedestrian.ped_curb_wait_mean__x"],
            "hidden_degradation": True,
        },
        {"family": "main", "scenario": "S3", "global_better": True, "flagged": []},
        {"family": "main", "scenario": "S4", "global_better": False, "flagged": ["car.x"]},
    ]
    _patch_json(root, "data/results/evaluation/summary.json", hidden_degradation=rows)
    f = collect_facts(root)["eval.equity.sentence"]
    assert f["text"].startswith("1 of the 2 scenarios (family and controller")
    assert "main S1 (bus, pedestrian)" in f["text"] and "S3" not in f["text"]
    assert "each of them" not in f["text"]  # S3 lowers the delay without a flag
    assert f["value"] == {"main S1": ["bus", "pedestrian"]}
    rows[1].update(flagged=["car.delay_total_mean__arm_S"], hidden_degradation=True)
    _patch_json(root, "data/results/evaluation/summary.json", hidden_degradation=rows)
    text = collect_facts(root)["eval.equity.sentence"]["text"]
    assert text.startswith("2 of the 2") and "main S3 (car)" in text and "each of them" in text
    _patch_json(root, "data/results/evaluation/summary.json", hidden_degradation=[])
    assert collect_facts(root)["eval.equity.sentence"]["text"].startswith("No controller")


def test_a05_and_person_delay_verbs_follow_the_sign(tmp_path):
    from traffic_twin.report.facts import a05_sentence, change_verb

    assert a05_sentence(-12.5) == (
        f"shortens the ambulance travel time by a Hodges-Lehmann shift of 12.5{NBSP}s"
    )
    assert a05_sentence(4.0).startswith("lengthens") and "unchanged" in a05_sentence(0.0)
    assert (change_verb(-3), change_verb(2), change_verb(0)) == (
        "lowers",
        "raises",
        "leaves unchanged",
    )
    facts = collect_facts(make_root(tmp_path))
    assert facts["eval.a05.sentence"]["text"].startswith("shortens")
    assert facts["eval.person_delay.s2_verb"]["text"] == "lowers"


def test_part7_sentences_come_from_the_verdict_file(tmp_path):
    facts = collect_facts(make_root(tmp_path))
    crit = facts["rl7.criteria_sentence"]["text"]
    assert "C1, the person-weighted delay against the actuated baseline S1, is a pass" in crit
    assert "C2, equity against S1 within 5" in crit and "is a fail" in crit
    assert "C3, safety, is a pass with 0 violations" in crit  # a check, not a win over S1
    assert facts["rl7.failed"]["text"] == "C2"
    c2 = facts["rl7.c2.sentence"]["text"]
    assert "the car time loss by -10.0" in c2 and "(lower, better)" in c2
    assert f"the pedestrian mean curb wait by +6.0{NBSP}% (higher, worse)" in c2
    assert "ambulance" not in c2  # no fact covers it
    assert facts["rl7.verdict_sentence"]["text"].startswith("Spec 10.6 applies")
    # another verdict file gives other sentences: nothing is typed in the templates
    root = make_root(tmp_path / "b")
    _patch_json(
        root,
        "data/rl7/devtest/verdict.json",
        verdict="pass",
        C1={"hl": 2.0, "p": 0.3, "p_holm": 0.6, "pass": False},
    )
    facts = collect_facts(root)
    assert "S1, is a fail (HL +2.0" in facts["rl7.criteria_sentence"]["text"]
    assert facts["rl7.failed"]["text"] == "C1, C2"
    # the frozen "pass" is never stated alone: C1 and C2 fail, so the page says fallback
    assert facts["rl7.verdict_sentence"]["text"].startswith("Spec 10.6 applies")
    # every criterion met and reproduce7 identical: the verdict is a pass
    root = make_root(tmp_path / "c")
    _patch_json(
        root,
        "data/rl7/devtest/verdict.json",
        C2={"limit_pct": 5.0, "pass": True, "rows": [{"kpi": "car/time_loss_mean", "rel_pct": -1}]},
    )
    _w(root, "data/rl7/_reproduce/abc1234/reproduce7.json", {"head": "abc1234", "same": True})
    facts = collect_facts(root)
    assert facts["rl7.verdict_sentence"]["text"].startswith("RL beats")
    assert facts["rl7.c4.pass"]["text"] == "pass"


def test_part7_c4_states_the_reproduce7_result_not_the_frozen_pending(tmp_path):
    """C4 is completed with the newest reproduce7 result (rl.study.final_verdict), as
    evaluation/final.cite_c4 does: the frozen `pending: [C4.repro]` is never stated alone."""
    root = make_root(tmp_path)
    _patch_json(
        root,
        "data/rl7/devtest/verdict.json",
        pending=["C4.repro"],
        C4={"runs_below_s1": 5, "of": 5, "runs_pass": True, "pass": None},
    )
    crit = collect_facts(root)["rl7.criteria_sentence"]["text"]
    assert "C4, reproducibility, is pending (the stage-F runs pass, its repeat check" in crit
    assert "rl7.c4.pass" not in {k for k, f in collect_facts(root).items() if f.get("text")}
    _w(root, "data/rl7/_reproduce/2b41e22/reproduce7.json", {"head": "2b41e22", "same": True})
    facts = collect_facts(root)
    crit = facts["rl7.criteria_sentence"]["text"]
    assert "C4, reproducibility, is a pass (reproduce7 at 2b41e22: identical)" in crit
    assert "pending" not in crit
    assert facts["rl7.c4.pass"]["text"] == "pass"


def test_calibration_rescaling_is_read_from_the_validation_summary(tmp_path):
    root = make_root(tmp_path)
    rel = "data/results/validation/pitie_fixed/summary.json"
    _patch_json(root, rel, calibration={"status": "converged", "recommended": {"N": 1.0}})
    assert collect_facts(root)["val.calibration.rescaled"]["text"].startswith("no arm needed")
    scales = {"N": 1.0, "S": 1.1, "W": 1.0}
    _patch_json(root, rel, calibration={"status": "converged", "recommended": scales})
    f = collect_facts(root)["val.calibration.rescaled"]
    assert f["text"] == "arm scales applied: S 1.10 (the other arms at 1)"
    assert f["source"] == rel


def test_signal_program_facts_are_the_tls_check_of_the_committed_program(tmp_path):
    from traffic_twin.config import get, load_scenario
    from traffic_twin.sim.tls_check import check_scenario, load_program

    cfg = load_scenario(ROOT / "configs" / "pitie_fixed.yaml")
    net = get(cfg, "network.net_file")
    root = tmp_path / "root"
    shutil.copytree(ROOT / "configs", root / "configs")
    (root / net).parent.mkdir(parents=True)
    shutil.copy(ROOT / net, root / net)
    facts = collect_facts(root)
    cfg["network"]["net_file"] = str(ROOT / net)
    report, prog = check_scenario(cfg), load_program(ROOT / net, get(cfg, "network.tls_id"))
    assert facts["net.tls_check"]["text"] == ("OK" if report.ok else "FAILED")
    assert facts["net.phases"]["value"] == len(prog.states)
    assert facts["net.cycle"]["value"] == prog.cycle
    assert facts["net.foe_greens"]["value"] == report.summary["foe_greens"]
    assert facts["net.worst_red"]["value"] == report.worst_crossing_red
    worst = facts["net.worst_red"]["text"]
    for name, red in report.worst_crossing_red.items():
        assert name in worst and f"{red:.0f}{NBSP}s" in worst
    assert facts["net.red_bound"]["value"] == report.crossing_bound_s
    assert facts["net.cycle"]["source"] == net
    # without the program the facts are PENDING, never a typed number
    assert collect_facts(make_root(tmp_path / "b"))["net.cycle"]["text"] is None


def test_a01_evidence_holds_only_what_verify_json_records(tmp_path):
    root = make_root(tmp_path)
    _w(root, "data/results/_rebaseline/abc/verify.json", {"head": "abc", "identical": 70, "of": 70})
    for facts in (collect_facts(make_root(tmp_path / "b")), collect_facts(root)):
        assert "grid regression" not in _status(facts)["A-01"]["evidence"]
    assert _status(collect_facts(root))["A-01"]["evidence"].startswith("70 of 70")


# -- demo evidence: the contract of demo.rehearsal (D116) ------------------------------
def _demo_result(kind: str) -> dict:
    from traffic_twin.demo.readiness import CLONE_CHECKS

    rows = [{"group": "dom home", "check": "page", "result": "PASS"}]
    if kind == "readiness":
        rows += [{"group": "clone", "check": c, "result": "PASS"} for c in CLONE_CHECKS]
    return {
        "kind": kind,
        "status": "PASS",
        "passed": True,
        "browser": "chrome (synthetic)",
        "git": {"code_version": "abc1234", "dirty": False},
        "steps": [{"id": "home", "front": "mock"}],
        "rows": rows,
    }


@pytest.mark.parametrize("kind", ["rehearsal", "readiness"])
def test_demo_facts_read_a_rehearsal_json_written_by_record_rehearsal(tmp_path, kind):
    from traffic_twin.demo.rehearsal import record_rehearsal

    root = make_root(tmp_path)
    doc = record_rehearsal(_demo_result(kind), root / "docs/demo/rehearsal.json", root)
    facts = collect_facts(root)
    for k in ("a08", "a09"):
        assert facts[f"demo.{k}"]["text"] == doc[k]["status"], k
        assert facts[f"demo.{k}.reason"]["value"] == doc[k].get("reason"), k
    assert facts["demo.a08.commit"]["text"] == "abc1234"
    assert facts["demo.a08.date"]["text"] == doc["a08"]["recorded_at"][:10]
    assert facts["demo.fresh_clone"]["value"] is (kind == "readiness")
    row = _status(facts)["A-09"]
    if kind == "readiness":
        assert facts["demo.a09.commit"]["text"] == facts["demo.commit"]["text"] == "abc1234"
        assert facts["demo.a09.date"]["text"] == doc["a09"]["recorded_at"][:10]
        assert "on a fresh clone" in row["evidence"] and "code abc1234" in row["evidence"]
    else:  # no readiness rehearsal: A-09 waits, and says why
        assert facts["demo.a09"]["text"] == "PENDING" and facts["demo.commit"]["text"] is None
        assert doc["a09"]["reason"] in row["evidence"]


@pytest.mark.parametrize(
    ("entries", "valid"),
    [
        ([], 0),
        ([{"date": "2026-11-01", "commit": "abc", "manual_code_edits": None}], 0),
        ([{"date": "2026-11-01", "commit": "abc"}], 0),
        ([{"commit": "abc", "manual_code_edits": 0}], 0),
        ([{"date": "2026-11-01", "manual_code_edits": 0}], 0),
        ([{"date": "2026-11-01", "commit": "abc", "manual_code_edits": 1}], 0),
        ([{"date": "2026-11-01", "commit": "abc", "manual_code_edits": "zero"}], 0),
        ([{"date": "2026-11-01", "commit": "abc", "manual_code_edits": 0}], 1),
    ],
)
def test_human_rehearsals_count_with_the_demo_rule(tmp_path, entries, valid):
    import yaml

    from traffic_twin.demo.rehearsal import human_rehearsals
    from traffic_twin.report.facts import valid_human_rehearsals

    root = make_root(tmp_path)
    log = root / "docs/demo/rehearsal_log.yaml"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(yaml.safe_dump({"entries": entries}), encoding="utf-8")
    assert valid_human_rehearsals({"entries": entries}) == valid
    assert human_rehearsals(log)["valid"] == valid
    facts = collect_facts(root)
    assert facts["demo.human_rehearsals"]["value"] == valid
    assert facts["demo.rehearsed"]["text"] == ("DONE" if valid else None)

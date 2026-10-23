"""Demo prerequisites (configs/demo.yaml `prerequisites`), each with a fix hint. Checked
against the API settings in use (configs/api.yaml, or `TT_API_CONFIG`), so they also hold
for the fixtures and scratch settings.

Each check answers {name, status, detail, hint} with status
- `ok`;
- `pending`: an output only the Part 8/9 integration makes (S3/S4 demo runs, the Part 8
  evaluation files and page): the demo runs, honestly showing "pas encore évalué";
- `missing`: the demo would not work as rehearsed; `demo serve` refuses (unless
  `--allow-missing`) and the rehearsal fails.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from traffic_twin.api.settings import Settings
from traffic_twin.paths import DEMO_PACK, PITIE_RL_S3_CONFIG

PART8_HEADING = "Final comparison table (Part 8)"  # D99 section 1 (text after the number)
PART8_EVAL_FILES = ("final_table.csv", "a06.json", "series.csv")  # what the API reads (D98)
EVAL_FILES = ("main/kpis.csv", "main/equity.csv")  # what /kpi reads (Part 4 outputs)
BOOTSTRAP_HINT = f"python -m traffic_twin.demo bootstrap --from-pack {DEMO_PACK.as_posix()}"


def _row(name: str, status: str, detail: str, hint: str = "") -> dict:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "hint": hint if status != "ok" else "",
    }


def _run_dir(settings: Settings, scenario: str, seed: int) -> Path | None:
    for d in settings.results_dirs:
        p = settings.path(d) / f"{scenario}_seed{seed}"
        if (p / "run.json").is_file():
            return p
    return None


def demo_runs(cfg: dict, settings: Settings) -> dict:
    part8 = set(cfg.get("part8_scenarios") or [])
    missing, pending = [], []
    for name, r in cfg["runs"].items():
        if _run_dir(settings, r["scenario"], r["seed"]) is None:
            (pending if r["scenario"] in part8 else missing).append(
                f"{name}: {r['scenario']} seed {r['seed']}"
            )
    if missing:
        return _row(
            "demo_runs",
            "missing",
            "no run folder for " + ", ".join(missing),
            f"python tasks.py demo-runs (Part 5), or {BOOTSTRAP_HINT}",
        )
    if pending:
        return _row(
            "demo_runs",
            "pending",
            "not made yet: " + ", ".join(pending),
            "python tasks.py demo-runs8 (integration step I3, on the registration commit)",
        )
    return _row("demo_runs", "ok", f"{len(cfg['runs'])} scripted runs found")


def evaluation_outputs(cfg: dict, settings: Settings) -> dict:
    d = settings.path(settings.evaluation_dir)
    missing = [f for f in EVAL_FILES if not (d / f).is_file()]
    if missing:
        return _row(
            "evaluation_outputs",
            "missing",
            f"{d}: no {', '.join(missing)}",
            f"python tasks.py part4 (Part 4 outputs), or {BOOTSTRAP_HINT}",
        )
    p8 = [f for f in PART8_EVAL_FILES if not (d / f).is_file()]
    if p8:
        return _row(
            "evaluation_outputs",
            "pending",
            f"Part 8 outputs not made yet: {', '.join(p8)}",
            "python tasks.py part8 (integration step I2 only)",
        )
    return _row("evaluation_outputs", "ok", f"{d}: Part 4 and Part 8 outputs")


def eta_reference(cfg: dict, settings: Settings) -> dict:
    models = settings.path(settings.eta_models_dir)
    latest = models / "LATEST"
    hint = f"python tasks.py part5 (ETA bundle), or {BOOTSTRAP_HINT}"
    if not latest.is_file():
        return _row("eta_reference", "missing", f"{latest} not found", hint)
    version = latest.read_text(encoding="utf-8").strip()
    ref = models / version / "reference.parquet"
    if not ref.is_file():
        return _row("eta_reference", "missing", f"{ref} not found", hint)
    return _row("eta_reference", "ok", f"ETA bundle {version} (reference table)")


def s3_model_sha(cfg: dict, settings: Settings) -> dict:
    root = settings.repo_root
    hint = "git checkout -- models/rl configs/pitie_rl_s3.yaml (the S3 model is committed, D86)"
    try:
        raw = yaml.safe_load((root / PITIE_RL_S3_CONFIG).read_text(encoding="utf-8")) or {}
        model, want = raw["rl"]["model"], str(raw["rl"]["model_sha256"])
    except (OSError, KeyError, TypeError, yaml.YAMLError) as e:
        return _row("s3_model_sha", "missing", f"{PITIE_RL_S3_CONFIG}: {e}", hint)
    path = root / model
    if not path.is_file():
        return _row("s3_model_sha", "missing", f"{model} not found", hint)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        return _row("s3_model_sha", "missing", f"{model} sha256 {got[:12]} != {want[:12]}", hint)
    return _row("s3_model_sha", "ok", f"{model} sha256 {want[:12]}")


def evaluation_page(cfg: dict, settings: Settings) -> dict:
    page = settings.repo_root / "docs" / "evaluation.md"
    if not page.is_file():
        return _row(
            "evaluation_page",
            "missing",
            "docs/evaluation.md not found",
            "git checkout -- docs/evaluation.md",
        )
    text = page.read_text(encoding="utf-8")
    if not any(ln.startswith("## ") and ln.endswith(PART8_HEADING) for ln in text.splitlines()):
        return _row(
            "evaluation_page",
            "pending",
            f"docs/evaluation.md predates the Part 8 registration (no {PART8_HEADING!r})",
            "python tasks.py part8 (integration step I2 only)",
        )
    return _row("evaluation_page", "ok", "docs/evaluation.md is the Part 8 page")


CHECKS = {
    "demo_runs": demo_runs,
    "evaluation_outputs": evaluation_outputs,
    "eta_reference": eta_reference,
    "s3_model_sha": s3_model_sha,
    "evaluation_page": evaluation_page,
}


def prerequisites(cfg: dict, settings: Settings) -> list[dict]:
    """One row per configured prerequisite (`cfg["prerequisites"]` order)."""
    return [CHECKS[name](cfg, settings) for name in cfg["prerequisites"]]


def format_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        line = f"  [{r['status'].upper():7}] {r['name']}: {r['detail']}"
        if r["hint"]:
            line += f"\n            fix: {r['hint']}"
        out.append(line)
    return "\n".join(out)


def dumps(rows: list[dict]) -> str:
    return json.dumps(rows, indent=1, ensure_ascii=False)

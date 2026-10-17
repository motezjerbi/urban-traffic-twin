"""Part 7 page (`rl.report7`, D83) without SUMO: it renders with every input missing
("Missing: ... Run `tasks.py ...`" placeholders) and with a full synthetic study (runs
of stages T, S and F, campaign, decision_T, selection, published model, stress panel,
sensitivity tables, dev-test look, reproduce7), is deterministic apart from its
code-version line, and computes the interim 10.6 text from verdict.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_rl_sensitivity import make_pick_run, sens_workspace

from traffic_twin.config import load_config
from traffic_twin.evaluation.compare import compare
from traffic_twin.rl import report7
from traffic_twin.rl import sensitivity as sens
from traffic_twin.rl.study import stress_summary
from traffic_twin.viz import rl7 as viz

T_CONFIGS = ("t0", "t1", "v1", "v2", "p1")
LEVEL = {"t0": 81_000.0, "t1": 79_500.0, "v1": 78_000.0, "v2": 79_800.0, "p1": 84_000.0}
BASELINES = {"S0": 90_000.0, "S1": 84_000.0, "S2": 79_000.0, "stay": 100_000.0}
VERSION = "rl7-dqn-t1-s2-0123abcd"
HEADLINE = [
    ("car", "time_loss_mean"),
    ("bus", "delay_mean"),
    ("pedestrian", "ped_curb_wait_mean"),
    ("pedestrian", "ped_curb_wait_max"),
    ("ambulance", "travel_time_total"),
    ("all", "person_delay_mean"),
    ("car", "depart_delay_mean"),
]


def _validation(level: float, steps: int, seed: int) -> pd.DataFrame:
    grid = np.arange(steps // 6, steps + 1, steps // 6)
    cost = level * (1.25 - 0.25 * grid / steps) * (1 + 0.004 * seed)
    return pd.DataFrame(
        {
            "step": grid,
            "cost": cost,
            **{f"cost_vs_{k}": 100 * (cost / v - 1) for k, v in BASELINES.items()},
            "car_s": cost * 0.5 / 1.3,
            "car_pending_s": 900.0,
            "bus_s": cost * 0.33 / 25,
            "pedestrian_s": cost * 0.13 / 1.2,
            "ambulance_s": 40.0,
            "penalty_s": 60.0 - 0.01 * grid,
            "interventions": 5.0,
            "iv_ped_wait": 4.0,
            "iv_max_green": 1.0,
            "danger": 0,
            "ped_curb_wait_max": 88.0,
            "decisions": 200.0,
            "advances": 70.0,
            "amb_seen_share": 0.0,
            "preempt_requests": 4,
            "preempt_served": 4,
            "preempt_timeouts": 0,
            "split_cut_per_h": 6.0,
            "split_replaced_per_h": 2.0,
            "split_cut_foreseeable_per_h": 1.5,
        }
    )


def _enrich(d: Path, variant: str, seed: int, steps: int) -> None:
    """Validation, monitor, baselines and manifest fields of a finished run folder."""
    val = _validation(LEVEL[variant], steps, seed)
    pick = val[val["step"] >= steps // 2].sort_values("cost").iloc[0]
    val["pick"] = val["step"] == pick["step"]
    val.to_csv(d / "validation.csv", index=False)
    mon = pd.DataFrame(
        {"r": -np.linspace(120, 80, 40) - seed, "l": [steps // 40] * 40, "t": np.arange(40.0)}
    )
    (d / "monitor.csv").write_text('#{"t_start": 0}\n' + mon.to_csv(index=False), encoding="utf-8")
    base = pd.DataFrame(
        [
            {
                "label": k,
                "policy": k,
                "seed": s,
                "cost": v,
                "car_s": v * 0.5 / 1.3,
                "bus_s": v * 0.38 / 25,
                "pedestrian_s": v * 0.11 / 1.2,
                "penalty_s": 20.0,
                "interventions": 0.0,
            }
            for k, v in BASELINES.items()
            for s in (5100, 5101)
        ]
    )
    base.to_csv(d / "baselines.csv", index=False)
    m = json.loads((d / "manifest.json").read_text())
    late = val[val["step"] >= steps // 2]["cost"]
    m |= {
        "algo": "ppo" if variant == "p1" else "dqn",
        "steps_done": steps,
        "n_envs": 2,
        "pick_step": int(pick["step"]),
        "pick_val_cost": float(pick["cost"]),
        "late_mean": float(late.mean()),
        "final_val_cost": float(val["cost"].iloc[-1]),
        "baselines": BASELINES,
        "wall_s": 1200.0 + seed,
        "steps_per_s": 380.0,
        "code_version": "abc1234",
        "versions": {"python": "3.12", "sb3": "2.9.0"},
    }
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    # rl.study holdout: the pick's episodes (run, step, mode, seed, cost, ...)
    hold = pd.DataFrame(
        {
            "run": d.name,
            "step": int(pick["step"]),
            "mode": ["train", "train", "full"],
            "seed": [5112, 5113, 5112],
            "cost": float(pick["cost"]) * np.array([1.2, 1.2, 4.0]),
        }
    )
    hold.to_csv(d / "holdout.csv", index=False)
    return {
        "run": d.name,
        "stage": d.parent.name,
        "variant": variant,
        "seed": seed,
        "holdout_w30": float(pick["cost"]) * 1.2,
        "holdout_full": float(pick["cost"]) * 4.0,
        **{f"w30_{k}": v * 1.2 for k, v in BASELINES.items()},
        **{f"w30_vs_{k}": 100 * (float(pick["cost"]) / v - 1) for k, v in BASELINES.items()},
    }


def _paired(seeds) -> pd.DataFrame:
    """Dev-test paired rows (`evaluation.compare`, S3 - ref) with a `ref` column."""
    rng = np.random.default_rng(0)
    out = []
    for ref, shift in (("S1", -6.0), ("S0", -9.0), ("S2", 1.0)):
        a, b = [], []
        for s in seeds:
            for u, k in HEADLINE:
                v = 50.0 + rng.normal(0, 1)
                a.append({"seed": s, "user_type": u, "kpi": k, "value": v})
                b.append(
                    {"seed": s, "user_type": u, "kpi": k, "value": v + shift + rng.normal(0, 0.1)}
                )
        out.append(compare(pd.DataFrame(a), pd.DataFrame(b), HEADLINE).assign(ref=ref))
    return pd.concat(out, ignore_index=True)


REPRO_HEAD = "abc1234"


def write_repro(study: Path, head: str, same: bool) -> Path:
    """A `tasks.py reproduce7` report: `_reproduce/<head>/reproduce7.json`."""
    d = study / "_reproduce" / head
    d.mkdir(parents=True, exist_ok=True)
    items = {
        "rl7 run dqn_t1_s0_400k_x": [],
        "golden tests/data/rl_s3_golden.json": [],
        "page docs/rl_training.md": [] if same else ["differs from line 9 of the original"],
    }
    p = d / "reproduce7.json"
    p.write_text(json.dumps({"head": head, "same": same, "items": items}), encoding="utf-8")
    return p


def full_study(
    stub_rl7_cfg,
    monkeypatch,
    verdict: str = "pass",
    repro_same: bool | None = True,
    applies: tuple[str, ...] = (REPRO_HEAD,),
) -> Path:
    """A stub workspace holding every Part 7 artifact; returns its rl7.yaml. verdict.json
    is written as the dev-test look writes it (before reproduce7: C4.repro pending); a
    reproduce7 report at REPRO_HEAD unless `repro_same` is None; reproduce7 results apply
    to this code only for the heads in `applies` (no git)."""
    monkeypatch.setattr(report7, "repro_applies", lambda head, code: head in applies)
    rl7, sens_out, _ = sens_workspace(stub_rl7_cfg, monkeypatch)
    sens.tables(rl7)
    cfg = load_config(rl7)
    study, runs = Path(cfg["outputs"]["study"]), Path(cfg["outputs"]["runs"])
    t_steps, f_steps = (
        int(cfg["protocol"]["stage_t"]["steps"]),
        int(cfg["protocol"]["stage_f"]["steps"]),
    )
    jobs, summary = [], []
    for c in T_CONFIGS:
        for seed in (0, 1, 2):
            existing = sorted((runs / "T").glob(f"dqn_{c}_s{seed}_*"))
            d = existing[0] if existing else make_pick_run(runs, rl7, c, seed, t_steps, stage="T")
            summary.append(_enrich(d, c, seed, t_steps))
            jobs.append(
                {
                    "run": d.name,
                    "stage": "T",
                    "status": "done",
                    "wall_s": 1200,
                    "steps_per_s": 380,
                    "attempts": 1,
                }
            )
    for seed in range(5):
        d = make_pick_run(runs, rl7, "t1", seed, f_steps, stage="F")
        summary.append(_enrich(d, "t1", seed, f_steps))
        val = pd.read_csv(d / "validation.csv")
        curve = pd.DataFrame(
            {
                "step": val["step"],
                "n": 4,
                "cost": val["cost"] * 1.15,
                "pick": val["pick"],
                "run": d.name,
            }
        )
        curve.to_csv(d / "holdout_curve.csv", index=False)
        jobs.append(
            {
                "run": d.name,
                "stage": "F",
                "status": "done",
                "wall_s": 3000,
                "steps_per_s": 375,
                "attempts": 2 if seed == 3 else 1,
            }
        )
    for d in sorted((runs / "S").glob("*")):
        jobs.append(
            {
                "run": d.name,
                "stage": "S",
                "status": "done",
                "wall_s": 1250,
                "steps_per_s": 370,
                "attempts": 1,
            }
        )
    (study / "holdout").mkdir()
    pd.DataFrame(summary).to_csv(study / "holdout" / "summary.csv", index=False)
    (study / "campaign.json").write_text(json.dumps({"slots": 4, "jobs": jobs}), encoding="utf-8")
    scores = {c: {str(s): LEVEL[c] * (1 + 0.003 * s) for s in range(3)} for c in T_CONFIGS}
    means = {c: float(np.mean(list(v.values()))) for c, v in scores.items()}
    decision = {
        "scores": scores,
        "means": means,
        "ped_s": {c: 7600.0 + 10 * i for i, c in enumerate(T_CONFIGS)},
        "rule": {"reference": "t1", "margin_rel": 0.015, "min_seed_wins": 2, "ped_guard_rel": 0.05},
        "outcome": {
            "v1": {"a": True, "b": True, "c": False, "qualifies": False},
            "t0": {"a": False},
        },
        "chosen": "t1",
        "created_at": "2026-10-17T00:00:00+00:00",
    }
    (study / "decision_T.json").write_text(json.dumps(decision), encoding="utf-8")
    f_runs = sorted((runs / "F").glob("*"))
    cands = [
        {
            "run": d.name,
            "pick_step": f_steps,
            "stage1": 79_000 + 100 * i,
            "stage2": 300_000 - 900 * i,
            "holdout_w30": 95_000.0,
            "holdout_full": 310_000.0,
            "late_mean": 79_400.0,
        }
        for i, d in enumerate(f_runs)
    ]
    (study / "select").mkdir()
    sel = {
        "candidates": cands,
        "chosen": f_runs[-1].name,
        "rule": {
            "text": "S3 = the stage-F pick with the lowest stage-2 cost (D75).",
            "tie_rel": 0.005,
        },
    }
    (study / "select" / "decision.json").write_text(json.dumps(sel), encoding="utf-8")
    model_dir = Path(cfg["outputs"]["models"]) / VERSION
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_version": VERSION,
                "model_sha256": "ab" * 32,
                "policy_sha256": "cd" * 32,
                "source_run": f_runs[-1].name,
            }
        )
    )
    (study / "stress").mkdir()
    cells = list(cfg["protocol"]["stress"]["cells"])
    ep = pd.DataFrame(
        [
            {
                "cell": c,
                "label": lab,
                "seed": s,
                "mode": "full" if c == "full_nominal" else "train",
                "override": "{}",
                "cost": 80_000
                * {"S3": 1.0 if c != "x0_6" else 1.2, "S0": 1.1, "S1": 1.08, "S2": 0.99}[lab]
                * (1 + 0.01 * (s - 5118)),
                "pedestrian_s": {"S3": 1_120.0, "S0": 1_000.0, "S1": 1_000.0, "S2": 1_050.0}[lab],
                "penalty_s": {"S3": 41.0, "S0": 30.0, "S1": 30.0, "S2": 35.0}[lab],
                "interventions": {"S3": 6.0, "S0": 0.0, "S1": 0.0, "S2": 2.0}[lab],
                "split_cut": 4.0 if lab == "S3" else 0.0,
                "split_replaced": 2.0 if lab == "S3" else 0.0,
                "split_cut_foreseeable": 1.0 if lab == "S3" else 0.0,
                "danger": 0,
                "ped_curb_wait_max": 88.0,
                "preempt_timeouts": 0,
            }
            for c in cells
            for lab in ("S3", "S0", "S1", "S2")
            for s in (5118, 5119)
        ]
    )
    ep.to_csv(study / "stress" / "episodes.csv", index=False)
    # the producer's own summary (`rl.study stress`), so the column contract is tested
    stress_summary(ep, float(cfg["protocol"]["stress"]["worse_than_s1_rel"]), cells).to_csv(
        study / "stress" / "summary.csv", index=False
    )
    dev = study / "devtest"
    dev.mkdir()
    seeds = [5210, 5211]
    _paired(seeds).to_csv(dev / "paired.csv", index=False)
    cmp = pd.DataFrame(
        [
            {"label": lab, "seed": s, "cost": c * (1 + 0.01 * (s - 5210)), "amb_seen_share": 0.0}
            for lab, c in (
                ("S3", 76_000),
                ("S0", 90_000),
                ("S1", 84_000),
                ("S2", 77_000),
                ("stay", 99_000),
                ("random", 95_000),
            )
            for s in seeds
        ]
    )
    cmp.to_csv(dev / "compare.csv", index=False)
    for sc in ("pitie_rl_s3", "pitie_actuated"):
        for s in seeds:
            d = dev / f"{sc}_seed{s}"
            d.mkdir()
            ctl = (
                {
                    "model_version": VERSION,
                    "iv_split": {"cut": 9, "replaced": 2, "cut_foreseeable": 4},
                }
                if sc.endswith("s3")
                else {}
            )
            meta = {
                "scenario": sc,
                "seed": s,
                "duration_s": 7200,
                "code_version": "abc1234",
                "safety": {
                    "interventions": {"ped_wait": 9, "max_green": 2} if ctl else {},
                    "danger": 0,
                },
                "health": {"collisions": 0, "teleports": 0},
                "preemption": {"requests": 1, "served": 1, "timeouts": 0},
                "control": ctl,
            }
            (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")
            pd.DataFrame(
                [{"user_type": "pedestrian", "kpi": "ped_curb_wait_max", "value": 88.0}]
            ).to_csv(d / "kpi_run.csv", index=False)
    v = {
        "C1": {"hl": -8.4, "p": 0.002, "pass": verdict == "pass"},
        "C2": {
            "rows": [
                {
                    "kpi": "car/time_loss_mean",
                    "s1_mean": 58.0,
                    "s3_mean": 55.0,
                    "rel_pct": -5.17,
                    "worse": False,
                }
            ],
            "limit_pct": 5.0,
            "pass": True,
        },
        "C3": {"violations": [], "pass": True},
        "C4": {
            "runs_below_s1": 5,
            "of": 5,
            "required": 4,
            "runs": [],
            "repro": {"status": "pending", "same": None},
            "pass": True,
        },
        "pending": ["C4.repro"],
        "stress_cells_worse": ["x0_6"],
        "s2_framing": {"framing": "approx", "text": "RL ≈ rules", "hl": 0.4, "p": 0.6},
        "verdict": verdict,
        "seeds": seeds,
        "model_version": VERSION,
    }
    (dev / "verdict.json").write_text(json.dumps(v), encoding="utf-8")
    if repro_same is not None:
        write_repro(study, REPRO_HEAD, repro_same)
    return rl7


def _sections(text: str) -> list[str]:
    return re.findall(r"^## \d+\. (.+)$", text, flags=re.M)


def test_page_with_every_input_missing(stub_rl7_cfg):
    rl7 = stub_rl7_cfg()
    page = report7.build_docs(rl7)
    cfg = load_config(rl7)
    assert page == Path(cfg["outputs"]["docs"])
    text = page.read_text(encoding="utf-8")
    assert text.startswith(report7.TITLE + "\n")
    heads = _sections(text)
    assert len(heads) == len(report7.SECTIONS) == 13
    for want, got in zip(report7.SECTIONS, heads, strict=True):
        assert got.startswith(want.split(" (")[0]), (want, got)
    for task in (
        "rl7-campaign",
        "rl7-holdout",
        "rl7-stress",
        "rl7-sens",
        "rl7-devtest",
        "reproduce7",
    ):
        assert f"Run `tasks.py {task}`" in text, task
    assert text.count("_Missing:") >= 10
    for word in (
        "10.6",
        "10.3",
        "10.4",
        "10.5",
        "Learning curves",
        "ASSUMED",
        "provisional",
        "field visit",
    ):
        assert word in text, word
    assert "![" not in text and not Path(cfg["outputs"]["figures"]).exists()
    assert "Interim 10.6 verdict" not in text  # no statement without verdict.json
    assert "\r\n" not in page.read_bytes().decode("utf-8")


def test_page_with_full_synthetic_inputs(stub_rl7_cfg, monkeypatch):
    rl7 = full_study(stub_rl7_cfg, monkeypatch)
    page = report7.build_docs(rl7)
    text = page.read_text(encoding="utf-8")
    assert "_Missing:" not in text, [ln for ln in text.splitlines() if "Missing" in ln]
    assert len(_sections(text)) == 13
    assert "**Interim 10.6 verdict: PASS**" in text and "main candidate" in text
    assert "`x0_6`" in text and "RL ≈ rules (person delay HL 0.4 s, p 0.600)" in text
    assert "5 of 5 (need 4); reproduce7 identical at `abc1234` | yes |" in text
    assert "PASS, provisional" not in text and "Pending:" not in text
    assert "Holdout vs S2" in text and "Holdout vs S1" in text
    assert "Rule: S3 = the stage-F pick" in text
    assert f"S3 = `{VERSION}`" in text and "**C\\* = T1.**" in text
    figs = Path(load_config(rl7)["outputs"]["figures"])
    for name in viz.FIGURES:
        assert f"({_rel(figs / f'{name}.png', page)})" in text, name
        assert (figs / f"{name}.png").is_file()
    # tables: stage T, stage F, selection, sensitivity, stress, dev-test pairs
    assert "| T1 | 0 |" in text and "| V1 | 2 |" in text
    assert "**S3**" in text and "| **t1** |" in text and "| wb50 |" in text
    assert "**S3 vs S1**" in text and "**S3 vs S2**" in text
    assert "`reproduce7` at `abc1234`: **identical** (rl7 run dqn_t1_s0_400k_x: same;" in text
    # stress panel (D78): paired tests, pedestrian and penalty seconds, interventions
    assert "| Cell | Mode | Wins vs S1 | p vs S1 | Wins vs S2 | p vs S2 | Ped s vs S1 |" in text
    row = next(ln for ln in text.splitlines() if ln.startswith("| `x0_6` | train |"))
    assert "| 0/2 |" in row and "| +12.0 % | +6.7 % |" in row, row
    row = next(ln for ln in text.splitlines() if ln.startswith("| `x0_6` | 41 / 30 / 35 |"))
    assert "| 6.0 / 0.0 / 2.0 | 4.0 / 2.0 / 1.0 | 0 | 88 | 0 |" in row, row
    assert "| `full_nominal` | full |" in text
    assert "Concurrent jobs: 4" in text
    for p in re.findall(r"!\[[^]]*\]\(([^)]+)\)", text):
        assert (page.parent / p).resolve().is_file(), p


def _rel(target: Path, page: Path) -> str:
    import os

    return Path(os.path.relpath(target.resolve(), page.resolve().parent)).as_posix()


def test_page_is_deterministic_apart_from_the_code_line(stub_rl7_cfg, monkeypatch):
    rl7 = full_study(stub_rl7_cfg, monkeypatch)
    a = report7.build_docs(rl7).read_text(encoding="utf-8")
    monkeypatch.setattr("traffic_twin.sim.results.code_version", lambda: "fffffff-dirty")
    b = report7.build_docs(rl7).read_text(encoding="utf-8")
    diff = [(x, y) for x, y in zip(a.splitlines(), b.splitlines(), strict=True) if x != y]
    assert len(diff) == 1 and diff[0][1].startswith(report7.CODE_LINE)


def test_published_model_found_from_the_selection(stub_rl7_cfg, monkeypatch):
    """Before the dev-test look the S3 identity comes from the published manifest whose
    source run is the selected one."""
    rl7 = full_study(stub_rl7_cfg, monkeypatch)
    (Path(load_config(rl7)["outputs"]["study"]) / "devtest" / "verdict.json").unlink()
    text = report7.build_docs(rl7).read_text(encoding="utf-8")
    assert f"**S3 = `{VERSION}`**" in text and "Missing: the dev-test verdict" in text


def test_fallback_verdict_text(stub_rl7_cfg, monkeypatch):
    rl7 = full_study(stub_rl7_cfg, monkeypatch, verdict="fallback")
    text = report7.build_docs(rl7).read_text(encoding="utf-8")
    assert "**Interim 10.6 verdict: FALLBACK**" in text
    assert "Spec 10.6 applies" in text and "S2) become the main solution" in text
    # the C2 row names each equity KPI's change against S1 and the limit
    assert "| C2 equity vs S1 | `car/time_loss_mean` -5.2 %; limit +5 % |" in text


def test_partial_inputs(stub_rl7_cfg, monkeypatch, tmp_path):
    """Only a few stage-T runs: the page names what is missing and draws what it can."""
    rl7 = stub_rl7_cfg()
    cfg = load_config(rl7)
    runs = Path(cfg["outputs"]["runs"])
    steps = int(cfg["protocol"]["stage_t"]["steps"])
    for seed in (0, 1):
        _enrich(make_pick_run(runs, rl7, "t1", seed, steps, stage="T"), "t1", seed, steps)
    docs = tmp_path / "page" / "rl_training.md"
    text = report7.build_docs(rl7, docs=docs, figures=tmp_path / "page" / "fig").read_text(
        encoding="utf-8"
    )
    assert "learning_bands.png" in text and "kpi_curves" not in text.split("## 5.")[0]
    assert "Missing: the stage-T decision" in text and "Missing: the stage-F runs" in text
    assert (tmp_path / "page" / "fig" / "learning_bands.png").is_file()
    assert (tmp_path / "page" / "fig" / "kpi_curves.png").is_file()  # T1 runs stand in for F


def test_stress_frame_formats():
    ep = pd.DataFrame(
        {
            "cell": ["a", "a", "a", "b", "b", "b"],
            "label": ["S3", "S1", "S2"] * 2,
            "seed": 1,
            "cost": [100.0, 80.0, 100.0, 90.0, 100.0, 90.0],
        }
    )
    fr = report7.stress_frame(None, ep).set_index(["cell", "ref"])["pct"]
    assert fr[("a", "S1")] == pytest.approx(25.0) and fr[("b", "S1")] == pytest.approx(-10.0)
    summ = pd.DataFrame({"cell": ["a"], "vs_S1_pct": [3.0], "vs_S2_pct": [-1.0]})
    fr2 = report7.stress_frame(summ, None)
    assert fr2.to_dict("records") == [
        {"cell": "a", "ref": "S1", "pct": 3.0},
        {"cell": "a", "ref": "S2", "pct": -1.0},
    ]
    assert report7.stress_frame(None, None).empty


def test_cli_help_and_paths(stub_rl7_cfg, tmp_path, capsys):
    rl7 = stub_rl7_cfg()
    out = tmp_path / "x" / "page.md"
    assert (
        report7.main(["--config", str(rl7), "--docs", str(out), "--figures", str(tmp_path / "f")])
        == 0
    )
    assert capsys.readouterr().out.strip() == out.as_posix()
    with pytest.raises(SystemExit):
        report7.build_parser().parse_args(["--help"])


def _study(rl7: Path) -> Path:
    return Path(load_config(rl7)["outputs"]["study"])


@pytest.mark.parametrize("form", ["pass", "pending"])
def test_verdict_is_provisional_while_reproduce7_is_pending(stub_rl7_cfg, monkeypatch, form):
    """verdict.json is written at the dev-test look, before the commit and so before
    reproduce7 (C4.repro pending): the page calls it provisional and C4 pending, whether
    verdict.json says "pass" (C4 pass = runs only) or "pending" (`runs_pass`, C4 None)."""
    rl7 = full_study(stub_rl7_cfg, monkeypatch, repro_same=None)
    if form == "pending":
        path = _study(rl7) / "devtest" / "verdict.json"
        v = json.loads(path.read_text(encoding="utf-8"))
        v["verdict"] = "pending"
        v["C4"] |= {"runs_pass": True, "pass": None}
        path.write_text(json.dumps(v), encoding="utf-8")
    text = report7.build_docs(rl7).read_text(encoding="utf-8")
    head = text.split("## 2.")[0]
    assert "**Interim 10.6 verdict: PASS, provisional**" in head
    assert "**Interim 10.6 verdict: PASS**" not in text
    assert "reproduce7 pending (runs after the commit) | pending |" in head
    assert "Pending: C4.repro." in head and "If reproduce7 reproduces" in head
    assert "Missing: the reproduce7 check" in text


def test_failed_reproduce7_turns_a_stored_pass_into_fallback(stub_rl7_cfg, monkeypatch):
    """D79: C4 needs reproduce7's check. A result that did not reproduce fails C4 and the
    page says FALLBACK in section 1 as in section 12; verdict.json stays as written."""
    rl7 = full_study(stub_rl7_cfg, monkeypatch, repro_same=False)
    verdict = _study(rl7) / "devtest" / "verdict.json"
    before = verdict.read_bytes()
    text = report7.build_docs(rl7).read_text(encoding="utf-8")
    head = text.split("## 2.")[0]
    assert "**Interim 10.6 verdict: FALLBACK**" in head and "PASS**" not in head
    assert "reproduce7 NOT identical at `abc1234` | no |" in head
    assert "C4 therefore fails" in head and "Spec 10.6 applies" in head
    assert "main candidate" not in head and "Pending:" not in head
    assert "`reproduce7` at `abc1234`: **NOT identical**" in text
    assert "page docs/rl_training.md: DIFFERENT (differs from line 9" in text
    assert verdict.read_bytes() == before


def test_effective_verdict_edges():
    base = {
        "C1": {"pass": True},
        "C2": {"pass": True},
        "C3": {"pass": True},
        "C4": {"runs_below_s1": 4, "required": 4, "repro": {"same": None}, "pass": True},
        "verdict": "pass",
        "pending": ["C4.repro"],
    }
    e = report7.effective_verdict(base, [])
    assert (e["verdict"], e["provisional"], e["c4"], e["pending"]) == (
        "pending",
        True,
        "pending",
        ["C4.repro"],
    )
    ok, bad = {"head": "a", "same": True}, {"head": "b", "same": False}
    e = report7.effective_verdict(base, [ok])
    assert (e["verdict"], e["provisional"], e["c4"], e["pending"]) == ("pass", False, "yes", [])
    e = report7.effective_verdict(base, [ok, bad])  # one failed reproduction fails C4
    assert (e["verdict"], e["downgraded"], e["c4"]) == ("fallback", True, "no")
    # a stored fallback never becomes a pass; too few runs below S1 fail C4 whatever repro
    fb = {**base, "verdict": "fallback", "C4": {**base["C4"], "runs_below_s1": 3, "pass": False}}
    e = report7.effective_verdict(fb, [ok])
    assert (e["verdict"], e["downgraded"], e["c4"]) == ("fallback", False, "no")
    # a repro stored in verdict.json (reproduce7 before the look) counts as well
    stored = {**base, "C4": {**base["C4"], "repro": {"head": "c", "same": False}, "pass": False}}
    stored["verdict"] = "fallback"
    e = report7.effective_verdict(stored, [])
    assert e["c4"] == "no" and "NOT identical at `c`" in e["repro"]
    # rl.study's "pending" form (runs_pass, C4 pass None): provisional, then settled
    new = {**base, "verdict": "pending", "C4": {**base["C4"], "runs_pass": True, "pass": None}}
    e = report7.effective_verdict(new, [])
    assert e["verdict"] == "pending" and e["provisional"]
    assert report7.effective_verdict(new, [ok])["verdict"] == "pass"
    assert report7.effective_verdict(new, [bad])["downgraded"]
    # another criterion not evaluated: pending, not a provisional pass
    e = report7.effective_verdict({**new, "C1": {"pass": None}}, [ok])
    assert e["verdict"] == "pending" and not e["provisional"]


def _masked(text: str) -> list[str]:
    """The page as `tasks.py reproduce7` compares it (code-version lines aside)."""
    return [ln for ln in text.splitlines() if not re.search(r"code[ -]version", ln, re.I)]


def test_page_ignores_reproduce7_results_of_other_code(stub_rl7_cfg, monkeypatch):
    """D82: reproduce7 rebuilds the page and compares it with the committed one. A
    reproduce7 report left by an earlier commit whose code differs (a failed first try
    fixed in code), and the new backup folder of the running reproduce7, must not change
    the page."""
    rl7 = full_study(stub_rl7_cfg, monkeypatch, repro_same=None, applies=())
    study = _study(rl7)
    committed = report7.build_docs(rl7).read_text(encoding="utf-8")
    write_repro(study, "aaaa111", same=False)  # reproduce7 at A, then a code fix -> B
    backup = study / "_reproduce" / "bbbb222"  # reproduce7 at B: prepare + golden
    backup.mkdir(parents=True)
    (backup / "rl_s3_golden.json").write_text(json.dumps({"q": [1.0]}), encoding="utf-8")
    rebuilt = report7.build_docs(rl7).read_text(encoding="utf-8")
    assert _masked(rebuilt) == _masked(committed)
    assert "aaaa111" not in rebuilt and "PASS, provisional" in rebuilt


def test_page_with_a_reproduce7_result_is_stable_for_docs_only_commits(stub_rl7_cfg, monkeypatch):
    """After reproduce7 at A, the page rebuilt (and committed as B, docs only) states A's
    result; reproduce7 at B rebuilds it before writing B's report: the same page."""
    rl7 = full_study(stub_rl7_cfg, monkeypatch, applies=(REPRO_HEAD, "bbbb222"))
    study = _study(rl7)
    committed = report7.build_docs(rl7).read_text(encoding="utf-8")
    assert "**Interim 10.6 verdict: PASS**" in committed
    (study / "_reproduce" / "bbbb222").mkdir()
    rebuilt = report7.build_docs(rl7).read_text(encoding="utf-8")
    assert _masked(rebuilt) == _masked(committed)


def test_repro_applies(monkeypatch):
    assert not report7.repro_applies("abc1234", "abc1234-dirty")
    assert not report7.repro_applies("abc1234", "unknown")
    assert not report7.repro_applies("", "abc1234")
    assert report7.repro_applies("abc1234", "abc1234")  # same clean commit, no git needed
    assert not report7.repro_applies("0000000", "abc1234")  # not a commit of this repo
    calls = []

    class Done:
        returncode = 0

    def run(cmd, **kw):
        calls.append(cmd)
        return Done()

    monkeypatch.setattr("subprocess.run", run)
    assert report7.repro_applies("aaaa111", "bbbb222")  # same files outside docs/
    assert calls[0][-4:] == ["aaaa111", "--", ".", ":(exclude)docs"]


def test_stress_details_missing_summary():
    lines = report7.stress_details(None, ["a"])
    assert lines[0].startswith("_Missing: the stress panel's paired tests")
    summ = pd.DataFrame({"cell": ["a"], "vs_S1_pct": [3.0]})  # an older, thinner summary
    text = "\n".join(report7.stress_details(summ, ["a"]))
    assert "| `a` | – | – | – | – | – | – | – |" in text


# -- post-campaign reporting fixes (Part 7 phase 3 review) --------------------------------
def _v1_study(stub_rl7_cfg, monkeypatch) -> Path:
    """The full synthetic study with S3 a V1 policy (pedestrian group dropped, 62 features),
    the sensitivity policies on the full observation (74), a failed pedestrian C2 row, a
    collision on one seed for a policy and S0, and a dirty dev-test code version."""
    rl7 = full_study(stub_rl7_cfg, monkeypatch, verdict="fallback")
    cfg = load_config(rl7)
    study = Path(cfg["outputs"]["study"])
    man = Path(cfg["outputs"]["models"]) / VERSION / "manifest.json"
    m = json.loads(man.read_text(encoding="utf-8"))
    m |= {"variant": "v1", "obs_drop": ["pedestrian"], "n_features": 62}
    man.write_text(json.dumps(m), encoding="utf-8")
    plan = json.loads((study / "sens" / "plan.json").read_text(encoding="utf-8"))
    runs = Path(cfg["outputs"]["runs"])
    for p in plan["policies"]:
        for f in runs.glob(f"*/{p['run']}/manifest.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            r["observation"] = {"version": "rl-obs-1", "drop": [], "n_features": 74}
            f.write_text(json.dumps(r), encoding="utf-8")
    vpath = study / "devtest" / "verdict.json"
    v = json.loads(vpath.read_text(encoding="utf-8"))
    v["C2"]["rows"].append({"kpi": "pedestrian/ped_curb_wait_mean", "rel_pct": 6.2, "worse": True})
    v["C2"]["pass"] = False
    v["code_version"] = "abc1234-dirty"
    vpath.write_text(json.dumps(v), encoding="utf-8")
    saf_path = study / "sens" / "safety.csv"
    saf = pd.read_csv(saf_path)
    seed = int(saf["seed"].iloc[0])
    pol = plan["policies"][1]["label"]
    hit = saf["seed"].eq(seed) & saf["label"].isin([pol, "S0"])
    saf.loc[hit, "collisions"] = 1
    saf.to_csv(saf_path, index=False)
    return rl7


def test_page_states_what_s3_observes(stub_rl7_cfg, monkeypatch):
    """Plan section 11 (V1 adoption): when S3 drops the pedestrian group the page says so
    in the summary, the sensitivity, the interventions split and the limitations, and that
    the sensitivity agents (full observation) are a different agent."""
    text = report7.build_docs(_v1_study(stub_rl7_cfg, monkeypatch)).read_text(encoding="utf-8")
    head = text.split("## 2.")[0]
    assert "S3 does not observe pedestrians: V1 drops the pedestrian feature group" in head
    assert "none of its 62 features is a pedestrian count, curb wait or shield slack" in head
    assert "C2's pedestrian result (`pedestrian/ped_curb_wait_mean` +6.2 %)" in head
    sens_part = text.split("## 8.")[1].split("## 9.")[0]
    assert "**Not S3's agent.** The sensitivity policies observe 74 features (the full" in (
        sens_part
    )
    assert "62 features (group pedestrian dropped)" in sens_part
    assert "describe a different agent from S3" in sens_part
    assert "S3 (V1) does not observe it" in text  # foreseeable cuts, section 11
    limits = text.split("## 13.")[1]
    assert "- S3 (V1, 62 features) reads every signal-controlled lane" in limits
    assert "no pedestrian count, curb wait or shield slack" in limits
    assert "the pedestrian curb waits from the simulator" not in text
    assert "different agent from S3" in limits
    assert "could not be checked on them" in limits  # the collision STOP clause


def test_page_with_full_observation_keeps_the_plain_text(stub_rl7_cfg, monkeypatch):
    text = report7.build_docs(full_study(stub_rl7_cfg, monkeypatch)).read_text(encoding="utf-8")
    assert "S3 does not observe pedestrians" not in text and "Not S3's agent" not in text
    assert "the pedestrian curb waits from the simulator (ASSUMED ideal detection)" in text


def test_outcome_text_shows_the_margin_to_the_limit():
    """D73: the stage-T ratio printed to 5 decimals against its limit (V1 passed by 0.0002,
    which 3 significant figures hide)."""
    rule = {"margin_rel": 0.015, "min_seed_wins": 2, "ped_guard_rel": 0.05}
    o = {
        "score": 94_901.98,
        "ratio": 0.9847835765,
        "ped_s": 8547.5,
        "ped_ratio": 1.0007447594,
        "reference": False,
        "wins": 3,
        "seeds": 3,
        "a": True,
        "b": True,
        "c": True,
        "qualifies": True,
    }
    t = report7.outcome_text(o, rule, "t1")
    assert "ratio to T1 0.98478 against the limit 0.98500 (below it by 0.00022); (a) yes" in t
    assert "late mean below T1 on 3 of 3 seeds (need 2); (b) yes" in t
    assert "pedestrian waiting ratio 1.00074 against the limit 1.05000" in t
    assert t.endswith("qualifies: yes")
    assert report7.outcome_text({"reference": True, "score": 96_368.4}, rule, "t1") == (
        "the reference (score 96,368)"
    )
    assert "ratio to T1 1.00956 against the limit 0.98500 (above it by 0.02456)" in (
        report7.outcome_text({"ratio": 1.009559586}, rule, "t1")
    )
    assert report7.outcome_text({"a": False, "x": 2}, {}, "t1") == "(a) no; x 2"


def test_resolution_text_is_measured():
    boot = pd.DataFrame(
        {
            "setting": ["wb12", "wb50", "lam100", "wp48"],
            "metric": "bus/delay_mean",
            "diff": [6.22, -7.47, -2.98, 3.25],
            "ci_low": [2.78, -10.76, -6.20, -0.43],
            "ci_high": [9.52, -4.45, 0.09, 6.78],
            "beyond_noise": [True, True, False, False],
            "n_policies": [5, 5, 3, 3],
        }
    )
    t = report7.resolution_text(boot, "bus/delay_mean", "Bus delay")
    assert t == (
        "Bus delay (measured): the smallest change whose CI excludes 0 is `wb12` +6.2 s "
        "[+2.8, +9.5], 5 policies; the largest whose CI includes 0 is `wp48` +3.2 s "
        "[-0.4, +6.8], 3 policies. "
    )
    assert report7.resolution_text(None, "bus/delay_mean", "Bus delay") == ""
    assert report7.resolution_text(boot, "car/x", "Car") == ""


def test_sensitivity_safety_and_dirty_versions(stub_rl7_cfg, monkeypatch):
    rl7 = _v1_study(stub_rl7_cfg, monkeypatch)
    text = report7.build_docs(rl7).read_text(encoding="utf-8")
    sens_part = text.split("## 8.")[1].split("## 9.")[0]
    assert "**Safety of the sensitivity runs**" in sens_part
    assert "Runs with a SUMO collision (count in brackets):" in sens_part
    assert "`S0` seed" in sens_part and "A baseline without RL also collides on seed" in sens_part
    assert "Bus effects below about 7 s" not in text and "Bus delay (measured):" in sens_part
    # dev-test code version: a dirty one not in DIRTY_NOTES is stated without a story
    dev = text.split("## 10.")[1].split("## 11.")[0]
    assert "(`verdict.json`): `abc1234-dirty`: the tree differed from that commit" in dev
    repro = text.split("## 12.")[1].split("## 13.")[0]
    assert "Code versions of the training runs: `abc1234`." in repro
    assert "- dev-test look (`devtest/verdict.json`): `abc1234-dirty`" in repro
    assert "stress panel (`stress/`): the outputs record no code version." in repro
    assert report7.version_text("5bec559-dirty").startswith(
        "`5bec559-dirty`: the tree was `5bec559` plus untracked planned outputs only"
    )
    assert report7.version_text("5bec559") == "`5bec559`."


def test_prototype_tables():
    """Section 6 cites every prototype the plan names (masks, adaptive hold, penalty,
    VecNormalize) with its numbers, against A0 rather than "the Part 6 DQN"."""
    lines = "\n".join(report7.variants_section({"decision_t": {}}))
    assert "Cost vs A0 (binary DQN, 6 envs, 200k)" in lines and "Part 6 DQN" not in lines
    assert "Intervention penalty κ = 1000" in lines and "−69 %" in lines
    assert "Mask during training" in lines and "+7.8 %" in lines
    assert "Inference-time mask" in lines and "+2.4 %" in lines
    assert "γ 0.99" in lines and "Update-to-data 0.25" in lines and "+9.7 ± 3.1 %" in lines
    assert "VecNormalize in training" not in lines.split("Never run:")[1].split(".")[0]

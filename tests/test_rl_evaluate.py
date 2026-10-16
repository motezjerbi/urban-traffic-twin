"""RL evaluation (Part 6, WS5; D60): seed guards, model selection, the dev-test window
comparison and its Wilcoxon test, the full-run KPI and safety summary, the S6 criterion
at its edges, the reproducibility check and the docs/rl.md generator, all on synthetic
run folders or the SUMO-free stub env (no SUMO)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from traffic_twin.config import ConfigError, load_config, text_sha256
from traffic_twin.evaluation.stats import wilcoxon
from traffic_twin.paths import RL_CONFIG
from traffic_twin.rl import evaluate as ev
from traffic_twin.rl.train import BASELINES, TOTAL_KEYS, file_sha256, stub_backend

RL_SHA = text_sha256(RL_CONFIG)
BASE = {"S0": 185_000.0, "S1": 172_000.0, "S2": 146_000.0, "stay": 184_000.0}
SCENARIOS = {"S0": "pitie_fixed", "S1": "pitie_actuated", "S2": "pitie_rules", "RL": "pitie_rl"}


# -- synthetic artifacts ---------------------------------------------------------------
def make_run(
    root: Path,
    algo: str = "dqn",
    seed: int = 0,
    best: float = 160_000.0,
    steps: int = 200_000,
    backend: str = "sumo",
    sha: str = RL_SHA,
    n_val: int = 4,
    preempt: bool = True,
) -> Path:
    """A finished training run folder as `rl.train` writes it (fake model bytes);
    `preempt` False: a validation.csv without the preemption columns (older runs)."""
    name = f"{algo}_s{seed}_{steps // 1000}k_{sha[:8]}" + ("" if backend == "sumo" else "_stub")
    d = Path(root) / name
    d.mkdir(parents=True)
    (d / "best_model.zip").write_bytes(f"model {name}".encode())
    (d / "final_model.zip").write_bytes(f"final {name}".encode())
    step = np.linspace(steps / n_val, steps, n_val).astype(int)
    cost = np.linspace(BASE["stay"], best, n_val)
    val = pd.DataFrame({"step": step, "cost": cost})
    for k, v in BASE.items():
        val[f"cost_vs_{k}"] = 100 * (cost / v - 1)
    for i, k in enumerate(TOTAL_KEYS):
        val[k] = cost / (10 + i)
    val["interventions"], val["iv_ped_wait"], val["danger"] = 12.0, 10.0, 0
    val["ped_curb_wait_max"], val["decisions"], val["advances"] = 88.0, 180.0, 60.0
    val["amb_seen_share"], val["best"] = 0.0, True
    if preempt:
        val["preempt_requests"], val["preempt_served"], val["preempt_timeouts"] = 6, 5, 1
    val.to_csv(d / "validation.csv", index=False)
    rng = np.random.default_rng(seed)
    n_ep = 120
    mon = pd.DataFrame(
        {
            "r": -300 + np.cumsum(rng.uniform(0, 1.5, n_ep)),
            "l": 160,
            "t": np.arange(n_ep, dtype=float),
        }
    )
    (d / "monitor.csv").write_text('#{"t_start": 0}\n' + mon.to_csv(index=False), encoding="utf-8")
    rows = [
        {
            "label": lab,
            "policy": pol,
            "seed": s,
            "cost": BASE[lab] + s,
            **{k: BASE[lab] / (10 + i) for i, k in enumerate(TOTAL_KEYS)},
        }
        for lab, pol in BASELINES.items()
        for s in (3000, 3001)
    ]
    pd.DataFrame(rows).to_csv(d / "baselines.csv", index=False)
    manifest = {
        "run": name,
        "algo": algo,
        "seed": seed,
        "steps": steps,
        "steps_done": steps,
        "n_envs": 8,
        "threads": 1,
        "backend": backend,
        "rl_yaml_sha256": sha,
        "obs_version": "rl-obs-1",
        "reward_version": "rl-reward-1",
        "code_version": "abc1234-dirty",
        "versions": {"python": "3.12", "torch": "2.14.1"},
        "wall_s": 600.0,
        "eval_s": 90.0,
        "steps_per_s": 420.0,
        "baselines": BASE,
        "best_step": int(step[-1]),
        "best_val_cost": float(best),
        "best_vs": {k: 100 * (best / v - 1) for k, v in BASE.items()},
        "best_model_sha256": file_sha256(d / "best_model.zip"),
        "best_policy_sha256": f"{seed:02d}" + "ab" * 31,
        "final_policy_sha256": "cd" * 32,
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def compare_rows(seeds=range(3020, 3030), rl_shift: float = -0.03) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for s in seeds:
        base = 90_000 + 1_000 * (s - 3020)
        costs = {
            "S0": base * 1.08,
            "S1": base,
            "S2": base * 0.88,
            "random": base * 2.3,
            "stay": base * 1.1,
            "RL": base * (1 + rl_shift + rng.normal(0, 0.02)),
        }
        for lab, c in costs.items():
            rows.append(
                {
                    "label": lab,
                    "policy": ev.COMPARE_POLICIES[lab],
                    "seed": s,
                    "cost": c,
                    **dict.fromkeys(TOTAL_KEYS, c / 10),
                    "interventions": 5.0,
                    "iv_ped_wait": 4.0,
                    "iv_max_green": 1.0,
                    "danger": 0,
                    "ped_curb_wait_max": 86.0,
                    "decisions": 170,
                    "advances": 60,
                    # random times a preemption out on every other window (D24 bound)
                    "amb_seen_share": 0.4 if lab == "random" and s % 2 else 0.0,
                    "preempt_requests": 1,
                    "preempt_served": 0 if lab == "random" and s % 2 else 1,
                    "preempt_timeouts": 1 if lab == "random" and s % 2 else 0,
                }
            )
    return pd.DataFrame(rows)


def fake_model(root: Path, content: bytes = b"policy-a", version: str = "rl1-dqn-s0-00abab"):
    """A model.zip with its manifest.json (only its sha256 matters to `kpi`)."""
    d = Path(root) / f"model_{version}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.zip").write_bytes(content)
    sha = file_sha256(d / "model.zip")
    manifest = {"algo": "dqn", "model_version": version, "model_sha256": sha}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d / "model.zip", sha


def make_kpi_run(
    root: Path,
    label: str,
    seed: int,
    danger: int = 0,
    curb: float = 88.0,
    model_sha256: str | None = None,
    preemption: dict | None = None,
) -> None:
    d = Path(root) / f"{SCENARIOS[label]}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    shift = {"S0": 10.0, "S1": 0.0, "S2": -8.0, "RL": -2.0}[label] + seed % 3
    kpis = [
        ("car", "time_loss_mean", 60 + shift),
        ("car", "delay_total_mean", 61 + shift),
        ("car", "depart_delay_mean", 0.1),
        ("bus", "delay_mean", 55 + shift),
        ("pedestrian", "ped_curb_wait_max", curb),
        ("pedestrian", "ped_curb_wait_mean", 18.0),
        ("ambulance", "travel_time_total", 100 + shift),
    ]
    pd.DataFrame(
        [
            {
                "run_id": "x",
                "scenario": SCENARIOS[label],
                "seed": seed,
                "user_type": u,
                "kpi": k,
                "value": v,
            }
            for u, k, v in kpis
        ]
    ).to_csv(d / "kpi_run.csv", index=False)
    pd.DataFrame(
        {
            "id": ["c1", "c2", "b1", "p1"],
            "type": ["car", "car", "bus", "pedestrian"],
            "time_loss_s": [50 + shift, 70 + shift, 40.0, 10.0],
            "depart_delay_s": [0.0, 1.0, 0.0, None],
        }
    ).to_parquet(d / "trip.parquet", index=False)
    meta = {
        "run_id": "x",
        "scenario": SCENARIOS[label],
        "seed": seed,
        "code_version": "abc1234",
        "config_hash": "h",
        "demand_hash": "83a4f220fd20",
        "health": {
            "collisions": 0,
            "teleports": 0,
            "drain_timeout": False,
            "depart_delay_mean": 0.1,
            "depart_delay_max": 20.0,
        },
        "safety": {"danger": danger, "interventions": {"ped_wait": 3 if label == "RL" else 0}},
        "preemption": preemption
        or (
            {"enabled": True, "requests": 1, "served": 1, "timeouts": 0, "active_s": 30.0}
            if label in ("S2", "RL")  # S0 / S1 run without preemption
            else {"enabled": False, "requests": 0, "served": 0, "timeouts": 0}
        ),
        "control": {
            "model_version": "rl1-dqn-s0-00abab",
            "model_sha256": model_sha256,
            "decisions": 800,
        }
        if label == "RL"
        else {},
    }
    (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")


def timing_doc() -> dict:
    eps = []
    for pol in ev.TIMING_POLICIES:
        for s in (3000, 3001, 3002):
            row = {
                "policy": pol,
                "seed": s,
                "wall_s": 2.0 + 0.3 * (s - 3000),
                "decisions": 0 if pol in ("fixed", "actuated", "rules") else 820,
            }
            if pol not in ("fixed", "actuated", "rules"):
                row |= {"reset_s": 0.2, "step_s": 1.8, "nn_s": 0.1, "decisions_per_s": 350.0}
            eps.append(row)
    eps[0]["construct_s"] = 1.4
    scaling = [
        {
            "n_envs": n,
            "steps": 5000,
            "wall_s": 5000 / (200 * n**0.7),
            "steps_per_s": 200 * n**0.7,
            "episodes": n,
            "startup_s": 2.0 + n / 4,
        }
        for n in (1, 4, 8, 12)
    ]
    return {
        "created_at": "2026-10-16T00:00:00+00:00",
        "code_version": "abc1234-dirty",
        "seeds": [3000, 3001, 3002],
        "cpu_count": 32,
        "platform": "Windows-11",
        "episodes": eps,
        "summary": ev.timing_summary(eps),
        "scaling": scaling,
        "scaling_steps": 5000,
    }


def synthetic_artifacts(root: Path) -> dict[str, Path]:
    """A complete set of RL artifacts under `root`: 3 DQN S6 runs (2 pass), 1 PPO run, a
    stub run, a selected model, a dev-test comparison, full 2 h runs, timing, repro."""
    runs = root / "rl_runs"
    make_run(runs, "dqn", 0, best=160_000)  # -13.5 % vs S0, -13.0 % vs stay: pass
    make_run(runs, "dqn", 1, best=170_000)  # -8.1 % / -7.6 %: pass
    make_run(runs, "dqn", 2, best=178_000)  # -3.8 %: fail
    make_run(runs, "ppo", 0, best=175_000)
    make_run(runs, "dqn", 3, best=1.0, backend="stub")
    selected = root / "selected"
    ev.select(RL_CONFIG, ev.find_runs(runs), "dqn", selected)
    devtest = runs / "devtest"
    devtest.mkdir()
    rows = compare_rows()
    rows.to_csv(devtest / "compare.csv", index=False)
    (devtest / "compare_summary.json").write_text(
        json.dumps(
            {
                "seeds": list(range(3020, 3030)),
                "code_version": "abc1234-dirty",
                "model": {"model_version": "rl1-dqn-s0-00abab", "algo": "dqn"},
                "summary": ev.compare_summary(rows),
            },
            default=ev._jsonable,
        ),
        encoding="utf-8",
    )
    model = selected / "model.zip"
    for lab in SCENARIOS:
        for s in (3020, 3021, 3022):
            make_kpi_run(devtest, lab, s, model_sha256=file_sha256(model))
    ev.kpi(RL_CONFIG, [3020, 3021, 3022], model=model, out_root=devtest, summary_only=True)
    (runs / "timing").mkdir()
    (runs / "timing" / "timing.json").write_text(json.dumps(timing_doc()), encoding="utf-8")
    (runs / "repro.json").write_text(
        json.dumps(
            {
                "same": True,
                "n_envs": [4, 4],
                "checks": {"final_policy_sha256": True, "monitor_r_l": True},
            }
        ),
        encoding="utf-8",
    )
    return {"runs": runs, "selected": selected, "devtest": devtest}


# -- seeds -----------------------------------------------------------------------------
def test_parse_seeds():
    assert ev.parse_seeds(["3020..3022"]) == [3020, 3021, 3022]
    assert ev.parse_seeds(["3020-3021", "3025,3020", 3030]) == [3020, 3021, 3025, 3030]
    with pytest.raises(ValueError):
        ev.parse_seeds(["3022..3020"])
    with pytest.raises(ValueError):
        ev.parse_seeds(["30x"])


def test_check_seeds(rl_train_scenario):
    cfg = rl_train_scenario
    ev.check_seeds([3020, 3049], cfg, ("dev_test",), "compare")
    ev.check_seeds([3000, 3020], cfg, ("dev_test", "selection"), "kpi")
    for seeds in ([1], [101], [205]):  # Part 3-4 seeds: always refused
        with pytest.raises(ConfigError, match="Parts 3-4"):
            ev.check_seeds(seeds, cfg, ("dev_test",), "kpi")
    with pytest.raises(ConfigError, match="outside"):
        ev.check_seeds([3010], cfg, ("dev_test", "selection"), "compare")  # burned
    with pytest.raises(ConfigError, match="outside"):
        ev.check_seeds([3020], cfg, ("selection", "train"), "timing")
    with pytest.raises(ConfigError, match="no seeds"):
        ev.check_seeds([], cfg, ("dev_test",), "kpi")


def test_never_writes_under_data_results(tmp_path):
    with pytest.raises(ConfigError, match="data/results"):
        ev.check_out_root(Path("data/results/rl"))
    ev.check_out_root(tmp_path)
    with pytest.raises(ConfigError, match="data/results"):
        ev.kpi_runs([3020], Path("data/results"), {})


@pytest.mark.parametrize("key", ["runs_root", "out"])
def test_timing_refuses_data_results(tmp_path, monkeypatch, key):
    """rl.yaml `outputs.runs` (the timing work_root) or `--out` under data/results is
    refused before any episode runs (cwd is a scratch folder)."""
    cfg = load_config(RL_CONFIG)
    out = None
    if key == "runs_root":
        cfg.setdefault("outputs", {})["runs"] = "data/results"
    else:
        out = Path("data/results/timing.json")
    config = tmp_path / "rl.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "configs").mkdir()  # check_seeds reads the Part 3-4 seeds from it
    shutil.copy(ev.EVALUATION_CONFIG, tmp_path / "configs" / "evaluation.yaml")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="data/results"):
        ev.timing(config, n_envs=(), out=out, backend=stub_backend(window_s=300))
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    ("ep", "text"),
    [
        (
            {"begin_s": 27000, "warmup_s": 0, "window_s": 7200},
            "07:30-09:30, no warm-up, truncated at 7,200 s, no drain",
        ),
        (
            {"begin_s": 27000, "warmup_s": 300, "window_s": 6900, "warmup_controller": "fixed"},
            "07:35-09:30, 300 s of `fixed` warm-up from 07:30, truncated at 6,900 s, no drain",
        ),
        (None, "full-mode episodes (rl.yaml `episode.full`), no drain"),
    ],
)
def test_full_episode_text_follows_the_parameters(ep, text):
    assert ev.full_episode_text(ep) == text


def test_full_episode_times_of_the_config():
    ep = ev.full_episode_times(load_config(RL_CONFIG))
    assert ep is not None and ep["warmup_s"] == 0 and ep["window_s"] == 7200


def test_docs_note_validation_without_preemption_columns(tmp_path):
    art = synthetic_artifacts(tmp_path)
    make_run(art["runs"], "dqn", 4, best=150_000, steps=100_000, preempt=False)
    docs = tmp_path / "docs" / "rl.md"
    text = ev.build_docs(
        RL_CONFIG, docs, tmp_path / "figs", art["runs"], art["selected"]
    ).read_text(encoding="utf-8")
    assert "`dqn_s4_100k_" in text and "predate the preemption columns" in text
    assert "preemptions 5/6 (1 timeouts)" in text


# -- selection -------------------------------------------------------------------------
def test_select_picks_the_lowest_eligible_run(tmp_path):
    runs = tmp_path / "runs"
    make_run(runs, "dqn", 0, best=170_000)
    best = make_run(runs, "dqn", 1, best=160_000)
    make_run(runs, "dqn", 2, best=150_000, sha="0" * 64)  # other rl.yaml
    make_run(runs, "dqn", 3, best=100_000, backend="stub")
    ppo = make_run(runs, "ppo", 0, best=155_000)
    (runs / "_work").mkdir()
    assert len(ev.find_runs(runs)) == 5
    out = ev.select(RL_CONFIG, ev.find_runs(runs), "dqn", tmp_path / "sel")
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["source_run"] == best.name and m["algo"] == "dqn"
    assert (out / "model.zip").read_bytes() == (best / "best_model.zip").read_bytes()
    assert m["model_sha256"] == file_sha256(out / "model.zip")
    # the keys control.rl.RLController reads
    assert {m["obs_version"], m["reward_version"]} == {"rl-obs-1", "rl-reward-1"}
    assert m["model_version"].startswith("rl1-dqn-s1-")
    excluded = {c["run"]: c["excluded"] for c in m["selection"]["candidates"]}
    assert excluded[best.name] is None and "rl.yaml" in excluded["dqn_s2_200k_00000000"]
    assert "backend" in excluded[next(k for k in excluded if k.endswith("_stub"))]
    out2 = ev.select(RL_CONFIG, ev.find_runs(runs), None, tmp_path / "sel_any")
    assert json.loads((out2 / "manifest.json").read_text())["source_run"] == ppo.name
    with pytest.raises(ConfigError, match="no eligible"):
        ev.select(RL_CONFIG, [runs / "dqn_s2_200k_00000000"], "dqn", tmp_path / "none")


def test_select_refuses_a_tampered_model(tmp_path):
    d = make_run(tmp_path / "runs", "dqn", 0)
    (d / "best_model.zip").write_bytes(b"changed")
    with pytest.raises(ConfigError, match="sha256"):
        ev.select(RL_CONFIG, [d], "dqn", tmp_path / "sel")


# -- dev-test comparison ---------------------------------------------------------------
def test_compare_summary_uses_the_exact_wilcoxon():
    rows = compare_rows()
    s = ev.compare_summary(rows)
    assert set(s) == {"RL", "S0", "S1", "S2", "random", "stay"}
    rl = rows[rows["label"] == "RL"].sort_values("seed")["cost"].to_numpy()
    s1 = rows[rows["label"] == "S1"].sort_values("seed")["cost"].to_numpy()
    v = s["RL"]["vs_S1"]
    assert v["n"] == 10 and v["p"] == pytest.approx(wilcoxon(rl, s1).pvalue)
    assert v["wins"] == int((rl < s1).sum())
    assert v["pct_mean"] == pytest.approx(np.mean(100 * (rl / s1 - 1)))
    assert "vs_S1" not in s["S1"] and s["S2"]["vs_S1"]["pct_mean"] == pytest.approx(-12.0)
    assert s["S2"]["vs_S1"]["p"] == pytest.approx(2 / 1024)  # 10 of 10 wins
    assert s["RL"]["iv_ped_wait"] == 4.0 and s["RL"]["danger_max"] == 0


def test_paired_vs_ignores_unpaired_seeds():
    rows = compare_rows(seeds=range(3020, 3026))
    rows = rows[~((rows["label"] == "RL") & (rows["seed"] == 3020))]
    assert ev.paired_vs(rows, "RL", "S1")["n"] == 5
    assert ev.paired_vs(rows, "RL", "nobody") == {"n": 0}


def test_compare_on_the_stub(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import DQN

    from traffic_twin.rl.stub import StubTrafficEnv

    model = DQN(
        "MlpPolicy",
        StubTrafficEnv(window_s=300),
        seed=0,
        device="cpu",
        policy_kwargs={"net_arch": [8]},
        learning_starts=10,
    )
    mdir = tmp_path / "model"
    mdir.mkdir()
    model.save(str(mdir / "model.zip"))
    (mdir / "manifest.json").write_text(json.dumps({"algo": "dqn", "model_version": "t"}))
    out = ev.compare(
        RL_CONFIG,
        [3020, 3021],
        mdir / "model.zip",
        1,
        tmp_path / "devtest",
        backend=stub_backend(window_s=300),
    )
    doc = json.loads(out.read_text(encoding="utf-8"))
    rows = pd.read_csv(tmp_path / "devtest" / "compare.csv")
    assert len(rows) == 2 * len(ev.COMPARE_POLICIES)
    assert doc["summary"]["RL"]["vs_S1"]["n"] == 2 and doc["model"]["algo"] == "dqn"
    with pytest.raises(ConfigError, match="Parts 3-4"):
        ev.compare(
            RL_CONFIG,
            [5],
            mdir / "model.zip",
            1,
            tmp_path / "x",
            backend=stub_backend(window_s=300),
        )


def test_resolve_model_needs_an_algo(tmp_path, rl_train_scenario):
    (tmp_path / "m.zip").write_bytes(b"x")
    with pytest.raises(ConfigError, match="algo"):
        ev.resolve_model(tmp_path / "m.zip", rl_train_scenario)
    assert ev.resolve_model(tmp_path / "m.zip", rl_train_scenario, "ppo")["algo"] == "ppo"
    with pytest.raises(ConfigError, match="no RL model"):
        ev.resolve_model(tmp_path / "missing.zip", rl_train_scenario)


# -- full 2 h runs ---------------------------------------------------------------------
def test_kpi_summary_and_safety(tmp_path):
    model, sha = fake_model(tmp_path / "m")
    for lab in SCENARIOS:
        for s in (3020, 3021, 3022):
            make_kpi_run(tmp_path, lab, s, model_sha256=sha)
    summary = ev.kpi(
        RL_CONFIG, [3020, 3021, 3022], model=model, out_root=tmp_path, summary_only=True
    )
    assert summary["safety_ok"] and summary["violations"] == []
    k = summary["kpis"]["car/time_loss_mean"]
    assert set(k) == {"S0", "S1", "S2", "RL"} and k["RL"]["n"] == 3
    assert k["RL"]["vs_S1"]["diff_mean"] == pytest.approx(-2.0)
    assert "all/person_delay_mean" in summary["kpis"]
    assert summary["safety"]["RL"]["iv_ped_wait"] == 3.0
    assert (tmp_path / "kpi.csv").is_file() and (tmp_path / "kpi_summary.json").is_file()
    safety = pd.read_csv(tmp_path / "kpi_safety.csv")
    assert set(safety["model_version"].dropna()) == {"rl1-dqn-s0-00abab"}
    # model and code identity of the runs, preemption outcomes per scenario
    assert summary["models"]["RL"]["model_sha256"] == sha
    assert summary["models"]["RL"]["model_version"] == "rl1-dqn-s0-00abab"
    assert summary["runs"]["code_version"] == ["abc1234"]
    assert summary["safety"]["RL"]["preempt_requests"] == 3
    assert summary["safety"]["S1"]["preempt_enabled"] is False
    assert summary["safety"]["RL"]["preempt_enabled"] is True
    assert {"preempt_requests", "preempt_served", "preempt_timeouts"} <= set(safety.columns)


def test_kpi_refuses_rl_runs_of_another_model(tmp_path):
    """config_hash holds only the model path: a run of an older policy written to the same
    path must not be reported under the newly selected model."""
    old_model, old_sha = fake_model(tmp_path / "old", b"policy-a")
    for lab in SCENARIOS:
        make_kpi_run(tmp_path, lab, 3020, model_sha256=old_sha)
    new_model, _ = fake_model(tmp_path / "new", b"policy-b", "rl1-dqn-s1-11cdcd")
    with pytest.raises(ConfigError, match="model sha256"):
        ev.kpi(RL_CONFIG, [3020], model=new_model, out_root=tmp_path, summary_only=True)
    ev.kpi(RL_CONFIG, [3020], model=old_model, out_root=tmp_path, summary_only=True)


def test_current_needs_the_same_model_sha(tmp_path, monkeypatch):
    """`kpi_runs` reruns an rl run whose run.json model sha256 differs from the model the
    scenario resolves to, even when the config_hash is unchanged."""
    _, sha_a = fake_model(tmp_path / "a", b"policy-a")
    model_b, sha_b = fake_model(tmp_path / "b", b"policy-b", "rl1-dqn-s1-11cdcd")
    make_kpi_run(tmp_path, "RL", 3020, model_sha256=sha_a)
    cfgs = ev.kpi_configs(model_b, {"RL": ev.KPI_CONFIGS["RL"]})
    cfg = cfgs["RL"][1]
    meta_path = tmp_path / "pitie_rl_seed3020" / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["config_hash"] = ev.config_hash(cfg)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    run_dir = tmp_path / "pitie_rl_seed3020"
    ident = ev.rl_model_identity(cfg)
    assert ident["model_sha256"] == sha_b
    assert not ev._current(run_dir, cfg, ident)
    meta["control"]["model_sha256"] = sha_b
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert ev._current(run_dir, cfg, ident)

    ran = []
    import traffic_twin.sim.run_scenario as rs

    monkeypatch.setattr(rs, "run", lambda cfg, seed, out_root, **kw: ran.append(seed))
    meta["control"]["model_sha256"] = sha_a
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    done = ev.kpi_runs([3020], tmp_path, cfgs)
    assert ran == [3020] and done[0]["skipped"] is False


@pytest.mark.parametrize(
    ("danger", "curb", "why"), [(1, 88.0, "danger 1"), (0, 91.0, "curb wait 91 s > 90 s")]
)
def test_kpi_safety_violation_stops(tmp_path, danger, curb, why):
    model, sha = fake_model(tmp_path / "m")
    for lab in SCENARIOS:
        args = (danger, curb) if lab == "RL" else (0, 88.0)
        make_kpi_run(tmp_path, lab, 3020, *args, model_sha256=sha)
    argv = ["kpi", "--seeds", "3020", "--out", str(tmp_path), "--summary-only"]
    argv += ["--model", str(model)]
    assert ev.main(argv) == ev.SAFETY_EXIT
    v = json.loads((tmp_path / "kpi_summary.json").read_text(encoding="utf-8"))["violations"]
    assert v == [f"RL seed 3020: {why}"]


def test_kpi_missing_run_is_an_error(tmp_path):
    model, _ = fake_model(tmp_path / "m")
    with pytest.raises(ConfigError, match="missing run"):
        ev.kpi(RL_CONFIG, [3020], model=model, out_root=tmp_path, summary_only=True)


# -- S6 criterion ----------------------------------------------------------------------
def _m(seed: int, best: float, steps: int = 200_000, algo: str = "dqn") -> dict:
    return {
        "algo": algo,
        "seed": seed,
        "steps": steps,
        "backend": "sumo",
        "run": f"r{seed}",
        "rl_yaml_sha256": RL_SHA,
        "best_val_cost": best,
        "baselines": {"S0": 100.0, "stay": 100.0},
    }


@pytest.mark.parametrize(
    ("bests", "verdict", "n_pass"),
    [
        ((95.0, 95.0, 99.0), "PASS", 2),  # exactly 5 % below twice: passes
        ((95.0, 95.01, 99.0), "FAIL", 1),  # 4.99 % is not enough
        ((90.0, 90.0, 90.0), "PASS", 3),
        ((95.0, None, 99.0), "INCOMPLETE", 1),  # the missing run could still pass
        ((99.0, None, 99.0), "FAIL", 0),  # it could not reach 2 any more
        ((90.0, 90.0, None), "PASS", 2),
        ((None, None, None), "INCOMPLETE", 0),
    ],
)
def test_s6_verdict_edges(rl_train_scenario, bests, verdict, n_pass):
    ms = [_m(seed, b) for seed, b in zip((0, 1, 2), bests, strict=True) if b is not None]
    ms += [_m(0, 50.0, steps=60_000), _m(1, 50.0, algo="ppo")]  # not S6 runs: ignored
    v = ev.s6_verdict(ms, rl_train_scenario, RL_SHA)
    assert (v["verdict"], v["n_pass"]) == (verdict, n_pass)
    assert v["of"] == 3 and v["required"] == 2


def test_s6_verdict_needs_both_references(rl_train_scenario):
    ms = [_m(s, 90.0) for s in (0, 1, 2)]
    for m in ms:
        m["baselines"] = {"S0": 100.0, "stay": 93.0}  # only 3.2 % below always-stay
    assert ev.s6_verdict(ms, rl_train_scenario)["verdict"] == "FAIL"


# -- reproducibility -------------------------------------------------------------------
def test_repro_check(tmp_path):
    a = make_run(tmp_path / "a", "dqn", 0)
    b = make_run(tmp_path / "b", "dqn", 0)
    res = ev.repro_check(a, b)
    assert res["same"], res["checks"]
    mon = ev.read_monitor(b / "monitor.csv").drop(columns="steps")
    mon.loc[3, "r"] += 0.5
    (b / "monitor.csv").write_text("#{}\n" + mon.to_csv(index=False), encoding="utf-8")
    assert not ev.repro_check(a, b)["checks"]["monitor_r_l"]
    c = make_run(tmp_path / "c", "dqn", 1)
    assert not ev.repro_check(a, c)["checks"]["best_policy_sha256"]


# -- docs ------------------------------------------------------------------------------
def test_docs_from_synthetic_artifacts(tmp_path):
    art = synthetic_artifacts(tmp_path)
    docs = tmp_path / "docs" / "rl.md"
    figs = tmp_path / "docs" / "figures" / "rl"
    out = ev.build_docs(RL_CONFIG, docs, figs, art["runs"], art["selected"])
    text = out.read_text(encoding="utf-8")
    for n, title in enumerate(
        [
            "Environment",
            "Pre-registration",
            "Episode timing (S6 measurement)",
            "Trainings",
            "Learning curves",
            "S6 criterion",
            "Dev-test window comparison",
            "Full 2 h runs",
            "Safety",
            "Ambulance in the state",
            "Reproducibility",
            "Limitations and the 10.6 fallback",
            "Part 7 plan",
        ],
        start=1,
    ):
        assert f"## {n}. {title}" in text
    assert "**Verdict: PASS** (2 of 3 runs pass)" in text
    assert "dqn_s3_200k" not in text  # stub runs are never reported
    assert "Not available yet" not in text
    assert "rl1-dqn-s0-" in text and "10.6" in text and "ASSUMED" in text
    for name in ("learning_curves", "validation_totals", "episode_timing", "devtest_compare"):
        assert (figs / f"{name}.png").is_file()
        assert f"(figures/rl/{name}.png)" in text
    rl_cfg = load_config(RL_CONFIG)
    assert f"pre-registered **{rl_cfg['preregistered']}**" in text
    # timing protocol from the episode parameters (rl.yaml full: no warm-up)
    assert "300 s warm-up" not in text
    assert "Full episodes (07:30-09:30, no warm-up, truncated at 7,200 s, no drain)" in text
    # preemption is bounded; its outcomes are reported per policy
    assert "bounded, not an unconditional guarantee" in text and "120 s" in text
    assert "| random | 10 | 5 | 5 | 5/10 |" in text
    assert "Preemptions served/requests (timeouts)" in text
    assert "S0 off, S1 off, S2 3/3 (0), RL 3/3 (0)" in text
    assert "predate the preemption columns" not in text
    # the 10.6 status carries its caveats; the ideal detection is declared
    assert "significantly" not in text
    assert "exploratory and unadjusted; env cost only, on 10 30-min windows" in text
    assert "picked from 3 runs" in text and "Against S2 (rules)" in text
    assert "Safety Guard interventions per 2 h run" in text
    assert "ASSUMED ideal lane detection" in text and "information advantage" in text
    # the full runs name the model they were checked against and their code versions
    assert "checked against every run.json" in text and "`abc1234`" in text


def test_docs_without_artifacts(tmp_path):
    docs = tmp_path / "rl.md"
    ev.main(
        [
            "docs",
            "--docs",
            str(docs),
            "--figures",
            str(tmp_path / "fig"),
            "--runs",
            str(tmp_path / "empty"),
        ]
    )
    text = docs.read_text(encoding="utf-8")
    assert "S6 criterion **INCOMPLETE**" in text and "Episode timing" in text
    assert text.count("Not available yet") >= 5

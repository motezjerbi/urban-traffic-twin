"""Part 7 weight sensitivity (D76, `rl.sensitivity`) without SUMO: the hierarchical
bootstrap on synthetic nested data, the no-retraining reweighting against the reward
tracker, pick discovery, and the run -> tables pipeline on fake run folders (the
simulation is replaced by a writer of run.json / kpi_run.csv / trip.parquet /
reward_totals.json). Seeds are the stub workspace's development seeds."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from traffic_twin.config import ConfigError, config_hash, load_config
from traffic_twin.control.base import EXTRA_PED_COUNT, EXTRA_PED_WAIT
from traffic_twin.rl import sensitivity as sens
from traffic_twin.rl.reward import RewardTracker
from traffic_twin.rl.train import run_config_sha

# per-setting effects of the fake runs (s): car time loss, bus delay, ped curb mean
EFFECTS = {
    "t1": (0.0, 0.0, 0.0),
    "wb12": (-4.0, 9.0, 0.0),
    "wb50": (9.0, -5.0, 0.5),
    "lam100": (5.0, 1.0, -0.6),
    "S0": (8.0, 6.0, -1.5),
    "S1": (4.0, 12.0, -1.0),
    "S2": (-3.0, 7.0, -0.2),
}
BASE_TOTALS = {
    "car_s": 100_000.0,
    "car_pending_s": 900.0,
    "bus_s": 4_000.0,
    "pedestrian_s": 30_000.0,
    "ambulance_s": 80.0,
    "penalty_s": 150.0,
}


# -- bootstrap -------------------------------------------------------------------------
def test_bootstrap_paired_seed_effects_cancel():
    """Shared evaluation-seed effects cancel (paired design): a pure shift d has the
    degenerate CI [d, d] whatever the seed effects and the number of policies."""
    rng = np.random.default_rng(1)
    u = rng.normal(50, 10, size=12)  # per evaluation seed
    a = np.tile(u + 3.0, (3, 1))
    b = np.tile(u, (5, 1))
    d, lo, hi = sens.hierarchical_bootstrap(a, b, n=500, seed=0)
    assert d == pytest.approx(3.0)
    assert lo == pytest.approx(3.0) and hi == pytest.approx(3.0)


def test_bootstrap_training_seed_level_known_ci():
    """Two policies 0 and 2 against a constant base: the resampled mean is 0, 1 or 2
    (p 1/4, 1/2, 1/4), so the 95 % percentile CI is exactly [0, 2]; one evaluation seed."""
    a = np.array([[0.0], [2.0]])
    b = np.array([[0.0]])
    d, lo, hi = sens.hierarchical_bootstrap(a, b, n=2000, seed=0)
    assert (d, lo, hi) == (1.0, 0.0, 2.0)


def test_bootstrap_nested_and_deterministic():
    rng = np.random.default_rng(7)
    seeds = rng.normal(0, 5, size=20)
    a = 2.0 + rng.normal(0, 1, size=(5, 1)) + seeds + rng.normal(0, 1, size=(5, 20))
    b = rng.normal(0, 1, size=(5, 1)) + seeds + rng.normal(0, 1, size=(5, 20))
    r1 = sens.hierarchical_bootstrap(a, b, n=2000, seed=0)
    r2 = sens.hierarchical_bootstrap(a, b, n=2000, seed=0)
    assert r1 == r2
    d, lo, hi = r1
    assert lo < d < hi and lo > 0  # a true +2 effect, policy SD 1: beyond noise
    assert sens.hierarchical_bootstrap(a, b, n=2000, seed=1) != r1
    # the training-seed level widens the interval against a single policy per arm
    _, lo1, hi1 = sens.hierarchical_bootstrap(a[:1], b[:1], n=2000, seed=0)
    assert hi - lo > 0 and hi1 - lo1 > 0


def test_bootstrap_nan_and_shape():
    a = np.array([[1.0, np.nan], [3.0, 5.0]])
    b = np.array([[1.0, 1.0]])
    d, lo, hi = sens.hierarchical_bootstrap(a, b, n=200)
    assert d == pytest.approx(3.0 - 1.0)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert all(math.isnan(x) for x in sens.hierarchical_bootstrap(np.full((2, 2), np.nan), b))
    with pytest.raises(ValueError):
        sens.hierarchical_bootstrap(np.ones((2, 3)), np.ones((2, 2)))


# -- reweighting -----------------------------------------------------------------------
class _FakeSim:
    """The TraCI surface RewardTracker.accrue reads, replaying a fixed step sequence."""

    def __init__(self, steps):
        self.steps, self.i, self.dt = steps, 0, 1.0
        types = {"c1": "car", "c2": "car", "b1": "bus", "a1": "ambulance", "c3": "car"}
        self.traci = SimpleNamespace(
            vehicle=SimpleNamespace(
                getIDList=lambda: list(self.steps[self.i]["wait"]),
                getWaitingTime=lambda v: self.steps[self.i]["wait"][v],
                getTypeID=lambda v: types[v],
            ),
            simulation=SimpleNamespace(getPendingVehicles=lambda: self.steps[self.i]["pending"]),
        )

    def observe(self):
        s = self.steps[self.i]
        return SimpleNamespace(extras={EXTRA_PED_COUNT: s["ped"], EXTRA_PED_WAIT: s["curb"]})


def _accrue(reward: dict, steps) -> dict:
    sim = _FakeSim(steps)
    t = RewardTracker(reward)
    for i in range(len(steps)):
        sim.i = i
        t.accrue(sim)
    return t.totals


def test_reweighting_equals_recomputed_cost(rl7_cfg):
    rng = np.random.default_rng(3)
    steps = []
    for _ in range(300):
        wait = {v: float(rng.integers(0, 2)) for v in ("c1", "c2", "b1", "a1")}
        steps.append(
            {
                "wait": wait,
                "pending": ["c3"] if rng.random() < 0.3 else [],
                "ped": tuple(int(x) for x in rng.integers(0, 4, size=4)),
                "curb": tuple(float(x) for x in rng.uniform(0, 95, size=4)),
            }
        )
    base = rl7_cfg["reward"]
    totals = _accrue(base, steps)
    assert sens.reweighted_cost(totals, base) == pytest.approx(totals["cost"])
    for setting in ("wb12", "wb50", "wp48", "lam0", "lam100", "transit_ped", "car_centric"):
        other = sens.reward_block(Path("configs/rl7") / f"{setting}.yaml")
        retrained = _accrue(other, steps)  # the same steps accrued under the other weights
        assert sens.reweighted_cost(totals, other) == pytest.approx(retrained["cost"])
    frame = pd.DataFrame([totals, totals])
    assert list(sens.reweighted_cost(frame, base)) == pytest.approx([totals["cost"]] * 2)


def test_weights_and_direction():
    base = sens.reward_block(Path("configs/rl7/t1.yaml"))
    assert sens.weights_row(base) == {
        "w_car": 1.3,
        "w_bus": 25.0,
        "w_pedestrian": 1.2,
        "w_ambulance": 20.0,
        "lambda": 30.0,
    }
    want = {
        "t1": "base",
        "wb12": "car",
        "wb50": "transit_ped",
        "wp06": "car",
        "wp24": "transit_ped",
        "wp48": "transit_ped",
        "lam0": "car",
        "lam100": "transit_ped",
        "transit_ped": "transit_ped",
        "car_centric": "car",
    }
    for setting, d in want.items():
        assert sens.direction(sens.reward_block(Path(f"configs/rl7/{setting}.yaml")), base) == d
    assert sens.direction({"weights": {"bus": 50.0, "pedestrian": 0.6}}, base) == "mixed"


# -- config, seeds, picks --------------------------------------------------------------
def test_settings_and_seeds(rl7_cfg):
    assert sens.settings(rl7_cfg) == [
        "t1",
        "wb12",
        "wb50",
        "wp06",
        "wp24",
        "wp48",
        "lam0",
        "lam100",
        "transit_ped",
        "car_centric",
    ]
    assert sens.setting_seeds(rl7_cfg, "wb50") == [0, 1, 2, 3, 4]
    assert sens.setting_seeds(rl7_cfg, "t1") == [0, 1, 2, 3, 4]
    assert sens.setting_seeds(rl7_cfg, "lam0") == [0, 1, 2]
    assert sum(len(sens.setting_seeds(rl7_cfg, s)) for s in sens.settings(rl7_cfg)) == 36
    assert sens.eval_seeds(rl7_cfg) == list(range(3160, 3180))


def test_eval_seeds_refuse_other_ranges(rl7_cfg):
    assert sens.eval_seeds(rl7_cfg, [5000, 5001]) == [5000, 5001]  # dev: training seeds
    for bad in ([1, 2], [3020], [3050], [3110], [2500]):
        with pytest.raises(ConfigError):
            sens.eval_seeds(rl7_cfg, bad)


def make_pick_run(
    root: Path,
    rl7: Path,
    setting: str,
    seed: int,
    steps: int,
    stale: bool = False,
    stage: str = "T",
) -> Path:
    """A finished training run folder (manifest, provenance, pick_model.zip) of a stub
    workspace config."""
    config = sens.setting_config(rl7, setting)
    sha = run_config_sha(load_config(config))
    d = root / stage / f"dqn_{setting}_s{seed}_{steps // 1000}k_{sha[:8]}"
    d.mkdir(parents=True)
    (d / sens.PICK_FILE).write_bytes(f"pick {setting} {seed}".encode())
    base = {"S0": 90_000.0, "S1": 84_000.0, "S2": 79_000.0, "stay": 100_000.0}
    pick = 79_000.0 * (1 + 0.01 * (seed - 1)) * (1.02 if setting == "lam100" else 1.0)
    manifest = {
        "run": d.name,
        "algo": "dqn",
        "seed": seed,
        "steps": steps,
        "variant": setting,
        "obs_version": "rl-obs-1",
        "reward_version": "rl-reward-1",
        "backend": "sumo",
        "pick_step": steps,
        "pick_val_cost": pick,
        "late_mean": pick * 1.01,
        "baselines": base,
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    prov = {"run_config_sha": "0" * 64 if stale else sha, "variant": setting}
    (d / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    return d


def test_find_picks(stub_rl7_cfg):
    rl7 = stub_rl7_cfg()
    cfg = load_config(rl7)
    root = Path(cfg["outputs"]["runs"])
    steps = int(cfg["protocol"]["sensitivity"]["steps"])
    for seed in range(5):
        make_pick_run(root, rl7, "t1", seed, steps, stage="T" if seed < 3 else "S")
    make_pick_run(root, rl7, "wb50", 0, steps, stage="S")
    make_pick_run(root, rl7, "wb50", 1, steps, stale=True, stage="S")
    make_pick_run(root, rl7, "wb50", 2, steps + 1000, stage="S")  # another budget
    picks, missing = sens.find_picks(rl7, only=["t1", "wb50"])
    assert [p.label for p in picks] == ["t1_s0", "t1_s1", "t1_s2", "t1_s3", "t1_s4", "wb50_s0"]
    assert [m.split(" (")[0] for m in missing] == ["wb50 s1", "wb50 s2", "wb50 s3", "wb50 s4"]
    assert picks[0].model.name == sens.PICK_FILE and picks[3].run_dir.parent.name == "S"
    _, missing_all = sens.find_picks(rl7)
    assert len(missing_all) == 36 - 6
    with pytest.raises(ConfigError, match="no run config"):
        sens.find_picks(rl7, only=["nope"])
    with pytest.raises(ConfigError, match="picks missing"):
        sens.run(rl7, workers=1, only=["wb50"])


# -- run -> tables on fake runs ----------------------------------------------------------
def _kpi_rows(label: str, setting: str, tseed: int, seed: int) -> pd.DataFrame:
    car, bus, ped = EFFECTS[setting]
    noise = 0.3 * tseed + 0.1 * (seed % 7)
    rows = [
        ("car", "count", 2300.0),
        ("car", "time_loss_mean", 55.0 + car + noise),
        ("car", "delay_total_mean", 55.5 + car + noise),
        ("car", "depart_delay_mean", 0.05),
        ("car", "delay_total_mean__arm_N", 50.0 + car),
        ("car", "delay_total_mean__arm_W", 85.0 + car),
        ("bus", "count", 94.0),
        ("bus", "delay_mean", 48.0 + bus + noise),
        ("bus", "headway_std", 17.0 + 0.1 * bus),
        ("bus", "delay_mean__flow_57_N", 37.0 + bus),
        ("pedestrian", "count", 1570.0),
        ("pedestrian", "ped_curb_wait_mean", 20.0 + ped + 0.05 * noise),
        ("pedestrian", "ped_curb_wait_p95", 76.0 + 2 * ped),
        ("pedestrian", "ped_curb_wait_max", 88.0),
        ("pedestrian", "ped_curb_wait_mean__crossing_NB", 11.0 + ped),
        ("pedestrian", "ped_curb_wait_mean__crossing_S", 30.0 + ped),
        ("ambulance", "count", 1.0),
        ("ambulance", "travel_time_total", 60.0 + 10 * (seed % 3)),
    ]
    return pd.DataFrame(rows, columns=["user_type", "kpi", "value"])


def fake_simulator(calls: list):
    """A `sensitivity._simulate` replacement writing the files the tables read."""

    def simulate(cfg, seed, out_root, config_path, reward):
        from traffic_twin.sim.run_scenario import run_dir

        d = run_dir(cfg, seed, out_root)
        d.mkdir(parents=True, exist_ok=True)
        label = Path(out_root).name
        if label == sens.BASELINES_DIR:
            label = {"fixed": "S0", "actuated": "S1", "rules": "S2"}[cfg["controller"]]
        setting, _, tseed = label.rpartition("_s") if "_s" in label else (label, "", "0")
        setting = setting or label
        tseed = int(tseed) if tseed.isdigit() else 0
        _kpi_rows(label, setting, tseed, seed).to_csv(d / "kpi_run.csv", index=False)
        car, bus, ped = EFFECTS[setting]
        trips = pd.DataFrame(
            {
                "id": ["c.0", "b.0", "p.0", "p.0"],
                "type": ["car", "bus", "pedestrian", "pedestrian"],
                "time_loss_s": [50.0 + car, 45.0 + bus, 10.0 + ped, 5.0],
                "depart_delay_s": [0.0, 0.0, np.nan, np.nan],
            }
        )
        trips.to_parquet(d / "trip.parquet", index=False)
        is_rl = cfg.get("controller") == "rl"
        control = {}
        if is_rl:
            control = {
                "model_sha256": cfg["rl"]["model_sha256"],
                "model_version": None,
                "iv_split": {"cut": 8 + tseed, "replaced": 2, "cut_foreseeable": 3},
            }
        meta = {
            "scenario": cfg["scenario"],
            "seed": seed,
            "config_hash": config_hash(cfg),
            "duration_s": 7200,
            "code_version": "test",
            "safety": {
                "interventions": {"ped_wait": 9 if is_rl else 0, "max_green": 1 if is_rl else 0},
                "danger": 0,
            },
            "health": {"collisions": 0, "teleports": 0},
            "preemption": {"enabled": True, "requests": 1, "served": 1, "timeouts": 0},
            "control": control,
        }
        (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")
        totals = {
            k: v * (1 + 0.01 * car if k.startswith("car") else 1 + 0.01 * bus)
            for k, v in BASE_TOTALS.items()
        }
        totals["cost"] = sens.reweighted_cost(totals, reward)
        (d / sens.REWARD_FILE).write_text(json.dumps({"totals": totals, "reward": reward}))
        calls.append((label, seed))

    return simulate


def sens_workspace(stub_rl7_cfg, monkeypatch, settings=("t1", "wb12", "wb50", "lam100")):
    """A stub rl7 workspace with fake picks of `settings` and fake sensitivity runs
    (`run` with the simulation replaced); returns (rl7 path, sens folder, calls)."""
    rl7 = stub_rl7_cfg()
    cfg = load_config(rl7)
    root = Path(cfg["outputs"]["runs"])
    steps = int(cfg["protocol"]["sensitivity"]["steps"])
    for s in settings:
        for seed in sens.setting_seeds(cfg, s):
            make_pick_run(root, rl7, s, seed, steps, stage="T" if s == "t1" and seed < 3 else "S")
    calls: list = []
    monkeypatch.setattr(sens, "_simulate", fake_simulator(calls))
    out = sens.run(rl7, workers=1, only=list(settings))
    return rl7, out, calls


def test_run_layout_and_skip(stub_rl7_cfg, monkeypatch):
    rl7, out, calls = sens_workspace(stub_rl7_cfg, monkeypatch)
    cfg = load_config(rl7)
    assert out == Path(cfg["outputs"]["study"]) / sens.SENS_DIR
    seeds = list(range(5116, 5118))  # the stub's sensitivity range
    n_picks = 5 + 5 + 5 + 3
    assert len(calls) == (n_picks + 3) * len(seeds)
    plan = json.loads((out / sens.PLAN_FILE).read_text())
    assert plan["seeds"] == seeds and plan["base"] == "t1"
    assert [p["label"] for p in plan["policies"]][:2] == ["t1_s0", "t1_s1"]
    assert set(plan["reward"]) == {"t1", "wb12", "wb50", "lam100"}
    assert plan["runs"] == {
        "total": len(calls),
        "ran": len(calls),
        "skipped": 0,
        "wall_s": plan["runs"]["wall_s"],
    }
    assert (out / "wb50_s3" / "pitie_rl_seed5116" / sens.REWARD_FILE).is_file()
    assert (out / sens.BASELINES_DIR / "pitie_actuated_seed5117" / "run.json").is_file()
    # the in-memory scenario pins the pick and names the setting's run config
    pol = sens.plan_policies(plan)
    _, scen, _ = pol["wb50_s0"]
    assert scen["rl"]["config"].endswith("rl7/wb50.yaml")
    assert scen["rl"]["model"].endswith("pick_model.zip") and len(scen["rl"]["model_sha256"]) == 64
    # a rerun keeps every current run
    calls.clear()
    sens.run(rl7, workers=1, only=["t1", "wb12", "wb50", "lam100"])
    assert calls == []
    sens.run(rl7, workers=1, only=["lam100"], force=True)
    assert len(calls) == (3 + 3) * len(seeds)


def test_run_refuses_data_results(stub_rl7_cfg):
    with pytest.raises(ConfigError, match="data/results"):
        sens.run(stub_rl7_cfg(), workers=1, out=Path("data/results/sens"))


def test_tables(stub_rl7_cfg, monkeypatch, tmp_path):
    rl7, out, _ = sens_workspace(stub_rl7_cfg, monkeypatch)
    paths = sens.tables(rl7)
    assert list(paths) == [*sens.TABLES, "operator"]
    cfg = load_config(rl7)
    assert paths["operator"] == Path(cfg["outputs"]["figures"]) / sens.OPERATOR_CSV
    kpis = pd.read_csv(paths["kpis"])
    k = kpis.set_index(["setting", "metric"])
    assert list(dict.fromkeys(kpis["setting"])) == [
        "t1",
        "wb12",
        "wb50",
        "lam100",
        "S0",
        "S1",
        "S2",
    ]
    assert k.loc[("wb50", "car/time_loss_mean"), "diff_vs_base"] == pytest.approx(9.0)
    assert k.loc[("wb50", "car/time_loss_mean"), "n_policies"] == 5
    assert k.loc[("lam100", "car/time_loss_mean"), "n_policies"] == 3
    assert k.loc[("S1", "car/time_loss_mean"), "vs_S1_pct"] == pytest.approx(0.0)
    assert k.loc[("t1", "pedestrian/ped_curb_wait_max"), "note"].startswith("capped")
    assert ("t1", "safety/cut") in k.index and ("t1", "reward/penalty_s") in k.index
    assert ("t1", "all/person_delay_mean") in k.index
    boot = pd.read_csv(paths["bootstrap"]).set_index(["setting", "metric"])
    assert set(boot.index.get_level_values(0)) == {"wb12", "wb50", "lam100"}
    car = boot.loc[("wb50", "car/time_loss_mean")]
    assert car["diff"] == pytest.approx(9.0) and car["ci_low"] > 0 and bool(car["beyond_noise"])
    amb = boot.loc[("wb50", "ambulance/travel_time_total")]
    assert not bool(amb["beyond_noise"]) and "ambulance" in amb["note"]
    assert ("wb50", "pedestrian/ped_curb_wait_mean__crossing_NB") in boot.index
    eq = pd.read_csv(paths["equity"])
    assert set(eq["role"]) == {"primary", "subgroup", "global"}
    wb50 = set(eq[(eq["setting"] == "wb50") & eq["worse_flag"]]["kpi"])
    # pedestrians +2 s on 9.5 / 28.5 s vs S0 (flagged); car +1 s on 63 s (not flagged)
    assert {"ped_curb_wait_mean__crossing_NB", "ped_curb_wait_mean__crossing_S"} <= wb50
    assert "time_loss_mean" not in wb50
    assert not eq.loc[eq["role"] == "global", "worse_flag"].any()
    rew = pd.read_csv(paths["reweight"])
    assert set(rew["weights"]) == {"t1", "wb12", "wb50", "lam100"}
    own = rew[rew["own"]]
    assert set(own["controller"]) == {"t1", "wb12", "wb50", "lam100"}
    assert (rew.groupby("weights")["rank"].min() == 1).all()
    conv = pd.read_csv(paths["convergence"])
    assert len(conv) == 18 and conv["beats_S2"].dtype == bool
    c = conv.set_index(["setting", "train_seed"])
    assert bool(c.loc[("t1", 0), "beats_S2"]) and not bool(c.loc[("lam100", 0), "beats_S2"])
    assert c.loc[("t1", 0), "pick_vs_S2_pct"] == pytest.approx(-1.0)
    op = pd.read_csv(paths["operator"])
    assert list(op["setting"]) == ["t1", "wb12", "wb50", "lam100", "S0", "S1", "S2"]
    assert op.loc[op["setting"] == "wb50", "w_bus"].item() == 50.0
    assert op.loc[op["setting"] == "wb50", "direction"].item() == "transit_ped"
    assert op.loc[op["setting"] == "S1", "kind"].item() == "baseline"
    assert b"\r\n" not in paths["operator"].read_bytes()
    for name in (sens.RUNS_CSV, sens.SAFETY_CSV, sens.TOTALS_CSV):
        assert (out / name).is_file()
    saf = pd.read_csv(out / sens.SAFETY_CSV)
    assert saf.loc[saf["label"] == "t1_s2", "cut"].tolist() == [10, 10]
    assert saf.loc[saf["label"] == "S1", "cut"].isna().all()
    # a second build gives the same tables (deterministic bootstrap, no timestamps)
    before = {n: p.read_bytes() for n, p in paths.items()}
    again = sens.tables(rl7)
    assert {n: p.read_bytes() for n, p in again.items()} == before


def test_tables_need_a_plan(stub_rl7_cfg):
    with pytest.raises(ConfigError, match="plan.json"):
        sens.tables(stub_rl7_cfg())


def test_tables_detect_a_stale_model(stub_rl7_cfg, monkeypatch):
    rl7, out, _ = sens_workspace(stub_rl7_cfg, monkeypatch, settings=("t1",))
    plan = json.loads((out / sens.PLAN_FILE).read_text())
    Path(plan["policies"][0]["model"]).write_bytes(b"retrained")
    with pytest.raises(ConfigError, match="model sha256"):
        sens.tables(rl7)


def test_cli(stub_rl7_cfg, capsys):
    rl7 = stub_rl7_cfg()
    assert sens.build_parser().parse_args(["run", "--workers", "3"]).workers == 3
    with pytest.raises(SystemExit, match="plan.json"):
        sens.main(["--config", str(rl7), "tables"])
    with pytest.raises(SystemExit, match="picks missing"):
        sens.main(["--config", str(rl7), "run", "--workers", "1", "--settings", "t1"])
    with pytest.raises(SystemExit, match="Parts 3-4"):
        sens.main(["--config", str(rl7), "run", "--seeds", "1..3"])

"""ETA dataset (EF-12; D38, D43, D46): world sampling, splits, world configs, the dataset
id, the build bookkeeping (resume, manifest counts, refusals) and, with SUMO, one short
world run end to end."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_twin.config import ConfigError, config_hash, demand_hash, load_config
from traffic_twin.ml import eta_dataset as eds
from traffic_twin.ml.eta_dataset import (
    SAMPLE_COLUMNS,
    SNAPSHOT_COLUMNS,
    World,
    base_config,
    dataset_id,
    part_path,
    plan_worlds,
    read_parts,
    run_world,
    sample_world,
    split_of,
    world_config,
)
from traffic_twin.ml.features import DIAGNOSTIC, ETA_FEATURES, TRAJETS
from traffic_twin.paths import ETA_CONFIG
from traffic_twin.sim.demand import emergency_trips

# Pure draws only (no simulation): seeds outside every registered range
DRAW_SEEDS = range(100_000, 102_000)
TOL = 0.03  # +-3 points over 2,000 draws


@pytest.fixture(scope="module")
def worlds(eta_scenario) -> list[World]:
    return [sample_world(s, eta_scenario) for s in DRAW_SEEDS]


def share(values, key) -> float:
    values = list(values)
    return sum(v == key for v in values) / len(values)


# -- worlds ----------------------------------------------------------------------------
def test_sample_world_is_deterministic(eta_cfg):
    assert sample_world(2000, eta_cfg) == sample_world(2000, copy.deepcopy(eta_cfg))
    assert sample_world(2000, eta_cfg) != sample_world(2001, eta_cfg)


def test_world_kinds_and_day_types_follow_the_config(worlds, eta_cfg):
    w = eta_cfg["world"]
    kinds = [x.kind for x in worlds]
    assert share(kinds, "realistic") == pytest.approx(w["kinds"]["realistic"], abs=TOL)
    realistic = [x for x in worlds if x.kind == "realistic"]
    stress = [x for x in worlds if x.kind == "stress"]
    for day_type, p in w["realistic"]["day_types"].items():
        assert share((x.day_type for x in realistic), day_type) == pytest.approx(p, abs=TOL)
    assert {x.day_type for x in stress} == {"tue_wed_thu_term"}
    log_m = np.log([x.multiplier for x in realistic])
    assert abs(log_m.mean()) < 0.01
    assert np.std(log_m) == pytest.approx(w["realistic"]["multiplier_lognormal_sd"], abs=0.01)
    m = np.array([x.multiplier for x in stress])
    assert m.min() >= 0.6 and m.max() <= 1.4 and m.mean() == pytest.approx(1.0, abs=0.03)
    scales = np.array([[x.arm_scale[a] for a in "NSW"] for x in realistic])
    assert scales.std() == pytest.approx(w["realistic"]["arm_scale_sd"], abs=0.01)
    scales = np.array([[x.arm_scale[a] for a in "NSW"] for x in stress])
    assert scales.std() == pytest.approx(w["stress"]["arm_scale_sd"], abs=0.01)


def test_ambulances_and_incidents_follow_the_config(worlds, eta_cfg):
    w = eta_cfg["world"]
    for k, p in enumerate(w["ambulances"]["count_p"]):
        assert share((len(x.ambulances) for x in worlds), k) == pytest.approx(p, abs=TOL)
    for k, p in enumerate(w["incidents"]["count_p"]):
        assert share((len(x.incidents) for x in worlds), k) == pytest.approx(p, abs=TOL)
    trips = [t for x in worlds for t in x.ambulances]
    routes = [tuple(t["route"]) for t in trips]
    assert share(routes, ("n_in", "s_out")) == pytest.approx(0.6, abs=TOL)
    assert set(routes) == {("n_in", "s_out"), ("w_in", "s_out")}
    departs = [t["depart_s"] for t in trips]
    assert min(departs) >= 300 and max(departs) < 6600
    assert all(isinstance(d, int) for d in departs)
    for x in worlds:
        assert [t["depart_s"] for t in x.ambulances] == sorted(t["depart_s"] for t in x.ambulances)
        assert [i.start_s for i in x.incidents] == sorted(i.start_s for i in x.incidents)
        assert all(300 <= i.start_s < 6000 for i in x.incidents)
        assert all(i.duration_s in (300, 600, 900, 1200) for i in x.incidents)


def test_split_boundaries(eta_cfg):
    expected = {
        2000: "train", 2299: "train", 2300: "val", 2349: "val", 2350: "calib",
        2449: "calib", 2450: "reserve", 2899: "reserve", 2900: "test", 2999: "test",
    }  # fmt: skip
    assert {s: split_of(s, eta_cfg) for s in expected} == expected
    for seed in (1999, 3000, 5, 101):
        with pytest.raises(ValueError, match="no registered"):
            split_of(seed, eta_cfg)


def test_plan_worlds_refuses_test_reserve_and_foreign_seeds(eta_cfg):
    assert plan_worlds(eta_cfg, ["val"]) == list(range(2300, 2350))
    assert len(plan_worlds(eta_cfg, ["train", "val", "calib"])) == 450
    assert plan_worlds(eta_cfg, ["train"], [2001, 2000, 2001]) == [2000, 2001]
    with pytest.raises(ValueError, match="test"):
        plan_worlds(eta_cfg, ["train", "val", "calib"], [2000, 2900])
    with pytest.raises(ValueError, match="calib"):
        plan_worlds(eta_cfg, ["train"], [2400])
    with pytest.raises(ValueError, match="cannot generate"):
        plan_worlds(eta_cfg, ["reserve"])
    with pytest.raises(ValueError, match="no registered"):
        plan_worlds(eta_cfg, ["train"], [7])
    assert plan_worlds(eta_cfg, ["test"], [2950]) == [2950]


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out |= flatten(v, key + ".")
        else:
            out[key] = v
    return out


OVERRIDES = {
    "scenario", "seed", "demand.day_type", "demand.calibration", "demand.multiplier",
    "demand.arm_scale.N", "demand.arm_scale.S", "demand.arm_scale.W", "outputs.dir",
}  # fmt: skip


@pytest.mark.parametrize("controller", ["fixed", "actuated", "rules"])
def test_world_config_changes_only_the_d43_keys(eta_cfg, controller):
    base = base_config(eta_cfg, controller)
    frozen = copy.deepcopy(base)
    world = sample_world(2002, eta_cfg)  # realistic, 2 ambulances
    assert len(world.ambulances) == 2
    cfg = world_config(base, world, controller, eta_cfg)
    assert base == frozen  # not mutated
    a, b = flatten(base), flatten(cfg)
    changed = {k for k in a.keys() | b.keys() if not k.startswith("emergency.")}
    changed = {k for k in changed if a.get(k, "<absent>") != b.get(k, "<absent>")}
    assert changed <= OVERRIDES
    assert cfg["scenario"] == f"eta_{controller}_w2002" and cfg["seed"] == 2002
    assert cfg["controller"] == controller and cfg["duration_s"] == 7200
    assert cfg["demand"]["calibration"] == eta_cfg["calibration_files"][world.day_type]
    assert cfg["demand"]["multiplier"] == world.multiplier
    assert cfg["outputs"]["dir"] == "data/eta"
    assert cfg["emergency"]["trips"] == [dict(t) for t in world.ambulances]
    assert [d for d, _, _ in emergency_trips(cfg)] == [t["depart_s"] for t in world.ambulances]
    demand_hash(cfg)  # the calibration file exists


def test_world_config_without_ambulances_and_short(eta_cfg):
    world = sample_world(2001, eta_cfg)  # no ambulance
    assert world.ambulances == ()
    cfg = world_config(base_config(eta_cfg, "rules"), world, "rules", eta_cfg)
    assert cfg["emergency"]["enabled"] is False and emergency_trips(cfg) == []
    world = sample_world(2002, eta_cfg)
    short = world_config(base_config(eta_cfg, "rules"), world, "rules", eta_cfg, 1700)
    assert short["duration_s"] == 1700
    assert [t["depart_s"] for t in short["emergency"]["trips"]] == [1685]  # 4121 dropped
    assert config_hash(short) != config_hash(
        world_config(base_config(eta_cfg, "rules"), world, "rules", eta_cfg)
    )


def test_paired_controllers_share_the_world_demand(eta_cfg):
    world = sample_world(2003, eta_cfg)
    hashes = {
        c: demand_hash(world_config(base_config(eta_cfg, c), world, c, eta_cfg))
        for c in eta_cfg["controllers"]
    }
    assert len(set(hashes.values())) == 1


def test_dataset_id_is_stable_and_scoped():
    a, b = load_config(ETA_CONFIG), load_config(ETA_CONFIG)
    assert dataset_id(a) == dataset_id(b)
    assert dataset_id(a).startswith("eta1-") and len(dataset_id(a)) == 13
    b["workers"], b["changes"] = 2, [{"date": "2026-11-01", "what": "x"}]
    b["outputs"] = {**b["outputs"], "data": "elsewhere"}
    assert dataset_id(b) == dataset_id(a)
    b["world"]["incidents"]["count_p"] = [0.4, 0.4, 0.2]
    assert dataset_id(b) != dataset_id(a)


# -- build bookkeeping (no SUMO: run_world faked) -----------------------------------------
def fake_run_world(seed, controller, eta_cfg, dataset_dir, work_dir, code=None, **kw):
    """Writes parts and the run row like run_world, with 2 trips per trajet."""
    split = split_of(seed, eta_cfg)
    world = sample_world(seed, eta_cfg)
    cfg = world_config(base_config(eta_cfg, controller), world, controller, eta_cfg)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for k, j in enumerate(TRAJETS * 2):
        rows.append(dict.fromkeys(SAMPLE_COLUMNS, 0.0) | {
            "run_id": "r", "world_seed": seed, "controller": controller, "split": split,
            "kind": world.kind, "veh_id": f"v{k}", "trajet": j, "teleported": k == 0,
            "exit_t": math.nan if k == 1 else 5.0,
        })  # fmt: skip
    samples = pd.DataFrame(rows)[list(SAMPLE_COLUMNS)]
    eds._write_parquet(samples, part_path(dataset_dir, "samples", split, controller, seed))
    snaps = pd.DataFrame({"run_id": ["r"], "t": [5.0]})
    eds._write_parquet(snaps, part_path(dataset_dir, "snapshots", split, controller, seed))
    row = {
        "run_id": "r", "world_seed": seed, "controller": controller, "split": split,
        "kind": world.kind, "config_hash": config_hash(cfg), "code_version": code,
        "sampler_version": eds.SAMPLER_VERSION, "n_samples": len(samples),
        "feature_schema_version": eds.FEATURE_SCHEMA_VERSION, "n_missing_exit": 1,
        "teleports": 1, "wall_s": 0.0,
    }  # fmt: skip
    eds._write_json(row, part_path(dataset_dir, "runs", split, controller, seed))
    return row


@pytest.fixture
def fake_build(monkeypatch, tmp_path, eta_cfg):
    calls = []

    def fake(seed, controller, *args, **kw):
        calls.append((seed, controller))
        return fake_run_world(seed, controller, *args, **kw)

    monkeypatch.setattr(eds, "run_world", fake)
    monkeypatch.setattr(eds, "code_version", lambda: "abc1234")

    def _build(**kw):
        args = {"splits": ["train"], "out": tmp_path / "eta", "worlds": [2000, 2001]} | kw
        return eds.build(eta_cfg, workers=1, argv=["test"], **args)

    return _build, calls


def test_build_writes_manifest_counts_and_resumes(fake_build, eta_cfg, tmp_path):
    build, calls = fake_build
    dataset_dir, done, errors = build()
    assert errors == [] and len(done) == 6 and len(calls) == 6
    assert dataset_dir == tmp_path / "eta" / dataset_id(eta_cfg)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == dataset_dir.name and manifest["code_version"] == "abc1234"
    assert manifest["counts"] == {
        "train": {c: dict.fromkeys(TRAJETS, 4) for c in ("fixed", "actuated", "rules")}
    }
    assert manifest["n_samples"] == 72 and manifest["n_runs"] == 6
    assert manifest["excluded"]["train"]["rules"] == {"teleported": 2, "missing_exit": 2}
    assert manifest["splits"] == {"train": [2000, 2001]}
    assert manifest["features"]["names"] == list(ETA_FEATURES)
    assert manifest["features"]["diagnostic"] == list(DIAGNOSTIC)
    assert set(manifest["base_configs"]) == {"fixed", "actuated", "rules"}
    assert all(v["sha256"] for v in manifest["calibration_files"].values())
    # D65: library versions (SUMO first: the samples depend on it) in the manifest and build
    versions = manifest["versions"]
    assert versions == manifest["builds"][0]["versions"] == eds.library_versions()
    assert set(versions) >= {"python", "sumo", "libsumo", "traci", "numpy", "pandas", "pyarrow"}
    assert versions["python"] and versions["numpy"] == np.__version__
    assert versions["pandas"] == pd.__version__
    runs = pd.read_parquet(dataset_dir / "runs.parquet")
    assert len(runs) == 6 and set(runs["world_seed"]) == {2000, 2001}
    assert not (tmp_path / "eta" / "_work").exists()
    # resume: complete runs are skipped, a missing part is redone
    part_path(dataset_dir, "samples", "train", "actuated", 2001).unlink()
    calls.clear()
    _, done, _ = build(worlds=[2000, 2001, 2002])
    assert sorted(calls) == [
        (2001, "actuated"),
        *((2002, c) for c in ("actuated", "fixed", "rules")),
    ]
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["builds"]) == 2 and manifest["n_runs"] == 9
    assert manifest["versions"] == eds.library_versions()
    assert all(b["versions"] == eds.library_versions() for b in manifest["builds"])
    counted = read_parts(dataset_dir, "samples").groupby("trajet").size().to_dict()
    assert counted == dict.fromkeys(TRAJETS, 2 * 9)


def test_build_refuses_dirty_code_unless_allowed(fake_build, monkeypatch, eta_cfg, tmp_path):
    build, calls = fake_build
    monkeypatch.setattr(eds, "code_version", lambda: "abc1234-dirty")
    with pytest.raises(ConfigError, match="allow-dirty"):
        build()
    assert calls == []
    dataset_dir, done, _ = build(allow_dirty=True, controllers=["rules"])
    assert dataset_dir.name == dataset_id(eta_cfg) + "-dev" and len(done) == 2
    with pytest.raises(ConfigError, match="controllers"):
        build(controllers=["rl"], allow_dirty=True)


def test_build_reports_run_errors(fake_build, monkeypatch):
    build, _ = fake_build

    def boom(*a, **k):
        raise RuntimeError("SUMO died")

    monkeypatch.setattr(eds, "run_world", boom)
    _, done, errors = build(worlds=[2000], controllers=["fixed"])
    assert done == [] and len(errors) == 1 and "SUMO died" in errors[0]["error"]


def test_cli_refuses_test_seeds_without_the_test_split(capsys):
    assert eds.main(["--splits", "train", "--worlds", "2900", "--allow-dirty"]) == 2
    assert "test" in capsys.readouterr().err


# -- one world with SUMO ------------------------------------------------------------------
@pytest.mark.sumo
def test_run_world_mini(eta_cfg, tmp_path):
    """World 2000 under S2 for 900 s: samples and snapshots, sane targets, a leak-free
    join, byte-identical reruns, and nothing written outside the given folders."""
    demand_dir = Path("sumo/demand")
    before = sorted((p.name, p.stat().st_mtime_ns) for p in demand_dir.glob("*"))
    ds = tmp_path / "ds"
    row = run_world(2000, "rules", eta_cfg, ds, tmp_path / "work", duration_s=900, code="x")
    samples = pd.read_parquet(part_path(ds, "samples", "train", "rules", 2000))
    snaps = pd.read_parquet(part_path(ds, "snapshots", "train", "rules", 2000))
    assert tuple(samples.columns) == SAMPLE_COLUMNS
    assert row["n_samples"] == len(samples) > 100 and row["n_missing_exit"] == 0
    assert row["teleports"] == 0 and row["n_missing_features"] == 0
    assert set(samples["trajet"]) <= set(TRAJETS) and samples["run_id"].eq(row["run_id"]).all()
    assert (samples["y_zone_s"] >= 1).all() and (samples["exit_t"] > samples["insert_t"]).all()
    assert (samples["stopline_t"] <= samples["exit_t"]).all()
    assert (samples["insert_t"] >= 0).all() and (samples["insert_t"] < 900).all()
    assert (samples["t_feat"] <= np.floor(samples["insert_t"]) - 1).all()
    assert (samples["y_junction_s"] >= samples["y_zone_s"]).all()
    assert tuple(snaps.columns) == ("run_id", "world_seed", "split", "kind", *SNAPSHOT_COLUMNS[1:])
    assert (snaps["t"] % 5 == 0).all() and snaps["t"].iloc[0] == 5.0
    assert snaps["t"].max() < 900 + 600 + 1  # until the window end + 600 s
    marker = json.loads(part_path(ds, "runs", "train", "rules", 2000).read_text("utf-8"))
    assert marker["config_hash"] == row["config_hash"]
    raw = [p.name for p in (tmp_path / "work" / "out").iterdir()]
    assert raw == [], f"raw SUMO outputs left: {raw}"
    # rerun into another dataset folder: byte-identical parts
    ds2 = tmp_path / "ds2"
    run_world(2000, "rules", eta_cfg, ds2, tmp_path / "work2", duration_s=900, code="x")
    for what in ("samples", "snapshots"):
        a = part_path(ds, what, "train", "rules", 2000).read_bytes()
        assert a == part_path(ds2, what, "train", "rules", 2000).read_bytes()
    after = sorted((p.name, p.stat().st_mtime_ns) for p in demand_dir.glob("*"))
    assert after == before  # no route file written into sumo/demand
    written = {p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*") if p.is_file()}
    assert written <= {"ds", "ds2", "work", "work2"}


def test_shared_versions_only_when_every_run_making_build_agrees():
    v1, v2 = {"python": "3.12", "sumo": "1.28.0"}, {"python": "3.12", "sumo": "1.29.0"}
    made = {"runs_made": 3}
    assert eds.shared_versions([{**made, "versions": v1}]) == v1
    # a no-op build (resume with nothing left) does not vouch for the runs
    assert eds.shared_versions([{**made, "versions": v1}, {"runs_made": 0, "versions": v2}]) == v1
    assert eds.shared_versions([{**made, "versions": v1}, {**made, "versions": v2}]) is None
    assert eds.shared_versions([{**made}, {**made, "versions": v1}]) is None  # legacy build
    assert eds.shared_versions([{"runs_made": 0, "versions": v2}]) == v2
    assert eds.shared_versions([]) is None


def test_manifest_versions_null_after_a_legacy_build(fake_build, eta_cfg, tmp_path):
    build, _ = fake_build
    dataset_dir, _, _ = build(worlds=[2000])
    path = dataset_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for b in manifest["builds"]:
        b.pop("versions")  # a manifest written before D65 versions were recorded
    path.write_text(json.dumps(manifest), encoding="utf-8")
    build(worlds=[2000, 2001])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["versions"] is None
    assert manifest["builds"][-1]["versions"] == eds.library_versions()

"""ETA model training (Part 5, D47-D49, D51): split loading, reference table and fallback,
bundle round trip with a fake predictor (version string, sha256 checks, tampering),
freeze rules, and (with xgboost only) a real fit with early stopping.

The synthetic dataset helpers here are shared with the other ETA test modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_twin.config import ConfigError
from traffic_twin.ml.features import CATEGORIES, ETA_FEATURES, TRAJETS
from traffic_twin.ml.predict import BundleError, latest_bundle, load_bundle, predict_frame
from traffic_twin.ml.train_eta import (
    FALLBACK_DAY_TYPE,
    FALLBACK_HOUR,
    LATEST_FILE,
    META_FILE,
    MODEL_FILE,
    REFERENCE_FILE,
    apply_reference,
    bundle_sha256,
    check_freeze,
    fit_reference,
    freeze,
    hours_of,
    load_split,
    model_features,
    model_version,
    reference_lookup,
    reference_population,
    reference_rows,
    save_bundle,
    stop_rules,
)

TRAJET_BASE = {"N_S": 40.0, "N_W": 20.0, "S_N": 25.0, "S_W": 55.0, "W_N": 45.0, "W_S": 60.0}
CONTROLLERS = ("fixed", "actuated", "rules")


# -- synthetic data (shared) -----------------------------------------------------------
def synthetic_rows(
    n: int,
    rng: np.random.Generator,
    world: int = 2000,
    controller: str = "fixed",
    kind: str = "realistic",
    day_type: str = "tue_wed_thu_term",
) -> pd.DataFrame:
    """`n` sample rows with every ETA feature, the identity columns and an integer
    target that depends on the trajet and on `halt_N` (learnable)."""
    df = pd.DataFrame(
        {
            "run_id": f"run-{world}-{controller}",
            "world_seed": np.int64(world),
            "controller": controller,
            "split": "train",
            "kind": kind,
            "veh_id": [f"car_N_S_{world}.{i}" for i in range(n)],
            "trajet": rng.choice(TRAJETS, n),
            "insert_t": np.sort(rng.uniform(0, 7199, n)).round(0),
        }
    )
    for f in ETA_FEATURES:
        if f in ("trajet", "controller"):
            continue
        if f == "day_type":
            df[f] = day_type
        elif f in CATEGORIES:
            df[f] = rng.choice(CATEGORIES[f], n)
        else:
            df[f] = rng.uniform(0, 10, n)
    for a in ("N", "S", "W"):
        df[f"pending_{a}"] = 0.0
        df[f"inc_{a}"] = (rng.uniform(size=n) < 0.05).astype(float)
    df["amb_present"] = (rng.uniform(size=n) < 0.05).astype(float)
    df["t_feat"] = np.floor(df["insert_t"]) - 1
    base = df["trajet"].map(TRAJET_BASE).to_numpy()
    noise = rng.gamma(2.0, 4.0, n)
    df["y_zone_s"] = np.round(base + 3.0 * df["halt_N"].to_numpy() + noise)
    df["exit_t"] = df["insert_t"] + df["y_zone_s"]
    df["teleported"] = False
    return df


def write_dataset(
    root: Path,
    splits: dict[str, list[int]],
    n: int = 120,
    seed: int = 0,
    stress_every: int = 3,
) -> Path:
    """A dataset folder with sample parts `samples/split=<s>/controller=<c>/part-<w>.parquet`
    for every world (realistic unless world % stress_every == 1) and controller."""
    rng = np.random.default_rng(seed)
    day_types = list(CATEGORIES["day_type"])
    for split, worlds in splits.items():
        for w in worlds:
            kind = "stress" if stress_every and w % stress_every == 1 else "realistic"
            day_type = day_types[w % len(day_types)] if kind == "realistic" else day_types[0]
            for c in CONTROLLERS:
                df = synthetic_rows(n, rng, w, c, kind, day_type).assign(split=split)
                p = root / "samples" / f"split={split}" / f"controller={c}" / f"part-{w}.parquet"
                p.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(p, index=False)
    (root / "manifest.json").write_text(
        json.dumps({"dataset_id": "eta1-synthetic", "code_version": "test"}), encoding="utf-8"
    )
    return root


class LinearPredictor:
    """Fake QuantilePredictor: trajet base + 3 x halt_N + fixed offsets per quantile."""

    OFFSETS = (2.0, 8.0, 18.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base = X["trajet"].astype(str).map(TRAJET_BASE).to_numpy(float)
        core = base + 3.0 * X["halt_N"].to_numpy(float)
        # deliberately unsorted columns: callers sort each row
        return np.column_stack([core + self.OFFSETS[2], core + self.OFFSETS[0], core + 8.0])

    def to_bytes(self) -> bytes:
        return json.dumps({"offsets": self.OFFSETS}).encode("utf-8")


def linear_loader(raw: bytes, meta: dict) -> LinearPredictor:
    assert json.loads(raw.decode("utf-8")) == {"offsets": list(LinearPredictor.OFFSETS)}
    return LinearPredictor()


def fake_info(eta_cfg: dict) -> dict:
    return {
        "features": model_features(eta_cfg),
        "quantiles": [0.1, 0.5, 0.9],
        "params": {**eta_cfg["model"]["params"], "max_depth": 6},
        "grid": [{"max_depth": 6, "val_pinball": 1.0}],
        "best_iteration": 10,
        "xgboost": "fake",
    }


def make_fake_bundle(models_dir: Path, eta_cfg: dict, train: pd.DataFrame, qhat=-0.5) -> Path:
    extra = {
        "begin_s": 27000,
        "seen": {"controller": list(CONTROLLERS), "day_type": list(CATEGORIES["day_type"])},
        "n": {"train": len(train)},
    }
    version = save_bundle(
        LinearPredictor(), fit_reference(train), qhat, fake_info(eta_cfg), eta_cfg,
        "eta1-synthetic", models_dir, extra,
    )  # fmt: skip
    return models_dir / version


@pytest.fixture
def train_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    parts = [
        synthetic_rows(150, rng, w, c, "realistic", dt)
        for w, dt in ((2000, "tue_wed_thu_term"), (2001, "mon_fri_term"))
        for c in CONTROLLERS
    ]
    return pd.concat(parts, ignore_index=True)


# -- loading ---------------------------------------------------------------------------
def test_load_split_drops_and_counts(tmp_path):
    ds = write_dataset(tmp_path / "ds", {"train": [2000, 2002], "val": [2300]}, n=50)
    p = ds / "samples/split=train/controller=fixed/part-2000.parquet"
    df = pd.read_parquet(p)
    df.loc[0, "teleported"] = True
    df.loc[1, ["y_zone_s", "exit_t"]] = np.nan
    df.loc[2, "t_feat"] = np.nan
    df.to_parquet(p, index=False)
    tr = load_split(ds, "train")
    assert len(tr) == 2 * 3 * 50 - 3
    assert tr.attrs["dropped"] == {"teleported": 1, "missing_exit": 1, "unjoined": 1}
    assert tr["halt_N"].dtype == np.float32 and tr["y_zone_s"].dtype == np.float64
    stress = load_split(ds, "train", kind="stress")  # world 2002 (2002 % 3 == 1)
    assert set(stress["kind"]) == {"stress"} and set(stress["world_seed"]) == {2002}
    assert set(load_split(ds, "*", worlds=[2300])["world_seed"]) == {2300}
    with pytest.raises(FileNotFoundError):
        load_split(ds, "test")


def test_load_split_columns_keep_the_drops(tmp_path):
    ds = write_dataset(tmp_path / "ds", {"train": [2000, 2002]}, n=20)
    p = ds / "samples/split=train/controller=fixed/part-2000.parquet"
    df = pd.read_parquet(p)
    df.loc[0, "teleported"] = True
    df.loc[1, "t_feat"] = np.nan
    df.to_parquet(p, index=False)
    full = load_split(ds, "train")
    few = load_split(ds, "train", columns=["trajet", "insert_t", "day_type", "y_zone_s"])
    assert len(few) == len(full) == 2 * 3 * 20 - 2
    assert few.attrs["dropped"] == full.attrs["dropped"]
    assert "halt_N" not in few and {"kind", "trajet", "y_zone_s"} <= set(few.columns)
    assert few["y_zone_s"].tolist() == full["y_zone_s"].tolist()


def test_hours_of_clock():
    assert list(hours_of(pd.Series([0.0, 1799.0, 1800.0, 7199.0]))) == [7, 7, 8, 9]


# -- reference -------------------------------------------------------------------------
def test_reference_population_rule():
    """`reference.population` (review of Parts 5-6): `all` is the frozen bundle's rule
    (stress worlds included); `realistic` fits the historical mean of normal days."""
    assert reference_population({}) == "all"
    assert reference_population({"reference": {"keys": ["route"]}}) == "all"
    assert reference_population({"reference": {"population": "realistic"}}) == "realistic"
    with pytest.raises(ConfigError, match="reference.population"):
        reference_population({"reference": {"population": "stress"}})
    train = pd.DataFrame(
        {
            "trajet": ["N_S"] * 4,
            "insert_t": [0.0] * 4,
            "day_type": ["tue_wed_thu_term"] * 4,
            "kind": ["realistic", "realistic", "stress", "stress"],
            "y_zone_s": [10.0, 20.0, 100.0, 200.0],
        }
    )
    assert len(reference_rows(train, "all")) == 4
    ref = fit_reference(reference_rows(train, "realistic"))
    row = reference_lookup(ref, "N_S", 7, "tue_wed_thu_term")
    assert row["mean_s"] == 15.0 and row["n"] == 2  # the stress cars do not inflate it
    assert reference_lookup(fit_reference(train), "N_S", 7, "tue_wed_thu_term")["mean_s"] == 82.5


def test_reference_keys_and_fallback():
    train = pd.DataFrame(
        {
            "trajet": ["N_S"] * 4 + ["W_N"] * 2,
            "insert_t": [0.0, 10.0, 1900.0, 1950.0, 0.0, 5.0],
            "day_type": ["tue_wed_thu_term"] * 4 + ["mon_fri_term"] * 2,
            "y_zone_s": [10.0, 20.0, 30.0, 50.0, 7.0, 9.0],
        }
    )
    ref = fit_reference(train)
    assert list(ref.columns) == ["route", "hour", "day_type", "mean_s", "p10_s", "p90_s", "n"]
    main = ref[ref["hour"] != FALLBACK_HOUR].set_index(["route", "hour", "day_type"])
    assert main.loc[("N_S", 7, "tue_wed_thu_term"), "mean_s"] == 15.0
    assert main.loc[("N_S", 8, "tue_wed_thu_term"), "n"] == 2
    assert main.loc[("N_S", 7, "tue_wed_thu_term"), "p10_s"] == pytest.approx(11.0)
    fb = ref[ref["hour"] == FALLBACK_HOUR].set_index("route")
    assert set(fb["day_type"]) == {FALLBACK_DAY_TYPE}
    assert fb.loc["N_S", "mean_s"] == 27.5 and fb.loc["W_N", "n"] == 2
    query = pd.DataFrame(
        {
            "trajet": ["N_S", "N_S", "W_N", "S_N"],
            "insert_t": [100.0, 6000.0, 0.0, 0.0],
            "day_type": ["tue_wed_thu_term", "tue_wed_thu_term", "school_holiday", "x"],
        },
        index=[10, 11, 12, 13],
    )
    got = apply_reference(ref, query)
    assert list(got.index) == [10, 11, 12, 13]
    assert got.loc[10, "mean_s"] == 15.0  # exact key
    assert got.loc[11, "mean_s"] == 27.5  # 09:10: no key at hour 9 -> fallback
    assert got.loc[12, "mean_s"] == 8.0  # unseen day type -> fallback
    assert np.isnan(got.loc[13, "mean_s"])  # route never seen
    assert reference_lookup(ref, "N_S", 7, "tue_wed_thu_term")["mean_s"] == 15.0
    assert reference_lookup(ref, "N_S", 9, "school_holiday")["hour"] == FALLBACK_HOUR
    assert reference_lookup(ref, "S_W", 7, "tue_wed_thu_term") is None


# -- bundle ----------------------------------------------------------------------------
def test_bundle_round_trip(tmp_path, eta_cfg, train_frame):
    models = tmp_path / "models"
    bundle_dir = make_fake_bundle(models, eta_cfg, train_frame)
    version = bundle_dir.name
    assert version.startswith("eta1-") and len(version) == len("eta1-") + 8
    assert latest_bundle(models) == bundle_dir
    assert (models / LATEST_FILE).read_text().strip() == version
    meta = json.loads((bundle_dir / META_FILE).read_text())
    assert meta["features"] == model_features(eta_cfg) and meta["qhat"] == -0.5
    assert meta["dtypes"]["trajet"] == list(TRAJETS) and meta["dtypes"]["halt_N"] == "float"
    again = model_version("eta1-synthetic", meta["params"], -0.5, meta["model_sha256"])
    assert again == version
    assert model_version("eta1-synthetic", meta["params"], -0.4, meta["model_sha256"]) != version

    b = load_bundle(bundle_dir, loader=linear_loader)
    assert b.version == version and b.qhat == -0.5
    assert set(b.reference["model_version"]) == {version}
    out = predict_frame(b, train_frame.iloc[:40])
    assert list(out.columns) == ["eta_median_s", "eta_low_s", "eta_high_s"]
    assert (out.dtypes == np.float64).all()
    assert (out["eta_low_s"] <= out["eta_median_s"]).all()
    assert (out["eta_median_s"] <= out["eta_high_s"]).all()
    assert (out == np.round(out)).all().all()


def test_bundle_without_latest(tmp_path):
    assert latest_bundle(tmp_path) is None
    (tmp_path / LATEST_FILE).write_text("eta1-missing\n")
    assert latest_bundle(tmp_path) is None


@pytest.mark.parametrize("victim", [MODEL_FILE, REFERENCE_FILE])
def test_tampered_bundle_raises(tmp_path, eta_cfg, train_frame, victim):
    bundle_dir = make_fake_bundle(tmp_path, eta_cfg, train_frame)
    p = bundle_dir / victim
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(BundleError, match="sha256"):
        load_bundle(bundle_dir, loader=linear_loader)


def test_inconsistent_bundles_raise(tmp_path, eta_cfg, train_frame):
    bundle_dir = make_fake_bundle(tmp_path, eta_cfg, train_frame)
    meta_path = bundle_dir / META_FILE
    good = json.loads(meta_path.read_text())

    def broken(**changes) -> None:
        meta_path.write_text(json.dumps({**good, **changes}))
        with pytest.raises(BundleError):
            load_bundle(bundle_dir, loader=linear_loader)

    broken(qhat=good["qhat"] + 1.0)  # version string no longer matches the content
    broken(features=list(reversed(good["features"])))
    broken(feature_schema_version="eta-features-0")
    broken(categories={**good["categories"], "stage": ["HOP"]})
    broken(bundle_format="other")
    meta_path.write_text(json.dumps(good))
    renamed = bundle_dir.with_name("eta1-00000000")
    bundle_dir.rename(renamed)
    with pytest.raises(BundleError, match="folder"):
        load_bundle(renamed, loader=linear_loader)
    with pytest.raises(BundleError, match="meta.json"):
        load_bundle(tmp_path / "nowhere", loader=linear_loader)


def test_freeze_rules(tmp_path, eta_cfg, train_frame):
    models = tmp_path / "models"
    bundle_dir = make_fake_bundle(models, eta_cfg, train_frame)
    with pytest.raises(ValueError, match="no freeze.json"):
        check_freeze(bundle_dir, models)
    ds = write_dataset(tmp_path / "ds", {"train": [2000]}, n=10)
    rec = freeze(bundle_dir, models, ds)
    assert rec["model_version"] == bundle_dir.name
    assert rec["bundle_sha256"] == bundle_sha256(bundle_dir)
    assert check_freeze(bundle_dir, models)["model_version"] == bundle_dir.name
    # eval.json is written into the bundle after the test run: not part of its sha256
    (bundle_dir / "eval.json").write_text("{}")
    check_freeze(bundle_dir, models)
    # a second, different bundle: refused without --refreeze and a dated change
    other = make_fake_bundle(models, eta_cfg, train_frame, qhat=1.0)
    with pytest.raises(ValueError, match="already freezes"):
        freeze(other, models, ds)
    with pytest.raises(ValueError, match="changes"):
        freeze(other, models, ds, refreeze=True, eta_cfg=eta_cfg)
    with pytest.raises(ValueError, match="frozen model"):
        check_freeze(other, models)
    # tampering after the freeze is caught
    ref = bundle_dir / REFERENCE_FILE
    ref.write_bytes(ref.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed since the freeze"):
        check_freeze(bundle_dir, models)


def test_freeze_refused_once_test_worlds_exist(tmp_path, eta_cfg, train_frame):
    models = tmp_path / "models"
    bundle_dir = make_fake_bundle(models, eta_cfg, train_frame)
    ds = write_dataset(tmp_path / "ds", {"train": [2000], "test": [2900]}, n=10)
    with pytest.raises(ValueError, match="test worlds"):
        freeze(bundle_dir, models, ds)
    assert not (models / "freeze.json").exists()


def test_freeze_refuses_world_override(tmp_path, eta_cfg, train_frame):
    models = tmp_path / "models"
    version = save_bundle(
        LinearPredictor(), fit_reference(train_frame), 0.0, fake_info(eta_cfg), eta_cfg,
        "eta1-synthetic", models, {"worlds_override": {"train": [2000]}},
    )  # fmt: skip
    with pytest.raises(ValueError, match="overridden"):
        freeze(models / version, models)


def test_stop_rules():
    assert stop_rules({"qhat": -0.5, "val_mae": 10.0, "val_ref_mae": 40.0}) == []
    assert len(stop_rules({"qhat": 25.0, "val_mae": 50.0, "val_ref_mae": 40.0})) == 2


# -- real fit (local only: xgboost is not installed in CI) ------------------------------
def small_cfg(eta_cfg: dict) -> dict:
    eta_cfg["model"]["params"].update(n_estimators=300, early_stopping_rounds=15, n_jobs=2)
    eta_cfg["model"]["grid"] = {"max_depth": [2, 3]}
    return eta_cfg


def test_xgboost_fit_early_stopping_and_sorted_output(eta_cfg):
    pytest.importorskip("xgboost")
    from traffic_twin.ml.features import model_frame
    from traffic_twin.ml.train_eta import fit_quantile_model, pinball_loss

    rng = np.random.default_rng(3)
    train = pd.concat(
        [synthetic_rows(350, rng, 2000 + i, c) for i in range(2) for c in CONTROLLERS],
        ignore_index=True,
    )  # 2,100 rows
    val = pd.concat([synthetic_rows(200, rng, 2300, c) for c in CONTROLLERS], ignore_index=True)
    cfg = small_cfg(eta_cfg)
    model, info = fit_quantile_model(train, val, cfg)
    assert [g["max_depth"] for g in info["grid"]] == [2, 3]
    assert info["chosen"]["max_depth"] in (2, 3)
    best = min(info["grid"], key=lambda g: g["val_pinball"])
    assert info["chosen"]["max_depth"] == best["max_depth"]
    assert all(g["best_iteration"] < g["rounds"] for g in info["grid"])  # early stopping
    pred = model.predict(model_frame(val)[info["features"]])
    assert pred.shape == (len(val), 3)
    s = np.sort(pred, axis=1)
    assert pinball_loss(val["y_zone_s"].to_numpy(), s) == pytest.approx(info["val_pinball"])
    # the trajet effect is learnt: median MAE well below the spread of the target
    assert np.mean(np.abs(s[:, 1] - val["y_zone_s"])) < 0.5 * val["y_zone_s"].std()
    raw = model.to_bytes()
    assert raw == model.to_bytes()  # deterministic serialisation


def test_train_end_to_end_with_xgboost(tmp_path, eta_cfg):
    pytest.importorskip("xgboost")
    from traffic_twin.ml.train_eta import train

    ds = write_dataset(
        tmp_path / "ds", {"train": [2000, 2003, 2006], "val": [2300], "calib": [2351, 2354]},
        n=200,
    )  # fmt: skip
    cfg = small_cfg(eta_cfg)
    models = tmp_path / "models"
    summary = train(cfg, ds, models)
    assert summary["val_mae"] < summary["val_ref_mae"]
    assert summary["calib_coverage"] >= 0.80
    b = load_bundle(models / summary["model_version"])
    assert b.meta["calibration"]["calib_population"] == "realistic"
    assert b.meta["seen"]["controller"] == sorted(CONTROLLERS)
    assert b.meta["n"]["train"] == 3 * 3 * 200
    again = train(cfg, ds, tmp_path / "models2")
    assert again["model_version"] == summary["model_version"]  # deterministic (D65)

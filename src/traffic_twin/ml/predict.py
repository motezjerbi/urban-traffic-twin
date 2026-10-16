"""ETA bundle loading and precomputed predictions (Part 5, D51; contract 4.5).

    python -m traffic_twin.ml.predict demo [--seeds 4000] [--out data/demo] [--bundle DIR]
    python -m traffic_twin.ml.predict runs <run_dir>... [--bundle DIR]

Serving is precomputed (D51): a run recorded with the passive `EtaRecorder` has
`<run_dir>/eta_snapshots.parquet` (feature rows every 5 s); `predict_run` turns it into
`<run_dir>/eta_prediction.parquet` with the exact columns of Supabase
`public.eta_prediction`: run_id (the uuid of run.json), route (trajet), t (float64, s since
scenario start, 5, 10, ..., 7195), eta_median_s, eta_low_s, eta_high_s (float64, whole
seconds, low <= median <= high: the served bounds of D48) and model_version, sorted by
(route, t). A row at t predicts for a car entering the zone at t from the features of
t - 1 (the training join, D41).

An unseen controller (e.g. `rl`) or day type gets no prediction file: the API then serves
the bundle's reference table as an approximate estimate. The API never imports this
module's xgboost path (`load_bundle` imports xgboost only to build the model).

Demo runs (`demo`): `pitie_{fixed,actuated,rules}` on the demo seeds of
`configs/api.yaml` (4000-4009; Part 5 uses 4000) into `data/demo/`, route files in a
temporary folder, never into `data/results/` or `sumo/demand/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_twin.ml.train_eta import (
    BUNDLE_FORMAT,
    ETA_MODELS_DIR,
    LATEST_FILE,
    META_FILE,
    MODEL_FILE,
    MODEL_FORMAT,
    REFERENCE_FILE,
    QuantilePredictor,
    XGBQuantilePredictor,
    model_version,
    served_bounds,
    sha256_bytes,
)
from traffic_twin.paths import (
    API_CONFIG,
    DEMO_DIR,
    PITIE_ACTUATED_CONFIG,
    PITIE_FIXED_CONFIG,
    PITIE_RULES_CONFIG,
)

log = logging.getLogger(__name__)

PREDICTION_FILE = "eta_prediction.parquet"
SNAPSHOT_FILE = "eta_snapshots.parquet"
PREDICTION_COLUMNS = (
    "run_id", "route", "t", "eta_median_s", "eta_low_s", "eta_high_s", "model_version",
)  # fmt: skip
DEMO_SEEDS = (4000, 4009)  # configs/api.yaml seeds.demo (D38), used when it is absent
PROTECTED_DIR = Path("data") / "results"  # Part 3-4 results: never written by Part 5

ModelLoader = Callable[[bytes, dict], QuantilePredictor]


class BundleError(Exception):
    """A bundle that is missing, tampered with or inconsistent."""


@dataclass(frozen=True)
class EtaBundle:
    version: str
    meta: dict
    model: QuantilePredictor
    reference: pd.DataFrame
    qhat: float

    @property
    def features(self) -> list[str]:
        return list(self.meta["features"])


def _xgboost_loader(raw: bytes, meta: dict) -> QuantilePredictor:
    if meta.get("model_format") != MODEL_FORMAT:
        raise BundleError(f"unknown model format {meta.get('model_format')!r}")
    try:
        import xgboost as xgb
    except ImportError as e:  # pragma: no cover - CI has no xgboost
        raise BundleError("xgboost is not installed (pip install -e .[ml])") from e
    want = (meta.get("versions") or {}).get("xgboost")
    if want and want != xgb.__version__:
        log.warning("bundle made with xgboost %s, running %s", want, xgb.__version__)
    n_jobs = (meta.get("params") or {}).get("n_jobs")
    try:
        return XGBQuantilePredictor.from_bytes(raw, n_jobs)
    except Exception as e:  # xgboost raises its own error types on a corrupt model
        raise BundleError(f"cannot load the model: {e}") from e


def _expected_features(meta: dict) -> list[str]:
    from traffic_twin.ml.features import ETA_FEATURES

    excluded = set(meta.get("excluded_features") or ())
    return [f for f in ETA_FEATURES if f not in excluded]


def load_bundle(path: Path, loader: ModelLoader | None = None) -> EtaBundle:
    """sha256 and feature-list checks -> BundleError. Checks: meta.json present and of
    this bundle format; model and reference sha256 as recorded; the version string
    recomputed from (dataset_id, params, qhat, model sha256) and equal to the folder
    name; the feature schema, ordered feature list and category lists equal to the
    current `ml.features`. A different xgboost version is only a warning. `loader`
    builds the predictor from the model bytes (default: xgboost)."""
    from traffic_twin.ml.features import CATEGORIES, FEATURE_SCHEMA_VERSION

    path = Path(path)
    meta_path = path / META_FILE
    if not meta_path.is_file():
        raise BundleError(f"no {META_FILE} in {path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise BundleError(f"unreadable {meta_path}: {e}") from e
    if meta.get("bundle_format") != BUNDLE_FORMAT:
        raise BundleError(f"bundle format {meta.get('bundle_format')!r} != {BUNDLE_FORMAT}")
    files = {}
    for name, key in ((MODEL_FILE, "model_sha256"), (REFERENCE_FILE, "reference_sha256")):
        f = path / name
        if not f.is_file():
            raise BundleError(f"missing {name} in {path}")
        files[name] = f.read_bytes()
        if sha256_bytes(files[name]) != meta.get(key):
            raise BundleError(f"{name}: sha256 mismatch (tampered or partial bundle)")
    version = meta.get("model_version")
    expected = model_version(
        meta.get("dataset_id", ""), meta.get("params") or {}, meta.get("qhat", 0.0),
        meta.get("model_sha256", ""),
    )  # fmt: skip
    if version != expected or path.name != version:
        raise BundleError(f"version {version!r} does not match its content ({expected}) or folder")
    if meta.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise BundleError(
            f"feature schema {meta.get('feature_schema_version')} != {FEATURE_SCHEMA_VERSION}"
        )
    if list(meta.get("features") or []) != _expected_features(meta):
        raise BundleError("feature list differs from ml.features.ETA_FEATURES")
    cats = meta.get("categories") or {}
    for k, v in CATEGORIES.items():
        if k in meta["features"] and list(cats.get(k) or []) != list(v):
            raise BundleError(f"categories of {k} differ from ml.features.CATEGORIES")
    model = (loader or _xgboost_loader)(files[MODEL_FILE], meta)
    reference = pd.read_parquet(path / REFERENCE_FILE)
    return EtaBundle(str(version), meta, model, reference, float(meta["qhat"]))


def latest_bundle(models_dir: Path = ETA_MODELS_DIR) -> Path | None:
    """The bundle named by `<models_dir>/LATEST`, or None (no file, or no such folder)."""
    latest = Path(models_dir) / LATEST_FILE
    if not latest.is_file():
        return None
    version = latest.read_text(encoding="utf-8").strip()
    path = Path(models_dir) / version
    return path if version and path.is_dir() else None


def predict_quantiles(bundle: EtaBundle, frame: pd.DataFrame) -> np.ndarray:
    """Raw quantile predictions (n, 3), sorted per row (crossing repair)."""
    from traffic_twin.ml.features import model_frame

    if frame.empty:
        return np.empty((0, 3))
    x = model_frame(frame)[bundle.features]
    raw = np.asarray(bundle.model.predict(x), dtype=float)
    if raw.shape != (len(frame), len(bundle.meta["quantiles"])):
        raise BundleError(f"model returned shape {raw.shape} for {len(frame)} rows")
    return np.sort(raw, axis=1)


def predict_frame(bundle: EtaBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """eta_median_s, eta_low_s, eta_high_s (served integer bounds, float64), index of
    `frame`."""
    low, median, high = served_bounds(predict_quantiles(bundle, frame), bundle.qhat)
    return pd.DataFrame(
        {
            "eta_median_s": median.astype(np.float64),
            "eta_low_s": low.astype(np.float64),
            "eta_high_s": high.astype(np.float64),
        },
        index=frame.index,
    )


def _is_protected(path: Path) -> bool:
    try:
        Path(path).resolve().relative_to(PROTECTED_DIR.resolve())
        return True
    except ValueError:
        return False


def unseen_reason(bundle: EtaBundle, controller: str | None, day_type: str | None) -> str | None:
    """Why a run cannot get predictions (unseen controller or day type), or None."""
    cats = bundle.meta.get("categories") or {}
    seen = bundle.meta.get("seen") or {}
    for key, value in (("controller", controller), ("day_type", day_type)):
        allowed = seen.get(key) or cats.get(key) or []
        if value not in allowed:
            return f"{key} {value!r} not seen by the model {bundle.version} ({allowed})"
    return None


def predict_run(run_dir: Path, bundle: EtaBundle) -> Path | None:
    """Reads run.json + eta_snapshots.parquet; refuses an unseen controller or day type
    (returns None, logs the reason). Writes `<run_dir>/eta_prediction.parquet` (module
    docstring) for t inside the target window and returns its path."""
    from traffic_twin.ml.features import prediction_frame

    run_dir = Path(run_dir)
    if _is_protected(run_dir):
        log.warning("%s: Part 3-4 results are never written; no prediction", run_dir)
        return None
    meta_path, snap_path = run_dir / "run.json", run_dir / SNAPSHOT_FILE
    if not meta_path.is_file():
        log.warning("%s: no run.json; no prediction", run_dir)
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = meta.get("config") or {}
    controller = cfg.get("controller")
    day_type = (cfg.get("demand") or {}).get("day_type")
    reason = unseen_reason(bundle, controller, day_type)
    if reason is not None:
        log.warning("%s: %s; no prediction (the API serves the reference)", run_dir, reason)
        return None
    if not snap_path.is_file():
        log.warning("%s: no %s (run without EtaRecorder); no prediction", run_dir, SNAPSHOT_FILE)
        return None
    rows = pd.read_parquet(snap_path)
    for key, value in (("controller", controller), ("day_type", day_type)):
        if key not in rows:
            rows[key] = value
    frame = prediction_frame(rows)
    lo, hi = (bundle.meta.get("target") or {}).get("window_s", (0, 7200))
    t = frame["t"].astype(float)
    frame = frame[(t >= float(lo)) & (t < float(hi))]
    bounds = predict_frame(bundle, frame)
    out = pd.DataFrame(
        {
            "run_id": str(meta["run_id"]),
            "route": frame["trajet"].astype(str).to_numpy(),
            "t": frame["t"].astype(np.float64).to_numpy(),
            "eta_median_s": bounds["eta_median_s"].to_numpy(),
            "eta_low_s": bounds["eta_low_s"].to_numpy(),
            "eta_high_s": bounds["eta_high_s"].to_numpy(),
            "model_version": bundle.version,
        },
        columns=list(PREDICTION_COLUMNS),
    )
    out = out.sort_values(["route", "t"], kind="mergesort").reset_index(drop=True)
    path = run_dir / PREDICTION_FILE
    out.to_parquet(path, index=False)
    log.info("%s: %d predictions (%s)", path, len(out), bundle.version)
    return path


def demo_seed_range() -> tuple[int, int]:
    """`seeds.demo` of configs/api.yaml (D38), else 4000-4009."""
    from traffic_twin.config import load_config

    try:
        lo, hi = load_config(API_CONFIG)["seeds"]["demo"]
        return int(lo), int(hi)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return DEMO_SEEDS


def demo_runs(
    seeds: Sequence[int] = (4000,),
    configs: Sequence[Path] = (PITIE_FIXED_CONFIG, PITIE_ACTUATED_CONFIG, PITIE_RULES_CONFIG),
    out_root: Path = DEMO_DIR,
    bundle: Path | None = None,
) -> list[Path]:
    """run_scenario.run(..., demand_dir=<tmp>, observers=[EtaRecorder(5, False)]) for
    every config and demo seed, then `predict_run` with `bundle` (default: LATEST; no
    bundle -> runs without predictions, logged). Refuses seeds outside the demo range
    and an `out_root` inside data/results."""
    from traffic_twin.config import load_scenario
    from traffic_twin.ml.eta_dataset import EtaRecorder
    from traffic_twin.sim.run_scenario import run, run_dir

    lo, hi = demo_seed_range()
    bad = [s for s in seeds if not lo <= int(s) <= hi]
    if bad:
        raise ValueError(f"demo seeds must be in {lo}-{hi} (D38), got {bad}")
    if _is_protected(Path(out_root)):
        raise ValueError(f"{out_root} is inside {PROTECTED_DIR}: demo runs go to data/demo")
    dirs: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="tt_demo_demand_") as demand:
        for config in configs:
            cfg = load_scenario(config)
            for seed in seeds:
                run(
                    cfg, int(seed), Path(out_root), config_path=Path(config),
                    demand_dir=Path(demand), observers=[EtaRecorder(5, False)],
                )  # fmt: skip
                dirs.append(run_dir(cfg, int(seed), Path(out_root)))
    bundle_dir = bundle or latest_bundle()
    if bundle_dir is None:
        log.warning("no ETA bundle (data/models/eta/LATEST): demo runs have no predictions")
        return dirs
    b = load_bundle(bundle_dir)
    for d in dirs:
        predict_run(d, b)
    return dirs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_demo = sub.add_parser("demo", help="demo runs with ETA snapshots and predictions")
    p_demo.add_argument("--seeds", type=int, nargs="+", default=[4000])
    p_demo.add_argument("--out", type=Path, default=DEMO_DIR)
    p_demo.add_argument(
        "--configs", type=Path, nargs="+",
        default=[PITIE_FIXED_CONFIG, PITIE_ACTUATED_CONFIG, PITIE_RULES_CONFIG],
    )  # fmt: skip
    p_demo.add_argument("--bundle", type=Path, default=None, help="default: LATEST")
    p_runs = sub.add_parser("runs", help="predict existing runs that have eta_snapshots")
    p_runs.add_argument("run_dirs", type=Path, nargs="+")
    p_runs.add_argument("--bundle", type=Path, default=None, help="default: LATEST")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "demo":
        try:
            dirs = demo_runs(args.seeds, args.configs, args.out, args.bundle)
        except (ValueError, BundleError) as e:
            print(f"demo refused: {e}", file=sys.stderr)
            return 1
        for d in dirs:
            has = (d / PREDICTION_FILE).is_file()
            print(f"{d.as_posix()}: {'eta_prediction' if has else 'no prediction'}")
        return 0

    bundle_dir = args.bundle or latest_bundle()
    if bundle_dir is None:
        print("no ETA bundle (data/models/eta/LATEST)", file=sys.stderr)
        return 1
    try:
        bundle = load_bundle(bundle_dir)
    except BundleError as e:
        print(f"bad bundle {bundle_dir}: {e}", file=sys.stderr)
        return 1
    missing = 0
    for d in args.run_dirs:
        path = predict_run(d, bundle)
        print(f"{Path(d).as_posix()}: {path.as_posix() if path else 'no prediction'}")
        missing += path is None
    return 1 if missing == len(args.run_dirs) else 0


if __name__ == "__main__":
    raise SystemExit(main())

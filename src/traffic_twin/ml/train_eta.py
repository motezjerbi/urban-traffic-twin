"""ETA model (Part 5, EF-13; D47-D49, D51; contract 4.5): reference table, XGBoost
multi-quantile model, conformal (CQR) widening on calibration worlds, versioned bundle.

    python -m traffic_twin.ml.train_eta --config configs/eta.yaml [--dataset DIR]
    python -m traffic_twin.ml.train_eta --config configs/eta.yaml --freeze [--bundle DIR]

- **Reference (D49, spec 11.4):** mean, p10 and p90 of `y_zone_s` per (route = trajet,
  hour, day type) on the train worlds, `hour = floor((begin_s + insert_t) / 3600)`, plus
  one fallback row per route (`hour = -1`, `day_type = "*"`). It is also the
  degraded-mode table of the API. `reference.population` in eta.yaml picks the worlds:
  `all` (default, what the frozen eta1-56926796 bundle used: realistic **and** stress
  train worlds, and every stress world is labelled `tue_wed_thu_term`, so that day
  type's cells are about half stress) or `realistic` (the historical mean of spec 11.4;
  meant for the post-field-visit retrain, D67, with a dated `changes` entry).
- **Model (D47):** one XGBoost model, `reg:quantileerror` with the 10/50/90 % quantiles,
  early stopping on the val worlds, grid `max_depth` chosen on the val pinball loss of the
  sorted outputs. Nothing else is tuned.
- **Interval (D48):** served bounds `low = floor(q10 - qhat)`, `median = round(q50)`,
  `high = ceil(q90 + qhat)`, clipped to `0 <= low <= median <= high`; qhat is the smallest
  value of the grid whose pooled coverage (inclusive bounds) on the realistic calib worlds
  reaches the level (0.80). The same bounds are stored, served and evaluated.
- **Bundle (D51):** `data/models/eta/<model_version>/{model.ubj, meta.json,
  reference.parquet}` and `data/models/eta/LATEST`; `model_version = "eta1-" +
  sha8(dataset_id || params || qhat || model sha256)`. `--freeze` writes
  `data/models/eta/freeze.json` (version and bundle sha256) and refuses once test worlds
  exist in the dataset: `evaluate_eta --split test` needs it.

xgboost is imported inside the fit and the loader only: it is not installed in CI (D66),
and everything else (reference, conformal step, bounds, bundle files) is tested through
the `QuantilePredictor` protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import platform
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from traffic_twin.config import ConfigError, load_config, text_sha256
from traffic_twin.paths import ETA_CONFIG, ETA_DIR, MODELS_DIR

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)

ETA_MODELS_DIR = MODELS_DIR / "eta"
BUNDLE_FORMAT = "eta-bundle-1"
MODEL_FORMAT = "xgboost-ubj"
MODEL_FILE = "model.ubj"
META_FILE = "meta.json"
REFERENCE_FILE = "reference.parquet"
EVAL_FILE = "eval.json"
LATEST_FILE = "LATEST"
FREEZE_FILE = "freeze.json"
BUNDLE_FILES = (MODEL_FILE, REFERENCE_FILE, META_FILE)  # what the bundle sha256 covers
VERSION_PREFIX = "eta1-"
QUANTILES = (0.1, 0.5, 0.9)
TARGET = "y_zone_s"
BEGIN_S = 27000  # 07:30, `begin_s` of the Pitié configs
REFERENCE_COLUMNS = ("route", "hour", "day_type", "mean_s", "p10_s", "p90_s", "n")
FALLBACK_HOUR = -1
FALLBACK_DAY_TYPE = "*"
# Rows a split drops before fit / evaluation (D39), counted in `df.attrs["dropped"]`
DROP_REASONS = ("teleported", "missing_exit", "unjoined")
REFERENCE_POPULATIONS = ("all", "realistic")
# Columns `load_split` needs to apply the D39 drops (read even when `columns` is given)
_DROP_COLUMNS = ("teleported", TARGET, "t_feat", "tod_sin", "kind")


class QuantilePredictor(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """(n, 3) quantile predictions, any order (callers sort each row)."""
        ...


@runtime_checkable
class SerializablePredictor(Protocol):
    """A predictor the bundle can store: `to_bytes()` is the content of `model.ubj`."""

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def to_bytes(self) -> bytes: ...


class XGBQuantilePredictor:
    """An XGBoost booster cut at its best iteration (all its trees are used)."""

    def __init__(self, booster) -> None:
        self.booster = booster

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.asarray(self.booster.inplace_predict(X), dtype=float)
        return out.reshape(len(X), -1)

    def to_bytes(self) -> bytes:
        return bytes(self.booster.save_raw("ubj"))

    @classmethod
    def from_bytes(cls, raw: bytes, n_jobs: int | None = None) -> XGBQuantilePredictor:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(bytearray(raw))
        if n_jobs:
            booster.set_param({"nthread": int(n_jobs)})
        return cls(booster)


# -- small helpers ---------------------------------------------------------------------
def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def model_features(eta_cfg: Mapping | None = None) -> list[str]:
    """Model inputs in order: `ETA_FEATURES` minus `model.excluded_features` (D41)."""
    from traffic_twin.ml.features import ETA_FEATURES

    excluded = set(((eta_cfg or {}).get("model") or {}).get("excluded_features") or ())
    return [f for f in ETA_FEATURES if f not in excluded]


def quantiles_of(eta_cfg: Mapping) -> tuple[float, ...]:
    q = tuple(float(x) for x in eta_cfg["model"]["quantiles"])
    if len(q) != 3 or list(q) != sorted(q):
        raise ConfigError(f"model.quantiles must be 3 increasing levels, got {list(q)}")
    return q


def pinball_loss(y: np.ndarray, pred: np.ndarray, quantiles: Sequence[float] = QUANTILES) -> float:
    """Mean pinball (quantile) loss over the quantile columns of `pred`."""
    y = np.asarray(y, dtype=float)
    losses = []
    for i, a in enumerate(quantiles):
        d = y - pred[:, i]
        losses.append(np.mean(np.maximum(a * d, (a - 1.0) * d)))
    return float(np.mean(losses))


def interval_coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Share of `y` inside [low, high], bounds included (D48)."""
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((y >= low) & (y <= high)))


# -- data --------------------------------------------------------------------------------
def _part_files(dataset_dir: Path, split: str, worlds: Sequence[int] | None) -> list[Path]:
    root = Path(dataset_dir) / "samples"
    files = sorted(root.glob(f"split={split}/controller=*/part-*.parquet"))
    if worlds is not None:
        wanted = {int(w) for w in worlds}
        files = [f for f in files if int(f.stem.removeprefix("part-")) in wanted]
    return files


def load_split(
    dataset_dir: Path,
    split: str,
    kind: str | None = None,
    worlds: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """The sample rows of one split (`samples/split=<split>/controller=*/part-*.parquet`;
    `split="*"` reads every split, e.g. with `worlds` for a development fit), optionally
    of one world kind (realistic / stress). Teleported cars, cars without an exit time
    and rows without a feature row are dropped (D39); their counts are in
    `df.attrs["dropped"]`. Float feature columns are stored as float32 (what XGBoost
    uses anyway). `columns` reads only those columns (plus the ones the drops need).
    Raises FileNotFoundError when no part matches."""
    files = _part_files(dataset_dir, split, worlds)
    if not files:
        raise FileNotFoundError(
            f"no sample parts for split {split!r} under {Path(dataset_dir) / 'samples'}"
            + (f" (worlds {sorted(worlds)})" if worlds is not None else "")
        )
    cols = None
    if columns is not None:
        import pyarrow.parquet as pq

        present = set(pq.read_schema(files[0]).names)
        wanted = list(dict.fromkeys([*columns, *_DROP_COLUMNS]))
        cols = [c for c in wanted if c in present]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    if kind is not None:
        df = df[df["kind"].astype(str) == kind]
    teleported = (
        df["teleported"].astype("boolean").fillna(False).astype(bool)
        if "teleported" in df
        else pd.Series(False, index=df.index)
    )
    missing = df[TARGET].isna() & ~teleported
    joined_col = "t_feat" if "t_feat" in df else "tod_sin"  # NaN: no feature row in reach
    unjoined = (
        df[joined_col].isna() & ~teleported & ~missing
        if joined_col in df
        else pd.Series(False, index=df.index)
    )
    keep = ~(teleported | missing | unjoined)
    dropped = {
        "teleported": int(teleported.sum()),
        "missing_exit": int(missing.sum()),
        "unjoined": int(unjoined.sum()),
    }
    df = df[keep].reset_index(drop=True)
    floats = [c for c in model_features() if c in df and df[c].dtype == np.float64]
    if floats:
        df[floats] = df[floats].astype(np.float32)
    df.attrs["dropped"] = dropped
    df.attrs["files"] = len(files)
    return df


def hours_of(insert_t: pd.Series | np.ndarray, begin_s: int = BEGIN_S) -> np.ndarray:
    """Clock hour of an insertion time (s since scenario start)."""
    return np.floor((begin_s + np.asarray(insert_t, dtype=float)) / 3600.0).astype(int)


# -- reference (D49) ---------------------------------------------------------------------
def reference_population(eta_cfg: Mapping | None) -> str:
    """`reference.population` of eta.yaml: the train worlds the reference (and so the
    degraded-mode table) is fitted on; `all` when absent (the frozen bundle's rule)."""
    pop = str(((eta_cfg or {}).get("reference") or {}).get("population", "all"))
    if pop not in REFERENCE_POPULATIONS:
        raise ConfigError(f"reference.population must be one of {REFERENCE_POPULATIONS}: {pop!r}")
    return pop


def reference_rows(train: pd.DataFrame, population: str) -> pd.DataFrame:
    """The train rows the reference is fitted on (`all`, or one world kind)."""
    if population == "all":
        return train
    return train[train["kind"].astype(str) == population]


def fit_reference(train: pd.DataFrame, begin_s: int = BEGIN_S) -> pd.DataFrame:
    """route, hour, day_type, mean_s, p10_s, p90_s, n (+ fallback rows with hour = -1,
    day_type = "*") (D49)."""
    df = pd.DataFrame(
        {
            "route": train["trajet"].astype(str).to_numpy(),
            "hour": hours_of(train["insert_t"], begin_s),
            "day_type": train["day_type"].astype(str).to_numpy(),
            "y": train[TARGET].astype(float).to_numpy(),
        }
    )
    if df.empty:
        raise ValueError("fit_reference: no training rows")

    def agg(keys: list[str]) -> pd.DataFrame:
        g = df.groupby(keys, sort=True)["y"]
        return pd.DataFrame(
            {
                "mean_s": g.mean(),
                "p10_s": g.quantile(0.1),
                "p90_s": g.quantile(0.9),
                "n": g.size().astype(np.int64),
            }
        ).reset_index()

    main = agg(["route", "hour", "day_type"])
    fallback = agg(["route"])
    fallback["hour"] = FALLBACK_HOUR
    fallback["day_type"] = FALLBACK_DAY_TYPE
    out = pd.concat([main, fallback[list(REFERENCE_COLUMNS)]], ignore_index=True)
    out["hour"] = out["hour"].astype(np.int64)
    return out[list(REFERENCE_COLUMNS)].reset_index(drop=True)


def apply_reference(
    reference: pd.DataFrame, df: pd.DataFrame, begin_s: int = BEGIN_S
) -> pd.DataFrame:
    """mean_s, p10_s, p90_s of the reference for each row of `df` (index kept); a key
    missing from the table falls back to the route's row (hour -1, day type "*"), a
    route missing entirely gives NaN."""
    keys = pd.DataFrame(
        {
            "route": df["trajet"].astype(str).to_numpy(),
            "hour": hours_of(df["insert_t"], begin_s),
            "day_type": df["day_type"].astype(str).to_numpy(),
        }
    )
    cols = ["mean_s", "p10_s", "p90_s"]
    exact = reference[reference["hour"] != FALLBACK_HOUR]
    fb = reference[reference["hour"] == FALLBACK_HOUR].set_index("route")[cols]
    m = keys.merge(
        exact[["route", "hour", "day_type", *cols]], how="left", on=keys.columns.tolist()
    )
    f = fb.reindex(keys["route"]).reset_index(drop=True)
    out = m[cols].fillna(f)
    out.index = df.index
    return out


def reference_lookup(reference: pd.DataFrame, route: str, hour: int, day_type: str) -> dict | None:
    """One reference row as a dict (exact key, else the route's fallback, else None);
    `approximate` estimate of the degraded mode."""
    r = reference
    hit = r[(r["route"] == route) & (r["hour"] == hour) & (r["day_type"] == day_type)]
    if hit.empty:
        hit = r[(r["route"] == route) & (r["hour"] == FALLBACK_HOUR)]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in hit.columns}


# -- quantile model (D47) ----------------------------------------------------------------
def _grid_points(grid: Mapping | None) -> list[dict]:
    grid = dict(grid or {})
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*grid.values())]


def fit_quantile_model(
    train: pd.DataFrame, val: pd.DataFrame, eta_cfg: dict
) -> tuple[QuantilePredictor, dict]:
    """Grid on val pinball loss (D47): one `reg:quantileerror` model per grid point,
    early-stopped on val; the point with the lowest val pinball loss of its sorted
    outputs wins (first one on a tie). Returns the predictor (booster cut at its best
    iteration) and the fit record (grid, chosen params, best iteration, sizes)."""
    import xgboost as xgb

    from traffic_twin.ml.features import model_frame

    mcfg = eta_cfg["model"]
    quantiles = quantiles_of(eta_cfg)
    features = model_features(eta_cfg)
    params = dict(mcfg["params"])
    x_tr, x_va = model_frame(train)[features], model_frame(val)[features]
    y_tr, y_va = train[TARGET].to_numpy(float), val[TARGET].to_numpy(float)
    records, best = [], None
    for point in _grid_points(mcfg.get("grid")):
        t0 = time.perf_counter()
        reg = xgb.XGBRegressor(
            **params, **point, quantile_alpha=np.asarray(quantiles), enable_categorical=True
        )
        reg.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
        best_it = int(reg.best_iteration)
        predictor = XGBQuantilePredictor(reg.get_booster()[: best_it + 1])
        pred = np.sort(predictor.predict(x_va), axis=1)
        rec = {
            **point,
            "best_iteration": best_it,
            "rounds": int(reg.get_booster().num_boosted_rounds()),
            "hit_cap": best_it + 1 >= int(params.get("n_estimators", 0)),
            "val_pinball": pinball_loss(y_va, pred, quantiles),
            "val_mae": float(np.mean(np.abs(pred[:, 1] - y_va))),
            "fit_s": round(time.perf_counter() - t0, 1),
        }
        records.append(rec)
        log.info("grid %s: %s", point, rec)
        if best is None or rec["val_pinball"] < best[0]["val_pinball"]:
            best = (rec, point, predictor)
    assert best is not None
    rec, point, predictor = best
    info = {
        "features": features,
        "quantiles": list(quantiles),
        "params": {**params, **point},
        "grid": records,
        "chosen": point,
        "best_iteration": rec["best_iteration"],
        "val_pinball": rec["val_pinball"],
        "val_mae": rec["val_mae"],
        "n_train": len(train),
        "n_val": len(val),
        "xgboost": xgb.__version__,
    }
    return predictor, info


# -- conformal step and served bounds (D48) ----------------------------------------------
def conformal_grid(grid: tuple[float, float, float] = (-20.0, 60.0, 0.1)) -> np.ndarray:
    """The q-hat grid lo, lo + step, ..., hi (rounded to 6 decimals: no float drift)."""
    lo, hi, step = (float(x) for x in grid)
    if step <= 0 or hi < lo:
        raise ValueError(f"bad conformal grid {grid}")
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 6)


def served_bounds(pred: np.ndarray, qhat: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(low, median, high) whole seconds: floor(q10 - qhat), round(q50), ceil(q90 + qhat),
    clipped to 0 <= low <= median <= high (D48). Rows of `pred` are sorted first (quantile
    crossing repair); `round` is half-to-even (numpy `rint`)."""
    p = np.sort(np.asarray(pred, dtype=float).reshape(-1, 3), axis=1)
    if not np.isfinite(p).all():
        raise ValueError("served_bounds: non-finite quantile prediction")
    median = np.maximum(np.rint(p[:, 1]), 0.0)
    low = np.minimum(np.maximum(np.floor(p[:, 0] - qhat), 0.0), median)
    high = np.maximum(np.ceil(p[:, 2] + qhat), median)
    return low.astype(np.int64), median.astype(np.int64), high.astype(np.int64)


def fit_conformal(
    pred_calib: np.ndarray,
    y_calib: np.ndarray,
    level: float = 0.80,
    grid: tuple[float, float, float] = (-20.0, 60.0, 0.1),
) -> float:
    """Smallest q-hat on the grid whose coverage of the served integer bounds reaches
    `level` (D48). Coverage is non-decreasing in q-hat, so a bisection over the grid finds
    exactly the first grid value. ValueError when the calib set is empty or even the last
    grid value stays below `level`."""
    y = np.asarray(y_calib, dtype=float)
    if len(y) == 0:
        raise ValueError("fit_conformal: no calibration rows")
    p = np.sort(np.asarray(pred_calib, dtype=float).reshape(-1, 3), axis=1)
    if len(p) != len(y):
        raise ValueError(f"fit_conformal: {len(p)} predictions for {len(y)} targets")
    values = conformal_grid(grid)

    def covered(q: float) -> bool:
        low, _, high = served_bounds(p, q)
        return interval_coverage(y, low, high) >= level

    if not covered(float(values[-1])):
        raise ValueError(f"fit_conformal: coverage stays below {level} up to q-hat {values[-1]} s")
    lo, hi = 0, len(values) - 1  # invariant: covered(values[hi])
    if covered(float(values[0])):
        return float(values[0])
    while hi - lo > 1:  # invariant: not covered(values[lo])
        mid = (lo + hi) // 2
        if covered(float(values[mid])):
            hi = mid
        else:
            lo = mid
    return float(values[hi])


def conformal_by_group(
    pred: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    level: float = 0.80,
    grid: tuple[float, float, float] = (-20.0, 60.0, 0.1),
) -> dict[str, float]:
    """Per-group q-hat (the per-trajet variant of D48: reported, never served)."""
    groups = np.asarray(groups).astype(str)
    out = {}
    for g in sorted(set(groups)):
        m = groups == g
        out[g] = fit_conformal(np.asarray(pred)[m], np.asarray(y)[m], level, grid)
    return out


# -- bundle (D51) ------------------------------------------------------------------------
def model_version(dataset_id: str, params: Mapping, qhat: float, model_sha256: str) -> str:
    """`eta1-` + sha8(dataset_id || params || q-hat || model sha256) (D51)."""
    payload = _canonical([dataset_id, dict(params), round(float(qhat), 6), model_sha256])
    return VERSION_PREFIX + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def bundle_sha256(bundle_dir: Path) -> str:
    """sha256 of the bundle: of `name sha256` lines of model.ubj, reference.parquet and
    meta.json (eval.json, written later, is not part of it)."""
    lines = [f"{name} {sha256_file(Path(bundle_dir) / name)}" for name in BUNDLE_FILES]
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def _feature_dtypes(features: Sequence[str]) -> dict[str, object]:
    from traffic_twin.ml.features import CATEGORIES

    return {f: list(CATEGORIES[f]) if f in CATEGORIES else "float" for f in features}


def save_bundle(
    predictor: SerializablePredictor,
    reference: pd.DataFrame,
    qhat: float,
    info: Mapping,
    eta_cfg: Mapping,
    dataset_id: str,
    models_dir: Path = ETA_MODELS_DIR,
    extra: Mapping | None = None,
) -> str:
    """Write data/models/eta/<model_version>/ and LATEST; returns model_version (D51).

    `info` is the fit record of `fit_quantile_model` (features, quantiles, params, ...);
    `extra` adds keys to meta.json (splits, sizes, calib coverage, provenance)."""
    from traffic_twin.ml.features import CATEGORIES, FEATURE_SCHEMA_VERSION
    from traffic_twin.sim.results import code_version

    raw = predictor.to_bytes()
    model_sha = sha256_bytes(raw)
    params = dict(info["params"])
    qhat = round(float(qhat), 6)
    version = model_version(dataset_id, params, qhat, model_sha)
    out = Path(models_dir) / version
    out.mkdir(parents=True, exist_ok=True)
    (out / MODEL_FILE).write_bytes(raw)
    ref = reference[list(REFERENCE_COLUMNS)].copy()
    ref["model_version"] = version
    ref.to_parquet(out / REFERENCE_FILE, index=False)
    features = list(info["features"])
    mcfg = eta_cfg.get("model") or {}
    meta = {
        "bundle_format": BUNDLE_FORMAT,
        "model_version": version,
        "model_format": MODEL_FORMAT,
        "model_sha256": model_sha,
        "reference_sha256": sha256_file(out / REFERENCE_FILE),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": features,
        "dtypes": _feature_dtypes(features),
        "categories": {k: list(v) for k, v in CATEGORIES.items()},
        "excluded_features": list(mcfg.get("excluded_features") or []),
        "quantiles": list(info["quantiles"]),
        "qhat": qhat,
        "conformal": dict(eta_cfg.get("conformal") or {}),
        "target": dict(eta_cfg.get("target") or {"name": TARGET}),
        "params": params,
        "grid": list(info.get("grid") or []),
        "best_iteration": info.get("best_iteration"),
        "dataset_id": dataset_id,
        "splits": dict(eta_cfg.get("seeds") or {}),
        "code_version": code_version(),
        "versions": {
            "python": platform.python_version(),
            "xgboost": info.get("xgboost"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **dict(extra or {}),
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=1, default=str) + "\n", encoding="utf-8")
    (Path(models_dir) / LATEST_FILE).write_text(version + "\n", encoding="utf-8")
    return version


# -- freeze (D51) ------------------------------------------------------------------------
def read_freeze(models_dir: Path = ETA_MODELS_DIR) -> dict | None:
    p = Path(models_dir) / FREEZE_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def check_freeze(bundle_dir: Path, models_dir: Path = ETA_MODELS_DIR) -> dict:
    """The freeze record when it names this bundle and its sha256 still matches; raises
    ValueError otherwise (no test evaluation without a matching freeze, D51)."""
    freeze = read_freeze(models_dir)
    if freeze is None:
        raise ValueError(
            f"no {FREEZE_FILE} in {models_dir}: freeze the bundle (train_eta --freeze) "
            "before any test world is generated"
        )
    meta = json.loads((Path(bundle_dir) / META_FILE).read_text(encoding="utf-8"))
    if freeze.get("model_version") != meta.get("model_version"):
        raise ValueError(
            f"frozen model {freeze.get('model_version')} != bundle {meta.get('model_version')}"
        )
    sha = bundle_sha256(bundle_dir)
    if freeze.get("bundle_sha256") != sha:
        raise ValueError(f"bundle {bundle_dir} changed since the freeze (sha256 {sha[:12]})")
    return freeze


def _test_parts(dataset_dir: Path | None) -> list[Path]:
    if dataset_dir is None:
        return []
    return sorted((Path(dataset_dir) / "samples").glob("split=test/controller=*/part-*.parquet"))


def freeze(
    bundle_dir: Path,
    models_dir: Path = ETA_MODELS_DIR,
    dataset_dir: Path | None = None,
    refreeze: bool = False,
    eta_cfg: Mapping | None = None,
) -> dict:
    """Write `<models_dir>/freeze.json` for the bundle (D51). Refuses a bundle fitted on
    overridden worlds, a dataset that already holds test worlds, and a second freeze of a
    different version unless `refreeze` and a dated `changes` entry in eta.yaml (D50)."""
    from traffic_twin.sim.results import code_version

    bundle_dir = Path(bundle_dir)
    meta = json.loads((bundle_dir / META_FILE).read_text(encoding="utf-8"))
    if meta.get("worlds_override"):
        raise ValueError("a bundle fitted on overridden worlds (development) cannot be frozen")
    if _test_parts(dataset_dir):
        raise ValueError(
            f"{dataset_dir} already holds test worlds: the freeze must precede them (D51)"
        )
    old = read_freeze(models_dir)
    if old is not None and old.get("model_version") != meta["model_version"]:
        if not refreeze:
            raise ValueError(
                f"{FREEZE_FILE} already freezes {old.get('model_version')}; a second attempt "
                "needs a dated `changes` entry in eta.yaml and --refreeze (D50)"
            )
        if not (eta_cfg or {}).get("changes"):
            raise ValueError("--refreeze needs a dated `changes` entry in eta.yaml (D50)")
    record = {
        "model_version": meta["model_version"],
        "bundle_sha256": bundle_sha256(bundle_dir),
        "model_sha256": meta["model_sha256"],
        "dataset_id": meta.get("dataset_id"),
        "test_seeds": (meta.get("splits") or {}).get("test"),
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": code_version(),
        "previous": old if refreeze else None,
    }
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    (Path(models_dir) / FREEZE_FILE).write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    return record


# -- CLI ---------------------------------------------------------------------------------
def resolve_dataset(eta_cfg: Mapping, dataset: Path | None = None) -> Path:
    """`--dataset`, else `<outputs.data>/<dataset_id>` (or its `-dev` variant, made from a
    dirty tree with --allow-dirty)."""
    if dataset is not None:
        if not Path(dataset).is_dir():
            raise FileNotFoundError(f"dataset {dataset} not found")
        return Path(dataset)
    from traffic_twin.ml.eta_dataset import dataset_id

    root = Path((eta_cfg.get("outputs") or {}).get("data", ETA_DIR))
    base = dataset_id(dict(eta_cfg))
    for name in (base, f"{base}-dev"):
        if (root / name).is_dir():
            return root / name
    raise FileNotFoundError(f"no dataset {base} (or {base}-dev) under {root}: run eta-data")


def read_manifest(dataset_dir: Path) -> dict:
    p = Path(dataset_dir) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def _begin_s(eta_cfg: Mapping) -> int:
    """`begin_s` of the base configs (they share it; 07:30 at Pitié)."""
    from traffic_twin.config import load_scenario

    begins = set()
    for path in (eta_cfg.get("base_configs") or {}).values():
        if Path(path).is_file():
            begins.add(int(load_scenario(path).get("begin_s", BEGIN_S)))
    if len(begins) > 1:
        raise ConfigError(f"base configs disagree on begin_s: {sorted(begins)}")
    return begins.pop() if begins else BEGIN_S


def train(
    eta_cfg: dict,
    dataset_dir: Path,
    models_dir: Path = ETA_MODELS_DIR,
    worlds: Mapping[str, Sequence[int]] | None = None,
    population: str | None = None,
) -> dict:
    """Fit the reference, the quantile model and q-hat, write the bundle; returns a
    summary. `worlds` ({train, val, calib} -> world seeds, from any split) is the
    development override of the gate (the bundle cannot be frozen)."""
    begin_s = _begin_s(eta_cfg)
    manifest = read_manifest(dataset_dir)
    data_id = manifest.get("dataset_id") or Path(dataset_dir).name
    population = population or (eta_cfg.get("conformal") or {}).get("population", "realistic")

    def load(name: str) -> pd.DataFrame:
        if worlds is not None:
            return load_split(dataset_dir, "*", worlds=worlds[name])
        return load_split(dataset_dir, name)

    tr, va, ca = load("train"), load("val"), load("calib")
    log.info("rows: train %d, val %d, calib %d", len(tr), len(va), len(ca))
    ref_population = reference_population(eta_cfg)
    reference = fit_reference(reference_rows(tr, ref_population), begin_s)
    predictor, info = fit_quantile_model(tr, va, eta_cfg)
    from traffic_twin.ml.features import model_frame

    cal = ca if population == "all" else ca[ca["kind"].astype(str) == population]
    if cal.empty:
        raise ValueError(f"no calib rows of population {population!r}")
    ccfg = eta_cfg.get("conformal") or {}
    level = float(ccfg.get("level", 0.80))
    grid = tuple(ccfg.get("grid_s", (-20.0, 60.0, 0.1)))
    pred_cal = np.sort(predictor.predict(model_frame(cal)[info["features"]]), axis=1)
    y_cal = cal[TARGET].to_numpy(float)
    qhat = fit_conformal(pred_cal, y_cal, level, grid)
    low, _, high = served_bounds(pred_cal, qhat)
    calib_cov = interval_coverage(y_cal, low, high)
    raw_cov = interval_coverage(y_cal, pred_cal[:, 0], pred_cal[:, 2])
    pred_va = np.sort(predictor.predict(model_frame(va)[info["features"]]), axis=1)
    y_va = va[TARGET].to_numpy(float)
    _, med_va, _ = served_bounds(pred_va, qhat)
    ref_va = apply_reference(reference, va, begin_s)["mean_s"].to_numpy()
    summary = {
        "val_mae": float(np.mean(np.abs(med_va - y_va))),
        "val_ref_mae": float(np.nanmean(np.abs(ref_va - y_va))),
        "qhat": qhat,
        "calib_coverage": calib_cov,
        "calib_raw_coverage": raw_cov,
        "calib_population": population,
        "n_calib": len(cal),
    }
    seen = {
        "controller": sorted({str(x) for x in tr["controller"].unique()}),
        "day_type": sorted({str(x) for x in tr["day_type"].unique()}),
    }
    extra = {
        "begin_s": begin_s,
        "eta_yaml_sha256": text_sha256(ETA_CONFIG) if ETA_CONFIG.is_file() else None,
        "dataset_code_version": manifest.get("code_version"),
        "n": {"train": len(tr), "val": len(va), "calib": len(ca)},
        "dropped": {
            "train": tr.attrs.get("dropped"),
            "val": va.attrs.get("dropped"),
            "calib": ca.attrs.get("dropped"),
        },  # fmt: skip
        "seen": seen,
        "reference_population": ref_population,
        "calibration": summary,
        "worlds_override": {k: [int(s) for s in v] for k, v in worlds.items()} if worlds else None,
    }
    version = save_bundle(predictor, reference, qhat, info, eta_cfg, data_id, models_dir, extra)
    summary["model_version"] = version
    summary["bundle"] = (Path(models_dir) / version).as_posix()
    summary["best_iteration"] = info["best_iteration"]
    summary["chosen"] = info["chosen"]
    return summary


def stop_rules(summary: Mapping) -> list[str]:
    """Phase 3 stop rules of `eta-train` (plan section 8, step 3)."""
    out = []
    if not -10.0 <= float(summary["qhat"]) <= 20.0:
        out.append(f"q-hat {summary['qhat']} s outside [-10, 20]")
    if not float(summary["val_mae"]) < float(summary["val_ref_mae"]):
        out.append(
            f"val MAE {summary['val_mae']:.2f} s >= reference {summary['val_ref_mae']:.2f} s"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ETA_CONFIG)
    parser.add_argument("--dataset", type=Path, default=None, help="data/eta/<dataset_id>")
    parser.add_argument("--models", type=Path, default=None, help="default outputs.models")
    parser.add_argument("--freeze", action="store_true", help="freeze a bundle (no fit)")
    parser.add_argument("--bundle", type=Path, default=None, help="bundle to freeze (LATEST)")
    parser.add_argument("--refreeze", action="store_true", help="second attempt (D50)")
    for name in ("train", "val", "calib"):
        parser.add_argument(
            f"--{name}-worlds", type=int, nargs="+", default=None,
            help=f"development override: {name} world seeds from any split (gate only)",
        )  # fmt: skip
    parser.add_argument(
        "--conformal-population", choices=("realistic", "stress", "all"), default=None,
        help="calib population of the CQR step (default conformal.population)",
    )  # fmt: skip
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    eta_cfg = load_config(args.config)
    models_dir = args.models or Path((eta_cfg.get("outputs") or {}).get("models", ETA_MODELS_DIR))

    if args.freeze:
        bundle = args.bundle
        if bundle is None:
            from traffic_twin.ml.predict import latest_bundle

            bundle = latest_bundle(models_dir)
            if bundle is None:
                print(f"no bundle under {models_dir} (LATEST)", file=sys.stderr)
                return 1
        try:
            dataset = resolve_dataset(eta_cfg, args.dataset)
        except FileNotFoundError:
            dataset = None
        try:
            record = freeze(bundle, models_dir, dataset, args.refreeze, eta_cfg)
        except ValueError as e:
            print(f"freeze refused: {e}", file=sys.stderr)
            return 1
        print(json.dumps(record, indent=1))
        return 0

    over = {k: getattr(args, f"{k}_worlds") for k in ("train", "val", "calib")}
    worlds = None
    if any(v is not None for v in over.values()):
        if any(v is None for v in over.values()):
            parser.error("--train-worlds, --val-worlds and --calib-worlds go together")
        worlds = over
    dataset = resolve_dataset(eta_cfg, args.dataset)
    summary = train(eta_cfg, dataset, models_dir, worlds, args.conformal_population)
    print(json.dumps(summary, indent=1, default=float))
    problems = stop_rules(summary)
    for p in problems:
        print(f"STOP: {p} (investigate on val / calib only)", file=sys.stderr)
    if math.isnan(summary["val_ref_mae"]):
        print("warning: reference undefined on some val rows", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

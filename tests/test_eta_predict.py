"""Precomputed ETA predictions (D51): `predict_run` writes exactly the Supabase
`public.eta_prediction` columns (uuid run_id from run.json, t on the 5 s grid of the
target window, low <= median <= high, whole seconds); an unseen controller or day type,
or a run without snapshots, writes nothing; demo runs refuse foreign seeds and
data/results."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_eta_train import linear_loader, make_fake_bundle, synthetic_rows

from traffic_twin.ml.features import CATEGORIES, ROW_FEATURES, TRAJETS
from traffic_twin.ml.predict import (
    PREDICTION_COLUMNS,
    PREDICTION_FILE,
    SNAPSHOT_FILE,
    demo_runs,
    load_bundle,
    predict_run,
)

MIGRATION = Path("supabase/migrations/20261011000000_init_schema.sql")


@pytest.fixture
def bundle(tmp_path, eta_cfg):
    rng = np.random.default_rng(5)
    train = pd.concat([synthetic_rows(100, rng, 2000, c) for c in CATEGORIES["controller"]])
    return load_bundle(make_fake_bundle(tmp_path / "models", eta_cfg, train), linear_loader)


def make_run(root: Path, controller: str = "rules", day_type: str = "tue_wed_thu_term",
             snapshots: bool = True, t_max: int = 7800) -> tuple[Path, str]:  # fmt: skip
    """A run folder with run.json and 5 s eta_snapshots (beyond the 7200 s window)."""
    run_dir = root / f"pitie_{controller}_seed4000"
    run_dir.mkdir(parents=True)
    run_id = str(uuid.uuid4())
    meta = {"run_id": run_id, "scenario": run_dir.name[:-9], "seed": 4000,
            "config": {"controller": controller, "demand": {"day_type": day_type}}}  # fmt: skip
    (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    if snapshots:
        rng = np.random.default_rng(6)
        t = np.arange(5, t_max + 1, 5, dtype=float)
        rows = synthetic_rows(len(t), rng, 4000, controller, "realistic", day_type)
        snap = rows[[c for c in ROW_FEATURES if c in rows]].assign(t=t, t_feat=t - 1)
        snap.insert(0, "run_id", run_id)
        snap.drop(columns=["controller", "day_type"]).to_parquet(run_dir / SNAPSHOT_FILE)
    return run_dir, run_id


def supabase_columns() -> list[str]:
    sql = MIGRATION.read_text(encoding="utf-8")
    body = re.search(r"create table if not exists public\.eta_prediction \((.*?)\n\);", sql, re.S)
    assert body, "eta_prediction table not found in the migration"
    cols = []
    for line in body.group(1).splitlines():
        m = re.match(r"\s+([a-z_]+)\s+(uuid|text|real)\b", line)
        if m:
            cols.append(m.group(1))
    return cols


def test_prediction_file_has_the_supabase_columns(tmp_path, bundle):
    run_dir, run_id = make_run(tmp_path)
    path = predict_run(run_dir, bundle)
    assert path == run_dir / PREDICTION_FILE
    df = pd.read_parquet(path)
    assert list(df.columns) == list(PREDICTION_COLUMNS) == supabase_columns()
    assert set(df["run_id"]) == {run_id} and uuid.UUID(run_id)
    assert set(df["model_version"]) == {bundle.version}
    for c in ("t", "eta_median_s", "eta_low_s", "eta_high_s"):
        assert df[c].dtype == np.float64
    assert (
        (df["eta_low_s"] <= df["eta_median_s"]) & (df["eta_median_s"] <= df["eta_high_s"])
    ).all()
    for c in ("eta_median_s", "eta_low_s", "eta_high_s"):
        assert (df[c] == np.round(df[c])).all() and (df[c] >= 0).all()
    # t on the 5 s grid of the target window [0, 7200): 5, 10, ..., 7195 per route
    assert sorted(set(df["route"])) == sorted(TRAJETS)
    for _, g in df.groupby("route"):
        assert list(g["t"]) == list(np.arange(5.0, 7200.0, 5.0))
    assert df.equals(df.sort_values(["route", "t"]).reset_index(drop=True))
    assert not df.duplicated(["run_id", "route", "t", "model_version"]).any()  # primary key


def test_unseen_controller_or_day_type_writes_nothing(tmp_path, bundle, caplog):
    run_dir, _ = make_run(tmp_path / "a", controller="rl")
    assert predict_run(run_dir, bundle) is None
    assert not (run_dir / PREDICTION_FILE).exists()
    assert "controller 'rl'" in caplog.text
    run_dir, _ = make_run(tmp_path / "b", day_type="saturday")
    assert predict_run(run_dir, bundle) is None
    assert not (run_dir / PREDICTION_FILE).exists()


def test_run_without_snapshots_or_run_json(tmp_path, bundle):
    run_dir, _ = make_run(tmp_path, snapshots=False)
    assert predict_run(run_dir, bundle) is None
    assert not (run_dir / PREDICTION_FILE).exists()
    assert predict_run(tmp_path / "nothing", bundle) is None


def test_results_dir_is_never_written(tmp_path, bundle, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir, _ = make_run(tmp_path / "data" / "results")
    assert predict_run(run_dir, bundle) is None
    assert not (run_dir / PREDICTION_FILE).exists()


def test_demo_runs_refuse_foreign_seeds_and_results(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="demo seeds"):
        demo_runs(seeds=(1,), out_root=tmp_path)
    with pytest.raises(ValueError, match="demo seeds"):
        demo_runs(seeds=(4000, 4010), out_root=tmp_path)
    with pytest.raises(ValueError, match="data/demo"):
        demo_runs(seeds=(4000,), out_root=Path("data/results"))

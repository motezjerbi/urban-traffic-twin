"""Run staleness (evaluation.runs): hashes as in Part 3, plus the Part 4 KPI version rule
(a run made with an older KPI set, or before Part 4 without `kpi_version`, is stale for a
caller that requires the new KPIs and is rerun by `run_missing`), and the optional
events table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from traffic_twin.evaluation.runs import (
    LEGACY_KPI_VERSION,
    TABLES,
    StaleRunsError,
    check_current,
    expected_hashes,
    kpi_version,
    load_runs,
    run_missing,
    stale_reason,
    stale_seeds,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
NOW = {"config_hash": "c1", "demand_hash": "d1"}


def test_kpi_version_of_a_run():
    assert kpi_version({"kpi_version": 2}) == 2
    assert kpi_version({}) == LEGACY_KPI_VERSION == 1  # Part 3 run.json: no key


def test_stale_reason_adds_the_kpi_version_only_when_required():
    old = {**NOW, "kpi_version": 1}
    assert stale_reason(old, NOW) is None  # Part 3 callers: hashes only (unchanged)
    assert stale_reason(old, NOW, min_kpi_version=1) is None
    assert stale_reason(old, NOW, min_kpi_version=2) == "kpi_version 1 < required 2"
    assert stale_reason({**NOW}, NOW, 2) == "kpi_version 1 < required 2"
    assert stale_reason({**NOW, "kpi_version": 3}, NOW, 2) is None
    both = stale_reason({"config_hash": "c0", "demand_hash": "d1"}, NOW, 2)
    assert both == "config_hash c0 != current c1; kpi_version 1 < required 2"


def _write(results: Path, scenario: str, seed: int, meta: dict) -> None:
    d = results / f"{scenario}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"user_type": "car", "kpi": "time_loss_mean", "value": 1.0}]).to_csv(
        d / "kpi_run.csv", index=False
    )
    (d / "run.json").write_text(json.dumps({"seed": seed, **meta}), encoding="utf-8")


def test_check_current_refuses_old_kpi_versions(tmp_path, pitie_cfg):
    h = expected_hashes(pitie_cfg)
    _write(tmp_path, "pitie_fixed", 1, {**h, "kpi_version": 2})
    _write(tmp_path, "pitie_fixed", 2, {**h})
    runs = load_runs(tmp_path, "pitie_fixed", [1, 2])
    assert check_current(runs, pitie_cfg) == h  # hashes only: current
    assert stale_seeds(runs, h, 2) == {2: "kpi_version 1 < required 2"}
    with pytest.raises(StaleRunsError, match="seed 2: kpi_version 1 < required 2"):
        check_current(runs, pitie_cfg, min_kpi_version=2)


def test_run_missing_reruns_old_kpi_versions_only_when_required(tmp_path, pitie_cfg, monkeypatch):
    import traffic_twin.sim.run_scenario as rs

    h = expected_hashes(pitie_cfg)
    _write(tmp_path, "pitie_fixed", 1, {**h, "kpi_version": 2})
    _write(tmp_path, "pitie_fixed", 2, {**h, "kpi_version": 1})
    rerun: list[int] = []

    def fake_run(cfg, seed, out_root, gui=False, config_path=None):
        rerun.append(seed)
        _write(Path(out_root), cfg["scenario"], seed, {**expected_hashes(cfg), "kpi_version": 2})

    monkeypatch.setattr(rs, "run", fake_run)
    assert run_missing(pitie_cfg, [1, 2, 3], tmp_path) == [3]  # absent seed only
    assert run_missing(pitie_cfg, [1, 2, 3], tmp_path, min_kpi_version=2) == [2]
    assert rerun == [3, 2]
    assert run_missing(pitie_cfg, [1, 2, 3], tmp_path, min_kpi_version=2) == []


def test_events_table_is_optional(tmp_path):
    assert TABLES["events"] == "events.parquet"
    _write(tmp_path, "s", 1, NOW)
    _write(tmp_path, "s", 2, NOW)
    ev = pd.DataFrame(
        [{"run_id": "r", "t": 3.0, "source": "guard", "rule": "ped_wait", "stage_from": 0,
          "stage_to": 1, "detail": ""}]
    )  # fmt: skip
    ev.to_parquet(tmp_path / "s_seed2" / "events.parquet", index=False)
    runs = load_runs(tmp_path, "s", [1, 2], tables=("kpis", "events"))
    assert runs.events["seed"].tolist() == [2] and runs.events["rule"].tolist() == ["ped_wait"]
    assert load_runs(tmp_path, "s", [1], tables=("events",)).events.empty

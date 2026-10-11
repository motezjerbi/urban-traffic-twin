"""The grid scenario is frozen (Part 3, decision D2): its demand files and its KPIs for
seeds 1-3 must stay identical to the fixture captured before any Part 3 change
(tests/data/grid_baseline_kpis.csv, tests/data/grid_baseline_meta.json).

The KPI comparison is exact (rel 1e-9) on the platform the fixture was captured on and
within 5 % elsewhere (floating point differs across platforms), with a warning.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

from traffic_twin.sim.demand import build_routes, write_demand

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_KPIS = Path(__file__).parent / "data" / "grid_baseline_kpis.csv"
FIXTURE_META = Path(__file__).parent / "data" / "grid_baseline_meta.json"
KEY = ["seed", "user_type", "kpi"]

# sha256 of the grid demand text ("\n" line ends), captured with the KPI fixture:
# build_routes(scenario_base) and build_additional(scenario_base, 239.6), 239.6 m being
# the length of the bus stop lane e_out_1 in sumo/network/intersection.net.xml.
ROUTES_SHA256 = "19abd33bbcf5e3cad05fcd5f935d61ccb128c8879984e33e7152eb5fe31e7605"
STOPS_SHA256 = "f2343e36ed8bca1d09ed9348ccf258392f37dc1338b8c23291f301e3058705cc"

EXACT_REL = 1e-9
CROSS_PLATFORM_REL = 0.05


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def baseline() -> tuple[pd.DataFrame, dict]:
    kpis = pd.read_csv(FIXTURE_KPIS)
    meta = json.loads(FIXTURE_META.read_text(encoding="utf-8"))
    return kpis, meta


def test_fixture_is_well_formed(baseline):
    kpis, meta = baseline
    assert list(kpis.columns) == [*KEY, "value"]
    assert not kpis.duplicated(KEY).any()
    assert sorted(kpis["seed"].unique()) == [1, 2, 3]
    assert {"car", "bus", "pedestrian", "ambulance"} <= set(kpis["user_type"])
    assert {"platform", "sumo_version", "git"} <= set(meta)


def test_grid_routes_unchanged(base_scenario):
    assert sha256(build_routes(base_scenario)) == ROUTES_SHA256


def test_grid_demand_files_unchanged(base_scenario, tmp_path):
    """The files the simulation reads (whatever their names) keep the frozen content."""
    pytest.importorskip("sumolib")
    rou, stops = write_demand(base_scenario, tmp_path)
    assert sha256(rou.read_text(encoding="utf-8")) == ROUTES_SHA256
    assert sha256(stops.read_text(encoding="utf-8")) == STOPS_SHA256


def _sumo_version() -> str:
    try:
        import sumolib

        return sumolib.version.gitDescribe()
    except Exception:  # version lookup only feeds a message
        return "unknown"


@pytest.mark.sumo
def test_grid_kpis_match_baseline(base_scenario, baseline, tmp_path, monkeypatch):
    """Full 2 h runs of seeds 1-3 from a scratch copy of the grid net; every fixture row
    must be reproduced (extra KPI rows are allowed)."""
    from traffic_twin.sim.run_scenario import run, run_dir

    expected, meta = baseline
    cfg = copy.deepcopy(base_scenario)
    net = Path(cfg["network"]["net_file"])
    (tmp_path / net).parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / net, tmp_path / net)
    monkeypatch.chdir(tmp_path)

    out_root = tmp_path / "results"
    frames = []
    for seed in sorted(int(s) for s in expected["seed"].unique()):
        run(cfg, seed, out_root)
        frames.append(pd.read_csv(run_dir(cfg, seed, out_root) / "kpi_run.csv"))
    got = pd.concat(frames, ignore_index=True)[[*KEY, "value"]]

    exact = sys.platform == meta["platform"]
    rel = EXACT_REL if exact else CROSS_PLATFORM_REL
    if not exact:
        warnings.warn(
            f"grid fixture captured on {meta['platform']}, running on {sys.platform}: "
            f"KPIs compared within rel={rel}",
            UserWarning,
            stacklevel=1,
        )

    rows = expected.merge(got, on=KEY, how="left", suffixes=("_expected", "_got"))
    missing = rows[rows["value_got"].isna()]
    assert missing.empty, f"KPI rows missing from the run:\n{missing[KEY].to_string()}"
    bad = [
        f"seed {r.seed} {r.user_type}.{r.kpi}: expected {r.value_expected!r}, got {r.value_got!r}"
        for r in rows.itertuples()
        if r.value_got != pytest.approx(r.value_expected, rel=rel)
    ]
    assert not bad, (
        f"grid KPIs changed (rel={rel}; fixture SUMO {meta['sumo_version']} at {meta['git']}, "
        f"running SUMO {_sumo_version()}):\n" + "\n".join(bad)
    )

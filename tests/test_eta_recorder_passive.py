"""The ETA recorder is a passive run observer (D63): S2 on the Pitié junction gives the
same KPI rows, signal table and events with and without it, and the recorder writes its
5 s feature snapshots next to the run. Marked `sumo`."""

import copy

import pandas as pd
import pytest

from traffic_twin.ml.eta_dataset import SNAPSHOT_COLUMNS, EtaRecorder
from traffic_twin.ml.features import TRAJETS, prediction_frame
from traffic_twin.sim.run_scenario import run, run_dir

pytestmark = pytest.mark.sumo

SEED = 101  # passivity checks only (plan section 3)


@pytest.fixture(scope="module")
def runs(rules_scenario, tmp_path_factory):
    """pitie_rules, seed 101, 900 s with an ambulance at 300 s (preemption events), run
    without and with the recorder; (out_root, cfg, recorder)."""
    cfg = copy.deepcopy(rules_scenario)
    cfg["duration_s"] = 900
    cfg["emergency"] = {"enabled": True, "depart_s": 300, "route": ["n_in", "s_out"]}
    root = tmp_path_factory.mktemp("eta_passive")
    run(cfg, SEED, root / "plain", demand_dir=root / "demand_plain")
    recorder = EtaRecorder(persist_every_s=5, record_samples=True)
    run(cfg, SEED, root / "rec", demand_dir=root / "demand_rec", observers=[recorder])
    return root, cfg, recorder


def table(root, cfg, name: str) -> pd.DataFrame:
    path = run_dir(cfg, SEED, root) / name
    df = pd.read_csv(path) if name.endswith(".csv") else pd.read_parquet(path)
    return df.drop(columns=["run_id"], errors="ignore")


@pytest.mark.parametrize("name", ["kpi_run.csv", "signal.parquet", "events.parquet"])
def test_recorder_leaves_the_run_unchanged(runs, name):
    root, cfg, _ = runs
    plain, rec = table(root / "plain", cfg, name), table(root / "rec", cfg, name)
    assert len(plain) > 0
    pd.testing.assert_frame_equal(plain, rec, check_exact=True)


def test_recorder_writes_snapshots_on_the_5_s_grid(runs):
    root, cfg, recorder = runs
    snaps = pd.read_parquet(run_dir(cfg, SEED, root / "rec") / "eta_snapshots.parquet")
    assert not (run_dir(cfg, SEED, root / "plain") / "eta_snapshots.parquet").exists()
    assert tuple(snaps.columns) == SNAPSHOT_COLUMNS
    run_id = pd.read_csv(run_dir(cfg, SEED, root / "rec") / "kpi_run.csv")["run_id"].iloc[0]
    assert snaps["run_id"].eq(run_id).all()
    assert snaps["t"].iloc[0] == 5.0 and snaps["t"].diff().dropna().eq(5.0).all()
    assert (snaps["t"] == snaps["t_feat"] + 1).all()
    assert snaps["controller"].eq("rules").all() and snaps["day_type"].eq("tue_wed_thu_term").all()
    assert snaps["amb_present"].max() == 1 and snaps["preempt_active"].max() == 1
    assert len(recorder.rows) >= 900 and len(recorder.exit_t) > 100
    pred = prediction_frame(snaps)
    assert len(pred) == len(snaps) * len(TRAJETS)

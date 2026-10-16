"""End-to-end headless run with the real SUMO binaries (pip install eclipse-sumo).
Marked `sumo`: skipped automatically when SUMO is missing (see conftest.py), and
deselectable with `pytest -m "not sumo"`. The Parts 5-6 hook tests at the end use the
fake TraCI of test_session.py instead."""

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_session import CFG, FakeTraci  # the fake TraCI module (no SUMO)

import traffic_twin.sim.run_scenario as rs
import traffic_twin.sim.session as session_mod
from traffic_twin.control.fixed import FixedTimeController
from traffic_twin.sim.session import SimulationClosed

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def short_scenario(scenario_cfg, tmp_path, monkeypatch):
    """A 10-minute copy of the base scenario, run from a scratch copy of the repo's
    configs and network so nothing under sumo/ or data/ is touched."""
    scenario_cfg["duration_s"] = 600
    scenario_cfg["emergency"]["depart_s"] = 100
    scenario_cfg["outputs"]["dir"] = "results"
    shutil.copytree(REPO_ROOT / "sumo/network", tmp_path / "sumo/network")
    path = tmp_path / "configs" / "scenario_short.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(scenario_cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return path


@pytest.mark.sumo
def test_short_headless_run(short_scenario, tmp_path, capsys):
    from traffic_twin.sim.run_scenario import main

    main(["--config", str(short_scenario), "--seed", "7"])

    out = tmp_path / "results" / "baseline_fixed_seed7"
    names = {p.name for p in out.iterdir()}
    assert names >= {
        "kpi_run.csv", "trip.parquet", "state_log.parquet", "run.json",
        "tripinfo.xml", "stops.xml", "statistics.xml",
    }  # fmt: skip
    # the grid has no detectors and no measurement section
    assert not names & {"detectors.add.xml", "detector.parquet", "queue.parquet", "signal.parquet"}
    kpis = pd.read_csv(out / "kpi_run.csv")
    assert {"car", "bus", "pedestrian", "ambulance"} <= set(kpis["user_type"])
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert (kpis["run_id"] == meta["run_id"]).all()
    assert meta["seed"] == 7 and meta["duration_s"] == 600
    assert meta["config_sources"] == ["configs/scenario_short.yaml"]
    health = meta["health"]
    assert health["teleports"] == health["collisions"] == 0 and not health["drain_timeout"]
    assert health["vehicles_waiting_end"] == 0 and health["depart_delay_max"] >= 0
    assert set(meta["safety"]["interventions"].values()) == {0}
    assert set(meta["safety"]["audit"].values()) == {0}
    counts = kpis[kpis["kpi"] == "count"].set_index("user_type")["value"]
    assert counts["car"] > 50 and counts["ambulance"] == 1
    assert (kpis["value"] >= 0).all()
    get = kpis.set_index(["user_type", "kpi"])["value"]
    assert get["pedestrian", "ped_wait_single_max"] <= get["pedestrian", "ped_wait_max"]
    assert ("car", "depart_delay_mean") in get.index and ("bus", "dwell_mean") in get.index

    state = pd.read_parquet(out / "state_log.parquet")
    assert {"intersection", "car"} <= set(state["entity"])
    assert state["t"].max() < 600  # logged during the demand window only
    assert "travel_time" in capsys.readouterr().out  # summary table
    assert (tmp_path / "sumo" / "demand" / "baseline_fixed.rou.xml").is_file()


# -- Parts 5-6 hooks (contract 4.1): Controller.bind, observers, demand_dir, cleanup ----


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """The fake TraCI; `write_demand` records the folder it is given."""
    traci = FakeTraci()
    traci.demand_dirs = []

    def write_demand(cfg, d):
        traci.demand_dirs.append(Path(d))
        return d / "r.xml", d / "a.xml"

    monkeypatch.setattr(session_mod, "traci_module", lambda gui: traci)
    monkeypatch.setattr(session_mod, "write_demand", write_demand)
    monkeypatch.setattr("traffic_twin.sim.sumo.find_binary", lambda name: name)
    return traci


class Recorder:
    """A RunObserver logging its calls into a shared list."""

    def __init__(self, name: str, log: list):
        self.name, self.log = name, log

    def bind(self, sim):
        self.log.append((self.name, "bind", sim.t))

    def after_step(self, sim):
        self.log.append((self.name, "after_step", sim.t))

    def finish(self, sim, out_dir, run_id):
        assert (out_dir / "run.json").is_file() and (out_dir / "kpi_run.csv").is_file()
        self.log.append((self.name, "finish", run_id))


class BindingController(FixedTimeController):
    def __init__(self, log: list):
        super().__init__([6, 4])
        self.log = log

    def bind(self, sim):
        assert sim.plan is not None  # the session has started
        self.log.append(("ctrl", "bind", sim.t))

    def reset(self, seed):
        self.log.append(("ctrl", "reset", seed))
        super().reset(seed)

    def decide(self, obs):
        self.log.append(("ctrl", "decide", obs.t))
        return super().decide(obs)


def test_observers_and_bind_are_called_in_the_documented_order(
    fake, sumo_outputs, monkeypatch, tmp_path
):
    fake.tripinfo_text = sumo_outputs[0].read_text(encoding="utf-8")
    log: list = []
    monkeypatch.setattr(rs, "make_controller", lambda cfg: BindingController(log))
    observers = [Recorder("a", log), Recorder("b", log)]
    meta = rs.run(CFG, 1, tmp_path, demand_dir=tmp_path / "demand", observers=observers)
    n = fake.steps  # 20 steps: the demand window, the fake network is empty at its end
    assert n == 20
    assert log[:4] == [("ctrl", "bind", 0.0), ("ctrl", "reset", 1), ("a", "bind", 0.0),
                       ("b", "bind", 0.0)]  # fmt: skip
    body = log[4:-2]
    assert len(body) == 3 * n
    for k in range(n):  # decide at t, then each observer after the step (t + 1)
        assert body[3 * k : 3 * k + 3] == [
            ("ctrl", "decide", float(k)), ("a", "after_step", k + 1.0), ("b", "after_step", k + 1.0)
        ]  # fmt: skip
    assert log[-2:] == [("a", "finish", meta["run_id"]), ("b", "finish", meta["run_id"])]


def test_default_controller_bind_is_a_no_op(fake, sumo_outputs, tmp_path):
    fake.tripinfo_text = sumo_outputs[0].read_text(encoding="utf-8")
    assert FixedTimeController([6, 4]).bind(object()) is None
    rs.run(CFG, 1, tmp_path / "res", demand_dir=tmp_path / "demand")
    assert (tmp_path / "res" / "fake_seed1" / "kpi_run.csv").is_file()


def test_demand_dir_is_honoured(fake, sumo_outputs, tmp_path):
    fake.tripinfo_text = sumo_outputs[0].read_text(encoding="utf-8")
    rs.run(CFG, 1, tmp_path / "res", demand_dir=tmp_path / "my_demand")
    assert fake.demand_dirs == [tmp_path / "my_demand"]
    assert fake.cmd[fake.cmd.index("-r") + 1] == str(tmp_path / "my_demand" / "r.xml")


def test_demand_dir_defaults_to_sumo_demand(fake, sumo_outputs, tmp_path):
    from traffic_twin.paths import DEMAND_DIR

    fake.tripinfo_text = sumo_outputs[0].read_text(encoding="utf-8")
    rs.run(CFG, 1, tmp_path / "res")  # the fake write_demand writes nothing
    assert fake.demand_dirs == [DEMAND_DIR]


def test_cleanup_after_an_early_stop_keeps_files_this_run_did_not_write(fake, tmp_path):
    """A run into an existing run folder that SUMO leaves early removes only the raw
    outputs it created or rewrote; the earlier results and the folder stay."""
    out = tmp_path / "fake_seed1"
    out.mkdir()
    old = {"stops.xml": "<old stops/>", "e1.xml": "<old e1/>", "kpi_run.csv": "old"}
    for name, text in old.items():
        (out / name).write_text(text)
        os.utime(out / name, (1_000_000_000, 1_000_000_000))  # 2001
    fake.close_at_step = 3
    with pytest.raises(SimulationClosed):
        rs.run(CFG, 1, tmp_path, demand_dir=tmp_path / "demand")
    # the fake SUMO wrote tripinfo.xml and statistics.xml: partial, removed
    assert sorted(p.name for p in out.iterdir()) == sorted(old)
    assert {p.name: p.read_text() for p in out.iterdir()} == old


def test_cleanup_removes_a_raw_output_this_run_rewrote(fake, tmp_path):
    out = tmp_path / "fake_seed1"
    out.mkdir()
    (out / "tripinfo.xml").write_text("<complete earlier run/>")
    os.utime(out / "tripinfo.xml", (1_000_000_000, 1_000_000_000))
    fake.close_at_step = 3
    with pytest.raises(SimulationClosed):
        rs.run(CFG, 1, tmp_path, demand_dir=tmp_path / "demand")
    assert out.is_dir() and not (out / "tripinfo.xml").exists()  # SUMO had truncated it

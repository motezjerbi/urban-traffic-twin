"""SimSession stepping and GUI-close handling against a fake TraCI module."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import traffic_twin.sim.run_scenario as rs
import traffic_twin.sim.session as session_mod
from traffic_twin.config import ConfigError
from traffic_twin.control.fixed import FixedTimeController
from traffic_twin.sim.session import SimSession, SimulationClosed

# phases: 0 green A, 1 yellow A, 2 all-red, 3 green B, 4 yellow B
DURATIONS = [30, 3, 2, 30, 3]
CFG = {
    "scenario": "fake",
    "seed": 1,
    "begin_s": 100,
    "duration_s": 20,
    "controller": "fixed",
    "network": {
        "net_file": "unused.net.xml",
        "tls_id": "C",
        "stages": [
            {"name": "A", "green_phase": 0, "clearance_phases": [1, 2]},
            {"name": "B", "green_phase": 3, "clearance_phases": [4]},
        ],
        "approaches": {"W": "w_in", "N": "n_in"},
    },
    "fixed": {"green_s": {"A": 6, "B": 4}},
    "outputs": {"dir": "unused", "state_log_every_s": 5},
}


class FakeTraci:
    class FatalTraCIError(Exception):
        pass

    def __init__(self):
        self.time, self.steps, self.shown = 100.0, 0, []
        self.close_at_step: int | None = None
        self.pending, self.connected = 0, False
        self.tripinfo_text = "<partial"  # what SUMO has written when the connection opens
        logic = SimpleNamespace(phases=[SimpleNamespace(duration=d) for d in DURATIONS])
        self.trafficlight = SimpleNamespace(
            getAllProgramLogics=lambda tls: [logic],
            setPhase=lambda tls, p: self.shown.append(p),
            setPhaseDuration=lambda tls, d: None,
        )
        self.edge = SimpleNamespace(getLastStepHaltingNumber=lambda e: 2)
        self.simulation = SimpleNamespace(
            getMinExpectedNumber=lambda: self.pending, getTime=lambda: self.time
        )
        self.vehicle = SimpleNamespace(
            getIDList=lambda: ["v0"],
            getPosition=lambda v: (1.0, 2.0),
            getTypeID=lambda v: "bus",
            getSpeed=lambda v: 3.0,
        )
        self.person = SimpleNamespace(getIDList=list)

    def start(self, cmd):
        self.connected = True
        self.cmd = cmd
        Path(cmd[cmd.index("--tripinfo-output") + 1]).write_text(self.tripinfo_text)

    def simulationStep(self):
        if not self.connected:
            raise self.FatalTraCIError("Not connected.")
        if self.steps == self.close_at_step:
            self.connected = False
            raise self.FatalTraCIError("Connection closed by SUMO.")
        self.steps += 1
        self.time += 1.0

    def close(self):
        if not self.connected:
            raise self.FatalTraCIError("Not connected.")
        self.connected = False


@pytest.fixture
def fake(monkeypatch):
    traci = FakeTraci()
    monkeypatch.setattr(session_mod, "traci_module", lambda gui: traci)
    monkeypatch.setattr(session_mod, "write_demand", lambda cfg, d: (d / "r.xml", d / "a.xml"))
    monkeypatch.setattr("traffic_twin.sim.sumo.find_binary", lambda name: name)
    return traci


def test_controller_loop_runs_demand_window(fake, tmp_path):
    ctrl = FixedTimeController([6, 4])
    with SimSession(CFG, 1, tmp_path) as sim:
        assert sim.observe().queues == {"W": 2, "N": 2}
        while not sim.done:
            sim.step(ctrl.decide(sim.observe()))
    assert fake.steps == 20 and not fake.connected
    # initial phase, then 6 s green A, 5 s clearance, 4 s green B, 3 s clearance, green A
    assert fake.shown == [0] + [0] * 6 + [1, 1, 1, 2, 2] + [3] * 4 + [4] * 3 + [0, 0]
    inter = [r for r in sim.state_rows if r["entity"] == "intersection"]
    assert [r["t"] for r in inter] == [0, 5, 10, 15]
    assert inter[0]["queue"] == 4
    assert {r["entity"] for r in sim.state_rows} == {"intersection", "bus"}


def test_observation_is_cached_per_step(fake, tmp_path):
    with SimSession(CFG, 1, tmp_path, log_state=False) as sim:
        first = sim.observe()
        assert sim.observe() is first
        sim.step(0)
        assert sim.observe() is not first and sim.observe().t == 1.0
    assert sim.state_rows == []


def test_drain_continues_while_vehicles_remain(fake, tmp_path):
    fake.pending = 1
    with SimSession(CFG, 1, tmp_path, log_state=False) as sim:
        for _ in range(20):
            sim.step(0)
        assert not sim.done
        fake.pending = 0
        assert sim.done


def test_gui_close_raises_simulation_closed(fake, tmp_path):
    fake.close_at_step = 3
    with pytest.raises(SimulationClosed), SimSession(CFG, 1, tmp_path) as sim:
        while not sim.done:
            sim.step(0)
    assert not fake.connected


def test_run_writes_nothing_when_sumo_closes(fake, tmp_path):
    fake.close_at_step = 3
    with pytest.raises(SimulationClosed):
        rs.run(CFG, 1, tmp_path)
    assert not (tmp_path / "fake_seed1").exists()  # partial tripinfo removed


def test_cli_exits_cleanly_when_sumo_closes(monkeypatch, caplog):
    def closed(*a, **kw):
        raise SimulationClosed("SUMO stopped before the end")

    monkeypatch.setattr(rs, "load_scenario", lambda path: CFG)
    monkeypatch.setattr(rs, "run", closed)
    with pytest.raises(SystemExit) as exc:
        rs.main(["--seed", "1", "--gui"])
    assert exc.value.code == 1
    assert "No results written" in caplog.text


def test_sumo_command_uses_scenario_window(fake, tmp_path):
    with SimSession(CFG, 9, tmp_path, log_state=False):
        cmd = fake.cmd
    assert cmd[cmd.index("--seed") + 1] == "9"
    assert cmd[cmd.index("--begin") + 1] == "100"
    assert float(cmd[cmd.index("--end") + 1]) > 120  # demand window plus drain time
    assert cmd[cmd.index("--tripinfo-output") + 1] == str(tmp_path / "tripinfo.xml")


def test_step_after_close_raises(fake, tmp_path):
    sim = SimSession(CFG, 1, tmp_path, log_state=False).start()
    sim.close()
    sim.close()  # idempotent
    with pytest.raises(SimulationClosed):
        sim.step(0)


def test_run_and_cli_write_results(fake, sumo_outputs, tmp_path, monkeypatch, capsys):
    fake.tripinfo_text = sumo_outputs[0].read_text(encoding="utf-8")
    meta = rs.run(CFG, 1, tmp_path)
    out = tmp_path / "fake_seed1"
    assert {p.name for p in out.iterdir()} >= {"kpi_run.csv", "run.json", "trip.parquet"}
    assert meta["scenario"] == "fake" and meta["duration_s"] == 20

    cfg = {**CFG, "outputs": {**CFG["outputs"], "dir": str(tmp_path / "cli")}}
    monkeypatch.setattr(rs, "load_scenario", lambda path: cfg)
    fake.time, fake.steps = 100.0, 0
    rs.main(["--seed", "1"])
    assert (tmp_path / "cli" / "fake_seed1" / "kpi_run.csv").is_file()
    assert "time_loss_mean" in capsys.readouterr().out


def test_cli_reports_config_errors(monkeypatch):
    def bad(path):
        raise ConfigError("missing keys network.tls_id")

    monkeypatch.setattr(rs, "load_scenario", bad)
    with pytest.raises(SystemExit, match="config error: missing keys"):
        rs.main([])


def test_run_with_no_finished_trip(fake, tmp_path):
    fake.tripinfo_text = "<tripinfos/>"
    rs.run(CFG, 1, tmp_path)

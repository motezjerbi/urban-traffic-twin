"""SimSession stepping, observation extras, guard, measurements and GUI-close handling
against a fake TraCI module."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import traffic_twin.sim.run_scenario as rs
import traffic_twin.sim.session as session_mod
from traffic_twin.config import ConfigError
from traffic_twin.control.base import EXTRA_CALLS, EXTRA_GAP_S
from traffic_twin.control.fixed import FixedTimeController
from traffic_twin.control.safety import INTERVENTION_KEYS
from traffic_twin.sim.audit import AUDIT_KEYS
from traffic_twin.sim.detectors import Area, DetectorLayout, Loop
from traffic_twin.sim.results import HEALTH_KEYS
from traffic_twin.sim.session import SimSession, SimulationClosed
from traffic_twin.sim.sumo import sumo_command

# phases: 0 green A, 1 yellow A, 2 all-red, 3 green B, 4 yellow B (links: a_0, b_0)
DURATIONS = [30, 3, 2, 30, 3]
STATES = ["Gr", "yr", "rr", "rG", "ry"]
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

# A Pitié-like variant: multi-edge approaches, control loops and a measurement section
LAYOUT = DetectorLayout(
    loops=(
        Loop("det_a", "a_0", 10.0, "control"),
        Loop("det_b", "b_0", 10.0, "control"),
        Loop("det_c", "c_0", 10.0, "control"),
        Loop("e1_S_s_0", "s_0", 5.0, "station"),
    ),
    stage_loops=(("det_a",), ("det_b", "det_c")),
    stations={"S": ("e1_S_s_0",)},
    queues={"N": ("q_N_n_0", "q_N_n_1")},
    areas=(Area("q_N_n_0", "n_0", 1.0, 50.0), Area("q_N_n_1", "n_1", 1.0, 50.0)),
    switch_links={("a_0", "x_0"): 0, ("b_0", "y_0"): 1},
)
MEAS_CFG = copy.deepcopy(CFG)
MEAS_CFG["network"]["approaches"] = {
    "N": {"in_edges": ["n_in", "n2_in"], "out_edges": ["n_out"]},
    "W": {"in_edges": ["w_in"], "out_edges": ["w_out"]},
}
MEAS_CFG["detectors"] = {"control": {"distance_m": None}}
MEAS_CFG["measurement"] = {
    "period_s": 900,
    "stations": [{"id": "S", "lanes": ["s_0"], "pos_m": 5.0}],
    "queues": {"N": {"lanes": ["n_0", "n_1"], "length_m": 50}},
    "tls_switch": True,
}
E1 = """<detector>
  <interval begin="100.00" end="110.00" id="e1_S_s_0" nVehContrib="3" flow="1080.00"
            occupancy="5.00" speed="10.00" harmonicMeanSpeed="9.00" length="5.00"/>
  <interval begin="110.00" end="120.00" id="e1_S_s_0" nVehContrib="0" flow="0.00"
            occupancy="0.00" speed="-1.00" harmonicMeanSpeed="-1.00" length="-1.00"/>
</detector>"""
SWITCHES = """<tlsSwitches>
  <tlsSwitch id="C" programID="0" fromLane="a_0" toLane="x_0" begin="100.00" end="106.00"
             duration="6.00"/>
  <tlsSwitch id="C" programID="0" fromLane="b_0" toLane="y_0" begin="111.00" end="115.00"
             duration="4.00"/>
</tlsSwitches>"""
STATISTICS = """<statistics>
  <vehicles loaded="3" inserted="3" running="0" waiting="0"/>
  <teleports total="0" jam="0" yield="0" wrongLane="0"/>
  <safety collisions="0" emergencyStops="0" emergencyBraking="0"/>
  <persons loaded="1" running="0" jammed="0"/>
</statistics>"""


class FakeTraci:
    class FatalTraCIError(Exception):
        pass

    def __init__(self):
        self.time, self.steps, self.shown = 100.0, 0, []
        self.close_at_step: int | None = None
        self.pending, self.connected = 0, False
        self.tripinfo_text = "<partial"  # what SUMO has written when the connection opens
        self.halting = {"w_in": 2, "n_in": 2, "n2_in": 3}
        self.gap = {"det_a": 0.5, "det_b": 4.0, "det_c": 2.5}  # time since detection
        self.hits: dict[int, set[str]] = {}  # step -> loops hit during that step
        self.waits: dict[int, dict[str, float]] = {}  # step -> person -> waiting time
        self.persons: list[str] = []
        logic = SimpleNamespace(
            phases=[
                SimpleNamespace(duration=d, state=s) for d, s in zip(DURATIONS, STATES, strict=True)
            ]
        )
        self.trafficlight = SimpleNamespace(
            getAllProgramLogics=lambda tls: [logic],
            setPhase=lambda tls, p: self.shown.append(p),
            setPhaseDuration=lambda tls, d: None,
        )
        self.edge = SimpleNamespace(getLastStepHaltingNumber=lambda e: self.halting[e])
        self.simulation = SimpleNamespace(
            getMinExpectedNumber=lambda: self.pending, getTime=lambda: self.time
        )
        self.vehicle = SimpleNamespace(
            getIDList=lambda: ["v0"],
            getPosition=lambda v: (1.0, 2.0),
            getTypeID=lambda v: "bus",
            getSpeed=lambda v: 3.0,
        )
        self.person = SimpleNamespace(
            getIDList=lambda: list(self.persons),
            getWaitingTime=lambda p: self.waits.get(self.steps, {}).get(p, 0.0),
            getPosition=lambda p: (5.0, 6.0),
            getSpeed=lambda p: 1.2,
        )
        self.inductionloop = SimpleNamespace(
            getTimeSinceDetection=lambda d: self.gap[d],
            getLastStepVehicleNumber=lambda d: int(d in self.hits.get(self.steps, ())),
        )
        self.lanearea = SimpleNamespace(
            getJamLengthVehicle=lambda d: 2,
            getJamLengthMeters=lambda d: 15.0,
            getLastStepHaltingNumber=lambda d: 3,
        )

    def start(self, cmd):
        self.connected = True
        self.cmd = cmd

        def out(option: str) -> Path:
            return Path(cmd[cmd.index(option) + 1])

        out("--tripinfo-output").write_text(self.tripinfo_text)
        stats = out("--statistic-output")
        stats.write_text(STATISTICS)
        if "detectors.add.xml" in cmd[cmd.index("-a") + 1]:  # SUMO's detector outputs
            (stats.parent / "e1.xml").write_text(E1)
            (stats.parent / "tls_switch.xml").write_text(SWITCHES)

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


@pytest.fixture
def detectors(fake, monkeypatch):
    """write_detectors returns LAYOUT and writes a placeholder detector file."""

    def write(cfg, out_dir):
        path = Path(out_dir) / "detectors.add.xml"
        path.write_text("<additional/>")
        return path, LAYOUT

    monkeypatch.setattr(session_mod, "write_detectors", write)
    return fake


def test_controller_loop_runs_demand_window(fake, tmp_path):
    ctrl = FixedTimeController([6, 4])
    with SimSession(CFG, 1, tmp_path) as sim:
        assert sim.observe().queues == {"W": 2, "N": 2}
        assert sim.observe().extras == {}  # no control loops
        while not sim.done:
            sim.step(ctrl.decide(sim.observe()))
    assert fake.steps == 20 and not fake.connected
    # initial phase, then 6 s green A, 5 s clearance, 4 s green B, 3 s clearance, green A
    assert fake.shown == [0] + [0] * 6 + [1, 1, 1, 2, 2] + [3] * 4 + [4] * 3 + [0, 0]
    assert sim.phases_shown == fake.shown[1:]  # one per step, for the signal audit
    assert sim.phase_states == STATES
    inter = [r for r in sim.state_rows if r["entity"] == "intersection"]
    assert [r["t"] for r in inter] == [0, 5, 10, 15]
    assert inter[0]["queue"] == 4
    assert {r["entity"] for r in sim.state_rows} == {"intersection", "bus"}
    assert sim.queue_rows == [] and not sim.drain_timeout


def test_observation_is_cached_per_step(fake, tmp_path):
    with SimSession(CFG, 1, tmp_path, log_state=False) as sim:
        first = sim.observe()
        assert sim.observe() is first
        sim.step(0)
        assert sim.observe() is not first and sim.observe().t == 1.0
    assert sim.state_rows == []


def test_queues_sum_over_in_edges(detectors, tmp_path):
    with SimSession(MEAS_CFG, 1, tmp_path, log_state=False) as sim:
        assert sim.approaches == {"N": ["n_in", "n2_in"], "W": ["w_in"]}
        assert sim.observe().queues == {"N": 5, "W": 2}


def test_gap_and_calls_per_stage(detectors, tmp_path):
    detectors.hits = {0: {"det_b"}, 1: {"det_a"}}  # before step 1, before step 2
    with SimSession(MEAS_CFG, 1, tmp_path, log_state=False) as sim:
        obs = sim.observe()
        assert obs.extras[EXTRA_GAP_S] == (0.5, 2.5)  # min over the stage's loops
        assert obs.extras[EXTRA_CALLS] == (False, True)  # A green: no call for A
        sim.step(0)
        assert sim.observe().extras[EXTRA_CALLS] == (False, True)  # A is green
        for _ in range(6):  # A holds its 6 s, then the 5 s clearance into B
            sim.step(0)
        sim.step(1)
        assert sim.observe().in_transition
        assert sim.observe().extras[EXTRA_CALLS] == (False, True)  # call kept
        while sim.observe().in_transition:
            sim.step(1)
        assert sim.observe().stage == 1
        assert sim.observe().extras[EXTRA_CALLS] == (False, False)  # reset when B is green


def test_calls_latch_while_red(detectors, tmp_path):
    detectors.hits = {3: {"det_a"}}  # A is green then: no call
    with SimSession(MEAS_CFG, 1, tmp_path, log_state=False) as sim:
        for _ in range(6):
            sim.step(0)
        while sim.observe().stage == 0 or sim.observe().in_transition:
            sim.step(1)
        assert sim.observe().extras[EXTRA_CALLS] == (False, False)
        sim.step(1)
        detectors.hits = {detectors.steps: {"det_a"}}  # a vehicle on A's loop, A red
        assert sim.observe().extras[EXTRA_CALLS] == (True, False)
        sim.step(1)
        assert sim.observe().extras[EXTRA_CALLS] == (True, False)  # kept until A is green


def test_gap_is_inf_for_a_stage_without_loops(detectors, monkeypatch, tmp_path):
    layout = DetectorLayout(LAYOUT.loops[:1], (("det_a",), ()), {}, {})
    monkeypatch.setattr(session_mod, "write_detectors", lambda cfg, d: (None, layout))
    with SimSession(MEAS_CFG, 1, tmp_path, log_state=False) as sim:
        assert sim.observe().extras[EXTRA_GAP_S] == (0.5, math.inf)


def test_guard_is_called_once_per_step_and_wins(fake, monkeypatch, tmp_path):
    calls = []

    class StickyGuard:
        interventions: dict = {}

        @classmethod
        def from_config(cls, cfg, stages, phase_states):
            assert [s.name for s in stages] == ["A", "B"] and phase_states == STATES
            return cls()

        def apply(self, proposed, state):
            calls.append((proposed, state.t, state.stage))
            return 0  # always keep stage A

    monkeypatch.setattr(session_mod, "SafetyGuard", StickyGuard)
    with SimSession(CFG, 1, tmp_path, log_state=False) as sim:
        for _ in range(12):
            sim.step(1)
    assert len(calls) == 12
    assert [c[1] for c in calls] == list(range(12))  # state at the requesting step
    assert set(sim.phases_shown) == {0}  # the guard's stage was applied


def test_additional_files_list_stops_and_detectors(detectors, tmp_path):
    with SimSession(MEAS_CFG, 1, tmp_path, log_state=False) as sim:
        cmd = detectors.cmd
    adds = cmd[cmd.index("-a") + 1].split(",")
    assert adds == [str(sim.demand_dir / "a.xml"), str(tmp_path / "detectors.add.xml")]
    assert cmd[cmd.index("--statistic-output") + 1] == str(tmp_path / "statistics.xml")


def test_grid_config_has_no_detector_file(fake, tmp_path):
    with SimSession(CFG, 3, tmp_path, log_state=False) as sim:
        cmd = fake.cmd
        assert sim.detector_file is None and sim.layout.empty
    assert not (tmp_path / "detectors.add.xml").exists()
    legacy = sumo_command(CFG, 3, sim.routes, sim.demand_dir / "a.xml", sim.tripinfo,
                          sim.stops, False)  # fmt: skip
    i = cmd.index("--statistic-output")
    assert cmd[i + 1] == str(tmp_path / "statistics.xml")
    assert cmd[:i] + cmd[i + 2 :] == legacy  # unchanged apart from the statistics


def test_queue_samples_follow_the_state_log_ticks(detectors, tmp_path):
    with SimSession(MEAS_CFG, 1, tmp_path, log_state=False) as sim:
        while not sim.done:
            sim.step(0)
    assert [r["t"] for r in sim.queue_rows] == [0, 5, 10, 15]  # even without the state log
    assert sim.queue_rows[0] == {"t": 0, "approach": "N", "jam_veh": 4, "jam_m": 30.0,
                                 "halting_veh": 6}  # fmt: skip
    assert sim.state_rows == []


def test_ped_single_wait_is_the_max_over_steps(fake, tmp_path):
    fake.persons = ["p0", "p1"]
    fake.waits = {1: {"p0": 1.0}, 2: {"p0": 2.0}, 3: {"p0": 0.0, "p1": 1.0}, 4: {"p0": 1.0}}
    with SimSession(CFG, 1, tmp_path, log_state=False) as sim:
        for _ in range(5):
            sim.step(0)
    assert sim.ped_wait_max == {"p0": 2.0, "p1": 1.0}


def test_drain_continues_while_vehicles_remain(fake, tmp_path):
    fake.pending = 1
    with SimSession(CFG, 1, tmp_path, log_state=False) as sim:
        for _ in range(20):
            sim.step(0)
        assert not sim.done
        fake.pending = 0
        assert sim.done
    assert not sim.drain_timeout


def test_drain_timeout_is_flagged(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "DRAIN_S", 10)
    fake.pending = 1
    with SimSession(CFG, 1, tmp_path, log_state=False) as sim:
        while not sim.done:
            sim.step(0)
    assert fake.steps == 30 and sim.drain_timeout


def test_gui_close_raises_simulation_closed(fake, tmp_path):
    fake.close_at_step = 3
    with pytest.raises(SimulationClosed), SimSession(CFG, 1, tmp_path) as sim:
        while not sim.done:
            sim.step(0)
    assert not fake.connected


@pytest.mark.parametrize(("cfg", "traci"), [(CFG, "fake"), (MEAS_CFG, "detectors")])
def test_run_writes_nothing_when_sumo_closes(request, tmp_path, cfg, traci):
    request.getfixturevalue(traci).close_at_step = 3
    with pytest.raises(SimulationClosed):
        rs.run(cfg, 1, tmp_path)
    assert not (tmp_path / "fake_seed1").exists()  # partial outputs and detector file removed


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


def test_sumo_command_options_and_gui_settings(monkeypatch):
    monkeypatch.setattr("traffic_twin.sim.sumo.find_binary", lambda name: name)
    cfg = copy.deepcopy(CFG)
    cfg["sumo"] = {"options": ["--collision.check-junctions", "true"]}
    cfg["network"]["gui_settings"] = "sumo/gui/pitie_view.xml"
    p = Path("x")
    headless = sumo_command(cfg, 1, p, [p, Path("y")], p, p, False, statistics=Path("s.xml"))
    assert headless[headless.index("-a") + 1] == "x,y"
    assert headless[-4:] == ["--statistic-output", "s.xml", "--collision.check-junctions", "true"]
    gui = sumo_command(cfg, 1, p, p, p, p, True)
    assert gui[gui.index("--gui-settings-file") + 1] == "sumo/gui/pitie_view.xml"
    assert gui.index("--collision.check-junctions") < gui.index("--gui-settings-file")


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
    assert not (out / "detector.parquet").exists()  # no measurement section
    assert meta["scenario"] == "fake" and meta["duration_s"] == 20

    cfg = {**CFG, "outputs": {**CFG["outputs"], "dir": str(tmp_path / "cli")}}
    monkeypatch.setattr(rs, "load_scenario", lambda path: cfg)
    fake.time, fake.steps = 100.0, 0
    rs.main(["--seed", "1"])
    assert (tmp_path / "cli" / "fake_seed1" / "kpi_run.csv").is_file()
    assert "time_loss_mean" in capsys.readouterr().out
    meta = json.loads((tmp_path / "cli" / "fake_seed1" / "run.json").read_text(encoding="utf-8"))
    assert meta["config_sources"] == ["configs/scenario_base.yaml"]  # the CLI's --config


def test_run_json_has_hashes_health_and_safety(fake, sumo_outputs, tmp_path):
    fake.tripinfo_text = sumo_outputs[0].read_text(encoding="utf-8")
    fake.persons, fake.waits = ["p0"], {2: {"p0": 7.0}}
    rs.run(CFG, 1, tmp_path)
    out = tmp_path / "fake_seed1"
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert len(meta["config_hash"]) == 64 and len(meta["demand_hash"]) == 64
    assert meta["config_sources"] is None  # run() without the config path
    health = meta["health"]
    assert list(health)[: len(HEALTH_KEYS)] == list(HEALTH_KEYS)
    assert health["teleports"] == health["collisions"] == 0 and not health["drain_timeout"]
    assert health["depart_delay_mean"] is None  # fixture trips carry no departDelay
    assert meta["safety"]["interventions"] == dict.fromkeys(INTERVENTION_KEYS, 0)
    assert meta["safety"]["audit"] == dict.fromkeys(AUDIT_KEYS, 0)
    kpis = pd.read_csv(out / "kpi_run.csv").set_index(["user_type", "kpi"])["value"]
    assert kpis["pedestrian", "ped_wait_single_max"] == 7.0


def test_run_writes_measurement_tables(detectors, sumo_outputs, tmp_path):
    detectors.tripinfo_text = sumo_outputs[0].read_text(encoding="utf-8")
    rs.run(MEAS_CFG, 1, tmp_path)
    out = tmp_path / "fake_seed1"
    names = {p.name for p in out.iterdir()}
    assert names >= {"kpi_run.csv", "trip.parquet", "state_log.parquet", "run.json",
                     "detector.parquet", "queue.parquet", "signal.parquet",
                     "statistics.xml", "detectors.add.xml", "e1.xml", "tls_switch.xml"}  # fmt: skip
    det = pd.read_parquet(out / "detector.parquet")
    assert det["begin_s"].tolist() == [0.0, 10.0]  # relative to the scenario start
    assert det["n_veh"].tolist() == [3, 0] and det["station"].unique().tolist() == ["S"]
    assert pd.isna(det["mean_speed"].iloc[1])
    queues = pd.read_parquet(out / "queue.parquet")
    assert queues["t"].tolist() == [0, 5, 10, 15] and set(queues["approach"]) == {"N"}
    signal = pd.read_parquet(out / "signal.parquet")
    assert signal[["link", "begin_s", "duration_s"]].values.tolist() == [[0, 0, 6], [1, 11, 4]]


def test_cli_reports_config_errors(monkeypatch):
    def bad(path):
        raise ConfigError("missing keys network.tls_id")

    monkeypatch.setattr(rs, "load_scenario", bad)
    with pytest.raises(SystemExit, match="config error: missing keys"):
        rs.main([])


def test_run_with_no_finished_trip(fake, tmp_path):
    fake.tripinfo_text = "<tripinfos/>"
    rs.run(CFG, 1, tmp_path)

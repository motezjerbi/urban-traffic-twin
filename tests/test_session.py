"""SimSession stepping, observation extras, guard, measurements and GUI-close handling
against a fake TraCI module."""

import copy
import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import traffic_twin.sim.run_scenario as rs
import traffic_twin.sim.session as session_mod
from traffic_twin.config import ConfigError
from traffic_twin.control.base import (
    EXTRA_BUS_CALLS,
    EXTRA_CALLS,
    EXTRA_EMERGENCY,
    EXTRA_EXIT_BLOCKED,
    EXTRA_GAP_S,
    EXTRA_GAP_S_OPEN,
    EXTRA_PED_COUNT,
    EXTRA_PED_SLACK,
    EXTRA_PED_WAIT,
    EXTRA_PREEMPT,
    Controller,
    EmergencyCall,
)
from traffic_twin.control.fixed import FixedTimeController
from traffic_twin.control.safety import INTERVENTION_KEYS
from traffic_twin.sim.audit import AUDIT_KEYS
from traffic_twin.sim.detectors import Area, DetectorLayout, Loop
from traffic_twin.sim.kpi import KPI_VERSION
from traffic_twin.sim.results import EVENT_COLUMNS, HEALTH_KEYS, PREEMPTION_REPORT_KEYS
from traffic_twin.sim.session import SimSession, SimulationClosed
from traffic_twin.sim.signal import SignalState
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

    def __init__(self, durations=DURATIONS, states=STATES):
        self.time, self.steps, self.shown = 100.0, 0, []
        self.close_at_step: int | None = None
        self.pending, self.connected = 0, False
        self.tripinfo_text = "<partial"  # what SUMO has written when the connection opens
        self.halting = {"w_in": 2, "n_in": 2, "n2_in": 3}
        self.gap = {"det_a": 0.5, "det_b": 4.0, "det_c": 2.5}  # time since detection
        self.hits: dict[int, set[str]] = {}  # step -> loops hit during that step
        self.waits: dict[int, dict[str, float]] = {}  # step -> person -> waiting time
        self.persons: list[str] = []
        # Part 4: person -> next edge / current edge, TLS links, vehicles per lane
        self.next_edge: dict[str, str] = {}
        self.next_edge_calls = 0
        self.road: dict[str, str] = {}
        self.links: list = [[("a_0", "x_0", ":C_0_0")], [("b_0", "y_0", ":C_1_0")]]
        self.lane_vehicles: dict[str, list[str]] = {}  # lane -> vehicles (exit lanes, box)
        self.veh_pos: dict[str, float] = {}
        self.veh_speed: dict[str, float] = {}
        logic = SimpleNamespace(
            phases=[
                SimpleNamespace(duration=d, state=s) for d, s in zip(durations, states, strict=True)
            ]
        )
        self.trafficlight = SimpleNamespace(
            getAllProgramLogics=lambda tls: [logic],
            setPhase=lambda tls, p: self.shown.append(p),
            setPhaseDuration=lambda tls, d: None,
            getControlledLinks=lambda tls: self.links,
        )
        self.edge = SimpleNamespace(getLastStepHaltingNumber=lambda e: self.halting[e])
        self.simulation = SimpleNamespace(
            getMinExpectedNumber=lambda: self.pending,
            getTime=lambda: self.time,
            getPendingVehicles=list,
        )
        self.vehicle = SimpleNamespace(
            getIDList=lambda: ["v0"],
            getPosition=lambda v: (1.0, 2.0),
            getTypeID=lambda v: "bus",
            getSpeed=lambda v: self.veh_speed.get(v, 3.0),
            getLanePosition=lambda v: self.veh_pos.get(v, 50.0),
            getNextTLS=lambda v: [],
            getStops=lambda v, n=1: [],
        )
        self.lane = SimpleNamespace(
            getLinks=lambda lane: [],
            getLastStepVehicleIDs=lambda lane: list(self.lane_vehicles.get(lane, [])),
        )

        def next_edge(p: str) -> str:
            self.next_edge_calls += 1
            return self.next_edge.get(p, "")

        self.person = SimpleNamespace(
            getIDList=lambda: list(self.persons),
            getWaitingTime=lambda p: self.waits.get(self.steps, {}).get(p, 0.0),
            getPosition=lambda p: (5.0, 6.0),
            getSpeed=lambda p: 1.2,
            getNextEdge=next_edge,
            getRoadID=lambda p: self.road.get(p, "w_in"),
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
        def from_config(cls, cfg, stages, phase_states, durations=None):
            assert [s.name for s in stages] == ["A", "B"] and phase_states == STATES
            assert durations == DURATIONS  # the program's, read through TraCI
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
    assert meta["safety"]["danger"] == 0 and meta["safety"]["ped_forcing"] == "off"
    assert meta["kpi_version"] == KPI_VERSION == 3
    assert meta["preemption"]["enabled"] is False
    assert {k: v for k, v in meta["preemption"].items() if k != "enabled"} == dict.fromkeys(
        (*PREEMPTION_REPORT_KEYS, "hold_s", "jumps", "ped_yield"), 0
    )
    assert meta["control"] == {}  # fixed time reports nothing
    assert not (out / "events.parquet").exists()  # no event in the run
    kpis = pd.read_csv(out / "kpi_run.csv").set_index(["user_type", "kpi"])["value"]
    assert kpis["pedestrian", "ped_wait_single_max"] == 7.0
    assert ("car", "throughput_vph") in kpis.index  # KPI version 2 rows follow


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


# -- Part 4: junction extras, guard inputs, box clearing, run outputs --------------------
# Links: 0 a_0 -> x_0 (stage A), 1 b_0 -> y_0 (B), 2 crossing X (green with A), 3 c_0 -> z_0
# (B). The crossing's vehicle foe is link 1.
P4_STATES = ["GrGr", "yrrr", "rrrr", "rGrG", "ryry"]
P4_LINKS = [
    [("a_0", "x_0", ":C_0_0")],
    [("b_0", "y_0", ":C_1_0")],
    [(":C_w0_0", ":C_c0_0", "")],
    [("c_0", "z_0", ":C_3_0")],
]
P4_FOES = {0: frozenset(), 1: frozenset({2}), 2: frozenset({1}), 3: frozenset()}
P4_CFG = copy.deepcopy(MEAS_CFG)
P4_CFG["network"]["crossings"] = {"X": {"edge": ":C_c0", "link": 2}}
PART4_KEYS = {EXTRA_GAP_S, EXTRA_CALLS, EXTRA_GAP_S_OPEN, EXTRA_PED_WAIT, EXTRA_PED_COUNT,
              EXTRA_PED_SLACK, EXTRA_EMERGENCY, EXTRA_PREEMPT, EXTRA_BUS_CALLS,
              EXTRA_EXIT_BLOCKED}  # fmt: skip
AMBULANCE = EmergencyCall("amb", 1, 1, 80.0, 0.0, False, "N")


class FakeDetector:
    """Stands for EmergencyDetector / BusDetector: records the times it is asked."""

    def __init__(self, traci, tls_id, stages, phase_states, approaches, **kw):
        self.kw, self.asked = kw, []
        self.answer: tuple = ()

    def calls(self, t):
        self.asked.append(t)
        return self.answer


class FakeManager:
    """Stands for PreemptionManager: target stage 1 while an emergency call is present."""

    def __init__(self):
        self.updates: list = []
        self.events: list = []

    def update(self, t, calls):
        self.updates.append((t, tuple(calls)))
        return 1 if calls else None

    def report(self):
        return {"requests": 1, "served": 1, "timeouts": 0, "active_s": 3.0}


class FakeShield:
    def __init__(self, mode):
        self.mode = mode
        self.states: list = []

    def slack(self, state):
        self.states.append(state)
        return tuple(88.0 - w for w in state.ped_wait)


class RecordingGuard:
    """Passes every request through and records the SignalState it was given."""

    shield = None
    allowed = [frozenset({1}), frozenset({0})]

    def __init__(self):
        self.states: list = []
        self.interventions: dict = {}

    @classmethod
    def from_config(cls, cfg, stages, phase_states, durations=None):
        return cls()

    def apply(self, proposed, state):
        self.states.append(state)
        return proposed


@pytest.fixture
def part4(detectors, monkeypatch):
    """Pitié-like session parts: a 4-link program with crossing X, recorded detectors."""
    fake = FakeTraci(states=P4_STATES)
    fake.links = P4_LINKS
    fake.gap = {"det_a": 0.5, "det_b": 1.0, "det_c": 2.5}
    monkeypatch.setattr(session_mod, "traci_module", lambda gui: fake)
    monkeypatch.setattr(session_mod, "program_foes", lambda net, tls: P4_FOES)
    found = SimpleNamespace(fake=fake, emergency=[], buses=[])

    def emergency(*a, **kw):
        found.emergency.append(FakeDetector(*a, **kw))
        return found.emergency[-1]

    def buses(*a, **kw):
        found.buses.append(FakeDetector(*a, **kw))
        return found.buses[-1]

    monkeypatch.setattr(session_mod, "EmergencyDetector", emergency)
    monkeypatch.setattr(session_mod, "BusDetector", buses)
    return found


def test_part4_extras_and_ped_waits_of_the_previous_step(part4, tmp_path):
    fake = part4.fake
    fake.persons = ["p0", "p1", "p2"]
    fake.next_edge = {"p0": ":C_c0", "p1": "w_in", "p2": ":C_c0"}
    fake.waits = {1: {"p0": 3.0, "p1": 9.0}, 2: {"p0": 4.0, "p2": 6.0}}
    with SimSession(P4_CFG, 1, tmp_path, log_state=False) as sim:
        assert sim.emergency is part4.emergency[0] and sim.buses is part4.buses[0]
        assert part4.emergency[0].kw == {"pending": True, "max_dist_m": None}  # defaults
        assert sim.preemption is None  # no `preemption.enabled`
        assert sim.foes == P4_FOES and sim.junction is not None
        obs = sim.observe()
        assert set(obs.extras) == PART4_KEYS
        assert obs.extras[EXTRA_PED_WAIT] == (0.0,) and obs.extras[EXTRA_PED_COUNT] == (0,)
        assert obs.extras[EXTRA_PED_SLACK] == (math.inf,)  # no shield
        assert obs.extras[EXTRA_PREEMPT] is None and obs.extras[EXTRA_EMERGENCY] == ()
        assert obs.extras[EXTRA_EXIT_BLOCKED] == frozenset()
        sim.step(0)
        obs = sim.observe()  # waits sampled after step 1; p1 waits elsewhere
        assert obs.extras[EXTRA_PED_WAIT] == (3.0,) and obs.extras[EXTRA_PED_COUNT] == (1,)
        sim.step(0)
        assert sim.observe().extras[EXTRA_PED_WAIT] == (6.0,)
        assert sim.observe().extras[EXTRA_PED_COUNT] == (2,)
    assert part4.emergency[0].asked == [0.0, 1.0, 2.0]  # once per step (observe is cached)
    assert part4.buses[0].asked == [0.0, 1.0, 2.0]
    assert sim.ped_wait_max == {"p0": 4.0, "p1": 9.0, "p2": 6.0}  # unchanged KPI input


def test_preemption_detection_config_reaches_the_detector(part4, tmp_path):
    cfg = copy.deepcopy(P4_CFG)
    cfg["preemption"] = {"enabled": False, "detection": {"pending": False, "max_dist_m": 150}}
    with SimSession(cfg, 1, tmp_path, log_state=False) as sim:
        assert sim.preemption is None
    assert part4.emergency[0].kw == {"pending": False, "max_dist_m": 150}


@pytest.mark.parametrize("mode", ["detected", "red", "off"])
def test_signal_state_carries_the_waits_and_the_preemption_target(part4, monkeypatch, tmp_path,
                                                                  mode):  # fmt: skip
    guard, manager, shield = RecordingGuard(), FakeManager(), FakeShield(mode)
    guard.shield = shield
    monkeypatch.setattr(session_mod.SafetyGuard, "from_config", lambda *a, **kw: guard)
    monkeypatch.setattr(session_mod.PreemptionManager, "from_config", lambda cfg: manager)
    fake = part4.fake
    fake.persons, fake.next_edge = ["p0"], {"p0": ":C_c0"}
    fake.waits = {k: {"p0": float(k)} for k in range(1, 20)}
    with SimSession(P4_CFG, 1, tmp_path, log_state=False) as sim:
        assert sim.preemption is manager
        sim.step(0)
        sim.step(0)
        part4.emergency[0].answer = (AMBULANCE,)
        obs = sim.observe()
        assert sim.observe() is obs
        assert obs.extras[EXTRA_PREEMPT] == 1 and obs.extras[EXTRA_EMERGENCY] == (AMBULANCE,)
        plain = sim.plan.state(obs.t)
        sim.step(0)
        for _ in range(3):  # A's clearance from step 3: crossing X turns red
            sim.step(1)
    want = {"detected": [(0.0,), (1.0,), (2.0,)], "red": [(0.0,), (0.0,), (0.0,)],
            "off": [(), (), ()]}[mode]  # fmt: skip
    assert [st.ped_wait for st in guard.states[:3]] == want
    assert [st.preempt_target for st in guard.states] == [None, None, 1, 1, 1, 1]
    assert dataclasses.replace(guard.states[2], ped_wait=(), preempt_target=None) == plain
    if mode == "red":
        assert [st.ped_wait for st in guard.states[3:]] == [(0.0,), (1.0,), (2.0,)]
    assert [u[0] for u in manager.updates] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # once per step
    # the shield's slack is computed from the same waits
    assert [st.ped_wait for st in shield.states[:3]] == want
    assert obs.extras[EXTRA_PED_SLACK] == tuple(88.0 - w for w in want[2])


def test_curb_wait_survives_creeps_until_the_crossing_turns_green(part4, tmp_path):
    """SUMO resets getWaitingTime when a waiting pedestrian creeps (> 0.1 m/s) at the
    curb. The shield's w_c and `ped_curb_wait_max` keep counting from the first wait at
    the red crossing (an S2 pedestrian stood 91 s at a red crossing while
    getWaitingTime peaked at 88); `ped_wait_max` stays the Part 3 getWaitingTime value."""
    fake = part4.fake
    fake.persons = ["p0", "p1"]
    fake.next_edge = {"p0": ":C_c0", "p1": ":C_c0"}
    # getWaitingTime after step k: p0 creeps after steps 3 and 6; p1 after step 4
    fake.waits = {
        1: {"p0": 1.0},
        2: {"p0": 2.0, "p1": 1.0},
        3: {"p0": 0.0, "p1": 2.0},
        4: {"p0": 1.0, "p1": 0.0},
        5: {"p0": 2.0},
        6: {"p0": 0.0, "p1": 1.0},
        7: {"p0": 1.0, "p1": 2.0},
        8: {"p0": 2.0, "p1": 3.0},
        9: {"p0": 3.0, "p1": 4.0},
        10: {"p0": 4.0, "p1": 5.0},
    }
    seen, shown = [], []
    with SimSession(P4_CFG, 1, tmp_path, log_state=False) as sim:
        for k in range(1, 13):
            if k == 5:
                fake.next_edge["p1"] = "w_in"  # p1 turns away: its curb wait ends
            if k == 6:
                fake.next_edge["p1"] = ":C_c0"  # back at the curb: a new wait starts
            if k == 12:
                fake.next_edge["p0"] = ":C_w1"  # p0 is on the crossing
            shown.append(sim.step(1 if k <= 7 else 0))
            extras = sim.observe().extras
            seen.append((extras[EXTRA_PED_WAIT][0], extras[EXTRA_PED_COUNT][0]))
    # crossing X (link 2) is red in phases 1-4 and green in phase 0
    assert shown == [1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 0, 0]
    assert [w for w, _ in seen] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0]
    assert [n for _, n in seen] == [1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 0, 0]
    assert sim._curb == {}  # green served p1, p0 left the curb
    assert sim.ped_wait_max == {"p0": 4.0, "p1": 5.0}  # Part 3 KPI input, unchanged
    # p0: 10 s at the red curb; p1: 3 s, then (after turning away) 5 s
    assert sim.ped_curb_wait_max == {"p0": 10.0, "p1": 5.0}


def test_curb_wait_counts_the_red_time_of_a_wait_begun_under_green(part4, tmp_path):
    """A person still waiting when its crossing turns red keeps its getWaitingTime: the
    curb clock starts at now - getWaitingTime, never below the Part 3 value."""
    fake = part4.fake
    fake.persons, fake.next_edge = ["p0"], {"p0": ":C_c0"}
    fake.waits = {1: {"p0": 1.0}, 2: {"p0": 2.0}, 3: {"p0": 0.0}}
    waits = []
    with SimSession(P4_CFG, 1, tmp_path, log_state=False) as sim:
        for _ in range(3):
            sim.step(0)  # X green: no curb clock, the wait is getWaitingTime
            waits.append(sim.observe().extras[EXTRA_PED_WAIT][0])
        fake.waits.update({4: {"p0": 3.0}, 5: {"p0": 0.0}, 6: {"p0": 1.0}})
        for _ in range(3):
            sim.step(1)  # X red from here: the clock starts at 104 - 3
            waits.append(sim.observe().extras[EXTRA_PED_WAIT][0])
    assert waits == [1.0, 2.0, 0.0, 3.0, 4.0, 5.0]
    assert sim.ped_wait_max == {"p0": 3.0} and sim.ped_curb_wait_max == {"p0": 5.0}


def test_grid_extras_and_guard_input_are_unchanged(detectors, monkeypatch, tmp_path):
    guard = RecordingGuard()
    monkeypatch.setattr(session_mod.SafetyGuard, "from_config", lambda *a, **kw: guard)
    detectors.persons, detectors.waits = ["p0"], {1: {"p0": 5.0}}
    with SimSession(MEAS_CFG, 1, tmp_path, log_state=False) as sim:  # no crossings
        assert set(sim.observe().extras) == {EXTRA_GAP_S, EXTRA_CALLS}
        sim.step(0)
        sim.step(0)
        assert set(sim.observe().extras) == {EXTRA_GAP_S, EXTRA_CALLS}
        assert sim.emergency is sim.buses is sim.preemption is sim.junction is sim.foes is None
    assert len(guard.states) == 2
    assert all(st.ped_wait == () and st.preempt_target is None for st in guard.states)
    assert guard.states[0] == SignalState(0.0, 0, 0.0, False, (0.0, 0.0))
    assert detectors.next_edge_calls == 0  # no per-crossing sampling on the grid
    assert sim.ped_wait_max == {"p0": 5.0} and sim.ped_curb_wait_max == {}  # no curb waits


def test_gap_s_open_ignores_loops_that_only_feed_blocked_exits(part4, tmp_path):
    fake = part4.fake
    with SimSession(P4_CFG, 1, tmp_path, log_state=False) as sim:
        extras = sim.observe().extras
        assert extras[EXTRA_GAP_S] == (0.5, 1.0) and extras[EXTRA_GAP_S_OPEN] == (0.5, 1.0)
        fake.lane_vehicles = {"y_0": ["q0"]}  # det_b's lane feeds y_0 only: now blocked
        fake.veh_pos, fake.veh_speed = {"q0": 4.0}, {"q0": 0.0}
        sim.step(0)
        extras = sim.observe().extras
        assert extras[EXTRA_EXIT_BLOCKED] == frozenset({"y_0"})
        assert extras[EXTRA_GAP_S] == (0.5, 1.0) and extras[EXTRA_GAP_S_OPEN] == (0.5, 2.5)
        fake.lane_vehicles = {"y_0": ["q0"], "z_0": ["q1"]}  # det_c's exit too
        fake.veh_pos["q1"], fake.veh_speed["q1"] = 14.0, 0.2
        sim.step(0)
        assert sim.observe().extras[EXTRA_GAP_S_OPEN] == (0.5, math.inf)
        fake.veh_pos["q1"] = 15.0  # beyond the 15 m exit zone
        sim.step(0)
        assert sim.observe().extras[EXTRA_GAP_S_OPEN] == (0.5, 2.5)


def test_without_the_monitor_no_exit_is_blocked(part4, tmp_path):
    fake = part4.fake
    fake.lane_vehicles, fake.veh_pos, fake.veh_speed = {"y_0": ["q0"]}, {"q0": 1.0}, {"q0": 0.0}
    with SimSession(P4_CFG, 1, tmp_path, log_state=False, junction_monitor=False) as sim:
        sim.step(0)
        extras = sim.observe().extras
        assert sim.junction is None and extras[EXTRA_EXIT_BLOCKED] == frozenset()
        assert extras[EXTRA_GAP_S_OPEN] == extras[EXTRA_GAP_S]


class LoggingController(Controller):
    """Fixed A/B alternation that logs one event per switch and reports a count."""

    def reset(self, seed):
        self.clear_events()
        self.switches = 0

    def decide(self, obs):
        if obs.in_transition or obs.time_in_stage < 6:
            return obs.stage
        self.switches += 1
        self.log_event(obs.t, "switch", obs.stage, 1 - obs.stage, "test")
        return 1 - obs.stage

    def report(self):
        return {"switches": self.switches}


def test_run_writes_part4_outputs(part4, sumo_outputs, monkeypatch, tmp_path):
    part4.fake.tripinfo_text = sumo_outputs[0].read_text(encoding="utf-8")
    manager = FakeManager()
    monkeypatch.setattr(session_mod.PreemptionManager, "from_config", lambda cfg: manager)
    monkeypatch.setattr(rs, "make_controller", lambda cfg: LoggingController())
    requirements = SimpleNamespace(intergreen={}, foes=P4_FOES)
    monkeypatch.setattr(rs.tls_check, "cached_requirements", lambda net, cfg: requirements)
    meta = rs.run(P4_CFG, 1, tmp_path)
    out = tmp_path / "fake_seed1"
    assert meta["control"] == {"switches": 2}
    assert meta["preemption"] == {"enabled": True, "requests": 1, "served": 1, "timeouts": 0,
                                  "active_s": 3.0, "hold_s": 0, "jumps": 0,
                                  "ped_yield": 0}  # fmt: skip
    assert set(meta["safety"]) == {"interventions", "audit", "danger", "ped_forcing"}
    events = pd.read_parquet(out / "events.parquet")
    assert list(events.columns) == EVENT_COLUMNS
    assert events[["t", "source", "rule", "stage_from", "stage_to"]].values.tolist() == [
        [6.0, "controller", "switch", 0, 1], [17.0, "controller", "switch", 1, 0],
    ]  # fmt: skip
    kpis = pd.read_csv(out / "kpi_run.csv")
    junction = kpis[kpis["user_type"] == "junction"]["kpi"].tolist()
    assert junction == ["box_halt_red_s", "box_foe_onsets", "exit_blocked_s",
                        "exit_blocked_green_s"]  # fmt: skip

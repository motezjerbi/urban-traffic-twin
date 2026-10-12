"""Bus detection for TSP (Part 4, sim/transit.py): BusCall fields on a fake TraCI, and
on the Pitié net in SUMO the remaining-dwell rule of plan section 2."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic_twin.config import approach_in_edges
from traffic_twin.control.registry import make_controller
from traffic_twin.sim.signal import Stage
from traffic_twin.sim.transit import BusDetector

# links 0-3: 0 and 1 green (G) in A, 2 G in B, 3 G in A and g in B
PHASES = ["GGrG", "yyry", "rrGg", "rryy"]
STAGES = [Stage("A", 0, (1,)), Stage("B", 2, (3,))]
APPROACHES = {"W": ["w_in"], "NB": ["nb_in"]}


class FakeVehicle:
    def __init__(self, vehicles: dict):
        self.v = vehicles  # vid -> dict(cls, line, next_tls, speed, stops)
        self.class_reads: list[str] = []

    def getVehicleClass(self, vid):
        self.class_reads.append(vid)
        return self.v[vid]["cls"]

    def getNextTLS(self, vid):
        return self.v[vid]["next_tls"]

    def getLine(self, vid):
        return self.v[vid]["line"]

    def getSpeed(self, vid):
        return self.v[vid]["speed"]

    def getStops(self, vid, limit):
        assert limit == 1
        return self.v[vid]["stops"][:1]


def stop(lane: str, duration: float):
    return SimpleNamespace(lane=lane, duration=duration, stoppingPlaceID="s")


def veh(cls: str, line: str, next_tls: list, speed: float, stops: list) -> dict:
    return {"cls": cls, "line": line, "next_tls": next_tls, "speed": speed, "stops": stops}


@pytest.fixture
def fake():
    vehicles = {
        "bus_91_E.0": veh("bus", "91", [("C", 1, 40.0, "r")], 6.5, [stop("w_in_1", 20.0)]),
        "bus_91_W.0": veh("bus", "91", [("X", 0, 5.0, "G"), ("C", 3, 12.0, "r")], 0.0,
                          [stop("wb_out_0", 20.0)]),
        "car_W.3": veh("passenger", "", [("C", 0, 3.0, "G")], 3.0, []),
        "bus_57_S.0": veh("bus", "57", [], 8.0, []),
        "bus_24_E.0": veh("bus", "24", [("C", 2, 80.0, "G")], 8.0, []),
    }  # fmt: skip
    on_edge = {"w_in": ["car_W.3", "bus_91_E.0", "bus_57_S.0"], "nb_in": ["bus_91_W.0"],
               "s_in": ["bus_24_E.0"]}  # fmt: skip
    return SimpleNamespace(
        vehicle=FakeVehicle(vehicles),
        edge=SimpleNamespace(getLastStepVehicleIDs=lambda e: on_edge.get(e, [])),
    )


def test_bus_calls_fields(fake):
    det = BusDetector(fake, "C", STAGES, PHASES, APPROACHES)
    calls = det.calls(10.0)
    assert [c.vid for c in calls] == ["bus_91_W.0", "bus_91_E.0"]  # by distance
    wb, eb = calls
    assert (eb.line, eb.link, eb.stages, eb.approach) == ("91", 1, (0,), "W")
    assert (eb.dist_m, eb.speed_mps, eb.dwell_left_s) == (40.0, 6.5, 20.0)  # stop on w_in
    # the next TLS of junction C (not X); its next stop is downstream: no dwell here
    assert (wb.link, wb.stages, wb.approach, wb.dwell_left_s) == (3, (0, 1), "NB", 0.0)


def test_cars_buses_off_the_approaches_and_past_the_stop_line_are_ignored(fake):
    det = BusDetector(fake, "C", STAGES, PHASES, APPROACHES)
    ids = {c.vid for c in det.calls(0.0)}
    assert "car_W.3" not in ids  # not a bus
    assert "bus_57_S.0" not in ids  # no stop line of C ahead
    assert "bus_24_E.0" not in ids  # s_in is not an approach in-edge here
    det.calls(1.0)
    reads = fake.vehicle.class_reads
    assert len(reads) == len(set(reads)) == 4  # vClass read once per vehicle


def test_stages_of_a_link():
    det = BusDetector(SimpleNamespace(), "C", STAGES, PHASES, APPROACHES)
    assert det.stages_of(0) == (0,) and det.stages_of(2) == (1,) and det.stages_of(3) == (0, 1)


@pytest.mark.sumo
def test_pitie_bus_91_east_call_and_remaining_dwell(pitie_scenario, tmp_path: Path):
    """Seed 101, 120 s: bus_91_E.0 (wb_in -> nb_out, stop_SM_E 8.8 m before the stop line)
    is served by link 9 in BUS_91 only; while it dwells its remaining dwell counts down
    20 -> 0, then it passes the stop line and leaves the calls."""
    from traffic_twin.sim.session import SimSession

    cfg = copy.deepcopy(pitie_scenario)
    cfg["duration_s"] = 300
    ctrl = make_controller(cfg)
    sim = SimSession(cfg, 101, tmp_path / "run", log_state=False, demand_dir=tmp_path / "dem")
    seen: dict[str, list] = {}
    with sim:
        det = BusDetector(sim.traci, sim.tls, sim.plan.stages, sim.phase_states,
                          approach_in_edges(cfg))  # fmt: skip
        ctrl.reset(101)
        while sim.t < 120:
            for c in det.calls(sim.t):
                seen.setdefault(c.vid, []).append((sim.t, c))
            sim.step(ctrl.decide(sim.observe()))
    east = seen["bus_91_E.0"]
    assert {(c.line, c.link, c.stages, c.approach) for _, c in east} == {("91", 9, (3,), "WB")}
    dwell = [c.dwell_left_s for _, c in east]
    assert dwell[0] == 20.0 and dwell[-1] == 0.0
    assert all(0 <= a - b <= 1.0 for a, b in zip(dwell, dwell[1:], strict=False))
    assert set(range(21)) <= set(dwell)  # every remaining second, one per step
    dists = [c.dist_m for _, c in east]
    assert all(b <= a + 1e-9 for a, b in zip(dists, dists[1:], strict=False))
    times = [t for t, _ in east]
    assert times == [float(t) for t in range(int(times[0]), int(times[-1]) + 1)]
    assert times[-1] < 119  # left the calls after the stop line, before the end
    assert "bus_24_W.0" in seen  # nb_in: link 3, BUS_91 and BUS_57, no stop on nb_in
    assert {(c.link, c.stages, c.approach, c.dwell_left_s) for _, c in seen["bus_24_W.0"]} == {
        (3, (3, 4), "NB", 0.0)
    }

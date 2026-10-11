"""Shared fixtures. Tests run from the repo root (pytest rootdir), so the repo configs
and the SUMO network are reachable by their relative paths."""

from __future__ import annotations

import copy
import functools
from pathlib import Path

import pytest

from traffic_twin.config import load_scenario
from traffic_twin.paths import SCENARIO_CONFIG

REPO_ROOT = Path(__file__).resolve().parent.parent

# Small SUMO outputs: two cars, a bus, a pedestrian walk, and three bus stop arrivals
TRIPINFO = """<tripinfos>
  <tripinfo id="car_W_straight.0" vType="car" depart="27000" arrival="27060" duration="60"
            timeLoss="20" waitingTime="10" waitingCount="1"/>
  <tripinfo id="car_W_straight.1" vType="car" depart="27010" arrival="27050" duration="40"
            timeLoss="0" waitingTime="0" waitingCount="0"/>
  <tripinfo id="bus_B1.0" vType="bus" depart="27000" arrival="27100" duration="100"
            timeLoss="30" waitingTime="15" waitingCount="1"/>
  <personinfo id="ped_w_in_e_out.0" depart="27000">
    <walk depart="27000" arrival="27300" duration="300" routeLength="400" timeLoss="45"/>
  </personinfo>
</tripinfos>"""

STOPS = """<stops>
  <stopinfo id="bus_B1.0" started="100"/><stopinfo id="bus_B1.1" started="500"/>
  <stopinfo id="bus_B1.2" started="940"/>
</stops>"""


@functools.cache
def sumo_available() -> bool:
    """True when the `sumo` binary and a TraCI backend can be found."""
    try:
        import traci  # noqa: F401

        from traffic_twin.sim.sumo import find_binary

        find_binary("sumo")
    except (ImportError, FileNotFoundError):
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip `@pytest.mark.sumo` tests when the SUMO binaries are not installed."""
    if sumo_available():
        return
    skip = pytest.mark.skip(reason="SUMO not installed (pip install -e '.[sim]')")
    for item in items:
        if "sumo" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def base_scenario() -> dict:
    """configs/scenario_base.yaml, loaded once. Do not mutate: use `scenario_cfg`."""
    return load_scenario(REPO_ROOT / SCENARIO_CONFIG)


@pytest.fixture
def scenario_cfg(base_scenario: dict) -> dict:
    """A private deep copy of the base scenario config that a test may modify."""
    return copy.deepcopy(base_scenario)


@pytest.fixture
def write_file(tmp_path: Path):
    """Write text to a file under tmp_path and return its path."""

    def _write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def sumo_outputs(write_file) -> tuple[Path, Path]:
    """(tripinfo.xml, stops.xml) written from TRIPINFO and STOPS."""
    return write_file("tripinfo.xml", TRIPINFO), write_file("stops.xml", STOPS)

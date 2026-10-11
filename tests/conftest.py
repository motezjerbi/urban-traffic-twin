"""Shared fixtures. Tests run from the repo root (pytest rootdir), so the repo configs
and the SUMO network are reachable by their relative paths."""

from __future__ import annotations

import copy
import functools
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from traffic_twin.config import load_scenario
from traffic_twin.paths import PITIE_FIXED_CONFIG, SCENARIO_CONFIG

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


@pytest.fixture(scope="session")
def pitie_scenario() -> dict:
    """configs/pitie_fixed.yaml (S0 on the Pitié junction, `extends` merged), loaded
    once. Do not mutate: use `pitie_cfg`."""
    return load_scenario(REPO_ROOT / PITIE_FIXED_CONFIG)


@pytest.fixture
def pitie_cfg(pitie_scenario: dict) -> dict:
    """A private deep copy of the Pitié S0 scenario config that a test may modify."""
    return copy.deepcopy(pitie_scenario)


def scenario_inputs(cfg: dict) -> list[Path]:
    """Repo-relative files a headless run of `cfg` reads besides its configs: the net
    and, for calibrated demand, the calibration file. Absolute paths are left out."""
    files = [Path(cfg["network"]["net_file"])]
    calibration = (cfg.get("demand") or {}).get("calibration")
    if calibration:
        files.append(Path(calibration))
    return [f for f in files if not f.is_absolute()]


@pytest.fixture(scope="session")
def make_workspace() -> Callable[..., Path]:
    """`make_workspace(root, *cfgs) -> root`: copy configs/ and the inputs of the given
    scenario configs (net, calibration) from the repo into `root`, at the same relative
    paths. Runs started with `root` as the working directory read the committed inputs
    and write nothing into the repo (demand files, results), and their `demand_hash`
    equals the one computed in the repo."""

    def _make(root: Path, *cfgs: dict) -> Path:
        root = Path(root)
        shutil.copytree(REPO_ROOT / "configs", root / "configs", dirs_exist_ok=True)
        for cfg in cfgs:
            for rel in scenario_inputs(cfg):
                src = REPO_ROOT / rel
                if not src.is_file():
                    raise FileNotFoundError(f"scenario input {rel.as_posix()} is not in the repo")
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, root / rel)
        return root

    return _make


@pytest.fixture
def pitie_workdir(make_workspace, pitie_scenario: dict, tmp_path: Path, monkeypatch) -> Path:
    """tmp_path holding the Pitié configs, net and calibration (see `make_workspace`), as
    the working directory: `load_scenario("configs/pitie_actuated.yaml")` and runs work
    from there."""
    make_workspace(tmp_path, pitie_scenario)
    monkeypatch.chdir(tmp_path)
    return tmp_path


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

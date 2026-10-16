"""Shared fixtures. Tests run from the repo root (pytest rootdir), so the repo configs
and the SUMO network are reachable by their relative paths."""

from __future__ import annotations

import copy
import functools
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from traffic_twin.config import load_config, load_scenario, merge
from traffic_twin.paths import (
    CONFIG_DIR,
    ETA_CONFIG,
    PITIE_FIXED_CONFIG,
    PITIE_RL_CONFIG,
    PITIE_RULES_CONFIG,
    RL7_CONFIG,
    RL7_CONFIG_DIR,
    RL_CONFIG,
    SCENARIO_CONFIG,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_RULES_CONFIG = CONFIG_DIR / "pitie_amb_rules.yaml"  # ambulance-probe family, S2

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


@pytest.fixture(scope="session")
def rules_scenario() -> dict:
    """configs/pitie_rules.yaml (S2 rule based on the Pitié junction, `extends` merged),
    loaded once. Do not mutate: use `rules_cfg`."""
    return load_scenario(REPO_ROOT / PITIE_RULES_CONFIG)


@pytest.fixture
def rules_cfg(rules_scenario: dict) -> dict:
    """A private deep copy of the Pitié S2 scenario config that a test may modify."""
    return copy.deepcopy(rules_scenario)


@pytest.fixture(scope="session")
def probe_scenario() -> dict:
    """configs/pitie_amb_rules.yaml (ambulance-probe family, S2 with the 7 trips of
    `emergency.trips`), loaded once. Do not mutate: use `probe_cfg`."""
    return load_scenario(REPO_ROOT / PROBE_RULES_CONFIG)


@pytest.fixture
def probe_cfg(probe_scenario: dict) -> dict:
    """A private deep copy of the ambulance-probe S2 scenario config."""
    return copy.deepcopy(probe_scenario)


@pytest.fixture(scope="session")
def rl_scenario() -> dict:
    """configs/pitie_rl.yaml (the RL controller on the Pitié junction: S0 demand, S2
    preemption and safety, `controller: rl`), loaded once. Do not mutate: use `rl_cfg`."""
    return load_scenario(REPO_ROOT / PITIE_RL_CONFIG)


@pytest.fixture
def rl_cfg(rl_scenario: dict) -> dict:
    """A private deep copy of the Pitié RL scenario config that a test may modify."""
    return copy.deepcopy(rl_scenario)


@pytest.fixture(scope="session")
def rl_train_scenario() -> dict:
    """configs/rl.yaml (environment and training, pre-registered D55-D62), loaded once.
    Do not mutate."""
    return load_config(REPO_ROOT / RL_CONFIG)


@pytest.fixture(scope="session")
def rl7_train_scenario() -> dict:
    """configs/rl7.yaml (Part 7 pre-registration D68-D90, `extends` rl.yaml merged),
    loaded once. Do not mutate: use `rl7_cfg`."""
    return load_config(REPO_ROOT / RL7_CONFIG)


@pytest.fixture
def rl7_cfg(rl7_train_scenario: dict) -> dict:
    """A private deep copy of the merged configs/rl7.yaml that a test may modify."""
    return copy.deepcopy(rl7_train_scenario)


# Part 7 stub-workspace settings (`stub_rl7_cfg`): development seeds only (training range
# 5000-5099, validation 5100-5109, the other roles above them; never a selection,
# holdout, sensitivity, stress or dev-test seed of rl7.yaml), tiny budgets, one
# validation worker, no clean-tree requirement.
STUB_RL7 = {
    "seeds": {
        "train": [5000, 5099], "selection": [5100, 5109], "select2": [5110, 5111],
        "holdout": [5112, 5115], "sensitivity": [5116, 5117], "stress": [5118, 5119],
        "reserve": [5120, 5199], "burned": [5200, 5209], "dev_test": [5210, 5219],
    },
    "train": {"n_envs": 2, "eval_every_steps": 200, "eval_workers": 1,
              "checkpoint_every_steps": 400, "require_clean": False},
    "dqn": {"exploration_steps": 300},
    "protocol": {
        "stage_t": {"steps": 600}, "stage_f": {"steps": 1200}, "sensitivity": {"steps": 600},
        "holdout": {"full": [5112, 5113]}, "devtest": {"seeds": [5210, 5211]}, "slots": 2,
    },
}  # fmt: skip


@pytest.fixture
def stub_rl7_cfg(tmp_path: Path) -> Callable[..., Path]:
    """`stub_rl7_cfg(**overrides) -> Path`: a Part 7 config tree under tmp_path for
    SUMO-free (stub) and small tests, returning its `rl7.yaml`:

    - `<tmp>/rl7cfg/rl7.yaml` extends the repo's configs/rl7.yaml (absolute path) with
      STUB_RL7, `outputs` under `<tmp>/rl7out/` (runs, study, models, docs, figures) and
      `overrides` merged on top (`config.merge`: mappings key by key);
    - `<tmp>/rl7cfg/rl7/*.yaml` are verbatim copies of configs/rl7/*.yaml, so each run
      config (`.../rl7/t1.yaml`, ...) extends the stub rl7.yaml instead of the repo's.

    Nothing is written into the repo; rl.yaml `base_config` stays repo-relative (tests run
    from the repo root)."""

    def _make(**overrides) -> Path:
        root = tmp_path / "rl7cfg"
        (root / "rl7").mkdir(parents=True, exist_ok=True)
        out = tmp_path / "rl7out"
        outputs = {
            "runs": out / "runs",
            "study": out,
            "models": out / "models" / "rl",
            "docs": out / "docs" / "rl_training.md",
            "figures": out / "docs" / "figures",
            "selected": out / "selected",  # rl.yaml's Part 6 key, never data/models here
        }
        cfg = merge(STUB_RL7, {"outputs": {k: v.as_posix() for k, v in outputs.items()}})
        cfg = {"extends": (REPO_ROOT / RL7_CONFIG).as_posix(), **merge(cfg, overrides)}
        path = root / "rl7.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        for src in sorted((REPO_ROOT / RL7_CONFIG_DIR).glob("*.yaml")):
            shutil.copy2(src, root / "rl7" / src.name)
        return path

    return _make


@pytest.fixture(scope="session")
def eta_scenario() -> dict:
    """configs/eta.yaml (ETA dataset, model and evaluation, pre-registered D38-D52),
    loaded once. Do not mutate: use `eta_cfg`."""
    return load_config(REPO_ROOT / ETA_CONFIG)


@pytest.fixture
def eta_cfg(eta_scenario: dict) -> dict:
    """A private deep copy of configs/eta.yaml that a test may modify."""
    return copy.deepcopy(eta_scenario)


@pytest.fixture
def api_results(tmp_path: Path) -> Path:
    """A results root under tmp_path with synthetic run folders
    (`api.fixtures.make_results`, 3 runs): the API tests never read data/results."""
    from traffic_twin.api.fixtures import make_results

    root = tmp_path / "results"
    make_results(root, n_runs=3)
    return root


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

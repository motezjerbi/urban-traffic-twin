"""tasks.py, the cross-platform stand-in for the Makefile."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tasks():
    spec = importlib.util.spec_from_file_location("tasks", ROOT / "tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def commands(tasks, *argv):
    return tasks.build(tasks.parser().parse_args(list(argv)))


def test_every_makefile_target_has_a_task(tasks):
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    phony = " ".join(re.findall(r"^\.PHONY:(.*)$", makefile, re.M)).split()
    later = {"train", "eval", "app"}  # placeholders for later parts of the roadmap
    assert set(phony) - later <= set(tasks.TASKS)
    assert {"plots", "baseline", "check"} <= set(phony)


def test_tasks_use_the_running_interpreter(tasks):
    for name in tasks.TASKS:
        if name == "db-runs":
            continue  # needs finished runs on disk
        for cmd in commands(tasks, name):
            assert cmd[0] == sys.executable


def test_sim_passes_scenario_and_seeds(tasks):
    (cmd,) = commands(tasks, "sim", "--seeds", "1", "2", "3", "--scenario", "x.yaml")
    assert cmd[1:] == ["-m", "traffic_twin.sim.run_scenario", "--config", "x.yaml",
                       "--seed", "1", "2", "3"]  # fmt: skip
    (gui,) = commands(tasks, "sim-gui", "--seeds", "5", "6")
    assert gui[-3:] == ["--seed", "5", "--gui"]


def test_bad_seed_rejected(tasks):
    with pytest.raises(SystemExit, match="seed"):
        commands(tasks, "sim", "--seeds", "one")


def test_composite_tasks(tasks):
    modules = [c[2] for c in commands(tasks, "baseline")]
    assert modules == ["traffic_twin.sim.network", "traffic_twin.sim.run_scenario",
                       "traffic_twin.viz.plots"]  # fmt: skip
    check = [c[2:] for c in commands(tasks, "check")]
    assert check == [["ruff", "check", "."], ["ruff", "format", "--check", "."], ["pytest", "-q"]]


def test_data_config_option(tasks):
    for cmd in commands(tasks, "data", "explore", "db-load", "--config", "c.yaml"):
        assert cmd[-2:] == ["--config", "c.yaml"]


def test_db_runs_lists_finished_runs(tasks, tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="no finished runs"):
        commands(tasks, "db-runs")
    for name in ("b_seed2", "a_seed1"):
        (tmp_path / "data/results" / name).mkdir(parents=True)
        (tmp_path / "data/results" / name / "run.json").write_text("{}")
    (tmp_path / "data/results/partial").mkdir()
    (cmd,) = commands(tasks, "db-runs")
    assert cmd[-2:] == ["data/results/a_seed1", "data/results/b_seed2"]


def test_run_stops_at_first_failure(tasks, capsys):
    ok = [sys.executable, "-c", "pass"]
    fail = [sys.executable, "-c", "raise SystemExit(3)"]
    never = [sys.executable, "-c", "raise SystemExit(9)"]
    assert tasks.run([ok, fail, never]) == 3
    out = capsys.readouterr().out
    assert out.count("+ ") == 2  # each command is printed before it runs


def test_dry_run_runs_nothing(tasks, capsys):
    assert tasks.main(["sim", "net", "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and all(line.startswith("+ ") for line in out)

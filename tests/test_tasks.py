"""tasks.py, the cross-platform stand-in for the Makefile."""

import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
S0, S1 = "configs/pitie_fixed.yaml", "configs/pitie_actuated.yaml"
GRID = "configs/scenario_base.yaml"


@pytest.fixture(scope="module")
def tasks():
    spec = importlib.util.spec_from_file_location("tasks", ROOT / "tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def commands(tasks, *argv):
    return tasks.build(tasks.parser().parse_args(list(argv)))


def modules(cmds):
    return [c[2] for c in cmds]


# -- Makefile mirror ---------------------------------------------------------------------
def makefile_rules() -> dict[str, tuple[list[str], list[str]]]:
    """target -> (prerequisites, recipe lines). Target-specific variable lines
    (`t: VAR = value`) and conditionals are skipped."""
    rules: dict[str, tuple[list[str], list[str]]] = {}
    current: list[str] = []
    for line in (ROOT / "Makefile").read_text(encoding="utf-8").splitlines():
        if line.startswith("\t"):
            for t in current:
                rules[t][1].append(line.strip())
            continue
        m = re.match(r"^([A-Za-z][\w.-]*(?: [A-Za-z][\w.-]*)*)\s*:(?!=)(.*)$", line)
        if m and "=" not in m.group(2):
            current = m.group(1).split()
            for t in current:
                rules.setdefault(t, ([], []))[0].extend(m.group(2).split())
        elif line.strip() and not line.startswith("#"):
            current = []
    return rules


def make_calls(rules, target) -> list[tuple[str, list[str]]]:
    """(module, --flags) of every `$(PY) -m module ...` the target runs, prerequisites
    first. Flags inside `$(if ...)` (optional arguments) are left out."""
    prereqs, recipe = rules[target]
    calls = [c for p in prereqs for c in make_calls(rules, p)]
    for line in recipe:
        m = re.search(r"\$\(PY\) -m (\S+)(.*)", line)
        if m:
            args = re.sub(r"\$\(if .*$", "", m.group(2))
            calls.append((m.group(1), re.findall(r"(?<!\S)--[\w-]+", args)))
    return calls


def test_every_makefile_target_has_a_task(tasks):
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    phony = " ".join(re.findall(r"^\.PHONY:(.*)$", makefile, re.M)).split()
    later = {"train", "eval", "app"}  # placeholders for later parts of the roadmap
    assert set(phony) - later == set(tasks.TASKS)
    assert {"plots", "baseline", "check", "part3"} <= set(phony)


def test_makefile_mirrors_tasks(tasks, monkeypatch):
    """Each Make target runs the same modules, in the same order, with the same options."""
    monkeypatch.setattr(tasks, "_run_dirs", lambda results: ["data/results/a_seed1"])
    rules = makefile_rules()
    for name in tasks.TASKS:
        expected = [(c[2], [a for a in c[3:] if a.startswith("--")]) for c in commands(tasks, name)]
        assert make_calls(rules, name) == expected, name


# -- every task ----------------------------------------------------------------------------
def test_tasks_use_the_running_interpreter(tasks):
    for name in tasks.TASKS:
        if name == "db-runs":
            continue  # needs finished runs on disk
        for cmd in commands(tasks, name):
            assert cmd[0] == sys.executable


def test_default_configs_exist(tasks):
    for path in (tasks.GRID_SCENARIO, tasks.PITIE_S0, tasks.PITIE_S1):
        assert (ROOT / path).is_file(), path


def test_help_lists_every_task(tasks):
    text = tasks.parser().format_help()
    assert all(name in text for name in tasks.TASKS)
    assert text.isascii()  # printable on any Windows console code page


# -- Part 2 tasks ----------------------------------------------------------------------------
def test_sim_passes_scenario_and_seeds(tasks):
    (cmd,) = commands(tasks, "sim", "--seeds", "1", "2", "3", "--scenario", "x.yaml")
    assert cmd[1:] == ["-m", "traffic_twin.sim.run_scenario", "--config", "x.yaml",
                       "--seed", "1", "2", "3"]  # fmt: skip
    (gui,) = commands(tasks, "sim-gui", "--seeds", "5", "6")
    assert gui[-3:] == ["--seed", "5", "--gui"]


def test_grid_tasks_default_to_the_grid(tasks):
    (sim,) = commands(tasks, "sim")
    assert sim[3:] == ["--config", GRID, "--seed", "1", "2", "3"]
    (gui,) = commands(tasks, "sim-gui")
    assert gui[3:] == ["--config", GRID, "--seed", "1", "--gui"]
    (demand,) = commands(tasks, "demand")
    assert demand[3:] == ["--config", GRID]


def test_net_passes_the_scenario(tasks):
    (grid,) = commands(tasks, "net")
    assert grid[1:] == ["-m", "traffic_twin.sim.network", "--config", GRID]
    (pitie,) = commands(tasks, "net", "--scenario", S0)
    assert pitie[3:] == ["--config", S0]


def test_bad_seed_rejected(tasks):
    with pytest.raises(SystemExit, match="seed"):
        commands(tasks, "sim", "--seeds", "one")
    with pytest.raises(SystemExit, match="seed"):
        commands(tasks, "calibrate", "--seeds", "-1")


def test_composite_tasks(tasks):
    assert modules(commands(tasks, "baseline")) == [
        "traffic_twin.sim.network", "traffic_twin.sim.run_scenario", "traffic_twin.viz.plots",
    ]  # fmt: skip
    check = [c[2:] for c in commands(tasks, "check")]
    assert check == [["ruff", "check", "."], ["ruff", "format", "--check", "."], ["pytest", "-q"]]


def test_data_config_option(tasks):
    for cmd in commands(tasks, "data", "explore", "db-load", "calib-data", "--config", "c.yaml"):
        assert cmd[3:5] == ["--config", "c.yaml"]


# -- Part 3 tasks ----------------------------------------------------------------------------
def test_part3_tasks_default_to_pitie(tasks):
    got = {name: cmd[2:] for name in ("osm", "tls-check", "field-check", "calibrate",
                                      "validate", "compare")
           for (cmd,) in [commands(tasks, name)]}  # fmt: skip
    assert got == {
        "osm": ["traffic_twin.sim.osm_import", "--config", S0],
        "tls-check": ["traffic_twin.sim.tls_check", "--config", S0],
        "field-check": ["traffic_twin.validation.field", "--check", "--scenario", S0],
        "calibrate": ["traffic_twin.validation.calibration_loop", "--scenario", S0],
        "validate": ["traffic_twin.validation.report", "--scenario", S0, "--run-missing",
                     "--docs", "docs/validation.md"],
        "compare": ["traffic_twin.evaluation.compare", "--baseline-config", S0,
                    "--scenario-config", S1, "--run-missing"],
    }  # fmt: skip


def test_part3_tasks_take_the_scenario(tasks):
    for name in ("osm", "tls-check", "field-check", "calibrate", "validate"):
        (cmd,) = commands(tasks, name, "--scenario", "x.yaml")
        assert "x.yaml" in cmd and S0 not in cmd, name
    (cmp,) = commands(tasks, "compare", "--scenario", "s1.yaml", "--baseline", "s0.yaml")
    assert cmp[3:] == ["--baseline-config", "s0.yaml", "--scenario-config", "s1.yaml",
                       "--run-missing"]  # fmt: skip


def test_calibrate_passes_seeds_only_when_given(tasks):
    (default,) = commands(tasks, "calibrate")
    assert "--seeds" not in default  # the calibration seeds of configs/validation.yaml
    (cmd,) = commands(tasks, "calibrate", "--seeds", "101", "102")
    assert cmd[-3:] == ["--seeds", "101", "102"]


def test_calib_data_reads_raw_with_the_data_config(tasks):
    (cmd,) = commands(tasks, "calib-data")
    assert cmd[2:] == ["traffic_twin.data.calibrate", "--config", "configs/data.yaml",
                       "--from-raw"]  # fmt: skip


def test_plots_pitie(tasks):
    (cmd,) = commands(tasks, "plots-pitie")
    assert cmd[2:] == ["traffic_twin.viz.plots", "--scenario", "pitie_fixed", "pitie_actuated",
                       "--out", "docs/figures/pitie"]  # fmt: skip


def test_part3_runs_in_order(tasks):
    cmds = commands(tasks, "part3")
    assert modules(cmds) == [
        "traffic_twin.sim.tls_check",
        "traffic_twin.validation.field",
        "traffic_twin.validation.calibration_loop",
        "traffic_twin.evaluation.compare",
        "traffic_twin.validation.report",
        "traffic_twin.viz.plots",
    ]
    tls, field, calib, cmp, report, _ = cmds
    for cmd in (tls, field, calib, report):  # every step but compare runs on S0
        assert S0 in cmd and S1 not in cmd
    assert cmp[3:] == ["--baseline-config", S0, "--scenario-config", S1, "--run-missing"]


def test_part3_scenario_is_s1_and_baseline_is_s0(tasks):
    cmds = commands(tasks, "part3", "--scenario", "s1.yaml", "--baseline", "s0.yaml")
    tls, field, calib, cmp, report, _ = cmds
    for cmd in (tls, field, calib, report):
        assert "s0.yaml" in cmd and "s1.yaml" not in cmd
    assert cmp[3:] == ["--baseline-config", "s0.yaml", "--scenario-config", "s1.yaml",
                       "--run-missing"]  # fmt: skip


# -- runner ------------------------------------------------------------------------------------
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


def test_part3_dry_run(tasks, capsys):
    assert tasks.main(["part3", "-n"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == len(tasks.PART3_STEPS) and all(line.startswith("+ ") for line in out)


# -- pins (D17) --------------------------------------------------------------------------------
def test_sumo_packages_pinned_to_the_fixture_version():
    """The [sim] extra pins every SUMO package to the version the grid regression fixture
    was captured with (netconvert output and link order depend on it)."""
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sim = extras["project"]["optional-dependencies"]["sim"]
    pins = dict(re.fullmatch(r"([\w-]+)==([\d.]+)", req).groups() for req in sim)
    assert set(pins) == {"eclipse-sumo", "traci", "sumolib", "libsumo"}
    meta = json.loads((ROOT / "tests/data/grid_baseline_meta.json").read_text(encoding="utf-8"))
    assert set(pins.values()) == {meta["sumo_version"].removeprefix("v")}

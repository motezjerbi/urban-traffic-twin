"""tasks.py, the cross-platform stand-in for the Makefile."""

import importlib
import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
S0, S1 = "configs/pitie_fixed.yaml", "configs/pitie_actuated.yaml"
S2 = "configs/pitie_rules.yaml"
EVAL = "configs/evaluation.yaml"
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
    assert {"plots", "baseline", "check", "part3", "part4"} <= set(phony)


def test_makefile_mirrors_tasks(tasks, monkeypatch):
    """Each Make target runs the same modules, in the same order, with the same options.
    A task with in-process steps (stdlib Python inside tasks.py) is delegated whole."""
    monkeypatch.setattr(tasks, "_run_dirs", lambda results: ["data/results/a_seed1"])
    rules = makefile_rules()
    for name in tasks.TASKS:
        cmds = commands(tasks, name)
        if any(isinstance(c, tasks.Step) for c in cmds):
            assert rules[name] == ([], [f"$(PY) tasks.py {name}"]), name
            continue
        expected = [(c[2], [a for a in c[3:] if a.startswith("--")]) for c in cmds]
        assert make_calls(rules, name) == expected, name


# -- every task ----------------------------------------------------------------------------
def test_tasks_use_the_running_interpreter(tasks):
    for name in tasks.TASKS:
        if name == "db-runs":
            continue  # needs finished runs on disk
        for cmd in commands(tasks, name):
            assert isinstance(cmd, tasks.Step) or cmd[0] == sys.executable


def test_default_configs_exist(tasks):
    for path in (
        tasks.GRID_SCENARIO,
        tasks.PITIE_S0,
        tasks.PITIE_S1,
        tasks.PITIE_S2,
        tasks.EVALUATION_CONFIG,
    ):
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


def test_field_sheets_task(tasks):
    (cmd,) = commands(tasks, "field-sheets")
    assert cmd[2:] == ["traffic_twin.validation.field_sheets", "--scenario", S0]
    (cmd,) = commands(tasks, "field-sheets", "--scenario", "x.yaml")
    assert cmd[-1] == "x.yaml"


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


# -- Part 4 tasks ----------------------------------------------------------------------------
PART4_MODULES = ["traffic_twin.sim.tls_check", "traffic_twin.evaluation.evaluate",
                 "traffic_twin.viz.evaluation"]  # fmt: skip


def test_evaluate_and_plots_eval(tasks):
    (ev,) = commands(tasks, "evaluate")
    assert ev[2:] == ["traffic_twin.evaluation.evaluate", "--config", EVAL, "--run-missing",
                      "--docs", "docs/evaluation.md"]  # fmt: skip
    (fig,) = commands(tasks, "plots-eval")
    assert fig[2:] == ["traffic_twin.viz.evaluation", "--config", EVAL,
                       "--out", "docs/figures/pitie/eval"]  # fmt: skip


def test_evaluation_ignores_scenario_options(tasks):
    """The evaluated scenarios are pre-registered in configs/evaluation.yaml (D33): no
    command-line option of tasks.py changes them."""
    for name in ("evaluate", "plots-eval"):
        assert commands(tasks, name, "--scenario", "x.yaml", "--baseline", "y.yaml",
                        "--seeds", "1") == commands(tasks, name), name  # fmt: skip


@pytest.mark.parametrize("name", ["evaluate", "plots-eval"])
def test_part4_commands_parse_with_their_cli(tasks, name):
    """Every flag the task passes is one the module's own parser accepts."""
    (cmd,) = commands(tasks, name)
    args = importlib.import_module(cmd[2]).build_parser().parse_args(cmd[3:])
    assert args.config == Path(EVAL)
    assert vars(args).get("run_missing", True)  # evaluate: run the missing seeds


def test_part4_runs_in_order(tasks):
    cmds = commands(tasks, "part4")
    assert modules(cmds) == PART4_MODULES
    tls, ev, fig = cmds
    assert tls[3:] == ["--config", S2]  # tls-check on S2 inside part4 ...
    assert ev == commands(tasks, "evaluate")[0] and fig == commands(tasks, "plots-eval")[0]
    (alone,) = commands(tasks, "tls-check")
    assert alone[3:] == ["--config", S0]  # ... and on S0 on its own (Part 3 default)


def test_part4_scenario_is_s2(tasks):
    tls, ev, fig = commands(tasks, "part4", "--scenario", "s2.yaml", "--baseline", "s0.yaml")
    assert tls[3:] == ["--config", "s2.yaml"]
    assert "s2.yaml" not in ev + fig and "s0.yaml" not in ev + fig


def test_part3_part4_chain(tasks):
    """`tasks.py part3 part4` (the full Pitié rerun after the field visit) keeps each
    chain's own defaults."""
    cmds = commands(tasks, "part3", "part4")
    assert len(cmds) == len(tasks.PART3_STEPS) + len(tasks.PART4_STEPS)
    assert cmds[0][3:] == ["--config", S0] and cmds[len(tasks.PART3_STEPS)][3:] == ["--config", S2]


# -- provenance of the published runs ------------------------------------------------------------
SCENARIOS = {"main": {"S0": "pitie_fixed", "S2": "pitie_rules"}, "probe": {"S0": "pitie_amb_fixed"}}


@pytest.fixture
def results(tasks, tmp_path, monkeypatch):
    """A fake repo: provenance.json of 3 scenarios x 2 seeds, every run made from
    `abc1234-dirty`, docs pages naming that version; git HEAD abc1234, clean."""
    monkeypatch.setattr(tasks, "ROOT", tmp_path)
    head = {"value": ("abc1234", False)}
    monkeypatch.setattr(tasks, "git_head", lambda: head["value"])
    prov = {
        "seeds": [1, 2],
        "families": {
            fam: {"scenarios": {k: {"scenario": v} for k, v in sc.items()}}
            for fam, sc in SCENARIOS.items()
        },
    }
    (tmp_path / tasks.EVAL_PROVENANCE).parent.mkdir(parents=True)
    (tmp_path / tasks.EVAL_PROVENANCE).write_text(json.dumps(prov), encoding="utf-8")
    for d in tasks.evaluation_run_dirs():
        write_run(tmp_path / d, "abc1234-dirty", kpis=f"kpi,value\nx,{d.name}\n")
    (tmp_path / "docs").mkdir()
    for doc in tasks.PUBLISHED_DOCS:
        (tmp_path / doc).write_text("| Code version (git) | abc1234-dirty |\n", encoding="utf-8")
    return tmp_path, head


def write_run(run_dir: Path, version: str, kpis: str, **extra) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "kpi_run.csv").write_text(kpis, encoding="utf-8")
    meta = {"run_id": f"id-{version}", "code_version": version, "started_at": version,
            "wall_time_s": 1.0, "config_hash": "c", "demand_hash": "d", **extra}  # fmt: skip
    (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")


def test_evaluation_run_dirs_cover_every_scenario_and_seed(tasks, results):
    dirs = {d.as_posix() for d in tasks.evaluation_run_dirs()}
    assert dirs == {f"data/results/{s}_seed{n}" for fam in SCENARIOS.values()
                    for s in fam.values() for n in (1, 2)}  # fmt: skip


def test_provenance_refuses_runs_and_pages_of_an_uncommitted_tree(tasks, results, capsys):
    """The Part 4 runs and pages were published as `18c6cc6-dirty`: `provenance` must
    fail on a dirty tree, on runs not made from HEAD and on a page naming `-dirty`."""
    root, head = results
    assert tasks.check_provenance() == 1
    err = capsys.readouterr().err
    assert "not every run was made from abc1234" in err
    assert all(f"{doc} names an uncommitted tree" in err for doc in tasks.PUBLISHED_DOCS)
    head["value"] = ("abc1234", True)
    assert tasks.check_provenance() == 1
    assert "uncommitted changes" in capsys.readouterr().err
    # runs and pages from the clean HEAD: accepted
    head["value"] = ("abc1234", False)
    for d in tasks.evaluation_run_dirs():
        write_run(root / d, "abc1234", kpis="kpi,value\n")
    for doc in tasks.PUBLISHED_DOCS:
        (root / doc).write_text("| Code version (git) | abc1234 |\n", encoding="utf-8")
    assert tasks.check_provenance() == 0


def test_rebaseline_refuses_an_uncommitted_tree(tasks, results):
    root, head = results
    head["value"] = ("abc1234", True)
    assert tasks.rebaseline_backup() == 1
    assert all((root / d / "run.json").is_file() for d in tasks.evaluation_run_dirs())


def rerun_all(tasks, root: Path, version: str, change: str | None = None) -> None:
    """Stand-in for `compare validate part4 --run-missing`: re-make every moved run
    from its backup (identical KPI rows unless it is `change`) and fix the pages."""
    backup = root / tasks.REBASELINE_DIR / "abc1234"
    for old in backup.iterdir():
        kpis = (old / "kpi_run.csv").read_text(encoding="utf-8")
        write_run(root / tasks.RESULTS / old.name, version,
                  kpis=kpis + ("y,1\n" if old.name == change else ""))  # fmt: skip
    for doc in tasks.PUBLISHED_DOCS:
        (root / doc).write_text(f"| Code version (git) | {version} |\n", encoding="utf-8")


def test_rebaseline_moves_reruns_and_accepts_identical_runs(tasks, results, capsys):
    root, _ = results
    assert tasks.rebaseline_backup() == 0
    backup = root / tasks.REBASELINE_DIR / "abc1234"
    assert len(list(backup.iterdir())) == 6
    assert not any((root / d).exists() for d in tasks.evaluation_run_dirs())
    rerun_all(tasks, root, "abc1234")
    assert tasks.rebaseline_verify() == 0
    assert "6 of 6 re-made runs identical" in capsys.readouterr().out
    assert tasks.rebaseline_backup() == 1  # the moved runs are kept, never overwritten


def test_rebaseline_fails_when_a_rerun_differs(tasks, results, capsys):
    root, _ = results
    assert tasks.rebaseline_backup() == 0
    rerun_all(tasks, root, "abc1234", change="pitie_rules_seed2")
    assert tasks.rebaseline_verify() == 1
    assert "pitie_rules_seed2: kpi_run.csv differs" in capsys.readouterr().err


def test_rebaseline_compares_the_run_summary(tasks, results, capsys):
    root, _ = results
    assert tasks.rebaseline_backup() == 0
    rerun_all(tasks, root, "abc1234")
    write_run(root / "data/results/pitie_fixed_seed1", "abc1234",
              kpis="kpi,value\nx,pitie_fixed_seed1\n", config_hash="other")  # fmt: skip
    assert tasks.rebaseline_verify() == 1
    assert "pitie_fixed_seed1: run.json differs" in capsys.readouterr().err


def test_rebaseline_chain(tasks):
    """Move aside, re-make the published pages with every default, then compare."""
    cmds = commands(tasks, "rebaseline", "--scenario", "x.yaml", "--baseline", "y.yaml")
    first, *middle, last = cmds
    assert (first.name, last.name) == ("rebaseline-backup", "rebaseline-verify")
    expected = commands(tasks, *tasks.REBASELINE_STEPS)
    assert middle == expected
    assert modules(middle) == ["traffic_twin.evaluation.compare",
                               "traffic_twin.validation.report", *PART4_MODULES]  # fmt: skip
    assert all("--run-missing" in c for c in middle if "compare" in c[2] or "report" in c[2])
    assert "x.yaml" not in str(cmds) and "y.yaml" not in str(cmds)


def test_steps_print_and_dry_run(tasks, capsys):
    assert tasks.main(["rebaseline", "-n"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "+ tasks.py:rebaseline-backup (in-process)"
    assert out[-1] == "+ tasks.py:rebaseline-verify (in-process)"
    calls = []
    step = tasks.Step("probe", lambda: calls.append(1) or 4)
    assert tasks.run([step, [sys.executable, "-c", "pass"]]) == 4 and calls == [1]


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


def test_part4_dry_run(tasks, capsys):
    assert tasks.main(["part4", "-n"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == len(tasks.PART4_STEPS) and all(line.startswith("+ ") for line in out)
    assert "evaluation.yaml" in out[1] and "evaluation.md" in out[1]


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

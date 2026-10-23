"""tasks.py, the cross-platform stand-in for the Makefile."""

import argparse
import importlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
S0, S1 = "configs/pitie_fixed.yaml", "configs/pitie_actuated.yaml"
S2 = "configs/pitie_rules.yaml"
EVAL = "configs/evaluation.yaml"
GRID = "configs/scenario_base.yaml"
ETA, RL = "configs/eta.yaml", "configs/rl.yaml"
DEVTEST = [str(s) for s in range(3020, 3030)]
# Tasks whose commands are read from a config at run time: the Makefile hands them to
# tasks.py whole (Make cannot read YAML)
DELEGATED = {"rl-train"}


@pytest.fixture(scope="module")
def tasks():
    spec = importlib.util.spec_from_file_location("tasks", ROOT / "tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def commands(tasks, *argv):
    return tasks.build(tasks.parser().parse_args(list(argv)))


def modules(cmds):
    """The modules the commands run (in-process steps left out)."""
    return [c[2] for c in cmds if isinstance(c, list)]


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
        m = re.search(r"\$\(PY\) (?:-m (\S+)|(tasks\.py \S+?))(?=\s|\$|$)(.*)", line)
        if m:
            args = re.sub(r"\$\(if .*$", "", m.group(3))
            calls.append((m.group(1) or m.group(2), re.findall(r"(?<!\S)--[\w.-]+", args)))
    return calls


def expected_calls(tasks, name) -> list[tuple[str, list[str]]]:
    """(module, --flags) of a task's commands, as the Makefile must run them: a delegated
    task is one `tasks.py <name>` call, also inside a chain."""
    if name in DELEGATED:
        return [(f"tasks.py {name}", [])]
    chains = {"part5": tasks.PART5_STEPS, "part6": tasks.PART6_STEPS}
    if name in chains:
        return [c for step in chains[name] for c in expected_calls(tasks, step)]
    return [(c[2], [a for a in c[3:] if a.startswith("--")]) for c in commands(tasks, name)]


def test_every_makefile_target_has_a_task(tasks):
    """Every phony target is a task and the reverse; the placeholders of the roadmap
    (`train eval app`) are gone now that Parts 8-10 exist."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    phony = " ".join(re.findall(r"^\.PHONY:(.*)$", makefile, re.M)).split()
    assert len(phony) == len(set(phony))
    assert set(phony) == set(tasks.TASKS)
    assert {"plots", "baseline", "check", "part3", "part4", "part5", "part6", "part7", "part8",
            "part9", "part10"} <= set(phony)  # fmt: skip
    assert not {"train", "eval", "app"} & set(phony) and "later part" not in makefile


def test_makefile_mirrors_tasks(tasks, monkeypatch):
    """Each Make target runs the same modules, in the same order, with the same options.
    A task with in-process steps (stdlib Python inside tasks.py) is delegated whole."""
    monkeypatch.setattr(tasks, "_run_dirs", lambda results: ["data/results/a_seed1"])
    rules = makefile_rules()
    for name in tasks.TASKS:
        cmds = commands(tasks, name)
        if any(isinstance(c, tasks.Step) for c in cmds):
            prereqs, recipe = rules[name]
            options = [re.sub(r"\$\(if .*$", "", line) for line in recipe]  # optional flags
            assert (prereqs, options) == ([], [f"$(PY) tasks.py {name}"]), name
            continue
        assert make_calls(rules, name) == expected_calls(tasks, name), name


def test_makefile_delegates_config_driven_tasks_with_their_options():
    """`make rl-train ALGO=dqn SEED=5 STEPS=8000 N_ENVS=4` reaches tasks.py's options."""
    (line,) = makefile_rules()["rl-train"][1]
    assert line.startswith("$(PY) tasks.py rl-train")
    for var, flag in (("ALGO", "--algo"), ("SEED", "--seed"), ("STEPS", "--steps"),
                      ("N_ENVS", "--n-envs")):  # fmt: skip
        assert f"$(if $({var}), {flag} $({var}))" in line, var
    for target in ("eta-data", "eta-data-test", "rl7-campaign", "part7"):
        (line,) = makefile_rules()[target][1]
        assert line.endswith("$(if $(ALLOW_DIRTY), --allow-dirty)"), target
    for target in ("rl7-campaign", "part7"):  # `make part7 SLOTS=6 ALLOW_DIRTY=1`
        (line,) = makefile_rules()[target][1]
        assert "$(if $(SLOTS), --slots $(SLOTS))" in line, target


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
        tasks.ETA_CONFIG,
        tasks.RL_CONFIG,
    ):
        assert (ROOT / path).is_file(), path


def test_help_lists_every_task(tasks):
    text = tasks.parser().format_help()
    assert all(name in text for name in tasks.TASKS)
    assert text.isascii()  # printable on any Windows console code page


# -- Part 2 tasks ----------------------------------------------------------------------------
def test_sim_passes_scenario_and_seeds(tasks):
    lock, cmd = commands(tasks, "sim", "--seeds", "1", "2", "3", "--scenario", "x.yaml")
    assert isinstance(lock, tasks.Step) and lock.name == "run-lock"
    assert cmd[1:] == ["-m", "traffic_twin.sim.run_scenario", "--config", "x.yaml",
                       "--seed", "1", "2", "3"]  # fmt: skip
    lock, gui = commands(tasks, "sim-gui", "--seeds", "5", "6")
    assert lock.name == "run-lock" and gui[-3:] == ["--seed", "5", "--gui"]


def test_grid_tasks_default_to_the_grid(tasks):
    _, sim = commands(tasks, "sim")
    assert sim[3:] == ["--config", GRID, "--seed", "1", "2", "3"]
    _, gui = commands(tasks, "sim-gui")
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


def modules_cmds(cmds) -> list:
    """The subprocess commands (in-process steps left out)."""
    return [c for c in cmds if isinstance(c, list)]


def plain(tasks, cmds) -> list:
    """Commands with each in-process step written as its name."""
    return [c.name if isinstance(c, tasks.Step) else c for c in cmds]


def test_composite_tasks(tasks):
    baseline = plain(tasks, commands(tasks, "baseline"))
    assert [c if isinstance(c, str) else c[2] for c in baseline] == [
        "traffic_twin.sim.network", "run-lock", "traffic_twin.sim.run_scenario",
        "traffic_twin.viz.plots",
    ]  # fmt: skip
    check = [c[2:] for c in commands(tasks, "check")]
    assert check == [["ruff", "check", "."], ["ruff", "format", "--check", "."], ["pytest", "-q"]]


def test_data_config_option(tasks):
    for cmd in commands(tasks, "data", "explore", "db-load", "calib-data", "--config", "c.yaml"):
        assert cmd[3:5] == ["--config", "c.yaml"]


# -- Part 3 tasks ----------------------------------------------------------------------------
def test_part3_tasks_default_to_pitie(tasks):
    got = {name: cmds[-1][2:] for name in ("osm", "tls-check", "field-check", "calibrate",
                                           "validate", "compare")
           for cmds in [commands(tasks, name)]}  # fmt: skip
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
        *_, cmd = commands(tasks, name, "--scenario", "x.yaml")
        assert "x.yaml" in cmd and S0 not in cmd, name
    _, cmp = commands(tasks, "compare", "--scenario", "s1.yaml", "--baseline", "s0.yaml")
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
    names = plain(tasks, cmds)
    # the run lock first (before the calibration loop), then again before compare, validate
    assert [names[i] for i in (0, 4, 6)] == ["run-lock"] * 3 and names.count("run-lock") == 3
    tls, field, calib, cmp, report, _ = [c for c in cmds if isinstance(c, list)]
    for cmd in (tls, field, calib, report):  # every step but compare runs on S0
        assert S0 in cmd and S1 not in cmd
    assert cmp[3:] == ["--baseline-config", S0, "--scenario-config", S1, "--run-missing"]


def test_part3_scenario_is_s1_and_baseline_is_s0(tasks):
    cmds = commands(tasks, "part3", "--scenario", "s1.yaml", "--baseline", "s0.yaml")
    tls, field, calib, cmp, report, _ = [c for c in cmds if isinstance(c, list)]
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
    cmds = [c for c in commands(tasks, "part3", "part4") if isinstance(c, list)]
    assert len(cmds) == len(tasks.PART3_STEPS) + len(tasks.PART4_STEPS)
    assert cmds[0][3:] == ["--config", S0] and cmds[len(tasks.PART3_STEPS)][3:] == ["--config", S2]


# -- Parts 5 and 6 tasks -----------------------------------------------------------------------
PART5_MODULES = [
    "traffic_twin.ml.daytypes", "traffic_twin.ml.eta_dataset", "traffic_twin.ml.train_eta",
    "traffic_twin.ml.train_eta", "traffic_twin.ml.eta_dataset", "traffic_twin.ml.evaluate_eta",
    "traffic_twin.ml.predict", "traffic_twin.api.bench",
]  # fmt: skip


def test_part5_task_defaults(tasks):
    got = {name: [c[2:] for c in commands(tasks, name)] for name in tasks.PART5_STEPS + ("api",)}
    assert got == {
        "eta-daytypes": [["traffic_twin.ml.daytypes", "--check"]],
        "eta-data": [["traffic_twin.ml.eta_dataset", "--config", ETA, "--splits", "train", "val",
                      "calib"]],
        "eta-train": [["traffic_twin.ml.train_eta", "--config", ETA],
                      ["traffic_twin.ml.train_eta", "--config", ETA, "--freeze"]],
        "eta-data-test": [["traffic_twin.ml.eta_dataset", "--config", ETA, "--splits", "test"]],
        "eta-eval": [["traffic_twin.ml.evaluate_eta", "--config", ETA, "--split", "test",
                      "--docs", "docs/eta.md", "--figures", "docs/figures/eta"]],
        "demo-runs": [["traffic_twin.ml.predict", "demo", "--seeds", "4000"]],
        "api-bench": [["traffic_twin.api.bench", "--results", "data/results", "data/demo",
                       "--docs", "docs/api.md"]],
        "api": [["uvicorn", "traffic_twin.api.main:app", "--no-access-log", "--host",
                 "127.0.0.1", "--port", "8000"]],
    }  # fmt: skip


def test_part5_runs_in_order_and_freezes_before_test_data(tasks):
    cmds = commands(tasks, "part5")
    assert modules(cmds) == PART5_MODULES
    assert cmds == [c for t in tasks.PART5_STEPS for c in commands(tasks, t)]
    fit_data = next(i for i, c in enumerate(cmds) if "calib" in c)
    freeze = next(i for i, c in enumerate(cmds) if "--freeze" in c)
    test_data = next(i for i, c in enumerate(cmds) if c[-2:] == ["--splits", "test"])
    assert fit_data < freeze < test_data  # D51: no test world exists before the freeze
    assert "test" not in cmds[fit_data]  # the fit splits never include the test worlds


def test_allow_dirty_reaches_only_the_dataset_commands(tasks):
    cmds = commands(tasks, "part5", "--allow-dirty")
    dirty = [c[2] for c in cmds if "--allow-dirty" in c]
    assert dirty == ["traffic_twin.ml.eta_dataset"] * 2
    assert [[a for a in c if a != "--allow-dirty"] for c in cmds] == commands(tasks, "part5")
    assert all(c[-1] == "--allow-dirty" for c in cmds if "--allow-dirty" in c)


def test_rl_train_runs_the_s6_plan_of_the_config(tasks, rl_train_scenario):
    s6 = rl_train_scenario["train"]["s6"]
    assert tasks.rl_s6_runs() == s6
    cmds = commands(tasks, "rl-train")
    expected = [(algo, str(seed), str(plan["steps"])) for algo, plan in s6.items()
                for seed in plan["seeds"]]  # fmt: skip
    assert [(c[c.index("--algo") + 1], c[c.index("--seed") + 1], c[c.index("--steps") + 1])
            for c in cmds] == expected  # fmt: skip
    assert expected == [("dqn", "0", "200000"), ("dqn", "1", "200000"), ("dqn", "2", "200000"),
                        ("ppo", "0", "200000")]  # D61  # fmt: skip
    assert all(c[2:5] == ["traffic_twin.rl.train", "--config", RL] for c in cmds)
    assert all("--n-envs" not in c for c in cmds)  # train.n_envs of the config (D61)


def test_rl_train_overrides(tasks):
    (ppo,) = commands(tasks, "rl-train", "--algo", "ppo")
    assert ppo[5:] == ["--algo", "ppo", "--seed", "0", "--steps", "200000"]
    cmds = commands(tasks, "rl-train", "--algo", "dqn", "--seed", "5", "6", "--steps", "8000",
                    "--n-envs", "4")  # fmt: skip
    assert [c[5:] for c in cmds] == [
        ["--algo", "dqn", "--seed", s, "--steps", "8000", "--n-envs", "4"] for s in ("5", "6")
    ]
    every = commands(tasks, "rl-train", "--seed", "9", "--steps", "5000")
    assert [c[6] for c in every] == ["dqn", "ppo"] and all(c[8] == "9" for c in every)


@pytest.mark.parametrize(
    "argv", [["--steps", "0"], ["--steps", "1e5"], ["--seed", "-1"], ["--n-envs", "0"]]
)
def test_rl_train_rejects_bad_numbers(tasks, argv):
    with pytest.raises(SystemExit, match="must be"):
        commands(tasks, "rl-train", *argv)


def test_rl_train_without_an_s6_plan_needs_seed_and_steps(tasks, monkeypatch):
    monkeypatch.setattr(tasks, "rl_s6_runs", lambda: {"dqn": {"seeds": [0], "steps": 10}})
    with pytest.raises(SystemExit, match="no ppo run"):
        commands(tasks, "rl-train", "--algo", "ppo")
    (cmd,) = commands(tasks, "rl-train", "--algo", "ppo", "--seed", "1", "--steps", "100")
    assert cmd[5:] == ["--algo", "ppo", "--seed", "1", "--steps", "100"]


def test_part6_task_defaults_and_order(tasks):
    got = {name: [c[2:] for c in commands(tasks, name)]
           for name in ("rl-timing", "rl-select", "rl-eval")}  # fmt: skip
    assert got == {
        "rl-timing": [["traffic_twin.rl.evaluate", "timing"]],
        "rl-select": [["traffic_twin.rl.evaluate", "select"]],
        "rl-eval": [["traffic_twin.rl.evaluate", "compare", "--seeds", *DEVTEST],
                    ["traffic_twin.rl.evaluate", "kpi", "--seeds", *DEVTEST],
                    ["traffic_twin.rl.evaluate", "docs", "--docs", "docs/rl.md",
                     "--figures", "docs/figures/rl"]],
    }  # fmt: skip
    cmds = commands(tasks, "part6")
    assert cmds == [c for t in tasks.PART6_STEPS for c in commands(tasks, t)]
    assert [c[3] if c[2] == "traffic_twin.rl.evaluate" else "train" for c in cmds] == [
        "timing", "train", "train", "train", "train", "select", "compare", "kpi", "docs",
    ]  # fmt: skip


@pytest.mark.parametrize("name", ["part5", "part6", "demo-runs", "rl-eval", "eta-data"])
def test_pre_registered_tasks_ignore_scenario_and_seed_options(tasks, name):
    """The Parts 5-6 seeds, splits and configs are pre-registered (D38): `--scenario`,
    `--baseline`, `--seeds` and `--config` never reach them."""
    assert commands(tasks, name, "--scenario", "x.yaml", "--baseline", "y.yaml", "--seeds", "1",
                    "--config", "c.yaml") == commands(tasks, name), name  # fmt: skip


def test_parts_5_6_seeds_are_the_pre_registered_ones(tasks, rl_train_scenario):
    """Demo seed 4000 inside api.yaml seeds.demo; the dev-test seeds inside rl.yaml
    seeds.dev_test; no Parts 5-6 task passes an evaluation (1-10) or calibration
    (101-103) seed (D38)."""
    import yaml

    api = yaml.safe_load((ROOT / "configs/api.yaml").read_text(encoding="utf-8"))
    lo, hi = api["seeds"]["demo"]
    assert all(lo <= int(s) <= hi for s in tasks.DEMO_SEEDS)
    lo, hi = rl_train_scenario["seeds"]["dev_test"]
    assert all(lo <= int(s) <= hi for s in tasks.RL_DEVTEST_SEEDS)
    assert tasks.RL_DEVTEST_SEEDS == tuple(DEVTEST)  # Part 6 uses 3020-3029 (D38, D60)
    forbidden = {str(s) for s in [*range(1, 11), 101, 102, 103]}
    for name in (*tasks.PART5_STEPS, *tasks.PART6_STEPS):
        for cmd in commands(tasks, name):
            if "--seeds" in cmd:
                seeds = [s for s in cmd[cmd.index("--seeds") + 1 :] if s.isdigit()]
                assert seeds and not forbidden & set(seeds), (name, seeds)


# Import names of the core dependencies (pyproject `dependencies`): never optional
CORE_IMPORTS = frozenset({"pandas", "pyarrow", "yaml", "requests", "matplotlib", "numpy"})


def optional_dependency_missing(exc: ModuleNotFoundError) -> bool:
    """A module missing because an optional extra (`ml`, `api`, `rl`, or one of their own
    dependencies such as pydantic or starlette) is not installed: skip, as the tests of
    those modules do. A missing project module or core dependency is a real failure."""
    top = (exc.name or "").partition(".")[0]
    return bool(top) and top != "traffic_twin" and top not in CORE_IMPORTS


def test_optional_dependency_missing_covers_every_optional_extra():
    """`api-bench` imports pydantic before fastapi: every module of the optional extras
    (and their own dependencies) must skip, not only a hand-written list."""
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = extras["project"]["dependencies"]
    core = {re.match(r"[\w-]+", d).group(0).lower().replace("pyyaml", "yaml") for d in deps}
    assert core == CORE_IMPORTS
    optional = ("pydantic", "starlette", "uvicorn", "duckdb", "httpx", "fastapi", "xgboost",
                "torch", "stable_baselines3", "gymnasium", "pydantic_core.core_schema")  # fmt: skip
    for name in optional:
        assert optional_dependency_missing(ModuleNotFoundError(name=name)), name
    for name in ("traffic_twin.api.schemas", "pandas", "yaml", "numpy.typing", None):
        assert not optional_dependency_missing(ModuleNotFoundError(name=name)), name


class _Parsed(Exception):
    """Raised instead of running a CLI once its top-level parser has parsed argv."""


def parse_only(monkeypatch, module_name: str, argv: list[str]) -> tuple:
    """(namespace, extra args) of `python -m module_name *argv`, without running it: the
    first top-level argparse parse raises _Parsed, whatever the module calls (parse_args,
    parse_known_args, subparsers inside)."""
    original = argparse.ArgumentParser.parse_known_args
    depth = [0]

    def parse_known_args(self, args=None, namespace=None):
        depth[0] += 1
        try:
            result = original(self, args, namespace)
        finally:
            depth[0] -= 1
        if depth[0] == 0:
            raise _Parsed(result)
        return result

    monkeypatch.setattr(argparse.ArgumentParser, "parse_known_args", parse_known_args)
    monkeypatch.setattr(sys, "argv", [module_name, *argv])
    try:
        module = importlib.import_module(module_name)
        if not hasattr(module, "main"):  # a package: `python -m pkg` runs pkg/__main__.py
            module = importlib.import_module(f"{module_name}.__main__")
    except ModuleNotFoundError as exc:
        if optional_dependency_missing(exc):
            pytest.skip(f"{module_name} needs {exc.name}, not installed here")
        raise
    try:
        module.main(argv)
    except _Parsed as parsed:
        return parsed.args[0]
    except NotImplementedError as exc:
        pytest.fail(f"{module_name} CLI is still a stub ({exc}): its workstream implements it")
    pytest.fail(f"{module_name}.main returned without parsing its arguments with argparse")


PARTS_5_6_CLI = [
    (name, i)
    for name in ("eta-daytypes", "eta-data", "eta-train", "eta-data-test", "eta-eval",
                 "demo-runs", "api-bench", "rl-timing", "rl-train", "rl-select", "rl-eval")
    for i in range({"eta-train": 2, "rl-train": 4, "rl-eval": 3}.get(name, 1))
]  # fmt: skip


@pytest.mark.parametrize(("name", "i"), PARTS_5_6_CLI)
def test_parts_5_6_commands_parse_with_their_cli(tasks, monkeypatch, name, i):
    """Every argument a Parts 5-6 task passes is one its module's own parser accepts, so
    tasks.py and the Makefile cannot drift from the CLIs. Nothing runs: the parse is
    intercepted before the module does any work."""
    cmd = commands(tasks, name)[i]
    namespace, extra = parse_only(monkeypatch, cmd[2], cmd[3:])
    assert extra == [], f"{name}: {cmd[2]} does not accept {extra}"
    values = vars(namespace)
    if "--config" in cmd:
        assert Path(values["config"]) == Path(cmd[cmd.index("--config") + 1])
    if "--seeds" in cmd:
        given = [int(s) for s in cmd[cmd.index("--seeds") + 1 :] if s.isdigit()]
        assert [int(s) for s in values["seeds"]] == given


def test_part5_part6_dry_run(tasks, capsys):
    assert tasks.main(["part5", "part6", "-n"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == len(commands(tasks, "part5")) + len(commands(tasks, "part6"))
    assert all(line.startswith("+ ") for line in out)


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
    for old in (p for p in backup.iterdir() if p.is_dir()):  # not verify.json
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


def test_rebaseline_masks_each_runs_own_run_id(tasks, results, capsys):
    """kpi_run.csv carries the run's uuid run_id in every row, and a re-made run gets a new
    one: the rows are compared with each run's own run_id masked, nothing else."""
    root, _ = results
    assert tasks.rebaseline_backup() == 0
    backup = root / tasks.REBASELINE_DIR / "abc1234"
    for old in backup.iterdir():  # moved rows name their own run_id; the re-made rows theirs
        write_run(old, "abc1234-dirty", kpis=f"run_id,kpi\nid-abc1234-dirty,{old.name}\n")
        write_run(root / tasks.RESULTS / old.name, "abc1234",
                  kpis=f"run_id,kpi\nid-abc1234,{old.name}\n")  # fmt: skip
    for doc in tasks.PUBLISHED_DOCS:
        (root / doc).write_text("| Code version (git) | abc1234 |\n", encoding="utf-8")
    assert tasks.rebaseline_verify() == 0
    assert "6 of 6 re-made runs identical" in capsys.readouterr().out
    # a row naming another run's id is a real difference
    write_run(root / "data/results/pitie_fixed_seed1", "abc1234",
              kpis="run_id,kpi\nid-other,pitie_fixed_seed1\n")  # fmt: skip
    assert tasks.rebaseline_verify() == 1
    assert "pitie_fixed_seed1: kpi_run.csv differs" in capsys.readouterr().err


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
    lock, first, *middle, last = cmds
    assert (lock.name, first.name, last.name) == ("run-lock", "rebaseline-backup",
                                                  "rebaseline-verify")  # fmt: skip
    expected = commands(tasks, *tasks.REBASELINE_STEPS)
    assert plain(tasks, middle) == plain(tasks, expected)
    assert plain(tasks, middle)[0] == "run-lock" and plain(tasks, middle)[2] == "run-lock"
    assert modules(middle) == ["traffic_twin.evaluation.compare",
                               "traffic_twin.validation.report", *PART4_MODULES]  # fmt: skip
    assert all("--run-missing" in c for c in modules_cmds(middle)
               if "compare" in c[2] or "report" in c[2])  # fmt: skip
    assert "x.yaml" not in str(cmds) and "y.yaml" not in str(cmds)


def test_steps_print_and_dry_run(tasks, capsys):
    assert tasks.main(["rebaseline", "-n"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:2] == ["+ tasks.py:run-lock (in-process)",
                       "+ tasks.py:rebaseline-backup (in-process)"]  # fmt: skip
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
    assert len(out) == 3 and all(line.startswith("+ ") for line in out)
    assert out[0] == "+ tasks.py:run-lock (in-process)"


def test_part3_dry_run(tasks, capsys):
    assert tasks.main(["part3", "-n"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == len(tasks.PART3_STEPS) + 3 and all(line.startswith("+ ") for line in out)
    # the run lock of the chain, then those of compare and validate
    assert out[0] == out[4] == out[6] == "+ tasks.py:run-lock (in-process)"


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


# -- install (the documented install-then-check path) --------------------------------------------
def test_install_installs_every_extra_the_suite_and_the_tasks_use(tasks):
    """`tasks.py install` then `tasks.py check` (README Quick start, "what CI runs") must
    give the suite its optional extras: at least every extra CI installs, plus `ml` for
    `part5`. PyTorch comes first from the CPU index, at the `rl` extra's pin, as in CI."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    assert set(tasks.INSTALL_EXTRAS) == set(extras)  # every extra of the package
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    (ci_extras,) = re.findall(r'pip install -e "\.\[([\w,]+)\]"', ci)
    assert set(ci_extras.split(",")) <= set(tasks.INSTALL_EXTRAS)
    assert {"ml", "api", "rl"} <= set(tasks.INSTALL_EXTRAS)
    assert tasks.TORCH_PIN in extras["rl"]
    assert f"pip install {tasks.TORCH_PIN} --index-url {tasks.TORCH_CPU_INDEX}" in ci
    torch, package = [c[3:] for c in commands(tasks, "install")]
    assert torch == ["install", tasks.TORCH_PIN, "--index-url", tasks.TORCH_CPU_INDEX]
    assert package == ["install", "-e", f".[{','.join(tasks.INSTALL_EXTRAS)}]"]
    # the Makefile runs the very same two commands
    assert makefile_rules()["install"][1] == [
        f"$(PY) -m pip install {tasks.TORCH_PIN} --index-url {tasks.TORCH_CPU_INDEX}",
        '$(PY) -m pip install -e ".[' + ",".join(tasks.INSTALL_EXTRAS) + ']"',
    ]


# -- reproduction of Parts 5 and 6 after a commit (D65) ----------------------------------------
def test_reproduce56_chain(tasks):
    """Move aside, rerun `part5 part6` with every default (no development dataset, the
    train.s6 plan unchanged, nothing forced), then compare."""
    cmds = commands(tasks, "reproduce56", "--allow-dirty", "--algo", "ppo", "--seed", "7",
                    "--steps", "100", "--n-envs", "2")  # fmt: skip
    first, *middle, last = cmds
    assert (first.name, last.name) == ("reproduce56-backup", "reproduce56-verify")
    assert tasks.REPRODUCE_STEPS == ("part5", "part6")
    assert middle == commands(tasks, "part5") + commands(tasks, "part6")
    assert not any(a in ("--allow-dirty", "--force", "--refreeze") for c in middle for a in c)


def test_reproduce56_output_roots_are_the_configs_outputs(tasks):
    """The moved-aside roots are where `part5 part6` write (eta.yaml / rl.yaml outputs,
    paths.DEMO_DIR), so the rerun finds no freeze and no finished run of its own."""
    import yaml

    from traffic_twin import paths

    eta = yaml.safe_load((ROOT / ETA).read_text(encoding="utf-8"))["outputs"]
    rl = yaml.safe_load((ROOT / RL).read_text(encoding="utf-8"))["outputs"]
    assert Path(eta["data"]) == tasks.ETA_DATA_DIR == paths.ETA_DIR
    assert Path(eta["models"]) == tasks.ETA_MODELS_DIR
    assert Path(rl["runs"]) == tasks.RL_RUNS_DIR == paths.RL_RUNS_DIR
    assert Path(rl["selected"]) == tasks.RL_SELECTED_DIR
    assert tasks.DEMO_DIR == paths.DEMO_DIR
    for d in (tasks.ETA_DATA_DIR, tasks.ETA_MODELS_DIR, tasks.RL_SELECTED_DIR, tasks.DEMO_DIR,
              tasks.RL_RUNS_DIR):  # fmt: skip
        assert any(d == p or p in d.parents for p in tasks.REPRODUCE_PARENTS), d
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert all(f"{p.as_posix()}/*" in gitignore for p in tasks.REPRODUCE_PARENTS)  # backups too


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


DEVTEST_TABLES = ("compare.csv", "kpi.csv", "kpi_safety.csv")


def make_parts56(root: Path, version: str, *, dev: bool, change: str | None = None) -> None:
    """A fake Phase 3 output tree of `part5 part6`, made from `version`. A clean tree
    (`dev` False) drops the `-dev` suffix, so dataset id, model version and bundle sha256
    change, as do code versions, timestamps, timings and the SB3 zip sha256. `change`
    names one real difference to plant."""
    dataset = "eta1-ce9b6451" + ("-dev" if dev else "")
    model_version = f"eta1-{'56926796' if dev else '0badf00d'}"
    _write_json(root / "data/eta" / dataset / "manifest.json", {
        "dataset_id": dataset, "code_version": version, "created_at": version,
        "builds": [{"wall_s": 1.0 if dev else 2.0}], "n_samples": 10,
        "counts": {"train": {"fixed": {"N_S": 4 if change != "counts" else 5}}},
    })  # fmt: skip
    (root / "data/eta" / dataset / "samples").mkdir()
    models = root / "data/models/eta"
    bundle = models / model_version
    bundle.mkdir(parents=True)
    (bundle / "model.ubj").write_bytes(b"model" + (b"!" if change == "model" else b""))
    (bundle / "reference.parquet").write_bytes(b"reference")
    _write_json(bundle / "meta.json", {
        "model_version": model_version, "dataset_id": dataset, "code_version": version,
        "dataset_code_version": version, "created_at": version, "qhat": -0.5,
        "grid": [{"max_depth": 6, "val_pinball": 3.69, "fit_s": 1.0 if dev else 2.0}],
        "params": {"max_depth": 8}, "best_iteration": 5999, "model_sha256": "m",
    })  # fmt: skip
    freeze = {"model_version": model_version, "frozen_at": version, "code_version": version}
    _write_json(bundle / "eval.json", {
        "a07": {"coverage": 0.806 if change != "coverage" else 0.807, "pass": True},
        "rmse": 26.4, "bias": float("nan"), "dataset_id": dataset,
        "model_version": model_version, "bundle_sha256": f"b-{version}",
        "code_version": version, "freeze": freeze,
    })  # fmt: skip
    _write_json(models / "freeze.json", freeze)
    (models / "LATEST").write_text(model_version + "\n", encoding="utf-8")
    zip_sha = f"zip-{version}"
    policy = "p" + ("x" if change == "policy" else "")
    _write_json(root / "data/models/rl/selected/manifest.json", {
        "model_version": "rl1-dqn-s0-3e1539b9", "policy_sha256": policy,
        "model_sha256": zip_sha, "code_version": version, "selected_at": version,
        "best_val_cost": 74453.34,
    })  # fmt: skip
    (root / "data/models/rl/selected/model.zip").write_bytes(zip_sha.encode())
    write_run(root / "data/demo/pitie_rules_seed4000", version, kpis="kpi,value\nx,1\n")
    for algo in ("dqn", "ppo"):
        _write_json(root / f"data/rl_runs/{algo}_s0_200k_c0b7dd7b/manifest.json", {
            "run": f"{algo}_s0_200k_c0b7dd7b", "seed": 0, "steps_done": 200000,
            "code_version": version, "dirty": dev, "started_at": version,
            "wall_s": 872.2 if dev else 801.0, "steps_per_s": 321.6 if dev else 350.0,
            "best_model_sha256": zip_sha, "best_policy_sha256": policy, "best_step": 200000,
            "baselines": {"S0": 86483.88}, "argv": [version],
        })  # fmt: skip
    dev_root = root / "data/rl_runs/devtest"
    dev_root.mkdir(parents=True, exist_ok=True)
    for name in DEVTEST_TABLES:
        rows = (
            "label,cost\nRL,1.0\n"
            if change == "devtest" and name == "kpi.csv"
            else "label,cost\nRL,68555.2\n"
        )
        (dev_root / name).write_text(rows, encoding="utf-8")
    for name in ("compare_summary.json", "kpi_summary.json"):
        _write_json(dev_root / name, {"created_at": version, "code_version": version,
                                      "dirty": dev, "summary": {"RL": {"cost_mean": 68555.2}},
                                      "model": {"model_sha256": zip_sha}})  # fmt: skip
    write_run(dev_root / "pitie_rl_seed3020", version, kpis="kpi,value\nrl,2\n",
              control={"model_version": "rl1-dqn-s0-3e1539b9", "model_sha256": zip_sha,
                       "decisions": 821})  # fmt: skip
    _write_json(root / "data/rl_runs/timing/timing.json", {"wall_s": 1.0 if dev else 2.0})
    (root / "docs").mkdir(exist_ok=True)
    for doc in ("docs/eta.md", "docs/api.md", "docs/rl.md"):
        (root / doc).write_text(f"code `{version}` (`-dirty` for an uncommitted tree)\n",
                                encoding="utf-8")  # fmt: skip


@pytest.fixture
def parts56(tasks, tmp_path, monkeypatch):
    """The Phase 3 outputs, made from `abc1234-dirty`; HEAD abc1234, clean; `rl.evaluate
    repro` stubbed (its own checks are tested in test_rl_evaluate.py)."""
    monkeypatch.setattr(tasks, "ROOT", tmp_path)
    head = {"value": ("abc1234", False)}
    monkeypatch.setattr(tasks, "git_head", lambda: head["value"])
    repro = {"code": 0, "calls": []}
    monkeypatch.setattr(
        tasks, "call", lambda cmd, env=None: repro["calls"].append(cmd) or repro["code"]
    )
    make_parts56(tmp_path, "abc1234-dirty", dev=True)
    for keep in ("data/rl_runs/_work", "data/rl_runs/_repro_a/dqn_s5_8k_c0b7dd7b"):
        (tmp_path / keep).mkdir(parents=True)
    _write_json(tmp_path / "data/rl_runs/repro.json", {"same": True})
    for parent in tasks.REPRODUCE_PARENTS:
        (tmp_path / parent / ".gitkeep").write_text("", encoding="utf-8")
    return tmp_path, head, repro


def test_reproduce56_backup_refuses_a_dirty_tree_and_an_empty_one(
    tasks, parts56, tmp_path, monkeypatch
):
    root, head, _ = parts56
    head["value"] = ("abc1234", True)
    assert tasks.reproduce_backup() == 1
    assert (root / "data/models/eta/freeze.json").is_file()  # nothing moved
    head["value"] = ("abc1234", False)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(tasks, "ROOT", empty)
    assert tasks.reproduce_backup() == 1


def test_reproduce56_backup_clears_the_way_for_part5_and_part6(tasks, parts56):
    """Regression (D65): on top of its own outputs `part5` stops at the freeze (a second
    model needs --refreeze) and `part6` at the first finished run folder. The backup
    moves every output of the chains, and only those."""
    root, _, _ = parts56
    assert tasks.reproduce_backup() == 0
    assert not (root / "data/models/eta/freeze.json").exists()
    assert not (root / "data/models/eta/LATEST").exists()
    assert not list((root / "data/eta").glob("eta1-*"))
    assert not tasks._rl_run_dirs(root / "data/rl_runs")
    assert not (root / "data/rl_runs/devtest").exists()
    assert not (root / "data/models/rl/selected").exists()
    assert not (root / "data/demo/pitie_rules_seed4000").exists()
    for kept in ("data/rl_runs/_work", "data/rl_runs/_repro_a", "data/rl_runs/repro.json",
                 "data/eta/.gitkeep", "data/models/.gitkeep"):  # fmt: skip
        assert (root / kept).exists(), kept
    backup = root / "data/models/_reproduce/abc1234"
    assert (backup / "eta/freeze.json").is_file() and (backup / "rl/selected").is_dir()
    assert (root / "data/rl_runs/_reproduce/abc1234/dqn_s0_200k_c0b7dd7b/manifest.json").is_file()
    assert (root / "data/eta/_reproduce/abc1234/eta1-ce9b6451-dev/manifest.json").is_file()
    assert tasks.reproduce_backup() == 1  # the originals are kept, never overwritten


def test_reproduce56_accepts_a_faithful_reproduction(tasks, parts56, capsys):
    """The clean rebuild changes dataset id, model version, bundle sha256, code versions,
    timestamps, timings and the zip sha256: none of them is compared."""
    root, _, repro = parts56
    assert tasks.reproduce_backup() == 0
    make_parts56(root, "abc1234", dev=False)
    assert tasks.reproduce_verify() == 0, capsys.readouterr().err
    out = capsys.readouterr().out
    assert "eta bundle eta1-56926796 -> eta1-0badf00d: same" in out
    assert "eta dataset eta1-ce9b6451-dev -> eta1-ce9b6451: same" in out
    report = json.loads(
        (root / "data/models/_reproduce/abc1234/reproduce56.json").read_text(encoding="utf-8")
    )
    assert report["same"] and not any(report["items"].values())
    assert {"rl run dqn_s0_200k_c0b7dd7b", "rl run ppo_s0_200k_c0b7dd7b", "rl selection",
            "rl devtest tables", "rl devtest run pitie_rl_seed3020",
            "demo run pitie_rules_seed4000"} <= set(report["items"])  # fmt: skip
    called = [(c[2:4], c[4], c[5]) for c in repro["calls"]]
    assert called == [
        (["traffic_twin.rl.evaluate", "repro"], f"data/rl_runs/_reproduce/abc1234/{r}",
         f"data/rl_runs/{r}") for r in ("dqn_s0_200k_c0b7dd7b", "ppo_s0_200k_c0b7dd7b")
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("change", "item", "what"),
    [
        ("coverage", "eta bundle", "eval.json: a07.coverage"),
        ("model", "eta bundle", "model.ubj sha256 differs"),
        ("counts", "eta dataset", "manifest.json: counts.train.fixed.N_S"),
        ("policy", "rl run dqn_s0", "manifest.json: best_policy_sha256"),
        ("policy", "rl selection", "manifest.json: policy_sha256"),
        ("devtest", "rl devtest tables", "kpi.csv differs"),
    ],
)
def test_reproduce56_fails_on_a_real_difference(tasks, parts56, capsys, change, item, what):
    root, _, _ = parts56
    assert tasks.reproduce_backup() == 0
    make_parts56(root, "abc1234", dev=False, change=change)
    assert tasks.reproduce_verify() == 1
    err = [line for line in capsys.readouterr().err.splitlines() if item in line]
    assert err and what in err[0], err


def test_reproduce56_fails_when_rl_repro_or_a_page_differs(tasks, parts56, capsys):
    root, _, repro = parts56
    assert tasks.reproduce_backup() == 0
    make_parts56(root, "abc1234", dev=False)
    repro["code"] = 1  # policy parameters, episode returns or validation costs differ
    assert tasks.reproduce_verify() == 1
    assert "rl.evaluate repro: differences" in capsys.readouterr().err
    repro["code"] = 0
    (root / "docs/rl.md").write_text("Code versions of the runs: abc1234-dirty.\n")
    assert tasks.reproduce_verify() == 1
    assert "page docs/rl.md: DIFFERENT: names abc1234-dirty" in capsys.readouterr().err
    shutil.rmtree(root / "data/rl_runs/ppo_s0_200k_c0b7dd7b")
    assert tasks.reproduce_verify() == 1
    assert "rl run ppo_s0_200k_c0b7dd7b: DIFFERENT: not re-made" in capsys.readouterr().err


def test_reproduce56_dry_run_prints_its_steps(tasks, capsys):
    assert tasks.main(["reproduce56", "-n"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "+ tasks.py:reproduce56-backup (in-process)"
    assert out[-1] == "+ tasks.py:reproduce56-verify (in-process)"
    assert len(out) == 2 + len(commands(tasks, "part5", "part6"))


def test_json_diff(tasks):
    a = {"x": 1, "nan": float("nan"), "l": [1, {"t": 1, "keep": 2}], "skip": 1}
    assert tasks.json_diff(a, json.loads(json.dumps(a))) == []
    b = {"x": 1.0, "nan": float("nan"), "l": [1, {"t": 9, "keep": 3}], "skip": 2, "new": 0}
    assert tasks.json_diff(a, b, frozenset({"skip", "t"})) == ["l[1].keep", "new", "x"]
    assert tasks.json_diff([1, 2], [1]) == ["<root>"]


# -- Part 7 tasks (pre-registered D68-D90, configs/rl7.yaml) ------------------------------------
RL7 = "configs/rl7.yaml"
STUDY, SENS = "traffic_twin.rl.study", "traffic_twin.rl.sensitivity"
RL7_SINGLE = ("rl7-campaign", "rl7-select", "rl7-publish", "rl7-stress", "rl7-sens", "rl7-devtest",
              "rl7-docs")  # fmt: skip


def rl7_yaml() -> dict:
    import yaml

    return yaml.safe_load((ROOT / RL7).read_text(encoding="utf-8"))


def test_part7_task_defaults(tasks):
    got = {name: [c[2:] for c in commands(tasks, name)] for name in RL7_SINGLE}
    assert got == {
        "rl7-campaign": [["traffic_twin.rl.campaign", "--config", RL7, "--stages", "T", "S", "F"]],
        "rl7-select": [[STUDY, "select"]],
        "rl7-publish": [[STUDY, "publish"]],
        "rl7-stress": [[STUDY, "stress"]],
        "rl7-sens": [[SENS, "run"], [SENS, "tables"]],
        "rl7-devtest": [[STUDY, "devtest"]],
        "rl7-docs": [["traffic_twin.rl.report7", "--docs", "docs/rl_training.md",
                      "--figures", "docs/figures/rl7"]],
    }  # fmt: skip
    ((step,),) = [commands(tasks, "rl7-holdout")]
    assert isinstance(step, tasks.Step) and step.name == "rl7-holdout"


def test_rl7_campaign_options(tasks):
    (cmd,) = commands(tasks, "rl7-campaign", "--slots", "6", "--allow-dirty")
    assert cmd[3:] == ["--config", RL7, "--stages", "T", "S", "F", "--slots", "6",
                       "--allow-dirty"]  # fmt: skip
    for bad in ("0", "x", "-2"):
        with pytest.raises(SystemExit, match="--slots must be"):
            commands(tasks, "rl7-campaign", "--slots", bad)


def test_rl7_constants_are_the_configs_and_paths(tasks):
    """tasks.py's Part 7 paths are rl7.yaml's `outputs`, paths.py's constants and the
    contract's file names; the D82 run is stage-T `t1` seed 0 at the stage-T budget."""
    import yaml

    from traffic_twin import paths
    from traffic_twin.rl import campaign, study

    rl7 = rl7_yaml()
    out, protocol = rl7["outputs"], rl7["protocol"]
    assert (
        Path(tasks.RL7_CONFIG) == paths.RL7_CONFIG
        and Path(tasks.RL7_CONFIG_DIR) == paths.RL7_CONFIG_DIR
    )
    assert Path(tasks.PITIE_RL_S3) == paths.PITIE_RL_S3_CONFIG
    assert Path(out["study"]) == tasks.RL7_DIR == paths.RL7_DIR
    assert Path(out["runs"]) == tasks.RL7_RUNS_DIR
    assert out["docs"] == tasks.RL7_DOC and out["figures"] == tasks.RL7_FIGURES
    assert tuple(protocol["devtest"]["seeds"]) == tasks.RL7_DEVTEST_RANGE
    assert tasks.RL7_STAGES == campaign.STAGES and tasks.RL7_GOLDEN == study.GOLDEN_FILE
    run, stage_t = tasks.REPRODUCE7_RUN, protocol["stage_t"]
    assert run["variant"] == stage_t["reference"] and run["seed"] in stage_t["seeds"]
    assert run["steps"] == stage_t["steps"] and run["stage"] == "T"
    t1 = yaml.safe_load((ROOT / run["config"]).read_text(encoding="utf-8"))
    assert t1["variant"] == run["variant"] and t1.get("algo", "dqn") == run["algo"]
    for parent in tasks.REPRODUCE_PARENTS:  # reproduce56 never moves Part 7 outputs aside
        lineage = {parent, *parent.parents}
        assert tasks.RL7_DIR not in lineage and Path("models") not in lineage, parent
        assert parent not in tasks.RL7_DIR.parents, parent
    assert "data/rl7/*" in (ROOT / ".gitignore").read_text(encoding="utf-8").split()


def test_part7_chain_order_and_stop_before_s3(tasks):
    cmds = commands(tasks, "part7")
    names = [c.name if isinstance(c, tasks.Step) else c[2:4] for c in cmds]
    assert names == [
        ["traffic_twin.rl.campaign", "--config"], "rl7-holdout", [STUDY, "select"],
        [STUDY, "publish"], "part7-s3-config", [STUDY, "stress"], [SENS, "run"], [SENS, "tables"],
        [STUDY, "devtest"], ["traffic_twin.rl.report7", "--docs"],
    ]  # fmt: skip
    assert tasks.PART7_STEPS == ("rl7-campaign", "rl7-holdout", "rl7-select", "rl7-publish",
                                 "rl7-stress", "rl7-sens", "rl7-devtest", "rl7-docs")  # fmt: skip
    plain = [c for c in cmds if not isinstance(c, tasks.Step)]
    expected = [c for t in tasks.PART7_STEPS for c in commands(tasks, t)
                if not isinstance(c, tasks.Step)]  # fmt: skip
    assert plain == expected
    # --slots and --allow-dirty reach the campaign only
    opts = commands(tasks, "part7", "--slots", "6", "--allow-dirty")
    assert opts[0][-3:] == ["--slots", "6", "--allow-dirty"]
    assert [c for c in opts[1:] if not isinstance(c, tasks.Step)] == plain[1:]


@pytest.mark.parametrize("name", ["part7", "rl7-campaign", "rl7-sens", "rl7-devtest", "rl7-docs"])
def test_part7_tasks_ignore_scenario_seed_and_rl6_options(tasks, name):
    """Part 7 seeds, configs and budgets are pre-registered in configs/rl7.yaml (D70): no
    option of the earlier parts reaches them."""

    def plain(cmds):
        return [c if not isinstance(c, tasks.Step) else c.name for c in cmds]

    other = commands(tasks, name, "--scenario", "x.yaml", "--baseline", "y.yaml", "--seeds", "1",
                     "--config", "c.yaml", "--algo", "ppo", "--seed", "7", "--steps", "10",
                     "--n-envs", "2")  # fmt: skip
    assert plain(other) == plain(commands(tasks, name))


def test_part7_dry_run(tasks, capsys):
    assert tasks.main(["part7", "-n"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == len(commands(tasks, "part7")) and all(line.startswith("+ ") for line in out)
    assert "+ tasks.py:part7-s3-config (in-process)" in out
    assert out.index("+ tasks.py:part7-s3-config (in-process)") == 4  # right after rl7-publish


@pytest.fixture
def rl7_tree(tasks, tmp_path, monkeypatch):
    """A fake repo for the Part 7 steps: ROOT in tmp_path, git HEAD abc1234 (clean), no
    OMP_* / MKL_* / OPENBLAS_* variable in this environment (a default shell), the
    subprocess calls and their environments recorded (and answered by `calls["fn"](cmd,
    env)` when set)."""
    monkeypatch.setattr(tasks, "ROOT", tmp_path)
    for k in [k for k in os.environ if k.startswith(tasks.RL7_ENV_PREFIXES)]:
        monkeypatch.delenv(k)
    head = {"value": ("abc1234", False)}
    monkeypatch.setattr(tasks, "git_head", lambda: head["value"])
    calls = {"cmds": [], "envs": [], "fn": None, "code": 0}

    def call(cmd, env=None):
        calls["cmds"].append(cmd)
        calls["envs"].append(env)
        if calls["fn"] is not None:
            calls["fn"](cmd, env)
        return calls["code"]

    monkeypatch.setattr(tasks, "call", call)
    return tmp_path, head, calls


CAMPAIGN_ENV = {"OPENBLAS_NUM_THREADS": "1"}  # what rl.campaign.train_env() adds


def launch_record(env: dict | None, eval_pool: str = "persistent") -> dict:
    """The launch keys rl.train records in provenance.json (and a Part 7 manifest) for a
    learner started in `env` (None: this process's): train.machine_info() `env` and
    `child_env`, and `eval_pool` (6 eval workers)."""
    env = dict(os.environ) if env is None else env
    return {
        "machine": {
            "cpu": "cpu", "cpu_count": 32,
            "env": {k: v for k, v in sorted(env.items()) if k.startswith(("OMP_", "MKL_",
                                                                            "OPENBLAS_"))},
            "child_env": {k: env.get(k, v) for k, v in CAMPAIGN_ENV.items()},
        },
        "eval_pool": eval_pool,
    }  # fmt: skip


def make_run(run_dir: Path, version: str = "abc1234", *, timing: float = 1.0,
             change: str | None = None, launch: dict | None = None) -> None:  # fmt: skip
    """A finished Part 7 run folder made from `version`, with `timing` in its wall-time
    fields; `change` plants one real difference; `launch` (launch_record) the launch keys,
    by default those of a campaign launch (CAMPAIGN_ENV, persistent pool)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    launch = launch_record(CAMPAIGN_ENV) if launch is None else launch
    zip_sha = f"zip-{version}-{timing}"
    _write_json(run_dir / "manifest.json", {
        "run": run_dir.name, "algo": "dqn", "seed": 0, "steps": 400000, "steps_done": 400000,
        "best_step": 375000, "best_val_cost": 70000.5,
        "best_policy_sha256": "best",
        "final_policy_sha256": "final" + ("x" if change == "final" else ""),
        "best_model_sha256": zip_sha, "final_model_sha256": zip_sha, "model_bytes": 460000,
        "code_version": version, "dirty": version.endswith("-dirty"), "started_at": version,
        "wall_s": 1200.0 * timing, "steps_per_s": 380.0 / timing, "argv": [version],
        "config": f"{'' if version.endswith('-dirty') else 'c:/repo/'}configs/rl7/t1.yaml",
        **launch,
    })  # fmt: skip
    _write_json(run_dir / "provenance.json", {
        "run_config_sha": "abcd1234" * 8, "variant": "t1",
        "allow_dirty": version.endswith("-dirty"),
        "pick_step": 375000 if change != "pick" else 400000, "pick_val_cost": 70000.5,
        "pick_policy_sha256": "pick", "pick_model_sha256": zip_sha, "late_mean": 71000.25,
        "code_version": version, "created_at": version,
        "torch": {"threads_learner": 1, "threads_worker": 1},
        **launch,
    })  # fmt: skip
    cost = "70000.5" if change != "validation" else "70001.5"
    (run_dir / "validation.csv").write_text(
        f"step,cost,cost_vs_S1,pedestrian_s,eval_s\n200000,72000.0,-8.1,1500.0,{7.5 * timing}\n"
        f"375000,{cost},-9.9,1490.0,{6.5 * timing}\n",
        encoding="utf-8",
    )
    reward = "-75.5" if change != "monitor" else "-75.4"
    (run_dir / "monitor.csv").write_text(
        f'#{{"t_start": {timing}, "env_id": "None"}}\nr,l,t\n-80.25,1800,{3.0 * timing}\n'
        f"{reward},1800,{6.0 * timing}\n",
        encoding="utf-8",
    )


RUN_NAME = "dqn_t1_s0_400k_abcd1234"
PAGE = ("# RL training (Part 7)\n\nCode version of the runs: `{v}`.\n\n"
        "![Learning curves]({fig}/learning_bands.png)\n\n| Run | Pick |\n| --- | --- |\n"
        "| t1 s0 | 375000 |\n")  # fmt: skip
GOLDEN = {"q_values": [[0.125, -0.5], [1.0, 2.25]], "actions": [1, 0], "phase_hash": "ph"}


@pytest.fixture
def rl7_reproducible(tasks, rl7_tree):
    """The Phase 3 outputs reproduce7 needs, made from `abc1234-dirty` (an --allow-dirty
    campaign), and a stand-in for the three reruns that makes a faithful reproduction
    from HEAD abc1234 (`change` in the returned dict plants one difference)."""
    root, head, calls = rl7_tree
    make_run(root / "data/rl7/runs/T" / RUN_NAME, "abc1234-dirty")
    make_run(root / "data/rl7/runs/T/dqn_t1_s1_400k_abcd1234", "abc1234-dirty")  # another seed
    (root / "configs").mkdir()
    (root / tasks.PITIE_RL_S3).write_text("extends: pitie_rl.yaml\n", encoding="utf-8")
    _write_json(root / tasks.RL7_GOLDEN, {**GOLDEN, "created_at": "t0", "code_version": "x-dirty"})
    (root / "docs").mkdir()
    (root / tasks.RL7_DOC).write_text(
        PAGE.format(v="abc1234-dirty", fig="docs/figures/rl7"), encoding="utf-8"
    )
    plant = {"change": None, "name": RUN_NAME}

    def rerun(cmd, env):
        args = cmd[3:]
        module, change = cmd[2], plant["change"]
        if module == "traffic_twin.rl.train":
            out = root / args[args.index("--out") + 1] / plant["name"]
            pool = args[args.index("--eval-pool") + 1] if "--eval-pool" in args else "persistent"
            # records the environment it was launched in, as rl.train does
            make_run(out, "abc1234", timing=2.0, change=change, launch=launch_record(env, pool))
        elif args[0] == "golden":
            golden = {**GOLDEN, "created_at": "t1", "code_version": "abc1234"}
            if change == "golden":
                golden["q_values"] = [[0.125, -0.5], [1.0, 2.2500001]]
            _write_json(root / args[args.index("--out") + 1], golden)
        elif module == "traffic_twin.rl.report7":
            page = root / args[args.index("--docs") + 1]
            page.parent.mkdir(parents=True, exist_ok=True)
            figures = args[args.index("--figures") + 1]
            text = PAGE.format(v="abc1234", fig=figures)
            if change == "page":
                text = text.replace("375000", "400000")
            page.write_text(text, encoding="utf-8")

    calls["fn"] = rerun
    return root, head, calls, plant


def test_reproduce7_chain(tasks, capsys):
    cmds = commands(tasks, "reproduce7", "--slots", "6", "--allow-dirty", "--seed", "3")
    assert [c.name for c in cmds] == [
        "reproduce7-prepare", "reproduce7-train", "reproduce7-golden", "reproduce7-docs",
        "reproduce7-verify",
    ]  # fmt: skip
    assert tasks.main(["reproduce7", "-n"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [f"+ tasks.py:{c.name} (in-process)" for c in cmds]


def test_reproduce7_reruns_into_its_own_folder_and_accepts_a_faithful_rerun(
    tasks, rl7_reproducible, capsys
):
    """Code versions, the clean-tree flag, timings, zip sha256, timestamps, the
    code-version line of the page and its figure folder may differ; nothing else."""
    root, _, calls, _ = rl7_reproducible
    assert tasks.run(commands(tasks, "reproduce7")) == 0, capsys.readouterr().err
    backup = "data/rl7/_reproduce/abc1234"
    assert [c[2:] for c in calls["cmds"]] == [
        ["traffic_twin.rl.train", "--config", "configs/rl7/t1.yaml", "--algo", "dqn", "--seed",
         "0", "--steps", "400000", "--out", f"{backup}/runs", "--eval-pool", "persistent"],
        [STUDY, "golden", "--config", "configs/pitie_rl_s3.yaml", "--out",
         f"{backup}/rl_s3_golden.json"],
        ["traffic_twin.rl.report7", "--docs", f"{backup}/docs/rl_training.md", "--figures",
         f"{backup}/docs/figures/rl7"],
    ]  # fmt: skip
    # the training runs in the original learner's thread variables, the others as they are
    train_env, *others = calls["envs"]
    assert {k: v for k, v in train_env.items() if k.startswith(tasks.RL7_ENV_PREFIXES)} == (
        CAMPAIGN_ENV
    )
    assert others == [None, None]
    report = json.loads((root / backup / "reproduce7.json").read_text(encoding="utf-8"))
    assert report["same"] and report["head"] == "abc1234" and not any(report["items"].values())
    assert set(report["items"]) == {f"rl7 run {RUN_NAME}", "golden tests/data/rl_s3_golden.json",
                                    "page docs/rl_training.md"}  # fmt: skip
    # the originals are untouched
    assert (root / "data/rl7/runs/T" / RUN_NAME / "manifest.json").is_file()
    assert "abc1234-dirty" in (root / "docs/rl_training.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "item", "what"),
    [
        ("final", "rl7 run", "manifest.json: final_policy_sha256"),
        ("pick", "rl7 run", "provenance.json: pick_step"),
        ("validation", "rl7 run", "validation.csv differs"),
        ("monitor", "rl7 run", "monitor.csv differs"),
        ("golden", "golden", "rl_s3_golden.json: q_values[1][1]"),
        ("page", "page docs/rl_training.md", "differs from line 9"),
    ],
)
def test_reproduce7_fails_on_a_real_difference(tasks, rl7_reproducible, capsys, change, item, what):
    root, _, _, plant = rl7_reproducible
    plant["change"] = change
    assert tasks.run(commands(tasks, "reproduce7")) == 1
    err = [line for line in capsys.readouterr().err.splitlines() if item in line]
    assert err and what in err[0], err
    report = json.loads(
        (root / "data/rl7/_reproduce/abc1234/reproduce7.json").read_text(encoding="utf-8")
    )
    assert not report["same"]


@pytest.mark.parametrize(
    ("recorded", "pool", "shell"),
    [
        (CAMPAIGN_ENV, "persistent", {}),  # the campaign on a default shell
        ({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}, "transient", {}),
        (CAMPAIGN_ENV, "persistent", {"MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "8"}),
    ],
)
def test_reproduce7_launches_the_training_as_the_campaign_launched_the_original(
    tasks, rl7_reproducible, monkeypatch, capsys, recorded, pool, shell
):
    """The campaign starts every learner with rl.campaign.train_env() (OPENBLAS_NUM_THREADS=1
    where unset) and maybe `--eval-pool transient`; rl.train records its OMP_* / MKL_* /
    OPENBLAS_* variables (machine.env, child_env) and the pool in provenance.json and the
    manifest. The rerun gets the original's variables (the shell's own ones replaced) and
    pool, so a faithful rerun records the same provenance and passes."""
    root, _, calls, _ = rl7_reproducible
    make_run(root / "data/rl7/runs/T" / RUN_NAME, "abc1234-dirty",
             launch=launch_record(recorded, pool))  # fmt: skip
    for k, v in shell.items():
        monkeypatch.setenv(k, v)
    assert tasks.run(commands(tasks, "reproduce7")) == 0, capsys.readouterr().err
    train = calls["cmds"][0]
    assert train[train.index("--eval-pool") + 1] == pool
    env = calls["envs"][0]
    assert {k: v for k, v in env.items() if k.startswith(tasks.RL7_ENV_PREFIXES)} == recorded
    assert env["PATH"] == os.environ["PATH"]  # the rest of this environment is kept


@pytest.mark.parametrize(
    ("pool", "stale", "what"),
    [
        ("persistent", "reproduce7_train_env", "machine.env.OPENBLAS_NUM_THREADS"),
        ("transient", "reproduce7_train_options", "eval_pool"),
    ],
)
def test_reproduce7_still_reports_a_launch_difference(
    tasks, rl7_reproducible, monkeypatch, capsys, pool, stale, what
):
    """machine.env and eval_pool are not volatile: a rerun launched in this shell's
    environment as it is, or without the original's `--eval-pool` (what reproduce7 did
    before), is reported in both files, not hidden."""
    root, _, _, _ = rl7_reproducible
    make_run(root / "data/rl7/runs/T" / RUN_NAME, "abc1234-dirty",
             launch=launch_record(CAMPAIGN_ENV, pool))  # fmt: skip
    old = {"reproduce7_train_env": lambda original: None,
           "reproduce7_train_options": lambda original: []}[stale]  # fmt: skip
    monkeypatch.setattr(tasks, stale, old)
    assert tasks.run(commands(tasks, "reproduce7")) == 1
    err = capsys.readouterr().err
    assert f"manifest.json: {what}" in err and f"provenance.json: {what}" in err


def test_reproduce7_train_env_without_a_record_is_the_campaigns(tasks, monkeypatch, tmp_path):
    """A run that recorded no machine.env is rerun in what rl.campaign.train_env() gives:
    RL7_CHILD_ENV where unset, a variable already set kept."""
    for k in [k for k in os.environ if k.startswith(tasks.RL7_ENV_PREFIXES)]:
        monkeypatch.delenv(k)
    _write_json(tmp_path / "run/provenance.json", {"machine": {"cpu": "x"}})
    for original in (None, tmp_path / "run", tmp_path / "absent"):
        env = tasks.reproduce7_train_env(original)
        assert env == {**os.environ, **tasks.RL7_CHILD_ENV}
        assert tasks.reproduce7_train_options(original) == []
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    assert tasks.reproduce7_train_env(None)["OPENBLAS_NUM_THREADS"] == "3"
    _write_json(tmp_path / "run/provenance.json", {"eval_pool": "in-process"})
    assert tasks.reproduce7_train_options(tmp_path / "run") == []  # 1 eval worker: no pool


def test_reproduce7_launch_constants_mirror_rl_train_and_campaign(tasks, monkeypatch):
    """tasks.py is stdlib only: its copies of rl.train ENV_PREFIXES, CHILD_ENV and
    EVAL_POOLS are the module's, and its fallback environment is rl.campaign.train_env()."""
    train = pytest.importorskip("traffic_twin.rl.train")
    campaign = pytest.importorskip("traffic_twin.rl.campaign")
    assert tasks.RL7_ENV_PREFIXES == train.ENV_PREFIXES
    assert tasks.RL7_CHILD_ENV == train.CHILD_ENV
    assert tasks.RL7_EVAL_POOLS == train.EVAL_POOLS
    for k in [k for k in os.environ if k.startswith(tasks.RL7_ENV_PREFIXES)]:
        monkeypatch.delenv(k)
    assert tasks.reproduce7_train_env(None) == campaign.train_env()
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert tasks.reproduce7_train_env(None) == campaign.train_env()


def test_reproduce7_rerun_of_a_campaign_launched_stub_run_is_identical(
    tasks, stub_rl7_cfg, tmp_path, monkeypatch
):
    """End to end on the stub (no SUMO, a few hundred steps, training seed range only):
    a run launched as rl.campaign launches it (train_command, train_env) and its rerun
    launched as reproduce7 launches it (reproduce7_train_env/_options, tasks.call) give
    rl7_run_diff == []; the same rerun in the shell's environment as it is differs exactly
    on machine.env, which is what reproduce7 reported before."""
    pytest.importorskip("stable_baselines3")
    from traffic_twin.rl import campaign

    for k in [k for k in os.environ if k.startswith(tasks.RL7_ENV_PREFIXES)]:
        monkeypatch.delenv(k)
    t1 = stub_rl7_cfg().parent / "rl7" / "t1.yaml"
    seeds = ["--stub", "--validation-seeds", "5100"]

    def base(out: Path) -> list[str]:
        return [sys.executable, "-m", "traffic_twin.rl.train", "--config", t1.as_posix(),
                "--algo", "dqn", "--seed", "0", "--steps", "400",
                "--out", out.as_posix()]  # fmt: skip

    job = campaign.Job("T", t1, "dqn", 0, 400, 0, tmp_path / "orig")
    cmd = campaign.train_command(job, validation_seeds=[5100], stub=True)
    assert cmd == base(tmp_path / "orig") + ["--validation-seeds", "5100", "--stub"]
    with (tmp_path / "orig.log").open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=campaign.train_env())
        assert proc.wait() == 0, (tmp_path / "orig.log").read_text(encoding="utf-8")
    (original,) = [p for p in (tmp_path / "orig").iterdir() if (p / "manifest.json").is_file()]
    rerun = base(tmp_path / "rerun") + seeds + tasks.reproduce7_train_options(original)
    assert tasks.call(rerun, env=tasks.reproduce7_train_env(original)) == 0
    assert tasks.rl7_run_diff(original, tmp_path / "rerun" / original.name) == []
    assert tasks.call(base(tmp_path / "plain") + seeds) == 0
    assert tasks.rl7_run_diff(original, tmp_path / "plain" / original.name) == [
        "manifest.json: machine.env.OPENBLAS_NUM_THREADS",
        "provenance.json: machine.env.OPENBLAS_NUM_THREADS",
    ]


def test_reproduce7_fails_when_the_rerun_is_another_run(tasks, rl7_reproducible, capsys):
    """A config edited since the campaign changes the run name (run_config_sha): the
    original is not reproduced."""
    _, _, _, plant = rl7_reproducible
    plant["name"] = "dqn_t1_s0_400k_ffff0000"
    assert tasks.run(commands(tasks, "reproduce7")) == 1
    assert "not re-made (the rerun is dqn_t1_s0_400k_ffff0000)" in capsys.readouterr().err


def test_reproduce7_requires_the_pick_to_be_recorded(tasks, rl7_reproducible, capsys):
    root, _, _, _ = rl7_reproducible
    for run in (root / "data/rl7/runs/T" / RUN_NAME,):
        prov = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        del prov["pick_step"]
        _write_json(run / "provenance.json", prov)
    state = {}
    assert tasks.reproduce7_prepare(state) == 0
    for what in ("train", "golden", "docs"):
        assert tasks.reproduce7_run(state, what) == 0
    # the rerun records it, the original does not
    assert tasks.reproduce7_verify(state) == 1
    assert "no pick_step recorded" in capsys.readouterr().err


def test_reproduce7_prepare_refuses(tasks, rl7_reproducible, capsys):
    root, head, calls, _ = rl7_reproducible
    head["value"] = ("abc1234", True)
    assert tasks.run(commands(tasks, "reproduce7")) == 1
    assert "uncommitted changes" in capsys.readouterr().err and not calls["cmds"]
    head["value"] = ("abc1234", False)
    (root / tasks.RL7_GOLDEN).unlink()
    assert tasks.reproduce7_prepare({}) == 1
    assert "missing tests/data/rl_s3_golden.json" in capsys.readouterr().err
    _write_json(root / tasks.RL7_GOLDEN, GOLDEN)
    make_run(root / "data/rl7/runs/T/dqn_t1_s0_400k_ffff0000")  # two candidates
    assert tasks.reproduce7_prepare({}) == 1
    assert "exactly one finished run" in capsys.readouterr().err
    shutil.rmtree(root / "data/rl7/runs/T/dqn_t1_s0_400k_ffff0000")
    assert tasks.reproduce7_prepare({}) == 0
    (root / "data/rl7/_reproduce/abc1234/runs").mkdir(parents=True)
    assert tasks.reproduce7_prepare({}) == 1  # a reproduction is never overwritten
    assert "already holds a reproduction" in capsys.readouterr().err
    assert not calls["cmds"]


def test_reproduce7_stops_when_a_rerun_fails(tasks, rl7_reproducible):
    _, _, calls, _ = rl7_reproducible
    calls["code"] = 3
    assert tasks.run(commands(tasks, "reproduce7")) == 3
    assert len(calls["cmds"]) == 1  # the training failed: nothing else ran


def test_rl7_holdout_lists_the_runs_when_it_runs(tasks, rl7_tree, capsys):
    """The step lists the finished runs at run time (the campaign makes them inside
    `part7`): T and S runs on their pick, F runs with --curves; partial, `_`-prefixed and
    non-run folders are skipped."""
    root, _, calls = rl7_tree
    (cmd,) = commands(tasks, "rl7-holdout")  # built before any run exists
    assert (
        cmd.fn() == 1 and "no finished run under data/rl7/runs/<stage>/" in capsys.readouterr().err
    )
    runs = root / "data/rl7/runs"
    for stage, name in (("T", "dqn_t1_s0_400k_a"), ("T", "ppo_p1_s0_400k_a"),
                        ("S", "dqn_wb12_s3_400k_b"), ("F", "dqn_t1_s1_1000k_c"),
                        ("F", "dqn_t1_s0_1000k_c")):  # fmt: skip
        _write_json(runs / stage / name / "manifest.json", {"run": name})
    (runs / "T/dqn_t0_s0_400k_partial").mkdir()  # no manifest: still running or crashed
    _write_json(runs / "S/_gate/x/manifest.json", {})
    assert cmd.fn() == 0
    assert [c[2:] for c in calls["cmds"]] == [
        [STUDY, "holdout", "data/rl7/runs/T/dqn_t1_s0_400k_a", "data/rl7/runs/T/ppo_p1_s0_400k_a",
         "data/rl7/runs/S/dqn_wb12_s3_400k_b"],
        [STUDY, "holdout", "data/rl7/runs/F/dqn_t1_s0_1000k_c",
         "data/rl7/runs/F/dqn_t1_s1_1000k_c", "--curves"],
    ]  # fmt: skip
    calls["cmds"].clear()
    calls["code"] = 2
    assert cmd.fn() == 2 and len(calls["cmds"]) == 1  # stops at the first failure


def test_part7_stops_until_the_s3_config_exists(tasks, rl7_tree, capsys):
    root, _, calls = rl7_tree
    assert tasks.part7_s3_check() == 1
    err = capsys.readouterr().err
    assert "configs/pitie_rl_s3.yaml is missing" in err and "rl7-stress rl7-sens" in err
    cmds = commands(tasks, "part7")
    (guard,) = [c for c in cmds if isinstance(c, tasks.Step) and c.name == "part7-s3-config"]
    assert cmds.index(guard) == 4 and guard.fn() == 1  # `run` stops there, after rl7-publish
    assert not calls["cmds"]
    (root / "configs").mkdir()
    (root / tasks.PITIE_RL_S3).write_text("extends: pitie_rl.yaml\n", encoding="utf-8")
    assert tasks.part7_s3_check() == 0


def _part7_cli_commands(tasks, root: Path) -> list[list[str]]:
    """Every subprocess command of the Part 7 tasks, the in-process ones included."""
    cmds = [c for name in RL7_SINGLE for c in commands(tasks, name)]
    for stage, name in (("T", "dqn_t1_s0_400k_a"), ("F", "dqn_t1_s0_1000k_c")):
        _write_json(root / "data/rl7/runs" / stage / name / "manifest.json", {"run": name})
    _write_json(root / "data/rl7/runs/T/dqn_t1_s0_400k_a/provenance.json",
                {"eval_pool": "transient"})  # fmt: skip
    cmds += tasks.rl7_holdout_commands()
    cmds += list(
        tasks.reproduce7_commands(
            root / "data/rl7/_reproduce/abc1234", root / "data/rl7/runs/T/dqn_t1_s0_400k_a"
        ).values()
    )
    return cmds


def test_part7_commands_parse_with_their_cli(tasks, rl7_tree, monkeypatch):
    """Every argument a Part 7 task passes, including the commands its in-process steps
    build (holdout, reproduce7), is one its module's own parser accepts. Nothing runs."""
    root, _, _ = rl7_tree
    cmds = _part7_cli_commands(tasks, root)
    assert len(cmds) == 13
    for cmd in cmds:
        with monkeypatch.context() as m:
            namespace, extra = parse_only(m, cmd[2], cmd[3:])
        assert extra == [], f"{cmd[2]} does not accept {extra}"
        values = vars(namespace)
        if "--config" in cmd and cmd[3] != "golden":
            assert Path(values["config"]) == Path(cmd[cmd.index("--config") + 1]), cmd
        if cmd[2] == STUDY and cmd[3] == "golden":  # the subcommand's own --config
            assert Path(values["scenario"]) == Path("configs/pitie_rl_s3.yaml")
        if "--curves" in cmd:
            assert values["curves"] and [Path(r) for r in values["runs"]] == [
                Path(r) for r in cmd[4:-1]
            ]
        if cmd[2] == "traffic_twin.rl.train":
            assert (values["seed"], values["steps"]) == (0, 400000)
            assert values["eval_pool"] == "transient"  # the original's pool


# -- Parts 8-10 tasks (pre-registered D91-D120) -------------------------------------------------
PREDICT, DEMO, USERTESTS, REPORT = (
    "traffic_twin.ml.predict", "traffic_twin.demo", "traffic_twin.usertests", "traffic_twin.report",
)  # fmt: skip
S2_NOP, S3, S4 = (
    "configs/pitie_rules_nopreempt.yaml", "configs/pitie_rl_s3.yaml",
    "configs/pitie_rl_s3_nopreempt.yaml",
)  # fmt: skip
PARTS_8_10_MODULE_TASKS = {
    "demo-runs8": [PREDICT, "demo", "--seeds", "4000", "--configs", S2_NOP, S3, S4],
    "operator": ["streamlit", "run", "app.py", "--server.address", "127.0.0.1",
                 "--server.port", "8501"],
    "usertest-kit": [USERTESTS, "kit"],
    "usertest-serve": [USERTESTS, "serve"],
    "usertest-check": [USERTESTS, "check"],
    "usertest-docs": [USERTESTS, "docs"],
    "test-browser": ["pytest", "-q", "-m", "browser"],
    "report-facts": [REPORT, "facts"],
    "report": [REPORT, "report"],
    "slides": [REPORT, "slides"],
    "app-shots": [REPORT, "shots"],
    "report-check": [REPORT, "check"],
    "demo": [DEMO, "serve"],
    "demo-check": [DEMO, "check"],
    "demo-rehearsal": [DEMO, "rehearsal"],
    "demo-clean": [DEMO, "clean"],
    "demo-pack": [DEMO, "pack"],
    "bootstrap": [DEMO, "bootstrap"],
    "demo-readiness": [DEMO, "readiness"],
}  # fmt: skip


def test_parts_8_10_task_defaults(tasks):
    got = {name: [c[2:] for c in commands(tasks, name)] for name in PARTS_8_10_MODULE_TASKS}
    assert got == {name: [cmd] for name, cmd in PARTS_8_10_MODULE_TASKS.items()}
    for name in ("part8-guard", "rebaseline-verify"):
        ((step,),) = [commands(tasks, name)]
        assert isinstance(step, tasks.Step) and step.name == name


def test_part8_chain(tasks):
    """D97: the guard, then `rebaseline` with every default (move aside, `compare validate
    part4` re-make the 70 runs and make the 40 S3 / S4 runs, verify), the figures and the
    provenance check. No option of tasks.py reaches it."""
    assert tasks.PART8_STEPS == ("part8-guard", "rebaseline", "plots-eval", "provenance")
    cmds = commands(tasks, "part8")
    names = [c.name if isinstance(c, tasks.Step) else c[2] for c in cmds]
    assert names == [
        "part8-guard", "run-lock", "rebaseline-backup", "run-lock",
        "traffic_twin.evaluation.compare",
        "run-lock", "traffic_twin.validation.report", *PART4_MODULES, "rebaseline-verify",
        "traffic_twin.viz.evaluation", "provenance",
    ]  # fmt: skip
    rebaseline = [c for c in commands(tasks, "rebaseline") if not isinstance(c, tasks.Step)]
    assert [c for c in cmds if not isinstance(c, tasks.Step)][:-1] == rebaseline
    other = commands(tasks, "part8", "--scenario", "x.yaml", "--baseline", "y.yaml", "--seeds",
                     "1", "--config", "c.yaml", "--fixtures", "--record")  # fmt: skip
    assert plain(tasks, other) == plain(tasks, cmds)
    (ev,) = [c for c in cmds if not isinstance(c, tasks.Step) and c[2].endswith(".evaluate")]
    assert ev == commands(tasks, "evaluate")[0] and "--run-missing" in ev


def test_part8_dry_run(tasks, capsys):
    assert tasks.main(["part8", "-n"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "+ tasks.py:part8-guard (in-process)"
    assert out[-1] == "+ tasks.py:provenance (in-process)"
    assert len(out) == len(commands(tasks, "part8"))


def test_part9_part10_chains(tasks):
    assert tasks.PART9_STEPS == ("demo-runs8", "api-bench", "usertest-check", "usertest-docs")
    assert tasks.PART10_STEPS == ("report-facts", "app-shots", "report", "slides", "demo-pack",
                                  "demo-rehearsal")  # fmt: skip
    for chain, steps in (("part9", tasks.PART9_STEPS), ("part10", tasks.PART10_STEPS)):
        assert commands(tasks, chain) == [c for t in steps for c in commands(tasks, t)], chain
    # the rehearsal options reach the chain's rehearsal only
    *_, rehearsal = commands(tasks, "part10", "--fixtures", "--no-browser")
    assert rehearsal[3:] == ["rehearsal", "--fixtures", "--no-browser"]


def test_parts_9_10_options_reach_only_their_tasks(tasks):
    (cmd,) = commands(tasks, "demo-rehearsal", "--fixtures", "--no-browser", "--record")
    assert cmd[3:] == ["rehearsal", "--fixtures", "--no-browser", "--record"]
    for name, sub in (("demo", "serve"), ("demo-check", "check")):
        (cmd,) = commands(tasks, name, "--no-browser", "--fixtures", "--record")
        assert cmd[3:] == [sub, "--no-browser"], name
    (kit,) = commands(tasks, "usertest-kit", "--allow-pending-s3", "--allow-dirty")
    assert kit[3:] == ["kit", "--allow-pending-s3"]
    (serve,) = commands(tasks, "usertest-serve", "--allow-dirty", "--allow-pending-s3")
    assert serve[3:] == ["serve", "--allow-dirty"]
    flags = ["--fixtures", "--no-browser", "--record", "--allow-pending-s3", "--allow-dirty"]
    untouched = ("usertest-check", "usertest-docs", "demo-clean", "demo-pack", "bootstrap",
                 "demo-readiness", "report-facts", "report", "slides", "app-shots", "demo-runs8",
                 "operator", "part9", "evaluate", "sim")  # fmt: skip
    for name in untouched:
        assert plain(tasks, commands(tasks, name, *flags)) == plain(tasks, commands(tasks, name))


def test_makefile_passes_the_parts_9_10_options():
    """`make demo-rehearsal FIXTURES=1 NO_BROWSER=1 RECORD=1`, `make usertest-kit
    ALLOW_PENDING_S3=1` and `make usertest-serve ALLOW_DIRTY=1` reach the module options;
    `make sim`, `sim-gui` and `baseline` go through tasks.py for the run lock."""
    rules = makefile_rules()
    (line,) = rules["demo-rehearsal"][1]
    for var, flag in (("FIXTURES", "--fixtures"), ("NO_BROWSER", "--no-browser"),
                      ("RECORD", "--record")):  # fmt: skip
        assert f"$(if $({var}), {flag})" in line, var
    for target in ("demo", "demo-check"):
        (line,) = rules[target][1]
        assert line.endswith("$(if $(NO_BROWSER), --no-browser)"), target
    (line,) = rules["usertest-kit"][1]
    assert line.endswith("$(if $(ALLOW_PENDING_S3), --allow-pending-s3)")
    (line,) = rules["usertest-serve"][1]
    assert line.endswith("$(if $(ALLOW_DIRTY), --allow-dirty)")
    for target in ("sim", "sim-gui", "baseline"):
        (line,) = rules[target][1]
        assert line.startswith(f"$(PY) tasks.py {target}$(if $(SCENARIO), --scenario $(SCENARIO))")
        assert line.endswith("$(if $(SEEDS), --seeds $(SEEDS))"), target


def test_parts_8_10_constants_are_the_configs_and_paths(tasks):
    """tasks.py's Part 8-9 configs are those configs/evaluation.yaml registers (S2_noP, S3,
    S4 of the main family) and paths.py names; the demo seeds are api.yaml's; the Operator
    address is that of .streamlit/config.toml (and configs/demo.yaml once it exists); the
    registration key is the one `evaluate.registration_commit` looks for."""
    import yaml

    from traffic_twin import paths
    from traffic_twin.evaluation import evaluate

    ecfg = yaml.safe_load((ROOT / EVAL).read_text(encoding="utf-8"))
    main = ecfg["families"]["main"]["scenarios"]
    assert tasks.DEMO_RUNS8_CONFIGS == (main["S2_noP"], main["S3"], main["S4"])
    assert Path(tasks.PITIE_S4) == paths.PITIE_RL_S4_CONFIG and Path(tasks.PITIE_RL_S3) == Path(S3)
    assert all((ROOT / c).is_file() for c in tasks.DEMO_RUNS8_CONFIGS)
    assert tasks.REGISTRATION_KEY in ecfg and tasks.REGISTRATION_KEY == "a06"
    assert '"a06:"' in Path(evaluate.__file__).read_text(encoding="utf-8")
    assert Path(tasks.EVAL_PROVENANCE).parent == paths.EVAL_DIR
    assert tasks.RESULTS == paths.RESULTS_DIR
    api = yaml.safe_load((ROOT / "configs/api.yaml").read_text(encoding="utf-8"))
    lo, hi = api["seeds"]["demo"]
    assert all(lo <= int(s) <= hi for s in tasks.DEMO_SEEDS)  # never an evaluation seed (D101)
    server = tomllib.loads((ROOT / ".streamlit/config.toml").read_text(encoding="utf-8"))["server"]
    assert (server["address"], str(server["port"])) == (tasks.OPERATOR_HOST, tasks.OPERATOR_PORT)
    assert (ROOT / tasks.OPERATOR_SCRIPT).is_file()
    demo = ROOT / "configs/demo.yaml"  # WS7 (D116)
    if demo.is_file():
        d = yaml.safe_load(demo.read_text(encoding="utf-8"))
        assert (d["api"]["host"], str(d["api"]["port"])) == (tasks.API_HOST, tasks.API_PORT)
        assert (d["operator"]["host"], str(d["operator"]["port"])) == (
            tasks.OPERATOR_HOST, tasks.OPERATOR_PORT,
        )  # fmt: skip
        assert d["operator"]["script"] == tasks.OPERATOR_SCRIPT
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    for ignored in ("data/demo_check/*", "data/report/*", "data/exports/*", "last_dom.html"):
        assert ignored in gitignore, ignored  # a stray file there would dirty every I2 run


PARTS_8_10_CLI = [
    ("demo-runs8", ()), ("usertest-kit", ()), ("usertest-kit", ("--allow-pending-s3",)),
    ("usertest-serve", ()), ("usertest-serve", ("--allow-dirty",)), ("usertest-check", ()),
    ("usertest-docs", ()), ("report-facts", ()), ("report", ()), ("slides", ()),
    ("app-shots", ()), ("report-check", ()), ("demo", ()), ("demo", ("--no-browser",)),
    ("demo-check", ()), ("demo-check", ("--no-browser",)), ("demo-rehearsal", ()),
    ("demo-rehearsal", ("--fixtures", "--no-browser", "--record")), ("demo-clean", ()),
    ("demo-pack", ()), ("bootstrap", ()), ("demo-readiness", ()),
]  # fmt: skip


@pytest.mark.parametrize(("name", "flags"), PARTS_8_10_CLI)
def test_parts_8_10_commands_parse_with_their_cli(tasks, monkeypatch, name, flags):
    """Every argument a Parts 8-10 task passes, with each of its options, is one the
    module's own parser accepts (parsed only, nothing runs): the contract between tasks.py
    and the usertests, report and demo CLIs."""
    (cmd,) = commands(tasks, name, *flags)
    namespace, extra = parse_only(monkeypatch, cmd[2], cmd[3:])
    assert extra == [], f"{name}: {cmd[2]} does not accept {extra}"
    values = vars(namespace)
    if "--seeds" in cmd:
        assert [int(s) for s in values["seeds"]] == [4000]
        assert [Path(c) for c in values["configs"]] == [Path(c) for c in tasks.DEMO_RUNS8_CONFIGS]


# The packages the Parts 9-10 tasks run (Phase 0 stubs until their workstreams land)
TASK_PACKAGES = ("demo", "usertests", "report")
STUB_MARKERS = ("raise NotImplementedError", "not implemented yet", "Phase 0 stub")


def test_task_packages_have_no_stub_left():
    """Phase 2 gate ("no stubs left"): no module of the packages the Parts 9-10 tasks call
    still raises NotImplementedError or answers "not implemented yet"."""
    left = [
        f"{path.relative_to(ROOT).as_posix()}: {marker!r}"
        for pkg in TASK_PACKAGES
        for path in sorted((ROOT / "src/traffic_twin" / pkg).glob("*.py"))
        for marker in STUB_MARKERS
        if marker in path.read_text(encoding="utf-8")
    ]
    assert not left, "stubs left: " + "; ".join(left)


# -- Part 8 guard and verify.json (D92, D97) ------------------------------------------------------
REGISTERED = """\
seeds: {evaluation: [1, 2]}
families:
  main: {scenarios: {S0: configs/pitie_fixed.yaml, S2: configs/pitie_rules.yaml,
                     S3: configs/pitie_rl_s3.yaml}}
  probe: {scenarios: {S0: configs/pitie_amb_fixed.yaml}}
a06:   # Part 8
  candidate: S3
"""


@pytest.fixture
def part8_repo(tasks, results, monkeypatch):
    """The fake repo of `results` (provenance of pitie_fixed, pitie_rules, pitie_amb_fixed
    on seeds 1-2), with `git show HEAD:configs/evaluation.yaml` answering `head["text"]`."""
    root, head = results
    head["text"] = REGISTERED
    calls = []

    def git(*args):
        calls.append(args)
        if args[0] == "show":
            return head["text"]
        return "abc1234 2026-10-23T12:00:00+02:00\n" if "a06:" in head["text"] else ""

    monkeypatch.setattr(tasks, "_git", git)
    return root, head, calls


def test_part8_guard_accepts_the_clean_registration_commit(tasks, part8_repo, capsys):
    _, _, calls = part8_repo
    assert tasks.part8_guard() == 0
    assert "registration committed (abc1234 2026-10-23T12:00:00+02:00)" in capsys.readouterr().out
    assert calls[0] == ("show", "HEAD:configs/evaluation.yaml")


@pytest.mark.parametrize(
    "case", ["dirty", "no_a06", "comment_only", "ran", "early_s3", "no_provenance"]
)
def test_part8_guard_refuses(tasks, part8_repo, capsys, case):
    root, head, _ = part8_repo
    expected = {
        "dirty": "uncommitted changes",
        "no_a06": "has no `a06` block",
        "comment_only": "has no `a06` block",
        "ran": "part8 (its rebaseline) already ran from abc1234",
        "early_s3": "data/results/pitie_rl_s3_seed2): only part8 makes them",
        "no_provenance": "the Part 4 evaluation is missing",
    }[case]
    if case == "dirty":
        head["value"] = ("abc1234", True)
    elif case == "no_a06":
        head["text"] = REGISTERED.split("a06:")[0]
    elif case == "comment_only":  # the comment naming the Part 8 keys is not the block
        head["text"] = REGISTERED.split("a06:")[0] + "# ... a06, final_table, part8 ...\n"
    elif case == "ran":
        (root / tasks.REBASELINE_DIR / "abc1234").mkdir(parents=True)
    elif case == "early_s3":
        write_run(root / "data/results/pitie_rl_s3_seed2", "abc1234-dirty", kpis="kpi\n")
        write_run(root / "data/results/pitie_rl_s3_seed4000", "abc1234-dirty", kpis="kpi\n")
    else:
        (root / tasks.EVAL_PROVENANCE).unlink()
    assert tasks.part8_guard() == 1
    err = capsys.readouterr().err
    assert expected in err, err
    if case == "early_s3":
        assert "seed4000" not in err  # demo seeds are not evaluation seeds


def test_part8_guard_ignores_runs_of_already_evaluated_scenarios(tasks, part8_repo):
    """The S0-S2 runs on the evaluation seeds exist (the rebaseline re-makes them): only
    the scenarios the last evaluation did not run must have no run yet."""
    import yaml

    root, _, _ = part8_repo
    assert (root / "data/results/pitie_fixed_seed1/run.json").is_file()
    assert tasks.part8_new_runs(yaml.safe_load(REGISTERED)) == []
    assert tasks.part8_guard() == 0


def test_rebaseline_verify_writes_the_a01_evidence(tasks, results, capsys):
    root, _ = results
    verify = root / tasks.REBASELINE_DIR / "abc1234" / tasks.VERIFY_FILE
    assert tasks.rebaseline_backup() == 0
    rerun_all(tasks, root, "abc1234", change="pitie_rules_seed2")
    assert tasks.rebaseline_verify() == 1
    assert json.loads(verify.read_text(encoding="utf-8")) == {
        "head": "abc1234", "identical": 5, "of": 6,
        "differing": [{"run": "pitie_rules_seed2", "what": ["kpi_run.csv differs"]}],
    }  # fmt: skip
    rerun_all(tasks, root, "abc1234")
    assert tasks.rebaseline_verify() == 0
    assert json.loads(verify.read_text(encoding="utf-8")) == {
        "head": "abc1234", "identical": 6, "of": 6, "differing": [],
    }  # fmt: skip
    out = capsys.readouterr().out
    assert "rebaseline: report data/results/_rebaseline/abc1234/verify.json" in out


def test_rebaseline_verify_without_a_backup_writes_nothing(tasks, results, capsys):
    root, _ = results
    tasks.rebaseline_verify()
    assert not (root / tasks.REBASELINE_DIR).exists()  # rebaseline-backup keeps its own check
    assert "no verify.json written" in capsys.readouterr().out


# -- the D92 run lock of `sim` --------------------------------------------------------------------
@pytest.fixture
def evaluate_state(monkeypatch):
    """traffic_twin.evaluation.evaluate with git answering a dirty tree and no
    registration commit; the returned dict switches them."""
    evaluate = pytest.importorskip("traffic_twin.evaluation.evaluate")
    state = {"dirty": True, "registered": False}
    monkeypatch.setattr(evaluate, "git_head", lambda: ("abc1234", state["dirty"]))
    commit = {"commit": "abc1234", "committed_at": "2026-10-23T12:00:00+02:00"}
    monkeypatch.setattr(
        evaluate, "registration_commit", lambda path=None: commit if state["registered"] else None
    )
    return state


def test_sim_run_lock_refuses_an_evaluation_run_before_the_registration(
    tasks, evaluate_state, capsys
):
    assert tasks.sim_run_lock(S3, ["3", "4000"]) == 2
    err = capsys.readouterr().err
    assert "run lock: refusing 1 run(s)" in err and "pitie_rl_s3 seed 3" in err
    assert tasks.sim_run_lock("configs/pitie_fixed.yaml", ["10"]) == 2  # would overwrite S0
    evaluate_state.update(dirty=False, registered=True)  # the clean registration commit
    assert tasks.sim_run_lock(S3, ["3"]) == 0


@pytest.mark.parametrize(
    ("scenario", "seeds"),
    [(GRID, ["1", "2", "3"]), (S3, ["4000"]), ("configs/pitie_rl.yaml", ["1"]),
     ("configs/missing.yaml", ["1"])],
)  # fmt: skip
def test_sim_run_lock_passes_other_scenarios_and_seeds(tasks, evaluate_state, scenario, seeds):
    """The grid on its seeds 1-3, the demo seed, an unregistered Pitie config, and a config
    run_scenario rejects itself."""
    assert tasks.sim_run_lock(scenario, seeds) == 0


def test_sim_launches_nothing_when_the_lock_refuses(tasks, evaluate_state, monkeypatch, capsys):
    launched = []
    monkeypatch.setattr(tasks.subprocess, "call", lambda cmd, **kw: launched.append(cmd) or 0)
    assert tasks.main(["sim", "--scenario", S4, "--seeds", "1"]) == 2
    assert launched == []
    assert "run lock" in capsys.readouterr().err
    assert tasks.main(["sim-gui", "--scenario", S4, "--seeds", "4001"]) == 0
    assert [c[2] for c in launched] == ["traffic_twin.sim.run_scenario"]


# -- the D92 run lock of `compare`, `validate`, `part3` and `rebaseline` ----------------------
@pytest.fixture
def run_folders(tmp_path, monkeypatch):
    """runs.* as the lock reads them, on tmp folders: a run exists (complete) when its
    `<scenario>_seed<n>` name is in the returned `fresh` set, and is stale when it is in
    `stale` too. The results folder the lock checks stays data/results."""
    runs = pytest.importorskip("traffic_twin.evaluation.runs")
    state: dict[str, set[str]] = {"fresh": set(), "stale": set()}
    monkeypatch.setattr(runs, "run_path", lambda results, sc, sd: tmp_path / f"{sc}_seed{sd}")
    monkeypatch.setattr(runs, "is_complete", lambda d: d.name in state["fresh"])
    monkeypatch.setattr(runs, "read_meta", lambda d: {"name": d.name})
    monkeypatch.setattr(runs, "expected_hashes", lambda cfg: {})
    monkeypatch.setattr(
        runs, "stale_reason",
        lambda meta, expected, mkv=None: "stale" if meta["name"] in state["stale"] else None,
    )  # fmt: skip
    return state


def all_runs(scenario: str) -> set[str]:
    return {f"{scenario}_seed{s}" for s in range(1, 11)}


@pytest.mark.parametrize(
    "argv",
    [["compare", "--scenario", S3], ["part3", "--scenario", S3], ["validate", "--scenario", S3],
     ["compare", "--scenario", S4], ["compare"], ["rebaseline"]],
)  # fmt: skip
@pytest.mark.parametrize("why", ["dirty", "unregistered"])
def test_missing_runs_on_seeds_1_10_launch_nothing_before_the_registration(
    tasks, evaluate_state, run_folders, monkeypatch, capsys, argv, why
):
    """compare.py and validation/report.py (frozen) run the missing seeds 1-10 into
    data/results with no lock of their own: the tasks refuse them, exit 2, and launch
    nothing (part3 before its calibration loop, rebaseline before moving any run)."""
    if why == "unregistered":
        evaluate_state.update(dirty=False)  # a clean tree, but no commit adds a06: yet
    launched, moved = [], []
    monkeypatch.setattr(tasks.subprocess, "call", lambda cmd, **kw: launched.append(cmd) or 0)
    monkeypatch.setattr(tasks, "rebaseline_backup", lambda: moved.append(1) or 0)
    run_folders["fresh"] |= all_runs("pitie_fixed")  # only S1 (default) or S3 / S4 missing
    assert tasks.main(argv) == 2
    assert launched == [] and moved == []
    err = capsys.readouterr().err
    assert "run lock: refusing" in err and "nothing was launched" in err


def test_missing_run_lock_names_the_runs_compare_would_make(tasks, evaluate_state, run_folders,
                                                            capsys):  # fmt: skip
    """Only the absent and the stale runs count; fresh runs launch nothing, so re-making a
    page from them works on a dirty tree; after the registration commit everything passes."""
    run_folders["fresh"] |= all_runs("pitie_fixed") | all_runs("pitie_rl_s3")
    assert tasks.missing_run_lock([S0, S3], S0) == 0  # every run is there and fresh
    run_folders["stale"] |= {"pitie_rl_s3_seed4"}
    assert tasks.missing_run_lock([S0, S3], S0) == 2
    assert "refusing 1 run(s)" in capsys.readouterr().err  # the stale S3 seed 4 only
    run_folders["fresh"] -= {"pitie_fixed_seed10"}
    assert tasks.missing_run_lock([S0], S0) == 2  # S0 too: every controller (D92)
    assert "pitie_fixed seed 10" in capsys.readouterr().err
    assert tasks.missing_run_lock([S0], S0, every=True) == 2  # rebaseline: all of them
    assert "refusing 10 run(s)" in capsys.readouterr().err
    evaluate_state.update(dirty=False, registered=True)  # the clean registration commit
    assert tasks.missing_run_lock([S0, S3], S0, every=True) == 0
    assert tasks.missing_run_lock(["configs/missing.yaml"], S0) == 0  # compare reports it


def test_compare_validate_part3_run_after_the_lock(tasks, evaluate_state, run_folders,
                                                   monkeypatch):  # fmt: skip
    """With fresh runs (nothing to launch) the dirty tree passes and the modules run."""
    launched = []
    monkeypatch.setattr(tasks.subprocess, "call", lambda cmd, **kw: launched.append(cmd) or 0)
    run_folders["fresh"] |= all_runs("pitie_fixed") | all_runs("pitie_actuated")
    assert tasks.main(["compare", "validate"]) == 0
    assert modules(launched) == ["traffic_twin.evaluation.compare",
                                 "traffic_twin.validation.report"]  # fmt: skip
    launched.clear()
    assert tasks.main(["part3"]) == 0
    assert modules(launched) == modules(commands(tasks, "part3"))


def test_ci_runs_the_parts_9_10_checks(tasks):
    """D118: CI installs the `app` and `report` extras, runs the suite without the
    headless-browser tests, then the fixtures rehearsal through tasks.py (no real data, no
    SUMO, no browser), with options the task really passes."""
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    (ci_extras,) = re.findall(r'pip install -e "\.\[([\w,]+)\]"', ci)
    assert {"dev", "sim", "api", "rl", "app", "report"} <= set(ci_extras.split(","))
    runs = re.findall(r"^\s+run: (python .*)$", ci, re.M)
    assert 'python -m pytest -q -m "not browser"' in runs
    (rehearsal,) = [r for r in runs if "tasks.py" in r]
    argv = shlex.split(rehearsal)[2:]
    assert argv == ["demo-rehearsal", "--fixtures", "--no-browser"]
    (cmd,) = commands(tasks, *argv)
    assert cmd[2:] == [DEMO, "rehearsal", "--fixtures", "--no-browser"]
    assert runs.index(rehearsal) > runs.index('python -m pytest -q -m "not browser"')

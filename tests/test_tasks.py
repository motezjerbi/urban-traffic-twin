"""tasks.py, the cross-platform stand-in for the Makefile."""

import argparse
import importlib
import importlib.util
import json
import re
import shutil
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
        m = re.search(r"\$\(PY\) (?:-m (\S+)|(tasks\.py \S+?))(?=\s|\$|$)(.*)", line)
        if m:
            args = re.sub(r"\$\(if .*$", "", m.group(3))
            calls.append((m.group(1) or m.group(2), re.findall(r"(?<!\S)--[\w-]+", args)))
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
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    phony = " ".join(re.findall(r"^\.PHONY:(.*)$", makefile, re.M)).split()
    later = {"train", "eval", "app"}  # placeholders for later parts of the roadmap
    assert set(phony) - later == set(tasks.TASKS)
    assert {"plots", "baseline", "check", "part3", "part4", "part5", "part6"} <= set(phony)


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
        assert make_calls(rules, name) == expected_calls(tasks, name), name


def test_makefile_delegates_config_driven_tasks_with_their_options():
    """`make rl-train ALGO=dqn SEED=5 STEPS=8000 N_ENVS=4` reaches tasks.py's options."""
    (line,) = makefile_rules()["rl-train"][1]
    assert line.startswith("$(PY) tasks.py rl-train")
    for var, flag in (("ALGO", "--algo"), ("SEED", "--seed"), ("STEPS", "--steps"),
                      ("N_ENVS", "--n-envs")):  # fmt: skip
        assert f"$(if $({var}), {flag} $({var}))" in line, var
    for target in ("eta-data", "eta-data-test"):
        (line,) = makefile_rules()[target][1]
        assert line.endswith("$(if $(ALLOW_DIRTY), --allow-dirty)"), target


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
    monkeypatch.setattr(tasks, "call", lambda cmd: repro["calls"].append(cmd) or repro["code"])
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

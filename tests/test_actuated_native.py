"""Oracle for S1 (D12): the Python ActuatedController, run through SimSession, must take
the same decision as SUMO's own `type="actuated"` logic at every step.

The native side loads the net's program as an actuated tlLogic from an additional file
(the program loaded last is the active one) with the same control loops
(`<param key=lane value=loop>`), SUMO's gap parameters and `offset = begin mod cycle`
so that it starts in phase 0. SUMO switches at the start of a step, so its `getPhase()`
read before step t+1 is the phase SimSession showed during step t.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import traffic_twin.sim.session as session_mod
from traffic_twin.config import load_scenario
from traffic_twin.control.registry import make_controller
from traffic_twin.paths import PITIE_ACTUATED_CONFIG
from traffic_twin.sim.detectors import DetectorLayout
from traffic_twin.sim.session import SimSession
from traffic_twin.sim.sumo import traci_module

pytestmark = pytest.mark.sumo

REPO_ROOT = Path(__file__).resolve().parent.parent
STEPS = 900


def native_program(cfg: dict, layout: DetectorLayout) -> str:
    """The scenario's signal program as a native actuated tlLogic (programID "act")."""
    net, act = cfg["network"], cfg["actuated"]
    control = cfg["detectors"]["control"]
    root = ET.parse(net["net_file"]).getroot()
    tl = next(t for t in root.iter("tlLogic") if t.get("id") == net["tls_id"])
    green = {s["green_phase"]: s["name"] for s in net["stages"]}
    params = {
        "max-gap": act["max_gap_s"],
        "detector-gap": control.get("detector_gap_s", 2.0),
        "passing-time": control.get("passing_time_s", 1.9),
        **{lp.lane: lp.id for lp in layout.control_loops},
    }
    lines = [f'        <param key="{k}" value="{v}"/>' for k, v in params.items()]
    cycle = 0.0
    for i, p in enumerate(tl.iter("phase")):
        state = p.get("state")
        if i in green:
            lo, hi = act["min_green_s"][green[i]], act["max_green_s"][green[i]]
            lines.append(
                f'        <phase duration="{lo}" minDur="{lo}" maxDur="{hi}" state="{state}"/>'
            )
            cycle += lo
        else:
            lines.append(f'        <phase duration="{p.get("duration")}" state="{state}"/>')
            cycle += float(p.get("duration"))
    offset = int(cfg["begin_s"]) % round(cycle)  # SUMO starts at (begin - offset) mod cycle
    head = f'    <tlLogic id="{net["tls_id"]}" type="actuated" programID="act" offset="{offset}">'
    return "\n".join(["<additional>", head, *lines, "    </tlLogic>", "</additional>", ""])


def run_python(cfg: dict, seed: int, tmp_path: Path, monkeypatch):
    """(phases shown, sumo command, session) of SimSession + ActuatedController."""
    commands = []
    build = session_mod.sumo_command

    def record(*args, **kwargs):
        commands.append(build(*args, **kwargs))
        return commands[-1]

    monkeypatch.setattr(session_mod, "sumo_command", record)
    ctrl = make_controller(cfg)
    sim = SimSession(cfg, seed, tmp_path / "py", log_state=False, demand_dir=tmp_path / "demand")
    with sim:
        ctrl.reset(seed)
        for _ in range(STEPS):
            sim.step(ctrl.decide(sim.observe()))
    return list(sim.phases_shown), commands[0], sim


def run_native(cfg: dict, cmd: list[str], layout: DetectorLayout, out: Path) -> list[int]:
    """getPhase() before each step of the same run with SUMO's actuated program."""
    out.mkdir(parents=True)
    program = out / "tls_actuated.add.xml"
    program.write_text(native_program(cfg, layout), encoding="utf-8")
    cmd = list(cmd)
    i = cmd.index("-a") + 1
    cmd[i] = f"{cmd[i]},{program}"
    for opt in ("--tripinfo-output", "--stop-output", "--statistic-output"):
        if opt in cmd:
            j = cmd.index(opt) + 1
            cmd[j] = str(out / Path(cmd[j]).name)
    tls = cfg["network"]["tls_id"]
    traci = traci_module(gui=False)
    traci.start(cmd)
    try:
        assert traci.trafficlight.getProgram(tls) == "act"
        # SUMO uses our loops and builds none of its own
        assert set(traci.inductionloop.getIDList()) == {lp.id for lp in layout.loops}
        seen = []
        for _ in range(STEPS + 1):
            seen.append(traci.trafficlight.getPhase(tls))
            traci.simulationStep()
    finally:
        traci.close()
    return seen


def green_lengths(shown: list[int], cfg: dict) -> dict[str, list[int]]:
    """Complete green durations (steps) per stage."""
    green = {s["green_phase"]: s["name"] for s in cfg["network"]["stages"]}
    runs: list[list[int]] = []
    for p in shown:
        if runs and runs[-1][0] == p:
            runs[-1][1] += 1
        else:
            runs.append([p, 1])
    out: dict[str, list[int]] = {name: [] for name in green.values()}
    for p, n in runs[:-1]:
        if p in green:
            out[green[p]].append(n)
    return out


def grid_actuated(base_scenario: dict) -> dict:
    """The grid with S1 sections (the analyst's variant, max greens 40/12/25/10)."""
    names = [s["name"] for s in base_scenario["network"]["stages"]]
    return {
        **base_scenario,
        "scenario": "grid_actuated_oracle",
        "controller": "actuated",
        "detectors": {
            "control": {"distance_m": None, "detector_gap_s": 2.0, "passing_time_s": 1.9}
        },
        "actuated": {
            "max_gap_s": 3.0,
            "min_green_s": dict.fromkeys(names, 6),
            "max_green_s": dict(zip(names, [40, 12, 25, 10], strict=True)),
        },
    }


@pytest.mark.parametrize(
    "scenario, seed",
    [("grid", 1), ("pitie", 101)],
)
def test_python_actuated_matches_native_sumo(scenario, seed, base_scenario, tmp_path, monkeypatch):
    if scenario == "grid":
        cfg = grid_actuated(base_scenario)
    else:
        cfg = load_scenario(REPO_ROOT / PITIE_ACTUATED_CONFIG)
    shown, cmd, sim = run_python(cfg, seed, tmp_path, monkeypatch)
    assert sum(sim.guard.interventions.values()) == 0

    seen = run_native(cfg, cmd, sim.layout, tmp_path / "native")
    native = seen[1:]  # phase used during step t, as SimSession shows it
    diff = next((t for t, (a, b) in enumerate(zip(shown, native, strict=True)) if a != b), None)
    if diff is not None:
        window = slice(max(0, diff - 5), diff + 10)
        pytest.fail(f"first difference at t={diff}: py {shown[window]} native {native[window]}")

    # the comparison covers real actuation: greens were extended and gapped out
    act = cfg["actuated"]
    actuated = [s for s in act["min_green_s"] if act["min_green_s"][s] < act["max_green_s"][s]]
    greens = [
        (n, act["min_green_s"][s], act["max_green_s"][s])
        for s, ns in green_lengths(shown, cfg).items()
        if s in actuated
        for n in ns
    ]
    assert any(n > lo for n, lo, _ in greens), "no green was extended"
    assert any(n < hi for n, _, hi in greens), "no green gapped out"

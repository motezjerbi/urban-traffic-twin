"""Locate the SUMO binaries, pick the TraCI backend and build the `sumo` command line."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import ModuleType

from traffic_twin.paths import SUMO_DIR

DRAIN_S = 1800  # extra time after the demand window to let the network empty
GUI_SETTINGS = SUMO_DIR / "gui" / "view.xml"  # sumo-gui view: colours, zoom on the junction, delay


def find_binary(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    try:  # pip install eclipse-sumo ships the binaries inside the package
        import sumo

        for exe in (name, name + ".exe"):  # Windows wheels ship sumo.exe
            candidate = Path(sumo.SUMO_HOME) / "bin" / exe
            if candidate.exists():
                return str(candidate)
    except ImportError:
        pass
    raise FileNotFoundError(f"{name} not found: install SUMO or `pip install eclipse-sumo`")


def traci_module(gui: bool = False) -> ModuleType:
    """libsumo is much faster; TraCI is needed for sumo-gui (and is the fallback)."""
    if not gui:
        try:
            import libsumo

            return libsumo
        except ImportError:
            pass
    import traci

    return traci


def sumo_command(
    cfg: dict, seed: int, routes: Path, additional: Path, tripinfo: Path, stops: Path, gui: bool
) -> list[str]:
    begin = int(cfg["begin_s"])
    end = begin + int(cfg["duration_s"])
    cmd = [
        find_binary("sumo-gui" if gui else "sumo"),
        "-n", cfg["network"]["net_file"], "-r", str(routes), "-a", str(additional),
        "--begin", str(begin), "--end", str(end + DRAIN_S),
        "--step-length", str(float(cfg.get("step_length_s", 1.0))), "--seed", str(seed),
        "--tripinfo-output", str(tripinfo), "--stop-output", str(stops),
        "--device.emissions.probability", "0",
        "--no-step-log", "true", "--no-warnings", "true",
        "--time-to-teleport", "300",
    ]  # fmt: skip
    if gui:  # display only: the GUI view never changes what is simulated
        cmd += ["--gui-settings-file", str(GUI_SETTINGS), "--start", "true"]  # no play button
    return cmd

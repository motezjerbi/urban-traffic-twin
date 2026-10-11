"""Locate the SUMO binaries, pick the TraCI backend and build the `sumo` command line."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from traffic_twin.paths import SUMO_DIR

DRAIN_S = 1800  # extra time after the demand window to let the network empty
GUI_SETTINGS = SUMO_DIR / "gui" / "view.xml"  # default sumo-gui view (`network.gui_settings`)


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
    cfg: dict,
    seed: int,
    routes: Path,
    additional: Path | Sequence[Path],
    tripinfo: Path,
    stops: Path,
    gui: bool,
    statistics: Path | None = None,
) -> list[str]:
    """The `sumo` (or `sumo-gui`) command of one run. `additional` may list several
    files (`-a a,b`); `cfg["sumo"]["options"]` are appended to the simulation options."""
    begin = int(cfg["begin_s"])
    end = begin + int(cfg["duration_s"])
    adds = [additional] if isinstance(additional, str | Path) else list(additional)
    cmd = [
        find_binary("sumo-gui" if gui else "sumo"),
        "-n", cfg["network"]["net_file"], "-r", str(routes), "-a", ",".join(map(str, adds)),
        "--begin", str(begin), "--end", str(end + DRAIN_S),
        "--step-length", str(float(cfg.get("step_length_s", 1.0))), "--seed", str(seed),
        "--tripinfo-output", str(tripinfo), "--stop-output", str(stops),
        "--device.emissions.probability", "0",
        "--no-step-log", "true", "--no-warnings", "true",
        "--time-to-teleport", "300",
    ]  # fmt: skip
    if statistics is not None:  # passive: teleports, collisions, insertion backlog
        cmd += ["--statistic-output", str(statistics)]
    cmd += [str(o) for o in (cfg.get("sumo") or {}).get("options", [])]
    if gui:  # display only: the GUI view never changes what is simulated
        settings = cfg["network"].get("gui_settings") or GUI_SETTINGS
        cmd += ["--gui-settings-file", str(settings), "--start", "true"]  # no play button
    return cmd

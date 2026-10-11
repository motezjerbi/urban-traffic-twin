"""Build the SUMO network from the plain XML description with netconvert.

    python -m traffic_twin.sim.network                       the Part 2 grid (legacy build)
    python -m traffic_twin.sim.network --config <scenario>   the scenario's network

A scenario with a `network.build` section (plain-XML prefix, signal program `.tll.xml`
and netconvert options) is built from it into `network.net_file`; a scenario without one
(the grid) uses the legacy build from sumo/network/plain/.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from traffic_twin.config import get, load_config
from traffic_twin.paths import NET_FILE, PLAIN_NET_DIR
from traffic_twin.sim.sumo import find_binary

NETCONVERT_OPTIONS = [
    "--sidewalks.guess",
    "--crossings.guess",
    "--tls.default-type", "static",
    "--tls.layout", "opposites",
    "--tls.yellow.time", "3",
    "--tls.allred.time", "2",
    "--tls.left-green.time", "10",  # protected left phase for the main axis
    "--no-turnarounds",
]  # fmt: skip


def netconvert_command(plain_dir: Path, out_file: Path) -> list[str]:
    # POSIX paths: netconvert copies them into the .net.xml header, which must not
    # depend on the OS that built it
    return [
        find_binary("netconvert"),
        "-n", (plain_dir / "intersection.nod.xml").as_posix(),
        "-e", (plain_dir / "intersection.edg.xml").as_posix(),
        "-x", (plain_dir / "intersection.con.xml").as_posix(),
        *NETCONVERT_OPTIONS,
        "-o", out_file.as_posix(),
    ]  # fmt: skip


def build_command(cfg: dict) -> list[str]:
    """netconvert command for a scenario: from `network.build` (plain files
    `<plain_prefix>.{nod,edg,con}.xml`, optional `tll` program and `options`) into
    `network.net_file`; the legacy grid command when there is no build section."""
    out_file = Path(get(cfg, "network.net_file"))
    build = get(cfg, "network.build", None)
    if not build:
        return netconvert_command(PLAIN_NET_DIR, out_file)
    prefix = Path(build["plain_prefix"]).as_posix()
    cmd = [
        find_binary("netconvert"),
        "-n", f"{prefix}.nod.xml", "-e", f"{prefix}.edg.xml", "-x", f"{prefix}.con.xml",
    ]  # fmt: skip
    if build.get("tll"):
        cmd += ["-i", Path(build["tll"]).as_posix()]
    return [*cmd, *(str(o) for o in build.get("options", [])), "-o", out_file.as_posix()]


def netconvert_env() -> dict[str, str]:
    """Environment for SUMO tools: SUMO_HOME set to the pip package when unset (without
    it netconvert warns that it cannot validate its XML inputs)."""
    env = dict(os.environ)
    if not env.get("SUMO_HOME"):
        try:
            import sumo

            env["SUMO_HOME"] = str(sumo.SUMO_HOME)
        except ImportError:
            pass
    return env


def _netconvert(cmd: list[str]) -> None:
    run = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=netconvert_env(),
    )
    if run.returncode:  # e.g. "--" inside an XML comment of a plain file
        raise RuntimeError(f"netconvert failed ({run.returncode}):\n{run.stdout}{run.stderr}")


def build_network(plain_dir: Path, out_file: Path) -> Path:
    _netconvert(netconvert_command(plain_dir, out_file))
    return out_file


def build(cfg: dict) -> Path:
    """Build the scenario's network; returns `network.net_file`."""
    _netconvert(build_command(cfg))
    return Path(get(cfg, "network.net_file"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, help="scenario config (default: the grid)")
    parser.add_argument("--plain", type=Path, default=PLAIN_NET_DIR, help="grid plain XML folder")
    parser.add_argument("--out", type=Path, default=NET_FILE, help="grid .net.xml")
    args = parser.parse_args(argv)
    if args.config is None:
        print(build_network(args.plain, args.out))
        return
    # network section only: building the net does not need a complete scenario
    print(build(load_config(args.config, ("network.net_file",))))


if __name__ == "__main__":
    main()

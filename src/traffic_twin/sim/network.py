"""Build the SUMO network from the plain XML description with netconvert."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

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


def build_network(plain_dir: Path, out_file: Path) -> Path:
    subprocess.run(
        netconvert_command(plain_dir, out_file), check=True, capture_output=True, text=True
    )
    return out_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", type=Path, default=PLAIN_NET_DIR)
    parser.add_argument("--out", type=Path, default=NET_FILE)
    args = parser.parse_args(argv)
    print(build_network(args.plain, args.out))


if __name__ == "__main__":
    main()

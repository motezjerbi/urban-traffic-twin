"""Build the SUMO network from the plain XML description with netconvert."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

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


def find_binary(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    try:  # pip install eclipse-sumo ships the binaries inside the package
        import sumo

        candidate = Path(sumo.SUMO_HOME) / "bin" / name
        if candidate.exists():
            return str(candidate)
    except ImportError:
        pass
    raise FileNotFoundError(f"{name} not found: install SUMO or `pip install eclipse-sumo`")


def build_network(plain_dir: Path, out_file: Path) -> Path:
    cmd = [
        find_binary("netconvert"),
        "-n", str(plain_dir / "intersection.nod.xml"),
        "-e", str(plain_dir / "intersection.edg.xml"),
        "-x", str(plain_dir / "intersection.con.xml"),
        *NETCONVERT_OPTIONS,
        "-o", str(out_file),
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", default="sumo/network/plain")
    parser.add_argument("--out", default="sumo/network/intersection.net.xml")
    args = parser.parse_args(argv)
    print(build_network(Path(args.plain), Path(args.out)))


if __name__ == "__main__":
    main()

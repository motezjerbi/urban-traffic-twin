"""`tasks.py bootstrap`: make a fresh clone demo-ready (D116). It never rebuilds the
evaluation (that would run seeds 1-10 outside the registered Part 8 run).

- `--from-pack [docs/demo/demo_pack.zip]` (default): verify every entry against the
  manifest (sha256, size, safe paths), refuse to overwrite any existing file (nothing is
  written then), and unpack at the same repo-relative paths;
- `--demo-only`: rebuild the missing demo runs at seed 4000 by simulation
  (`ml.predict demo --seeds 4000 --configs ...`, about 5-10 s per run with SUMO); an
  existing run folder is never overwritten. No evaluation output is made.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from traffic_twin.demo.common import DemoError, child_env
from traffic_twin.demo.pack import verify
from traffic_twin.paths import CONFIG_DIR, DEMO_DIR

DEMO_ONLY_SEED = 4000


def bootstrap(pack: Path | None, root: Path, *, demo_only: bool = False) -> list[Path]:
    """The files restored (from the pack) or the run folders made (`demo_only`)."""
    root = Path(root).resolve()
    if demo_only:
        return rebuild_demo_runs(root)
    if pack is None:
        raise DemoError("bootstrap needs a pack (--from-pack) or --demo-only")
    pack = Path(pack) if Path(pack).is_absolute() else root / pack
    manifest = verify(pack)
    names = [e["path"] for e in manifest["files"]]
    exists = [n for n in names if (root / n).exists()]
    if exists:
        raise DemoError(
            f"bootstrap refuses to overwrite {len(exists)} existing file(s), e.g. "
            f"{exists[:3]}; remove them first (nothing was written)"
        )
    written: list[Path] = []
    with zipfile.ZipFile(pack) as z:
        for name in names:
            target = (root / name).resolve()
            if not str(target).startswith(str(root)):
                raise DemoError(f"unsafe pack path {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
            written.append(target)
    return written


def demo_only_configs(cfg: dict | None = None) -> list[str]:
    """The scenario configs of the pack's demo runs (configs/<scenario>.yaml)."""
    from traffic_twin.demo.config import load_demo_config

    cfg = cfg or load_demo_config()
    return [
        (CONFIG_DIR / f"{s}.yaml").as_posix() for s in (cfg.get("pack") or {}).get("runs") or []
    ]


def rebuild_demo_runs(root: Path, cfg: dict | None = None) -> list[Path]:
    """`ml.predict demo --seeds 4000` for the pack's scenarios that have no run folder in
    data/demo yet (existing folders are never overwritten); the folders made."""
    from traffic_twin.demo.config import load_demo_config

    cfg = cfg or load_demo_config(root=root)
    seed = int((cfg.get("pack") or {}).get("seed", DEMO_ONLY_SEED))
    if seed != DEMO_ONLY_SEED:
        raise DemoError(f"--demo-only rebuilds seed {DEMO_ONLY_SEED} only (D101), got {seed}")
    configs = demo_only_configs(cfg)
    missing = [c for c in configs if not (root / c).is_file()]
    if missing:
        raise DemoError(f"scenario configs not found: {missing}")
    todo = [c for c in configs if not (root / DEMO_DIR / f"{Path(c).stem}_seed{seed}").exists()]
    dirs = [root / DEMO_DIR / f"{Path(c).stem}_seed{seed}" for c in todo]
    if not todo:
        print(f"every demo run already exists under {DEMO_DIR.as_posix()}: nothing to do")
        return []
    cmd = [
        sys.executable,
        "-m",
        "traffic_twin.ml.predict",
        "demo",
        "--seeds",
        str(seed),
        "--configs",
        *todo,
    ]
    cp = subprocess.run(cmd, cwd=str(root), env=child_env())
    if cp.returncode != 0:
        raise DemoError(f"ml.predict demo exited with code {cp.returncode}")
    return [d for d in dirs if (d / "run.json").is_file()]

"""`tasks.py demo-clean`: remove demo leftovers. Never touches data/results, data/demo,
data/models or models.

- stops the processes listed in data/demo_check/.pids.json (a demo that crashed), after
  checking that each pid is still a Python / Streamlit process;
- removes old data/demo_check/<ts>-<kind>/ folders (the newest `keep` stay) and the
  `stray/` folder of moved `last_dom.html` files;
- removes a stray `last_dom.html` in the repo root (the desktop browser pane writes one);
- removes the demo's temporary folders (`tt_demo_*` in the system temp folder) older than
  an hour (a younger one may belong to a running rehearsal).
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from traffic_twin.demo.common import (
    PIDS_FILE,
    WINDOWS,
    DemoError,
    image_name,
    kill_tree,
    pid_alive,
    protected,
    under,
)
from traffic_twin.paths import DEMO_CHECK_DIR

STAMP_DIR = re.compile(r"^\d{8}-\d{6}-[a-z]+$")
TMP_PREFIXES = ("tt_demo_",)
OUR_IMAGES = ("python", "streamlit", "uvicorn")
TMP_MIN_AGE_S = 3600.0  # a younger temporary folder may belong to a running rehearsal


def _allowed(path: Path, root: Path) -> bool:
    """Only the demo's own leftovers: under data/demo_check, the root last_dom.html, or a
    `tt_demo_*` temporary folder; never a protected folder."""
    path = Path(path)
    if protected(path, root):
        return False
    if under(path, root / DEMO_CHECK_DIR):
        return True
    if path.resolve() == (root / "last_dom.html").resolve():
        return True
    tmp = Path(tempfile.gettempdir())
    return (
        under(path, tmp)
        and path.name.startswith(TMP_PREFIXES)
        and path.parent.resolve() == tmp.resolve()
    )


def _remove(path: Path, root: Path, dry_run: bool) -> bool:
    if not _allowed(path, root):
        raise DemoError(f"demo clean refused to remove {path}")
    if dry_run:
        return True
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError:
            return False
    return not path.exists()


def stop_leftovers(root: Path, dry_run: bool = False) -> list[int]:
    """Stop the processes of a crashed demo (pids file); the pids stopped."""
    f = root / DEMO_CHECK_DIR / PIDS_FILE
    if not f.is_file():
        return []
    try:
        children = json.loads(f.read_text(encoding="utf-8")).get("children") or []
    except ValueError:
        children = []
    stopped = []
    for c in children:
        pid = int(c.get("pid") or 0)
        if pid <= 0 or not pid_alive(pid):
            continue
        if WINDOWS:
            name = (image_name(pid) or "").lower()
            if not name.startswith(OUR_IMAGES):
                continue  # the pid was reused by another program
        if not dry_run:
            kill_tree(pid, grace_s=3.0)
        stopped.append(pid)
    if not dry_run:
        f.unlink(missing_ok=True)
    return stopped


def clean(root: Path, *, keep: int = 3, dry_run: bool = False) -> list[Path]:
    """The paths removed (or that would be, with `dry_run`)."""
    root = Path(root).resolve()
    stop_leftovers(root, dry_run)
    removed: list[Path] = []
    check = root / DEMO_CHECK_DIR
    if check.is_dir():
        runs = sorted(
            (d for d in check.iterdir() if d.is_dir() and STAMP_DIR.match(d.name)),
            key=lambda d: d.name,
        )
        old = runs[: max(0, len(runs) - max(0, keep))]
        for d in [*old, check / "stray"]:
            if d.exists() and _remove(d, root, dry_run):
                removed.append(d)
    stray = root / "last_dom.html"
    if stray.is_file() and _remove(stray, root, dry_run):
        removed.append(stray)
    tmp = Path(tempfile.gettempdir())
    for d in sorted(tmp.glob("tt_demo_*")):
        try:
            young = time.time() - d.stat().st_mtime < TMP_MIN_AGE_S
        except OSError:
            continue
        if d.is_dir() and not young and _remove(d, root, dry_run):
            removed.append(d)
    return removed

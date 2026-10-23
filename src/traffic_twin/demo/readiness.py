"""`tasks.py demo-readiness`: the A-09 proof on a clean checkout (D116).

1. `git clone --local --no-hardlinks` of the repo into a temporary folder, detached at
   HEAD (uncommitted changes are therefore not part of the test: the result names the
   commit and whether the source tree was dirty);
2. the clone's own code is used: `PYTHONPATH=<clone>/src` (checked: `traffic_twin` is
   imported from the clone, not from the editable install);
3. `python -m traffic_twin.demo bootstrap --from-pack docs/demo/demo_pack.zip` in the clone
   (the committed pack; the source repo's pack when it is not committed yet, recorded);
4. `python -m traffic_twin.demo rehearsal --fresh-clone` in the clone;
5. the clone's `git status` must still be clean (no tracked or untracked change);
6. the result is recorded as the `readiness` block of docs/demo/rehearsal.json (A-09 PASS
   once it passed on a clean commit), unless `--no-record`.

It never rebuilds the evaluation and runs no simulation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from traffic_twin.demo.common import DemoError, Report, child_env, git_info, now_iso
from traffic_twin.paths import DEMO_PACK

# the clone rows A-09 needs as PASS (rehearsal.evidence): no clone, no A-09
CLONE_CHECKS = (
    "git clone --local",
    "traffic_twin is imported from the clone",
    "bootstrap --from-pack",
    "clone tree still clean after the rehearsal",
)


def _run(cmd: list[str], cwd: Path, env: dict, log: Path, timeout: float) -> int:
    with log.open("w", encoding="utf-8") as fh:
        try:
            cp = subprocess.run(
                cmd, cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            fh.write(f"\nTIMEOUT after {timeout:.0f} s\n")
            return -1
    return cp.returncode


def readiness(
    cfg: dict,
    *,
    root: Path | None = None,
    browser: bool = True,
    record: bool = True,
    keep: bool = False,
    pack: Path | None = None,
    quiet: bool = False,
) -> dict:
    """The readiness report {status, passed, summary, rows, git, clone, ...}. Recording
    (the default) needs `browser`: a run without DOM checks is no A-09 evidence."""
    from traffic_twin.api.settings import find_repo_root
    from traffic_twin.demo.common import check_dir

    if record and not browser:
        raise DemoError(
            "readiness records A-09 and needs the DOM checks: drop --no-browser "
            "(or add --no-record)"
        )
    root = Path(root or find_repo_root()).resolve()
    out = check_dir(root, "readiness")
    rep = Report(quiet=quiet)
    t0 = time.monotonic()
    info = git_info(root)
    result: dict = {
        "kind": "readiness",
        "mode": "real",
        "fresh_clone": True,
        "started_at": now_iso(),
        "source_dirty": info["dirty"],
    }
    if info["full"] is None:
        raise DemoError("readiness needs a git checkout")
    rep.add(
        "clone",
        "source tree",
        "PASS" if not info["dirty"] else "PENDING",
        f"HEAD {info['commit']}"
        + (" (dirty: uncommitted changes are not tested)" if info["dirty"] else ""),
    )
    work = Path(tempfile.mkdtemp(prefix="tt_demo_readiness_"))
    clone = work / "clone"
    try:
        cp = subprocess.run(
            ["git", "clone", "--quiet", "--local", "--no-hardlinks", str(root), str(clone)],
            capture_output=True,
            text=True,
        )
        if not rep.check("clone", CLONE_CHECKS[0], cp.returncode == 0, cp.stderr.strip()[:200]):
            return _finish(result, rep, t0, out, root, record)
        subprocess.run(
            ["git", "-C", str(clone), "checkout", "--quiet", "--detach", info["full"]],
            capture_output=True,
        )
        env = child_env(
            PYTHONPATH=str(clone / "src"),
            TT_API_CONFIG=None,
            TT_RESULTS_DIRS=None,
            TT_API_INPROCESS=None,
            TT_API_URL=None,
        )
        who = subprocess.run(
            [sys.executable, "-c", "import traffic_twin; print(traffic_twin.__file__)"],
            cwd=str(clone),
            env=env,
            capture_output=True,
            text=True,
        )
        where = Path(who.stdout.strip() or ".").resolve()
        rep.check(
            "clone",
            CLONE_CHECKS[1],
            str(where).startswith(str(clone.resolve())),
            who.stdout.strip() or who.stderr.strip()[:200],
        )
        committed = clone / DEMO_PACK
        src_pack = Path(pack) if pack else root / DEMO_PACK
        use = committed if committed.is_file() else src_pack
        result["pack_committed"] = committed.is_file()
        rep.add(
            "clone",
            "demo pack",
            "PASS" if committed.is_file() else "PENDING",
            f"{DEMO_PACK.as_posix()} is committed"
            if committed.is_file()
            else f"not committed: using {use.name} from the source tree",
        )
        if not use.is_file():
            rep.add(
                "clone",
                "demo pack present",
                "FAIL",
                f"{DEMO_PACK.as_posix()} not found (python tasks.py demo-pack)",
            )
            return _finish(result, rep, t0, out, root, record)
        code = _run(
            [sys.executable, "-m", "traffic_twin.demo", "bootstrap", "--from-pack", str(use)],
            clone,
            env,
            out / "bootstrap.log",
            300,
        )
        if not rep.check("clone", CLONE_CHECKS[2], code == 0, f"exit {code}"):
            return _finish(result, rep, t0, out, root, record)
        rj = work / "rehearsal_report"
        cmd = [
            sys.executable,
            "-m",
            "traffic_twin.demo",
            "rehearsal",
            "--fresh-clone",
            "--out",
            str(rj),
        ] + ([] if browser else ["--no-browser"])
        code = _run(cmd, clone, env, out / "rehearsal.log", 900)
        try:
            inner = json.loads((rj / "report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            inner = None
        if inner is None:
            rep.add("clone", "rehearsal in the clone", "FAIL", f"exit {code}, no report")
            return _finish(result, rep, t0, out, root, record)
        rep.rows.extend(inner["rows"])
        for k in ("steps", "degraded", "prerequisites", "browser"):
            result[k] = inner.get(k)
        if browser and not inner.get("browser"):
            rep.add(
                "clone",
                "headless browser in the clone",
                "PENDING" if record else "SKIP",
                "no Chrome or Edge found: no DOM check, no A-09 evidence",
            )
        st = subprocess.run(
            ["git", "-C", str(clone), "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True,
            text=True,
        ).stdout
        rep.check(
            "clone",
            CLONE_CHECKS[3],
            not st.strip(),
            st.strip().replace("\n", "; ")[:200],
        )
    finally:
        if keep:
            result["clone"] = str(clone)
        else:
            shutil.rmtree(work, ignore_errors=True)
    return _finish(result, rep, t0, out, root, record)


def _finish(result: dict, rep: Report, t0: float, out: Path, root: Path, record: bool) -> dict:
    from traffic_twin.demo.common import write_report
    from traffic_twin.demo.rehearsal import REHEARSAL_JSON, record_rehearsal

    info = git_info(root)
    result.update(
        status=rep.status,
        passed=rep.passed,
        summary=rep.summary(),
        git=info,
        duration_s=round(time.monotonic() - t0, 1),
        rows=rep.rows,
    )
    write_report(
        out,
        result,
        rep.markdown(
            "Demo readiness (fresh clone)",
            [f"- commit: {info['code_version']}", f"- duration: {result['duration_s']} s"],
        ),
    )
    if record:
        record_rehearsal(result, root / REHEARSAL_JSON, root)
    return result

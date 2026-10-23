"""Shared demo plumbing: ports and the impostor probe, process groups with a tree
teardown (Windows: CTRL_BREAK, then `taskkill /T /F`), HTTP waits, git facts and the
check report (PASS / FAIL / PENDING / SKIP rows).

No psutil: process trees are stopped with `taskkill /T` on Windows and process groups
(`os.killpg`) elsewhere.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from traffic_twin.paths import DEMO_CHECK_DIR

PIDS_FILE = ".pids.json"
RESULTS = ("PASS", "FAIL", "PENDING", "SKIP")
WINDOWS = sys.platform == "win32"
PROTECTED = ("data/results", "data/demo", "data/models", "models")  # never written by demo


class DemoError(RuntimeError):
    """A demo command cannot go on (prerequisite, busy port, impostor, bad pack)."""


# -- time and git ------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def git_info(root: Path) -> dict:
    """{commit (short), full, dirty, code_version}: `sim.results.code_version` semantics
    (ignored files and docs/ do not count)."""
    git = ["git", "--no-optional-locks", "-C", str(root)]
    try:
        full = subprocess.run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            [
                *git,
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--",
                ".",
                ":(exclude)docs",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "full": None, "dirty": True, "code_version": "unknown"}
    dirty = bool(status.strip())
    short = full[:7]
    return {
        "commit": short,
        "full": full,
        "dirty": dirty,
        "code_version": short + ("-dirty" if dirty else ""),
    }


def check_dir(root: Path, kind: str, out_dir: Path | None = None) -> Path:
    """`out_dir`, or `<root>/data/demo_check/<timestamp>-<kind>/` (created)."""
    d = Path(out_dir) if out_dir else Path(root) / DEMO_CHECK_DIR / f"{stamp()}-{kind}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# -- ports and probes --------------------------------------------------------------------
def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """True when something accepts TCP connections on host:port."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def probe_api(base_url: str, title: str, timeout: float = 3.0) -> tuple[str, str]:
    """("free" | "api" | "impostor", detail) for the server at `base_url`: the real API
    answers /openapi.json with `title` ("Urban Traffic Twin API"); run_mock.py and any other
    server do not (D107)."""
    u = httpx.URL(base_url)
    if not port_open(u.host, u.port or 80):
        return "free", "nothing listening"
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/openapi.json", timeout=timeout)
        got = (r.json().get("info") or {}).get("title") if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError, AttributeError) as e:
        return "impostor", f"no valid /openapi.json ({type(e).__name__})"
    if got == title:
        return "api", f"/openapi.json title {got!r}"
    return "impostor", f"/openapi.json title {got!r} (expected {title!r}; run_mock.py?)"


def probe_streamlit(base_url: str, timeout: float = 3.0) -> tuple[str, str]:
    """("free" | "streamlit" | "other", detail) for the server at `base_url`."""
    u = httpx.URL(base_url)
    if not port_open(u.host, u.port or 80):
        return "free", "nothing listening"
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/_stcore/health", timeout=timeout)
    except httpx.HTTPError as e:
        return "other", type(e).__name__
    if r.status_code == 200 and r.text.strip() == "ok":
        return "streamlit", "/_stcore/health ok"
    return "other", f"/_stcore/health -> http {r.status_code}"


def wait_http(
    url: str, *, timeout_s: float, proc: subprocess.Popen | None = None, expect: str | None = None
) -> float:
    """Seconds until `url` answers 200 (and `expect` in the body); DemoError on timeout or
    when `proc` exits first."""
    t0 = time.monotonic()
    while True:
        if proc is not None and proc.poll() is not None:
            raise DemoError(f"process exited with code {proc.returncode} before {url} answered")
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200 and (expect is None or expect in r.text):
                return time.monotonic() - t0
        except httpx.HTTPError:
            pass
        if time.monotonic() - t0 > timeout_s:
            raise DemoError(f"{url} did not answer within {timeout_s:.0f} s")
        time.sleep(0.2)


# -- process groups ----------------------------------------------------------------------
def image_name(pid: int) -> str | None:
    """The executable name of a live Windows process (None when it is gone)."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.strip().splitlines()[:1]
    if not line or not line[0].startswith('"'):
        return None
    return line[0].split('","')[0].strip('"')


def pid_alive(pid: int) -> bool:
    if WINDOWS:
        return image_name(pid) is not None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_tree(pid: int, *, grace_s: float = 5.0, graceful: bool = True) -> None:
    """Stop process `pid` and its children: CTRL_BREAK / SIGTERM to the group first, then
    `taskkill /T /F` / SIGKILL after `grace_s`."""
    if WINDOWS:
        if graceful:
            try:
                os.kill(int(pid), signal.CTRL_BREAK_EVENT)  # the child's own process group
            except OSError:
                pass
            t0 = time.monotonic()
            while time.monotonic() - t0 < grace_s and pid_alive(pid):
                time.sleep(0.2)
        # the tree, even when the parent already left (its children are then orphans
        # that taskkill /T cannot find: hence the graceful break first)
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(int(pid))], capture_output=True, timeout=30
        )
        return
    try:
        pgid = os.getpgid(int(pid))
    except ProcessLookupError:
        return
    if graceful:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        t0 = time.monotonic()
        while time.monotonic() - t0 < grace_s and pid_alive(pid):
            time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@dataclass
class Child:
    name: str
    proc: subprocess.Popen
    port: int | None
    log: Path | None
    cmd: list[str] = field(default_factory=list)


class ProcessGroup:
    """Children started in their own process group, stopped as trees; the pids are kept
    in `pids_file` so `demo clean` can stop leftovers after a crash."""

    def __init__(self, pids_file: Path | None = None, *, stop_s: float = 10.0) -> None:
        self.children: list[Child] = []
        self.pids_file = pids_file
        self.stop_s = stop_s

    def start(
        self,
        name: str,
        cmd: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        port: int | None = None,
        log: Path | None = None,
    ) -> Child:
        kwargs: dict = {"cwd": str(cwd), "env": dict(env), "stdin": subprocess.DEVNULL}
        if WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        fh = None
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = log.open("w", encoding="utf-8")
            kwargs.update(stdout=fh, stderr=subprocess.STDOUT)
        else:
            kwargs.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            proc = subprocess.Popen(list(cmd), **kwargs)
        finally:
            if fh is not None:
                fh.close()  # the child keeps its own handle
        child = Child(name, proc, port, log, list(cmd))
        self.children.append(child)
        self._write_pids()
        return child

    def _write_pids(self) -> None:
        if self.pids_file is None:
            return
        live = [
            {"name": c.name, "pid": c.proc.pid, "port": c.port, "cmd": c.cmd[:4]}
            for c in self.children
            if c.proc.poll() is None
        ]
        self.pids_file.parent.mkdir(parents=True, exist_ok=True)
        if live:
            self.pids_file.write_text(
                json.dumps({"started_at": now_iso(), "children": live}, indent=1), encoding="utf-8"
            )
        elif self.pids_file.is_file():
            self.pids_file.unlink()

    def dead(self) -> list[Child]:
        return [c for c in self.children if c.proc.poll() is not None]

    def stop(self) -> dict[str, bool]:
        """Stop every child tree; {name: port freed (True when no port)}."""
        for c in reversed(self.children):
            if c.proc.poll() is None:
                kill_tree(c.proc.pid, grace_s=min(self.stop_s, 5.0))
            elif WINDOWS:  # parent gone: still sweep its tree in case
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(c.proc.pid)],
                    capture_output=True,
                    timeout=30,
                )
            try:
                c.proc.wait(timeout=self.stop_s)
            except subprocess.TimeoutExpired:
                c.proc.kill()
        freed = {}
        for c in self.children:
            ok = True
            if c.port:
                t0 = time.monotonic()
                while port_open("127.0.0.1", c.port) and time.monotonic() - t0 < self.stop_s:
                    time.sleep(0.2)
                ok = not port_open("127.0.0.1", c.port)
            freed[c.name] = ok
        self.children = []
        self._write_pids()
        return freed

    def __enter__(self) -> ProcessGroup:
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def api_cmd(port: int, host: str = "127.0.0.1") -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "traffic_twin.api.main:app",
        "--no-access-log",
        "--host",
        host,
        "--port",
        str(port),
    ]


def operator_cmd(script: str, port: int, host: str = "127.0.0.1") -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        script,
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.fileWatcherType",
        "none",
    ]


def child_env(**extra: str | None) -> dict[str, str]:
    """os.environ plus `extra` (None removes a variable); UTF-8 output on Windows."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    for k, v in extra.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return env


# -- the check report --------------------------------------------------------------------
class Report:
    """Rows {group, check, result, detail, ms}; `passed` = no FAIL row."""

    def __init__(self, quiet: bool = False) -> None:
        self.rows: list[dict] = []
        self.quiet = quiet

    def add(
        self, group: str, check: str, result: str, detail: str = "", ms: float | None = None
    ) -> str:
        if result not in RESULTS:
            raise ValueError(result)
        detail = " ".join(str(detail).split())
        row = {"group": group, "check": check, "result": result, "detail": detail}
        if ms is not None:
            row["ms"] = round(ms, 1)
        self.rows.append(row)
        if not self.quiet:
            print(f"[{result:7}] {group}: {check} {detail}".rstrip(), flush=True)
        return result

    def check(
        self, group: str, check: str, ok: bool, detail: str = "", ms: float | None = None
    ) -> bool:
        self.add(group, check, "PASS" if ok else "FAIL", detail, ms)
        return ok

    def extend(self, other: Report) -> None:
        self.rows.extend(other.rows)

    def count(self, result: str) -> int:
        return sum(r["result"] == result for r in self.rows)

    @property
    def passed(self) -> bool:
        return self.count("FAIL") == 0

    @property
    def status(self) -> str:
        """PASS (no FAIL, no PENDING), PENDING (only pending rows fail to pass) or FAIL."""
        if self.count("FAIL"):
            return "FAIL"
        return "PENDING" if self.count("PENDING") else "PASS"

    def summary(self) -> dict:
        return {r.lower(): self.count(r) for r in RESULTS} | {"rows": len(self.rows)}

    def markdown(self, title: str, header: Sequence[str] = ()) -> str:
        s = self.summary()
        lines = [
            f"# {title}",
            "",
            *header,
            "",
            f"**{self.status}**: {s['pass']} PASS, {s['fail']} FAIL, {s['pending']} "
            f"PENDING, {s['skip']} SKIP.",
            "",
            "| group | check | result | detail |",
            "|---|---|---|---|",
        ]
        for r in self.rows:
            res = r["result"] if r["result"] in ("PASS", "SKIP") else f"**{r['result']}**"
            detail = str(r["detail"]).replace("|", "/").replace("\n", " ")
            lines.append(f"| {r['group']} | `{r['check']}` | {res} | {detail} |")
        return "\n".join(lines) + "\n"


def write_report(out: Path, payload: dict, md: str) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    j, m = out / "report.json", out / "report.md"
    j.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    m.write_text(md, encoding="utf-8")
    return j, m


def under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def protected(path: Path, root: Path) -> bool:
    """True when `path` is (inside) one of the folders the demo never writes or removes."""
    return any(under(path, Path(root) / p) for p in PROTECTED)

"""Headless screenshots of every configs/demo.yaml step into docs/figures/app/ (D116).

Each `/mock` step becomes its deep link (D106: `view`, `scenario` and `seed` from the
step's run, then `t`, `crossing`, `speed`, `line`, `trajet`, `exam`, `kiosk`) and is
captured by `traffic_twin.app.browser.screenshot` (Chrome first, then Edge, `TT_BROWSER`
overrides). One extra shot, `error.png`, shows the error state of a bad deep link (an
unknown scenario sends the app home with its error banner).

A `front: streamlit` step captures the Operator through the Chrome DevTools protocol:
Chrome's `--virtual-time-budget` does not wait for Streamlit's websocket, so the page is
driven over a debugging connection, given real time to render, and shot once per tab
(`<id>.png` is the first tab, `<id>_<tab>.png` every tab). This needs the `websockets`
client that the app stack installs; without it the step is skipped with a message.

Without `--api-url` the API is started on a free local port (uvicorn, the current
`TT_API_CONFIG` or configs/api.yaml) and the Operator (Streamlit) on another, both torn
down afterwards. No simulation runs; nothing is written outside `out`.

    python -m traffic_twin.report shots [--out docs/figures/app] [--api-url URL]
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

import yaml

from traffic_twin.paths import APP_FIGURES, DEMO_CONFIG

SIZE = (1280, 800)
MOCK_WAIT_MS = 5000  # the 2D/3D twin needs a few frames
STREAMLIT_SETTLE_S = 45.0  # real seconds for the first Streamlit run
TAB_SETTLE_S = 2.0
STREAMLIT_SIZE = (1280, 1000)
# Streamlit rendered: its app container, no loading skeleton, no 'running' status
ST_READY_JS = (
    "!!document.querySelector('[data-testid=stApp]') && "
    "!document.querySelector('[data-testid=stSkeleton]') && "
    "!document.querySelector('[data-testid=stStatusWidget]') && "
    "document.querySelectorAll('[data-testid=stMarkdownContainer]').length > 0"
)
DEEP_LINK_KEYS = ("t", "crossing", "speed", "line", "trajet", "exam", "kiosk", "play")
ERROR_SHOT = {"id": "error", "view": "pedestrian", "scenario": "unknown_scenario", "seed": 4000}


class ShotsError(RuntimeError):
    """Screenshots cannot be taken (no browser, the API does not start)."""


def load_demo_config(path: Path = DEMO_CONFIG) -> dict:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ShotsError(f"{Path(path).as_posix()} not found ({exc})") from exc


def step_url(step: dict, cfg: dict, base: str) -> str:
    """The `/mock` deep link of a demo step (D106). Raises ShotsError on an unknown run."""
    params: dict[str, object] = {"view": step.get("view", "home")}
    run = step.get("run")
    if run is not None:
        runs = cfg.get("runs") or {}
        if run not in runs:
            raise ShotsError(f"step {step.get('id')}: unknown run {run!r} ({', '.join(runs)})")
        params["scenario"] = runs[run]["scenario"]
        params["seed"] = int(runs[run]["seed"])
    for k in ("scenario", "seed"):
        if k in step:
            params[k] = step[k]
    for k in DEEP_LINK_KEYS:
        if k in step:
            params[k] = step[k]
    params.setdefault("play", 0)  # a still frame
    return f"{base.rstrip('/')}/mock?{urlencode(params)}"


def plan(cfg: dict, api_url: str, operator_url: str | None) -> list[tuple[str, str, int]]:
    """(name, url, wait_ms) of the `/mock` shots: every demo step plus the error state
    (the Streamlit step is taken by `operator_shots`)."""
    shots = []
    for step in cfg.get("steps") or []:
        sid = str(step.get("id"))
        if step.get("front") == "streamlit":
            continue  # taken over the DevTools protocol (operator_shots)
        shots.append((sid, step_url(step, cfg, api_url), MOCK_WAIT_MS))
    shots.append(("error", step_url(ERROR_SHOT, cfg, api_url), MOCK_WAIT_MS))
    return shots


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_http(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status < 500:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.3)
    raise ShotsError(f"{url} did not answer within {timeout_s:.0f} s")


@contextlib.contextmanager
def servers(with_operator: bool = True):
    """Start the API (and the Operator) on free local ports; yields (api_url,
    operator_url or None) and stops both."""
    procs: list[subprocess.Popen] = []
    try:
        port = free_port()
        api = f"http://127.0.0.1:{port}"
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "traffic_twin.api.main:app",
                    "--no-access-log",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
        wait_http(f"{api}/health")
        operator = None
        if with_operator and Path("app.py").is_file():
            oport = free_port()
            env = {**os.environ, "TT_API_URL": api}
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        "app.py",
                        "--server.port",
                        str(oport),
                        "--server.address",
                        "127.0.0.1",
                        "--server.headless",
                        "true",
                        "--browser.gatherUsageStats",
                        "false",
                    ],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
            operator = f"http://127.0.0.1:{oport}"
            wait_http(operator, timeout_s=60)
        yield api, operator
    finally:
        for p in reversed(procs):
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def take_shots(
    demo_cfg: dict,
    out: Path = APP_FIGURES,
    *,
    api_url: str | None = None,
    operator_url: str | None = None,
    size: tuple[int, int] = SIZE,
) -> list[Path]:
    """The PNG files written: one per demo step (`<id>.png`) plus `error.png`."""
    from traffic_twin.app import browser

    if browser.find_browser() is None:
        raise ShotsError("no headless Chrome or Edge found (set TT_BROWSER)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        if api_url is None:
            api_url, started_op = stack.enter_context(servers(with_operator=operator_url is None))
            operator_url = operator_url or started_op
        written = []
        for name, url, wait_ms in plan(demo_cfg, api_url, operator_url):
            written.append(browser.screenshot(url, out / f"{name}.png", size=size, wait_ms=wait_ms))
        streamlit = [s for s in demo_cfg.get("steps") or [] if s.get("front") == "streamlit"]
        if streamlit and operator_url:
            try:
                written += operator_shots(operator_url, out, str(streamlit[0].get("id")))
            except ImportError:
                print("websockets not installed: Streamlit Operator screenshots skipped")
    return [Path(p) for p in written]


# -- the Streamlit Operator over the DevTools protocol ---------------------------------
def _slug(text: str) -> str:
    """'Sécurité et ambulance' -> 'securite_et_ambulance'."""
    ascii_ = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_.lower()).strip("_")


def operator_shots(
    url: str, out: Path, name: str = "streamlit", size: tuple[int, int] = STREAMLIT_SIZE
) -> list[Path]:
    """`<name>.png` (first tab) and `<name>_<tab>.png` for every tab of the Operator."""
    import websockets  # noqa: F401 - fail early (ImportError) when missing

    from traffic_twin.app import browser

    exe = browser.find_browser()
    if exe is None:
        raise ShotsError("no headless Chrome or Edge found (set TT_BROWSER)")
    profile = tempfile.mkdtemp(prefix="tt_shots_")
    proc = subprocess.Popen(
        [
            str(exe),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--hide-scrollbars",
            f"--user-data-dir={profile}",
            f"--window-size={size[0]},{size[1]}",
            "--remote-debugging-port=0",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=profile,
    )
    try:
        port = _devtools_port(Path(profile))
        return asyncio.run(_operator(port, url, Path(out), name, size))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


def _devtools_port(profile: Path, timeout_s: float = 20.0) -> int:
    f = profile / "DevToolsActivePort"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if f.is_file():
            first = f.read_text(encoding="utf-8").splitlines()[:1]
            if first and first[0].strip().isdigit():
                return int(first[0])
        time.sleep(0.2)
    raise ShotsError("the browser did not open its DevTools port")


TABS_JS = "Array.from(document.querySelectorAll('[role=tab]')).map(b => b.innerText.trim())"
# click tab i and scroll the tab bar to the top, so its panel fills the screenshot
TAB_CLICK_JS = (
    "(() => {{ const t = document.querySelectorAll('[role=tab]')[{i}]; t.click(); "
    "t.scrollIntoView({{block: 'start'}}); return true; }})()"
)


async def _operator(port: int, url: str, out: Path, name: str, size) -> list[Path]:
    import websockets

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as r:
        page = next(t for t in json.loads(r.read()) if t.get("type") == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
        seq = 0

        async def call(method: str, **params):
            nonlocal seq
            seq += 1
            await ws.send(json.dumps({"id": seq, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == seq:
                    if "error" in msg:
                        raise ShotsError(f"DevTools {method}: {msg['error']}")
                    return msg.get("result", {})

        async def js(expr: str):
            res = await call("Runtime.evaluate", expression=expr, returnByValue=True)
            return (res.get("result") or {}).get("value")

        async def shot(path: Path) -> Path:
            data = await call("Page.captureScreenshot", format="png")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(data["data"]))
            return path

        await call("Page.enable")
        await call(
            "Emulation.setDeviceMetricsOverride",
            width=size[0],
            height=size[1],
            deviceScaleFactor=1,
            mobile=False,
        )
        await call("Page.navigate", url=url)
        deadline = time.monotonic() + STREAMLIT_SETTLE_S
        while time.monotonic() < deadline and not await js(ST_READY_JS):
            await asyncio.sleep(0.5)
        await asyncio.sleep(TAB_SETTLE_S)
        written = [await shot(out / f"{name}.png")]
        for i, label in enumerate(await js(TABS_JS) or []):
            await js(TAB_CLICK_JS.format(i=i))
            await asyncio.sleep(TAB_SETTLE_S)
            written.append(await shot(out / f"{name}_{_slug(label) or i}.png"))
        return written

"""Shared headless-browser helper (D110, D116): Chrome first, then Edge, `TT_BROWSER`
overrides. Used by the browser tests (`@pytest.mark.browser`), `report.shots` and the demo
check and rehearsal (`traffic_twin.demo`).

- `find_browser()`: `TT_BROWSER` (a path or a name on the PATH), then the usual Chrome and
  Edge install paths, then `google-chrome` / `chromium` / `msedge` on the PATH. Chrome is
  preferred: headless Edge `--dump-dom` printed nothing on the development machine.
- `dump_dom(url)`: the DOM after the page's scripts ran for `wait_ms` of virtual time.
- `screenshot(url, out)`: a PNG of the page.
- Both take `cookie=(name, value)`: the page is then opened over the DevTools protocol with
  that cookie set first (the session of the user app's sign-in, `api.admin`), and read once
  the mock view has left `loading` (`MOCK_READY`) or after `wait_ms` x 3. This mode needs
  `websockets` (installed with Streamlit, the `app` extra).
- `fields(dom)` and `view_states(dom)`: the `data-field` hooks and `data-view-state` roots
  of the mock views (plan §4.4), parsed with the standard library.

Every call uses a fresh temporary profile and runs with that profile as its working
directory, so nothing lands in the repo. A stray `last_dom.html` that appears in the repo
root during a call (the desktop browser pane writes one) is moved to
`data/demo_check/stray/`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

ENV_BROWSER = "TT_BROWSER"
CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)
NAMES = (
    "google-chrome",
    "google-chrome-stable",
    "chrome",
    "chromium",
    "chromium-browser",
    "msedge",
)
PROFILE_PREFIX = "tt_demo_browser_"  # `demo clean` removes leftovers with this prefix
STRAY_FILE = "last_dom.html"
TIMEOUT_S = 120.0
# the mock view is ready: a view root that left `loading` and no request in flight
MOCK_READY = (
    "(() => { const v = document.getElementById('view'); const b = document.getElementById('busy');"
    " return !!v && !!v.dataset.view && v.dataset.viewState !== 'loading'"
    " && !(b && b.classList.contains('on')); })()"
)
SETTLE_S = 0.8  # once ready, the page keeps this long to finish its last paint
VOID_TAGS = frozenset(
    "area base br col embed hr img input link meta param source track wbr".split()
)
INLINE_TAGS = frozenset("a abbr b code em i kbd label mark q s small span strong sub sup u".split())


class BrowserError(RuntimeError):
    """No headless browser, or the browser failed (timeout, empty output)."""


def find_browser() -> Path | None:
    """The headless browser executable (`TT_BROWSER`, Chrome, then Edge), or None."""
    env = os.environ.get(ENV_BROWSER, "").strip()
    if env:
        if Path(env).is_file():
            return Path(env)
        found = shutil.which(env)
        return Path(found) if found else None
    for cand in CANDIDATES:
        if Path(cand).is_file():
            return Path(cand)
    found = next((p for p in map(shutil.which, NAMES) if p), None)
    return Path(found) if found else None


def _repo_root() -> Path:
    from traffic_twin.api.settings import find_repo_root

    return find_repo_root()


def _move_stray(since: float) -> Path | None:
    """Move a `last_dom.html` written into the repo root after `since` out of the way."""
    root = _repo_root()
    stray = root / STRAY_FILE
    try:
        if not stray.is_file() or stray.stat().st_mtime < since - 1.0:
            return None
        dest = root / "data" / "demo_check" / "stray"
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / f"last_dom_{time.strftime('%Y%m%d-%H%M%S')}.html"
        shutil.move(str(stray), str(target))
        return target
    except OSError:
        return None


def _rmtree(path: str, tries: int = 4) -> None:
    """Remove a profile folder; Chrome's helper processes may hold it a moment after exit
    (a leftover is removed later by `demo clean`)."""
    for i in range(tries):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        time.sleep(0.25 * (i + 1))


def _args(profile: str, wait_ms: int) -> list[str]:
    args = [
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-sync",
        "--mute-audio",
        "--hide-scrollbars",
        f"--user-data-dir={profile}",
        f"--virtual-time-budget={int(wait_ms)}",
    ]
    if sys.platform.startswith("linux"):
        args.append("--no-sandbox")  # GitHub ubuntu runners
    return args


def _run(cmd_tail: list[str], wait_ms: int, browser: Path | None, timeout_s: float):
    exe = browser or find_browser()
    if exe is None:
        raise BrowserError("no headless Chrome or Edge found (set TT_BROWSER)")
    since = time.time()
    profile = tempfile.mkdtemp(prefix=PROFILE_PREFIX)
    try:
        cmd = [str(exe), *_args(profile, wait_ms), *cmd_tail]
        try:
            cp = subprocess.run(cmd, capture_output=True, timeout=timeout_s, cwd=profile)
        except subprocess.TimeoutExpired as e:
            raise BrowserError(f"{Path(exe).name} did not finish in {timeout_s:.0f} s") from e
        except OSError as e:
            raise BrowserError(f"cannot start {exe}: {e}") from e
    finally:
        _rmtree(profile)
        _move_stray(since)
    return exe, cp


def dump_dom(
    url: str,
    *,
    wait_ms: int = 4000,
    browser: Path | None = None,
    timeout_s: float = TIMEOUT_S,
    cookie: tuple[str, str] | None = None,
) -> str:
    """The DOM of `url` after `wait_ms` of virtual time (`--dump-dom`); with `cookie`, over
    DevTools with that cookie set (see the module doc). BrowserError when no browser is
    found or it prints nothing."""
    if cookie is not None:
        return _cdp_page(url, cookie, wait_ms=wait_ms, browser=browser, timeout_s=timeout_s)
    exe, cp = _run(["--dump-dom", url], wait_ms, browser, timeout_s)
    dom = cp.stdout.decode("utf-8", "replace")
    if not dom.strip():
        err = cp.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
        raise BrowserError(
            f"{Path(exe).name} --dump-dom printed nothing (exit {cp.returncode}) {err[0]}".strip()
        )
    return dom


def screenshot(
    url: str,
    out: Path,
    *,
    size: tuple[int, int] = (1280, 800),
    wait_ms: int = 4000,
    browser: Path | None = None,
    timeout_s: float = TIMEOUT_S,
    cookie: tuple[str, str] | None = None,
) -> Path:
    """A PNG screenshot of `url` written to `out` (parents created); with `cookie`, over
    DevTools with that cookie set. BrowserError when no browser is found or no image was
    written."""
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    w, h = size
    if cookie is not None:
        _cdp_page(url, cookie, wait_ms=wait_ms, browser=browser, timeout_s=timeout_s,
                  png=out, size=size)  # fmt: skip
        if not out.is_file() or out.stat().st_size == 0:
            raise BrowserError("DevTools wrote no screenshot")
        return out
    exe, cp = _run(
        [f"--window-size={int(w)},{int(h)}", f"--screenshot={out}", url],
        wait_ms,
        browser,
        timeout_s,
    )
    if not out.is_file() or out.stat().st_size == 0:
        raise BrowserError(f"{Path(exe).name} wrote no screenshot (exit {cp.returncode})")
    return out


# -- DevTools mode (a session cookie set before the page loads) --------------------------
def _cdp_page(
    url: str,
    cookie: tuple[str, str],
    *,
    wait_ms: int,
    browser: Path | None,
    timeout_s: float,
    png: Path | None = None,
    size: tuple[int, int] = (1280, 800),
) -> str:
    """The DOM of `url` (and a PNG into `png`) opened with `cookie` set."""
    try:
        import websockets  # noqa: F401
    except ImportError as e:
        raise BrowserError("the DevTools mode needs websockets (pip install -e .[app])") from e
    exe = browser or find_browser()
    if exe is None:
        raise BrowserError("no headless Chrome or Edge found (set TT_BROWSER)")
    since = time.time()
    profile = tempfile.mkdtemp(prefix=PROFILE_PREFIX)
    args = [a for a in _args(profile, wait_ms) if not a.startswith("--virtual-time-budget")]
    w, h = size
    cmd = [str(exe), *args, f"--window-size={int(w)},{int(h)}", "--remote-debugging-port=0",
           "about:blank"]  # fmt: skip
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=profile
        )
    except OSError as e:
        _rmtree(profile)
        raise BrowserError(f"cannot start {exe}: {e}") from e
    try:
        port_file = Path(profile) / "DevToolsActivePort"
        deadline = time.monotonic() + 30
        while not (port_file.is_file() and port_file.read_text().split("\n")[0].isdigit()):
            if time.monotonic() > deadline or proc.poll() is not None:
                raise BrowserError(f"{Path(exe).name}: no DevTools port")
            time.sleep(0.1)
        port = int(port_file.read_text().split("\n")[0])
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as r:
            pages = [t for t in json.loads(r.read()) if t.get("type") == "page"]
        if not pages:
            raise BrowserError(f"{Path(exe).name}: no page target")
        limit = max(20.0, 3 * wait_ms / 1000)
        coro = _cdp_session(pages[0]["webSocketDebuggerUrl"], url, cookie, limit, png)
        try:
            return asyncio.run(asyncio.wait_for(coro, timeout_s))
        except (TimeoutError, OSError) as e:
            raise BrowserError(f"{Path(exe).name} (DevTools): {type(e).__name__} {e}") from e
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        _rmtree(profile)
        _move_stray(since)


async def _cdp_session(
    ws_url: str, url: str, cookie: tuple[str, str], limit_s: float, png: Path | None
) -> str:
    import websockets

    async with websockets.connect(ws_url, max_size=None) as ws:
        seq = 0

        async def call(method: str, **params):
            nonlocal seq
            seq += 1
            await ws.send(json.dumps({"id": seq, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == seq:
                    if "error" in msg:
                        raise BrowserError(f"DevTools {method}: {msg['error']}")
                    return msg.get("result") or {}

        async def js(expr: str):
            res = await call("Runtime.evaluate", expression=expr, returnByValue=True)
            return (res.get("result") or {}).get("value")

        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        await call("Network.enable")
        await call("Network.setCookie", name=cookie[0], value=cookie[1], url=origin,
                   path="/", httpOnly=True, sameSite="Strict")  # fmt: skip
        await call("Page.navigate", url=url)
        deadline = time.monotonic() + limit_s
        await asyncio.sleep(0.3)
        while time.monotonic() < deadline:
            try:
                if await js(MOCK_READY):
                    break
            except BrowserError:
                pass  # the document is being replaced: ask again
            await asyncio.sleep(0.2)
        await asyncio.sleep(SETTLE_S)
        if png is not None:
            shot = await call("Page.captureScreenshot", format="png")
            png.write_bytes(base64.b64decode(shot.get("data") or ""))
        return await js("document.documentElement.outerHTML") or ""


# -- DOM parsing (stdlib only) -----------------------------------------------------------
class _Collector:
    """The text of one open element: a `data-field` hook or a `data-view-state` root."""

    __slots__ = ("kind", "key", "attrs", "parts")

    def __init__(self, kind: str, key: str, attrs: dict[str, str]) -> None:
        self.kind, self.key, self.attrs, self.parts = kind, key, attrs, []

    @property
    def text(self) -> str:
        return " ".join("".join(self.parts).split())


class _HookParser(HTMLParser):
    """Collects the text of every `data-field` element and every `data-view-state` root."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, list[_Collector]]] = []  # (tag, collectors opened there)
        self.done: list[_Collector] = []  # in document (start-tag) order
        self.skip = 0  # inside <script> / <style>

    def _space(self, tag: str) -> None:
        if tag not in INLINE_TAGS:  # block boundaries separate words
            for _tag, opened in self.stack:
                for c in opened:
                    c.parts.append(" ")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag in ("script", "style"):
            self.skip += 1
        self._space(tag)
        opened = []
        if a.get("data-field"):
            c = _Collector("field", a["data-field"], a)
            if tag in ("input", "textarea", "select") and a.get("value"):
                c.parts.append(a["value"])
            opened.append(c)
        if "data-view-state" in a:
            opened.append(_Collector("state", a["data-view-state"], a))
        self.done.extend(opened)
        if tag not in VOID_TAGS:
            self.stack.append((tag, opened))

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        self._space(tag)
        # close up to the matching open tag (tolerates unclosed children)
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        for _tag, opened in self.stack:
            for c in opened:
                c.parts.append(data)


def _parse(dom: str) -> _HookParser:
    p = _HookParser()
    p.feed(dom)
    p.close()
    return p


def fields(dom: str) -> dict[str, str]:
    """`data-field` hook -> its text (whitespace collapsed; an input's `value`); the first
    occurrence wins when a hook appears more than once."""
    out: dict[str, str] = {}
    for c in _parse(dom).done:
        if c.kind == "field":
            out.setdefault(c.key, c.text)
    return out


def view_states(dom: str) -> list[dict]:
    """Every element carrying `data-view-state`: {state, attrs, text}, in document order."""
    return [
        {"state": c.key, "attrs": c.attrs, "text": c.text}
        for c in _parse(dom).done
        if c.kind == "state"
    ]


def text(dom: str) -> str:
    """The visible text of `dom` (scripts and styles dropped, whitespace collapsed)."""

    class _Text(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []
            self.skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self.skip += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self.skip = max(0, self.skip - 1)

        def handle_data(self, data):
            if not self.skip:
                self.parts.append(data)

    p = _Text()
    p.feed(dom)
    p.close()
    return " ".join(" ".join(p.parts).split())

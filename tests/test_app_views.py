"""Rendered user app views (Part 9, A-08; D104-D106, plan 4.4): headless Chrome (or Edge)
opens each deep link of `/mock` served by uvicorn, and the dumped DOM must show the API's
answer. Each `data-field` hook carries the raw API value in `data-value`, compared with the
same request made in process. Runs on FAKE fixture data (always) and on the real demo run
`pitie_rules` seed 4000 when `data/demo` has it; an API that answers nothing gives the
`error` state, never a blank view.

`@pytest.mark.browser`: skipped when no browser is found (`TT_BROWSER` overrides).
"""

from __future__ import annotations

import html
import re
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from importlib import resources
from pathlib import Path

import pytest
import uvicorn
from conftest import browser_path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.testclient import TestClient

from traffic_twin.api.fixtures import make_fake_eta, make_results, write_settings
from traffic_twin.api.main import create_app
from traffic_twin.api.settings import load_settings

pytestmark = pytest.mark.browser

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_RUN = REPO_ROOT / "data" / "demo" / "pitie_rules_seed4000"
VIRTUAL_TIME_MS = 8000
OK_STATES = {"ok", "degraded"}


# -- servers ------------------------------------------------------------------------------
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Served:
    """An ASGI app on a free localhost port, in a daemon thread."""

    def __init__(self, app) -> None:
        self.port = free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 20
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.05)
        self.url = f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(10)


class Site:
    """A served API plus an in-process client on the same settings (the expected answers)."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.api = TestClient(create_app(settings))
        self.served = Served(create_app(settings))

    def get(self, path: str) -> dict:
        return self.api.get(path).json()


@pytest.fixture(scope="module")
def fake_site(tmp_path_factory) -> Iterator[Site]:
    """FAKE data (`api.fixtures`): 3 runs of 1800 s at seed 4000, FAKE ETA files."""
    root = tmp_path_factory.mktemp("views") / "fake"
    make_results(root, n_runs=3)
    runs = sorted(p.parent for p in root.glob("*/run.json"))
    for d in runs:
        make_fake_eta(d)
    make_fake_eta(root / "models" / "eta")
    cfg = write_settings(root, runs, root / "models" / "eta")
    site = Site(load_settings(path=cfg, repo_root=REPO_ROOT, environ={}))
    yield site
    site.served.stop()


@pytest.fixture(scope="module")
def real_site() -> Iterator[Site]:
    """The repo settings (data/results + data/demo), default run pitie_rules seed 4000."""
    if not (DEMO_RUN / "run.json").is_file():
        pytest.skip("no demo run data/demo/pitie_rules_seed4000 (tasks.py demo-runs)")
    settings = replace(load_settings(repo_root=REPO_ROOT, environ={}), mock=True)
    site = Site(settings)
    yield site
    site.served.stop()


@pytest.fixture(scope="module")
def down_site() -> Iterator[str]:
    """Serves the page and three.js, and answers every API path with a non-JSON 503."""
    app = FastAPI()
    page = resources.files("traffic_twin.app").joinpath("mock.html").read_text(encoding="utf-8")
    js = resources.files("traffic_twin.app").joinpath("three.min.js").read_bytes()
    app.add_api_route("/mock", lambda: HTMLResponse(page), methods=["GET"])
    app.add_api_route(
        "/three.min.js",
        lambda: Response(js, media_type="application/javascript"),
        methods=["GET"],
    )

    @app.api_route("/{path:path}", methods=["GET"])
    def down(path: str):
        return PlainTextResponse("service unavailable", status_code=503)

    served = Served(app)
    yield served.url
    served.stop()


# -- the browser --------------------------------------------------------------------------
class Dom:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.text_only = re.sub(r"<script.*?</script>|<style.*?</style>", "", raw, flags=re.S)
        m = re.search(r"<main id=\"view\"([^>]*)>(.*?)</main>", self.text_only, flags=re.S)
        assert m, "no <main id=view> in the dumped DOM"
        self.main_attrs = m.group(1)
        self.main_html = m.group(2)
        self.main_text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", m.group(2))).split())
        f = re.search(r"<footer>(.*?)</footer>", self.text_only, flags=re.S)
        self.footer = " ".join(
            html.unescape(re.sub(r"<[^>]+>", " ", f.group(1) if f else "")).split()
        )

    def attr(self, name: str) -> str | None:
        m = re.search(rf'{name}="([^"]*)"', self.main_attrs)
        return html.unescape(m.group(1)) if m else None

    @property
    def view(self) -> str | None:
        return self.attr("data-view")

    @property
    def state(self) -> str | None:
        return self.attr("data-view-state")

    def hook(self, field: str) -> str | None:
        m = re.search(rf'data-field="{re.escape(field)}"[^>]*?data-value="([^"]*)"', self.text_only)
        return html.unescape(m.group(1)) if m else None

    def has(self, field: str) -> bool:
        return f'data-field="{field}"' in self.text_only


def dump(url: str) -> Dom:
    exe = browser_path()
    with tempfile.TemporaryDirectory(prefix="tt_views_") as prof:
        cmd = [
            exe, "--headless=new", "--disable-gpu", "--no-first-run",
            "--no-default-browser-check", "--disable-extensions", f"--user-data-dir={prof}",
            f"--virtual-time-budget={VIRTUAL_TIME_MS}", "--dump-dom", url,
        ]  # fmt: skip
        out = subprocess.run(cmd, capture_output=True, timeout=90, cwd=prof)
    raw = out.stdout.decode("utf-8", "replace")
    assert raw.strip(), f"empty DOM from the browser: {out.stderr[-300:]!r}"
    return Dom(raw)


# -- expected values ----------------------------------------------------------------------
def same(shown: str | None, expected) -> bool:
    """A data-value (JS String()) against an API value."""
    if shown is None:
        return False
    if expected is None:
        return shown == ""
    if isinstance(expected, bool):
        return shown == str(expected).lower()
    if isinstance(expected, int | float):
        return shown != "" and abs(float(shown) - float(expected)) < 1e-6
    return shown == str(expected)


def run_id(site: Site, scenario: str, seed: int) -> str:
    for grp in site.get("/scenarios")["scenarios"]:
        if grp["scenario"] == scenario:
            return next(r["run_id"] for r in grp["runs"] if r["seed"] == seed)
    raise LookupError(scenario)


def expected_hooks(site: Site, view: str, run: str, params: dict) -> dict:
    t = params.get("t")
    q = f"t={t}&run_id={run}"
    if view == "pedestrian":
        b = site.get(f"/crossing/{params['crossing']}?{q}&speed={params['speed']}")
        green = b["signal"] == "green"
        verdict = "unknown" if b["can_cross"] is None else str(b["can_cross"]).lower()
        remaining = b["green_remaining_s"] if green else b["next_green_in_s"]
        return {"ped-verdict": verdict, "ped-remaining": remaining, "ped-walk": b["walk_time_s"]}
    if view == "student":
        b = site.get(f"/eta?{q}&trajet={params['trajet']}&student=true&deadline={params['exam']}")
        rest = b["rest_of_trip_s"]
        med = None if b["eta_median_s"] is None else b["eta_median_s"] + rest
        interval = (
            f"{_js(b['eta_low_s'] + rest)},{_js(b['eta_high_s'] + rest)}" if med is not None else ""
        )
        return {
            "stu-eta": med,
            "stu-interval": interval,
            "stu-advice": (b["advice"] or {}).get("code", ""),
        }
    if view == "bus":
        b = site.get(f"/bus/{params['line']}/status?{q}")
        prio = "unavailable"
        if b["priority_available"]:
            prio = "unknown" if b["priority_active"] is None else str(b["priority_active"]).lower()
        return {
            "bus-delay": b["delay_s"],
            "bus-headway": b["headway_s"],
            "bus-priority": prio,
            "bus-message": b["message"],
        }
    if view == "ambulance":
        b = site.get(f"/ambulance/next-intersections?{q}")
        hooks = {"amb-eta": b["eta_destination_s"]}
        for k, jn in enumerate(b["intersections"][:3], start=1):
            hooks[f"amb-j{k}-status"] = jn["status"] if jn["modelled"] else "not_modelled"
        return hooks
    if view == "replay":
        b = site.get(f"/replay/{run}?t={t}&positions=true")
        c = b["counts"]
        return {
            "replay-t": b["sim_t_s"],
            "replay-count": sum(c.get(k, 0) for k in ("car", "bus", "ambulance")),
        }
    if view == "operator":
        ev = site.get("/evaluation?family=main")
        labels = [d["label"] for d in ev.get("labels") or []]
        roles = {d["label"]: d.get("role") for d in ev.get("labels") or []}
        if not labels:
            groups = site.get("/scenarios")["scenarios"]
            main = [g for g in groups if g["family"] == "main" and g["label"]]
            labels = sorted({g["label"] for g in main})
            roles = {g["label"]: g.get("role") for g in main}
        best = None
        for label in labels:
            rows = site.get(f"/kpi?scenario={label}&user=all&family=main").get("rows") or []
            row = next((r for r in rows if r["kpi"] == "person_delay_mean"), None)
            if (
                row is None
                or row["mean"] is None
                or roles.get(label) in ("ablation", "exploratory")
            ):
                continue
            if best is None or row["mean"] < best[1]:
                best = (label, row["mean"])
        return {"op-best": best[0] if best else ""}
    return {}


def _js(x: float) -> str:
    """JS String() of a number (integers without the trailing .0)."""
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def query(view: str, params: dict) -> str:
    items = {"view": view, **params}
    return "?" + "&".join(f"{k}={v}" for k, v in items.items())


def check_view(site: Site, view: str, params: dict, *, scenario: str, seed: int) -> Dom:
    run = run_id(site, scenario, seed)
    dom = dump(
        f"{site.served.url}/mock{query(view, {'scenario': scenario, 'seed': seed, **params})}"
    )
    assert dom.view == view, (dom.view, dom.main_text[:300])
    assert dom.state in OK_STATES, (dom.state, dom.main_text[:300])
    assert "Chargement" not in dom.main_text
    assert "Données simulées" in dom.footer and "calibrées sur capteurs réels" in dom.footer
    calibrated = site.get("/health")["calibrated_on"]
    assert calibrated and calibrated in dom.footer
    assert dom.has("footer-source") and dom.has("footer-calibrated")
    for field, expected in expected_hooks(site, view, run, params).items():
        assert same(dom.hook(field), expected), (field, dom.hook(field), expected)
    if view == "operator":
        assert dom.state == expected_operator_state(site), (dom.state, dom.main_text[:300])
    return dom


def expected_operator_state(site: Site) -> str:
    """The card is degraded when /evaluation is not ok or a registered label has no /kpi
    answer, while some row is shown (plan 4.4 root state)."""
    ev = site.get("/evaluation?family=main")
    labels = [d["label"] for d in ev.get("labels") or []]
    kpis = [site.get(f"/kpi?scenario={label}&user=all&family=main") for label in labels]
    if any(k.get("status") == "degraded" for k in kpis):
        return "degraded"
    if ev.get("status") in ("degraded", "unavailable"):
        return "degraded"
    if any(k.get("error") or k.get("status") == "unavailable" for k in kpis):
        return "degraded"
    return "ok"


# -- FAKE fixtures --------------------------------------------------------------------------
FAKE_CASES = {
    "pedestrian": ("pedestrian", {"crossing": "S", "speed": "normal", "t": 600}),
    "pedestrian_reduced": ("pedestrian", {"crossing": "W", "speed": "reduced", "t": 905}),
    "student": ("student", {"trajet": "N_S", "exam": "07:50", "t": 900}),
    "bus": ("bus", {"line": "91", "t": 1305}),
    "ambulance": ("ambulance", {"t": 340, "kiosk": 1}),
    "replay": ("replay", {"t": 600}),
    "operator": ("operator", {}),
}


@pytest.mark.parametrize("case", sorted(FAKE_CASES))
def test_views_show_the_api_answer_on_fixtures(fake_site, case):
    view, params = FAKE_CASES[case]
    check_view(fake_site, view, params, scenario="pitie_rules", seed=4000)


def test_home_lists_the_five_profiles_and_replay(fake_site):
    dom = dump(f"{fake_site.served.url}/mock")
    assert dom.view == "home" and dom.state in OK_STATES
    for key in ("pedestrian", "student", "bus", "ambulance", "operator", "replay"):
        assert dom.has(f"home-card-{key}"), key
    assert "Données simulées" in dom.footer


def test_kiosk_hides_the_navigation(fake_site):
    dom = dump(f"{fake_site.served.url}/mock?view=ambulance&t=340&kiosk=1")
    assert re.search(r'<body[^>]*class="[^"]*kiosk', dom.raw)


@pytest.mark.parametrize(
    "bad", ["view=nowhere", "crossing=Z", "seed=4999&scenario=pitie_rules", "t=99999999"]
)
def test_bad_deep_link_opens_home_with_an_error(fake_site, bad):
    qs = bad if bad.startswith("view=") else f"view=pedestrian&{bad}"
    dom = dump(f"{fake_site.served.url}/mock?{qs}")
    assert dom.view == "home"
    assert dom.has("deeplink-error"), dom.main_text[:300]


def test_api_down_gives_the_error_state_never_a_blank_view(down_site):
    for view in ("home", "pedestrian", "operator"):
        dom = dump(f"{down_site}/mock?view={view}")
        assert dom.state == "error", (view, dom.state)
        assert dom.has("banner-error") and len(dom.main_text) > 20, dom.main_text
        assert "Données simulées" in dom.footer


# -- the real demo run (pitie_rules seed 4000; plan 4.5 / D111 instants) ----------------------
REAL_CASES = {
    "pedestrian": ("pedestrian", {"crossing": "S", "speed": "normal", "t": 1805}),
    "pedestrian_reduced": ("pedestrian", {"crossing": "S", "speed": "reduced", "t": 1805}),
    "student": ("student", {"trajet": "N_S", "exam": "08:18", "t": 2700}),
    "bus": ("bus", {"line": "91", "t": 1305}),
    "ambulance": ("ambulance", {"t": 2440, "kiosk": 1}),
    "replay": ("replay", {"t": 2440}),
    "operator": ("operator", {}),
}


@pytest.mark.parametrize("case", sorted(REAL_CASES))
def test_views_show_the_api_answer_on_the_demo_run(real_site, case):
    view, params = REAL_CASES[case]
    check_view(real_site, view, params, scenario="pitie_rules", seed=4000)


# -- the replay clock and the operator card under partial failures (12.1) --------------------
def patched(site: Site, override) -> Served:
    """The site's API with some answers replaced: `override(request)` returns a response or
    None (the API answers)."""
    app = create_app(site.settings)

    @app.middleware("http")
    async def _override(request: Request, call_next):
        r = override(request)
        return r if r is not None else await call_next(request)

    return Served(app)


def test_replay_frames_failing_never_show_the_old_map_as_current(fake_site):
    """The opening instant answers, every other /replay frame fails (API down mid-replay):
    the view turns `error` with the error banner, the vehicle count is cleared and the
    footer says the API is unreachable, instead of keeping the last fleet as `ok`."""
    opening = 602  # between two frames: the map needs the frames 600 and 605

    def frames_down(request: Request):
        path = request.url.path
        if path.startswith("/replay/") and not path.endswith("/events"):
            if request.query_params.get("t") != str(opening):
                return PlainTextResponse("service unavailable", status_code=503)
        return None

    served = patched(fake_site, frames_down)
    try:
        dom = dump(f"{served.url}/mock?view=replay&scenario=pitie_rules&seed=4000&t={opening}")
    finally:
        served.stop()
    assert dom.view == "replay"
    assert dom.state == "error", (dom.state, dom.main_text[:300])
    assert dom.has("banner-error"), dom.main_text[:300]
    assert dom.hook("replay-count") == "" and dom.hook("replay-t") == ""
    assert "API injoignable" in dom.footer, dom.footer


def test_replay_playback_moves_the_footer_with_the_instant(fake_site):
    """While the replay plays, the footer timestamp is the one of the frame on the map."""
    run = run_id(fake_site, "pitie_rules", 4000)
    dom = dump(
        f"{fake_site.served.url}/mock?view=replay&scenario=pitie_rules&seed=4000&t=600&play=1"
    )
    assert dom.state in OK_STATES, (dom.state, dom.main_text[:300])
    shown = dom.hook("replay-t")
    assert shown and float(shown) > 600, shown  # playback moved on
    b = fake_site.get(f"/replay/{run}?t={int(float(shown))}&positions=true")
    assert f"Horodatage {b['timestamp'][11:19]}" in dom.footer, (shown, dom.footer)


def test_operator_card_is_degraded_when_a_registered_label_is_missing(fake_site):
    """S3 registered but not evaluated: /evaluation degraded and /kpi unavailable for S3."""
    notice = "S3 : pas encore évalué"

    def s3_missing(request: Request):
        path, qp = request.url.path, request.query_params
        if path == "/kpi" and qp.get("scenario") == "S3":
            b = fake_site.get(f"/kpi?{request.url.query}")
            b.update(status="unavailable", notice=notice, rows=[])
            return JSONResponse(b)
        if path == "/evaluation":
            b = fake_site.get(f"/evaluation?{request.url.query}")
            b.update(status="degraded", notice=notice)
            return JSONResponse(b)
        return None

    served = patched(fake_site, s3_missing)
    try:
        dom = dump(f"{served.url}/mock?view=operator")
    finally:
        served.stop()
    assert dom.view == "operator"
    assert dom.state == "degraded", (dom.state, dom.main_text[:300])
    assert dom.has("banner-degraded") and notice in dom.main_text, dom.main_text[:400]

"""Static contract of the user app `app/mock.html` (Part 9, D104-D107; plan 4.4), without a
browser: what the page fetches and reads must exist in API v1, every view handles the 12.1
states and the ENF-05 footer, and nothing on the page is an invented result (the D107
honesty ban list). The rendered views are checked against the API in test_app_views.py.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import replace
from importlib import resources
from pathlib import Path

import pydantic
import pytest
from fastapi.testclient import TestClient

from traffic_twin.api import schemas
from traffic_twin.api.fixtures import make_results
from traffic_twin.api.main import create_app
from traffic_twin.api.settings import load_settings
from traffic_twin.api.site import load_site

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE = resources.files("traffic_twin.app").joinpath("mock.html").read_text(encoding="utf-8")
SCRIPTS = re.findall(r"<script>(.*?)</script>", PAGE, flags=re.S)
SCRIPT = max(SCRIPTS, key=len)  # the app (the other inline script is the three.js loader)
STYLE = "\n".join(re.findall(r"<style>(.*?)</style>", PAGE, flags=re.S))
BODY = re.sub(r"<script.*?</script>|<style>.*?</style>", "", PAGE, flags=re.S)
TEXT = BODY + SCRIPT  # everything but the CSS

VIEW_FUNCTIONS = (
    "home",
    "pedestrian",
    "student",
    "bus",
    "ambulance",
    "replay",
    "operator",
    "admin",
    "login",
)
# plan 4.4 test hooks (data-field); every hook carries the raw API value in data-value
HOOKS = (
    "home-card-pedestrian", "home-card-student", "home-card-bus", "home-card-ambulance",
    "home-card-operator", "home-card-replay",
    "ped-verdict", "ped-remaining", "ped-walk",
    "stu-eta", "stu-interval", "stu-advice",
    "bus-delay", "bus-headway", "bus-priority", "bus-message",
    "amb-eta", "replay-t", "replay-count", "op-best",
    "footer-source", "footer-calibrated",
    "home-admin", "admin-state", "admin-session", "admin-tab", "admin-table", "admin-moments",
    "admin-timeline", "admin-events", "admin-maps", "admin-radar", "admin-raw",
    "login-state", "login-role", "auth-role",
)  # fmt: skip
VIEW_STATES = ("loading", "ok", "degraded", "stale", "unavailable", "error")
DEEP_LINK_PARAMS = (
    "view", "scenario", "seed", "t", "crossing", "speed", "line", "trajet", "exam", "play",
    "kiosk", "operator", "run",
)  # fmt: skip
# D107 ban list: fabricated results, live-feed or certification claims, client-side fakes
BANNED = (
    "S3_DATA", "s3_rl_dqn", "s3_", "next_arrival", "countdown_text", "unit_text", "00:28",
    "Garantie", "certifié", "en direct", "V2I", "Télématique", "Capteurs réels 30",
    "SUMO 60 FPS", "Math.random", "getPhase", "Simuler", "cyclist", "Q-Value", "XGBoost CQR v1",
)  # fmt: skip
# API objects as the page names them -> the schema that defines their fields
PREFIX_MODELS = {
    "jn": schemas.Intersection,
    "kr": schemas.KpiRow,
    "row": schemas.KpiRow,
    "grp": schemas.ScenarioGroup,
    "sr": schemas.ScenarioRun,
    "apiPos": schemas.Position,
}
JS_MEMBERS = {"length", "map", "filter", "find", "some", "slice", "join", "forEach", "toFixed"}


def envelope_fields() -> set[str]:
    """Every field of every response model (the page names any response `b`)."""
    fields = {"http"}  # added by the page's api() helper
    for name in dir(schemas):
        obj = getattr(schemas, name)
        if isinstance(obj, type) and issubclass(obj, pydantic.BaseModel):
            fields |= set(obj.model_fields)
    return fields


def routes() -> list[str]:
    return [getattr(r, "path", "") for r in create_app(load_settings(repo_root=REPO_ROOT)).routes]


def function_body(name: str) -> str:
    m = re.search(rf"^(?:async )?function {name}\(.*?^\}}$", SCRIPT, flags=re.S | re.M)
    assert m, f"function {name} not found"
    return m.group(0)


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    """The API over 3 FAKE runs (`api.fixtures`), mock screens on."""
    root = make_results(tmp_path_factory.mktemp("mock") / "results", n_runs=3)
    settings = replace(
        load_settings(repo_root=REPO_ROOT, environ={}),
        results_dirs=(root,),
        evaluation_dir=root / "evaluation",
        default_run=(root / "pitie_rules_seed4000",),
        mock=True,
    )
    return TestClient(create_app(settings))


def test_the_page_is_served_with_three_js(client):
    r = client.get("/mock")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert r.text == PAGE
    assert client.get("/three.min.js").status_code == 200
    root = client.get("/", follow_redirects=False)
    assert root.status_code in (302, 307) and root.headers["location"] == "/mock"


def test_every_fetched_path_is_an_api_route():
    calls = re.findall(r"\bapi\(\s*([`\"'])(.*?)\1", SCRIPT)
    assert len(calls) >= 10
    patterns = [re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", p) + "$") for p in routes()]
    for _, raw in calls:
        path = re.sub(r"\$\{[^}]*\}", "X", raw.split("?", 1)[0])
        assert path.startswith("/"), raw
        assert any(p.match(path) for p in patterns), f"{raw!r} is not an API route"
    # only api() talks to the network: no other fetch, no direct file reads (D102)
    assert SCRIPT.count("fetch(") == 1
    assert "XMLHttpRequest" not in SCRIPT and "EventSource" not in SCRIPT
    for literal in re.findall(r"[`\"']([^`\"'\n]*)[`\"']", SCRIPT):
        assert not re.search(r"(^|/)(docs|data)/", literal), literal


def api_code() -> str:
    """The script without Motez's canvas drawing and 2D/3D toggles, which never read the
    API (their local `b` is a road branch or a button)."""
    code = SCRIPT
    for name in ("drawSim", "switchTo2D", "switchTo3D"):
        code = code.replace(function_body(name), "")
    return code


def test_response_fields_read_exist_in_the_schemas():
    fields = envelope_fields()
    code = api_code()
    read = set(re.findall(r"\bb\.([a-z_][a-z0-9_]*)", code))
    assert {"can_cross", "eta_median_s", "delay_s", "intersections", "positions"} <= read
    # the other names of whole responses: /evaluation (ev), /health (h), init replay (rb)
    for name in ("ev", "h", "rb"):
        read |= set(re.findall(rf"\b{name}\.([a-z_][a-z0-9_]*)", code)) - JS_MEMBERS
    assert read - fields == set(), "read from a response but in no schema"
    nested = {
        "advice": schemas.EtaAdvice,
        "run": schemas.RunRef,
        "error": schemas.ErrorInfo,
    }
    for key, model in nested.items():
        got = set(re.findall(rf"\bb\.{key}\.([a-z_][a-z0-9_]*)", SCRIPT))
        assert got <= set(model.model_fields), (key, got - set(model.model_fields))
    for prefix, model in PREFIX_MODELS.items():
        got = set(re.findall(rf"\b{prefix}\.([a-z_][a-z0-9_]*)", SCRIPT)) - JS_MEMBERS
        assert got, f"no {prefix}.<field> read"
        assert got <= set(model.model_fields), (prefix, got - set(model.model_fields))
    # events are dicts: their keys are the ones documented by ReplayEventsResponse
    doc = schemas.ReplayEventsResponse.model_fields["events"].description
    keys = set(re.findall(r"[a-z_]+", doc))
    got = set(re.findall(r"\bevt\.([a-z_][a-z0-9_]*)", SCRIPT))
    assert got and got <= keys, got - keys
    signals = schemas.ReplayStateResponse.model_fields["signals"].description
    for approach in re.findall(r'sigOf\("(\w+)"\)', SCRIPT):
        assert f'"{approach}"' in signals


def test_every_view_handles_the_states_and_banners():
    render = re.search(r"const RENDER = \{([^}]*)\}", SCRIPT)
    assert render and {s.strip() for s in render.group(1).split(",")} == set(VIEW_FUNCTIONS)
    for name in VIEW_FUNCTIONS:
        body = function_body(name)
        assert "banners(" in body, f"view {name} does not show the 12.1 banners"
        assert re.search(r"\bapi\(", body), f"view {name} does not read the API"
    for st in VIEW_STATES:
        assert f'"{st}"' in SCRIPT, st
    assert "data-view-state" in BODY and "dataset.viewState" in SCRIPT
    assert "AbortController" in SCRIPT and re.search(r"FETCH_TIMEOUT_MS = 4000\b", SCRIPT)
    assert "LAST_GOOD" in SCRIPT and "dernière valeur connue" in SCRIPT  # 12.1 last value
    assert "Chargement" in SCRIPT and 'id="busy"' in BODY  # loading indicator


@pytest.mark.parametrize("hook", HOOKS)
def test_view_hooks(hook):
    if hook.startswith("home-card-") and hook != "home-card-pedestrian":
        # the other profile cards come from one template: card("<view>", ...)
        assert 'data-field="home-card-${key}"' in SCRIPT
        assert f'card("{hook.removeprefix("home-card-")}",' in SCRIPT
        return
    assert f'data-field="{hook}"' in TEXT
    if not hook.startswith(("home-card", "footer")):
        assert re.search(rf'data-field="{hook}"[^>]*data-value=', TEXT), hook


def test_ambulance_junction_hooks():
    """amb-j1..3-status are numbered by the template, one per corridor junction."""
    assert 'data-field="amb-j${k + 1}-status" data-value=' in SCRIPT
    assert ".slice(0, 3).map((jn, k)" in SCRIPT


def test_enf05_footer():
    assert "Données simulées" in BODY  # before any API answer
    assert "calibrées sur capteurs réels jusqu'au" in SCRIPT
    assert "programme de feux provisoire" in SCRIPT
    assert "certifi" not in PAGE.lower()


@pytest.mark.parametrize("banned", BANNED)
def test_honesty_ban_list(banned):
    assert banned not in PAGE


def test_no_hard_coded_results():
    assert not re.findall(r"[-+−]\s?\d+(?:[.,]\d+)?\s?%", TEXT)  # signed percentages
    assert not re.findall(r"\d+[.,]\d+\s?%", TEXT)  # decimal percentages
    assert not re.findall(r"\bp\s?[=<≤]\s?0?[.,]\d", TEXT)  # p-values
    assert not re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}", PAGE)  # run ids


def test_no_invented_fallbacks():
    """`b.<field> || <literal>` invents a value when the API has none (allow-list: notice and
    message texts, which are wording, not data)."""
    # an empty container ([] / {}) is no invented value
    pattern = r"\bb\.([a-z_]+)(?:\.[a-z_]+)*\s*(?:\|\||\?\?)\s*(?!\[\]|\{\})([\"'`\d\[{-])"
    hits = re.findall(pattern, SCRIPT)
    assert [h for h in hits if h[0] not in ("notice", "message")] == []


def test_no_backslash_r_escapes():
    assert "\\r" not in SCRIPT and "\r" not in SCRIPT.replace("\r\n", "\n")


def test_no_hard_coded_runs_or_reserved_seeds():
    assert not re.search(r"<option[^>]*>[^<$]*graine[^<$]*\d", TEXT)
    assert not re.search(r"<option value=\"[^\"$]", TEXT)  # options come from /scenarios


def test_storage_is_optional():
    for m in re.finditer(r"localStorage", SCRIPT):
        before = SCRIPT[: m.start()].rsplit("\n", 3)[-3:]
        assert any("try {" in ln for ln in before), "localStorage outside try/catch"


def test_deep_link_parameters():
    used = set(re.findall(r"P\.(?:get|has)\(\"(\w+)\"\)", SCRIPT))
    used |= set(re.findall(r"pick\(\"(\w+)\"", SCRIPT))
    used |= {"play", "kiosk"} if 'for (const key of ["play", "kiosk"])' in SCRIPT else set()
    assert set(DEEP_LINK_PARAMS) <= used, set(DEEP_LINK_PARAMS) - used


def test_constants_match_the_api_settings():
    s = load_settings(repo_root=REPO_ROOT, environ={})
    site = load_site(s.site_config, s.repo_root, s.field_file)
    crossings = re.search(r"const CROSSINGS = \[([^\]]*)\]", SCRIPT).group(1)
    assert set(re.findall(r'"(\w+)"', crossings)) == set(site.crossings)
    lines = re.search(r"const LINES = \[([^\]]*)\]", SCRIPT).group(1)
    assert set(re.findall(r'"(\w+)"', lines)) <= {b.line for b in site.bus_lines.values()}
    speeds = re.search(r"const SPEEDS = \[(.*?)\n\];", SCRIPT, flags=re.S).group(1)
    assert set(re.findall(r'\["(\w+)"', speeds)) == set(s.walking_speeds_mps)
    for key, mps in s.walking_speeds_mps.items():
        assert re.search(rf'\["{key}", "[^"]*\({mps:.1f} m/s\)"\]', speeds), key
    demo = re.search(r"const DEMO_SEEDS = \[(\d+), (\d+)\]", SCRIPT)
    assert s.demo_seeds is None or tuple(map(int, demo.groups())) == tuple(s.demo_seeds)


def test_net_geometry_is_the_simulated_net():
    """The 2D plan draws the simulated net (lanes, crossings, junction), so the state-log
    positions sit on their lanes: the page's <net-geometry> block is the one `python -m
    traffic_twin.app.netmap` writes from the scenario's net (after a net rebuild: rerun it)."""
    from traffic_twin.app import netmap

    geo = netmap.geometry(netmap.DEFAULT_SCENARIO, REPO_ROOT)
    assert netmap.replace_block(PAGE, netmap.block(geo)) == PAGE
    lanes = {ln["id"]: ln for ln in geo["lanes"]}
    assert set(geo["crossings"].values()) <= set(lanes)
    # every approach lane leads somewhere through the junction
    for ln in lanes.values():
        if ln.get("a") and ln["m"] & 7:
            assert ln.get("n") and all(n in lanes for n in ln["n"]), ln["id"]


def test_inline_script_parses_with_node(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = tmp_path / "mock.js"
    js.write_text(SCRIPT, encoding="utf-8")
    out = subprocess.run([node, "--check", str(js)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr


def test_style_is_only_in_the_head():
    assert STYLE and "@keyframes spin" in STYLE  # the loading spinner is defined


def test_replay_view_has_one_main_item_and_two_secondary_items():
    """Plan 4.4 / spec 12: the Relecture view is the map at t (replay-t) plus 2 secondary
    items, the vehicle count and the event ticker; no signal-stage pill, and the canvas
    badge does not repeat the count."""
    body = function_body("replay")
    assert set(re.findall(r'id="(hud-[\w-]+)"', body)) == {"hud-t", "hud-count"}
    assert 'id="ticker"' in body
    assert "hud-phase" not in SCRIPT
    assert "SIM.vehicles.length" not in function_body("drawSim")


def test_replay_clock_never_shows_an_old_instant_as_current():
    """12.1 during playback: the replay clock (not render()) writes the root state, the
    banners and the footer of the instant shown; a missing or failed frame empties the map
    and greys the lights; a failed frame is retried with a back-off, not on every tick."""
    sync = function_body("syncFleetFromReplay")
    assert re.search(r"if \(!f0\) \{\s*replayMissing\(k0\);\s*return false;", sync)
    assert "setReplayState(viewState(f0.b), f0.b)" in sync
    missing = function_body("replayMissing")
    for clear in ("SIM.vehicles = [];", "SIM.peds = [];", "REPLAY.signals = null;"):
        assert clear in missing, clear
    assert "FRAME_GRACE_MS" in missing and "setReplayState(" in missing
    load = function_body("loadFrame")
    assert "REPLAY.failed.set(k" in load and "FRAME_RETRY_MS" in load
    put = function_body("setReplayState")
    assert "dataset.viewState = st" in put and "banners(b)" in put and "footer(b)" in put
    assert re.search(r"FRAME_GRACE_MS = 1000\b", SCRIPT)
    assert re.search(r"FRAME_RETRY_MS = 2000\b", SCRIPT)


def test_operator_state_reflects_the_final_evaluation():
    """The Opérateur card is degraded when /evaluation is degraded or unavailable, or when a
    registered label has no /kpi answer (plan 4.4 root state), not only from /kpi."""
    body = function_body("operator")
    assert 'ev.status === "degraded" || ev.status === "unavailable"' in body
    assert re.search(r'main = \{ \.\.\.main, status: "degraded"', body)


# -- sign-in, one account per role (`api.admin`) ----------------------------------------------
def test_every_view_is_gated_by_the_role_session():
    """Every render first applies the role guard: no session -> the sign-in view; a view of
    another role -> the landing view of the session's role. The allowed views are the
    API's `views`, never a list of the page."""
    render = function_body("render")
    assert render.index("guardView();") < render.index("RENDER[state.view]")
    guard = function_body("guardView")
    assert 'state.view = "login"' in guard and "allowed[0]" in guard
    allowed = function_body("allowedViews")
    assert "state.session.views" in allowed and "gateOn()" in allowed
    assert re.search(r"const gateOn = \(\) => !state\.session \|\|", SCRIPT)  # unknown: gate on
    init = function_body("init")
    assert 'api("/auth/session")' in init and "setSession(sess)" in init
    # one view: no navigation bar; the user chip says the role and offers the logout
    nav = function_body("renderNav")
    assert "allowedViews()" in nav and "items.length < 2" in nav
    assert 'data-field="auth-role"' in function_body("renderUser")


def test_sign_in_form_and_requests():
    """The sign-in view asks /auth/session before anything else; the form has one
    identifier field and one password field that password managers fill, one button to
    send (and the show/hide eye of the password).
    The role comes from the address: the identifier (a full email, or its part before @)
    names an account of the API, shown as typed, and its email is posted to /auth/login;
    logout posts to /auth/logout. These are the only non-GET requests of the page; the
    password never stays in the page or storage."""
    body = function_body("login")
    first = re.search(r'\bapi\(\s*"([^"]+)"', body)
    assert first and first.group(1) == "/auth/session"
    form = function_body("loginHtml")
    assert "sess.accounts" in form and 'name="role"' not in form  # no role picker
    assert 'id="auth-email"' in form and 'type="password"' in form
    assert 'autocomplete="username"' in form and 'autocomplete="current-password"' in form
    assert form.count("<button") == 2 and form.count('type="submit"') == 1  # send + eye
    assert 'aria-label="Afficher le mot de passe"' in form and 'role="alert"' in form
    assert 'aria-describedby="auth-role"' in form and 'aria-describedby="auth-pw-msg"' in form
    assert 'split("@")[0] === t' in function_body("accountOf")  # "pieton" -> pieton@...
    assert "accounts-test" in form  # how to fix "aucun compte"
    submit = function_body("authSubmit")
    assert 'api("/auth/login", false, { method: "POST"' in submit
    assert "accountOf(email.value)" in submit and "acc.email" in submit
    assert 'pw.value = ""' in submit and "localStorage" not in submit
    assert 'api("/auth/logout", false, { method: "POST" })' in function_body("authLogout")
    assert re.findall(r'method: "(\w+)"', SCRIPT) == ["POST", "POST"]
    assert 'const keep = state.view !== "admin" && state.view !== "login";' in function_body(
        "render"
    )
    # the session ends at its expiry: read again, back to the sign-in screen
    expiry = function_body("authExpiry")
    assert "setTimeout" in expiry and 'api("/auth/session", true)' in expiry
    assert "render()" in expiry


def test_admin_view_needs_the_administrator_role():
    """The cockpit reads /auth/session before its data and shows the sign-in form (admin
    picked) unless the session's role is admin; it is never served from the last known
    value."""
    body = function_body("admin")
    first = re.search(r'\bapi\(\s*"([^"]+)"', body)
    assert first and first.group(1) == "/auth/session"
    assert 'sess.role !== "admin"' in body
    assert body.index("adminLogin(sess, banners(sess))") < body.index("adminData()")
    assert 'loginHtml(sess, alerts, "admin")' in function_body("adminLogin")


def test_admin_cockpit_reads_only_api_v1():
    """Both tabs read the public API v1 (no hard-coded result): the 10-seed evaluation,
    /kpi equity flags, /compare Holm, the demo runs of /scenarios, /replay frames and
    events, /ambulance; an absent demo run is shown as such."""
    code = "\n".join(function_body(n) for n in (
        "adminData", "adminSyncData", "adminSync", "adminTable", "holmOf", "flagOf",
    ))  # fmt: skip
    paths = (
        "/evaluation?family=main",
        "/kpi?scenario=",
        "/compare?a=S0",
        "/scenarios",
        "/events?limit=",
        "/ambulance/next-intersections?run_id=",
        "positions=true",
    )
    for path in paths:
        assert path in code, path
    assert 'seed_role === "demo"' in function_body("adminSyncData")
    assert "run indisponible" in function_body("adminSync")
    assert "main_solution" in function_body("adminData")
    assert "final_table" in function_body("adminRows")
    assert "a06" in function_body("holmOf") and "p_holm" in function_body("holmOf")
    assert "moyenne des graines" in function_body("adminSync")


def test_home_has_the_admin_entry_and_its_unconfigured_state():
    body = function_body("home")
    assert "state.session" in body and 'sess.role === "admin"' in body
    assert 'data-field="home-admin"' in body and "Se connecter en admin" in body
    assert "not_configured" in body and "admin-set-password --role admin" in body
    assert "Sans compte" not in body


def test_served_page_contains_the_admin_view(client):
    page = client.get("/mock").text
    assert "admin: { label:" in page and "/auth/session" in page and 'id="auth-form"' in page
    assert "Cockpit d'administration" in page and "Comparaison synchronisée" in page

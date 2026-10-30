"""Admin cockpit sign-in (`api.admin`; a local-only addition, not a user account system):
PBKDF2 hashing, the .env credential, POST /admin/login, /admin/logout, GET /admin/session,
the signed 30 min session cookie, the rate limit, and the CLI behind
`tasks.py admin-set-password`. Every test uses its own temporary .env, never the
developer's."""

from __future__ import annotations

import io
import re
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from traffic_twin.api import admin
from traffic_twin.api.main import create_app
from traffic_twin.api.settings import load_settings

REPO_ROOT = Path(__file__).resolve().parent.parent
EMAIL = "admin@urban-traffic-twin.local"
PASSWORD = "correct horse battery"
FAST = admin.MIN_ITERATIONS  # the lowest accepted cost keeps the suite fast


@pytest.fixture(autouse=True)
def _no_admin_environment(monkeypatch):
    for key in admin.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def configure(env: Path, password: str = PASSWORD, email: str = EMAIL) -> dict[str, str]:
    values = {
        admin.ENV_EMAIL: email,
        admin.ENV_HASH: admin.hash_password(password, iterations=FAST),
        admin.ENV_SECRET: "s" * 48,
    }
    admin.write_env_keys(env, values)
    return values


def client_for(env: Path) -> TestClient:
    settings = replace(load_settings(repo_root=REPO_ROOT, environ={}), admin_env_file=env)
    return TestClient(create_app(settings))


@pytest.fixture
def env(tmp_path) -> Path:
    path = tmp_path / ".env"
    configure(path)
    return path


@pytest.fixture
def client(env) -> TestClient:
    return client_for(env)


def login(c: TestClient, email: str = EMAIL, password: str = PASSWORD):
    return c.post("/admin/login", json={"email": email, "password": password})


# -- hashing ---------------------------------------------------------------------------------
def test_hash_format_and_verify():
    stored = admin.hash_password("s3cret-pass", iterations=FAST)
    scheme, iterations, salt, digest = stored.split("$")
    assert scheme == "pbkdf2_sha256" and int(iterations) == FAST
    assert salt and digest and "s3cret-pass" not in stored
    assert admin.verify_password("s3cret-pass", stored)
    assert not admin.verify_password("s3cret-pasS", stored)
    assert admin.hash_password("same", iterations=FAST) != admin.hash_password(
        "same", iterations=FAST
    )


def test_default_cost_and_weak_or_malformed_hashes():
    assert admin.ITERATIONS >= 200_000
    with pytest.raises(ValueError):
        admin.hash_password("x" * 10, iterations=1000)
    weak = admin.hash_password("pw123456", iterations=FAST).replace(f"${FAST}$", "$1000$")
    for bad in (weak, "", "plain", "md5$1$a$b", "pbkdf2_sha256$x$a$b"):
        assert admin.parse_hash(bad) is None
        assert not admin.verify_password("pw123456", bad)


# -- .env ------------------------------------------------------------------------------------
def test_env_file_keeps_other_lines_and_the_environment_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# local\nOTHER=1\nexport TT_ADMIN_EMAIL='old@x.y'\n", encoding="utf-8")
    admin.write_env_keys(env, {admin.ENV_EMAIL: "new@x.y", admin.ENV_SECRET: "k"})
    text = env.read_text(encoding="utf-8")
    assert "# local" in text and "OTHER=1" in text and "old@x.y" not in text
    assert admin.read_env_file(env)[admin.ENV_EMAIL] == "new@x.y"
    cfg = admin.load_config(env)
    assert cfg.email == "new@x.y" and not cfg.configured  # no hash yet
    assert admin.load_config(tmp_path / "missing.env").email == admin.DEFAULT_EMAIL
    monkeypatch.setenv(admin.ENV_EMAIL, "Env@X.Y")
    assert admin.load_config(env).email == "env@x.y"


# -- endpoints ---------------------------------------------------------------------------------
def test_unconfigured_admin_answers_503_and_says_so(tmp_path):
    c = client_for(tmp_path / "absent.env")
    r = login(c)
    assert r.status_code == 503
    body = r.json()
    assert body["type"] == "admin_session" and body["status"] == "unavailable"
    assert body["error"] == {"code": "admin_not_configured", "message": "admin non configuré"}
    s = c.get("/admin/session").json()
    assert s["configured"] is False and s["authenticated"] is False
    assert s["notice"] == "admin non configuré" and s["data_source"] == "simulation"


def test_login_sets_a_strict_httponly_cookie_and_opens_a_30_min_session(client):
    r = login(client, email="  ADMIN@urban-traffic-twin.local ")
    assert r.status_code == 200
    body = r.json()
    assert body["authenticated"] is True and body["email"] == EMAIL
    assert body["ttl_s"] == 1800 and 1790 <= body["remaining_s"] <= 1800
    assert body["calibrated_on"] and body["provisional"] in (True, False)  # the envelope
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{admin.COOKIE}=")
    for flag in ("HttpOnly", "Max-Age=1800", "Path=/", "SameSite=strict"):
        assert flag.lower() in cookie.lower(), flag
    assert "secure" not in cookie.lower()  # plain-http localhost; https adds Secure
    assert r.headers["cache-control"] == "no-store"
    s = client.get("/admin/session").json()
    assert s["authenticated"] is True and s["expires_at"] == body["expires_at"]


def test_cookie_is_secure_over_https(env):
    settings = replace(load_settings(repo_root=REPO_ROOT, environ={}), admin_env_file=env)
    c = TestClient(create_app(settings), base_url="https://testserver")
    assert "secure" in login(c).headers["set-cookie"].lower()


@pytest.mark.parametrize(
    "email, password", [(EMAIL, PASSWORD + "!"), ("other@x.y", PASSWORD), ("", ""), (EMAIL, "")]
)
def test_bad_credentials_are_401_without_a_cookie(client, email, password):
    r = login(client, email, password)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "bad_credentials"
    assert "set-cookie" not in r.headers
    assert client.get("/admin/session").json()["authenticated"] is False


def test_a_bad_body_is_the_usual_404_bad_request(client):
    r = client.post(
        "/admin/login",
        content="email=a&password=b",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 404 and r.json()["error"]["code"] == "bad_request"


def test_five_failures_per_minute_then_429(client, monkeypatch):
    now = [1_000_000.0]
    monkeypatch.setattr(client.app.state.admin, "clock", lambda: now[0])
    for _ in range(admin.MAX_FAILURES):
        assert login(client, password="wrong").status_code == 401
    r = login(client)  # even the right password waits
    assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_attempts"
    now[0] += admin.FAILURE_WINDOW_S + 1
    assert login(client).status_code == 200
    assert login(client, password="wrong").status_code == 401  # a success resets the count


def test_session_expires_after_30_minutes(client, monkeypatch):
    now = [2_000_000.0]
    monkeypatch.setattr(client.app.state.admin, "clock", lambda: now[0])
    assert login(client).status_code == 200
    now[0] += admin.SESSION_TTL_S - 5
    s = client.get("/admin/session").json()
    assert s["authenticated"] is True and s["remaining_s"] == 5
    now[0] += 10
    assert client.get("/admin/session").json()["authenticated"] is False


def test_logout_clears_the_cookie(client):
    login(client)
    r = client.post("/admin/logout")
    assert r.status_code == 200 and r.json()["authenticated"] is False
    assert "max-age=0" in r.headers["set-cookie"].lower()
    assert client.get("/admin/session").json()["authenticated"] is False


def test_tampered_foreign_or_stale_tokens_are_refused(client, env):
    token = login(client).cookies[admin.COOKIE]
    payload, sig = token.split(".")
    for bad in (f"{payload}.{sig[:-2]}AA", f"{payload}x.{sig}", "garbage", ""):
        client.cookies.set(admin.COOKIE, bad)
        assert client.get("/admin/session").json()["authenticated"] is False
    client.cookies.set(admin.COOKIE, token)
    assert client.get("/admin/session").json()["authenticated"] is True
    configure(env, password="a new password")  # a new password ends every open session
    assert client.get("/admin/session").json()["authenticated"] is False


def test_no_secret_in_any_response(client, env):
    values = admin.read_env_file(env)
    texts = [login(client, password="wrong").text, login(client).text]
    texts += [client.get("/admin/session").text, client.post("/admin/logout").text]
    for text in texts:
        assert PASSWORD not in text
        assert values[admin.ENV_HASH] not in text and values[admin.ENV_SECRET] not in text
        assert "pbkdf2" not in text


def test_admin_routes_leave_the_public_api_unchanged(client):
    """ENF-07: the public endpoints never need or set a cookie; CORS stays GET-only."""
    r = client.get("/health")
    assert r.status_code == 200 and "set-cookie" not in r.headers
    pre = client.options(
        "/admin/login",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "POST" not in pre.headers.get("access-control-allow-methods", "")


# -- CLI (tasks.py admin-set-password) ---------------------------------------------------------
def test_cli_writes_the_env_and_prints_no_secret(tmp_path, capsys, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.setattr("sys.stdin", io.StringIO("piped-password\n"))
    assert admin.main(["set-password", "--email", "Boss@Example.org", "--env-file", str(env)]) == 0
    out = capsys.readouterr().out
    values = admin.read_env_file(env)
    assert values[admin.ENV_EMAIL] == "boss@example.org"
    assert admin.verify_password("piped-password", values[admin.ENV_HASH])
    assert len(values[admin.ENV_SECRET]) >= 32
    assert "piped-password" not in out and values[admin.ENV_HASH] not in out
    assert values[admin.ENV_SECRET] not in out
    secret = values[admin.ENV_SECRET]
    admin.main(["set-password", "--password", "another-one", "--env-file", str(env)])
    again = admin.read_env_file(env)
    assert again[admin.ENV_SECRET] == secret  # kept; only created when missing
    assert admin.verify_password("another-one", again[admin.ENV_HASH])
    assert again[admin.ENV_EMAIL] == admin.DEFAULT_EMAIL


def test_cli_refuses_short_passwords_and_bad_emails(tmp_path):
    env = tmp_path / ".env"
    with pytest.raises(SystemExit):
        admin.main(["set-password", "--password", "short", "--env-file", str(env)])
    with pytest.raises(SystemExit):
        admin.main(
            [
                "set-password",
                "--email",
                "nobody",
                "--password",
                "long enough",
                "--env-file",
                str(env),
            ]
        )
    assert not env.exists()


def test_env_is_gitignored():
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in ignored]
    assert not re.search(
        r"pbkdf2_sha256\$\d+\$", (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    )

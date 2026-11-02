"""Sign-in of the user app, one account per role (`api.admin`; a local-only deviation from
ENF-07): PBKDF2 hashing, the .env accounts, POST /auth/login, /auth/logout, GET
/auth/session, the signed 30 min session cookie and the views of each role, the rate limit,
and the CLI behind `tasks.py admin-set-password` and `tasks.py accounts-test`. Every test
uses its own temporary .env, never the developer's."""

from __future__ import annotations

import io
import json
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
AMB_EMAIL = "ambulance@urban-traffic-twin.local"
AMB_PASSWORD = "sirene et gyrophare"
FAST = admin.MIN_ITERATIONS  # the lowest accepted cost keeps the suite fast


@pytest.fixture(autouse=True)
def _no_auth_environment(monkeypatch):
    for key in admin.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def configure(env: Path, password: str = PASSWORD, email: str = EMAIL, role: str = "admin"):
    r = admin.ROLE_BY_KEY[role]
    values = {
        r.email_key: email,
        r.hash_key: admin.hash_password(password, iterations=FAST),
        admin.ENV_SECRET: "s" * 48,
    }
    admin.write_env_keys(env, values)
    return values


def client_for(env: Path, auth: bool = True) -> TestClient:
    settings = replace(
        load_settings(repo_root=REPO_ROOT, environ={}), admin_env_file=env, auth=auth
    )
    return TestClient(create_app(settings))


@pytest.fixture
def env(tmp_path) -> Path:
    path = tmp_path / ".env"
    configure(path)
    configure(path, AMB_PASSWORD, AMB_EMAIL, "ambulance")
    return path


@pytest.fixture
def client(env) -> TestClient:
    return client_for(env)


def login(c: TestClient, email: str = EMAIL, password: str = PASSWORD):
    return c.post("/auth/login", json={"email": email, "password": password})


# -- roles -----------------------------------------------------------------------------------
def test_each_role_opens_its_own_views():
    views = {r.key: r.views for r in admin.ROLES}
    assert set(views) == {"pedestrian", "student", "bus", "ambulance", "operator", "admin"}
    for role in ("pedestrian", "student", "bus", "ambulance"):
        assert views[role] == (role,)
    assert views["operator"] == ("operator", "replay")
    assert set(views["admin"]) == set(admin.PUBLIC_VIEWS) | {"admin"}
    assert all("admin" not in v for k, v in views.items() if k != "admin")
    emails = [r.email for r in admin.ROLES]
    assert len(set(emails)) == len(emails)


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
    assert cfg.account("admin").email == "new@x.y" and not cfg.configured  # no hash yet
    absent = admin.load_config(tmp_path / "missing.env")
    assert absent.account("admin").email == admin.DEFAULT_EMAIL
    assert absent.account("pedestrian").email == "pieton@urban-traffic-twin.local"
    monkeypatch.setenv(admin.ENV_EMAIL, "Env@X.Y")
    assert admin.load_config(env).account("admin").email == "env@x.y"


# -- endpoints ---------------------------------------------------------------------------------
def test_no_account_answers_503_and_says_so(tmp_path):
    c = client_for(tmp_path / "absent.env")
    r = login(c)
    assert r.status_code == 503
    body = r.json()
    assert body["type"] == "auth_session" and body["status"] == "unavailable"
    assert body["error"] == {"code": "auth_not_configured", "message": "aucun compte configuré"}
    s = c.get("/auth/session").json()
    assert s["configured"] is False and s["authenticated"] is False and s["required"] is True
    assert s["notice"] == "aucun compte configuré" and s["data_source"] == "simulation"
    assert s["views"] == [] and [a["role"] for a in s["accounts"]] == [r.key for r in admin.ROLES]
    assert not any(a["configured"] for a in s["accounts"])


def test_session_lists_the_accounts_but_no_secret(client):
    s = client.get("/auth/session").json()
    acc = {a["role"]: a for a in s["accounts"]}
    assert acc["admin"]["configured"] and acc["ambulance"]["configured"]
    assert not acc["pedestrian"]["configured"]
    assert acc["ambulance"]["email"] == AMB_EMAIL and acc["ambulance"]["views"] == ["ambulance"]
    assert s["required"] is True and s["views"] == [] and s["role"] is None


def test_login_sets_a_strict_httponly_cookie_and_opens_a_30_min_session(client):
    r = login(client, email="  ADMIN@urban-traffic-twin.local ")
    assert r.status_code == 200
    body = r.json()
    assert body["authenticated"] is True and body["email"] == EMAIL
    assert body["role"] == "admin" and body["role_label"] == "Administrateur"
    assert "admin" in body["views"] and body["views"][0] == "home"
    assert body["ttl_s"] == 1800 and 1790 <= body["remaining_s"] <= 1800
    assert body["calibrated_on"] and body["provisional"] in (True, False)  # the envelope
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{admin.COOKIE}=")
    for flag in ("HttpOnly", "Max-Age=1800", "Path=/", "SameSite=strict"):
        assert flag.lower() in cookie.lower(), flag
    assert "secure" not in cookie.lower()  # plain-http localhost; https adds Secure
    assert r.headers["cache-control"] == "no-store"
    s = client.get("/auth/session").json()
    assert s["authenticated"] is True and s["expires_at"] == body["expires_at"]


def test_a_role_account_opens_only_its_views(client):
    r = login(client, AMB_EMAIL, AMB_PASSWORD)
    assert r.status_code == 200
    s = client.get("/auth/session").json()
    assert s["role"] == "ambulance" and s["views"] == ["ambulance"]
    assert s["role_label"] == "Ambulance" and s["email"] == AMB_EMAIL
    # the admin password does not open the ambulance account, and vice versa
    assert login(client, AMB_EMAIL, PASSWORD).status_code == 401
    assert login(client, EMAIL, AMB_PASSWORD).status_code == 401


def test_without_the_gate_the_public_views_open_without_a_session(env):
    s = client_for(env, auth=False).get("/auth/session").json()
    assert s["required"] is False and s["authenticated"] is False
    assert s["views"] == list(admin.PUBLIC_VIEWS) and "admin" not in s["views"]


def test_cookie_is_secure_over_https(env):
    settings = replace(load_settings(repo_root=REPO_ROOT, environ={}), admin_env_file=env)
    c = TestClient(create_app(settings), base_url="https://testserver")
    assert "secure" in login(c).headers["set-cookie"].lower()


@pytest.mark.parametrize(
    "email, password",
    [
        (EMAIL, PASSWORD + "!"),
        ("other@x.y", PASSWORD),
        ("", ""),
        (EMAIL, ""),
        ("pieton@urban-traffic-twin.local", PASSWORD),  # a role without a password
    ],
)
def test_bad_credentials_are_401_without_a_cookie(client, email, password):
    r = login(client, email, password)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "bad_credentials"
    assert "set-cookie" not in r.headers
    assert client.get("/auth/session").json()["authenticated"] is False


def test_a_bad_body_is_the_usual_404_bad_request(client):
    r = client.post(
        "/auth/login",
        content="email=a&password=b",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 404 and r.json()["error"]["code"] == "bad_request"


def test_five_failures_per_minute_then_429(client, monkeypatch):
    now = [1_000_000.0]
    monkeypatch.setattr(client.app.state.auth, "clock", lambda: now[0])
    for _ in range(admin.MAX_FAILURES):
        assert login(client, password="wrong").status_code == 401
    r = login(client)  # even the right password waits
    assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_attempts"
    now[0] += admin.FAILURE_WINDOW_S + 1
    assert login(client).status_code == 200
    assert login(client, password="wrong").status_code == 401  # a success resets the count


def test_session_expires_after_30_minutes(client, monkeypatch):
    now = [2_000_000.0]
    monkeypatch.setattr(client.app.state.auth, "clock", lambda: now[0])
    assert login(client).status_code == 200
    now[0] += admin.SESSION_TTL_S - 5
    s = client.get("/auth/session").json()
    assert s["authenticated"] is True and s["remaining_s"] == 5
    now[0] += 10
    assert client.get("/auth/session").json()["authenticated"] is False


def test_logout_clears_the_cookie(client):
    login(client)
    r = client.post("/auth/logout")
    assert r.status_code == 200 and r.json()["authenticated"] is False
    assert "max-age=0" in r.headers["set-cookie"].lower()
    assert client.get("/auth/session").json()["authenticated"] is False


def test_tampered_foreign_or_stale_tokens_are_refused(client, env):
    token = login(client).cookies[admin.COOKIE]
    payload, sig = token.split(".")
    for bad in (f"{payload}.{sig[:-2]}AA", f"{payload}x.{sig}", "garbage", ""):
        client.cookies.set(admin.COOKIE, bad)
        assert client.get("/auth/session").json()["authenticated"] is False
    # a token claiming another role is refused (its signature is the admin's)
    claims = json.loads(admin._unb64(payload))
    forged = admin._b64(json.dumps({**claims, "role": "ambulance"}).encode())
    client.cookies.set(admin.COOKIE, f"{forged}.{sig}")
    assert client.get("/auth/session").json()["authenticated"] is False
    client.cookies.set(admin.COOKIE, token)
    assert client.get("/auth/session").json()["authenticated"] is True
    configure(env, password="a new password")  # a new password ends the role's sessions
    assert client.get("/auth/session").json()["authenticated"] is False


def test_a_new_password_ends_only_that_roles_sessions(client, env):
    token = login(client, AMB_EMAIL, AMB_PASSWORD).cookies[admin.COOKIE]
    configure(env, password="another admin password")
    client.cookies.set(admin.COOKIE, token)
    assert client.get("/auth/session").json()["role"] == "ambulance"


def test_no_secret_in_any_response(client, env):
    values = admin.read_env_file(env)
    texts = [login(client, password="wrong").text, login(client).text]
    texts += [client.get("/auth/session").text, client.post("/auth/logout").text]
    for text in texts:
        assert PASSWORD not in text and AMB_PASSWORD not in text
        assert values[admin.ENV_HASH] not in text and values[admin.ENV_SECRET] not in text
        assert "pbkdf2" not in text


def test_auth_routes_leave_the_public_api_unchanged(client):
    """The public endpoints never need or set a cookie; CORS stays GET-only."""
    r = client.get("/health")
    assert r.status_code == 200 and "set-cookie" not in r.headers
    pre = client.options(
        "/auth/login",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "POST" not in pre.headers.get("access-control-allow-methods", "")


# -- accounts made by the admin (several per role) -------------------------------------------
def create(c: TestClient, name: str, role: str = "pedestrian", password: str = "mot de passe 1"):
    return c.post("/auth/accounts", json={"name": name, "role": role, "password": password})


def test_only_the_admin_session_manages_accounts(client, monkeypatch):
    monkeypatch.setattr(admin, "ITERATIONS", FAST)
    for r in (client.get("/auth/accounts"), create(client, "sami"),
              client.delete("/auth/accounts/sami")):  # fmt: skip
        assert r.status_code == 401 and r.json()["error"]["code"] == "not_signed_in"
        assert r.json()["type"] == "auth_accounts"
    login(client, AMB_EMAIL, AMB_PASSWORD)  # another role
    assert client.get("/auth/accounts").status_code == 403
    assert create(client, "sami").json()["error"]["code"] == "forbidden"
    login(client)
    r = client.get("/auth/accounts")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    body = r.json()
    assert [x["role"] for x in body["roles"]] == [x.key for x in admin.ROLES]
    assert [x["name"] for x in body["accounts"]] == [
        "pieton", "etudiant", "bus", "ambulance", "operateur", "admin"
    ]  # fmt: skip
    assert not any(x["created"] for x in body["accounts"]) and "pbkdf2" not in r.text


def test_the_admin_creates_several_accounts_per_role_that_sign_in_by_name(client, env, monkeypatch):
    monkeypatch.setattr(admin, "ITERATIONS", FAST)
    login(client)
    r = create(client, " Sami ", "Piéton", "sami password")  # role by label, name any case
    assert r.status_code == 200 and r.json()["created"] == "sami"
    assert create(client, "lina", "pedestrian", "lina password").status_code == 200
    accounts = {x["name"]: x for x in r.json()["accounts"]}
    assert accounts["sami"]["role"] == "pedestrian" and accounts["sami"]["created"] is True
    assert accounts["sami"]["email"] == f"sami@{admin.DOMAIN}" and accounts["sami"]["created_at"]
    # stored beside the .env, the hash only
    users = admin.users_file_for(env)
    assert users == env.parent / "data" / "auth" / "users.json"
    text = users.read_text(encoding="utf-8")
    assert "sami password" not in text and "pbkdf2_sha256$" in text
    # the public session never lists them
    client.post("/auth/logout")
    assert "sami" not in client.get("/auth/session").text
    # each signs in by name (or email) and opens the pedestrian view
    for ident, pw in (("sami", "sami password"), (f"LINA@{admin.DOMAIN}", "lina password")):
        r = login(client, ident, pw)
        assert r.status_code == 200 and r.json()["role"] == "pedestrian", ident
        assert r.json()["views"] == ["pedestrian"]
        assert client.get("/auth/session").json()["email"] == ident.split("@")[0].lower() + (
            f"@{admin.DOMAIN}"
        )
    assert login(client, "sami", "lina password").status_code == 401
    assert login(client, "pieton", PASSWORD).status_code == 401  # role account, no password


def test_account_creation_refusals_and_removal(client, env, monkeypatch):
    monkeypatch.setattr(admin, "ITERATIONS", FAST)
    login(client)
    cases = [
        ("admin", "pedestrian", "long enough", 409, "name_taken"),  # a role account's name
        ("ambulance", "bus", "long enough", 409, "name_taken"),
        ("a", "pedestrian", "long enough", 404, "bad_request"),
        ("jean pierre", "pedestrian", "long enough", 404, "bad_request"),
        ("élise", "pedestrian", "long enough", 404, "bad_request"),
        ("sami", "chauffeur", "long enough", 404, "bad_request"),
        ("sami", "pedestrian", "short", 404, "bad_request"),
    ]
    for name, role, pw, status, code in cases:
        r = create(client, name, role, pw)
        assert (r.status_code, r.json()["error"]["code"]) == (status, code), name
    assert create(client, "sami").status_code == 200
    assert create(client, "SAMI", "bus").json()["error"]["code"] == "name_taken"
    # removal: the created account only
    r = client.delete("/auth/accounts/admin")
    assert r.status_code == 409 and r.json()["error"]["code"] == "role_account"
    assert client.delete("/auth/accounts/nobody").json()["error"]["code"] == "not_found"
    token = None
    with TestClient(client.app) as other:
        token = login(other, "sami", "mot de passe 1").cookies[admin.COOKIE]
    r = client.delete("/auth/accounts/sami")
    assert r.status_code == 200 and "sami" not in [x["name"] for x in r.json()["accounts"]]
    assert login(client, "sami", "mot de passe 1").status_code == 401
    client.cookies.set(admin.COOKIE, token)  # its open session ends with it
    assert client.get("/auth/session").json()["authenticated"] is False


def test_cli_adds_lists_and_removes_accounts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(admin, "ITERATIONS", FAST)
    env = tmp_path / ".env"
    configure(env)
    args = ["--env-file", str(env)]
    assert admin.main(["add-user", "--name", "sami", "--role", "bus", "--password",
                       "sami password", *args]) == 0  # fmt: skip
    cfg = admin.load_config(env)
    acc = cfg.by_email(f"sami@{admin.DOMAIN}")
    assert acc and acc.created and acc.role.key == "bus" and cfg.usable(acc)
    admin.main(["list", *args])
    out = capsys.readouterr().out
    assert "sami" in out and "créé par l'admin" in out and "sami password" not in out
    with pytest.raises(SystemExit):  # a role account's name
        admin.main(["add-user", "--name", "admin", "--role", "bus", "--password", "x" * 9, *args])
    with pytest.raises(SystemExit):  # the role names cannot take a created account's name
        admin.main(["test-accounts", "--name", "pieton=sami", "--file",
                    str(tmp_path / "t.json"), *args])  # fmt: skip
    assert admin.main(["remove-user", "--name", "sami", *args]) == 0
    assert admin.load_config(env).by_email(f"sami@{admin.DOMAIN}") is None


# -- CLI (tasks.py admin-set-password, accounts-test) ----------------------------------------
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
    assert again[admin.ENV_EMAIL] == "boss@example.org"  # the email in .env is kept


def test_cli_sets_one_role_with_its_default_email(tmp_path):
    env = tmp_path / ".env"
    args = ["set-password", "--role", "bus", "--password", "bus-password", "--env-file"]
    assert admin.main([*args, str(env)]) == 0
    values = admin.read_env_file(env)
    assert values["TT_BUS_EMAIL"] == "bus@urban-traffic-twin.local"
    assert admin.verify_password("bus-password", values["TT_BUS_PASSWORD_HASH"])
    assert admin.ENV_HASH not in values  # the other roles are untouched


def test_cli_refuses_short_passwords_bad_emails_and_shared_emails(tmp_path):
    env = tmp_path / ".env"
    with pytest.raises(SystemExit):
        admin.main(["set-password", "--password", "short", "--env-file", str(env)])
    with pytest.raises(SystemExit):
        admin.main(
            ["set-password", "--email", "nobody", "--password", "long enough",
             "--env-file", str(env)]
        )  # fmt: skip
    assert not env.exists()
    with pytest.raises(SystemExit):  # the admin's email for the bus account
        admin.main(
            ["set-password", "--role", "bus", "--email", EMAIL, "--password", "long enough",
             "--env-file", str(env)]
        )  # fmt: skip
    with pytest.raises(SystemExit):  # the admin's name (the identifier typed) at another domain
        admin.main(
            ["set-password", "--role", "bus", "--email", "admin@example.org", "--password",
             "long enough", "--env-file", str(env)]
        )  # fmt: skip


def test_test_accounts_are_kept_in_a_file_and_reused(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(admin, "ITERATIONS", FAST)
    env, file = tmp_path / ".env", tmp_path / "auth" / "test_accounts.json"
    assert admin.main(["test-accounts", "--file", str(file), "--env-file", str(env)]) == 0
    rows = json.loads(file.read_text(encoding="utf-8"))["accounts"]
    assert [r["role"] for r in rows] == [r.key for r in admin.ROLES]
    cfg = admin.load_config(env)
    for row in rows:
        acc = cfg.account(row["role"])
        assert cfg.usable(acc) and acc.email == row["email"]
        assert admin.verify_password(row["password"], acc.password_hash)
        assert len(row["password"]) >= admin.MIN_PASSWORD_LEN
    out = capsys.readouterr().out
    assert all(r["password"] in out for r in rows)  # the table the tester reads
    # a second run reuses the file; an edited password is written into .env
    rows[0]["password"] = "edited-password"
    file.write_text(json.dumps({"accounts": rows}), encoding="utf-8")
    admin.main(["test-accounts", "--file", str(file), "--env-file", str(env)])
    again = json.loads(file.read_text(encoding="utf-8"))["accounts"]
    assert [r["password"] for r in again] == [r["password"] for r in rows]
    acc = admin.load_config(env).account(rows[0]["role"])
    assert admin.verify_password("edited-password", acc.password_hash)
    # --renew draws new passwords
    admin.main(["test-accounts", "--renew", "--file", str(file), "--env-file", str(env)])
    renewed = json.loads(file.read_text(encoding="utf-8"))["accounts"]
    assert all(a["password"] != b["password"] for a, b in zip(renewed, rows, strict=True))


def test_test_accounts_take_one_name_per_role(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(admin, "ITERATIONS", FAST)
    env, file = tmp_path / ".env", tmp_path / "test_accounts.json"
    first = {r["role"]: r for r in admin.seed_test_accounts(file, env)}
    args = ["--file", str(file), "--env-file", str(env)]
    names = ["--name", "Piéton=Ali", "--name", "bus=karim"]  # role by label or key
    assert admin.main(["test-accounts", *names, *args]) == 0
    rows = {r["role"]: r for r in json.loads(file.read_text(encoding="utf-8"))["accounts"]}
    assert rows["pedestrian"]["email"] == f"ali@{admin.DOMAIN}"
    assert rows["bus"]["email"] == f"karim@{admin.DOMAIN}"
    assert rows["student"]["email"] == first["student"]["email"]  # the others keep theirs
    pw = rows["pedestrian"]["password"]
    assert pw == first["pedestrian"]["password"]  # a new name keeps the password
    assert "ali" in capsys.readouterr().out
    c = client_for(env)
    r = login(c, f"ali@{admin.DOMAIN}", pw)
    assert r.status_code == 200 and r.json()["role"] == "pedestrian"
    assert r.json()["email"] == f"ali@{admin.DOMAIN}"
    assert login(c, f"pieton@{admin.DOMAIN}", pw).status_code == 401  # one name per role
    sess = c.get("/auth/session").json()
    assert [a["email"] for a in sess["accounts"] if a["role"] == "pedestrian"] == [
        f"ali@{admin.DOMAIN}"
    ]
    # --renew draws new passwords and keeps the names
    admin.main(["test-accounts", "--renew", *args])
    renewed = {r["role"]: r for r in json.loads(file.read_text(encoding="utf-8"))["accounts"]}
    assert renewed["pedestrian"]["email"] == f"ali@{admin.DOMAIN}"
    assert renewed["pedestrian"]["password"] != pw
    # refused: no "=", too short, a space, an accent, an unknown role, a name already taken
    for bad in ("ali", "pieton=a", "pieton=jean pierre", "pieton=élise", "chauffeur=ali",
                "bus=ali"):  # fmt: skip
        with pytest.raises(SystemExit):
            admin.main(["test-accounts", "--name", bad, *args])
    assert json.loads(file.read_text(encoding="utf-8"))["accounts"] == list(renewed.values())
    # the file holds the names; a name edited there renames the account at the next run
    assert [r["name"] for r in renewed.values()] == [
        "ali", "etudiant", "karim", "ambulance", "operateur", "admin"
    ]  # fmt: skip
    renewed["student"]["name"] = "sara"
    file.write_text(json.dumps({"accounts": list(renewed.values())}), encoding="utf-8")
    admin.main(["test-accounts", *args])
    rows = {r["role"]: r for r in json.loads(file.read_text(encoding="utf-8"))["accounts"]}
    assert rows["student"]["name"] == "sara" and rows["student"]["email"] == f"sara@{admin.DOMAIN}"
    assert admin.load_config(env).account("student").email == f"sara@{admin.DOMAIN}"


def test_test_accounts_sign_in_through_the_api(tmp_path, monkeypatch):
    monkeypatch.setattr(admin, "ITERATIONS", FAST)
    env, file = tmp_path / ".env", tmp_path / "test_accounts.json"
    rows = admin.seed_test_accounts(file, env)
    c = client_for(env)
    for row in rows:
        r = login(c, row["email"], row["password"])
        assert r.status_code == 200 and r.json()["role"] == row["role"], row["role"]


def test_env_and_test_accounts_are_gitignored():
    ignored = [ln.strip() for ln in (REPO_ROOT / ".gitignore").read_text("utf-8").splitlines()]
    assert ".env" in ignored and "data/auth/*" in ignored
    assert admin.TEST_ACCOUNTS.as_posix().startswith("data/auth/")
    assert not re.search(
        r"pbkdf2_sha256\$\d+\$", (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    )

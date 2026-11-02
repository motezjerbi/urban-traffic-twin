"""Sign-in of the user app, one account per role (a deliberate, local-only deviation from
ENF-07: every view of `mock.html` is behind a sign-in; the public GET endpoints of API v1
stay account-free and cookie-free).

Roles and the views each one opens (`ROLES`):

- `pedestrian` (Piéton), `student` (Étudiant), `bus` (Conducteur de bus) and `ambulance`:
  their own view, with the map;
- `operator` (Opérateur): the operator card and the replay;
- `admin` (Administrateur): every view and the admin cockpit.

Each account is read at each request from the environment, else from the gitignored `.env`
of the repo root (the environment wins):

- `TT_<ROLE>_EMAIL` (default `<role>@urban-traffic-twin.local`, e.g. `pieton@...`);
- `TT_<ROLE>_PASSWORD_HASH`: `pbkdf2_sha256$<iterations>$<salt b64>$<hash b64>` (hashlib,
  random 16-byte salt, at least 200 000 iterations);
- `TT_ADMIN_SECRET`: the random key that signs the session cookies of every role.

An account without a hash, or any account without the secret, cannot sign in; with no
usable account at all, `/auth/login` answers 503. The passwords and their hashes are never
committed, logged or sent back.

    python -m traffic_twin.api.admin set-password --role ambulance
    python -m traffic_twin.api.admin test-accounts [--renew]
    python -m traffic_twin.api.admin list

`set-password` reads the password with getpass (or one line of a piped stdin; `--password`
for scripts) and writes the role's keys into `.env`, creating the secret when missing. It
prints no secret. `test-accounts` gives every role a test account whose password is kept in
clear in the gitignored `data/auth/test_accounts.json` (created with one random password per
role, then reused, so the accounts survive every restart; edit a password there and run it
again to change it; `--renew` draws new ones) and prints the table. `list` prints each
role, its email and whether it is configured.

Sessions: an HMAC-SHA256-signed token `<payload b64>.<signature b64>` whose payload holds
the email, the role and the expiry (now + 30 min), in an HttpOnly, SameSite=Strict cookie
on `/`. The signing key mixes the secret with the role and its password hash, so a new
password ends every open session of that role. Failed logins are rate-limited in memory per
client (5 per minute -> 429).
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_SECRET = "TT_ADMIN_SECRET"  # signs the sessions of every role (historic name)
DOMAIN = "urban-traffic-twin.local"
ENV_FILE = Path(".env")  # repo root, gitignored

SCHEME = "pbkdf2_sha256"
ITERATIONS = 600_000  # OWASP 2023 for PBKDF2-HMAC-SHA256
MIN_ITERATIONS = 200_000  # a stored hash below this is refused (not configured)
SALT_BYTES = 16
MIN_PASSWORD_LEN = 8

COOKIE = "tt_session"
SESSION_TTL_S = 30 * 60
MAX_FAILURES = 5  # failed logins per client ...
FAILURE_WINDOW_S = 60.0  # ... within this window, then 429


# -- roles ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Role:
    key: str  # role id, as sent to the page
    env: str  # TT_<env>_EMAIL, TT_<env>_PASSWORD_HASH
    label: str  # French label shown by the page
    email: str  # default email of the account
    views: tuple[str, ...]  # the views of mock.html this role opens (first = landing view)

    @property
    def email_key(self) -> str:
        return f"TT_{self.env}_EMAIL"

    @property
    def hash_key(self) -> str:
        return f"TT_{self.env}_PASSWORD_HASH"


PUBLIC_VIEWS = ("home", "pedestrian", "student", "bus", "ambulance", "replay", "operator")
ALL_VIEWS = (*PUBLIC_VIEWS, "admin")
ROLES = (
    Role("pedestrian", "PEDESTRIAN", "Piéton", f"pieton@{DOMAIN}", ("pedestrian",)),
    Role("student", "STUDENT", "Étudiant", f"etudiant@{DOMAIN}", ("student",)),
    Role("bus", "BUS", "Conducteur de bus", f"bus@{DOMAIN}", ("bus",)),
    Role("ambulance", "AMBULANCE", "Ambulance", f"ambulance@{DOMAIN}", ("ambulance",)),
    Role("operator", "OPERATOR", "Opérateur", f"operateur@{DOMAIN}", ("operator", "replay")),
    Role("admin", "ADMIN", "Administrateur", f"admin@{DOMAIN}", ALL_VIEWS),
)
ROLE_BY_KEY = {r.key: r for r in ROLES}
ADMIN = ROLE_BY_KEY["admin"]
DEFAULT_EMAIL = ADMIN.email
ENV_EMAIL, ENV_HASH = ADMIN.email_key, ADMIN.hash_key  # the admin account
ENV_KEYS = (*(k for r in ROLES for k in (r.email_key, r.hash_key)), ENV_SECRET)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# -- password hashing ----------------------------------------------------------------------
def hash_password(password: str, iterations: int | None = None, salt: bytes | None = None) -> str:
    """`pbkdf2_sha256$<iterations>$<salt b64>$<hash b64>` of `password` (random salt;
    `ITERATIONS` by default, read at the call)."""
    iterations = ITERATIONS if iterations is None else iterations
    if iterations < MIN_ITERATIONS:
        raise ValueError(f"iterations must be >= {MIN_ITERATIONS}")
    salt = secrets.token_bytes(SALT_BYTES) if salt is None else salt
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{SCHEME}${iterations}${_b64(salt)}${_b64(dk)}"


def parse_hash(stored: str) -> tuple[int, bytes, bytes] | None:
    """(iterations, salt, hash) of a stored hash; None when malformed or too weak."""
    parts = stored.strip().split("$")
    if len(parts) != 4 or parts[0] != SCHEME or not parts[1].isdigit():
        return None
    iterations = int(parts[1])
    try:
        salt, dk = _unb64(parts[2]), _unb64(parts[3])
    except (ValueError, TypeError):
        return None
    if iterations < MIN_ITERATIONS or not salt or len(dk) != 32:
        return None
    return iterations, salt, dk


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of `password` against a stored hash (False when malformed)."""
    parsed = parse_hash(stored)
    if parsed is None:
        return False
    iterations, salt, dk = parsed
    got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(got, dk)


# -- .env ----------------------------------------------------------------------------------
def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines of a .env file (comments, blank lines and `export` allowed; one pair
    of surrounding quotes stripped); {} when absent or unreadable."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def write_env_keys(path: Path, values: Mapping[str, str]) -> None:
    """Set `values` in the .env file, replacing their lines and keeping every other line."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    left = dict(values)
    out = []
    for line in lines:
        key = line.strip().removeprefix("export ").split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in left:
            out.append(f"{key}={left.pop(key)}")
        else:
            out.append(line)
    out.extend(f"{k}={v}" for k, v in left.items())
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:  # best effort (Windows keeps its ACLs)
        pass


# -- configuration and sessions ------------------------------------------------------------
@dataclass(frozen=True)
class Account:
    role: Role
    email: str
    password_hash: str | None

    @property
    def has_password(self) -> bool:
        return self.password_hash is not None and parse_hash(self.password_hash) is not None


@dataclass(frozen=True)
class AuthConfig:
    accounts: tuple[Account, ...]  # one per role, in ROLES order
    secret: str | None

    def account(self, role: str) -> Account:
        return next(a for a in self.accounts if a.role.key == role)

    def usable(self, account: Account) -> bool:
        """The account can sign in: a valid hash and the signing secret."""
        return bool(self.secret) and account.has_password

    @property
    def configured(self) -> bool:
        """At least one account can sign in."""
        return any(self.usable(a) for a in self.accounts)

    def signing_key(self, account: Account) -> bytes:
        """The secret mixed with the role and its password hash: a new password ends every
        session of that role."""
        assert self.secret and account.password_hash
        material = f"{account.role.key}:{account.password_hash}".encode()
        return hmac.new(self.secret.encode("utf-8"), material, hashlib.sha256).digest()


def load_config(env_file: Path, environ: Mapping[str, str] | None = None) -> AuthConfig:
    """Every role account: the environment, else the .env file."""
    env = os.environ if environ is None else environ
    file = read_env_file(env_file)

    def get(key: str) -> str | None:
        value = env.get(key) or file.get(key)
        return value.strip() if value and value.strip() else None

    accounts = tuple(
        Account(role=r, email=(get(r.email_key) or r.email).lower(), password_hash=get(r.hash_key))
        for r in ROLES
    )
    return AuthConfig(accounts=accounts, secret=get(ENV_SECRET))


def make_token(
    cfg: AuthConfig, account: Account, now: float, ttl_s: int = SESSION_TTL_S
) -> tuple[str, int]:
    """(signed token, expiry epoch s) of a session of `account` opened at `now`."""
    exp = int(now) + ttl_s
    claims = {"sub": account.email, "role": account.role.key, "exp": exp}
    payload = _b64(json.dumps(claims, separators=(",", ":")).encode())
    sig = hmac.new(cfg.signing_key(account), payload.encode("ascii"), hashlib.sha256).digest()
    return f"{payload}.{_b64(sig)}", exp


def read_token(cfg: AuthConfig, token: str | None, now: float) -> tuple[Account, int] | None:
    """(account, expiry epoch s) of a valid, unexpired token of a usable account, else None."""
    if not token or token.count(".") != 1:
        return None
    payload, sig = token.split(".")
    try:
        data = json.loads(_unb64(payload))
        got = _unb64(sig)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("role") not in ROLE_BY_KEY:
        return None
    account = cfg.account(data["role"])
    if not cfg.usable(account):
        return None
    want = hmac.new(cfg.signing_key(account), payload.encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(got, want):
        return None
    exp = data.get("exp")
    if not isinstance(exp, int) or exp <= now or data.get("sub") != account.email:
        return None
    return account, exp


class RateLimiter:
    """In-memory failed-login counter per client: `limit` failures within `window_s`
    block further attempts until the oldest one leaves the window."""

    def __init__(self, limit: int = MAX_FAILURES, window_s: float = FAILURE_WINDOW_S) -> None:
        self.limit, self.window_s = limit, window_s
        self._fails: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, client: str, now: float) -> deque[float]:
        q = self._fails.setdefault(client, deque())
        while q and now - q[0] >= self.window_s:
            q.popleft()
        return q

    def blocked(self, client: str, now: float) -> float | None:
        """Seconds until the client may try again, None when not blocked."""
        with self._lock:
            q = self._recent(client, now)
            if len(q) < self.limit:
                return None
            return max(0.0, self.window_s - (now - q[0]))

    def failed(self, client: str, now: float) -> None:
        with self._lock:
            self._recent(client, now).append(now)

    def reset(self, client: str) -> None:
        with self._lock:
            self._fails.pop(client, None)


class Auth:
    """The sign-in gate of one app: account lookup, login check and rate limit."""

    def __init__(
        self,
        env_file: Path,
        environ: Mapping[str, str] | None = None,
        clock=time.time,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.env_file = env_file
        self.environ = environ
        self.clock = clock
        self.limiter = limiter or RateLimiter()

    def config(self) -> AuthConfig:
        return load_config(self.env_file, self.environ)

    def check(self, cfg: AuthConfig, email: str, password: str) -> Account | None:
        """The usable account of `email` when `password` matches it, else None. Every email
        is compared in constant time and one hash is always computed, so the answer time
        does not tell whether the email exists."""
        given = email.strip().lower().encode("utf-8")
        usable = [a for a in cfg.accounts if cfg.usable(a)]
        hit = None
        for a in usable:
            if hmac.compare_digest(given, a.email.encode("utf-8")) and hit is None:
                hit = a
        stored = hit or (usable[0] if usable else None)
        ok = verify_password(password, stored.password_hash or "") if stored else False
        return hit if hit is not None and ok else None


# -- CLI -------------------------------------------------------------------------------------
def _read_password(given: str | None, role: Role) -> str:
    if given is not None:
        return given
    if not sys.stdin.isatty():  # piped: one line
        return sys.stdin.readline().rstrip("\r\n")
    first = getpass.getpass(f"Mot de passe {role.label} : ")
    if getpass.getpass("Confirmer : ") != first:
        raise SystemExit("admin: les deux mots de passe diffèrent")
    return first


def set_password(
    email: str | None, password: str, env_file: Path, role: str = ADMIN.key
) -> list[str]:
    """Write the role's TT_<ROLE>_EMAIL and TT_<ROLE>_PASSWORD_HASH (and TT_ADMIN_SECRET when
    missing) into `env_file`; the names of the keys written. Without `email`, the email
    already in the file is kept, else the role's default."""
    if role not in ROLE_BY_KEY:
        raise SystemExit(f"admin: unknown role {role!r} ({', '.join(ROLE_BY_KEY)})")
    r = ROLE_BY_KEY[role]
    if len(password) < MIN_PASSWORD_LEN:
        raise SystemExit(f"admin: password must have at least {MIN_PASSWORD_LEN} characters")
    file = read_env_file(env_file)
    email = (email or file.get(r.email_key) or r.email).strip().lower()
    if "@" not in email:
        raise SystemExit(f"admin: {email!r} is not an email address")
    for other in ROLES:
        if other is not r and (file.get(other.email_key) or other.email).lower() == email:
            raise SystemExit(f"admin: {email} is already the email of the role {other.key}")
    values = {r.email_key: email, r.hash_key: hash_password(password)}
    if not file.get(ENV_SECRET):
        values[ENV_SECRET] = secrets.token_urlsafe(48)
    write_env_keys(env_file, values)
    return list(values)


# -- test accounts: one known password per role, kept in a local file ------------------------
TEST_ACCOUNTS = Path("data/auth/test_accounts.json")  # gitignored: passwords in clear
TEST_NOTE = (
    "Comptes de TEST de la connexion par profil (mots de passe en clair, locaux, jamais "
    "versionnés). Modifier un mot de passe ici puis relancer `python tasks.py accounts-test` "
    "le réécrit dans .env ; `--renew` en tire de nouveaux."
)
_SLUG = {"pedestrian": "pieton", "student": "etudiant", "bus": "bus", "ambulance": "ambulance",
         "operator": "operateur", "admin": "admin"}  # fmt: skip


def _test_password(role: Role) -> str:
    return f"{_SLUG[role.key]}-{secrets.token_hex(3)}"  # e.g. pieton-3fa9c1 (>= 8 characters)


def read_test_accounts(path: Path) -> dict[str, dict[str, str]]:
    """role -> {email, password} of the test accounts file ({} when absent or unreadable)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    rows = raw.get("accounts") if isinstance(raw, dict) else None
    out: dict[str, dict[str, str]] = {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("role") in ROLE_BY_KEY and row.get("password"):
            role = ROLE_BY_KEY[row["role"]]
            out[role.key] = {"email": str(row.get("email") or role.email).strip().lower(),
                             "password": str(row["password"])}  # fmt: skip
    return out


def seed_test_accounts(path: Path, env_file: Path, renew: bool = False) -> list[dict[str, str]]:
    """Every role gets the account of the test file (created, or completed, with a random
    password per missing role; all new with `renew`); the file is rewritten and each
    account's hash goes into `env_file`. The rows (role, label, email, password, views)."""
    known = {} if renew else read_test_accounts(path)
    rows = []
    for role in ROLES:
        acc = known.get(role.key) or {"email": role.email, "password": _test_password(role)}
        rows.append({"role": role.key, "label": role.label, "email": acc["email"],
                     "password": acc["password"], "views": ", ".join(role.views)})  # fmt: skip
    emails = [r["email"] for r in rows]
    if len(set(emails)) != len(emails):
        raise SystemExit(f"admin: {path}: two roles share an email")
    # every email first (set_password refuses an email still held by another role)
    write_env_keys(env_file, {ROLE_BY_KEY[r["role"]].email_key: r["email"] for r in rows})
    for r in rows:
        set_password(r["email"], r["password"], env_file, r["role"])
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "_note": TEST_NOTE,
        "accounts": [{k: r[k] for k in ("role", "label", "email", "password")} for r in rows],
    }
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m traffic_twin.api.admin", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    tp = sub.add_parser(
        "test-accounts", help=f"one test account per role, passwords kept in {TEST_ACCOUNTS}"
    )
    tp.add_argument("--file", type=Path, default=TEST_ACCOUNTS, help=f"default: {TEST_ACCOUNTS}")
    tp.add_argument("--renew", action="store_true", help="draw new passwords for every role")
    tp.add_argument("--env-file", type=Path, default=ENV_FILE, help="default: .env")
    sp = sub.add_parser("set-password", help="write a role account into .env")
    sp.add_argument("--role", default=ADMIN.key, choices=list(ROLE_BY_KEY), help="default: admin")
    sp.add_argument("--email", help="account email (default: the one in .env, else the role's)")
    sp.add_argument(
        "--password", help="the password (default: getpass, or one line of a piped stdin)"
    )
    sp.add_argument("--env-file", type=Path, default=ENV_FILE, help="default: .env")
    lp = sub.add_parser("list", help="each role, its email and whether it can sign in")
    lp.add_argument("--env-file", type=Path, default=ENV_FILE, help="default: .env")
    args = p.parse_args(argv)
    if args.cmd == "test-accounts":
        rows = seed_test_accounts(args.file, args.env_file, renew=args.renew)
        print(f"admin: {len(rows)} comptes de test écrits dans {args.env_file} ; "
              f"mots de passe dans {args.file} (local, jamais versionné)")  # fmt: skip
        for r in rows:
            print(f"  {r['label']:<18} {r['email']:<38} {r['password']:<20} vues : {r['views']}")
        return 0
    if args.cmd == "list":
        cfg = load_config(args.env_file)
        for a in cfg.accounts:
            state = "configuré" if cfg.usable(a) else "non configuré"
            print(f"{a.role.key:<10} {a.email:<40} {', '.join(a.role.views):<30} {state}")
        return 0
    role = ROLE_BY_KEY[args.role]
    keys = set_password(args.email, _read_password(args.password, role), args.env_file, role.key)
    print(f"admin: {', '.join(keys)} written to {args.env_file} (keep this file out of git)")
    if any(k in os.environ for k in (role.hash_key, ENV_SECRET)):
        print("admin: note: the environment sets TT_* keys and overrides the .env file")
    return 0


if __name__ == "__main__":
    sys.exit(main())

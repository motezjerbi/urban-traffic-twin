"""Local admin sign-in of the admin cockpit (a deliberate, local-only addition; not a user
account system: ENF-07 keeps the five public profiles account-free).

One admin credential, read at each request from the environment, else from the
gitignored `.env` of the repo root (the environment wins):

- `TT_ADMIN_EMAIL` (default `admin@urban-traffic-twin.local`);
- `TT_ADMIN_PASSWORD_HASH`: `pbkdf2_sha256$<iterations>$<salt b64>$<hash b64>` (hashlib,
  random 16-byte salt, at least 200 000 iterations);
- `TT_ADMIN_SECRET`: the random key that signs the session cookies.

Without a hash or a secret the admin is "non configuré" (`/admin/login` answers 503).
The password and its hash are never committed, logged or sent back.

    python -m traffic_twin.api.admin set-password --email admin@urban-traffic-twin.local

reads the password with getpass (or one line of a piped stdin; `--password` for scripts)
and writes the three keys into `.env`, creating the secret when missing. It prints no
secret.

Sessions: an HMAC-SHA256-signed token `<payload b64>.<signature b64>` whose payload holds
the email and the expiry (now + 30 min), in an HttpOnly, SameSite=Strict cookie on `/`.
The signing key mixes the secret with the password hash, so a new password ends every
open session. Failed logins are rate-limited in memory per client (5 per minute -> 429).
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

ENV_EMAIL = "TT_ADMIN_EMAIL"
ENV_HASH = "TT_ADMIN_PASSWORD_HASH"
ENV_SECRET = "TT_ADMIN_SECRET"
ENV_KEYS = (ENV_EMAIL, ENV_HASH, ENV_SECRET)
DEFAULT_EMAIL = "admin@urban-traffic-twin.local"
ENV_FILE = Path(".env")  # repo root, gitignored

SCHEME = "pbkdf2_sha256"
ITERATIONS = 600_000  # OWASP 2023 for PBKDF2-HMAC-SHA256
MIN_ITERATIONS = 200_000  # a stored hash below this is refused (not configured)
SALT_BYTES = 16
MIN_PASSWORD_LEN = 8

COOKIE = "tt_admin"
SESSION_TTL_S = 30 * 60
MAX_FAILURES = 5  # failed logins per client ...
FAILURE_WINDOW_S = 60.0  # ... within this window, then 429


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# -- password hashing ----------------------------------------------------------------------
def hash_password(password: str, iterations: int = ITERATIONS, salt: bytes | None = None) -> str:
    """`pbkdf2_sha256$<iterations>$<salt b64>$<hash b64>` of `password` (random salt)."""
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
class AdminConfig:
    email: str
    password_hash: str | None
    secret: str | None

    @property
    def configured(self) -> bool:
        return (
            bool(self.secret)
            and self.password_hash is not None
            and (parse_hash(self.password_hash) is not None)
        )

    def signing_key(self) -> bytes:
        """The secret mixed with the password hash: a new password ends every session."""
        assert self.secret and self.password_hash
        return hmac.new(
            self.secret.encode("utf-8"), self.password_hash.encode("utf-8"), hashlib.sha256
        ).digest()


def load_config(env_file: Path, environ: Mapping[str, str] | None = None) -> AdminConfig:
    """The admin credential: the environment, else the .env file."""
    env = os.environ if environ is None else environ
    file = read_env_file(env_file)

    def get(key: str) -> str | None:
        value = env.get(key) or file.get(key)
        return value.strip() if value and value.strip() else None

    return AdminConfig(
        email=(get(ENV_EMAIL) or DEFAULT_EMAIL).lower(),
        password_hash=get(ENV_HASH),
        secret=get(ENV_SECRET),
    )


def make_token(cfg: AdminConfig, now: float, ttl_s: int = SESSION_TTL_S) -> tuple[str, int]:
    """(signed token, expiry epoch s) of a session opened at `now`."""
    exp = int(now) + ttl_s
    payload = _b64(json.dumps({"sub": cfg.email, "exp": exp}, separators=(",", ":")).encode())
    sig = hmac.new(cfg.signing_key(), payload.encode("ascii"), hashlib.sha256).digest()
    return f"{payload}.{_b64(sig)}", exp


def read_token(cfg: AdminConfig, token: str | None, now: float) -> int | None:
    """The expiry of a valid, unexpired token of the configured admin, else None."""
    if not token or not cfg.configured or token.count(".") != 1:
        return None
    payload, sig = token.split(".")
    want = hmac.new(cfg.signing_key(), payload.encode("ascii"), hashlib.sha256).digest()
    try:
        ok = hmac.compare_digest(_unb64(sig), want)
        data = json.loads(_unb64(payload))
    except (ValueError, TypeError):
        return None
    if not ok or not isinstance(data, dict):
        return None
    exp = data.get("exp")
    if not isinstance(exp, int) or exp <= now or data.get("sub") != cfg.email:
        return None
    return exp


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


class AdminAuth:
    """The admin gate of one app: credential lookup, login check and rate limit."""

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

    def config(self) -> AdminConfig:
        return load_config(self.env_file, self.environ)

    def check(self, cfg: AdminConfig, email: str, password: str) -> bool:
        """Email and password match, in constant time (the hash is always computed)."""
        same_email = hmac.compare_digest(
            email.strip().lower().encode("utf-8"), cfg.email.encode("utf-8")
        )
        same_password = verify_password(password, cfg.password_hash or "")
        return same_email and same_password


# -- CLI -------------------------------------------------------------------------------------
def _read_password(given: str | None) -> str:
    if given is not None:
        return given
    if not sys.stdin.isatty():  # piped: one line
        return sys.stdin.readline().rstrip("\r\n")
    first = getpass.getpass("Mot de passe admin : ")
    if getpass.getpass("Confirmer : ") != first:
        raise SystemExit("admin: les deux mots de passe diffèrent")
    return first


def set_password(email: str, password: str, env_file: Path) -> list[str]:
    """Write TT_ADMIN_EMAIL and TT_ADMIN_PASSWORD_HASH (and TT_ADMIN_SECRET when missing)
    into `env_file`; the names of the keys written."""
    if len(password) < MIN_PASSWORD_LEN:
        raise SystemExit(f"admin: password must have at least {MIN_PASSWORD_LEN} characters")
    email = email.strip().lower()
    if "@" not in email:
        raise SystemExit(f"admin: {email!r} is not an email address")
    values = {ENV_EMAIL: email, ENV_HASH: hash_password(password)}
    if not read_env_file(env_file).get(ENV_SECRET):
        values[ENV_SECRET] = secrets.token_urlsafe(48)
    write_env_keys(env_file, values)
    return list(values)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m traffic_twin.api.admin", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("set-password", help="write the admin credential into .env")
    sp.add_argument("--email", default=DEFAULT_EMAIL, help=f"admin email ({DEFAULT_EMAIL})")
    sp.add_argument(
        "--password", help="the password (default: getpass, or one line of a piped stdin)"
    )
    sp.add_argument("--env-file", type=Path, default=ENV_FILE, help="default: .env")
    args = p.parse_args(argv)
    keys = set_password(args.email, _read_password(args.password), args.env_file)
    print(f"admin: {', '.join(keys)} written to {args.env_file} (keep this file out of git)")
    if ENV_HASH in os.environ or ENV_SECRET in os.environ:
        print("admin: note: the environment sets TT_ADMIN_* and overrides the .env file")
    return 0


if __name__ == "__main__":
    sys.exit(main())

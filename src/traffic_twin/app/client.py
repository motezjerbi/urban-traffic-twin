"""API client of the Streamlit Operator (`app.py`, D102): every read goes through the
API, never through the repository files.

- base URL `TT_API_URL` (default http://127.0.0.1:8000), 3 s timeout, envelope aware;
- `TT_API_INPROCESS=1`: an in-process `TestClient(create_app(load_settings()))` (tests,
  AppTest; `TT_API_CONFIG` selects the settings as for uvicorn);
- `get(path, **params) -> (payload | None, state)` with state ok / degraded / stale /
  unavailable / error; `fetch` returns the same plus a short French message. Neither
  raises to the UI: an unreachable API, a timeout, an error envelope (404, 503) or an
  unreadable answer all give `(None, "error")`.

State of an envelope (spec 12.1): `error` when it carries `error`, else `unavailable` /
`degraded` from `status`, else `stale` when the API flags the data as old, else `ok`.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Literal, NamedTuple

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"
TIMEOUT_S = 3.0
URL_ENV = "TT_API_URL"
INPROCESS_ENV = "TT_API_INPROCESS"
CONFIG_ENV = "TT_API_CONFIG"  # read by api.settings.load_settings
State = Literal["ok", "degraded", "stale", "unavailable", "error"]
STATES: tuple[str, ...] = ("ok", "degraded", "stale", "unavailable", "error")

UNREACHABLE = "API injoignable"
TIMED_OUT = f"l'API ne répond pas (délai de {TIMEOUT_S:g} s dépassé)"
UNREADABLE = "réponse illisible de l'API"
INTERNAL = "données temporairement indisponibles"


class Reply(NamedTuple):
    """One API answer: the envelope (None on error), its state, a short message for the
    UI (the error message, else the envelope's notice) and the HTTP status (None when
    the API could not be reached)."""

    payload: dict[str, Any] | None
    state: State
    message: str | None = None
    status_code: int | None = None


def api_url() -> str:
    """The API base URL (`TT_API_URL`, else DEFAULT_API_URL), without a trailing slash."""
    return (os.environ.get(URL_ENV) or DEFAULT_API_URL).rstrip("/")


def inprocess() -> bool:
    """True when `TT_API_INPROCESS` asks for the in-process API (tests)."""
    return os.environ.get(INPROCESS_ENV, "").strip().lower() not in ("", "0", "false", "no")


def describe() -> str:
    """Where the answers come from, for the UI footer."""
    if inprocess():
        return "API en processus (tests)"
    return api_url()


def identity() -> tuple[str, str]:
    """A key naming the API being read (cache keys in the UI)."""
    if inprocess():
        return ("inprocess", os.environ.get(CONFIG_ENV, ""))
    return ("http", api_url())


_clients: dict[tuple[str, str], Any] = {}
_lock = threading.Lock()


def _build(key: tuple[str, str]) -> Any:
    if key[0] == "inprocess":
        from fastapi.testclient import TestClient

        from traffic_twin.api.main import create_app
        from traffic_twin.api.settings import load_settings

        return TestClient(create_app(load_settings()), raise_server_exceptions=False)
    return httpx.Client(base_url=key[1], timeout=TIMEOUT_S, headers={"Accept": "application/json"})


def _client() -> Any:
    key = identity()
    with _lock:
        client = _clients.get(key)
        if client is None:
            client = _clients[key] = _build(key)
    return client


def reset() -> None:
    """Close and forget the cached clients (tests, a changed TT_API_URL)."""
    with _lock:
        clients = list(_clients.values())
        _clients.clear()
    for c in clients:
        try:
            c.close()
        except Exception:  # closing must never fail the UI
            pass


def state_of(payload: dict[str, Any]) -> State:
    """The 12.1 state of an envelope."""
    if payload.get("error"):
        return "error"
    status = payload.get("status")
    if status == "unavailable":
        return "unavailable"
    if status == "degraded":
        return "degraded"
    if payload.get("stale"):
        return "stale"
    return "ok"


def _params(params: dict[str, Any]) -> dict[str, str]:
    out = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = "true" if v else "false"
        elif isinstance(v, list | tuple | set):
            v = ",".join(str(x) for x in v)
        out[k] = str(v)
    return out


def fetch(path: str, **params: Any) -> Reply:
    """GET `path` (e.g. "/kpi") with query `params` (None dropped, lists joined by
    commas). Never raises."""
    if not path.startswith("/"):
        path = "/" + path
    try:
        resp = _client().get(path, params=_params(params))
    except httpx.TimeoutException:
        return Reply(None, "error", TIMED_OUT)
    except httpx.HTTPError:
        return Reply(None, "error", f"{UNREACHABLE} ({describe()})")
    except Exception:  # the in-process API failed to start (bad settings, missing files)
        return Reply(None, "error", INTERNAL)
    try:
        payload = resp.json()
    except ValueError:
        return Reply(None, "error", UNREADABLE, resp.status_code)
    if not isinstance(payload, dict):
        return Reply(None, "error", UNREADABLE, resp.status_code)
    if resp.status_code >= 400 or payload.get("error"):
        err = payload.get("error") or {}
        message = err.get("message") if isinstance(err, dict) else None
        return Reply(None, "error", message or f"erreur HTTP {resp.status_code}", resp.status_code)
    return Reply(payload, state_of(payload), payload.get("notice"), resp.status_code)


def get(path: str, **params: Any) -> tuple[dict | None, State]:
    """GET `path` with `params`: (envelope payload, state); (None, "error") when the API
    cannot be reached or answers an error."""
    reply = fetch(path, **params)
    return reply.payload, reply.state

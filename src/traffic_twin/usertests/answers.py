"""Expected answers of each task, computed through an in-process API (D111).

The run is resolved by (scenario, seed) through `/scenarios`, never by a run_id written in
advance (a run_id changes when a run is remade). Each `expect` entry of the protocol is
asked of the same endpoint the view reads, at the task's instant, so the answer key
cannot drift from what the app shows. The run_id and the sha256 of its `signal.parquet`
are recorded, so `usertest-serve` can refuse a campaign whose run changed (D111: the app
and the run are frozen for the campaign).

T-OP's answer is computed: the label with the lowest `/kpi` value among
`operator.labels`. A label missing from `/kpi` (S3 before the Part 8 run) leaves the
answer pending, and the kit refuses to build unless told so.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from traffic_twin.usertests.protocol import tasks_by_id

SIGNAL_FILE = "signal.parquet"


class AnswerError(RuntimeError):
    """The frozen run cannot be found, or the API cannot answer."""


def api_client(settings=None):
    """An in-process client of the API (`TT_API_CONFIG`, else configs/api.yaml)."""
    with warnings.catch_warnings():  # Starlette's TestClient warns that it builds on httpx
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient

    from traffic_twin.api.main import create_app
    from traffic_twin.api.settings import load_settings

    return TestClient(create_app(settings or load_settings()), raise_server_exceptions=False)


def _settings(client):
    return client.app.state.ctx.settings


def resolve_run(client, scenario: str, seed: int) -> dict:
    """The run of `scenario` at `seed` listed by `/scenarios` (run_id, dir, current)."""
    body = client.get("/scenarios").json()
    for group in body.get("scenarios") or []:
        if group.get("scenario") != scenario:
            continue
        for run in group.get("runs") or []:
            if run.get("seed") == seed:
                return {"scenario": scenario, "label": group.get("label"), "seed": seed, **run}
    raise AnswerError(
        f"no {scenario} run at seed {seed} in /scenarios: make the demo runs first "
        "(`tasks.py demo-runs`)"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_identity(client, run: dict) -> dict:
    """run_id, scenario, seed, code version and the sha256 of signal.parquet."""
    settings = _settings(client)
    d = settings.path(run["dir"])
    meta = {}
    if (d / "run.json").is_file():
        meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    sig = d / SIGNAL_FILE
    return {
        "run_id": run["run_id"],
        "scenario": run["scenario"],
        "label": run.get("label"),
        "seed": run["seed"],
        "dir": run["dir"],
        "current": run.get("current"),
        "code_version": meta.get("code_version"),
        "signal_sha256": sha256_file(sig) if sig.is_file() else None,
    }


def _field(body: Any, path: str) -> Any:
    """`intersections[0].status` read from a JSON body (None when absent)."""
    value = body
    for part in re.findall(r"[^.\[\]]+", path):
        if isinstance(value, list):
            idx = int(part) if part.isdigit() else None
            value = value[idx] if idx is not None and idx < len(value) else None
        elif isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def request_url(task: dict, expect: dict, run_id: str) -> str:
    """The API call behind one expectation, at the task's instant on the frozen run."""
    path = expect["endpoint"].replace("{run_id}", run_id)
    query = {k: str(v) for k, v in (expect.get("query") or {}).items()}
    if "{run_id}" not in expect["endpoint"]:
        query["run_id"] = run_id
    t = (task.get("start") or {}).get("t")
    if t is not None and "t" not in query:
        query["t"] = str(t)
    return f"{path}?{urlencode(query)}" if query else path


def operator_values(client, protocol: dict) -> dict:
    """`/kpi` value of `operator.kpi` per label, the argmin and what is missing."""
    op = protocol["operator"]
    values: dict[str, float | None] = {}
    for label in op["labels"]:
        body = client.get("/kpi", params={"scenario": label, "family": op["family"]}).json()
        rows = [
            r for r in body.get("rows") or []
            if r.get("user") == op["user"] and r.get("kpi") == op["kpi"]
        ]  # fmt: skip
        values[label] = rows[0]["mean"] if rows and rows[0].get("mean") is not None else None
    missing = [k for k, v in values.items() if v is None]
    present = {k: v for k, v in values.items() if v is not None}
    best = min(present, key=present.get) if present else None
    pending = [k for k in op["required"] if values.get(k) is None]
    return {
        "family": op["family"],
        "kpi": f"{op['user']}/{op['kpi']}",
        "values": values,
        "missing": missing,
        "pending": pending,
        "argmin": None if pending or missing else best,
    }


def expected_answers(protocol: dict, client=None) -> dict:
    """Task id -> expected answer and the API calls behind it, for the frozen run.

    Returns `{"run": identity, "api": {...}, "operator": {...}, "tasks": {id: {...}},
    "problems": [...]}`; a problem is an expectation that no longer holds."""
    client = client or api_client()
    run = resolve_run(client, protocol["run"]["scenario"], protocol["run"]["seed"])
    identity = run_identity(client, run)
    health = client.get("/health").json()
    out: dict[str, Any] = {
        "run": identity,
        "api": {k: health.get(k) for k in ("data_source", "calibrated_on", "provisional")},
        "operator": operator_values(client, protocol),
        "tasks": {},
        "problems": [],
    }
    for tid, task in tasks_by_id(protocol).items():
        checks = []
        for e in task["expect"]:
            if e.get("computed") == "operator":
                op = out["operator"]
                checks.append({
                    "url": "/kpi?scenario=<label>", "field": op["kpi"], "value": op["argmin"],
                    "expected": "argmin", "holds": op["argmin"] is not None, "pending": bool(
                        op["pending"]),
                })  # fmt: skip
                continue
            url = request_url(task, e, run["run_id"])
            resp = client.get(url)
            body = resp.json() if resp.headers.get("content-type", "").startswith(
                "application/json") else {}  # fmt: skip
            value = _field(body, e["field"])
            if resp.status_code != 200 or body.get("status") in ("error", "unavailable"):
                holds = False
            elif "equals" in e:
                holds = value == e["equals"]
            else:
                holds = value is not None
            checks.append({
                "url": url, "field": e["field"], "value": value, "expected": e.get("equals"),
                "tolerance": e.get("tolerance"), "unit": e.get("unit"), "holds": holds,
                "status": body.get("status"),
            })  # fmt: skip
            if not holds:
                out["problems"].append(
                    f"{tid}: {url} {e['field']} = {value!r} "
                    + (f"(expected {e['equals']!r})" if "equals" in e else "(expected a value)")
                )
        out["tasks"][tid] = {"checks": checks, "holds": all(c["holds"] for c in checks)}
    return out


def answer_text(task: dict, answer: dict) -> str:
    """The answer key line of one task (facilitator only)."""
    parts = []
    for c in answer["checks"]:
        if c["expected"] == "argmin":
            parts.append(f"lowest {c['field']}: {c['value'] or 'PENDING'}")
        elif c.get("tolerance") is not None:
            unit = f" {c['unit']}" if c.get("unit") else ""
            parts.append(f"{c['field']} = {c['value']}{unit} (± {c['tolerance']}{unit})")
        else:
            parts.append(f"{c['field']} = {json.dumps(c['value'], ensure_ascii=False)}")
    return "; ".join(parts)

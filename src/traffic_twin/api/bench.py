"""API latency bench (Part 5, ENF-02; contract 4.6): every endpoint, in process
(Starlette TestClient) and over real HTTP (uvicorn subprocess + httpx), on the run
folders given; rewrites the block between `<!-- latency:begin -->` and
`<!-- latency:end -->` of docs/api.md.

    python -m traffic_twin.api.bench --results data/results data/demo --n 200 --docs docs/api.md

ENF-02: "the API endpoints answer in less than 500 ms on precomputed results". The
percentiles leave out one warm-up request per endpoint; the cold start (building the
app, the run index and the site description, then the first request) is reported apart.
Request parameters are drawn with a fixed seed (no simulation is run). The bench only
reads the results.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from traffic_twin.api.clock import PARIS
from traffic_twin.api.settings import ENV_RESULTS_DIRS, Settings, load_settings

BEGIN, END = "<!-- latency:begin -->", "<!-- latency:end -->"
ENF02_LIMIT_MS = 500.0
BENCH_SEED = 4000  # draws request parameters only
CROSSINGS = ("NB", "S", "WB", "W")
SPEEDS = ("normal", "slow", "reduced")
BUSES = ("91", "24", "57", "91_E")
TRAJETS = ("N_S", "N_W", "S_N", "S_W", "W_N", "W_S")


def endpoints(runs: list[dict], rng: random.Random) -> dict[str, Callable[[], str]]:
    """Endpoint name -> URL factory (random run, time and parameters)."""
    ids = [r["run_id"] for r in runs] or ["default"]

    def t() -> str:
        return f"{rng.uniform(0, 7195):.0f}"

    return {
        "health": lambda: "/health",
        "scenarios": lambda: "/scenarios",
        "kpi": lambda: f"/kpi?scenario={rng.choice(['S0', 'S1', 'S2', 'S2_noP'])}",
        "replay (default run)": lambda: f"/replay/default?t={t()}",
        "replay (positions)": lambda: f"/replay/default?t={t()}&positions=true",
        "replay (random run)": lambda: f"/replay/{rng.choice(ids)}?t={t()}",
        "crossing": lambda: (
            f"/crossing/{rng.choice(CROSSINGS)}?speed={rng.choice(SPEEDS)}&t={t()}"
            f"&run_id={rng.choice(ids)}"
        ),
        "bus": lambda: f"/bus/{rng.choice(BUSES)}/status?t={t()}&run_id={rng.choice(ids)}",
        "ambulance": lambda: f"/ambulance/next-intersections?t={t()}&run_id={rng.choice(ids)}",
        "eta": lambda: f"/eta?trajet={rng.choice(TRAJETS)}&t={t()}&deadline=09:00",
        "error (404)": lambda: "/crossing/ZZ",
    }


def _stats(lat_ms: list[float], answers: Counter) -> dict:
    a = np.asarray(lat_ms)
    return {
        "p50_ms": round(float(np.percentile(a, 50)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1),
        "max_ms": round(float(a.max()), 1),
        "answers": dict(sorted(answers.items())),
    }


def _measure(get: Callable[[str], tuple[int, str]], urls: dict, n: int) -> dict:
    out = {}
    for name, make in urls.items():
        get(make())  # warm-up (left out)
        lat, answers = [], Counter()
        for _ in range(n):
            url = make()
            t0 = time.perf_counter()
            code, status = get(url)
            lat.append((time.perf_counter() - t0) * 1000.0)
            answers[f"{code} {status}"] += 1
        out[name] = _stats(lat, answers)
    return out


def _status(body: bytes) -> str:
    try:
        return str(json.loads(body).get("status"))
    except (ValueError, AttributeError):
        return "?"


def bench_inprocess(settings: Settings, n: int) -> tuple[dict, dict]:
    """(per-endpoint stats, info) with Starlette's TestClient."""
    from fastapi.testclient import TestClient

    from traffic_twin.api.main import create_app

    t0 = time.perf_counter()
    client = TestClient(create_app(settings), raise_server_exceptions=False)
    health = client.get("/health").json()
    cold = (time.perf_counter() - t0) * 1000.0
    runs = [r for g in client.get("/scenarios").json().get("scenarios", []) for r in g["runs"]]
    info = {
        "cold_start_ms": round(cold, 1),
        "runs_indexed": health.get("runs_indexed"),
        "default_run": health.get("default_run"),
        "eta_model_version": health.get("eta_model_version"),
    }

    def get(url: str) -> tuple[int, str]:
        r = client.get(url)
        return r.status_code, _status(r.content)

    return _measure(get, endpoints(runs, random.Random(BENCH_SEED)), n), info


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def bench_http(settings: Settings, n: int, timeout_s: float = 60.0) -> dict:
    """Per-endpoint stats over HTTP: `uvicorn traffic_twin.api.main:app --no-access-log`
    in a subprocess (results dirs passed through TT_RESULTS_DIRS)."""
    import httpx

    port = _free_port()
    env = dict(os.environ)
    env[ENV_RESULTS_DIRS] = os.pathsep.join(str(settings.path(d)) for d in settings.results_dirs)
    if settings.source is not None:
        env["TT_API_CONFIG"] = str(settings.source)
    cmd = [
        sys.executable, "-m", "uvicorn", "traffic_twin.api.main:app", "--no-access-log",
        "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning",
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, cwd=settings.repo_root, env=env)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                httpx.get(f"{base}/health", timeout=5.0)
                break
            except httpx.TransportError:
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("uvicorn did not start") from None
                time.sleep(0.2)
        with httpx.Client(base_url=base, timeout=30.0) as c:
            runs = [r for g in c.get("/scenarios").json().get("scenarios", []) for r in g["runs"]]

            def get(url: str) -> tuple[int, str]:
                r = c.get(url)
                return r.status_code, _status(r.content)

            return _measure(get, endpoints(runs, random.Random(BENCH_SEED + 1)), n)
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def verdict(result: dict) -> dict:
    worst = max(
        s["p95_ms"] for mode in ("inprocess", "http") for s in (result.get(mode) or {}).values()
    )
    return {"limit_ms": ENF02_LIMIT_MS, "worst_p95_ms": worst, "pass": worst < ENF02_LIMIT_MS}


def latency_block(result: dict) -> str:
    """The markdown that goes between the latency markers of docs/api.md."""
    http = result.get("http") or {}
    info = result["info"]
    dirs = ", ".join(f"`{d}`" for d in result["results_dirs"])
    v = result["enf02"]
    lines = [
        f"Measured {result['generated_at'][:16].replace('T', ' ')} (Europe/Paris) by "
        f"`python -m traffic_twin.api.bench`: code `{result['code_version']}`, Python "
        f"{result['python']}, {result['platform']}. {info['runs_indexed']} runs indexed from "
        f"{dirs}; default run `{(info.get('default_run') or {}).get('scenario')}` seed "
        f"{(info.get('default_run') or {}).get('seed')}; ETA model "
        f"`{info.get('eta_model_version')}`.",
        "",
        f"{result['n']} requests per endpoint after one warm-up request, parameters drawn "
        f"with seed {BENCH_SEED}. Cold start (app, run index, site, first request): "
        f"{info['cold_start_ms']:.0f} ms.",
        "",
        "| Endpoint | In-process p50 (ms) | p95 | max | HTTP p50 (ms) | p95 | max | Answers |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in result["inprocess"].items():
        h = http.get(name)
        hcols = f"{h['p50_ms']} | {h['p95_ms']} | {h['max_ms']}" if h else "n/a | n/a | n/a"
        answers = ", ".join(f"{k}: {c}" for k, c in s["answers"].items())
        lines.append(
            f"| {name} | {s['p50_ms']} | {s['p95_ms']} | {s['max_ms']} | {hcols} | {answers} |"
        )
    word = "PASS" if v["pass"] else "FAIL"
    lines += [
        "",
        f"**ENF-02 (< {v['limit_ms']:.0f} ms on precomputed results): {word}**, worst p95 "
        f"{v['worst_p95_ms']} ms.",
    ]
    return "\n".join(lines)


def rewrite_block(text: str, block: str) -> str:
    """`text` with the content between the latency markers replaced by `block`."""
    i, j = text.find(BEGIN), text.find(END)
    if i < 0 or j < 0 or j < i:
        raise ValueError(f"markers {BEGIN} ... {END} not found")
    return text[: i + len(BEGIN)] + "\n" + block.strip() + "\n" + text[j:]


def run_bench(settings: Settings, n: int, http: bool = True) -> dict:
    from traffic_twin.sim.results import code_version

    inproc, info = bench_inprocess(settings, n)
    result = {
        "generated_at": datetime.now(PARIS).isoformat(timespec="seconds"),
        "code_version": code_version(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "results_dirs": [Path(d).as_posix() for d in settings.results_dirs],
        "n": n,
        "info": info,
        "inprocess": inproc,
        "http": bench_http(settings, n) if http else None,
    }
    result["enf02"] = verdict(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", nargs="+", type=Path, help="run folders to index")
    parser.add_argument("--n", type=int, default=200, help="requests per endpoint")
    parser.add_argument("--docs", type=Path, help="rewrite the latency block of this page")
    parser.add_argument("--json", type=Path, help="also write the raw result here")
    parser.add_argument("--no-http", action="store_true", help="in-process only")
    a = parser.parse_args(argv)
    if a.n < 2:
        parser.error("--n must be >= 2")
    settings = load_settings()
    if a.results:
        settings = replace(settings, results_dirs=tuple(a.results))
    result = run_bench(settings, a.n, http=not a.no_http)
    block = latency_block(result)
    print(block)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    if a.docs:
        text = a.docs.read_text(encoding="utf-8")
        a.docs.write_text(rewrite_block(text, block), encoding="utf-8")
        print(f"latency block written to {a.docs}")
    return 0 if result["enf02"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

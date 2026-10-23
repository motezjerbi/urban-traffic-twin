"""`configs/demo.yaml` (plan §4.5): loading, validation, run resolution and the deep link
and API calls of every step. Shared by the demo commands and `report.shots`.

    cfg = load_demo_config()                         # ConfigError listing every problem
    runs = resolve_runs(cfg, scenarios_payload)      # {"main": {"run_id", ...}, "s3": None}
    step_url(cfg, step, runs, base_url)              # /mock?view=...&scenario=&seed=&t=
    step_calls(cfg, step, runs, labels)              # the API requests the view makes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import yaml

from traffic_twin.config import ConfigError
from traffic_twin.paths import DEMO_CONFIG

VIEWS = ("home", "pedestrian", "student", "bus", "ambulance", "replay", "operator")
FRONTS = ("mock", "streamlit")
PREREQUISITES = (
    "demo_runs",
    "evaluation_outputs",
    "eta_reference",
    "s3_model_sha",
    "evaluation_page",
)
DEGRADED = ("no_eta_bundle", "no_evaluation", "no_runs", "api_down")
DEMO_SEEDS = (4000, 4009)  # api.yaml seeds.demo (D101: the demo uses 4000 only)
STEP_KEYS = {
    "id",
    "view",
    "front",
    "run",
    "t",
    "crossing",
    "speed",
    "trajet",
    "exam",
    "line",
    "kiosk",
    "play",
}
SPEEDS = ("normal", "slow", "reduced")
DEFAULTS = {
    "part8_scenarios": [],
    "part8_labels": [],
    "operator_pair": {"a": "S1", "b": "S3"},
    "series_labels": ["S0", "S1", "S2", "S3"],
    "budgets": {
        "api_ms": 500,
        "api_start_s": 60,
        "operator_start_s": 90,
        "browser_wait_ms": 8000,
        "stop_s": 10,
    },
    "pack": {},
}
# The required `data-field` hooks of each view's main and secondary items (plan §4.4)
HOOKS = {
    "home": ("home-card-",),  # a prefix: at least one profile card
    "pedestrian": ("ped-verdict", "ped-remaining", "ped-walk"),
    "student": ("stu-eta", "stu-interval", "stu-advice"),
    "bus": ("bus-delay", "bus-headway", "bus-priority", "bus-message"),
    "ambulance": ("amb-j1-status", "amb-j2-status", "amb-j3-status", "amb-eta"),
    "replay": ("replay-t", "replay-count"),
    "operator": ("op-best",),
}
FOOTER_HOOKS = ("footer-source", "footer-calibrated")


@dataclass(frozen=True)
class Call:
    """One API request of a step: the path with its query, the expected message type,
    the statuses that pass, and whether `unavailable` means "not evaluated yet"."""

    group: str
    path: str
    type: str
    ok: frozenset[str] = frozenset({"ok", "degraded"})
    http: int = 200
    part8: bool = False  # `unavailable` "pas encore évalué" -> PENDING (Part 8 output)
    expect: dict[str, Any] = field(default_factory=dict)  # extra field checks


def _check_step(i: int, s: Any, runs: dict, problems: list[str]) -> None:
    where = f"steps[{i}]"
    if not isinstance(s, dict):
        problems.append(f"{where} must be a mapping")
        return
    unknown = set(s) - STEP_KEYS
    if unknown:
        problems.append(f"{where} unknown keys {sorted(unknown)}")
    if not s.get("id"):
        problems.append(f"{where} needs an id")
    view, front = s.get("view"), s.get("front", "mock")
    if front not in FRONTS:
        problems.append(f"{where} front must be one of {FRONTS}, got {front!r}")
    if front == "mock" and view not in VIEWS:
        problems.append(f"{where} view must be one of {VIEWS}, got {view!r}")
    if "run" in s and s["run"] not in runs:
        problems.append(f"{where} run {s['run']!r} is not in runs {sorted(runs)}")
    if "t" in s and (isinstance(s["t"], bool) or not isinstance(s["t"], int) or s["t"] < 0):
        problems.append(f"{where} t must be an integer >= 0 (s since the scenario start)")
    if "speed" in s and s["speed"] not in SPEEDS:
        problems.append(f"{where} speed must be one of {SPEEDS}")
    if view in ("pedestrian", "student", "bus", "ambulance", "replay") and "run" not in s:
        problems.append(f"{where} ({view}) needs a run")


def validate(cfg: dict) -> dict:
    """`cfg` with the WS7 defaults filled in; ConfigError listing every problem."""
    problems: list[str] = []
    for key in ("api", "operator", "runs", "prerequisites", "steps", "degraded"):
        if key not in cfg:
            problems.append(f"missing key {key!r}")
    if problems:
        raise ConfigError("configs/demo.yaml: " + "; ".join(problems))
    api, op = cfg["api"], cfg["operator"]
    for name, block in (("api", api), ("operator", op)):
        if not isinstance(block, dict) or not {"host", "port"} <= set(block):
            problems.append(f"{name} needs host and port")
        elif block["host"] not in ("127.0.0.1", "localhost"):
            problems.append(f"{name}.host must be local (127.0.0.1), got {block['host']!r}")
    if isinstance(api, dict) and not api.get("title"):
        problems.append("api.title is needed (the impostor check)")
    if isinstance(op, dict) and not op.get("script"):
        problems.append("operator.script is needed")
    runs = cfg["runs"] if isinstance(cfg["runs"], dict) else {}
    if not runs:
        problems.append("runs must be a non-empty mapping")
    for name, r in runs.items():
        if not isinstance(r, dict) or not {"scenario", "seed"} <= set(r):
            problems.append(f"runs.{name} needs scenario and seed")
            continue
        lo, hi = DEMO_SEEDS
        if (
            isinstance(r["seed"], bool)
            or not isinstance(r["seed"], int)
            or not lo <= r["seed"] <= hi
        ):
            problems.append(
                f"runs.{name}.seed must be a demo seed {lo}-{hi} (D101), got {r['seed']!r}"
            )
    bad = [p for p in cfg["prerequisites"] or [] if p not in PREREQUISITES]
    if bad:
        problems.append(f"unknown prerequisites {bad} (known: {list(PREREQUISITES)})")
    bad = [d for d in cfg["degraded"] or [] if d not in DEGRADED]
    if bad:
        problems.append(f"unknown degraded cases {bad} (known: {list(DEGRADED)})")
    steps = cfg["steps"] if isinstance(cfg["steps"], list) else []
    if not steps:
        problems.append("steps must be a non-empty list")
    for i, s in enumerate(steps):
        _check_step(i, s, runs, problems)
    ids = [s.get("id") for s in steps if isinstance(s, dict)]
    if len(set(ids)) != len(ids):
        problems.append("step ids must be unique")
    if problems:
        raise ConfigError("configs/demo.yaml: " + "; ".join(problems))
    out = {**DEFAULTS, **cfg}
    out["budgets"] = {**DEFAULTS["budgets"], **(cfg.get("budgets") or {})}
    return out


def load_demo_config(path: str | Path = DEMO_CONFIG, root: Path | None = None) -> dict:
    """The validated demo script (`path` relative to `root`, default: the repo root)."""
    p = Path(path)
    if not p.is_absolute():
        from traffic_twin.api.settings import find_repo_root

        p = (root or find_repo_root(DEMO_CONFIG)) / p
    if not p.is_file():
        raise ConfigError(f"demo config {p} not found")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: expected a mapping")
    return validate(raw)


# -- runs --------------------------------------------------------------------------------
def resolve_runs(cfg: dict, scenarios: dict | None) -> dict[str, dict | None]:
    """{run name: {run_id, scenario, seed, label, controller, dir} or None} from a
    /scenarios payload, by (scenario, seed); the first current run wins."""
    groups = {g.get("scenario"): g for g in (scenarios or {}).get("scenarios") or []}
    out: dict[str, dict | None] = {}
    for name, want in cfg["runs"].items():
        g = groups.get(want["scenario"])
        hits = [r for r in (g or {}).get("runs") or [] if r.get("seed") == want["seed"]]
        hits.sort(key=lambda r: r.get("current") is not True)
        out[name] = (
            {
                **hits[0],
                "scenario": want["scenario"],
                "label": g.get("label"),
                "controller": g.get("controller"),
            }
            if hits
            else None
        )
    return out


def is_part8_run(cfg: dict, name: str) -> bool:
    """True when run `name` is made only by the integration (`demo-runs8`)."""
    return cfg["runs"][name]["scenario"] in set(cfg.get("part8_scenarios") or [])


# -- deep links (D106) -------------------------------------------------------------------
def step_query(cfg: dict, step: dict, operator_url: str | None = None) -> dict[str, str]:
    """The deep-link parameters of a mock step (scenario + seed, never the run_id)."""
    q: dict[str, str] = {"view": step["view"]}
    if "run" in step:
        r = cfg["runs"][step["run"]]
        q.update(scenario=r["scenario"], seed=str(r["seed"]))
    for key in ("t", "crossing", "speed", "trajet", "exam", "line", "kiosk", "play"):
        if key in step:
            q[key] = str(step[key])
    if step["view"] == "replay" and "play" not in step:
        q["play"] = "0"  # a still frame: the DOM check and screenshots are deterministic
    if operator_url:
        q["operator"] = operator_url
    return q


def step_url(cfg: dict, step: dict, base_url: str, operator_url: str | None = None) -> str:
    """`<base>/mock?view=...` for a mock step; the Operator URL for a Streamlit step."""
    if step.get("front", "mock") == "streamlit":
        return operator_url or operator_base(cfg)
    return f"{base_url.rstrip('/')}/mock?{urlencode(step_query(cfg, step, operator_url))}"


def api_base(cfg: dict, port: int | None = None) -> str:
    return f"http://{cfg['api']['host']}:{port or cfg['api']['port']}"


def operator_base(cfg: dict, port: int | None = None) -> str:
    return f"http://{cfg['operator']['host']}:{port or cfg['operator']['port']}"


# -- API calls of each step --------------------------------------------------------------
def _q(**params: Any) -> str:
    return urlencode({k: v for k, v in params.items() if v is not None})


def step_calls(
    cfg: dict, step: dict, runs: dict[str, dict | None], labels: list[str]
) -> list[Call] | str:
    """The API requests the step's view makes, or a reason string when its run is not
    resolved (the caller reports it)."""
    front, view = step.get("front", "mock"), step.get("view")
    run_id = None
    if "run" in step:
        r = runs.get(step["run"])
        if r is None:
            return (
                f"run {step['run']!r} ({cfg['runs'][step['run']]['scenario']} seed "
                f"{cfg['runs'][step['run']]['seed']}) not found by /scenarios"
            )
        run_id = r["run_id"]
    t = step.get("t")
    p8 = set(cfg.get("part8_labels") or [])
    if front == "streamlit":
        pair = cfg["operator_pair"]
        sel = ",".join(cfg["series_labels"])
        return [
            Call("operator", "/evaluation?" + _q(family="main"), "evaluation", part8=True),
            Call(
                "operator",
                "/compare?" + _q(a=pair["a"], b=pair["b"], family="main"),
                "kpi_pair",
                part8=True,
            ),
            Call(
                "operator",
                "/series?" + _q(family="main", metric="halting_veh", approach="all", labels=sel),
                "kpi_series",
                part8=True,
            ),
            Call("operator", "/sensitivity", "sensitivity"),
        ]
    if view == "home":
        return [
            Call("home", "/health", "health"),
            Call("home", "/scenarios", "scenario_list"),
            Call("home", "/evaluation?" + _q(family="main"), "evaluation", part8=True),
        ]
    if view == "pedestrian":
        return [
            Call(
                view,
                f"/crossing/{step['crossing']}?"
                + _q(speed=step.get("speed", "normal"), t=t, run_id=run_id),
                "crossing_status",
            ),
        ]
    if view == "student":
        return [
            Call(
                view,
                "/eta?"
                + _q(
                    trajet=step["trajet"],
                    student="true",
                    deadline=step.get("exam"),
                    t=t,
                    run_id=run_id,
                ),
                "eta",
            ),
        ]
    if view == "bus":
        return [Call(view, f"/bus/{step['line']}/status?" + _q(t=t, run_id=run_id), "bus_status")]
    if view == "ambulance":
        return [
            Call(view, "/ambulance/next-intersections?" + _q(t=t, run_id=run_id), "ambulance_route")
        ]
    if view == "replay":
        lo = None if t is None else max(0, int(t) - 60)
        return [
            Call(
                view,
                f"/replay/{run_id}?" + _q(t=t, positions="true"),
                "replay_state",
                ok=frozenset({"ok"}),
            ),
            Call(
                view,
                f"/replay/{run_id}/events?" + _q(**{"from": lo, "to": t}),
                "replay_events",
                ok=frozenset({"ok"}),
                part8=True,
            ),
        ]
    if view == "operator":
        return [
            Call(
                view,
                "/kpi?" + _q(scenario=lab, user="all", family="main"),
                "kpi_table",
                part8=lab in p8,
            )
            for lab in labels
        ]
    return []


def registered_labels(root: Path, evaluation_config: Path | None = None) -> list[str]:
    """The family-main labels of the evaluation config (S0, S1, S2, S2_noP, S3, S4)."""
    from traffic_twin.paths import EVALUATION_CONFIG

    p = evaluation_config or EVALUATION_CONFIG
    p = p if Path(p).is_absolute() else root / p
    try:
        raw = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
        return list(((raw.get("families") or {}).get("main") or {}).get("scenarios") or {})
    except (OSError, yaml.YAMLError):
        return []

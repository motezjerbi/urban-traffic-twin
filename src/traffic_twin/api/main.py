"""API v1 (Part 5; spec 9.2, 9.3; D53; contract 4.6): FastAPI app over the local run
folders. Every JSON response carries the 9.2 envelope, errors included.

    uvicorn traffic_twin.api.main:app --no-access-log --host 127.0.0.1 --port 8000

`app` is built at import from configs/api.yaml (+ the TT_API_CONFIG and TT_RESULTS_DIRS
environment variables, see `api.settings`). The store, the site description and the ETA
spec are built on the first request.

Replay clock: time-dependent endpoints take `run_id` (default: the configured default
run; `/replay/default` is its alias) and `t` (seconds since the scenario start or
`HH:MM[:SS]`); "now" is that instant of the run (`api.clock`).

Error policy (ENF-03): 404 with an envelope and a code for an unknown id or a bad
parameter (FastAPI's 422 is remapped to `bad_request`); 200 with `status: degraded` or
`unavailable` and a notice when data is missing; 503 without a traceback for anything
unexpected. Never 500.

Honesty labels (D53): `remaining_basis` (`program` for the fixed-time S0, whose future
greens are known; `replay` for adaptive controllers, whose replayed countdown is future
knowledge a live system would not have); bus `delay_basis: junction_time_loss` (the
simulation has no timetable); 2 of 3 corridor junctions `unknown`; `obstructed: null`
(not logged in v1).
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections.abc import Mapping
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from traffic_twin.api.clock import (
    PARIS,
    age_s,
    parse_deadline,
    parse_t,
    replay_instant,
    wall_now,
)
from traffic_twin.api.db import LocalStore, RunInfo, Store
from traffic_twin.api.schemas import (
    AmbulanceRouteResponse,
    ApiError,
    BusStatusResponse,
    CrossingStatusResponse,
    Envelope,
    ErrorInfo,
    EtaAdvice,
    EtaResponse,
    HealthResponse,
    Intersection,
    KpiRow,
    KpiTableResponse,
    Position,
    ReplayStateResponse,
    RunRef,
    ScenarioGroup,
    ScenarioListResponse,
    ScenarioRun,
)
from traffic_twin.api.settings import Settings, load_settings
from traffic_twin.api.site import EtaSpec, Site, load_eta_spec, load_site, phase_map

log = logging.getLogger(__name__)

API_VERSION = "1.0.0"
USERS = ("car", "bus", "pedestrian", "ambulance")  # state_log entities besides the junction
# Message type of each path (used for error envelopes too)
PATH_TYPES = (
    ("/health", "health"), ("/scenarios", "scenario_list"), ("/kpi", "kpi_table"),
    ("/replay", "replay_state"), ("/crossing", "crossing_status"), ("/bus", "bus_status"),
    ("/ambulance", "ambulance_route"), ("/eta", "eta"),
)  # fmt: skip
SPEED_RANGE_MPS = (0.3, 3.0)  # numeric walking speeds accepted by /crossing
APPROX = "estimation approximative"  # spec 11.4 wording of the degraded ETA
CORRIDOR_NOTICE = "1 carrefour modélisé sur {n} : les autres sont inconnus"
TSP_RE = re.compile(r"line=([^,\s]+).*?allow=([\d.]+)")
VID_RE = re.compile(r"vid=([^,\s]+)")
LINK_RE = re.compile(r"link=(\d+)")
HALT_SPEED_MPS = 0.1  # SUMO's halting threshold: an ambulance slower than this is stopped


def _f(x: Any, ndigits: int | None = 1) -> float | None:
    """A finite float (rounded) or None: NaN never reaches the JSON encoder."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return round(x, ndigits) if ndigits is not None else x


def message_type(path: str) -> str:
    return next((kind for prefix, kind in PATH_TYPES if path.startswith(prefix)), "error")


def run_ref(run: RunInfo) -> RunRef:
    return RunRef(
        run_id=run.run_id,
        scenario=run.scenario,
        seed=run.seed,
        controller=run.controller,
        code_version=run.code_version,
        current=run.current,
    )


class Context:
    """Per-app resources, built on first use (thread safe)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._store: Store | None = None
        self._site: Site | None = None
        self._eta: EtaSpec | None = None

    @property
    def store(self) -> Store:
        if self._store is None:
            with self._lock:
                if self._store is None:
                    self._store = LocalStore(self.settings)
        return self._store

    @property
    def site(self) -> Site:
        if self._site is None:
            with self._lock:
                if self._site is None:
                    s = self.settings
                    self._site = load_site(s.site_config, s.repo_root, s.field_file)
        return self._site

    @property
    def eta_spec(self) -> EtaSpec:
        if self._eta is None:
            with self._lock:
                if self._eta is None:
                    self._eta = load_eta_spec(self.settings.path(self.settings.eta_config))
        return self._eta

    def site_or_none(self) -> Site | None:
        try:
            return self.site
        except Exception:  # error envelopes must not fail on a broken site config
            return None

    # -- envelope helpers ----------------------------------------------------------------
    def envelope(
        self,
        kind: str,
        *,
        timestamp: datetime,
        age: float | None = None,
        run: RunInfo | None = None,
        t: float | None = None,
        status: str = "ok",
        notice: str | None = None,
        limit_s: float | None = None,
    ) -> dict[str, Any]:
        limit = limit_s if limit_s is not None else self.settings.freshness_s.get(kind)
        site = self.site
        return {
            "calibrated_on": site.calibrated_on,
            "provisional": site.provisional,
            "timestamp": timestamp,
            "data_age_s": age,
            "stale": age is not None and limit is not None and age > limit,
            "run": run_ref(run) if run is not None else None,
            "sim_t_s": _f(t),
            "status": status,
            "notice": notice,
        }

    def run_and_t(self, run_id: str | None, t: str | None) -> tuple[RunInfo, float]:
        if run_id and run_id != "default":
            run = self.store.run(run_id)
        else:
            run = self.store.default_run()
            if run is None:
                raise ApiError("unknown_run", "no simulation run is indexed")
        s = self.settings
        tq = parse_t(
            t,
            begin_s=run.begin_s,
            duration_s=run.duration_s,
            default_t=s.default_t,
            mode=s.clock_mode,
        )
        return run, tq

    def instant(self, run: RunInfo, t: float) -> datetime:
        return replay_instant(run.begin_s, t, self.settings.replay_date)

    def unavailable(self, kind: str, run: RunInfo, t: float, notice: str) -> dict[str, Any]:
        return self.envelope(
            kind, timestamp=self.instant(run, t), run=run, t=t, status="unavailable", notice=notice
        )

    def other_site(self, run: RunInfo) -> str | None:
        """A notice when the run simulates another network than the site (e.g. the grid
        baseline): its crossing, bus, ambulance and ETA answers would be meaningless."""
        net = (run.config.get("network") or {}).get("net_file")
        if net is None or Path(str(net)).as_posix() == Path(self.site.net_file).as_posix():
            return None
        return f"ce run simule un autre réseau ({Path(str(net)).name}) que le carrefour Pitié"

    def stage_of(self, run: RunInfo, phase: int | None):
        """(stage index, name, in clearance) of a SUMO phase, from the run's own
        `network.stages` (the site's when the run.json has none)."""
        stages = (run.config.get("network") or {}).get("stages")
        if not stages:
            return self.site.stage_of(phase)
        try:
            k, clearing = phase_map(stages)[phase]
            return k, str(stages[k]["name"]), clearing
        except (KeyError, TypeError, ValueError, IndexError):
            return None, None, None


def error_response(
    ctx: Context, request: Request, status_code: int, code: str, message: str
) -> JSONResponse:
    site = ctx.site_or_none()
    body = Envelope(
        type=message_type(request.url.path),
        timestamp=wall_now(),
        status="unavailable",
        calibrated_on=site.calibrated_on if site else None,
        provisional=site.provisional if site else True,
        error=ErrorInfo(code=code, message=message),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


# -- endpoint logic (plain functions of the context; the routes below wrap them) -----------
def latest_rows(df: pd.DataFrame | None, t: float, key: str) -> pd.DataFrame:
    """The rows of the last time <= t of a table with a `t` column (one per `key`)."""
    if df is None or df.empty or "t" not in df.columns:
        return pd.DataFrame()
    past = df[df["t"] <= t]
    if past.empty:
        return past
    return past[past["t"] == past["t"].max()].drop_duplicates(key, keep="last")


def queue_by_approach(ctx: Context, run: RunInfo, t: float) -> dict[str, int]:
    q = latest_rows(ctx.store.queues(run.run_id), t, "approach")
    if q.empty or "halting_veh" not in q.columns:
        return {}
    return {str(a): int(h) for a, h in zip(q["approach"], q["halting_veh"], strict=True)}


def crossing_signal(
    site: Site, sig: pd.DataFrame, link: int, t: float
) -> tuple[str, float | None, float | None]:
    """(signal, green remaining, next green in) of a crossing link from the green
    intervals of signal.parquet. `clearance`: the crossing's green just ended and the
    junction is in the intergreen of the stage that gave it (the program phase whose
    green links match the logged ones)."""
    g = sig[sig["link"] == link]
    cur = g[(g["begin_s"] <= t) & (g["end_s"] > t)]
    nxt = g.loc[g["begin_s"] > t, "begin_s"]
    next_in = float(nxt.min() - t) if len(nxt) else None
    if not cur.empty:
        return "green", float(cur["end_s"].max() - t), next_in
    on = sig[(sig["begin_s"] <= t) & (sig["end_s"] > t)]
    stage, clearing = site.phase_of_green(frozenset(int(x) for x in on["link"]))
    if clearing and stage is not None and site.link_green_in_stage(link, stage):
        return "clearance", None, next_in
    return "red", None, next_in


def open_requests(events: pd.DataFrame | None, t: float) -> dict[str, int | None]:
    """The preemption requests open at t (a request at or before t without a later
    served / timeout): vehicle id -> the TLS link it asked for (None when not logged)."""
    if events is None or events.empty or not {"source", "rule", "detail", "t"} <= set(events):
        return {}
    pe = events[(events["source"] == "preemption") & (events["t"] <= t)]
    if pe.empty:
        return {}
    detail = pe["detail"].astype(str)
    pe = pe.assign(vid=detail.str.extract(VID_RE)[0], link=detail.str.extract(LINK_RE)[0])
    last = pe.groupby("vid", sort=False).tail(1)
    last = last[last["rule"] == "request"]
    return {
        str(v): (int(k) if isinstance(k, str) else None)
        for v, k in zip(last["vid"], last["link"], strict=True)
    }


def open_preemption(events: pd.DataFrame | None, t: float) -> bool:
    """A preemption request at or before t without a later served / timeout."""
    return bool(open_requests(events, t))


def links_green(sig: pd.DataFrame | None, links: tuple[int, ...], t: float) -> bool | None:
    """True when one of `links` is green at t in the signal log, False when none is,
    None without a signal log or without links."""
    if not links or sig is None or sig.empty or not {"link", "begin_s", "end_s"} <= set(sig):
        return None
    on = sig[sig["link"].isin(list(links)) & (sig["begin_s"] <= t) & (sig["end_s"] > t)]
    return not on.empty


def tsp_active(events: pd.DataFrame | None, line: str, t: float) -> bool:
    """A `tsp_extend` of this line whose window [t_event, t_event + allow] holds t."""
    if events is None or events.empty or "rule" not in events.columns:
        return False
    e = events[(events["rule"] == "tsp_extend") & (events["t"] <= t)]
    for te, detail in zip(e["t"], e["detail"].astype(str), strict=True):
        m = TSP_RE.search(detail)
        if m and m[1] == line and t <= float(te) + float(m[2]):
            return True
    return False


def reference_row(ref: pd.DataFrame | None, trajet: str, hour: int, day_type: str | None):
    """The reference row of (trajet, hour, day type), else the trajet's fallback row
    (hour -1, day type "*"), else None."""
    if ref is None or ref.empty:
        return None
    r = ref[ref["route"] == trajet]
    hit = r[(r["hour"] == hour) & (r["day_type"] == day_type)]
    if hit.empty:
        hit = r[(r["hour"] == -1) & (r["day_type"] == "*")]
    return None if hit.empty else hit.iloc[0]


def served_reference(row) -> tuple[float, float, float] | None:
    """Whole-second (low, median, high) of a reference row: p10 floored, mean rounded,
    p90 ceiled, ordered and >= 0 (the served form of the model's bounds, D48)."""
    mean, p10, p90 = _f(row["mean_s"], None), _f(row["p10_s"], None), _f(row["p90_s"], None)
    if mean is None or p10 is None or p90 is None:
        return None
    med = max(0.0, float(round(mean)))
    low = min(max(0.0, float(math.floor(p10))), med)
    high = max(float(math.ceil(p90)), med)
    return low, med, high


def kpi_table(ctx: Context, scenario: str, user: str | None, family: str) -> KpiTableResponse:
    store, now = ctx.store, wall_now()
    labels = store.labels()
    fam = labels.get(family)
    if labels and fam is None:
        raise ApiError("bad_request", f"unknown family {family!r}: {', '.join(labels)}")
    fam = fam or {}
    if scenario in fam:
        name, label = Path(fam[scenario]).stem, scenario
    else:
        name = scenario
        label = next((lab for lab, cfg in fam.items() if Path(cfg).stem == scenario), None)
    runs = [r for r in store.runs().values() if r.scenario == name]
    if label is None and not runs:
        raise ApiError("bad_request", f"unknown scenario {scenario!r} in family {family!r}")
    df, mtime = store.kpis(family, name)
    source, notice, status = "evaluation", None, "ok"
    if df is None or df.empty:
        df, mtime = store.run_kpis(name), None
        source, status = "runs", "degraded"
        notice = "évaluation absente : moyennes des runs, sans test ni comparaison à S0"
    base = {"scenario": name, "label": label, "family": family}
    if df is None or df.empty:
        env = ctx.envelope("kpi_table", timestamp=now, status="unavailable")
        env["notice"] = f"aucun résultat pour {scenario}"
        return KpiTableResponse(**env, **base)
    if user:
        df = df[df["user_type"] == user]
        if df.empty:
            env = ctx.envelope("kpi_table", timestamp=now, status="unavailable")
            env["notice"] = f"aucun KPI pour l'usager {user!r}"
            return KpiTableResponse(**env, **base, source=source)
    g = (
        df.groupby(["user_type", "kpi"], sort=True)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    eq_rows: dict[tuple[str, str], Mapping] = {}
    eq = store.equity(family) if source == "evaluation" and label else None
    if eq is not None and {"scenario", "user_type", "kpi"} <= set(eq.columns):
        e = eq[eq["scenario"] == label]
        recs = e.to_dict("records")
        eq_rows = {(r["user_type"], r["kpi"]): r for r in recs}
    rows = []
    for rec in g.itertuples(index=False):
        q = eq_rows.get((rec.user_type, rec.kpi), {})
        worse = q.get("worse_flag")
        rows.append(
            KpiRow(
                user=str(rec.user_type),
                kpi=str(rec.kpi),
                mean=_f(rec.mean, 3),
                std=_f(rec.std, 3),
                n=int(rec.count),
                rel_change_vs_s0_pct=_f(q.get("rel_change_pct"), 2),
                p_value=_f(q.get("p_value"), 4),
                worse_flag=None if worse is None or pd.isna(worse) else bool(worse),
            )
        )
    if mtime is not None:
        ts = datetime.fromtimestamp(mtime, PARIS)
    else:
        started = [r.started_at for r in runs if r.started_at is not None]
        ts = max(started).astimezone(PARIS) if started else now
    env = ctx.envelope(
        "kpi_table",
        timestamp=ts,
        age=_f((now - ts).total_seconds()),
        status=status,
        notice=notice,
    )
    return KpiTableResponse(**env, **base, source=source, rows=rows)


def eta_answer(
    ctx: Context,
    trajet: str,
    t: str | None,
    run_id: str | None,
    deadline: str | None,
    student: bool,
) -> EtaResponse:
    from traffic_twin.ml.advice import departure_advice

    spec = ctx.eta_spec
    if trajet not in spec.trajets:
        raise ApiError("unknown_trajet", f"trajets: {', '.join(spec.trajets)}")
    rest = 0.0
    if student:
        if trajet not in spec.rest_s:
            raise ApiError("bad_request", f"student routes: {', '.join(spec.rest_s)}")
        rest = spec.rest_s[trajet]
    run, tq = ctx.run_and_t(run_id, t)
    now = ctx.instant(run, tq)
    dl = parse_deadline(deadline, now)
    store = ctx.store
    if other := ctx.other_site(run):
        return EtaResponse(
            **ctx.unavailable("eta", run, tq, other), trajet=trajet, rest_of_trip_s=rest
        )

    queue_beyond = None
    snap = store.snapshot_at(run.run_id, tq)
    col = f"pending_{trajet.split('_')[0]}"
    if snap is not None and _f(snap.get(col)) is not None:
        queue_beyond = float(snap[col]) > 0

    values = model_version = None
    approximate, status, age, ts, notice = False, "ok", None, now, None
    pred = store.eta_at(run.run_id, trajet, tq) if run.has_eta else None
    if pred is not None:
        vals = [_f(pred.get(k), None) for k in ("eta_low_s", "eta_median_s", "eta_high_s")]
        if None not in vals:
            values = tuple(vals)
            model_version = str(pred.get("model_version"))
            age = age_s(tq, float(pred["t"]))
            ts = ctx.instant(run, float(pred["t"]))
    if values is None:
        if run.controller not in spec.controllers:
            reason = f"contrôleur {run.controller} non vu par le modèle"
        elif run.day_type not in spec.day_types:
            reason = f"type de jour {run.day_type} non vu par le modèle"
        elif not run.has_eta:
            reason = "pas de prévision pour ce run"
        else:
            reason = "pas de prévision à cet instant"
        ref, version = store.reference()
        hour = int((run.begin_s + tq) // 3600) % 24
        row = reference_row(ref, trajet, hour, run.day_type)
        served = served_reference(row) if row is not None else None
        if served is not None:
            values = served
            model_version = str(row.get("model_version") or version)
            approximate, status, notice = True, "degraded", f"{APPROX} ({reason})"
        else:
            env = ctx.envelope("eta", timestamp=now, run=run, t=tq, status="unavailable")
            env["notice"] = f"aucun modèle d'ETA disponible ({reason})"
            return EtaResponse(
                **env, trajet=trajet, rest_of_trip_s=rest, queue_beyond_zone=queue_beyond
            )
    low, median, high = values
    advice = None
    if dl is not None:
        try:
            a = departure_advice(now, median, high, dl, rest, approximate)
            advice = EtaAdvice(
                code=str(a.advice.value),
                message=a.message,
                arrival_median=a.arrival_median,
                arrival_high=a.arrival_high,
                margin_s=round(a.margin_s, 1),
            )
        except ValueError as e:
            log.warning("no departure advice: %s", e)
    env = ctx.envelope("eta", timestamp=ts, age=age, run=run, t=tq, status=status)
    if notice is not None:
        env["notice"] = notice
    return EtaResponse(
        **env,
        trajet=trajet,
        eta_median_s=median,
        eta_low_s=low,
        eta_high_s=high,
        approximate=approximate,
        model_version=model_version,
        queue_beyond_zone=queue_beyond,
        rest_of_trip_s=rest,
        advice=advice,
    )


# -- app -----------------------------------------------------------------------------------
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    ctx = Context(settings)
    app = FastAPI(
        title="Urban Traffic Twin API",
        version=API_VERSION,
        description="Pitié-Salpêtrière digital twin: simulated data (spec 9.2, 9.3).",
    )
    app.state.ctx = ctx
    # ENF-07: no accounts, no cookies; read-only GET from any origin
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_credentials=False
    )

    @app.middleware("http")
    async def never_500(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception:  # ENF-03: 503 with an envelope, never a traceback to the client
            log.exception("unexpected error on %s", request.url.path)
            return error_response(
                ctx, request, 503, "internal", "données temporairement indisponibles"
            )

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return error_response(ctx, request, exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError):
        parts = []
        for err in exc.errors()[:3]:
            where = ".".join(str(x) for x in err.get("loc", ()) if x not in ("query", "path"))
            parts.append(f"{where}: {err.get('msg', 'invalid')}")
        return error_response(ctx, request, 404, "bad_request", "; ".join(parts) or "bad request")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "bad_request")
        return error_response(ctx, request, exc.status_code, code, str(exc.detail))

    # -- /health -------------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    def health():
        runs = ctx.store.runs()
        default = ctx.store.default_run()
        status, notice = "ok", None
        if not runs:
            status, notice = "degraded", "aucun résultat de simulation trouvé"
        env = ctx.envelope("health", timestamp=wall_now(), status=status, notice=notice)
        return HealthResponse(
            **env,
            runs_indexed=len(runs),
            eta_model_version=ctx.store.eta_model_version(),
            default_run=run_ref(default) if default else None,
        )

    # -- /scenarios ----------------------------------------------------------------------
    @app.get("/scenarios", response_model=ScenarioListResponse)
    def scenarios():
        store, now = ctx.store, wall_now()
        names: dict[str, tuple[str, str]] = {}  # scenario -> (family, label), first family wins
        for fam, labels in store.labels().items():
            for label, cfg in labels.items():
                names.setdefault(Path(cfg).stem, (fam, label))
        groups: dict[str, ScenarioGroup] = {}
        runs = sorted(store.runs().values(), key=lambda r: (r.scenario, r.seed, str(r.dir)))
        for r in runs:
            fam, label = names.get(r.scenario, (None, None))
            g = groups.setdefault(
                r.scenario,
                ScenarioGroup(
                    scenario=r.scenario, label=label, family=fam, controller=r.controller
                ),
            )
            try:
                d = r.dir.relative_to(settings.repo_root).as_posix()
            except ValueError:
                d = r.dir.as_posix()
            g.runs.append(
                ScenarioRun(
                    run_id=r.run_id, seed=r.seed, current=r.current, has_eta=r.has_eta, dir=d
                )
            )
        started = [r.started_at for r in runs if r.started_at is not None]
        ts = max(started).astimezone(PARIS) if started else now
        status, notice = ("ok", None) if groups else ("degraded", "aucun résultat de simulation")
        env = ctx.envelope(
            "scenario_list",
            timestamp=ts,
            age=_f((now - ts).total_seconds()) if started else None,
            status=status,
            notice=notice,
        )
        return ScenarioListResponse(**env, scenarios=list(groups.values()))

    # -- /kpi ----------------------------------------------------------------------------
    @app.get("/kpi", response_model=KpiTableResponse)
    def kpi(
        scenario: str = Query(..., description="evaluation label (S0) or scenario name"),
        user: str | None = Query(None, description="car, bus, pedestrian, ambulance, all"),
        family: str = Query("main"),
    ):
        return kpi_table(ctx, scenario, user, family)

    # -- /replay -------------------------------------------------------------------------
    @app.get("/replay/{run_id}", response_model=ReplayStateResponse)
    def replay(run_id: str, t: str | None = None, positions: bool = False):
        run, tq = ctx.run_and_t(run_id, t)
        tick, rows = ctx.store.state_at(run.run_id, tq)
        if tick is None:
            env = ctx.envelope(
                "replay_state", timestamp=ctx.instant(run, tq), run=run, t=tq, status="unavailable"
            )
            env["notice"] = "état de la simulation non disponible à cet instant"
            return ReplayStateResponse(**env)
        inter = rows[rows["entity"] == "intersection"]
        phase = None if inter.empty else _f(inter["phase"].iloc[0], None)
        phase = None if phase is None else int(phase)
        stage, stage_name, clearing = ctx.stage_of(run, phase)
        others = rows[rows["entity"] != "intersection"]
        counts = dict.fromkeys(USERS, 0)
        counts.update({str(k): int(v) for k, v in others["entity"].value_counts().items()})
        resp = ReplayStateResponse(
            **ctx.envelope(
                "replay_state",
                timestamp=ctx.instant(run, tick),
                age=age_s(tq, tick),
                run=run,
                t=tq,
            ),
            phase=phase,
            stage=stage,
            stage_name=stage_name,
            in_clearance=clearing,
            queue_total=None if inter.empty else _f(inter["queue"].iloc[0]),
            queues=queue_by_approach(ctx, run, tq),
            counts=counts,
        )
        if positions:
            resp.positions = [
                Position(entity=str(e), id=str(i), x=_f(x, 2), y=_f(y, 2), speed=_f(v, 2))
                for e, i, x, y, v in zip(
                    others["entity"],
                    others["id"],
                    others["x"],
                    others["y"],
                    others["speed"],
                    strict=True,
                )  # fmt: skip
            ]
        return resp

    # -- /crossing -----------------------------------------------------------------------
    @app.get("/crossing/{crossing_id}", response_model=CrossingStatusResponse)
    def crossing(
        crossing_id: str,
        speed: str = Query("normal", description="normal | slow | reduced | m/s"),
        t: str | None = None,
        run_id: str | None = None,
    ):
        site = ctx.site
        c = site.crossings.get(crossing_id.strip().upper())
        if c is None:
            raise ApiError("unknown_crossing", f"crossings: {', '.join(site.crossings)}")
        v = settings.walking_speeds_mps.get(speed)
        if v is None:
            try:
                v = float(speed)
            except ValueError as e:
                names = " | ".join(settings.walking_speeds_mps)
                raise ApiError("bad_speed", f"speed: {names} or m/s") from e
            lo, hi = SPEED_RANGE_MPS
            if not lo <= v <= hi:
                raise ApiError("bad_speed", f"speed must be within {lo}-{hi} m/s")
        run, tq = ctx.run_and_t(run_id, t)
        walk = None if c.length_m is None else round(c.length_m / v, 1)
        base = {"crossing": c.id, "speed_mps": v, "length_m": c.length_m, "walk_time_s": walk}
        if other := ctx.other_site(run):
            return CrossingStatusResponse(
                **ctx.unavailable("crossing_status", run, tq, other), **base
            )
        sig = ctx.store.signal(run.run_id)
        if sig is None or sig.empty or not {"link", "begin_s", "end_s"} <= set(sig.columns):
            return crossing_from_state(ctx, run, tq, c.link, base)
        signal, remaining, next_in = crossing_signal(site, sig, c.link, tq)
        can = walk is not None and remaining is not None and walk <= remaining
        last = float(sig["end_s"].max())
        env = ctx.envelope(
            "crossing_status", timestamp=ctx.instant(run, tq), age=age_s(tq, last), run=run, t=tq
        )
        return CrossingStatusResponse(
            **env,
            **base,
            signal=signal,
            green_remaining_s=_f(remaining),
            remaining_basis="program" if run.controller == "fixed" else "replay",
            next_green_in_s=_f(next_in),
            can_cross=can if walk is not None else None,
        )

    # -- /bus ----------------------------------------------------------------------------
    @app.get("/bus/{bus_id}/status", response_model=BusStatusResponse)
    def bus(bus_id: str, t: str | None = None, run_id: str | None = None):
        return bus_status(ctx, bus_id, t, run_id)

    # -- /ambulance ----------------------------------------------------------------------
    @app.get("/ambulance/next-intersections", response_model=AmbulanceRouteResponse)
    def ambulance(t: str | None = None, run_id: str | None = None):
        return ambulance_route(ctx, t, run_id)

    # -- /eta ----------------------------------------------------------------------------
    @app.get("/eta", response_model=EtaResponse)
    def eta(
        trajet: str = Query(..., description="N_S, N_W, S_N, S_W, W_N or W_S"),
        t: str | None = None,
        run_id: str | None = None,
        deadline: str | None = Query(None, description="HH:MM or ISO datetime"),
        student: bool = Query(False, description="add the student route's rest of trip"),
    ):
        return eta_answer(ctx, trajet, t, run_id, deadline, student)

    # -- /mock (dev) ---------------------------------------------------------------------
    @app.get("/mock", include_in_schema=False, response_class=HTMLResponse)
    def mock():
        if not settings.mock:
            raise ApiError("not_found", "mock screens are off (api.yaml dev.mock)")
        page = resources.files("traffic_twin.app").joinpath("mock.html")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/", include_in_schema=False)
    def root():
        if settings.mock:
            return RedirectResponse("/mock")
        raise ApiError("not_found", "see /docs for the endpoints")

    return app


def crossing_from_state(
    ctx: Context, run: RunInfo, tq: float, link: int, base: dict
) -> CrossingStatusResponse:
    """Degraded crossing status without signal.parquet: the colour from the phase of the
    5 s state log and the program's phase states; remaining time unknown."""
    site = ctx.site
    tick, rows = ctx.store.state_at(run.run_id, tq)
    inter = rows[rows["entity"] == "intersection"] if len(rows) else rows
    phase = None if inter.empty else _f(inter["phase"].iloc[0], None)
    if tick is None or phase is None or int(phase) >= len(site.phases):
        env = ctx.envelope(
            "crossing_status", timestamp=ctx.instant(run, tq), run=run, t=tq, status="unavailable"
        )
        env["notice"] = "état du feu non disponible"
        return CrossingStatusResponse(**env, **base)
    phase = int(phase)
    stage, _, clearing = site.stage_of(phase)
    if link in site.phases[phase].green:
        signal = "green"
    elif clearing and stage is not None and site.link_green_in_stage(link, stage):
        signal = "clearance"
    else:
        signal = "red"
    env = ctx.envelope(
        "crossing_status",
        timestamp=ctx.instant(run, tick),
        age=age_s(tq, tick),
        run=run,
        t=tq,
        status="degraded",
        notice="temps restant inconnu (pas de journal des feux)",
    )
    return CrossingStatusResponse(
        **env, **base, signal=signal, can_cross=None if signal == "green" else False
    )


def bus_status(ctx: Context, bus_id: str, t: str | None, run_id: str | None):
    site = ctx.site
    lines = site.lines_of(bus_id)
    if not lines:
        ids = sorted({b.line for b in site.bus_lines.values()} | set(site.bus_lines))
        raise ApiError("unknown_bus", f"bus ids: {', '.join(ids)}")
    run, tq = ctx.run_and_t(run_id, t)
    line = lines[0].line
    flows = {f"bus_{b.flow}": b for b in lines}
    tsp = ((run.config.get("rules") or {}).get("tsp")) or {}
    prio = (
        run.controller == "rules"
        and bool(tsp.get("enabled", False))
        and line in {str(x) for x in tsp.get("lines", [])}
    )
    base = {"bus_id": bus_id, "line": line, "priority_available": prio}
    if other := ctx.other_site(run):
        return BusStatusResponse(**ctx.unavailable("bus_status", run, tq, other), **base)
    now = ctx.instant(run, tq)
    trips = ctx.store.trips(run.run_id)
    if trips is None or not {"flow", "arrival_s", "time_loss_s"} <= set(trips.columns):
        env = ctx.envelope("bus_status", timestamp=now, run=run, t=tq, status="unavailable")
        env["notice"] = "données des bus non disponibles"
        return BusStatusResponse(**env, **base)
    b = trips[trips["flow"].isin(list(flows))]
    arr = b["arrival_s"] - run.begin_s
    done = b.assign(arr_t=arr)[arr <= tq].sort_values("arr_t", kind="stable")  # past only
    if done.empty:
        env = ctx.envelope("bus_status", timestamp=now, run=run, t=tq, status="degraded")
        env["notice"] = "aucun bus de cette ligne n'est encore passé"
        return BusStatusResponse(**env, **base)
    last = done.iloc[-1]
    flow = str(last["flow"])
    same = done[done["flow"] == flow]
    headway = float(same["arr_t"].iloc[-1] - same["arr_t"].iloc[-2]) if len(same) > 1 else None
    delay = (_f(last["time_loss_s"], None) or 0.0) + (_f(last.get("depart_delay_s"), None) or 0.0)
    sched = flows[flow].headway_s
    active = tsp_active(ctx.store.events(run.run_id), line, tq) if prio else None
    msg = f"Ligne {line} : dernier bus {delay:+.0f} s au carrefour"
    if headway is not None:
        msg += f", écart {headway:.0f} s (prévu {sched:.0f} s)"
    if prio:
        msg += ", priorité active" if active else ", priorité disponible"
    arr_t = float(last["arr_t"])
    env = ctx.envelope(
        "bus_status",
        timestamp=ctx.instant(run, arr_t),
        age=age_s(tq, arr_t),
        run=run,
        t=tq,
        limit_s=ctx.settings.bus_headway_factor * sched,
    )
    return BusStatusResponse(
        **env,
        **base,
        flow=flow.removeprefix("bus_"),
        vehicle_id=str(last["id"]) if "id" in last.index else None,
        delay_s=round(delay, 1),
        headway_s=_f(headway),
        scheduled_headway_s=sched,
        priority_active=active,
        message=msg,
    )


def ambulance_route(ctx: Context, t: str | None, run_id: str | None) -> AmbulanceRouteResponse:
    from traffic_twin.config import ConfigError
    from traffic_twin.sim.demand import emergency_trips

    site, settings, store = ctx.site, ctx.settings, ctx.store
    run, tq = ctx.run_and_t(run_id, t)
    if other := ctx.other_site(run):
        return AmbulanceRouteResponse(**ctx.unavailable("ambulance_route", run, tq, other))
    tick, rows = store.state_at(run.run_id, tq)
    try:
        trips = emergency_trips(run.config)
    except (ConfigError, KeyError, TypeError) as e:
        log.warning("emergency trips of %s: %s", run.run_id, e)
        trips = []
    routes = {vid: route for _, vid, route in trips}

    def movement(route):
        return site.movement_of_route(route) if route else None

    vid = dist = passed = speed = None
    amb = rows[rows["entity"] == "ambulance"] if len(rows) else rows
    if not amb.empty:
        cands = []
        for i in amb.index:
            x, y = _f(amb.at[i, "x"], None), _f(amb.at[i, "y"], None)
            d = along = None
            if x is not None and y is not None and site.junction_xy is not None:
                cx, cy = site.junction_xy
                d = math.hypot(x - cx, y - cy)
            mv = movement(routes.get(str(amb.at[i, "id"])))
            sl = site.stop_lines.get(mv[0]) if mv else None
            if sl is not None and x is not None and y is not None:
                along = sl.along(x, y)
            p = None if along is None else along > 0
            # the nearest ambulance still before its stop line, else the nearest one
            cands.append((p is True, math.inf if d is None else d, i, d, p))
        _, _, i, d, passed = min(cands, key=lambda c: (c[0], c[1]))
        vid = str(amb.at[i, "id"])
        dist = _f(d, 0) if d is not None else None
        speed = _f(amb.at[i, "speed"], None) if "speed" in amb.columns else None
    if vid in routes:
        route = routes[vid]
    else:  # no ambulance on the network: the next one, else the last one
        upcoming = [r for dep, _, r in trips if dep >= tq]
        route = upcoming[0] if upcoming else (trips[-1][2] if trips else None)
    mv = movement(route)
    approach = site.approach_of_edge(mv[0] if mv else route[0]) if route else None

    status, reason = "unknown", "pas de données de file"
    if tick is not None and passed:
        status, reason = "clear", "carrefour franchi"
    elif tick is not None:
        reqs = open_requests(store.events(run.run_id), tq)
        requested = (vid in reqs) if vid is not None else bool(reqs)
        req_link = reqs.get(vid) if vid is not None else None
        links = (req_link,) if req_link is not None else ()
        if not links and mv and mv[1]:
            links = site.movement_links.get(mv, ())
        green = links_green(store.signal(run.run_id), links, tq)
        if requested and green:
            status, reason = "green_wave", "préemption en cours, feu vert"
        else:
            notes = []
            if requested:
                notes.append(
                    "préemption demandée, feu pas encore vert"
                    if green is False
                    else "préemption demandée, état du feu inconnu"
                )
            queues = queue_by_approach(ctx, run, tq)
            if approach is not None and approach in queues:
                h = queues[approach]
                status = "blocked" if h >= settings.ambulance_blocked_halting else "clear"
                notes.append(f"{h} véhicules arrêtés sur l'approche {approach}")
            if green is False and speed is not None and speed < HALT_SPEED_MPS:
                status = "blocked"
                notes.insert(0, "ambulance arrêtée au feu rouge")
            if notes:
                reason = " ; ".join(notes)
    out = []
    for j in settings.corridor:
        if j["modelled"]:
            out.append(
                Intersection(
                    id=j["id"],
                    name=j["name"],
                    modelled=True,
                    status=status,
                    distance_m=dist,
                    passed=passed,
                    reason=reason,
                )  # fmt: skip
            )
        else:
            out.append(
                Intersection(
                    id=j["id"],
                    name=j["name"],
                    modelled=False,
                    status="unknown",
                    reason="non modélisé dans le jumeau numérique",
                )  # fmt: skip
            )
    modelled = sum(j["modelled"] for j in settings.corridor)
    notice = CORRIDOR_NOTICE.format(n=len(settings.corridor))
    if modelled != 1:
        notice = f"{modelled} carrefour(s) modélisé(s) sur {len(settings.corridor)}"
    if tick is None:
        env = ctx.envelope(
            "ambulance_route", timestamp=ctx.instant(run, tq), run=run, t=tq, status="unavailable"
        )
        env["notice"] = "état de la simulation non disponible à cet instant"
    else:
        env = ctx.envelope(
            "ambulance_route",
            timestamp=ctx.instant(run, tick),
            age=age_s(tq, tick),
            run=run,
            t=tq,
            status="degraded",
            notice=notice,
        )
    return AmbulanceRouteResponse(**env, vehicle_id=vid, approach=approach, intersections=out)


app = create_app()  # settings from configs/api.yaml; the store is built on the first request

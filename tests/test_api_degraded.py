"""API v1 degraded modes and error policy (Part 5; ENF-03, D53): missing or corrupt data
gives 200 with `status: degraded` or `unavailable`, an unknown id or a bad parameter a
404 envelope with a code, anything unexpected a 503 envelope without a traceback.
Never a 500."""

import json
import os
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
# pytest.importorskip("duckdb")

from fastapi.testclient import TestClient  # noqa: E402

from traffic_twin.api import clock  # noqa: E402
from traffic_twin.api.db import LocalStore  # noqa: E402
from traffic_twin.api.fixtures import make_fake_eta, make_results, make_run  # noqa: E402
from traffic_twin.api.main import create_app  # noqa: E402
from traffic_twin.api.schemas import ApiError  # noqa: E402
from traffic_twin.api.settings import ENV_RESULTS_DIRS, load_settings  # noqa: E402
from traffic_twin.config import ConfigError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TIME_ENDPOINTS = (
    "/replay/default?t=310",
    "/crossing/S?t=310",
    "/bus/91/status?t=310",
    "/ambulance/next-intersections?t=310",
    "/eta?trajet=N_S&t=310",
)


def make_client(results: Path, models: Path | None = None, **overrides) -> TestClient:
    settings = replace(
        load_settings(repo_root=REPO_ROOT, environ={}),
        results_dirs=(results,),
        evaluation_dir=results / "evaluation",
        eta_models_dir=models or results.parent / "no_models",
        default_run=(results / "pitie_rules_seed4000",),
        replay_date=date(2026, 10, 14),
        **overrides,
    )
    return TestClient(create_app(settings), raise_server_exceptions=False)


@pytest.fixture(scope="module")
def shared_client(tmp_path_factory) -> TestClient:
    """A client on synthetic results that no test changes."""
    return make_client(make_results(tmp_path_factory.mktemp("api") / "results", n_runs=3))


def corrupt(path: Path) -> None:
    path.write_bytes(b"this is not a parquet file")


# -- ETA cascade -----------------------------------------------------------------------
def test_eta_cascade_predictions_then_reference_then_unavailable(api_results, tmp_path):
    run = api_results / "pitie_rules_seed4000"
    models = tmp_path / "models"
    make_fake_eta(run)
    make_fake_eta(models)
    ok = make_client(api_results, models).get("/eta?trajet=N_S&t=08:00").json()
    assert ok["status"] == "ok" and ok["approximate"] is False and ok["model_version"] == "fake-0"

    (run / "eta_prediction.parquet").unlink()
    ref = make_client(api_results, models).get("/eta?trajet=N_S&t=08:00&deadline=09:00").json()
    assert ref["status"] == "degraded" and ref["approximate"] is True
    assert ref["notice"].startswith("estimation approximative")
    # fake reference N_S, 8 h: mean 60, p10 42, p90 90 -> served whole seconds
    assert (ref["eta_low_s"], ref["eta_median_s"], ref["eta_high_s"]) == (42.0, 60.0, 90.0)
    assert ref["model_version"] == "fake-ref-0" and ref["data_age_s"] is None
    assert ref["advice"]["message"].endswith("(estimation approximative)")

    corrupt(models / "fake-ref-0" / "reference.parquet")
    gone = make_client(api_results, models).get("/eta?trajet=N_S&t=08:00").json()
    assert gone["status"] == "unavailable" and gone["eta_median_s"] is None
    assert gone["advice"] is None and gone["notice"]


def test_eta_corrupt_predictions_fall_back_to_the_reference(api_results, tmp_path):
    run = api_results / "pitie_rules_seed4000"
    make_fake_eta(run)
    make_fake_eta(tmp_path / "models")
    corrupt(run / "eta_prediction.parquet")
    b = make_client(api_results, tmp_path / "models").get("/eta?trajet=W_N&t=07:45").json()
    assert b["status"] == "degraded" and b["approximate"] is True


def test_eta_unseen_controller_and_day_type(api_results, tmp_path):
    make_run(api_results / "pitie_rl_seed4000", controller="rl")
    make_run(
        api_results / "pitie_fixed_seed4001", controller="fixed", seed=4001, day_type="saturday"
    )
    make_fake_eta(tmp_path / "models")
    c = make_client(api_results, tmp_path / "models")
    groups = {g["scenario"]: g["runs"] for g in c.get("/scenarios").json()["scenarios"]}
    rl = groups["pitie_rl"][0]["run_id"]
    b = c.get(f"/eta?trajet=N_S&t=08:00&run_id={rl}").json()
    assert b["status"] == "degraded" and b["approximate"] is True
    assert "contrôleur rl non vu par le modèle" in b["notice"]
    sat = next(r["run_id"] for r in groups["pitie_fixed"] if r["seed"] == 4001)
    d = c.get(f"/eta?trajet=N_S&t=08:00&run_id={sat}").json()
    assert "type de jour saturday" in d["notice"]
    # no (N_S, 8, saturday) row: the trajet's fallback row (mean 55, p10 38, p90 85)
    assert (d["eta_low_s"], d["eta_median_s"], d["eta_high_s"]) == (38.0, 55.0, 85.0)


def test_run_of_another_network(api_results):
    """A run of another net (the grid baseline) gets no Pitié answers; its replay uses
    its own stages."""
    meta_path = api_results / "pitie_fixed_seed4000" / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["scenario"] = "baseline_fixed"
    meta["config"]["network"]["net_file"] = "sumo/network/intersection.net.xml"
    meta["config"]["network"]["stages"] = [
        {"name": "EW", "green_phase": 0, "clearance_phases": [1, 2, 3]},
        {"name": "NS", "green_phase": 4, "clearance_phases": [5, 6]},
    ]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    c = make_client(api_results)
    rid = meta["run_id"]
    for url in (
        f"/crossing/S?t=20&run_id={rid}", f"/bus/91/status?t=20&run_id={rid}",
        f"/ambulance/next-intersections?t=20&run_id={rid}", f"/eta?trajet=N_S&t=20&run_id={rid}",
    ):  # fmt: skip
        b = c.get(url).json()
        assert b["status"] == "unavailable" and "autre réseau" in b["notice"], url
    r = c.get(f"/replay/{rid}?t=20").json()
    assert r["status"] == "ok" and r["stage_name"] == "EW" and r["phase"] == 0


# -- corrupt and missing files ---------------------------------------------------------
def test_corrupt_files_never_give_5xx(api_results):
    run = api_results / "pitie_rules_seed4000"
    make_fake_eta(run)
    for name in (
        "state_log.parquet", "signal.parquet", "trip.parquet", "events.parquet",
        "queue.parquet", "eta_prediction.parquet", "eta_snapshots.parquet",
    ):  # fmt: skip
        corrupt(run / name)
    (api_results / "evaluation" / "main" / "kpis.csv").write_text("garbage\n\x00\x01", "utf-8")
    c = make_client(api_results)
    for url in (*TIME_ENDPOINTS, "/kpi?scenario=S2", "/scenarios", "/health"):
        r = c.get(url)
        assert r.status_code == 200, (url, r.text)
        assert r.json()["data_source"] == "simulation"
    assert c.get("/replay/default?t=310").json()["status"] == "unavailable"
    assert c.get("/crossing/S?t=310").json()["status"] == "unavailable"
    assert c.get("/bus/91/status?t=310").json()["status"] == "unavailable"
    assert c.get("/ambulance/next-intersections?t=310").json()["status"] == "unavailable"


def test_corrupt_signal_log_degrades_the_crossing(api_results):
    corrupt(api_results / "pitie_rules_seed4000" / "signal.parquet")
    b = make_client(api_results).get("/crossing/NB?t=20").json()
    assert b["status"] == "degraded" and b["signal"] == "green" and b["green_remaining_s"] is None


def test_unreadable_run_json_is_skipped(api_results):
    (api_results / "pitie_fixed_seed4000" / "run.json").write_text("{not json", "utf-8")
    b = make_client(api_results).get("/health").json()
    assert b["status"] == "ok" and b["runs_indexed"] == 2


def test_empty_results_dir(tmp_path):
    c = make_client(tmp_path / "nothing")
    h = c.get("/health").json()
    assert h["status"] == "degraded" and h["runs_indexed"] == 0 and h["default_run"] is None
    assert c.get("/scenarios").json()["status"] == "degraded"
    k = c.get("/kpi?scenario=S0")
    assert k.status_code == 200 and k.json()["status"] == "unavailable"
    for url in TIME_ENDPOINTS:
        r = c.get(url)
        assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_run", url


# -- 404 codes, 422 remap, 503 ---------------------------------------------------------
@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("/replay/nope", "unknown_run"),
        ("/crossing/S?run_id=nope", "unknown_run"),
        ("/crossing/ZZ", "unknown_crossing"),
        ("/bus/999/status", "unknown_bus"),
        ("/eta?trajet=X_Y", "unknown_trajet"),
        ("/crossing/S?speed=fast", "bad_speed"),
        ("/crossing/S?speed=12", "bad_speed"),
        ("/replay/default?t=abc", "bad_t"),
        ("/replay/default?t=25:00", "bad_t"),
        ("/replay/default?t=-5", "bad_t"),
        ("/replay/default?t=07:00", "bad_t"),  # before the 07:30 start
        ("/replay/default?t=nan", "bad_t"),
        ("/eta?trajet=N_S&deadline=tomorrow", "bad_deadline"),
        ("/eta?trajet=N_W&student=true", "bad_request"),  # not a student route
        ("/kpi?scenario=S9", "bad_request"),
        ("/kpi?scenario=S0&family=nope", "bad_request"),
        ("/kpi", "bad_request"),  # 422 remapped
        ("/eta", "bad_request"),  # 422 remapped
        ("/replay/default?positions=maybe", "bad_request"),  # 422 remapped
        ("/no/such/path", "not_found"),
    ],
)
def test_404_codes_with_an_envelope(shared_client, url, code):
    r = shared_client.get(url)
    assert r.status_code == 404, r.text
    body = r.json()
    assert body["error"]["code"] == code and body["error"]["message"]
    assert body["status"] == "unavailable" and body["data_source"] == "simulation"
    assert body["calibrated_on"] == "2026-10-08" and "timestamp" in body


@pytest.mark.parametrize("t", ["1e12", "1e300", "99999999"])
@pytest.mark.parametrize("endpoint", [u.split("?")[0] for u in TIME_ENDPOINTS])
def test_absurd_replay_time_is_bad_t_not_503(shared_client, endpoint, t):
    """A t far past the end of the run (beyond clock.MAX_PAST_END_S) is a bad parameter
    on every time-dependent endpoint, not an internal error (it used to overflow the
    replay instant and give 503 internal)."""
    sep = "&" if endpoint == "/eta" else "?"
    url = f"{endpoint}{'?trajet=N_S' if endpoint == '/eta' else ''}{sep}t={t}"
    r = shared_client.get(url)
    assert r.status_code == 404, (url, r.text)
    assert r.json()["error"]["code"] == "bad_t"


def test_time_past_the_end_within_a_day_is_stale_not_an_error(shared_client):
    r = shared_client.get("/replay/default?t=5000")  # the fixture runs last 1800 s
    assert r.status_code == 200 and r.json()["stale"] is True


def test_method_not_allowed_has_an_envelope(shared_client):
    r = shared_client.post("/health")
    assert r.status_code == 405 and r.json()["error"]["code"] == "method_not_allowed"


def test_unexpected_error_is_503_without_traceback(api_results, monkeypatch):
    def boom(*args, **kwargs):
        raise ZeroDivisionError("division by zero in a secret place")

    monkeypatch.setattr(LocalStore, "state_at", boom)
    r = make_client(api_results).get("/replay/default?t=10")
    assert r.status_code == 503
    assert "Traceback" not in r.text and "secret" not in r.text
    body = r.json()
    assert body["error"]["code"] == "internal" and body["type"] == "replay_state"
    assert body["data_source"] == "simulation"


# -- settings and clock ----------------------------------------------------------------
def test_settings_env_overrides(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    env = {ENV_RESULTS_DIRS: os.pathsep.join([str(a), str(b)])}
    s = load_settings(repo_root=REPO_ROOT, environ=env)
    assert s.results_dirs == (a, b) and s.evaluation_dir == a / "evaluation"
    base = load_settings(repo_root=REPO_ROOT, environ={})
    assert base.results_dirs == (Path("data/results"), Path("data/demo"))
    assert base.walking_speeds_mps == {"normal": 1.2, "slow": 1.0, "reduced": 0.8}
    assert base.bus_headway_factor == 1.5 and base.mock is True
    with pytest.raises(ConfigError):
        load_settings(repo_root=REPO_ROOT, environ={"TT_API_CONFIG": str(tmp_path / "x.yaml")})
    bad = tmp_path / "bad.yaml"
    raw = (REPO_ROOT / "configs" / "api.yaml").read_text(encoding="utf-8")
    bad.write_text(raw.replace("mode: fixed", "mode: sometimes"), encoding="utf-8")
    with pytest.raises(ConfigError, match="clock.mode"):
        load_settings(bad, repo_root=REPO_ROOT, environ={})


def test_clock_parsing():
    kw = {"begin_s": 27000.0, "duration_s": 7200.0, "default_t": "08:15"}
    assert clock.parse_t(None, **kw) == 2700.0
    assert clock.parse_t("08:15:30", **kw) == 2730.0 and clock.parse_t("12.5", **kw) == 12.5
    loop_now = clock.replay_instant(0.0, 9 * 3600 + 40 * 60, date(2026, 10, 14))
    assert clock.parse_t("", **kw, mode="loop", now=loop_now) == (34800.0 - 27000.0) % 7200.0
    assert clock.parse_t(str(7200.0 + clock.MAX_PAST_END_S), **kw) == 7200.0 + 86400.0
    for raw in ("abc", "inf", "-1", "24:00", "07:29:59", "93600.5", "1e12", "1e300"):
        with pytest.raises(ApiError) as e:
            clock.parse_t(raw, **kw)
        assert e.value.code == "bad_t"
    with pytest.raises(ApiError) as e:  # not representable: bad_t, never OverflowError
        clock.replay_instant(27000.0, 1e12, date(2026, 10, 14))
    assert e.value.code == "bad_t"
    now = clock.replay_instant(27000.0, 2700.0, date(2026, 10, 14))
    assert now.isoformat() == "2026-10-14T08:15:00+02:00"
    assert clock.parse_deadline("09:00", now).isoformat() == "2026-10-14T09:00:00+02:00"
    assert clock.parse_deadline("2026-10-14T09:00", now).utcoffset() is not None
    assert clock.parse_deadline(None, now) is None
    with pytest.raises(ApiError) as e:
        clock.parse_deadline("9h", now)
    assert e.value.code == "bad_deadline"

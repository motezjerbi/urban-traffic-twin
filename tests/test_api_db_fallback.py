"""The pandas fallback of `api.db.LocalStore` (Motez Jerbi's addition, kept; D120): with
`duckdb` absent (`db.duckdb = None`), every endpoint answers exactly as with DuckDB on
the same synthetic run folders (`api.fixtures`), so the API works without DuckDB."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("duckdb")  # the reference answers

from fastapi.testclient import TestClient  # noqa: E402

from traffic_twin.api import db  # noqa: E402
from traffic_twin.api.fixtures import make_fake_eta, make_results  # noqa: E402
from traffic_twin.api.main import create_app  # noqa: E402
from traffic_twin.api.settings import load_settings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
URLS = (
    "/health", "/scenarios", "/kpi?scenario=S0", "/kpi?scenario=S1", "/kpi?scenario=S2",
    "/kpi?scenario=S3", "/kpi?scenario=S2&user=car", "/kpi?scenario=S2&user=all",
    "/kpi?scenario=pitie_fixed", "/kpi?scenario=S2&family=ambulance_probe",
    "/replay/default?t=62", "/replay/default?t=310&positions=true", "/replay/default?t=5000",
    "/crossing/NB?t=20&speed=normal", "/crossing/S?t=10", "/bus/91/status?t=122",
    "/bus/91_E/status?t=600", "/ambulance/next-intersections?t=350",
    "/eta?trajet=N_S&t=600&deadline=08:00&student=true", "/eta?trajet=W_S&t=700",
    "/evaluation", "/compare?a=S1&b=S3", "/series?labels=S0,S3", "/sensitivity",
    "/replay/default/events?from=100&to=600",
)  # fmt: skip
WALL = ("timestamp", "data_age_s")  # wall-clock ages of the aggregates
AGGREGATES = (
    "health", "scenario_list", "kpi_table", "evaluation", "kpi_pair", "kpi_series", "sensitivity",
)  # fmt: skip


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> Path:
    root = make_results(tmp_path_factory.mktemp("fallback") / "results", n_runs=3)
    make_fake_eta(root / "pitie_rules_seed4000")
    make_fake_eta(root.parent / "models")
    return root


def client(results: Path) -> TestClient:
    settings = replace(
        load_settings(repo_root=REPO_ROOT, environ={}),
        results_dirs=(results,),
        evaluation_dir=results / "evaluation",
        eta_models_dir=results.parent / "models",
        default_run=(results / "pitie_rules_seed4000",),
        replay_date=date(2026, 10, 13),
    )
    return TestClient(create_app(settings), raise_server_exceptions=False)


def answers(c: TestClient) -> dict:
    out = {}
    for url in URLS:
        r = c.get(url)
        body = r.json()
        if body.get("type") in AGGREGATES:
            body = {k: v for k, v in body.items() if k not in WALL}
        out[url] = (r.status_code, body)
    return out


def test_pandas_fallback_answers_like_duckdb(results, monkeypatch):
    with_duckdb = answers(client(results))
    monkeypatch.setattr(db, "duckdb", None)
    store = client(results).app.state.ctx.store
    assert store._con is None  # the fallback path is the one exercised
    without = answers(client(results))
    for url in URLS:
        assert with_duckdb[url][0] == 200, (url, with_duckdb[url])
        assert without[url] == with_duckdb[url], url


def test_pandas_fallback_reads_the_evaluation_kpis(results, monkeypatch):
    monkeypatch.setattr(db, "duckdb", None)
    b = client(results).get("/kpi?scenario=S2").json()
    assert b["status"] == "ok" and b["source"] == "evaluation"
    rows = {(r["user"], r["kpi"]): r for r in b["rows"]}
    car = rows[("car", "time_loss_mean")]
    assert car["mean"] == 56.0 and car["rel_change_vs_s0_pct"] == -20.0

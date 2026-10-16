"""API v1 contract tests (Part 5; spec 9.2, 9.3; D53) on synthetic run folders
(`api.fixtures`): CI has no data/results, and these tests never read it.

The fixture runs replay the net's own 123 s program from t = 0 (HOP green 0-36, its
clearance 36-45, S_LEFT 45-51, STM 61-81, ...), so the expected signal answers below
follow from configs/pitie_fixed.yaml and its net file."""

from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("duckdb")

from fastapi.testclient import TestClient  # noqa: E402

from traffic_twin.api.fixtures import make_fake_eta, make_results  # noqa: E402
from traffic_twin.api.main import create_app  # noqa: E402
from traffic_twin.api.settings import load_settings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ENVELOPE = {
    "data_source", "calibrated_on", "timestamp", "data_age_s", "stale", "type", "status",
    "notice", "provisional", "run", "sim_t_s", "error",
}  # fmt: skip
REPLAY_DATE = date(2026, 10, 14)


def make_client(results: Path, models: Path | None = None, **overrides) -> TestClient:
    settings = replace(
        load_settings(repo_root=REPO_ROOT, environ={}),
        results_dirs=(results,),
        evaluation_dir=results / "evaluation",
        eta_models_dir=models or results.parent / "no_models",
        default_run=(results / "pitie_rules_seed4000",),
        replay_date=REPLAY_DATE,
        **overrides,
    )
    return TestClient(create_app(settings), raise_server_exceptions=False)


@pytest.fixture(scope="module")
def shared_results(tmp_path_factory) -> Path:
    """Synthetic results shared by the read-only tests (tests that change files use the
    per-test `api_results`)."""
    return make_results(tmp_path_factory.mktemp("api") / "results", n_runs=3)


@pytest.fixture(scope="module")
def client(shared_results: Path) -> TestClient:
    return make_client(shared_results)


def run_id_of(client: TestClient, scenario: str) -> str:
    groups = client.get("/scenarios").json()["scenarios"]
    return next(g for g in groups if g["scenario"] == scenario)["runs"][0]["run_id"]


def check_envelope(body: dict, kind: str) -> None:
    assert ENVELOPE <= body.keys()
    assert body["data_source"] == "simulation"
    assert body["calibrated_on"] == "2026-10-08"  # last calibration day (versioned yaml)
    assert body["provisional"] is True  # the field file has no measured timing yet
    assert body["type"] == kind
    assert datetime.fromisoformat(body["timestamp"]).utcoffset() is not None


@pytest.mark.parametrize(
    ("url", "kind"),
    [
        ("/health", "health"),
        ("/scenarios", "scenario_list"),
        ("/kpi?scenario=S2", "kpi_table"),
        ("/replay/default?t=100", "replay_state"),
        ("/crossing/S?speed=slow&t=60", "crossing_status"),
        ("/bus/91/status?t=560", "bus_status"),
        ("/ambulance/next-intersections?t=310", "ambulance_route"),
        ("/eta?trajet=N_S&t=08:00&deadline=08:30", "eta"),
    ],
)
def test_every_endpoint_has_the_envelope(client, url, kind):
    r = client.get(url)
    assert r.status_code == 200, r.text
    body = r.json()
    check_envelope(body, kind)
    assert body["error"] is None


def test_errors_carry_the_envelope(client):
    r = client.get("/crossing/ZZ")
    assert r.status_code == 404
    body = r.json()
    check_envelope(body, "crossing_status")
    assert body["status"] == "unavailable"
    assert body["error"]["code"] == "unknown_crossing"


def test_replay_clock_and_age(client):
    b = client.get("/replay/default?t=07:31:02").json()  # 62 s after 07:30: sample at 60 s
    assert b["sim_t_s"] == 62.0 and b["data_age_s"] == 2.0 and not b["stale"]
    assert b["timestamp"] == "2026-10-14T07:31:00+02:00"
    assert b["run"]["scenario"] == "pitie_rules" and b["run"]["seed"] == 4000
    # 60 s: S_LEFT (45-51) is over, its clearance phase 6 (54-61) runs
    assert (b["phase"], b["stage"], b["stage_name"], b["in_clearance"]) == (6, 1, "S_LEFT", True)
    assert b["queues"] == {"N": 2, "S": 4, "W": 3} and b["queue_total"] == 9.0
    assert set(b["counts"]) >= {"car", "bus", "pedestrian", "ambulance"}
    assert b["positions"] is None
    assert client.get("/replay/default?t=62").json()["sim_t_s"] == 62.0
    late = client.get("/replay/default?t=2000").json()  # past the end of a 1800 s run
    assert late["stale"] and late["notice"] and late["data_age_s"] == 205.0


def test_replay_positions(client):
    b = client.get("/replay/default?t=310&positions=true").json()
    ents = {p["entity"] for p in b["positions"]}
    assert {"car", "ambulance"} <= ents and "intersection" not in ents
    assert len(b["positions"]) == sum(b["counts"].values())


def test_crossing_remaining_green_and_speed_verdict(client):
    g = client.get("/crossing/NB?t=20&speed=normal").json()  # NB green 0-36, 7.96 m
    assert g["signal"] == "green" and g["green_remaining_s"] == 16.0 and g["can_cross"]
    assert g["walk_time_s"] == 6.6 and g["length_m"] == 7.96 and g["obstructed"] is None
    assert client.get("/crossing/NB?t=29").json()["can_cross"] is True  # 6.6 s <= 7 s
    slow = client.get("/crossing/NB?t=29&speed=slow").json()  # 8.0 s > 7 s left
    assert slow["speed_mps"] == 1.0 and slow["can_cross"] is False
    num = client.get("/crossing/NB?t=29&speed=1.5").json()  # numeric speed, m/s
    assert num["speed_mps"] == 1.5 and num["can_cross"] is True
    assert client.get("/crossing/NB?t=31&speed=normal").json()["can_cross"] is False


def test_crossing_clearance_and_red(client):
    c = client.get("/crossing/NB?t=37").json()  # HOP's intergreen 36-45
    assert c["signal"] == "clearance" and c["can_cross"] is False
    assert c["next_green_in_s"] == 24.0  # STM green at 61
    red = client.get("/crossing/S?t=10").json()  # S is green from STM (61) to 98
    assert red["signal"] == "red" and red["next_green_in_s"] == 51.0
    assert red["green_remaining_s"] is None and red["can_cross"] is False


def test_remaining_basis_program_for_fixed_replay_for_adaptive(client):
    basis = {
        s: client.get(f"/crossing/S?t=70&run_id={run_id_of(client, s)}").json()["remaining_basis"]
        for s in ("pitie_fixed", "pitie_actuated", "pitie_rules")
    }
    assert basis == {"pitie_fixed": "program", "pitie_actuated": "replay", "pitie_rules": "replay"}


def test_crossing_degraded_without_signal_log(api_results):
    (api_results / "pitie_fixed_seed4000" / "signal.parquet").unlink()
    c = make_client(api_results)
    rid = run_id_of(c, "pitie_fixed")
    b = c.get(f"/crossing/NB?t=20&run_id={rid}").json()  # phase 0 (HOP) at the 20 s sample
    assert b["status"] == "degraded" and b["signal"] == "green"
    assert b["green_remaining_s"] is None and b["remaining_basis"] == "unknown"
    assert b["can_cross"] is None
    assert c.get(f"/crossing/S?t=20&run_id={rid}").json()["can_cross"] is False


def test_bus_uses_only_past_data(client):
    early = client.get("/bus/91/status?t=100").json()  # the first 91 arrives at 120
    assert early["status"] == "degraded" and early["delay_s"] is None
    b = client.get("/bus/91/status?t=130").json()
    assert b["vehicle_id"] == "bus_91_E.0" and b["flow"] == "91_E"  # not a later bus
    assert b["delay_s"] == 30.0 and b["headway_s"] is None and b["data_age_s"] == 10.0
    assert b["delay_basis"] == "junction_time_loss" and b["scheduled_headway_s"] == 360.0
    later = client.get("/bus/91_E/status?t=600").json()  # 91_E.1 arrived at 495
    assert later["vehicle_id"] == "bus_91_E.1" and later["headway_s"] == 375.0
    assert later["message"].startswith("Ligne 91")


def test_bus_priority_active_inside_its_window(client):
    b = client.get("/bus/91/status?t=122").json()  # tsp_extend at 115, allow 10
    assert b["priority_available"] is True and b["priority_active"] is True
    assert client.get("/bus/91/status?t=130").json()["priority_active"] is False
    fixed = run_id_of(client, "pitie_fixed")
    f = client.get(f"/bus/91/status?t=122&run_id={fixed}").json()
    assert f["priority_available"] is False and f["priority_active"] is None


def test_bus_freshness_is_headway_based(client):
    # last 91 arrival 1755 s; limit 1.5 x 360 = 540 s
    ok = client.get("/bus/91/status?t=2055").json()
    assert ok["data_age_s"] == 300.0 and ok["stale"] is False
    old = client.get("/bus/91/status?t=2400").json()
    assert old["data_age_s"] == 645.0 and old["stale"] is True and old["notice"]


def ambulance_c(client, t: float, run_id: str | None = None) -> dict:
    url = f"/ambulance/next-intersections?t={t}" + (f"&run_id={run_id}" if run_id else "")
    b = client.get(url).json()
    return {**b["intersections"][0], "vehicle_id": b["vehicle_id"], "approach": b["approach"]}


def test_ambulance_one_of_three_junctions_modelled(client):
    b = client.get("/ambulance/next-intersections?t=310").json()
    assert b["status"] == "degraded" and "1 carrefour" in b["notice"]
    by_id = {i["id"]: i for i in b["intersections"]}
    assert list(by_id) == ["C", "U1", "U2"]
    assert by_id["C"]["modelled"] and by_id["C"]["passed"] is False
    assert 150.0 < by_id["C"]["distance_m"] < 200.0  # 155 m before the n_in stop line
    assert all(by_id[j]["status"] == "unknown" and not by_id[j]["modelled"] for j in ("U1", "U2"))
    assert all(by_id[j]["passed"] is None for j in ("U1", "U2"))
    assert b["vehicle_id"] == "ambulance_0" and b["approach"] == "N"
    blocked = client.get("/ambulance/next-intersections?t=700").json()  # 12 halting on N
    assert blocked["intersections"][0]["status"] == "blocked" and blocked["vehicle_id"] is None
    clear = client.get("/ambulance/next-intersections?t=200").json()
    assert clear["intersections"][0]["status"] == "clear"


def test_ambulance_green_wave_only_on_its_own_green(client):
    """An open preemption request is not a green wave while the ambulance's own link
    (link 2, n_in -> s_out) is red: fixture request at 301, served at 371, link 2 red
    275-361 and green 361-390 in the rules run."""
    approaching = ambulance_c(client, 310)  # moving, red, 4 halting on N
    assert approaching["status"] == "clear" and approaching["passed"] is False
    assert "préemption demandée, feu pas encore vert" in approaching["reason"]
    waiting = ambulance_c(client, 350)  # stopped 5 m before the stop line, red
    assert waiting["status"] == "blocked" and waiting["passed"] is False
    assert waiting["reason"].startswith("ambulance arrêtée au feu rouge")
    assert "préemption demandée, feu pas encore vert" in waiting["reason"]
    wave = ambulance_c(client, 370)  # request still open, link 2 green, before the line
    assert wave["status"] == "green_wave" and wave["passed"] is False
    assert wave["reason"] == "préemption en cours, feu vert"


def test_ambulance_past_the_junction_is_clear(shared_results):
    """Once past its stop line the junction is behind the ambulance: `clear`, `passed`,
    whatever the queue on its approach (threshold 1 makes every N queue 'blocking')."""
    c = make_client(shared_results, ambulance_blocked_halting=1)
    after = ambulance_c(c, 380)  # 75 m past the n_in stop line
    assert after["vehicle_id"] == "ambulance_0" and after["approach"] == "N"
    assert after["status"] == "clear" and after["passed"] is True
    assert after["reason"] == "carrefour franchi"
    before = ambulance_c(c, 310)  # same threshold, before the line: the queue counts
    assert before["status"] == "blocked" and before["passed"] is False


def test_ambulance_without_preemption_uses_the_net_link(client):
    """A run without a preemption log (fixed) still knows the ambulance's link from the
    net (n_in -> s_out = link 2): stopped at its red is blocked, with no green wave."""
    fixed = run_id_of(client, "pitie_fixed")
    waiting = ambulance_c(client, 350, fixed)  # fixed program: link 2 red 283-369
    assert waiting["status"] == "blocked" and "préemption" not in waiting["reason"]
    assert waiting["reason"].startswith("ambulance arrêtée au feu rouge")
    moving = ambulance_c(client, 375, fixed)  # link 2 green, past the line
    assert moving["status"] == "clear" and moving["passed"] is True


def test_scenarios_carry_the_evaluation_labels(client):
    b = client.get("/scenarios").json()
    groups = {g["scenario"]: g for g in b["scenarios"]}
    assert {s: (g["label"], g["family"]) for s, g in groups.items()} == {
        "pitie_actuated": ("S1", "main"),
        "pitie_fixed": ("S0", "main"),
        "pitie_rules": ("S2", "main"),
    }
    run = groups["pitie_rules"]["runs"][0]
    assert run["seed"] == 4000 and run["has_eta"] is False and run["dir"].endswith("seed4000")
    assert run["current"] is not True  # a fixture is never a run of the current config
    assert groups["pitie_fixed"]["controller"] == "fixed"


def test_kpi_from_the_evaluation_csv(client):
    b = client.get("/kpi?scenario=S2").json()
    assert b["status"] == "ok" and b["source"] == "evaluation"
    assert (b["scenario"], b["label"], b["family"]) == ("pitie_rules", "S2", "main")
    rows = {(r["user"], r["kpi"]): r for r in b["rows"]}
    car = rows[("car", "time_loss_mean")]
    assert car["mean"] == 56.0 and car["n"] == 1 and car["std"] is None
    assert car["rel_change_vs_s0_pct"] == -20.0 and car["worse_flag"] is False
    s0 = client.get("/kpi?scenario=pitie_fixed").json()
    assert s0["label"] == "S0" and all(r["rel_change_vs_s0_pct"] is None for r in s0["rows"])
    only = client.get("/kpi?scenario=S2&user=car").json()
    assert {r["user"] for r in only["rows"]} == {"car"}


def test_kpi_falls_back_to_the_runs(api_results):
    for f in (api_results / "evaluation" / "main").iterdir():
        f.unlink()
    b = make_client(api_results).get("/kpi?scenario=S1").json()
    assert b["status"] == "degraded" and b["source"] == "runs" and b["notice"]
    rows = {(r["user"], r["kpi"]): r for r in b["rows"]}
    assert rows[("car", "time_loss_mean")]["mean"] == 63.0  # 70 x 0.9
    assert rows[("car", "time_loss_mean")]["rel_change_vs_s0_pct"] is None


def test_health(shared_results, tmp_path):
    b = make_client(shared_results).get("/health").json()
    assert b["status"] == "ok" and b["runs_indexed"] == 3
    assert b["default_run"]["scenario"] == "pitie_rules" and b["eta_model_version"] is None
    make_fake_eta(tmp_path / "models")
    assert (
        make_client(shared_results, tmp_path / "models").get("/health").json()["eta_model_version"]
        == "fake-ref-0"
    )


def test_eta_from_precomputed_predictions_and_advice(api_results):
    make_fake_eta(api_results / "pitie_rules_seed4000")
    c = make_client(api_results)
    b = c.get("/eta?trajet=N_S&t=08:00").json()
    assert b["status"] == "ok" and not b["approximate"] and b["model_version"] == "fake-0"
    assert b["eta_low_s"] <= b["eta_median_s"] <= b["eta_high_s"]
    assert b["data_age_s"] <= 5.0 and b["advice"] is None and b["rest_of_trip_s"] == 0.0
    now = datetime.fromisoformat("2026-10-14T08:00:00+02:00")
    late = c.get("/eta?trajet=N_S&t=08:00&deadline=08:00:30").json()
    assert late["advice"]["code"] == "risk_of_delay"
    ok = c.get("/eta?trajet=N_S&t=08:00&deadline=09:00&student=true").json()
    assert ok["rest_of_trip_s"] == 120.0 and ok["advice"]["code"] == "comfortable"
    arrival = datetime.fromisoformat(ok["advice"]["arrival_median"])
    assert arrival == now + timedelta(seconds=ok["eta_median_s"] + 120.0)
    assert c.get("/scenarios").json()["scenarios"][2]["runs"][0]["has_eta"] is True


def test_queue_beyond_zone_from_the_snapshots(api_results):
    make_fake_eta(api_results / "pitie_rules_seed4000")  # pending_N = 2 in [600, 900)
    c = make_client(api_results)
    assert c.get("/eta?trajet=N_S&t=700").json()["queue_beyond_zone"] is True
    assert c.get("/eta?trajet=W_S&t=700").json()["queue_beyond_zone"] is False
    assert c.get("/eta?trajet=N_S&t=1000").json()["queue_beyond_zone"] is False
    fixed = run_id_of(c, "pitie_fixed")  # no snapshots: unknown
    assert c.get(f"/eta?trajet=N_S&t=700&run_id={fixed}").json()["queue_beyond_zone"] is None


def test_mock_page(client, shared_results):
    r = client.get("/mock")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "Données simulées" in r.text and "/crossing/" in r.text
    off = make_client(shared_results, mock=False).get("/mock")
    assert off.status_code == 404 and off.json()["error"]["code"] == "not_found"


def test_cors_allows_get_only(client):
    r = client.options(
        "/health",
        headers={"Origin": "http://example.org", "Access-Control-Request-Method": "POST"},
    )
    assert "POST" not in r.headers.get("access-control-allow-methods", "")
    g = client.get("/health", headers={"Origin": "http://example.org"})
    assert g.headers["access-control-allow-origin"] == "*"
    assert "set-cookie" not in g.headers

"""API additions of Parts 8-9 (D103, D105; contract 4.3) on synthetic run folders and the
FAKE evaluation outputs of `api.fixtures` (never data/results):

- `/evaluation`, `/compare`, `/series`, `/sensitivity`: envelopes, statuses (a missing
  output file is `unavailable`, a label registered but not in the outputs "pas encore
  évalué", outputs of an older config `degraded`) and 404 `bad_request` for an unknown
  family, label or pair; never a 500;
- `/replay` time bounds and signal heads, `/replay/{run_id}/events`;
- `/scenarios` roles and seed roles;
- the Part 8 fixture outputs leave every Part 5 answer unchanged.

The fixture runs replay the net's own 123 s program from t = 0 (HOP green 0-36, its
clearance 36-45 with link 2 yellow 37-40, S_LEFT 45-51, STM 61-81 with links 10-11
yellow 81-84, ...)."""

import json
import shutil
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from traffic_twin.api.fixtures import MAIN_PAIRS, make_results  # noqa: E402
from traffic_twin.api.main import FAKE_NOTICE, PART8_PENDING, create_app  # noqa: E402
from traffic_twin.api.settings import load_settings  # noqa: E402
from traffic_twin.config import load_scenario  # noqa: E402
from traffic_twin.evaluation.runs import expected_hashes  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SENSITIVITY = REPO_ROOT / "docs" / "figures" / "rl7" / "sensitivity.csv"
MAIN_LABELS = ["S0", "S1", "S2", "S2_noP", "S3", "S4"]
NEW_ENDPOINTS = (
    ("/evaluation", "evaluation"),
    ("/evaluation?family=main", "evaluation"),
    ("/compare?a=S1&b=S3", "kpi_pair"),
    ("/series?family=main&approach=all&labels=S0,S2,S3", "kpi_series"),
    ("/sensitivity", "sensitivity"),
    ("/replay/default/events?from=0&to=600", "replay_events"),
)


def make_client(results: Path, **overrides) -> TestClient:
    settings = replace(
        load_settings(repo_root=REPO_ROOT, environ={}),
        results_dirs=(results,),
        evaluation_dir=results / "evaluation",
        eta_models_dir=results.parent / "no_models",
        default_run=(results / "pitie_rules_seed4000",),
        replay_date=date(2026, 10, 13),
        **overrides,
    )
    return TestClient(create_app(settings), raise_server_exceptions=False)


@pytest.fixture(scope="module")
def shared_results(tmp_path_factory) -> Path:
    return make_results(tmp_path_factory.mktemp("api9") / "results", n_runs=3)


@pytest.fixture(scope="module")
def client(shared_results: Path) -> TestClient:
    return make_client(shared_results)


@pytest.fixture
def results(tmp_path) -> Path:
    """Per-test results with the Part 8 FAKE outputs (tests that change files)."""
    return make_results(tmp_path / "results", n_runs=3)


@pytest.fixture
def part4_results(tmp_path) -> Path:
    """Results with the Part 4 evaluation layout only: S3 and S4 are registered in
    configs/evaluation.yaml but absent from the outputs (the state before Part 8)."""
    return make_results(tmp_path / "results", n_runs=3, part8=False)


def run_id_of(client: TestClient, scenario: str) -> str:
    groups = client.get("/scenarios").json()["scenarios"]
    return next(g for g in groups if g["scenario"] == scenario)["runs"][0]["run_id"]


# -- envelopes and errors ----------------------------------------------------------------
@pytest.mark.parametrize(("url", "kind"), NEW_ENDPOINTS)
def test_new_endpoints_have_the_envelope(client, url, kind):
    r = client.get(url)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["type"] == kind and b["data_source"] == "simulation" and b["error"] is None
    assert b["calibrated_on"] == "2026-10-08" and b["provisional"] is True
    assert b["status"] == "ok", b["notice"]
    assert datetime.fromisoformat(b["timestamp"]).utcoffset() is not None


@pytest.mark.parametrize(
    ("url", "kind", "code"),
    [
        ("/evaluation?family=nope", "evaluation", "bad_request"),
        ("/compare?a=S1&b=S3&family=nope", "kpi_pair", "bad_request"),
        ("/compare?a=S1&b=S9", "kpi_pair", "bad_request"),
        ("/compare?a=S3&b=S1", "kpi_pair", "bad_request"),  # pairs are oriented
        ("/compare?a=S1", "kpi_pair", "bad_request"),  # b missing (422 remapped)
        ("/series?family=nope", "kpi_series", "bad_request"),
        ("/series?labels=S0,S9", "kpi_series", "bad_request"),
        ("/series?approach=NE", "kpi_series", "bad_request"),
        ("/series?metric=jam_m", "kpi_series", "bad_request"),
        ("/replay/nope/events", "replay_events", "unknown_run"),
        ("/replay/default/events?from=600&to=100", "replay_events", "bad_t"),
        ("/replay/default/events?from=xx", "replay_events", "bad_t"),
        ("/replay/default/events?limit=0", "replay_events", "bad_request"),
    ],
)
def test_new_endpoints_404_with_an_envelope(client, url, kind, code):
    r = client.get(url)
    assert r.status_code == 404, r.text
    b = r.json()
    assert b["type"] == kind and b["status"] == "unavailable"
    assert b["error"]["code"] == code and b["data_source"] == "simulation"


def test_unregistered_pair_lists_the_registered_ones(client):
    msg = client.get("/compare?a=S3&b=S1").json()["error"]["message"]
    assert "S1-S3" in msg and "S2_noP-S4" in msg


# -- /evaluation -------------------------------------------------------------------------
def test_evaluation_from_the_outputs(client):
    b = client.get("/evaluation").json()
    assert b["family"] == "main" and b["preregistered"] == "2026-10-12"
    assert b["seeds"] == [4000] and b["notice"] == FAKE_NOTICE
    labels = {x["label"]: x for x in b["labels"]}
    assert list(labels) == MAIN_LABELS
    assert {k: v["role"] for k, v in labels.items()} == {
        "S0": "main", "S1": "main", "S2": "main", "S3": "main",
        "S2_noP": "ablation", "S4": "ablation",
    }  # fmt: skip
    assert labels["S3"]["scenario"] == "pitie_rl_s3" and labels["S3"]["controller"] == "rl"
    assert labels["S0"]["controller"] == "fixed" and all(x["evaluated"] for x in b["labels"])
    assert b["a03"]["text"] == "4/4" and b["a04"]["pass"] is True
    assert b["a05"]["verdict"] and "secondary" in b["a05"]  # a05.json, not the summary's
    a06 = b["a06"]
    assert a06["status"] == "PASS" and a06["primary"] == "S1" and a06["fake"] is True
    assert [c["reference"] for c in a06["comparisons"]] == ["S1", "S2", "S0"]
    assert b["replication_10_6"]["main_solution"] == "S2"
    assert b["registration"]["order"] == "before"
    rows = b["final_table"]
    assert {r["label"] for r in rows} == set(MAIN_LABELS) and all(r["fake"] for r in rows)
    pd_s3 = next(r for r in rows if (r["label"], r["kpi"]) == ("S3", "person_delay_mean"))
    assert pd_s3["section"] == "kpi" and pd_s3["ablation"] is False and pd_s3["n"] == 1
    assert next(r for r in rows if r["label"] == "S4")["ablation"] is True
    assert isinstance(b["hidden_degradation"], list)


def test_evaluation_of_a_family_without_outputs(client):
    """ambulance_probe is registered but the fixtures evaluate main only."""
    b = client.get("/evaluation?family=ambulance_probe").json()
    assert b["status"] == "degraded" and PART8_PENDING in b["notice"]
    assert b["final_table"] == [] and not any(x["evaluated"] for x in b["labels"])


def test_evaluation_without_summary_is_unavailable(results):
    (results / "evaluation" / "summary.json").unlink()
    b = make_client(results).get("/evaluation").json()
    assert b["status"] == "unavailable" and PART8_PENDING in b["notice"]
    assert [x["label"] for x in b["labels"]] == MAIN_LABELS and b["a06"] is None


def test_evaluation_without_a06_is_degraded(results):
    (results / "evaluation" / "a06.json").unlink()
    summary = results / "evaluation" / "summary.json"
    s = json.loads(summary.read_text(encoding="utf-8"))
    del s["a06"]
    summary.write_text(json.dumps(s), encoding="utf-8")
    b = make_client(results).get("/evaluation").json()
    assert b["status"] == "degraded" and b["notice"].startswith(f"A-06 : {PART8_PENDING}")
    assert b["a06"] is None and b["replication_10_6"] is not None  # from the summary


def test_corrupt_outputs_never_give_5xx(results):
    ev = results / "evaluation"
    for name in ("summary.json", "a06.json", "a05.json"):
        (ev / name).write_text("{not json", encoding="utf-8")
    for name in ("final_table.csv", "series.csv", "main/pairs/S1__vs__S3.csv"):
        (ev / name).write_bytes(b"\x00\x01\x02")
    c = make_client(results)
    for url, _ in NEW_ENDPOINTS:
        r = c.get(url)
        assert r.status_code == 200, (url, r.text)
    assert c.get("/evaluation").json()["status"] == "unavailable"
    assert c.get("/series").json()["status"] == "unavailable"


# -- S3 registered but not in the outputs (before Part 8) ---------------------------------
def test_registered_but_not_evaluated_is_unavailable_never_500(part4_results):
    c = make_client(part4_results)
    for url in ("/kpi?scenario=S3", "/kpi?scenario=S4", "/kpi?scenario=pitie_rl_s3"):
        r = c.get(url)
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["status"] == "unavailable" and b["notice"].endswith(PART8_PENDING)
        assert b["rows"] == [] and b["label"] in ("S3", "S4")
    pair = c.get("/compare?a=S1&b=S3").json()
    assert pair["status"] == "unavailable" and pair["notice"] == f"S1-S3 : {PART8_PENDING}"
    assert pair["role"] == "confirmatory" and pair["rows"] == []
    ev = c.get("/evaluation").json()
    assert ev["status"] == "unavailable"  # the Part 4 fixtures have no summary.json
    series = c.get("/series?labels=S0,S3").json()
    assert series["status"] == "unavailable" and series["points"] == []
    assert c.get("/kpi?scenario=S2").json()["status"] == "ok"  # evaluated labels still answer


def test_series_with_a_missing_label_is_degraded(results):
    path = results / "evaluation" / "series.csv"
    df = pd.read_csv(path)
    df[df["label"] != "S4"].to_csv(path, index=False)
    b = make_client(results).get("/series?labels=S3,S4").json()
    assert b["status"] == "degraded" and b["notice"].startswith(f"S4 : {PART8_PENDING}")
    assert {p["label"] for p in b["points"]} == {"S3"}
    only = make_client(results).get("/series?labels=S4").json()
    assert only["status"] == "unavailable" and only["points"] == []


def test_summary_listing_a_missing_label_is_degraded(results):
    summary = results / "evaluation" / "summary.json"
    s = json.loads(summary.read_text(encoding="utf-8"))
    s["families"]["main"] = ["S0", "S1", "S2", "S2_noP"]
    summary.write_text(json.dumps(s), encoding="utf-8")
    b = make_client(results).get("/evaluation").json()
    assert b["status"] == "degraded" and b["notice"].startswith(f"S3, S4 : {PART8_PENDING}")
    assert {x["label"]: x["evaluated"] for x in b["labels"]}["S3"] is False


# -- stale outputs -----------------------------------------------------------------------
def write_provenance(results: Path, hashes: dict[str, dict]) -> None:
    prov = {"families": {"main": {"scenarios": {
        lb: {"controller": "rl" if lb in ("S3", "S4") else None, **h}
        for lb, h in hashes.items()
    }}}}  # fmt: skip
    (results / "evaluation" / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")


def test_outputs_of_an_older_config_are_degraded(results, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)  # the current hashes read the net by its relative path
    s0 = expected_hashes(load_scenario(REPO_ROOT / "configs" / "pitie_fixed.yaml"))
    s3 = {"config_hash": "0" * 64, "demand_hash": s0["demand_hash"]}  # an older S3 config
    write_provenance(results, {"S0": s0, "S3": s3})
    c = make_client(results)
    ev = c.get("/evaluation").json()
    assert ev["status"] == "degraded" and "configuration antérieure" in ev["notice"]
    assert ev["notice"].split(" : ")[1].startswith("S3")  # S0 is current
    assert c.get("/compare?a=S1&b=S3").json()["status"] == "degraded"
    assert c.get("/compare?a=S0&b=S1").json()["status"] == "ok"  # S0 current, S1 unknown
    assert c.get("/series?labels=S0,S1").json()["status"] == "ok"
    assert c.get("/series?labels=S3").json()["status"] == "degraded"


# -- /compare, /series, /sensitivity -------------------------------------------------------
def test_compare_rows_and_roles(client):
    roles = {(a, b): role for a, b, role in MAIN_PAIRS}
    for (a, b), role in roles.items():
        body = client.get(f"/compare?a={a}&b={b}").json()
        assert body["status"] == "ok" and body["role"] == role, (a, b)
        assert body["rows"] and {f"{a.lower()}_mean", f"{b.lower()}_mean"} <= set(body["rows"][0])
    rows = client.get("/compare?a=S1&b=S3").json()["rows"]
    pd_row = next(r for r in rows if (r["user_type"], r["kpi"]) == ("all", "person_delay_mean"))
    assert pd_row["s3_mean"] < pd_row["s1_mean"] and pd_row["fake"] is True
    assert pd_row["p_value"] is None  # NaN in the CSV -> null


def test_series_points(client):
    b = client.get("/series?family=main&metric=halting_veh&approach=N&labels=S3,S0").json()
    assert b["metric"] == "halting_veh" and b["approach"] == "N" and b["step_s"] == 60
    pts = b["points"]
    assert set(pts[0]) == {"label", "t_s", "mean", "p10", "p90", "n"}
    labels = [p["label"] for p in pts]
    assert labels == sorted(labels, key=["S3", "S0"].index)  # the order asked for
    s3 = [p for p in pts if p["label"] == "S3"]
    assert [p["t_s"] for p in s3] == list(range(0, 1800, 60))
    assert all(p["p10"] <= p["mean"] <= p["p90"] and p["n"] == 1 for p in pts)
    default = client.get("/series").json()
    assert {p["label"] for p in default["points"]} == set(MAIN_LABELS)


def test_sensitivity_is_the_committed_table(client, tmp_path, shared_results):
    b = client.get("/sensitivity").json()
    assert b["source"] == "docs/figures/rl7/sensitivity.csv"
    want = pd.read_csv(SENSITIVITY)
    assert len(b["rows"]) == len(want) and list(b["rows"][0]) == list(want.columns)
    assert b["rows"][0]["setting"] == want["setting"].iloc[0]
    gone = make_client(shared_results, sensitivity=tmp_path / "none.csv").get("/sensitivity")
    assert gone.status_code == 200 and gone.json()["status"] == "unavailable"


def test_without_an_evaluation_config_the_new_endpoints_are_unavailable(results, tmp_path):
    c = make_client(results, evaluation_config=tmp_path / "missing.yaml")
    for url in ("/evaluation", "/compare?a=S1&b=S3", "/series"):
        b = c.get(url).json()
        assert b["status"] == "unavailable" and b["notice"], url


# -- /scenarios ----------------------------------------------------------------------------
def test_scenarios_roles_and_seed_roles(client):
    groups = {g["scenario"]: g for g in client.get("/scenarios").json()["scenarios"]}
    assert {s: g["role"] for s, g in groups.items()} == {
        "pitie_fixed": "main", "pitie_actuated": "main", "pitie_rules": "main",
    }  # fmt: skip
    assert all(r["seed_role"] == "demo" for g in groups.values() for r in g["runs"])


def test_seed_roles_evaluation_and_other(tmp_path):
    root = make_results(tmp_path / "results", n_runs=3)
    for name, seed in (("pitie_fixed_seed4000", 1), ("pitie_actuated_seed4000", 3210)):
        meta_path = root / name / "run.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["seed"] = seed
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    groups = make_client(root).get("/scenarios").json()["scenarios"]
    roles = {g["scenario"]: g["runs"][0]["seed_role"] for g in groups}
    assert roles == {"pitie_fixed": "evaluation", "pitie_actuated": "other", "pitie_rules": "demo"}


# -- /replay extras --------------------------------------------------------------------------
def test_replay_bounds(client):
    b = client.get("/replay/default?t=100").json()
    assert (b["t_min_s"], b["t_max_s"], b["step_s"]) == (0.0, 1795.0, 5.0)
    late = client.get("/replay/default?t=5000").json()  # stale, bounds still given
    assert late["stale"] and late["t_max_s"] == 1795.0


@pytest.mark.parametrize(
    ("t", "heads"),
    [
        (20, {"N": "green", "S": "green", "W": "red"}),  # HOP
        (38, {"N": "yellow", "S": "green", "W": "red"}),  # link 2 yellow 37-40
        (41, {"N": "red", "S": "green", "W": "red"}),
        (70, {"N": "green", "S": "red", "W": "green"}),  # STM
        (82, {"N": "green", "S": "red", "W": "yellow"}),  # links 10-11 yellow 81-84
        (86, {"N": "green", "W": "red"}),
    ],
)
def test_replay_signal_heads(client, t, heads):
    b = client.get(f"/replay/{run_id_of(client, 'pitie_fixed')}?t={t}").json()
    sig = b["signals"]
    assert {"N", "S", "W", "crossings"} <= set(sig)
    assert {k: sig[k] for k in heads} == heads, sig
    assert set(sig["crossings"]) == {"NB", "S", "WB", "W"}


def test_replay_crossing_heads_match_the_crossing_endpoint(client):
    walk = {"green": "walk", "clearance": "flash", "red": "dont_walk"}
    rid = run_id_of(client, "pitie_fixed")
    for t in range(0, 250, 7):
        heads = client.get(f"/replay/{rid}?t={t}").json()["signals"]["crossings"]
        for cid, head in heads.items():
            sig = client.get(f"/crossing/{cid}?t={t}&run_id={rid}").json()["signal"]
            assert head == walk[sig], (t, cid)


def test_replay_signals_without_signal_log_use_the_program(results):
    (results / "pitie_fixed_seed4000" / "signal.parquet").unlink()
    c = make_client(results)
    rid = run_id_of(c, "pitie_fixed")
    sig = c.get(f"/replay/{rid}?t=20").json()["signals"]  # phase 0 (HOP) at the 20 s sample
    assert (sig["N"], sig["S"], sig["W"]) == ("green", "green", "red")
    assert sig["crossings"]["NB"] == "walk" and sig["crossings"]["S"] == "dont_walk"
    clear = c.get(f"/replay/{rid}?t=38").json()["signals"]  # the 35 s sample: phase 0
    assert clear["N"] == "green"  # 5 s state-log resolution without the signal log


def test_replay_of_another_network_has_no_signals(results):
    meta_path = results / "pitie_fixed_seed4000" / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["config"]["network"]["net_file"] = "sumo/network/intersection.net.xml"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    b = make_client(results).get(f"/replay/{meta['run_id']}?t=20").json()
    assert b["status"] == "ok" and b["signals"] is None and b["t_max_s"] == 1795.0


# -- /replay/{run_id}/events --------------------------------------------------------------
def test_events_of_a_window(client):
    b = client.get("/replay/default/events?from=290&to=08:00").json()  # 08:00 = 1800 s
    assert b["sim_t_s"] == 290.0 and b["run"]["scenario"] == "pitie_rules"
    ev = b["events"]
    assert all(set(e) == {"t", "source", "rule", "stage_from", "stage_to", "detail"} for e in ev)
    assert [e["t"] for e in ev] == sorted(e["t"] for e in ev) and min(e["t"] for e in ev) >= 290
    pre = [(e["t"], e["rule"]) for e in ev if e["source"] == "preemption"]
    assert pre == [(301.0, "request"), (371.0, "served")]
    assert b["timestamp"] == "2026-10-13T08:00:00+02:00"
    whole = client.get("/replay/default/events").json()["events"]
    assert len(whole) > len(ev) and whole[0]["rule"] == "tsp_extend"


def test_events_limit(client):
    b = client.get("/replay/default/events?limit=2").json()
    assert len(b["events"]) == 2 and b["status"] == "ok" and "2 premiers" in b["notice"]


def test_events_of_a_controller_without_events(client):
    b = client.get(f"/replay/{run_id_of(client, 'pitie_fixed')}/events").json()
    assert b["status"] == "ok" and b["events"] == [] and "fixed" in b["notice"]


def test_events_missing_for_an_adaptive_controller(results):
    (results / "pitie_rules_seed4000" / "events.parquet").unlink()
    b = make_client(results).get("/replay/default/events").json()
    assert b["status"] == "unavailable" and b["notice"] and b["events"] == []


# -- the Part 5 answers do not change ------------------------------------------------------
OLD_URLS = (
    "/health", "/scenarios", "/kpi?scenario=S0", "/kpi?scenario=S1", "/kpi?scenario=S2",
    "/kpi?scenario=S2&user=car", "/replay/default?t=62", "/replay/default?t=310&positions=true",
    "/crossing/NB?t=20", "/crossing/S?t=10", "/bus/91/status?t=122",
    "/ambulance/next-intersections?t=310", "/eta?trajet=N_S&t=600",
)  # fmt: skip
NEW_KEYS = {"t_min_s", "t_max_s", "step_s", "signals", "role", "seed_role"}


def strip(obj, tmp: str):
    """A response without wall-clock fields, the temp dir and the Part 8-9 keys."""
    if isinstance(obj, dict):
        return {
            k: strip(v, tmp) for k, v in obj.items()
            if k not in NEW_KEYS and k not in ("timestamp", "data_age_s", "dir")
        }  # fmt: skip
    if isinstance(obj, list):
        return [strip(v, tmp) for v in obj]
    return obj


def test_part8_fixtures_leave_the_part5_answers_unchanged(tmp_path):
    with_p8 = make_client(make_results(tmp_path / "a" / "results", n_runs=3))
    without = make_client(make_results(tmp_path / "b" / "results", n_runs=3, part8=False))
    for url in OLD_URLS:
        a, b = with_p8.get(url), without.get(url)
        assert a.status_code == b.status_code == 200, url
        assert strip(a.json(), str(tmp_path)) == strip(b.json(), str(tmp_path)), url


def test_fixture_outputs_are_marked_fake(shared_results):
    ev = shared_results / "evaluation"
    assert (ev / "FAKE.txt").is_file()
    for name in ("summary.json", "a05.json", "a06.json"):
        assert json.loads((ev / name).read_text(encoding="utf-8"))["fake"] is True, name
    for name in (
        "final_table.csv",
        "series.csv",
        *(f"main/pairs/{a}__vs__{b}.csv" for a, b, _ in MAIN_PAIRS),
    ):
        assert pd.read_csv(ev / name)["fake"].all(), name
    a06 = json.loads((ev / "a06.json").read_text(encoding="utf-8"))
    assert all(c["sentence"].startswith("FAKE") for c in a06["comparisons"])


def test_fixture_cli_writes_the_part8_outputs(tmp_path):
    from traffic_twin.api.fixtures import main

    out = tmp_path / "fake"
    assert main(["--out", str(out), "--duration-s", "600"]) == 0
    for name in ("summary.json", "a06.json", "final_table.csv", "series.csv"):
        assert (out / "evaluation" / name).is_file(), name
    shutil.rmtree(out)

"""The Streamlit Operator (`app.py`, D102, D108) and its API client (`app/client.py`):
AppTest runs against the in-process API (`TT_API_INPROCESS=1`) on the FAKE fixtures, a
closed port (API down), degraded and stale settings, and canned FAKE payloads covering the
Part 8 outputs (final table, A-06, pairs, series). Plus the run_mock.py quarantine (D107).

Nothing here reads data/results or runs SUMO; every number is FAKE."""

from __future__ import annotations

import importlib.util
import json
import re
import socket
import subprocess
import threading
from http.server import HTTPServer
from pathlib import Path

import pytest
import yaml

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from traffic_twin.app import client  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
APP = REPO_ROOT / "app.py"
D108_TABS = [
    "Comparaison",
    "Tous les KPI",
    "Tests et paires",
    "Équité",
    "Sécurité et ambulance",
    "Courbes",
    "Sensibilité des poids",
    "Exports",
]
# D107: not allowed in app.py
BANNED = ("Seed 5", "0,0020", "0,0010", "parties 6-7", "cycle 123 s", "EXPORT_DIR", "evaluation.md")


# -- helpers ----------------------------------------------------------------------------
@pytest.fixture
def api_env(monkeypatch):
    """`api_env(config=None, url=None)`: point the client at the in-process API on
    `config`, or at `url` over HTTP; the cached clients are dropped before and after."""

    def _set(config: Path | None = None, url: str | None = None) -> None:
        client.reset()
        monkeypatch.delenv("TT_API_URL", raising=False)
        monkeypatch.delenv("TT_API_INPROCESS", raising=False)
        if config is not None:
            monkeypatch.setenv("TT_API_CONFIG", str(config))
            monkeypatch.setenv("TT_API_INPROCESS", "1")
        if url is not None:
            monkeypatch.setenv("TT_API_URL", url)

    yield _set
    client.reset()


def closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def edited_config(config: Path, **changes) -> Path:
    """A copy of the FAKE api.yaml with top-level `changes` merged (one level deep)."""
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    out = config.with_name("api_edited.yaml")
    out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def run_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60)
    return at.run()


def texts(at: AppTest) -> str:
    parts = []
    for kind in ("markdown", "caption", "info", "warning", "error", "success", "title"):
        parts += [str(e.value) for e in getattr(at, kind)]
    parts += [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(parts)


def no_exception(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert "Traceback" not in texts(at)


def repo_snapshot() -> tuple[set[str], str]:
    """Top-level repo entries and the git status of the folders the app could write."""
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--ignored=matching",
            "--untracked-files=normal",
            "--",
            "data",
            ".streamlit",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {p.name for p in REPO_ROOT.iterdir()}, status


@pytest.fixture
def downloads(monkeypatch):
    """The payloads of the AppTest download buttons: `downloads(at) -> {label: bytes}`."""
    import streamlit.testing.v1.app_test as app_test
    from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage

    stores: list[MemoryMediaFileStorage] = []

    class Recording(MemoryMediaFileStorage):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            stores.append(self)

    monkeypatch.setattr(app_test, "MemoryMediaFileStorage", Recording)

    def _read(at: AppTest) -> dict[str, bytes]:
        out = {}
        for b in at.get("download_button"):
            file_id = b.proto.url.rsplit("/", 1)[1].split(".")[0]
            out[b.proto.label] = stores[-1].get_file(file_id).content
        return out

    return _read


# -- client -----------------------------------------------------------------------------
def test_api_url_default_and_override(monkeypatch):
    monkeypatch.delenv("TT_API_URL", raising=False)
    assert client.api_url() == "http://127.0.0.1:8000"
    monkeypatch.setenv("TT_API_URL", "http://127.0.0.1:9000/")
    assert client.api_url() == "http://127.0.0.1:9000"


def test_state_of_follows_spec_12_1():
    assert client.state_of({"status": "ok"}) == "ok"
    assert client.state_of({"status": "ok", "stale": True}) == "stale"
    assert client.state_of({"status": "degraded", "stale": True}) == "degraded"
    assert client.state_of({"status": "unavailable"}) == "unavailable"
    assert client.state_of({"status": "unavailable", "error": {"code": "x"}}) == "error"
    assert set(client.STATES) == {"ok", "degraded", "stale", "unavailable", "error"}


def test_client_in_process_reads_the_envelopes(api_env, fake_api_config):
    api_env(config=fake_api_config)
    payload, state = client.get("/health")
    assert state == "ok" and payload["type"] == "health" and payload["data_source"] == "simulation"
    rep = client.fetch("/kpi", scenario="S0", family="no_such_family", user=None)
    assert rep.payload is None and rep.state == "error" and rep.status_code == 404
    assert rep.message and "no_such_family" in rep.message


def test_client_never_raises_when_the_api_is_down(api_env):
    api_env(url=f"http://127.0.0.1:{closed_port()}")
    rep = client.fetch("/health")
    assert rep.payload is None and rep.state == "error"
    assert rep.message.startswith(client.UNREACHABLE)
    assert client.get("/health") == (None, "error")


def test_client_times_out(api_env, monkeypatch):
    with socket.socket() as srv:  # accepts the connection (backlog) and never answers
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        monkeypatch.setattr(client, "TIMEOUT_S", 0.3)
        api_env(url=f"http://127.0.0.1:{srv.getsockname()[1]}")
        rep = client.fetch("/health")
    assert rep == client.Reply(None, "error", client.TIMED_OUT)


def test_client_params_drop_none_and_join_lists():
    assert client._params({"a": None, "b": True, "c": ["S0", "S2"], "d": 3}) == {
        "b": "true",
        "c": "S0,S2",
        "d": "3",
    }


# -- app.py: static honesty and isolation (D102, D107) ---------------------------------
def test_app_source_reads_and_writes_no_repository_file():
    src = APP.read_text(encoding="utf-8")
    for banned in BANNED:
        assert banned not in src, banned
    for call in (
        "open(",
        "read_csv",
        "read_text",
        "read_parquet",
        "read_json",
        "write_text",
        "mkdir",
        "Path(",
    ):
        assert call not in src, call
    assert re.findall(r"\.to_csv\((?!index=False\))", src) == []  # in-memory CSV only
    assert "docs/" not in src and "data/" not in src
    assert "traffic_twin.api" not in src  # through the client only
    assert "from traffic_twin.app import client" in src


# -- app.py: pure helpers ----------------------------------------------------------------
@pytest.fixture(scope="module")
def operator_app():
    spec = importlib.util.spec_from_file_location("operator_app", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_s3_colour_is_the_viz_rl_green(operator_app):
    from traffic_twin.viz.style import CONTROLLER_COLORS

    assert operator_app.COLORS["S3"] == CONTROLLER_COLORS["rl"]


def test_labels_come_from_evaluation_then_scenarios(operator_app):
    ev = {
        "labels": [
            {"label": "S0", "controller": "fixed", "role": None},
            {"label": "S4", "role": "exploratory"},
        ]
    }
    sc = {
        "scenarios": [
            {"label": "S1", "family": "main"},
            {"label": "S0", "family": "ambulance_probe"},
        ]
    }
    assert [(r["label"], r["role"]) for r in operator_app.label_info(ev, sc, "main")] == [
        ("S0", "main"),
        ("S4", "exploratory"),
    ]
    assert [r["label"] for r in operator_app.label_info({}, sc, "main")] == ["S1"]
    assert operator_app.families_of(sc) == ["main", "ambulance_probe"]
    assert operator_app.families_of({}) == ["main"]


def test_final_frames_split_kpi_and_extra_rows(operator_app):
    rows = [
        {
            "section": "main",
            "user": "all",
            "kpi": "person_delay_mean",
            "label": "S3",
            "rel_vs_s0_pct": -1.0,
        },
        {"section": "probe", "user": "ambulance", "kpi": "stops", "label": "S3"},
        {"section": "activity", "user": "junction", "kpi": "interventions_per_run", "label": "S3"},
    ]
    kpis, extra = operator_app.final_frames(rows, "main")
    assert list(kpis.kpi) == ["person_delay_mean"] and kpis.rel_pct.iloc[0] == -1.0
    assert list(extra.section) == ["activity"]
    probe, _ = operator_app.final_frames(rows, "ambulance_probe")
    assert list(probe.kpi) == ["stops"]
    assert all(f.empty for f in operator_app.final_frames([], "main"))


def test_footer_is_honest_about_the_program(operator_app):
    env = {
        "data_source": "simulation",
        "calibrated_on": "2026-10-08",
        "provisional": True,
        "timestamp": "2026-10-13T08:15:00+02:00",
        "data_age_s": 30,
    }
    text = operator_app.footer_text(env)
    assert text.startswith("Données simulées") and "2026-10-08" in text
    assert "programme de feux provisoire" in text and "13/10/2026 08:15:00" in text
    final = operator_app.footer_text({**env, "provisional": False})
    assert "provisoire" not in final and "certifié" not in final


def test_said_does_not_repeat_the_prefix(operator_app):
    assert operator_app.said("Courbes", "courbes : pas encore évalué", "x") == (
        "Courbes : pas encore évalué."
    )
    assert operator_app.said("Courbes", None, "pas encore évaluées") == (
        "Courbes : pas encore évaluées."
    )


# -- AppTest on the in-process API ---------------------------------------------------------
def test_nominal_render_on_fake_data(api_env, fake_api_config, downloads):
    api_env(config=fake_api_config)
    calibrated = client.get("/health")[0]["calibrated_on"]
    before = repo_snapshot()
    at = run_app()
    no_exception(at)
    assert [t.label for t in at.tabs] == D108_TABS
    text = texts(at)
    assert "Données simulées" in text and calibrated in text
    assert "programme de feux provisoire" in text
    assert at.toggle(key="scn_S0").value is True
    # S3 and S4 are registered (evaluation.yaml) and shown, evaluated or not
    keys = {t.key for t in at.toggle}
    assert {"scn_S3", "scn_S4"} <= keys
    assert "S3" in text and "S4" in text
    files = downloads(at)
    assert files, "no CSV download offered"
    for label, data in files.items():
        lines = data.decode("utf-8-sig").strip().splitlines()
        assert len(lines) >= 2, f"{label}: empty CSV"
    # toggling a scenario off reruns cleanly
    at.toggle(key="scn_S0").set_value(False).run()
    no_exception(at)
    assert repo_snapshot() == before, "the Operator wrote into the repository"


def test_api_down_gives_a_simple_message(api_env):
    api_env(url=f"http://127.0.0.1:{closed_port()}")
    at = run_app()
    no_exception(at)
    errors = [e.value for e in at.error]
    assert errors and errors[0].startswith("Données temporairement indisponibles")
    assert "Données simulées" in texts(at)
    assert not at.tabs


def test_kpis_from_runs_show_the_degraded_banner(api_env, fake_api_config, tmp_path):
    empty = tmp_path / "no_evaluation"
    empty.mkdir()
    api_env(config=edited_config(fake_api_config, evaluation={"dir": empty.as_posix()}))
    at = run_app()
    no_exception(at)
    warnings = [w.value for w in at.warning]
    assert any(w.startswith("Mode dégradé") for w in warnings), warnings


def test_old_answers_show_the_stale_banner(api_env, fake_api_config):
    fresh = yaml.safe_load(fake_api_config.read_text(encoding="utf-8"))["freshness_s"]
    limits = {**fresh, "kpi_table": 0.001, "evaluation": 0.001}
    api_env(config=edited_config(fake_api_config, freshness_s=limits))
    at = run_app()
    no_exception(at)
    assert any(w.value.startswith("Dernière valeur connue") for w in at.warning)


# -- AppTest on canned FAKE Part 8 payloads (final table, A-06, pairs, series) ------------
ENV = {
    "data_source": "simulation",
    "calibrated_on": "2026-10-08",
    "provisional": True,
    "timestamp": "2026-10-13T08:15:00+02:00",
    "data_age_s": 5.0,
    "stale": False,
    "status": "ok",
    "notice": None,
    "error": None,
}
FAKE_LABELS = [
    {"label": "S0", "scenario": "pitie_fixed", "controller": "fixed", "role": "main"},
    {"label": "S1", "scenario": "pitie_actuated", "controller": "actuated", "role": "main"},
    {"label": "S2", "scenario": "pitie_rules", "controller": "rules", "role": "main"},
    {"label": "S3", "scenario": "pitie_rl_s3", "controller": "rl", "role": "main"},
    {"label": "S4", "scenario": "pitie_rl_s3_nopreempt", "controller": "rl", "role": "exploratory"},
]
FAKE_MEANS = {"S0": 60.0, "S1": 55.0, "S2": 48.0, "S3": 47.0, "S4": 46.0}  # FAKE


def fake_payloads(path: str, params: dict) -> dict | None:
    """FAKE answers in the §4.3 shapes; None = unknown pair (404)."""
    labels = [r["label"] for r in FAKE_LABELS]
    if path == "/scenarios":
        return {
            **ENV,
            "type": "scenario_list",
            "scenarios": [{**r, "family": "main", "runs": []} for r in FAKE_LABELS],
        }
    if path == "/kpi":
        m = FAKE_MEANS[params["scenario"]]
        rel = None if params["scenario"] == "S0" else (m / 60.0 - 1) * 100
        rows = [
            {
                "user": "all",
                "kpi": "person_delay_mean",
                "mean": m,
                "std": 1.0,
                "n": 10,
                "rel_change_vs_s0_pct": rel,
                "p_value": 0.01,
                "worse_flag": False,
            },
            {
                "user": "pedestrian",
                "kpi": "ped_curb_wait_mean__crossing_S",
                "mean": m / 3,
                "std": 0.5,
                "n": 10,
                "rel_change_vs_s0_pct": rel and -rel,
                "p_value": 0.04,
                "worse_flag": params["scenario"] == "S3",
            },
        ]
        return {
            **ENV,
            "type": "kpi_table",
            "label": params["scenario"],
            "source": "evaluation",
            "rows": rows,
        }
    if path == "/evaluation":
        final = [
            {
                "section": "main",
                "user": "all",
                "kpi": "person_delay_mean",
                "label": lab,
                "scenario": lab,
                "ablation": lab == "S4",
                "n": 10,
                "mean": FAKE_MEANS[lab],
                "sd": 1.0,
                "rel_vs_s0_pct": (FAKE_MEANS[lab] / 60.0 - 1) * 100,
                "unit": "s",
                "better_if": "lower",
            }
            for lab in labels
        ] + [
            {
                "section": "activity",
                "user": "junction",
                "kpi": "interventions_per_run",
                "label": lab,
                "scenario": lab,
                "ablation": False,
                "n": 10,
                "mean": 2.0,
                "sd": None,
            }
            for lab in labels
        ]
        comps = [
            {
                "reference": ref,
                "n": 10,
                "mean_ref": FAKE_MEANS[ref],
                "mean_s3": 47.0,
                "hl": -1.0,
                "ci_low": -2.0,
                "ci_high": 0.5,
                "p": 0.1,
                "p_holm": 0.2,
                "outcome": "no significant difference",
                "sentence": f"FAKE sentence vs {ref}.",
            }
            for ref in ("S1", "S2", "S0")
        ]
        a06 = {
            "status": "PASS",
            "problems": [],
            "family": "main",
            "primary": "S1",
            "comparisons": comps,
            "replication_10_6": {
                "c1": {"pass": False},
                "c2": {"pass": False},
                "c3": {"pass": True},
                "c4": {"cited": "fail"},
                "main_solution": "S2",
                "text": "FAKE text.",
            },
            "equity_vs": {"S1": [{"kpi": "FAKE", "rel_pct": 6.0}], "S2": []},
        }
        a05 = {
            "verdict": "PASS",
            "primary": {
                "family": "main",
                "a": "S0",
                "b": "S2",
                "n": 3,
                "hl": -5.0,
                "hl_ci": [-9.0, 1.0],
                "p_one_sided": 0.125,
                "seeds": [1, 2, 3],
                "a_values": [100.0, 110.0, 120.0],
                "b_values": [90.0, 115.0, 100.0],
            },
        }
        return {
            **ENV,
            "type": "evaluation",
            "family": "main",
            "seeds": list(range(1, 11)),
            "labels": FAKE_LABELS,
            "a03": {"text": "4/4", "controllers": ["fixed", "rl"]},
            "a04": {"verdict": "PASS", "runs": 110, "failures": []},
            "a05": a05,
            "a06": a06,
            "replication_10_6": a06["replication_10_6"],
            "hidden_degradation": [],
            "final_table": final,
        }
    if path == "/compare":
        a, b = params["a"], params["b"]
        if (a, b) not in {("S0", "S3"), ("S1", "S3")}:
            return None
        rows = [
            {
                "user_type": "all",
                "kpi": "person_delay_mean",
                "headline": True,
                "better_if": "lower",
                "n_pairs": 10,
                f"{a.lower()}_mean": FAKE_MEANS[a],
                f"{b.lower()}_mean": FAKE_MEANS[b],
                "hl_diff": -1.0,
                "hl_ci_low": -2.0,
                "hl_ci_high": 0.5,
                "rel_change_pct": -2.0,
                "p_value": 0.1,
                "p_holm": 0.2,
                "effect": "same",
                "equity_flag": a == "S1",
            },
        ]
        return {
            **ENV,
            "type": "kpi_pair",
            "a": a,
            "b": b,
            "family": "main",
            "role": "confirmatory",
            "rows": rows,
        }
    if path == "/series":
        pts = [
            {
                "label": lab,
                "t_s": t,
                "mean": FAKE_MEANS[lab] / 10 + t / 600,
                "p10": 1.0,
                "p90": 9.0,
                "n": 10,
            }
            for lab in labels
            for t in (0, 60, 120)
        ]
        return {
            **ENV,
            "type": "kpi_series",
            "metric": "halting_veh",
            "approach": params.get("approach"),
            "step_s": 60,
            "points": pts,
        }
    if path == "/sensitivity":
        return {
            **ENV,
            "type": "sensitivity",
            "rows": [{"setting": "t1", "person_delay_s": 1.0}],
            "source": "FAKE.csv",
        }
    return {**ENV, "type": "x"}


@pytest.fixture
def canned_api(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_fetch(path: str, **params) -> client.Reply:
        calls.append((path, params))
        payload = fake_payloads(path, params)
        if payload is None:
            return client.Reply(None, "error", "unknown pair", 404)
        payload = json.loads(json.dumps(payload))  # a fresh copy, as from the wire
        return client.Reply(payload, client.state_of(payload), payload.get("notice"), 200)

    monkeypatch.setattr(client, "fetch", fake_fetch)
    monkeypatch.setattr(client, "identity", lambda: ("canned", "FAKE"))
    return calls


def test_part8_outputs_render(canned_api, downloads):
    at = run_app()
    no_exception(at)
    text = texts(at)
    assert "Tableau comparatif final (Partie 8)" in text
    assert "A-06 PASS" in text and "FAKE sentence vs S1." in text
    assert "Solution principale : S2" in text
    assert "10 graines d'évaluation (1 à 10)" in text
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["A-06 RL (S3)"] == "PASS" and metrics["A-04 sécurité"] == "PASS"
    assert "exploratoire" in text  # S4's role, dimmed
    assert len(at.get("plotly_chart")) == 1  # Courbes
    # the two FAKE pairs were found by probing /compare, and each has its CSV
    files = downloads(at)
    assert {"Paire S0 vs S3 · confirmatory (CSV)", "Paire S1 vs S3 · confirmatory (CSV)"} <= set(
        files
    )
    assert files["Tableau comparatif final (CSV)"].decode("utf-8-sig").startswith("section,")
    assert all(path.startswith("/") for path, _ in canned_api)


def test_part8_tabs_rerun_on_other_choices(canned_api):
    at = run_app()
    at.radio(key="eq_ref").set_value("S1").run()
    no_exception(at)
    at.selectbox(key="series_approach").set_value("N").run()
    no_exception(at)
    assert any(p == "/series" and q.get("approach") == "N" for p, q in canned_api)


# -- run_mock.py quarantine (D107, Q36) ------------------------------------------------------
@pytest.fixture(scope="module")
def run_mock():
    spec = importlib.util.spec_from_file_location("run_mock", REPO_ROOT / "run_mock.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_mock_is_quarantined(run_mock):
    import httpx

    assert run_mock.PORT == 8001
    html = run_mock.with_fake_banner("<html><BODY class='x'><p>hi</p></body></html>")
    assert html.index("FAUSSES DONNÉES (run_mock.py)") > html.index("<BODY class='x'>")
    server = HTTPServer(("127.0.0.1", 0), run_mock.MockAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        for path in ("/health", "/kpi?scenario=S0", "/eta?trajet=N_S"):
            body = httpx.get(base + path, timeout=5).json()
            assert body["data_source"] == "simulation", path
            assert body["provisional"] is True and body["calibrated_on"] == "2026-10-08", path
        page = httpx.get(base + "/mock", timeout=5).text
        assert "FAUSSES DONNÉES (run_mock.py)" in page
    finally:
        server.shutdown()
        server.server_close()

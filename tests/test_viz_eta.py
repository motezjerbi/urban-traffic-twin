"""ETA evaluation figures render from an eval.json (synthetic, no SUMO, no xgboost), and
the CLI writes all four."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from test_eta_evaluate import _frame, _reference, _with_coverage  # noqa: E402

from traffic_twin.ml.evaluate_eta import evaluate  # noqa: E402
from traffic_twin.viz.eta import FIGURES, main, make_figures  # noqa: E402


def _check_png(path):
    assert path.exists() and path.stat().st_size > 5_000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def ev(eta_cfg) -> dict:
    rng = np.random.default_rng(0)
    df, pred = _with_coverage(_frame(12, 60), 0.8, rng)
    df["trajet"] = np.resize(["N_S", "W_S", "S_N"], len(df))
    df.loc[df["trajet"] == "W_S", "y_zone_s"] = 200.0  # a flagged group
    out = evaluate(df, pred, -0.5, _reference(80.0), eta_cfg, split="val")
    out["model_version"] = "eta1-test0000"
    return json.loads(json.dumps(out))  # as read back from eval.json


def test_every_figure_renders(ev, tmp_path):
    paths = make_figures(ev, tmp_path)
    assert [p.name for p in paths] == list(FIGURES)
    for p in paths:
        _check_png(p)


def test_figures_survive_empty_sections(tmp_path):
    ev = {"split": "val", "a07": {"coverage": None, "n": 0}, "qhat": None}
    for p in make_figures(ev, tmp_path):
        assert p.exists()


def test_cli_writes_the_figures(ev, tmp_path):
    src = tmp_path / "eval.json"
    src.write_text(json.dumps(ev), encoding="utf-8")
    main(["--eval", str(src), "--out", str(tmp_path / "fig")])
    for name in FIGURES:
        _check_png(tmp_path / "fig" / name)
    with pytest.raises(SystemExit, match="evaluate_eta"):
        main(["--eval", str(tmp_path / "none.json"), "--out", str(tmp_path / "x")])


def _captured_coverage_figure(ev, tmp_path, monkeypatch):
    """Draw coverage_by_group.png and return the matplotlib figure it saved."""
    from traffic_twin.viz import eta as viz_eta

    seen = {}
    real = viz_eta.style.savefig

    def capture(fig, path, *a, **k):
        seen["fig"] = fig
        seen["legend"] = [t.get_text() for leg in fig.legends for t in leg.get_texts()] + [
            t.get_text() for sf in fig.subfigs for leg in sf.legends for t in leg.get_texts()
        ]
        return real(fig, path, *a, **k)

    monkeypatch.setattr(viz_eta.style, "savefig", capture)
    viz_eta.plot_coverage_by_group(ev, tmp_path / "coverage_by_group.png")
    return seen


def _group_rows(*groups):
    return [{"group": g, "n": 10, "n_worlds": 2, "coverage": c, "mae": 1.0, "ref_mae": 2.0,
             "width": 3.0, "flag": not 0.75 <= c <= 0.85} for g, c in groups]  # fmt: skip


def test_coverage_figure_draws_every_judged_subgroup(tmp_path, monkeypatch):
    """Review of Parts 5-6: the figure captioned 'per subgroup' left out the incident and
    ambulance breakdowns, the only flagged subgroup among them, while its legend showed a
    flag marker no point used."""
    ev = {
        "split": "test", "a07": {"band": [0.75, 0.85], "population": "realistic", "n": 50},
        "by_trajet": _group_rows(("N_S", 0.8), ("W_S", 0.79)),
        "by_controller": _group_rows(("fixed", 0.82)),
        "by_day_type": _group_rows(("tue_wed_thu_term", 0.81)),
        "by_incident_active": _group_rows(("no", 0.81), ("yes", 0.719)),
        "by_ambulance_present": _group_rows(("no", 0.81), ("yes", 0.765)),
        "by_kind": _group_rows(("realistic", 0.8), ("stress", 0.6)),
    }  # fmt: skip
    seen = _captured_coverage_figure(ev, tmp_path, monkeypatch)
    axes = [ax for sf in seen["fig"].subfigs for ax in sf.axes]

    def title(ax):
        return next((ax.get_title(loc) for loc in ("left", "center") if ax.get_title(loc)), "")

    axes = [ax for ax in axes if title(ax)]
    assert [title(ax) for ax in axes] == [
        "Trajet",
        "Controller",
        "Day type",
        "Active incident",
        "Ambulance",
    ]  # world kind is not judged: not drawn
    incident = axes[3]
    colors = incident.collections[0].get_facecolors()
    assert len(colors) == 2 and tuple(colors[0]) != tuple(colors[1])  # "yes" is flagged red
    assert any("flag" in t for t in seen["legend"])
    for key in ("by_incident_active", "by_ambulance_present"):
        for r in ev[key]:
            r["coverage"], r["flag"] = 0.8, False
    seen = _captured_coverage_figure(ev, tmp_path, monkeypatch)
    assert not any("flag" in t for t in seen["legend"])  # no unused legend entry

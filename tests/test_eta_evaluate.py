"""ETA evaluation (EF-14, A-07, D50): world bootstrap, the PASS/FAIL rule at the band
edges and the MAE CI rule, the evaluation of a synthetic split end to end (eval json,
page), and the refusal of a test evaluation without a matching freeze."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from test_eta_train import linear_loader, make_fake_bundle, write_dataset

from traffic_twin.ml.evaluate_eta import (
    GROUPS,
    POST_HOC_KEYS,
    a07_pass,
    evaluate,
    main,
    percentile_ci,
    reference_sensitivity,
    render,
    render_only,
    run_evaluation,
    unfinished_cars,
    world_bootstrap,
)
from traffic_twin.ml.train_eta import fit_reference, freeze, load_split


def _frame(n_worlds: int, per_world: int, kind: str = "realistic") -> pd.DataFrame:
    rows = []
    for w in range(n_worlds):
        for i in range(per_world):
            rows.append(
                {
                    "world_seed": 2900 + w,
                    "run_id": f"r{w}-{i % 3}",
                    "trajet": "N_S",
                    "controller": ("fixed", "actuated", "rules")[i % 3],
                    "day_type": "tue_wed_thu_term",
                    "kind": kind,
                    "insert_t": 100.0,
                }  # fmt: skip
            )
    return pd.DataFrame(rows)


def _reference(mean: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "route": ["N_S"],
            "hour": [-1],
            "day_type": ["*"],
            "mean_s": [mean],
            "p10_s": [mean - 20],
            "p90_s": [mean + 20],
            "n": [100],
        }  # fmt: skip
    )


def _with_coverage(df: pd.DataFrame, inside_share: float, rng) -> tuple[pd.DataFrame, np.ndarray]:
    """Predicted interval [40, 60] (median 50) and targets inside it for exactly
    `inside_share` of the rows of every world."""
    n = len(df)
    pred = np.tile([40.0, 50.0, 60.0], (n, 1))
    y = np.empty(n)
    for _, idx in df.groupby("world_seed").groups.items():
        idx = np.asarray(idx)
        k = int(round(inside_share * len(idx)))
        y[idx[:k]] = rng.integers(45, 56, k)
        y[idx[k:]] = rng.choice([30.0, 70.0], len(idx) - k)
    return df.assign(y_zone_s=y), pred


def test_world_bootstrap_ci_contains_the_known_coverage():
    rng = np.random.default_rng(0)
    worlds = np.repeat(np.arange(100), 200)
    p = rng.uniform(0.7, 0.9, 100)  # per-world coverage, mean 0.8
    inside = rng.uniform(size=len(worlds)) < p[worlds]
    est = world_bootstrap(inside.astype(float), np.ones(len(worlds)), worlds, 2000, 0)
    lo, hi = percentile_ci(est, 0.95)
    assert lo < inside.mean() < hi and lo < 0.8 < hi
    assert hi - lo < 0.05
    again = world_bootstrap(inside.astype(float), np.ones(len(worlds)), worlds, 2000, 0)
    assert np.array_equal(est, again)  # rng seed 0: reproducible


def test_bootstrap_resamples_worlds_not_rows():
    # two worlds, coverage 1 and 0: a row bootstrap would give a tight CI around 0.5
    worlds = np.repeat([0, 1], 500)
    inside = np.r_[np.ones(500), np.zeros(500)]
    est = world_bootstrap(inside, np.ones(1000), worlds, 2000, 0)
    assert set(np.round(est, 6)) == {0.0, 0.5, 1.0}


@pytest.mark.parametrize(
    "coverage,ci_hi,expected",
    [
        (0.75, -1.0, True),
        (0.85, -1.0, True),
        (0.80, -0.001, True),
        (0.7499, -1.0, False),
        (0.8501, -1.0, False),
        (0.80, 0.0, False),  # the CI must lie entirely below 0
        (0.80, 2.0, False),
        (0.80, None, False),
    ],
)
def test_a07_rule_edges(coverage, ci_hi, expected):
    assert a07_pass(coverage, [-5.0, ci_hi], (0.75, 0.85), True) is expected


def test_a07_exclusive_band():
    assert not a07_pass(0.75, [-5.0, -1.0], (0.75, 0.85), inclusive=False)


@pytest.mark.parametrize("share,expected", [(0.8, True), (0.75, True), (0.85, True), (0.7, False)])
def test_evaluate_known_coverage(eta_cfg, share, expected):
    rng = np.random.default_rng(1)
    df, pred = _with_coverage(_frame(20, 60), share, rng)
    ev = evaluate(df, pred, 0.0, _reference(80.0), eta_cfg)
    a = ev["a07"]
    assert a["coverage"] == pytest.approx(share)
    assert a["n"] == 1200 and a["n_worlds"] == 20 and a["n_runs"] == 60
    assert a["mae_diff_ci"][1] < 0  # reference mean 80 is far off
    assert a["pass"] is expected
    assert ev["width"] == 20.0 and ev["qhat"] == 0.0
    assert ev["stress"] == {"n": 0}
    assert [r["group"] for r in ev["by_controller"]] == ["actuated", "fixed", "rules"]
    assert all(f"by_{g}" in ev for g in GROUPS)
    assert ev["per_run_coverage"]["summary"]["n_runs"] == 60


def test_mae_rule_fails_when_the_model_is_not_better(eta_cfg):
    rng = np.random.default_rng(2)
    df, pred = _with_coverage(_frame(20, 60), 0.8, rng)
    ev = evaluate(df, pred, 0.0, _reference(50.0), eta_cfg)  # reference = model median
    assert ev["a07"]["coverage"] == pytest.approx(0.8)
    assert ev["a07"]["mae_diff"] == 0.0 and ev["a07"]["mae_diff_ci"][1] >= 0
    assert ev["a07"]["pass"] is False


def test_stress_worlds_are_reported_not_judged(eta_cfg):
    rng = np.random.default_rng(3)
    real, pr = _with_coverage(_frame(10, 30), 0.8, rng)
    stress_rows = _frame(5, 30, "stress").assign(world_seed=lambda d: d.world_seed + 50)
    stress, ps = _with_coverage(stress_rows, 0.4, rng)
    df = pd.concat([real, stress], ignore_index=True)
    ev = evaluate(df, np.vstack([pr, ps]), 0.0, _reference(90.0), eta_cfg)
    assert ev["a07"]["coverage"] == pytest.approx(0.8) and ev["a07"]["pass"]
    assert ev["stress"]["coverage"] == pytest.approx(0.4) and ev["stress"]["n_worlds"] == 5
    kinds = {r["group"]: r for r in ev["by_kind"]}
    assert kinds["stress"]["flag"] and not kinds["realistic"]["flag"]
    assert not any(f.startswith("kind=") for f in ev["flags"])


def test_subgroup_flags(eta_cfg):
    rng = np.random.default_rng(4)
    df, pred = _with_coverage(_frame(20, 60), 0.8, rng)
    df.loc[df["controller"] == "rules", "y_zone_s"] = 200.0  # rules never covered
    ev = evaluate(df, pred, 0.0, _reference(80.0), eta_cfg)
    flagged = {r["group"] for r in ev["by_controller"] if r["flag"]}
    assert "rules" in flagged
    assert any(f.startswith("controller=rules") for f in ev["flags"])


# -- end to end with a fake bundle -------------------------------------------------------
@pytest.fixture
def synthetic_setup(tmp_path, eta_cfg):
    ds = write_dataset(
        tmp_path / "ds",
        {"train": [2000, 2003], "val": [2300, 2303], "calib": [2351, 2354], "test": [2900, 2902]},
        n=60,
    )  # fmt: skip
    models = tmp_path / "models"
    bundle = make_fake_bundle(models, eta_cfg, load_split(ds, "train"))
    return ds, models, bundle


def test_val_evaluation_and_page(synthetic_setup, eta_cfg, tmp_path):
    ds, models, bundle = synthetic_setup
    ev, meta, manifest = run_evaluation(eta_cfg, "val", ds, bundle, models, loader=linear_loader)
    assert ev["split"] == "val" and ev["freeze"] is None
    assert ev["model_version"] == bundle.name and ev["dataset_id"] == "eta1-synthetic"
    assert ev["n_rows"] == 2 * 3 * 60
    assert set(ev["trajet_variant"]["by_trajet"]) <= {"N_S", "N_W", "S_N", "S_W", "W_N", "W_S"}
    json.dumps(ev)  # serialisable
    text = render(ev, meta, manifest, eta_cfg, "figures/eta")
    for needle in (
        "A-07",
        "EF-14",
        "Development split `val`",
        "## 9. Limitations",
        "simulated",
        "provisional",
        "ASSUMED",
        "figures/eta/calibration_curve.png",
    ):
        assert needle in text  # fmt: skip
    for i in range(1, 10):
        assert f"## {i}. " in text


def test_test_split_refused_without_or_with_a_stale_freeze(synthetic_setup, eta_cfg, capsys):
    ds, models, bundle = synthetic_setup
    args = ["--split", "test", "--dataset", str(ds), "--models", str(models),
            "--bundle", str(bundle)]  # fmt: skip
    assert main(args) == 1
    assert "no freeze.json" in capsys.readouterr().err
    with pytest.raises(ValueError, match="no freeze.json"):
        run_evaluation(eta_cfg, "test", ds, bundle, models, loader=linear_loader)
    # a freeze made for another bundle sha (the bundle changed afterwards)
    clean = write_dataset(ds.parent / "clean", {"train": [2000]}, n=5)
    freeze(bundle, models, clean)
    meta = bundle / "meta.json"
    meta.write_text(meta.read_text() + " ")
    assert main(args) == 1
    assert "changed since the freeze" in capsys.readouterr().err
    assert not (bundle / "eval.json").exists()


def test_test_split_with_a_matching_freeze(synthetic_setup, eta_cfg, tmp_path):
    ds, models, bundle = synthetic_setup
    clean = write_dataset(tmp_path / "clean", {"train": [2000]}, n=5)  # no test worlds yet
    freeze(bundle, models, clean)
    ev, _, _ = run_evaluation(eta_cfg, "test", ds, bundle, models, loader=linear_loader)
    assert ev["freeze"]["model_version"] == bundle.name
    assert ev["a07"]["n_worlds"] == 1  # world 2902 is a stress world (2902 % 3 == 1)


def test_worlds_override_for_development_only(synthetic_setup, eta_cfg):
    ds, models, bundle = synthetic_setup
    ev, _, _ = run_evaluation(
        eta_cfg, "val", ds, bundle, models, loader=linear_loader, worlds=[2000, 2003]
    )
    assert ev["worlds_override"] == [2000, 2003] and ev["n_rows"] == 2 * 3 * 60
    with pytest.raises(ValueError, match="without a worlds override"):
        run_evaluation(eta_cfg, "test", ds, bundle, models, loader=linear_loader, worlds=[2900])


# -- post-hoc disclosures (review of Parts 5-6) --------------------------------------------
def write_runs(ds, rows: list[dict]) -> None:
    """A `runs.parquet` with the censoring columns of `eta_dataset`."""
    base = {"controller": "fixed", "drain_timeout": False, "n_unfinished_cars": 0,
            "n_missing_exit": 0}  # fmt: skip
    pd.DataFrame([{**base, **r} for r in rows]).to_parquet(ds / "runs.parquet", index=False)


def test_unfinished_cars_are_counted_per_split_and_kind(tmp_path):
    """Cars still in the network at the drain cut have no sample: `n_missing_exit` (and
    the D39 `dropped` counts) stay 0, so they must be reported from runs.parquet."""
    assert unfinished_cars(tmp_path) is None
    write_runs(tmp_path, [
        {"split": "test", "kind": "realistic", "drain_timeout": True, "n_unfinished_cars": 29},
        {"split": "test", "kind": "realistic", "controller": "rules"},
        {"split": "test", "kind": "stress", "drain_timeout": True, "n_unfinished_cars": 300,
         "controller": "actuated"},
        {"split": "test", "kind": "stress", "drain_timeout": True, "n_unfinished_cars": 6},
        {"split": "train", "kind": "stress", "drain_timeout": True, "n_unfinished_cars": 2},
    ])  # fmt: skip
    u = unfinished_cars(tmp_path)
    assert u["cars"] == 337 and u["drain_timeout_runs"] == 4
    real, stress = u["by_split"]["test"]["realistic"], u["by_split"]["test"]["stress"]
    assert (real["cars"], real["runs"], real["drain_timeout_runs"], real["missing_exit"]) == (
        29, 2, 1, 0,
    )  # fmt: skip
    assert stress["cars"] == 306 and stress["cars_by_controller"] == {"actuated": 300, "fixed": 6}
    assert u["by_split"]["train"]["stress"]["cars"] == 2


def _sens_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    def rows(kind, day_type, y, n=10, world=2000):
        return pd.DataFrame({
            "world_seed": world, "trajet": "S_W", "insert_t": 2000.0, "day_type": day_type,
            "kind": kind, "y_zone_s": float(y),
        }, index=range(n))  # fmt: skip

    train = pd.concat(
        [rows("realistic", "tue_wed_thu_term", 50), rows("stress", "tue_wed_thu_term", 150),
         rows("realistic", "mon_fri_term", 30)], ignore_index=True,
    )  # fmt: skip
    test = pd.concat(
        [rows("realistic", "tue_wed_thu_term", 52, world=2900),
         rows("realistic", "mon_fri_term", 30, world=2901)], ignore_index=True,
    )  # fmt: skip
    return train, test


def test_reference_sensitivity_exposes_the_stress_mix():
    """Every stress world is labelled tue_wed_thu_term: a reference fitted on all train
    worlds inflates that day type's cells (and the degraded table); the realistic-only
    refit shows by how much, on the evaluated rows, without any model output."""
    train, test = _sens_frames()
    shipped = fit_reference(train)
    sens = reference_sensitivity(train, test, shipped, mae=1.0, used="all")
    assert sens["train_stress_share_by_day_type"] == {"mon_fri_term": 0.0, "tue_wed_thu_term": 0.5}
    cells = {(r["day_type"], r["hour"]): r for r in sens["table_shift"]}
    assert set(cells) == {("tue_wed_thu_term", 8), ("*", -1)}  # mon_fri_term does not move
    assert cells[("tue_wed_thu_term", 8)]["d_mean_s"] == 50.0  # 100 shipped vs 50 realistic
    assert sens["max_shift"]["d_mean_s"] == 50.0
    assert sens["shipped"]["ref_mae"] == pytest.approx((48 + 0) / 2)
    assert sens["alternative"]["ref_mae"] == pytest.approx((2 + 0) / 2)
    assert sens["alternative"]["mae_diff"] == pytest.approx(0.0)
    assert sens["by_day_type"]["tue_wed_thu_term"]["ref_mae_alt"] == 2.0
    same = reference_sensitivity(train, test, shipped, 1.0, used="realistic")
    assert "alternative" not in same and "note" in same


def test_post_hoc_keys_and_render_only(synthetic_setup, eta_cfg, tmp_path, monkeypatch, capsys):
    """run_evaluation ends with the post-hoc keys; `--render-only` adds exactly those keys
    to an eval json written before they existed (never re-evaluating, never loading the
    model), and the page reports the censoring, the reference mix and its sensitivity."""
    ds, models, bundle = synthetic_setup
    write_runs(ds, [
        {"split": "val", "kind": "realistic", "drain_timeout": True, "n_unfinished_cars": 4},
        {"split": "val", "kind": "stress", "drain_timeout": True, "n_unfinished_cars": 9},
    ])  # fmt: skip
    ev, _, _ = run_evaluation(eta_cfg, "val", ds, bundle, models, loader=linear_loader)
    assert list(ev)[-len(POST_HOC_KEYS) :] == list(POST_HOC_KEYS)
    assert ev["dropped"]["missing_exit"] == 0 and ev["unfinished"]["cars"] == 13
    assert ev["reference_sensitivity"]["shipped_population"] == "all"
    assert ev["reference_sensitivity"]["shipped"]["ref_mae"] == pytest.approx(ev["a07"]["ref_mae"])
    # an eval json from before the review: same content without the post-hoc keys
    old = {k: v for k, v in ev.items() if k not in POST_HOC_KEYS}
    path = bundle / "eval_val.json"
    path.write_text(json.dumps(old, indent=1) + "\n", encoding="utf-8")

    def no_model(*a, **k):
        raise AssertionError("render-only must not load the model")

    monkeypatch.setattr("traffic_twin.ml.predict.load_bundle", no_model)
    docs, figs = tmp_path / "docs" / "eta.md", tmp_path / "docs" / "figures" / "eta"
    args = ["--split", "val", "--dataset", str(ds), "--models", str(models), "--bundle",
            str(bundle), "--render-only", "--docs", str(docs), "--figures", str(figs)]  # fmt: skip
    assert main(args) == 0
    back = json.loads(path.read_text(encoding="utf-8"))
    assert back == json.loads(json.dumps(ev))  # identical to a fresh evaluation
    assert list(back) == list(ev)
    assert "added post-hoc keys" in capsys.readouterr().out
    text = docs.read_text(encoding="utf-8")
    for needle in ("Censored, not in the dataset", "realistic 4 cars", "stress 9 cars",
                   "realistic and stress", "Sensitivity (post hoc", "realistic-only fit",
                   "figures/eta/coverage_by_group.png"):  # fmt: skip
        assert needle in text, needle
    assert (figs / "coverage_by_group.png").is_file()
    stamp = path.stat().st_mtime_ns
    assert main(args) == 0  # nothing missing any more: the json is not rewritten
    assert path.stat().st_mtime_ns == stamp
    assert "added post-hoc keys" not in capsys.readouterr().out


def test_render_only_test_split_still_needs_the_freeze(synthetic_setup, eta_cfg, tmp_path, capsys):
    ds, models, bundle = synthetic_setup
    (bundle / "eval.json").write_text(json.dumps({"split": "test"}), encoding="utf-8")
    args = ["--split", "test", "--dataset", str(ds), "--models", str(models), "--bundle",
            str(bundle), "--render-only"]  # fmt: skip
    assert main(args) == 1
    assert "no freeze.json" in capsys.readouterr().err
    with pytest.raises(FileNotFoundError, match="evaluate the split first"):
        render_only(eta_cfg, "val", ds, bundle, models, tmp_path / "none.json")


def test_page_stress_censoring_bound(eta_cfg):
    """Section 7 says how far the censored stress trips could pull the stress coverage."""
    rng = np.random.default_rng(0)
    frame = pd.concat([_frame(4, 30), _frame(2, 30, "stress")], ignore_index=True)
    frame["world_seed"] = np.repeat(np.arange(6) + 2900, 30)
    df, pred = _with_coverage(frame, 0.8, rng)
    ev = evaluate(df, pred, 0.0, _reference(50.0), eta_cfg, split="test")
    ev["a07"]["n_runs"] = 3
    ev["unfinished"] = {"cars": 60, "drain_timeout_runs": 2, "by_split": {"test": {
        "realistic": {"cars": 0, "runs": 12, "drain_timeout_runs": 0},
        "stress": {"cars": 60, "runs": 6, "drain_timeout_runs": 2}}}}  # fmt: skip
    text = render(ev, {}, {}, eta_cfg)
    low = ev["stress"]["coverage"] * ev["stress"]["n"] / (ev["stress"]["n"] + 60)
    assert f"it would be {low:.4f}" in text
    assert "stress 60 cars (2 of 6 runs with a drain timeout)" in text

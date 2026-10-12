"""Paired S0 vs S1 comparison: identical runs, a uniform degradation (equity flag), Holm
over the headline KPIs only, untested counts, absolute differences in the markdown, the
pairing checks (seeds, demand, stale runs of an older config) and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_twin.config import load_scenario
from traffic_twin.evaluation.compare import (
    ComparisonError,
    better_if,
    check_pairing,
    compare,
    incomplete_rows,
    main,
    markdown,
    provenance_path,
    unit,
    worse_than,
)
from traffic_twin.evaluation.runs import expected_hashes, load_runs
from traffic_twin.evaluation.stats import holm

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADLINE = [
    ["car", "time_loss_mean"],
    ["car", "wait_p95"],
    ["bus", "delay_mean"],
    ["pedestrian", "ped_wait_single_max"],
]
SEEDS = list(range(1, 11))


def kpis(scale: dict[tuple[str, str], float] | None = None) -> pd.DataFrame:
    """Ten seeds of KPIs; `scale` multiplies some of them (another controller)."""
    rng = np.random.default_rng(0)
    base = {
        ("car", "time_loss_mean"): 40.0,
        ("car", "wait_p95"): 60.0,
        ("car", "count"): 900.0,
        ("bus", "delay_mean"): 30.0,
        ("pedestrian", "ped_wait_single_max"): 80.0,
        ("ambulance", "time_loss_mean"): 50.0,
    }
    noise = {k: rng.uniform(0.9, 1.1, len(SEEDS)) for k in base}
    rows = [
        {
            "seed": s,
            "user_type": u,
            "kpi": k,
            "value": v * noise[(u, k)][i] * (scale or {}).get((u, k), 1.0),
        }
        for (u, k), v in base.items()
        for i, s in enumerate(SEEDS)
    ]
    return pd.DataFrame(rows)


def test_identical_runs_give_p_1_and_no_flags():
    t = compare(kpis(), kpis(), HEADLINE).set_index(["user_type", "kpi"])
    tested = t[t["p_value"].notna()]
    assert (tested["p_value"] == 1.0).all() and not t["equity_flag"].any()
    assert (t["diff_mean"] == 0).all() and set(t["effect"]) == {"same"}
    assert not t["significant"].any()


def test_uniform_6_percent_degradation_sets_the_equity_flag():
    worse = kpis({("car", "time_loss_mean"): 1.06, ("ambulance", "time_loss_mean"): 0.9})
    t = compare(kpis(), worse, HEADLINE, equity_pct=5.0).set_index(["user_type", "kpi"])
    car = t.loc[("car", "time_loss_mean")]
    assert car["p_value"] == pytest.approx(2 / 1024)
    assert car["rel_change_pct"] == pytest.approx(6.0)
    assert car["equity_flag"] and car["equity_flag_significant"] and car["effect"] == "worse"
    assert not car["underpowered"]
    amb = t.loc[("ambulance", "time_loss_mean")]
    assert amb["effect"] == "better" and not amb["equity_flag"] and not amb["headline"]
    assert amb["p_value"] == pytest.approx(2 / 1024) and np.isnan(amb["p_holm"])


def test_holm_family_is_the_headline_only_and_counts_are_not_tested():
    worse = kpis(
        {("car", "time_loss_mean"): 1.06, ("bus", "delay_mean"): 1.2, ("car", "count"): 0.9}
    )
    t = compare(kpis(), worse, HEADLINE)
    assert t["kpi"].tolist()[:4] == [k for _, k in HEADLINE]  # headline first, in order
    head = t[t["headline"]]
    assert head["p_holm"].tolist() == pytest.approx(holm(head["p_value"]).tolist())
    count = t.set_index("kpi").loc["count"]
    assert np.isnan(count["p_value"]) and np.isnan(count["p_holm"])
    assert count["better_if"] == "higher" and count["equity_flag"]  # 10 % fewer trips
    assert better_if("dwell_mean") == "n/a" and better_if("ped_wait_single_max") == "lower"
    md = markdown(t, "pitie_fixed", "pitie_actuated")
    assert md.count("\n| ") == 1 + len(HEADLINE)  # header row + one row per headline KPI
    assert "count" in md  # flagged non-headline KPI listed below the table


def test_qualified_kpis_take_the_direction_and_unit_of_their_base():
    """Part 4 KPIs `<base>__<qualifier>` (per arm, flow, line, crossing): direction, test
    and unit come from the base name; the new rates and counts have their own sense."""
    assert better_if("time_loss_mean__arm_N") == "lower" and unit("delay_mean__flow_91_E") == "s"
    assert better_if("count__arm_N") == "higher"
    assert better_if("throughput_vph") == "higher" and unit("throughput_vph") == "veh/h"
    assert better_if("crossing_complete_rate") == "higher" and unit("crossing_complete_rate") == ""
    assert unit("crossing_red_entries") == "" and unit("box_foe_onsets") == ""
    assert better_if("crossing_red_entries") == "lower" and unit("exit_blocked_s") == "s"
    assert better_if("dwell_mean__line_91") == "n/a"
    base = pd.concat([_rows("car", "count__arm_N", [100.0] * 10),
                      _rows("car", "time_loss_mean__arm_N", np.arange(10.0) + 50)])  # fmt: skip
    other = base.assign(value=base["value"] * 1.1)
    t = compare(base, other, []).set_index("kpi")
    assert np.isnan(t.loc["count__arm_N", "p_value"])  # counts stay untested
    arm = t.loc["time_loss_mean__arm_N"]
    assert arm["effect"] == "worse" and arm["equity_flag"] and arm["better_if"] == "lower"


def _rows(user: str, kpi: str, values) -> pd.DataFrame:
    rows = zip(SEEDS, values, strict=True)
    return pd.DataFrame([{"seed": s, "user_type": user, "kpi": kpi, "value": v} for s, v in rows])


def test_markdown_gives_absolute_differences_next_to_relative_changes():
    """A near-zero baseline (bus insertion delay 0.15 s) must not read as a large loss:
    the headline rows and the flagged lines carry S1 - S0 with its CI, in seconds."""
    car0 = [40.0 + 0.1 * s for s in SEEDS]
    base = pd.concat(
        [
            _rows("car", "time_loss_mean", car0),
            _rows("bus", "depart_delay_mean", [0.15] * 10),
            _rows("car", "stops", [1.0] * 10),
        ]
    )
    other = pd.concat(
        [
            _rows("car", "time_loss_mean", [v + 2.4 for v in car0]),
            _rows("bus", "depart_delay_mean", [0.19] * 10),
            _rows("car", "stops", [1.2] * 10),
        ]
    )
    t = compare(base, other, [["car", "time_loss_mean"]])
    md = markdown(t, "pitie_fixed", "pitie_actuated")
    assert "S1 - S0 [95 % CI]" in md
    car = next(line for line in md.splitlines() if "`time_loss_mean`" in line)
    assert "| +2.4 [2.4, 2.4] s | 5.9 % |" in car
    bus = next(line for line in md.splitlines() if "`depart_delay_mean`" in line)
    assert "0.150 → 0.190 s, S1 - S0 +0.040 [0.040, 0.040] s (+26.7 %, p 0.0020)" in bus
    stops = next(line for line in md.splitlines() if "`stops`" in line)
    assert "1.00 → 1.20, S1 - S0 +0.20 [0.20, 0.20] (+20.0 %" in stops  # a count: no unit


def _meta(seeds, demand_hash="d1", config_hash="c1"):
    return {s: {"seed": s, "demand_hash": demand_hash, "config_hash": config_hash} for s in seeds}


def test_pairing_checks():
    assert check_pairing(_meta(SEEDS), _meta(SEEDS), SEEDS) == "d1"
    with pytest.raises(ComparisonError, match="seed"):
        check_pairing(_meta(SEEDS), _meta(SEEDS[:-1]), SEEDS)
    with pytest.raises(ComparisonError, match="demand_hash"):
        check_pairing(_meta(SEEDS), _meta(SEEDS, "d2"), SEEDS)
    with pytest.raises(ComparisonError, match="no demand_hash"):
        check_pairing(_meta(SEEDS), _meta(SEEDS, None), SEEDS)
    mixed = {**_meta(SEEDS), 3: {"seed": 3, "demand_hash": "d1", "config_hash": "c0"}}
    with pytest.raises(ComparisonError, match="baseline runs differ in config_hash"):
        check_pairing(mixed, _meta(SEEDS), SEEDS)


def test_pairing_refuses_runs_of_an_older_config():
    """The runs agree with each other but not with the current configs (a program or a
    turn-share change since they ran): they are never compared."""
    now = {"config_hash": "c1", "demand_hash": "d1"}
    assert check_pairing(_meta(SEEDS), _meta(SEEDS), SEEDS, now, now) == "d1"
    with pytest.raises(ComparisonError, match=r"baseline runs are stale.*seed 1: config_hash c0"):
        check_pairing(_meta(SEEDS, config_hash="c0"), _meta(SEEDS), SEEDS, now, now)
    new_demand = {"config_hash": "c1", "demand_hash": "d2"}
    with pytest.raises(ComparisonError, match="demand_hash d1 != current d2"):
        check_pairing(_meta(SEEDS), _meta(SEEDS), SEEDS, new_demand, new_demand)
    with pytest.raises(ComparisonError, match="do not share one demand"):
        check_pairing(_meta(SEEDS), _meta(SEEDS, "d2"), SEEDS, now, new_demand)


def write_runs(
    results: Path,
    scenario: str,
    table: pd.DataFrame,
    demand_hash: str = "d1",
    config_hash: str = "c1",
    seeds: list[int] | None = None,
) -> None:
    for seed, g in table.groupby("seed"):
        if seeds is not None and seed not in seeds:
            continue
        d = results / f"{scenario}_seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        g.assign(run_id="r", scenario=scenario).to_csv(d / "kpi_run.csv", index=False)
        meta = {
            "scenario": scenario,
            "seed": int(seed),
            "demand_hash": demand_hash,
            "config_hash": config_hash,
        }
        (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")


def write_current(results: Path, config: Path, table: pd.DataFrame) -> dict:
    """Runs of `config` that carry the current config's hashes."""
    cfg = load_scenario(config)
    h = expected_hashes(cfg)
    write_runs(results, cfg["scenario"], table, h["demand_hash"], h["config_hash"])
    return h


def test_load_runs_reads_existing_seeds_only(tmp_path):
    write_runs(tmp_path, "a", kpis())
    runs = load_runs(tmp_path, "a", [1, 2, 99])
    assert runs.seeds == [1, 2] and sorted(runs.meta) == [1, 2]
    assert set(runs.kpis["seed"]) == {1, 2}


S0 = REPO_ROOT / "configs/pitie_fixed.yaml"
S1 = REPO_ROOT / "configs/pitie_actuated.yaml"


def _cli(results: Path, figures: Path, s0: Path = S0) -> list[str]:
    return [
        "--baseline-config", str(s0), "--scenario-config", str(S1), "--results", str(results),
        "--figures", str(figures),
    ]  # fmt: skip


def test_cli_writes_csv_markdown_provenance_and_figure(tmp_path, capsys):
    results, figures = tmp_path / "results", tmp_path / "fig"
    h0 = write_current(results, S0, kpis())
    h1 = write_current(results, S1, kpis({("car", "time_loss_mean"): 0.9}))
    main(_cli(results, figures))
    csv = results / "comparison" / "pitie_fixed__vs__pitie_actuated.csv"
    t = pd.read_csv(csv)
    assert t["n_pairs"].eq(10).all()
    car = t.set_index(["user_type", "kpi"]).loc[("car", "time_loss_mean")]
    assert car["effect"] == "better" and car["p_value"] == pytest.approx(2 / 1024)
    assert (results / "comparison" / "pitie_fixed__vs__pitie_actuated.md").is_file()
    prov = json.loads(provenance_path(csv).read_text(encoding="utf-8"))
    assert prov["seeds"] == SEEDS
    assert prov["baseline"] == {"scenario": "pitie_fixed", "config": S0.as_posix(), **h0}
    assert prov["scenario"]["config_hash"] == h1["config_hash"]
    png = figures / "comparison_pitie_fixed_pitie_actuated.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "time_loss_mean" in capsys.readouterr().out

    other = tmp_path / "other.yaml"
    other.write_text(f"extends: {S0.as_posix()}\nscenario: pitie_other\n", encoding="utf-8")
    write_runs(results, "pitie_other", kpis(), demand_hash="d2")  # not the same demand
    with pytest.raises(SystemExit, match="demand_hash"):
        main(_cli(results, figures, other))


def test_stale_runs_are_rerun_or_refused_never_reused(tmp_path, monkeypatch):
    """After a program change (fixed.green_s HOP 36 -> 30) the stored S0 runs carry the
    old config_hash: without --run-missing the CLI refuses and names the stale seeds; with
    it, every stale seed is rerun before the comparison (the old runs are never reused)."""
    results, figures = tmp_path / "results", tmp_path / "fig"
    changed = tmp_path / "changed_fixed.yaml"
    changed.write_text(
        f"extends: {S0.as_posix()}\nfixed:\n  green_s: {{HOP: 30}}\n", encoding="utf-8"
    )
    old = expected_hashes(load_scenario(S0))
    now = expected_hashes(load_scenario(changed))
    assert old["config_hash"] != now["config_hash"] and old["demand_hash"] == now["demand_hash"]
    write_runs(results, "pitie_fixed", kpis(), old["demand_hash"], old["config_hash"])
    write_current(results, S1, kpis({("car", "time_loss_mean"): 0.9}))
    args = _cli(results, figures, changed)

    with pytest.raises(SystemExit) as err:
        main(args)
    msg = str(err.value)
    assert "baseline runs are stale" in msg and "seed 1: config_hash" in msg
    assert f"current {now['config_hash'][:12]}" in msg
    assert not (results / "comparison").exists()  # nothing published

    import traffic_twin.sim.run_scenario as rs

    rerun: list[int] = []
    fresh = kpis({("car", "time_loss_mean"): 1.2})

    def fake_run(cfg, seed, out_root, gui=False, config_path=None):
        rerun.append(seed)  # a new run of the changed config: its hashes, other KPIs
        h = expected_hashes(cfg)
        write_runs(out_root, cfg["scenario"], fresh, h["demand_hash"], h["config_hash"], [seed])

    monkeypatch.setattr(rs, "run", fake_run)
    main([*args, "--run-missing"])
    assert rerun == SEEDS  # every stale S0 seed; no S1 seed (they are current)
    meta = json.loads((results / "pitie_fixed_seed1" / "run.json").read_text(encoding="utf-8"))
    assert meta["config_hash"] == now["config_hash"]
    t = pd.read_csv(results / "comparison" / "pitie_fixed__vs__pitie_actuated.csv")
    car = t.set_index(["user_type", "kpi"]).loc[("car", "time_loss_mean")]
    expected = fresh.query("user_type == 'car' and kpi == 'time_loss_mean'")["value"].mean()
    assert car["s0_mean"] == pytest.approx(expected)  # from the reruns, not the old runs


# -- review fixes ----------------------------------------------------------------------
def test_a_zero_baseline_is_flagged_on_any_move_in_the_bad_direction():
    """Relative change is undefined on a 0 baseline: 0 -> 0.1 red entries was never
    flagged before (rel NaN)."""
    base = _rows("pedestrian", "crossing_red_entries", [0.0] * 10)
    worse = _rows("pedestrian", "crossing_red_entries", [0.0] * 9 + [1.0])
    t = compare(base, worse, [])
    assert np.isnan(t.loc[0, "rel_change_pct"]) and t.loc[0, "equity_flag"]
    assert not compare(base, base, []).loc[0, "equity_flag"]
    rate0 = _rows("pedestrian", "crossing_complete_rate", [0.0] * 10)
    up = _rows("pedestrian", "crossing_complete_rate", [0.1] * 10)
    assert not compare(rate0, up, []).loc[0, "equity_flag"]  # higher is better
    assert worse_than(0.0, 0.1, float("nan"), -1, 5.0)
    assert not worse_than(0.0, 0.1, float("nan"), 0, 5.0)  # neutral KPI
    assert worse_than(10.0, 10.6, 6.0, -1, 5.0) and not worse_than(10.0, 10.4, 4.0, -1, 5.0)


def test_incomplete_rows_lists_kpis_paired_on_fewer_seeds():
    a = kpis()
    b = a[~((a["user_type"] == "ambulance") & (a["seed"] == 3))]
    t = compare(a, b, HEADLINE)
    short = incomplete_rows(t, len(SEEDS))
    assert list(short["kpi"]) == ["time_loss_mean"] and list(short["user_type"]) == ["ambulance"]
    assert int(short["n_pairs"].iloc[0]) == 9
    assert incomplete_rows(compare(a, a, HEADLINE), len(SEEDS)).empty

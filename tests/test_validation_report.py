"""Validation report end to end on synthetic run folders (no SUMO, no data/processed):
every table, the summary, the figures and a docs page with every section; stale runs and
out-of-date comparisons are never published; capacity, pre-registration, dated changes,
calibration status and limitations come from the configs and the runs."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from traffic_twin.config import load_scenario  # noqa: E402
from traffic_twin.evaluation.compare import compare, provenance, provenance_path  # noqa: E402
from traffic_twin.evaluation.runs import expected_hashes  # noqa: E402
from traffic_twin.validation.report import SECTIONS, main  # noqa: E402
from traffic_twin.validation.signal import link_intervals  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO = REPO_ROOT / "configs/pitie_fixed.yaml"
S1 = REPO_ROOT / "configs/pitie_actuated.yaml"
V7 = [
    (32, "rrGrrrGGrrrrGrGG"), (1, "rrGrrrGGrrrrrrGr"), (3, "rryrrrGGrrrrrrGr"),
    (5, "rrrrrrGGrrrrrrGr"), (6, "rrrrrGGGGrrrrrGr"), (3, "rrrrryyyyrrrrrGr"),
    (7, "rrrrrrrrrrrrrrGr"), (18, "GGrrrrrrrrgGGGGr"), (3, "GGrrrrrrrryyrGrr"),
    (4, "GGrrrrrrrrrrrGrr"), (6, "GGrGrrrrrGrrrGrr"), (2, "GGrGrrrrrGrrrrrr"),
    (3, "GGrGrrrrryrrrrrr"), (7, "GGrGrrrrrrrrrrrr"), (6, "GGrGGrrrrrrrrrrr"),
    (3, "yyryyrrrrrrrrrrr"), (8, "rrrrrrrrrrrrrrrr"),
]  # fmt: skip
# Real flows (veh/h) per arc; the simulated stations reproduce them within a few %
ARC_FLOW = {"508": 624, "510": 1300, "511": 368, "513": 208, "509": 368, "512": 340, "514": 248}
STATION_FLOW = {"N_in": 624, "S_in": 368, "W_in": 208, "N_out": 368, "S_out": 340, "W_out": 248}
CALENDAR = {
    "weekdays": [1, 2, 3],
    "school_holidays": [{"from": "2026-10-17", "to": "2026-11-01", "name": "toussaint"}],
    "public_holidays": ["2026-11-11"],
}


def reference_csv(path: Path) -> Path:
    days = pd.date_range("2026-09-01", "2026-10-15", freq="D").date
    rows = []
    for iu, q in ARC_FLOW.items():
        for day in days:
            f = 1 + 0.02 * (day.toordinal() % 5 - 2)
            for h in range(5, 12):
                end = pd.Timestamp(day) + pd.Timedelta(hours=h + 1)
                rows.append(
                    {
                        "iu_ac": iu,
                        "libelle": "x",
                        "date": str(day),
                        "hour_start": h,
                        "weekday": day.weekday(),
                        "day_type": "term" if day.weekday() < 5 else "weekend",
                        "flow_veh_h": q * f,
                        "occupancy_pct": 10.0,
                        "flow_valid": True,
                        "occ_valid": True,
                        "etat_barre": "Ouvert",
                        "ts_end": end.tz_localize("Europe/Paris").isoformat(),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def trips_table(seed: int, scenario: str) -> pd.DataFrame:
    """Trips with depart delays: arm N backs up on seed 10 of S0 only (60 s mean, growing
    over the window), the ambulance waits 40 s on that seed."""
    rows = []
    backlog = seed == 10 and scenario == "pitie_fixed"
    for arm in ("N", "S", "W"):
        for k in range(40):
            planned = 27000.0 + 180.0 * k
            delay = (3.0 * k if arm == "N" and backlog else 0.0) + 0.5
            rows.append((f"car_{arm}_S_00.{k}", "car", planned + delay, delay))
    rows.append(("bus_91_E.0", "bus", 27100.0, 0.0))
    amb = 40.0 if backlog else 2.0
    rows.append(("ambulance_0", "ambulance", 29400.0 + amb, amb))
    rows.append(("ped_S_0.0", "pedestrian", 27010.0, np.nan))
    t = pd.DataFrame(rows, columns=["id", "type", "depart_s", "depart_delay_s"])
    return t.assign(run_id=f"r{seed}", arrival_s=t["depart_s"] + 60, time_loss_s=10.0)


def write_runs(
    results: Path,
    cfg: dict,
    scenario: str,
    seeds,
    full: bool = True,
    hashes: dict | None = None,
    started: str = "2026-10-2{d}T08:00:00+00:00",
) -> None:
    """Synthetic runs of `scenario`; run.json carries `hashes` (default: those of the
    current `cfg`, so the runs are current)."""
    rng = np.random.default_rng(len(scenario))
    stations = {s["id"]: s["lanes"] for s in cfg["measurement"]["stations"]}
    durations = [float(d) for d, _ in V7]
    states = [s for _, s in V7]
    hashes = hashes or expected_hashes(cfg)
    for seed in seeds:
        d = results / f"{scenario}_seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": f"r{seed}",
            "scenario": scenario,
            "seed": seed,
            "code_version": "abc1234",
            "started_at": started.format(d=seed % 3),
            **hashes,
            "health": {
                "teleports": 0,
                "collisions": 0,
                "emergency_stops": 0,
                "emergency_braking": 0,
                "vehicles_waiting_end": 0,
                "drain_timeout": False,
                "depart_delay_mean": 1.0,
                "depart_delay_max": 30.0 + seed,
            },
            "safety": {
                "interventions": {"min_green": 0, "order": 0, "max_green": 0, "starvation": 0},
                "audit": {"yellow_to_green": 0, "out_of_order_green": 0},
            },
        }
        (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")
        kpis = pd.DataFrame(
            [
                ("car", "time_loss_mean", 40 + rng.normal()),
                ("car", "wait_p95", 60 + rng.normal()),
                ("bus", "delay_mean", 30 + rng.normal()),
                ("pedestrian", "ped_wait_single_max", 75 + rng.normal()),
                ("pedestrian", "ped_wait_max", 95 + rng.normal()),
            ],
            columns=["user_type", "kpi", "value"],
        ).assign(run_id=f"r{seed}", scenario=scenario, seed=seed)
        kpis.to_csv(d / "kpi_run.csv", index=False)
        trips_table(seed, scenario).to_parquet(d / "trip.parquet")
        if not full:
            continue
        det = []
        for st, lanes in stations.items():
            for b in range(0, 7200, 900):
                for lane in lanes:
                    n = STATION_FLOW[st] / 4 / len(lanes) + rng.integers(-1, 2)
                    det.append(
                        (f"r{seed}", f"e1_{st}_{lane}", st, lane, float(b), b + 900.0, n, 5.0, 10.0)
                    )
        pd.DataFrame(
            det,
            columns=[
                "run_id",
                "det_id",
                "station",
                "lane",
                "begin_s",
                "end_s",
                "n_veh",
                "occupancy_pct",
                "mean_speed",
            ],
        ).to_parquet(d / "detector.parquet")
        t = np.arange(0, 7200, 5.0)
        q = pd.concat(
            [
                pd.DataFrame(
                    {
                        "run_id": f"r{seed}",
                        "t": t,
                        "approach": a,
                        "jam_veh": np.floor((t % 117) / 117 * k),
                        "halting_veh": 0,
                    }
                )
                for a, k in (("N", 16), ("S", 9), ("W", 7))
            ]
        )
        q.assign(jam_m=q["jam_veh"] * 7.5).to_parquet(d / "queue.parquet")
        sig = link_intervals(states, durations, n_cycles=62)
        sig.assign(run_id=f"r{seed}").to_parquet(d / "signal.parquet")


@pytest.fixture
def setup(tmp_path, monkeypatch, pitie_cfg):
    monkeypatch.chdir(tmp_path)  # the net file below stands in for sumo/network/pitie
    net = tmp_path / pitie_cfg["network"]["net_file"]
    net.parent.mkdir(parents=True)
    phases = "".join(f'<phase duration="{d}" state="{s}"/>' for d, s in V7)
    net.write_text(f'<net><tlLogic id="C" type="static">{phases}</tlLogic></net>', "utf-8")
    vcfg = yaml.safe_load((REPO_ROOT / "configs/validation.yaml").read_text(encoding="utf-8"))
    vcfg["reference"]["file"] = reference_csv(tmp_path / "hours.csv").as_posix()
    field = yaml.safe_load((REPO_ROOT / "data/field/pitie_measurements.yaml").read_text("utf-8"))
    field["sessions"][0]["date"] = dt.date(2026, 10, 13)
    field["signal"]["cycle_s"] = [117, 118, 117, 116, 117]
    field["signal"]["heads"]["N_straight"]["green_s"] = [33, 34, 33]
    field["signal"]["heads"]["ped_NB"]["green_s"] = {"HOP": [32, 33], "STM": [25]}
    field["queues"][0].update(time="08:15:00", vehicles=12, visible_m=100)
    (tmp_path / "field.yaml").write_text(yaml.safe_dump(field), encoding="utf-8")
    vcfg["field"] = (tmp_path / "field.yaml").as_posix()
    (tmp_path / "validation.yaml").write_text(yaml.safe_dump(vcfg), encoding="utf-8")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"calendar": CALENDAR}), encoding="utf-8")
    return tmp_path


def _args(root: Path, *extra: str) -> list[str]:
    return [
        "--scenario",
        str(SCENARIO),
        "--results",
        str(root / "results"),
        "--validation",
        str(root / "validation.yaml"),
        "--data-config",
        str(root / "data.yaml"),
        "--figures",
        str(root / "fig"),
        "--docs",
        str(root / "docs/validation.md"),
        "--calibration-log",
        str(root / "calibration_log.yaml"),
        *extra,
    ]


def write_comparison(root: Path, pitie_cfg: dict, seeds, s1_hashes: dict | None = None) -> Path:
    """A comparison CSV of the synthetic runs with its provenance file (as written by
    evaluation.compare)."""
    base, other = (
        pd.concat(pd.read_csv(p) for p in (root / "results").glob(f"{s}_seed*/kpi_run.csv"))
        for s in ("pitie_fixed", "pitie_actuated")
    )
    out = root / "results/comparison"
    out.mkdir(exist_ok=True)
    csv = out / "pitie_fixed__vs__pitie_actuated.csv"
    headline = [["car", "time_loss_mean"], ["pedestrian", "ped_wait_single_max"]]
    compare(base, other, headline).to_csv(csv, index=False)
    s1 = load_scenario(S1)
    sides = [
        (pitie_cfg, SCENARIO, expected_hashes(pitie_cfg)),
        (s1, S1, s1_hashes or expected_hashes(s1)),
    ]
    provenance_path(csv).write_text(json.dumps(provenance(sides, list(seeds))), encoding="utf-8")
    return csv


def test_report_on_synthetic_runs(setup, pitie_cfg, capsys):
    root = setup
    seeds = range(1, 11)
    write_runs(root / "results", pitie_cfg, "pitie_fixed", seeds)
    s1 = load_scenario(S1)
    write_runs(root / "results", s1, "pitie_actuated", seeds, full=False)
    write_comparison(root, pitie_cfg, seeds)

    main(_args(root))

    out = root / "results/validation/pitie_fixed"
    for name in ("flow_validation.csv", "signal_validation.csv", "queue_validation.csv"):
        assert (out / name).is_file(), name
    flows = pd.read_csv(out / "flow_validation.csv", dtype={"iu_ac": str})
    a02 = flows[flows["role"] == "A-02"]
    assert len(a02) == 6 and set(a02["status"]) == {"PASS"}
    assert set(flows[flows["role"] == "sensitivity"]["status"]) == {"off"}  # 510 = 2 x 508
    signal = pd.read_csv(out / "signal_validation.csv").set_index("item")
    assert signal.loc["cycle_s", "sim_mean"] == pytest.approx(117.0)
    assert signal.loc["cycle_s", "status"] == "PASS"
    assert signal.loc["N_straight.green_s", "status"] == "PASS"
    assert float(signal.loc["N_straight.yellow_s", "model_value"]) == 3
    assert signal.loc["W.green_s", "status"] == "NOT_MEASURED"
    # ped_NB is green twice per cycle (HOP 32 s, STM 18 s here): one row per stage
    assert "ped_NB.green_s" not in signal.index
    assert signal.loc["ped_NB.green_s@HOP", "sim_mean"] == pytest.approx(32.0)
    assert signal.loc["ped_NB.green_s@STM", "sim_mean"] == pytest.approx(18.0)
    assert signal.loc["ped_NB.green_s@HOP", "status"] == "PASS"  # 32.5 vs 32
    assert signal.loc["ped_NB.green_s@STM", "status"] == "FAIL"  # 25 vs 18
    queues = pd.read_csv(out / "queue_validation.csv")
    assert queues["status"].iloc[0] in ("PASS", "FAIL") and queues["sim_n"].iloc[0] > 0
    assert set(queues["status"].iloc[1:]) == {"NOT_MEASURED"}

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["a02"]["verdict"] == "PASS" and summary["seeds"] == list(seeds)
    assert summary["config_hash"] == expected_hashes(pitie_cfg)["config_hash"]
    assert summary["preregistered_before_runs"] is True
    assert summary["comparison"]["current"] is True
    assert summary["safety"]["pitie_fixed"]["health"]["collisions"] == 0
    assert summary["safety"]["pitie_actuated"]["runs"] == 10
    assert summary["pedestrians"]["pitie_fixed"]["within_target"] is True
    cap = {(r["scenario"], r["vehicles"]): r for r in summary["capacity"]}
    assert cap[("pitie_fixed", "car N")]["status"] == "capacity-limited"
    assert cap[("pitie_fixed", "car N")]["seeds_over"] == [10]
    assert (
        cap[("pitie_fixed", "car N")]["worst_15min_s"]
        > cap[("pitie_fixed", "car N")]["worst_seed_mean_s"]
    )  # the backlog grows within the window
    assert cap[("pitie_actuated", "car N")]["status"] == "ok"
    assert cap[("pitie_fixed", "ambulance")]["max_s"] == 40.0

    page = (root / "docs/validation.md").read_text(encoding="utf-8")
    for heading in SECTIONS:
        assert f"## {heading}" in page, heading
    assert "**A-02 verdict: PASS**" in page and "Circularity" in page
    assert "ASSUMED: signal program" in page  # network.assumptions rendered
    assert "pitie_actuated" in page and "Holm" in page  # S0 vs S1 section filled
    assert "S1 - S0 [95 % CI]" in page
    assert "fig/validation_flows.png" in page.replace("../", "")
    assert (root / "fig/validation_flows.png").is_file()
    assert (root / "fig/validation_queues.png").is_file()  # one queue is measured
    # section 2: pre-registration and the dated changes from configs/validation.yaml
    assert "Pre-registered on 2026-10-11" in page and "Warning" not in page
    assert "Changes since pre-registration:" in page and "capacity_depart_delay_s" in page
    # section 7: the capacity table; section 10: limitations derived from the configs
    assert "**Insertion delay and capacity.**" in page and "| car N |" in page
    limits = page.split("## 10.")[1]
    assert "can only gap out" not in limits
    fixed, maxg = pitie_cfg["fixed"]["green_s"], pitie_cfg["actuated"]["max_green_s"]
    longer = {st: maxg[st] - fixed[st] for st in maxg if maxg[st] > fixed[st]}
    assert longer  # S1 can extend some greens beyond S0's
    for st, d in longer.items():
        assert f"{st} {maxg[st]:g} s" in limits and f"{st} by up to {d:g} s" in limits
    assert "car N on seed 10 of `pitie_fixed`" in limits and "ambulance waited up to 40 s" in limits
    assert f"tasks.py net tls-check --scenario {SCENARIO.as_posix()}" in limits
    assert "No calibration log of this scenario" in page
    out_text = capsys.readouterr().out
    assert "A-02 PASS" in out_text and "capacity-limited: pitie_fixed car N seeds [10]" in out_text


def test_report_without_field_data_or_comparison(setup, pitie_cfg):
    root = setup
    write_runs(root / "results", pitie_cfg, "pitie_fixed", range(1, 4))
    main(_args(root, "--field", str(root / "absent.yaml")))
    page = (root / "docs/validation.md").read_text(encoding="utf-8")
    for heading in SECTIONS:
        assert f"## {heading}" in page, heading
    assert "No comparison yet" in page and "0 measured; all rows NOT_MEASURED" in page
    signal = pd.read_csv(root / "results/validation/pitie_fixed/signal_validation.csv")
    assert set(signal["status"]) == {"NOT_MEASURED"}
    assert {"ped_NB.green_s@HOP", "ped_NB.green_s@STM"} <= set(signal["item"])
    assert not (root / "fig/validation_queues.png").exists()


def test_report_needs_runs(setup):
    with pytest.raises(SystemExit, match="no runs"):
        main(_args(setup))


def test_stale_runs_are_refused_or_rerun_never_reused(setup, pitie_cfg, monkeypatch):
    """Runs made before a program change carry another config_hash: the report refuses
    them (naming the seeds), and --run-missing reruns them first."""
    root = setup
    old = {**expected_hashes(pitie_cfg), "config_hash": "0ld" + "0" * 61}
    write_runs(root / "results", pitie_cfg, "pitie_fixed", range(1, 11), hashes=old)
    with pytest.raises(SystemExit, match=r"10 run\(s\) were not made with the current config.*"):
        main(_args(root))
    assert not (root / "docs/validation.md").exists()

    import traffic_twin.sim.run_scenario as rs

    rerun: list[int] = []

    def fake_run(cfg, seed, out_root, gui=False, config_path=None):
        rerun.append(seed)
        write_runs(Path(out_root), cfg, cfg["scenario"], [seed])  # current hashes

    monkeypatch.setattr(rs, "run", fake_run)
    main(_args(root, "--run-missing"))
    assert rerun == list(range(1, 11))
    summary = json.loads(
        (root / "results/validation/pitie_fixed/summary.json").read_text(encoding="utf-8")
    )
    assert summary["config_hash"] == expected_hashes(pitie_cfg)["config_hash"]


def test_out_of_date_comparison_is_not_shown(setup, pitie_cfg):
    """A comparison computed with another S1 config (or other S1 runs) is not published:
    section 9 says why, and the other scenario's totals are left out."""
    root = setup
    seeds = range(1, 11)
    write_runs(root / "results", pitie_cfg, "pitie_fixed", seeds)
    write_runs(root / "results", load_scenario(S1), "pitie_actuated", seeds, full=False)
    old_s1 = {**expected_hashes(load_scenario(S1)), "config_hash": "c" * 64}
    write_comparison(root, pitie_cfg, seeds, s1_hashes=old_s1)
    main(_args(root))
    page = (root / "docs/validation.md").read_text(encoding="utf-8")
    s9 = page.split("## 9. S0 vs S1")[1].split("## 10.")[0]
    assert "**Not shown: the comparison" in s9 and "config changed since (config_hash)" in s9
    assert "Holm" not in s9 and "`pitie_actuated`" not in page.split("## 7.")[1].split("## 8.")[0]
    summary = json.loads(
        (root / "results/validation/pitie_fixed/summary.json").read_text(encoding="utf-8")
    )
    assert summary["comparison"]["current"] is False
    assert list(summary["safety"]) == ["pitie_fixed"]

    (root / "results/comparison/pitie_fixed__vs__pitie_actuated.json").unlink()
    main(_args(root))
    page = (root / "docs/validation.md").read_text(encoding="utf-8")
    assert "no provenance file" in page


def test_preregistration_after_the_runs_is_flagged(setup, pitie_cfg, capsys):
    root = setup
    early = "2026-10-05T08:00:00+00:00"  # before the pre-registration date
    write_runs(root / "results", pitie_cfg, "pitie_fixed", range(1, 4), started=early)
    main(_args(root))
    page = (root / "docs/validation.md").read_text(encoding="utf-8")
    assert "**Warning: the pre-registration date (2026-10-11) is later than the first " in page
    summary = json.loads(
        (root / "results/validation/pitie_fixed/summary.json").read_text(encoding="utf-8")
    )
    assert summary["preregistered_before_runs"] is False
    assert summary["first_run_date"] == "2026-10-05"


def test_calibration_log_status_is_reported(setup, pitie_cfg):
    root = setup
    write_runs(root / "results", pitie_cfg, "pitie_fixed", range(1, 4))
    log = {
        "calibration": {
            "scenario_name": "pitie_fixed",
            "seeds": [101, 102, 103],
            "status": "stopped: capacity-limited",
            "frozen": {"W": "capacity-limited: slope 0.02 < 0.5"},
            "recommended_arm_scale": {"N": 1.0, "S": 1.0, "W": 1.0},
            "rows": [{"config_hash": expected_hashes(pitie_cfg)["config_hash"]}],
        }
    }
    (root / "calibration_log.yaml").write_text(yaml.safe_dump(log), encoding="utf-8")
    main(_args(root))
    page = (root / "docs/validation.md").read_text(encoding="utf-8")
    assert "**Warning: the demand is not calibrated.**" in page and "frozen arms: W" in page

    log["calibration"].update(status="converged", frozen={})
    (root / "calibration_log.yaml").write_text(yaml.safe_dump(log), encoding="utf-8")
    main(_args(root))
    page = (root / "docs/validation.md").read_text(encoding="utf-8")
    assert "converged on seeds 101, 102, 103; recommended scale = configured scale" in page
    assert "its last iteration ran this config" in page


def test_threshold_changes_live_in_the_config(tmp_path):
    """The dated change list is part of configs/validation.yaml (docs/validation.md is
    regenerated, so a note written there would be lost); every entry needs date, what, why."""
    from traffic_twin.config import ConfigError
    from traffic_twin.validation import load_validation

    vcfg = load_validation(REPO_ROOT / "configs/validation.yaml")
    assert vcfg["preregistered"] == dt.date(2026, 10, 11)  # the day the thresholds were written
    assert vcfg["changes"] and all(set(c) >= {"date", "what", "why"} for c in vcfg["changes"])
    assert all(str(c["date"]) >= str(vcfg["preregistered"]) for c in vcfg["changes"])
    raw = yaml.safe_load((REPO_ROOT / "configs/validation.yaml").read_text(encoding="utf-8"))
    path = tmp_path / "v.yaml"
    path.write_text(yaml.safe_dump({**raw, "changes": None}), encoding="utf-8")
    assert load_validation(path)["changes"] == []
    path.write_text(yaml.safe_dump({**raw, "changes": [{"date": "2026-11-05", "what": "x"}]}))
    with pytest.raises(ConfigError, match=r"changes\[0\] needs why"):
        load_validation(path)

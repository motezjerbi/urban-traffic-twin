"""Synthetic run folders for the API tests and the mock screens (Part 5, D54; contract
4.6). Nothing here is a simulation result: the folders follow the run output schemas
(docs/architecture.md, Run outputs) with made-up but self-consistent values, so CI (which
has no `data/results`) can test the API, and the mock screens can run on "fake data".

    python -m traffic_twin.api.fixtures --out data/fake [--n-runs 3]

writes `<out>/pitie_<controller>_seed<seed>/` run folders (with FAKE ETA predictions),
`<out>/evaluation/main/{kpis,equity}.csv`, a FAKE ETA reference bundle under
`<out>/models/eta/` and `<out>/api.yaml`, the settings to serve them:

    TT_API_CONFIG=<out>/api.yaml uvicorn traffic_twin.api.main:app --no-access-log

- The signal log replays the net's own program (phase states and durations), so the
  crossing, phase and stage answers are consistent; adaptive controllers shorten the HOP
  green every other cycle.
- The ambulance drives down its N approach (n_in, along the net's stop line geometry),
  stops 5 m before the stop line at t = 340 s while its link 2 is red (red 275-361 s in
  the adaptive runs) under an open preemption request (rules runs), and crosses from
  t = 370 s, once link 2 is green again: AMBULANCE_PHASES.
- Seeds are labels only (demo range 4000-4009, D38): no simulation is run.
- Every FAKE file carries `fake` in its model version (`fake-0`, `fake-ref-0`).
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from traffic_twin.api.db import (
    ETA_PREDICTION_FILE,
    ETA_SNAPSHOTS_FILE,
    EVENTS_FILE,
    KPI_FILE,
    LATEST_FILE,
    QUEUE_FILE,
    REFERENCE_FILE,
    RUN_FILE,
    SIGNAL_FILE,
    STATE_FILE,
    TRIPS_FILE,
)
from traffic_twin.api.settings import find_repo_root
from traffic_twin.api.site import Site, StopLine, load_site
from traffic_twin.config import config_hash, demand_hash, load_scenario
from traffic_twin.paths import (
    API_CONFIG,
    FIELD_FILE,
    PITIE_ACTUATED_CONFIG,
    PITIE_FIXED_CONFIG,
    PITIE_RL_CONFIG,
    PITIE_RULES_CONFIG,
)

CONFIGS = {
    "fixed": PITIE_FIXED_CONFIG,
    "actuated": PITIE_ACTUATED_CONFIG,
    "rules": PITIE_RULES_CONFIG,
    "rl": PITIE_RL_CONFIG,
}
LABELS = {"fixed": "S0", "actuated": "S1", "rules": "S2"}  # evaluation.yaml family main
FIXTURE_SEED = 4000  # demo range (D38): a label, no simulation is run
FAKE_MODEL_VERSION = "fake-0"
FAKE_REFERENCE_VERSION = "fake-ref-0"
TRAJETS = ("N_S", "N_W", "S_N", "S_W", "W_N", "W_S")
DAY_TYPES = ("tue_wed_thu_term", "mon_fri_term", "school_holiday")
STARTED_AT = "2026-10-14T06:00:00+00:00"
SAMPLE_S = 5.0  # state_log, queue and ETA files every 5 s, as the real runs
AMBULANCE = {
    "vid": "ambulance_0",
    "depart_s": 300.0,
    "arrival_s": 400.0,
    "route": ["n_in", "s_out"],
}
PREEMPT = (301.0, 371.0)  # request and served (rules runs only)
# ambulance along its stop line (m, < 0 before it): approach at 5 m/s to 5 m before the
# line (t = 340), halted until 370, then across at 8 m/s
AMBULANCE_PHASES = {"halt": (340.0, 370.0), "approach_mps": 5.0, "cross_mps": 8.0, "gap_m": 5.0}
N_BLOCKED = (600.0, 900.0)  # N approach queue >= the ambulance 'blocked' threshold
BUS_TRAVEL_S = 120.0


@functools.lru_cache(maxsize=8)
def _site(root: Path) -> Site:
    return load_site(PITIE_FIXED_CONFIG, root, FIELD_FILE)


@functools.lru_cache(maxsize=16)
def _config(root: Path, controller: str) -> dict:
    """The scenario config a fixture run embeds (cached: callers deep-copy it)."""
    return load_scenario(root / CONFIGS[controller])


def _timeline(site: Site, duration_s: float, adaptive: bool) -> list[tuple[float, float, int]]:
    """(begin, end, phase) segments of the program over [0, duration); an adaptive
    controller shortens the HOP green (phase 0) to 28 s every other cycle."""
    out, t, cycle = [], 0.0, 0
    while t < duration_s:
        for i, p in enumerate(site.phases):
            d = p.duration_s
            if adaptive and i == 0 and cycle % 2 == 1:
                d = max(6.0, d - 8.0)
            out.append((t, t + d, i))
            t += d
        cycle += 1
    return out


def ambulance_along(t: float) -> tuple[float, float]:
    """(signed distance to the n_in stop line, speed) of the fixture ambulance at t."""
    a = AMBULANCE_PHASES
    h0, h1 = a["halt"]
    if t < h0:
        return -a["gap_m"] - a["approach_mps"] * (h0 - t), a["approach_mps"]
    if t < h1:
        return -a["gap_m"], 0.0
    return -a["gap_m"] + a["cross_mps"] * (t - h1), a["cross_mps"]


def _phase_at(segments, t: float) -> int:
    return next(p for b, e, p in segments if b <= t < e)


def _signal(site: Site, segments, run_id: str, duration_s: float) -> pd.DataFrame:
    rows = []
    n_links = max((len(p.state) for p in site.phases), default=0)
    for link in range(n_links):
        start = None
        for b, _e, p in segments:
            green = link in site.phases[p].green
            if green and start is None:
                start = b
            if not green and start is not None:
                rows.append((link, start, b))
                start = None
        if start is not None:
            rows.append((link, start, min(segments[-1][1], duration_s)))
    df = pd.DataFrame(rows, columns=["link", "begin_s", "end_s"])
    df.insert(0, "run_id", run_id)
    df["duration_s"] = df["end_s"] - df["begin_s"]
    return df.sort_values(["begin_s", "link"], kind="stable").reset_index(drop=True)


def _bus_trips(site: Site, run_id: str, begin_s: float, duration_s: float) -> list[dict]:
    rows = []
    for b in site.bus_lines.values():
        offset = {"91_E": 0, "91_W": 180, "24_E": 300, "24_W": 60, "57_N": 120, "57_S": 360}
        k, dep = 0, float(offset.get(b.flow, 0))
        while dep + BUS_TRAVEL_S < duration_s:
            travel = BUS_TRAVEL_S + 15.0 * (k % 3)
            rows.append({
                "run_id": run_id, "id": f"bus_{b.flow}.{k}", "type": "bus",
                "depart_s": begin_s + dep, "arrival_s": begin_s + dep + travel,
                "duration_s": travel, "time_loss_s": 30.0 + 10.0 * (k % 4), "waiting_s": 10.0,
                "stops": 1.0, "depart_delay_s": float(k % 2), "depart_edge": b.from_edge,
                "arrival_edge": b.to_edge, "flow": f"bus_{b.flow}",
            })  # fmt: skip
            k += 1
            dep += b.headway_s
    return rows


def _kpis(run_id: str, scenario: str, seed: int, controller: str) -> pd.DataFrame:
    gain = {"fixed": 1.0, "actuated": 0.9, "rules": 0.8, "rl": 0.85}.get(controller, 1.0)
    jitter = 1.0 + 0.01 * (seed - FIXTURE_SEED)
    base = [
        ("car", "time_loss_mean", 70.0), ("car", "wait_p95", 120.0),
        ("bus", "delay_mean", 45.0), ("pedestrian", "ped_curb_wait_max", 85.0),
        ("pedestrian", "ped_curb_wait_mean", 30.0), ("ambulance", "travel_time_total", 150.0),
        ("all", "person_delay_mean", 50.0),
    ]  # fmt: skip
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "scenario": scenario,
                "seed": seed,
                "user_type": u,
                "kpi": k,
                "value": round(v * gain * jitter, 3),
            }
            for u, k, v in base
        ]  # fmt: skip
    )


def make_run(
    dir: Path,
    *,
    scenario: str | None = None,
    seed: int = FIXTURE_SEED,
    controller: str = "rules",
    duration_s: float = 1800.0,
    with_signal: bool = True,
    day_type: str = "tue_wed_thu_term",
    repo_root: Path | None = None,
) -> Path:
    """One synthetic run folder `dir` (run.json, kpi_run.csv, state_log, queue, signal,
    trip and, for `rules`, events); returns `dir`. `controller` picks the scenario config
    the run.json embeds (fixed, actuated, rules or rl)."""
    root = repo_root or find_repo_root(API_CONFIG)
    if controller not in CONFIGS:
        raise ValueError(f"controller must be one of {sorted(CONFIGS)}, got {controller!r}")
    scenario = scenario or f"pitie_{controller}"
    site = _site(Path(root))
    cfg = copy.deepcopy(_config(Path(root), controller))
    cfg.update(scenario=scenario, seed=seed, duration_s=int(duration_s))
    cfg["demand"]["day_type"] = day_type
    cfg["emergency"] = {
        "enabled": True, "depart_s": AMBULANCE["depart_s"], "route": AMBULANCE["route"],
    }  # fmt: skip
    begin_s = float(cfg["begin_s"])
    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"traffic-twin/api-fixture/{scenario}/{seed}"))
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)

    segments = _timeline(site, duration_s, adaptive=controller != "fixed")
    if with_signal:
        _signal(site, segments, run_id, duration_s).to_parquet(d / SIGNAL_FILE, index=False)
    trips = _bus_trips(site, run_id, begin_s, duration_s)
    ts = np.arange(0.0, duration_s, SAMPLE_S)
    queue_rows, state_rows = [], []
    cx, cy = site.junction_xy or (0.0, 0.0)
    stop = site.stop_lines.get(AMBULANCE["route"][0]) or StopLine(cx, cy + 20.0, 0.0, -1.0)
    for k, t in enumerate(ts):
        halting = {"N": 12 if N_BLOCKED[0] <= t < N_BLOCKED[1] else 2 + k % 3, "S": 4, "W": 3}
        for a, h in halting.items():
            queue_rows.append({"run_id": run_id, "t": t, "approach": a, "jam_veh": h,
                               "jam_m": 7.5 * h, "halting_veh": h})  # fmt: skip
        state_rows.append({"run_id": run_id, "t": t, "entity": "intersection", "id": site.tls_id,
                           "x": None, "y": None, "speed": None,
                           "phase": float(_phase_at(segments, t)),
                           "queue": float(sum(halting.values()))})  # fmt: skip
        for j in range(3 + k % 4):
            state_rows.append({"run_id": run_id, "t": t, "entity": "car",
                               "id": f"car_N_S_00.{k}_{j}", "x": cx, "y": cy + 20.0 + 8.0 * j,
                               "speed": 4.0, "phase": None, "queue": None})  # fmt: skip
        if k % 2 == 0:
            state_rows.append({"run_id": run_id, "t": t, "entity": "pedestrian", "id": f"ped_S.{k}",
                               "x": cx - 5.0, "y": cy - 20.0, "speed": 1.2, "phase": None,
                               "queue": None})  # fmt: skip
        for trip in trips:
            if trip["depart_s"] - begin_s <= t < trip["arrival_s"] - begin_s:
                state_rows.append({"run_id": run_id, "t": t, "entity": "bus", "id": trip["id"],
                                   "x": cx - 30.0, "y": cy, "speed": 6.0, "phase": None,
                                   "queue": None})  # fmt: skip
        if AMBULANCE["depart_s"] <= t < AMBULANCE["arrival_s"]:
            along, v = ambulance_along(t)
            state_rows.append({"run_id": run_id, "t": t, "entity": "ambulance",
                               "id": AMBULANCE["vid"], "x": stop.x + stop.ux * along,
                               "y": stop.y + stop.uy * along, "speed": v,
                               "phase": None, "queue": None})  # fmt: skip
    pd.DataFrame(state_rows).to_parquet(d / STATE_FILE, index=False)
    pd.DataFrame(queue_rows).to_parquet(d / QUEUE_FILE, index=False)
    cars = [
        {"run_id": run_id, "id": f"car_N_S_00.{k}", "type": "car", "depart_s": begin_s + 30.0 * k,
         "arrival_s": begin_s + 30.0 * k + 70.0, "duration_s": 70.0, "time_loss_s": 25.0,
         "waiting_s": 12.0, "stops": 1.0, "depart_delay_s": 0.0, "depart_edge": "n_in",
         "arrival_edge": "s_out", "flow": "car_N_S_00"}
        for k in range(int(duration_s // 60))
    ]  # fmt: skip
    amb = {"run_id": run_id, "id": AMBULANCE["vid"], "type": "ambulance",
           "depart_s": begin_s + AMBULANCE["depart_s"],
           "arrival_s": begin_s + AMBULANCE["arrival_s"],
           "duration_s": AMBULANCE["arrival_s"] - AMBULANCE["depart_s"], "time_loss_s": 20.0,
           "waiting_s": 0.0, "stops": 0.0, "depart_delay_s": 0.0, "depart_edge": "n_in",
           "arrival_edge": "s_out", "flow": AMBULANCE["vid"]}  # fmt: skip
    pd.DataFrame([*trips, *cars, amb]).to_parquet(d / TRIPS_FILE, index=False)
    if controller == "rules":
        events = [
            (PREEMPT[0], "preemption", "request", f"vid={AMBULANCE['vid']}, link=2, dist=200.0"),
            (PREEMPT[1], "preemption", "served", f"vid={AMBULANCE['vid']}, link=2, dist=3.0"),
        ]
        for trip in trips:  # a TSP extension for every 91 bus, 5 s before its arrival
            if trip["flow"].startswith("bus_91"):
                te = trip["arrival_s"] - begin_s - 5.0
                events.append((te, "controller", "tsp_extend",
                               f"vid={trip['id']}, line=91, eta=4.0, allow=10"))  # fmt: skip
        ev = pd.DataFrame(events, columns=["t", "source", "rule", "detail"])
        ev.insert(0, "run_id", run_id)
        ev.insert(4, "stage_from", 0)
        ev.insert(5, "stage_to", 0)
        ev.sort_values("t", kind="stable").to_parquet(d / EVENTS_FILE, index=False)
    _kpis(run_id, scenario, seed, controller).to_csv(d / KPI_FILE, index=False)
    meta = {
        "run_id": run_id, "scenario": scenario, "seed": seed, "code_version": "fixture",
        "config": cfg, "started_at": STARTED_AT, "duration_s": int(duration_s),
        "wall_time_s": 0.0, "config_hash": config_hash(cfg), "demand_hash": demand_hash(cfg),
        "kpi_version": 3, "fixture": True,
    }  # fmt: skip
    (d / RUN_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return d


def _evaluation(root: Path, run_dirs: list[Path]) -> None:
    """`<root>/evaluation/main/{kpis,equity}.csv` from the synthetic runs of the three
    evaluation controllers (same columns as evaluation.evaluate writes)."""
    frames = []
    for d in run_dirs:
        meta = json.loads((d / RUN_FILE).read_text(encoding="utf-8"))
        controller = meta["config"]["controller"]
        if controller not in LABELS:
            continue
        df = pd.read_csv(d / KPI_FILE)
        frames.append(df.assign(family="main", label=LABELS[controller], controller=controller))
    if not frames:
        return
    kpis = pd.concat(frames, ignore_index=True)[
        ["family", "label", "scenario", "controller", "seed", "user_type", "kpi", "value"]
    ]
    out = root / "evaluation" / "main"
    out.mkdir(parents=True, exist_ok=True)
    kpis.to_csv(out / "kpis.csv", index=False)
    means = kpis.groupby(["label", "user_type", "kpi"])["value"].mean()
    rows = []
    for label in sorted(set(kpis["label"]) - {"S0"}):
        for (lab, user, kpi), mean in means.items():
            if lab != label or ("S0", user, kpi) not in means:
                continue
            base = means[("S0", user, kpi)]
            rel = 100.0 * (mean - base) / base if base else math.nan
            rows.append({
                "family": "main", "scenario": label, "user_type": user, "kpi": kpi,
                "base_kpi": kpi, "qualifier": None, "baseline_mean": base, "mean": mean,
                "abs_change": mean - base, "rel_change_pct": rel, "pp_change": None,
                "p_value": None, "worse_flag": bool(rel > 5.0), "significant_flag": False,
                "primary": False, "subgroup": False,
            })  # fmt: skip
    pd.DataFrame(rows).to_csv(out / "equity.csv", index=False)


def make_results(
    root: Path, n_runs: int, *, evaluation: bool = True, duration_s: float = 1800.0
) -> Path:
    """A results root with `n_runs` synthetic run folders: controllers fixed, actuated,
    rules in turn (`pitie_<controller>_seed<4000 + k // 3>`), plus the main family's
    evaluation CSVs. Returns `root`."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dirs = []
    for k in range(n_runs):
        controller = ("fixed", "actuated", "rules")[k % 3]
        seed = FIXTURE_SEED + k // 3
        dirs.append(
            make_run(
                root / f"pitie_{controller}_seed{seed}",
                controller=controller,
                seed=seed,
                duration_s=duration_s,
            )
        )
    if evaluation:
        _evaluation(root, dirs)
    return root


def _fake_predictions(run_dir: Path, model_version: str) -> Path:
    meta = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    duration = float(meta.get("duration_s") or meta["config"]["duration_s"])
    ts = np.arange(SAMPLE_S, duration, SAMPLE_S)
    frames = []
    for k, trajet in enumerate(TRAJETS):
        med = np.round(45.0 + 5.0 * k + 15.0 * np.sin(ts / 600.0 + k))
        frames.append(
            pd.DataFrame(
                {
                    "run_id": meta["run_id"],
                    "route": trajet,
                    "t": ts,
                    "eta_median_s": med,
                    "eta_low_s": np.floor(med * 0.75),
                    "eta_high_s": np.ceil(med * 1.4),
                    "model_version": model_version,
                }
            )  # fmt: skip
        )
    path = run_dir / ETA_PREDICTION_FILE
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    t_all = np.arange(0.0, duration, SAMPLE_S)
    pending_n = np.where((t_all >= N_BLOCKED[0]) & (t_all < N_BLOCKED[1]), 2, 0)
    pd.DataFrame({
        "t": t_all, "pending_N": pending_n, "pending_S": 0, "pending_W": 0,
    }).to_parquet(run_dir / ETA_SNAPSHOTS_FILE, index=False)  # fmt: skip
    return path


def _fake_reference(models_dir: Path, version: str) -> Path:
    rows = []
    for k, trajet in enumerate(TRAJETS):
        for day_type in DAY_TYPES:
            for hour in (7, 8, 9):
                mean = 50.0 + 5.0 * k + (10.0 if hour == 8 else 0.0)
                rows.append((trajet, hour, day_type, mean, 0.7 * mean, 1.5 * mean, 500))
        rows.append((trajet, -1, "*", 55.0 + 5.0 * k, 38.0, 85.0, 4500))
    ref = pd.DataFrame(rows, columns=["route", "hour", "day_type", "mean_s", "p10_s", "p90_s", "n"])
    ref["model_version"] = version
    bundle = models_dir / version
    bundle.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(bundle / REFERENCE_FILE, index=False)
    meta = {"model_version": version, "fake": True, "note": "FAKE reference table (fixtures)"}
    (bundle / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (models_dir / LATEST_FILE).write_text(version + "\n", encoding="utf-8")
    return bundle


def make_fake_eta(target: Path, *, model_version: str | None = None) -> Path:
    """FAKE ETA files (plumbing only, not a model).

    - `target` is a run folder (it has run.json): writes `eta_prediction.parquet` (the
      Supabase columns, every 5 s, 6 trajets) and a minimal `eta_snapshots.parquet`
      (`t`, `pending_{N,S,W}`); returns the prediction file.
    - otherwise `target` is an ETA models dir: writes `<target>/<version>/reference.parquet`
      and `meta.json`, and `<target>/LATEST`; returns the bundle folder.
    """
    target = Path(target)
    if (target / RUN_FILE).is_file():
        return _fake_predictions(target, model_version or FAKE_MODEL_VERSION)
    target.mkdir(parents=True, exist_ok=True)
    return _fake_reference(target, model_version or FAKE_REFERENCE_VERSION)


def write_settings(out: Path, run_dirs: list[Path], models_dir: Path) -> Path:
    """`<out>/api.yaml`: configs/api.yaml with the results, evaluation, models and
    default run pointing at the fake data."""
    root = find_repo_root(API_CONFIG)
    raw = yaml.safe_load((root / API_CONFIG).read_text(encoding="utf-8"))
    rules = [d for d in run_dirs if d.name.startswith("pitie_rules")]
    raw["results_dirs"] = [out.as_posix()]
    raw["evaluation"]["dir"] = (out / "evaluation").as_posix()
    raw["eta"]["models_dir"] = models_dir.as_posix()
    raw["default_run"] = [d.as_posix() for d in (rules or run_dirs)[:1]]
    raw.setdefault("dev", {})["mock"] = True
    path = out / "api.yaml"
    header = "# FAKE data settings written by `python -m traffic_twin.api.fixtures`\n"
    path.write_text(header + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), "utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data/fake"), help="output folder")
    parser.add_argument("--n-runs", type=int, default=3, help="synthetic runs (default 3)")
    parser.add_argument(
        "--duration-s", type=float, default=7200.0, help="run length (default 7200, 07:30-09:30)"
    )
    a = parser.parse_args(argv)
    if a.n_runs < 1 or a.duration_s < 600:
        parser.error("--n-runs must be >= 1 and --duration-s >= 600")
    results = (find_repo_root(API_CONFIG) / "data" / "results").resolve()
    out = a.out
    if out.resolve() == results or results in out.resolve().parents:
        parser.error("refusing to write fake runs into data/results")
    make_results(out, a.n_runs, duration_s=a.duration_s)
    runs = sorted(p.parent for p in out.glob(f"*/{RUN_FILE}"))
    for d in runs:
        make_fake_eta(d)
    models = out / "models" / "eta"
    make_fake_eta(models)
    settings = write_settings(out, runs, models)
    print(f"{len(runs)} FAKE runs in {out}; settings {settings}")
    serve = "uvicorn traffic_twin.api.main:app --no-access-log"
    print(f"serve: TT_API_CONFIG={settings.as_posix()} {serve}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

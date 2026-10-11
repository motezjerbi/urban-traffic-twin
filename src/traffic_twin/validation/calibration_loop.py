"""Simulation-in-the-loop demand scaling of S0 on the calibration seeds (13.4: "tout
écart ... est corrigé par ajustement de la demande ... puis documenté").

    python -m traffic_twin.validation.calibration_loop --scenario configs/pitie_fixed.yaml
        [--seeds 101 102 103]

Each iteration runs S0 on the calibration seeds (never the evaluation seeds) in a
temporary directory with `demand.arm_scale` changed in memory, measures each inbound arm's
07:30-09:30 loop flow (mean over seeds) and rescales the arm:
scale <- round(scale x clip(real / sim, 0.5, 2.0), 3). It stops when every arm is within
`tolerance` (5 %), or after `max_iter` updates. An arm whose loop flow stops following
its demand (slope d sim / d demand < freeze_slope after a demand change of at least
min_change) is capacity-limited: it is frozen at its last responsive scale and the run
reports "stopped: capacity-limited" (fix the network, signal or driving parameters, not
the demand). The log goes to docs/calibration_log.yaml (deterministic: no wall-clock
time). Configs are never edited: copy the printed `arm_scale` into
configs/pitie/demand.yaml by hand. The command exits 1 unless the loop converged with the
configured `arm_scale` (capacity-limited, not converged, or a new scale to copy), so the
`tasks.py part3` chain stops before publishing results on uncalibrated demand.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from traffic_twin.config import ConfigError, config_hash, load_config, load_scenario
from traffic_twin.evaluation.stats import geh
from traffic_twin.paths import DATA_CONFIG, PITIE_FIXED_CONFIG, VALIDATION_CONFIG
from traffic_twin.validation import load_validation, slot_bounds
from traffic_twin.validation.flows import sim_flows

log = logging.getLogger(__name__)

LOG_FILE = Path("docs/calibration_log.yaml")
Flows = dict[str, float]


@dataclass
class LoopResult:
    status: str  # converged | stopped: capacity-limited | stopped: max iterations
    scale: dict[str, float]  # recommended arm_scale
    frozen: dict[str, str] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)  # one per iteration and arm
    iterations: int = 0  # scale updates made


def _r(x: float | None, nd: int = 1) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def calibrate(
    simulate: Callable[[dict[str, float]], Flows],
    real: Flows,
    base_demand: Flows,
    scale: dict[str, float],
    *,
    tolerance: float = 0.05,
    max_iter: int = 4,
    clip: tuple[float, float] = (0.5, 2.0),
    freeze_slope: float = 0.5,
    min_change: float = 0.10,
    hash_of: Callable[[dict[str, float]], str] | None = None,
) -> LoopResult:
    """Run the loop. `simulate(scale)` returns the simulated flow (veh/h) per arm,
    `real` the targets, `base_demand` the arm's demand at scale 1 (veh/h, for the slope)."""
    arms = sorted(real)
    scale = {a: float(scale.get(a, 1.0)) for a in arms}
    frozen: dict[str, str] = {}
    prev: dict[str, tuple[float, float, float]] = {}  # arm -> (scale, demand, sim)
    rows: list[dict] = []
    updates = 0
    status = "stopped: max iterations"
    for it in range(max_iter + 1):
        sim = simulate(dict(scale))
        h = hash_of(dict(scale)) if hash_of else None
        within, actions = {}, {}
        for a in arms:
            demand = base_demand[a] * scale[a]
            err = sim[a] / real[a] - 1
            within[a] = abs(err) <= tolerance + 1e-12
            slope = None
            if a in prev and abs(demand / prev[a][1] - 1) >= min_change - 1e-12:
                slope = (sim[a] - prev[a][2]) / (demand - prev[a][1])
                if slope < freeze_slope and not within[a]:
                    frozen[a] = f"capacity-limited: slope {slope:.2f} < {freeze_slope}"
                    actions[a] = f"frozen, back to scale {prev[a][0]:.3f}"
            rows.append(
                {
                    "iteration": it,
                    "arm": a,
                    "scale": round(scale[a], 3),
                    "demand_veh_h": _r(demand),
                    "sim_veh_h": _r(sim[a]),
                    "real_veh_h": _r(real[a]),
                    "rel_err_pct": _r(100 * err, 2),
                    "geh": _r(geh(sim[a], real[a]), 2),
                    "slope": _r(slope, 3),
                    "action": None,
                    "config_hash": h,
                }
            )
        it_rows = rows[-len(arms) :]
        active = [a for a in arms if a not in frozen]
        for a in frozen:  # revert to the last responsive scale (the one before the change)
            if a in actions:
                scale[a] = prev[a][0]
        prev = {a: (scale[a], base_demand[a] * scale[a], sim[a]) for a in active}
        done = all(within[a] for a in active)
        for r in it_rows:
            a = r["arm"]
            if a in actions:
                r["action"] = actions[a]
            elif a in frozen:
                r["action"] = "frozen"
            elif within[a]:
                r["action"] = "within tolerance"
        if done or it == max_iter:
            status = "stopped: capacity-limited" if frozen else "converged" if done else status
            for r in it_rows:
                r["action"] = r["action"] or "max iterations reached"
            break
        for r in it_rows:
            a = r["arm"]
            if r["action"] is None:
                f = min(max(real[a] / sim[a] if sim[a] > 0 else clip[1], clip[0]), clip[1])
                scale[a] = round(scale[a] * f, 3)
                r["action"] = f"scale x{f:.3f} -> {scale[a]:.3f}"
        updates += 1
    return LoopResult(status, {a: round(scale[a], 3) for a in arms}, frozen, rows, updates)


# -- the real loop ---------------------------------------------------------------------
def calibration_arms(vcfg: dict) -> dict[str, str]:
    """Arm -> station of the A-02 inbound stations."""
    return {
        st["arm"]: st["station"]
        for st in vcfg["stations"]
        if st.get("direction") == "in" and st.get("a02")
    }


def window_slot(vcfg: dict, cfg: dict) -> dict:
    """The validation slot that spans the simulated window (07:30-09:30)."""
    for slot in vcfg["slots"]:
        if slot_bounds(slot, float(cfg["begin_s"])) == (0.0, float(cfg["duration_s"])):
            return slot
    raise ConfigError("configs/validation.yaml has no slot spanning the simulated window")


def base_demand(cfg: dict, arms: list[str]) -> Flows:
    """Each arm's window-mean car demand at arm_scale 1 (veh/h), from the calibration
    file times `demand.multiplier`."""
    demand = cfg["demand"]
    path = Path(demand["calibration"])
    if not path.is_file():
        raise ConfigError(f"calibration file {path} not found (run `tasks.py calib-data`)")
    cal = yaml.safe_load(path.read_text(encoding="utf-8"))
    mult = float(demand.get("multiplier", 1.0))
    out = {}
    for a in arms:
        slots = cal["arms"][a]["slots"]
        out[a] = mult * float(np.mean([s["flow_veh_h"] for s in slots]))
    return out


def with_scale(cfg: dict, scale: dict[str, float]) -> dict:
    out = copy.deepcopy(cfg)
    out["demand"]["arm_scale"] = {**out["demand"].get("arm_scale", {}), **scale}
    return out


def measure(
    cfg: dict, seeds: list[int], out_root: Path, stations: dict[str, str], slot: dict
) -> Flows:
    """Mean loop flow per arm over the calibration seeds (runs S0 into out_root)."""
    from traffic_twin.sim.run_scenario import run, run_dir

    frames = []
    for seed in seeds:
        run(cfg, seed, out_root)
        det = pd.read_parquet(run_dir(cfg, seed, out_root) / "detector.parquet")
        frames.append(det.assign(seed=seed))
    flows = sim_flows(pd.concat(frames, ignore_index=True), [slot], float(cfg["begin_s"]))
    mean = flows.groupby("station")["flow_veh_h"].mean()
    return {a: float(mean[s]) for a, s in stations.items()}


def write_log(path: Path, header: dict, result: LoopResult) -> Path:
    doc = {
        "calibration": {
            **header,
            "status": result.status,
            "iterations": result.iterations,
            "frozen": result.frozen,
            "recommended_arm_scale": result.scale,
            "rows": result.rows,
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Generated by `python -m traffic_twin.validation.calibration_loop`; do not edit.\n"
    text += yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(text, encoding="utf-8")
    return path


def gate(result: LoopResult, configured: dict[str, float], tol: float = 5e-4) -> list[str]:
    """Why the configured demand is not calibrated ([] when it is): the loop did not
    converge (an arm capacity-limited, or the iteration limit reached), or the recommended
    arm_scale is not the one in the config (copy it into configs/pitie/demand.yaml and
    rerun, which then converges at iteration 0)."""
    problems = []
    if result.status != "converged":
        if result.frozen:
            arms = ", ".join(result.frozen)
            problems.append(
                f"{result.status} (arm {arms}): fix the network, signal or driving parameters, "
                "never inflate the demand; report it before publishing results"
            )
        else:
            problems.append(f"{result.status}: the demand is not within tolerance of the sensors")
    diff = {
        a: (configured.get(a, 1.0), v)
        for a, v in result.scale.items()
        if abs(v - float(configured.get(a, 1.0))) > tol
    }
    if diff:
        text = ", ".join(f"{a} {old} -> {new}" for a, (old, new) in diff.items())
        problems.append(
            f"the recommended arm_scale differs from demand.arm_scale ({text}): copy it into "
            "configs/pitie/demand.yaml, then rerun"
        )
    return problems


def main(argv: list[str] | None = None) -> None:
    """Exits 1 (after writing the log) unless the loop converged with the configured
    arm_scale, so `tasks.py part3` never publishes results on uncalibrated demand."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", type=Path, default=PITIE_FIXED_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", help="default: calibration seeds")
    parser.add_argument("--validation", type=Path, default=VALIDATION_CONFIG)
    parser.add_argument("--data-config", type=Path, default=DATA_CONFIG)
    parser.add_argument("--log", type=Path, default=LOG_FILE)
    parser.add_argument("--work-dir", type=Path, help="keep the runs here (default: temp dir)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from traffic_twin.validation.reference import read_hours, real_flows

    try:
        cfg = load_scenario(args.scenario)
        vcfg = load_validation(args.validation)
        cal = load_config(args.data_config).get("calendar", {})
        seeds = args.seeds or [int(s) for s in vcfg["seeds"]["calibration"]]
        overlap = set(seeds) & {int(s) for s in vcfg["seeds"]["evaluation"]}
        if overlap:
            raise ConfigError(f"calibration seeds overlap the evaluation seeds: {sorted(overlap)}")
        stations = calibration_arms(vcfg)
        slot = window_slot(vcfg, cfg)
        st_cfg = [s for s in vcfg["stations"] if s["station"] in stations.values()]
        ref = real_flows(read_hours(vcfg["reference"]["file"]), st_cfg, [slot], cal, vcfg)
        ref = ref[~ref["sensitivity"]].set_index("station")
        if (ref["status"] != "ok").any():
            raise ConfigError(f"too few sensor days for {list(ref.index[ref['status'] != 'ok'])}")
        real = {a: float(ref.loc[s, "real_mean"]) for a, s in stations.items()}
        base = base_demand(cfg, list(stations))
    except (ConfigError, FileNotFoundError) as e:
        raise SystemExit(f"calibration error: {e}") from None

    loop = vcfg["calibration_loop"]
    start = {a: float(cfg["demand"].get("arm_scale", {}).get(a, 1.0)) for a in stations}
    calib_cfg = copy.deepcopy(cfg)
    calib_cfg["scenario"] = f"{cfg['scenario']}_calib"  # never clobber the S0 demand/results
    with tempfile.TemporaryDirectory(prefix="calib_") as tmp:
        root = args.work_dir or Path(tmp)
        counter = iter(range(10_000))

        def simulate(scale: dict[str, float]) -> Flows:
            out = root / f"iter{next(counter)}"
            return measure(with_scale(calib_cfg, scale), seeds, out, stations, slot)

        result = calibrate(
            simulate,
            real,
            base,
            start,
            tolerance=float(loop["tolerance"]),
            max_iter=int(loop["max_iter"]),
            clip=tuple(loop["clip"]),
            freeze_slope=float(loop["freeze_slope"]),
            min_change=float(loop["min_change"]),
            hash_of=lambda s: config_hash(with_scale(cfg, s)),
        )
    header = {
        "scenario": args.scenario.as_posix(),
        "scenario_name": cfg["scenario"],
        "controller": cfg["controller"],
        "seeds": seeds,
        "slot": slot["name"],
        "targets": {
            "source": Path(vcfg["reference"]["file"]).as_posix(),
            "arcs": {a: str(ref.loc[s, "iu_ac"]) for a, s in stations.items()},
            "n_days": {a: int(ref.loc[s, "n_days"]) for a, s in stations.items()},
            "days": {
                a: f"{ref.loc[s, 'first_day']}..{ref.loc[s, 'last_day']}"
                for a, s in stations.items()
            },
            "real_veh_h": {a: round(v, 1) for a, v in real.items()},
        },
        "base_demand_veh_h": {a: round(v, 1) for a, v in base.items()},
        "start_arm_scale": start,
        "settings": {k: (list(v) if isinstance(v, tuple | list) else v) for k, v in loop.items()},
    }
    write_log(args.log, header, result)
    print(f"status: {result.status} after {result.iterations} update(s); log {args.log.as_posix()}")
    for a, why in result.frozen.items():
        print(f"  {a}: {why} -> do not inflate demand; check the network / signal / driving")
    print(
        "recommended demand.arm_scale: {"
        + ", ".join(f"{a}: {v}" for a, v in result.scale.items())
        + "}"
    )
    problems = gate(result, start)
    if problems:  # a non-zero exit stops `tasks.py part3` before compare and validate
        for p in problems:
            print(f"calibration not done: {p}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

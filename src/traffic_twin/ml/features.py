"""ETA features `eta-features-1` (Part 5, D41, D42; contract 4.4). One definition used by
the dataset generator (`ml.eta_dataset`) and by prediction (`ml.predict`), so training
and serving cannot diverge.

A feature row describes what is known at time t (s since scenario start): it is built by
`FeatureState.update` from the `JunctionSnapshot` taken after the step that ended at t,
and carries `t_feat = t`. A car physically inserted at `insert_t` is joined to the row at
`t_feat = floor(insert_t) - 1` (`join_samples`), so nothing seen after its entry leaks in.
For serving, the row with `t_feat` predicts for cars entering at `t = t_feat + 1`
(`prediction_frame`), the same pairing as in training.

Features (D41; ASSUMED sensors in brackets):
- time and context: `tod_sin`, `tod_cos` (absolute clock), `trajet`, `controller`,
  `day_type` (categorical, fixed category lists);
- per arm A in N, S, W: `halt_A`, `nveh_A` (in-edges), `wait_max_A` (max vehicle waiting
  time on the in-edges, ASSUMED camera), `jam_m_A` (E2 jam length), `halt_out_A` (halted
  on the out-edges), `occ60_A` / `occ300_A` (mean E1 occupancy of station `A_in` over the
  last 60 / 300 s, 0..1), `flow300_A` (vehicles newly seen at `A_in` over the last 300 s,
  in veh/h);
- signal: `stage` (stage name), `in_transition`, `time_in_stage`, `clear_left_s`,
  `rem_min_s` / `rem_max_s` (D42: min / nominal green left, 0 during a transition),
  `gap_cur_s` (current stage's loop gap, capped at 60 s); per trajet j: `green_j`,
  `ttg_min_j`, `ttg_nom_j` (D42, see `time_to_green`);
- bus: `bus_N`, `bus_S`, `bus_NB`, `bus_WB` (bus calls per approach), `bus_dwell_s_out`
  (1 while a bus is halted on the S out-edge, the in-lane stop_57_S);
- ambulance (ASSUMED V2I): `amb_present`, `amb_pending`, `amb_dist_bin` (0 none,
  1 pending only, 2 nearest < 50 m, 3 < 150 m, 4 farther), `amb_arm` (approach of the
  nearest call, "none" without one), `preempt_active`;
- incidents: `inc_A` (an incident in force on arm A) and `inc_age_s_A` (s since the
  oldest one in force started, 0 without one);
- pedestrians: `ped_wait_max` (longest detected curb wait), `ped_count` (persons waiting).

`DIAGNOSTIC` (`pending_A`: vehicles waiting for insertion by arm) is stored with every
row but is simulation-only and never a model input (D41). World parameters (multiplier,
arm scales, incident durations, seeds) are never features.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from traffic_twin.config import approach_in_edges, approach_out_edges, get
from traffic_twin.ml.daytypes import DAY_TYPES

if TYPE_CHECKING:
    from traffic_twin.sim.sampler import JunctionSnapshot
    from traffic_twin.sim.session import SimSession

FEATURE_SCHEMA_VERSION = "eta-features-1"
TRAJETS = ("N_S", "N_W", "S_N", "S_W", "W_N", "W_S")
ARMS = ("N", "S", "W")
CONTROLLERS = ("fixed", "actuated", "rules")
STAGES = ("HOP", "S_LEFT", "STM", "BUS_91", "BUS_57")
DIAGNOSTIC = ("pending_N", "pending_S", "pending_W")  # stored, never model inputs (D41)
BUS_APPROACHES = ("N", "S", "NB", "WB")
NO_AMBULANCE = "none"

OCC_SHORT_S = 60  # occ60_A window (s, one sample per second)
ROLLING_S = 300  # occ300_A and flow300_A window
GAP_CAP_S = 60.0  # gap_cur_s cap (no detection for 60 s or more, inf included)
AMB_NEAR_M = (50.0, 150.0)  # amb_dist_bin edges: < 50 m -> 2, < 150 m -> 3, farther -> 4
EXIT_BUS_ARM = "S"  # bus_dwell_s_out: halted bus on the S out-edge (stop_57_S, in-lane)

# Nominal green per controller (D42): the program green for fixed, the max greens of the
# adaptive controllers. Any other controller (e.g. rl, never seen by the model) uses the
# actuated max greens.
NOMINAL_GREEN_KEY = {
    "fixed": "fixed.green_s",
    "actuated": "actuated.max_green_s",
    "rules": "rules.max_green_s",
}
DEFAULT_NOMINAL_KEY = "actuated.max_green_s"

# Model inputs in order (D41), "trajet" included.
ETA_FEATURES: tuple[str, ...] = (
    "trajet", "controller", "day_type", "tod_sin", "tod_cos",
    *(f"{k}_{a}" for a in ARMS for k in (
        "halt", "nveh", "wait_max", "jam_m", "halt_out", "occ60", "occ300", "flow300"
    )),
    "stage", "in_transition", "time_in_stage", "clear_left_s", "rem_min_s", "rem_max_s",
    "gap_cur_s",
    *(f"{k}_{j}" for j in TRAJETS for k in ("green", "ttg_min", "ttg_nom")),
    "bus_N", "bus_S", "bus_NB", "bus_WB", "bus_dwell_s_out",
    "amb_present", "amb_pending", "amb_dist_bin", "amb_arm", "preempt_active",
    *(f"{k}_{a}" for a in ARMS for k in ("inc", "inc_age_s")),
    "ped_wait_max", "ped_count",
)  # fmt: skip

# Fixed category lists of the categorical features
CATEGORIES: dict[str, tuple[str, ...]] = {
    "trajet": TRAJETS,
    "controller": CONTROLLERS,
    "day_type": DAY_TYPES,
    "stage": STAGES,
    "amb_arm": (NO_AMBULANCE, *ARMS),
}

# Columns of a feature row besides t_feat: every feature except trajet, then DIAGNOSTIC
ROW_FEATURES: tuple[str, ...] = tuple(f for f in ETA_FEATURES if f != "trajet")
ROW_COLUMNS: tuple[str, ...] = ("t_feat", *ROW_FEATURES, *DIAGNOSTIC)
NUMERIC_FEATURES: tuple[str, ...] = tuple(f for f in ETA_FEATURES if f not in CATEGORIES)
MODEL_FLOAT = "float32"  # numeric dtype of model_frame (XGBoost works in float32)


def feature_dtypes() -> dict[str, str]:
    """Feature name -> dtype of `model_frame` ("category" or MODEL_FLOAT), for manifests."""
    return {f: "category" if f in CATEGORIES else MODEL_FLOAT for f in ETA_FEATURES}


# -- site facts ------------------------------------------------------------------------
@dataclass(frozen=True)
class EtaSite:
    """Per-site signal facts for the time-to-green features (D42)."""

    trajet_stages: Mapping[str, tuple[int, ...]]  # trajet -> stages serving it (cycle order)
    min_green_s: tuple[float, ...]  # per stage
    nominal_green_s: tuple[float, ...]  # per stage: fixed program / actuated max / rules max
    stage_names: tuple[str, ...] = STAGES  # per stage, cycle order
    clearance_s: tuple[float, ...] = ()  # per stage: total duration of its clearance phases


def _per_stage(cfg: dict, key: str, names: Sequence[str], fallback: float | None) -> tuple:
    values = get(cfg, key, None)
    if not isinstance(values, Mapping):
        if fallback is None:
            raise KeyError(f"config has no {key}")
        return (float(fallback),) * len(names)
    missing = [n for n in names if n not in values]
    if missing:
        raise KeyError(f"{key}: no value for stage {', '.join(missing)}")
    return tuple(float(values[n]) for n in names)


def build_site(
    cfg: dict,
    controller: str,
    links: Sequence[tuple[str, str]],
    phase_states: Sequence[str],
    durations: Sequence[float],
) -> EtaSite:
    """EtaSite from the scenario config and the TLS program: `links[i]` = (from lane, to
    lane) of TLS link i, `phase_states` / `durations` the program's phases.

    A trajet A_B is served by a stage when its green phase shows G or g on a link from an
    in-edge of approach A to an out-edge of approach B. Min greens: `rules.min_green_s`
    for rules, else `actuated.min_green_s`, else `safety.min_green_s`; nominal greens:
    `NOMINAL_GREEN_KEY` (D42)."""
    stages = cfg["network"]["stages"]
    names = tuple(str(s["name"]) for s in stages)
    in_edges, out_edges = approach_in_edges(cfg), approach_out_edges(cfg)
    arm_of_in = {e: a for a, es in in_edges.items() for e in es}
    arm_of_out = {e: a for a, es in out_edges.items() for e in es}
    served: dict[str, list[int]] = {j: [] for j in TRAJETS}
    for k, st in enumerate(stages):
        state = phase_states[int(st["green_phase"])]
        for i, (from_lane, to_lane) in enumerate(links):
            if state[i] not in "Gg":
                continue
            a = arm_of_in.get(from_lane.rpartition("_")[0])
            b = arm_of_out.get(to_lane.rpartition("_")[0])
            j = f"{a}_{b}"
            if j in served and k not in served[j]:
                served[j].append(k)
    min_key = "rules.min_green_s" if controller == "rules" else "actuated.min_green_s"
    min_green = _per_stage(cfg, min_key, names, get(cfg, "safety.min_green_s", 0.0))
    nominal = _per_stage(
        cfg, NOMINAL_GREEN_KEY.get(controller, DEFAULT_NOMINAL_KEY), names, fallback=None
    )
    clearance = tuple(float(sum(durations[int(p)] for p in s["clearance_phases"])) for s in stages)
    return EtaSite(
        trajet_stages={j: tuple(sorted(ks)) for j, ks in served.items()},
        min_green_s=min_green,
        nominal_green_s=nominal,
        stage_names=names,
        clearance_s=clearance,
    )


def eta_site(sim: SimSession, controller: str) -> EtaSite:
    """EtaSite of a started session, from its TLS links and program and the config."""
    if sim.plan is None:
        raise RuntimeError("eta_site needs a started SimSession")
    links = []
    for conns in sim.traci.trafficlight.getControlledLinks(sim.tls):
        links.append((conns[0][0], conns[0][1]) if conns else ("", ""))
    return build_site(sim.cfg, controller, links, sim.phase_states, sim.plan.durations)


def time_to_green(
    site: EtaSite,
    stages: Sequence[int],
    stage: int,
    time_in_stage: float,
    in_transition: bool,
    clearance_left_s: float,
    target: int | None,
    greens: Sequence[float],
) -> float:
    """Seconds until one of `stages` shows green if every stage runs `greens` in cycle
    order with its full clearance, from the current state (D42). 0 when one of them is
    green now; during a transition the remaining clearance leads to `target` (the
    successor when unknown). NaN when no stage serves the trajet."""
    if not stages:
        return math.nan
    n = len(site.stage_names)
    if not in_transition:
        if stage in stages:
            return 0.0
        left = max(greens[stage] - time_in_stage, 0.0) + site.clearance_s[stage]
        k = (stage + 1) % n
    else:
        left = float(clearance_left_s)
        k = target if target is not None else (stage + 1) % n
    while k not in stages:
        left += greens[k] + site.clearance_s[k]
        k = (k + 1) % n
    return float(left)


# -- rolling feature state -------------------------------------------------------------
class FeatureState:
    """Rolling 1 Hz feature state; one per run. Call `update` once per simulated second
    (the occupancy and flow windows count samples, one per second)."""

    def __init__(self, cfg: dict, controller: str, site: EtaSite) -> None:
        self.cfg, self.controller, self.site = cfg, str(controller), site
        self.day_type = str(get(cfg, "demand.day_type", DAY_TYPES[0]))
        self.stations = {a: f"{a}_in" for a in ARMS}
        self._occ = {a: deque(maxlen=ROLLING_S) for a in ARMS}
        self._new = {a: deque(maxlen=ROLLING_S) for a in ARMS}

    def _rolling(self, snap: JunctionSnapshot, row: dict) -> None:
        for a in ARMS:
            station = self.stations[a]
            occ, new = self._occ[a], self._new[a]
            if station in snap.station_occupancy:
                occ.append(float(snap.station_occupancy[station]))
                new.append(int(snap.station_new_vehicles.get(station, 0)))
            if occ:
                last = list(occ)[-OCC_SHORT_S:]
                row[f"occ60_{a}"] = float(sum(last) / len(last))
                row[f"occ300_{a}"] = float(sum(occ) / len(occ))
                row[f"flow300_{a}"] = float(sum(new)) * 3600.0 / len(new)
            else:  # no measurement station on this arm
                row[f"occ60_{a}"] = row[f"occ300_{a}"] = row[f"flow300_{a}"] = math.nan

    def _signal(self, snap: JunctionSnapshot, row: dict) -> None:
        site, s = self.site, snap.stage
        trans, tis = bool(snap.in_transition), float(snap.time_in_stage)
        names = site.stage_names
        row["stage"] = names[s] if 0 <= s < len(names) else str(s)
        row["in_transition"] = int(trans)
        row["time_in_stage"] = tis
        row["clear_left_s"] = float(snap.clearance_left_s)
        row["rem_min_s"] = 0.0 if trans else max(site.min_green_s[s] - tis, 0.0)
        row["rem_max_s"] = 0.0 if trans else max(site.nominal_green_s[s] - tis, 0.0)
        gap = snap.gap_s[s] if 0 <= s < len(snap.gap_s) else math.inf
        row["gap_cur_s"] = float(min(gap, GAP_CAP_S))
        args = (s, tis, trans, snap.clearance_left_s, snap.target_stage)
        for j in TRAJETS:
            ks = site.trajet_stages.get(j, ())
            row[f"green_{j}"] = int(not trans and s in ks)
            row[f"ttg_min_{j}"] = time_to_green(site, ks, *args, site.min_green_s)
            row[f"ttg_nom_{j}"] = time_to_green(site, ks, *args, site.nominal_green_s)

    @staticmethod
    def _ambulance(snap: JunctionSnapshot, row: dict) -> None:
        calls = snap.emergency
        inserted = [c for c in calls if not c.pending]
        nearest = min(inserted, key=lambda c: c.dist_m) if inserted else None
        if nearest is not None:
            d = nearest.dist_m
            row["amb_dist_bin"] = 2 if d < AMB_NEAR_M[0] else 3 if d < AMB_NEAR_M[1] else 4
        else:
            row["amb_dist_bin"] = 1 if calls else 0
        first = nearest or (calls[0] if calls else None)
        arm = first.approach if first is not None else None
        row["amb_present"] = int(bool(calls))
        row["amb_pending"] = int(any(c.pending for c in calls))
        row["amb_arm"] = arm if arm in ARMS else NO_AMBULANCE
        row["preempt_active"] = int(snap.preempt_target is not None)

    def update(self, snap: JunctionSnapshot) -> dict:
        """{"t_feat": snap.t, <every non-trajet feature>, <DIAGNOSTIC>} (ROW_COLUMNS)."""
        t = float(snap.t)
        tod = 2.0 * math.pi * float(snap.clock_s) / 86400.0
        row: dict = {
            "t_feat": t,
            "controller": self.controller,
            "day_type": self.day_type,
            "tod_sin": math.sin(tod),
            "tod_cos": math.cos(tod),
        }
        for a in ARMS:
            row[f"halt_{a}"] = int(snap.arm_halting.get(a, 0))
            row[f"nveh_{a}"] = int(snap.arm_vehicles.get(a, 0))
            row[f"wait_max_{a}"] = float(snap.arm_wait_max_s.get(a, 0.0))
            jam = snap.queue_jam_m.get(a)
            row[f"jam_m_{a}"] = float(jam) if jam is not None else math.nan
            row[f"halt_out_{a}"] = int(snap.exit_halting.get(a, 0))
        self._rolling(snap, row)
        self._signal(snap, row)
        for a in BUS_APPROACHES:
            row[f"bus_{a}"] = sum(1 for b in snap.buses if b.approach == a)
        row["bus_dwell_s_out"] = int(snap.exit_bus_halting.get(EXIT_BUS_ARM, 0) > 0)
        self._ambulance(snap, row)
        for a in ARMS:
            since = [i.since_s for i in snap.incidents if i.arm == a]
            row[f"inc_{a}"] = int(bool(since))
            row[f"inc_age_s_{a}"] = float(t - min(since)) if since else 0.0
        row["ped_wait_max"] = float(max(snap.ped_wait_s, default=0.0))
        row["ped_count"] = int(sum(snap.ped_count))
        for a in ARMS:
            row[f"pending_{a}"] = int(snap.pending.get(a, 0))
        return {c: row[c] for c in ROW_COLUMNS}


# -- frames ----------------------------------------------------------------------------
def rows_frame(rows: Sequence[Mapping]) -> pd.DataFrame:
    """Feature rows (FeatureState.update dicts) as a frame with a fixed schema: ROW_COLUMNS
    in order, categorical features as strings, everything else float64."""
    df = pd.DataFrame(list(rows), columns=list(ROW_COLUMNS))
    for c in ROW_COLUMNS:
        if c in CATEGORIES:
            df[c] = df[c].astype(str)
        else:
            df[c] = df[c].astype("float64")
    return df


def join_samples(
    samples: pd.DataFrame, rows: pd.DataFrame, lag_s: int = 1, tol_s: float = 5.0
) -> pd.DataFrame:
    """Join each trip to the feature row at t_feat = floor(insert_t) - lag_s (merge_asof
    backward, tolerance tol_s): the latest row at or before that second, never one
    after it. Columns of `rows` already in `samples` (e.g. `controller`) are taken from
    `samples`. Trips without a row within the tolerance keep NaN features. The result
    keeps the order of `samples` and gains `t_feat` and the row features."""
    left = samples.reset_index(drop=True).copy()
    left["_order"] = np.arange(len(left))
    left["_key"] = np.floor(left["insert_t"].astype("float64")) - float(lag_s)
    cols = ["t_feat", *(c for c in rows.columns if c not in left.columns and c != "t_feat")]
    right = rows[cols].copy()
    right["t_feat"] = right["t_feat"].astype("float64")
    right = right.sort_values("t_feat", kind="stable")
    right["_key"] = right["t_feat"]
    out = pd.merge_asof(
        left.sort_values("_key", kind="stable"),
        right,
        on="_key",
        direction="backward",
        tolerance=float(tol_s),
        allow_exact_matches=True,
    )
    out = out.sort_values("_order", kind="stable").drop(columns=["_order", "_key"])
    return out.reset_index(drop=True)


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """ETA_FEATURES in order with fixed dtypes: categoricals with the CATEGORIES lists (a
    value outside a list becomes NaN), numeric features as float32."""
    missing = [f for f in ETA_FEATURES if f not in df.columns]
    if missing:
        raise KeyError(f"missing ETA features: {', '.join(missing)}")
    out = pd.DataFrame(index=df.index)
    for f in ETA_FEATURES:
        if f in CATEGORIES:
            cats = list(CATEGORIES[f])
            values = df[f].astype(str)
            out[f] = pd.Categorical(values.where(values.isin(cats)), categories=cats)
        else:
            out[f] = pd.to_numeric(df[f], errors="coerce").astype(MODEL_FLOAT)
    return out


def prediction_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (row t = t_feat + 1, trajet): the feature rows (e.g. a run's
    `eta_snapshots.parquet`) crossed with TRAJETS, sorted by (trajet, t). Columns: t,
    t_feat, then every ETA_FEATURES column (trajet included) and DIAGNOSTIC when
    present."""
    base = rows.copy()
    if "t_feat" not in base.columns:
        base["t_feat"] = base["t"].astype("float64") - 1.0
    base["t"] = base["t_feat"].astype("float64") + 1.0
    keep = ["t", "t_feat", *(c for c in (*ROW_FEATURES, *DIAGNOSTIC) if c in base.columns)]
    base = base[keep]
    parts = [base.assign(trajet=j) for j in TRAJETS]
    out = pd.concat(parts, ignore_index=True)
    cols = ["t", "t_feat", *ETA_FEATURES, *(d for d in DIAGNOSTIC if d in out.columns)]
    out = out[[c for c in cols if c in out.columns]]
    order = {j: k for k, j in enumerate(TRAJETS)}
    out = out.sort_values(["trajet", "t"], key=lambda s: s.map(order) if s.name == "trajet" else s)
    return out.reset_index(drop=True)

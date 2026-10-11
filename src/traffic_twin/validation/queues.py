"""Queue validation (13.4, D10): queues seen on the video at 2-3 instants per approach
against the simulated queue distribution around the same clock time.

The simulated cycle has no fixed phase relation to the real clock, so an observation at
time tau is compared with a sample pooled over all seeds within tau +- window_s:

- `cycle_max` (vehicles queued at the end of red): the maximum queue of each (seed,
  cycle) for the cycles overlapping the window; cycles start at the green onsets of the
  reference head;
- `instant`: every logged sample in the window.

Simulated queues (`queue.parquet`: jam_veh, jam_m per approach) are capped to what the
camera sees: q x min(1, visible_m x n_lanes / jam_m), with jam_m summed over the
approach's n_lanes lanes. PASS if p10 <= observed <= p90, or |median - observed| <=
queue_veh. Instants in the first `warmup_s` of the run or after its end are
OUT_OF_WINDOW.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from traffic_twin.validation import clock_s
from traffic_twin.validation.field import FieldData

QUEUE_COLUMNS = [
    "session", "time", "approach", "kind", "obs_veh", "visible_m", "sim_n", "sim_mean",
    "sim_p10", "sim_p50", "sim_p90", "obs_percentile", "abs_err", "truncated_share", "status",
]  # fmt: skip
WARMUP_S = 300.0


def capped(
    jam_veh: pd.Series, jam_m: pd.Series, visible_m: float | None, n_lanes: int = 1
) -> tuple[pd.Series, pd.Series]:
    """(queue seen by the camera, whether the cap applied) per sample."""
    jam_veh, jam_m = jam_veh.astype(float), jam_m.astype(float)
    if visible_m is None or pd.isna(visible_m):
        return jam_veh, pd.Series(False, index=jam_veh.index)
    seen = float(visible_m) * max(int(n_lanes), 1)
    factor = np.where(jam_m > seen, seen / jam_m.where(jam_m > 0, 1.0), 1.0)
    return jam_veh * factor, pd.Series(jam_m > seen, index=jam_veh.index)


def cycle_of(t: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """Cycle index of each time: 0 before the first onset, k after the k-th."""
    return np.searchsorted(np.asarray(onsets, dtype=float), np.asarray(t, dtype=float), "right")


def queue_sample(
    q: pd.DataFrame, t: float, kind: str, window_s: float, onsets: dict[int, np.ndarray]
) -> pd.DataFrame:
    """Sample (columns value, truncated) of one approach for an observation at t (s since
    the scenario start). `q` holds seed, t, value, truncated."""
    lo, hi = t - window_s, t + window_s
    if kind == "instant":
        return q[(q["t"] >= lo) & (q["t"] <= hi)][["value", "truncated"]]
    parts = []
    for seed, s in q.groupby("seed"):
        on = onsets.get(int(seed), np.array([]))
        cyc = cycle_of(s["t"].to_numpy(), on)
        edges = np.r_[-np.inf, on, np.inf]  # cycle k spans [edges[k], edges[k + 1])
        keep = [k for k in np.unique(cyc) if edges[k] <= hi and edges[k + 1] > lo]
        s = s.assign(cycle=cyc)
        s = s[s["cycle"].isin(keep)]
        if s.empty:
            continue
        idx = s.groupby("cycle")["value"].idxmax()
        parts.append(s.loc[idx, ["value", "truncated"]])
    return pd.concat(parts) if parts else pd.DataFrame(columns=["value", "truncated"])


def queue_table(
    queue: pd.DataFrame,
    field: FieldData,
    cfg: dict,
    thresholds: dict,
    onsets: dict[int, np.ndarray] | None = None,
) -> pd.DataFrame:
    """The `queue_validation.csv` table: one row per field queue observation. `queue`
    holds every run's queue.parquet rows with a `seed` column."""
    begin, duration = float(cfg["begin_s"]), float(cfg["duration_s"])
    window = float(thresholds.get("queue_window_s", 300))
    tol = float(thresholds.get("queue_veh", 2))
    lanes = {
        a: len(q.get("lanes", [])) for a, q in cfg.get("measurement", {}).get("queues", {}).items()
    }
    controller = str(cfg.get("controller", "fixed"))
    onsets = onsets or {}
    rows = []
    for obs in field.queues:
        row = {k: obs.get(k) for k in ("session", "time", "approach", "kind", "visible_m")}
        row["obs_veh"] = obs.get("vehicles")
        if obs.get("time") is None:
            rows.append({**row, "status": "NOT_MEASURED"})
            continue
        t = clock_s(obs["time"]) - begin
        if t < WARMUP_S or t > duration:
            rows.append({**row, "status": "OUT_OF_WINDOW"})
            continue
        if row["obs_veh"] is None:
            rows.append({**row, "status": "NOT_MEASURED"})
            continue
        a = obs["approach"]
        q = queue[queue["approach"] == a] if len(queue) else queue
        if q.empty:
            rows.append({**row, "status": "NOT_IN_SIM"})
            continue
        value, trunc = capped(q["jam_veh"], q["jam_m"], obs.get("visible_m"), lanes.get(a, 1))
        q = q.assign(value=value, truncated=trunc)
        sample = queue_sample(q, t, obs.get("kind", "instant"), window, onsets)
        if sample.empty:
            rows.append({**row, "status": "NOT_IN_SIM"})
            continue
        v = sample["value"].to_numpy(float)
        p10, p50, p90 = np.quantile(v, [0.10, 0.50, 0.90])
        o = float(row["obs_veh"])
        ok = p10 - 1e-9 <= o <= p90 + 1e-9 or abs(p50 - o) <= tol + 1e-9
        rows.append(
            {
                **row,
                "sim_n": len(v),
                "sim_mean": float(v.mean()),
                "sim_p10": p10,
                "sim_p50": p50,
                "sim_p90": p90,
                "obs_percentile": float((v <= o).mean()),
                "abs_err": p50 - o,
                "truncated_share": float(sample["truncated"].astype(bool).mean()),
                "status": ("PASS" if ok else "FAIL") if controller == "fixed" else "INFO",
            }
        )
    return pd.DataFrame(rows).reindex(columns=QUEUE_COLUMNS)

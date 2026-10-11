"""Flow validation (A-02, D9): simulated station flows from the E1 loops of each run
(`detector.parquet`) against the typical-day sensor flows (`reference.real_flows`).

Simulated flow of a station in a slot = vehicles counted on all its lanes during the
slot's detector intervals x 3600 / slot length, per seed; the table reports the mean
over the evaluation seeds. Status: PASS / FAIL on A-02 rows (inbound arm, A-02 slot,
|sim / real - 1| <= flow_rel), ok / off on control, information and sensitivity rows,
NO_SENSOR (no arc) and NO_DATA (too few sensor days, or no simulated value).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from traffic_twin.evaluation.stats import geh
from traffic_twin.validation import slot_bounds

FLOW_COLUMNS = [
    "station", "arm", "direction", "iu_ac", "alt_iu_ac", "group", "edges", "slot", "role",
    "real_mean", "real_std", "real_p10", "real_p90", "n_days", "sim_mean", "sim_std", "sim_min",
    "sim_max", "n_seeds", "abs_err", "rel_err_pct", "geh", "in_real_p10_p90", "seed_pass_share",
    "status",
]  # fmt: skip
COUNTED = ("PASS", "FAIL", "ok", "off")  # statuses with both a real and a simulated value
_EPS = 1e-6


class SlotNotCovered(ValueError):
    """The detector intervals of a run do not reach that far (e.g. a short run)."""


def station_counts(detectors: pd.DataFrame) -> pd.DataFrame:
    """Vehicles per station and detector interval, summed over the station's lanes."""
    keys = [k for k in ("seed", "station", "begin_s", "end_s") if k in detectors]
    return detectors.groupby(keys, as_index=False)["n_veh"].sum()


def slot_flow(counts: pd.DataFrame, begin: float, end: float) -> pd.Series:
    """Hourly-equivalent flow (veh/h) per station over [begin, end) (seconds since the
    scenario start). The slot must be a union of whole detector intervals."""
    if counts.empty or begin < counts["begin_s"].min() - _EPS or end > counts["end_s"].max() + _EPS:
        raise SlotNotCovered(f"slot [{begin:g}, {end:g}) s is outside the detector intervals")
    out = {}
    for station, g in counts.groupby("station"):
        inside = g[(g["begin_s"] >= begin - _EPS) & (g["end_s"] <= end + _EPS)]
        covered = float((inside["end_s"] - inside["begin_s"]).sum())
        if not math.isclose(covered, end - begin, abs_tol=1e-3):
            raise ValueError(
                f"slot [{begin:g}, {end:g}) s is not a union of detector intervals of {station}"
            )
        out[station] = float(inside["n_veh"].sum()) * 3600.0 / (end - begin)
    return pd.Series(out, name="flow_veh_h", dtype=float)


def sim_flows(detectors: pd.DataFrame, slots: list[dict], begin_s: float) -> pd.DataFrame:
    """Long table seed, station, slot, flow_veh_h. `detectors` holds every run's
    detector.parquet rows with a `seed` column; slots a run does not cover are left out."""
    rows = []
    counts = station_counts(detectors)
    for seed, c in counts.groupby("seed"):
        for slot in slots:
            try:
                flows = slot_flow(c, *slot_bounds(slot, begin_s))
            except SlotNotCovered:
                continue
            rows += [
                {"seed": seed, "station": st, "slot": slot["name"], "flow_veh_h": q}
                for st, q in flows.items()
            ]
    return pd.DataFrame(rows, columns=["seed", "station", "slot", "flow_veh_h"])


def _edges(lanes: list[str]) -> str:
    return " ".join(dict.fromkeys(lane.rsplit("_", 1)[0] for lane in lanes))


def _rows(stations: list[dict], slots: list[dict], lanes: dict[str, list[str]]) -> list[dict]:
    """One row per station arc (and sensitivity arc) and slot, with its role."""
    rows = []
    for st in stations:
        variants = [(st.get("iu_ac"), st.get("alt_iu_ac"), st.get("group"), False)]
        if st.get("alt_iu_ac") not in (None, ""):
            alt_group = st.get("alt_group") or f"alt_{st['alt_iu_ac']}"
            variants.append((st["alt_iu_ac"], None, alt_group, True))
        for iu, alt, group, sensitivity in variants:
            for slot in slots:
                if sensitivity:
                    role = "sensitivity"
                elif st.get("a02") and slot.get("role") == "A-02":
                    role = "A-02"
                else:
                    role = "control" if st.get("a02") else "info"
                rows.append(
                    {
                        "station": st["station"],
                        "arm": st.get("arm"),
                        "direction": st.get("direction"),
                        "iu_ac": None if iu in (None, "") else str(iu),
                        "alt_iu_ac": None if alt in (None, "") else str(alt),
                        "group": group,
                        "edges": _edges(lanes.get(st["station"], [])),
                        "slot": slot["name"],
                        "role": role,
                    }
                )
    return rows


def flow_table(
    real: pd.DataFrame,
    sim: pd.DataFrame,
    stations: list[dict],
    slots: list[dict],
    thresholds: dict,
    lanes: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """The `flow_validation.csv` table. `real` comes from `reference.real_flows`, `sim`
    from `sim_flows`; `lanes` maps station ids to their loop lanes (for the edges)."""
    tol = float(thresholds.get("flow_rel", 0.15))
    t = pd.DataFrame(_rows(stations, slots, lanes or {}))
    real_cols = ["station", "iu_ac", "slot", "real_mean", "real_std", "real_p10", "real_p90"]
    real = real.assign(real_status=real["status"])[[*real_cols, "n_days", "real_status"]]
    t = t.merge(real, on=["station", "iu_ac", "slot"], how="left")
    agg = sim.groupby(["station", "slot"])["flow_veh_h"].agg(
        sim_mean="mean", sim_std="std", sim_min="min", sim_max="max", n_seeds="count"
    )
    t = t.merge(agg.reset_index(), on=["station", "slot"], how="left")
    t["n_seeds"] = t["n_seeds"].fillna(0).astype(int)
    t["n_days"] = t["n_days"].fillna(0).astype(int)
    t["abs_err"] = t["sim_mean"] - t["real_mean"]
    t["rel_err_pct"] = 100.0 * t["abs_err"] / t["real_mean"]
    t["geh"] = [
        geh(s, r) if pd.notna(s) and pd.notna(r) else np.nan
        for s, r in zip(t["sim_mean"], t["real_mean"], strict=True)
    ]
    t["in_real_p10_p90"] = [
        bool(lo <= s <= hi) if pd.notna(s) and pd.notna(lo) and pd.notna(hi) else None
        for s, lo, hi in zip(t["sim_mean"], t["real_p10"], t["real_p90"], strict=True)
    ]
    per_seed = sim.merge(t[["station", "slot", "iu_ac", "real_mean"]], on=["station", "slot"])
    per_seed["ok"] = (per_seed["flow_veh_h"] / per_seed["real_mean"] - 1).abs() <= tol + 1e-12
    share = per_seed.dropna(subset=["real_mean"]).groupby(["station", "iu_ac", "slot"])["ok"]
    t = t.merge(
        share.mean().rename("seed_pass_share").reset_index(),
        on=["station", "iu_ac", "slot"],
        how="left",
    )

    def status(r) -> str:
        if r["iu_ac"] is None or pd.isna(r["iu_ac"]):
            return "NO_SENSOR"
        if r["real_status"] != "ok" or pd.isna(r["real_mean"]) or pd.isna(r["sim_mean"]):
            return "NO_DATA"
        ok = abs(r["sim_mean"] / r["real_mean"] - 1) <= tol + 1e-12
        if r["role"] == "A-02":
            return "PASS" if ok else "FAIL"
        return "ok" if ok else "off"

    t["status"] = t.apply(status, axis=1) if len(t) else []
    return t.reindex(columns=FLOW_COLUMNS)


def a02_summary(table: pd.DataFrame) -> dict:
    """A-02 verdict: PASS when every A-02 row passes, FAIL when one fails, INCOMPLETE
    when a row lacks data (and none fails)."""
    rows = table[table["role"] == "A-02"]
    n_fail = int((rows["status"] == "FAIL").sum())
    n_pass = int((rows["status"] == "PASS").sum())
    if rows.empty:
        verdict = "INCOMPLETE"
    elif n_fail:
        verdict = "FAIL"
    else:
        verdict = "PASS" if n_pass == len(rows) else "INCOMPLETE"
    return {"verdict": verdict, "rows": len(rows), "pass": n_pass, "fail": n_fail}


def geh_summary(table: pd.DataFrame, slots: list[dict], thresholds: dict) -> dict:
    """Share of distinct counts with GEH below the threshold (D9). A count is one station
    group in one A-02 slot: arcs sharing a flow station count once (first row in table
    order), sensitivity rows and rows without data are left out."""
    a02_slots = {s["name"] for s in slots if s.get("role") == "A-02"}
    rows = table[
        table["slot"].isin(a02_slots)
        & (table["role"] != "sensitivity")
        & table["status"].isin(COUNTED)
    ].drop_duplicates(["group", "slot"])
    limit, need = float(thresholds.get("geh", 5.0)), float(thresholds.get("geh_share", 0.85))
    share = float((rows["geh"] < limit).mean()) if len(rows) else math.nan
    return {
        "n_counts": len(rows),
        "n_below": int((rows["geh"] < limit).sum()),
        "share": share,
        "ok": bool(len(rows) and share >= need),
        "counts": [f"{g} {s}" for g, s in zip(rows["group"], rows["slot"], strict=True)],
    }

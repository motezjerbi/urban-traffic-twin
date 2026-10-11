"""Signal validation (13.4, D10): simulated cycle and per-head timings against the field
timing in the field file.

- Simulated greens come from each run's `signal.parquet` (green intervals per TLS link,
  from SaveTLSSwitchTimes). A signal head (`network.signal_groups`) is green while any of
  its links is green (G or g), so a vehicle head's green includes the trailing clearance
  phases where it stays green. The cycle is the time between successive green onsets of
  `network.reference_group`.
- A head green more than once per cycle (ped_NB: in HOP and again in STM) is validated
  per stage, `<head>.green_s@<stage>` (the stage whose phase the green starts in, cycle
  order from the reference onset), never as the mean of different greens; its field
  values are given per stage too (`green_s: {HOP: [...], STM: [...]}`).
- Yellow and all-red come from the program (SignalPlan replays its clearance phases): the
  net's phase states with each stage's green set to `fixed.green_s`. The all-red after a
  vehicle head is the time from the end of its yellow to the next green onset of any
  vehicle link (the stage transition's all-red).
- Status: PASS / FAIL (S0 only, tolerances in `thresholds`), INFO (other controllers,
  descriptive only), NOT_MEASURED (no field value), ASSUMED (`signal.source:
  assumption`), NOT_IN_SIM (measured but not modelled, e.g. flashing pedestrian green).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_twin.config import ConfigError
from traffic_twin.validation.field import FieldData, multi_green_message, numbers

GREEN, YELLOW = "Gg", "y"
SIGNAL_COLUMNS = [
    "item", "group", "kind", "model_value", "sim_mean", "sim_std", "sim_min", "sim_max",
    "sim_n", "field_value", "field_n", "field_source", "abs_err", "tol", "status",
]  # fmt: skip
_TOL_KEY = {"cycle": "cycle_s", "green": "green_s", "yellow": "yellow_s", "all_red": "all_red_s"}


# -- program ---------------------------------------------------------------------------
def read_program(net_file: str | Path, tls_id: str) -> tuple[list[str], list[float]]:
    """(phase states, phase durations) of the TLS program in a .net.xml."""
    for _, el in ET.iterparse(net_file, events=("end",)):
        if el.tag == "tlLogic" and el.get("id") == tls_id:
            phases = el.findall("phase")
            return [p.get("state") for p in phases], [float(p.get("duration")) for p in phases]
    raise ConfigError(f"{net_file}: no tlLogic {tls_id!r}")


def model_durations(
    durations: list[float], stages: list[dict], green_s: dict | None
) -> list[float]:
    """Program durations with each stage's green phase set to its fixed green."""
    out = list(durations)
    for st in stages:
        if green_s and st["name"] in green_s:
            out[int(st["green_phase"])] = float(green_s[st["name"]])
    return out


def link_intervals(
    states: list[str],
    durations: list[float],
    chars: str = GREEN,
    n_cycles: int = 1,
    t0: float = 0.0,
) -> pd.DataFrame:
    """Intervals (link, begin_s, end_s, duration_s) during which each link shows one of
    `chars`, with the program cycling n_cycles times from phase 0 at t0. Like
    SaveTLSSwitchTimes, an interval is written only once it ends (so the last one, still
    open at the end, is not)."""
    n_links = len(states[0])
    start: dict[int, float] = {}
    rows, t = [], float(t0)
    for _ in range(n_cycles):
        for state, dur in zip(states, durations, strict=True):
            for i in range(n_links):
                on = state[i] in chars
                if on and i not in start:
                    start[i] = t
                elif not on and i in start:
                    b = start.pop(i)
                    rows.append((i, b, t, t - b))
            t += dur
    df = pd.DataFrame(rows, columns=["link", "begin_s", "end_s", "duration_s"])
    return df.sort_values(["begin_s", "link"], ignore_index=True)


def group_intervals(intervals: pd.DataFrame, links: list[int]) -> pd.DataFrame:
    """Union of the intervals of a group's links (begin_s, end_s, duration_s)."""
    g = intervals[intervals["link"].isin(links)].sort_values("begin_s")
    merged: list[list[float]] = []
    for b, e in zip(g["begin_s"], g["end_s"], strict=True):
        if merged and b <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([b, e])
    out = pd.DataFrame(merged, columns=["begin_s", "end_s"])
    out["duration_s"] = out["end_s"] - out["begin_s"]
    return out


def ped_links_of(cfg: dict) -> set[int]:
    return {int(c["link"]) for c in cfg.get("network", {}).get("crossings", {}).values()}


def phase_stages(stages: list[dict], n_phases: int) -> list[str]:
    """Stage name of every phase index (its green phase and its clearance phases)."""
    out = [f"phase{i}" for i in range(n_phases)]
    for st in stages:
        for p in [st["green_phase"], *st.get("clearance_phases", [])]:
            if 0 <= int(p) < n_phases:
                out[int(p)] = str(st["name"])
    return out


def _unique(tags: list[str]) -> list[str]:
    """Stage tags made unique: a second green in the same stage is '<stage>.2'."""
    seen: dict[str, int] = {}
    out = []
    for t in tags:
        seen[t] = seen.get(t, 0) + 1
        out.append(t if seen[t] == 1 else f"{t}.{seen[t]}")
    return out


def _cycle_greens(
    states: list[str],
    durations: list[float],
    groups: dict[str, list[int]],
    stages: list[dict],
    reference_group: str | None,
) -> dict[str, pd.DataFrame]:
    """Per head: its green intervals (begin_s, duration_s, stage) over one program cycle,
    in cycle order from the reference head's green onset."""
    cycle = float(sum(durations))
    names = phase_stages(stages, len(states))
    starts = np.cumsum([0.0, *durations[:-1]])
    green = link_intervals(states, durations, GREEN, n_cycles=3)
    ref = group_intervals(green, list(groups.get(reference_group or "", [])))
    onsets = ref["begin_s"][ref["begin_s"] >= cycle - 1e-6]
    r = float(onsets.iloc[0]) if len(onsets) else cycle
    out = {}
    for name, links in groups.items():
        g = group_intervals(green, links)
        g = g[(g["begin_s"] >= r - 1e-6) & (g["begin_s"] < r + cycle - 1e-6)].copy()
        phases = np.searchsorted(starts, (g["begin_s"] % cycle) + 1e-6, side="right") - 1
        g["stage"] = _unique([names[int(p)] for p in phases])
        out[name] = g.reset_index(drop=True)
    return out


def head_stages(
    states: list[str],
    durations: list[float],
    groups: dict[str, list[int]],
    stages: list[dict],
    reference_group: str | None = None,
) -> dict[str, list[str]]:
    """Per head: the stage in which each of its green intervals starts, in cycle order from
    the reference head's green onset (e.g. ped_NB -> [HOP, STM])."""
    greens = _cycle_greens(states, durations, groups, stages, reference_group)
    return {name: g["stage"].tolist() for name, g in greens.items()}


def program_items(
    states: list[str],
    durations: list[float],
    groups: dict[str, list[int]],
    ped_links: set[int],
    stages: list[dict] | None = None,
    reference_group: str | None = None,
) -> dict[str, float]:
    """Model timing per item from the (model) program: cycle_s, <group>.green_s, and for
    vehicle heads yellow_s and all_red_after_s. With `stages`, a head green more than once
    per cycle gets one item per green, <group>.green_s@<stage> (never their mean)."""
    cycle = float(sum(durations))
    green = link_intervals(states, durations, GREEN, n_cycles=3)
    yellow = link_intervals(states, durations, YELLOW, n_cycles=3)
    per_stage = _cycle_greens(states, durations, groups, stages, reference_group) if stages else {}

    def middle(df: pd.DataFrame) -> pd.DataFrame:  # whole intervals, away from the edges
        return df[(df["begin_s"] >= cycle - 1e-6) & (df["begin_s"] < 2 * cycle - 1e-6)]

    vehicle = [i for i in range(len(states[0])) if i not in ped_links]
    onsets = np.sort(green[green["link"].isin(vehicle)]["begin_s"].unique())
    items = {"cycle_s": cycle}
    for name, links in groups.items():
        g = middle(group_intervals(green, links))
        stage_greens = per_stage.get(name)
        if stage_greens is not None and len(stage_greens) > 1:
            for stage, d in zip(stage_greens["stage"], stage_greens["duration_s"], strict=True):
                items[f"{name}.green_s@{stage}"] = float(d)
        else:
            items[f"{name}.green_s"] = float(g["duration_s"].mean()) if len(g) else 0.0
        if set(links) <= ped_links:
            continue
        y = middle(group_intervals(yellow, links))
        items[f"{name}.yellow_s"] = float(y["duration_s"].mean()) if len(y) else 0.0
        ends = y["end_s"] if len(y) else g["end_s"]  # no yellow: red right after green
        reds = [
            float(onsets[onsets >= e - 1e-6][0] - e) for e in ends if (onsets >= e - 1e-6).any()
        ]
        items[f"{name}.all_red_after_s"] = float(np.mean(reds)) if reds else np.nan
    return items


def ped_only_stage(states: list[str], stages: list[dict], ped_links: set[int]) -> bool:
    """True if a stage's green phase gives green to pedestrian links only."""
    for st in stages:
        s = states[int(st["green_phase"])]
        if not any(c in GREEN for i, c in enumerate(s) if i not in ped_links):
            return True
    return False


# -- simulated side --------------------------------------------------------------------
def sim_items(
    signal: pd.DataFrame,
    groups: dict[str, list[int]],
    reference_group: str,
    duration_s: float,
    stage_tags: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Long table seed, item, value: every green interval of every head and every cycle
    of the reference head, for intervals starting inside the demand window. A head with
    several stage tags (`head_stages`, green more than once per cycle) gives one item per
    stage, <head>.green_s@<stage>: within each complete cycle of the reference head its
    greens are matched to the tags in order (a cycle with another number of greens for
    that head is left out)."""
    rows = []
    for seed, s in signal.groupby("seed"):
        s = s[(s["begin_s"] >= 0) & (s["begin_s"] < duration_s)]
        ref = group_intervals(s, list(groups.get(reference_group, [])))
        onsets = np.sort(ref["begin_s"].to_numpy())
        for name, links in groups.items():
            g = group_intervals(s, links)
            tags = (stage_tags or {}).get(name) or []
            if len(tags) > 1:
                for b, e in zip(onsets[:-1], onsets[1:], strict=True):
                    w = g[(g["begin_s"] >= b - 1e-6) & (g["begin_s"] < e - 1e-6)]
                    if len(w) == len(tags):
                        rows += [
                            (seed, f"{name}.green_s@{t}", v)
                            for t, v in zip(tags, w["duration_s"], strict=True)
                        ]
            else:
                rows += [(seed, f"{name}.green_s", v) for v in g["duration_s"]]
            if name == reference_group:
                rows += [(seed, "cycle_s", v) for v in np.diff(np.sort(g["begin_s"].to_numpy()))]
    return pd.DataFrame(rows, columns=["seed", "item", "value"])


def cycle_onsets(signal: pd.DataFrame, links: list[int]) -> dict[int, np.ndarray]:
    """Per seed: green onsets (s since the scenario start) of the reference head."""
    return {
        int(seed): np.sort(group_intervals(s, links)["begin_s"].to_numpy())
        for seed, s in signal.groupby("seed")
    }


# -- table -----------------------------------------------------------------------------
def _stage_tags(item: str, model: dict[str, float], sim: pd.DataFrame) -> list[str]:
    """The stage tags of a head validated per stage (items '<item>@<stage>' of the model,
    in cycle order, then of the simulation); [] for a head green once per cycle."""
    prefix = f"{item}@"
    keys = [k for k in model if k.startswith(prefix)]
    if len(sim):
        keys += [k for k in sim["item"].unique() if k.startswith(prefix) and k not in keys]
    return [k[len(prefix) :] for k in keys]


def green_field_values(item: str, tags: list[str], raw: object) -> list[tuple[str, list[float]]]:
    """(item, field values) rows of one head's green. `raw` is the field `green_s`: a list
    (one value per green), or a mapping stage -> list for a head green more than once per
    cycle. A head the model shows green in several stages (`tags`) needs the mapping (a
    plain list cannot be matched to a stage: ConfigError). A stage the model does not give
    the head becomes its own row (NOT_IN_SIM once measured)."""
    head = item.split(".")[0]
    by_stage = {str(k): numbers(v) for k, v in raw.items()} if isinstance(raw, dict) else None
    if tags:
        if by_stage is None:
            if numbers(raw):
                raise ConfigError(
                    f"field signal.heads.{head}.green_s: {multi_green_message(head, tags)}"
                )
            by_stage = {}
        rows = [(f"{item}@{t}", by_stage.get(t, [])) for t in tags]
        return rows + [(f"{item}@{t}", v) for t, v in by_stage.items() if t not in tags]
    if by_stage is None:
        return [(item, numbers(raw))]
    measured = {t: v for t, v in by_stage.items() if v}
    if len(measured) <= 1:
        return [(item, next(iter(measured.values()), []))]
    # the field shows several greens per cycle where the model has one
    return [(item, []), *((f"{item}@{t}", v) for t, v in measured.items())]


def _status(measured: bool, in_sim: bool, source: str, controller: str, ok: bool) -> str:
    if not measured:
        return "NOT_MEASURED"
    if not in_sim:
        return "NOT_IN_SIM"
    if source == "assumption":
        return "ASSUMED"
    if controller != "fixed":
        return "INFO"
    return "PASS" if ok else "FAIL"


def signal_table(
    cfg: dict,
    field: FieldData,
    sim: pd.DataFrame,
    model: dict[str, float],
    thresholds: dict,
    ped_only: bool | None = None,
) -> pd.DataFrame:
    """The `signal_validation.csv` table for scenario `cfg`: one row per item (cycle,
    head green / yellow / all-red / flashing, stage structure)."""
    net = cfg["network"]
    groups = {k: list(v) for k, v in net.get("signal_groups", {}).items()}
    peds = ped_links_of(cfg)
    controller = str(cfg.get("controller", "fixed"))
    source = field.signal_source
    rows = []

    def numeric(item: str, group: str | None, kind: str, field_values: list[float]) -> None:
        s = sim.loc[sim["item"] == item, "value"] if len(sim) else pd.Series(dtype=float)
        m = model.get(item)
        n = len(s)
        if s.empty and kind in ("yellow", "all_red") and m is not None and pd.notna(m):
            s = pd.Series([m])  # clearances are replayed from the program (sim_n 0)
        in_sim = not s.empty
        fmean = float(np.mean(field_values)) if field_values else np.nan
        smean = float(s.mean()) if in_sim else np.nan
        tol = float(thresholds.get(_TOL_KEY.get(kind, ""), np.nan))
        err = smean - fmean
        ok = bool(abs(err) <= tol + 1e-9) if pd.notna(err) else False
        rows.append(
            {
                "item": item,
                "group": group,
                "kind": kind,
                "model_value": m,
                "sim_mean": smean,
                "sim_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0 if in_sim else np.nan,
                "sim_min": float(s.min()) if in_sim else np.nan,
                "sim_max": float(s.max()) if in_sim else np.nan,
                "sim_n": n,
                "field_value": fmean,
                "field_n": len(field_values),
                "field_source": source if field_values else None,
                "abs_err": err,
                "tol": tol,
                "status": _status(bool(field_values), in_sim, source, controller, ok),
            }
        )

    numeric("cycle_s", net.get("reference_group"), "cycle", field.cycle_values())
    for name, links in groups.items():
        item = f"{name}.green_s"
        tags = _stage_tags(item, model, sim)
        for green_item, values in green_field_values(item, tags, field.head_raw(name, "green_s")):
            numeric(green_item, name, "green", values)
        if set(links) <= peds:
            numeric(f"{name}.flashing_s", name, "flashing", field.head_values(name, "flashing_s"))
        else:
            numeric(f"{name}.yellow_s", name, "yellow", field.head_values(name, "yellow_s"))
            numeric(
                f"{name}.all_red_after_s",
                name,
                "all_red",
                field.head_values(name, "all_red_after_s"),
            )

    # Stage structure (qualitative): model value vs field answer
    names = [s["name"] for s in net.get("stages", [])]
    sig = field.signal
    cyc = sim.loc[sim["item"] == "cycle_s", "value"] if len(sim) else pd.Series(dtype=float)
    varies = bool(len(cyc) > 1 and cyc.std(ddof=1) > float(thresholds.get("cycle_s", 2.0)))
    structure = [
        ("stage_order", " > ".join(names), sig.get("stage_order") or None),
        ("varies_between_cycles", varies, sig.get("varies_between_cycles")),
        (
            "bus_stages_on_demand",
            False,
            sig.get("bus_stages_on_demand"),
        ),  # every stage, every cycle
        ("ped_only_stage", ped_only, sig.get("ped_only_stage")),
    ]
    for item, model_value, observed in structure:
        if isinstance(observed, list):
            in_sim = all(s in names for s in observed)
            observed = " > ".join(observed)
        else:
            in_sim = model_value is not None
        measured = observed is not None
        rows.append(
            {
                "item": item,
                "group": None,
                "kind": "structure",
                "model_value": model_value,
                "sim_mean": None,
                "sim_n": 0,
                "field_value": observed,
                "field_n": int(measured),
                "field_source": source if measured else None,
                "status": _status(measured, in_sim, source, controller, observed == model_value),
            }
        )
    return pd.DataFrame(rows).reindex(columns=SIGNAL_COLUMNS)

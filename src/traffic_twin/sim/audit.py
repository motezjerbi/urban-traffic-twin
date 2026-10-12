"""Audit of the signal phases actually shown during a run (evidence for A-04, D31).

Counts, over the per-step sequence of phase indices:
- `veh_green_to_red_without_yellow`: a vehicle link going from G/g to r without a yellow;
- `ped_green_to_red_outside_clearance`: a pedestrian link going from G to r into a
  phase that is not a clearance phase of the stage being left;
- `yellow_to_green`: a link going from yellow to green;
- `green_shorter_than_min`: a complete green shorter than the minimum green;
- `out_of_order_green`: a green stage served out of cycle order. Information only
  (`INFO_KEYS`): preemption and free-order controllers skip stages legitimately;
- `clearance_incomplete`: between two greens, the phases shown are not exactly the
  leaving stage's clearance phases, each for its programmed duration (a cut yellow or
  all-red); needs `durations`;
- `intergreen_short`: a link turning green less than the required intergreen after a
  foe link's green ended (`tls_check.intergreen_requirements`); needs `intergreen`;
- `foe_vehicle_greens`: two foe vehicle links both G or g in one phase; needs `foes`;
- `illegal_transition`: green b after green a with b not in the guard's link-safe
  `allowed[a]`; needs `allowed`;
- `unknown_phase`: a phase index outside the program.

A safe run (any controller behind the SafetyGuard) has every `DANGER_KEYS` count at 0.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence

from traffic_twin.sim.signal import Stage

AUDIT_KEYS = (
    "veh_green_to_red_without_yellow",
    "ped_green_to_red_outside_clearance",
    "yellow_to_green",
    "green_shorter_than_min",
    "out_of_order_green",
    "clearance_incomplete",
    "intergreen_short",
    "foe_vehicle_greens",
    "illegal_transition",
    "unknown_phase",
)
INFO_KEYS = ("out_of_order_green",)
DANGER_KEYS = tuple(k for k in AUDIT_KEYS if k not in INFO_KEYS)
GREEN, YELLOW = "Gg", "y"  # SUMO link states


def _runs(shown: Sequence[int]) -> list[tuple[int, int]]:
    """Run-length encoding: [(phase, steps), ...]."""
    out: list[tuple[int, int]] = []
    for p in shown:
        if out and out[-1][0] == p:
            out[-1] = (p, out[-1][1] + 1)
        else:
            out.append((p, 1))
    return out


def phase_steps(duration_s: float, step_s: float) -> int:
    """Steps a clearance phase of `duration_s` is shown for by SignalPlan: it is shown at
    least once and dropped as soon as its remaining time reaches 0."""
    return max(1, math.ceil((duration_s - 1e-9) / step_s))


def audit_phases(
    shown: Sequence[int],
    phase_states: Sequence[str],
    stages: Sequence[Stage],
    min_green_s: float,
    step_s: float = 1.0,
    ped_links: Collection[int] | None = None,
    *,
    durations: Sequence[float] | None = None,
    allowed: Sequence[Collection[int]] | None = None,
    intergreen: Mapping[tuple[int, int], float] | None = None,
    foes: Mapping[int, Collection[int]] | None = None,
) -> dict[str, int]:
    """Violation counts per AUDIT_KEYS for the phases shown at each step. `ped_links`
    are the TLS link indices of pedestrian crossings (default: links never y or g in the
    program). A key whose input (`durations`, `allowed`, `intergreen`, `foes`) is None
    stays 0.

    `shown` runs from the start of the run: its last green may be cut by the end of the
    run and is not checked against the min green, nor is a trailing transition. A link
    change counts once per link, a foe pair once per phase shown, a short intergreen
    once per (ended link, starting link) pair."""
    counts = dict.fromkeys(AUDIT_KEYS, 0)
    n_phases = len(phase_states)
    if ped_links is None:
        n_links = len(phase_states[0]) if phase_states else 0
        ped_links = [i for i in range(n_links) if not any(s[i] in "yg" for s in phase_states)]
    peds = set(ped_links)
    stage_of = {st.green_phase: k for k, st in enumerate(stages)}
    clearance_of = [set(st.clearance_phases) for st in stages]
    any_clearance = set().union(*clearance_of)

    known = [p for p in shown if 0 <= p < n_phases]
    all_runs = _runs(shown)
    counts["unknown_phase"] = sum(not 0 <= p < n_phases for p, _ in all_runs)

    # -- link changes (Part 3 keys), over the known phases ----------------------------
    leaving: int | None = None  # stage whose green was shown last
    runs = _runs(known)
    for (a, _), (b, _) in zip(runs, runs[1:], strict=False):
        leaving = stage_of.get(a, leaving)
        into_clearance = b in (any_clearance if leaving is None else clearance_of[leaving])
        for i, (x, y) in enumerate(zip(phase_states[a], phase_states[b], strict=True)):
            if x in GREEN and y not in GREEN:
                if i in peds:
                    if not into_clearance:
                        counts["ped_green_to_red_outside_clearance"] += 1
                elif y not in YELLOW:
                    counts["veh_green_to_red_without_yellow"] += 1
            elif x in YELLOW and y in GREEN:
                counts["yellow_to_green"] += 1

    greens = [(stage_of[p], n) for p, n in runs if p in stage_of]
    complete = greens if runs and runs[-1][0] not in stage_of else greens[:-1]
    counts["green_shorter_than_min"] = sum(n * step_s < min_green_s - 1e-9 for _, n in complete)
    n = len(stages)
    counts["out_of_order_green"] = sum(
        b != (a + 1) % n for (a, _), (b, _) in zip(greens, greens[1:], strict=False)
    )

    # -- transitions between greens (D31) ---------------------------------------------
    if durations is not None or allowed is not None:
        expected = None
        if durations is not None:
            expected = [
                [(q, phase_steps(float(durations[q]), step_s)) for q in st.clearance_phases]
                for st in stages
            ]
        last_green: int | None = None
        between: list[tuple[int, int]] = []  # phases shown since the last green
        for p, k in all_runs:
            if p not in stage_of:
                between.append((p, k))
                continue
            b = stage_of[p]
            if last_green is not None:
                if expected is not None and between != expected[last_green]:
                    counts["clearance_incomplete"] += 1
                if allowed is not None and b not in allowed[last_green]:
                    counts["illegal_transition"] += 1
            last_green, between = b, []

    if foes is not None:
        counts["foe_vehicle_greens"] = _foe_vehicle_greens(runs, phase_states, peds, foes)
    if intergreen is not None:
        counts["intergreen_short"] = _intergreen_short(shown, phase_states, intergreen, step_s)
    return counts


def _foe_vehicle_greens(
    runs: Sequence[tuple[int, int]],
    phase_states: Sequence[str],
    peds: Collection[int],
    foes: Mapping[int, Collection[int]],
) -> int:
    """Foe vehicle-link pairs green (G or g) together, once per phase shown."""
    per_phase: dict[int, int] = {}
    total = 0
    for p, _ in runs:
        if p not in per_phase:
            green = [i for i, ch in enumerate(phase_states[p]) if ch in GREEN and i not in peds]
            per_phase[p] = sum(
                1
                for x, a in enumerate(green)
                for b in green[x + 1 :]
                if b in foes.get(a, ()) or a in foes.get(b, ())
            )
        total += per_phase[p]
    return total


def _intergreen_short(
    shown: Sequence[int],
    phase_states: Sequence[str],
    intergreen: Mapping[tuple[int, int], float],
    step_s: float,
) -> int:
    """(ended link i, starting link j) pairs where j turned green less than
    intergreen[(i, j)] after i's green ended, i not green again meanwhile."""
    n_phases = len(phase_states)
    last_end: dict[int, float] = {}  # link -> time its last green ended
    count = 0
    prev: int | None = None
    for step, p in enumerate(shown):
        if not 0 <= p < n_phases:
            continue
        if prev is not None and p != prev:
            t = step * step_s
            before, now = phase_states[prev], phase_states[p]
            starts = []
            for i, (x, y) in enumerate(zip(before, now, strict=True)):
                if x in GREEN and y not in GREEN:
                    last_end[i] = t
                elif y in GREEN and x not in GREEN:
                    starts.append(i)
            for j in starts:
                for i, end in last_end.items():
                    need = intergreen.get((i, j))
                    if need is not None and now[i] not in GREEN and t - end < need - 1e-9:
                        count += 1
        prev = p
    return count

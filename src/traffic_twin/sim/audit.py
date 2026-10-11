"""Audit of the signal phases actually shown during a run (evidence for A-04).

Counts, over the per-step sequence of phase indices:
- a vehicle link going from G/g to r without a yellow in between;
- a pedestrian link going from G to r into a phase that is not a clearance phase of
  the stage being left;
- a link going from yellow to green;
- a complete green shorter than the minimum green;
- a green stage served out of the configured order.

A safe run (any controller behind the SafetyGuard) has every count at 0.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from traffic_twin.sim.signal import Stage

AUDIT_KEYS = (
    "veh_green_to_red_without_yellow",
    "ped_green_to_red_outside_clearance",
    "yellow_to_green",
    "green_shorter_than_min",
    "out_of_order_green",
)
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


def audit_phases(
    shown: Sequence[int],
    phase_states: Sequence[str],
    stages: Sequence[Stage],
    min_green_s: float,
    step_s: float = 1.0,
    ped_links: Collection[int] | None = None,
) -> dict[str, int]:
    """Violation counts per AUDIT_KEYS for the phases shown at each step. `ped_links`
    are the TLS link indices of pedestrian crossings (default: links never y or g in the
    program).

    `shown` runs from the start of the run: its last green may be cut by the end of the
    run and is not checked against the min green. A link change counts once per link."""
    counts = dict.fromkeys(AUDIT_KEYS, 0)
    if ped_links is None:
        n_links = len(phase_states[0]) if phase_states else 0
        ped_links = [i for i in range(n_links) if not any(s[i] in "yg" for s in phase_states)]
    peds = set(ped_links)
    stage_of = {st.green_phase: k for k, st in enumerate(stages)}
    clearance_of = [set(st.clearance_phases) for st in stages]
    any_clearance = set().union(*clearance_of)

    leaving: int | None = None  # stage whose green was shown last
    runs = _runs(shown)
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
    return counts

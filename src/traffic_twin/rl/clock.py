"""Semi-MDP decision clock (Part 6, D56; contract 4.7), shared by `TrafficEnv` and
`RLController` so that both ask the policy at the same seconds and decode its action the
same way.

A decision point needs all of:
- not in transition (yellow and all-red run automatically, from `SignalPlan`);
- `time_in_stage >= min_green_s` (before, a switch request would be ignored);
- no active preemption (`EXTRA_PREEMPT is None`), when `skip_during_preemption`;
- at least `interval_s` since the last decision.

At a decision point the action (D55, `Discrete(2)` under `safety.stage_order: strict`)
decodes to a stage request for that step only: 1 -> the cycle successor, 0 -> the
current stage. On every other step the request is the current stage (hold), so a request
the guard replaced is never replayed later. With a 6 s min green and 5 s spacing, greens
fall on 6 + 5k s.

    clock = DecisionClock(interval_s=5, min_green_s=6)
    clock.reset()
    while ...:
        obs = sim.observe()
        action = policy(...) if clock.is_decision_point(obs) else None
        sim.step(clock.request(obs, action, n_stages))

`InterventionSplit` (Part 7, D77) classifies the Safety Guard's interventions of an
episode or run, for the env summary and `RLController.report()`:
- `cut`: key `ped_wait` or `max_green` on a step whose request was the current stage
  (the policy held it): the shield or max green ended a held green at the binding
  second, the guard's designed job;
- `replaced`: every other intervention (total minus `cut`, so the two sum to the guard's
  count; warm-up interventions, if any, are `replaced`);
- `cut_foreseeable` (part of `cut`): the green's last decision was "stay" while the
  observed minimum pedestrian slack (`EXTRA_PED_SLACK`) was below `interval_s`, or
  time_in_stage + interval_s >= the stage's max green.
It reads the guard's counters (one intervention per rule per green) after each step.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from traffic_twin.control.base import EXTRA_PED_SLACK, EXTRA_PREEMPT, Observation

EPS = 1e-9
STAY, ADVANCE = 0, 1  # the two actions (D55)
IV_SPLIT_KEYS = ("cut", "replaced", "cut_foreseeable")  # D77
CUT_KEYS = ("ped_wait", "max_green")  # guard keys that end a held green


class DecisionClock:
    """When to ask the policy (D56) and how to turn its action into a stage request."""

    def __init__(
        self, interval_s: float, min_green_s: float, skip_during_preemption: bool = True
    ) -> None:
        if not interval_s > 0:
            raise ValueError(f"interval_s must be > 0, got {interval_s}")
        if min_green_s < 0:
            raise ValueError(f"min_green_s must be >= 0, got {min_green_s}")
        self.interval_s = float(interval_s)
        self.min_green_s = float(min_green_s)
        self.skip_during_preemption = bool(skip_during_preemption)
        self.last_decision_t = -math.inf

    def reset(self) -> None:
        """Forget the last decision (start of an episode or run)."""
        self.last_decision_t = -math.inf

    def is_decision_point(self, obs: Observation) -> bool:
        """True when the policy must be asked at this step (D56)."""
        if obs.in_transition or obs.time_in_stage < self.min_green_s - EPS:
            return False
        if self.skip_during_preemption and obs.extras.get(EXTRA_PREEMPT) is not None:
            return False
        return obs.t - self.last_decision_t >= self.interval_s - EPS

    def request(self, obs: Observation, action: int | None, n_stages: int) -> int:
        """The stage to request for this step.

        At a decision point with an action: 1 -> `(obs.stage + 1) % n_stages`, 0 ->
        `obs.stage`, and the decision time is recorded. Otherwise `obs.stage` (hold): an
        action given outside a decision point is ignored and records nothing."""
        if action is None or not self.is_decision_point(obs):
            return obs.stage
        action = int(action)
        if action not in (STAY, ADVANCE):
            raise ValueError(f"action must be 0 (stay) or 1 (advance), got {action}")
        self.last_decision_t = obs.t
        return (obs.stage + 1) % n_stages if action == ADVANCE else obs.stage


class InterventionSplit:
    """D77 split of the guard interventions (module docstring). Call `reset(guard)` at
    the start of the controlled part, `decided(obs, action, max_green_s)` at each
    decision, `after_step(guard, obs, requested)` after each `sim.step` (with the
    observation before it and the stage requested), `result(guard)` at the end."""

    def __init__(self, interval_s: float) -> None:
        if not interval_s > 0:
            raise ValueError(f"interval_s must be > 0, got {interval_s}")
        self.interval_s = float(interval_s)
        self.reset(None)

    def reset(self, guard) -> None:
        """Start counting from the guard's current counters (None: from zero)."""
        counts = getattr(guard, "interventions", None) or {}
        self._seen = {k: int(v) for k, v in counts.items()}
        self.cut = self.cut_foreseeable = 0
        self._stay: tuple[int, bool] | None = None  # (stage, foreseeable) of a "stay"

    def foreseeable(self, obs: Observation, max_green_s: Sequence[float] | None) -> bool:
        """At a decision: would a stay meet the shield or max green before the next one?"""
        slack = obs.extras.get(EXTRA_PED_SLACK) or ()
        if min(slack, default=math.inf) < self.interval_s - EPS:
            return True
        if max_green_s is not None:
            return obs.time_in_stage + self.interval_s >= float(max_green_s[obs.stage]) - EPS
        return False

    def decided(self, obs: Observation, action: int, max_green_s: Sequence[float] | None) -> None:
        """Record a decision (0 stay, 1 advance) taken on `obs`."""
        if int(action) == STAY:
            self._stay = (obs.stage, self.foreseeable(obs, max_green_s))
        else:
            self._stay = None

    def after_step(self, guard, obs: Observation, requested: int) -> None:
        """Classify the interventions the guard counted on the step just simulated."""
        if obs.in_transition or (self._stay is not None and self._stay[0] != obs.stage):
            self._stay = None  # a decision only speaks for its own green
        counts: Mapping[str, int] = guard.interventions
        held = int(requested) == obs.stage
        for key, value in counts.items():
            new = int(value) - self._seen.get(key, 0)
            if new <= 0:
                continue
            self._seen[key] = int(value)
            if held and key in CUT_KEYS:
                self.cut += new
                if self._stay is not None and self._stay[1]:
                    self.cut_foreseeable += new

    def result(self, guard) -> dict[str, int]:
        """{cut, replaced, cut_foreseeable}; cut + replaced = the guard's total."""
        total = sum(int(v) for v in guard.interventions.values())
        return {
            "cut": self.cut,
            "replaced": total - self.cut,
            "cut_foreseeable": self.cut_foreseeable,
        }

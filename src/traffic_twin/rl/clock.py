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
"""

from __future__ import annotations

import math

from traffic_twin.control.base import EXTRA_PREEMPT, Observation

EPS = 1e-9
STAY, ADVANCE = 0, 1  # the two actions (D55)


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

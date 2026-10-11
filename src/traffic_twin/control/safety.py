"""Safety guard between any controller and the signal plan (spec 8.3).

Every stage request goes through `SafetyGuard.apply` inside `SimSession.step`, so no
controller can bypass it. Part 3 rules, checked in this order:

1. min green: no switch before the green has lasted `safety.min_green_s`;
2. anti-starvation: once a stage has been red for `safety.max_red_s`, head for it
   (after min green);
3. stage order: a switch goes to a link-safe stage, which with `safety.stage_order:
   strict` (the default) is the next stage in cycle order;
4. max green: a green ends after `safety.max_green_s`.

`SignalPlan` still runs every clearance phase. A rule that fires counts one
intervention per green, however many steps it acts on. The fixed and actuated
controllers must cause none (D13). Part 4 adds pedestrian forcing and emergency
pre-emption.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path

from traffic_twin.config import ConfigError, get
from traffic_twin.sim.signal import SignalState, Stage

INTERVENTION_KEYS = ("min_green", "order", "max_green", "starvation")
STAGE_ORDERS = ("strict", "free")
GREEN, YELLOW = "Gg", "y"  # SUMO link states
_EPS = 1e-9


# -- transitions -----------------------------------------------------------------------
def _end_state(stage: Stage, phase_states: Sequence[str]) -> str:
    """Link states at the end of a stage's clearance (its green if it has none)."""
    phases = stage.clearance_phases or (stage.green_phase,)
    return phase_states[phases[-1]]


def unsafe_links(before: str, after: str) -> list[int]:
    """Links whose change from `before` to `after` needs a yellow first: green to
    anything but green, or yellow to green."""
    if len(before) != len(after):
        raise ConfigError(f"phase states of different lengths: {before!r} vs {after!r}")
    return [
        i
        for i, (x, y) in enumerate(zip(before, after, strict=True))
        if (x in GREEN and y not in GREEN) or (x in YELLOW and y in GREEN)
    ]


def safe_transitions(stages: Sequence[Stage], phase_states: Sequence[str]) -> list[frozenset[int]]:
    """allowed[a] = stages b reachable from a through a's clearance phases without any
    link going from green to red, or to another green, without yellow.

    That is: every link still green at the end of a's clearance is green in b's green
    phase, and no link yellow there turns green. A stage is never in its own set."""
    allowed = []
    for a, sa in enumerate(stages):
        end = _end_state(sa, phase_states)
        allowed.append(
            frozenset(
                b
                for b, sb in enumerate(stages)
                if b != a and not unsafe_links(end, phase_states[sb.green_phase])
            )
        )
    return allowed


# -- config helpers --------------------------------------------------------------------
def stage_values(value: object, names: Sequence[str], key: str) -> list[float] | None:
    """Per-stage values from a config entry: None, one number for every stage, or a
    mapping stage name -> number (every stage needed)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{key} must be a number or a mapping by stage name")
    if isinstance(value, int | float):
        return [float(value)] * len(names)
    if isinstance(value, Mapping):
        missing = [n for n in names if n not in value]
        if missing:
            raise ConfigError(f"{key} has no value for stages {missing}")
        return [float(value[n]) for n in names]
    raise ConfigError(f"{key} must be a number or a mapping by stage name")


def program_durations(net_file: str | Path, tls_id: str) -> list[float]:
    """Phase durations of the first program of `tls_id` in a .net.xml."""
    try:
        root = ET.parse(net_file).getroot()
    except (OSError, ET.ParseError) as e:
        msg = f"cannot read the signal program of '{tls_id}' in {net_file}: {e}"
        raise ConfigError(msg) from None
    for tl in root.iter("tlLogic"):
        if tl.get("id") == tls_id:
            return [float(p.get("duration")) for p in tl.iter("phase")]
    raise ConfigError(f"{net_file} has no tlLogic '{tls_id}'")


def _plan_greens(cfg: dict, names: Sequence[str]) -> dict[str, list[float]]:
    """Green times of the configured fixed plan and actuated maxima, by config key."""
    out = {}
    for key in ("fixed.green_s", "actuated.max_green_s"):
        greens = stage_values(get(cfg, key, None), names, key)
        if greens is not None:
            out[key] = greens
    return out


# -- guard -----------------------------------------------------------------------------
class SafetyGuard:
    """Checks each stage request against the Part 3 rules and substitutes a safe one."""

    def __init__(
        self,
        n_stages: int,
        min_green_s: float,
        max_green_s: Sequence[float] | None,
        max_red_s: float | None,
        allowed: Sequence[frozenset[int]],
        strict_order: bool = True,
    ):
        if n_stages < 1 or len(allowed) != n_stages:
            raise ConfigError(f"need one allowed set per stage ({len(allowed)} for {n_stages})")
        if max_green_s is not None and len(max_green_s) != n_stages:
            raise ConfigError(f"need one max green per stage ({len(max_green_s)} for {n_stages})")
        if min_green_s < 0:
            raise ConfigError(f"min green must be >= 0, got {min_green_s}")
        if max_green_s is not None and any(g < min_green_s for g in max_green_s):
            raise ConfigError(f"max greens {list(max_green_s)} below the min green {min_green_s}")
        if max_red_s is not None and max_red_s <= 0:
            raise ConfigError(f"max red must be > 0, got {max_red_s}")
        for s in range(n_stages if n_stages > 1 else 0):
            nxt = (s + 1) % n_stages
            if nxt not in allowed[s]:
                raise ConfigError(f"stage {s} -> {nxt} (cycle order) is not link-safe")
        self.n_stages = n_stages
        self.min_green_s = float(min_green_s)
        self.max_green_s = None if max_green_s is None else [float(g) for g in max_green_s]
        self.max_red_s = None if max_red_s is None else float(max_red_s)
        self.allowed = [frozenset(a) for a in allowed]
        self.strict_order = strict_order
        self.reset()

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        stages: Sequence[Stage],
        phase_states: Sequence[str],
        durations: Sequence[float] | None = None,
    ) -> SafetyGuard:
        """Guard for a scenario (`safety.*`). Grid: no caps, strict order.

        Checks that the cycle order is link-safe, that `safety.max_red_s` exceeds the
        longest legal cycle (fixed greens or actuated max greens, plus every clearance)
        and that those greens fit under `safety.max_green_s`, so that the configured
        plans never trigger the guard. `durations` are the program's phase durations
        (read from `network.net_file` when not given and `max_red_s` is set)."""
        safety = cfg.get("safety") or {}
        names = [s.name for s in stages]
        order = safety.get("stage_order", "strict")
        if order not in STAGE_ORDERS:
            raise ConfigError(f"safety.stage_order must be one of {STAGE_ORDERS}, got {order!r}")
        allowed = safe_transitions(stages, phase_states)
        n = len(stages)
        for i, st in enumerate(stages if n > 1 else ()):
            nxt = stages[(i + 1) % n]
            if (i + 1) % n not in allowed[i]:
                bad = unsafe_links(_end_state(st, phase_states), phase_states[nxt.green_phase])
                raise ConfigError(
                    f"network.stages: {st.name} -> {nxt.name} is not link-safe (links {bad} "
                    "change without yellow after the clearance)"
                )
        max_green = stage_values(safety.get("max_green_s"), names, "safety.max_green_s")
        max_red = safety.get("max_red_s")
        plans = _plan_greens(cfg, names)
        if max_green is not None:
            for key, greens in plans.items():
                over = [n for n, g, cap in zip(names, greens, max_green, strict=True) if g > cap]
                if over:
                    raise ConfigError(f"{key} of stages {over} exceed safety.max_green_s")
        if max_red is not None and plans:
            if durations is None:
                durations = program_durations(
                    get(cfg, "network.net_file"), get(cfg, "network.tls_id")
                )
            clearance = sum(durations[p] for s in stages for p in s.clearance_phases)
            key, greens = max(plans.items(), key=lambda kv: sum(kv[1]))
            cycle = sum(greens) + clearance
            if max_red <= cycle:
                raise ConfigError(
                    f"safety.max_red_s ({max_red}) must exceed the longest legal cycle, "
                    f"{cycle:g} s ({key} + {clearance:g} s of clearance)"
                )
        min_green = float(safety.get("min_green_s", 0.0))
        return cls(n, min_green, max_green, max_red, allowed, order == "strict")

    def reset(self) -> None:
        """Forget the interventions of a previous run."""
        self.interventions: Counter[str] = Counter(dict.fromkeys(INTERVENTION_KEYS, 0))
        self._green = 0  # number of the current green (an intervention counts once per green)
        self._in_green = False
        self._stage, self._tis = -1, 0.0
        self._fired: dict[str, int] = {}

    def apply(self, proposed: int, state: SignalState) -> int:
        """The stage to request instead of `proposed`, given the plan state."""
        if not 0 <= proposed < self.n_stages:
            raise ValueError(f"unknown stage {proposed}")
        self._track(state)
        s, tis = state.stage, state.time_in_stage
        if state.in_transition:  # SignalPlan ignores requests during clearance
            return s
        if proposed != s and tis < self.min_green_s - _EPS:
            return self._intervene("min_green", s)
        starving = self._starving(state)
        if starving is not None and tis >= self.min_green_s - _EPS:
            target = self._next_hop(s, starving)
            return proposed if proposed == target else self._intervene("starvation", target)
        if proposed != s:
            if proposed in self._legal(s):
                return proposed
            return self._intervene("order", self._succ(s))
        if self.max_green_s is not None and tis >= self.max_green_s[s] - _EPS:
            return self._intervene("max_green", self._succ(s))
        return s

    # -- internals -------------------------------------------------------------------
    def _track(self, state: SignalState) -> None:
        """Number the greens: a new one starts after a clearance, on a change of stage
        or when the green time restarts."""
        if state.in_transition:
            self._in_green = False
        elif not self._in_green or state.stage != self._stage or state.time_in_stage < self._tis:
            self._green += 1
            self._in_green = True
        self._stage, self._tis = state.stage, state.time_in_stage

    def _intervene(self, rule: str, stage: int) -> int:
        if self._fired.get(rule) != self._green:
            self._fired[rule] = self._green
            self.interventions[rule] += 1
        return stage

    def _succ(self, s: int) -> int:
        return (s + 1) % self.n_stages

    def _legal(self, s: int) -> frozenset[int]:
        return frozenset({self._succ(s)}) if self.strict_order else self.allowed[s]

    def _starving(self, state: SignalState) -> int | None:
        """The stage red the longest among those red for max_red_s or more (ties: the
        first one in cycle order after the current stage)."""
        if self.max_red_s is None:
            return None
        s, n = state.stage, self.n_stages
        late = [k for k in range(n) if k != s and state.red_s[k] >= self.max_red_s - _EPS]
        if not late:
            return None
        return max(late, key=lambda k: (state.red_s[k], -((k - s) % n)))

    def _next_hop(self, s: int, target: int) -> int:
        """First stage on a shortest legal path from s to target (breadth first, in
        cycle order). Always reachable: the successor is always legal."""
        first = {s: s}  # stage -> first hop of the path that reached it
        todo: deque[int] = deque([s])
        while todo:
            k = todo.popleft()
            for j in sorted(self._legal(k), key=lambda j: (j - k) % self.n_stages):
                if j not in first:
                    first[j] = j if k == s else first[k]
                    if j == target:
                        return first[j]
                    todo.append(j)
        return self._succ(s)

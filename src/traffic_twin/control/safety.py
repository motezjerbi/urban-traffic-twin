"""Safety guard between any controller and the signal plan (spec 8.3, EF-11, ENF-11).

Every stage request goes through `SafetyGuard.apply` inside `SimSession.step`, so no
controller can bypass it. Rule order (D20, binding):

- R0 unknown stage (or preemption target): ValueError;
- R1 in a transition: the current stage. `SignalPlan` runs every clearance phase in
  full, so yellow, all-red and pedestrian clearance are never cut;
- R2 min green: no switch before the green has lasted `safety.min_green_s`. Hard, also
  for preemption (counted only when no preemption is active);
- R3 emergency preemption, when `SignalState.preempt_target` is set: the first hop on
  the cheapest link-safe path to the target (`control.preemption.next_hop`, even under
  strict order), held once there. R4 is suspended; every hop is checked by R5 and a hop
  the shield refuses is replaced by the cycle successor (`ped_yield`);
- R4 the Part 3 rules: anti-starvation (once a stage has been red for
  `safety.max_red_s`, head for it), stage order (a switch goes to a link-safe stage,
  the cycle successor under `safety.stage_order: strict`), max green
  (`safety.max_green_s`);
- R5 pedestrian shield (`control.pedestrian`, D21), hard and last: a stay or a
  non-successor jump that would break w_c + T_c^fb <= theta becomes the successor.

A rule that fires counts one intervention per green however many steps it acts on, and
logs one `Event(source="guard")`. The fixed and actuated controllers cause none, on any
key (D13, D22). Preemption activity is counted apart (`PREEMPTION_KEYS`): seconds held
on the target, non-successor jumps and shield cuts.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path

from traffic_twin.config import ConfigError, get
from traffic_twin.control.base import Event
from traffic_twin.control.pedestrian import (
    PedestrianShield,
    crossing_links,
    cycle_red,
    forcing_mode,
    ped_threshold,
)
from traffic_twin.control.preemption import next_hop
from traffic_twin.sim.signal import SignalState, Stage

INTERVENTION_KEYS = ("min_green", "order", "max_green", "starvation", "ped_wait")
PREEMPTION_KEYS = ("hold_s", "jumps", "ped_yield")
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


def rules_greens(cfg: dict, names: Sequence[str]) -> list[float] | None:
    """Longest greens of the rules plan: `rules.max_green_s`, plus
    `rules.tsp.extension_max_s` on `rules.tsp.extend_stages` when TSP is enabled. None
    without `rules.max_green_s`."""
    greens = stage_values(get(cfg, "rules.max_green_s", None), names, "rules.max_green_s")
    if greens is None or not get(cfg, "rules.tsp.enabled", False):
        return greens
    ext = float(get(cfg, "rules.tsp.extension_max_s", 0.0) or 0.0)
    if ext < 0:
        raise ConfigError(f"rules.tsp.extension_max_s must be >= 0, got {ext:g}")
    extended = get(cfg, "rules.tsp.extend_stages", None) or []
    unknown = [s for s in extended if s not in names]
    if unknown:
        raise ConfigError(f"rules.tsp.extend_stages: unknown stage(s) {unknown}")
    return [g + ext if n in extended else g for n, g in zip(names, greens, strict=True)]


def _plan_greens(cfg: dict, names: Sequence[str]) -> dict[str, list[float]]:
    """Longest green of each stage under each configured plan, by config key: the fixed
    greens, the actuated maxima and the rules maxima (+ TSP extension)."""
    out = {}
    for key in ("fixed.green_s", "actuated.max_green_s"):
        greens = stage_values(get(cfg, key, None), names, key)
        if greens is not None:
            out[key] = greens
    rules = rules_greens(cfg, names)
    if rules is not None:
        tsp = get(cfg, "rules.tsp.enabled", False)
        out["rules.max_green_s" + (" + rules.tsp.extension_max_s" if tsp else "")] = rules
    return out


# -- guard -----------------------------------------------------------------------------
class SafetyGuard:
    """Checks each stage request against the D20 rules and substitutes a safe one."""

    def __init__(
        self,
        n_stages: int,
        min_green_s: float,
        max_green_s: Sequence[float] | None,
        max_red_s: float | None,
        allowed: Sequence[frozenset[int]],
        strict_order: bool = True,
        *,
        shield: PedestrianShield | None = None,
        clearance_s: Sequence[float] | None = None,
        dt: float = 1.0,
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
        if clearance_s is not None and len(clearance_s) != n_stages:
            raise ConfigError(f"need one clearance per stage ({len(clearance_s)} for {n_stages})")
        if dt <= 0:
            raise ConfigError(f"step length must be > 0, got {dt}")
        if shield is not None and len(shield.stages) != n_stages:
            raise ConfigError(f"shield built for {len(shield.stages)} stages, not {n_stages}")
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
        self.shield = shield
        self.clearance_s = None if clearance_s is None else [float(c) for c in clearance_s]
        self.dt = float(dt)
        self.reset()

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        stages: Sequence[Stage],
        phase_states: Sequence[str],
        durations: Sequence[float] | None = None,
    ) -> SafetyGuard:
        """Guard for a scenario (`safety.*`). Grid: no caps, strict order, no shield.

        Checks that the cycle order is link-safe; that every plan green (fixed greens,
        actuated maxima, rules maxima plus the TSP extension) fits under
        `safety.max_green_s`; that `safety.max_red_s` exceeds the longest legal cycle of
        the largest plan, so that the configured plans never trigger the guard; that
        without the pedestrian shield no plan's worst-case crossing red exceeds the
        pedestrian bound; and refuses `preemption.respects_ped_max_wait: false`.
        `durations` are the program's phase durations (read from `network.net_file`
        when needed and not given)."""
        safety = cfg.get("safety") or {}
        names = [s.name for s in stages]
        order = safety.get("stage_order", "strict")
        if order not in STAGE_ORDERS:
            raise ConfigError(f"safety.stage_order must be one of {STAGE_ORDERS}, got {order!r}")
        if get(cfg, "preemption.respects_ped_max_wait", True) is False:
            raise ConfigError(
                "preemption.respects_ped_max_wait: false is refused: A-04 requires every "
                "pedestrian wait <= safety.ped_max_wait_s, so preemption must yield to the "
                "pedestrian shield (D20)"
            )
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

        crossings = crossing_links(cfg)
        theta = ped_threshold(cfg)
        shield_on = forcing_mode(cfg) != "off" and bool(crossings)
        preempt = bool(get(cfg, "preemption.enabled", False))
        check_reds = bool(crossings) and theta is not None and bool(plans)
        need_durations = (max_red is not None and bool(plans)) or shield_on or preempt or check_reds
        if durations is None and need_durations:
            durations = program_durations(get(cfg, "network.net_file"), get(cfg, "network.tls_id"))

        if max_red is not None and plans:
            clearance = sum(durations[p] for s in stages for p in s.clearance_phases)
            key, greens = max(plans.items(), key=lambda kv: sum(kv[1]))
            cycle = sum(greens) + clearance
            if max_red <= cycle:
                raise ConfigError(
                    f"safety.max_red_s ({max_red}) must exceed the longest legal cycle, "
                    f"{cycle:g} s ({key} + {clearance:g} s of clearance)"
                )
        shield = None
        if durations is not None:
            shield = PedestrianShield.from_config(cfg, stages, durations, phase_states)
        if shield is None and crossings and theta is not None:
            for key, greens in plans.items():
                for name, link in crossings.items():
                    red = cycle_red(stages, durations, phase_states, greens, link)
                    if red > theta + _EPS:
                        raise ConfigError(
                            f"{key}: crossing {name} worst-case red {red:g} s > {theta:g} s "
                            "and the pedestrian shield is off: set safety.ped_forcing to "
                            "'detected' or 'red', or shorten the greens"
                        )
        clearance_s = None
        if durations is not None:
            clearance_s = [sum(durations[p] for p in s.clearance_phases) for s in stages]
        min_green = float(safety.get("min_green_s", 0.0))
        return cls(
            n,
            min_green,
            max_green,
            max_red,
            allowed,
            order == "strict",
            shield=shield,
            clearance_s=clearance_s,
            dt=float(cfg.get("step_length_s", 1.0)),
        )

    def reset(self) -> None:
        """Forget the interventions, preemption counts and events of a previous run."""
        self.interventions: Counter[str] = Counter(dict.fromkeys(INTERVENTION_KEYS, 0))
        self.preemption: Counter[str] = Counter(dict.fromkeys(PREEMPTION_KEYS, 0))
        self.preemption["hold_s"] = 0.0
        self.events: list[Event] = []
        self._green = 0  # number of the current green (an intervention counts once per green)
        self._in_green = False
        self._stage, self._tis = -1, 0.0
        self._fired: dict[str, int] = {}
        self._state: SignalState | None = None

    def apply(self, proposed: int, state: SignalState) -> int:
        """The stage to request instead of `proposed`, given the plan state (D20)."""
        if not 0 <= proposed < self.n_stages:  # R0
            raise ValueError(f"unknown stage {proposed}")
        target = state.preempt_target
        if target is not None and not 0 <= target < self.n_stages:
            raise ValueError(f"unknown preemption target {target}")
        self._track(state)
        self._state = state
        s, tis = state.stage, state.time_in_stage
        if state.in_transition:  # R1: SignalPlan ignores requests during clearance
            return s
        if target is not None:
            if tis < self.min_green_s - _EPS:  # R2, hard; not an intervention
                return s
            return self._preempt(target, state)  # R3 (R4 suspended, R5 inside)
        if proposed != s and tis < self.min_green_s - _EPS:  # R2
            return self._intervene("min_green", s, f"proposed={proposed} green={tis:g}s")
        return self._protect(self._part3(proposed, state), state)  # R4, then R5

    # -- rules -----------------------------------------------------------------------
    def _part3(self, proposed: int, state: SignalState) -> int:
        """R4: starvation, then order, then max green (Part 3, unchanged)."""
        s, tis = state.stage, state.time_in_stage
        starving = self._starving(state)
        if starving is not None and tis >= self.min_green_s - _EPS:
            target = self._next_hop(s, starving)
            if proposed == target:
                return proposed
            detail = f"starving={starving} red={state.red_s[starving]:g}s"
            return self._intervene("starvation", target, detail)
        if proposed != s:
            if proposed in self._legal(s):
                return proposed
            return self._intervene("order", self._succ(s), f"proposed={proposed}")
        if self.max_green_s is not None and tis >= self.max_green_s[s] - _EPS:
            return self._intervene("max_green", self._succ(s), f"green={tis:g}s")
        return s

    def _protect(self, r: int, state: SignalState) -> int:
        """R5: the pedestrian shield on the R4 result."""
        shield, s = self.shield, state.stage
        if shield is None or state.time_in_stage < self.min_green_s - _EPS:
            return r
        if r == s:
            excess = shield.stay_excess(state)
        elif r != self._succ(s):
            excess = shield.jump_excess(state, r)
        else:
            return r
        if all(x <= _EPS for x in excess):
            return r
        name, over = shield.worst(excess)
        return self._intervene(
            "ped_wait", self._succ(s), f"crossing={name} over={over:g}s want={r}"
        )

    def _preempt(self, target: int, state: SignalState) -> int:
        """R3: next hop to the target, checked by the shield; the successor otherwise."""
        s, succ = state.stage, self._succ(state.stage)
        h = self._preempt_hop(s, target)
        shield = self.shield
        if h == s:
            excess = () if shield is None else shield.stay_excess(state)
        elif h == succ:
            excess = ()
        else:
            excess = () if shield is None else shield.jump_excess(state, h)
        if all(x <= _EPS for x in excess):
            if h == s:
                self.preemption["hold_s"] += self.dt
            elif h != succ:
                self.preemption["jumps"] += 1
            return h
        name, over = shield.worst(excess)
        if self._once("ped_yield"):
            self.preemption["ped_yield"] += 1
            detail = f"target={target} hop={h} crossing={name} over={over:g}s"
            self._log("ped_yield", succ, detail)
        return succ

    def _preempt_hop(self, s: int, target: int) -> int:
        """`control.preemption.next_hop` over the link-safe graph, also under strict
        order (D20): cost = clearance + min green per hop."""
        clearance = self.clearance_s or [0.0] * self.n_stages
        return next_hop(s, target, self.allowed, clearance, self.min_green_s)

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

    def _once(self, rule: str) -> bool:
        """True the first time `rule` fires in the current green."""
        if self._fired.get(rule) == self._green:
            return False
        self._fired[rule] = self._green
        return True

    def _log(self, rule: str, stage_to: int, detail: str) -> None:
        state = self._state
        t, stage = (0.0, stage_to) if state is None else (state.t, state.stage)
        self.events.append(Event(t, "guard", rule, stage, stage_to, detail))

    def _intervene(self, rule: str, stage: int, detail: str = "") -> int:
        if self._once(rule):
            self.interventions[rule] += 1
            self._log(rule, stage, detail)
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

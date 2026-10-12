"""Pedestrian shield of the Safety Guard (Part 4, D21; spec EF-09, EF-11, A-04).

Hard bound on every pedestrian's wait at a signalised crossing. With
theta = `safety.ped_max_wait_s` - `safety.ped_red_margin_s` (88 s at Pitié), the guard
keeps, for every crossing c, the invariant

    w_c + T_c^fb <= theta

where w_c is the longest current wait at c (`SignalState.ped_wait`) and T_c^fb the time
until c shows green if the current stage ends its min green now, then every stage runs
at min green in cycle order with its full clearance (the fallback policy, which the
guard can always follow because the cycle successor is always legal):

- a stay is allowed iff w_c + T_c^fb + dt <= theta for every c (after one more step of
  green the fallback still serves everybody in time);
- a jump to a non-successor stage j iff w_c + T_via(c, s, j) <= theta for every c, with
  T_via the clearance of s then T^fb from the start of j's green;
- otherwise the guard requests the successor, which keeps w_c + T_c^fb constant.

The invariant holds at the start (w = 0) and for every new pedestrian as long as each
crossing's worst-case red at min greens is <= theta (`from_config` checks it), so no
pedestrian waits more than theta. Crossing c is green when its TLS link shows G or g,
also inside a clearance phase (a crossing that stays green into the yellow is served).

`safety.ped_forcing`:
- `detected`: w_c = the longest current curb wait of a person whose next edge is the
  crossing, sampled after the previous step (ideal detection, ASSUMED). The curb wait is
  the session's per-person curb clock (`SimSession._sample_ped_waits`): it keeps running
  through the small creeps of a pedestrian standing at a red curb, which reset SUMO's
  `person.getWaitingTime`, and is never below `getWaitingTime`; the bound therefore
  holds on the real curb wait (KPI `ped_curb_wait_max`, A-04);
- `red`: w_c = seconds since the crossing link last showed G (no detector: the shield
  then reproduces the static bound);
- `off` or absent: no shield (the grid). YAML reads a bare `off` as false: same thing.

The fallback timing uses `safety.min_green_s` for every stage and the program's
clearance durations, so it holds for any controller behind the guard. Crossings are
indexed in `network.crossings` key order, like `SignalState.ped_wait`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from traffic_twin.config import ConfigError, get
from traffic_twin.sim.signal import SignalState, Stage

FORCING_MODES = ("detected", "red", "off")
EPS = 1e-9
GREEN = "Gg"  # SUMO link states that let pedestrians cross


def forcing_mode(cfg: dict) -> str:
    """`safety.ped_forcing` normalised to one of FORCING_MODES ('off' when absent)."""
    mode = get(cfg, "safety.ped_forcing", None)
    if mode is None or mode is False:  # absent, or YAML's bare `off`
        return "off"
    if mode not in FORCING_MODES:
        raise ConfigError(f"safety.ped_forcing must be one of {FORCING_MODES}, got {mode!r}")
    return mode


def ped_threshold(cfg: dict) -> float | None:
    """theta = safety.ped_max_wait_s - safety.ped_red_margin_s, or None without a max wait."""
    max_wait = get(cfg, "safety.ped_max_wait_s", None)
    if max_wait is None:
        return None
    return float(max_wait) - float(get(cfg, "safety.ped_red_margin_s", 0.0) or 0.0)


def crossing_links(cfg: dict) -> dict[str, int]:
    """`network.crossings` as {name: TLS link}, in key order ({} without crossings)."""
    crossings = get(cfg, "network.crossings", None) or {}
    try:
        return {name: int(c["link"]) for name, c in crossings.items()}
    except (KeyError, TypeError, ValueError):
        raise ConfigError("network.crossings: every crossing needs an integer `link`") from None


def cycle_red(
    stages: Sequence[Stage],
    durations: Sequence[float],
    phase_states: Sequence[str],
    greens: Sequence[float],
    link: int,
) -> float:
    """Longest red (s) of `link` over the stage cycle actually shown by SignalPlan: each
    stage's green for `greens[k]`, then its clearance phases at their programmed
    durations, in cycle order. Red is any state but G/g; inf if the link is never green."""
    if len(greens) != len(stages):
        raise ValueError(f"{len(greens)} greens for {len(stages)} stages")
    cycle: list[tuple[float, bool]] = []
    for stage, g in zip(stages, greens, strict=True):
        cycle.append((float(g), phase_states[stage.green_phase][link] in GREEN))
        cycle += [
            (float(durations[p]), phase_states[p][link] in GREEN) for p in stage.clearance_phases
        ]
    if not any(green for _, green in cycle):
        return math.inf
    best = run = 0.0
    for d, green in cycle * 2:  # wrap around the end of the cycle
        run = 0.0 if green else run + d
        best = max(best, run)
    return best


class PedestrianShield:
    """Checks a stay or a jump against the pedestrian max-wait invariant (D21).

    `crossing_links[c]` is the TLS link of crossing c; `names` label the crossings in
    events and errors (default: `link<i>`). `mode` only tells the simulation which wait
    to feed (`detected` or `red`): the arithmetic is the same."""

    def __init__(
        self,
        stages: Sequence[Stage],
        durations: Sequence[float],
        phase_states: Sequence[str],
        crossing_links: Sequence[int],
        min_green_s: float,
        threshold_s: float,
        dt: float = 1.0,
        *,
        mode: str = "detected",
        names: Sequence[str] | None = None,
    ):
        if mode not in FORCING_MODES or mode == "off":
            raise ConfigError(f"shield mode must be 'detected' or 'red', got {mode!r}")
        if not stages:
            raise ConfigError("the pedestrian shield needs at least one stage")
        if min_green_s < 0 or dt <= 0:
            raise ConfigError(f"need min green >= 0 and dt > 0, got {min_green_s}, {dt}")
        n_phases = len(phase_states)
        for st in stages:
            bad = [p for p in (st.green_phase, *st.clearance_phases) if not 0 <= p < n_phases]
            if bad:
                raise ConfigError(f"stage {st.name}: phase(s) {bad} not in the program")
        if len(durations) != n_phases:
            raise ConfigError(f"{len(durations)} durations for {n_phases} phases")
        n_links = len(phase_states[0]) if phase_states else 0
        bad_links = [i for i in crossing_links if not 0 <= i < n_links]
        if bad_links:
            raise ConfigError(f"crossing links {bad_links} outside the {n_links}-link program")
        names = [f"link{i}" for i in crossing_links] if names is None else list(names)
        if len(names) != len(crossing_links):
            raise ConfigError(f"{len(names)} names for {len(crossing_links)} crossings")
        self.stages = list(stages)
        self.durations = [float(d) for d in durations]
        self.phase_states = list(phase_states)
        self.crossing_links = tuple(int(i) for i in crossing_links)
        self.names = tuple(names)
        self.min_green_s = float(min_green_s)
        self.threshold_s = float(threshold_s)
        self.dt = float(dt)
        self.mode = mode
        n = len(self.stages)
        # per crossing and stage: green in the stage's green phase; clearance as
        # [(duration, crossing green)] in order
        self._green = [
            [self.phase_states[st.green_phase][i] in GREEN for st in self.stages]
            for i in self.crossing_links
        ]
        self._clear = [
            [
                [(self.durations[p], self.phase_states[p][i] in GREEN) for p in st.clearance_phases]
                for st in self.stages
            ]
            for i in self.crossing_links
        ]
        self._n = n

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        stages: Sequence[Stage],
        durations: Sequence[float],
        phase_states: Sequence[str],
    ) -> PedestrianShield | None:
        """Shield for a scenario: None if `safety.ped_forcing` is absent or 'off', or the
        scenario has no `network.crossings`. ConfigError for an unknown mode, a missing
        `safety.ped_max_wait_s`, or a crossing whose worst-case red with every stage at
        `safety.min_green_s` exceeds theta (the invariant could not even start)."""
        mode = forcing_mode(cfg)
        links = crossing_links(cfg)
        if mode == "off" or not links:
            return None
        theta = ped_threshold(cfg)
        if theta is None:
            raise ConfigError("safety.ped_forcing needs safety.ped_max_wait_s")
        if theta <= 0:
            raise ConfigError(f"pedestrian threshold must be > 0, got {theta:g} s")
        shield = cls(
            stages,
            durations,
            phase_states,
            list(links.values()),
            float(get(cfg, "safety.min_green_s", 0.0) or 0.0),
            theta,
            float(cfg.get("step_length_s", 1.0)),
            mode=mode,
            names=list(links),
        )
        for c, name in enumerate(shield.names):
            red = shield.worst_red(c)
            if red > theta + EPS:
                raise ConfigError(
                    f"crossing {name}: worst-case red {red:g} s with every stage at "
                    f"safety.min_green_s exceeds the pedestrian bound {theta:g} s "
                    "(safety.ped_max_wait_s - safety.ped_red_margin_s): no controller can "
                    "keep it"
                )
        return shield

    # -- timing ----------------------------------------------------------------------
    def worst_red(self, c: int, greens: Sequence[float] | None = None) -> float:
        """Worst-case red of crossing c over the cycle, every stage at its green in
        `greens` (default: the min green)."""
        greens = [self.min_green_s] * self._n if greens is None else greens
        link = self.crossing_links[c]
        return cycle_red(self.stages, self.durations, self.phase_states, greens, link)

    def time_to_green(self, c: int, stage: int, time_in_stage: float) -> float:
        """T_c^fb: seconds until crossing c shows green if `stage` (green for
        `time_in_stage` s) ends its min green, then every stage runs at min green in cycle
        order with full clearance. 0 if c is green in `stage`'s green phase; a green
        inside a clearance phase counts from the start of that phase."""
        green, clear = self._green[c], self._clear[c]
        if green[stage]:
            return 0.0
        t = max(0.0, self.min_green_s - time_in_stage)
        k = stage
        for _ in range(self._n + 1):
            for d, g in clear[k]:
                if g:
                    return t
                t += d
            k = (k + 1) % self._n
            if green[k]:
                return t
            t += self.min_green_s
        return math.inf

    def time_via(self, c: int, stage: int, nxt: int) -> float:
        """Seconds until crossing c shows green if `stage` ends now and `nxt` is the next
        green: `stage`'s clearance, then T^fb(nxt, 0)."""
        t = 0.0
        for d, g in self._clear[c][stage]:
            if g:
                return t
            t += d
        return t + self.time_to_green(c, nxt, 0.0)

    # -- checks ----------------------------------------------------------------------
    def waits(self, state: SignalState) -> tuple[float, ...]:
        """`state.ped_wait`, or zeros when the caller gives none (Part 3 callers)."""
        w = state.ped_wait
        if not w:
            return (0.0,) * len(self.crossing_links)
        if len(w) != len(self.crossing_links):
            raise ValueError(f"{len(w)} pedestrian waits for {len(self.crossing_links)} crossings")
        return tuple(float(x) for x in w)

    def slack(self, state: SignalState) -> tuple[float, ...]:
        """Per crossing: theta - (w_c + T_c^fb) now; inf if T_c^fb == 0 (green now). In
        a transition T is computed as if the green of `state.stage` were still running,
        which overestimates it (conservative)."""
        out = []
        for c, w in enumerate(self.waits(state)):
            t = self.time_to_green(c, state.stage, state.time_in_stage)
            out.append(math.inf if t == 0.0 else self.threshold_s - (w + t))
        return tuple(out)

    def stay_excess(self, state: SignalState) -> tuple[float, ...]:
        """Per crossing: w_c + T_c^fb + dt - theta (a stay is allowed iff all <= 0)."""
        s, tis = state.stage, state.time_in_stage
        return tuple(
            w + self.time_to_green(c, s, tis) + self.dt - self.threshold_s
            for c, w in enumerate(self.waits(state))
        )

    def jump_excess(self, state: SignalState, nxt: int) -> tuple[float, ...]:
        """Per crossing: w_c + T_via(c, s, nxt) - theta (a jump is allowed iff all <= 0)."""
        s = state.stage
        return tuple(
            w + self.time_via(c, s, nxt) - self.threshold_s for c, w in enumerate(self.waits(state))
        )

    def stay_ok(self, state: SignalState) -> bool:
        """True if the current green may last one more step: always in a transition or
        before min green (nobody can leave anyway, and T counts the remaining min green)."""
        if state.in_transition or state.time_in_stage < self.min_green_s - EPS:
            return True
        return all(x <= EPS for x in self.stay_excess(state))

    def jump_ok(self, state: SignalState, nxt: int) -> bool:
        """True if a switch from `state.stage` to `nxt` keeps the invariant at the start
        of nxt's green. The cycle successor (the fallback itself) always does; `nxt` equal
        to the current stage is a stay; in a transition the plan ignores requests."""
        if state.in_transition or nxt == (state.stage + 1) % self._n:
            return True
        if nxt == state.stage:
            return self.stay_ok(state)
        return all(x <= EPS for x in self.jump_excess(state, nxt))

    def worst(self, excess: Sequence[float]) -> tuple[str, float]:
        """(crossing name, excess) of the largest excess, for event details."""
        c = max(range(len(excess)), key=lambda i: excess[i])
        return self.names[c], float(excess[c])

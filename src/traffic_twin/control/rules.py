"""S2 rule-based controller (Part 4, D19, D23; spec annex B `rules`, EF-08, EF-09).

Actuated gap logic (as S1) with max greens relaxed beyond the static pedestrian bound,
which the Safety Guard's pedestrian shield makes safe, plus three explainable rules.
`decide(obs)` applies them in this order (plan WS3, the order tested on seeds 101-103):

1. In transition, or before the stage's min green: hold.
2. **Gap logic / box clearing.** gaps = `extras["gap_s_open"]` when `rules.box.enabled`
   (only the loops whose lane can still discharge into an exit that is not blocked),
   else `extras["gap_s"]`. stay = green < max green and gap < `max_gap_s`. `box_gapout`
   is logged when the box rule turns an extension into a gap-out.
3. **TSP truncation** (paid priority, the configurable penalty cap; off at Pitié): while a
   called bus waits for another stage and none is served by this one, the green ends at
   min(max green, `tsp.truncation.max_green_s`).
4. **TSP extension.** A green about to end (gap-out or max-out) of a stage in
   `tsp.extend_stages` is held while a bus of `tsp.lines` served by it can reach the stop
   line in time: ETA = dist / `approach_speed_mps` + remaining dwell
   (+ `start_penalty_s` if halted or dwelling) <= max green + `extension_max_s` - green.
5. **Pedestrian anticipation.** Leave the green as soon as `min(extras["ped_slack"])` <
   dt, i.e. one more second would make the guard's pedestrian shield force the switch.
6. Hold the stage, or move to the next one in cycle order (never skips).

Every rule logs one `Event(source="controller")` per green when it changes the decision
(`box_gapout`, `tsp_truncate`, `tsp_extend`, `ped_protect`), and `report()` sums them
for run.json `control`. Parameters: `rules.*` (configs/pitie/rules.yaml). With S1's max
greens and every rule off it takes exactly the decisions of `ActuatedController`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from traffic_twin.config import ConfigError, get, stage_names
from traffic_twin.control.base import (
    EXTRA_BUS_CALLS,
    EXTRA_GAP_S,
    EXTRA_GAP_S_OPEN,
    EXTRA_PED_SLACK,
    BusCall,
    Controller,
    Observation,
)
from traffic_twin.control.registry import register
from traffic_twin.control.safety import stage_values

_EPS = 1e-9
HALTED_MPS = 0.1  # a bus slower than this pays the start penalty

RULES_KEYS = {"max_gap_s", "min_green_s", "max_green_s", "pedestrian", "tsp", "box"}
TSP_KEYS = {
    "enabled", "lines", "extend_stages", "extension_max_s", "approach_speed_mps",
    "start_penalty_s", "truncation",
}  # fmt: skip
EVENT_RULES = ("box_gapout", "tsp_truncate", "tsp_extend", "ped_protect")


@dataclass(frozen=True)
class TspParams:
    """Transit signal priority of the rules controller (`rules.tsp`)."""

    enabled: bool = False
    lines: tuple[str, ...] = ()
    extend_stages: frozenset[int] = frozenset()
    extension_max_s: float = 0.0
    approach_speed_mps: float = 7.0
    start_penalty_s: float = 4.0
    truncation: bool = False
    truncation_max_green_s: tuple[float, ...] | None = None  # per stage (penalty cap)

    def eta(self, call: BusCall) -> float:
        """Seconds until the bus can reach the stop line: distance at the approach speed,
        plus the remaining dwell, plus the start penalty when halted or dwelling."""
        eta = call.dist_m / self.approach_speed_mps + call.dwell_left_s
        if call.speed_mps < HALTED_MPS or call.dwell_left_s > 0:
            eta += self.start_penalty_s
        return eta


def _mapping(cfg: dict, key: str, allowed: Iterable[str]) -> dict:
    value = get(cfg, key, None)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ConfigError(f"{key}: unknown keys {unknown} (allowed: {sorted(allowed)})")
    return dict(value)


def _per_stage(cfg: dict, key: str, names: list[str]) -> list[float]:
    values = stage_values(get(cfg, key, None), names, key)
    if values is None:
        raise ConfigError(f"missing config key '{key}'")
    return values


def bus_lines(cfg: dict) -> set[str]:
    """Line names of the scenario's buses (`demand.bus.lines[].line`, or the grid's single
    `demand.bus.line`)."""
    bus = get(cfg, "demand.bus", None) or {}
    out = {str(ln["line"]) for ln in bus.get("lines") or [] if "line" in ln}
    if bus.get("line") is not None:
        out.add(str(bus["line"]))
    return out


def _tsp_from_config(cfg: dict, names: list[str], maxs: list[float], mins: list[float]):
    t = _mapping(cfg, "rules.tsp", TSP_KEYS)
    if not t:
        return TspParams()
    floor_cap = stage_values(get(cfg, "safety.max_green_s", None), names, "safety.max_green_s")
    lines = tuple(str(x) for x in t.get("lines") or [])
    known = bus_lines(cfg)
    unknown = sorted(set(lines) - known)
    if unknown:
        raise ConfigError(
            f"rules.tsp.lines {unknown} not in the demand's bus lines {sorted(known)}"
        )
    ext_names = [str(x) for x in t.get("extend_stages") or []]
    bad = [n for n in ext_names if n not in names]
    if bad:
        raise ConfigError(f"rules.tsp.extend_stages: unknown stages {bad} (stages: {names})")
    ext = float(t.get("extension_max_s", 0.0))
    speed = float(t.get("approach_speed_mps", 7.0))
    penalty = float(t.get("start_penalty_s", 4.0))
    if ext < 0 or penalty < 0 or speed <= 0:
        raise ConfigError(
            "rules.tsp needs extension_max_s >= 0, start_penalty_s >= 0, approach_speed_mps > 0"
        )
    extend = frozenset(names.index(n) for n in ext_names)
    over = [names[k] for k in sorted(extend) if floor_cap and maxs[k] + ext > floor_cap[k]]
    if over:
        raise ConfigError(
            f"rules.max_green_s + tsp.extension_max_s of {over} above safety.max_green_s"
        )
    trunc = _mapping(cfg, "rules.tsp.truncation", {"enabled", "max_green_s"})
    caps = stage_values(trunc.get("max_green_s"), names, "rules.tsp.truncation.max_green_s")
    if trunc.get("enabled") and caps is None:
        raise ConfigError("rules.tsp.truncation.enabled needs truncation.max_green_s")
    if caps is not None:
        high = [n for n, c, m in zip(names, caps, maxs, strict=True) if c > m + _EPS]
        if high:
            raise ConfigError(f"rules.tsp.truncation.max_green_s of {high} above rules.max_green_s")
        low = [n for n, c, m in zip(names, caps, mins, strict=True) if c < m - _EPS]
        if low:
            raise ConfigError(f"rules.tsp.truncation.max_green_s of {low} below rules.min_green_s")
    enabled = bool(t.get("enabled", False))
    if enabled and not lines:
        raise ConfigError("rules.tsp.enabled needs at least one line in rules.tsp.lines")
    return TspParams(
        enabled=enabled,
        lines=lines,
        extend_stages=extend,
        extension_max_s=ext,
        approach_speed_mps=speed,
        start_penalty_s=penalty,
        truncation=bool(trunc.get("enabled", False)),
        truncation_max_green_s=None if caps is None else tuple(caps),
    )


@register("rules")
class RuleBasedController(Controller):
    def __init__(
        self,
        min_green_s: Sequence[float],
        max_green_s: Sequence[float],
        max_gap_s: float = 3.0,
        *,
        tsp: TspParams | None = None,
        box: bool = False,
        anticipate: bool = False,
        dt: float = 1.0,
        stage_names: Sequence[str] | None = None,
    ):
        if not min_green_s or len(min_green_s) != len(max_green_s):
            raise ValueError("need one min and one max green per stage")
        if any(not 0 < lo <= hi for lo, hi in zip(min_green_s, max_green_s, strict=True)):
            raise ValueError("need 0 < min green <= max green for every stage")
        if max_gap_s <= 0 or dt <= 0:
            raise ValueError("max gap and step length must be positive")
        n = len(min_green_s)
        self.min_green_s = [float(g) for g in min_green_s]
        self.max_green_s = [float(g) for g in max_green_s]
        self.max_gap_s = float(max_gap_s)
        self.tsp = tsp or TspParams()
        if any(not 0 <= k < n for k in self.tsp.extend_stages):
            raise ValueError("tsp.extend_stages outside the stages")
        caps = self.tsp.truncation_max_green_s
        if caps is not None and len(caps) != n:
            raise ValueError("need one truncation cap per stage")
        if self.tsp.truncation and caps is None:
            raise ValueError("truncation needs a cap per stage")
        self.box = bool(box)
        self.anticipate = bool(anticipate)
        self.dt = float(dt)
        self.stage_names = list(stage_names) if stage_names else [str(k) for k in range(n)]
        if len(self.stage_names) != n:
            raise ValueError("need one name per stage")
        self.reset(0)

    @classmethod
    def from_config(cls, cfg: dict) -> RuleBasedController:
        """`rules.{max_gap_s, min_green_s, max_green_s, pedestrian, tsp, box}` (greens by
        stage name); the greens must lie within `safety.{min_green_s, max_green_s}`."""
        if get(cfg, "detectors.control", None) is None:
            raise ConfigError("rules needs detectors.control (control loops per stage)")
        _mapping(cfg, "rules", RULES_KEYS)
        names = stage_names(cfg)
        mins = _per_stage(cfg, "rules.min_green_s", names)
        maxs = _per_stage(cfg, "rules.max_green_s", names)
        floor = float(get(cfg, "safety.min_green_s", 0.0))
        low = [n for n, g in zip(names, mins, strict=True) if g < floor]
        if low:
            raise ConfigError(f"rules.min_green_s of {low} below safety.min_green_s ({floor:g})")
        caps = stage_values(get(cfg, "safety.max_green_s", None), names, "safety.max_green_s")
        high = [n for n, g, c in zip(names, maxs, caps or maxs, strict=True) if g > c]
        if high:
            raise ConfigError(f"rules.max_green_s of {high} above safety.max_green_s")
        bad = [n for n, lo, hi in zip(names, mins, maxs, strict=True) if lo > hi]
        if bad:
            raise ConfigError(f"rules min green above max green for {bad}")
        max_gap = float(get(cfg, "rules.max_gap_s", 3.0))
        if max_gap <= 0:
            raise ConfigError("rules.max_gap_s must be positive")
        tsp = _tsp_from_config(cfg, names, maxs, mins)
        box = _mapping(cfg, "rules.box", {"enabled"})
        ped = _mapping(cfg, "rules.pedestrian", {"anticipate"})
        return cls(
            mins,
            maxs,
            max_gap,
            tsp=tsp,
            box=bool(box.get("enabled", False)),
            anticipate=bool(ped.get("anticipate", False)),
            dt=float(cfg.get("step_length_s", 1.0)),
            stage_names=names,
        )

    # -- run state ---------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        """Deterministic: forget the counters, events and green tracking of a run."""
        self.clear_events()
        self._green = 0  # id of the current green (increments at each new green)
        self._last: tuple[int, float] | None = None  # (stage, green time) of the last step
        self._fired: dict[str, int] = {}  # rule -> green id it last fired in
        self._counts: Counter = Counter()  # rule -> greens
        self._tsp_bus: dict[str, str] = {}  # vid -> line of every TSP call seen
        self._ext_s: Counter = Counter()  # line -> extension seconds
        self._ext_greens: Counter = Counter()  # line -> extended greens
        self._ext_fired: set[tuple[int, str]] = set()  # (green id, line)
        self._trunc_s = 0.0

    def _track_green(self, obs: Observation) -> None:
        if obs.in_transition:
            self._last = None  # a transition always ends in a new green
            return
        if (
            self._last is None
            or obs.stage != self._last[0]
            or obs.time_in_stage < self._last[1] - _EPS
        ):
            self._green += 1
        self._last = (obs.stage, obs.time_in_stage)

    def _once(self, obs: Observation, rule: str, to: int, detail: str) -> bool:
        """Log `rule` the first time it fires in the current green; True if logged."""
        if self._fired.get(rule) == self._green:
            return False
        self._fired[rule] = self._green
        self._counts[rule] += 1
        self.log_event(obs.t, rule, obs.stage, to, detail)
        return True

    # -- decision ----------------------------------------------------------------------
    def decide(self, obs: Observation) -> int:
        self._track_green(obs)
        s, tis = obs.stage, obs.time_in_stage
        calls = self._tsp_calls(obs) if self.tsp.enabled else ()
        if obs.in_transition or tis < self.min_green_s[s] - _EPS:
            return s
        nxt = (s + 1) % len(self.min_green_s)
        mx = self.max_green_s[s]

        # 2. gap logic, with box clearing on the open-exit loops
        gaps = self._extra(obs, EXTRA_GAP_S, "detectors.control")
        if self.box:
            open_gaps = self._extra(obs, EXTRA_GAP_S_OPEN, "network.crossings (box clearing)")
            maxed = tis >= mx - _EPS
            stay = not maxed and open_gaps[s] < self.max_gap_s
            box_fired = not maxed and gaps[s] < self.max_gap_s <= open_gaps[s]
        else:
            stay = tis < mx - _EPS and gaps[s] < self.max_gap_s
            box_fired = False

        # 3. paid truncation (penalty cap) while a bus waits for another stage
        trunc = None
        if stay and self.tsp.truncation and calls:
            served_here = any(s in c.stages for c in calls)
            waiting = [c for c in calls if c.stages and s not in c.stages]
            cap = self.tsp.truncation_max_green_s[s]
            if waiting and not served_here and tis >= min(mx, cap) - _EPS:
                stay, trunc = False, waiting[0]

        # 4. TSP extension of a bus-serving stage about to end
        ext = None
        if not stay and trunc is None and s in self.tsp.extend_stages and calls:
            limit = mx + self.tsp.extension_max_s
            if tis < limit - _EPS:
                allow = limit - tis
                ok = [(self.tsp.eta(c), c.vid, c) for c in calls if s in c.stages]
                ok = [x for x in ok if x[0] <= allow + _EPS]
                if ok:
                    stay, ext = True, min(ok)

        # 5. pedestrian anticipation: leave before the shield has to force it
        protect = False
        if stay and self.anticipate:
            slack = self._extra(obs, EXTRA_PED_SLACK, "network.crossings (pedestrian shield)")
            low = min(slack, default=math.inf)
            if low < self.dt - _EPS:
                stay, protect = False, True

        out = s if stay else nxt
        self._log(obs, out, gaps, box_fired, trunc, ext, protect)
        return out

    def _log(
        self,
        obs: Observation,
        out: int,
        gaps: Sequence[float],
        box_fired: bool,
        trunc: BusCall | None,
        ext: tuple[float, str, BusCall] | None,
        protect: bool,
    ) -> None:
        """Events (once per green and rule) and report counters of one decision. A rule
        is logged only when it changed the decision: the box rule when the green ends,
        TSP extension when the bus hold survives the pedestrian anticipation."""
        s, tis = obs.stage, obs.time_in_stage
        name = self.stage_names[s]
        if box_fired and out != s:  # the box rule ended the green (no TSP hold)
            open_gap = obs.extras[EXTRA_GAP_S_OPEN][s]
            self._once(
                obs, "box_gapout", out, f"stage={name}, gap={gaps[s]:g}, gap_open={open_gap:g}"
            )
        if trunc is not None:
            cap = self.tsp.truncation_max_green_s[s]
            if self._once(
                obs, "tsp_truncate", out, f"vid={trunc.vid}, line={trunc.line}, cap={cap:g}"
            ):
                self._trunc_s += max(self.max_green_s[s] - tis, 0.0)
        if ext is not None and not protect:
            eta, vid, call = ext
            line = call.line
            self._ext_s[line] += self.dt
            if (self._green, line) not in self._ext_fired:
                self._ext_fired.add((self._green, line))
                self._ext_greens[line] += 1
            allow = self.max_green_s[s] + self.tsp.extension_max_s - tis
            self._once(
                obs, "tsp_extend", out, f"vid={vid}, line={line}, eta={eta:.1f}, allow={allow:g}"
            )
        if protect:
            low = min(obs.extras[EXTRA_PED_SLACK])
            self._once(obs, "ped_protect", out, f"stage={name}, slack={low:g}")

    def _tsp_calls(self, obs: Observation) -> tuple[BusCall, ...]:
        """The bus calls of the TSP lines (recorded for `report()`)."""
        raw = self._extra(obs, EXTRA_BUS_CALLS, "network.crossings (bus detection)")
        lines = self.tsp.lines
        calls = tuple(c for c in raw if c.line in lines)
        for c in calls:
            self._tsp_bus.setdefault(c.vid, c.line)
        return calls

    @staticmethod
    def _extra(obs: Observation, key: str, needs: str):
        value = obs.extras.get(key)
        if value is None:
            raise ConfigError(f"rules needs {needs}: the observation has no {key}")
        return value

    # -- explainability ----------------------------------------------------------------
    def report(self) -> dict:
        """Rule activity of the run (run.json `control`, plan section 4.6):
        - `ped_protect`, `box_gapouts`: greens ended by the pedestrian anticipation / by
          the box rule;
        - `tsp.calls`: buses of the TSP lines detected; `extended_greens`: greens held
          for a bus; `extension_s`: seconds of those holds (also `by_line`, attributed to
          the bus with the smallest ETA);
        - `tsp.truncated_greens`: greens cut at the penalty cap; `truncation_s`: the green
          they could still have had up to their max green."""
        by_line = {}
        for line in self.tsp.lines:
            by_line[line] = {
                "calls": sum(1 for x in self._tsp_bus.values() if x == line),
                "extended_greens": int(self._ext_greens[line]),
                "extension_s": float(self._ext_s[line]),
            }
        return {
            "ped_protect": int(self._counts["ped_protect"]),
            "box_gapouts": int(self._counts["box_gapout"]),
            "tsp": {
                "calls": len(self._tsp_bus),
                "extended_greens": int(self._counts["tsp_extend"]),
                "extension_s": float(sum(self._ext_s.values())),
                "by_line": by_line,
                "truncated_greens": int(self._counts["tsp_truncate"]),
                "truncation_s": float(self._trunc_s),
            },
        }

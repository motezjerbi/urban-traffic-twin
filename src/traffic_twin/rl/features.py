"""RL observation `rl-obs-1` (Part 6, D57; contract 4.7): a pure function of a
`JunctionSnapshot` (`sim/sampler.py`), Box[0, 1] float32, 74 values at Pitié. The
environment and the `rl` controller both call `observation`, so the trained policy and
the evaluated one see the same numbers (no train/serve skew; `tests/test_rl_parity.py`).

Layout (`feature_names`), every value divided by its `observation` norm of
`configs/rl.yaml` and clipped to [0, 1]:
- per TLS-controlled vehicle lane (TLS order; 7 at Pitié): halting / cap, vehicles / cap,
  max vehicle waiting time / `wait_norm_s`, with cap = length / `veh_length_m` (>= 1);
- pending insertions per general arm N, S, W / `pending_norm` (ASSUMED upstream
  detection: the insertion backlog stands for the queue beyond the modelled approach);
- signal: stage one-hot, in_transition, time_in_stage / `tis_norm_s`, per stage
  red_s / `red_norm_s`;
- per crossing (network.crossings order): waiting count / `ped_count_norm`, curb wait /
  `ped_wait_norm_s`, shield slack / `ped_wait_norm_s` (1 when +inf, i.e. no bound);
- ambulance: present (any emergency call), pending (a call not yet inserted), distance
  one-hot of the nearest inserted call (< 50 m, < 150 m, farther; all 0 when none is
  inserted), arm one-hot N / S / W of the calls, preempt_active. ASSUMED V2I. The block is
  kept for spec 10.2 but is expected to be ~0 at decision points: pending detection
  starts the preemption first and the clock skips decisions during it (D56, D57);
- per bus approach S, NB, WB: present, nearest distance / `bus_dist_norm_m` (1 when no
  bus), delay one-hot of its most delayed bus by `vehicle.getTimeLoss` (< 30 s, 30-90 s,
  >= 90 s; ASSUMED proxy, there is no timetable);
- time of day: 0.5 + 0.5 sin, 0.5 + 0.5 cos of the absolute SUMO clock.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from traffic_twin.sim.sampler import JunctionSnapshot
    from traffic_twin.sim.session import SimSession

OBS_VERSION = "rl-obs-1"
VEH_LENGTH_M = 7.5  # observation.veh_length_m default (lane capacity = length / 7.5 m)
DAY_S = 86400.0
AMB_DIST_NAMES = ("near", "mid", "far")  # < bins[0], < bins[1], farther
BUS_DELAY_NAMES = ("low", "mid", "high")  # < bins[0], < bins[1], >= bins[1]

# observation norms (configs/rl.yaml `observation`) read by `observation`
NORM_KEYS = (
    "wait_norm_s", "pending_norm", "tis_norm_s", "red_norm_s", "ped_count_norm",
    "ped_wait_norm_s", "bus_dist_norm_m", "amb_dist_bins_m", "bus_delay_bins_s",
)  # fmt: skip


@dataclass(frozen=True)
class SiteSpec:
    """What the observation layout depends on: one per junction (Pitié: 7 lanes, 5
    stages, 4 crossings)."""

    stage_names: tuple[str, ...]
    lanes: tuple[str, ...]  # TLS-controlled vehicle lanes, TLS order
    lane_cap: tuple[float, ...]  # length / veh_length_m, at least 1
    crossings: tuple[str, ...]
    bus_approaches: tuple[str, ...] = ("S", "NB", "WB")
    amb_arms: tuple[str, ...] = ("N", "S", "W")
    pending_arms: tuple[str, ...] = ("N", "S", "W")  # general arms (insertion backlog)


def _site(cfg: dict, lanes: list[str], lengths: list[float], veh_length_m: float) -> SiteSpec:
    if not veh_length_m > 0:
        raise ValueError(f"veh_length_m must be > 0, got {veh_length_m}")
    return SiteSpec(
        stage_names=tuple(str(s["name"]) for s in cfg["network"]["stages"]),
        lanes=tuple(lanes),
        lane_cap=tuple(max(length / veh_length_m, 1.0) for length in lengths),
        crossings=tuple((cfg["network"].get("crossings") or {}).keys()),
    )


def site_spec(sim: SimSession, veh_length_m: float = VEH_LENGTH_M) -> SiteSpec:
    """SiteSpec of a started session. The lanes are the TLS-controlled vehicle lanes in
    TLS link order, unique (the `JunctionSampler` order), read through TraCI."""
    tr = sim.traci
    lanes: list[str] = []
    for conns in tr.trafficlight.getControlledLinks(sim.tls):
        for from_lane, _to, _via in conns:
            # crossings start on walking areas (internal lanes ':...')
            if not from_lane.startswith(":") and from_lane not in lanes:
                lanes.append(from_lane)
    lengths = [float(tr.lane.getLength(lane)) for lane in lanes]
    return _site(sim.cfg, lanes, lengths, veh_length_m)


def site_spec_from_config(cfg: dict, veh_length_m: float = VEH_LENGTH_M) -> SiteSpec:
    """The same SiteSpec read from the scenario's .net.xml, without starting SUMO (the
    environment sizes its observation space with it; `site_spec` checks it at reset)."""
    from traffic_twin.sim.tls_check import load_program

    net_file = cfg["network"]["net_file"]
    prog = load_program(net_file, cfg["network"]["tls_id"])
    lanes: list[str] = []
    for i in sorted(prog.links):
        lane = prog.links[i].from_lane
        if not lane.startswith(":") and lane not in lanes:
            lanes.append(lane)
    lengths = _lane_lengths(str(net_file))
    return _site(cfg, lanes, [lengths[lane] for lane in lanes], veh_length_m)


def _lane_lengths(net_file: str) -> dict[str, float]:
    import xml.etree.ElementTree as ET

    root = ET.parse(net_file).getroot()
    return {ln.get("id"): float(ln.get("length")) for ln in root.iter("lane")}


def feature_names(site: SiteSpec) -> tuple[str, ...]:
    """Observation names in order (74 at Pitié)."""
    names: list[str] = []
    for lane in site.lanes:
        names += [f"halt_{lane}", f"veh_{lane}", f"wait_{lane}"]
    names += [f"pending_{a}" for a in site.pending_arms]
    names += [f"stage_{s}" for s in site.stage_names]
    names += ["in_transition", "time_in_stage"]
    names += [f"red_{s}" for s in site.stage_names]
    for c in site.crossings:
        names += [f"ped_count_{c}", f"ped_wait_{c}", f"ped_slack_{c}"]
    names += ["amb_present", "amb_pending"]
    names += [f"amb_dist_{b}" for b in AMB_DIST_NAMES]
    names += [f"amb_arm_{a}" for a in site.amb_arms]
    names += ["preempt_active"]
    for a in site.bus_approaches:
        names += [f"bus_present_{a}", f"bus_dist_{a}"]
        names += [f"bus_delay_{b}_{a}" for b in BUS_DELAY_NAMES]
    names += ["tod_sin", "tod_cos"]
    return tuple(names)


def _bin(value: float, edges: tuple[float, float]) -> int:
    """0 if value < edges[0], 1 if < edges[1], else 2."""
    return 0 if value < edges[0] else 1 if value < edges[1] else 2


def _edges(norm: Mapping, key: str) -> tuple[float, float]:
    edges = tuple(float(x) for x in norm[key])
    if len(edges) != 2 or not edges[0] < edges[1]:
        raise ValueError(f"observation.{key} must be two increasing values, got {norm[key]}")
    return edges  # type: ignore[return-value]


def observation(snap: JunctionSnapshot, site: SiteSpec, norm: Mapping) -> np.ndarray:
    """The observation vector of a snapshot: float32, clipped to [0, 1], in
    `feature_names(site)` order. `norm` is `configs/rl.yaml: observation`."""
    if tuple(ls.lane for ls in snap.lanes) != site.lanes:
        raise ValueError(
            f"snapshot lanes {[ls.lane for ls in snap.lanes]} differ from the site's {site.lanes}"
        )
    n_stages, n_cross = len(site.stage_names), len(site.crossings)
    if len(snap.red_s) != n_stages:
        raise ValueError(f"snapshot has {len(snap.red_s)} stages, the site {n_stages}")
    wait_n, pend_n = float(norm["wait_norm_s"]), float(norm["pending_norm"])
    ped_n, pedw_n = float(norm["ped_count_norm"]), float(norm["ped_wait_norm_s"])
    bus_n = float(norm["bus_dist_norm_m"])
    amb_edges, delay_edges = _edges(norm, "amb_dist_bins_m"), _edges(norm, "bus_delay_bins_s")

    x: list[float] = []
    for ls, cap in zip(snap.lanes, site.lane_cap, strict=True):
        x += [ls.halting / cap, ls.vehicles / cap, ls.max_wait_s / wait_n]
    x += [snap.pending.get(a, 0) / pend_n for a in site.pending_arms]

    onehot = [0.0] * n_stages
    onehot[snap.stage] = 1.0
    x += onehot
    x += [float(snap.in_transition), snap.time_in_stage / float(norm["tis_norm_s"])]
    x += [r / float(norm["red_norm_s"]) for r in snap.red_s]

    counts = snap.ped_count or (0,) * n_cross
    waits = snap.ped_wait_s or (0.0,) * n_cross
    slacks = snap.ped_slack_s or (math.inf,) * n_cross
    for c, w, s in zip(counts, waits, slacks, strict=True):
        x += [c / ped_n, w / pedw_n, 1.0 if s == math.inf else s / pedw_n]

    calls = snap.emergency
    inserted = [e.dist_m for e in calls if not e.pending and math.isfinite(e.dist_m)]
    dist = [0.0] * len(AMB_DIST_NAMES)
    if inserted:
        dist[_bin(min(inserted), amb_edges)] = 1.0
    arms = {e.approach for e in calls}
    x += [float(bool(calls)), float(any(e.pending for e in calls)), *dist]
    x += [1.0 if a in arms else 0.0 for a in site.amb_arms]
    x += [float(snap.preempt_target is not None)]

    for a in site.bus_approaches:
        mine = [b for b in snap.buses if b.approach == a]
        delay = [0.0] * len(BUS_DELAY_NAMES)
        if mine:
            near = min(b.dist_m for b in mine) / bus_n
            loss = max(snap.bus_time_loss_s.get(b.vid, 0.0) for b in mine)
            delay[_bin(loss, delay_edges)] = 1.0
            x += [1.0, near, *delay]
        else:
            x += [0.0, 1.0, *delay]

    angle = 2.0 * math.pi * (snap.clock_s % DAY_S) / DAY_S
    x += [0.5 + 0.5 * math.sin(angle), 0.5 + 0.5 * math.cos(angle)]
    out = np.asarray(x, dtype=np.float32)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0, out=out)

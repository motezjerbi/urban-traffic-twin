"""Simulation vs reality validation (spec 13.4, A-02): reference sensor flows, field
measurements, flow / signal / queue tables, the calibration loop and docs/validation.md.

The settings are pre-registered in `configs/validation.yaml` (`load_validation`). Clock
times ("07:30", "08:15:20") are seconds since midnight via `clock_s`; simulation times
are relative to the scenario start (`begin_s`).
"""

from __future__ import annotations

import re
from pathlib import Path

from traffic_twin.config import ConfigError, load_config
from traffic_twin.paths import VALIDATION_CONFIG

VALIDATION_KEYS = (
    "preregistered", "reference.file", "reference.min_days", "reference.mad_k",
    "seeds.evaluation", "seeds.calibration", "slots", "stations", "thresholds",
    "calibration_loop", "compare.headline",
)  # fmt: skip

CHANGE_KEYS = ("date", "what", "why")

_CLOCK = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?$")


def load_validation(path: str | Path = VALIDATION_CONFIG) -> dict:
    """configs/validation.yaml with its required keys checked. `changes` (the dated
    changes since pre-registration, rendered in docs/validation.md) defaults to [], and
    every entry needs a date, what and why."""
    vcfg = load_config(path, VALIDATION_KEYS)
    changes = vcfg.get("changes") or []
    if not isinstance(changes, list):
        raise ConfigError(f"{path}: changes must be a list of {{date, what, why}}")
    bad = []
    for i, c in enumerate(changes):
        missing = [k for k in CHANGE_KEYS if not (isinstance(c, dict) and c.get(k))]
        if missing:
            bad.append(f"changes[{i}] needs {', '.join(missing)}")
    if bad:
        raise ConfigError(f"{path}: {'; '.join(bad)}")
    vcfg["changes"] = changes
    return vcfg


def clock_s(text: str) -> float:
    """Seconds since midnight of a clock time "HH:MM" or "HH:MM:SS"."""
    m = _CLOCK.match(text.strip()) if isinstance(text, str) else None
    if not m or int(m.group(1)) > 24 or int(m.group(2)) > 59:
        raise ConfigError(f"bad clock time {text!r} (expected a quoted 'HH:MM' or 'HH:MM:SS')")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3) or 0)


def clock_str(seconds: float) -> str:
    """'HH:MM' (or 'HH:MM:SS' when the seconds are not zero) of seconds since midnight."""
    s = int(round(seconds))
    hh, mm, ss = s // 3600, s // 60 % 60, s % 60
    return f"{hh:02d}:{mm:02d}" + (f":{ss:02d}" if ss else "")


def slot_bounds(slot: dict, begin_s: float) -> tuple[float, float]:
    """A slot's [begin, end) in seconds relative to the scenario start."""
    return clock_s(slot["begin"]) - begin_s, clock_s(slot["end"]) - begin_s

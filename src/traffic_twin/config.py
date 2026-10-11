"""YAML config loading with checks for required keys (dotted paths, e.g. "network.tls_id")."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Missing or invalid configuration."""


# Keys each config file must define (more specific checks live with the code using them)
SCENARIO_KEYS = (
    "scenario", "seed", "begin_s", "duration_s", "controller",
    "network.net_file", "network.tls_id", "network.stages", "network.approaches",
    "demand.approaches", "demand.bus", "demand.pedestrians", "outputs.dir",
)  # fmt: skip
DATA_KEYS = (
    "source.base_url", "source.dataset", "source.months", "zones",
    "cleaning.weekdays", "cleaning.timezone", "cleaning.max_flow_veh_h",
    "cleaning.max_occupancy_pct", "paths.raw", "paths.processed", "paths.figures",
)  # fmt: skip
INTERSECTIONS_KEYS = ("weights", "candidates")

_MISSING = object()


def get(cfg: dict, key: str, default: Any = _MISSING) -> Any:
    """Value at a dotted key; raises ConfigError if absent and no default is given."""
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise ConfigError(f"missing config key '{key}'")
            return default
        node = node[part]
    return node


def load_config(path: str | Path, required: Iterable[str] = ()) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    missing = [k for k in required if get(cfg, k, None) is None]  # absent or empty
    if missing:
        raise ConfigError(f"{path}: missing keys {', '.join(missing)}")
    return cfg


def load_scenario(path: str | Path) -> dict[str, Any]:
    return load_config(path, SCENARIO_KEYS)


def stage_names(cfg: dict) -> list[str]:
    """Green stage names of a scenario config, in cycle order."""
    return [s["name"] for s in get(cfg, "network.stages")]

"""API settings (Part 5, D53): `configs/api.yaml`, with two environment overrides.

- `TT_API_CONFIG`: another settings file (e.g. the `data/fake/api.yaml` that
  `python -m traffic_twin.api.fixtures --out data/fake` writes for the mock screens).
- `TT_RESULTS_DIRS`: run folders to index instead of `results_dirs`, separated by
  `os.pathsep` (`;` on Windows). The evaluation outputs are then read from
  `<first dir>/evaluation`.

Relative paths resolve against the repo root: the working directory when it holds the
settings file (every CLI runs from the repo root), else the checkout this package lives
in (an editable install run from elsewhere).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

from traffic_twin.config import ConfigError, load_config
from traffic_twin.paths import API_CONFIG, FIELD_FILE

# Part 7 operator weight-sensitivity table (spec 12; committed, so /sensitivity works on a
# fresh clone): the default when api.yaml has no `sensitivity` key
SENSITIVITY_CSV = Path("docs") / "figures" / "rl7" / "sensitivity.csv"

ENV_CONFIG = "TT_API_CONFIG"
ENV_RESULTS_DIRS = "TT_RESULTS_DIRS"
CLOCK_MODES = ("fixed", "loop")  # fixed: `clock.default_t`; loop: wall clock folded onto the run
FRESHNESS_TYPES = ("replay_state", "crossing_status", "ambulance_route", "eta")
REQUIRED_KEYS = (
    "results_dirs", "evaluation.config", "evaluation.dir", "site_config", "eta.config",
    "eta.models_dir", "default_run", "clock.mode", "clock.default_t", "freshness_s",
    "walking_speeds_mps", "corridor", "ambulance_blocked_halting",
)  # fmt: skip
_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


def package_root() -> Path:
    """The checkout this package lives in (`src/traffic_twin/api/` -> repo root)."""
    return Path(__file__).resolve().parents[3]


def find_repo_root(config: Path = API_CONFIG) -> Path:
    """The working directory when it holds `config` (or `config` is absolute), else the
    package checkout when that one does."""
    cwd = Path.cwd()
    if config.is_absolute() or (cwd / config).is_file():
        return cwd
    root = package_root()
    return root if (root / config).is_file() else cwd


@dataclass(frozen=True)
class Settings:
    """Everything the API reads, resolved from `configs/api.yaml` (see `load_settings`).
    Tests build variants with `dataclasses.replace`."""

    repo_root: Path
    results_dirs: tuple[Path, ...]
    evaluation_config: Path
    evaluation_dir: Path
    site_config: Path
    eta_config: Path
    eta_models_dir: Path
    default_run: tuple[Path, ...]  # first that is indexed wins
    clock_mode: str = "fixed"
    default_t: str = "08:15"
    replay_date: date | None = None  # date stamped on replay instants (None: today, Paris)
    freshness_s: Mapping[str, float] = field(default_factory=dict)
    bus_headway_factor: float = 1.5  # bus freshness = factor x scheduled headway
    walking_speeds_mps: Mapping[str, float] = field(default_factory=dict)
    corridor: tuple[Mapping[str, Any], ...] = ()
    ambulance_blocked_halting: int = 10
    mock: bool = False  # dev: serve app/mock.html at /mock
    demo_seeds: tuple[int, int] | None = None
    field_file: Path = FIELD_FILE  # field inventory: `provisional` until timings are measured
    index_ttl_s: float = 30.0  # the run index is rebuilt at most this often
    sensitivity: Path = SENSITIVITY_CSV  # /sensitivity table (Part 7, D103)
    source: Path | None = None  # the settings file

    def path(self, p: str | Path) -> Path:
        """`p` resolved against the repo root (absolute paths are kept)."""
        p = Path(p)
        return p if p.is_absolute() else self.repo_root / p


def _paths(value: Any, key: str) -> tuple[Path, ...]:
    if isinstance(value, str | Path):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str | Path) for v in value):
        raise ConfigError(f"api settings: {key} must be a path or a list of paths")
    return tuple(Path(v) for v in value)


def _positive_map(value: Any, key: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"api settings: {key} must be a mapping")
    out = {}
    for k, v in value.items():
        if isinstance(v, bool) or not isinstance(v, int | float) or v <= 0:
            raise ConfigError(f"api settings: {key}.{k} must be a number > 0, got {v!r}")
        out[str(k)] = float(v)
    return out


def _corridor(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("api settings: corridor must be a non-empty list")
    out = []
    for i, j in enumerate(value):
        if not isinstance(j, Mapping) or not {"id", "name", "modelled"} <= set(j):
            raise ConfigError(f"api settings: corridor[{i}] needs id, name and modelled")
        out.append({"id": str(j["id"]), "name": str(j["name"]), "modelled": bool(j["modelled"])})
    if len({j["id"] for j in out}) != len(out):
        raise ConfigError("api settings: corridor ids must be unique")
    return tuple(out)


def settings_from_dict(
    raw: Mapping[str, Any], repo_root: Path, source: Path | None = None
) -> Settings:
    """Settings from a parsed `api.yaml` mapping (checked; ConfigError on a bad value)."""
    clock = raw["clock"]
    mode = clock["mode"]
    if mode not in CLOCK_MODES:
        raise ConfigError(f"api settings: clock.mode must be one of {CLOCK_MODES}, got {mode!r}")
    default_t = str(clock["default_t"])
    if not _CLOCK_RE.match(default_t):
        raise ConfigError(f"api settings: clock.default_t must be HH:MM[:SS], got {default_t!r}")
    replay_date = clock.get("replay_date")
    if replay_date is not None and not isinstance(replay_date, date):
        try:
            replay_date = date.fromisoformat(str(replay_date))
        except ValueError as e:
            raise ConfigError(f"api settings: clock.replay_date {replay_date!r}: {e}") from e
    fresh = dict(raw["freshness_s"])
    factor = fresh.pop("bus_headway_factor", 1.5)
    freshness = _positive_map(fresh, "freshness_s")
    missing = [k for k in FRESHNESS_TYPES if k not in freshness]
    if missing:
        raise ConfigError(f"api settings: freshness_s misses {missing}")
    halting = raw["ambulance_blocked_halting"]
    if isinstance(halting, bool) or not isinstance(halting, int) or halting < 1:
        raise ConfigError("api settings: ambulance_blocked_halting must be an integer >= 1")
    demo = (raw.get("seeds") or {}).get("demo")
    return Settings(
        repo_root=repo_root,
        results_dirs=_paths(raw["results_dirs"], "results_dirs"),
        evaluation_config=Path(raw["evaluation"]["config"]),
        evaluation_dir=Path(raw["evaluation"]["dir"]),
        site_config=Path(raw["site_config"]),
        eta_config=Path(raw["eta"]["config"]),
        eta_models_dir=Path(raw["eta"]["models_dir"]),
        default_run=_paths(raw["default_run"], "default_run"),
        clock_mode=mode,
        default_t=default_t,
        replay_date=replay_date,
        freshness_s=freshness,
        bus_headway_factor=_positive_map({"f": factor}, "freshness_s.bus_headway_factor")["f"],
        walking_speeds_mps=_positive_map(raw["walking_speeds_mps"], "walking_speeds_mps"),
        corridor=_corridor(raw["corridor"]),
        ambulance_blocked_halting=halting,
        mock=bool((raw.get("dev") or {}).get("mock", False)),
        demo_seeds=tuple(demo) if demo else None,
        sensitivity=_paths(raw.get("sensitivity") or SENSITIVITY_CSV, "sensitivity")[0],
        source=source,
    )


def load_settings(
    path: str | Path | None = None,
    *,
    repo_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Settings from `path` (default: `TT_API_CONFIG`, else configs/api.yaml), with the
    `TT_RESULTS_DIRS` override. ConfigError when the file is absent or invalid."""
    env = os.environ if environ is None else environ
    cfg_path = Path(path or env.get(ENV_CONFIG) or API_CONFIG)
    root = repo_root if repo_root is not None else find_repo_root(cfg_path)
    full = cfg_path if cfg_path.is_absolute() else root / cfg_path
    if not full.is_file():
        raise ConfigError(f"api settings: {full} not found")
    raw = load_config(full, REQUIRED_KEYS)
    settings = settings_from_dict(raw, root, source=full)
    dirs = env.get(ENV_RESULTS_DIRS)
    if dirs:
        results = tuple(Path(d) for d in dirs.split(os.pathsep) if d.strip())
        if results:
            settings = replace(
                settings, results_dirs=results, evaluation_dir=results[0] / "evaluation"
            )
    return settings

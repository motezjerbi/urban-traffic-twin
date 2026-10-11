"""YAML config loading with checks for required keys (dotted paths, e.g. "network.tls_id").

A config file may build on others with `extends:` (one path or a list, relative to the
including file's folder). The bases are merged in order, then the including file on
top: mappings merge key by key (recursively), lists and scalars are replaced. The
`extends` key is removed from the result and required keys are checked after the
merge. `configs/pitie_fixed.yaml` is assembled this way from `configs/pitie/*.yaml`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Missing or invalid configuration."""


# Keys each config file must define (more specific checks live with the code using them).
# A tuple entry lists alternatives: at least one of them must be present.
SCENARIO_KEYS = (
    "scenario", "seed", "begin_s", "duration_s", "controller",
    "network.net_file", "network.tls_id", "network.stages", "network.approaches",
    ("demand.approaches", "demand.calibration"), "demand.bus", "demand.pedestrians",
    "outputs.dir",
)  # fmt: skip
DATA_KEYS = (
    "source.base_url", "source.dataset", "source.months", "zones",
    "cleaning.weekdays", "cleaning.timezone", "cleaning.max_flow_veh_h",
    "cleaning.max_occupancy_pct", "paths.raw", "paths.processed", "paths.figures",
)  # fmt: skip
INTERSECTIONS_KEYS = ("weights", "candidates")

EXTENDS_KEY = "extends"

_MISSING = object()
# Configs returned by load_config -> the files they were assembled from (bounded; the
# config is kept alive so its id is never reused while listed)
_LOADED: OrderedDict[int, tuple[dict, list[Path]]] = OrderedDict()
_LOADED_MAX = 256


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


# -- loading and `extends` -------------------------------------------------------------
def _read_mapping(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return cfg


def _base_paths(path: Path, cfg: dict) -> list[Path]:
    """Paths named by the `extends` key of `cfg` (loaded from `path`), resolved against
    the folder of `path`. The key is removed from `cfg`."""
    bases = cfg.pop(EXTENDS_KEY, None)
    if bases is None:
        return []
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(b, str) and b for b in bases):
        raise ConfigError(f"{path}: '{EXTENDS_KEY}' must be a path or a list of paths")
    return [Path(os.path.normpath(path.parent / b)) for b in bases]


def merge(base: dict, override: dict) -> dict:
    """Deep merge: mappings merge key by key, anything else in `override` replaces the
    value in `base`. Neither input is modified."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve(path: Path, chain: tuple[Path, ...], sources: list[Path]) -> dict[str, Any]:
    """Load `path` with its bases merged in. `chain` holds the files being resolved
    (to detect cycles); every file read is appended to `sources` once, bases first."""
    key = path.resolve()
    if key in chain:
        names = " -> ".join(str(p) for p in (*chain, key))
        raise ConfigError(f"'{EXTENDS_KEY}' cycle: {names}")
    if not path.is_file():
        via = f" (extended by {chain[-1]})" if chain else ""
        raise ConfigError(f"config file not found: {path}{via}")
    cfg = _read_mapping(path)
    merged: dict[str, Any] = {}
    for base in _base_paths(path, cfg):
        merged = merge(merged, _resolve(base, (*chain, key), sources))
    if all(p.resolve() != key for p in sources):
        sources.append(path)
    return merge(merged, cfg)


def _missing_keys(cfg: dict, required: Iterable[str | Sequence[str]]) -> list[str]:
    """Required keys that are absent or empty (None). A non-string entry lists
    alternatives, reported as "a or b" when none of them is set."""
    missing = []
    for k in required:
        alternatives = [k] if isinstance(k, str) else list(k)
        if all(get(cfg, a, None) is None for a in alternatives):
            missing.append(" or ".join(alternatives))
    return missing


def load_config(path: str | Path, required: Iterable[str | Sequence[str]] = ()) -> dict[str, Any]:
    """Load a YAML config, merging the files it `extends`, and check required keys."""
    path = Path(path)
    sources: list[Path] = []
    cfg = _resolve(path, (), sources)
    missing = _missing_keys(cfg, required)
    if missing:
        raise ConfigError(f"{path}: missing keys {', '.join(missing)}")
    _LOADED[id(cfg)] = (cfg, sources)
    while len(_LOADED) > _LOADED_MAX:
        _LOADED.popitem(last=False)
    return cfg


def config_sources(path: str | Path) -> list[Path]:
    """The files a config is assembled from, in merge order (bases first, the file
    itself last), each listed once. Paths are relative when `path` is."""
    sources: list[Path] = []
    _resolve(Path(path), (), sources)
    return sources


def loaded_sources(cfg: dict) -> list[Path] | None:
    """`config_sources` of the file `cfg` came from, when `cfg` is the very object
    `load_config` returned (possibly modified in place since); None for a config built or
    deep-copied in memory."""
    entry = _LOADED.get(id(cfg))
    return list(entry[1]) if entry is not None and entry[0] is cfg else None


def load_scenario(path: str | Path) -> dict[str, Any]:
    return load_config(path, SCENARIO_KEYS)


# -- scenario helpers ------------------------------------------------------------------
def stage_names(cfg: dict) -> list[str]:
    """Green stage names of a scenario config, in cycle order."""
    return [s["name"] for s in get(cfg, "network.stages")]


def _approach_edges(cfg: dict, key: str) -> dict[str, list[str]]:
    out = {}
    for name, a in get(cfg, "network.approaches").items():
        if isinstance(a, str):  # grid form {W: w_in}: out edge by the <leg>_in/<leg>_out rule
            if key == "in_edges":
                edges = [a]
            elif a.endswith("_in"):
                edges = [a[: -len("_in")] + "_out"]
            else:
                raise ConfigError(
                    f"network.approaches.{name}: cannot derive the out edge of '{a}' "
                    "(name it <leg>_in or give {in_edges: [...], out_edges: [...]})"
                )
        elif isinstance(a, dict) and isinstance(a.get(key), list) and a[key]:
            edges = [str(e) for e in a[key]]
        else:
            raise ConfigError(f"network.approaches.{name} needs a non-empty '{key}' list")
        out[name] = edges
    return out


def approach_in_edges(cfg: dict) -> dict[str, list[str]]:
    """Incoming edges per approach. Accepts `{W: "w_in"}` (grid) or
    `{W: {in_edges: [...], out_edges: [...]}}`."""
    return _approach_edges(cfg, "in_edges")


def approach_out_edges(cfg: dict) -> dict[str, list[str]]:
    """Outgoing edges per approach. For the grid form `{W: "w_in"}` the out edge follows
    the naming rule `<leg>_in` -> `<leg>_out`."""
    return _approach_edges(cfg, "out_edges")


# -- hashes ----------------------------------------------------------------------------
def _canonical(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace; dates and paths as strings."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text_bytes(path: str | Path | None) -> bytes | None:
    """A text file's bytes with LF line ends and no UTF-8 BOM, or None when no path is
    given or the file is absent. Git autocrlf checks text files out with CRLF on Windows
    and LF elsewhere: the same content must hash the same on every clone."""
    if not path or not Path(path).is_file():
        return None
    return Path(path).read_bytes().removeprefix(b"\xef\xbb\xbf").replace(b"\r\n", b"\n")


def _xml_body(text: bytes) -> bytes:
    """An XML document from its root element on: drops the prolog (XML declaration,
    comments, whitespace). netconvert writes its build time and its options (input paths)
    in a comment there, so a rebuild that changes nothing else keeps the body."""
    rest = text.lstrip()
    while rest.startswith((b"<?", b"<!--")):
        close = b"?>" if rest.startswith(b"<?") else b"-->"
        end = rest.find(close)
        if end < 0:
            return text  # malformed prolog: hash the whole text
        rest = rest[end + len(close) :].lstrip()
    return rest


def text_sha256(path: str | Path | None) -> str | None:
    """sha256 of a text file with LF line ends (stable across git autocrlf checkouts),
    or None when no path is given or the file is absent."""
    text = _text_bytes(path)
    return None if text is None else hashlib.sha256(text).hexdigest()


def xml_sha256(path: str | Path | None) -> str | None:
    """`text_sha256` of an XML file from its root element on (`_xml_body`): ignores line
    ends and the generator's header comment, e.g. netconvert's `<!-- generated on ...`."""
    text = _text_bytes(path)
    return None if text is None else hashlib.sha256(_xml_body(text)).hexdigest()


def config_hash(cfg: dict) -> str:
    """sha256 of the whole merged config as canonical JSON, except `outputs` (where the
    results are written does not change them)."""
    return _sha256_text(_canonical({k: v for k, v in cfg.items() if k != "outputs"}))


def demand_hash(cfg: dict) -> str:
    """sha256 identifying the demand a run sees (spec 13.3: S0 and S1 must share it).

    Covers `begin_s`, `duration_s`, `step_length_s`, `demand`, `emergency` and
    `network.{net_file, approaches, crossings, bus_stops}` as canonical JSON, plus the
    sha256 of the calibration file (`demand.calibration`, `text_sha256`) and of the net
    file (`xml_sha256`: from the `<net>` element on). Both are hashed as text with LF line
    ends, so the hash is the same on a Windows (CRLF) and a Linux (LF) clone, and a net
    rebuild that only rewrites netconvert's header comment (build time) keeps it. Paths
    are relative to the working directory (the repo root); an absent file hashes as None.
    """
    net = cfg.get("network", {}) or {}
    demand = cfg.get("demand", {}) or {}
    payload = {
        "begin_s": cfg.get("begin_s"),
        "duration_s": cfg.get("duration_s"),
        "step_length_s": cfg.get("step_length_s", 1.0),
        "demand": demand,
        "emergency": cfg.get("emergency"),
        "network": {k: net.get(k) for k in ("net_file", "approaches", "crossings", "bus_stops")},
        "calibration_sha256": text_sha256(demand.get("calibration")),
        "net_sha256": xml_sha256(net.get("net_file")),
    }
    return _sha256_text(_canonical(payload))

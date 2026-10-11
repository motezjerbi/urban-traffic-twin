"""Controllers by name. A controller module registers its class with `@register("name")`;
every module of this package is imported on first lookup, so adding a controller is
just adding a module (keep heavy imports, e.g. torch, inside functions). The scenario
config selects one with `controller: <name>`."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable

from traffic_twin.config import ConfigError, get
from traffic_twin.control.base import Controller

_REGISTRY: dict[str, type[Controller]] = {}


def register(name: str) -> Callable[[type[Controller]], type[Controller]]:
    def deco(cls: type[Controller]) -> type[Controller]:
        if _REGISTRY.get(name, cls) is not cls:
            raise ValueError(f"controller '{name}' is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _import_package_modules() -> None:
    import traffic_twin.control as pkg

    for m in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{m.name}")


def available() -> list[str]:
    _import_package_modules()
    return sorted(_REGISTRY)


def make_controller(cfg: dict) -> Controller:
    """Build the controller named by `cfg["controller"]` from the scenario config."""
    name = get(cfg, "controller")
    if name not in _REGISTRY:
        _import_package_modules()
    if name not in _REGISTRY:
        raise ConfigError(f"unknown controller '{name}' (available: {', '.join(available())})")
    return _REGISTRY[name].from_config(cfg)

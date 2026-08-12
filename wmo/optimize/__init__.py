"""Lazy public optimizer contracts and artifact-specific optimizer packages.

Offline guarded router fitting lives in :mod:`wmo.optimize.router`, model
training lives in :mod:`wmo.optimize.model`, and prompt evolution remains an
optional package-root export. Importing a focused optimizer must not initialize
GEPA or simulation modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmo.optimize.base import (
        ArtifactRef as ArtifactRef,
    )
    from wmo.optimize.base import (
        OptimizeMetrics as OptimizeMetrics,
    )
    from wmo.optimize.base import (
        Optimizer as Optimizer,
    )
    from wmo.optimize.base import (
        OptimizeResult as OptimizeResult,
    )
    from wmo.optimize.gepa import GEPAOptimizer as GEPAOptimizer

_EXPORT_MODULES = {
    "ArtifactRef": "wmo.optimize.base",
    "OptimizeMetrics": "wmo.optimize.base",
    "Optimizer": "wmo.optimize.base",
    "OptimizeResult": "wmo.optimize.base",
    "GEPAOptimizer": "wmo.optimize.gepa",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> object:
    """Resolve one supported optimizer export on first access.

    Args:
        name: Package attribute requested by Python.

    Returns:
        The supported optimizer object loaded from its owning module.

    Raises:
        AttributeError: The name is not part of the supported public API.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus supported lazy exports for introspection."""
    return sorted(set(globals()) | set(__all__))

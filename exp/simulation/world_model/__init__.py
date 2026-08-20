"""Persisted grounded world-model build artifacts and executable runtime."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exp.simulation.world_model.application import WorldModel as WorldModel
    from exp.simulation.world_model.application import (
        WorldModelLoadError as WorldModelLoadError,
    )
    from exp.simulation.world_model.application import (
        WorldModelObservation as WorldModelObservation,
    )
    from exp.simulation.world_model.application import WorldModelSession as WorldModelSession
    from exp.simulation.world_model.application import (
        WorldModelSessionError as WorldModelSessionError,
    )
    from exp.simulation.world_model.application import (
        WorldModelSessionLimits as WorldModelSessionLimits,
    )
    from exp.simulation.world_model.application import load_world_model as load_world_model
from exp.simulation.world_model.artifact import (
    GROUNDED_WORLD_MODEL_ARTIFACT_TYPE,
    GroundedWorldModelArtifact,
    PersistedGroundedWorldModel,
    persist_grounded_world_model,
)
from exp.simulation.world_model.runtime import (
    GroundedWorldModel,
    bind_fit_grounded_world_model,
    load_grounded_world_model,
    load_grounded_world_model_artifact,
    verify_grounded_world_model_artifact,
)

_APPLICATION_EXPORTS = frozenset(
    {
        "WorldModel",
        "WorldModelLoadError",
        "WorldModelObservation",
        "WorldModelSession",
        "WorldModelSessionError",
        "WorldModelSessionLimits",
        "load_world_model",
    }
)

__all__ = [
    "GROUNDED_WORLD_MODEL_ARTIFACT_TYPE",
    "GroundedWorldModel",
    "GroundedWorldModelArtifact",
    "PersistedGroundedWorldModel",
    "WorldModel",
    "WorldModelLoadError",
    "WorldModelObservation",
    "WorldModelSession",
    "WorldModelSessionError",
    "WorldModelSessionLimits",
    "bind_fit_grounded_world_model",
    "load_grounded_world_model",
    "load_grounded_world_model_artifact",
    "load_world_model",
    "persist_grounded_world_model",
    "verify_grounded_world_model_artifact",
]


def __getattr__(name: str) -> object:
    """Resolve one public application service without loading it during package import.

    Args:
        name: Package attribute requested by Python.

    Returns:
        The supported public application object loaded from its owning module.

    Raises:
        AttributeError: The name is not a lazy public application export.
    """
    if name not in _APPLICATION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("exp.simulation.world_model.application"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus supported lazy application exports."""
    return sorted(set(globals()) | _APPLICATION_EXPORTS)

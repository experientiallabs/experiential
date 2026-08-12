"""World Model Optimizer public API with lazy package-root exports.

Importing a focused subpackage must not initialize the simulation or offline
optimizer graphs. Package-root attributes remain available for supported
Python callers and are imported only when accessed.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmo.common.core.types import (
        Action as Action,
    )
    from wmo.common.core.types import (
        ActionKind as ActionKind,
    )
    from wmo.common.core.types import (
        EnvState as EnvState,
    )
    from wmo.common.core.types import (
        Observation as Observation,
    )
    from wmo.common.core.types import (
        Session as Session,
    )
    from wmo.common.core.types import (
        Step as Step,
    )
    from wmo.common.core.types import (
        Trace as Trace,
    )
    from wmo.runtime import (
        DONE_SIGNAL as DONE_SIGNAL,
    )
    from wmo.runtime import (
        Agent as Agent,
    )
    from wmo.runtime import (
        Env as Env,
    )
    from wmo.runtime import (
        EpisodeResult as EpisodeResult,
    )
    from wmo.runtime import (
        StopReason as StopReason,
    )
    from wmo.runtime import (
        run_episode as run_episode,
    )
    from wmo.simulation.environment import WorldModelEnv as WorldModelEnv
    from wmo.simulation.model.world_model import WorldModel as WorldModel

_EXPORT_MODULES = {
    "Action": "wmo.common.core.types",
    "ActionKind": "wmo.common.core.types",
    "EnvState": "wmo.common.core.types",
    "Observation": "wmo.common.core.types",
    "Session": "wmo.common.core.types",
    "Step": "wmo.common.core.types",
    "Trace": "wmo.common.core.types",
    "DONE_SIGNAL": "wmo.runtime",
    "Agent": "wmo.runtime",
    "Env": "wmo.runtime",
    "EpisodeResult": "wmo.runtime",
    "StopReason": "wmo.runtime",
    "run_episode": "wmo.runtime",
    "WorldModelEnv": "wmo.simulation.environment",
    "WorldModel": "wmo.simulation.model.world_model",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> object:
    """Resolve one supported package-root export on first access.

    Args:
        name: Package attribute requested by Python.

    Returns:
        The supported public object loaded from its owning module.

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

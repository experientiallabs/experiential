"""Frozen online router runtime and loopback HTTP adapter."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmo.runtime.router.application import (
        RouterApplicationError as RouterApplicationError,
    )
    from wmo.runtime.router.application import (
        create_project_completion_service as create_project_completion_service,
    )
    from wmo.runtime.router.application import (
        create_project_router_app as create_project_router_app,
    )
    from wmo.runtime.router.endpoint import create_router_endpoint as create_router_endpoint

from wmo.runtime.router.completion import (
    JournaledRouterCompletionService,
    RouterCompletionConflictError,
    RouterCompletionFailedError,
    RouterCompletionInProgressError,
    RouterCompletionService,
)
from wmo.runtime.router.journal import (
    JournaledRouterRuntime,
    RuntimeAcceptedEvent,
    RuntimeAttemptFailedEvent,
    RuntimeCompletedEvent,
    RuntimeIdempotencyConflictError,
    RuntimeInteractionFailedError,
    RuntimeInteractionInProgressError,
    RuntimeInteractionJournal,
    RuntimeJournalError,
    RuntimeJournalEvent,
)
from wmo.runtime.router.runtime import (
    RoutedModelResponse,
    RouterEpisodeConflictError,
    RouterModelCapabilityError,
    RouterRuntime,
    RouterRuntimeIntegrityError,
)
from wmo.runtime.router.snapshot import (
    LoadedRuntimeTraceSnapshot,
    PersistedRuntimeTraceExport,
    RuntimeTraceAttempt,
    RuntimeTraceInteraction,
    RuntimeTraceSnapshot,
    RuntimeTraceSnapshotError,
    load_runtime_trace_snapshot,
    routed_task_text,
    seal_runtime_trace_snapshot,
)

_SERVER_EXPORT_MODULES = {
    "RouterApplicationError": "wmo.runtime.router.application",
    "create_project_completion_service": "wmo.runtime.router.application",
    "create_project_router_app": "wmo.runtime.router.application",
    "create_router_endpoint": "wmo.runtime.router.endpoint",
}

__all__ = [
    "RoutedModelResponse",
    "JournaledRouterRuntime",
    "JournaledRouterCompletionService",
    "RuntimeAcceptedEvent",
    "RuntimeAttemptFailedEvent",
    "RuntimeCompletedEvent",
    "RuntimeIdempotencyConflictError",
    "RuntimeInteractionFailedError",
    "RuntimeInteractionInProgressError",
    "RuntimeInteractionJournal",
    "RuntimeJournalError",
    "RuntimeJournalEvent",
    "RouterEpisodeConflictError",
    "RouterModelCapabilityError",
    "RouterCompletionService",
    "RouterCompletionConflictError",
    "RouterCompletionInProgressError",
    "RouterCompletionFailedError",
    "RouterRuntime",
    "RouterRuntimeIntegrityError",
    "LoadedRuntimeTraceSnapshot",
    "PersistedRuntimeTraceExport",
    "RuntimeTraceAttempt",
    "RuntimeTraceInteraction",
    "RuntimeTraceSnapshot",
    "RuntimeTraceSnapshotError",
    "load_runtime_trace_snapshot",
    "routed_task_text",
    "seal_runtime_trace_snapshot",
    "create_router_endpoint",
    "RouterApplicationError",
    "create_project_completion_service",
    "create_project_router_app",
]


def __getattr__(name: str) -> object:
    """Resolve one server-facing router service without loading FastAPI at package import.

    Args:
        name: Package attribute requested by Python.

    Returns:
        The supported public server object loaded from its owning module.

    Raises:
        AttributeError: The name is not a lazy server-facing export.
    """
    module_name = _SERVER_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus supported lazy server exports."""
    return sorted(set(globals()) | set(_SERVER_EXPORT_MODULES))

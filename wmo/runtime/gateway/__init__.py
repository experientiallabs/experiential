"""Provider-neutral contracts for the local identity-aware model gateway."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from wmo.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    CompatibilityDisposition,
    CompatibilityField,
    CompatibilityManifest,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayTarget,
    GatewayToolDefinition,
    GatewayUsage,
    ProjectSelection,
    ProjectTarget,
    StructuredTextFormat,
)
from wmo.runtime.gateway.interfaces import (
    AttemptLedger,
    GatewayClock,
    GatewayControlStore,
    ProjectTargetResolver,
    ProviderStream,
    SecretResolver,
)

if TYPE_CHECKING:
    from wmo.runtime.gateway.composition import GatewayRuntime as GatewayRuntime
    from wmo.runtime.gateway.composition import GatewayRuntimeConfig as GatewayRuntimeConfig
    from wmo.runtime.gateway.composition import create_gateway_runtime as create_gateway_runtime

_LAZY_EXPORT_MODULES = {
    "GatewayRuntime": "wmo.runtime.gateway.composition",
    "GatewayRuntimeConfig": "wmo.runtime.gateway.composition",
    "create_gateway_runtime": "wmo.runtime.gateway.composition",
}

__all__ = [
    "AttemptLedger",
    "AuthorizationSnapshot",
    "CompatibilityDisposition",
    "CompatibilityField",
    "CompatibilityManifest",
    "DirectTarget",
    "ExecutionSnapshot",
    "GatewayApiSurface",
    "GatewayClock",
    "GatewayControlStore",
    "GatewayEvent",
    "GatewayEventKind",
    "GatewayFailure",
    "GatewayFailureClass",
    "GatewayMessage",
    "GatewayRequest",
    "GatewayRuntime",
    "GatewayRuntimeConfig",
    "GatewayTarget",
    "GatewayToolDefinition",
    "GatewayUsage",
    "ProjectSelection",
    "ProjectTarget",
    "ProjectTargetResolver",
    "ProviderStream",
    "SecretResolver",
    "StructuredTextFormat",
    "create_gateway_runtime",
]


def __getattr__(name: str) -> object:
    """Resolve one composition export without widening gateway import dependencies."""
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eager contracts plus supported lazy composition exports."""
    return sorted(set(globals()) | set(__all__))

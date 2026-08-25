"""Process host for the native gateway data plane."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from exp_gateway_native import ShutdownHandle

    from exp.runtime.gateway.native_bridge import NativeControlPlane


class NativeServerControlPlane(Protocol):
    """Control-plane value required by the process host."""

    @property
    def request_timeout_seconds(self) -> float:
        """Return the shared request deadline."""
        ...


class NativeGatewayServerError(RuntimeError):
    """The native extension could not serve."""


def serve_native_gateway(
    control_plane: NativeServerControlPlane,
    *,
    host: str,
    port: int,
    max_active_requests: int = 64,
    graceful_timeout_seconds: float = 10.0,
    native_usage_enabled: bool = True,
    shutdown: ShutdownHandle | None = None,
    on_listening: Callable[[], None] | None = None,
) -> None:
    """Serve the native data plane until a stop signal, blocking this thread.

    Args:
        control_plane: Shared authority and accounting callbacks.
        host: Public listener host.
        port: Public listener port.
        max_active_requests: Native concurrent-admission bound.
        graceful_timeout_seconds: Bound for graceful shutdown.
        native_usage_enabled: Whether Rust owns ``/usage.json``. Hosted,
            multi-tenant callers should disable it so their own surface owns
            usage.
        shutdown: Optional embedder-owned stop handle from
            ``exp_gateway_native.shutdown_handle()``. A host serving on a
            background thread calls ``request_shutdown()`` to stop the plane
            gracefully, since threads cannot receive SIGINT.
        on_listening: Optional callback the native server invokes exactly
            once, after its listener socket is bound and queuing
            connections and before any request is accepted, so the embedder
            can announce readiness truthfully. A callback failure aborts
            the launch.

    Raises:
        NativeGatewayServerError: The extension is unavailable or the native
            server fails.
    """
    try:
        native = importlib.import_module("exp_gateway_native")
    except ModuleNotFoundError as exc:
        raise NativeGatewayServerError("the exp_gateway_native extension is not installed") from exc

    config: dict[str, object] = {
        "host": host,
        "port": port,
        "max_active_requests": max_active_requests,
        "request_timeout_seconds": control_plane.request_timeout_seconds,
        "graceful_timeout_seconds": graceful_timeout_seconds,
        "native_usage_enabled": native_usage_enabled,
    }
    try:
        native.serve(
            cast("NativeControlPlane", control_plane),
            json.dumps(config),
            shutdown,
            on_listening,
        )
    except KeyboardInterrupt:
        # The native server drains on SIGINT before returning control to Python.
        pass
    except RuntimeError as exc:
        raise NativeGatewayServerError(f"the native gateway failed: {exc}") from exc

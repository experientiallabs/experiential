"""Process host for the native gateway with an internal ASGI fallback."""

from __future__ import annotations

import importlib
import json
import socket
import threading
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, cast

import uvicorn

if TYPE_CHECKING:
    from exp.runtime.gateway.native_bridge import NativeControlPlane

_LOOPBACK_HOST = "127.0.0.1"
_FALLBACK_START_TIMEOUT_SECONDS = 30.0


AsgiApplication = Callable[..., Awaitable[None]]


class NativeServerControlPlane(Protocol):
    """Control-plane value required by the process host."""

    @property
    def request_timeout_seconds(self) -> float:
        """Return the shared request deadline."""
        ...


class NativeGatewayServerError(RuntimeError):
    """The native extension or its embedded fallback could not serve."""


def serve_native_gateway(
    fallback_app: AsgiApplication,
    control_plane: NativeServerControlPlane,
    *,
    host: str,
    port: int,
    max_active_requests: int = 64,
    graceful_timeout_seconds: float = 10.0,
    native_usage_enabled: bool = True,
) -> None:
    """Serve Rust publicly and proxy unsupported routes to an internal ASGI app.

    Args:
        fallback_app: Python ASGI application for unsupported and escalated routes.
        control_plane: Shared authority and accounting callbacks.
        host: Public listener host.
        port: Public listener port.
        max_active_requests: Native concurrent-admission bound.
        graceful_timeout_seconds: Bound for both native and fallback shutdown.
        native_usage_enabled: Whether Rust owns ``/usage.json``. Hosted,
            multi-tenant callers should disable it so their ASGI app owns usage.

    Raises:
        NativeGatewayServerError: The extension is unavailable, the fallback
            cannot start, or the native server fails.
    """
    try:
        native = importlib.import_module("exp_gateway_native")
    except ModuleNotFoundError as exc:
        raise NativeGatewayServerError("the exp_gateway_native extension is not installed") from exc

    fallback_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    fallback_socket.bind((_LOOPBACK_HOST, 0))
    fallback_port = fallback_socket.getsockname()[1]
    fallback = uvicorn.Server(
        uvicorn.Config(
            fallback_app,
            host=_LOOPBACK_HOST,
            port=fallback_port,
            log_level="warning",
        )
    )
    fallback_thread = threading.Thread(
        target=lambda: fallback.run(sockets=[fallback_socket]),
        name="exp-fallback-engine",
        daemon=True,
    )
    fallback_thread.start()
    deadline = time.monotonic() + _FALLBACK_START_TIMEOUT_SECONDS
    while not fallback.started:
        if not fallback_thread.is_alive() or time.monotonic() > deadline:
            fallback.should_exit = True
            fallback_socket.close()
            raise NativeGatewayServerError("the embedded Python fallback failed to start")
        time.sleep(0.05)
    config = json.dumps(
        {
            "host": host,
            "port": port,
            "max_active_requests": max_active_requests,
            "request_timeout_seconds": control_plane.request_timeout_seconds,
            "fallback_port": fallback_port,
            "graceful_timeout_seconds": graceful_timeout_seconds,
            "native_usage_enabled": native_usage_enabled,
        },
        separators=(",", ":"),
    )
    try:
        native.serve(cast("NativeControlPlane", control_plane), config)
    except RuntimeError as exc:
        raise NativeGatewayServerError(f"the native gateway failed: {exc}") from exc
    finally:
        fallback.should_exit = True
        fallback_thread.join(timeout=graceful_timeout_seconds + 5.0)
        fallback_socket.close()

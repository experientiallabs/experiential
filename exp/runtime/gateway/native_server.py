"""Process host for the native gateway data plane."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, cast

from exp.runtime.gateway.guardrails.deterministic import NativeDetector

if TYPE_CHECKING:
    from exp_gateway_native import CallbackScopes, CaptureCollector, ShutdownHandle

    from exp.runtime.gateway.native_bridge import NativeControlPlane


class NativeServerControlPlane(Protocol):
    """Control-plane value required by the process host."""

    @property
    def request_timeout_seconds(self) -> float:
        """Return the shared request deadline."""
        ...

    @property
    def guardrail_detectors(self) -> dict[str, NativeDetector]:
        """Return the compiled deterministic guardrail rules, keyed by adapter."""
        ...


class NativeGatewayServerError(RuntimeError):
    """The native extension could not serve."""


PRE_FIRST_BYTE_RESERVE_ENV = "EXP_GATEWAY_PRE_FIRST_BYTE_RESERVE_SECONDS"
"""Operator override of what a raised pre-first-byte budget keeps past the authored allowance."""


def pre_first_byte_reserve_seconds(environ: Mapping[str, str] | None = None) -> float | None:
    """Read the operator's pre-first-byte reserve override, if one is set.

    A request whose route authors a first-byte allowance longer than the
    configured budget gets that allowance plus this reserve (the data plane's
    default is 15 s), so a rung that stalls out its whole allowance still
    leaves a successor time to answer.

    Args:
        environ: The environment to read; ``None`` reads ``os.environ``.

    Returns:
        The override in seconds, or ``None`` to keep the data plane's default.

    Raises:
        ValueError: The override does not parse or is negative.
    """
    raw = (os.environ if environ is None else environ).get(PRE_FIRST_BYTE_RESERVE_ENV)
    if not raw:
        return None
    seconds = float(raw)
    if seconds < 0:
        raise ValueError(f"{PRE_FIRST_BYTE_RESERVE_ENV} must not be negative")
    return seconds


PRE_FIRST_BYTE_MAX_ENV = "EXP_GATEWAY_PRE_FIRST_BYTE_MAX_SECONDS"
"""Operator override of the ceiling on a pre-first-byte budget an authored allowance raised."""


def pre_first_byte_max_seconds(environ: Mapping[str, str] | None = None) -> float | None:
    """Read the operator's ceiling on a raised pre-first-byte budget, if one is set.

    An authored first-byte allowance raises a request's budget (see
    ``pre_first_byte_reserve_seconds``) to at most this many seconds (the data
    plane's default is 600 s); it never lowers the configured budget. Zero
    removes the ceiling.

    Args:
        environ: The environment to read; ``None`` reads ``os.environ``.

    Returns:
        The override in seconds, or ``None`` to keep the data plane's default.

    Raises:
        ValueError: The override does not parse or is negative.
    """
    raw = (os.environ if environ is None else environ).get(PRE_FIRST_BYTE_MAX_ENV)
    if not raw:
        return None
    seconds = float(raw)
    if seconds < 0:
        raise ValueError(f"{PRE_FIRST_BYTE_MAX_ENV} must not be negative")
    return seconds


def serve_native_gateway(
    control_plane: NativeServerControlPlane,
    *,
    host: str,
    port: int,
    max_active_requests: int = 64,
    max_active_per_organization: int | None = None,
    callback_permits: int = 4,
    graceful_timeout_seconds: float = 10.0,
    connect_timeout_seconds: float = 5.0,
    public_upstreams_only: bool = False,
    time_to_first_byte_seconds: float = 15.0,
    time_to_first_byte_seconds_per_million_input_tokens: float = 240.0,
    time_to_first_token_seconds: float = 120.0,
    pre_first_byte_budget_seconds: float = 90.0,
    native_usage_enabled: bool = True,
    maximum_retained_output_bytes: int | None = None,
    capture: CaptureCollector | None = None,
    callback_scopes: CallbackScopes | None = None,
    shutdown: ShutdownHandle | None = None,
    on_listening: Callable[[], None] | None = None,
) -> None:
    """Serve the native data plane until a stop signal, blocking this thread.

    Args:
        control_plane: Shared authority and accounting callbacks.
        host: Public listener host.
        port: Public listener port.
        max_active_requests: Native concurrent-admission bound.
        max_active_per_organization: Maximum active requests for a trusted organization,
            held through upload or stream completion. Requires callback_scopes; missing
            trusted scopes fail closed. None preserves legacy local/embedded capacity.
        callback_permits: Serving callback workers, 1 through 256. The native
            bridge reserves a separate settlement pool of max(2, permits/4)
            workers and bounds each admitted organization to half each pool.
        graceful_timeout_seconds: Bound for graceful shutdown.
        connect_timeout_seconds: Fail-fast bound on the TCP+TLS connect phase
            of every provider call, so a lane whose host never accepts the
            connection fails over in seconds instead of hanging on the
            per-deployment request timeout.
        time_to_first_byte_seconds: Fail-fast bound on the wait for a
            provider's response headers per attempt (the flat part; the
            input-scaled allowance below is added). Deployments may
            override it through their gateway capabilities.
        time_to_first_byte_seconds_per_million_input_tokens: Input-scaled
            first-byte allowance added on top of the flat bound, in seconds
            per million approximate input tokens (request bytes over four),
            so a very large prompt's prefill is not misread as a dead lane.
            Added to both the header and the first-token bounds.
            Deployments may override it through their gateway capabilities.
        time_to_first_token_seconds: Fail-fast bound on the wait for a
            provider's first TOKEN per attempt: the first semantic event
            (content, reasoning, a tool call), absolute from the dial, with
            the same input-scaled allowance added. Response headers, SSE
            keepalive comments and role-only frames do not satisfy it (a
            lane that sends those at once and then stalls for minutes is
            exactly the case it exists for), and a thinking model on a chat
            wire streams nothing until its first content token, so the
            default is two minutes rather than the header bound's fifteen
            seconds (clamped to three quarters of ``request_timeout_seconds``
            so the stall is caught while the request can still fail over).
            It never caps total generation time: once the attempt
            commits, reads are paced by the deployment's own per-chunk
            timeout. A stall fails over to the next rung. Deployments may
            override the flat bound through their gateway capabilities.
        pre_first_byte_budget_seconds: Bound on everything before the
            caller's first byte, across every attempt and failover, from the
            request's start. Kept under an edge proxy's origin timeout
            (Cloudflare's is 100 s) so a chain of slow dials ends in the
            gateway's own retryable 503 instead of the proxy's 504 page. Zero
            disables it. A request whose route AUTHORS a longer first-byte
            allowance gets that allowance plus a reserve instead (15 s, or
            ``EXP_GATEWAY_PRE_FIRST_BYTE_RESERVE_SECONDS``), so an authored
            self-hosted lane is never cut at the default.
        public_upstreams_only: Restrict native provider connections to public HTTPS
            addresses, including connection-time DNS checks and proxy bypass.
            Hosted embedders accepting customer endpoints must enable this. It
            defaults to False for local/private model servers and does not cover
            Python control-plane callbacks that make their own HTTP requests.
        native_usage_enabled: Whether Rust owns ``/usage.json``. Hosted,
            multi-tenant callers should disable it so their own surface owns
            usage.
        maximum_retained_output_bytes: Worker-wide bound on provider output
            retained across all in-flight streams (tool arguments, buffered
            Messages blocks, non-streaming aggregations, Responses
            continuations). A stream that would cross it fails with a
            retryable 503 instead of growing worker memory; a continuation
            that would cross it is simply not retained. None (the default)
            leaves the budget off, which is the behavior before this
            setting existed; 1 GiB (a quarter of a 4 GiB worker) is the
            suggested value when enabling it.
        capture: The collector shared with the admission controller; None retains no content.
        callback_scopes: In-process registry populated by the trusted host front
            with each ephemeral relay token and its verified organization before
            native authentication. Caller headers cannot register a scope.
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
    if not 1 <= callback_permits <= 256:
        raise ValueError("callback_permits must be between 1 and 256")
    if (
        max_active_per_organization is not None
        and not 1 <= max_active_per_organization <= max_active_requests
    ):
        raise ValueError("max_active_per_organization must be between 1 and max_active_requests")
    if max_active_per_organization is not None and callback_scopes is None:
        raise ValueError("max_active_per_organization requires a trusted callback_scopes registry")
    try:
        native = importlib.import_module("exp_gateway_native")
    except ModuleNotFoundError as exc:
        raise NativeGatewayServerError("the exp_gateway_native extension is not installed") from exc

    config: dict[str, object] = {
        "host": host,
        "port": port,
        "max_active_requests": max_active_requests,
        "max_active_per_organization": max_active_per_organization,
        "callback_permits": callback_permits,
        "request_timeout_seconds": control_plane.request_timeout_seconds,
        "graceful_timeout_seconds": graceful_timeout_seconds,
        "connect_timeout_seconds": connect_timeout_seconds,
        "public_upstreams_only": public_upstreams_only,
        "time_to_first_byte_seconds": time_to_first_byte_seconds,
        "time_to_first_byte_seconds_per_million_input_tokens": (
            time_to_first_byte_seconds_per_million_input_tokens
        ),
        "time_to_first_token_seconds": time_to_first_token_seconds,
        "pre_first_byte_budget_seconds": pre_first_byte_budget_seconds,
        "native_usage_enabled": native_usage_enabled,
    }
    if maximum_retained_output_bytes is not None:
        config["maximum_retained_output_bytes"] = maximum_retained_output_bytes
    ceiling = pre_first_byte_max_seconds()
    if ceiling is not None:
        config["pre_first_byte_max_seconds"] = ceiling
    reserve = pre_first_byte_reserve_seconds()
    if reserve is not None:
        config["pre_first_byte_reserve_seconds"] = reserve
    try:
        native.serve(
            cast("NativeControlPlane", control_plane),
            json.dumps(config),
            shutdown,
            on_listening,
            control_plane.guardrail_detectors,
            capture,
            callback_scopes,
        )
    except KeyboardInterrupt:
        # The native server drains on SIGINT before returning control to Python.
        pass
    except RuntimeError as exc:
        raise NativeGatewayServerError(f"the native gateway failed: {exc}") from exc

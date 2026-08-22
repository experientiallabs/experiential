"""Embeddable gateway application composition and owned worker lifecycle."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum

from fastapi import FastAPI, Header, Response
from fastapi.responses import HTMLResponse, JSONResponse

from exp.runtime.gateway.contracts import ExecutionSnapshot
from exp.runtime.gateway.execution import GatewayExecutor
from exp.runtime.gateway.interfaces import AttemptLedger, GatewayClock, GatewayControlStore
from exp.runtime.gateway.routing import CatalogRouteResolver
from exp.runtime.gateway.service import (
    GatewayService,
    _bearer_key,
    _exception_response,
    create_gateway_app,
)
from exp.runtime.gateway.usage import GatewayUsageReport, usage_html
from exp.runtime.openai_protocol.state import ResponseContinuationStore, ResponseReplayStore

GatewayReadinessProbe = Callable[[], Awaitable[ExecutionSnapshot]]
GatewayUsageSupplier = Callable[[str | None], GatewayUsageReport]
GatewayTerminalFlusher = Callable[[], Awaitable[None]]


class _GatewayLifecyclePhase(StrEnum):
    """Process-local admission and shutdown phases for one owned runtime."""

    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"


@dataclass(frozen=True)
class GatewayRuntimeConfig:
    """Finite worker-owned gateway timing and application metadata."""

    graceful_timeout_seconds: float
    request_timeout_seconds: float = 120
    title: str = "EXP gateway"

    def __post_init__(self) -> None:
        """Reject timing bounds that cannot provide a finite worker lifecycle."""
        if not math.isfinite(self.graceful_timeout_seconds) or self.graceful_timeout_seconds <= 0:
            raise ValueError("graceful_timeout_seconds must be finite and positive")
        if not math.isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be finite and positive")
        if not self.title:
            raise ValueError("title must not be empty")


@dataclass
class GatewayLifecycleState:
    """Track whether an owned application accepts readiness traffic."""

    phase: _GatewayLifecyclePhase = _GatewayLifecyclePhase.STARTING

    @property
    def ready(self) -> bool:
        """Return whether the runtime most recently passed preflight and still admits work."""
        return self.phase is _GatewayLifecyclePhase.READY


@dataclass
class GatewayRuntime:
    """Own one injected gateway service, ASGI application, and bounded lifecycle."""

    service: GatewayService
    state: GatewayLifecycleState
    config: GatewayRuntimeConfig
    app: FastAPI = field(init=False)
    _drain_task: asyncio.Task[bool] | None = field(default=None, init=False, repr=False)

    async def preflight(self) -> ExecutionSnapshot:
        """Prove one route ready and expose readiness only after proof succeeds."""
        try:
            proof = await self.service.preflight()
        except Exception:
            if self.state.phase is _GatewayLifecyclePhase.READY:
                self.state.phase = _GatewayLifecyclePhase.STARTING
            raise
        if self.state.phase not in {
            _GatewayLifecyclePhase.DRAINING,
            _GatewayLifecyclePhase.STOPPED,
        }:
            self.state.phase = _GatewayLifecyclePhase.READY
        return proof

    async def readiness(self) -> bool:
        """Re-probe recoverably unless this runtime has begun fail-closed shutdown."""
        if self.state.phase in {
            _GatewayLifecyclePhase.DRAINING,
            _GatewayLifecyclePhase.STOPPED,
        }:
            return False
        try:
            await self.preflight()
        except Exception:  # noqa: BLE001 - readiness converts all failures to not-ready.
            return False
        return self.state.phase is _GatewayLifecyclePhase.READY

    async def drain(self, *, timeout_seconds: float | None = None) -> bool:
        """Stop admission once and drain owned work within an explicit finite bound."""
        bound = self.config.graceful_timeout_seconds if timeout_seconds is None else timeout_seconds
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        task = self._drain_task
        if task is None:
            self.state.phase = _GatewayLifecyclePhase.DRAINING
            task = asyncio.create_task(self._drain_once(bound))
            self._drain_task = task
        if task.done():
            return task.result()
        return await asyncio.shield(task)

    async def shutdown(self) -> bool:
        """Drain this worker using its configured graceful shutdown bound."""
        return await self.drain()

    async def _drain_once(self, timeout_seconds: float) -> bool:
        """Run the sole service drain and permanently close lifecycle admission."""
        try:
            return await self.service.drain(timeout_seconds=timeout_seconds)
        finally:
            self.state.phase = _GatewayLifecyclePhase.STOPPED


def create_gateway_runtime(
    *,
    config: GatewayRuntimeConfig,
    authority: GatewayControlStore,
    ledger: AttemptLedger,
    routes: CatalogRouteResolver,
    executor: GatewayExecutor,
    clock: GatewayClock,
    readiness: GatewayReadinessProbe,
    usage: GatewayUsageSupplier,
    replay: ResponseReplayStore | None = None,
    continuations: ResponseContinuationStore | None = None,
    wall_clock: Callable[[], float] | None = None,
    terminal_flusher: GatewayTerminalFlusher | None = None,
) -> GatewayRuntime:
    """Compose one hosted-safe gateway runtime entirely from injected dependencies.

    Callers own provider construction, secret resolution, catalog snapshots, project selection,
    authority, accounting, and durable flushing. This factory performs no filesystem, environment,
    database, process, or server access.

    Args:
        config: Finite request and shutdown bounds plus application metadata.
        authority: Injected key, grant, alias, and revision authority.
        ledger: Injected content-free request and attempt accounting.
        routes: Injected direct and project route resolver.
        executor: Injected bounded provider executor.
        clock: Injected authority and deadline clock.
        readiness: Credential-free proof for one granted executable route.
        usage: Content-free usage report supplier for the public usage routes.
            It receives the caller's optional Bearer key: ``None`` returns the
            organization-wide report, and a virtual key returns the report
            scoped to that key's identity.
        replay: Optional bounded Chat and Responses replay state.
        continuations: Optional bounded Responses continuation state.
        wall_clock: Optional epoch clock for public OpenAI object timestamps.
        terminal_flusher: Optional bounded durable-accounting flush hook.

    Returns:
        An explicit owned lifecycle handle containing the composed FastAPI application.
    """
    service = GatewayService(
        control_store=authority,
        ledger=ledger,
        routes=routes,
        executor=executor,
        clock=clock,
        readiness_probe=readiness,
        request_timeout_seconds=config.request_timeout_seconds,
        replay_store=replay,
        continuation_store=continuations,
        wall_clock=wall_clock or time.time,
        terminal_flusher=terminal_flusher,
    )
    state = GatewayLifecycleState()
    runtime = GatewayRuntime(
        service=service,
        state=state,
        config=config,
    )
    runtime.app = _create_managed_app(runtime, usage=usage)
    return runtime


def _create_managed_app(
    runtime: GatewayRuntime,
    *,
    usage: GatewayUsageSupplier,
) -> FastAPI:
    """Wrap the single OpenAI data plane with worker health and usage routes."""

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        """Own readiness and bounded shutdown for one ASGI application lifetime."""
        await runtime.preflight()
        try:
            yield
        finally:
            await runtime.shutdown()

    application = FastAPI(title=runtime.config.title, lifespan=lifespan)

    @application.get("/health/live")
    async def health_live() -> JSONResponse:
        """Return liveness after an ASGI listener can reach this worker."""
        return JSONResponse({"status": "live"})

    @application.get("/health/ready")
    async def health_ready() -> JSONResponse:
        """Return readiness only while service preflight remains healthy."""
        ready = await runtime.readiness()
        status = 200 if ready else 503
        return JSONResponse({"status": "ready" if ready else "not_ready"}, status_code=status)

    @application.get("/usage.json")
    async def usage_json(authorization: str | None = Header(default=None)) -> Response:
        """Return versioned content-free usage, scoped when a key is presented."""
        try:
            report = usage(_usage_scope_key(authorization))
        except Exception as exc:  # noqa: BLE001 - HTTP boundary sanitizes every failure.
            return _exception_response(exc)
        return JSONResponse(report.model_dump(mode="json"))

    @application.get("/usage")
    async def usage_page(authorization: str | None = Header(default=None)) -> Response:
        """Return the minimal content-free usage page, scoped when a key is presented."""
        try:
            report = usage(_usage_scope_key(authorization))
        except Exception as exc:  # noqa: BLE001 - HTTP boundary sanitizes every failure.
            return _exception_response(exc)
        return HTMLResponse(usage_html(report))

    application.mount("/", create_gateway_app(runtime.service))
    return application


def _usage_scope_key(authorization: str | None) -> str | None:
    """Return the optional Bearer credential scoping one usage request.

    Args:
        authorization: Raw Authorization header value, if any.

    Returns:
        The presented virtual key, or ``None`` for the anonymous
        organization-wide report.

    Raises:
        OpenAIProtocolError: A present header does not carry a Bearer key.
    """
    if authorization is None:
        return None
    return _bearer_key(authorization)

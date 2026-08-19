"""Embeddable gateway application composition and owned worker lifecycle."""

from __future__ import annotations

import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from wmo.runtime.gateway.contracts import ExecutionSnapshot
from wmo.runtime.gateway.execution import GatewayExecutor
from wmo.runtime.gateway.interfaces import AttemptLedger, GatewayClock, GatewayControlStore
from wmo.runtime.gateway.routing import CatalogRouteResolver
from wmo.runtime.gateway.service import GatewayService, create_gateway_app
from wmo.runtime.gateway.usage import GatewayUsageReport, usage_html
from wmo.runtime.openai_protocol.state import BoundedContinuationStore, BoundedReplayStore

GatewayReadinessProbe = Callable[[], Awaitable[ExecutionSnapshot]]
GatewayUsageSupplier = Callable[[], GatewayUsageReport]
GatewayTerminalFlusher = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class GatewayRuntimeConfig:
    """Finite worker-owned gateway timing and application metadata."""

    graceful_timeout_seconds: float
    request_timeout_seconds: float = 120
    title: str = "WMO gateway"

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

    ready: bool = False


@dataclass
class GatewayRuntime:
    """Own one injected gateway service, ASGI application, and bounded lifecycle."""

    service: GatewayService
    state: GatewayLifecycleState
    config: GatewayRuntimeConfig
    app: FastAPI = field(init=False)

    async def preflight(self) -> ExecutionSnapshot:
        """Prove one route ready and expose readiness only after proof succeeds."""
        proof = await self.service.preflight()
        self.state.ready = True
        return proof

    async def readiness(self) -> bool:
        """Return current readiness after rechecking process-local service health."""
        if not self.state.ready:
            return False
        try:
            await self.service.preflight()
        except Exception:  # noqa: BLE001 - readiness converts all failures to not-ready.
            self.state.ready = False
        return self.state.ready

    async def drain(self, *, timeout_seconds: float | None = None) -> bool:
        """Stop admission and drain owned work within an explicit finite bound."""
        bound = self.config.graceful_timeout_seconds if timeout_seconds is None else timeout_seconds
        self.state.ready = False
        return await self.service.drain(timeout_seconds=bound)

    async def shutdown(self) -> bool:
        """Drain this worker using its configured graceful shutdown bound."""
        return await self.drain()


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
    replay: BoundedReplayStore | None = None,
    continuations: BoundedContinuationStore | None = None,
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
    async def usage_json() -> JSONResponse:
        """Return versioned content-free usage from the injected supplier."""
        report = usage()
        return JSONResponse(report.model_dump(mode="json"))

    @application.get("/usage")
    async def usage_page() -> HTMLResponse:
        """Return the minimal content-free usage page."""
        return HTMLResponse(usage_html(usage()))

    application.mount("/", create_gateway_app(runtime.service))
    return application

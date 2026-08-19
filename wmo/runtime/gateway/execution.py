"""Execute one frozen exact-model route through bounded physical provider attempts."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from wmo.common.core.artifacts import stable_id
from wmo.common.models.gateway_catalog import ExactModelDeployment
from wmo.runtime.gateway.budgets import (
    BudgetReservationRejected,
    BudgetScopeKind,
    maximum_attempt_cost_micro_usd,
)
from wmo.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
    GatewayUsage,
)
from wmo.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry
from wmo.runtime.gateway.interfaces import AttemptLedger, ProviderStream
from wmo.runtime.gateway.routing import GatewayRoute
from wmo.runtime.models import ResolvedModel, RuntimeModelCatalog
from wmo.runtime.models.providers import (
    AsyncGatewayProvider,
    RequestDeadline,
    preflight_gateway_request,
    require_gateway_provider,
)
from wmo.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from wmo.runtime.models.providers.errors import normalized_provider_failure
from wmo.runtime.models.providers.transport import RetryPolicy

_SINGLE_DISPATCH = RetryPolicy(
    maximum_attempts=1,
    initial_delay_seconds=0,
    maximum_delay_seconds=0,
)
_SEMANTIC_EVENTS = {
    GatewayEventKind.TEXT_DELTA,
    GatewayEventKind.REFUSAL_DELTA,
    GatewayEventKind.TOOL_CALL_STARTED,
    GatewayEventKind.TOOL_ARGUMENTS_DELTA,
    GatewayEventKind.TOOL_CALL_COMPLETED,
}
_TERMINAL_EVENTS = {
    GatewayEventKind.COMPLETED,
    GatewayEventKind.INCOMPLETE,
    GatewayEventKind.FAILED,
}
_MAX_WITHHELD_REFUSAL_BYTES = 65_536
_MAX_WITHHELD_REFUSAL_EVENTS = 256


class GatewayExecutionError(RuntimeError):
    """A route failed before a normalized provider stream could be returned."""

    def __init__(self, failure: GatewayFailure, *, request_finalized: bool = False) -> None:
        """Retain one sanitized failure for the public protocol boundary."""
        super().__init__(failure.safe_message)
        self.failure = failure
        self.request_finalized = request_finalized


@dataclass(frozen=True)
class _ResolvedDeployment:
    """One verified provider binding for a frozen deployment."""

    deployment: ExactModelDeployment
    provider: AsyncGatewayProvider
    health_key: DeploymentHealthKey
    idempotency_key: str


@dataclass
class _PhysicalAttempt:
    """Mutable stream and usage state for one durably recorded dispatch."""

    route_index: int
    attempt_id: str
    stream: ProviderStream | None = None
    iterator: AsyncIterator[GatewayEvent] | None = None
    latest_usage: GatewayUsage | None = None
    last_provider_sequence: int = -1
    withheld_refusals: list[GatewayEvent] = field(default_factory=list)
    withheld_refusal_bytes: int = 0
    visible_refusal: bool = False
    settled: bool = False


class GatewayExecutionStream:
    """Expose one logical stream while accounting every precommit physical attempt."""

    def __init__(
        self,
        *,
        route: GatewayRoute,
        request: GatewayRequest,
        deadline: RequestDeadline,
        resolved: tuple[_ResolvedDeployment, ...],
        ledger: AttemptLedger,
        health: DeploymentHealthRegistry,
        release: Callable[[], None],
        accounting_failure: Callable[[], None],
        maximum_total_attempts: int,
        maximum_same_deployment_attempts: int,
        refusal_failover: bool,
    ) -> None:
        """Bind one frozen logical route to its finite physical-attempt policy.

        Args:
            route: Ordered certified deployment route.
            request: Canonical request forced into provider streaming mode.
            deadline: One request-wide absolute deadline.
            resolved: Verified runtime providers in route order.
            ledger: Content-free request and attempt ledger.
            health: Revision-isolated circuit and throttle registry.
            release: Callback releasing one logical execution permit.
            accounting_failure: Callback latching durable accounting failures.
            maximum_total_attempts: Hard cap across retries and deployments.
            maximum_same_deployment_attempts: Initial dispatch plus safe retries per deployment.
            refusal_failover: Whether a typed precommit refusal may advance to another deployment.
        """
        self._route = route
        self._request = request
        self._deadline = deadline
        self._resolved = resolved
        self._ledger = ledger
        self._health = health
        self._release = release
        self._accounting_failure = accounting_failure
        self._maximum_total_attempts = maximum_total_attempts
        self._maximum_same_deployment_attempts = maximum_same_deployment_attempts
        self._refusal_failover = refusal_failover
        self._attempt_counts = [0 for _ in resolved]
        self._total_attempts = 0
        self._current: _PhysicalAttempt | None = None
        self._committed = False
        self._settled = False
        self._parent_finalized = False
        self._next_sequence = 0
        self._pending_outward: deque[GatewayEvent] = deque()
        self._settlement_lock = asyncio.Lock()

    @property
    def deployment(self) -> ExactModelDeployment:
        """Return the active or terminal physical deployment."""
        current = self._require_current()
        return self._resolved[current.route_index].deployment

    @property
    def route_depth(self) -> int:
        """Return the zero-based deployment position serving the outward result."""
        return self._require_current().route_index

    async def open(self) -> None:
        """Open the first healthy attempt and terminalize final opening exhaustion."""
        candidate = self._initial_candidate()
        if candidate is None:
            raise GatewayExecutionError(_all_routes_unavailable())
        failure = await self._open_from(candidate, finalize_on_exhaustion=True)
        if failure is not None:
            raise GatewayExecutionError(failure, request_finalized=self._parent_finalized)

    def __aiter__(self) -> AsyncIterator[GatewayEvent]:
        """Return this one-pass logical event iterator."""
        return self

    async def __anext__(self) -> GatewayEvent:
        """Yield one logical event, advancing routes only before semantic commitment."""
        if self._pending_outward:
            return self._outward(self._pending_outward.popleft())
        if self._settled:
            raise StopAsyncIteration
        while True:
            current = self._require_current()
            iterator = self._require_iterator(current)
            try:
                event = await iterator.__anext__()
            except StopAsyncIteration:
                if self._settled:
                    raise
                failure = GatewayFailure(
                    failure_class=GatewayFailureClass.MALFORMED_RESPONSE,
                    safe_message="provider stream ended without a terminal event",
                    retryable_same_deployment=True,
                    failover_eligible=True,
                )
                event = GatewayEvent(
                    kind=GatewayEventKind.FAILED,
                    sequence_number=current.last_provider_sequence + 1,
                    failure=failure,
                )
            except asyncio.CancelledError:
                await self.abort(
                    GatewayFailure(
                        failure_class=GatewayFailureClass.CANCELLED,
                        safe_message="provider request was cancelled",
                    )
                )
                raise
            except BaseException as exc:  # noqa: BLE001 - provider taxonomy owns conversion.
                failure = normalized_provider_failure(exc)
                event = GatewayEvent(
                    kind=GatewayEventKind.FAILED,
                    sequence_number=current.last_provider_sequence + 1,
                    failure=failure,
                )
            if self._settled:
                raise StopAsyncIteration
            current.last_provider_sequence = event.sequence_number
            if event.usage is not None:
                current.latest_usage = event.usage
            if (
                event.kind == GatewayEventKind.REFUSAL_DELTA
                and self._refusal_failover
                and not self._committed
            ):
                event_bytes = len((event.text_delta or "").encode("utf-8"))
                if (
                    current.withheld_refusal_bytes + event_bytes > _MAX_WITHHELD_REFUSAL_BYTES
                    or len(current.withheld_refusals) + 1 > _MAX_WITHHELD_REFUSAL_EVENTS
                ):
                    self._commit_withheld_refusal(current, event)
                    return self._outward(self._pending_outward.popleft())
                current.withheld_refusals.append(event)
                current.withheld_refusal_bytes += event_bytes
                continue
            if current.withheld_refusals and event.kind in _SEMANTIC_EVENTS:
                self._commit_withheld_refusal(current, event)
                return self._outward(self._pending_outward.popleft())
            if event.kind in _SEMANTIC_EVENTS:
                if event.kind == GatewayEventKind.REFUSAL_DELTA:
                    current.visible_refusal = True
                self._committed = True
                return self._outward(event)
            if event.kind not in _TERMINAL_EVENTS:
                if self._committed:
                    return self._outward(event)
                continue
            terminal = _with_latest_usage(event, current.latest_usage)
            withheld_non_refusal_failure = False
            typed_refusal = (
                terminal.kind == GatewayEventKind.FAILED
                and terminal.failure is not None
                and terminal.failure.failure_class == GatewayFailureClass.REFUSAL
            )
            if current.withheld_refusals and terminal.kind != GatewayEventKind.FAILED:
                terminal = GatewayEvent(
                    kind=GatewayEventKind.FAILED,
                    sequence_number=terminal.sequence_number,
                    failure=GatewayFailure(
                        failure_class=GatewayFailureClass.REFUSAL,
                        safe_message="provider refused the request",
                        safe_details={"signal": "provider_refusal"},
                    ),
                    usage=terminal.usage,
                )
                typed_refusal = True
            elif current.withheld_refusals and not typed_refusal:
                withheld_non_refusal_failure = True
            if terminal.kind != GatewayEventKind.FAILED:
                self._health.succeeded(self._health_key(current.route_index))
                await self._finish_current(terminal=terminal, failure=None, finalize_request=True)
                self._settle()
                return self._outward(terminal)
            failure = terminal.failure or GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="provider execution failed",
            )
            self._health.failed(self._health_key(current.route_index), failure)
            candidate = None if self._committed else self._next_candidate(failure)
            if candidate is None:
                await self._finish_current(
                    terminal=terminal,
                    failure=failure,
                    finalize_request=True,
                )
                self._settle()
                if withheld_non_refusal_failure:
                    self._commit_withheld_refusal(current, terminal)
                    return self._outward(self._pending_outward.popleft())
                if typed_refusal and current.withheld_refusals:
                    self._commit_withheld_refusal(
                        current,
                        GatewayEvent(
                            kind=GatewayEventKind.COMPLETED,
                            sequence_number=terminal.sequence_number,
                            usage=terminal.usage,
                        ),
                    )
                    return self._outward(self._pending_outward.popleft())
                if typed_refusal and current.visible_refusal:
                    return self._outward(
                        GatewayEvent(
                            kind=GatewayEventKind.COMPLETED,
                            sequence_number=terminal.sequence_number,
                            usage=terminal.usage,
                        )
                    )
                return self._outward(terminal)
            if current.withheld_refusals:
                current.withheld_refusals.clear()
                current.withheld_refusal_bytes = 0
            await self._finish_current(
                terminal=terminal,
                failure=failure,
                finalize_request=False,
            )
            try:
                opening_failure = await self._open_from(
                    candidate,
                    finalize_on_exhaustion=True,
                )
            except asyncio.CancelledError:
                await self._finish_parent(
                    GatewayFailure(
                        failure_class=GatewayFailureClass.CANCELLED,
                        safe_message="provider request was cancelled",
                    )
                )
                self._settle()
                raise
            except GatewayExecutionError as exc:
                await self._finish_parent(exc.failure)
                self._settle()
                return self._outward(
                    GatewayEvent(
                        kind=GatewayEventKind.FAILED,
                        sequence_number=0,
                        failure=exc.failure,
                    )
                )
            if opening_failure is not None:
                self._settle()
                return self._outward(
                    GatewayEvent(
                        kind=GatewayEventKind.FAILED,
                        sequence_number=0,
                        failure=opening_failure,
                    )
                )

    async def cancel(self) -> None:
        """Cancel active upstream work and durably retain observed usage if available."""
        await self.abort(
            GatewayFailure(
                failure_class=GatewayFailureClass.CANCELLED,
                safe_message="provider request was cancelled",
            )
        )

    async def abort(self, failure: GatewayFailure) -> None:
        """Claim terminal ownership before bounded upstream cancellation.

        Args:
            failure: Sanitized reason the owning gateway stopped this logical request.
        """
        current = self._current
        if current is None:
            self._settle()
            return
        accounting_error: BaseException | None = None
        owns_cancel = False
        async with self._settlement_lock:
            if self._settled or current.settled:
                return
            owns_cancel = True
            terminal = GatewayEvent(
                kind=GatewayEventKind.FAILED,
                sequence_number=current.last_provider_sequence + 1,
                failure=failure,
                usage=current.latest_usage,
            )
            try:
                self._ledger.finish_attempt(
                    attempt_id=current.attempt_id,
                    terminal_event=terminal,
                    failure=failure,
                    finalize_request=True,
                )
                current.settled = True
                self._parent_finalized = True
            except Exception as exc:  # noqa: BLE001 - latch any durable ledger failure.
                accounting_error = exc
                self._accounting_failure()
            finally:
                self._settle()
        if owns_cancel and current.stream is not None:
            await current.stream.cancel()
        if accounting_error is not None:
            raise accounting_error

    async def _open_from(
        self,
        candidate: int,
        *,
        finalize_on_exhaustion: bool,
    ) -> GatewayFailure | None:
        """Open the next physical attempt, consuming safe failures within the total cap."""
        current_candidate: int | None = candidate
        while current_candidate is not None:
            try:
                await self._dispatch(current_candidate)
                return None
            except asyncio.CancelledError:
                raise
            except _BudgetRouteSkipped as exc:
                if exc.scope_kind is not BudgetScopeKind.DEPLOYMENT:
                    failure = _budget_quota_failure()
                    if finalize_on_exhaustion:
                        await self._finish_parent(failure)
                    return failure
                next_candidate = self._next_budget_candidate(current_candidate)
                if next_candidate is None:
                    failure = (
                        _budget_quota_failure()
                        if current_candidate == len(self._resolved) - 1
                        else _all_routes_unavailable()
                    )
                    if finalize_on_exhaustion:
                        await self._finish_parent(failure)
                    return failure
                current_candidate = next_candidate
            except _DispatchFailure as exc:
                failure = exc.failure
                self._health.failed(self._health_key(current_candidate), failure)
                next_candidate = self._next_candidate(failure, current_candidate)
                finalize = finalize_on_exhaustion and next_candidate is None
                await self._finish_open_failure(exc, finalize_request=finalize)
                if exc.cancelled:
                    raise asyncio.CancelledError from exc
                if next_candidate is None:
                    return failure
                current_candidate = next_candidate
        return _all_routes_unavailable()

    async def _dispatch(self, route_index: int) -> _PhysicalAttempt:
        """Persist and open exactly one physical provider dispatch."""
        binding = self._resolved[route_index]
        attempt_id: str | None = None
        try:
            self._deadline.attempt_timeout()
            attempt_id = self._ledger.start_attempt(
                snapshot=self._route.snapshot,
                deployment=binding.deployment,
                attempt_ordinal=self._total_attempts,
                route_depth=route_index,
                maximum_cost_micro_usd=maximum_attempt_cost_micro_usd(
                    self._request,
                    binding.deployment,
                ),
            )
            self._attempt_counts[route_index] += 1
            self._total_attempts += 1
            self._ledger.record_route_context(
                attempt_id=attempt_id,
                route_reason=self._route.route_reason,
                fallback_reason=self._route.fallback_reason,
            )
            current = _PhysicalAttempt(
                route_index=route_index,
                attempt_id=attempt_id,
            )
            self._current = current
            stream = await binding.provider.stream(
                self._request,
                deadline=self._deadline,
                idempotency_key=binding.idempotency_key,
                retry_policy=_SINGLE_DISPATCH,
            )
            if self._settled or current.settled:
                await stream.cancel()
                raise asyncio.CancelledError
            current.stream = stream
            current.iterator = stream.__aiter__()
        except BudgetReservationRejected as exc:
            self._health.release_probe(binding.health_key)
            raise _BudgetRouteSkipped(exc.scope_kind) from exc
        except BaseException as exc:
            if attempt_id is None:
                self._health.release_probe(binding.health_key)
                if isinstance(exc, ProviderDeadlineExceeded):
                    raise GatewayExecutionError(normalized_provider_failure(exc)) from exc
                if not isinstance(exc, asyncio.CancelledError):
                    self._accounting_failure()
                    raise GatewayExecutionError(normalized_provider_failure(exc)) from exc
                raise
            failure = normalized_provider_failure(exc)
            raise _DispatchFailure(
                attempt_id,
                failure,
                cancelled=isinstance(exc, asyncio.CancelledError),
            ) from exc
        return current

    async def _finish_open_failure(
        self,
        dispatch_failure: _DispatchFailure,
        *,
        finalize_request: bool,
    ) -> None:
        """Settle one provider opening failure before another physical dispatch."""
        try:
            self._ledger.finish_attempt(
                attempt_id=dispatch_failure.attempt_id,
                terminal_event=None,
                failure=dispatch_failure.failure,
                finalize_request=finalize_request,
            )
            current = self._current
            if current is not None and current.attempt_id == dispatch_failure.attempt_id:
                current.settled = True
            if finalize_request:
                self._parent_finalized = True
        except Exception:  # noqa: BLE001 - preserve the primary provider failure.
            self._accounting_failure()
            raise

    async def _finish_current(
        self,
        *,
        terminal: GatewayEvent,
        failure: GatewayFailure | None,
        finalize_request: bool,
    ) -> None:
        """Durably settle the active attempt once under the shared settlement lock."""
        current = self._require_current()
        async with self._settlement_lock:
            if current.settled:
                return
            try:
                self._ledger.finish_attempt(
                    attempt_id=current.attempt_id,
                    terminal_event=terminal,
                    failure=failure,
                    finalize_request=finalize_request,
                )
                current.settled = True
                if finalize_request:
                    self._parent_finalized = True
            except BaseException:
                self._accounting_failure()
                raise

    def _initial_candidate(self) -> int | None:
        """Claim the first currently healthy deployment in authored order."""
        for route_index in range(len(self._resolved)):
            if self._health.claim(self._health_key(route_index)):
                return route_index
        return None

    def _next_candidate(
        self,
        failure: GatewayFailure,
        current: int | None = None,
    ) -> int | None:
        """Choose a safe retry or later exact deployment without changing logical model."""
        if current is None:
            current = self._require_current_index_for_failure()
        if self._total_attempts >= self._maximum_total_attempts:
            return None
        if (
            failure.retryable_same_deployment
            and self._attempt_counts[current] < self._maximum_same_deployment_attempts
            and self._health.claim(self._health_key(current))
        ):
            return current
        refusal_eligible = (
            failure.failure_class == GatewayFailureClass.REFUSAL and self._refusal_failover
        )
        if not failure.failover_eligible and not refusal_eligible:
            return None
        for route_index in range(current + 1, len(self._resolved)):
            if self._health.claim(self._health_key(route_index)):
                return route_index
        return None

    def _next_budget_candidate(self, current: int) -> int | None:
        """Advance past a route whose hard monthly allocation cannot admit this call."""
        for route_index in range(current + 1, len(self._resolved)):
            if self._health.claim(self._health_key(route_index)):
                return route_index
        return None

    def _require_current_index_for_failure(self) -> int:
        """Return the last dispatched route index, including failed opening attempts."""
        if self._current is not None and not self._current.settled:
            return self._current.route_index
        for route_index in range(len(self._attempt_counts) - 1, -1, -1):
            if self._attempt_counts[route_index] > 0:
                return route_index
        raise RuntimeError("waterfall failure has no physical attempt")

    def _health_key(self, route_index: int) -> DeploymentHealthKey:
        """Return the revision-isolated health key for one ordered deployment."""
        return self._resolved[route_index].health_key

    def _outward(self, event: GatewayEvent) -> GatewayEvent:
        """Rewrite provider-local sequence numbers into one monotonic public sequence."""
        outward = event.model_copy(update={"sequence_number": self._next_sequence})
        self._next_sequence += 1
        return outward

    def _commit_withheld_refusal(
        self,
        current: _PhysicalAttempt,
        *following: GatewayEvent,
    ) -> None:
        """Flush bounded refusal output and permanently freeze the active route.

        Args:
            current: Physical attempt holding in-memory refusal deltas.
            *following: Semantic or terminal events that arrived after the refusal.
        """
        self._committed = True
        current.visible_refusal = True
        self._pending_outward.extend(current.withheld_refusals)
        self._pending_outward.extend(following)
        current.withheld_refusals.clear()
        current.withheld_refusal_bytes = 0

    def _require_current(self) -> _PhysicalAttempt:
        """Return the active or terminal attempt after successful stream opening."""
        if self._current is None:
            raise RuntimeError("gateway execution stream has no physical attempt")
        return self._current

    @staticmethod
    def _require_iterator(current: _PhysicalAttempt) -> AsyncIterator[GatewayEvent]:
        """Return the opened iterator for one active physical attempt."""
        if current.iterator is None:
            raise RuntimeError("gateway physical attempt is not open")
        return current.iterator

    async def _finish_parent(self, failure: GatewayFailure) -> None:
        """Terminalize a parent whose next attempt could not be durably started."""
        if self._parent_finalized:
            return
        try:
            self._ledger.finish_request(
                authorization=self._route.snapshot.authorization,
                failure=failure,
            )
            self._parent_finalized = True
        except Exception:
            self._accounting_failure()
            raise

    def _settle(self) -> None:
        """Release logical execution admission exactly once."""
        if self._settled:
            return
        self._settled = True
        self._release()


class _DispatchFailure(Exception):
    """One durably started attempt failed before returning its provider stream."""

    def __init__(
        self,
        attempt_id: str,
        failure: GatewayFailure,
        *,
        cancelled: bool,
    ) -> None:
        """Retain the attempt and sanitized opening failure."""
        super().__init__(failure.safe_message)
        self.attempt_id = attempt_id
        self.failure = failure
        self.cancelled = cancelled


class _BudgetRouteSkipped(Exception):
    """A physical route could not reserve beneath its applicable monthly limits."""

    def __init__(self, scope_kind: BudgetScopeKind) -> None:
        """Retain only the content-free scope classification for fallback policy."""
        self.scope_kind = scope_kind
        super().__init__(scope_kind.value)


class GatewayExecutor:
    """Open certified provider waterfalls after preflight, admission, and durable dispatch."""

    def __init__(
        self,
        catalogs: Mapping[tuple[str, str], RuntimeModelCatalog],
        ledger: AttemptLedger,
        *,
        maximum_active_requests: int = 64,
        maximum_total_attempts: int = 8,
        maximum_same_deployment_attempts: int = 2,
        health: DeploymentHealthRegistry | None = None,
    ) -> None:
        """Bind runtime providers, attempt policy, health, and finite request admission.

        Args:
            catalogs: Revision and catalog digests mapped to frozen runtime catalogs.
            ledger: Content-free request and attempt ledger.
            maximum_active_requests: Maximum active logical upstream requests.
            maximum_total_attempts: Hard physical dispatch cap for one logical request.
            maximum_same_deployment_attempts: Initial dispatch plus safe retries per deployment.
            health: Optional shared content-free deployment health registry.
        """
        if maximum_active_requests < 1:
            raise ValueError("maximum_active_requests must be at least one")
        if maximum_total_attempts < 1:
            raise ValueError("maximum_total_attempts must be at least one")
        if maximum_same_deployment_attempts < 1:
            raise ValueError("maximum_same_deployment_attempts must be at least one")
        self._catalogs = dict(catalogs)
        self._ledger = ledger
        self._permits = asyncio.Semaphore(maximum_active_requests)
        self._maximum_total_attempts = maximum_total_attempts
        self._maximum_same_deployment_attempts = maximum_same_deployment_attempts
        self._health = health or DeploymentHealthRegistry()
        self._accounting_healthy = True

    def require_healthy(self) -> None:
        """Fail readiness after any durable terminal accounting write is lost."""
        if not self._accounting_healthy:
            raise GatewayExecutionError(
                GatewayFailure(
                    failure_class=GatewayFailureClass.INTERNAL,
                    safe_message="gateway terminal accounting is unhealthy",
                )
            )

    def mark_accounting_unhealthy(self) -> None:
        """Latch an unhealthy state after a terminal accounting failure."""
        self._accounting_healthy = False

    async def start(
        self,
        *,
        route: GatewayRoute,
        request: GatewayRequest,
    ) -> GatewayExecutionStream:
        """Start one certified exact-model waterfall under the request-wide deadline.

        Args:
            route: Frozen ordered exact-model route.
            request: Canonical public request.

        Returns:
            Accounted logical stream frozen on its first semantic event.

        Raises:
            GatewayExecutionError: Preflight, admission, resolution, or all openings fail.
        """
        deadline = RequestDeadline(route.snapshot.authorization.deadline_monotonic)
        provider_request = request.model_copy(update={"stream": True, "include_usage": True})
        try:
            for deployment in route.deployments:
                require_gateway_provider(deployment.provider)
                preflight_gateway_request(provider_request, deployment.gateway.capabilities)
            resolved = self._resolve_route(route)
            await self._acquire(deadline)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise GatewayExecutionError(normalized_provider_failure(exc)) from exc
        execution = GatewayExecutionStream(
            route=route,
            request=provider_request,
            deadline=deadline,
            resolved=resolved,
            ledger=self._ledger,
            health=self._health,
            release=self._permits.release,
            accounting_failure=self.mark_accounting_unhealthy,
            maximum_total_attempts=self._maximum_total_attempts,
            maximum_same_deployment_attempts=self._maximum_same_deployment_attempts,
            refusal_failover=route.snapshot.authorization.refusal_failover,
        )
        try:
            await execution.open()
        except BaseException:
            execution._settle()  # noqa: SLF001 - executor owns admission until open succeeds.
            raise
        return execution

    def _resolve_route(self, route: GatewayRoute) -> tuple[_ResolvedDeployment, ...]:
        """Resolve and identity-check every deployment before the first billable dispatch."""
        authorization = route.snapshot.authorization
        catalog = self._catalogs.get(
            (authorization.alias_revision_id, authorization.catalog_sha256)
        )
        if catalog is None:
            raise ValueError("runtime catalog is not loaded for the authorized revision")
        resolved: list[_ResolvedDeployment] = []
        for deployment in route.deployments:
            runtime_model = catalog.resolve(deployment.source_alias)
            _require_deployment_identity(deployment, runtime_model)
            if getattr(runtime_model.client, "stream", None) is None:
                raise TypeError("resolved gateway deployment has no async stream capability")
            resolved.append(
                _ResolvedDeployment(
                    deployment=deployment,
                    provider=cast(AsyncGatewayProvider, runtime_model.client),
                    health_key=(
                        authorization.catalog_sha256,
                        deployment.deployment_id,
                        deployment.connection_sha256,
                    ),
                    idempotency_key=_deployment_idempotency_key(route, deployment),
                )
            )
        return tuple(resolved)

    async def _acquire(self, deadline: RequestDeadline) -> None:
        """Wait for logical execution admission within the request-wide deadline."""
        try:
            async with asyncio.timeout(deadline.attempt_timeout()):
                await self._permits.acquire()
        except TimeoutError as exc:
            raise ProviderDeadlineExceeded("gateway execution queue deadline exceeded") from exc


def _require_deployment_identity(
    deployment: ExactModelDeployment,
    resolved: ResolvedModel,
) -> None:
    """Fail before accounting or network work when runtime resolution drifts from authority.

    A catalog record may pin a response-only served-model identity that differs from the
    requested provider model. The frozen deployment names the requested model, so either the
    resolved requested identity or the pinned served identity may match it.
    """
    if (
        resolved.alias != deployment.source_alias
        or resolved.snapshot.provider != deployment.provider
        or deployment.provider_model not in {resolved.snapshot.model_id, resolved.served_model_id}
        or resolved.snapshot.revision != deployment.revision
        or resolved.snapshot.connection_sha256 != deployment.connection_sha256
        or resolved.snapshot.billing_source != deployment.billing_source
        or (
            deployment.capabilities is not None and resolved.capabilities != deployment.capabilities
        )
    ):
        raise ValueError("resolved runtime client differs from the frozen gateway deployment")


def _deployment_idempotency_key(
    route: GatewayRoute,
    deployment: ExactModelDeployment,
) -> str:
    """Derive one stable key reused only by physical retries of this deployment."""
    authorization = route.snapshot.authorization
    return stable_id(
        "gateway-provider-operation",
        {
            "request_id": authorization.request_id,
            "catalog_sha256": authorization.catalog_sha256,
            "deployment_id": deployment.deployment_id,
            "connection_sha256": deployment.connection_sha256,
        },
    )


def _with_latest_usage(
    event: GatewayEvent,
    latest_usage: GatewayUsage | None,
) -> GatewayEvent:
    """Attach the last observed usage to a terminal event that omitted it."""
    if latest_usage is None or event.usage is not None:
        return event
    return event.model_copy(update={"usage": latest_usage})


def _all_routes_unavailable() -> GatewayFailure:
    """Return the sanitized terminal failure for an exhausted certified pool."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
        safe_message="all exact-model deployments are unavailable",
    )


def _budget_quota_failure() -> GatewayFailure:
    """Return the sanitized quota failure after no route can reserve its maximum cost."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
        safe_message="monthly gateway allocation is exhausted",
    )

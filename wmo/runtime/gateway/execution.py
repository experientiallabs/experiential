"""Compose one authorized singleton route with provider execution and accounting."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from typing import cast

from wmo.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
    GatewayUsage,
)
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


class GatewayExecutionError(RuntimeError):
    """A route failed before a normalized provider stream could be returned."""

    def __init__(self, failure: GatewayFailure) -> None:
        """Retain one sanitized failure for the public protocol boundary."""
        super().__init__(failure.safe_message)
        self.failure = failure


class GatewayExecutionStream:
    """Account one normalized provider stream through its single terminal state."""

    def __init__(
        self,
        stream: ProviderStream,
        *,
        ledger: AttemptLedger,
        attempt_id: str,
        release: Callable[[], None],
        accounting_failure: Callable[[], None],
    ) -> None:
        """Bind provider iteration to durable terminal accounting.

        Args:
            stream: Active normalized provider stream.
            ledger: Content-free attempt ledger.
            attempt_id: Durable attempt identity.
            release: Callback releasing one execution admission permit.
            accounting_failure: Callback latching terminal accounting failures.
        """
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._ledger = ledger
        self._attempt_id = attempt_id
        self._release = release
        self._accounting_failure = accounting_failure
        self._latest_usage: GatewayUsage | None = None
        self._last_sequence = -1
        self._settled = False
        self._settlement_lock = asyncio.Lock()

    @property
    def committed(self) -> bool:
        """Return whether semantic provider output has committed this route."""
        return self._stream.committed

    def __aiter__(self) -> AsyncIterator[GatewayEvent]:
        """Return this one-pass accounted event iterator."""
        return self

    async def __anext__(self) -> GatewayEvent:
        """Return the next event after persisting terminal accounting."""
        if self._settled:
            raise StopAsyncIteration
        try:
            event = await self._iterator.__anext__()
        except StopAsyncIteration as exc:
            async with self._settlement_lock:
                if self._settled:
                    raise
                failure = GatewayFailure(
                    failure_class=GatewayFailureClass.MALFORMED_RESPONSE,
                    safe_message="provider stream ended without a terminal event",
                )
                try:
                    self._ledger.finish_attempt(
                        attempt_id=self._attempt_id,
                        terminal_event=None,
                        failure=failure,
                    )
                except BaseException:
                    self._accounting_failure()
                    raise
                finally:
                    self._settle()
                raise GatewayExecutionError(failure) from exc
        self._last_sequence = event.sequence_number
        if event.usage is not None:
            self._latest_usage = event.usage
        if event.kind in {
            GatewayEventKind.COMPLETED,
            GatewayEventKind.INCOMPLETE,
            GatewayEventKind.FAILED,
        }:
            terminal = (
                event
                if self._latest_usage is None or event.usage is not None
                else event.model_copy(update={"usage": self._latest_usage})
            )
            async with self._settlement_lock:
                if self._settled:
                    raise StopAsyncIteration
                try:
                    self._ledger.finish_attempt(
                        attempt_id=self._attempt_id,
                        terminal_event=terminal,
                        failure=event.failure,
                    )
                except BaseException:
                    self._accounting_failure()
                    raise
                finally:
                    self._settle()
                return terminal
        if self._settled:
            raise StopAsyncIteration
        return event

    async def cancel(self) -> None:
        """Cancel upstream work and durably retain observed usage if available."""
        await self.abort(
            GatewayFailure(
                failure_class=GatewayFailureClass.CANCELLED,
                safe_message="provider request was cancelled",
            )
        )

    async def abort(self, failure: GatewayFailure) -> None:
        """Cancel upstream work and settle it with the supplied primary failure.

        Args:
            failure: Sanitized reason the owning gateway stopped this attempt.
        """
        accounting_error: BaseException | None = None
        owns_cancel = False
        async with self._settlement_lock:
            if self._settled:
                return
            owns_cancel = True
            terminal = GatewayEvent(
                kind=GatewayEventKind.FAILED,
                sequence_number=self._last_sequence + 1,
                failure=failure,
                usage=self._latest_usage,
            )
            try:
                self._ledger.finish_attempt(
                    attempt_id=self._attempt_id,
                    terminal_event=terminal,
                    failure=failure,
                )
            except Exception as exc:  # noqa: BLE001 - latch any durable ledger failure
                accounting_error = exc
                self._accounting_failure()
            finally:
                self._settle()
        if owns_cancel:
            await self._stream.cancel()
        if accounting_error is not None:
            raise accounting_error

    def _settle(self) -> None:
        """Release execution admission exactly once."""
        if self._settled:
            return
        self._settled = True
        self._release()


class GatewayExecutor:
    """Open provider streams only after preflight, admission, and durable dispatch."""

    def __init__(
        self,
        catalogs: Mapping[tuple[str, str], RuntimeModelCatalog],
        ledger: AttemptLedger,
        *,
        maximum_active_requests: int = 64,
    ) -> None:
        """Bind runtime providers, ledger, and finite request admission.

        Args:
            catalogs: Revision and catalog digests mapped to frozen runtime catalogs.
            ledger: Content-free request and attempt ledger.
            maximum_active_requests: Maximum active upstream streams.
        """
        if maximum_active_requests < 1:
            raise ValueError("maximum_active_requests must be at least one")
        self._catalogs = dict(catalogs)
        self._ledger = ledger
        self._permits = asyncio.Semaphore(maximum_active_requests)
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
        """Durably start one singleton provider stream under the request deadline.

        Args:
            route: Frozen exact model and deployment route.
            request: Canonical public request.

        Returns:
            Accounted normalized provider stream.

        Raises:
            GatewayExecutionError: Preflight, admission, or provider opening fails.
        """
        deadline = RequestDeadline(route.snapshot.authorization.deadline_monotonic)
        provider_request = request.model_copy(update={"stream": True, "include_usage": True})
        try:
            require_gateway_provider(route.deployment.provider)
            preflight_gateway_request(
                provider_request,
                route.deployment.gateway.capabilities,
            )
            await self._acquire(deadline)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise GatewayExecutionError(normalized_provider_failure(exc)) from exc
        attempt_id: str | None = None
        try:
            authorization = route.snapshot.authorization
            catalog = self._catalogs.get(
                (authorization.alias_revision_id, authorization.catalog_sha256)
            )
            if catalog is None:
                raise ValueError("runtime catalog is not loaded for the authorized revision")
            resolved = catalog.resolve(route.deployment.source_alias)
            _require_deployment_identity(route, resolved)
            stream_method = getattr(resolved.client, "stream", None)
            if stream_method is None:
                raise TypeError("resolved gateway deployment has no async stream capability")
            attempt_id = self._ledger.start_attempt(
                snapshot=route.snapshot,
                deployment=route.deployment,
                route_depth=0,
            )
            self._ledger.record_route_context(
                attempt_id=attempt_id,
                route_reason=route.route_reason,
                fallback_reason=route.fallback_reason,
            )
            provider = cast(AsyncGatewayProvider, resolved.client)
            stream = await provider.stream(
                provider_request,
                deadline=deadline,
                idempotency_key=route.snapshot.authorization.request_id,
            )
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                failure = GatewayFailure(
                    failure_class=GatewayFailureClass.CANCELLED,
                    safe_message="provider request was cancelled",
                )
            else:
                failure = normalized_provider_failure(exc)
            try:
                if attempt_id is not None:
                    self._ledger.finish_attempt(
                        attempt_id=attempt_id,
                        terminal_event=None,
                        failure=failure,
                    )
            except Exception:  # noqa: BLE001 - preserve the primary provider failure
                self.mark_accounting_unhealthy()
            finally:
                self._permits.release()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise GatewayExecutionError(failure) from exc
        return GatewayExecutionStream(
            stream,
            ledger=self._ledger,
            attempt_id=attempt_id,
            release=self._permits.release,
            accounting_failure=self.mark_accounting_unhealthy,
        )

    async def _acquire(self, deadline: RequestDeadline) -> None:
        """Wait for execution admission within the one request-wide deadline."""
        try:
            async with asyncio.timeout(deadline.attempt_timeout()):
                await self._permits.acquire()
        except TimeoutError as exc:
            raise ProviderDeadlineExceeded("gateway execution queue deadline exceeded") from exc


def _require_deployment_identity(route: GatewayRoute, resolved: ResolvedModel) -> None:
    """Fail before accounting or network work when runtime resolution drifts from authority."""
    deployment = route.deployment
    served_model = resolved.served_model_id or resolved.snapshot.model_id
    if (
        resolved.alias != deployment.source_alias
        or resolved.snapshot.provider != deployment.provider
        or served_model != deployment.provider_model
        or resolved.snapshot.revision != deployment.revision
        or resolved.snapshot.connection_sha256 != deployment.connection_sha256
        or (
            deployment.capabilities is not None and resolved.capabilities != deployment.capabilities
        )
    ):
        raise ValueError("resolved runtime client differs from the frozen gateway deployment")

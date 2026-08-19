"""Ordered exact-model routing and isolated deployment-health regressions."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import cast

import pytest

from wmo.common.models import ModelCapabilities, ModelClient, ModelSnapshot
from wmo.common.models.catalog import (
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    GatewayEquivalenceCertification,
)
from wmo.common.models.gateway_catalog import (
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
)
from wmo.runtime.gateway.contracts import (
    AuthorizationSnapshot,
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
    GatewayUsage,
    ProjectSelection,
    ProjectTarget,
)
from wmo.runtime.gateway.execution import GatewayExecutionError, GatewayExecutor
from wmo.runtime.gateway.health import DeploymentHealthRegistry
from wmo.runtime.gateway.routing import CatalogRouteResolver, GatewayRoute
from wmo.runtime.models import ResolvedModel, RuntimeModelCatalog
from wmo.runtime.models.providers import ProviderDeadlineExceeded, RequestDeadline
from wmo.runtime.models.providers.transport import ProviderTransportError, RetryPolicy

_DIGEST = "a" * 64


def test_direct_certified_pool_preserves_operational_deployment_order() -> None:
    """Direct resolution exposes every certified deployment in authored priority order."""
    first = _deployment("route-a", connection_sha256="b" * 64)
    second = _deployment("route-b", connection_sha256="c" * 64)
    certification = GatewayEquivalenceCertification(
        certification_id="certification-one",
        provenance="operator comparison run 2026-08-18",
        evidence_sha256=_DIGEST,
        certified_at=datetime(2026, 8, 18, tzinfo=UTC),
    )
    catalog = NormalizedGatewayCatalog(
        deployments=(first, second),
        pools=(
            ExactModelPool(
                pool_id="pool-one",
                exact_model_id="exact-one",
                deployment_ids=(first.deployment_id, second.deployment_id),
                equivalence=certification,
            ),
        ),
    )
    catalog_sha256 = catalog.identity_sha256()
    resolver = CatalogRouteResolver({("revision-one", catalog_sha256): catalog})

    async def scenario() -> None:
        """Resolve the direct target on one event loop."""
        route = await resolver.resolve(
            authorization=_authorization(catalog_sha256),
            request=GatewayRequest(
                surface=GatewayApiSurface.CHAT_COMPLETIONS,
                messages=(GatewayMessage(role="user", content="hello"),),
            ),
            episode_namespace=("org", "identity", "revision-one", "episode"),
        )

        assert route.deployments == (first, second)
        assert route.snapshot.deployment_ids == ("route-a", "route-b")

    asyncio.run(scenario())


def test_project_selection_expands_once_to_its_certified_ordered_pool() -> None:
    """One learned selection fixes the exact model before exposing its deployment waterfall."""
    first = _deployment("route-a", connection_sha256="b" * 64)
    second = _deployment("route-b", connection_sha256="c" * 64)
    catalog = NormalizedGatewayCatalog(
        deployments=(first, second),
        pools=(
            ExactModelPool(
                pool_id="pool-one",
                exact_model_id="exact-one",
                deployment_ids=(first.deployment_id, second.deployment_id),
                equivalence=GatewayEquivalenceCertification(
                    certification_id="certification-one",
                    provenance="operator comparison run 2026-08-18",
                    evidence_sha256=_DIGEST,
                    certified_at=datetime(2026, 8, 18, tzinfo=UTC),
                ),
            ),
        ),
    )
    catalog_sha256 = catalog.identity_sha256()
    project_resolver = _ProjectResolver(first.source_alias)
    resolver = CatalogRouteResolver(
        {("revision-one", catalog_sha256): catalog},
        project_resolver=project_resolver,
    )
    authorization = _authorization(catalog_sha256).model_copy(
        update={
            "target": ProjectTarget(
                project_ref="project-one",
                activation_ref="activation-one",
                catalog_sha256=catalog_sha256,
            )
        }
    )

    async def scenario() -> None:
        """Resolve one project decision into both certified physical deployments."""
        route = await resolver.resolve(
            authorization=authorization,
            request=_request(),
            episode_namespace=("org", "identity", "revision-one", "episode"),
        )

        assert route.deployments == (first, second)
        assert route.snapshot.pool_id == "pool-one"
        assert project_resolver.calls == 1

    asyncio.run(scenario())


def test_circuit_identity_is_catalog_and_connection_scoped() -> None:
    """One failed revision cannot suppress a same-named deployment in another catalog."""
    now = [100.0]
    health = DeploymentHealthRegistry(
        failure_threshold=1,
        open_seconds=10,
        throttle_seconds=5,
        clock=lambda: now[0],
    )
    first = ("catalog-a", "deployment", "connection-a")
    second = ("catalog-b", "deployment", "connection-b")

    assert health.claim(first)
    health.failed(
        first,
        GatewayFailure(
            failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
            safe_message="provider authentication failed",
            failover_eligible=True,
        ),
    )

    assert not health.claim(first)
    assert health.claim(second)
    health.succeeded(second)
    now[0] += 11
    assert health.claim(first)


def test_provider_auth_failover_opens_skips_and_recovers_the_primary() -> None:
    """Provider authentication failover opens one circuit and later probes recovery."""
    now = [100.0]
    first = _deployment("route-a", connection_sha256="b" * 64)
    second = _deployment("route-b", connection_sha256="c" * 64)
    first_provider = _ScriptedProvider(
        [
            ProviderTransportError("provider rejected credentials", status_code=401),
            _completed_stream("primary recovered"),
        ]
    )
    second_provider = _ScriptedProvider(
        [_completed_stream("fallback one"), _completed_stream("fallback two")]
    )
    health = DeploymentHealthRegistry(
        failure_threshold=1,
        open_seconds=10,
        throttle_seconds=5,
        clock=lambda: now[0],
    )
    executor = _executor(
        (first, second),
        {
            first.source_alias: first_provider,
            second.source_alias: second_provider,
        },
        _WaterfallLedger(),
        maximum_same_deployment_attempts=1,
        health=health,
    )

    async def consume() -> str:
        """Run one logical request and return its sole text delta."""
        stream = await executor.start(route=_route((first, second)), request=_request())
        events = [event async for event in stream]
        return "".join(event.text_delta or "" for event in events)

    assert asyncio.run(consume()) == "fallback one"
    assert asyncio.run(consume()) == "fallback two"
    assert len(first_provider.idempotency_keys) == 1
    now[0] += 11
    assert asyncio.run(consume()) == "primary recovered"
    assert len(first_provider.idempotency_keys) == 2
    assert len(second_provider.idempotency_keys) == 2


def test_throttle_window_recovers_without_opening_the_failure_circuit() -> None:
    """A throttle suppresses dispatch only for its independent bounded window."""
    now = [100.0]
    health = DeploymentHealthRegistry(
        failure_threshold=2,
        open_seconds=30,
        throttle_seconds=5,
        clock=lambda: now[0],
    )
    key = ("catalog", "deployment", "connection")

    assert health.claim(key)
    health.failed(
        key,
        GatewayFailure(
            failure_class=GatewayFailureClass.THROTTLED,
            safe_message="provider throttled the request",
            failover_eligible=True,
        ),
    )
    assert not health.claim(key)
    now[0] += 6
    assert health.claim(key)


def test_same_deployment_retry_reuses_identity_and_records_every_dispatch() -> None:
    """A safe retry keeps one provider key while each physical call gets an attempt."""

    async def scenario() -> None:
        """Run one opening transport retry followed by success."""
        first = _deployment("route-a", connection_sha256="b" * 64)
        provider = _ScriptedProvider(
            [
                ProviderTransportError("connection failed"),
                _completed_stream("recovered"),
            ]
        )
        ledger = _WaterfallLedger()
        executor = _executor((first,), {first.source_alias: provider}, ledger)

        stream = await executor.start(
            route=_route((first,)),
            request=_request(),
        )
        events = [event async for event in stream]

        assert [event.kind for event in events] == [
            GatewayEventKind.TEXT_DELTA,
            GatewayEventKind.COMPLETED,
        ]
        assert provider.idempotency_keys[0] == provider.idempotency_keys[1]
        assert provider.retry_attempts == [1, 1]
        assert ledger.started == [("route-a", 0, 0), ("route-a", 1, 0)]
        assert [entry[2] for entry in ledger.finished] == [False, True]

    asyncio.run(scenario())


def test_precommit_failover_changes_deployment_identity_and_finalizes_once() -> None:
    """A certified fallback uses a distinct provider key and one final parent owner."""

    async def scenario() -> None:
        """Fail the first deployment before commitment and complete on the second."""
        first = _deployment("route-a", connection_sha256="b" * 64)
        second = _deployment("route-b", connection_sha256="c" * 64)
        first_provider = _ScriptedProvider([ProviderTransportError("connection failed")])
        second_provider = _ScriptedProvider([_completed_stream("fallback")])
        ledger = _WaterfallLedger()
        executor = _executor(
            (first, second),
            {
                first.source_alias: first_provider,
                second.source_alias: second_provider,
            },
            ledger,
            maximum_same_deployment_attempts=1,
        )

        stream = await executor.start(route=_route((first, second)), request=_request())
        events = [event async for event in stream]

        assert events[0].text_delta == "fallback"
        assert stream.deployment == second
        assert stream.route_depth == 1
        assert first_provider.idempotency_keys != second_provider.idempotency_keys
        assert ledger.started == [("route-a", 0, 0), ("route-b", 1, 1)]
        assert [entry[2] for entry in ledger.finished] == [False, True]
        assert ledger.parent_finishes == []

    asyncio.run(scenario())


def test_all_opening_failures_settle_the_last_attempt_as_parent_owner() -> None:
    """Exhaustion terminalizes exactly the last physical attempt before surfacing failure."""

    async def scenario() -> None:
        """Exhaust two unavailable deployments without a second parent terminal write."""
        first = _deployment("route-a", connection_sha256="b" * 64)
        second = _deployment("route-b", connection_sha256="c" * 64)
        ledger = _WaterfallLedger()
        executor = _executor(
            (first, second),
            {
                first.source_alias: _ScriptedProvider([ProviderTransportError("first failed")]),
                second.source_alias: _ScriptedProvider([ProviderTransportError("second failed")]),
            },
            ledger,
            maximum_same_deployment_attempts=1,
        )

        with pytest.raises(GatewayExecutionError) as raised:
            await executor.start(route=_route((first, second)), request=_request())

        assert raised.value.request_finalized
        assert [entry[2] for entry in ledger.finished] == [False, True]
        assert ledger.parent_finishes == []

    asyncio.run(scenario())


def test_first_semantic_event_freezes_route_even_when_provider_later_fails() -> None:
    """Outward semantic output prevents a later provider failure from advancing routes."""

    async def scenario() -> None:
        """Emit text then a retryable terminal failure from the first deployment."""
        first = _deployment("route-a", connection_sha256="b" * 64)
        second = _deployment("route-b", connection_sha256="c" * 64)
        failure = GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="provider transport failed",
            retryable_same_deployment=True,
            failover_eligible=True,
        )
        first_provider = _ScriptedProvider(
            [
                _WaterfallStream(
                    (
                        GatewayEvent(
                            kind=GatewayEventKind.TEXT_DELTA,
                            sequence_number=0,
                            text_delta="committed",
                        ),
                        GatewayEvent(
                            kind=GatewayEventKind.FAILED,
                            sequence_number=1,
                            failure=failure,
                        ),
                    )
                )
            ]
        )
        second_provider = _ScriptedProvider([_completed_stream("must not run")])
        ledger = _WaterfallLedger()
        executor = _executor(
            (first, second),
            {
                first.source_alias: first_provider,
                second.source_alias: second_provider,
            },
            ledger,
        )

        stream = await executor.start(route=_route((first, second)), request=_request())
        events = [event async for event in stream]

        assert [event.kind for event in events] == [
            GatewayEventKind.TEXT_DELTA,
            GatewayEventKind.FAILED,
        ]
        assert second_provider.idempotency_keys == []
        assert ledger.started == [("route-a", 0, 0)]
        assert ledger.finished[0][2]

    asyncio.run(scenario())


def test_refusal_failover_requires_opt_in_and_withholds_the_failed_route() -> None:
    """A provider refusal stream advances only for an explicitly allowed alias revision."""

    async def scenario(*, opted_in: bool) -> tuple[list[GatewayEvent], int, _WaterfallLedger]:
        """Run one refusal with or without the injected revision policy."""
        first = _deployment("route-a", connection_sha256="b" * 64)
        second = _deployment("route-b", connection_sha256="c" * 64)
        first_provider = _ScriptedProvider(
            [
                _WaterfallStream(
                    (
                        GatewayEvent(
                            kind=GatewayEventKind.REFUSAL_DELTA,
                            sequence_number=0,
                            text_delta="provider refusal body",
                        ),
                        GatewayEvent(
                            kind=GatewayEventKind.USAGE,
                            sequence_number=1,
                            usage=GatewayUsage(input_tokens=5, output_tokens=2),
                        ),
                        GatewayEvent(
                            kind=GatewayEventKind.COMPLETED,
                            sequence_number=2,
                        ),
                    )
                )
            ]
        )
        second_provider = _ScriptedProvider([_completed_stream("allowed fallback")])
        ledger = _WaterfallLedger()
        executor = _executor(
            (first, second),
            {
                first.source_alias: first_provider,
                second.source_alias: second_provider,
            },
            ledger,
            maximum_same_deployment_attempts=1,
        )

        stream = await executor.start(
            route=_route((first, second), refusal_failover=opted_in),
            request=_request(),
        )
        events = [event async for event in stream]
        return events, len(second_provider.idempotency_keys), ledger

    default_events, default_fallbacks, default_ledger = asyncio.run(scenario(opted_in=False))
    opted_events, opted_fallbacks, opted_ledger = asyncio.run(scenario(opted_in=True))

    assert [event.kind for event in default_events] == [
        GatewayEventKind.REFUSAL_DELTA,
        GatewayEventKind.USAGE,
        GatewayEventKind.COMPLETED,
    ]
    assert default_fallbacks == 0
    assert [event.kind for event in opted_events] == [
        GatewayEventKind.TEXT_DELTA,
        GatewayEventKind.COMPLETED,
    ]
    assert all(event.kind is not GatewayEventKind.REFUSAL_DELTA for event in opted_events)
    assert opted_fallbacks == 1
    assert default_ledger.finished[0][1] is None
    opted_failure = opted_ledger.finished[0][1]
    assert opted_failure is not None
    assert opted_failure.failure_class is GatewayFailureClass.REFUSAL
    assert opted_ledger.finished[0][2] is False
    assert opted_ledger.finished_events[0] is not None
    assert opted_ledger.finished_events[0].usage == GatewayUsage(input_tokens=5, output_tokens=2)


@pytest.mark.parametrize(
    "refusals",
    [
        ("r" * 65_536,),
        ("a" * 32_768, "b" * 32_768),
        tuple("" for _index in range(256)),
    ],
)
def test_refusal_buffer_accepts_exact_byte_and_event_bounds(
    refusals: tuple[str, ...],
) -> None:
    """Exact refusal bounds remain precommit and can advance without exposure."""

    async def scenario() -> tuple[list[GatewayEvent], int]:
        """Complete one exactly bounded refusal-only route and its fallback."""
        first = _deployment("route-a", connection_sha256="b" * 64)
        second = _deployment("route-b", connection_sha256="c" * 64)
        first_events = tuple(
            GatewayEvent(
                kind=GatewayEventKind.REFUSAL_DELTA,
                sequence_number=index,
                text_delta=text,
            )
            for index, text in enumerate(refusals)
        ) + (
            GatewayEvent(
                kind=GatewayEventKind.USAGE,
                sequence_number=len(refusals),
                usage=GatewayUsage(input_tokens=5, output_tokens=2),
            ),
            GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=len(refusals) + 1,
            ),
        )
        second_provider = _ScriptedProvider([_completed_stream("bounded fallback")])
        executor = _executor(
            (first, second),
            {
                first.source_alias: _ScriptedProvider([_WaterfallStream(first_events)]),
                second.source_alias: second_provider,
            },
            _WaterfallLedger(),
            maximum_same_deployment_attempts=1,
        )
        stream = await executor.start(
            route=_route((first, second), refusal_failover=True),
            request=_request(),
        )
        return [event async for event in stream], len(second_provider.idempotency_keys)

    events, fallback_count = asyncio.run(scenario())

    assert [event.kind for event in events] == [
        GatewayEventKind.TEXT_DELTA,
        GatewayEventKind.COMPLETED,
    ]
    assert events[0].text_delta == "bounded fallback"
    assert fallback_count == 1


@pytest.mark.parametrize(
    "first_events",
    [
        (
            GatewayEvent(
                kind=GatewayEventKind.REFUSAL_DELTA,
                sequence_number=0,
                text_delta="r" * 70_000,
            ),
            GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1),
        ),
        (
            GatewayEvent(
                kind=GatewayEventKind.REFUSAL_DELTA,
                sequence_number=0,
                text_delta="a" * 32_768,
            ),
            GatewayEvent(
                kind=GatewayEventKind.REFUSAL_DELTA,
                sequence_number=1,
                text_delta="b" * 32_769,
            ),
            GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=2),
        ),
        (
            GatewayEvent(
                kind=GatewayEventKind.REFUSAL_DELTA,
                sequence_number=0,
                text_delta="refused",
            ),
            GatewayEvent(
                kind=GatewayEventKind.TEXT_DELTA,
                sequence_number=1,
                text_delta="mixed output",
            ),
            GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=2),
        ),
        tuple(
            GatewayEvent(
                kind=GatewayEventKind.REFUSAL_DELTA,
                sequence_number=index,
                text_delta="",
            )
            for index in range(257)
        )
        + (GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=257),),
    ],
)
def test_refusal_buffer_overflow_or_mixed_output_commits_without_failover(
    first_events: tuple[GatewayEvent, ...],
) -> None:
    """A bounded refusal buffer flushes safely before semantic output can switch routes."""

    async def scenario() -> tuple[list[GatewayEvent], int]:
        """Consume one overflow or mixed-semantic provider stream."""
        first = _deployment("route-a", connection_sha256="b" * 64)
        second = _deployment("route-b", connection_sha256="c" * 64)
        first_provider = _ScriptedProvider([_WaterfallStream(first_events)])
        second_provider = _ScriptedProvider([_completed_stream("must not run")])
        executor = _executor(
            (first, second),
            {
                first.source_alias: first_provider,
                second.source_alias: second_provider,
            },
            _WaterfallLedger(),
            maximum_same_deployment_attempts=1,
        )
        stream = await executor.start(
            route=_route((first, second), refusal_failover=True),
            request=_request(),
        )
        return [event async for event in stream], len(second_provider.idempotency_keys)

    events, fallback_count = asyncio.run(scenario())

    assert [event.kind for event in events] == [event.kind for event in first_events]
    assert [
        event.text_delta for event in events if event.kind is GatewayEventKind.REFUSAL_DELTA
    ] == [
        event.text_delta for event in first_events if event.kind is GatewayEventKind.REFUSAL_DELTA
    ]
    assert fallback_count == 0


def test_deadline_between_attempts_finalizes_parent_and_releases_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired next dispatch releases admission and its claimed health probe."""
    calls = 0

    def attempt_timeout(
        _deadline: RequestDeadline,
        maximum_seconds: float | None = None,
        *,
        now_monotonic: float | None = None,
    ) -> float:
        """Expire only after admission and the first physical dispatch."""
        del maximum_seconds, now_monotonic
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise ProviderDeadlineExceeded("provider request deadline exceeded")
        return 30

    monkeypatch.setattr(RequestDeadline, "attempt_timeout", attempt_timeout)

    async def scenario() -> None:
        """Advance after a precommit failure into an already expired deadline."""
        first = _deployment("route-a", connection_sha256="b" * 64)
        second = _deployment("route-b", connection_sha256="c" * 64)
        failure = GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="provider transport failed",
            retryable_same_deployment=True,
            failover_eligible=True,
        )
        first_provider = _ScriptedProvider(
            [
                _WaterfallStream(
                    (
                        GatewayEvent(
                            kind=GatewayEventKind.FAILED,
                            sequence_number=0,
                            failure=failure,
                        ),
                    )
                )
            ]
        )
        second_provider = _ScriptedProvider([_completed_stream("must not run")])
        now = [100.0]
        health = DeploymentHealthRegistry(
            failure_threshold=1,
            open_seconds=10,
            throttle_seconds=5,
            clock=lambda: now[0],
        )
        second_key = (_DIGEST, second.deployment_id, second.connection_sha256)
        health.failed(
            second_key,
            GatewayFailure(
                failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
                safe_message="provider authentication failed",
            ),
        )
        now[0] += 11
        ledger = _WaterfallLedger()
        executor = _executor(
            (first, second),
            {
                first.source_alias: first_provider,
                second.source_alias: second_provider,
            },
            ledger,
            maximum_same_deployment_attempts=1,
            health=health,
        )

        stream = await executor.start(route=_route((first, second)), request=_request())
        events = [event async for event in stream]

        assert [event.kind for event in events] == [GatewayEventKind.FAILED]
        assert events[0].failure is not None
        assert events[0].failure.failure_class is GatewayFailureClass.TIMEOUT
        assert ledger.started == [("route-a", 0, 0)]
        assert ledger.finished == [("attempt-1", failure, False)]
        assert len(ledger.parent_finishes) == 1
        assert ledger.parent_finishes[0].failure_class is GatewayFailureClass.TIMEOUT
        assert second_provider.idempotency_keys == []
        await asyncio.wait_for(executor._permits.acquire(), timeout=0.1)  # noqa: SLF001
        executor._permits.release()  # noqa: SLF001
        assert health.claim(second_key)
        health.release_probe(second_key)

    asyncio.run(scenario())


class _WaterfallStream:
    """Yield one scripted normalized event sequence."""

    def __init__(self, events: tuple[GatewayEvent, ...]) -> None:
        """Store the provider-local events in delivery order."""
        self._events = iter(events)
        self._committed = False
        self.cancelled = False

    @property
    def committed(self) -> bool:
        """Return whether this physical stream emitted semantic output."""
        return self._committed

    def __aiter__(self) -> AsyncIterator[GatewayEvent]:
        """Return this one-pass asynchronous iterator."""
        return self

    async def __anext__(self) -> GatewayEvent:
        """Yield the next scripted provider event."""
        try:
            event = next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        if event.kind in {
            GatewayEventKind.TEXT_DELTA,
            GatewayEventKind.REFUSAL_DELTA,
            GatewayEventKind.TOOL_CALL_STARTED,
            GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            GatewayEventKind.TOOL_CALL_COMPLETED,
        }:
            self._committed = True
        return event

    async def cancel(self) -> None:
        """Record bounded cancellation of this provider stream."""
        self.cancelled = True


class _ProjectResolver:
    """Return one fixed learned selection while counting selection calls."""

    def __init__(self, selected_alias: str) -> None:
        """Bind the selected deployment alias."""
        self._selected_alias = selected_alias
        self.calls = 0

    async def select(
        self,
        *,
        target: GatewayTarget,
        request: GatewayRequest,
        episode_namespace: tuple[str, str, str, str],
        deadline_monotonic: float,
    ) -> ProjectSelection:
        """Return the fixed exact-model decision without provider work."""
        del request, episode_namespace, deadline_monotonic
        if not isinstance(target, ProjectTarget):
            raise AssertionError("project resolver received a direct target")
        self.calls += 1
        return ProjectSelection(
            exact_model_id="exact-one",
            selected_alias=self._selected_alias,
            activation_ref=target.activation_ref,
        )


class _ScriptedProvider:
    """Open scripted streams or failures while recording dispatch identity."""

    def __init__(self, outcomes: list[_WaterfallStream | BaseException]) -> None:
        """Retain one outcome per expected physical dispatch."""
        self._outcomes = outcomes
        self.idempotency_keys: list[str] = []
        self.retry_attempts: list[int] = []

    async def stream(
        self,
        request: GatewayRequest,
        *,
        deadline: RequestDeadline,
        idempotency_key: str,
        retry_policy: RetryPolicy | None = None,
    ) -> _WaterfallStream:
        """Return or raise the next scripted physical-dispatch result."""
        assert request.stream and request.include_usage
        assert deadline.remaining_seconds() > 0
        assert retry_policy is not None
        self.idempotency_keys.append(idempotency_key)
        self.retry_attempts.append(retry_policy.maximum_attempts)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _WaterfallRuntimeCatalog:
    """Resolve each route alias to its injected provider and frozen identity."""

    def __init__(
        self,
        deployments: tuple[ExactModelDeployment, ...],
        providers: dict[str, _ScriptedProvider],
    ) -> None:
        """Index deployments and providers by source alias."""
        self._deployments = {item.source_alias: item for item in deployments}
        self._providers = providers

    def resolve(self, alias: str, *, role: str | None = None) -> ResolvedModel:
        """Return one runtime binding matching the frozen deployment exactly."""
        del role
        deployment = self._deployments[alias]
        capabilities = ModelCapabilities()
        return ResolvedModel(
            alias=alias,
            snapshot=ModelSnapshot(
                provider=deployment.provider,
                model_id=deployment.provider_model,
                revision=deployment.revision,
                billing_source=deployment.billing_source,
                capabilities_sha256=capabilities.identity_sha256(),
                connection_sha256=deployment.connection_sha256,
            ),
            capabilities=capabilities,
            client=cast(ModelClient, self._providers[alias]),
            embedding_client=None,
        )


class _WaterfallLedger:
    """Capture physical and parent terminal ownership without request content."""

    def __init__(self) -> None:
        """Initialize empty ordered accounting records."""
        self.started: list[tuple[str, int, int]] = []
        self.finished: list[tuple[str, GatewayFailure | None, bool]] = []
        self.finished_events: list[GatewayEvent | None] = []
        self.parent_finishes: list[GatewayFailure] = []

    def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Accept a parent request when a service-level test requires it."""
        del authorization

    def start_attempt(
        self,
        *,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
    ) -> str:
        """Record one durable physical dispatch before provider work."""
        assert deployment.deployment_id in snapshot.deployment_ids
        self.started.append((deployment.deployment_id, attempt_ordinal, route_depth))
        return f"attempt-{len(self.started)}"

    def record_route_context(
        self,
        *,
        attempt_id: str,
        route_reason: str | None,
        fallback_reason: str | None,
    ) -> None:
        """Accept display-safe route context for the scripted attempt."""
        del attempt_id, route_reason, fallback_reason

    def finish_attempt(
        self,
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
    ) -> None:
        """Record physical settlement and whether it owns parent terminalization."""
        self.finished_events.append(terminal_event)
        self.finished.append((attempt_id, failure, finalize_request))

    def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Record terminalization without a new physical attempt."""
        del authorization
        self.parent_finishes.append(failure)


def _completed_stream(text: str) -> _WaterfallStream:
    """Build one semantic text event followed by successful terminal completion."""
    return _WaterfallStream(
        (
            GatewayEvent(
                kind=GatewayEventKind.TEXT_DELTA,
                sequence_number=0,
                text_delta=text,
            ),
            GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1),
        )
    )


def _request() -> GatewayRequest:
    """Build one canonical request for physical execution tests."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )


def _route(
    deployments: tuple[ExactModelDeployment, ...],
    *,
    refusal_failover: bool = False,
) -> GatewayRoute:
    """Build one frozen certified route with a live request deadline."""
    authorization = _authorization(_DIGEST).model_copy(
        update={
            "deadline_monotonic": time.monotonic() + 30,
            "refusal_failover": refusal_failover,
        }
    )
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


def _executor(
    deployments: tuple[ExactModelDeployment, ...],
    providers: dict[str, _ScriptedProvider],
    ledger: _WaterfallLedger,
    *,
    maximum_same_deployment_attempts: int = 2,
    health: DeploymentHealthRegistry | None = None,
) -> GatewayExecutor:
    """Compose one revision-pinned executor for the scripted route."""
    catalog = cast(
        RuntimeModelCatalog,
        _WaterfallRuntimeCatalog(deployments, providers),
    )
    return GatewayExecutor(
        {("revision-one", _DIGEST): catalog},
        ledger,
        maximum_same_deployment_attempts=maximum_same_deployment_attempts,
        health=health,
    )


def _deployment(deployment_id: str, *, connection_sha256: str) -> ExactModelDeployment:
    """Build one deployment in the shared certified exact-model pool."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai",
        provider_model="provider-model",
        connection_sha256=connection_sha256,
        capabilities_sha256="d" * 64,
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(supports_streaming=True)
        ),
    )


def _authorization(catalog_sha256: str) -> AuthorizationSnapshot:
    """Build one direct authority snapshot pinned to the test catalog."""
    return AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256=catalog_sha256,
        canonical_request_sha256=_DIGEST,
        deadline_monotonic=1.0,
    )

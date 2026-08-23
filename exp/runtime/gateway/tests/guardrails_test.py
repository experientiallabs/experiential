"""Service-level guardrail data-flow, waterfall, streaming, and fail-closed tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import cast

import pytest
from fastapi.responses import StreamingResponse

from exp.common.models import ModelCapabilities, ModelClient, ModelSnapshot, ToolCall
from exp.common.models.catalog import (
    GatewayEquivalenceCertification,
)
from exp.common.models.gateway_catalog import (
    ExactModelPool,
    NormalizedGatewayCatalog,
)
from exp.runtime.gateway.contracts import (
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.execution import GatewayExecutor
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, ScriptedClassifier
from exp.runtime.gateway.guardrails.client import (
    DirectClassifierClient,
    GuardrailRecursionError,
    classification_scope,
)
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailPolicy,
    GuardrailRejected,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.routing import CatalogRouteResolver
from exp.runtime.gateway.service import GatewayService, create_gateway_app
from exp.runtime.gateway.tests.data_plane_test import (
    _BlockingStream,
    _Clock,
    _ControlStore,
    _EventStream,
    _Ledger,
    _Provider,
    _service,
)
from exp.runtime.gateway.tests.waterfall_test import (
    _completed_stream,
    _deployment,
    _executor,
    _route,
    _ScriptedProvider,
    _WaterfallLedger,
    _WaterfallStream,
)
from exp.runtime.models import ResolvedModel, RuntimeModelCatalog
from exp.runtime.models.providers import RequestDeadline
from exp.runtime.models.providers.transport import ProviderTransportError, RetryPolicy
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses
from exp.runtime.openai_protocol.state import BoundedReplayStore


def _check(
    check_id: str,
    *,
    stage: GuardrailCheckStage = GuardrailCheckStage.INPUT,
    action: GuardrailAction = GuardrailAction.BLOCK,
) -> GuardrailCheck:
    """Build one content-safety check."""
    return GuardrailCheck(
        check_id=check_id,
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=stage,
        action=action,
        timeout_ms=250,
        adapter_id="scripted",
    )


def _engine(
    classifier: ScriptedClassifier,
    *,
    checks: tuple[GuardrailCheck, ...],
    protected: bool = True,
    identity_id: str = "identity-one",
) -> GuardrailEngine:
    """Compose one engine for the data-plane identity."""
    policy = GuardrailPolicy(
        policy_id="member-policy",
        organization_id="organization-one",
        identity_id=identity_id,
        protected=protected,
        checks=checks,
    )
    return GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"scripted": classifier})),
        monotonic=_Clock().monotonic,
    )


def _text_events(text: str) -> tuple[GatewayEvent, ...]:
    """Return one successful text completion."""
    return (
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta=text),
        GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=1,
            usage=GatewayUsage(input_tokens=3, output_tokens=1),
        ),
    )


class _RecordingProvider(_Provider):
    """Record every canonical request dispatched to the provider."""

    def __init__(self, factory: Callable[[], _EventStream | _BlockingStream]) -> None:
        """Bind the injected stream factory."""
        super().__init__(factory)
        self.requests: list[GatewayRequest] = []

    async def stream(
        self,
        request: GatewayRequest,
        *,
        deadline: RequestDeadline,
        idempotency_key: str,
        retry_policy: RetryPolicy | None = None,
    ) -> _EventStream | _BlockingStream:
        """Capture the request, then open the injected stream."""
        self.requests.append(request)
        return await super().stream(
            request,
            deadline=deadline,
            idempotency_key=idempotency_key,
            retry_policy=retry_policy,
        )


def test_unguarded_traffic_never_calls_classifiers() -> None:
    """No assigned policy leaves the existing hot path and call counts at zero."""

    async def scenario() -> None:
        """Serve one request without a guardrail engine."""
        classifier = ScriptedClassifier()
        provider = _RecordingProvider(lambda: _EventStream(_text_events("hello")))
        service, _control, ledger, _proof = _service(provider)

        response = await service.complete(
            raw_key="caller-secret",
            decoded=decode_chat(
                {"model": "public-model", "messages": [{"role": "user", "content": "hello"}]}
            ),
        )

        assert response.status_code == 200
        assert classifier.input_calls == 0
        assert classifier.output_calls == 0
        assert ledger.accepted
        assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_other_organization_policy_does_not_apply() -> None:
    """The same identity ID in another organization does not share a policy."""

    async def scenario() -> None:
        """Serve one request whose identity matches only a foreign organization."""
        classifier = ScriptedClassifier(input_verdict=ClassifierVerdict(flagged=True))
        policy = GuardrailPolicy(
            policy_id="foreign-policy",
            organization_id="organization-two",
            identity_id="identity-one",
            protected=True,
            checks=(_check("input-safety"),),
        )
        engine = GuardrailEngine(
            store=MappingGuardrailStore((policy,)),
            client=DirectClassifierClient(ClassifierRegistry({"scripted": classifier})),
            monotonic=_Clock().monotonic,
        )
        provider = _RecordingProvider(lambda: _EventStream(_text_events("hello")))
        service, _control, ledger, _proof = _service(provider, guardrails=engine)

        response = await service.complete(
            raw_key="caller-secret",
            decoded=decode_chat(
                {"model": "public-model", "messages": [{"role": "user", "content": "hello"}]}
            ),
        )

        assert response.status_code == 200
        assert classifier.input_calls == 0
        assert engine.policy_for("organization-one", "identity-one") is None
        assert ledger.accepted

    asyncio.run(scenario())


def test_input_block_does_not_accept_or_dispatch() -> None:
    """A blocked input chain never reaches ledger acceptance or the provider."""

    async def scenario() -> None:
        """Serve one blocked request."""
        classifier = ScriptedClassifier(input_verdict=ClassifierVerdict(flagged=True))
        provider = _RecordingProvider(lambda: _EventStream(_text_events("hello")))
        service, _control, ledger, _proof = _service(
            provider,
            guardrails=_engine(classifier, checks=(_check("input-safety"),)),
        )

        with pytest.raises(GuardrailRejected):
            await service.complete(
                raw_key="caller-secret",
                decoded=decode_chat(
                    {
                        "model": "public-model",
                        "messages": [{"role": "user", "content": "blocked-prompt"}],
                    }
                ),
            )

        assert classifier.input_calls == 1
        assert provider.requests == []
        assert ledger.accepted == []
        assert ledger.started == []

    asyncio.run(scenario())


def test_http_guardrail_block_is_a_sanitized_content_filter() -> None:
    """The public HTTP boundary returns 400 content_filter and no prompt text."""

    async def scenario() -> None:
        """Dispatch one blocked request through the ASGI application."""
        classifier = ScriptedClassifier(input_verdict=ClassifierVerdict(flagged=True))
        provider = _RecordingProvider(lambda: _EventStream(_text_events("hello")))
        service, _control, _ledger, _proof = _service(
            provider,
            guardrails=_engine(classifier, checks=(_check("input-safety"),)),
        )
        transport = __import__("httpx").ASGITransport(app=create_gateway_app(service))
        async with __import__("httpx").AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer caller-secret"},
                json={
                    "model": "public-model",
                    "messages": [{"role": "user", "content": "blocked-prompt"}],
                },
            )

        body = response.json()
        assert response.status_code == 400
        assert body["error"]["code"] == "content_filter"
        assert "blocked-prompt" not in json.dumps(body)

    asyncio.run(scenario())


def test_input_runs_after_continuation_and_once_per_request() -> None:
    """Continuation expansion happens before the input chain sees the request."""

    async def scenario() -> None:
        """Continue a Responses turn and inspect the expanded messages."""
        seen: list[tuple[str, ...]] = []

        class _Capture(ScriptedClassifier):
            """Record message contents seen by the input adapter."""

            def inspect_input(
                self,
                *,
                request: GatewayRequest,
                check: GuardrailCheck,
            ) -> ClassifierVerdict:
                """Capture contents, then allow."""
                seen.append(tuple(message.content or "" for message in request.messages))
                return super().inspect_input(request=request, check=check)

        classifier = _Capture()
        provider = _RecordingProvider(lambda: _EventStream(_text_events("hello")))
        service, _control, _ledger, _proof = _service(
            provider,
            guardrails=_engine(classifier, checks=(_check("input-safety"),)),
        )
        first = await service.complete(
            raw_key="caller-secret",
            decoded=decode_responses({"model": "public-model", "input": "parent-turn"}),
        )
        first_id = json.loads(bytes(first.body))["id"]
        await service.complete(
            raw_key="caller-secret",
            decoded=decode_responses(
                {
                    "model": "public-model",
                    "input": "child-turn",
                    "previous_response_id": first_id,
                }
            ),
        )

        assert seen[0] == ("parent-turn",)
        assert "parent-turn" in seen[1]
        assert "child-turn" in seen[1]
        assert classifier.input_calls == 2

    asyncio.run(scenario())


def test_output_runs_once_before_delivery_or_replay() -> None:
    """Streaming output checks finish before the first public byte or replay publish."""

    async def scenario() -> None:
        """Stream one guarded completion and inspect ordering."""
        classifier = ScriptedClassifier()
        engine = _engine(
            classifier,
            checks=(_check("output-safety", stage=GuardrailCheckStage.OUTPUT),),
        )
        provider = _RecordingProvider(lambda: _EventStream(_text_events("hello")))
        replays = BoundedReplayStore()
        service, _control, _ledger, _proof = _service(
            provider,
            guardrails=engine,
            replay_store=replays,
        )
        response = await service.complete(
            raw_key="caller-secret",
            decoded=decode_chat(
                {
                    "model": "public-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
                idempotency_key="operation-one",
            ),
        )

        assert isinstance(response, StreamingResponse)
        assert engine.output_invocations == 1
        assert classifier.output_calls == 1
        frames: list[bytes] = []
        async for frame in cast(AsyncIterator[bytes], response.body_iterator):
            frames.append(frame)
        assert b"hello" in b"".join(frames)

    asyncio.run(scenario())


def test_output_block_exposes_no_partial_bytes() -> None:
    """A blocked completion becomes a sanitized error before StreamingResponse starts."""

    async def scenario() -> None:
        """Block one streaming completion."""
        classifier = ScriptedClassifier(output_verdict=ClassifierVerdict(flagged=True))
        provider = _RecordingProvider(lambda: _EventStream(_text_events("unsafe-output")))
        service, _control, _ledger, _proof = _service(
            provider,
            guardrails=_engine(
                classifier,
                checks=(_check("output-safety", stage=GuardrailCheckStage.OUTPUT),),
            ),
        )

        with pytest.raises(GuardrailRejected) as raised:
            await service.complete(
                raw_key="caller-secret",
                decoded=decode_chat(
                    {
                        "model": "public-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    }
                ),
            )

        assert raised.value.failure.failover_eligible is False
        assert "unsafe-output" not in raised.value.failure.safe_message

    asyncio.run(scenario())


def test_textless_streaming_modify_encodes_without_duplicate_sequences() -> None:
    """A refusal-only modify inserts text without 502 invalid_provider_stream."""

    async def scenario() -> None:
        """Stream one refusal-only completion through an output modify check."""
        classifier = ScriptedClassifier(
            output_verdict=ClassifierVerdict(flagged=True, replacement_text="safe")
        )
        events = (
            GatewayEvent(
                kind=GatewayEventKind.REFUSAL_DELTA,
                sequence_number=0,
                text_delta="I cannot",
            ),
            GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=1,
                usage=GatewayUsage(input_tokens=3, output_tokens=1),
            ),
        )
        provider = _RecordingProvider(lambda: _EventStream(events))
        service, _control, _ledger, _proof = _service(
            provider,
            guardrails=_engine(
                classifier,
                checks=(
                    _check(
                        "output-safety",
                        stage=GuardrailCheckStage.OUTPUT,
                        action=GuardrailAction.MODIFY,
                    ),
                ),
            ),
        )

        response = await service.complete(
            raw_key="caller-secret",
            decoded=decode_chat(
                {
                    "model": "public-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                }
            ),
        )

        assert isinstance(response, StreamingResponse)
        frames: list[bytes] = []
        async for frame in cast(AsyncIterator[bytes], response.body_iterator):
            frames.append(frame)
        body = b"".join(frames)
        assert b"invalid_provider_stream" not in body
        assert b"safe" in body

    asyncio.run(scenario())


def test_refusal_only_responses_modify_encodes_without_mixed_deltas() -> None:
    """A Responses refusal-only modify is text-only and does not return 502."""

    async def scenario() -> None:
        """Stream one refusal-only Responses completion through output modify."""
        classifier = ScriptedClassifier(
            output_verdict=ClassifierVerdict(flagged=True, replacement_text="safe")
        )
        events = (
            GatewayEvent(
                kind=GatewayEventKind.REFUSAL_DELTA,
                sequence_number=0,
                text_delta="I cannot",
            ),
            GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=1,
                usage=GatewayUsage(input_tokens=3, output_tokens=1),
            ),
        )
        provider = _RecordingProvider(lambda: _EventStream(events))
        service, _control, _ledger, _proof = _service(
            provider,
            guardrails=_engine(
                classifier,
                checks=(
                    _check(
                        "output-safety",
                        stage=GuardrailCheckStage.OUTPUT,
                        action=GuardrailAction.MODIFY,
                    ),
                ),
            ),
        )

        response = await service.complete(
            raw_key="caller-secret",
            decoded=decode_responses(
                {
                    "model": "public-model",
                    "input": "hello",
                    "stream": True,
                }
            ),
        )

        assert isinstance(response, StreamingResponse)
        frames: list[bytes] = []
        async for frame in cast(AsyncIterator[bytes], response.body_iterator):
            frames.append(frame)
        body = b"".join(frames)
        assert b"invalid_provider_stream" not in body
        assert b"safe" in body
        assert b"I cannot" not in body

    asyncio.run(scenario())


def test_tool_call_arguments_are_blocked_not_rewritten() -> None:
    """A modify action on a tool-calling completion becomes a block."""

    async def scenario() -> None:
        """Complete one tool-calling response under a modify output check."""
        classifier = ScriptedClassifier(
            output_verdict=ClassifierVerdict(flagged=True, replacement_text="safe")
        )
        events = (
            GatewayEvent(
                kind=GatewayEventKind.TOOL_CALL_STARTED,
                sequence_number=0,
                tool_call_index=0,
                tool_call_id="call-1",
                tool_name="lookup",
            ),
            GatewayEvent(
                kind=GatewayEventKind.TOOL_CALL_COMPLETED,
                sequence_number=1,
                tool_call_index=0,
                tool_call=ToolCall(
                    call_id="call-1",
                    name="lookup",
                    arguments={"q": "secret"},
                    raw_arguments='{"q":"secret"}',
                ),
            ),
            GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=2),
        )
        provider = _RecordingProvider(lambda: _EventStream(events))
        service, _control, _ledger, _proof = _service(
            provider,
            guardrails=_engine(
                classifier,
                checks=(
                    _check(
                        "output-safety",
                        stage=GuardrailCheckStage.OUTPUT,
                        action=GuardrailAction.MODIFY,
                    ),
                ),
            ),
        )

        with pytest.raises(GuardrailRejected) as raised:
            await service.complete(
                raw_key="caller-secret",
                decoded=decode_chat(
                    {"model": "public-model", "messages": [{"role": "user", "content": "hello"}]}
                ),
            )

        assert raised.value.failure.safe_details["action"] == "block"

    asyncio.run(scenario())


def test_public_route_rejects_classifier_recursion() -> None:
    """complete() refuses to run while an internal classification is active."""

    async def scenario() -> None:
        """Re-enter the public service from a classification scope."""
        provider = _RecordingProvider(lambda: _EventStream(_text_events("hello")))
        service, _control, _ledger, _proof = _service(provider)
        with classification_scope():
            with pytest.raises(GuardrailRecursionError):
                await service.complete(
                    raw_key="caller-secret",
                    decoded=decode_chat(
                        {
                            "model": "public-model",
                            "messages": [{"role": "user", "content": "hello"}],
                        }
                    ),
                )

    asyncio.run(scenario())


def test_waterfall_reuses_one_transformed_request_and_keeps_provider_failover() -> None:
    """Input runs once; every physical attempt sees the same transformed request."""

    async def scenario() -> None:
        """Fail the primary provider, then succeed on the fallback."""
        replacement = (GatewayMessage(role="user", content="redacted"),)
        classifier = ScriptedClassifier(
            input_verdict=ClassifierVerdict(flagged=True, replacement_messages=replacement)
        )
        first = _deployment("route-a", connection_sha256="b" * 64)
        second = _deployment("route-b", connection_sha256="c" * 64)
        first_provider = _RecordingScripted(
            [ProviderTransportError("primary failed", status_code=503)]
        )
        second_provider = _RecordingScripted([_completed_stream("fallback")])
        ledger = _WaterfallLedger()
        executor = _executor(
            (first, second),
            {first.source_alias: first_provider, second.source_alias: second_provider},
            ledger,
            maximum_same_deployment_attempts=1,
        )
        engine = _engine(
            classifier,
            checks=(_check("input-safety", action=GuardrailAction.MODIFY),),
        )
        policy = engine.policy_for("organization-one", "identity-one")
        assert policy is not None
        request = GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="original"),),
        )
        transformed = engine.enforce_input(
            policy=policy,
            request=request,
            deadline_monotonic=_Clock().monotonic() + 30,
        )
        stream = await executor.start(route=_route((first, second)), request=transformed)
        events = [event async for event in stream]

        assert classifier.input_calls == 1
        assert len(first_provider.requests) == 1
        assert first_provider.requests == second_provider.requests
        dispatched = first_provider.requests[0]
        assert dispatched.messages == transformed.messages
        assert dispatched.messages[0].content == "redacted"
        assert dispatched.stream is True
        assert dispatched.include_usage is True
        assert events[-1].kind is GatewayEventKind.COMPLETED

    asyncio.run(scenario())


def test_output_block_does_not_advance_the_waterfall() -> None:
    """A successful primary completion that fails output does not dispatch fallback."""

    async def scenario() -> None:
        """Block the winning completion after the first physical attempt."""
        classifier = ScriptedClassifier(output_verdict=ClassifierVerdict(flagged=True))
        first = _deployment("route-a", connection_sha256="b" * 64)
        second = _deployment("route-b", connection_sha256="c" * 64)
        first_provider = _RecordingScripted([_completed_stream("unsafe")])
        second_provider = _RecordingScripted([_completed_stream("fallback")])
        catalog = NormalizedGatewayCatalog(
            deployments=(first, second),
            pools=(
                ExactModelPool(
                    pool_id="pool-one",
                    exact_model_id="exact-one",
                    deployment_ids=(first.deployment_id, second.deployment_id),
                    equivalence=GatewayEquivalenceCertification(
                        certification_id="certification-one",
                        provenance="operator comparison",
                        evidence_sha256="a" * 64,
                        certified_at=datetime(2026, 8, 18, tzinfo=UTC),
                    ),
                ),
            ),
        )
        control = _ControlStore(catalog.identity_sha256())
        ledger = _Ledger()
        routes = CatalogRouteResolver({("revision-one", catalog.identity_sha256()): catalog})

        class _MapCatalog:
            """Resolve each source alias to its scripted provider."""

            def resolve(self, alias: str) -> ResolvedModel:
                """Return the matching runtime binding."""
                deployment = {first.source_alias: first, second.source_alias: second}[alias]
                provider = {
                    first.source_alias: first_provider,
                    second.source_alias: second_provider,
                }[alias]
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
                    client=cast(ModelClient, provider),
                    embedding_client=None,
                )

        service = GatewayService(
            control_store=control,
            ledger=ledger,
            routes=routes,
            executor=GatewayExecutor(
                {
                    ("revision-one", catalog.identity_sha256()): cast(
                        RuntimeModelCatalog, _MapCatalog()
                    )
                },
                ledger,
            ),
            clock=_Clock(),
            readiness_probe=_unused_readiness,
            guardrails=_engine(
                classifier,
                checks=(_check("output-safety", stage=GuardrailCheckStage.OUTPUT),),
            ),
        )
        with pytest.raises(GuardrailRejected):
            await service.complete(
                raw_key="caller-secret",
                decoded=decode_chat(
                    {"model": "public-model", "messages": [{"role": "user", "content": "hello"}]}
                ),
            )

        assert first_provider.requests
        assert second_provider.requests == []

    asyncio.run(scenario())


def test_protected_adapter_failure_is_terminal() -> None:
    """A protected identity fail-closes before provider dispatch."""

    class _Boom(ScriptedClassifier):
        """Raise on every input inspection."""

        def inspect_input(
            self,
            *,
            request: GatewayRequest,
            check: GuardrailCheck,
        ) -> ClassifierVerdict:
            """Fail the adapter."""
            del request, check
            self.input_calls += 1
            raise RuntimeError("detector down")

    async def scenario() -> None:
        """Serve one request whose classifier cannot complete."""
        provider = _RecordingProvider(lambda: _EventStream(_text_events("hello")))
        service, _control, ledger, _proof = _service(
            provider,
            guardrails=_engine(_Boom(), checks=(_check("input-safety"),), protected=True),
        )

        with pytest.raises(GuardrailRejected) as raised:
            await service.complete(
                raw_key="caller-secret",
                decoded=decode_chat(
                    {"model": "public-model", "messages": [{"role": "user", "content": "hello"}]}
                ),
            )

        assert raised.value.failure.failure_class is GatewayFailureClass.GUARDRAIL
        assert provider.requests == []
        assert ledger.started == []

    asyncio.run(scenario())


async def _unused_readiness() -> ExecutionSnapshot:
    """Readiness is unused by complete() in the isolated waterfall service."""
    raise AssertionError("readiness is not used")


class _RecordingScripted(_ScriptedProvider):
    """Scripted waterfall provider that records canonical requests."""

    def __init__(self, outcomes: list[_WaterfallStream | BaseException]) -> None:
        """Retain scripted outcomes."""
        super().__init__(outcomes)
        self.requests: list[GatewayRequest] = []

    async def stream(
        self,
        request: GatewayRequest,
        *,
        deadline: RequestDeadline,
        idempotency_key: str,
        retry_policy: RetryPolicy | None = None,
    ) -> _WaterfallStream:
        """Record the request, then delegate."""
        self.requests.append(request)
        return await super().stream(
            request,
            deadline=deadline,
            idempotency_key=idempotency_key,
            retry_policy=retry_policy,
        )

"""Native-boundary JSON contract tests."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.guardrails.bounded import BoundedInspect
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, ScriptedClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailPolicy,
    GuardrailRejected,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native import (
    encode_output_decision,
    enforce_native_input,
    enforce_native_output,
    parse_output_payload,
)
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore


def _authorization() -> AuthorizationSnapshot:
    """Return one identity-scoped authority snapshot."""
    return AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256="a" * 64,
        canonical_request_sha256="b" * 64,
        deadline_monotonic=200.0,
    )


def _engine() -> GuardrailEngine:
    """Compose one blocking input-and-output engine."""
    policy = GuardrailPolicy(
        policy_id="member-policy",
        organization_id="organization-one",
        identity_id="identity-one",
        checks=(
            GuardrailCheck(
                check_id="input-safety",
                capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                stage=GuardrailCheckStage.INPUT,
                action=GuardrailAction.BLOCK,
                timeout_ms=100,
                adapter_id="scripted",
            ),
            GuardrailCheck(
                check_id="output-safety",
                capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                stage=GuardrailCheckStage.OUTPUT,
                action=GuardrailAction.BLOCK,
                timeout_ms=100,
                adapter_id="scripted",
            ),
        ),
    )
    return GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(
            ClassifierRegistry(
                {
                    "scripted": ScriptedClassifier(
                        input_verdict=ClassifierVerdict(flagged=False),
                        output_verdict=ClassifierVerdict(flagged=True),
                    )
                }
            )
        ),
        monotonic=lambda: 0.0,
    )


def test_unguarded_native_input_does_not_call_classifiers() -> None:
    """No engine, or no assigned policy, leaves the request unchanged."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )

    unchanged, policy = enforce_native_input(
        None,
        authorization=_authorization(),
        request=request,
        deadline_monotonic=200.0,
    )

    assert unchanged is request
    assert policy is None


def test_native_input_runs_the_async_chain_on_a_private_loop() -> None:
    """The native callback awaits enforcement without a caller event loop."""
    engine = _engine()
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )

    rewritten, policy = enforce_native_input(
        engine,
        authorization=_authorization(),
        request=request,
        deadline_monotonic=200.0,
    )

    assert policy is not None
    assert rewritten.messages[0].content == "hello"


def test_native_output_payload_round_trips_tool_calls() -> None:
    """The JSON boundary carries tool-call arguments without rewriting them."""
    completion = parse_output_payload(
        {
            "text": "hello",
            "reasoning_content": "hidden provider reasoning",
            "refusal": False,
            "tool_calls": [{"call_id": "call-1", "name": "lookup", "arguments": '{"q":"x"}'}],
        }
    )

    assert completion.tool_calls[0].arguments == '{"q":"x"}'
    assert completion.reasoning_content == "hidden provider reasoning"


def test_native_output_block_returns_sanitized_failure_json() -> None:
    """A blocked output chain returns action plus a content-free failure."""
    engine = _engine()
    policy = engine.policy_for("organization-one", "identity-one")
    decision = json.loads(
        enforce_native_output(
            engine,
            policy,
            json.dumps({"text": "unsafe", "tool_calls": []}),
            deadline_monotonic=200.0,
        )
    )

    assert decision["action"] == "block"
    assert decision["failure"]["failure_class"] == "guardrail"
    assert "unsafe" not in json.dumps(decision)


def test_missing_policy_on_output_callback_fail_closes() -> None:
    """Invoking the output callback without a captured policy is a terminal error."""
    decision = json.loads(
        enforce_native_output(
            None,
            None,
            json.dumps({"text": "ok"}),
            deadline_monotonic=200.0,
        )
    )

    assert decision["action"] == "error"
    assert encode_output_decision(action="allow") == '{"action":"allow"}'


class _NativeCancelSwallow:
    """Native-loop adapter that swallows cancellation until teardown."""

    def __init__(self) -> None:
        """Start with an empty call count and a closed hold."""
        self.input_calls = 0
        self._hold = threading.Event()

    def release(self) -> None:
        """Allow every abandoned inspect to finish."""
        self._hold.set()

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Ignore cancellation and wait for teardown."""
        del request, check
        self.input_calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            deadline = time.monotonic() + 5.0
            while not self._hold.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            return ClassifierVerdict(flagged=False)
        return ClassifierVerdict(flagged=False)

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Native tests do not use the output path."""
        del completion, check
        raise AssertionError("output inspect is unused")


def test_native_callback_returns_while_adapter_ignores_cancellation() -> None:
    """A native worker returns at the timeout even when the inspect stays live."""
    rogue = _NativeCancelSwallow()
    healthy = ScriptedClassifier()
    inspects = BoundedInspect(max_inflight=2)
    rogue_policy = GuardrailPolicy(
        policy_id="rogue-policy",
        organization_id="organization-one",
        identity_id="identity-one",
        protected=True,
        checks=(
            GuardrailCheck(
                check_id="rogue-input",
                capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                stage=GuardrailCheckStage.INPUT,
                action=GuardrailAction.BLOCK,
                timeout_ms=40,
                adapter_id="rogue",
            ),
        ),
    )
    healthy_policy = GuardrailPolicy(
        policy_id="healthy-policy",
        organization_id="organization-one",
        identity_id="identity-healthy",
        checks=(
            GuardrailCheck(
                check_id="healthy-input",
                capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                stage=GuardrailCheckStage.INPUT,
                action=GuardrailAction.ALLOW,
                timeout_ms=200,
                adapter_id="healthy",
            ),
        ),
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore((rogue_policy, healthy_policy)),
        client=DirectClassifierClient(ClassifierRegistry({"rogue": rogue, "healthy": healthy})),
        monotonic=time.monotonic,
        inspects=inspects,
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )
    try:
        started = time.monotonic()
        with pytest.raises(GuardrailRejected):
            enforce_native_input(
                engine,
                authorization=_authorization(),
                request=request,
                deadline_monotonic=time.monotonic() + 30,
            )
        assert time.monotonic() - started < 0.5
        assert inspects.detached_inspect_count() == 1
        assert rogue.input_calls == 1

        started = time.monotonic()
        with pytest.raises(GuardrailRejected):
            enforce_native_input(
                engine,
                authorization=_authorization(),
                request=request,
                deadline_monotonic=time.monotonic() + 30,
            )
        assert time.monotonic() - started < 0.2
        assert inspects.detached_inspect_count() == 1
        assert rogue.input_calls == 1

        healthy_auth = _authorization().model_copy(update={"identity_id": "identity-healthy"})
        started = time.monotonic()
        rewritten, policy = enforce_native_input(
            engine,
            authorization=healthy_auth,
            request=request,
            deadline_monotonic=time.monotonic() + 30,
        )
        assert time.monotonic() - started < 0.2
        assert policy is not None
        assert rewritten.messages[0].content == "hello"
        assert healthy.input_calls == 1
    finally:
        rogue.release()
        for _ in range(50):
            if inspects.detached_inspect_count() == 0:
                break
            time.sleep(0.02)
        assert inspects.detached_inspect_count() == 0

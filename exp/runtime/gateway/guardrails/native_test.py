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
    OutputGuardrailMode,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native import (
    encode_output_decision,
    enforce_native_output,
    enforce_native_output_segment,
    native_output_mode,
    parse_output_payload,
    require_unguarded_surface,
    upstream_requests_reasoning,
)
from exp.runtime.gateway.guardrails.regex import (
    BuiltinPattern,
    RegexAdapterDocument,
    RegexClassifier,
)
from exp.runtime.gateway.guardrails.session import GuardrailSession
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.guardrails.store_test import _authorization
from exp.runtime.gateway.native_accounting_errors import NativeBridgeError
from exp.runtime.gateway.tests.mandatory_guardrails_test import _Guard


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


def _inspect_input(
    engine: GuardrailEngine | None,
    *,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
    deadline_monotonic: float,
) -> tuple[GatewayRequest, GuardrailSession | None]:
    """Open the request lifecycle and perform its input phase."""
    session = (
        None
        if engine is None
        else engine.open_request(authorization, deadline_monotonic=deadline_monotonic)
    )
    return (request if session is None else session.inspect_input(request), session)


def _session(
    engine: GuardrailEngine | None, policy: GuardrailPolicy | None, deadline: float = 200.0
) -> GuardrailSession | None:
    """Bind an explicit output policy fixture to the common lifecycle."""
    return (
        None
        if engine is None or policy is None
        else GuardrailSession(
            engine, (policy.model_copy(update={"checks": policy.output_checks}),), deadline
        )
    )


def test_unguarded_native_input_does_not_call_classifiers() -> None:
    """No engine, or no assigned policy, leaves the request unchanged."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )

    unchanged, policy = _inspect_input(
        None,
        authorization=_authorization(),
        request=request,
        deadline_monotonic=200.0,
    )

    assert unchanged is request
    assert policy is None


def test_pool_scoped_platform_policy_fences_only_its_target_pool() -> None:
    """A store that reads the alias target guards one exact-model pool and not another."""
    platform = GuardrailPolicy(policy_id="pool-content-safety", protected=True)

    class PoolScopedStore:
        """Apply the platform policy only to aliases that target the guarded pool."""

        def policies_for(self, authorization: AuthorizationSnapshot) -> tuple[GuardrailPolicy, ...]:
            """Bind the platform policy when the alias targets ``guarded-pool``."""
            target = authorization.target
            if not isinstance(target, DirectTarget) or target.pool_id != "guarded-pool":
                return ()
            return (platform.bind(authorization.organization_id, authorization.identity_id),)

    engine = GuardrailEngine(
        store=PoolScopedStore(),
        client=DirectClassifierClient(ClassifierRegistry()),
        monotonic=lambda: 0.0,
    )
    guarded = _authorization(pool_id="guarded-pool")
    unguarded = _authorization(pool_id="standard-pool")

    assert engine.open_request(guarded, deadline_monotonic=200.0) is not None
    assert engine.open_request(unguarded, deadline_monotonic=200.0) is None
    with pytest.raises(NativeBridgeError, match="do not support native images"):
        require_unguarded_surface(engine, guarded, "images")
    require_unguarded_surface(engine, unguarded, "images")


def test_mandatory_policy_checks_customer_rewrites_before_dispatch() -> None:
    """A customer-controlled replacement cannot introduce uninspected instructions."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="original request"),),
    )
    replacement = (GatewayMessage(role="user", content="rewritten request"),)
    policy = GuardrailPolicy(
        policy_id="rewrite",
        organization_id="organization-one",
        identity_id="identity-one",
        checks=(
            GuardrailCheck(
                check_id="rewrite-input",
                capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                stage=GuardrailCheckStage.INPUT,
                action=GuardrailAction.MODIFY,
                timeout_ms=100,
                adapter_id="scripted",
            ),
        ),
    )
    engine = _Guard(
        policies=(policy,),
        adapters={
            "scripted": ScriptedClassifier(
                input_verdict=ClassifierVerdict(flagged=True, replacement_messages=replacement)
            )
        },
    )
    transformed, session = _inspect_input(
        engine,
        authorization=_authorization(),
        request=request,
        deadline_monotonic=time.monotonic() + 10,
    )
    assert transformed.messages == replacement
    assert [item.messages for item in engine.requests] == [request.messages, replacement]
    assert session is not None
    assert len(session.policies) == 2


def test_native_input_runs_the_async_chain_on_a_private_loop() -> None:
    """The native callback awaits enforcement without a caller event loop."""
    engine = _engine()
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )

    rewritten, policy = _inspect_input(
        engine,
        authorization=_authorization(),
        request=request,
        deadline_monotonic=time.monotonic() + 10,
    )

    assert policy is not None
    assert rewritten.messages[0].content == "hello"


@pytest.mark.parametrize("blocked_stage", ["original", "replacement"])
def test_customer_rewrite_cannot_erase_or_introduce_blocked_intent(blocked_stage: str) -> None:
    """The mandatory policy sees both sides of a customer transformation."""

    class IntentGuard(_Guard):
        """Classify a synthetic intent marker through the shared adapter registry."""

        async def inspect_input(
            self, *, request: GatewayRequest, check: GuardrailCheck
        ) -> ClassifierVerdict:
            """Reject only the explicit synthetic marker while retaining observed input."""
            self.requests.append(request)
            return ClassifierVerdict(
                flagged=any(message.content == "blocked intent" for message in request.messages)
            )

    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(
                role="user", content="blocked intent" if blocked_stage == "original" else "safe"
            ),
        ),
    )
    customer = ScriptedClassifier(
        input_verdict=ClassifierVerdict(
            flagged=True,
            replacement_messages=(
                GatewayMessage(
                    role="user",
                    content="blocked intent" if blocked_stage == "replacement" else "safe",
                ),
            ),
        )
    )
    policy = _engine().policies_for(_authorization())[0]
    assert policy is not None
    policy = policy.model_copy(
        update={
            "checks": (
                policy.input_checks[0].model_copy(update={"action": GuardrailAction.MODIFY}),
            )
        }
    )
    engine = IntentGuard(policies=(policy,), adapters={"scripted": customer})
    with pytest.raises(GuardrailRejected) as rejected:
        _inspect_input(
            engine,
            authorization=_authorization(),
            request=request,
            deadline_monotonic=time.monotonic() + 10,
        )
    assert rejected.value.failure.failure_class.value == "guardrail"
    assert customer.input_calls == (1 if blocked_stage == "replacement" else 0)
    assert len(engine.requests) == (2 if blocked_stage == "replacement" else 1)


def test_native_output_payload_round_trips_tool_calls() -> None:
    """The JSON boundary carries tool-call arguments without rewriting them."""
    completion = parse_output_payload(
        {
            "text": "hello",
            "refusal": False,
            "tool_calls": [{"call_id": "call-1", "name": "lookup", "arguments": '{"q":"x"}'}],
        }
    )

    assert completion.tool_calls[0].arguments == '{"q":"x"}'


def test_native_output_block_returns_sanitized_failure_json() -> None:
    """A blocked output chain returns action plus a content-free failure."""
    engine = _engine()
    policy = engine.policies_for(_authorization())[0]
    decision = json.loads(
        enforce_native_output(
            _session(engine, policy, 200.0), json.dumps({"text": "unsafe", "tool_calls": []})
        )
    )

    assert decision["action"] == "block"
    assert decision["failure"]["failure_class"] == "guardrail"
    assert "unsafe" not in json.dumps(decision)


def test_missing_policy_on_output_callback_fail_closes() -> None:
    """Invoking the output callback without a captured policy is a terminal error."""
    decision = json.loads(
        enforce_native_output(_session(None, None, 200.0), json.dumps({"text": "ok"}))
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
            _inspect_input(
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
            _inspect_input(
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
        rewritten, policy = _inspect_input(
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


def test_output_segment_without_an_engine_fails_closed() -> None:
    """An unguarded process cannot release bytes through the streaming seam."""
    decision = json.loads(
        enforce_native_output_segment(
            _session(None, None, 1e12),
            json.dumps({"pending": "text", "final": True, "settled_bytes": 0}),
        )
    )
    assert decision["action"] == "error"
    assert "release" not in decision
    assert decision["failure"]["failure_class"] == "unavailable"


def test_output_segment_releases_redacted_text() -> None:
    """A deterministic chain returns redacted release text and the held tail."""
    detector = RegexClassifier(
        RegexAdapterDocument(
            adapter_id="detector",
            builtin_patterns=(BuiltinPattern.EMAIL,),
            stream_window_characters=64,
        )
    )
    policy = GuardrailPolicy(
        policy_id="member-policy",
        organization_id="organization-one",
        identity_id="identity-one",
        protected=True,
        checks=(
            GuardrailCheck(
                check_id="output-redact",
                capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                stage=GuardrailCheckStage.OUTPUT,
                action=GuardrailAction.MODIFY,
                timeout_ms=100,
                adapter_id="detector",
            ),
        ),
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"detector": detector})),
        monotonic=lambda: 0.0,
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="Draw"),),
        stream=True,
    )
    assert (
        native_output_mode(_session(engine, policy, 200.0), request) == OutputGuardrailMode.STREAM
    )
    assert (
        native_output_mode(_session(engine, policy, 200.0), request, image_output=True)
        == OutputGuardrailMode.BUFFER
    )
    assert (
        native_output_mode(_session(engine, None, 200.0), request, image_output=True)
        == OutputGuardrailMode.OFF
    )
    decision = json.loads(
        enforce_native_output_segment(
            _session(engine, policy, 1e12),
            json.dumps(
                {
                    "pending": "mail ada@example.com now " + "y" * 200 + " done ",
                    "final": False,
                    "settled_bytes": 0,
                }
            ),
        )
    )
    assert decision["action"] == "allow"
    assert "ada@example.com" not in decision["release"]
    assert decision["flagged"] is True


def test_upstream_reasoning_defaults_rule_out_incremental_streaming() -> None:
    """A payload that asks for a summary or a thinking display is not reasoning-free."""
    assert upstream_requests_reasoning([{"upstream_payload": {"reasoning": {"summary": "auto"}}}])
    assert upstream_requests_reasoning(
        [{"upstream_payload": {"thinking": {"type": "adaptive", "display": "summarized"}}}]
    )
    assert not upstream_requests_reasoning(
        [{"upstream_payload": {"reasoning": {"effort": "high"}}}, {"upstream_payload": None}]
    )

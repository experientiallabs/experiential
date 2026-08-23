"""Native-boundary JSON contract tests."""

from __future__ import annotations

import json

from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, ScriptedClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailPolicy,
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
        store=MappingGuardrailStore({"identity-one": policy}),
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
    policy = engine.policy_for("identity-one")
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

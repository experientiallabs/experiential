"""Contract tests for immutable identity-scoped guardrail policies."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailPolicy,
    GuardrailToolCall,
    request_content_bytes,
)


def _check(
    check_id: str,
    *,
    stage: GuardrailCheckStage = GuardrailCheckStage.INPUT,
    action: GuardrailAction = GuardrailAction.BLOCK,
) -> GuardrailCheck:
    """Build one valid check with the requested stage and action."""
    return GuardrailCheck(
        check_id=check_id,
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=stage,
        action=action,
        timeout_ms=250,
        adapter_id="keyword-safety",
    )


def test_policy_splits_input_and_output_checks_in_authored_order() -> None:
    """Input and output chains keep authored order and unique check IDs."""
    policy = GuardrailPolicy(
        policy_id="member-policy",
        organization_id="organization-one",
        identity_id="identity-one",
        checks=(
            _check("input-one"),
            _check("output-one", stage=GuardrailCheckStage.OUTPUT),
            _check("input-two"),
        ),
    )

    assert [check.check_id for check in policy.input_checks] == ["input-one", "input-two"]
    assert [check.check_id for check in policy.output_checks] == ["output-one"]


def test_policy_rejects_duplicate_check_ids() -> None:
    """Repeated check IDs make chain order ambiguous and fail closed."""
    with pytest.raises(ValueError, match="unique"):
        GuardrailPolicy(
            policy_id="member-policy",
            organization_id="organization-one",
            identity_id="identity-one",
            checks=(_check("same-check"), _check("same-check")),
        )


def test_capability_kinds_name_jobs_not_providers() -> None:
    """Capability names stay provider-neutral."""
    assert {item.value for item in GuardrailCapabilityKind} == {
        "pii",
        "secret_leakage",
        "prompt_injection",
        "content_safety",
    }


def test_prompt_injection_is_input_only() -> None:
    """Output-stage prompt injection is rejected at check construction."""
    with pytest.raises(ValueError, match="input-only"):
        GuardrailCheck(
            check_id="output-injection",
            capability=GuardrailCapabilityKind.PROMPT_INJECTION,
            stage=GuardrailCheckStage.OUTPUT,
            action=GuardrailAction.BLOCK,
            timeout_ms=250,
            adapter_id="hosted-injection",
        )


def test_policy_allows_repeated_stage_capability_pairs() -> None:
    """Manual chains may run two classifiers for the same capability and stage."""
    policy = GuardrailPolicy(
        policy_id="member-policy",
        organization_id="organization-one",
        identity_id="identity-one",
        checks=(
            GuardrailCheck(
                check_id="input-pii-one",
                capability=GuardrailCapabilityKind.PII,
                stage=GuardrailCheckStage.INPUT,
                action=GuardrailAction.MODIFY,
                timeout_ms=250,
                adapter_id="hosted-pii-one",
            ),
            GuardrailCheck(
                check_id="input-pii-two",
                capability=GuardrailCapabilityKind.PII,
                stage=GuardrailCheckStage.INPUT,
                action=GuardrailAction.MODIFY,
                timeout_ms=250,
                adapter_id="hosted-pii-two",
            ),
        ),
    )

    assert [check.check_id for check in policy.input_checks] == [
        "input-pii-one",
        "input-pii-two",
    ]


def test_request_content_bytes_count_the_compact_json_subject() -> None:
    """The request bound is the compact JSON sent to classifiers, not message text."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="user", content="hi"),
            GatewayMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        call_id="call-1",
                        name="lookup",
                        arguments={"q": "ab"},
                        raw_arguments='{"q":"ab"}',
                    ),
                ),
            ),
        ),
        tools=(
            GatewayToolDefinition(
                name="lookup",
                description="look up a record",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            ),
        ),
    )
    completion = GuardrailCompletion(
        text="ok",
        tool_calls=(GuardrailToolCall(call_id="call-1", name="lookup", arguments='{"q":"ab"}'),),
    )

    assert request_content_bytes(request) == len(canonical_json_bytes(request))
    assert request_content_bytes(request) > len("hi") + len('{"q":"ab"}')
    assert completion.content_bytes() == len("ok") + len('{"q":"ab"}')

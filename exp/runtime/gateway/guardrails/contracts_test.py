"""Contract tests for immutable identity-scoped guardrail policies."""

from __future__ import annotations

import pytest

from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
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


def test_policy_rejects_duplicate_stage_capability_pairs() -> None:
    """Two checks for the same capability on one stage are ambiguous."""
    with pytest.raises(ValueError, match="same stage"):
        GuardrailPolicy(
            policy_id="member-policy",
            organization_id="organization-one",
            identity_id="identity-one",
            checks=(
                _check("input-one"),
                GuardrailCheck(
                    check_id="input-two",
                    capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                    stage=GuardrailCheckStage.INPUT,
                    action=GuardrailAction.BLOCK,
                    timeout_ms=250,
                    adapter_id="keyword-safety",
                ),
            ),
        )


def test_request_and_completion_byte_counts_cover_tool_arguments() -> None:
    """Byte bounds include tool-call arguments, not only visible text."""
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
    )
    completion = GuardrailCompletion(
        text="ok",
        tool_calls=(GuardrailToolCall(call_id="call-1", name="lookup", arguments='{"q":"ab"}'),),
    )

    assert request_content_bytes(request) == len("hi") + len('{"q":"ab"}')
    assert completion.content_bytes() == len("ok") + len('{"q":"ab"}')

"""Contract tests for immutable identity-scoped guardrail policies."""

from __future__ import annotations

from typing import Never

import pytest

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.guardrails import contracts
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailPolicy,
    GuardrailToolCall,
    request_content_bytes,
    request_exceeds_inspection_limit,
)
from exp.runtime.gateway.guardrails.subjects import observation_subject_bytes


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


@pytest.mark.parametrize(
    ("stage", "action"),
    [
        (GuardrailCheckStage.OUTPUT, GuardrailAction.BLOCK),
        (GuardrailCheckStage.INPUT, GuardrailAction.MODIFY),
    ],
)
def test_observation_rejects_output_and_rewrite_checks(
    stage: GuardrailCheckStage, action: GuardrailAction
) -> None:
    """Observation cannot promise output coverage or mutate the provider's input."""
    with pytest.raises(ValueError, match="observation requires read-only input checks"):
        GuardrailPolicy(
            policy_id="platform-observer",
            protected=True,
            mode="observe",
            checks=(_check("observer", stage=stage, action=action),),
        )


def test_observation_preserves_protected_platform_authority() -> None:
    """Rollout mode is separate from the policy's ownership and authored block action."""
    policy = GuardrailPolicy(
        policy_id="platform-observer",
        protected=True,
        mode="observe",
        checks=(_check("observer"),),
    )
    assert policy.mode == "observe"
    assert policy.bind("org", "identity").mode == "observe"
    assert policy.checks[0].action is GuardrailAction.BLOCK


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
    assert completion.content_bytes() == len(canonical_json_bytes(completion))


@pytest.mark.parametrize("text", ["benign", "\x00" * 30, "\U0001f642" * 30])
def test_inspection_bound_counts_private_context_at_its_exact_encoded_size(text: str) -> None:
    """The shared bound counts hidden carriers and JSON/UTF-8 expansion without changing HTTP."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(
            GatewayToolDefinition(
                name="lookup",
                parameters={},
                input_examples=({"nested": [text, None, True, 4]},),
                allowed_callers=(text,),
            ),
        ),
    )
    complete_size = len(observation_subject_bytes(request))
    assert request_content_bytes(request) == len(canonical_json_bytes(request))
    assert request_content_bytes(request) < complete_size
    assert not request_exceeds_inspection_limit(request, complete_size)
    assert request_exceeds_inspection_limit(request, complete_size - 1)


def test_oversized_private_context_is_rejected_before_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An attacker-sized private example cannot allocate an encoded copy during admission."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(
            GatewayToolDefinition(
                name="lookup", parameters={}, input_examples=({"private": "x" * 65_536},)
            ),
        ),
    )

    def forbidden_encoding(_request: GatewayRequest) -> Never:
        """Fail if the early character bound permits a large private allocation."""
        raise AssertionError("oversized private context reached encoding")

    monkeypatch.setattr(contracts, "observation_subject_bytes", forbidden_encoding)
    assert request_exceeds_inspection_limit(request, 4096)


@pytest.mark.parametrize("length", [257, 65_536])
def test_output_guardrail_preserves_long_tool_id(length: int) -> None:
    """An output check accepts every bounded tool identifier emitted by the engine."""
    call = GuardrailToolCall(call_id="x" * length, name="terminal", arguments="{}")
    assert GuardrailCompletion(tool_calls=(call,)).tool_calls[0].call_id == "x" * length

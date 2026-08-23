"""Tests for one-shot output buffering."""

from __future__ import annotations

from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailPolicy,
)
from exp.runtime.gateway.guardrails.delivery import requires_output_buffer


def test_output_buffer_is_required_only_when_output_checks_exist() -> None:
    """Unguarded policies and input-only policies skip response buffering."""
    input_only = GuardrailPolicy(
        policy_id="member-policy",
        organization_id="organization-one",
        identity_id="identity-one",
        checks=(
            GuardrailCheck(
                check_id="input-safety",
                capability=GuardrailCapabilityKind.PII,
                stage=GuardrailCheckStage.INPUT,
                action=GuardrailAction.BLOCK,
                timeout_ms=50,
                adapter_id="scripted",
            ),
        ),
    )
    with_output = input_only.model_copy(
        update={
            "checks": (
                *input_only.checks,
                GuardrailCheck(
                    check_id="output-safety",
                    capability=GuardrailCapabilityKind.PII,
                    stage=GuardrailCheckStage.OUTPUT,
                    action=GuardrailAction.BLOCK,
                    timeout_ms=50,
                    adapter_id="scripted",
                ),
            )
        }
    )

    assert requires_output_buffer(None) is False
    assert requires_output_buffer(input_only) is False
    assert requires_output_buffer(with_output) is True

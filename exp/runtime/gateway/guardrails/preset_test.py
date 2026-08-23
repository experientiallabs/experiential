"""Tests for standard guardrail preset expansion and load-time rejection."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheckStage,
)
from exp.runtime.gateway.guardrails.preset import (
    STANDARD_DEFAULT_TIMEOUT_MS,
    STANDARD_PRESET_STEPS,
    expand_standard_checks,
    policy_from_authored,
)

_ADAPTERS = frozenset(
    {
        "hosted-pii",
        "hosted-secrets",
        "hosted-injection",
        "hosted-safety",
    }
)
_BINDINGS = {
    "pii": "hosted-pii",
    "secret_leakage": "hosted-secrets",
    "prompt_injection": "hosted-injection",
    "content_safety": "hosted-safety",
}


def _standard_item(**overrides: object) -> dict[str, object]:
    """Return one valid standard-preset policy object."""
    item: dict[str, object] = {
        "policy_id": "standard-member",
        "organization_id": "organization-one",
        "identity_id": "identity-one",
        "protected": True,
        "preset": "standard",
        "capability_adapters": dict(_BINDINGS),
    }
    item.update(overrides)
    return item


def test_standard_preset_expands_in_documented_order_and_actions() -> None:
    """The pack is input redact, input block, output redact, output block."""
    checks = expand_standard_checks(
        capability_adapters=_BINDINGS,
        adapter_ids=_ADAPTERS,
    )

    assert [check.check_id for check in checks] == [step.check_id for step in STANDARD_PRESET_STEPS]
    assert [(check.stage, check.capability, check.action) for check in checks] == [
        (GuardrailCheckStage.INPUT, GuardrailCapabilityKind.PII, GuardrailAction.MODIFY),
        (
            GuardrailCheckStage.INPUT,
            GuardrailCapabilityKind.SECRET_LEAKAGE,
            GuardrailAction.MODIFY,
        ),
        (
            GuardrailCheckStage.INPUT,
            GuardrailCapabilityKind.PROMPT_INJECTION,
            GuardrailAction.BLOCK,
        ),
        (
            GuardrailCheckStage.INPUT,
            GuardrailCapabilityKind.CONTENT_SAFETY,
            GuardrailAction.BLOCK,
        ),
        (GuardrailCheckStage.OUTPUT, GuardrailCapabilityKind.PII, GuardrailAction.MODIFY),
        (
            GuardrailCheckStage.OUTPUT,
            GuardrailCapabilityKind.SECRET_LEAKAGE,
            GuardrailAction.MODIFY,
        ),
        (
            GuardrailCheckStage.OUTPUT,
            GuardrailCapabilityKind.CONTENT_SAFETY,
            GuardrailAction.BLOCK,
        ),
    ]
    assert all(check.timeout_ms == STANDARD_DEFAULT_TIMEOUT_MS for check in checks)
    assert [check.adapter_id for check in checks] == [
        "hosted-pii",
        "hosted-secrets",
        "hosted-injection",
        "hosted-safety",
        "hosted-pii",
        "hosted-secrets",
        "hosted-safety",
    ]


def test_standard_preset_is_input_only_for_prompt_injection() -> None:
    """The expanded pack never schedules output-stage prompt injection."""
    checks = expand_standard_checks(
        capability_adapters=_BINDINGS,
        adapter_ids=_ADAPTERS,
    )

    assert not any(
        check.capability is GuardrailCapabilityKind.PROMPT_INJECTION
        and check.stage is GuardrailCheckStage.OUTPUT
        for check in checks
    )


def test_timeout_overrides_apply_by_check_id_and_stage_capability() -> None:
    """Operators can raise one check above the conservative 250 ms default."""
    checks = expand_standard_checks(
        capability_adapters=_BINDINGS,
        timeout_ms=250,
        timeouts={"standard-input-pii": 100, "output.content_safety": 400},
        adapter_ids=_ADAPTERS,
    )
    by_id = {check.check_id: check.timeout_ms for check in checks}

    assert by_id["standard-input-pii"] == 100
    assert by_id["standard-output-content-safety"] == 400
    assert by_id["standard-input-prompt-injection"] == 250


def test_ambiguous_timeout_keys_are_rejected() -> None:
    """The same check cannot be overridden twice under different keys."""
    with pytest.raises(ValueError, match="ambiguous timeout"):
        expand_standard_checks(
            capability_adapters=_BINDINGS,
            timeouts={"standard-input-pii": 80, "input.pii": 90},
            adapter_ids=_ADAPTERS,
        )


def test_missing_capability_binding_is_rejected() -> None:
    """Every standard capability must be bound to a registered adapter."""
    incomplete = dict(_BINDINGS)
    del incomplete["content_safety"]
    with pytest.raises(ValueError, match="missing content_safety"):
        expand_standard_checks(
            capability_adapters=incomplete,
            adapter_ids=_ADAPTERS,
        )


def test_unregistered_capability_adapter_is_rejected() -> None:
    """A binding that names an adapter outside the document fails closed."""
    with pytest.raises(ValueError, match="unknown adapter_id"):
        expand_standard_checks(
            capability_adapters={**_BINDINGS, "pii": "missing-pii"},
            adapter_ids=_ADAPTERS,
        )


def test_preset_combined_with_manual_checks_is_rejected() -> None:
    """Mixing a preset and a check list is ambiguous at load time."""
    with pytest.raises(ValueError, match="cannot be combined"):
        policy_from_authored(
            _standard_item(
                checks=[
                    {
                        "check_id": "input-safety",
                        "capability": "content_safety",
                        "stage": "input",
                        "action": "block",
                        "timeout_ms": 250,
                        "adapter_id": "hosted-safety",
                    }
                ]
            ),
            _ADAPTERS,
        )


def test_capability_adapters_without_preset_are_rejected() -> None:
    """Bindings without an explicit standard opt-in are not implied."""
    with pytest.raises(ValueError, match="requires the standard preset"):
        policy_from_authored(
            {
                "policy_id": "member-policy",
                "organization_id": "organization-one",
                "identity_id": "identity-one",
                "capability_adapters": dict(_BINDINGS),
            },
            _ADAPTERS,
        )


def test_unknown_preset_name_is_rejected() -> None:
    """Only the documented standard pack is defined."""
    with pytest.raises(ValueError, match="unknown guardrail preset"):
        policy_from_authored(_standard_item(preset="strict"), _ADAPTERS)


def test_preset_without_bindings_is_rejected() -> None:
    """An opt-in without capability adapters is incomplete."""
    item = _standard_item()
    del item["capability_adapters"]
    with pytest.raises(ValueError, match="adapter_id for every capability"):
        policy_from_authored(item, _ADAPTERS)


def test_manual_check_with_unknown_adapter_is_rejected() -> None:
    """Hand-authored checks must name an adapter from the same document."""
    with pytest.raises(ValueError, match="unknown adapter_id"):
        policy_from_authored(
            {
                "policy_id": "member-policy",
                "organization_id": "organization-one",
                "identity_id": "identity-one",
                "checks": [
                    {
                        "check_id": "input-safety",
                        "capability": "content_safety",
                        "stage": "input",
                        "action": "block",
                        "timeout_ms": 250,
                        "adapter_id": "missing-safety",
                    }
                ],
            },
            _ADAPTERS,
        )

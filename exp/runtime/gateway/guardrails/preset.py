"""Deterministic standard guardrail preset expansion at configuration load."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from pydantic import Field, ValidationError

from exp.common.core.artifacts import ArtifactId, ContractModel
from exp.runtime.gateway.contracts import IdentityId, OrganizationId
from exp.runtime.gateway.guardrails.contracts import (
    DEFAULT_MAX_REQUEST_BYTES,
    DEFAULT_MAX_RESPONSE_BYTES,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailPolicy,
)

STANDARD_PRESET_NAME: Final = "standard"
STANDARD_DEFAULT_TIMEOUT_MS: Final = 250
STANDARD_REQUIRED_CAPABILITIES: Final[frozenset[GuardrailCapabilityKind]] = frozenset(
    {
        GuardrailCapabilityKind.PII,
        GuardrailCapabilityKind.SECRET_LEAKAGE,
        GuardrailCapabilityKind.PROMPT_INJECTION,
        GuardrailCapabilityKind.CONTENT_SAFETY,
    }
)


class StandardPresetStep(ContractModel):
    """One documented check in the standard pack, in expansion order."""

    check_id: ArtifactId
    capability: GuardrailCapabilityKind
    stage: GuardrailCheckStage
    action: GuardrailAction


STANDARD_PRESET_STEPS: Final[tuple[StandardPresetStep, ...]] = (
    StandardPresetStep(
        check_id="standard-input-pii",
        capability=GuardrailCapabilityKind.PII,
        stage=GuardrailCheckStage.INPUT,
        action=GuardrailAction.MODIFY,
    ),
    StandardPresetStep(
        check_id="standard-input-secret-leakage",
        capability=GuardrailCapabilityKind.SECRET_LEAKAGE,
        stage=GuardrailCheckStage.INPUT,
        action=GuardrailAction.MODIFY,
    ),
    StandardPresetStep(
        check_id="standard-input-prompt-injection",
        capability=GuardrailCapabilityKind.PROMPT_INJECTION,
        stage=GuardrailCheckStage.INPUT,
        action=GuardrailAction.BLOCK,
    ),
    StandardPresetStep(
        check_id="standard-input-content-safety",
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=GuardrailCheckStage.INPUT,
        action=GuardrailAction.BLOCK,
    ),
    StandardPresetStep(
        check_id="standard-output-pii",
        capability=GuardrailCapabilityKind.PII,
        stage=GuardrailCheckStage.OUTPUT,
        action=GuardrailAction.MODIFY,
    ),
    StandardPresetStep(
        check_id="standard-output-secret-leakage",
        capability=GuardrailCapabilityKind.SECRET_LEAKAGE,
        stage=GuardrailCheckStage.OUTPUT,
        action=GuardrailAction.MODIFY,
    ),
    StandardPresetStep(
        check_id="standard-output-content-safety",
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=GuardrailCheckStage.OUTPUT,
        action=GuardrailAction.BLOCK,
    ),
)

STANDARD_CHECK_IDS: Final[frozenset[str]] = frozenset(
    step.check_id for step in STANDARD_PRESET_STEPS
)
_STAGE_CAPABILITY_TO_CHECK_ID: Final[dict[tuple[str, str], str]] = {
    (step.stage.value, step.capability.value): step.check_id for step in STANDARD_PRESET_STEPS
}


class AuthoredManualPolicy(ContractModel):
    """Hand-authored identity policy with an explicit check list."""

    policy_id: ArtifactId
    organization_id: OrganizationId
    identity_id: IdentityId
    protected: bool = False
    checks: tuple[GuardrailCheck, ...] = ()
    max_request_bytes: int = Field(default=DEFAULT_MAX_REQUEST_BYTES, ge=1, le=64 * 1024 * 1024)
    max_response_bytes: int = Field(default=DEFAULT_MAX_RESPONSE_BYTES, ge=1, le=64 * 1024 * 1024)


class AuthoredStandardPolicy(ContractModel):
    """Identity policy that opts into the documented standard pack."""

    policy_id: ArtifactId
    organization_id: OrganizationId
    identity_id: IdentityId
    protected: bool = False
    preset: str
    timeout_ms: int = Field(default=STANDARD_DEFAULT_TIMEOUT_MS, ge=1, le=30_000)
    timeouts: dict[str, int] = Field(default_factory=dict)
    capability_adapters: dict[str, ArtifactId]
    max_request_bytes: int = Field(default=DEFAULT_MAX_REQUEST_BYTES, ge=1, le=64 * 1024 * 1024)
    max_response_bytes: int = Field(default=DEFAULT_MAX_RESPONSE_BYTES, ge=1, le=64 * 1024 * 1024)


def policy_from_authored(
    item: Mapping[str, object],
    adapter_ids: frozenset[str],
) -> GuardrailPolicy:
    """Validate one authored policy object and expand a preset when requested.

    The standard pack is never implied. An identity opts in by setting
    ``preset`` to ``standard`` and binding every required capability to a
    registered ``adapter_id``.

    Args:
        item: One policy object from ``guardrails.json``.
        adapter_ids: Adapter identities registered in the same document.

    Returns:
        An immutable policy whose checks the engine can run in order.

    Raises:
        ValueError: The object is malformed, mixed ambiguously, or unbound.
    """
    has_preset = _has_value(item.get("preset"))
    has_checks = _has_value(item.get("checks"))
    has_bindings = _has_value(item.get("capability_adapters"))
    if has_preset and has_checks:
        raise ValueError("standard preset cannot be combined with authored checks")
    if has_bindings and not has_preset:
        raise ValueError("capability_adapters requires the standard preset")
    if has_preset and not has_bindings:
        raise ValueError("standard preset requires an adapter_id for every capability")
    if "timeouts" in item and not has_preset:
        raise ValueError("timeouts requires the standard preset")
    if "timeout_ms" in item and not has_preset:
        raise ValueError("timeout_ms requires the standard preset")
    if has_preset:
        return _expand_standard(item, adapter_ids)
    manual = dict(item)
    manual.pop("preset", None)
    manual.pop("capability_adapters", None)
    manual.pop("timeouts", None)
    manual.pop("timeout_ms", None)
    return _manual_policy(manual, adapter_ids)


def expand_standard_checks(
    *,
    capability_adapters: Mapping[str, str],
    timeout_ms: int = STANDARD_DEFAULT_TIMEOUT_MS,
    timeouts: Mapping[str, int] | None = None,
    adapter_ids: frozenset[str],
) -> tuple[GuardrailCheck, ...]:
    """Expand the standard pack into ordered checks.

    Args:
        capability_adapters: Explicit adapter identity for every capability.
        timeout_ms: Default per-check timeout used when no override is set.
        timeouts: Optional overrides keyed by check ID or ``stage.capability``.
        adapter_ids: Adapter identities that exist in the same document.

    Returns:
        The seven standard checks in documented order.

    Raises:
        ValueError: Bindings, adapters, or timeout keys are malformed.
    """
    adapters = _bound_adapters(capability_adapters, adapter_ids)
    overrides = _resolved_timeouts(timeouts or {})
    return tuple(
        GuardrailCheck(
            check_id=step.check_id,
            capability=step.capability,
            stage=step.stage,
            action=step.action,
            timeout_ms=overrides.get(step.check_id, timeout_ms),
            adapter_id=adapters[step.capability],
        )
        for step in STANDARD_PRESET_STEPS
    )


def _expand_standard(item: Mapping[str, object], adapter_ids: frozenset[str]) -> GuardrailPolicy:
    """Parse a standard-preset policy object and expand its checks."""
    try:
        authored = AuthoredStandardPolicy.model_validate(item)
    except ValidationError as exc:
        raise ValueError("standard guardrail preset is malformed") from exc
    if authored.preset != STANDARD_PRESET_NAME:
        raise ValueError("unknown guardrail preset; only standard is defined")
    checks = expand_standard_checks(
        capability_adapters=authored.capability_adapters,
        timeout_ms=authored.timeout_ms,
        timeouts=authored.timeouts,
        adapter_ids=adapter_ids,
    )
    return GuardrailPolicy(
        policy_id=authored.policy_id,
        organization_id=authored.organization_id,
        identity_id=authored.identity_id,
        protected=authored.protected,
        checks=checks,
        max_request_bytes=authored.max_request_bytes,
        max_response_bytes=authored.max_response_bytes,
    )


def _manual_policy(item: Mapping[str, object], adapter_ids: frozenset[str]) -> GuardrailPolicy:
    """Parse a hand-authored policy and require every adapter to be registered."""
    try:
        authored = AuthoredManualPolicy.model_validate(item)
    except ValidationError as exc:
        raise ValueError("guardrail policy is malformed") from exc
    missing = sorted(
        {check.adapter_id for check in authored.checks if check.adapter_id not in adapter_ids}
    )
    if missing:
        raise ValueError(
            "guardrail checks reference unknown adapter_id values: " + ", ".join(missing)
        )
    return GuardrailPolicy(
        policy_id=authored.policy_id,
        organization_id=authored.organization_id,
        identity_id=authored.identity_id,
        protected=authored.protected,
        checks=authored.checks,
        max_request_bytes=authored.max_request_bytes,
        max_response_bytes=authored.max_response_bytes,
    )


def _bound_adapters(
    capability_adapters: Mapping[str, str],
    adapter_ids: frozenset[str],
) -> dict[GuardrailCapabilityKind, ArtifactId]:
    """Require an explicit, registered adapter for every standard capability."""
    unknown = sorted(set(capability_adapters) - {item.value for item in GuardrailCapabilityKind})
    if unknown:
        raise ValueError("unknown capability binding: " + ", ".join(unknown))
    bound: dict[GuardrailCapabilityKind, ArtifactId] = {}
    for name, adapter_id in capability_adapters.items():
        bound[GuardrailCapabilityKind(name)] = adapter_id
    missing = sorted(
        capability.value for capability in STANDARD_REQUIRED_CAPABILITIES if capability not in bound
    )
    if missing:
        raise ValueError(
            "standard preset requires an adapter_id for every capability; missing "
            + ", ".join(missing)
        )
    extra = sorted(
        capability.value for capability in bound if capability not in STANDARD_REQUIRED_CAPABILITIES
    )
    if extra:
        raise ValueError("standard preset has unexpected capability bindings: " + ", ".join(extra))
    unregistered = sorted(
        adapter_id for adapter_id in bound.values() if adapter_id not in adapter_ids
    )
    if unregistered:
        raise ValueError(
            "standard preset binds unknown adapter_id values: " + ", ".join(unregistered)
        )
    return bound


def _resolved_timeouts(timeouts: Mapping[str, int]) -> dict[str, int]:
    """Map authored timeout keys onto unique standard check IDs."""
    resolved: dict[str, int] = {}
    for key, value in timeouts.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > 30_000:
            raise ValueError("preset timeouts must be integers from 1 to 30000")
        check_id = _timeout_check_id(key)
        if check_id in resolved:
            raise ValueError(f"ambiguous timeout override for {check_id}")
        resolved[check_id] = value
    return resolved


def _timeout_check_id(key: str) -> str:
    """Resolve one timeout key to a standard check identity."""
    if key in STANDARD_CHECK_IDS:
        return key
    if "." in key:
        stage, capability = key.split(".", 1)
        check_id = _STAGE_CAPABILITY_TO_CHECK_ID.get((stage, capability))
        if check_id is not None:
            return check_id
    raise ValueError(f"unknown standard preset timeout key: {key}")


def _has_value(value: object) -> bool:
    """Return whether an authored field is present and non-empty."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, dict, tuple)):
        return bool(value)
    return True

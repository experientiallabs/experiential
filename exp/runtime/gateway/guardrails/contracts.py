"""Immutable scoped guardrail policy and decision contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, JsonObject, canonical_json_bytes
from exp.common.models.model import MAXIMUM_TOOL_CALL_ID_CHARACTERS
from exp.runtime.gateway.contracts import (
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    IdentityId,
    OrganizationId,
)
from exp.runtime.gateway.guardrails.subjects import (
    observation_subject_bytes,
    observation_subject_exceeds,
)

DEFAULT_MAX_REQUEST_BYTES = 1_048_576
DEFAULT_MAX_RESPONSE_BYTES = 1_048_576


class GuardrailCapabilityKind(StrEnum):
    """Classifier capability a policy check may request.

    These names describe the inspection job. They do not name a vendor or a
    hosted detector, so operators can bind any adapter that implements the
    matching capability.
    """

    PII = "pii"
    SECRET_LEAKAGE = "secret_leakage"
    PROMPT_INJECTION = "prompt_injection"
    CONTENT_SAFETY = "content_safety"


class GuardrailAction(StrEnum):
    """Outcome a check may apply after one classifier verdict."""

    ALLOW = "allow"
    MODIFY = "modify"
    BLOCK = "block"
    ERROR = "error"


class GuardrailCheckStage(StrEnum):
    """Whether one check inspects the canonical request or the winning completion."""

    INPUT = "input"
    OUTPUT = "output"


class OutputGuardrailMode(StrEnum):
    """How the data plane must enforce one admission's output chain.

    ``OFF`` means the identity has no output check, so bytes flow untouched.
    ``BUFFER`` means the whole winning completion is collected and inspected
    once before any caller byte, which is the only safe shape for a check
    that can block or for a detector that needs the full text. ``STREAM``
    means the completion is redacted incrementally as it arrives, with only a
    bounded trailing window held back.
    """

    OFF = "off"
    BUFFER = "buffer"
    STREAM = "stream"


class GuardrailCheck(ContractModel):
    """One ordered, timed inspection step in a policy chain."""

    check_id: ArtifactId
    capability: GuardrailCapabilityKind
    stage: GuardrailCheckStage
    action: GuardrailAction
    timeout_ms: int = Field(ge=1, le=30_000)
    adapter_id: ArtifactId

    @model_validator(mode="after")
    def _reject_output_prompt_injection(self) -> GuardrailCheck:
        """Prompt injection inspects inbound text only.

        Returns:
            The validated check.

        Raises:
            ValueError: The check asks for output-stage prompt injection.
        """
        if (
            self.stage is GuardrailCheckStage.OUTPUT
            and self.capability is GuardrailCapabilityKind.PROMPT_INJECTION
        ):
            raise ValueError("prompt_injection is input-only")
        return self


class GuardrailPolicy(ContractModel):
    """Immutable guardrail policy, scoped to the platform or one identity.

    Omit both scope IDs for a protected platform policy. Scoped identity lookup
    is ``(organization_id, identity_id)``. Identities are unique only
    inside an organization, so the same identity ID in two organizations
    cannot share a policy. ``protected`` fail-closes on adapter timeout,
    missing adapter, oversized payload, or any other classifier uncertainty.
    Non-protected identities skip a failed check and continue the remaining
    chain.

    Attributes:
        policy_id: Unique operator-authored policy identity.
        organization_id: Identity's organization, or None for platform scope.
        identity_id: Assigned identity, or None for platform scope.
        revision: Detector and rollout revision used to invalidate cached replay.
        input_execution: Inspect before dispatch by default, or overlap safe generation.
        mode: Enforce by default, or observe read-only input checks without gating delivery.
        protected: Fail closed on classifier uncertainty; required for platform policies.
        checks: Ordered checks, empty by default.
        max_request_bytes: Complete in-memory input inspection bound, default 1 MiB.
        max_response_bytes: Total canonical output inspection bound, default 1 MiB.
    """

    policy_id: ArtifactId
    organization_id: OrganizationId | None = None
    identity_id: IdentityId | None = None
    revision: str = Field(default="configured", min_length=1, max_length=256)
    input_execution: Literal["before_dispatch", "parallel"] = "before_dispatch"
    mode: Literal["enforce", "observe"] = "enforce"
    protected: bool = False
    checks: tuple[GuardrailCheck, ...] = ()
    max_request_bytes: int = Field(default=DEFAULT_MAX_REQUEST_BYTES, ge=1, le=64 * 1024 * 1024)
    max_response_bytes: int = Field(default=DEFAULT_MAX_RESPONSE_BYTES, ge=1, le=64 * 1024 * 1024)

    @model_validator(mode="after")
    def _require_unique_check_ids(self) -> GuardrailPolicy:
        """Reject repeated check identities that would make chain order ambiguous.

        Returns:
            The validated policy.

        Raises:
            ValueError: A check_id appears more than once.
        """
        ids = tuple(check.check_id for check in self.checks)
        if len(set(ids)) != len(ids):
            raise ValueError("guardrail check IDs must be unique")
        if (self.organization_id is None) != (self.identity_id is None):
            raise ValueError("guardrail scope requires both organization_id and identity_id")
        if self.organization_id is None and not self.protected:
            raise ValueError("platform guardrail policies must be protected")
        if self.mode == "observe" and any(
            check.stage is GuardrailCheckStage.OUTPUT or check.action is GuardrailAction.MODIFY
            for check in self.checks
        ):
            raise ValueError("observation requires read-only input checks")
        return self

    def bind(self, organization_id: OrganizationId, identity_id: IdentityId) -> GuardrailPolicy:
        """Capture authenticated decision metadata without changing checks or execution mode."""
        return self.model_copy(
            update={"organization_id": organization_id, "identity_id": identity_id}
        )

    @property
    def input_checks(self) -> tuple[GuardrailCheck, ...]:
        """Return input-stage checks in authored order."""
        return tuple(check for check in self.checks if check.stage is GuardrailCheckStage.INPUT)

    @property
    def output_checks(self) -> tuple[GuardrailCheck, ...]:
        """Return output-stage checks in authored order."""
        return tuple(check for check in self.checks if check.stage is GuardrailCheckStage.OUTPUT)


class GuardrailToolCall(ContractModel):
    """One completed tool invocation presented to an output check."""

    call_id: str = Field(min_length=1, max_length=MAXIMUM_TOOL_CALL_ID_CHARACTERS)
    name: str = Field(min_length=1, max_length=256)
    arguments: str = Field(max_length=4_000_000)


class GuardrailCompletion(ContractModel):
    """Normalized output subject, including readable alternate channels in ``context``.

    Attributes:
        text: Collected visible text, empty when the provider emits no text.
        refusal: Whether the provider refused the request, false by default.
        tool_calls: Completed tool invocations, empty when none were emitted.
        context: Readable alternate output channels, empty when none were emitted.
    """

    text: str = ""
    refusal: bool = False
    tool_calls: tuple[GuardrailToolCall, ...] = ()
    context: tuple[str, ...] = ()

    def content_bytes(self) -> int:
        """Return the UTF-8 size of the complete serialized classifier subject.

        Include opaque tool IDs, names, and JSON framing as well as arguments
        so the policy bounds the actual completion sent to the classifier.
        """
        return len(canonical_json_bytes(self))


class ClassifierVerdict(ContractModel):
    """Content-free classifier result plus optional in-memory replacement.

    Replacements stay on this object only for the remainder of the request.
    They are never logged or persisted.
    """

    flagged: bool
    replacement_text: str | None = None
    replacement_messages: tuple[GatewayMessage, ...] | None = None


class GuardrailOutcome(StrEnum):
    """Content-free inspection outcomes, distinct from authored enforcement actions."""

    ALLOW = "allow"
    FLAGGED = "flagged"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ClassifierCoverageError(Exception):
    """The complete subject cannot be inspected; retries cannot restore coverage."""

    def __init__(self) -> None:
        """Carry no detector diagnostics or request content across the adapter boundary."""
        super().__init__("Content inspection does not support this complete request.")


class ClassifierUncertainError(Exception):
    """The classifier completed without establishing an allow or a violation."""

    def __init__(self) -> None:
        """Keep uncertainty distinguishable without retaining classifier diagnostics."""
        super().__init__("Content inspection could not determine a decision.")


def coverage_failure(*, check_id: str | None = None) -> GatewayFailure:
    """Return a content-free, nonretryable coverage error with no failover authority."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
        safe_message=(
            "Content inspection does not support this complete request. "
            "Use a supported context or modality."
        ),
        safe_details={"action": "error", **({} if check_id is None else {"check_id": check_id})},
        retryable_same_deployment=False,
        failover_eligible=False,
    )


class GuardrailRejected(Exception):
    """A guardrail chain decided to block or fail a request."""

    def __init__(self, failure: GatewayFailure) -> None:
        """Retain one sanitized, non-failover failure.

        Args:
            failure: Public-safe failure already stripped of request content.
        """
        super().__init__(failure.safe_message)
        self.failure = failure


def request_content_bytes(request: GatewayRequest) -> int:
    """Return UTF-8 size of the compact JSON request subject sent to classifiers.

    This count is the exact deterministic serialization used as the ``request``
    subject on the ``http_json`` contract: every canonical field, including
    messages, tool definitions, structured schemas, and metadata. Inspection
    admission separately bounds the complete in-memory subject, including exclusions.

    Args:
        request: Canonical request after optional continuation expansion.

    Returns:
        Byte count of the compact UTF-8 JSON subject.
    """
    return len(canonical_json_bytes(request))


def request_exceeds_inspection_limit(request: GatewayRequest, maximum_bytes: int) -> bool:
    """Bound every classifier-visible field before any adapter constructs its projection.

    The ordinary HTTP serialization excludes private provider carriers that local
    adapters can inspect. Apply a no-copy size lower bound to the complete subject
    first, then encode only a bounded candidate for exact UTF-8 JSON accounting.
    The HTTP adapter's narrower wire representation remains unchanged.
    """
    return observation_subject_exceeds(request, maximum_bytes) or (
        len(observation_subject_bytes(request)) > maximum_bytes
    )


def guardrail_failure(*, action: GuardrailAction, check_id: str | None = None) -> GatewayFailure:
    """Build one sanitized, terminal guardrail failure.

    Args:
        action: Block or error outcome that ended the chain.
        check_id: Optional content-free check identity for operator metadata.

    Returns:
        A failure that must not advance a provider waterfall.
    """
    if action is GuardrailAction.BLOCK:
        message = "The request was blocked by a gateway guardrail."
    else:
        message = "A gateway guardrail could not complete this request."
    details: JsonObject = {"action": action.value}
    if check_id is not None:
        details["check_id"] = check_id
    return GatewayFailure(
        failure_class=GatewayFailureClass.GUARDRAIL,
        safe_message=message,
        retryable_same_deployment=False,
        failover_eligible=False,
        safe_details=details,
    )

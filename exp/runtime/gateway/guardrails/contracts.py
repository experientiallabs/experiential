"""Immutable identity-scoped guardrail policy and decision contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, JsonObject, canonical_json_bytes
from exp.runtime.gateway.contracts import (
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    IdentityId,
    OrganizationId,
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
    """Immutable guardrail assignment for one organization-scoped identity.

    Lookup is ``(organization_id, identity_id)``. Identities are unique only
    inside an organization, so the same identity ID in two organizations
    cannot share a policy. ``protected`` fail-closes on adapter timeout,
    missing adapter, oversized payload, or any other classifier uncertainty.
    Non-protected identities skip a failed check and continue the remaining
    chain.
    """

    policy_id: ArtifactId
    organization_id: OrganizationId
    identity_id: IdentityId
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
        return self

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

    call_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    arguments: str = Field(max_length=4_000_000)


class GuardrailCompletion(ContractModel):
    """Winning normalized completion inspected once before any caller delivery."""

    text: str = ""
    reasoning_content: str = ""
    refusal: bool = False
    tool_calls: tuple[GuardrailToolCall, ...] = ()

    def content_bytes(self) -> int:
        """Return UTF-8 size of text, refusal flag, and raw tool arguments."""
        tool_bytes = sum(len(call.arguments.encode("utf-8")) for call in self.tool_calls)
        return (
            len(self.text.encode("utf-8"))
            + len(self.reasoning_content.encode("utf-8"))
            + tool_bytes
        )


class ClassifierVerdict(ContractModel):
    """Content-free classifier result plus optional in-memory replacement.

    Replacements stay on this object only for the remainder of the request.
    They are never logged or persisted.
    """

    flagged: bool
    replacement_text: str | None = None
    replacement_messages: tuple[GatewayMessage, ...] | None = None


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

    The bound is the exact deterministic serialization used as the ``request``
    subject on the ``http_json`` contract: every canonical field, including
    messages, tool definitions, structured schemas, and metadata.

    Args:
        request: Canonical request after optional continuation expansion.

    Returns:
        Byte count of the compact UTF-8 JSON subject.
    """
    carrier_bytes = sum(
        block.carrier_size_bytes
        if block.kind == "reasoning_content"
        else len(block.carrier.encode("ascii"))
        if block.kind == "sealed_reasoning_content"
        else 0
        for message in request.messages
        for block in message.provider_reasoning
    )
    return len(canonical_json_bytes(guardrail_request_subject(request))) + carrier_bytes


def guardrail_request_subject(request: GatewayRequest) -> JsonObject:
    """Return the classifier subject including authenticated hidden reasoning."""
    subject = request.model_dump(mode="json", by_alias=True, exclude_none=False)
    messages: list[JsonObject] = []
    for message in request.messages:
        item = message.model_dump(mode="json", by_alias=True, exclude_none=False)
        if message.provider_reasoning:
            item["provider_reasoning"] = [
                block.model_dump(mode="json", by_alias=True, exclude_none=False)
                for block in message.provider_reasoning
                if block.kind != "sealed_reasoning_content"
            ]
        messages.append(item)
    subject["messages"] = messages
    return subject


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

"""Explicit admission exports for request-owned native classifier execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailPolicy,
)
from exp.runtime.gateway.guardrails.text_coverage import text_input_context

if TYPE_CHECKING:
    from exp.runtime.gateway.guardrails.session import GuardrailSession


@runtime_checkable
class NativeOutputClassifier(Protocol):
    """A classifier with an explicit native output contract."""

    def native_output_configuration(self) -> JsonObject:
        """Export this adapter's immutable native configuration at admission."""
        ...


@runtime_checkable
class NativeInputClassifier(Protocol):
    """A classifier that freezes the exact read-only input it inspects."""

    def native_input_configuration(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> JsonObject:
        """Export an input program while admission still owns the request."""
        ...


@runtime_checkable
class NativeClassifierTransport(Protocol):
    """An injected classifier transport that exports its exact egress authority."""

    def native_classifier_transport(self) -> JsonObject:
        """Return private headers, egress restrictions and managed envelope additions."""
        ...


@runtime_checkable
class NativeClassifierClient(Protocol):
    """A classifier lookup that retains each check's resource scope."""

    def native_output_configuration(self, *, check: GuardrailCheck) -> JsonObject:
        """Resolve one scoped adapter's native output authority."""
        ...


@runtime_checkable
class NativeInputClient(Protocol):
    """A scoped lookup for the input adapter's native program."""

    def native_input_configuration(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> JsonObject:
        """Resolve one scoped adapter's read-only input program."""
        ...


@runtime_checkable
class NativeInputTemplate(Protocol):
    """Reusable input authority exported by an adapter or its scoped lookup."""

    def native_input_template(self, *, check: GuardrailCheck) -> JsonObject:
        """Resolve reusable input authority in the same admitted scope."""
        ...


def input_registration(
    session: GuardrailSession,
    subjects: tuple[GatewayRequest, ...],
    *,
    policies: tuple[GuardrailPolicy, ...] | None = None,
) -> list[JsonObject]:
    """Preserve subject order and check scope for overlapping read-only inspection."""
    return [
        session.engine.native_input_configuration(policy=policy, check=check, request=subject)
        for subject in subjects
        for policy in (session.enforcing_policies if policies is None else policies)
        for check in policy.checks
        if check.stage is GuardrailCheckStage.INPUT
    ]


def text_subjects(request: GatewayRequest) -> list[str]:
    """Freeze complete text context and immutable tool arguments for native checks."""
    context = text_input_context(request)
    result = [context] if context else []
    for message in request.messages:
        content = message.folded_tool_error_content()
        if content:
            result.append(content)
        result.extend(call.arguments_json() for call in message.tool_calls)
    return result


def output_registration(session: GuardrailSession | None) -> list[JsonObject] | None:
    """Freeze ordered checks and scopes independently of their classifier transport."""
    if session is None:
        return None
    result: list[JsonObject] = []
    for policy in session.output_policies:
        for check in policy.output_checks:
            result.append(session.engine.native_output_registration(policy=policy, check=check))
    return result

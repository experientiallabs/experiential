"""Carry frozen input inspection evidence through the existing durable write boundary."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from inspect import Parameter, signature
from typing import TYPE_CHECKING, TypedDict

from exp.common.core.artifacts import JsonObject

if TYPE_CHECKING:
    from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayFailure
    from exp.runtime.gateway.guardrails.session import GuardrailSession
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
    from exp.runtime.gateway.native_components import SyncWriteLedger
    from exp.runtime.gateway.native_execution import InflightRequest


class InputOutcomeKeyword(TypedDict, total=False):
    """Optional durable-ledger capability, absent when the ledger does not consume it.

    Attributes:
        input_guardrail_outcomes: Server-authored records, including request and organization.
    """

    input_guardrail_outcomes: tuple[JsonObject, ...]


def enable_input_outcomes(session: GuardrailSession, ledger: SyncWriteLedger) -> None:
    """Collect only when a terminal writer explicitly consumes the evidence."""
    if consumes_input_outcomes(ledger.finish_attempt) or consumes_input_outcomes(
        ledger.finish_request
    ):
        session.enable_input_outcomes()


def outcome_documents(
    entry: InflightRequest | None,
    *,
    authorization: AuthorizationSnapshot | None = None,
    guardrails: GuardrailSession | None = None,
) -> tuple[JsonObject, ...]:
    """Bind frozen evidence to authenticated ownership, including before registration."""
    if entry is not None:
        authorization, guardrails = entry.authorization, entry.guardrails
    if authorization is None or guardrails is None:
        return ()
    return tuple(
        {
            **outcome.model_dump(mode="json"),
            "request_id": authorization.request_id,
            "organization_id": authorization.organization_id,
        }
        for outcome in guardrails.input_outcomes()
    )


def outcome_keyword(
    write: Callable[..., object], documents: tuple[JsonObject, ...]
) -> InputOutcomeKeyword:
    """Offer the accounting extension only to a ledger explicitly declaring it."""
    if consumes_input_outcomes(write):
        return {"input_guardrail_outcomes": documents}
    return {}


def consumes_input_outcomes(write: Callable[..., object]) -> bool:
    """Probe the explicit capability without requiring a hashable callable host."""
    try:
        return _cached_consumes_input_outcomes(write)
    except TypeError:
        return _signature_consumes_input_outcomes(write)


@lru_cache(maxsize=128)
def _cached_consumes_input_outcomes(write: Callable[..., object]) -> bool:
    """Bound common function/method signature work across host requests."""
    return _signature_consumes_input_outcomes(write)


def _signature_consumes_input_outcomes(write: Callable[..., object]) -> bool:
    """A generic forwarding wrapper does not assert that its destination persists evidence."""
    try:
        parameter = signature(write).parameters.get("input_guardrail_outcomes")
    except (TypeError, ValueError):
        return False
    return (
        parameter is not None
        and parameter.default is not Parameter.empty
        and parameter.kind in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    )


def terminal_outcomes(
    write: Callable[..., object], entry: InflightRequest, *, finalize: bool = True
) -> InputOutcomeKeyword:
    """Intermediate attempts never freeze the request's eventual inspection outcome."""
    if not finalize or not consumes_input_outcomes(write):
        return {}
    return {"input_guardrail_outcomes": outcome_documents(entry)}


def finish_request_quietly(
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    failure: GatewayFailure,
    *,
    certify_no_effects: bool = False,
    web_search_requests: int = 0,
    guardrails: GuardrailSession | None = None,
) -> bool:
    """Retain failure, meters and input evidence together for the existing accounting sweep."""
    entry = accounting.entry(authorization.request_id)
    if entry is not None and not entry.total_attempts:
        web_search_requests = max(web_search_requests, entry.web_search_requests)
    try:
        return accounting.request_settlements.finish(
            authorization,
            failure,
            certify_no_effects=certify_no_effects,
            web_search_requests=web_search_requests,
            input_guardrail_outcomes=accounting.request_settlements.input_outcomes(
                entry, authorization=authorization, guardrails=guardrails
            ),
        )
    except Exception:  # noqa: BLE001 - the sanitized admission failure wins; recovery retains the write.
        return False


def write_request_outcomes(
    write: Callable[..., bool],
    *,
    authorization: AuthorizationSnapshot,
    failure: GatewayFailure,
    certify_no_effects: bool,
    web_search_requests: int,
    input_guardrail_outcomes: tuple[JsonObject, ...],
) -> bool:
    """Invoke the typed terminal write with its explicitly declared evidence extension."""
    return write(
        authorization=authorization,
        failure=failure,
        certify_no_effects=certify_no_effects,
        web_search_requests=web_search_requests,
        **outcome_keyword(write, input_guardrail_outcomes),
    )

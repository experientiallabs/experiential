"""Normalize native data-plane settlement payloads for durable accounting.

Also home to the sanitized failure vocabulary the accounting boundary answers
with (quota exhaustion, exhausted or throttled pools, transient roll
conditions) and the parser that turns one boundary failure payload into a
typed :class:`GatewayFailure`.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import cast

from exp.common.core.artifacts import JsonObject, stable_id
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRefusalReason,
    GatewayUsage,
)
from exp.runtime.gateway.rate_limit_headers import (
    RateLimitObservation,
    rate_limit_observation_from_payload,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.openai_protocol.errors import (
    THROTTLED_RETRY_AFTER_SECONDS,
    OpenAIProtocolError,
    public_failure_error,
)

_TERMINAL_KINDS = {
    "completed": GatewayEventKind.COMPLETED,
    "incomplete": GatewayEventKind.INCOMPLETE,
    "failed": GatewayEventKind.FAILED,
}


def budget_quota_failure() -> GatewayFailure:
    """Return the sanitized quota failure after no route can reserve its cost."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
        safe_message="monthly gateway allocation is exhausted",
    )


def all_routes_unavailable_failure() -> GatewayFailure:
    """Return the sanitized terminal failure for an exhausted certified pool."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
        safe_message="all exact-model deployments are unavailable",
    )


def all_routes_throttled_failure(remaining_seconds: float) -> GatewayFailure:
    """Return the throttle-window failure for a route the provider backed off.

    Every deployment sitting inside a provider throttle window is caller-facing
    rate limiting (the provider answered 429 and asked for backoff), not
    platform deadness: classing it provider_internal misfiled 429 storms as
    outages and paged operators for caller-driven load (2026-09-04 ledger,
    deepseek-v4-flash-vision-exp). One computed wait (the remaining window,
    floored at the default throttle backoff) rides both the message and
    ``retry_after_seconds`` so the Retry-After header a client honors never
    disagrees with the sentence it reads.

    Args:
        remaining_seconds: Longest remaining throttle window across the route.

    Returns:
        Sanitized throttled failure naming the retry window.
    """
    seconds = max(THROTTLED_RETRY_AFTER_SECONDS, math.ceil(remaining_seconds))
    return GatewayFailure(
        failure_class=GatewayFailureClass.THROTTLED,
        safe_message=(
            "all exact-model deployments are inside a provider throttle window; "
            f"retry in {seconds}s"
        ),
        retry_after_seconds=seconds,
    )


def gateway_updating_failure() -> GatewayFailure:
    """Return the sanitized retryable failure for a transient roll condition.

    A pod that cannot build the authorized catalog revision during a rolling
    deploy (a snapshot authored by another engine version it cannot reconcile)
    surfaces this instead of a closed INTERNAL: the condition clears on its own
    once the roll settles, so the honest answer is a retryable 503, never a bug
    signal that pages or opens a deployment circuit.
    """
    return GatewayFailure(
        failure_class=GatewayFailureClass.UNAVAILABLE,
        safe_message="the gateway is updating; retry the request",
    )


def failure_from_boundary_payload(payload: object) -> GatewayFailure | None:
    """Parse one optional classified failure from a boundary payload."""
    if not isinstance(payload, dict):
        return None
    data = cast("JsonObject", payload)
    rejected_parameter = data.get("rejected_parameter")
    provider_detail = data.get("provider_detail")
    retry_after = data.get("retry_after_seconds")
    return GatewayFailure(
        failure_class=GatewayFailureClass(str(data["failure_class"])),
        safe_message=str(data["safe_message"]),
        retryable_same_deployment=bool(data.get("retryable_same_deployment", False)),
        failover_eligible=bool(data.get("failover_eligible", False)),
        rejected_parameter=(
            rejected_parameter
            if isinstance(rejected_parameter, str) and rejected_parameter
            else None
        ),
        provider_detail=(
            provider_detail if isinstance(provider_detail, str) and provider_detail else None
        ),
        customer_owned=data.get("customer_owned") is True,
        retry_after_seconds=(
            retry_after
            if isinstance(retry_after, int)
            and not isinstance(retry_after, bool)
            and retry_after >= 1
            else None
        ),
        refusal_reason=refusal_reason_from_payload(data.get("refusal_reason")),
    )


def refusal_reason_from_payload(value: object) -> GatewayRefusalReason | None:
    """Parse one optional bounded refusal reason from a boundary payload.

    An unknown token fails closed to ``None`` rather than raising, so a future
    native reason a stale worker does not know never breaks settlement.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return GatewayRefusalReason(value)
    except ValueError:
        return None


def ledger_failure(failure: GatewayFailure) -> GatewayFailure:
    """The failure as the ledger records it.

    A customer-owned provider failure (their BYOK credential or account) keeps
    its provider class for ladder decisions, but the durable row files it as
    the caller's invalid request: it is their configuration, never operator
    deadness that pages or opens a house circuit.
    """
    if failure.customer_owned and failure.failure_class in {
        GatewayFailureClass.PROVIDER_AUTHENTICATION,
        GatewayFailureClass.PROVIDER_QUOTA,
    }:
        return failure.model_copy(update={"failure_class": GatewayFailureClass.INVALID_REQUEST})
    return failure


def terminal_from_settlement(
    data: JsonObject,
) -> tuple[GatewayEvent, GatewayFailure | None]:
    """Build a durable terminal event from one native settlement payload.

    Args:
        data: Parsed outcome, usage, tool names, and optional failure.

    Returns:
        The normalized terminal event and optional failure.
    """
    raw_usage = data.get("usage")
    raw_tool_names = data.get("tool_names")
    usage = _usage_from_payload(
        raw_usage if isinstance(raw_usage, dict) else None,
        [str(name) for name in raw_tool_names] if isinstance(raw_tool_names, list) else [],
    )
    failure_payload = data.get("failure")
    failure = None
    if isinstance(failure_payload, dict):
        provider_detail = failure_payload.get("provider_detail")
        failure = GatewayFailure(
            failure_class=GatewayFailureClass(str(failure_payload["failure_class"])),
            safe_message=str(failure_payload["safe_message"]),
            provider_detail=(
                provider_detail if isinstance(provider_detail, str) and provider_detail else None
            ),
            customer_owned=failure_payload.get("customer_owned") is True,
            retry_after_seconds=_optional_wait(failure_payload.get("retry_after_seconds")),
            # The bounded refusal category rides the settlement argument so the
            # control plane counts refusals by reason without parsing detail.
            refusal_reason=refusal_reason_from_payload(failure_payload.get("refusal_reason")),
        )
        if (
            failure.failure_class == GatewayFailureClass.THROTTLED
            and failure.retry_after_seconds is None
        ):
            # A throttled settlement whose failure names no wait still carries
            # the provider's own Retry-After when the data plane harvested the
            # rate-limit headers; sizing the throttle window from it is what
            # lets a daily-quota reset actually suppress the rung for hours.
            observed = settlement_rate_limit(data).retry_after_seconds
            if observed is not None:
                failure = failure.model_copy(update={"retry_after_seconds": observed})
        # A rejected credential or exhausted account on the customer's own
        # BYOK rung kept its ladder class in the data plane (so another
        # customer-managed rung could still serve), but the ledger files it
        # where it belongs: the caller's configuration, never operator
        # deadness that pages.
        failure = ledger_failure(failure)
    kind = _TERMINAL_KINDS[str(data["outcome"])]
    terminal = GatewayEvent(
        kind=kind,
        sequence_number=0,
        usage=usage,
        failure=failure if kind == GatewayEventKind.FAILED else None,
    )
    return terminal, failure


def first_token_at_from_settlement(data: JsonObject) -> datetime | None:
    """Return the winning attempt's first-token wall-clock time from a settlement payload.

    The native data plane includes ``first_token_at`` as an ISO-8601 timestamp only when it
    observed a first streamed token. A missing, non-string, or unparseable value yields
    ``None`` so accounting stays backward-compatible with engines that omit the field.

    Args:
        data: Parsed native settlement payload.

    Returns:
        The timezone-aware first-token time, or ``None`` when it is absent or malformed.
    """
    raw = data.get("first_token_at")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _usage_from_payload(
    payload: JsonObject | None,
    tool_names: list[str],
) -> GatewayUsage | None:
    """Build normalized usage from settlement scalars and tool names."""
    names = tuple(str(name) for name in tool_names)
    if payload is None or payload.get("input_tokens") is None:
        return GatewayUsage(tool_names=names) if names else None
    return GatewayUsage(
        input_tokens=_optional_count(payload.get("input_tokens")),
        output_tokens=_optional_count(payload.get("output_tokens")),
        cached_input_tokens=_optional_count(payload.get("cached_input_tokens")),
        # The billed cache-write subset of input_tokens: a reported zero is 0;
        # absent or null (a wire without the count, or a pre-0.3.52 data
        # plane) stays unknown, the one case a consumer may approximate.
        cache_creation_input_tokens=_optional_count(payload.get("cache_creation_input_tokens")),
        reasoning_tokens=_optional_count(payload.get("reasoning_tokens")),
        tool_names=names,
    )


def _optional_count(value: object) -> int | None:
    """Return one integer settlement token count or ``None``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_wait(value: object) -> int | None:
    """Return one positive integer wait in seconds or ``None``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def settlement_rate_limit(data: JsonObject) -> RateLimitObservation:
    """Parse the settlement's optional harvested rate-limit headers.

    The data plane forwards the allowlisted provider rate-limit response
    headers (successes and failures alike) as ``rate_limit_headers``; an
    engine that predates the field, or a response carrying none, yields the
    empty observation.

    Args:
        data: Parsed native settlement payload.

    Returns:
        The typed observation for the ledger and throttle calibration.
    """
    return rate_limit_observation_from_payload(data.get("rate_limit_headers"))


def deployment_operation_key(route: GatewayRoute, deployment: ExactModelDeployment) -> str:
    """Derive the stable per-deployment idempotency key used by dispatch.

    Mirrors the executor's provider-operation identity so retried physical
    dispatches of the same deployment reuse one caller operation while every
    later route position derives its own.

    Args:
        route: Resolved ordered route.
        deployment: The certified deployment being dispatched.

    Returns:
        Stable content-addressed operation identity.
    """
    authorization = route.snapshot.authorization
    return stable_id(
        "gateway-provider-operation",
        {
            "request_id": authorization.request_id,
            "catalog_sha256": authorization.catalog_sha256,
            "deployment_id": deployment.deployment_id,
            "connection_sha256": deployment.connection_sha256,
        },
    )


def optional_text(value: object) -> str | None:
    """Return one optional boundary string value or ``None``."""
    return value if isinstance(value, str) else None


def budget_quota_protocol_error() -> OpenAIProtocolError:
    """Return the public quota error for an exhausted monthly allocation."""
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
        safe_message="monthly gateway allocation is exhausted",
    )
    return public_failure_error(failure)

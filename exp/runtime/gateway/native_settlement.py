"""Normalize native data-plane settlement payloads for durable accounting.

Also home to the sanitized failure vocabulary the accounting boundary answers
with (quota exhaustion, exhausted or throttled pools, transient roll
conditions) and the parser that turns one boundary failure payload into a
typed :class:`GatewayFailure`.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
from collections.abc import Callable
from datetime import datetime
from typing import NotRequired, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field

from exp.common.core.artifacts import JsonObject, stable_id
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
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


def exhausted_attempt_payload(failure: GatewayFailure) -> str:
    """Serialize one sanitized terminal attempt-selection failure for the native boundary."""
    payload: JsonObject = {
        "failure_class": failure.failure_class.value,
        "safe_message": failure.safe_message,
    }
    if failure.customer_owned:
        payload["customer_owned"] = True
    if failure.rejected_parameter is not None:
        payload["rejected_parameter"] = failure.rejected_parameter
    if failure.provider_detail is not None:
        payload["provider_detail"] = failure.provider_detail
    if failure.refusal_reason is not None:
        payload["refusal_reason"] = failure.refusal_reason.value
    if failure.retry_after_seconds is not None:
        payload["retry_after_seconds"] = failure.retry_after_seconds
    return json.dumps({"exhausted": True, "failure": payload}, separators=(",", ":"))


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


class NativeSettlementPayload(BaseModel):
    """Strict native-only provenance that can retain an unresolved reservation.

    Remaining settlement fields keep their existing dedicated parsers. An
    absent marker never authorizes a hold, and a malformed marker is rejected
    rather than coerced from client-shaped strings or integers.
    """

    model_config = ConfigDict(extra="ignore")
    dispatched: bool = Field(default=False, strict=True)
    finalize: bool = Field(default=True, strict=True)
    usage_incomplete_due_to_disconnect: bool = Field(default=False, strict=True)

    def validate_disconnect(self, failure: GatewayFailure | None, kind: GatewayEventKind) -> bool:
        """Reject inconsistent hold evidence and return the trusted marker.

        Args:
            failure: Normalized failure attached to this settlement.
            kind: Normalized terminal event kind.

        Returns:
            Whether the cancelled dispatched attempt lacks a final meter.

        Raises:
            ValueError: A hold marker accompanies non-finalizing, non-cancelled,
                or undispatched work.
        """
        if self.usage_incomplete_due_to_disconnect and (
            not self.dispatched
            or not self.finalize
            or kind is not GatewayEventKind.FAILED
            or failure is None
            or failure.failure_class is not GatewayFailureClass.CANCELLED
        ):
            raise ValueError("incomplete disconnect usage requires dispatched cancelled work")
        return self.usage_incomplete_due_to_disconnect


class StreamedOutput(BaseModel):
    """Generated text the data plane observed before a caller disconnected.

    Verbatim text by output leg plus the characters past the data plane's
    retained bound, so an estimate can extrapolate what it could not keep.

    Attributes:
        text: Retained visible/refusal/tool argument text, empty by default.
        reasoning: Retained generated reasoning, empty by default.
        text_overflow_chars: Visible characters beyond the retained bound, default zero.
        reasoning_overflow_chars: Reasoning characters beyond the bound, default zero.
        images: Observed generated image count, default zero; any image disables estimation.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    text: str = ""
    reasoning: str = ""
    text_overflow_chars: int = Field(default=0, ge=0)
    reasoning_overflow_chars: int = Field(default=0, ge=0)
    images: int = Field(default=0, ge=0)


def streamed_output_from_settlement(data: JsonObject | None) -> StreamedOutput | None:
    """Parse the settlement's optional generated-output evidence.

    Only a dispatched cancellation without a provider terminal carries it,
    and the data plane always sends it then (an empty object when nothing was
    generated). A malformed object is dropped rather than estimated from, and
    an absent one means a data plane predating the field: either way the
    settlement keeps its unknown meter.

    Args:
        data: Parsed native settlement payload.

    Returns:
        The typed streamed output, or None when absent or malformed.
    """
    if data is None:
        return None
    payload = data.get("streamed_output")
    if not isinstance(payload, dict):
        return None
    try:
        return StreamedOutput.model_validate(payload)
    except ValueError:
        return None


def terminal_from_settlement(
    data: JsonObject,
    *,
    surface: GatewayApiSurface | None = None,
) -> tuple[GatewayEvent, GatewayFailure | None]:
    """Build a durable terminal event from one native settlement payload.

    Args:
        data: Parsed outcome, usage, tool names, and optional failure.
        surface: Frozen request surface for internal decision rejection evidence.

    Returns:
        The normalized terminal event and optional failure.
    """
    provenance = NativeSettlementPayload.model_validate(data)
    raw_usage = data.get("usage")
    raw_tool_names = data.get("tool_names")
    usage = _usage_from_payload(
        raw_usage if isinstance(raw_usage, dict) else None,
        [str(name) for name in raw_tool_names] if isinstance(raw_tool_names, list) else [],
        web_search_requests=web_search_requests_from_settlement(data),
        tool_search_requests=tool_search_requests_from_settlement(data),
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
        usage=_credible_usage(kind, usage),
        failure=failure if kind == GatewayEventKind.FAILED else None,
        usage_incomplete_due_to_disconnect=provenance.validate_disconnect(failure, kind),
        decision_provider_rejected=(
            surface is GatewayApiSurface.DECISIONS
            and kind is GatewayEventKind.FAILED
            and usage is None
            and data.get("opened") is False
            and data.get("decision_provider_rejected") is True
        ),
    )
    return terminal, failure


UPSTREAM_PROVIDER_MAX_CHARS = 128


@functools.lru_cache(maxsize=128)
def accepts_keyword(callable_object: Callable[..., object], name: str) -> bool:
    """Whether ``callable_object`` accepts keyword ``name`` (named or through ``**kwargs``).

    Capability detection for the hosted ledger seam: the engine may ship a new
    settle keyword before the host's ledger learns it, so the value is handed
    over only where the signature admits it. An unreadable signature is read
    as not accepting, never as accepting. Cached per callable (a bound method
    hashes by its function and receiver), so the probe runs once per ledger.
    """
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name and parameter.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return True
    return False


def upstream_provider_kwarg(
    settle: Callable[..., object], upstream_provider: str | None
) -> dict[str, str | None]:
    """The ``upstream_provider`` settle keyword for ``settle``, empty when it predates the field.

    ``settle`` is the hosted ledger's ``finish_attempt``; a host whose ledger
    has not learned the keyword gets no such argument (never a TypeError on
    every settle after an engine repin), one that has gets the named upstream.
    """
    if not accepts_keyword(settle, "upstream_provider"):
        return {}
    return {"upstream_provider": upstream_provider}


def web_search_requests_kwarg(
    settle: Callable[..., object], web_search_requests: int
) -> dict[str, int]:
    """The ``web_search_requests`` settle keyword for ``settle``, empty at zero or when unknown.

    Same seam as :func:`upstream_provider_kwarg`: a host whose ledger predates
    the keyword never sees it. A zero count is also withheld, so an attempt
    that ran no search settles byte-for-byte as before the field existed.
    """
    if web_search_requests <= 0 or not accepts_keyword(settle, "web_search_requests"):
        return {}
    return {"web_search_requests": web_search_requests}


def web_search_requests_from_settlement(data: JsonObject | None) -> int:
    """Return the gateway-executed web searches the settlement bills to the attempt.

    The native data plane puts ``web_search_requests`` at the top level of the
    settle argument, beside (not inside) ``usage``, and omits it at zero. A
    missing, non-integer, boolean, or negative value reads as zero so an
    engine that never searched settles exactly as before.

    Args:
        data: Parsed native settlement payload; ``None`` (a cancelled sweep) bills none.

    Returns:
        The non-negative search count, zero when the payload names none.
    """
    count = None if data is None else _optional_count(data.get("web_search_requests"))
    return count if count is not None and count > 0 else 0


def web_search_requests_from_terminal(terminal: GatewayEvent | None) -> int:
    """Return the web-search count the settled usage carries, zero without usage."""
    if terminal is None or terminal.usage is None:
        return 0
    return terminal.usage.web_search_requests


def tool_search_requests_kwarg(
    settle: Callable[..., object], tool_search_requests: int
) -> dict[str, int]:
    """The ``tool_search_requests`` settle keyword for ``settle``, empty at zero or when unknown.

    Same seam as :func:`web_search_requests_kwarg`: a host whose ledger
    predates the keyword never sees it, and a zero count is withheld so an
    attempt that ran no tool search settles byte-for-byte as before the field.
    """
    if tool_search_requests <= 0 or not accepts_keyword(settle, "tool_search_requests"):
        return {}
    return {"tool_search_requests": tool_search_requests}


def tool_search_requests_from_settlement(data: JsonObject | None) -> int:
    """Return the gateway-executed tool-search rounds the settlement bills to the attempt.

    The native data plane puts ``tool_search_requests`` at the top level of the
    settle argument, beside (not inside) ``usage``, and omits it at zero. A
    missing, non-integer, boolean, or negative value reads as zero so an
    engine that never ran a tool search settles exactly as before.

    Args:
        data: Parsed native settlement payload; ``None`` (a cancelled sweep) bills none.

    Returns:
        The non-negative tool-search count, zero when the payload names none.
    """
    count = None if data is None else _optional_count(data.get("tool_search_requests"))
    return count if count is not None and count > 0 else 0


def tool_search_requests_from_terminal(terminal: GatewayEvent | None) -> int:
    """Return the tool-search count the settled usage carries, zero without usage."""
    if terminal is None or terminal.usage is None:
        return 0
    return terminal.usage.tool_search_requests


"""Longest upstream label the settlement carries; anything longer is not a name."""


class SettlementMetadata(TypedDict):
    """Content-free observation fields forwarded identically on direct and swept writes."""

    first_token_at: datetime | None
    retry_after_seconds: int | None
    ratelimit_limit_requests: int | None
    ratelimit_remaining_requests: int | None
    ratelimit_limit_tokens: int | None
    ratelimit_remaining_tokens: int | None
    upstream_provider: NotRequired[str | None]


def settlement_metadata(
    data: JsonObject | None, settle: Callable[..., object]
) -> SettlementMetadata:
    """Project original observations while withholding unsupported host keywords."""
    observed = settlement_rate_limit(data)
    fields: SettlementMetadata = {
        "first_token_at": None if data is None else first_token_at_from_settlement(data),
        "retry_after_seconds": observed.retry_after_seconds,
        "ratelimit_limit_requests": observed.limit_requests,
        "ratelimit_remaining_requests": observed.remaining_requests,
        "ratelimit_limit_tokens": observed.limit_tokens,
        "ratelimit_remaining_tokens": observed.remaining_tokens,
    }
    if accepts_keyword(settle, "upstream_provider"):
        fields["upstream_provider"] = upstream_provider_from_settlement(data)
    return fields


def upstream_provider_from_settlement(data: JsonObject | None) -> str | None:
    """Return the upstream an aggregator rung named as serving the attempt.

    The native data plane includes ``upstream_provider`` when the provider's
    stream named the endpoint behind the answer (OpenRouter's per-chunk
    ``provider`` field, opted in by the ZDR constraint's metadata header). A
    missing, empty, non-string, or over-long value yields ``None`` so an
    engine or a provider that names nothing settles exactly as before.

    Args:
        data: Parsed native settlement payload; ``None`` (a cancelled sweep) names none.

    Returns:
        The provider label, or ``None`` when the attempt named none.
    """
    raw = None if data is None else data.get("upstream_provider")
    if not isinstance(raw, str):
        return None
    label = raw.strip()
    if not label or len(label) > UPSTREAM_PROVIDER_MAX_CHARS:
        return None
    return label


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


def _credible_usage(kind: GatewayEventKind, usage: GatewayUsage | None) -> GatewayUsage | None:
    """Drop a finished attempt's all-zero token report: it is not an observation.

    A provider that finished serving a request processed at least its prompt,
    so a usage object reporting zero input AND zero output tokens on a
    completed or incomplete terminal cannot be what the provider metered.
    Production 2026-09-15: 2.2% of the OpenAI lane's ``max_output_tokens``
    truncations (1,634 attempts across 192 organizations in 30 days) arrived
    with every count zero, while the identical prompt at the identical budget
    reported ~56k input / 64 reasoning tokens the other 98% of the time, at
    the same latency. Filing such a report as observed settles the attempt as
    provider-confirmed free; filing it as UNKNOWN (no usage) keeps it inside
    the ledger's unknown-usage review counters and its nightly invariant, and
    keeps the zero out of the cache-fraction calibration. Failed terminals are
    left alone: their zeros already settle at nothing and a billed refusal
    keys on positive counts. The whole usage goes, tool names included: the
    control plane files ANY non-null usage as observed (a tool-only usage is
    its convention for a provider that omitted the meter but streamed calls),
    and a tool call is output the meter should have counted, so tool names on
    an all-zero report describe a stream whose meter is not credible; losing
    ``tools_used`` on that row beats filing it as observed.

    Args:
        kind: The normalized terminal kind of the settlement.
        usage: The usage the data plane reported, if any.

    Returns:
        The usage the ledger should record.
    """
    if usage is None or kind not in {GatewayEventKind.COMPLETED, GatewayEventKind.INCOMPLETE}:
        return usage
    if usage.input_tokens != 0 or usage.output_tokens != 0:
        return usage
    return None


def _usage_from_payload(
    payload: JsonObject | None,
    tool_names: list[str],
    *,
    web_search_requests: int = 0,
    tool_search_requests: int = 0,
) -> GatewayUsage | None:
    """Build normalized usage without inventing absent token or TTL evidence.

    Args:
        payload: Native settlement usage object, or None.
        tool_names: Observed tool names in invocation order.
        web_search_requests: Gateway-executed searches billed to this attempt; rides on
            whichever usage shape the payload yields (a bare count is not usage and is dropped).
        tool_search_requests: Gateway-executed tool-search rounds billed to this attempt;
            rides on the usage exactly as ``web_search_requests`` does.

    Returns:
        Typed token or tool-only usage, or None when neither was observed.

    Raises:
        ValueError: The observed token totals or subsets are contradictory.
    """
    names = tuple(str(name) for name in tool_names)
    if payload is None or (
        payload.get("input_tokens") is None and payload.get("output_tokens") is None
    ):
        if not names:
            return None
        return GatewayUsage(
            tool_names=names,
            web_search_requests=web_search_requests,
            tool_search_requests=tool_search_requests,
        )
    return GatewayUsage(
        input_tokens=_optional_count(payload.get("input_tokens")),
        output_tokens=_optional_count(payload.get("output_tokens")),
        cached_input_tokens=_optional_count(payload.get("cached_input_tokens")),
        cache_creation_input_tokens=_optional_count(payload.get("cache_creation_input_tokens")),
        cache_creation_1h_input_tokens=_optional_count(
            payload.get("cache_creation_1h_input_tokens")
        ),
        reasoning_tokens=_optional_count(payload.get("reasoning_tokens")),
        tool_names=names,
        web_search_requests=web_search_requests,
        tool_search_requests=tool_search_requests,
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


def settlement_rate_limit(data: JsonObject | None) -> RateLimitObservation:
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
    if data is None:
        return RateLimitObservation()
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

"""Estimate the meter of a dispatched attempt that never received the provider's final usage.

Most OpenAI-shaped wires report token usage only in the stream's final frame.
When the caller disconnects first, the data plane closes the upstream and the
provider still bills the prompt it processed and the tokens it generated up to
the cut, so settling that attempt as unknown leaves real provider spend
uncharged. The gateway already tokenizes every prompt for its reservation and
already saw every generated delta, so it can price the work with its own
tokenizer: observed legs win, estimated legs fill the holes, and the result is
labelled ``estimated`` (never ``observed``) wherever it lands.

The same estimate fills a capped request's budget truncation that the
provider sent without any usage report. Meta's muse lanes end a stream that
hit ``max_tokens`` with ``finish_reason: stop`` and no usage frame
(2026-10-17); the data plane reads that capped stop as a truncation, metered
at no less than the output budget the attempt dispatched with, because the
provider generated (and billed) the whole budget even when its reasoning was
never streamed.

Every settled meter is then held to the attempt's dispatched output ceiling
when the caller set one (``max_tokens``, ``max_completion_tokens``,
``max_output_tokens``). OpenAI defines that ceiling as covering visible AND
reasoning tokens, but some reasoning lanes apply the wire field to visible
output only: Novita ``deepseek/deepseek-r1-turbo`` answered ``max_tokens: 16``
with 206 completion tokens, 203 of them reasoning (2026-10-20). The provider
may still bill its overshoot; the caller is never billed past the ceiling it
set, and each clamp is logged with its deployment so the lane's wire field can
be corrected in the catalog.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.attempt_tokens import counted_input_tokens, worst_case_output_tokens
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayFailure, GatewayRequest
from exp.runtime.gateway.embeddings_contracts import ServingRequest
from exp.runtime.gateway.native_execution import rung_load_key
from exp.runtime.gateway.native_settlement import (
    StreamedOutput,
    streamed_output_from_settlement,
    terminal_from_settlement,
    tool_search_requests_from_settlement,
    web_search_cost_from_settlement,
    web_search_requests_from_settlement,
    web_search_results_from_settlement,
)
from exp.runtime.gateway.reservation_tokenizer import reservation_encoder
from exp.runtime.gateway.rung_admission import RungLoadRegistry
from exp.runtime.gateway.stream_contracts import GatewayEvent, GatewayUsage

if TYPE_CHECKING:
    from exp.runtime.gateway.native_execution import InflightRequest

_logger = logging.getLogger(__name__)

FALLBACK_CHARACTERS_PER_TOKEN = 4
"""Characters per token assumed for overflow text when nothing was retained to calibrate on."""


def settled_terminal(
    data: JsonObject,
    entry: InflightRequest,
    *,
    parsed: tuple[GatewayEvent, GatewayFailure | None] | None = None,
    loads: RungLoadRegistry | None = None,
) -> tuple[GatewayEvent, GatewayFailure | None]:
    """Build one in-flight request's terminal from its settlement, disconnect estimate applied.

    Deterministic over the retained payload: the cache fraction read from the
    registry is frozen on the in-flight entry at first use, so the sweep's
    replay of a retained settlement reproduces the original meter even after
    later settlements moved the organization's live signal.

    Args:
        data: Parsed native settlement payload.
        entry: The owning in-flight request (its prompt and frozen surface).
        parsed: Already validated terminal used to stamp receipt time before tokenization.
        loads: The rung load registry whose per-organization cached-fraction
            EWMA (fed by this organization's observed settled meters on the rung)
            fills an unreported cache leg; None prices the prompt fresh.

    Returns:
        The normalized terminal event and optional failure.
    """
    terminal, failure = parsed or terminal_from_settlement(
        data, surface=entry.authorization.surface
    )
    terminal = estimate_disconnect_usage(
        terminal,
        request=entry.request,
        surface=entry.authorization.surface,
        opened=data.get("opened") is True,
        streamed=streamed_output_from_settlement(data),
        cached_fraction=_frozen_cached_fraction(data, loads, entry),
    )
    terminal = estimate_unreported_truncation_usage(
        terminal,
        request=entry.request,
        surface=entry.authorization.surface,
        opened=data.get("opened") is True,
        streamed=streamed_output_from_settlement(data),
        cached_fraction=_frozen_cached_fraction(data, loads, entry),
        output_budget=_truncated_output_budget(terminal, data, entry),
    )
    terminal = bill_within_output_ceiling(
        terminal,
        output_ceiling=_dispatched_output_ceiling(data, entry),
        deployment_id=_attempt_deployment_id(data, entry),
    )
    if terminal.usage_estimated and terminal.usage is not None:
        terminal = terminal.model_copy(
            update={
                "usage": terminal.usage.model_copy(
                    update={
                        "web_search_requests": web_search_requests_from_settlement(data),
                        "web_search_results": web_search_results_from_settlement(data),
                        "web_search_cost_nano_usd": web_search_cost_from_settlement(data),
                        "tool_search_requests": tool_search_requests_from_settlement(data),
                    }
                )
            }
        )
    denied = None if entry.guardrails is None else entry.guardrails.settlement_failure()
    return terminal, denied or failure


def _frozen_cached_fraction(
    data: JsonObject, loads: RungLoadRegistry | None, entry: InflightRequest
) -> float:
    """The fraction frozen on the entry for this attempt, freezing the live signal at first use."""
    attempt_id = data.get("attempt_id")
    if not isinstance(attempt_id, str):
        return 0.0
    with entry.recovery_observation_lock:
        frozen = entry.estimated_cache_fractions.get(attempt_id)
    if frozen is not None:
        return frozen
    observed = _recent_cached_fraction(loads, entry, attempt_id)
    with entry.recovery_observation_lock:
        return entry.estimated_cache_fractions.setdefault(attempt_id, observed)


def _truncated_output_budget(
    terminal: GatewayEvent, data: JsonObject, entry: InflightRequest
) -> int | None:
    """The output budget the attempt dispatched with, for an unreported truncation only.

    Recomputed exactly as the attempt's reservation sized it (the frozen
    per-depth bound when one was frozen), so the floor is the ``max_tokens``
    the provider generated up to. An uncapped request has no budget to floor at.
    """
    attempt_id = data.get("attempt_id")
    if not terminal.usage_unreported_truncation or not isinstance(attempt_id, str):
        return None
    depth = entry.attempt_depths.get(attempt_id)
    request = entry.request
    if (
        depth is None
        or depth >= len(entry.route.deployments)
        or not isinstance(request, GatewayRequest)
        or request.maximum_output_tokens is None
    ):
        return None
    if depth < len(entry.reserved_output_tokens_by_depth):
        request = request.model_copy(
            update={"maximum_output_tokens": entry.reserved_output_tokens_by_depth[depth]}
        )
    return worst_case_output_tokens(request, entry.route.deployments[depth])


def _dispatched_output_ceiling(data: JsonObject, entry: InflightRequest) -> int | None:
    """The output ceiling one attempt dispatched under a caller's explicit cap, else None.

    The caller's ceiling as the attempt carried it: the frozen per-depth bound
    when one was frozen (a lower rung maximum, or a disclosed floor repair),
    else the request's own value. A request without an explicit ceiling has
    none, so a catalog maximum never caps what a provider legitimately billed.
    """
    attempt_id = data.get("attempt_id")
    request = entry.request
    if (
        not isinstance(attempt_id, str)
        or not isinstance(request, GatewayRequest)
        or request.maximum_output_tokens is None
    ):
        return None
    depth = entry.attempt_depths.get(attempt_id)
    if depth is not None and depth < len(entry.reserved_output_tokens_by_depth):
        return entry.reserved_output_tokens_by_depth[depth]
    return request.maximum_output_tokens


def _attempt_deployment_id(data: JsonObject, entry: InflightRequest) -> str | None:
    """The deployment the settled attempt dispatched to, for operator logs only."""
    attempt_id = data.get("attempt_id")
    depth = None if not isinstance(attempt_id, str) else entry.attempt_depths.get(attempt_id)
    if depth is None or depth >= len(entry.route.deployments):
        return None
    return entry.route.deployments[depth].deployment_id


def bill_within_output_ceiling(
    terminal: GatewayEvent, *, output_ceiling: int | None, deployment_id: str | None = None
) -> GatewayEvent:
    """Hold a settled output meter to the caller's dispatched ceiling.

    The ceiling covers reasoning and visible output together (OpenAI's
    ``max_completion_tokens`` contract, which ``max_tokens`` shares for
    reasoning models). A provider that reports more output than it was
    allowed overshot its own wire contract, so the excess is the lane's, not
    the caller's: output is settled at the ceiling, and the reasoning and
    generated-image subsets are held inside it. Input legs are untouched.

    Args:
        terminal: The normalized terminal, estimates already applied.
        output_ceiling: The attempt's dispatched ceiling, or None when the
            caller set none.
        deployment_id: The attempt's deployment, named in the clamp log.

    Returns:
        The terminal with its output meter clamped, or the terminal unchanged.
    """
    usage = terminal.usage
    if (
        output_ceiling is None
        or usage is None
        or usage.output_tokens is None
        or usage.output_tokens <= output_ceiling
    ):
        return terminal
    _logger.warning(
        "provider output exceeded the caller's ceiling; billed at the ceiling",
        extra={
            "deployment_id": deployment_id,
            "output_ceiling": output_ceiling,
            "reported_output_tokens": usage.output_tokens,
            "reported_reasoning_tokens": usage.reasoning_tokens,
        },
    )
    return terminal.model_copy(
        update={
            "usage": usage.model_copy(
                update={
                    "output_tokens": output_ceiling,
                    "reasoning_tokens": _within(usage.reasoning_tokens, output_ceiling),
                    "image_output_tokens": _within(usage.image_output_tokens, output_ceiling),
                }
            )
        }
    )


def _within(subset: int | None, ceiling: int) -> int | None:
    """An output subset held inside a clamped output total."""
    return None if subset is None else min(subset, ceiling)


def _recent_cached_fraction(
    loads: RungLoadRegistry | None, entry: InflightRequest, attempt_id: object
) -> float:
    """The organization's live cached fraction on the rung that served the attempt, else 0."""
    depth = None if not isinstance(attempt_id, str) else entry.attempt_depths.get(attempt_id)
    if loads is None or depth is None or depth >= len(entry.route.deployments):
        return 0.0
    return loads.cached_fraction(
        rung_load_key(entry.route.deployments[depth]), entry.authorization.organization_id
    )


def estimate_disconnect_usage(
    terminal: GatewayEvent,
    *,
    request: ServingRequest,
    surface: GatewayApiSurface | None,
    opened: bool,
    streamed: StreamedOutput | None,
    cached_fraction: float = 0.0,
) -> GatewayEvent:
    """Fill a cancelled disconnect's unreported meter legs from gateway evidence.

    Applies only to the trusted ``usage_incomplete_due_to_disconnect`` marker on
    a completion attempt whose provider had accepted the request (``opened``)
    and whose data plane sent its generated-output evidence: a dispatch the
    provider never answered has no billed work to estimate, a data plane that
    predates the evidence (or sent it malformed) leaves the meter unknown, and
    generated images are billed per image, so any image keeps the meter
    unknown too. Each repair owns a separate reserved physical attempt, so
    the estimate never includes or releases another attempt's liability.
    Decisions, embeddings, and image requests keep their own contracts untouched.

    Args:
        terminal: The normalized cancelled terminal from the settlement.
        request: The admitted request, whose prompt the gateway tokenizes.
        surface: The frozen request surface.
        opened: Whether the provider's response headers arrived.
        streamed: Generated text the data plane observed before the cut.
        cached_fraction: The organization's recent cached share of input on
            this rung (its own settled meters' EWMA). An unreported cache leg
            is estimated at that share, so a caller whose conversation runs
            hot in the provider cache is not billed the whole prompt fresh
            (OpenAI-shaped wires report cache only in the final frame).

    Returns:
        The terminal with an ``estimated`` usage, or the terminal unchanged.
    """
    if not terminal.usage_incomplete_due_to_disconnect:
        return terminal
    return _estimated(terminal, request, surface, opened, streamed, cached_fraction, 0)


def estimate_unreported_truncation_usage(
    terminal: GatewayEvent,
    *,
    request: ServingRequest,
    surface: GatewayApiSurface | None,
    opened: bool,
    streamed: StreamedOutput | None,
    cached_fraction: float = 0.0,
    output_budget: int | None = None,
) -> GatewayEvent:
    """Fill a budget truncation's missing meter from gateway evidence.

    Applies only to the trusted ``usage_unreported_truncation`` marker, under
    the same evidence rules as a disconnect estimate. Output is metered at no
    less than ``output_budget``: the provider generated the whole budget,
    reasoning it never streamed included.

    Args:
        terminal: The normalized incomplete terminal.
        request: The admitted request, whose prompt the gateway tokenizes.
        surface: The frozen request surface.
        opened: Whether the provider's response headers arrived.
        streamed: Generated text the data plane observed.
        cached_fraction: The organization's recent cached share of input on this rung.
        output_budget: The dispatched output budget, or None when the
            request carried no cap to floor at.

    Returns:
        The terminal with an ``estimated`` usage, or the terminal unchanged.
    """
    if not terminal.usage_unreported_truncation:
        return terminal
    return _estimated(
        terminal, request, surface, opened, streamed, cached_fraction, output_budget or 0
    )


def _estimated(
    terminal: GatewayEvent,
    request: ServingRequest,
    surface: GatewayApiSurface | None,
    opened: bool,
    streamed: StreamedOutput | None,
    cached_fraction: float,
    output_floor: int,
) -> GatewayEvent:
    """Estimate one unreported meter, observed legs kept, output at least ``output_floor``."""
    if (
        not opened
        or surface is GatewayApiSurface.DECISIONS
        or not isinstance(request, GatewayRequest)
        or streamed is None
        or streamed.images > 0
    ):
        return terminal
    observed = terminal.usage
    reasoning_estimate = _text_tokens(streamed.reasoning, streamed.reasoning_overflow_chars)
    visible_estimate = _text_tokens(streamed.text, streamed.text_overflow_chars)
    input_tokens = (
        counted_input_tokens(request)
        if observed is None or observed.input_tokens is None
        else observed.input_tokens
    )
    # Reasoning is an output subset (the ledger's pricing contract), so the
    # estimate folds it into output. A provider's running report can trail
    # the text already streamed (Anthropic reports one output token at
    # message start), so the larger of report and estimate is the meter.
    reasoning_tokens = _largest(
        None if observed is None else observed.reasoning_tokens, reasoning_estimate
    )
    output_tokens = _largest(
        None if observed is None else observed.output_tokens,
        visible_estimate + reasoning_estimate,
    )
    if reasoning_tokens is not None and reasoning_tokens > output_tokens:
        output_tokens = reasoning_tokens
    output_tokens = max(output_tokens, output_floor)
    # Every observed cache leg is kept as reported, whether or not the
    # provider also reported the input total. When the total is estimated it
    # is raised to hold the observed subsets, and an unreported read leg is
    # estimated at the organization's recent cached share of input: reads
    # and writes are disjoint input subsets and settlement clamps reads
    # FIRST, so the estimate leaves room for every observed write (an
    # unknown-TTL write's unpriced status included) instead of displacing it.
    cached_input_tokens = None if observed is None else observed.cached_input_tokens
    observed_writes = None if observed is None else observed.cache_creation_input_tokens
    observed_1h_writes = None if observed is None else observed.cache_creation_1h_input_tokens
    observed_12h_writes = None if observed is None else observed.cache_creation_12h_input_tokens
    if observed is None or observed.input_tokens is None:
        input_tokens = max(input_tokens, (cached_input_tokens or 0) + (observed_writes or 0))
    room = input_tokens - (observed_writes or 0)
    if cached_input_tokens is None and cached_fraction > 0 and room > 0:
        cached_input_tokens = min(room, int(input_tokens * min(cached_fraction, 1.0)))
    usage = GatewayUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=observed_writes,
        cache_creation_1h_input_tokens=observed_1h_writes,
        cache_creation_12h_input_tokens=observed_12h_writes,
        reasoning_tokens=reasoning_tokens,
        tool_names=() if observed is None else observed.tool_names,
        web_search_requests=0 if observed is None else observed.web_search_requests,
        web_search_results=None if observed is None else observed.web_search_results,
        web_search_cost_nano_usd=None if observed is None else observed.web_search_cost_nano_usd,
        tool_search_requests=0 if observed is None else observed.tool_search_requests,
    )
    return terminal.model_copy(update={"usage": usage, "usage_estimated": True})


def _text_tokens(text: str, overflow_chars: int) -> int:
    """Count one output leg with the reservation tokenizer, extrapolating overflow."""
    counted = len(reservation_encoder().encode_ordinary(text)) if text else 0
    if overflow_chars == 0:
        return counted
    retained_chars = len(text)
    if retained_chars == 0 or counted == 0:
        return counted + -(-overflow_chars // FALLBACK_CHARACTERS_PER_TOKEN)
    return counted + -(-overflow_chars * counted // retained_chars)


def _largest(observed: int | None, estimate: int) -> int:
    """The observed count when it is at least the estimate, else the estimate."""
    return estimate if observed is None else max(observed, estimate)

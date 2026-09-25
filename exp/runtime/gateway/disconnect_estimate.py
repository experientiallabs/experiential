"""Estimate the meter of a dispatched attempt whose caller left before the provider's final usage.

Most OpenAI-shaped wires report token usage only in the stream's final frame.
When the caller disconnects first, the data plane closes the upstream and the
provider still bills the prompt it processed and the tokens it generated up to
the cut, so settling that attempt as unknown leaves real provider spend
uncharged. The gateway already tokenizes every prompt for its reservation and
already saw every generated delta, so it can price the work with its own
tokenizer: observed legs win, estimated legs fill the holes, and the result is
labelled ``estimated`` (never ``observed``) wherever it lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.attempt_tokens import counted_input_tokens
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayFailure, GatewayRequest
from exp.runtime.gateway.embeddings_contracts import ServingRequest
from exp.runtime.gateway.native_execution import rung_load_key
from exp.runtime.gateway.native_settlement import (
    StreamedOutput,
    streamed_output_from_settlement,
    terminal_from_settlement,
    tool_search_requests_from_settlement,
    web_search_requests_from_settlement,
)
from exp.runtime.gateway.reservation_tokenizer import reservation_encoder
from exp.runtime.gateway.rung_admission import RungLoadRegistry
from exp.runtime.gateway.stream_contracts import GatewayEvent, GatewayUsage

if TYPE_CHECKING:
    from exp.runtime.gateway.native_execution import InflightRequest

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
    if terminal.usage_estimated and terminal.usage is not None:
        terminal = terminal.model_copy(
            update={
                "usage": terminal.usage.model_copy(
                    update={
                        "web_search_requests": web_search_requests_from_settlement(data),
                        "tool_search_requests": tool_search_requests_from_settlement(data),
                    }
                )
            }
        )
    return terminal, failure


def _frozen_cached_fraction(
    data: JsonObject, loads: RungLoadRegistry | None, entry: InflightRequest
) -> float:
    """The fraction frozen on the entry for this attempt, freezing the live signal at first use."""
    attempt_id = data.get("attempt_id")
    if not isinstance(attempt_id, str):
        return 0.0
    with entry.execution_lock:
        frozen = entry.estimated_cache_fractions.get(attempt_id)
    if frozen is not None:
        return frozen
    observed = _recent_cached_fraction(loads, entry, attempt_id)
    with entry.execution_lock:
        return entry.estimated_cache_fractions.setdefault(attempt_id, observed)


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
    if (
        not terminal.usage_incomplete_due_to_disconnect
        or not opened
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
        reasoning_tokens=reasoning_tokens,
        tool_names=() if observed is None else observed.tool_names,
        web_search_requests=0 if observed is None else observed.web_search_requests,
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

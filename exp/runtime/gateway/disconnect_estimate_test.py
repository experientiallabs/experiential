"""Tests for the disconnect usage estimate."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.attempt_tokens import counted_input_tokens
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.disconnect_estimate import (
    FALLBACK_CHARACTERS_PER_TOKEN,
    estimate_disconnect_usage,
)
from exp.runtime.gateway.embeddings_contracts import EmbeddingsRequest
from exp.runtime.gateway.ledger_valuation import estimated_cost_nano_usd
from exp.runtime.gateway.native_settlement import StreamedOutput
from exp.runtime.gateway.reservation_tokenizer import reservation_encoder
from exp.runtime.gateway.stream_contracts import GatewayEvent, GatewayEventKind


def _request() -> GatewayRequest:
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="user", content="Explain why the sky is blue in two sentences."),
        ),
    )


def _disconnect(usage: GatewayUsage | None = None) -> GatewayEvent:
    return GatewayEvent(
        kind=GatewayEventKind.FAILED,
        sequence_number=0,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.CANCELLED, safe_message="caller disconnected"
        ),
        usage=usage,
        usage_incomplete_due_to_disconnect=True,
    )


def _tokens(text: str) -> int:
    return len(reservation_encoder().encode_ordinary(text))


def test_opened_disconnect_without_any_report_is_priced_from_prompt_and_streamed_text() -> None:
    """The counted prompt and the tokenized deltas replace the meter the provider never sent."""
    request = _request()
    streamed = StreamedOutput(text="Sunlight scatters off air molecules.", reasoning="short waves")
    estimated = estimate_disconnect_usage(
        _disconnect(),
        request=request,
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        opened=True,
        streamed=streamed,
    )
    assert estimated.usage_estimated is True
    assert estimated.usage_incomplete_due_to_disconnect is True
    usage = estimated.usage
    assert usage is not None
    assert usage.input_tokens == counted_input_tokens(request) > 0
    assert usage.reasoning_tokens == _tokens("short waves")
    assert usage.output_tokens == _tokens(streamed.text) + _tokens("short waves")
    assert usage.cached_input_tokens is None
    assert "usage_estimated" not in estimated.model_dump()


def test_observed_legs_win_unless_the_streamed_text_already_exceeds_them() -> None:
    """An Anthropic message-start report (input, cache, one output token) keeps its input legs."""
    request = _request()
    observed = GatewayUsage(
        input_tokens=1_200, output_tokens=1, cached_input_tokens=1_000, reasoning_tokens=None
    )
    streamed = StreamedOutput(text="word " * 40)
    estimated = estimate_disconnect_usage(
        _disconnect(observed),
        request=request,
        surface=GatewayApiSurface.MESSAGES,
        opened=True,
        streamed=streamed,
    )
    usage = estimated.usage
    assert usage is not None
    assert usage.input_tokens == 1_200
    assert usage.cached_input_tokens == 1_000
    assert usage.output_tokens == _tokens(streamed.text) > 1
    assert usage.reasoning_tokens == 0
    # A running cumulative report at or above the estimate is the meter.
    ahead = GatewayUsage(input_tokens=1_200, output_tokens=500, reasoning_tokens=200)
    kept = estimate_disconnect_usage(
        _disconnect(ahead),
        request=request,
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        opened=True,
        streamed=streamed,
    ).usage
    assert kept is not None
    assert (kept.output_tokens, kept.reasoning_tokens) == (500, 200)


def test_observed_cache_legs_survive_an_estimated_input_total() -> None:
    """A partial report's cache subsets are kept and the estimated total is raised to hold them."""
    request = _request()
    counted = counted_input_tokens(request)
    observed = GatewayUsage(output_tokens=3, cached_input_tokens=50)
    usage = estimate_disconnect_usage(
        _disconnect(observed),
        request=request,
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        opened=True,
        streamed=StreamedOutput(),
    ).usage
    assert usage is not None
    assert usage.cached_input_tokens == 50
    assert usage.input_tokens == max(counted, 50)
    assert usage.output_tokens == 3
    # Writes reported without an input total: kept, the estimated read
    # leaves room for them, and an unknown TTL split keeps the cost unknown.
    writes_only = GatewayUsage(output_tokens=2, cache_creation_input_tokens=counted + 100)
    usage = estimate_disconnect_usage(
        _disconnect(writes_only),
        request=request,
        surface=GatewayApiSurface.MESSAGES,
        opened=True,
        streamed=StreamedOutput(),
        cached_fraction=0.9,
    ).usage
    assert usage is not None
    assert usage.input_tokens == counted + 100
    assert usage.cache_creation_input_tokens == counted + 100
    assert usage.cache_creation_1h_input_tokens is None
    assert usage.cached_input_tokens is None
    assert (
        estimated_cost_nano_usd(
            usage,
            input_rate=3_000_000_000,
            cached_input_rate=300_000_000,
            cache_creation_input_rate=3_750_000_000,
            cache_creation_1h_input_rate=None,
            output_rate=15_000_000_000,
            reasoning_rate=None,
        )
        is None
    )
    # Writes smaller than the counted prompt leave room: the estimated read
    # fills only that room.
    assert counted > 5
    partial_writes = GatewayUsage(output_tokens=2, cache_creation_input_tokens=5)
    usage = estimate_disconnect_usage(
        _disconnect(partial_writes),
        request=request,
        surface=GatewayApiSurface.MESSAGES,
        opened=True,
        streamed=StreamedOutput(),
        cached_fraction=0.9,
    ).usage
    assert usage is not None
    assert usage.input_tokens == counted
    assert usage.cache_creation_input_tokens == 5
    assert usage.cached_input_tokens == min(counted - 5, int(counted * 0.9))


@pytest.mark.parametrize(
    ("observed", "fraction", "expected_cached"),
    [
        (None, 0.9, "estimate"),
        (None, 0.0, None),
        (None, 1.7, "all"),
        (GatewayUsage(input_tokens=1_000, output_tokens=1), 0.9, 900),
        (GatewayUsage(input_tokens=1_000, output_tokens=1, cached_input_tokens=0), 0.9, 0),
        (GatewayUsage(input_tokens=1_000, output_tokens=1, cached_input_tokens=250), 0.9, 250),
    ],
)
def test_unreported_cache_reads_take_the_recent_cached_fraction(
    observed: GatewayUsage | None, fraction: float, expected_cached: object
) -> None:
    """A missing cache-read leg is estimated at the organization's share; a reported one is kept."""
    request = _request()
    usage = estimate_disconnect_usage(
        _disconnect(observed),
        request=request,
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        opened=True,
        streamed=StreamedOutput(text="partial"),
        cached_fraction=fraction,
    ).usage
    assert usage is not None
    assert usage.input_tokens is not None
    if expected_cached == "estimate":
        assert usage.cached_input_tokens == int(usage.input_tokens * fraction) > 0
    elif expected_cached == "all":
        assert usage.cached_input_tokens == usage.input_tokens
    else:
        assert usage.cached_input_tokens == expected_cached
    assert usage.cache_creation_input_tokens is None


def test_estimated_cache_reads_never_displace_observed_cache_writes() -> None:
    """A read estimate leaves room for every reported write, so write liability survives pricing."""
    observed = GatewayUsage(input_tokens=1_000, output_tokens=1, cache_creation_input_tokens=300)
    usage = estimate_disconnect_usage(
        _disconnect(observed),
        request=_request(),
        surface=GatewayApiSurface.MESSAGES,
        opened=True,
        streamed=StreamedOutput(text="partial"),
        cached_fraction=0.9,
    ).usage
    assert usage is not None
    assert usage.cached_input_tokens == 700
    assert usage.cache_creation_input_tokens == 300
    # The read-first clamps keep all 300 writes, and an unknown TTL split
    # keeps the cost unknown instead of pricing the writes away.
    assert (
        estimated_cost_nano_usd(
            usage,
            input_rate=3_000_000_000,
            cached_input_rate=300_000_000,
            cache_creation_input_rate=3_750_000_000,
            cache_creation_1h_input_rate=None,
            output_rate=15_000_000_000,
            reasoning_rate=None,
        )
        is None
    )
    priced = estimated_cost_nano_usd(
        usage.model_copy(update={"cache_creation_1h_input_tokens": 0}),
        input_rate=3_000_000_000,
        cached_input_rate=300_000_000,
        cache_creation_input_rate=3_750_000_000,
        cache_creation_1h_input_rate=6_000_000_000,
        output_rate=15_000_000_000,
        reasoning_rate=None,
    )
    assert priced == round(
        (0 * 3_000 + 300 * 3_750 + 700 * 300 + 1 * 15_000) * 1_000_000 / 1_000_000
    )
    # Writes that already fill the input leave nothing to estimate.
    full = estimate_disconnect_usage(
        _disconnect(
            GatewayUsage(input_tokens=500, output_tokens=1, cache_creation_input_tokens=500)
        ),
        request=_request(),
        surface=GatewayApiSurface.MESSAGES,
        opened=True,
        streamed=StreamedOutput(),
        cached_fraction=0.9,
    ).usage
    assert full is not None
    assert full.cached_input_tokens is None


def test_overflow_extrapolates_from_the_retained_ratio_or_the_fallback_density() -> None:
    """Text past the data plane's bound is counted, never forgotten."""
    retained = "alpha beta gamma delta " * 8
    usage = estimate_disconnect_usage(
        _disconnect(),
        request=_request(),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        opened=True,
        streamed=StreamedOutput(text=retained, text_overflow_chars=len(retained)),
    ).usage
    assert usage is not None
    assert usage.output_tokens == 2 * _tokens(retained)
    bare = estimate_disconnect_usage(
        _disconnect(),
        request=_request(),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        opened=True,
        streamed=StreamedOutput(reasoning_overflow_chars=41),
    ).usage
    assert bare is not None
    assert bare.reasoning_tokens == -(-41 // FALLBACK_CHARACTERS_PER_TOKEN)
    assert bare.output_tokens == bare.reasoning_tokens


_PARTIAL = StreamedOutput(text="partial answer")


@pytest.mark.parametrize(
    ("opened", "surface", "marker", "streamed"),
    [
        (False, GatewayApiSurface.CHAT_COMPLETIONS, True, _PARTIAL),
        (True, GatewayApiSurface.DECISIONS, True, _PARTIAL),
        (True, GatewayApiSurface.CHAT_COMPLETIONS, False, _PARTIAL),
        # A data plane predating (or mis-sending) the evidence keeps unknown.
        (True, GatewayApiSurface.CHAT_COMPLETIONS, True, None),
        # Generated images are billed per image, never estimable from text.
        (True, GatewayApiSurface.CHAT_COMPLETIONS, True, StreamedOutput(text="ok", images=1)),
    ],
)
def test_unopened_decision_image_and_ordinary_settlements_are_left_alone(
    opened: bool, surface: GatewayApiSurface, marker: bool, streamed: StreamedOutput | None
) -> None:
    """Only an opened, dispatched text disconnect with data-plane evidence is estimated."""
    terminal = (
        _disconnect()
        if marker
        else GatewayEvent(
            kind=GatewayEventKind.FAILED,
            sequence_number=0,
            failure=GatewayFailure(
                failure_class=GatewayFailureClass.CANCELLED, safe_message="cancelled"
            ),
        )
    )
    unchanged = estimate_disconnect_usage(
        terminal,
        request=_request(),
        surface=surface,
        opened=opened,
        streamed=streamed,
    )
    assert unchanged is terminal
    assert unchanged.usage is None
    assert unchanged.usage_estimated is False


def test_non_completion_requests_are_left_alone() -> None:
    """Embeddings and image requests have their own billing shapes; nothing is estimated."""
    unchanged = estimate_disconnect_usage(
        _disconnect(),
        request=EmbeddingsRequest(inputs=("one",)),
        surface=GatewayApiSurface.EMBEDDINGS,
        opened=True,
        streamed=StreamedOutput(),
    )
    assert unchanged.usage is None
    assert unchanged.usage_estimated is False


def test_estimated_marker_requires_a_disconnect_with_both_totals() -> None:
    """The marker never rides a clean terminal or a half-empty meter."""
    with pytest.raises(ValueError, match="estimated usage requires"):
        GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=0,
            usage=GatewayUsage(input_tokens=1, output_tokens=1),
            usage_estimated=True,
        )
    with pytest.raises(ValueError, match="estimated usage requires"):
        GatewayEvent(
            kind=GatewayEventKind.FAILED,
            sequence_number=0,
            failure=GatewayFailure(
                failure_class=GatewayFailureClass.CANCELLED, safe_message="cancelled"
            ),
            usage=GatewayUsage(input_tokens=1),
            usage_incomplete_due_to_disconnect=True,
            usage_estimated=True,
        )

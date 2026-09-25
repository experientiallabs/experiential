"""Passive measurements preserve uncertainty, cache subsets, and observer timing."""

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.capture.metrics import observed_metrics


@pytest.mark.parametrize("terminal", [False, True])
def test_anthropic_cache_totals_match_gateway_capture_fields(terminal: bool) -> None:
    """Cache creation detail partitions its total and never adds another input charge."""
    metrics = observed_metrics(
        "messages",
        {
            "usage": {
                "input_tokens": 3,
                "output_tokens": 7,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 10,
                "cache_creation": {"ephemeral_1h_input_tokens": 6},
            }
        },
        started_ns=1_000_000_000,
        ended_ns=3_000_000_000,
        terminal=terminal,
    )
    assert metrics.usage is not None
    assert metrics.usage.input_tokens == 113
    assert metrics.usage.output_tokens == 7
    assert metrics.usage.cached_input_tokens == 100
    assert metrics.usage.cache_creation_input_tokens == 10
    assert metrics.usage.cache_creation_1h_input_tokens == 6
    assert metrics.first_token_at is None
    assert metrics.terminal_at == (3 if terminal else None)
    assert metrics.duration_ms == (2000 if terminal else None)
    assert metrics.usage_complete is terminal


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"usage": {}},
        {"usage": {"input_tokens": 1}},
        {"usage": {"input_tokens": True, "output_tokens": 2}},
        {"usage": {"input_tokens": 0, "output_tokens": 0}},
    ],
)
def test_absent_partial_invalid_and_zero_meters_are_not_complete(response: JsonObject) -> None:
    """The evidence never certifies missing or noncredible meters as final usage."""
    metrics = observed_metrics("responses", response, started_ns=1, ended_ns=2, terminal=True)
    assert not metrics.usage_complete
    assert metrics.first_token_at is None


def test_openai_cache_and_reasoning_are_subsets_of_provider_totals() -> None:
    """Responses details survive alongside the original totals without double counting."""
    metrics = observed_metrics(
        "responses",
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 90},
                "output_tokens_details": {"reasoning_tokens": 12},
            }
        },
        started_ns=1_000_000_000,
        ended_ns=2_000_000_000,
        terminal=True,
    )
    assert metrics.usage is not None
    assert metrics.usage.input_tokens == 100
    assert metrics.usage.output_tokens == 20
    assert metrics.usage.cached_input_tokens == 90
    assert metrics.usage.reasoning_tokens == 12

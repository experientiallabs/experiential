"""Tests for native Gemini response conversion and usage accounting."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models import BillingSource, ModelSnapshot, Usage
from exp.runtime.models.providers.errors import ProviderResponseError
from exp.runtime.models.providers.gemini import gemini_generate_response


def _snapshot() -> ModelSnapshot:
    """Return a frozen Gemini identity fixture."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="gemini",
        model_id="gemini-fixture",
        revision="fixture-revision",
        capabilities_sha256="a" * 64,
        connection_sha256="a" * 64,
    )


def _completed_usage(usage: JsonObject) -> Usage:
    """Parse usage from one completed generateContent payload.

    Args:
        usage: Native ``usageMetadata`` object.

    Returns:
        Observed provider-neutral usage.

    Raises:
        ProviderResponseError: A usage field is present but not a non-negative integer.
    """
    response = gemini_generate_response(
        {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": usage,
        },
        configured_model=_snapshot(),
        latency_seconds=0.1,
    )
    observed = response.economics.usage
    assert observed is not None
    return observed


def test_absent_thoughts_leave_output_at_candidates_token_count() -> None:
    """Omitting thoughtsTokenCount keeps output equal to candidatesTokenCount."""
    usage = _completed_usage(
        {
            "promptTokenCount": 11,
            "candidatesTokenCount": 5,
            "cachedContentTokenCount": 2,
        }
    )
    assert usage == Usage(input_tokens=11, output_tokens=5, cached_input_tokens=2)


def test_thoughts_fold_into_billed_output_tokens() -> None:
    """Google's thoughtsTokenCount is additive, so billed output is the sum."""
    usage = _completed_usage(
        {
            "promptTokenCount": 11,
            "candidatesTokenCount": 5,
            "cachedContentTokenCount": 2,
            "thoughtsTokenCount": 3,
            "totalTokenCount": 19,
        }
    )
    assert usage == Usage(input_tokens=11, output_tokens=8, cached_input_tokens=2)


def test_zero_thoughts_keep_output_at_candidates_token_count() -> None:
    """A reported zero thoughts count is valid and does not change output."""
    usage = _completed_usage(
        {
            "promptTokenCount": 11,
            "candidatesTokenCount": 5,
            "cachedContentTokenCount": 2,
            "thoughtsTokenCount": 0,
        }
    )
    assert usage == Usage(input_tokens=11, output_tokens=5, cached_input_tokens=2)


@pytest.mark.parametrize("bad", ("3", True, -1, 1.5, []))
def test_malformed_thoughts_token_count_is_a_provider_response_error(bad: JsonValue) -> None:
    """A present thoughtsTokenCount that is not a non-negative integer fails closed."""
    usage: JsonObject = {
        "promptTokenCount": 11,
        "candidatesTokenCount": 5,
        "cachedContentTokenCount": 2,
        "thoughtsTokenCount": bad,
    }
    with pytest.raises(ProviderResponseError, match="thoughtsTokenCount"):
        _completed_usage(usage)


def test_input_and_cached_counts_stay_subsets_of_prompt_tokens() -> None:
    """Prompt and cached-content counters stay unchanged when thoughts fold in."""
    usage = _completed_usage(
        {
            "promptTokenCount": 12,
            "candidatesTokenCount": 6,
            "cachedContentTokenCount": 4,
            "thoughtsTokenCount": 3,
        }
    )
    assert usage.input_tokens == 12
    assert usage.cached_input_tokens == 4
    assert usage.output_tokens == 9

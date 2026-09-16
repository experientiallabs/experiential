"""Tests for probability admission and exact effort preservation."""

from dataclasses import replace

import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.capability_policy import coerce_generation_parameters
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.generation_route_compat import (
    compatible_generation_parameter_profile_indexes,
)
from exp.runtime.models.providers.logprobs import require_chat_logprobs
from exp.runtime.models.providers.streaming_requests_test import _chat_request


def _profile() -> GatewayWireProfile:
    """Build an explicitly capable nonreasoning Chat deployment."""
    return GatewayWireProfile(
        dialect="openai_compatible",
        url="https://example.invalid",
        headers={},
        model_id="test",
        supports_logprobs=True,
    )


def test_probability_narrowing_keeps_order_and_rejects_unknown_support() -> None:
    """Unsupported fallbacks are removed without changing the requested probabilities."""
    request = _chat_request().model_copy(update={"logprobs": True, "top_logprobs": 0})
    profiles = (_profile(), replace(_profile(), supports_logprobs=False), _profile())
    assert compatible_generation_parameter_profile_indexes(profiles, request) == (0, 2)
    with pytest.raises(ProviderParameterError):
        require_chat_logprobs((profiles[1],), request)


def test_reasoning_logprobs_require_explicit_effective_effort_support() -> None:
    """Unknown combinations reject, and a declared default counts when effort is omitted."""
    request = _chat_request().model_copy(update={"logprobs": True})
    profile = replace(
        _profile(),
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_effort="low",
        supported_reasoning_efforts=("none", "low"),
    )
    with pytest.raises(ProviderParameterError):
        require_chat_logprobs((profile,), request)
    profile = replace(profile, logprobs_reasoning_efforts=("none",))
    with pytest.raises(ProviderParameterError):
        require_chat_logprobs((profile,), request)
    require_chat_logprobs((profile,), request.model_copy(update={"reasoning_effort": "none"}))
    assert (
        coerce_generation_parameters(
            (profile,), request.model_copy(update={"reasoning_effort": "high"})
        )
        is None
    )


def test_other_surface_and_count_without_opt_in_reject() -> None:
    """Direct canonical requests cannot bypass Chat-only and count-dependency gates."""
    with pytest.raises(ProviderParameterError):
        require_chat_logprobs((_profile(),), _chat_request().model_copy(update={"top_logprobs": 0}))
    with pytest.raises(ProviderParameterError):
        require_chat_logprobs(
            (_profile(),),
            _chat_request().model_copy(
                update={"logprobs": True, "surface": GatewayApiSurface.RESPONSES}
            ),
        )


def test_output_rewriting_rejects_only_active_probability_requests() -> None:
    """Output transforms must not silently invalidate the requested token records."""
    from exp.runtime.models.providers.logprobs import require_unmodified_probability_output

    request = _chat_request().model_copy(update={"logprobs": True})
    with pytest.raises(ProviderParameterError, match="output guardrails"):
        require_unmodified_probability_output(request, True)
    require_unmodified_probability_output(request, False)
    require_unmodified_probability_output(_chat_request(), True)

# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Tests for the pre-dispatch context-window narrowing and refusal."""

from __future__ import annotations

import base64

import pytest

from exp.common.models.content import ImageContentPart, TextContentPart
from exp.common.models.dispatch_policy import GatewayRungDispatchPolicy
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.common.models.model import ModelCapabilities, ToolCall
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.execution_snapshot import ExecutionSnapshot
from exp.runtime.gateway.prompt_size import (
    MAXIMUM_BYTES_PER_TOKEN,
    TEMPLATE_RESERVE_TOKENS,
    TYPICAL_BYTES_PER_TOKEN,
    clamp_output_to_context_window,
    context_window_compatible_indexes,
    minimum_prompt_tokens,
    prompt_cap_indexes,
    prompt_text_bytes,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.errors import ProviderParameterError


def _deployment(deployment_id: str, context_window_tokens: int | None) -> ExactModelDeployment:
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai",
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        capabilities=(
            None
            if context_window_tokens is None
            else ModelCapabilities(context_window_tokens=context_window_tokens)
        ),
    )


def _route(*deployments: ExactModelDeployment) -> GatewayRoute:
    authorization = AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256="a" * 64,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
            failover_mode="maximize_availability",
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


def _request(
    text: str,
    surface: GatewayApiSurface = GatewayApiSurface.CHAT_COMPLETIONS,
    *,
    max_tokens: int | None = None,
) -> GatewayRequest:
    return GatewayRequest(
        surface=surface,
        messages=(GatewayMessage(role="user", content=text),),
        maximum_output_tokens=max_tokens,
        maximum_output_tokens_parameter=None if max_tokens is None else "max_tokens",
    )


def _tokens(count: int) -> str:
    """Text whose lower-bound token count is exactly ``count``."""
    return "x" * (count * MAXIMUM_BYTES_PER_TOKEN)


def test_text_bytes_count_every_text_the_model_reads_and_no_media() -> None:
    """Message text, tool calls, and tool schemas count once each; inline images do not."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="system", content="sys"),
            GatewayMessage(
                role="user",
                # The decoders mirror text parts into ``content``; it is counted once.
                content="héllo",
                content_parts=(
                    TextContentPart(text="héllo"),
                    ImageContentPart(
                        media_type="image/png",
                        data=base64.b64encode(b"\x89PNG" * 100).decode("ascii"),
                    ),
                ),
            ),
            GatewayMessage(
                role="assistant",
                content=None,
                tool_calls=(ToolCall(call_id="call_1", name="lookup", arguments={"q": "x"}),),
            ),
        ),
        tools=(
            GatewayToolDefinition(name="lookup", description="d", parameters={"type": "object"}),
        ),
    )
    expected = (
        len(b"sys")
        + len("héllo".encode())
        + len(b"lookup")
        + len(b'{"q":"x"}')
        + len(b"lookup")
        + len(b"d")
        + len(b'{"type":"object"}')
    )
    assert prompt_text_bytes(request) == expected
    assert minimum_prompt_tokens(request) == expected // MAXIMUM_BYTES_PER_TOKEN


def test_raw_tool_arguments_are_counted_verbatim_when_present() -> None:
    """A provider's exact argument text outranks the parsed object for byte counting."""
    call = ToolCall(call_id="call_1", name="f", arguments={"a": 1}, raw_arguments='{"a":   1}')
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="assistant", content=None, tool_calls=(call,)),),
    )
    assert prompt_text_bytes(request) == len(b"f") + len(b'{"a":   1}')


def test_a_prompt_certain_to_overflow_every_window_is_refused_with_the_numbers() -> None:
    """The lower bound exceeds the largest window: refuse before dispatch, naming both numbers."""
    route = _route(_deployment("small", 1_000), _deployment("large", 2_000))
    # 2,001 tokens even at the most generous bytes-per-token; certain to fail on both rungs.
    request = _request(_tokens(2_001))

    with pytest.raises(ProviderParameterError) as caught:
        context_window_compatible_indexes(route, request)

    assert caught.value.code == "context_length_exceeded"
    assert caught.value.param == "messages"
    assert "at least 2,001 tokens" in str(caught.value)
    assert "largest context window on this model route is 2,000 tokens" in str(caught.value)


def test_a_prompt_over_a_small_rung_falls_to_the_larger_rung() -> None:
    """The Experiential Cloud shape: a 262k box beside 1M rungs skips the box, never refuses."""
    route = _route(_deployment("box", 1_000), _deployment("large", 2_000))
    assert context_window_compatible_indexes(route, _request(_tokens(1_500))) == (1,)
    # Exactly at a rung's bound is not over it.
    assert context_window_compatible_indexes(route, _request(_tokens(1_000))) == (0, 1)
    assert context_window_compatible_indexes(route, _request(_tokens(2_000))) == (1,)


def test_the_requested_output_budget_counts_against_each_rung() -> None:
    """116 of the 140 qwen refusals fit the box on prompt alone and died on prompt + max_tokens."""
    route = _route(_deployment("box", 1_000), _deployment("large", 2_000))
    # 600 prompt tokens fit the box; 600 + 500 requested output does not.
    assert context_window_compatible_indexes(route, _request(_tokens(600), max_tokens=500)) == (1,)
    assert context_window_compatible_indexes(route, _request(_tokens(600), max_tokens=400)) == (
        0,
        1,
    )
    # No requested budget reserves nothing: the provider sizes the output to its room.
    assert context_window_compatible_indexes(route, _request(_tokens(999))) == (0, 1)


def test_the_context_check_never_raises_a_small_caller_ceiling() -> None:
    """Provider-floor incompatibility belongs to generation policy, not a budget rewrite."""
    route = _route(_deployment("first", 1_000), _deployment("second", 1_000))
    request = _request(_tokens(990), max_tokens=1)
    assert context_window_compatible_indexes(route, request) == (0, 1)


def test_no_rung_holds_prompt_plus_budget_refuses_naming_the_budget_field() -> None:
    """When only max_tokens breaks it, the refusal points at max_tokens and says what fits."""
    route = _route(_deployment("box", 1_000), _deployment("large", 2_000))
    request = _request(_tokens(1_500), max_tokens=600)

    with pytest.raises(ProviderParameterError) as caught:
        context_window_compatible_indexes(route, request)

    assert caught.value.code == "context_length_exceeded"
    assert caught.value.param == "max_tokens"
    assert "reserves 600 output tokens" in str(caught.value)
    assert "Lower max_tokens to at most 500" in str(caught.value)
    assert "2,000 tokens" in str(caught.value)


def test_routes_without_a_declared_window_never_refuse() -> None:
    """No declaration means no bound: the provider's own count decides."""
    route = _route(_deployment("unknown", None))
    assert context_window_compatible_indexes(route, _request("x" * 10_000_000)) == (0,)


def test_an_undeclared_rung_stays_while_a_declared_small_rung_is_skipped() -> None:
    """An undeclared rung is permissive; a declared rung that cannot hold the request is not."""
    route = _route(_deployment("small", 100), _deployment("unknown", None))
    assert context_window_compatible_indexes(route, _request(_tokens(10_000))) == (1,)
    assert context_window_compatible_indexes(route, _request(_tokens(50))) == (0, 1)


def test_the_responses_surface_names_its_input_field() -> None:
    """The refusal points at the field the caller actually sent."""
    route = _route(_deployment("small", 10))
    request = _request(_tokens(11), GatewayApiSurface.RESPONSES)
    with pytest.raises(ProviderParameterError) as caught:
        context_window_compatible_indexes(route, request)
    assert caught.value.param == "input"


def test_an_oversized_ceiling_is_lowered_to_what_the_largest_window_leaves() -> None:
    """Production 2026-10-17: max_tokens=10,000,000 beside a two-token prompt on a 131k route."""
    route = _route(_deployment("small", 32_768), _deployment("large", 131_072))
    request = _request("hi there", max_tokens=10_000_000)

    clamped, disclosures = clamp_output_to_context_window(route, request)

    target = 131_072 - len(b"hi there") - TEMPLATE_RESERVE_TOKENS
    assert clamped.maximum_output_tokens == target
    assert disclosures == (f"max_tokens->{target}(context_window)",)
    # The lowered request now fits the large rung and narrows to it.
    assert context_window_compatible_indexes(route, clamped) == (1,)
    # The caller's own spelling is named.
    responses = request.model_copy(update={"maximum_output_tokens_parameter": "max_output_tokens"})
    assert clamp_output_to_context_window(route, responses)[1] == (
        f"max_output_tokens->{target}(context_window)",
    )


def test_a_large_prompt_is_sized_at_typical_bytes_when_one_per_byte_leaves_nothing() -> None:
    """The strict bound cannot fit 3,000 bytes in 2,000; the typical estimate can."""
    route = _route(_deployment("large", 2_000))
    request = _request("x" * 3_000, max_tokens=5_000)

    clamped, _ = clamp_output_to_context_window(route, request)

    typical = 3_000 // TYPICAL_BYTES_PER_TOKEN
    assert clamped.maximum_output_tokens == 2_000 - typical - TEMPLATE_RESERVE_TOKENS


def test_the_window_clamp_leaves_fitting_permissive_and_hopeless_requests_alone() -> None:
    """Only the ceiling ever changes, and only when it alone is the blocker."""
    route = _route(_deployment("box", 1_000), _deployment("large", 2_000))
    for request in (
        # Some rung already holds it verbatim.
        _request(_tokens(10), max_tokens=1_500),
        # No ceiling to lower.
        _request(_tokens(10)),
        # The prompt alone overflows every window: the refusal stands.
        _request(_tokens(2_001), max_tokens=10_000_000),
    ):
        assert clamp_output_to_context_window(route, request) == (request, ())
    with pytest.raises(ProviderParameterError) as caught:
        context_window_compatible_indexes(
            route,
            clamp_output_to_context_window(route, _request(_tokens(2_001), max_tokens=10))[0],
        )
    assert caught.value.param == "messages"
    # A rung that declares no window is permissive: the provider decides.
    permissive = _route(_deployment("open", None), _deployment("large", 2_000))
    request = _request("hi", max_tokens=10_000_000)
    assert clamp_output_to_context_window(permissive, request) == (request, ())


def _capped(deployment: ExactModelDeployment, cap: int) -> ExactModelDeployment:
    """``deployment`` with an authored prompt cap (a dedicated :fast replica)."""
    return deployment.model_copy(
        update={
            "gateway": deployment.gateway.model_copy(
                update={"dispatch": GatewayRungDispatchPolicy(max_prompt_tokens=cap)}
            )
        }
    )


def test_a_prompt_cap_skips_its_rung_by_prompt_only() -> None:
    """A long prompt leaves the capped replica for the main lane; a big max_tokens never does."""
    route = _route(_capped(_deployment("fast", 200_000), 1_000), _deployment("main", 200_000))
    assert prompt_cap_indexes(route, _request(_tokens(999))) is None
    assert prompt_cap_indexes(route, _request(_tokens(1_000))) == (1,)
    assert prompt_cap_indexes(route, _request(_tokens(10), max_tokens=100_000)) is None


def test_capped_rungs_stay_when_no_uncapped_rung_of_their_model_is_left() -> None:
    """Every rung capped: kept, never a refusal. A cap never moves a request to another model."""
    alone = _route(_capped(_deployment("fast", 200_000), 1_000))
    assert prompt_cap_indexes(alone, _request(_tokens(2_000))) is None
    other_model = _deployment("other", 200_000).model_copy(update={"exact_model_id": "exact-two"})
    mixed = _route(_capped(_deployment("fast", 200_000), 1_000), other_model)
    assert prompt_cap_indexes(mixed, _request(_tokens(2_000))) is None

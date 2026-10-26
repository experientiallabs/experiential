"""Tests for the disclosed route repairs admission applies after verbatim narrowing fails."""

from __future__ import annotations

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.capability_policy import coerce_generation_parameters
from exp.runtime.models.providers.route_repairs import (
    ENABLE_THINKING_DROPPED,
    ENCRYPTED_REASONING_HISTORY_DROPPED,
    ENCRYPTED_REASONING_INCLUDE_DROPPED,
    apply_route_repairs,
    clamp_output_ceiling,
    clamp_sampling_range,
    drop_unservable_enable,
    drop_unservable_encrypted_reasoning,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses


def _request(
    surface: GatewayApiSurface = GatewayApiSurface.MESSAGES, **fields: object
) -> GatewayRequest:
    """Build one minimal request on ``surface``."""
    base = GatewayRequest(surface=surface, messages=(GatewayMessage(role="user", content="go"),))
    return base.model_copy(update=fields)


_GEMINI = GatewayWireProfile(
    dialect="gemini_generate_content", url="https://g.test", maximum_output_tokens=32_768
)
_RESPONSES = GatewayWireProfile(dialect="openai_responses", url="https://o.test")


def test_a_ceiling_above_the_route_maximum_is_lowered_to_it() -> None:
    """Claude Code's 64000 on a 32768 route serves at 32768 (2,137 refusals a day)."""
    request = _request(maximum_output_tokens=64_000, maximum_output_tokens_parameter="max_tokens")
    coercion = coerce_generation_parameters((_GEMINI,), request)
    assert coercion is not None
    assert coercion.request.maximum_output_tokens == 32_768
    assert coercion.disclosures == ("max_tokens->32768(route_maximum)",)


def test_a_ceiling_below_the_route_floor_is_raised_to_it() -> None:
    """A probe below the Responses floor of 16 serves at 16 (2,644 refusals a day)."""
    request = _request(
        GatewayApiSurface.RESPONSES,
        maximum_output_tokens=1,
        maximum_output_tokens_parameter="max_output_tokens",
    )
    coercion = coerce_generation_parameters((_RESPONSES,), request)
    assert coercion is not None
    assert coercion.request.maximum_output_tokens == 16
    assert coercion.disclosures == ("max_output_tokens->16(route_minimum)",)


_THINKING_CLAUDE = GatewayWireProfile(
    dialect="anthropic_messages",
    url="https://a.test",
    minimum_temperature=1.0,
    maximum_temperature=1.0,
    supports_top_p=True,
    minimum_top_p=0.95,
    maximum_top_p=1.0,
)
"""A route whose sampling is locked, as Claude's thinking lanes are in the catalog."""


def test_a_temperature_below_a_locked_route_is_raised_onto_it() -> None:
    """claude-haiku-5.5 refused temperature 0.1 as "between 1.0 and 1.0" (41 a day)."""
    coercion = coerce_generation_parameters((_THINKING_CLAUDE,), _request(temperature=0.1))
    assert coercion is not None
    assert coercion.request.temperature == 1.0
    assert coercion.disclosures == ("temperature->1.0(route_minimum)",)


def test_a_top_p_outside_the_route_is_clamped_beside_temperature() -> None:
    """Both sampling controls repair together, each disclosed by name."""
    coercion = coerce_generation_parameters(
        (_THINKING_CLAUDE,), _request(temperature=1.5, top_p=0.5)
    )
    assert coercion is not None
    assert (coercion.request.temperature, coercion.request.top_p) == (1.0, 0.95)
    assert coercion.disclosures == (
        "temperature->1.0(route_maximum)",
        "top_p->0.95(route_minimum)",
    )


def test_a_sampling_value_some_rung_carries_is_never_clamped() -> None:
    """A rung that accepts the value, or drops an undeclared control, keeps it verbatim."""
    open_rung = GatewayWireProfile(dialect="openai_compatible", url="https://r.test")
    blind_rung = GatewayWireProfile(
        dialect="openai_compatible", url="https://b.test", supports_temperature=False
    )
    request = _request(temperature=0.1)
    assert clamp_sampling_range((_THINKING_CLAUDE, open_rung), request) is None
    assert clamp_sampling_range((_THINKING_CLAUDE, blind_rung), request) is None


def test_a_sampling_value_outside_the_public_bounds_is_still_refused() -> None:
    """Clamping is a route repair, never a validity repair: -1 and NaN stay 400s."""
    for value in (-1, float("nan"), 3):
        try:
            decode_chat(
                {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": value}
            )
        except OpenAIProtocolError as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError(f"temperature {value!r} was admitted")


def test_a_ceiling_some_rung_honors_is_never_clamped() -> None:
    """Narrowing keeps the honoring rung; the repair applies only when none does."""
    roomy = GatewayWireProfile(dialect="openai_compatible", url="https://r.test")
    request = _request(maximum_output_tokens=64_000)
    assert clamp_output_ceiling((_GEMINI, roomy), request) is None
    assert clamp_output_ceiling((_GEMINI,), _request()) is None
    plan = GatewayWireProfile(
        dialect="openai_responses", url="https://plan.test", omits_output_token_limit=True
    )
    assert clamp_output_ceiling((plan,), _request(maximum_output_tokens=1)) is None


def test_an_enable_toggle_on_a_route_that_cannot_reason_is_dropped() -> None:
    """deepseek-v4.1-flash with chat_template_kwargs.enable_thinking (873 refusals a day)."""
    plain = GatewayWireProfile(dialect="openai_compatible", url="https://d.test", model_id="m")
    request = _request(GatewayApiSurface.CHAT_COMPLETIONS, thinking_default_enable=True)
    repair = drop_unservable_enable((plain,), request)
    assert repair is not None
    assert repair.request.thinking_default_enable is False
    assert repair.disclosures == (ENABLE_THINKING_DROPPED,)
    reasoning = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://r.test",
        model_id="m",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
        supported_reasoning_efforts=("none", "low", "high"),
    )
    assert drop_unservable_enable((plain, reasoning), request) is None
    budgeted = request.model_copy(update={"thinking_budget": 2048})
    assert drop_unservable_enable((plain,), budgeted) is None


def test_the_encrypted_reasoning_selector_drops_only_without_any_carrier() -> None:
    """Claude routes cannot seal reasoning (575 refusals a day); Responses rungs can."""
    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://a.test")
    request = _request(GatewayApiSurface.RESPONSES, include_encrypted_reasoning=True)
    repair = drop_unservable_encrypted_reasoning((anthropic,), request)
    assert repair is not None
    assert repair.request.include_encrypted_reasoning is False
    assert repair.disclosures == (ENCRYPTED_REASONING_INCLUDE_DROPPED,)
    assert drop_unservable_encrypted_reasoning((anthropic, _RESPONSES), request) is None


def test_replayed_native_reasoning_is_removed_only_where_no_rung_can_read_it() -> None:
    """History from an OpenAI rung reaches a non-Responses route without the payload."""
    decoded = decode_responses(
        {
            "model": "m",
            "input": [
                {"role": "user", "content": "go"},
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "b=="},
                {"type": "message", "role": "assistant", "content": "done"},
                {"role": "user", "content": "again"},
            ],
        }
    )
    chat = GatewayWireProfile(dialect="openai_compatible", url="https://c.test")
    repair = drop_unservable_encrypted_reasoning((chat,), decoded.request)
    assert repair is not None
    assert repair.disclosures == (ENCRYPTED_REASONING_HISTORY_DROPPED,)
    assert not any(
        block.kind == "encrypted_reasoning"
        for message in repair.request.messages
        for block in message.provider_reasoning
    )
    assert [message.content for message in repair.request.messages] == ["go", "done", "again"]
    assert drop_unservable_encrypted_reasoning((_RESPONSES,), decoded.request) is None


def test_independent_repairs_compose() -> None:
    """One request can need several repairs; all apply and all are disclosed."""
    request = _request(
        GatewayApiSurface.RESPONSES,
        maximum_output_tokens=64_000,
        maximum_output_tokens_parameter="max_output_tokens",
        include_encrypted_reasoning=True,
    )
    repair = apply_route_repairs((_GEMINI,), request)
    assert repair is not None
    assert repair.request.maximum_output_tokens == 32_768
    assert repair.disclosures == (
        "max_output_tokens->32768(route_maximum)",
        ENCRYPTED_REASONING_INCLUDE_DROPPED,
    )

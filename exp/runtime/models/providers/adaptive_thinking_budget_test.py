# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Adaptive-to-budgeted thinking translation for budgeted-enabled Anthropic routes."""

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.adaptive_thinking_budget import budgeted_enabled_config
from exp.runtime.models.providers.base import GatewayWireProfile

_HAIKU = GatewayWireProfile(
    dialect="anthropic_messages",
    url="https://anthropic.test",
    model_id="claude-haiku-4-5",
    supports_reasoning=True,
    reasoning_wire_format="anthropic_adaptive",
)


def _request(config: JsonObject, maximum: int | None) -> GatewayRequest:
    """Build one Messages request with a thinking config and output ceiling."""
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
        provider_thinking_config=config,
        maximum_output_tokens=maximum,
    )


def test_adaptive_translates_with_a_derived_budget_or_defers_it() -> None:
    """A known ceiling derives the budget; an unknown one leaves it to the rung."""
    assert budgeted_enabled_config((_HAIKU,), _request({"type": "adaptive"}, 8_000)) == {
        "type": "enabled",
        "budget_tokens": 4_000,
    }
    assert budgeted_enabled_config((_HAIKU,), _request({"type": "adaptive"}, None)) == {
        "type": "enabled"
    }


def test_only_adaptive_configs_on_budgeted_routes_translate() -> None:
    """Other configs, other routes, and an impossible ceiling are left alone."""
    assert budgeted_enabled_config((_HAIKU,), _request({"type": "disabled"}, 8_000)) is None
    shim = GatewayWireProfile(dialect="openai_compatible", url="https://shim.test")
    assert budgeted_enabled_config((shim,), _request({"type": "adaptive"}, 8_000)) is None
    assert budgeted_enabled_config((_HAIKU,), _request({"type": "adaptive"}, 512)) is None

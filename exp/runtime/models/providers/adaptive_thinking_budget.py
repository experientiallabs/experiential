# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Translate an adaptive thinking config for a budgeted-enabled Anthropic route.

A budgeted-enabled Anthropic model (haiku-4-5) reasons and honors a
``thinking: {type: enabled, budget_tokens}`` config but rejects ``adaptive`` by
name. Capability coercion offers the translation built here, disclosed, after
every rung declined the verbatim request.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.reasoning_compat import (
    MINIMUM_THINKING_BUDGET_TOKENS,
    anthropic_budgeted_enabled_only,
    anthropic_thinking_budget_tokens,
)

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile


def _all_budgeted_enabled_anthropic(profiles: Sequence[GatewayWireProfile]) -> bool:
    """Whether every rung is an Anthropic budgeted-enabled-only reasoning route.

    A budgeted-enabled-only rung reasons and honors a ``thinking: {type:
    enabled, budget_tokens}`` config but rejects ``adaptive`` by name (haiku-4-5).
    A mixed, non-reasoning, or adaptive-accepting route (the effort generation)
    is not one, so an adaptive config there is dropped or left verbatim rather
    than translated.
    """
    if not profiles or not all(profile.dialect == "anthropic_messages" for profile in profiles):
        return False
    return all(
        profile.supports_reasoning and anthropic_budgeted_enabled_only(profile.model_id)
        for profile in profiles
    )


def _caller_or_derived_budget(
    config: Mapping[str, object],
    maximum_output_tokens: int | None,
) -> int | None:
    """Return a legal translated budget, or None to preserve the refusal.

    A caller-supplied budget is immutable. Only an omitted budget can be
    derived from a known output ceiling; an omitted ceiling is deferred to
    per-rung payload construction.
    """
    caller_budget = config.get("budget_tokens")
    if isinstance(caller_budget, int) and not isinstance(caller_budget, bool):
        ceiling = maximum_output_tokens if maximum_output_tokens is not None else caller_budget + 1
        if MINIMUM_THINKING_BUDGET_TOKENS <= caller_budget < ceiling:
            return caller_budget
    if "budget_tokens" in config or maximum_output_tokens is None:
        return None
    return anthropic_thinking_budget_tokens(maximum_output_tokens)


def budgeted_enabled_config(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
) -> JsonObject | None:
    """Translate an adaptive thinking config for a budgeted-enabled route.

    A budgeted-enabled Anthropic model rejects adaptive mode but accepts an
    enabled config. Translation preserves a legal explicit budget, or derives
    an omitted one from the caller's known ceiling. With neither a budget nor
    a ceiling it defers derivation to the selected rung's payload builder.
    An illegal explicit budget or an impossible ceiling keeps the refusal;
    neither can turn the requested reasoning off. History blocks stay intact.

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: Decoded public request that no rung accepted verbatim.

    Returns:
        The translated ``enabled`` config, or ``None`` when the request is not
        an adaptive config on a budgeted-enabled Anthropic route or no legal
        budget exists.
    """
    config = request.provider_thinking_config
    if config is None or config.get("type") != "adaptive":
        return None
    if not _all_budgeted_enabled_anthropic(profiles):
        return None
    budget = _caller_or_derived_budget(config, request.maximum_output_tokens)
    deferred = request.maximum_output_tokens is None and "budget_tokens" not in config
    if budget is None and not deferred:
        return None
    translated: JsonObject = {"type": "enabled"}
    if budget is not None:
        translated["budget_tokens"] = budget
    return translated

"""Freeze native cache plans while retaining durable authority checks at dispatch."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.models.providers.google_cache import GoogleCachePlan


def _plan(plan: GoogleCachePlan) -> JsonObject:
    return {
        "create_url": plan.create_url,
        "model": plan.model,
        "resource_prefix": plan.resource_prefix,
        "prefix_sha256": plan.prefix_sha256,
        "conservative_input_bound": plan.conservative_input_bound,
        "create_payload": plan.create_payload,
        "generation_payload": plan.generation_payload,
    }


def cache_registration(entry: InflightRequest) -> JsonObject | None:
    """Export bodies privately; no admission snapshot grants cache spending authority."""
    state = entry.explicit_cache_state
    if state is None:
        return None
    return {
        "automatic": state.automatic,
        "bindings": [
            None
            if binding is None
            else {
                "deployment_id": binding.deployment.deployment_id,
                "connection": binding.deployment.connection_sha256,
                "plan": _plan(binding.plan),
                "automatic_plans": [_plan(plan) for plan in binding.automatic_plans],
            }
            for binding in state.bindings
        ],
    }

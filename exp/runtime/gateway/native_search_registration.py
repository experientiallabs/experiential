"""Compile each search-capable rung once before native request ownership begins."""

from __future__ import annotations

from functools import partial

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_reasoning import rung_provider_request
from exp.runtime.gateway.native_rungs import build_rung_dispatch
from exp.runtime.gateway.native_search_counter import prompt_registration
from exp.runtime.gateway.native_search_subject import input_templates, subject_registration
from exp.runtime.gateway.native_search_wires import compile_wire, tool_registration
from exp.runtime.models.providers.protocol import emulated_gateway_capabilities


def search_registration(entry: InflightRequest) -> JsonObject | None:
    """Freeze search policy, provider builders, counters and scoped reinspection authority."""
    state = entry.tool_search
    if state is None:
        return None
    request = entry.request
    if (
        not isinstance(request, GatewayRequest)
        or entry.resolved_wires is None
        or entry.public_request is None
    ):
        message = "Native tool search requires its complete admitted route"
        raise ValueError(message)
    plans: list[JsonValue] = []
    for depth, _ in enumerate(entry.route.deployments):
        plans.append(
            compile_wire(
                request,
                state,
                partial(_build, entry, depth),
                profile=entry.resolved_wires[depth][0],
                strip_reasoning=entry.route.requires_reasoning_strip(
                    entry.route.deployments[depth]
                ),
                requirements=_requirements(entry, depth),
            )
        )
    tools = (*state.loaded, *state.deferred, state.gateway_tool())
    return {
        "search": {
            "mode": state.search.mode,
            "tool_name": state.tool_name,
            "declared_name": state.search.tool_name or state.search.tool_type,
            "max_rounds": state.search.max_rounds,
            "default_limit": state.search.default_limit,
            "loaded": [tool_registration(t) for t in state.loaded],
            "deferred": [tool_registration(t) for t in state.deferred],
            "gateway_tool": tool_registration(state.gateway_tool()),
        },
        "plans": plans,
        "counter": prompt_registration(request, tools),
        "subject": subject_registration(request, tools),
        "checks": list[JsonValue](input_templates(entry.guardrails)),
        "observations": list[JsonValue](input_templates(entry.guardrails, observing=True)),
    }


def _requirements(entry: InflightRequest, depth: int) -> JsonObject:
    """Retain model and deployment capability fences for newly rewritten messages."""
    assert entry.resolved_wires is not None
    assert entry.public_request is not None
    deployment = entry.route.deployments[depth]
    profile = entry.resolved_wires[depth][0]
    model = deployment.capabilities
    return {
        "provider": deployment.provider,
        "capabilities": deployment.gateway.capabilities.model_dump(mode="json"),
        "emulated": list[JsonValue](
            sorted(emulated_gateway_capabilities(profile.dialect, emulate_parallel_tool_calls=True))
        ),
        "model_tools": model is None or model.supports_tools is not False,
        "public_stream": entry.public_request.stream,
    }


def _build(entry: InflightRequest, depth: int, candidate: GatewayRequest) -> JsonObject:
    """Use the same rung shaping and signer binding as the ordinary dispatch builder."""
    if entry.resolved_wires is None or entry.public_request is None:
        message = "Native tool search requires its complete admitted route"
        raise ValueError(message)
    deployment = entry.route.deployments[depth]
    profile, client = entry.resolved_wires[depth]
    budget = (
        entry.throttle_redial_budgets[depth] if depth < len(entry.throttle_redial_budgets) else 0
    )
    return build_rung_dispatch(
        entry.route,
        deployment,
        profile,
        client,
        provider_request=rung_provider_request(entry.route, deployment, candidate),
        public_request=entry.public_request,
        authorization=entry.authorization,
        throttle_redial_budget=budget,
    ).wire_entry

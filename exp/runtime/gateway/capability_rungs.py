"""Capability-first rung selection for hard request features some rungs lack.

Admission ranks a route's rungs by how exactly they keep the caller's soft
preferences (sampling, thinking) before it checks the hard capabilities each
rung's catalog declares. When the rung that keeps the preferences exactly is
the one that cannot carry a hard feature, that ranking narrows the route to it
and the capability check then refuses a request another rung could serve
(qwen3.8-27b: a structured-output request with a ``temperature`` its
schema-enforcing rung does not take, 2026-10-17). A hard feature therefore
narrows the route FIRST, to the rungs that declare it, in route order; the
soft preferences are judged among those rungs only.

The hard features handled here are all decided by the decoded request alone:

- Function tools and media inputs (images, video, audio, PDFs): content the
  caller sent that no shaping removes, and that the per-rung preflight
  refuses on any rung declaring it unsupported. Narrowing to the rungs that
  declare it therefore never drops a rung that could have served; it only
  stops the sampling preference from picking one that cannot (gpt-6-astra:
  GitHub Copilot's tools with ``temperature`` and ``top_p``, which only the
  tool-less Novita rung takes, were refused as unsupported ``tools`` while
  the OpenAI rung serves tools, 2026-10-20). They narrow first.
- Structured output (``response_format`` / ``text.format`` json_schema): no
  later shaping or coercion can remove it. It narrows among the rungs the
  content features kept, so when no rung carries both the refusal names the
  feature the kept rungs lack rather than the one they carry.
- Explicit thinking (an active ``thinking`` config, an enable-thinking toggle,
  or an effort above ``none``): a rung that cannot reason would otherwise lead
  the per-rung ranking and the route-wide shaping would refuse a request the
  reasoning rungs serve (claude-sonnet-5 behind a non-reasoning Novita lead,
  deepseek-v4-flash with a rung whose effort ladder is undeclared, production
  2026-10-18). The route narrows to the reasoning rungs; when their ladders
  share no tier for a level-less enable, it narrows further to the rungs that
  declare one. Structured output narrows first, and thinking narrows only
  among the rungs it kept.
- A thinking off switch (``thinking: {type: disabled}``): it narrows to the
  rungs that honor it exactly (a rung that never reasons, one with a ``none``
  tier, or Anthropic's native wire), so a rung that cannot stop reasoning is
  only used when no rung can.
- OpenAI's 12-hour prompt cache lifetime (``prompt_cache_options.ttl:
  "12h"``): only rungs whose catalog declares ``supports_prompt_cache_ttl_12h``
  and whose wire carries it (OpenAI's own Responses API) write at that
  lifetime and price, so the route narrows to them first and a rung that would
  write at the default lifetime (Azure, OpenRouter, a ChatGPT plan rung, a
  pre-GPT-5.6 model) is never asked. With no such rung the request is refused.

When no rung carries a feature the route is left whole and the existing
checks and disclosed coercions decide. A reasoning-pinned continuation whose
issuing rung is still on the route, and a caller-pinned route, never move.
"""

from __future__ import annotations

from collections.abc import Callable

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayRequest
from exp.runtime.gateway.native_execution import select_route_deployments
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.generation_parameter_validation import (
    profile_reasoning_efforts,
    resolve_level_less_enable,
)
from exp.runtime.models.providers.protocol import NativeWireClient

STRUCTURED_OUTPUT_RUNG_SUFFIX = "->routed(structured_output_rung)"
"""Disclosure suffix recorded when structured output moves the request off the lead rung."""

TOOLS_RUNG_DISCLOSURE = "tools->routed(tools_rung)"
"""Disclosure recorded when function tools move the request off the lead rung."""

REASONING_RUNG_SUFFIX = "->routed(reasoning_rung)"
"""Disclosure suffix recorded when explicit thinking moves the request off the lead rung."""

REASONING_OFF_RUNG_DISCLOSURE = "thinking->routed(reasoning_off_rung)"
"""Disclosure recorded when a thinking off switch moves the request off the lead rung."""

PROMPT_CACHE_TTL_RUNG_DISCLOSURE = "prompt_cache_options.ttl->routed(prompt_cache_ttl_12h_rung)"
"""Disclosure recorded when the 12-hour cache lifetime moves the request off the lead rung."""

_ResolvedWires = tuple[tuple[GatewayWireProfile, NativeWireClient], ...]


def serves_structured_output(deployment: ExactModelDeployment) -> bool:
    """Whether one rung's catalog declares structured output, as preflight judges it.

    Mirrors ``preflight_gateway_request``: the deployment must declare
    structured text, and its model capabilities (when the catalog carries
    them) must declare structured output.
    """
    model = deployment.capabilities
    return deployment.gateway.capabilities.supports_structured_text and (
        model is None or model.supports_structured_output
    )


def serves_function_tools(deployment: ExactModelDeployment) -> bool:
    """Whether one rung's catalog declares function tools, as preflight judges it.

    Mirrors ``preflight_gateway_request``: only an explicit
    ``supports_tools=False`` refuses, and a rung whose catalog carries no model
    capabilities is not judged on tools at all.
    """
    model = deployment.capabilities
    return model is None or model.supports_tools is not False


def _media_requirements(
    request: GatewayRequest,
) -> tuple[tuple[str, Callable[[ExactModelDeployment], bool]], ...]:
    """Media inputs the request carries, each with the deployment flag preflight reads."""
    media: tuple[tuple[bool, str, Callable[[ExactModelDeployment], bool]], ...] = (
        (
            bool(request.images),
            "image_input",
            lambda deployment: deployment.gateway.capabilities.supports_image_input,
        ),
        (
            bool(request.videos),
            "video_input",
            lambda deployment: deployment.gateway.capabilities.supports_video_input,
        ),
        (
            bool(request.audios),
            "audio_input",
            lambda deployment: deployment.gateway.capabilities.supports_audio_input,
        ),
        (
            bool(request.documents),
            "pdf_input",
            lambda deployment: deployment.gateway.capabilities.supports_pdf_input,
        ),
    )
    return tuple((label, serves) for present, label, serves in media if present)


def requests_thinking(request: GatewayRequest) -> bool:
    """Whether the caller explicitly asked the model to reason.

    An active ``thinking`` config (``enabled`` or ``adaptive``), a level-less
    enable toggle, or an effort above ``none``. An off switch and an absent
    signal ask for nothing a non-reasoning rung cannot give; a numeric budget
    (``budget_tokens`` or the Chat ``thinking_budget``) already narrows to the
    budget-capable rungs in the per-rung check.
    """
    config = request.provider_thinking_config
    if config is not None and config.get("type") in ("enabled", "adaptive"):
        return "budget_tokens" not in config
    if request.reasoning_effort is not None:
        return request.reasoning_effort != "none"
    return request.thinking_default_enable


def _honors_thinking_off(profile: GatewayWireProfile) -> bool:
    """Whether one rung serves a thinking off switch without reasoning."""
    return (
        profile.dialect == "anthropic_messages"
        or not profile.supports_reasoning
        or "none" in profile_reasoning_efforts(profile)
    )


def _thinking_parameter(request: GatewayRequest) -> str:
    """The public field that asked for thinking, named in the reorder disclosure."""
    if request.provider_thinking_config is not None:
        return "thinking"
    if request.reasoning_effort is None and request.thinking_default_enable:
        return "enable_thinking"
    return request.caller_effort_parameter


def _reasoning_indexes(
    profiles: tuple[GatewayWireProfile, ...], request: GatewayRequest
) -> tuple[int, ...]:
    """Route positions that can carry the caller's explicit thinking, in route order.

    Every reasoning rung qualifies. A level-less enable additionally needs
    the kept rungs to share a tier it can pin (the route-wide shaping pins
    one effort for every rung): when the reasoning rungs share none because
    some declare no ladder at all, only the rungs that declare one are kept,
    and when even those share none the reasoning set stands and the shaping
    names the refusal.
    """
    reasoning = tuple(index for index, profile in enumerate(profiles) if profile.supports_reasoning)
    if not request.thinking_default_enable or request.reasoning_effort is not None:
        return reasoning
    for candidate in (
        reasoning,
        tuple(
            index
            for index in reasoning
            if set(profile_reasoning_efforts(profiles[index])) - {"none"}
        ),
    ):
        if not candidate:
            continue
        try:
            resolve_level_less_enable(
                tuple(profiles[index] for index in candidate),
                effort_path=request.caller_effort_parameter,
            )
        except ProviderParameterError:
            continue
        return candidate
    return reasoning


def narrow_to_capable_rungs(
    route: GatewayRoute, resolved_wires: _ResolvedWires, request: GatewayRequest
) -> tuple[GatewayRoute, _ResolvedWires, tuple[str, ...]]:
    """Keep only the rungs that can carry the request's hard features.

    Args:
        route: Frozen ordered route aligned with ``resolved_wires``.
        resolved_wires: Ordered wire profiles and clients per deployment.
        request: Decoded public request.

    Returns:
        The narrowed route and wires plus the reorder disclosures, each
        recorded only when its feature passed over the lead rung. The inputs
        come back unchanged when the request asks for no hard feature, every
        rung or no rung carries it, or the route is pinned to a live rung.
    """
    pinned = route.reasoning_pinned_deployment_id
    if route.resolved_route_id is not None or any(
        deployment.deployment_id == pinned for deployment in route.deployments
    ):
        return route, resolved_wires, ()
    disclosures: list[str] = []
    if request.prompt_cache_ttl == "12h":
        route, resolved_wires = _narrow(
            route,
            resolved_wires,
            tuple(
                index
                for index, deployment in enumerate(route.deployments)
                if deployment.gateway.capabilities.supports_prompt_cache_ttl_12h
                and resolved_wires[index][0].forwards_prompt_cache_ttl
            ),
            PROMPT_CACHE_TTL_RUNG_DISCLOSURE,
            disclosures,
        )
    if request.tools or request.provider_native_tools:
        route, resolved_wires = _narrow(
            route,
            resolved_wires,
            tuple(
                index
                for index, deployment in enumerate(route.deployments)
                if serves_function_tools(deployment)
            ),
            TOOLS_RUNG_DISCLOSURE,
            disclosures,
        )
    content_field = "input" if request.surface == GatewayApiSurface.RESPONSES else "messages"
    for label, serves in _media_requirements(request):
        route, resolved_wires = _narrow(
            route,
            resolved_wires,
            tuple(
                index for index, deployment in enumerate(route.deployments) if serves(deployment)
            ),
            f"{content_field}->routed({label}_rung)",
            disclosures,
        )
    if request.structured_text is not None:
        route, resolved_wires = _narrow(
            route,
            resolved_wires,
            tuple(
                index
                for index, deployment in enumerate(route.deployments)
                if serves_structured_output(deployment)
            ),
            ("text.format" if request.surface == GatewayApiSurface.RESPONSES else "response_format")
            + STRUCTURED_OUTPUT_RUNG_SUFFIX,
            disclosures,
        )
    if requests_thinking(request):
        route, resolved_wires = _narrow(
            route,
            resolved_wires,
            _reasoning_indexes(tuple(profile for profile, _client in resolved_wires), request),
            _thinking_parameter(request) + REASONING_RUNG_SUFFIX,
            disclosures,
        )
    config = request.provider_thinking_config
    if config is not None and config.get("type") == "disabled":
        profiles = tuple(profile for profile, _client in resolved_wires)
        route, resolved_wires = _narrow(
            route,
            resolved_wires,
            tuple(index for index, profile in enumerate(profiles) if _honors_thinking_off(profile)),
            REASONING_OFF_RUNG_DISCLOSURE,
            disclosures,
        )
    return route, resolved_wires, tuple(disclosures)


def _narrow(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    indexes: tuple[int, ...],
    disclosure: str,
    disclosures: list[str],
) -> tuple[GatewayRoute, _ResolvedWires]:
    """Select ``indexes`` unless they keep every rung or none, disclosing a moved lead."""
    if not indexes or len(indexes) == len(route.deployments):
        return route, resolved_wires
    if indexes[0] != 0:
        disclosures.append(disclosure)
    return (
        select_route_deployments(route, indexes),
        tuple(resolved_wires[index] for index in indexes),
    )

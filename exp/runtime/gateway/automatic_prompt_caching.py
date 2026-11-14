"""Gateway-added Anthropic prompt caching for Chat, Responses and generateContent callers.

Anthropic's API, Claude on Vertex AI, Claude on Azure AI Foundry and Claude on
Amazon Bedrock cache a prompt prefix only where the request marks one. A client
that speaks the Messages protocol (Claude Code) marks its prompts itself, but an
OpenAI-style or Google-style client sending Chat Completions, Responses or
generateContent requests never does (those wires have no block marker), so every
turn of its agent loop pays the full input price on a prefix the provider would
have served from cache. Providers that cache prefixes on their own (OpenAI,
DeepSeek) give the same clients most of their input from cache.

When such a request carries no cache marker of its own
and is dispatched to a rung on the Anthropic Messages wire or the Bedrock
Converse wire, this module asks the provider for automatic caching at the
five-minute lifetime:

* On the Anthropic Messages wire, the top-level ``cache_control``
  (``{"type": "ephemeral"}``): the provider places the breakpoint on the last
  cacheable block and moves it forward as the conversation grows.
* On the Bedrock Converse wire, a ``cachePoint`` after the last message's
  content, the same translation the Messages surface's top-level marker
  receives on Bedrock.

Any caller marker or explicit cache choice (``prompt_cache_options``) wins and
disables the gateway's own (a request is never given a second breakpoint the
caller did not place). Nothing is added to a request
routed under zero data retention (:func:`zdr_routed`), when the
organization turned prompt caching off
(``AuthorizationSnapshot.prompt_caching_disabled``), when the operator stopped the
feature for the process (:data:`AUTOMATIC_PROMPT_CACHING_ENV`), when the rung's
authored dispatch policy sets ``automatic_prompt_caching`` to ``False``, or when
the prompt is provably shorter than the smallest prefix any Claude model caches
(:data:`SMALLEST_CACHEABLE_PREFIX_TOKENS`). Messages-surface requests are never
touched: their caller already chose where to cache.

Cost: a cache write is billed at 1.25 times the input rate, a cache read at
about 0.1 times. A prompt below the model's minimum cacheable prefix is never
written by the provider, so it costs nothing extra; a one-off prompt above the
minimum pays the 25% premium once on its prefix, and every later turn that
shares the prefix reads it at about a tenth of the input rate.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayApiSurface, GatewayRequest
from exp.runtime.gateway.prompt_size import prompt_text_bytes
from exp.runtime.models.providers.anthropic import anthropic_official_origin
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.cache_policy import cache_markers

AUTOMATIC_CACHE_CONTROL: JsonObject = {"type": "ephemeral"}
"""The top-level marker sent on the Anthropic Messages wire (five-minute lifetime)."""

BEDROCK_CACHE_POINT: JsonObject = {"cachePoint": {"type": "default"}}
"""The Converse checkpoint appended after the last message's content on Bedrock."""

AUTOMATIC_PROMPT_CACHING_ENV = "EXP_GATEWAY_AUTOMATIC_PROMPT_CACHING"
"""Operator stop for the whole process: ``0``, ``false``, ``off`` or ``no`` disables it."""

SMALLEST_CACHEABLE_PREFIX_TOKENS = 512
"""The smallest minimum cacheable prefix Anthropic documents for any Claude model.

Older models need more (1,024, 2,048 or 4,096 tokens). Below a model's own
minimum the provider writes nothing and bills no write premium, so this floor
only keeps the field off requests that certainly cannot cache on any model.
"""

FRAMING_TOKENS_PER_MESSAGE = 8
"""Allowance for role and turn framing tokens that the text bytes do not count."""

FRAMING_TOKENS_PER_REQUEST = 64
"""Allowance for request-level framing (system wrapper, tool list wrapper)."""

FRAMING_TOKENS_PER_TOOL = 64
"""Allowance for each tool definition's own wrapper fields and schema structure."""

CACHING_WIRES = frozenset({"anthropic_messages", "bedrock_converse_stream"})
"""The wires whose cache checkpoints this module knows how to place."""

_AUTOMATIC_SURFACES = frozenset(
    {
        GatewayApiSurface.CHAT_COMPLETIONS,
        GatewayApiSurface.RESPONSES,
        GatewayApiSurface.GENERATE_CONTENT,
    }
)
"""The provider-neutral conversational surfaces whose clients never mark a prefix themselves."""
_OFF_VALUES = frozenset({"0", "false", "off", "no"})
# The advisor server tool switches its own prompt caching with this key.
_SERVER_TOOL_CACHING = "caching"


def automatic_prompt_caching_stopped(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the operator stopped gateway-added caching for this process.

    Args:
        environ: The environment to read; ``None`` reads ``os.environ``.

    Returns:
        True when :data:`AUTOMATIC_PROMPT_CACHING_ENV` holds an off value.
    """
    raw = (os.environ if environ is None else environ).get(AUTOMATIC_PROMPT_CACHING_ENV, "")
    return raw.strip().lower() in _OFF_VALUES


def caches_by_default(deployment: ExactModelDeployment, profile: GatewayWireProfile) -> bool:
    """Whether a rung that authored no setting is known to honor automatic caching.

    A provider 400 does not fail over, so the default covers only origins where
    the field is verified: Anthropic's own API (the top-level ``cache_control``
    was verified live there) and Claude models on Bedrock that take a
    ``cachePoint`` (Claude 3.5 Haiku and later; the Claude 3 generation and
    Claude 3.5 Sonnet reject one). Claude on Vertex AI and on Azure AI Foundry,
    proxies and other vendors' Anthropic-compatible origins stay off until a
    probe proves the field there; such a rung opts in by authoring
    ``automatic_prompt_caching=True``.

    Args:
        deployment: The rung about to be dispatched.
        profile: Its resolved wire profile.

    Returns:
        True when the rung caches automatically unless it opts out.
    """
    match profile.dialect:
        case "anthropic_messages":
            return deployment.provider == "anthropic" and anthropic_official_origin(profile.url)
        case "bedrock_converse_stream":
            return (
                deployment.provider == "bedrock"
                and _names_claude(deployment, profile)
                and not _names_pre_caching_claude(deployment, profile)
            )
        case _:
            return False


def automatic_prompt_caching_applies(
    deployment: ExactModelDeployment,
    profile: GatewayWireProfile,
    request: GatewayRequest,
    authorization: AuthorizationSnapshot,
    *,
    zdr_constrained: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Decide whether this rung's payload gets the gateway's own cache checkpoint.

    Args:
        deployment: The rung about to be dispatched.
        profile: Its resolved wire profile.
        request: The request this rung dispatches (after admission shaping).
        authorization: Frozen authority for the accepted request.
        zdr_constrained: Whether the host flagged this rung for zero-data-retention
            dispatch (``ExecutionSnapshot.zdr_constrained_deployment_ids``).
        environ: The environment to read for the operator stop; ``None`` reads
            ``os.environ``.

    Returns:
        True only for a Chat Completions, Responses or generateContent request with no caller
        cache marker, not routed under zero data retention, on a caching wire,
        with caching allowed by the organization, the operator and the rung,
        and a prompt that could reach the smallest cacheable prefix.
    """
    if request.surface not in _AUTOMATIC_SURFACES or profile.dialect not in CACHING_WIRES:
        return False
    if authorization.prompt_caching_disabled or zdr_routed(
        request, authorization, zdr_constrained=zdr_constrained
    ):
        return False
    policy = deployment.gateway.dispatch
    setting = None if policy is None else policy.automatic_prompt_caching
    if setting is False or (setting is None and not caches_by_default(deployment, profile)):
        return False
    # The attempt reservation prices a cache write only on a rung that reports
    # them; without it a gateway-requested write could settle above the hold.
    if not deployment.gateway.capabilities.reports_cache_creation_input_tokens:
        return False
    if automatic_prompt_caching_stopped(environ):
        return False
    if (
        cache_markers(request)
        or request.prompt_cache_ttl is not None
        # ``prompt_cache_options.mode: "explicit"`` with no breakpoint means cache nothing.
        or request.prompt_cache_mode is not None
        or any(_SERVER_TOOL_CACHING in tool for tool in request.provider_server_tools)
    ):
        return False
    return not certainly_below_cacheable_minimum(request)


def zdr_routed(
    request: GatewayRequest,
    authorization: AuthorizationSnapshot,
    *,
    zdr_constrained: bool = False,
) -> bool:
    """Whether this dispatch is routed under zero data retention.

    The gateway adds no cache state of its own to a ZDR-routed request, the same
    rule as automatic explicit caching (``native_explicit_cache``). Caller markers
    on such a request are untouched: they forward exactly as before.

    The engine sees three facts: the caller's per-request demand
    (``GatewayRequest.zdr_requested``, ``provider: {"zdr": true}``), the
    authority's ``AuthorizationSnapshot.zdr_requested``, and a rung the host
    flagged for ZDR-constrained dispatch. An organization-wide ZDR requirement
    is a host posture the engine does not see on its own; a host excludes such
    an organization by setting ``AuthorizationSnapshot.zdr_requested``.

    Args:
        request: The request this rung dispatches.
        authorization: Frozen authority for the accepted request.
        zdr_constrained: Whether the host flagged this rung for ZDR dispatch.

    Returns:
        True when any of the three facts holds.
    """
    return request.zdr_requested or authorization.zdr_requested or zdr_constrained


def certainly_below_cacheable_minimum(request: GatewayRequest) -> bool:
    """Whether the prompt is provably shorter than any Claude model's cacheable prefix.

    Claude's tokenizer never produces more than one token per UTF-8 byte of
    text, so the text bytes plus a framing allowance bound the prompt's token
    count from ABOVE. Media tokenizes by its own rules and cannot be bounded
    here, so a request carrying any is never skipped.

    Args:
        request: The request this rung dispatches.

    Returns:
        True when the upper bound is under :data:`SMALLEST_CACHEABLE_PREFIX_TOKENS`.
    """
    if any(part.kind != "text" for message in request.messages for part in message.content_parts):
        return False
    upper_bound = (
        prompt_text_bytes(request)
        + FRAMING_TOKENS_PER_REQUEST
        + FRAMING_TOKENS_PER_MESSAGE * len(request.messages)
        + FRAMING_TOKENS_PER_TOOL * (len(request.tools) + len(request.provider_server_tools))
    )
    return upper_bound < SMALLEST_CACHEABLE_PREFIX_TOKENS


def with_automatic_prompt_caching(payload: JsonObject, dialect: str) -> JsonObject:
    """Return the provider payload carrying the automatic five-minute cache checkpoint.

    Args:
        payload: The payload the rung's dialect built, left unmodified.
        dialect: The rung's wire dialect, one of :data:`CACHING_WIRES`.

    Returns:
        A copy with the top-level ``cache_control`` (Anthropic Messages) or a
        ``cachePoint`` after the last message's content (Bedrock Converse); the
        payload unchanged when it has no message content to mark or already
        ends in a checkpoint.

    Raises:
        ValueError: ``dialect`` is not a caching wire.
    """
    match dialect:
        case "anthropic_messages":
            if "cache_control" in payload:
                return payload
            return {**payload, "cache_control": dict(AUTOMATIC_CACHE_CONTROL)}
        case "bedrock_converse_stream":
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                return payload
            last = messages[-1]
            if not isinstance(last, dict):
                return payload
            content = last.get("content")
            if not isinstance(content, list) or not content:
                return payload
            final = content[-1]
            if isinstance(final, dict) and "cachePoint" in final:
                return payload
            marked: JsonObject = {**last, "content": [*content, dict(BEDROCK_CACHE_POINT)]}
            return {**payload, "messages": [*messages[:-1], marked]}
        case _:
            raise ValueError(f"automatic prompt caching has no placement on {dialect!r}")


_PRE_CACHING_CLAUDE = (
    "claude-v2",
    "claude-instant",
    "claude-3-haiku",
    "claude-3-sonnet",
    "claude-3-opus",
    "claude-3-5-sonnet",
)
"""Bedrock Claude model-id stems that predate Bedrock prompt caching (no ``cachePoint``)."""


def _names_pre_caching_claude(
    deployment: ExactModelDeployment, profile: GatewayWireProfile
) -> bool:
    """Whether the Bedrock rung serves a Claude model that rejects a ``cachePoint``."""
    return any(
        stem in name.lower()
        for name in (profile.model_id, deployment.provider_model, deployment.exact_model_id)
        for stem in _PRE_CACHING_CLAUDE
    )


def _names_claude(deployment: ExactModelDeployment, profile: GatewayWireProfile) -> bool:
    """Whether the Bedrock rung serves a Claude model, by its provider-side model id."""
    return any(
        "claude" in name.lower()
        for name in (profile.model_id, deployment.provider_model, deployment.exact_model_id)
    )

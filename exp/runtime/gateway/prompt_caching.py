"""The organization prompt-caching switch, applied once at native admission.

The host sets ``AuthorizationSnapshot.prompt_caching_disabled`` from the
organization's stored data-control policy. When it is set, admission makes
three promises on every route:

* The gateway adds no cache state of its own. No tenant cache-affinity key
  is derived (the ``prompt_cache_key`` OpenAI-family and OpenRouter rungs
  route by), no Google cache resource is created or read, and the rendezvous
  and sticky rung placement of ``maximize_cache_affinity`` pools is skipped,
  so the route keeps its certified order.
* Every caller cache marker is removed before any payload is built: Anthropic
  ``cache_control`` on the top level, content blocks, tool results, tool calls
  and tool definitions (Bedrock would translate these to ``cachePoint``; OpenAI
  ``prompt_cache_breakpoint`` markers ride the same carriers), and
  OpenAI's 12-hour ``prompt_cache_options.ttl``. The caller's
  ``prompt_cache_key`` stays in request identity but reaches no provider.
  Each removed path is disclosed in ``ignored_parameters``.
* A provider that caches prompt prefixes on its own, with no request field
  that turns it off, may still do so. Such rungs stay routable and the
  response discloses ``prompt_caching`` as best effort instead.
"""

from __future__ import annotations

from collections.abc import Sequence

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import MessageContentPart
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayMessage, GatewayRequest
from exp.runtime.gateway.prompt_cache_affinity import (
    provider_prompt_cache_key,
    provider_session_id,
)
from exp.runtime.models.providers.base import GatewayWireProfile

PROMPT_CACHING_OFF_SUFFIX = "->not_forwarded(organization_prompt_caching_off)"
"""Suffix for a caller cache marker or hint removed because the organization turned caching off."""

PROMPT_CACHING_BEST_EFFORT = "prompt_caching->best_effort(provider_caches_implicitly)"
"""Disclosure for a route where a rung's provider caches prefixes with no request opt-out."""

_MARKER = "cache_control"


def with_cache_affinity(
    provider_request: GatewayRequest, authorization: AuthorizationSnapshot
) -> GatewayRequest:
    """Attach the tenant's cache-affinity key and session id to the final dispatch request.

    Cache affinity is per tenant and per session, so it is derived once, from
    the request admission settled on and the frozen authority, and read by
    every rung's payload builder that forwards it. It is applied last so no
    admission-time rebuild (route narrowing, capability or schema coercion)
    can drop it; the public request keeps the caller's own ``prompt_cache_key``
    untouched. An organization with prompt caching off gets neither.

    Args:
        provider_request: The request admission settled on.
        authorization: Frozen authority for the accepted request.

    Returns:
        The request carrying the derived key, or unchanged when caching is off.
    """
    if authorization.prompt_caching_disabled:
        return provider_request
    organization_id = str(authorization.organization_id)
    identity_id = str(authorization.identity_id)
    return provider_request.model_copy(
        update={
            "provider_prompt_cache_key": provider_prompt_cache_key(
                provider_request, organization_id=organization_id, identity_id=identity_id
            ),
            "provider_session_id": provider_session_id(
                provider_request, organization_id=organization_id, identity_id=identity_id
            ),
        }
    )


def without_prompt_caching(
    request: GatewayRequest, authorization: AuthorizationSnapshot
) -> GatewayRequest:
    """Remove every caller cache marker when the organization turned caching off.

    Markers are cost hints: removing them changes what a provider is asked to
    cache, never what the model sees. Admission runs after the host froze the
    request digest, so the removal never alters authorized identity. The
    caller's ``prompt_cache_key`` is kept and disclosed, because
    :func:`with_cache_affinity` derives nothing from it.

    Args:
        request: The decoded public request.
        authorization: Frozen authority for the accepted request.

    Returns:
        The request without markers, each removed path disclosed in the
        caller's spelling; unchanged when caching is on or nothing was sent.
    """
    if not authorization.prompt_caching_disabled:
        return request
    paths = _marker_paths(request)
    if not paths:
        return request
    return request.model_copy(
        update={
            "provider_cache_control": None,
            "prompt_cache_ttl": None,
            "tools": tuple(
                tool.model_copy(update={_MARKER: None}) if tool.cache_control else tool
                for tool in request.tools
            ),
            "provider_server_tools": tuple(
                _unmarked(tool) for tool in request.provider_server_tools
            ),
            "messages": tuple(_unmarked_message(message) for message in request.messages),
            "ignored_parameters": tuple(
                dict.fromkeys(
                    (
                        *request.ignored_parameters,
                        *(f"{path}{PROMPT_CACHING_OFF_SUFFIX}" for path in paths),
                    )
                )
            ),
        }
    )


def disclose_best_effort_caching(
    request: GatewayRequest,
    deployments: Sequence[ExactModelDeployment],
    profiles: Sequence[GatewayWireProfile],
    authorization: AuthorizationSnapshot,
) -> GatewayRequest:
    """Disclose best-effort caching when any admitted rung caches on its own.

    Anthropic's API, and Claude on Vertex AI, cache only at the markers
    :func:`without_prompt_caching` already removed, so a route served only by
    those rungs caches nothing. Every other provider in the fleet either
    caches prefixes automatically with no documented per-request opt-out, or
    offers one only for some models, so the gateway cannot promise the
    provider keeps nothing.

    Args:
        request: The admitted public request.
        deployments: The admitted rungs in dispatch order.
        profiles: The wire profile of each admitted rung, aligned with ``deployments``.
        authorization: Frozen authority for the accepted request.

    Returns:
        The request with :data:`PROMPT_CACHING_BEST_EFFORT` disclosed, or
        unchanged when caching is on or every rung caches only on request.
    """
    if not authorization.prompt_caching_disabled or all(
        _caches_only_on_request(deployment, profile)
        for deployment, profile in zip(deployments, profiles, strict=True)
    ):
        return request
    return request.model_copy(
        update={
            "ignored_parameters": tuple(
                dict.fromkeys((*request.ignored_parameters, PROMPT_CACHING_BEST_EFFORT))
            )
        }
    )


def _caches_only_on_request(deployment: ExactModelDeployment, profile: GatewayWireProfile) -> bool:
    """Whether this rung's provider caches nothing unless a marker asks it to."""
    return deployment.provider == "anthropic" or (
        deployment.provider == "vertex" and profile.dialect == "anthropic_messages"
    )


def _marker_paths(request: GatewayRequest) -> tuple[str, ...]:
    """Name every cache marker or hint the caller sent, in the caller's spelling."""
    paths: list[str] = []
    if request.provider_cache_control is not None:
        paths.append(_MARKER)
    if any(tool.cache_control is not None for tool in request.tools) or any(
        _MARKER in tool for tool in request.provider_server_tools
    ):
        paths.append(f"tools.{_MARKER}")
    if any(call.cache_control is not None for m in request.messages for call in m.tool_calls):
        paths.append(f"messages.tool_calls.{_MARKER}")
    if any(_content_marked(message) for message in request.messages):
        # OpenAI breakpoints ride the same carriers; name them as the caller did.
        spelling = "prompt_cache_breakpoint" if request.prompt_cache_breakpoints else _MARKER
        paths.append(f"messages.content.{spelling}")
    if request.prompt_cache_key is not None:
        paths.append("prompt_cache_key")
    if request.prompt_cache_ttl is not None:
        paths.append("prompt_cache_options.ttl")
    return tuple(paths)


def _content_marked(message: GatewayMessage) -> bool:
    """Whether a message carries a marker on itself, a content part or a verbatim block."""
    blocks = (*message.provider_text_blocks, *(message.provider_anthropic_blocks or ()))
    if message.provider_anthropic_block is not None:
        blocks = (*blocks, message.provider_anthropic_block)
    return (
        message.cache_control is not None
        or any(_part_marker(part) is not None for part in message.content_parts)
        or any(_MARKER in block for block in blocks)
    )


def _part_marker(part: MessageContentPart) -> JsonObject | None:
    """Return the marker of a text, image or document part, the only parts that carry one."""
    if part.kind == "text" or part.kind == "image" or part.kind == "document":
        return part.cache_control
    return None


def _unmarked(block: JsonObject) -> JsonObject:
    """Return a verbatim provider block without its cache marker."""
    return {key: value for key, value in block.items() if key != _MARKER}


def _unmarked_message(message: GatewayMessage) -> GatewayMessage:
    """Return a message with every marker removed and its content otherwise intact."""
    if not _content_marked(message) and all(c.cache_control is None for c in message.tool_calls):
        return message
    return message.model_copy(
        update={
            _MARKER: None,
            "tool_calls": tuple(
                call.model_copy(update={_MARKER: None}) for call in message.tool_calls
            ),
            "content_parts": tuple(
                part.model_copy(update={_MARKER: None}) if _part_marker(part) else part
                for part in message.content_parts
            ),
            "provider_text_blocks": tuple(_unmarked(b) for b in message.provider_text_blocks),
            "provider_anthropic_blocks": None
            if message.provider_anthropic_blocks is None
            else tuple(_unmarked(b) for b in message.provider_anthropic_blocks),
            "provider_anthropic_block": None
            if message.provider_anthropic_block is None
            else _unmarked(message.provider_anthropic_block),
        }
    )
